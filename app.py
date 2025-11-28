import os
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import faiss
import openai   # ← legacy style

# =========================================================
# FILE PATHS (change only if your filenames differ)
# =========================================================
FAISS_INDEX_PATH = "echs_faiss.index"
SEGMENTS_PARQUET_PATH = "echs_segments_master.parquet"

TOP_K = 4  # retrieved chunks per answer

SYSTEM_PROMPT = """
You answer questions about ECHS based ONLY on the provided training-video transcripts.
If answer not found in retrieved context → say so politely. Avoid hallucination.
Answer clearly, short, helpful.
"""

# =========================================================
# 💡 Example Questions (appear as buttons)
# =========================================================
EXAMPLE_QUESTIONS = {
    "Basics of ECHS": [
        "What is ECHS and who is eligible?",
        "How do I register for ECHS?",
        "What documents are required for ECHS enrolment?",
    ],
    "Smart Card & Dependents": [
        "How do I apply for an ECHS smart card?",
        "How do I add or delete dependents?",
        "What to do if the smart card is lost?",
    ],
    "Polyclinic & Hospital Use": [
        "How does treatment at an ECHS polyclinic work?",
        "Procedure for referral to empanelled hospitals?",
        "How are emergency cases treated?",
    ],
    "Billing & Issues": [
        "How are hospital bills settled under ECHS?",
        "What if hospital refuses cashless treatment?",
        "Where to complain if problems occur?",
    ],
}

# =========================================================
# LOADING (cached = faster)
# =========================================================
@st.cache_resource
def load_faiss():
    return faiss.read_index(FAISS_INDEX_PATH)

@st.cache_data
def load_segments():
    return pd.read_parquet(SEGMENTS_PARQUET_PATH)

# =========================================================
# EMBEDDING + LLM (Legacy OpenAI)
# =========================================================
def embed(text: str) -> np.ndarray:
    resp = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp["data"][0]["embedding"]).astype("float32")

def ask_llm(context: str, question: str) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user",
             "content":f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2
    )
    return resp["choices"][0]["message"]["content"].strip()

# =========================================================
# SEARCH
# =========================================================
def retrieve(query, index, df, k=TOP_K):
    q = embed(query).reshape(1,-1)
    D,I = index.search(q,k)
    out=[]
    for idx,dist in zip(I[0],D[0]):
        row=df.iloc[idx].copy()
        row["_dist"]=float(dist)
        out.append(row)
    return pd.DataFrame(out)

def build_context(rows:pd.DataFrame) -> str:
    if rows.empty: return ""
    parts=[]
    for _,r in rows.iterrows():
        txt=r.get("text","") or r.get("chunk","")
        t=r.get("video_title","")
        tm=int(r.get("start_time",0))
        stamp=f"[{t} @ {tm//60:02d}:{tm%60:02d}] "
        parts.append(stamp+txt)
    return "\n\n".join(parts)

# =========================================================
# UI – Example Question Buttons
# =========================================================
def example_panel():
    with st.expander("📌 Browse example questions"):
        for topic,qs in EXAMPLE_QUESTIONS.items():
            st.markdown(f"**{topic}**")
            cols=st.columns(2)
            for i,q in enumerate(qs):
                if cols[i%2].button(q):
                    st.session_state["auto_q"]=q
                    st.experimental_rerun()

# =========================================================
# MAIN Q&A BLOCK
# =========================================================
def chat_block():
    example_panel()

    # if sample clicked → auto ask
    question = st.session_state.pop("auto_q",None) or st.chat_input("Ask here...")

    if not question: return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching ECHS training material..."):
            df = retrieve(question, st.session_state.idx, st.session_state.segs)
            ctx = build_context(df)
            if not ctx:
                st.write("No matching info in available training videos. Please confirm with official ECHS sources.")
                return
            ans = ask_llm(ctx, question)
            st.write(ans)

        st.markdown("### 🔗 Sources used")
        for _,r in df.iterrows():
            t=r.get("video_title","ECHS Video")
            tm=int(r.get("start_time",0))
            st.markdown(f"- **{t}** at `{tm//60:02d}:{tm%60:02d}` — {r['text'][:150]}...")

# =========================================================
# MAIN APP
# =========================================================
def main():
    st.set_page_config(page_title="ECHS BOT",page_icon="🩺",layout="wide")
    st.title("ECHS Assistant Bot")

    st.session_state.idx = load_faiss()
    st.session_state.segs = load_segments()

    chat_block()

if __name__=="__main__":
    main()
