import os
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

import faiss

# If you are using the **new** OpenAI client, comment the next 2 lines
# and use the section marked "NEW OPENAI CLIENT" below.
import openai


# =========================================================
# CONFIG
# =========================================================

FAISS_INDEX_PATH = "echs_faiss.index"
SEGMENTS_PARQUET_PATH = "echs_segments_master.parquet"

# Google Sheet – replace with your actual ID / URL or leave blank
GOOGLE_SHEET_ID = os.getenv("ECHS_FEEDBACK_SHEET_ID", "")

SYSTEM_PROMPT = """
You are an assistant that answers questions about the Ex-Servicemen Contributory Health Scheme (ECHS).

Use ONLY the provided context from official ECHS training videos and transcripts.
If the answer is not present or is unclear, say you are not sure and suggest asking the ECHS polyclinic or checking official ECHS instructions.

Be concise and clear. Where relevant, mention that rules can change and beneficiaries should confirm with official sources.
"""

# Curated example questions – based on your 6 ECHS videos
EXAMPLE_QUESTIONS: Dict[str, List[str]] = {
    "Basics of ECHS": [
        "What is ECHS and who is eligible?",
        "How do I register for ECHS as a veteran?",
        "What documents are required for ECHS enrolment?",
    ],
    "ECHS smart card & dependents": [
        "How do I apply for an ECHS smart card?",
        "How can I add or delete dependents in ECHS?",
        "What should I do if my ECHS smart card is lost or damaged?",
    ],
    "Using ECHS polyclinics & hospitals": [
        "How do I take treatment at an ECHS polyclinic?",
        "What is the procedure for referral to empanelled hospitals?",
        "How are emergency admissions handled under ECHS?",
    ],
    "Bills, claims & problems": [
        "How are hospital bills settled under ECHS?",
        "What can I do if a hospital refuses cashless treatment?",
        "Where can I complain if I face problems with ECHS services?",
    ],
}

TOP_K = 4  # number of segments to retrieve


# =========================================================
# CACHED LOADERS
# =========================================================

@st.cache_resource(show_spinner=True)
def load_faiss_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
    return faiss.read_index(path)


@st.cache_data(show_spinner=True)
def load_segments_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Segments parquet not found at {path}")
    df = pd.read_parquet(path)
    return df


# =========================================================
# EMBEDDINGS & LLM
# =========================================================

def get_openai_model_name() -> str:
    # adjust if you prefer a different model
    return os.getenv("ECHS_OPENAI_MODEL", "gpt-4o-mini")


def embed_query(text: str) -> np.ndarray:
    """
    Returns a 1D numpy array embedding for the query.
    Uses OpenAI embeddings. Adjust model name as needed.
    """
    # LEGACY OPENAI CLIENT
    emb_model = os.getenv("ECHS_EMBED_MODEL", "text-embedding-3-small")
    resp = openai.Embedding.create(
        model=emb_model,
        input=text,
    )
    vec = np.array(resp["data"][0]["embedding"], dtype="float32")
    return vec

    # --- NEW OPENAI CLIENT EXAMPLE (if you upgrade) ---
    # from openai import OpenAI
    # client = OpenAI()
    # emb_model = os.getenv("ECHS_EMBED_MODEL", "text-embedding-3-small")
    # resp = client.embeddings.create(model=emb_model, input=text)
    # vec = np.array(resp.data[0].embedding, dtype="float32")
    # return vec


def call_llm(context: str, question: str) -> str:
    """
    Calls OpenAI chat completion with RAG context.
    """
    model = get_openai_model_name()

    # LEGACY OPENAI CLIENT
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context from ECHS training materials:\n\n{context}\n\n"
                           f"User question: {question}",
            },
        ],
        temperature=0.2,
    )
    return completion["choices"][0]["message"]["content"].strip()

    # --- NEW OPENAI CLIENT EXAMPLE (if you upgrade) ---
    # from openai import OpenAI
    # client = OpenAI()
    # completion = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user",
    #          "content": f"Context from ECHS training materials:\n\n{context}\n\n"
    #                     f"User question: {question}"},
    #     ],
    #     temperature=0.2,
    # )
    # return completion.choices[0].message.content.strip()


# =========================================================
# RETRIEVAL
# =========================================================

def search_segments(
    query: str,
    index,
    segments_df: pd.DataFrame,
    top_k: int = TOP_K,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Embed query, search FAISS, return subset dataframe + distances list.
    """
    q_emb = embed_query(query).reshape(1, -1)
    distances, indices = index.search(q_emb, top_k)

    idxs = indices[0]
    dists = distances[0].tolist()

    valid_rows = []
    valid_dists = []
    for i, d in zip(idxs, dists):
        if i < 0 or i >= len(segments_df):
            continue
        row = segments_df.iloc[i].copy()
        row["_distance"] = d
        valid_rows.append(row)
        valid_dists.append(d)

    if not valid_rows:
        return pd.DataFrame(), []

    result_df = pd.DataFrame(valid_rows)
    return result_df, valid_dists


def format_time_seconds(sec: Any) -> str:
    try:
        sec = int(sec)
    except Exception:
        return ""
    mins = sec // 60
    s = sec % 60
    return f"{mins:02d}:{s:02d}"


def build_context_from_segments(segments_df: pd.DataFrame) -> str:
    """
    Concatenate retrieved segments into a context string for the LLM.
    """
    if segments_df.empty:
        return ""

    texts = []
    for _, row in segments_df.iterrows():
        text = row.get("text", "") or row.get("segment_text", "") or row.get("chunk", "")
        video_title = row.get("video_title", "") or row.get("source_title", "")
        start_time = format_time_seconds(row.get("start_time", row.get("start_sec", 0)))
        prefix = ""
        if video_title or start_time:
            prefix = f"[Source: {video_title} @ {start_time}] "
        texts.append(prefix + str(text))
    return "\n\n".join(texts)


# =========================================================
# FEEDBACK LOGGING (Google Sheets stub)
# =========================================================

def log_feedback_to_gsheet(
    question: str,
    answer: str,
    rating: int,
    comment: str,
    meta: Dict[str, Any],
):
    """
    Append a row to Google Sheet.
    This is wrapped in try/except so app never crashes
    if Sheets is not configured.
    """
    if not GOOGLE_SHEET_ID:
        # silently skip if not configured
        return

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        # Path to your service-account JSON in Streamlit secrets or env
        svc_json_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if not svc_json_path or not os.path.exists(svc_json_path):
            return

        creds = Credentials.from_service_account_file(svc_json_path, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(GOOGLE_SHEET_ID)
        ws = sh.sheet1

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        row = [
            ts,
            question,
            answer,
            rating,
            comment,
            meta.get("retrieved_titles", ""),
            meta.get("retrieved_distances", ""),
        ]
        ws.append_row(row)

    except Exception as e:
        # Avoid breaking the app; optionally show a small notice in admin mode.
        st.session_state["feedback_error"] = str(e)


# =========================================================
# UI HELPERS
# =========================================================

def render_example_questions():
    """
    Show curated example questions grouped by topic.
    Clicking a question sets it as the current question.
    """
    if not EXAMPLE_QUESTIONS:
        return

    with st.expander("Browse example questions by topic"):
        for topic, questions in EXAMPLE_QUESTIONS.items():
            st.markdown(f"**{topic}**")
            cols = st.columns(2)
            for i, q in enumerate(questions):
                col = cols[i % 2]
                key = f"exq_{abs(hash(topic + q))}_{i}"
                if col.button(q, key=key):
                    st.session_state["prefill_question"] = q
                    st.session_state["from_example"] = True
                    st.experimental_rerun()


def render_sources_block(retrieved_df: pd.DataFrame):
    """
    Show retrieved source segments under the answer (for end users).
    """
    if retrieved_df.empty:
        return

    st.markdown("###### Sources used from ECHS training materials")
    for _, row in retrieved_df.iterrows():
        title = row.get("video_title", "") or row.get("source_title", "ECHS training video")
        text = row.get("text", "") or row.get("segment_text", "") or row.get("chunk", "")
        start_time = format_time_seconds(row.get("start_time", row.get("start_sec", 0)))
        url = row.get("video_url", "") or row.get("source_url", "")

        # Build a timestamped URL if possible
        ts_link = ""
        if url and start_time:
            # assume start_time is in seconds for URL
            try:
                sec = int(row.get("start_time", row.get("start_sec", 0)))
            except Exception:
                sec = 0
            if "youtube" in url or "youtu.be" in url:
                sep = "&" if "?" in url else "?"
                ts_link = f"{url}{sep}t={sec}s"
            else:
                ts_link = url

        bullet = f"- **{title}**"
        if start_time:
            bullet += f" at `{start_time}`"
        if ts_link:
            bullet += f" ([open video]({ts_link}))"
        st.markdown(bullet)
        st.markdown(f"> {text[:300]}{'...' if len(text) > 300 else ''}")


def render_admin_debug(retrieved_df: pd.DataFrame):
    """
    Admin-only debug table with distances & raw segments.
    """
    if retrieved_df.empty:
        st.info("No segments retrieved.")
        return

    debug_cols = []
    for col in ["video_title", "start_time", "text", "_distance"]:
        if col in retrieved_df.columns:
            debug_cols.append(col)

    st.markdown("### 🔧 Admin – Retrieved segments & distances")
    st.dataframe(
        retrieved_df[debug_cols],
        use_container_width=True,
        hide_index=True,
    )


def render_feedback_block(
    question: str,
    answer: str,
    retrieved_df: pd.DataFrame,
):
    """
    Feedback UI: rating slider + comment box. Logs to Sheets.
