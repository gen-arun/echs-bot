"""
ECHS Assistant – Streamlit App
Uses:
- Precomputed FAISS index over ECHS video transcripts
- SentenceTransformers for query embeddings
- Groq LLM (llama-3.3-70b-versatile) for answers
- Google Apps Script for feedback logging
"""

from pathlib import Path
from datetime import datetime
import json
import csv

import streamlit as st
import numpy as np
import pandas as pd
import faiss
import requests
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================
#  CONFIG
# ============================================================

# Paths (relative to the repository root where app.py lives)
INDEX_PATH = Path("echs_faiss.index")
PARQUET_PATH = Path("echs_segments_master.parquet")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Google Apps Script endpoint for feedback logging
GOOGLE_APPS_SCRIPT_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbznEPzlIYwDDZWtXAoIJdUS2NdBaSVW11P0K9BshGlu2u-eHcVxdJhhE2cUm_4v7Vo4fg/exec"
)

# ------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="ECHS Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# Custom CSS (simple styling)
# ------------------------------------------------------------
st.markdown(
    """
<style>
.main-header {
    font-size: 2.0rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 1.0rem;
    color: #666666;
    margin-bottom: 0.5rem;
}
.answer-box {
    border-radius: 8px;
    padding: 0.8rem;
    background-color: #f5f9ff;
    border: 1px solid #d8e2ff;
}
.source-box {
    border-radius: 6px;
    padding: 0.6rem;
    background-color: #f9f9f9;
    border: 1px dashed #cccccc;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
#  SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    # list of dicts: {role: "user"|"assistant", content: str, sources: dict}
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

if "run_query" not in st.session_state:
    st.session_state.run_query = False

if "run_suggested" not in st.session_state:
    st.session_state.run_suggested = False

# ============================================================
#  INITIALIZATION – KB + LLM
# ============================================================

@st.cache_resource
def initialize_system():
    """
    Initialize:
    - Groq client
    - FAISS index
    - Segment dataframe from parquet
    - Embedding model
    """

    # 1) Groq client (API key in Streamlit secrets)
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        raise RuntimeError(
            "GROQ_API_KEY not found in Streamlit secrets. Please configure it."
        )
    client = Groq(api_key=groq_api_key)

    # 2) Check that required files exist
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base index file not found: {INDEX_PATH}. Please upload it."
        )
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base data file not found: {PARQUET_PATH}. Please upload it."
        )

    # 3) Load FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # 4) Load segments dataframe
    df = pd.read_parquet(PARQUET_PATH)  # columns: segment_id, source_file, rel_path, text

    # 5) Load embedding model (for queries)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return client, index, df, embedding_model


# ============================================================
#  RETRIEVAL + LLM
# ============================================================

def retrieve_context(question: str, index, df: pd.DataFrame, embedding_model, n_results: int = 5):
    """Embed question and retrieve top-k segments from FAISS."""
    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, n_results)

    # Get relevant rows
    rows = [df.iloc[i] for i in indices[0]]

    # Build context and simple "sources" (from file names)
    context_parts = []
    sources = {}
    for row, dist in zip(rows, distances[0]):
        src = str(row.get("source_file", "segment"))
        text = str(row.get("text", ""))
        context_parts.append(f"From file: {src}\n{text}")
        if src not in sources:
            sources[src] = f"(local segment, distance={dist:.3f})"

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def build_prompt(question: str, context: str) -> str:
    """Build the prompt for the LLM."""
    prompt = f"""You are an expert assistant helping people understand the Ex-Servicemen Contributory Health Scheme (ECHS).

You will answer based ONLY on the information given below from ECHS training videos and transcripts. 
If the answer is not clearly present in the context, say you are not sure and suggest checking with the ECHS polyclinic or official instructions.

================= CONTEXT START =================
{context}
================= CONTEXT END ===================

User question:
{question}

Please respond with:
1. A clear, direct answer in simple language.
2. Any important conditions, limitations, or exceptions.
3. A short bullet list summary at the end.

Format:

**Answer:**
<your answer>

**Important points:**
- point 1
- point 2
- point 3 (if any)
"""
    return prompt


def call_llm(prompt: str, client: Groq):
    """Call Groq LLM and return the answer text or a friendly error."""
    try:
        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "503" in error_msg:
            return "⏱️ The AI service is temporarily busy. Please try again in a moment."
        if "rate limit" in error_msg or "429" in error_msg:
            return "⚠️ Too many requests right now. Please wait 30 seconds and try again."
        if "connection" in error_msg:
            return "🔌 Connection issue. Please check your internet and try again."
        return "❌ Something went wrong. Please try rephrasing your question or contact support."


def get_answer(question: str, client, index, df, embedding_model, n_results: int = 5):
    """High-level: retrieve context and ask the LLM."""
    context, sources = retrieve_context(question, index, df, embedding_model, n_results)
    prompt = build_prompt(question, context)
    answer = call_llm(prompt, client)
    return answer, sources


# ============================================================
#  FEEDBACK LOGGING
# ============================================================

def log_feedback(question: str, answer: str, rating: int, comment: str):
    """
    Send feedback row to Google Apps Script, which writes to Google Sheet.
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment or "",
    }

    try:
        response = requests.post(
            GOOGLE_APPS_SCRIPT_URL,
            json=payload,
            timeout=10,
        )
        if response.status_code != 200:
            st.warning(
                f"Feedback not confirmed by server (status {response.status_code})."
            )
    except Exception as e:
        # Show a soft warning, but don't break the app
        st.warning(f"Could not send feedback to log: {e}")


# ============================================================
#  UI HELPERS
# ============================================================

def submit_question():
    """Common handler when user hits Enter or clicks Get Answer."""
    q = st.session_state.get("question_input", "").strip()
    if q:
        st.session_state.pending_question = q
        st.session_state.run_query = True


def handle_example_click(text: str):
    """When a suggested question is clicked."""
    st.session_state.question_input = text
    st.session_state.pending_question = text
    st.session_state.run_query = True


def render_header():
    st.markdown('<h1 class="main-header">🏥 ECHS Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Ask questions about Ex-Servicemen Contributory Health Scheme</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "ℹ️ This assistant answers questions based on a limited set of ECHS training videos and transcripts. "
        "It may not cover every edge case. For specific situations or final decisions, "
        "please consult your ECHS polyclinic or official ECHS instructions."
    )


def render_question_block():
    """Render text input + example questions, trigger run_query when needed."""

    st.text_input(
        "💬 Ask your question:",
        placeholder="e.g., What is ECHS? How do I register? What documents are required?",
        key="question_input",
        on_change=submit_question,  # Enter key
    )

    with st.expander("📝 Browse example questions by topic"):
        tab1, tab2, tab3 = st.tabs(
            ["Registration & Eligibility", "Emergency Care", "Referrals & Reimbursement"]
        )

        with tab1:
            st.markdown("**Registration & Eligibility**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("What is ECHS?", use_container_width=True, key="q1"):
                    handle_example_click("What is ECHS?")
                if st.button("Who is eligible for ECHS?", use_container_width=True, key="q2"):
                    handle_example_click("Who is eligible for ECHS?")
            with col2:
                if st.button("How do I register for ECHS?", use_container_width=True, key="q3"):
                    handle_example_click("How do I register for ECHS?")
                if st.button(
                    "What documents are required for ECHS registration?",
                    use_container_width=True,
                    key="q4",
                ):
                    handle_example_click("What documents are required for ECHS registration?")

        with tab2:
            st.markdown("**Emergency Care**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Can I get emergency treatment at AFMS hospital?",
                    use_container_width=True,
                    key="q5",
                ):
                    handle_example_click("Can I get emergency treatment at AFMS hospital?")
                if st.button(
                    "What is the procedure for emergency treatment in ECHS?",
                    use_container_width=True,
                    key="q6",
                ):
                    handle_example_click("What is the procedure for emergency treatment in ECHS?")
            with col2:
                if st.button(
                    "Can my parents be treated at AFMS in emergency?",
                    use_container_width=True,
                    key="q7",
                ):
                    handle_example_click(
                        "In an emergency, can my parents be treated at AFMS hospital on cashless basis?"
                    )
                if st.button(
                    "What about non-empanelled hospitals?",
                    use_container_width=True,
                    key="q8",
                ):
                    handle_example_click("Can I get treatment in non-empanelled hospitals?")

        with tab3:
            st.markdown("**Referrals & Reimbursement**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("What is HSR?", use_container_width=True, key="q9"):
                    handle_example_click("What is HSR?")
                if st.button(
                    "Can Non-MIL Polyclinic refer to AFMS?",
                    use_container_width=True,
                    key="q10",
                ):
                    handle_example_click("Can a Non MIL Polyclinic refer ESM to AFMS hospital?")
            with col2:
                if st.button(
                    "Can 26-year-old dependent be referred?",
                    use_container_width=True,
                    key="q11",
                ):
                    handle_example_click(
                        "Can my dependent who is 26 years old be referred by the polyclinic to AFMS Hospital?"
                    )
                if st.button(
                    "How do I claim reimbursement?",
                    use_container_width=True,
                    key="q12",
                ):
                    handle_example_click("How do I claim medical reimbursement in ECHS?")

    # Main "Get Answer" button (same handler as Enter)
    st.button(
        "🔍 Get Answer",
        use_container_width=True,
        type="primary",
        on_click=submit_question,
    )


def render_conversation_and_feedback():
    """Show conversation history and feedback controls."""
    if not st.session_state.messages:
        return

    st.markdown("---")
    st.markdown("### 💬 Conversation History")

    # Show latest Q–A pairs first
    for i in range(len(st.session_state.messages) - 1, -1, -2):
        if i < 0:
            break
        answer_msg = st.session_state.messages[i]
        if answer_msg["role"] != "assistant":
            continue

        question_msg = st.session_state.messages[i - 1] if i - 1 >= 0 else None
        question_text = question_msg["content"] if question_msg else ""
        answer_text = answer_msg["content"]

        st.markdown(f"**❓ Question:** {question_text}")
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(f"**💡 Answer:**\n\n{answer_text}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Sources (from retrieval)
        if "sources" in answer_msg and answer_msg["sources"]:
            st.markdown('<div class="source-box">', unsafe_allow_html=True)
            st.markdown("**📄 Information taken from these segments/files:**")
            for j, (src, desc) in enumerate(answer_msg["sources"].items(), start=1):
                st.markdown(f"{j}. `{src}` — {desc}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Rating + comment
        rating_key = f"rating_{i}"
        comment_key = f"comment_{i}"
        submitted_key = f"feedback_submitted_{i}"

        st.write("**Rate this answer:**")
        rating = st.radio(
            "How would you rate this answer?",
            [1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x,
            horizontal=True,
            key=rating_key,
        )

        comment = ""
        if rating <= 3:
            comment = st.text_area(
                "Tell us what was missing or confusing (optional):",
                key=comment_key,
            )

        if not st.session_state.get(submitted_key, False):
            if st.button("💾 Submit feedback", key=f"submit_feedback_{i}"):
                log_feedback(question_text, answer_text, rating, comment)
                st.session_state[submitted_key] = True
                st.success("Thank you! Your feedback has been recorded.")
        else:
            st.info("Feedback for this answer was already submitted.")

        st.markdown("---")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def render_footer():
    st.markdown("---")
    st.markdown(
        """
<div style="font-size: 0.8rem; color: #888888;">
ECHS Assistant · Experimental tool · Always verify critical decisions with official ECHS instructions.<br/>
Built with ❤️ for veterans and families.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
#  MAIN
# ============================================================

def main():
    render_header()

    # Initialize once (cached)
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing ECHS Assistant (loading knowledge base)..."):
            try:
                client, index, df, embedding_model = initialize_system()
                st.session_state.client = client
                st.session_state.index = index
                st.session_state.df = df
                st.session_state.embedding_model = embedding_model
                st.session_state.initialized = True
                st.success("✅ Ready to answer your questions!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {e}")
                st.stop()

    # Question input + examples
    render_question_block()

    # If a question is pending (from Enter, button, or example click)
    if st.session_state.run_query and st.session_state.pending_question:
        q_text = st.session_state.pending_question

        # reset flags before processing
        st.session_state.run_query = False
        st.session_state.pending_question = ""

        # Store user message
        st.session_state.messages.append({"role": "user", "content": q_text})

        with st.spinner("🔍 Searching ECHS knowledge base... 🤖 Generating answer..."):
            answer, sources = get_answer(
                q_text,
                st.session_state.client,
                st.session_state.index,
                st.session_state.df,
                st.session_state.embedding_model,
            )

        # Store assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    # Show conversation + feedback
    render_conversation_and_feedback()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
