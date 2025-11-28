import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq, BadRequestError

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

FAISS_INDEX_PATH = "echs_faiss.index"
SEGMENTS_PARQUET_PATH = "echs_segments_master.parquet"

TOP_K = 4  # how many chunks to retrieve for each question

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the Ex-Servicemen "
    "Contributory Health Scheme (ECHS). Use only the context provided from "
    "the training-video transcripts. If the answer is not in the context or "
    "is unclear, say that you are not sure and advise the user to check with "
    "their ECHS polyclinic or official ECHS instructions."
)

# Example questions grouped by topic (based on your 6 ECHS videos)
EXAMPLE_QUESTIONS: Dict[str, List[str]] = {
    "Basics of ECHS": [
        "What is ECHS and who is eligible?",
        "How do I register for ECHS?",
        "What documents are required for ECHS enrolment?",
    ],
    "Smart card and dependents": [
        "How do I apply for an ECHS smart card?",
        "How can I add or delete dependents in ECHS?",
        "What should I do if my ECHS smart card is lost?",
    ],
    "Polyclinics and hospitals": [
        "How do I take treatment at an ECHS polyclinic?",
        "What is the procedure for referral to empanelled hospitals?",
        "How are emergency cases handled under ECHS?",
    ],
    "Bills, issues and complaints": [
        "How are hospital bills settled under ECHS?",
        "What can I do if a hospital refuses cashless treatment?",
        "Where can I complain if I face problems with ECHS services?",
    ],
}


# -------------------------------------------------------------------
# GROQ CLIENT (LLM)
# -------------------------------------------------------------------

@st.cache_resource
def _create_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def get_groq_client() -> Groq | None:
    """
    Create and return a Groq client.
    If misconfigured, show a friendly error in the UI and return None.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        st.error(
            "The model backend is not configured yet (missing `GROQ_API_KEY`). "
            "Please contact the administrator."
        )
        return None

    try:
        return _create_groq_client(api_key)
    except Exception as e:
        st.error("Could not initialise connection to the Groq API.")
        if st.session_state.get("admin_mode", False):
            st.caption(f"Groq initialisation error: {e}")
        return None


# -------------------------------------------------------------------
# CACHED LOADERS – FAISS, SEGMENTS, EMBEDDING MODEL
# -------------------------------------------------------------------

@st.cache_resource
def load_faiss_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index file not found: {path}")
    return faiss.read_index(path)


@st.cache_data
def load_segments_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Segments parquet file not found: {path}")
    df = pd.read_parquet(path)
    return df


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    # IMPORTANT: use the SAME model that was used to build echs_faiss.index
    model_name = os.getenv("ECHS_EMBED_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


# -------------------------------------------------------------------
# EMBEDDINGS + LLM CALL
# -------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """Return a 1D embedding vector for the given text."""
    model = load_embedding_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype="float32")


def call_llm_with_context(context: str, question: str) -> str:
    """
    Call Groq chat model with context and question.
    Any API error is caught and turned into a friendly message so that
    the Streamlit UI never shows a long red traceback.
    """
    client = get_groq_client()
    if client is None:
        # get_groq_client already showed a message
        return (
            "The model service is not configured yet, so I cannot generate an answer. "
            "Please contact the administrator."
        )

    model_name = os.getenv("ECHS_CHAT_MODEL", "llama3-8b-8192")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Context from ECHS training materials:\n\n"
                f"{context}\n\n"
                f"User question: {question}"
            ),
        },
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
        )
        return (completion.choices[0].message.content or "").strip()
    except BadRequestError:
        # Configuration or request issue – explain briefly, no traceback.
        st.warning(
            "The model could not process this request (bad request to Groq API). "
            "This is usually due to an incorrect model name or account limits. "
            "Please contact the administrator to check the model configuration."
        )
        return (
            "I’m sorry, but I could not generate an answer because the model "
            "service rejected the request. Please try a shorter question or "
            "ask your ECHS polyclinic / administrator."
        )
    except Exception as e:
        st.error("Unexpected error while contacting the model service.")
        if st.session_state.get("admin_mode", False):
            st.caption(f"Groq runtime error: {e}")
        return (
            "I’m sorry, something went wrong while contacting the model service. "
            "Please try again later."
        )


# -------------------------------------------------------------------
# RETRIEVAL HELPERS
# -------------------------------------------------------------------

def search_segments(
    query: str,
    index,
    segments_df: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Embed query, search FAISS, and return top_k rows with distances."""
    q_vec = embed_text(query).reshape(1, -1)
    distances, indices = index.search(q_vec, top_k)

    rows = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(segments_df):
            continue
        row = segments_df.iloc[idx].copy()
        row["_distance"] = float(dist)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def format_timestamp(seconds_value: Any) -> str:
    """Convert seconds to mm:ss string (best-effort)."""
    try:
        sec = int(seconds_value)
    except Exception:
        return ""
    minutes = sec // 60
    seconds = sec % 60
    return f"{minutes:02d}:{seconds:02d}"


def build_context_from_rows(rows: pd.DataFrame) -> str:
    """Build a single context string from retrieved segments."""
    if rows.empty:
        return ""

    parts: List[str] = []
    for _, r in rows.iterrows():
        text = (
            r.get("text")
            or r.get("segment_text")
            or r.get("chunk")
            or ""
        )
        title = r.get("video_title") or r.get("source_title") or "ECHS training video"
        ts = format_timestamp(r.get("start_time", r.get("start_sec", 0)))
        prefix = f"[Source: {title}"
        if ts:
            prefix += f" @ {ts}"
        prefix += "] "
        parts.append(prefix + str(text))

    return "\n\n".join(parts)


# -------------------------------------------------------------------
# UI HELPERS – EXAMPLES, SOURCES, ADMIN
# -------------------------------------------------------------------

def render_example_questions_panel():
    """Show example questions by topic as clickable buttons."""
    if not EXAMPLE_QUESTIONS:
        return

    with st.expander("📌 Browse example questions by topic"):
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


def render_sources_for_user(rows: pd.DataFrame):
    """Show 'Sources used' section under the answer."""
    if rows.empty:
        return

    st.markdown("###### Sources used from ECHS training videos")
    for _, r in rows.iterrows():
        title = r.get("video_title") or r.get("source_title") or "ECHS training video"
        ts = format_timestamp(r.get("start_time", r.get("start_sec", 0)))
        text = (
            r.get("text")
            or r.get("segment_text")
            or r.get("chunk")
            or ""
        )

        bullet = f"- **{title}**"
        if ts:
            bullet += f" at `{ts}`"
        st.markdown(bullet)
        if text:
            snippet = text[:300] + ("..." if len(text) > 300 else "")
            st.markdown(f"> {snippet}")


def render_admin_debug_table(rows: pd.DataFrame):
    """Admin debug: show retrieved rows + distances."""
    if rows.empty:
        st.info("No segments retrieved.")
        return

    st.markdown("### 🔧 Admin: retrieved segments and distances")
    debug_cols = []
    for col in ["video_title", "start_time", "text", "segment_text", "chunk", "_distance"]:
        if col in rows.columns:
            debug_cols.append(col)

    st.dataframe(rows[debug_cols], use_container_width=True, hide_index=True)


# -------------------------------------------------------------------
# CORE QA PIPELINE
# -------------------------------------------------------------------

def answer_question(
    question: str,
    faiss_index,
    segments_df: pd.DataFrame,
) -> Tuple[str, Dict[str, Any]]:
    """Retrieve relevant segments and ask the LLM."""
    retrieved = search_segments(question, faiss_index, segments_df, TOP_K)
    context = build_context_from_rows(retrieved)

    if not context:
        answer = (
            "I could not find relevant information about this in the available ECHS "
            "training materials. Please confirm with your ECHS polyclinic or official "
            "ECHS instructions."
        )
        return answer, {"retrieved": retrieved, "context": context}

    answer = call_llm_with_context(context, question)
    return answer, {"retrieved": retrieved, "context": context}


# -------------------------------------------------------------------
# MAIN QUESTION BLOCK
# -------------------------------------------------------------------

def render_question_block():
    """Main interaction area: example buttons + chat input + answer."""
    render_example_questions_panel()

    # Decide where the question comes from
    if st.session_state.get("from_example"):
        question = st.session_state.get("prefill_question", "")
        st.session_state["from_example"] = False
    else:
        question = st.chat_input("Ask your question:")

    if not question:
        return

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching ECHS training material and preparing answer..."):
            answer, dbg = answer_question(
                question,
                st.session_state["faiss_index"],
                st.session_state["segments_df"],
            )
            st.markdown(answer)
            retrieved_rows = dbg.get("retrieved", pd.DataFrame())
            render_sources_for_user(retrieved_rows)

    # Admin debug view
    if st.session_state.get("admin_mode", False):
        render_admin_debug_table(dbg.get("retrieved", pd.DataFrame()))


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="ECHS Assistant",
        page_icon="🩺",
        layout="wide",
    )

    st.markdown("## Ask questions about Ex-Servicemen Contributory Health Scheme")

    st.info(
        "This assistant uses a limited set of ECHS training videos and transcripts. "
        "It may not cover every edge case. For specific situations or final decisions, "
        "please consult your ECHS polyclinic or official ECHS instructions."
    )

    # Sidebar – admin toggle
    with st.sidebar:
        st.header("Settings")
        st.session_state["admin_mode"] = st.checkbox("🛠 Admin debug mode", False)

        st.markdown("---")
        st.markdown("**Data files**")
        st.text(f"Index: {FAISS_INDEX_PATH}")
        st.text(f"Segments: {SEGMENTS_PARQUET_PATH}")

    # Load FAISS + segments (cached)
    try:
        faiss_index = load_faiss_index(FAISS_INDEX_PATH)
        segments_df = load_segments_df(SEGMENTS_PARQUET_PATH)
    except Exception as e:
        st.error(
            "Error while loading the knowledge base.\n\n"
            "Check that `echs_faiss.index` and `echs_segments_master.parquet` "
            "are present in the app directory."
        )
        st.exception(e)
        return

    st.session_state["faiss_index"] = faiss_index
    st.session_state["segments_df"] = segments_df

    # Main Q&A block
    render_question_block()


if __name__ == "__main__":
    main()
