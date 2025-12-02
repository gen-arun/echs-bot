"""
ECHS Assistant – Streamlit App

- Uses precomputed FAISS index over ECHS video segments
- SentenceTransformers for query embeddings
- Groq LLM (llama-3.3-70b-versatile) for answers
- Google Apps Script for feedback logging
- Video/time metadata parsed from segment headers:
  VIDEO ID   : <id>
  SEGMENT    : <n>
  TIME       : <start_sec> → <end_sec>
"""

from pathlib import Path
from datetime import datetime
import json
import re
import os

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

# Files (in same folder as app.py)
INDEX_PATH = Path("echs_faiss.index")
PARQUET_PATH = Path("echs_segments_master.parquet")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Google Apps Script endpoint for feedback logging
GOOGLE_APPS_SCRIPT_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbznEPzlIYwDDZWtXAoIJdUS2NdBaSVW11P0K9BshGlu2u-eHcVxdJhhE2cUm_4v7Vo4fg/exec"
)

# Video metadata mapping (from your table)
VIDEO_CATALOG = {
    "PptemekIV4s": {
        "title": "Reimbursement approval using High Power Committee (HPC)?",
        "url": "https://youtu.be/PptemekIV4s",
    },
    "Lul0rNeu-ys": {
        "title": "Purchase & reimbursement of NA medicines from market",
        "url": "https://youtube.com/shorts/Lul0rNeu-ys",
    },
    "WBo3Nles5ME": {
        "title": "How to find reimbursement claim status & meaning of status messages",
        "url": "https://youtu.be/WBo3Nles5ME",
    },
    "_A2FHFov-1s": {
        "title": "Reimbursement claims cannot be rejected at any level. Decision is with CO ECHS",
        "url": "https://youtube.com/shorts/_A2FHFov-1s",
    },
    "K0LmnRMxR3E": {
        "title": "How to – Non Availability (NA) certificate & reimbursement financial limit",
        "url": "https://youtu.be/K0LmnRMxR3E",
    },
    "TALESjqESn8": {
        "title": "Medicines reimbursement steps made easy",
        "url": "https://youtu.be/TALESjqESn8",
    },
}

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
# Custom CSS
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
    # list of dicts: {role: "user"|"assistant", content: str, sources: list}
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

if "run_query" not in st.session_state:
    st.session_state.run_query = False

if "run_suggested" not in st.session_state:
    st.session_state.run_suggested = False

if "segment_only" not in st.session_state:
    st.session_state.segment_only = False

if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False

# ============================================================
#  UTILS – METADATA & FORMATTING
# ============================================================


def parse_segment_header(text: str, source_file: str):
    """
    Parse VIDEO ID, SEGMENT, TIME from the header embedded in text.

    Returns dict:
    {
      "video_id", "segment", "start_sec", "end_sec",
      "body_text"
    }
    """
    video_id = None
    segment = None
    start_sec = None
    end_sec = None

    # Video ID
    m_vid = re.search(r"VIDEO ID\s*:\s*([A-Za-z0-9_-]+)", text)
    if m_vid:
        video_id = m_vid.group(1).strip()

    # Segment
    m_seg = re.search(r"SEGMENT\s*:\s*([0-9]+)", text)
    if m_seg:
        segment = int(m_seg.group(1))

    # Time in seconds like "0.0 → 13.84"
    m_time = re.search(r"TIME\s*:\s*([0-9.]+)\s*[^\d]+([0-9.]+)", text)
    if m_time:
        try:
            start_sec = float(m_time.group(1))
            end_sec = float(m_time.group(2))
        except ValueError:
            start_sec, end_sec = None, None

    # Fallback from filename if needed
    if video_id is None or segment is None:
        m_fn = re.match(r"([A-Za-z0-9_-]+)_([0-9]+)\.txt", source_file)
        if m_fn:
            if video_id is None:
                video_id = m_fn.group(1)
            if segment is None:
                segment = int(m_fn.group(2))

    # Body text: after WHISPER TRANSCRIPT marker if present
    body = text
    marker = "=============== WHISPER TRANSCRIPT ==============="
    if marker in text:
        body = text.split(marker, 1)[1].strip()

    return {
        "video_id": video_id,
        "segment": segment,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "body_text": body,
    }


def format_seconds(sec: float | None):
    """Convert seconds (float) to mm:ss or hh:mm:ss string."""
    if sec is None:
        return "unknown"
    s = int(sec)
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s2:02d}"
    return f"{m:02d}:{s2:02d}"


def get_video_info(video_id: str):
    """Return (title, base_url) for a given video_id."""
    if not video_id:
        return "ECHS Training Video", ""
    if video_id in VIDEO_CATALOG:
        return VIDEO_CATALOG[video_id]["title"], VIDEO_CATALOG[video_id]["url"]
    # Default fallback
    return f"ECHS Training Video ({video_id})", f"https://youtu.be/{video_id}"


def add_timestamp_to_url(url: str, start_sec: float | None):
    """Append ?t=seconds (or &t=) to YouTube URL."""
    if not url:
        return ""
    if start_sec is None:
        return url
    t = int(start_sec)
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={t}s"


def is_question_unclear(q: str) -> bool:
    """Heuristic check for very short / vague questions."""
    q = (q or "").strip()
    if not q:
        return True
    words = q.split()
    if len(words) <= 2:
        return True
    # Very short non-question phrases
    if len(q) < 10 and not q.lower().startswith(
        ("what", "how", "who", "why", "when", "where", "can", "is", "are")
    ):
        return True
    return False


def build_clarification_response(q: str) -> str:
    """Message shown when the question is too sketchy."""
    return f"""
Your question **\"{q}\"** is a bit short or unclear, so I'm not sure what exactly you need.

To help me give a precise answer, please:
- Mention **who** it is about (veteran, spouse, dependent, parent).
- Say whether it is about **registration, emergency care, referral, or reimbursement**.
- Add key details like *empanelled / non-empanelled hospital*, *age of dependent*, *emergency vs routine*, etc.

You can also try one of these more specific questions:
- "How do I register for ECHS?"
- "What is the procedure for emergency treatment in ECHS?"
- "How do I claim reimbursement for treatment in a non-empanelled hospital?"
- "Can my parents get emergency treatment at AFMS hospital on cashless basis?"
"""

# ============================================================
#  INITIALIZATION – KB + LLM
# ============================================================


@st.cache_resource
def initialize_system():
    """
    Initialize:
    - Groq client
    - FAISS index
    - Segments dataframe
    - Embedding model
    """
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        raise RuntimeError(
            "GROQ_API_KEY not found in Streamlit secrets. Please configure it."
        )

    client = Groq(api_key=groq_api_key)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base index file not found: {INDEX_PATH}. Please upload it."
        )
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base data file not found: {PARQUET_PATH}. Please upload it."
        )

    index = faiss.read_index(str(INDEX_PATH))
    df = pd.read_parquet(PARQUET_PATH)

    # Prefer local SentenceTransformer path if provided
    local_model_path = None
    try:
        # Try Streamlit secrets first
        local_model_path = st.secrets.get("LOCAL_SBERT_PATH", None)
    except Exception:
        local_model_path = None

    if not local_model_path:
        # Fallback to environment variable if secrets not set
        local_model_path = os.getenv("LOCAL_SBERT_PATH")

    if local_model_path and os.path.isdir(local_model_path):
        embedding_model = SentenceTransformer(local_model_path)
    else:
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

    context_parts = []
    hits = []

    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]
        meta = parse_segment_header(row["text"], row["source_file"])

        video_id = meta["video_id"]
        segment = meta["segment"]
        start_sec = meta["start_sec"]
        end_sec = meta["end_sec"]
        body_text = meta["body_text"]

        title, base_url = get_video_info(video_id)
        url_with_t = add_timestamp_to_url(base_url, start_sec)

        start_str = format_seconds(start_sec)
        end_str = format_seconds(end_sec)

        # Build context chunk for LLM
        header = f"Video: {title}\nTime: {start_str} → {end_str}"
        chunk = f"{header}\n{body_text}"
        context_parts.append(chunk)

        hits.append(
            {
                "video_id": video_id,
                "video_title": title,
                "video_url": url_with_t,
                "base_url": base_url,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_str": start_str,
                "end_str": end_str,
                "segment_id": int(row["segment_id"]),
                "source_file": row["source_file"],
                "distance": float(dist),
                "body_text": body_text,
            }
        )

    context = "\n\n---\n\n".join(context_parts)
    return context, hits


def build_prompt(question: str, context: str) -> str:
    """Build the prompt for the LLM."""
    prompt = f"""You are an expert assistant helping people understand the Ex-Servicemen Contributory Health Scheme (ECHS).

You must answer based ONLY on the information given below from ECHS training videos and transcripts.
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
    context, hits = retrieve_context(question, index, df, embedding_model, n_results)
    prompt = build_prompt(question, context)
    answer = call_llm(prompt, client)
    return answer, hits


def build_transcript_only_answer(question: str, hits, max_segments: int = 3):
    """
    Transcript-only response (no AI).
    Handles:
    - no hits
    - hits with no usable body text
    - normal case with readable transcript text
    Also suggests using "Answer with AI" when transcript-only is weak.
    """
    # Normalize question for simple keyword check
    q = (question or "").strip().lower()

    # Case 1: no hits at all
    if not hits:
        return (
            "### 📄 Transcript-only response (no AI used)\n\n"
            "**No transcript matches were found for this question.**\n"
            "You may try **Answer with AI** for some possible leads or rephrase your question."
        )

    # Limit to top-k hits
    limited_hits = hits[:max_segments]

    # Extract body_text and do a simple relevance heuristic
    tokens = [w for w in q.split() if len(w) >= 3]
    non_empty_segments = []
    any_token_match = False

    for h in limited_hits:
        body = str(h.get("body_text", "") or "").strip()
        if body:
            non_empty_segments.append((h, body))
            body_lower = body.lower()
            if tokens and any(t in body_lower for t in tokens):
                any_token_match = True

    out = ["### 📄 Transcript-only response (no AI used)\n"]

    # Case 2: hits exist but no good body text, or no token overlap with question
    if not non_empty_segments or (tokens and not any_token_match):
        out.append(
            "I found some video segments that might be loosely related, "
            "but I couldn't extract clear transcript text that directly answers this question.\n"
            "You may try **Answer with AI** for some possible leads or rephrase your question.\n"
        )
        out.append("")  # blank line

        for h in limited_hits:
            out.append(
                f"**{h['video_title']}**  \n"
                f"⏱ {h['start_str']} → {h['end_str']}  \n"
                f"🔗 [Watch this part]({h['video_url']})\n---\n"
            )

        return "\n".join(out)

    # Case 3: normal transcript-only display with body text
    for h, body in non_empty_segments:
        out.append(
            f"**{h['video_title']}**  \n"
            f"⏱ {h['start_str']} → {h['end_str']}  \n"
            f"🔗 [Watch this part]({h['video_url']})\n\n"
            f"{body}\n---\n"
        )

    return "\n".join(out)



def log_feedback(question: str, answer: str, rating: int, comment: str):
    """
    Send feedback row to Google Apps Script, which writes to Google Sheet.
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": "",
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
        st.warning(f"Could not send feedback to log: {e}")


# ============================================================
#  UI HELPERS
# ============================================================


def submit_question():
    """Standard LLM-answer handler"""
    q = st.session_state.get("question_input", "").strip()
    if q:
        st.session_state.pending_question = q
        st.session_state.run_query = True
        st.session_state.segment_only = False


def submit_segment_only_question():
    """Transcript-only answer handler"""
    q = st.session_state.get("question_input", "").strip()
    if q:
        st.session_state.pending_question = q
        st.session_state.run_query = True
        st.session_state.segment_only = True


def handle_example_click(text: str):
    """When a suggested question is clicked."""
    # IMPORTANT: do NOT modify question_input widget value here.
    # Just mark the question to be run; main loop will treat it
    # exactly like a typed question.
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
        placeholder="e.g., How do I claim medicines reimbursement? What is an NA certificate?",
        key="question_input",
        on_change=submit_question,  # Enter key
    )

    # Suggested questions – updated to latest reimbursement videos
    with st.expander("📝 Browse example questions by latest videos"):
        tab1, tab2, tab3 = st.tabs(
            [
                "NA medicines & NA certificate",
                "HPC / CO ECHS & approval",
                "Claim status & steps",
            ]
        )

        # ---------------- Tab 1: NA medicines & NA certificate ----------------
        with tab1:
            st.markdown("**Non-Availability (NA) medicines and NA certificate**")
            col1, col2 = st.columns(2)

            # Mode selection buttons (AI vs Transcript-only)
    col1, col2 = st.columns(2)

    with col1:
        st.button(
            "🧠 Answer with AI",
            type="primary",
            use_container_width=True,
            on_click=submit_question,
        )

    with col2:
        st.button(
            "📄 Transcript Only",
            use_container_width=True,
            on_click=submit_segment_only_question,
        )
def render_conversation_and_feedback():
    """Show conversation history and feedback controls."""
    if not st.session_state.messages:
        return

    st.markdown("---")
    st.markdown("### 💬 Conversation History")

    admin_mode = st.session_state.admin_mode

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

        sources = answer_msg.get("sources", [])

        if sources:
            st.markdown('<div class="source-box">', unsafe_allow_html=True)
            st.markdown("**📄 Information verified from these ECHS videos:**")
            for j, hit in enumerate(sources, start=1):
                title = hit["video_title"]
                url = hit["video_url"]
                start_str = hit["start_str"]
                end_str = hit["end_str"]
                video_id = hit["video_id"]
                st.markdown(
                    f"{j}. **{title}**  \n"
                    f"   ⏱ {start_str} → {end_str}  \n"
                    f"   🔗 [Watch this part]({url})"
                )
                if admin_mode:
                    st.markdown(
                        f"   🛠 `id={video_id}` | segment={hit['segment_id']} | "
                        f"file={hit['source_file']} | faiss={hit['distance']:.3f}"
                    )
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

    # Sidebar admin toggle
    st.session_state.admin_mode = st.sidebar.checkbox(
        "🛠 Admin / debug mode", value=st.session_state.admin_mode
    )

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

        # Unified clarification logic: applies to BOTH AI and transcript-only
        if is_question_unclear(q_text):
            clarification = build_clarification_response(q_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": clarification, "sources": []}
            )
        else:
            if st.session_state.segment_only:
                # Transcript-only mode: no LLM call
                with st.spinner("🔎 Retrieving transcript only (no AI)..."):
                    _, hits = retrieve_context(
                        q_text,
                        st.session_state.index,
                        st.session_state.df,
                        st.session_state.embedding_model,
                    )
                answer = build_transcript_only_answer(q_text, hits)
                st.session_state.segment_only = False
            else:
                # Normal LLM mode
                with st.spinner("🔍 Searching ECHS knowledge base... 🤖 Generating answer..."):
                    answer, hits = get_answer(
                        q_text,
                        st.session_state.client,
                        st.session_state.index,
                        st.session_state.df,
                        st.session_state.embedding_model,
                    )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": hits}
            )

            # PATCH6/9: Prepend 'Sources used: N' to AI answers only (not transcript-only)
            if hits and isinstance(answer, str) and "Transcript-only response (no AI used)" not in answer:
                src_count = len(hits)
                st.session_state.messages[-1]['content'] = f"**📄 Sources used: {src_count}**\n\n" + st.session_state.messages[-1]['content']


    # Show conversation + feedback
    render_conversation_and_feedback()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
# --- PATCH TEST SUCCESSFUL ---


# --- PATCH TEST SUCCESSFUL ---
