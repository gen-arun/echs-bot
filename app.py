"""
ECHS Q&A Bot - Streamlit Web App
Ex-Servicemen Contributory Health Scheme Assistant
Using precomputed FAISS embeddings for fast startup
"""

import streamlit as st
import json
from pathlib import Path
from groq import Groq
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import csv
from datetime import datetime

# ============================================================
#  CONFIG / CONSTANTS  (Lego: configuration block)
# ============================================================

INDEX_PATH = "index.faiss"
DOCS_META_PATH = "documents_metadata.json"
TRANSCRIPTS_META_PATH = Path("transcripts") / "metadata.json"
FEEDBACK_LOG_PATH = "feedback_log.csv"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Page configuration
st.set_page_config(
    page_title="ECHS Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
#  SESSION STATE INITIALIZATION  (Lego: state block)
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "run_suggested" not in st.session_state:
    st.session_state.run_suggested = False

# ============================================================
#  INITIALIZATION  (Lego: system bootstrap block)
# ============================================================

@st.cache_resource
def initialize_system():
    """Initialize the Q&A system with precomputed embeddings"""

    # ---- Groq client ----
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=groq_api_key)

    # ---- Check required files ----
    required_files = [INDEX_PATH, DOCS_META_PATH]
    for file in required_files:
        if not Path(file).exists():
            raise FileNotFoundError(
                f"⚠️ Knowledge base file '{file}' not found. Please contact administrator."
            )

    # ---- Load FAISS index ----
    index = faiss.read_index(INDEX_PATH)

    # ---- Load documents + metadatas ----
    with open(DOCS_META_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        documents = data["documents"]
        metadatas = data["metadatas"]

    # ---- Load embedding model (query only) ----
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # ---- Load original metadata (for reference / future use) ----
    if not TRANSCRIPTS_META_PATH.exists():
        raise FileNotFoundError(
            "⚠️ Video metadata not found. Please contact administrator."
        )

    with open(TRANSCRIPTS_META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return client, index, documents, metadatas, embedding_model, metadata

# ============================================================
#  CORE RETRIEVAL & LLM  (Lego: brain blocks)
# ============================================================

def retrieve_context(question, index, documents, metadatas, embedding_model, n_results=5):
    """Embed question and retrieve top-k relevant chunks from FAISS."""
    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, n_results)

    relevant_docs = [documents[i] for i in indices[0]]
    relevant_metadata = [metadatas[i] for i in indices[0]]

    context = "\n\n---\n\n".join(
        [
            f"From video: {meta['title']}\n{doc}"
            for doc, meta in zip(relevant_docs, relevant_metadata)
        ]
    )

    return context, relevant_metadata

def build_prompt(question, context):
    """Build the prompt for the LLM."""
    prompt = f"""You are an expert assistant helping people understand the Ex-Servicemen Contributory Health Scheme (ECHS). 

Based on the following information from ECHS training videos (including both spoken content and text from slides/visuals), please answer the user's question clearly and accurately.

CONTEXT FROM TRANSCRIPTS (Audio + Visual):
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Provide a clear, detailed answer based on the transcript information.
2. If the information isn't in the transcripts, say so clearly.
3. Include which video(s) the information came from when relevant.
4. IMPORTANT: Always end your answer with a "**Quick Summary:**" section that gives the key points in 2-3 short bullet points (use • for bullets).

Example format:
[Your detailed answer here...]

**Quick Summary:**
- Key point 1
- Key point 2
- Key point 3
"""
    return prompt

def call_llm(prompt, client):
    """Call Groq LLM and return the answer text or a friendly error."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL_NAME,
            temperature=0.3,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        error_msg = str(e).lower()

        if "timeout" in error_msg or "503" in error_msg:
            return "⏱️ The AI service is temporarily busy. Please try again in a moment."
        elif "rate limit" in error_msg or "429" in error_msg:
            return "⚠️ Too many requests right now. Please wait 30 seconds and try again."
        elif "connection" in error_msg:
            return "🔌 Connection issue. Please check your internet and try again."
        else:
            return "❌ Something went wrong. Please try rephrasing your question or contact support."

def get_answer(question, client, index, documents, metadatas, embedding_model, n_results=5):
    """Top-level helper that retrieves context and asks the LLM."""
    context, relevant_metadata = retrieve_context(
        question, index, documents, metadatas, embedding_model, n_results
    )
    prompt = build_prompt(question, context)
    answer = call_llm(prompt, client)

    # Build unique sources
    unique_sources = {}
    for meta in relevant_metadata:
        title = meta["title"]
        if title not in unique_sources:
            unique_sources[title] = meta["url"]

    return answer, unique_sources

# ============================================================
#  FEEDBACK LOGGER  (Lego: swappable I/O block)
# ============================================================

def log_feedback(question, answer, rating, comment):
    """
    Log feedback to a local CSV file.

    Later, this function can be swapped with a Google Sheets implementation
    without changing any UI code.
    """
    try:
        file_exists = Path(FEEDBACK_LOG_PATH).exists()
        with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "rating",
                        "comment",
                        "question",
                        "answer_snippet",
                    ]
                )
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    rating,
                    comment,
                    question,
                    (answer[:300] + "...") if answer else "",
                ]
            )
    except Exception:
        # Fail silently for user; you can add st.error for debugging if needed
        pass

# ============================================================
#  UI BLOCKS  (Lego: presentation layer)
# ============================================================

def render_header():
    st.markdown(
        '<h1 class="main-header">🏥 ECHS Assistant</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Ask questions about Ex-Servicemen Contributory Health Scheme</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "ℹ️ This assistant answers questions based on 5 official ECHS training videos. "
        "It may not cover every edge case. For specific situations or final decisions, "
        "please consult your ECHS polyclinic."
    )

def handle_example_click(label, question_text, key):
    """
    Helper to set question and trigger auto-run when example is clicked.
    """
    if st.button(label, key=key, use_container_width=True):
        st.session_state.question_input = question_text
        st.session_state.run_suggested = True
        st.rerun()

def render_question_input():
    """
    Render text input + example questions.

    Returns:
        question (str), should_run (bool)
    """
    question = st.text_input(
        "💬 Ask your question:",
        placeholder="e.g., What is ECHS? How do I register? What documents are required?",
        key="question_input",
    )

    with st.expander("📝 Browse questions by topic"):
        tab1, tab2, tab3 = st.tabs(
            ["Registration & Eligibility", "Emergency Care", "Referrals & Reimbursement"]
        )

        with tab1:
            st.markdown("**Registration & Eligibility Questions**")
            col1, col2 = st.columns(2)
            with col1:
                handle_example_click("What is ECHS?", "What is ECHS?", "q1")
                handle_example_click(
                    "Who is eligible for ECHS?",
                    "Who is eligible for ECHS?",
                    "q2",
                )
            with col2:
                handle_example_click(
                    "How do I register for ECHS?",
                    "How do I register for ECHS?",
                    "q3",
                )
                handle_example_click(
                    "What documents are required?",
                    "What documents are required for ECHS registration?",
                    "q4",
                )

        with tab2:
            st.markdown("**Emergency Care Questions**")
            col1, col2 = st.columns(2)
            with col1:
                handle_example_click(
                    "Can I get emergency treatment at AFMS hospital?",
                    "Can I get emergency treatment at AFMS hospital?",
                    "q5",
                )
                handle_example_click(
                    "What is the procedure for emergency treatment?",
                    "What is the procedure for emergency treatment in ECHS?",
                    "q6",
                )
            with col2:
                handle_example_click(
                    "Can my parents be treated at AFMS in emergency?",
                    "In an emergency, can my parents be treated at AFMS hospital on cashless basis?",
                    "q7",
                )
                handle_example_click(
                    "What about non-empanelled hospitals?",
                    "Can I get treatment in non-empanelled hospitals?",
                    "q8",
                )

        with tab3:
            st.markdown("**Referrals & Reimbursement Questions**")
            col1, col2 = st.columns(2)
            with col1:
                handle_example_click("What is HSR?", "What is HSR?", "q9")
                handle_example_click(
                    "Can Non-MIL Polyclinic refer to AFMS?",
                    "Can a Non MIL Polyclinic refer ESM to AFMS hospital?",
                    "q10",
                )
            with col2:
                handle_example_click(
                    "Can 26-year-old dependent be referred?",
                    "Can my dependent who is 26 years old be referred by the polyclinic to AFMS Hospital?",
                    "q11",
                )
                handle_example_click(
                    "How do I claim reimbursement?",
                    "How do I claim medical reimbursement in ECHS?",
                    "q12",
                )

    ask_button = st.button("🔍 Get Answer", use_container_width=True, type="primary")

    # New logic: run automatically if example question was clicked
    should_run = ask_button or st.session_state.get("run_suggested", False)

    return question, should_run

def render_answer_with_feedback():
    """
    Render conversation history with 5-star rating + optional comment box per answer.
    """
    if not st.session_state.messages:
        return

    st.markdown("---")
    st.markdown("### 💬 Conversation History")

    # Display in reverse order (latest first)
    for i in range(len(st.session_state.messages) - 1, -1, -2):
        if i >= 0 and st.session_state.messages[i]["role"] == "assistant":
            question_msg = st.session_state.messages[i - 1] if i > 0 else None
            answer_msg = st.session_state.messages[i]

            if question_msg:
                st.markdown(f"**❓ Question:** {question_msg['content']}")

            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(f"**💡 Answer:**\n\n{answer_msg['content']}")
            st.markdown("</div>", unsafe_allow_html=True)

            # ---- Feedback section ----
            fb_col1, fb_col2, fb_col3 = st.columns([2, 3, 7])

            with fb_col1:
                st.write("**Rate this answer:**")

            rating_key = f"rating_{i}"
            comment_key = f"comment_{i}"
            submitted_key = f"feedback_submitted_{i}"
            submit_btn_key = f"submit_feedback_{i}"

            with fb_col2:
                # 5-star rating (1–5)
                rating = st.slider(
                    "Stars",
                    min_value=1,
                    max_value=5,
                    value=5,
                    key=rating_key,
                    label_visibility="collapsed",
                )

            # Optional comment if rating <= 3
            comment = ""
            if rating <= 3:
                comment = st.text_area(
                    "What could be improved?",
                    key=comment_key,
                    placeholder="Optional: Please tell us what was unclear or missing.",
                )

            with fb_col3:
                submitted = st.session_state.get(submitted_key, False)
                if submitted:
                    st.success("Feedback recorded. Thank you! 🙏")
                else:
                    if st.button("Submit feedback", key=submit_btn_key):
                        q_text = question_msg["content"] if question_msg else ""
                        a_text = answer_msg["content"]
                        log_feedback(q_text, a_text, rating, comment)
                        st.session_state[submitted_key] = True
                        st.success("Feedback recorded. Thank you! 🙏")

            # Sources
            if "sources" in answer_msg and answer_msg["sources"]:
                st.markdown('<div class="source-box">', unsafe_allow_html=True)
                st.markdown("**📹 Information from these videos:**")
                for j, (title, url) in enumerate(
                    answer_msg["sources"].items(), start=1
                ):
                    st.markdown(f"{j}. [{title}]({url})")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

def render_footer():
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>📊 Currently covering 5 ECHS training videos</p>
        <p>✅ Includes both spoken content and text from slides/visuals</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">Powered by AI • Free to use • Beta Version</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============================================================
#  MAIN APP  (Lego: assembly block)
# ============================================================

def main():
    render_header()

    # ---- Initialize system (cached) ----
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing ECHS Assistant (loading knowledge base)..."):
            try:
                (
                    st.session_state.client,
                    st.session_state.index,
                    st.session_state.documents,
                    st.session_state.metadatas,
                    st.session_state.embedding_model,
                    st.session_state.metadata,
                ) = initialize_system()
                st.session_state.initialized = True
                st.success("✅ Ready to answer your questions!")
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                st.stop()

    # ---- Question input + example questions ----
    question, should_run = render_question_input()

    # Reset the run_suggested flag after reading it
    st.session_state.run_suggested = False

    # ---- Process question ----
    if should_run and question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🔍 Searching ECHS training videos... 🤖 Generating answer..."):
            answer, sources = get_answer(
                question,
                st.session_state.client,
                st.session_state.index,
                st.session_state.documents,
                st.session_state.metadatas,
                st.session_state.embedding_model,
            )

        # Add assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    # ---- Show conversation + feedback controls ----
    render_answer_with_feedback()

    # ---- Footer ----
    render_footer()

if __name__ == "__main__":
    main()
