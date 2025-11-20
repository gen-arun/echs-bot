"""
ECHS Q&A Bot - Streamlit Web App
Ex-Servicemen Contributory Health Scheme Assistant
Using precomputed FAISS embeddings for fast startup
"""

import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import faiss
import requests
from groq import Groq
from sentence_transformers import SentenceTransformer

# ========= CONFIG ========= #

# Your deployed Google Apps Script Web App URL
GOOGLE_APPS_SCRIPT_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbznEPzlIYwDDZWtXAoIJdUS2NdBaSVW11P0K9BshGlu2u-eHcVxdJhhE2cUm_4v7Vo4fg/exec"
)

# Page configuration
st.set_page_config(
    page_title="ECHS Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ========= STYLES ========= #

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

# ========= SESSION STATE ========= #

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources?}

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# For unified submission logic (button, Enter, example-click)
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

if "run_query" not in st.session_state:
    st.session_state.run_query = False


# ========= CORE INITIALIZATION ========= #

@st.cache_resource
def initialize_system():
    """Initialize the Q&A system with precomputed embeddings."""
    try:
        # Initialize Groq client
        groq_api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=groq_api_key)

        # Check if precomputed files exist
        required_files = ["index.faiss", "documents_metadata.json"]
        for file in required_files:
            if not Path(file).exists():
                raise FileNotFoundError(
                    f"⚠️ Knowledge base file '{file}' not found. "
                    f"Please contact administrator."
                )

        # Load precomputed FAISS index
        index = faiss.read_index("index.faiss")

        # Load documents and metadata
        with open("documents_metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            documents = data["documents"]
            metadatas = data["metadatas"]

        # Load embedding model (for query encoding only)
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load original video metadata (not strictly needed for Q&A)
        transcripts_path = Path("transcripts")
        metadata_file = transcripts_path / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(
                "⚠️ Video metadata not found. Please contact administrator."
            )

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return client, index, documents, metadatas, embedding_model, metadata

    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.stop()


# ========= ANSWER GENERATION ========= #

def get_answer(
    question,
    client,
    index,
    documents,
    metadatas,
    embedding_model,
    n_results=5,
):
    """Get answer for a question using FAISS + Groq."""
    try:
        # Create query embedding
        query_embedding = embedding_model.encode([question])
        query_embedding = np.array(query_embedding).astype("float32")

        # Search in FAISS
        distances, indices = index.search(query_embedding, n_results)

        # Get relevant documents
        relevant_docs = [documents[i] for i in indices[0]]
        relevant_metadata = [metadatas[i] for i in indices[0]]

        # Create context
        context = "\n\n---\n\n".join(
            [
                f"From video: {meta['title']}\n{doc}"
                for doc, meta in zip(relevant_docs, relevant_metadata)
            ]
        )

        # Prompt with quick-summary requirement
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
- Key point 3"""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
        )

        answer = chat_completion.choices[0].message.content

        # Unique sources
        unique_sources = {}
        for meta in relevant_metadata:
            title = meta["title"]
            if title not in unique_sources:
                unique_sources[title] = meta["url"]

        return answer, unique_sources

    except Exception as e:
        error_msg = str(e).lower()

        if "timeout" in error_msg or "503" in error_msg:
            return (
                "⏱️ The AI service is temporarily busy. Please try again in a moment.",
                {},
            )
        elif "rate limit" in error_msg or "429" in error_msg:
            return (
                "⚠️ Too many requests right now. Please wait 30 seconds and try again.",
                {},
            )
        elif "connection" in error_msg:
            return (
                "🔌 Connection issue. Please check your internet and try again.",
                {},
            )
        else:
            return (
                "❌ Something went wrong. Please try rephrasing your question or contact support.",
                {},
            )


# ========= GOOGLE SHEET LOGGING ========= #

def log_feedback(question, answer, rating, comment):
    """Send feedback row to Google Apps Script, which writes to Google Sheet."""
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
        # Optional: check response text if your script returns something
        if response.status_code != 200:
            st.warning(
                f"Feedback not confirmed by server (status {response.status_code})."
            )
    except Exception as e:
        st.warning(f"Could not send feedback to log: {e}")


# ========= SUBMISSION HELPERS ========= #

def submit_question():
    """Common handler for Enter key + Get Answer button."""
    q = st.session_state.get("question_input", "").strip()
    if q:
        st.session_state.pending_question = q
        st.session_state.run_query = True


def handle_example_click(question_text: str):
    """When a suggested question button is clicked."""
    st.session_state.question_input = question_text
    st.session_state.pending_question = question_text
    st.session_state.run_query = True


# ========= MAIN APP ========= #

def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🏥 ECHS Assistant</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Ask questions about Ex-Servicemen Contributory Health Scheme</p>',
        unsafe_allow_html=True,
    )

    # Disclaimer
    st.info(
        "ℹ️ This assistant answers questions based on 5 official ECHS training videos. "
        "It may not cover every edge case. For specific situations or final decisions, "
        "please consult your ECHS polyclinic."
    )

    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing ECHS Assistant (loading knowledge base)..."):
            result = initialize_system()
            st.session_state.client = result[0]
            st.session_state.index = result[1]
            st.session_state.documents = result[2]
            st.session_state.metadatas = result[3]
            st.session_state.embedding_model = result[4]
            st.session_state.metadata = result[5]
            st.session_state.initialized = True
            st.success("✅ Ready to answer your questions!")

    # ---- Question input ---- #

    question = st.text_input(
        "💬 Ask your question:",
        placeholder="e.g., What is ECHS? How do I register? What documents are required?",
        key="question_input",
        on_change=submit_question,  # ENTER submits
    )

    # Example questions organized by topic
    with st.expander("📝 Browse questions by topic"):
        tab1, tab2, tab3 = st.tabs(
            ["Registration & Eligibility", "Emergency Care", "Referrals & Reimbursement"]
        )

        with tab1:
            st.markdown("**Registration & Eligibility Questions**")
            col1, col2 = st.columns(2)

            with col1:
                st.button(
                    "What is ECHS?",
                    key="q1",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("What is ECHS?",),
                )
                st.button(
                    "Who is eligible for ECHS?",
                    key="q2",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("Who is eligible for ECHS?",),
                )

            with col2:
                st.button(
                    "How do I register for ECHS?",
                    key="q3",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("How do I register for ECHS?",),
                )
                st.button(
                    "What documents are required?",
                    key="q4",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("What documents are required for ECHS registration?",),
                )

        with tab2:
            st.markdown("**Emergency Care Questions**")
            col1, col2 = st.columns(2)

            with col1:
                st.button(
                    "Can I get emergency treatment at AFMS hospital?",
                    key="q5",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("Can I get emergency treatment at AFMS hospital?",),
                )
                st.button(
                    "What is the procedure for emergency treatment?",
                    key="q6",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("What is the procedure for emergency treatment in ECHS?",),
                )

            with col2:
                st.button(
                    "Can my parents be treated at AFMS in emergency?",
                    key="q7",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=(
                        "In an emergency, can my parents be treated at AFMS hospital on cashless basis?",
                    ),
                )
                st.button(
                    "What about non-empanelled hospitals?",
                    key="q8",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("Can I get treatment in non-empanelled hospitals?",),
                )

        with tab3:
            st.markdown("**Referrals & Reimbursement Questions**")
            col1, col2 = st.columns(2)

            with col1:
                st.button(
                    "What is HSR?",
                    key="q9",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("What is HSR?",),
                )
                st.button(
                    "Can Non-MIL Polyclinic refer to AFMS?",
                    key="q10",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("Can a Non MIL Polyclinic refer ESM to AFMS hospital?",),
                )

            with col2:
                st.button(
                    "Can 26-year-old dependent be referred?",
                    key="q11",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=(
                        "Can my dependent who is 26 years old be referred by the polyclinic to AFMS Hospital?",
                    ),
                )
                st.button(
                    "How do I claim reimbursement?",
                    key="q12",
                    use_container_width=True,
                    on_click=handle_example_click,
                    args=("How do I claim medical reimbursement in ECHS?",),
                )

    # Ask button (also uses same submit logic)
    st.button(
        "🔍 Get Answer",
        use_container_width=True,
        type="primary",
        on_click=submit_question,
    )

    # ---- Process a pending question (from Enter, button, or example click) ---- #

    if st.session_state.run_query and st.session_state.pending_question:
        q_text = st.session_state.pending_question

        # reset flags first to avoid duplicate runs if something inside reruns
        st.session_state.run_query = False
        st.session_state.pending_question = ""

        # Store user message
        st.session_state.messages.append({"role": "user", "content": q_text})

        with st.spinner("🔍 Searching ECHS training videos... 🤖 Generating answer..."):
            answer, sources = get_answer(
                q_text,
                st.session_state.client,
                st.session_state.index,
                st.session_state.documents,
                st.session_state.metadatas,
                st.session_state.embedding_model,
            )

        # Store assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    # ---- Conversation history + rating widgets ---- #

    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 💬 Conversation History")

        # Display in reverse order (latest first)
        for i in range(len(st.session_state.messages) - 1, -1, -2):
            if (
                i >= 0
                and st.session_state.messages[i]["role"] == "assistant"
            ):
                answer_msg = st.session_state.messages[i]
                question_msg = (
                    st.session_state.messages[i - 1] if i > 0 else None
                )

                question_text = (
                    question_msg["content"] if question_msg else ""
                )
                answer_text = answer_msg["content"]

                if question_msg:
                    st.markdown(f"**❓ Question:** {question_text}")

                st.markdown(
                    '<div class="answer-box">',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**💡 Answer:**\n\n{answer_text}")
                st.markdown("</div>", unsafe_allow_html=True)

                # ---- Rating & feedback ---- #
                rating_key = f"rating_{i}"
                comment_key = f"comment_{i}"
                submitted_key = f"feedback_submitted_{i}"

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

                if st.button(
                    "💾 Submit feedback",
                    key=f"submit_feedback_{i}",
                    use_container_width=False,
                ):
                    if not st.session_state.get(submitted_key, False):
                        log_feedback(
                            question_text,
                            answer_text,
                            rating,
                            comment,
                        )
                        st.session_state[submitted_key] = True
                        st.success("Thank you! Your feedback has been recorded.")
                    else:
                        st.info("Feedback for this answer was already submitted.")

                # Sources
                if "sources" in answer_msg and answer_msg["sources"]:
                    st.markdown(
                        '<div class="source-box">',
                        unsafe_allow_html=True,
                    )
                    st.markdown("**📹 Information from these videos:**")
                    for j, (title, url) in enumerate(
                        answer_msg["sources"].items(), 1
                    ):
                        st.markdown(f"{j}. [{title}]({url})")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Footer
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


if __name__ == "__main__":
    main()
