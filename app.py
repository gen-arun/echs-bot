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

# Page configuration
st.set_page_config(
    page_title="ECHS Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False


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
                    "Please contact the administrator."
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

        # Load original metadata for videos
        transcripts_path = Path("transcripts")
        metadata_file = transcripts_path / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(
                "⚠️ Video metadata not found. Please contact the administrator."
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


def get_answer(
    question,
    client,
    index,
    documents,
    metadatas,
    embedding_model,
    n_results: int = 5,
):
    """Get answer for a question."""

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

        # Create prompt with summary instruction
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
4. IMPORTANT: Always end your answer with a "**Quick Summary:**" section that gives the key points in 2–3 short bullet points (use • for bullets).

Example format:
[Your detailed answer here...]

**Quick Summary:**
- Key point 1
- Key point 2
- Key point 3"""

        # Get answer from Groq
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
        )

        answer = chat_completion.choices[0].message.content

        # Get unique sources
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
                "🔌 Connection issue. Please check your internet connection and try again.",
                {},
            )
        else:
            return (
                "❌ Something went wrong. Please try rephrasing your question or contact support.",
                {},
            )


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
        with st.spinner(
            "🔄 Initializing ECHS Assistant (loading knowledge base)..."
        ):
            try:
                result = initialize_system()
                (
                    st.session_state.client,
                    st.session_state.index,
                    st.session_state.documents,
                    st.session_state.metadatas,
                    st.session_state.embedding_model,
                    st.session_state.metadata,
                ) = result
                st.session_state.initialized = True
                st.success("✅ Ready to answer your questions!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                st.stop()

    # Text input
    question = st.text_input(
        "💬 Ask your question:",
        placeholder="e.g., What is ECHS? How do I register? What documents are required?",
        key="question_input",
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
                if st.button("What is ECHS?", key="q1", use_container_width=True):
                    st.session_state.question_input = "What is ECHS?"
                    st.rerun()
                if st.button(
                    "Who is eligible for ECHS?",
                    key="q2",
                    use_container_width=True,
                ):
                    st.session_state.question_input = "Who is eligible for ECHS?"
                    st.rerun()

            with col2:
                if st.button(
                    "How do I register for ECHS?",
                    key="q3",
                    use_container_width=True,
                ):
                    st.session_state.question_input = "How do I register for ECHS?"
                    st.rerun()
                if st.button(
                    "What documents are required?",
                    key="q4",
                    use_container_width=True,
                ):
                    st.session_state.question_input = (
                        "What documents are required for ECHS registration?"
                    )
                    st.rerun()

        with tab2:
            st.markdown("**Emergency Care Questions**")
            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Can I get emergency treatment at AFMS hospital?",
                    key="q5",
                    use_container_width=True,
                ):
                    st.session_state.question_input = (
                        "Can I get emergency treatment at AFMS hospital?"
                    )
                    st.rerun()
                if st.button(
                    "What is the procedure for emergency treatment?",
                    key="q6",
                    use_container_width=True,
                ):
                    st.session_state.question_input = (
                        "What is the procedure for emergency treatment in ECHS?"
                    )
                    st.rerun()

            with col2:
                if st.button(
                    "Can my parents be treated at AFMS in emergency?",
                    key="q7",
                    use_container_width=True,
                ):
                    st.session_state.question_input = (
                        "In an emergency, can my parents be treated at AFMS hospital on cashless basis?"
                    )
                    st.rerun()
                if st.button(
                    "What about non-empanelled hospitals?",
                    key="q8",
                    use_container_width=True,
                ):
                    st.session_state.question_input = (
                        "Can I get treatment in non-empanelled hospitals?"
                    )
                    st.rerun()

        with tab3:
            st.markdown("**Referrals & Reimbursement Questions**")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("What is HSR?", key="q9", use_container_width=True):
                    st.session_state.question_input = "What is HSR?"
                    st.rerun()
                if st.button(
                    "Can Non-MIL Polyclinic refer to AFMS?",
                    key="q10",
                    use_container_width=True,
                ):
                    st.session_state.question_input = (
                        "Can a Non MIL Polyclinic refer ESM to AFMS hospital?"
                    )
                    st.rerun()

            with col2:
                if st.button(
                    "Can 26-year-old dependent be referred?",
                    key="q11",
                    use_container_width=True,
                ):
                    st.session_state.question_input = (
                        "Can my dependent who is 26 years old be referred by the polyclinic to AFMS Hospital?"
                    )
                    st.rerun()
                if st.button(
                    "How do I claim reimbursement?",
                    key="q12",
                    use_container_width=True,
                ):
                    st.session_state.question_input = (
                        "How do I claim medical reimbursement in ECHS?"
                    )
                    st.rerun()

    # Ask button
    ask_button = st.button("🔍 Get Answer", use_container_width=True, type="primary")

    # Process question
    if (ask_button or question) and question:
        # Add to messages
        st.session_state.messages.append({"role": "user", "content": question})

        # Get answer with progress indicator
        with st.spinner("🔍 Searching ECHS training videos... 🤖 Generating answer..."):
            answer, sources = get_answer(
                question,
                st.session_state.client,
                st.session_state.index,
                st.session_state.documents,
                st.session_state.metadatas,
                st.session_state.embedding_model,
            )

        # Add answer to messages
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    # Display conversation
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 💬 Conversation History")

        # Display in reverse order (latest first)
        for i in range(len(st.session_state.messages) - 1, -1, -2):
            if i >= 0 and st.session_state.messages[i]["role"] == "assistant":
                # Get the question
                question_msg = (
                    st.session_state.messages[i - 1] if i > 0 else None
                )
                answer_msg = st.session_state.messages[i]

                if question_msg:
                    st.markdown(f"**❓ Question:** {question_msg['content']}")

                st.markdown(
                    '<div class="answer-box">', unsafe_allow_html=True
                )
                st.markdown(f"**💡 Answer:**\n\n{answer_msg['content']}")
                st.markdown("</div>", unsafe_allow_html=True)

                # Feedback buttons
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button(
                        "👍 Helpful",
                        key=f"helpful_{i}",
                        use_container_width=True,
                    ):
                        st.success("Thanks for your feedback!")
                with col2:
                    if st.button(
                        "👎 Not helpful",
                        key=f"not_helpful_{i}",
                        use_container_width=True,
                    ):
                        st.warning("Thanks! We'll work on improving.")

                # Sources
                if "sources" in answer_msg and answer_msg["sources"]:
                    st.markdown(
                        '<div class="source-box">', unsafe_allow_html=True
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
