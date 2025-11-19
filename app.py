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
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


# -------------------------------------------------------------------
# 🔧 DEBUG-ENABLED INITIALIZATION FUNCTION
# -------------------------------------------------------------------
@st.cache_resource
def initialize_system():
    """Initialize the Q&A system with precomputed embeddings (with debug logs)"""

    st.write("🔹 DEBUG: Starting initialization...")

    try:
        st.write("🔹 DEBUG: Loading Groq client...")
        groq_api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=groq_api_key)

        st.write("🔹 DEBUG: Checking required FAISS & metadata files...")
        required_files = ['index.faiss', 'documents_metadata.json']
        for file in required_files:
            if not Path(file).exists():
                st.error(f"❌ DEBUG: Missing required file: {file}")
                raise FileNotFoundError(f"Required file '{file}' not found.")

        st.write("🔹 DEBUG: Loading FAISS index...")
        index = faiss.read_index('index.faiss')

        st.write("🔹 DEBUG: Loading documents & metadata...")
        with open('documents_metadata.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            documents = data['documents']
            metadatas = data['metadatas']

        st.write("🔹 DEBUG: Loading SentenceTransformer model (may take time first run)...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        st.write("🔹 DEBUG: Loading video metadata...")
        transcripts_path = Path("transcripts")
        metadata_file = transcripts_path / "metadata.json"

        if not metadata_file.exists():
            st.error("❌ DEBUG: transcripts/metadata.json NOT FOUND")
            raise FileNotFoundError("Video metadata not found.")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        st.success("✅ DEBUG: Initialization completed successfully!")
        return client, index, documents, metadatas, embedding_model, metadata

    except Exception as e:
        st.error(f"❌ DEBUG: Initialization failed: {str(e)}")
        st.stop()


# -------------------------------------------------------------------
# 🤖 ANSWER GENERATION FUNCTION
# -------------------------------------------------------------------
def get_answer(question, client, index, documents, metadatas, embedding_model, n_results=5):
    """Get answer for a question"""

    try:
        # Create query embedding
        query_embedding = embedding_model.encode([question])
        query_embedding = np.array(query_embedding).astype('float32')

        # Search in FAISS
        distances, indices = index.search(query_embedding, n_results)

        # Get relevant documents
        relevant_docs = [documents[i] for i in indices[0]]
        relevant_metadata = [metadatas[i] for i in indices[0]]

        # Build context
        context = "\n\n---\n\n".join([
            f"From video: {meta['title']}\n{doc}"
            for doc, meta in zip(relevant_docs, relevant_metadata)
        ])

        # Prompt with summary requirement
        prompt = f"""You are an expert assistant helping people understand the Ex-Servicemen Contributory Health Scheme (ECHS). 

Based on the following information from ECHS training videos (including spoken content + slide text), answer clearly and accurately.

CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Base your answer ONLY on the transcripts.
- If information is missing, say so.
- Cite which videos the answer comes from.
- END your answer with:

**Quick Summary:**
• Bullet 1  
• Bullet 2  
• Bullet 3
"""

        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
        )

        answer = chat_completion.choices[0].message.content

        # Unique video sources
        unique_sources = {}
        for meta in relevant_metadata:
            title = meta['title']
            if title not in unique_sources:
                unique_sources[title] = meta['url']

        return answer, unique_sources

    except Exception as e:
        msg = str(e).lower()
        if "timeout" in msg or "503" in msg:
            return ("⏱️ AI service busy. Try again shortly.", {})
        if "rate limit" in msg or "429" in msg:
            return ("⚠️ Too many requests. Please wait 30 seconds.", {})
        if "connection" in msg:
            return ("🔌 Network issue. Check connection.", {})
        return ("❌ Something went wrong. Try again.", {})


# -------------------------------------------------------------------
# 🎨 MAIN APP
# -------------------------------------------------------------------
def main():

    # Header
    st.markdown('<h1 class="main-header">🏥 ECHS Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions about Ex-Servicemen Contributory Health Scheme</p>',
                unsafe_allow_html=True)

    # Disclaimer
    st.info("ℹ️ Based on 5 official ECHS training videos. For specific cases, consult your ECHS polyclinic.")

    # Initialization
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

    # Text input
    question = st.text_input(
        "💬 Ask your question:",
        placeholder="e.g., What is ECHS? How do I register? What documents are required?",
        key="question_input"
    )

    # Ask button
    ask_button = st.button("🔍 Get Answer", use_container_width=True)

    # If question submitted
    if (ask_button or question) and question:

        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🔍 Searching videos... Generating answer..."):
            answer, sources = get_answer(
                question,
                st.session_state.client,
                st.session_state.index,
                st.session_state.documents,
                st.session_state.metadatas,
                st.session_state.embedding_model
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

    # Conversation display
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 💬 Conversation History")

        for i in range(len(st.session_state.messages)-1, -1, -2):
            if i >= 0 and st.session_state.messages[i]["role"] == "assistant":
                question_msg = st.session_state.messages[i-1]
                answer_msg = st.session_state.messages[i]

                st.markdown(f"**❓ Question:** {question_msg['content']}")

                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown(f"{answer_msg['content']}")
                st.markdown('</div>', unsafe_allow_html=True)

                # Feedback
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.button("👍 Helpful", key=f"help_{i}")
                with col2:
                    st.button("👎 Not helpful", key=f"not_help_{i}")

                # Sources
                if answer_msg.get("sources"):
                    st.markdown('<div class="source-box">', unsafe_allow_html=True)
                    st.markdown("**📹 Video Sources:**")
                    for j, (title, url) in enumerate(answer_msg["sources"].items(), 1):
                        st.markdown(f"{j}. [{title}]({url})")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")

        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>📊 Currently covering 5 ECHS training videos</p>
        <p>✅ Includes spoken + slide text</p>
        <p style="font-size: 0.9rem;">Powered by AI • Beta Version</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
