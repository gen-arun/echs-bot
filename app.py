"""
ECHS Q&A Bot - Streamlit Web App (Simple Version)
Deploy this on Streamlit Cloud for public access
"""

import streamlit as st
import os
import json
from pathlib import Path
from groq import Groq
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Page configuration
st.set_page_config(
    page_title="ECHS Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
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
        margin-bottom: 3rem;
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

@st.cache_resource
def initialize_system():
    """Initialize the Q&A system once"""
    
    # Initialize Groq client
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=groq_api_key)
    
    # Load transcripts
    transcripts_path = Path("transcripts")
    
    # Load metadata
    with open(transcripts_path / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load all enriched transcripts
    documents = []
    metadatas = []
    ids = []
    
    for video_meta in metadata:
        video_id = video_meta['video_id']
        transcript_file = transcripts_path / f"{video_id}.txt"
        
        if transcript_file.exists():
            with open(transcript_file, 'r', encoding='utf-8') as f:
                enriched_content = f.read()
            
            # Split into chunks
            chunk_size = 1500
            chunks = [enriched_content[i:i+chunk_size] 
                     for i in range(0, len(enriched_content), chunk_size)]
            
            for j, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    'video_id': video_id,
                    'title': video_meta['title'],
                    'url': video_meta['url'],
                    'chunk_id': j
                })
                ids.append(f"{video_id}_chunk_{j}")
    
    # Create ChromaDB collection with proper settings
    chroma_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=False
    )
    chroma_client = chromadb.Client(chroma_settings)
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = chroma_client.create_collection(
        name="echs_enriched_transcripts",
        embedding_function=sentence_transformer_ef
    )
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return client, collection, metadata

def get_answer(question, client, collection):
    """Get answer for a question"""
    
    # Search for relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    
    relevant_docs = results['documents'][0]
    relevant_metadata = results['metadatas'][0]
    
    # Create context
    context = "\n\n---\n\n".join([
        f"From video: {meta['title']}\n{doc}" 
        for doc, meta in zip(relevant_docs, relevant_metadata)
    ])
    
    # Create prompt
    prompt = f"""You are an expert assistant helping people understand the Ex-Servicemen Comprehensive Health Scheme (ECHS). 

Based on the following information from ECHS training videos (including both spoken content and text from slides/visuals), please answer the user's question clearly and accurately.

CONTEXT FROM TRANSCRIPTS (Audio + Visual):
{context}

USER QUESTION:
{question}

Please provide a clear, helpful answer. If the information isn't in the transcripts, say so. Include which video(s) the information came from when relevant."""
    
    # Get answer from Groq
    try:
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
            title = meta['title']
            if title not in unique_sources:
                unique_sources[title] = meta['url']
        
        return answer, unique_sources
        
    except Exception as e:
        return f"Error: {str(e)}", {}

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 ECHS Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions about Ex-Servicemen Comprehensive Health Scheme</p>', 
                unsafe_allow_html=True)
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing ECHS Assistant..."):
            try:
                client, collection, metadata = initialize_system()
                st.session_state.client = client
                st.session_state.collection = collection
                st.session_state.metadata = metadata
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
                st.stop()
    
    # Text input
    question = st.text_input(
        "💬 Ask your question:",
        placeholder="e.g., What is ECHS? How do I register? What documents are required?",
        key="question_input"
    )
    
    # Example questions in expandable section
    with st.expander("📝 Click here for example questions"):
        st.markdown("**Common questions you can ask:**")
        examples = [
            "What is ECHS?",
            "Who is eligible for ECHS?",
            "How do I register for ECHS?",
            "What documents are required for ECHS registration?",
            "Can I get emergency treatment at AFMS hospital?",
            "What is HSR?",
            "Can my dependent who is 26 years old be referred to AFMS Hospital?",
            "Can a spouse be treated in AFMS hospital for planned treatment on a cashless basis?"
        ]
        
        col1, col2 = st.columns(2)
        for i, ex in enumerate(examples):
            if i % 2 == 0:
                with col1:
                    if st.button(ex, key=f"ex_{i}", use_container_width=True):
                        st.session_state.question_input = ex
                        st.rerun()
            else:
                with col2:
                    if st.button(ex, key=f"ex_{i}", use_container_width=True):
                        st.session_state.question_input = ex
                        st.rerun()
    
    # Ask button
    ask_button = st.button("🔍 Get Answer", use_container_width=True, type="primary")
    
    # Process question
    if (ask_button or question) and question:
        # Add to messages
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Get answer
        with st.spinner("🤔 Thinking..."):
            answer, sources = get_answer(
                question, 
                st.session_state.client, 
                st.session_state.collection
            )
        
        # Add answer to messages
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })
    
    # Display conversation
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 💬 Conversation History")
        
        # Display in reverse order (latest first)
        for i in range(len(st.session_state.messages)-1, -1, -2):
            if i >= 0 and st.session_state.messages[i]["role"] == "assistant":
                # Get the question (previous message)
                question_msg = st.session_state.messages[i-1] if i > 0 else None
                answer_msg = st.session_state.messages[i]
                
                if question_msg:
                    st.markdown(f"**❓ Question:** {question_msg['content']}")
                
                st.markdown(f'<div class="answer-box">', unsafe_allow_html=True)
                st.markdown(f"**💡 Answer:**\n\n{answer_msg['content']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Sources
                if "sources" in answer_msg and answer_msg["sources"]:
                    st.markdown('<div class="source-box">', unsafe_allow_html=True)
                    st.markdown("**📹 Information from these videos:**")
                    for j, (title, url) in enumerate(answer_msg["sources"].items(), 1):
                        st.markdown(f"{j}. [{title}]({url})")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>📊 Currently covering 5 ECHS training videos</p>
        <p>✅ Includes both spoken content and text from slides/visuals</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">Powered by AI • Free to use • Beta Version</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
