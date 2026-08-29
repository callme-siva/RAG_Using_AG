"""
Streamlit RAG Explorer - An interactive web application for Retrieval-Augmented Generation.
Deployable on Streamlit Community Cloud and fully customizable.
"""

import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

from config import (
    PROVIDERS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
    APP_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
)
from rag_engine import RAGEngine

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title=f"{APP_TITLE} | Streamlit RAG Sample",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern aesthetics
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-box {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.15), rgba(124, 58, 237, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(79, 70, 229, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .source-card {
        background-color: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .suggestion-btn {
        margin: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "total_chunks_count" not in st.session_state:
    st.session_state.total_chunks_count = 0


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_api_key_from_env_or_secrets(provider: str) -> str:
    """Checks secrets or environment variables for pre-configured API keys."""
    env_var = PROVIDERS[provider]["key_env_var"]
    # Check Streamlit Cloud secrets
    if hasattr(st, "secrets") and env_var in st.secrets:
        return st.secrets[env_var]
    # Check OS environment (.env)
    return os.getenv(env_var, "")


def reset_rag_state():
    """Clears indexed documents and conversation memory."""
    st.session_state.messages = []
    st.session_state.rag_engine = None
    st.session_state.indexed_chunks = []
    st.session_state.indexed_files = []
    st.session_state.total_chunks_count = 0


# ==============================================================================
# Sidebar UI & Controls
# ==============================================================================
with st.sidebar:
    st.markdown(f"### ⚙️ {APP_TITLE}")
    st.caption(f"v{APP_VERSION} • Production-Ready RAG Starter")

    st.divider()

    # 1. Model & Provider Selection
    st.subheader("1. AI Model Setup")
    selected_provider = st.selectbox(
        "Select Provider",
        options=list(PROVIDERS.keys()),
        index=0,
    )

    provider_info = PROVIDERS[selected_provider]
    selected_model = st.selectbox(
        "Select Model",
        options=provider_info["models"],
        index=0,
    )

    # API Key Input
    saved_key = get_api_key_from_env_or_secrets(selected_provider)
    api_key_input = st.text_input(
        f"{selected_provider} API Key",
        value=saved_key,
        type="password",
        help=f"Enter key or set {provider_info['key_env_var']} in .env or Streamlit Secrets.",
    )

    if not api_key_input:
        st.info(f"💡 Get API key: [{provider_info['key_env_var']}]({provider_info['key_doc_url']})")

    # 2. Hyperparameters
    with st.expander("🔧 RAG Tuning Parameters", expanded=False):
        chunk_size = st.slider("Chunk Size (characters)", 200, 2000, DEFAULT_CHUNK_SIZE, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 400, DEFAULT_CHUNK_OVERLAP, 25)
        top_k = st.slider("Retrieved Chunks (Top-K)", 1, 10, DEFAULT_TOP_K, 1)
        temperature = st.slider("Model Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)

    st.divider()

    # 3. Document Ingestion
    st.subheader("2. Knowledge Ingestion")

    # Upload files
    uploaded_files = st.file_uploader(
        "Upload Custom Documents (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    st.markdown("**Or choose a built-in demo dataset:**")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        load_demo_pdf = st.button("📄 Load Quantum Report (PDF)", use_container_width=True)
    with col_d2:
        load_demo_md = st.button("📝 Load AI Guide (MD)", use_container_width=True)

    load_both = st.button("📚 Load Both Demo Datasets", use_container_width=True)

    clear_kb = st.button("🗑️ Reset Knowledge Base", use_container_width=True)

    if clear_kb:
        reset_rag_state()
        st.success("Knowledge base reset!")
        st.rerun()

    # Ingestion Trigger
    process_docs = st.button("⚡ Index Uploaded Documents", type="primary", use_container_width=True)

    # Ingestion Logic
    load_any_demo = load_demo_pdf or load_demo_md or load_both
    if process_docs or load_any_demo:
        if not api_key_input:
            st.error(f"❌ Please provide an API key for {selected_provider} to proceed.")
        else:
            with st.spinner("Initializing RAG Engine and processing documents..."):
                try:
                    engine = RAGEngine(
                        provider=selected_provider,
                        api_key=api_key_input,
                        model_name=selected_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k,
                        temperature=temperature,
                    )

                    all_chunks = []
                    indexed_names = []

                    sample_dir = Path(__file__).parent / "sample_data"

                    if load_demo_pdf or load_both:
                        pdf_path = sample_dir / "quantum_computing_and_ai_report.pdf"
                        if pdf_path.exists():
                            with open(pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            pdf_chunks = engine.load_and_split_file("quantum_computing_and_ai_report.pdf", pdf_bytes)
                            all_chunks.extend(pdf_chunks)
                            indexed_names.append("quantum_computing_and_ai_report.pdf (Demo PDF)")

                    if load_demo_md or load_both:
                        md_path = sample_dir / "ai_agents_guide.md"
                        if md_path.exists():
                            with open(md_path, "rb") as f:
                                md_bytes = f.read()
                            md_chunks = engine.load_and_split_file("ai_agents_guide.md", md_bytes)
                            all_chunks.extend(md_chunks)
                            indexed_names.append("ai_agents_guide.md (Demo MD)")

                    if uploaded_files:
                        for uf in uploaded_files:
                            file_bytes = uf.read()
                            file_chunks = engine.load_and_split_file(uf.name, file_bytes)
                            all_chunks.extend(file_chunks)
                            indexed_names.append(uf.name)

                    if not all_chunks:
                        st.warning("⚠️ Please upload a file or choose a demo dataset to index.")
                    else:
                        total_indexed = engine.build_vector_store(all_chunks)
                        st.session_state.rag_engine = engine
                        st.session_state.indexed_chunks = all_chunks
                        st.session_state.indexed_files = indexed_names
                        st.session_state.total_chunks_count = total_indexed
                        st.success(f"Indexed {total_indexed} chunks successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error initializing RAG engine: {str(e)}")

    st.divider()

    # Knowledge Base Stats
    st.subheader("3. Knowledge Base Status")
    if st.session_state.total_chunks_count > 0:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="badge">ACTIVE INDEX</div>
                <h4 style="margin: 8px 0 4px 0;">{st.session_state.total_chunks_count} Chunks</h4>
                <p style="font-size: 0.82rem; color: #94a3b8; margin: 0;">
                    <b>Files:</b> {', '.join(st.session_state.indexed_files)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No documents indexed yet. Upload files or load demo to begin.")


# ==============================================================================
# Main Content Area
# ==============================================================================

# Header
st.title(f"🚀 {APP_TITLE}")
st.caption(APP_SUBTITLE)

tab_chat, tab_inspector, tab_architecture = st.tabs([
    "💬 Interactive RAG Chat",
    "📚 Document Chunks Inspector",
    "🧠 How RAG Works",
])

# ------------------------------------------------------------------------------
# TAB 1: Interactive Chat
# ------------------------------------------------------------------------------
with tab_chat:
    # If no documents are loaded, show a welcome card with quick starts
    if st.session_state.total_chunks_count == 0:
        st.markdown(
            """
            ### Welcome to the Streamlit RAG Sample! 👋
            Retrieval-Augmented Generation (RAG) allows you to ask questions about private documents with precise citations and zero hallucination.

            #### 🏁 Quick Start:
            1. **Enter API Key**: Add your Google Gemini or OpenAI API Key in the left sidebar.
            2. **Load Knowledge**: Click **'🚀 Load Demo Guide'** or upload your own PDF/TXT/Markdown files.
            3. **Ask Questions**: Ask anything about your documents and inspect exact citations!
            """
        )
        col_demo, _ = st.columns([1, 3])
        with col_demo:
            if st.button("✨ Click to Load Demo Knowledge Base Now", type="primary"):
                # Trigger demo load through rerun
                st.session_state.load_demo_trigger = True
                st.rerun()

    # Pre-populate sample question suggestions when knowledge base is loaded & history is empty
    if st.session_state.total_chunks_count > 0 and len(st.session_state.messages) == 0:
        st.markdown("##### 💡 Suggested Questions to Try:")
        cols = st.columns(3)
        suggestions = [
            "What are the NIST standards for Post-Quantum Cryptography (e.g. ML-KEM)?",
            "Compare Superconducting Transmons vs Trapped Ions hardware.",
            "What is an AI Agent and how does it plan?",
        ]
        for idx, col in enumerate(cols):
            with col:
                if st.button(suggestions[idx], key=f"sug_{idx}", use_container_width=True):
                    st.session_state.suggested_query = suggestions[idx]
                    st.rerun()

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("🔍 View Retrieved Source Citations", expanded=False):
                    for idx, src in enumerate(message["sources"]):
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <b>Chunk #{src.get('chunk_id', idx+1)}</b> | <i>Source: {src.get('source', 'Unknown')} (Page {src.get('page', 1)})</i>
                                <p style="margin-top: 6px; white-space: pre-wrap;">{src.get('content', '')}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Check for suggested query from button clicks
    prefilled_prompt = ""
    if "suggested_query" in st.session_state and st.session_state.suggested_query:
        prefilled_prompt = st.session_state.suggested_query
        st.session_state.suggested_query = ""

    # User Input
    user_query = st.chat_input("Ask a question about your indexed documents...") or prefilled_prompt

    if user_query:
        # Check if RAG engine is active
        if st.session_state.rag_engine is None:
            st.warning("⚠️ Please index documents or load the demo guide before asking questions.")
        else:
            # 1. Display User Message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # 2. Retrieve & Generate Response
            with st.chat_message("assistant"):
                try:
                    engine: RAGEngine = st.session_state.rag_engine

                    with st.spinner("Searching relevant document chunks..."):
                        start_time = time.time()
                        retrieved_docs = engine.retrieve_context(user_query)
                        retrieval_latency = round(time.time() - start_time, 2)

                    # Prepare sources for citation display
                    sources_meta = []
                    for doc in retrieved_docs:
                        sources_meta.append({
                            "source": doc.metadata.get("source", "Document"),
                            "page": doc.metadata.get("page", 1),
                            "chunk_id": doc.metadata.get("chunk_id", "?"),
                            "content": doc.page_content,
                        })

                    # Stream LLM generation
                    response_stream = engine.generate_rag_response_stream(user_query, retrieved_docs)
                    full_response = st.write_stream(response_stream)

                    # Display retrieved sources
                    with st.expander(f"🔍 Retrieved Citations ({len(retrieved_docs)} chunks in {retrieval_latency}s)", expanded=False):
                        for src in sources_meta:
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <b>Chunk #{src['chunk_id']}</b> | <i>Source: {src['source']} (Page {src['page']})</i>
                                    <p style="margin-top: 6px; white-space: pre-wrap;">{src['content']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources_meta,
                    })

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")


# ------------------------------------------------------------------------------
# TAB 2: Document Chunks Inspector
# ------------------------------------------------------------------------------
with tab_inspector:
    st.subheader("📚 In-Memory Vector Store Explorer")
    st.caption("Inspect how the Text Splitter partitioned your uploaded documents into searchable embeddings.")

    if not st.session_state.indexed_chunks:
        st.info("No chunks indexed yet. Ingest documents in the sidebar to inspect chunks.")
    else:
        st.write(f"Total chunks indexed: **{len(st.session_state.indexed_chunks)}**")

        search_chunk = st.text_input("Filter chunks by keyword:", "")
        filtered_chunks = [
            c for c in st.session_state.indexed_chunks
            if search_chunk.lower() in c.page_content.lower() or search_chunk.lower() in str(c.metadata).lower()
        ] if search_chunk else st.session_state.indexed_chunks

        st.caption(f"Showing {len(filtered_chunks)} of {len(st.session_state.indexed_chunks)} chunks")

        for idx, chunk in enumerate(filtered_chunks[:30]):
            with st.expander(f"Chunk #{chunk.metadata.get('chunk_id', idx+1)} — {chunk.metadata.get('source', 'Unknown')} (Page {chunk.metadata.get('page', 1)})", expanded=False):
                st.code(chunk.page_content, language="markdown")
                st.json(chunk.metadata)


# ------------------------------------------------------------------------------
# TAB 3: How RAG Works
# ------------------------------------------------------------------------------
with tab_architecture:
    st.subheader("🧠 Understanding Retrieval-Augmented Generation")
    st.markdown(
        """
        ### What makes RAG so powerful?
        Large Language Models (LLMs) are trained on general internet data, but cannot see your private company files, latest documentation, or internal notes. 
        RAG connects LLMs to your private data **without fine-tuning**.

        ---
        ### The 4 Stages of RAG:
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            #### 1. Ingestion & Chunking 📄
            - Raw documents (PDFs, Markdown, Word) are converted into raw text.
            - `RecursiveCharacterTextSplitter` breaks text into small, contextual pieces (e.g., 800 chars) with overlap to preserve sentences across boundaries.

            #### 2. Vector Embeddings & Indexing 🔢
            - Each text chunk is passed to an embedding model (e.g., Google `models/embedding-001` or OpenAI `text-embedding-3-small`).
            - Text is transformed into high-dimensional vectors representing semantic meaning and stored in **ChromaDB**.
            """
        )

    with col2:
        st.markdown(
            """
            #### 3. Semantic Retrieval 🔍
            - When a user asks a question, the query is embedded into a vector.
            - ChromaDB performs cosine similarity search to find the **Top-$k$ closest matching chunks**.

            #### 4. Grounded Generation 💬
            - The retrieved passages are placed into a strict system prompt.
            - The LLM synthesizes an accurate answer with exact citations, eliminating hallucinations.
            """
        )

    st.markdown("---")
    st.markdown("#### 📐 Architecture Diagram")
    st.code(
        """
[User Document (PDF/MD/TXT)] ───► [Text Splitter] ───► [Embeddings] ───► [ChromaDB Vector Store]
                                                                                ▲
                                                                                │ (Similarity Search)
[User Query] ───────────────────► [Query Embeddings] ───────────────────────────┤
                                                                                ▼
[Synthesized Answer with Sources] ◄── [LLM (Gemini/OpenAI)] ◄── [Prompt + Top-K Context]
        """,
        language="text",
    )
