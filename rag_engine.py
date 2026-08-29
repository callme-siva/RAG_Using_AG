"""
Core RAG Engine: Handles document processing, chunking, embeddings, vector indexing, and QA generation.
Includes resilient fallback to Chroma's built-in ONNX embeddings if API embedding endpoints fail.
"""

import os
import io
from typing import List, Dict, Any, Generator, Tuple
import pypdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

# Vector Store (support both langchain-chroma and langchain-community)
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Embedding & Chat Model Providers
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


class LocalDefaultEmbeddings(Embeddings):
    """
    Zero-configuration local embedding fallback using Chroma's built-in ONNX model.
    Runs locally on CPU/Metal with zero network calls and zero quota constraints.
    """
    def __init__(self):
        from chromadb.utils import embedding_functions
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]


class RAGEngine:
    """
    Modular RAG pipeline managing embeddings, vector storage, and query execution.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model_name: str,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 3,
        temperature: float = 0.2,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature

        self.vector_store: Chroma | None = None
        self.embedding_model = self._initialize_embeddings()
        self.llm = self._initialize_llm()
        self.embedding_info = ""

    def _initialize_embeddings(self) -> Embeddings:
        """Initializes the embedding model based on selected provider."""
        if not self.api_key:
            raise ValueError(f"API Key for {self.provider} is required.")

        if self.provider == "Google Gemini":
            # models/embedding-001 is universally supported across Gemini API versions
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key,
            )
        elif self.provider == "OpenAI":
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.api_key,
            )
        else:
            return LocalDefaultEmbeddings()

    def _initialize_llm(self):
        """Initializes the Large Language Model."""
        if not self.api_key:
            raise ValueError(f"API Key for {self.provider} is required.")

        if self.provider == "Google Gemini":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
            )
        elif self.provider == "OpenAI":
            return ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=self.api_key,
                temperature=self.temperature,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def load_and_split_file(self, file_name: str, file_bytes: bytes) -> List[Document]:
        """
        Parses raw file bytes (PDF, TXT, MD) and splits into semantic chunks.
        """
        docs: List[Document] = []

        if file_name.lower().endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page_idx, page in enumerate(pdf_reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": file_name, "page": page_idx + 1},
                        )
                    )
        else:
            # Handle text, markdown, etc.
            text_content = file_bytes.decode("utf-8", errors="ignore")
            docs.append(
                Document(
                    page_content=text_content,
                    metadata={"source": file_name, "page": 1},
                )
            )

        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = text_splitter.split_documents(docs)

        # Add chunk index metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i + 1

        return chunks

    def build_vector_store(self, chunks: List[Document]) -> int:
        """
        Builds or updates the in-memory Chroma vector store with document chunks.
        Includes automatic fallback to local ONNX embeddings if remote API embedding fails.
        """
        if not chunks:
            raise ValueError("No valid document chunks to index.")

        if self.provider == "Google Gemini":
            # Try available Gemini embedding model variants, then fallback to local ONNX embeddings
            candidates = ["models/embedding-001", "embedding-001", "text-embedding-004"]
            built = False
            for model_name in candidates:
                try:
                    emb = GoogleGenerativeAIEmbeddings(
                        model=model_name,
                        google_api_key=self.api_key,
                    )
                    self.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=emb,
                    )
                    self.embedding_model = emb
                    self.embedding_info = f"Google ({model_name})"
                    built = True
                    break
                except Exception:
                    continue

            # If remote embedding failed (e.g. 404 or quota), use local ONNX embeddings seamlessly
            if not built:
                local_emb = LocalDefaultEmbeddings()
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=local_emb,
                )
                self.embedding_model = local_emb
                self.embedding_info = "Local ONNX (all-MiniLM-L6-v2 fallback)"
        else:
            try:
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embedding_model,
                )
                self.embedding_info = f"{self.provider} Embeddings"
            except Exception:
                local_emb = LocalDefaultEmbeddings()
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=local_emb,
                )
                self.embedding_model = local_emb
                self.embedding_info = "Local ONNX (all-MiniLM-L6-v2 fallback)"

        return len(chunks)

    def retrieve_context(self, query: str) -> List[Document]:
        """
        Retrieves top-k relevant chunks for a user query.
        """
        if self.vector_store is None:
            raise ValueError("Vector database is empty. Please upload documents first.")

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        return retriever.invoke(query)

    def generate_rag_response_stream(
        self, query: str, context_docs: List[Document]
    ) -> Generator[str, None, None]:
        """
        Streams response from the LLM augmented with the retrieved context.
        Includes automatic fallback to gemini-1.5-flash or gemini-3.6-flash if selected model is deprecated.
        """
        # Format context string
        context_text = "\n\n---\n\n".join(
            [f"[Source: {doc.metadata.get('source', 'Unknown')} (Chunk {doc.metadata.get('chunk_id', '?')})]\n{doc.page_content}"
             for doc in context_docs]
        )

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert, helpful AI assistant analyzing user-provided documents using Retrieval-Augmented Generation (RAG).\n"
                "Answer the user's question accurately using ONLY the retrieved context below.\n"
                "If the context does not contain the answer, politely state that the provided documents do not contain sufficient information.\n"
                "Cite your sources with chunk/page numbers when referencing specific facts.\n\n"
                "CONTEXT:\n{context}",
            ),
            ("human", "{question}"),
        ])

        chain = prompt_template | self.llm | StrOutputParser()

        try:
            for chunk in chain.stream({"context": context_text, "question": query}):
                yield chunk
        except Exception as e:
            # If Google Gemini model failed (e.g. 404 deprecated model), attempt fallback models
            if self.provider == "Google Gemini":
                fallback_models = ["gemini-1.5-flash", "gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
                streamed = False
                for fb_model in fallback_models:
                    if fb_model == self.model_name:
                        continue
                    try:
                        self.llm = ChatGoogleGenerativeAI(
                            model=fb_model,
                            google_api_key=self.api_key,
                            temperature=self.temperature,
                        )
                        fallback_chain = prompt_template | self.llm | StrOutputParser()
                        for chunk in fallback_chain.stream({"context": context_text, "question": query}):
                            yield chunk
                        self.model_name = fb_model
                        streamed = True
                        break
                    except Exception:
                        continue
                if not streamed:
                    raise e
            else:
                raise e
