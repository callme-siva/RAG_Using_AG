"""
Core RAG Engine: Handles document processing, chunking, embeddings, vector indexing, and QA generation.
"""

import os
import io
import tempfile
from typing import List, Dict, Any, Generator, Tuple
import pypdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Vector Store
from langchain_community.vectorstores import Chroma

# Embedding & Chat Model Providers
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


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

    def _initialize_embeddings(self):
        """Initializes the embedding model based on selected provider."""
        if not self.api_key:
            raise ValueError(f"API Key for {self.provider} is required.")

        if self.provider == "Google Gemini":
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=self.api_key,
            )
        elif self.provider == "OpenAI":
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

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
        Returns the total number of chunks indexed.
        """
        if not chunks:
            raise ValueError("No valid document chunks to index.")

        # Create fresh Chroma in-memory store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
        )
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

        # Stream response tokens
        for chunk in chain.stream({"context": context_text, "question": query}):
            yield chunk
