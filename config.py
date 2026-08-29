"""
Configuration and constants for the RAG application.
"""

# Available LLM Providers & Models
PROVIDERS = {
    "Google Gemini": {
        "models": [
            "gemini-3.6-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        "default_model": "gemini-3.6-flash",
        "embedding_model": "models/embedding-001",
        "key_env_var": "GOOGLE_API_KEY",
        "key_doc_url": "https://aistudio.google.com/app/apikey",
    },
    "OpenAI": {
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
        ],
        "default_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "key_env_var": "OPENAI_API_KEY",
        "key_doc_url": "https://platform.openai.com/api-keys",
    },
}

# RAG Pipeline Defaults
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 3
DEFAULT_TEMPERATURE = 0.2

# App Metadata
APP_TITLE = "RAG Explorer 🚀"
APP_SUBTITLE = "Chat with your Documents using Retrieval-Augmented Generation"
APP_VERSION = "1.0.0"
