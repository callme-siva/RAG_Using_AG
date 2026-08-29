"""
Model Verification & Diagnostics Utility
Run this script to verify that your API keys and all configured models are operational.

Usage:
    python verify_models.py
    python verify_models.py --google-key AIzaSy...
    python verify_models.py --openai-key sk-...
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

from config import PROVIDERS
from rag_engine import LocalDefaultEmbeddings

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    ChatOpenAI = None
    OpenAIEmbeddings = None


def test_google_models(api_key: str):
    print("\n" + "=" * 60)
    print("🔍 Testing Google Gemini Models & Embeddings")
    print("=" * 60)

    if not api_key:
        print("❌ No Google API Key provided. Skipping Google tests.")
        return

    # 1. Test Embeddings
    print("\n1. Testing Embeddings:")
    emb_candidates = ["models/embedding-001", "embedding-001", "text-embedding-004"]
    emb_success = False
    for emb_name in emb_candidates:
        try:
            start = time.time()
            emb = GoogleGenerativeAIEmbeddings(model=emb_name, google_api_key=api_key)
            vec = emb.embed_query("Test embedding string")
            latency = round(time.time() - start, 2)
            print(f"  ✅ Embedding [{emb_name}]: SUCCESS (dim: {len(vec)}, {latency}s)")
            emb_success = True
            break
        except Exception as e:
            print(f"  ⚠️ Embedding [{emb_name}]: NOT AVAILABLE ({str(e)[:80]}...)")

    if not emb_success:
        print("  ℹ️ Testing Local ONNX Embedding Fallback:")
        try:
            start = time.time()
            local_emb = LocalDefaultEmbeddings()
            vec = local_emb.embed_query("Test local embedding")
            latency = round(time.time() - start, 2)
            print(f"  ✅ Local ONNX Fallback: SUCCESS (dim: {len(vec)}, {latency}s)")
        except Exception as e:
            print(f"  ❌ Local ONNX Fallback: FAILED ({e})")

    # 2. Test LLM Chat Models
    print("\n2. Testing Chat Models:")
    models = PROVIDERS["Google Gemini"]["models"]
    for model_name in models:
        try:
            start = time.time()
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)
            res = llm.invoke("Say 'READY' if you can read this message.")
            latency = round(time.time() - start, 2)
            reply = res.content.strip().replace("\n", " ")[:30]
            print(f"  ✅ LLM [{model_name}]: SUCCESS ({latency}s) — Response: \"{reply}\"")
        except Exception as e:
            print(f"  ❌ LLM [{model_name}]: FAILED — {e}")


def test_openai_models(api_key: str):
    print("\n" + "=" * 60)
    print("🔍 Testing OpenAI Models & Embeddings")
    print("=" * 60)

    if not api_key:
        print("❌ No OpenAI API Key provided. Skipping OpenAI tests.")
        return

    # 1. Test Embeddings
    print("\n1. Testing Embeddings:")
    try:
        start = time.time()
        emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        vec = emb.embed_query("Test embedding string")
        latency = round(time.time() - start, 2)
        print(f"  ✅ Embedding [text-embedding-3-small]: SUCCESS (dim: {len(vec)}, {latency}s)")
    except Exception as e:
        print(f"  ❌ Embedding [text-embedding-3-small]: FAILED — {e}")

    # 2. Test LLM Chat Models
    print("\n2. Testing Chat Models:")
    models = PROVIDERS["OpenAI"]["models"]
    for model_name in models:
        try:
            start = time.time()
            llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0.0)
            res = llm.invoke("Say 'READY' if you can read this message.")
            latency = round(time.time() - start, 2)
            reply = res.content.strip().replace("\n", " ")[:30]
            print(f"  ✅ LLM [{model_name}]: SUCCESS ({latency}s) — Response: \"{reply}\"")
        except Exception as e:
            print(f"  ❌ LLM [{model_name}]: FAILED — {e}")


def main():
    parser = argparse.ArgumentParser(description="Verify all RAG models and API keys.")
    parser.add_argument("--google-key", type=str, help="Google Gemini API Key", default=os.getenv("GOOGLE_API_KEY", ""))
    parser.add_argument("--openai-key", type=str, help="OpenAI API Key", default=os.getenv("OPENAI_API_KEY", ""))
    args = parser.parse_args()

    google_key = args.google_key
    openai_key = args.openai_key

    if not google_key and not openai_key:
        print("⚠️ No API keys detected in arguments or .env.")
        print("Tip: Pass keys via CLI:")
        print("   python verify_models.py --google-key YOUR_KEY")
        print("   python verify_models.py --openai-key YOUR_KEY")
        return

    if google_key:
        test_google_models(google_key)

    if openai_key:
        test_openai_models(openai_key)

    print("\n" + "=" * 60)
    print(" Diagnostic Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
