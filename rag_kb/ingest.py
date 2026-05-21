"""Ingest PDF documents into ChromaDB vector store for RAG.

Usage:  python -m rag_kb.ingest
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_kb.embedder import OllamaEmbedClient

load_dotenv()

# ── Config from .env ─────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:567m")
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent.parent / (
    os.getenv("CHROMA_DIR", "chroma_db")
)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in {DOCS_DIR}")
        print(f"    Place your PDFs in: {DOCS_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        print(f"  • {f.name}")
    print()

    all_docs = []
    for f in pdf_files:
        print(f"Loading: {f.name}...")
        loader = PyPDFLoader(str(f))
        docs = loader.load()
        print(f"  → {len(docs)} page(s)")
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"\nSplit into {len(chunks)} chunks")

    embed = OllamaEmbedClient(model=EMBED_MODEL, base_url=OLLAMA_URL)
    print(f"Embedding with {EMBED_MODEL} via Ollama...")

    from langchain_community.vectorstores import Chroma

    Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"\n✓ Done! {len(chunks)} chunks stored in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
