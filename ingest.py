"""Ingest PDF documents into ChromaDB vector store for RAG.

Usage:  python ingest.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Config from .env ─────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:567m")
DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / (os.getenv("CHROMA_DIR", "chroma_db"))
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# ─────────────────────────────────────────────────────────────────────────────


def main():
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

    # Use direct HTTP embedding (compatible with all Ollama versions)
    class _EmbedClient:
        def __init__(self, model, base_url):
            self.model = model
            self.base_url = base_url
        def embed_query(self, text):
            import httpx
            r = httpx.post(f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text}, timeout=30)
            r.raise_for_status()
            return r.json()["embedding"]
        def embed_documents(self, texts):
            return [self.embed_query(t) for t in texts]

    embed = _EmbedClient(model=EMBED_MODEL, base_url=OLLAMA_URL)
    print(f"Embedding with {EMBED_MODEL} via Ollama...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"\n✓ Done! {len(chunks)} chunks stored in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
