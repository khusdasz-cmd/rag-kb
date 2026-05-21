"""Ollama embedding client for RAG.

Provides a LangChain-compatible embedding interface using Ollama's
/api/embeddings endpoint — always runs locally, regardless of LLM backend.
"""

from __future__ import annotations

import httpx


class OllamaEmbedClient:
    """Embed documents via Ollama /api/embeddings.

    Parameters
    ----------
    model : str
        Ollama model name (e.g. "bge-m3:567m").
    base_url : str
        Ollama server URL (e.g. "http://localhost:11434").
    """

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self.embed_query(t) for t in texts]
