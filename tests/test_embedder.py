"""Tests for embedder.py — Ollama embedding client."""

from rag_kb.embedder import OllamaEmbedClient


class TestOllamaEmbedClient:
    def test_init(self):
        client = OllamaEmbedClient(model="bge-m3:567m", base_url="http://localhost:11434")
        assert client.model == "bge-m3:567m"
        assert client.base_url == "http://localhost:11434"

    def test_different_model(self):
        client = OllamaEmbedClient(model="nomic-embed-text", base_url="http://ollama:11434")
        assert client.model == "nomic-embed-text"
