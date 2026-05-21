"""Entry point for RAG proxy server. Delegates to rag_kb.rag_proxy."""

from rag_kb.rag_proxy import app

if __name__ == "__main__":
    import uvicorn

    from rag_kb.rag_proxy import PROXY_PORT

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
