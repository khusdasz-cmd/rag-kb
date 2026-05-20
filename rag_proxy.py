"""RAG Proxy — Chatbox ↔ ChromaDB ↔ 任意 LLM 后端（Ollama / LM Studio / OpenAI）。

通过 .env 配置后端类型，无需改代码即可切换。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

load_dotenv()

# ── Config from .env ─────────────────────────────────────────────────────────
LLM_TYPE = os.getenv("LLM_TYPE", "ollama")
PROXY_PORT = int(os.getenv("PROXY_PORT", "9124"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma_db"))
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = Path(__file__).parent / CHROMA_DIR
TOP_K = int(os.getenv("TOP_K", "4"))

# Ollama settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:567m")

# OpenAI-compatible settings
OPENAI_URL = os.getenv("OPENAI_URL", "http://localhost:1234/v1")
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag-proxy")

app = FastAPI(title="RAG Proxy")
vectorstore = None


# ── Embedding Client (always local Ollama) ───────────────────────────────────


class OllamaEmbedClient:
    """Embed via Ollama /api/embeddings — always local, regardless of LLM backend."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url

    def embed_query(self, text: str) -> list[float]:
        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


# ── LLM Backends ─────────────────────────────────────────────────────────────


def _stream_openai_chunks(resp: httpx.Response) -> AsyncGenerator[bytes, None]:
    """Convert any upstream SSE stream into Chatbox-compatible OpenAI chunks."""
    async def _gen():
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                yield "data: [DONE]\n\n"
                return
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices", [])
            delta = choices[0].get("delta", {}) if choices else {}
            finish = choices[0].get("finish_reason") if choices else None
            content = delta.get("content", "")
            yield f"data: {json.dumps({'id': 'rag-proxy', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': content} if content else {}, 'finish_reason': finish}]}, ensure_ascii=False)}\n\n"
            if finish:
                yield "data: [DONE]\n\n"
                return
    return _gen()


class OllamaBackend:
    def __init__(self):
        self.base = OLLAMA_URL
        self.model = OLLAMA_MODEL

    async def chat(self, messages: list[dict], stream: bool = True):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": 0.7},
        }
        async with httpx.AsyncClient(timeout=120) as c:
            resp = await c.post(f"{self.base}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            if not stream:
                return resp.json()
            return _stream_openai_chunks(resp)

    async def list_models(self) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base}/api/tags")
                models = r.json().get("models", [])
                return [{"id": m["name"], "object": "model", "created": 0, "owned_by": "ollama"} for m in models]
        except Exception:
            return [{"id": self.model, "object": "model", "created": 0, "owned_by": "ollama"}]

    async def health_extra(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base}/api/tags")
                models = r.json().get("models", [])
                return {"backend": "ollama", "available_models": [m["name"] for m in models]}
        except Exception:
            return {"backend": "ollama", "available_models": [], "error": "Ollama unreachable"}


class OpenAIBackend:
    def __init__(self):
        self.base = OPENAI_URL.rstrip("/")
        self.model = OPENAI_MODEL
        self.headers = {"Content-Type": "application/json", "Accept-Encoding": "identity, gzip, deflate"}
        if OPENAI_KEY:
            self.headers["Authorization"] = f"Bearer {OPENAI_KEY}"

    async def chat(self, messages: list[dict], stream: bool = True):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        async with httpx.AsyncClient(timeout=120) as c:
            resp = await c.post(
                f"{self.base}/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=120,
            )
            resp.raise_for_status()
            if not stream:
                return resp.json()
            return _stream_openai_chunks(resp)

    async def list_models(self) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base}/models", headers=self.headers)
                data = r.json().get("data", [])
                return data
        except Exception:
            return [{"id": self.model, "object": "model", "created": 0, "owned_by": "openai"}]

    async def health_extra(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base}/models", headers=self.headers)
                data = r.json().get("data", [])
                return {"backend": "openai", "available_models": [m["id"] for m in data[:10]]}
        except Exception:
            return {"backend": "openai", "available_models": [], "error": "Upstream unreachable"}


# Select backend
if LLM_TYPE == "openai":
    backend = OpenAIBackend()
    log.info("Backend: OpenAI-compatible (%s → %s)", OPENAI_URL, OPENAI_MODEL)
else:
    backend = OllamaBackend()
    log.info("Backend: Ollama (%s → %s)", OLLAMA_URL, OLLAMA_MODEL)


# ── Prompt ───────────────────────────────────────────────────────────────────


def build_system_prompt(contexts: list[str]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "你是一个基于本地知识库的问答助手。\n\n"
        "请用以下资料回答问题。如果资料与问题无关，直接基于你的知识回答。\n"
        "回答请用中文，简洁准确。\n\n"
        "参考资料：\n"
        f"{ctx}"
    ) if contexts else (
        "你是一个知识库问答助手。请简洁准确地回答问题，用中文回答。"
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    global vectorstore
    if CHROMA_DIR.exists():
        embed = OllamaEmbedClient(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_URL)
        try:
            from langchain_chroma import Chroma
            vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embed)
            log.info("ChromaDB loaded (%d docs)", vectorstore._collection.count())
        except Exception as e:
            log.warning("ChromaDB load failed: %s — RAG disabled", e)
    else:
        log.warning("ChromaDB not found — RAG disabled (run ingest.py first)")


@app.get("/")
async def root():
    return {
        "status": "running",
        "backend": LLM_TYPE,
        "rag_ready": vectorstore is not None,
        "chatbox_endpoint": f"http://localhost:{PROXY_PORT}/v1",
    }


@app.get("/health")
async def health():
    extra = await backend.health_extra()
    return {"status": "ok", "rag_ready": vectorstore is not None, **extra}


@app.get("/v1/models")
async def list_models():
    data = await backend.list_models()
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract user query for RAG
    user_msgs = [m for m in messages if m.get("role") == "user"]
    query = user_msgs[-1]["content"] if user_msgs else ""
    if isinstance(query, list):
        query = " ".join(p["text"] for p in query if isinstance(p, dict) and "text" in p)

    # Retrieve context
    contexts = []
    if vectorstore and query.strip():
        try:
            docs = vectorstore.similarity_search(query, k=TOP_K)
            contexts = [d.page_content for d in docs]
        except Exception as e:
            log.warning("Retrieval failed: %s", e)

    # Inject RAG context into system message
    sys_prompt = build_system_prompt(contexts)
    filtered = [m for m in messages if m.get("role") != "system"]
    filtered.insert(0, {"role": "system", "content": sys_prompt})

    log.info("Query: %.60s | Contexts: %d | Stream: %s", query.replace("\n", " "), len(contexts), stream)

    # Forward to selected backend
    try:
        result = await backend.chat(filtered, stream=stream)
    except Exception as e:
        log.error("LLM error: %s", e)
        return Response(
            content=json.dumps({"error": {"message": str(e), "type": "upstream_error"}}),
            status_code=502,
            media_type="application/json",
        )

    if stream:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: wrap OpenAI format
    body = await result
    choice = body.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    return {
        "id": "rag-proxy",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": body.get("usage", {}),
    }


# ── Entry ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
