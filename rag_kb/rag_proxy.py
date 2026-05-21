"""RAG Proxy — Chatbox -> ChromaDB -> any LLM backend (Ollama / LM Studio / OpenAI).

Configure via .env — no code changes needed to switch backends.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from rag_kb.adaptive_searcher import AdaptiveSearcher, SearchStrategy, update_strategy_stats
from rag_kb.embedder import OllamaEmbedClient

load_dotenv()

# ── Config from .env ─────────────────────────────────────────────────────────
LLM_TYPE = os.getenv("LLM_TYPE", "ollama")
PROXY_PORT = int(os.getenv("PROXY_PORT", "9124"))
ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma_db"))
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = ROOT_DIR / CHROMA_DIR
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

# Runtime configuration (adjustable via API, no restart needed)
RUNTIME_CONFIG: dict = {
    "top_k": TOP_K,
    "search_type": "similarity",
    "fetch_k": 20,
    "lambda_mult": 0.7,
    "score_threshold": 0.0,
    "auto_tune": True,
    "query_classify": False,
    "query_rewrite": False,
}
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag-proxy")

# ── First-run setup ──────────────────────────────────────────────────────────
env_path = ROOT_DIR / ".env"
env_example = ROOT_DIR / ".env.example"
if not env_path.exists() and env_example.exists():
    shutil.copy(env_example, env_path)
    log.info("Created .env from .env.example -- edit it before use")

docs_dir = ROOT_DIR / "docs"
docs_dir.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Proxy")
vectorstore = None


# ── Feedback Database ────────────────────────────────────────────────────────


def get_db() -> sqlite3.Connection:
    db_path = ROOT_DIR / "feedback.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            query TEXT,
            answer TEXT,
            contexts TEXT,
            context_count INTEGER DEFAULT 0,
            rating INTEGER DEFAULT NULL,
            comment TEXT DEFAULT '',
            category TEXT DEFAULT '',
            strategy_id TEXT DEFAULT '',
            created_at TEXT
        )"""
    )
    # Migration: add strategy_id if missing (existing DBs)
    try:
        conn.execute("ALTER TABLE feedback ADD COLUMN strategy_id TEXT DEFAULT ''")
    except Exception:
        pass
    conn.execute(
        """CREATE TABLE IF NOT EXISTS improvements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            reason TEXT,
            old_value TEXT DEFAULT '',
            new_value TEXT DEFAULT '',
            created_at TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS strategy_stats (
            strategy_id TEXT PRIMARY KEY,
            search_type TEXT DEFAULT 'similarity',
            top_k INTEGER DEFAULT 4,
            fetch_k INTEGER DEFAULT 20,
            lambda_mult REAL DEFAULT 0.7,
            score_threshold REAL DEFAULT 0.0,
            query_count INTEGER DEFAULT 0,
            thumbs_up INTEGER DEFAULT 0,
            thumbs_down INTEGER DEFAULT 0,
            total_score REAL DEFAULT 0,
            avg_score REAL DEFAULT 0,
            first_used TEXT DEFAULT (datetime('now')),
            last_used TEXT
        )"""
    )
    return conn


def log_query(query: str, answer: str, contexts: list[str], strategy_id: str = "") -> str:
    qid = str(uuid.uuid4())[:8]
    conn = get_db()
    conn.execute(
        "INSERT INTO feedback (id, query, answer, contexts, context_count, strategy_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (qid, query[:500], answer[:2000], json.dumps(contexts, ensure_ascii=False), len(contexts), strategy_id, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return qid


async def auto_categorize(query: str, answer: str, contexts: list[str]) -> str:
    """Use LLM to categorize a negative feedback."""
    ctx_text = "\n".join(f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts[:3]))
    prompt = (
        "Based on the user's question and the system's answer below, classify the issue into ONE category:\n"
        "- answer_wrong: 回答内容有误\n"
        "- retrieval_miss: 没有找到相关资料\n"
        "- too_long: 回答过于冗长\n"
        "- not_helpful: 回答没有解决用户问题\n"
        "- other: 其他\n\n"
        f"Question: {query[:300]}\n"
        f"Retrieved contexts ({len(contexts)}):\n{ctx_text[:500]}\n"
        f"Answer: {answer[:500]}\n\n"
        "Category (just output the category name, nothing else):"
    )
    try:
        if LLM_TYPE == "openai":
            async with httpx.AsyncClient(timeout=15) as c:
                headers = {"Content-Type": "application/json"}
                if OPENAI_KEY:
                    headers["Authorization"] = f"Bearer {OPENAI_KEY}"
                resp = await c.post(
                    f"{OPENAI_URL}/chat/completions",
                    json={"model": OPENAI_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False, "temperature": 0},
                    headers=headers,
                    timeout=15,
                )
                cat = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            async with httpx.AsyncClient(timeout=15) as c:
                resp = await c.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0}},
                    timeout=15,
                )
                cat = resp.json()["message"]["content"].strip()
        return cat if cat in ("answer_wrong", "retrieval_miss", "too_long", "not_helpful", "other") else "other"
    except Exception:
        return "other"



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
            yield (
                f"data: {json.dumps({'id': 'rag-proxy', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': content} if content else {}, 'finish_reason': finish}]}, ensure_ascii=False)}\n\n"
            )
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
    log.info("Backend: OpenAI-compatible (%s -> %s)", OPENAI_URL, OPENAI_MODEL)
else:
    backend = OllamaBackend()
    log.info("Backend: Ollama (%s -> %s)", OLLAMA_URL, OLLAMA_MODEL)

# Adaptive search engine (feedbacks-driven strategy tuning)
adaptive_searcher = AdaptiveSearcher(llm_backend=backend)
log.info("Adaptive search: search_type=%s top_k=%d", RUNTIME_CONFIG["search_type"], RUNTIME_CONFIG["top_k"])
# ── Prompt ───────────────────────────────────────────────────────────────────


def build_system_prompt(contexts: list[str]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "You are a Q&A assistant based on a local knowledge base.\n\n"
        "Use the reference materials below to answer the question. "
        "If they are unrelated, answer based on your own knowledge.\n"
        "Answer concisely and accurately in Chinese.\n\n"
        "Reference materials:\n"
        f"{ctx}"
    ) if contexts else (
        "You are a knowledge base Q&A assistant. Answer concisely and accurately in Chinese."
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
            log.warning("ChromaDB load failed: %s -- RAG disabled", e)
    else:
        log.warning("ChromaDB not found -- RAG disabled (run ingest.py first)")


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


@app.get("/v1/config")
async def get_config():
    return {"top_k": RUNTIME_CONFIG["top_k"]}


@app.post("/v1/config")
async def set_config(request: Request):
    body = await request.json()
    if "top_k" in body:
        k = int(body["top_k"])
        if 1 <= k <= 50:
            RUNTIME_CONFIG["top_k"] = k
    return {"top_k": RUNTIME_CONFIG["top_k"], "message": "应用成功"}


@app.get("/v1/strategy")
async def get_strategy():
    return {
        "search_type": RUNTIME_CONFIG.get("search_type", "similarity"),
        "top_k": RUNTIME_CONFIG.get("top_k", TOP_K),
        "fetch_k": RUNTIME_CONFIG.get("fetch_k", 20),
        "lambda_mult": RUNTIME_CONFIG.get("lambda_mult", 0.7),
        "score_threshold": RUNTIME_CONFIG.get("score_threshold", 0.0),
        "auto_tune": RUNTIME_CONFIG.get("auto_tune", True),
        "query_classify": RUNTIME_CONFIG.get("query_classify", False),
        "query_rewrite": RUNTIME_CONFIG.get("query_rewrite", False),
    }


@app.post("/v1/strategy")
async def set_strategy(request: Request):
    body = await request.json()
    overrides = {}
    for k in ("search_type", "top_k", "fetch_k", "lambda_mult", "score_threshold", "auto_tune", "query_classify", "query_rewrite"):
        if k not in body:
            continue
        v = body[k]
        if k == "search_type" and v in ("similarity", "mmr", "similarity_score_threshold", "similarity_with_score"):
            RUNTIME_CONFIG[k] = v
            overrides[k] = v
        elif k in ("top_k", "fetch_k") and isinstance(v, (int, float)):
            clamped = max(1, min(50, int(v)))
            RUNTIME_CONFIG[k] = clamped
            overrides[k] = clamped
        elif k == "lambda_mult" and isinstance(v, (int, float)):
            clamped = round(max(0.0, min(1.0, float(v))), 1)
            RUNTIME_CONFIG[k] = clamped
            overrides[k] = clamped
        elif k == "score_threshold" and isinstance(v, (int, float)):
            clamped = round(max(0.0, min(1.0, float(v))), 2)
            RUNTIME_CONFIG[k] = clamped
            overrides[k] = clamped
        elif k in ("auto_tune", "query_classify", "query_rewrite") and isinstance(v, bool):
            RUNTIME_CONFIG[k] = v
            overrides[k] = v

    if overrides:
        conn = get_db()
        conn.execute(
            "INSERT INTO improvements (action, reason, old_value, new_value, created_at) VALUES (?, ?, ?, ?, ?)",
            ("手动调参", f"手动调整: {', '.join(overrides.keys())}",
             "", json.dumps(overrides, ensure_ascii=False), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        log.info("Strategy manually changed: %s", overrides)

    return await get_strategy()


@app.get("/v1/strategy/stats")
async def get_strategy_stats():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM strategy_stats ORDER BY query_count DESC"
    ).fetchall()
    conn.close()
    return {
        "current": RUNTIME_CONFIG.get("search_type", "similarity"),
        "strategies": [dict(r) for r in rows],
    }


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

    # Retrieve context (adaptive search)
    contexts = []
    strategy_id_used = ""
    if vectorstore and query.strip():
        try:
            strategy = SearchStrategy(
                search_type=RUNTIME_CONFIG.get("search_type", "similarity"),
                top_k=RUNTIME_CONFIG.get("top_k", TOP_K),
                fetch_k=RUNTIME_CONFIG.get("fetch_k", 20),
                lambda_mult=RUNTIME_CONFIG.get("lambda_mult", 0.7),
                score_threshold=RUNTIME_CONFIG.get("score_threshold", 0.0),
            )
            result = await adaptive_searcher.search(
                query, vectorstore, strategy,
                classify=RUNTIME_CONFIG.get("query_classify", False),
                rewrite=RUNTIME_CONFIG.get("query_rewrite", False),
            )
            contexts = result.contexts
            strategy_id_used = result.strategy_id
        except Exception as e:
            log.warning("Adaptive search failed, fallback: %s", e)
            try:
                docs = vectorstore.similarity_search(query, k=RUNTIME_CONFIG.get("top_k", TOP_K))
                contexts = [d.page_content for d in docs]
            except Exception as e2:
                log.warning("Fallback search also failed: %s", e2)

    # Inject RAG context into system message
    sys_prompt = build_system_prompt(contexts)
    filtered = [m for m in messages if m.get("role") != "system"]
    filtered.insert(0, {"role": "system", "content": sys_prompt})

    log.info("Query: %.60s | Contexts: %d | Strategy: %s | Stream: %s", query.replace("\n", " "), len(contexts), strategy_id_used or "none", stream)

    # Log query immediately (get an ID for it)
    qid = log_query(query, "", contexts, strategy_id=strategy_id_used)

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
        # Wrap generator to capture full answer for logging
        full_answer = []

        async def _logged_stream():
            async for chunk in result:
                full_answer.append(chunk)
                yield chunk
            # After stream completes, log the full answer
            full_text = ""
            for c in full_answer:
                try:
                    s = c.decode("utf-8") if isinstance(c, bytes) else c
                    if s.startswith("data: ") and s != "data: [DONE]\n\n":
                        data = json.loads(s[6:])
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        full_text += content
                except Exception:
                    pass
            if full_text:
                conn = get_db()
                conn.execute("UPDATE feedback SET answer=? WHERE id=?", (full_text[:2000], qid))
                conn.commit()
                conn.close()

        return StreamingResponse(
            _logged_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: wrap OpenAI format
    body = result if isinstance(result, dict) else (await result)
    choice = body.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    # Update log with answer
    conn = get_db()
    conn.execute("UPDATE feedback SET answer=? WHERE id=?", (content[:2000], qid))
    conn.commit()
    conn.close()
    return {
        "id": qid,
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": body.get("usage", {}),
    }


# ── Feedback Endpoints ───────────────────────────────────────────────────────


@app.post("/v1/feedback")
async def submit_feedback(request: Request):
    body = await request.json()
    qid = body.get("id", "")
    rating = body.get("rating")  # 1 or -1
    comment = body.get("comment", "")

    conn = get_db()
    conn.execute("UPDATE feedback SET rating=?, comment=? WHERE id=?", (rating, comment, qid))
    conn.commit()

    # Track per-strategy performance
    row = conn.execute("SELECT strategy_id FROM feedback WHERE id=?", (qid,)).fetchone()
    if row and row["strategy_id"] and rating is not None:
        update_strategy_stats(conn, row["strategy_id"], rating)

    # Auto-categorize negative feedback + auto-tune
    if rating == -1:
        row = conn.execute("SELECT query, answer, contexts, strategy_id FROM feedback WHERE id=?", (qid,)).fetchone()
        if row:
            cat = await auto_categorize(row["query"], row["answer"], json.loads(row["contexts"] or "[]"))
            conn.execute("UPDATE feedback SET category=? WHERE id=?", (cat, qid))
            conn.commit()

            # Auto-tune if enabled
            if RUNTIME_CONFIG.get("auto_tune", True):
                current = SearchStrategy(
                    search_type=RUNTIME_CONFIG.get("search_type", "similarity"),
                    top_k=RUNTIME_CONFIG.get("top_k", TOP_K),
                    fetch_k=RUNTIME_CONFIG.get("fetch_k", 20),
                    lambda_mult=RUNTIME_CONFIG.get("lambda_mult", 0.7),
                    score_threshold=RUNTIME_CONFIG.get("score_threshold", 0.0),
                )
                new_strategy, action, reason = adaptive_searcher.adjust_for_feedback(cat, current, conn)
                if new_strategy and action:
                    # Apply new strategy to runtime config
                    old_values = {}
                    for k in ("search_type", "top_k", "fetch_k", "lambda_mult", "score_threshold"):
                        old_values[k] = RUNTIME_CONFIG.get(k)
                        RUNTIME_CONFIG[k] = getattr(new_strategy, k, RUNTIME_CONFIG.get(k))
                    # Log improvement
                    conn.execute(
                        "INSERT INTO improvements (action, reason, old_value, new_value, created_at) VALUES (?, ?, ?, ?, ?)",
                        ("auto_tune", reason,
                         json.dumps(old_values, ensure_ascii=False),
                         json.dumps(asdict(new_strategy), ensure_ascii=False),
                         datetime.now().isoformat()),
                    )
                    conn.commit()
                    log.info("Auto-tune: %s | reason: %s", action, reason)
    conn.close()
    return {"status": "ok"}


@app.get("/v1/feedback/logs")
async def get_logs(limit: int = 50):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, query, substr(answer,1,200) as answer_preview, context_count, rating, comment, category, created_at "
        "FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/v1/feedback/summary")
async def get_summary():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    rated = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating IS NOT NULL").fetchone()[0]
    thumbs_up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating=1").fetchone()[0]
    thumbs_down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating=-1").fetchone()[0]

    cats = conn.execute(
        "SELECT category, COUNT(*) as cnt FROM feedback WHERE rating=-1 AND category!='' GROUP BY category ORDER BY cnt DESC"
    ).fetchall()

    # Daily trend
    daily = conn.execute(
        "SELECT date(created_at) as day, COUNT(*) as total, SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) as up, "
        "SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) as down "
        "FROM feedback GROUP BY day ORDER BY day DESC LIMIT 14"
    ).fetchall()
    conn.close()

    return {
        "total_queries": total,
        "rated": rated,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "thumbs_up_rate": round(thumbs_up / rated, 3) if rated else 0,
        "categories": [{"name": r["category"], "count": r["cnt"]} for r in cats],
        "daily": [dict(r) for r in daily],
    }


@app.get("/v1/feedback/insights")
async def get_insights():
    """Analyze feedback data and generate optimization suggestions."""
    conn = get_db()

    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    miss = conn.execute("SELECT COUNT(*) FROM feedback WHERE context_count=0").fetchone()[0]
    low_ctx = conn.execute("SELECT COUNT(*) FROM feedback WHERE context_count BETWEEN 1 AND 2").fetchone()[0]
    rated = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating IS NOT NULL").fetchone()[0]
    up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating=1").fetchone()[0]
    down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating=-1").fetchone()[0]

    cats = conn.execute(
        "SELECT category, COUNT(*) as cnt FROM feedback WHERE rating=-1 AND category!='' GROUP BY category ORDER BY cnt DESC"
    ).fetchall()

    trend = conn.execute(
        "SELECT date(created_at) as day, COUNT(*) as total, "
        "SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) as up, "
        "SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) as down "
        "FROM feedback GROUP BY day ORDER BY day DESC LIMIT 14"
    ).fetchall()
    conn.close()

    # Generate suggestions
    suggestions = []
    miss_rate = miss / total if total > 0 else 0
    low_ctx_rate = low_ctx / total if total > 0 else 0

    if miss_rate > 0.3:
        suggestions.append({
            "type": "danger",
            "issue": "检索漏检率偏高",
            "detail": f"最近 {total} 次查询中 {miss} 次未找到相关文档（{miss_rate:.0%}）",
            "suggestion": f"调大 TOP_K（当前 {RUNTIME_CONFIG['top_k']}），或检查 docs/ 目录是否包含相关文档，或重新导入",
        })
    elif miss_rate > 0.1:
        suggestions.append({
            "type": "warning",
            "issue": "检索漏检率中等",
            "detail": f"最近 {total} 次查询中 {miss} 次未找到相关文档（{miss_rate:.0%}）",
            "suggestion": "检查文档导入是否完整，或适当调大 TOP_K",
        })

    if low_ctx_rate > 0.2:
        suggestions.append({
            "type": "info",
            "issue": "检索块数偏少",
            "detail": f"{low_ctx} 次查询只检索到 1-2 个文档块（{low_ctx_rate:.0%}）",
            "suggestion": f"调高 TOP_K（当前 {RUNTIME_CONFIG['top_k']}）或优化文档 chunk 分割策略",
        })

    if down > 0:
        cat_list = [{"name": r["category"], "count": r["cnt"]} for r in cats]
        if cat_list:
            top_cat = max(cat_list, key=lambda c: c["count"])
            if top_cat["name"] == "answer_wrong":
                suggestions.append({
                    "type": "danger",
                    "issue": "回答质量差",
                    "detail": f"差评集中在「回答内容有误」（{top_cat['count']} 次）",
                    "suggestion": "建议切换到更强的 LLM 模型，或优化 system prompt 中的上下文使用方式",
                })
            elif top_cat["name"] == "retrieval_miss":
                suggestions.append({
                    "type": "warning",
                    "issue": "检索遗漏导致差评",
                    "detail": f"差评集中在「没有找到相关资料」（{top_cat['count']} 次）",
                    "suggestion": f"调高 TOP_K 或重新导入文档",
                })
            elif top_cat["name"] == "too_long":
                suggestions.append({
                    "type": "info",
                    "issue": "回答过于冗长",
                    "detail": f"差评包含「回答过于冗长」（{top_cat['count']} 次）",
                    "suggestion": "在 system prompt 中加入「简洁回答，直击要点」的约束",
                })
            elif top_cat["name"] == "not_helpful":
                suggestions.append({
                    "type": "warning",
                    "issue": "回答未解决用户问题",
                    "detail": f"差评包含「没有解决用户问题」（{top_cat['count']} 次）",
                    "suggestion": "检查文档内容与用户实际需求的匹配度",
                })

    if total < 10 and total > 0:
        suggestions.append({
            "type": "info",
            "issue": "数据量不足",
            "detail": f"仅 {total} 条查询记录",
            "suggestion": "建议积累更多数据后再做分析，数据越多建议越准确",
        })

    return {
        "total_queries": total,
        "retrieval_miss": {"count": miss, "rate": round(miss_rate, 3)},
        "low_context": {"count": low_ctx, "rate": round(low_ctx_rate, 3)},
        "rated": rated,
        "thumbs_up": up,
        "thumbs_down": down,
        "thumbs_up_rate": round(up / rated, 3) if rated else 0,
        "categories": [{"name": r["category"], "count": r["cnt"]} for r in cats],
        "daily_trend": [dict(r) for r in trend],
        "suggestions": suggestions,
    }


@app.post("/v1/feedback/improvements")
async def log_improvement(request: Request):
    """Log an improvement action taken based on feedback."""
    body = await request.json()
    conn = get_db()
    conn.execute(
        "INSERT INTO improvements (action, reason, old_value, new_value, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            body.get("action", ""),
            body.get("reason", ""),
            body.get("old_value", ""),
            body.get("new_value", ""),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/v1/feedback/improvements")
async def get_improvements(limit: int = 20):
    """Get improvement action history."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM improvements ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page():
    html = ROOT_DIR / "feedback.html"
    if html.exists():
        return HTMLResponse(html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>feedback.html not found</h1>")


# ── Entry ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
