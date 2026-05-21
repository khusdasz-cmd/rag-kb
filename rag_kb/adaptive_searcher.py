"""Adaptive search engine — dispatches ChromaDB search strategies and self-tunes from feedback."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Literal, Optional

SearchType = Literal["similarity", "mmr", "similarity_score_threshold", "similarity_with_score"]
QueryType = Literal["factual", "overview", "precise"]

log = logging.getLogger("adaptive-searcher")


@dataclass
class SearchStrategy:
    """Parameters that define a retrieval strategy."""
    search_type: SearchType = "similarity"
    top_k: int = 4
    fetch_k: int = 20          # MMR candidate pool
    lambda_mult: float = 0.7   # MMR: 0=最多样, 1=最相关
    score_threshold: float = 0.0  # similarity_score_threshold 最低分

    def strategy_id(self) -> str:
        return (f"{self.search_type}_k{self.top_k}"
                f"_f{self.fetch_k}_l{self.lambda_mult}"
                f"_t{self.score_threshold}")


@dataclass
class SearchResult:
    contexts: list[str]
    doc_count: int
    strategy_id: str
    params: dict[str, Any]


# ── Query type classification ────────────────────────────────────────────────

CLASSIFY_PROMPT = (
    "Classify the following question into ONE type:\n"
    "- factual: 事实性查询，需要精确的、具体的答案（如日期、名称、数字、定义）\n"
    "- overview: 综述性查询，需要全面的、概括性的答案（如概述、总结、介绍、对比）\n"
    "- precise: 精确定位查询，需要从大量文本中找到特定信息（如原理、原因、机制）\n\n"
    "Question: {query}\n\n"
    "Type (just output factual/overview/precise):"
)

QUERY_TYPE_STRATEGY: dict[str, dict] = {
    "factual": {"search_type": "similarity", "top_k": 5},
    "overview": {"search_type": "mmr", "top_k": 6, "fetch_k": 25, "lambda_mult": 0.5},
    "precise": {"search_type": "similarity_score_threshold", "top_k": 6, "score_threshold": 0.15},
}

# ── Auto-tuning rules ────────────────────────────────────────────────────────

ADAPTATION_RULES: dict[str, list[tuple[str, Any, str]]] = {
    "retrieval_miss": [
        ("top_k", "+2", "检索遗漏，增加检索数量"),
        ("search_type", "mmr", "检索遗漏，切换到 MMR 增加覆盖率"),
    ],
    "answer_wrong": [
        ("top_k", "+1", "答案有误，增加上下文数量"),
    ],
    "too_long": [
        ("top_k", "-1", "回答冗长，减少上下文"),
    ],
    "not_helpful": [
        ("lambda_mult", "-0.1", "回答无帮助，增加多样性"),
    ],
    "other": [
        ("top_k", "+1", "保守增加上下文"),
    ],
}

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "top_k": (1, 15),
    "fetch_k": (5, 50),
    "lambda_mult": (0.3, 0.9),
    "score_threshold": (0.0, 0.5),
}


# ── AdaptiveSearcher ─────────────────────────────────────────────────────────

class AdaptiveSearcher:
    """Dispatch to the right ChromaDB search method and auto-tune from feedback."""

    def __init__(self, llm_backend: Optional[Any] = None):
        self.llm_backend = llm_backend
        self._classify_cache: dict[str, str] = {}

    # ── Search dispatch ───────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        vectorstore: Any,
        strategy: SearchStrategy,
        classify: bool = False,
        rewrite: bool = False,
    ) -> SearchResult:
        """Run retrieval with the given strategy.

        Args:
            rewrite: If True, expand query via LLM for better document matching.
            classify: If True, classify query type and adjust strategy.
        """
        search_query = query
        if rewrite and query.strip():
            search_query = await self._rewrite_query(query)

        if classify and query.strip():
            qtype = await self._classify_query(query)
            strategy = self._apply_type_overrides(qtype, strategy)
            log.info("Query classified as '%s', adjusted strategy: %s", qtype, strategy.strategy_id())

        docs = await self._execute_search(search_query, vectorstore, strategy)

        # Fallback guard: if strategy returned nothing, try basic similarity
        if not docs:
            log.warning("Strategy %s returned 0 docs, fallback to similarity k=2", strategy.strategy_id())
            docs = vectorstore.similarity_search(query, k=2)

        return SearchResult(
            contexts=[d.page_content for d in docs],
            doc_count=len(docs),
            strategy_id=strategy.strategy_id(),
            params=asdict(strategy),
        )

    async def _execute_search(self, query: str, vectorstore: Any, strategy: SearchStrategy) -> list:
        if strategy.search_type == "similarity":
            return vectorstore.similarity_search(query, k=strategy.top_k)

        if strategy.search_type == "mmr":
            return vectorstore.max_marginal_relevance_search(
                query, k=strategy.top_k,
                fetch_k=strategy.fetch_k,
                lambda_mult=strategy.lambda_mult,
            )

        if strategy.search_type == "similarity_score_threshold":
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
                query, k=strategy.top_k * 2,
            )
            filtered = [d for d, s in docs_with_scores if s >= strategy.score_threshold]
            return filtered[:strategy.top_k] if filtered else [d for d, s in docs_with_scores][:max(1, strategy.top_k // 2)]

        if strategy.search_type == "similarity_with_score":
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=strategy.top_k)
            docs_with_scores.sort(key=lambda x: x[1])
            return [d for d, _ in docs_with_scores]

        log.warning("Unknown search_type %s, fallback to similarity", strategy.search_type)
        return vectorstore.similarity_search(query, k=strategy.top_k)

    # ── Shared: extract text from backend response ───────────────────────

    def _extract_content(self, resp) -> str:
        """Parse LLM response — handles OpenAI-compatible and Ollama formats."""
        if not hasattr(resp, "get"):
            return str(resp)
        # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
        content_via_choices = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content_via_choices:
            return content_via_choices
        # Ollama format: {"message": {"content": "..."}}
        content_via_msg = resp.get("message", {}).get("content", "")
        return content_via_msg or ""

    # ── Query rewriting ──────────────────────────────────────────────────

    REWRITE_PROMPT = (
        "You are a search query optimizer. Rewrite the following user question "
        "to maximize retrieval accuracy in a RAG system.\n\n"
        "Rules:\n"
        "- Preserve the original intent\n"
        "- Add synonyms and terms likely to appear in academic/professional documents\n"
        "- Expand abbreviations and incomplete phrases\n"
        "- Output ONLY the rewritten query, nothing else\n"
        "- Keep under 200 characters\n\n"
        "Original: {query}\n"
        "Rewarded version:"
    )

    async def _rewrite_query(self, query: str) -> str:
        """Expand a short/natural-language query into a search-optimized one."""
        cache_key = f"rw:{query[:200]}"
        if cache_key in self._classify_cache:
            return self._classify_cache[cache_key]

        if not self.llm_backend:
            return query

        prompt = self.REWRITE_PROMPT.format(query=query[:300])
        try:
            resp = await self.llm_backend.chat(
                [{"role": "user", "content": prompt}], stream=False
            )
            content = self._extract_content(resp)
            rewritten = content.strip().strip('"\'')
            if rewritten and rewritten.lower() != query.lower():
                log.info("Query rewrite: '%s' → '%s'", query[:60], rewritten[:60])
                self._classify_cache[cache_key] = rewritten
                return rewritten
        except Exception:
            log.warning("Query rewrite failed, using original", exc_info=True)

        return query

    # ── Query classification ──────────────────────────────────────────────

    async def _classify_query(self, query: str) -> QueryType:
        cache_key = query[:200]
        if cache_key in self._classify_cache:
            return self._classify_cache[cache_key]

        if not self.llm_backend:
            return "factual"

        prompt = CLASSIFY_PROMPT.format(query=query[:300])
        try:
            resp = await self.llm_backend.chat(
                [{"role": "user", "content": prompt}], stream=False
            )
            content = self._extract_content(resp)
            qtype = content.strip().lower()
            if qtype in ("factual", "overview", "precise"):
                self._classify_cache[cache_key] = qtype
                return qtype
        except Exception:
            log.warning("Query classification failed, default to factual", exc_info=True)

        return "factual"

    def _apply_type_overrides(self, qtype: QueryType, base: SearchStrategy) -> SearchStrategy:
        overrides = QUERY_TYPE_STRATEGY.get(qtype, {})
        if not overrides:
            return base
        new = SearchStrategy(**asdict(base))
        for k, v in overrides.items():
            if hasattr(new, k):
                setattr(new, k, v)
        return new

    # ── Auto-tuning ───────────────────────────────────────────────────────

    def adjust_for_feedback(
        self,
        category: str,
        current: SearchStrategy,
        db_conn,
    ) -> tuple[Optional[SearchStrategy], str, str]:
        """Return (new_strategy, action_desc, reason) based on feedback category.

        Returns (None, "", "") if conditions are not met for adjustment.
        """
        rules = ADAPTATION_RULES.get(category)
        if not rules:
            return None, "", ""

        # Stability: only adjust if at least 2 thumbs-down in last 20 rated
        last_20 = db_conn.execute(
            "SELECT rating FROM feedback WHERE rating IS NOT NULL ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        down_count = sum(1 for r in last_20 if r["rating"] == -1)
        if down_count < 2:
            log.info("Skipping auto-tune: only %d thumbs-down in last 20", down_count)
            return None, "", ""

        new = SearchStrategy(**asdict(current))
        changes = []
        reasons = []

        for param, delta, reason_text in rules:
            if param == "search_type":
                if delta in ("mmr", "similarity", "similarity_score_threshold", "similarity_with_score"):
                    new.search_type = delta
                    changes.append(f"search_type: {current.search_type} → {delta}")
                    reasons.append(reason_text)
                    break

            elif param in ("top_k", "fetch_k"):
                old = getattr(new, param)
                new_val = max(
                    int(PARAM_BOUNDS[param][0]),
                    min(int(PARAM_BOUNDS[param][1]), old + int(delta)),
                )
                if new_val != old:
                    setattr(new, param, new_val)
                    changes.append(f"{param}: {old} → {new_val}")
                    reasons.append(reason_text)

            elif param == "lambda_mult":
                old = new.lambda_mult
                new_val = round(
                    max(PARAM_BOUNDS["lambda_mult"][0],
                        min(PARAM_BOUNDS["lambda_mult"][1], old + float(delta))),
                    1,
                )
                if new_val != old:
                    new.lambda_mult = new_val
                    changes.append(f"lambda_mult: {old} → {new_val}")
                    reasons.append(reason_text)

        if not changes:
            return None, "", ""

        return new, "; ".join(changes), "; ".join(reasons)


# ── Strategy stats tracking ──────────────────────────────────────────────────

def ensure_strategy_stats(conn, strategy_id: str):
    if not strategy_id:
        return
    # Parse strategy_id to get params
    conn.execute(
        """INSERT OR IGNORE INTO strategy_stats (strategy_id, search_type, top_k, fetch_k, lambda_mult, score_threshold, query_count, thumbs_up, thumbs_down)
           VALUES (?, 'similarity', 4, 20, 0.7, 0.0, 0, 0, 0)""",
        (strategy_id,),
    )
    # If row already existed, that's fine — INSERT OR IGNORE handles it


def update_strategy_stats(conn, strategy_id: str, rating: int):
    """Increment thumbs-up or thumbs-down for the given strategy."""
    if not strategy_id or rating is None:
        return
    # Make sure row exists
    conn.execute(
        """INSERT OR IGNORE INTO strategy_stats (strategy_id, search_type, top_k, fetch_k, lambda_mult, score_threshold, query_count, thumbs_up, thumbs_down)
           VALUES (?, 'similarity', 4, 20, 0.7, 0.0, 0, 0, 0)""",
        (strategy_id,),
    )
    # Update
    conn.execute(
        """UPDATE strategy_stats SET
           query_count = query_count + 1,
           thumbs_up = thumbs_up + ?,
           thumbs_down = thumbs_down + ?,
           last_used = datetime('now')
           WHERE strategy_id = ?""",
        (1 if rating == 1 else 0, 1 if rating == -1 else 0, strategy_id),
    )


def parse_strategy_id(sid: str) -> dict:
    """Parse 'mmr_k6_f25_l0.5_t0.0' → {'search_type': 'mmr', 'top_k': 6, ...}"""
    default = {"search_type": "similarity", "top_k": 4, "fetch_k": 20, "lambda_mult": 0.7, "score_threshold": 0.0}
    if not sid:
        return default
    parts = sid.split("_")
    if not parts or parts[0] not in ("similarity", "mmr", "similarity_score_threshold", "similarity_with_score"):
        return default
    result = {"search_type": parts[0]}
    for p in parts[1:]:
        if p.startswith("k"):
            try: result["top_k"] = int(p[1:])
            except: pass
        elif p.startswith("f"):
            try: result["fetch_k"] = int(p[1:])
            except: pass
        elif p.startswith("l"):
            try: result["lambda_mult"] = float(p[1:])
            except: pass
        elif p.startswith("t"):
            try: result["score_threshold"] = float(p[1:])
            except: pass
    return result
