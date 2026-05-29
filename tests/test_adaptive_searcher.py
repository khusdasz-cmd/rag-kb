"""Tests for adaptive_searcher.py — search strategies, feedback tuning, and parsing."""

import sqlite3

import pytest

from rag_kb.adaptive_searcher import (
    ADAPTATION_RULES,
    CLASSIFY_PROMPT,
    PARAM_BOUNDS,
    QUERY_TYPE_STRATEGY,
    AdaptiveSearcher,
    SearchResult,
    SearchStrategy,
    ensure_strategy_stats,
    parse_strategy_id,
    update_strategy_stats,
)


class TestSearchStrategy:
    def test_default_creation(self):
        s = SearchStrategy()
        assert s.search_type == "similarity"
        assert s.top_k == 4
        assert s.fetch_k == 20
        assert s.lambda_mult == 0.7
        assert s.score_threshold == 0.0

    def test_custom_strategy(self):
        s = SearchStrategy(search_type="mmr", top_k=6, fetch_k=25, lambda_mult=0.5)
        assert s.search_type == "mmr"
        assert s.top_k == 6

    def test_strategy_id_format(self):
        s = SearchStrategy(search_type="mmr", top_k=6, fetch_k=25, lambda_mult=0.5)
        sid = s.strategy_id()
        assert "mmr" in sid
        assert "k6" in sid
        assert "f25" in sid
        assert "l0.5" in sid


class TestSearchResult:
    def test_creation(self):
        r = SearchResult(
            contexts=["doc1", "doc2"],
            doc_count=2,
            strategy_id="mmr_k6_f25_l0.5_t0.0",
            params={"top_k": 6},
        )
        assert len(r.contexts) == 2
        assert r.doc_count == 2


class TestConstants:
    def test_QUERY_TYPE_STRATEGY_has_keys(self):
        for k in ("factual", "overview", "precise"):
            assert k in QUERY_TYPE_STRATEGY

    def test_ADAPTATION_RULES_has_keys(self):
        for k in ("retrieval_miss", "answer_wrong", "too_long", "not_helpful", "other"):
            assert k in ADAPTATION_RULES

    def test_PARAM_BOUNDS_has_keys(self):
        for k in ("top_k", "fetch_k", "lambda_mult", "score_threshold"):
            assert k in PARAM_BOUNDS

    def test_PARAM_BOUNDS_top_k(self):
        lo, hi = PARAM_BOUNDS["top_k"]
        assert lo == 1
        assert hi == 15

    def test_CLASSIFY_PROMPT_has_placeholder(self):
        assert "{query}" in CLASSIFY_PROMPT

    def test_REWRITE_PROMPT_has_placeholder(self):
        assert "{query}" in AdaptiveSearcher.REWRITE_PROMPT


class TestParseStrategyId:
    def test_full_strategy(self):
        result = parse_strategy_id("mmr_k6_f25_l0.5_t0.15")
        assert result["search_type"] == "mmr"
        assert result["top_k"] == 6
        assert result["fetch_k"] == 25
        assert result["lambda_mult"] == 0.5
        assert result["score_threshold"] == 0.15

    def test_empty_returns_default(self):
        result = parse_strategy_id("")
        assert result["search_type"] == "similarity"
        assert result["top_k"] == 4

    def test_none_returns_default(self):
        result = parse_strategy_id(None)
        assert result["search_type"] == "similarity"

    def test_partial_strategy(self):
        """Only search_type and top_k provided — only those keys in result."""
        result = parse_strategy_id("similarity_k8")
        assert result["search_type"] == "similarity"
        assert result["top_k"] == 8
        # Keys not present in the string won't be in the result dict
        assert "fetch_k" not in result

    def test_unknown_search_type_returns_default(self):
        result = parse_strategy_id("unknown_k6")
        assert result["search_type"] == "similarity"

    def test_malformed_k_value_ignored(self):
        """k_abc can't be parsed as int → top_k not added to result."""
        result = parse_strategy_id("similarity_k_abc")
        assert "top_k" not in result
        assert result["search_type"] == "similarity"


class TestAdaptiveSearcher:
    def test_init(self):
        searcher = AdaptiveSearcher()
        assert searcher.llm_backend is None
        assert searcher._classify_cache == {}

    def test_init_with_backend(self):
        searcher = AdaptiveSearcher(llm_backend="dummy")
        assert searcher.llm_backend == "dummy"

    # ── _extract_content ─────────────────────────────────────────────────

    def test_extract_content_openai_format(self):
        searcher = AdaptiveSearcher()
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        assert searcher._extract_content(resp) == "Hello"

    def test_extract_content_ollama_format(self):
        searcher = AdaptiveSearcher()
        resp = {"message": {"content": "World"}}
        assert searcher._extract_content(resp) == "World"

    def test_extract_content_openai_preferred(self):
        """OpenAI format takes precedence over Ollama format."""
        searcher = AdaptiveSearcher()
        resp = {
            "choices": [{"message": {"content": "OpenAI"}}],
            "message": {"content": "Ollama"},
        }
        assert searcher._extract_content(resp) == "OpenAI"

    def test_extract_content_empty(self):
        searcher = AdaptiveSearcher()
        assert searcher._extract_content({}) == ""

    def test_extract_content_non_dict(self):
        searcher = AdaptiveSearcher()
        assert searcher._extract_content("raw string") == "raw string"

    def test_extract_content_no_choices(self):
        searcher = AdaptiveSearcher()
        resp = {"choices": [{}]}
        assert searcher._extract_content(resp) == ""

    # ── _apply_type_overrides ─────────────────────────────────────────────

    def test_apply_factual_overrides(self):
        searcher = AdaptiveSearcher()
        base = SearchStrategy()
        result = searcher._apply_type_overrides("factual", base)
        assert result.search_type == "similarity"
        assert result.top_k == 5

    def test_apply_overview_overrides(self):
        searcher = AdaptiveSearcher()
        base = SearchStrategy()
        result = searcher._apply_type_overrides("overview", base)
        assert result.search_type == "mmr"
        assert result.top_k == 6

    def test_apply_precise_overrides(self):
        searcher = AdaptiveSearcher()
        base = SearchStrategy()
        result = searcher._apply_type_overrides("precise", base)
        assert result.search_type == "similarity_score_threshold"
        assert result.score_threshold == 0.15

    def test_apply_unknown_type_preserves_base(self):
        searcher = AdaptiveSearcher()
        base = SearchStrategy(search_type="mmr", top_k=8)
        result = searcher._apply_type_overrides("invalid_type", base)
        assert result.search_type == "mmr"
        assert result.top_k == 8

    def test_overrides_do_not_mutate_base(self):
        searcher = AdaptiveSearcher()
        base = SearchStrategy(top_k=4)
        searcher._apply_type_overrides("factual", base)
        assert base.top_k == 4  # unchanged


class TestStrategyStats:
    @pytest.fixture
    def db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
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
                last_used TEXT
            )"""
        )
        yield conn
        conn.close()

    def test_ensure_creates_new(self, db):
        ensure_strategy_stats(db, "test_id_1")
        row = db.execute(
            "SELECT * FROM strategy_stats WHERE strategy_id = ?", ("test_id_1",)
        ).fetchone()
        assert row is not None
        assert row["query_count"] == 0

    def test_ensure_idempotent(self, db):
        ensure_strategy_stats(db, "dup_id")
        ensure_strategy_stats(db, "dup_id")
        rows = db.execute(
            "SELECT * FROM strategy_stats WHERE strategy_id = ?", ("dup_id",)
        ).fetchall()
        assert len(rows) == 1

    def test_update_thumbs_up(self, db):
        update_strategy_stats(db, "s1", rating=1)
        row = db.execute(
            "SELECT * FROM strategy_stats WHERE strategy_id = ?", ("s1",)
        ).fetchone()
        assert row["thumbs_up"] == 1
        assert row["thumbs_down"] == 0
        assert row["query_count"] == 1

    def test_update_thumbs_down(self, db):
        update_strategy_stats(db, "s2", rating=-1)
        row = db.execute(
            "SELECT * FROM strategy_stats WHERE strategy_id = ?", ("s2",)
        ).fetchone()
        assert row["thumbs_up"] == 0
        assert row["thumbs_down"] == 1

    def test_update_none_rating_does_nothing(self, db):
        """None rating returns early — no row created."""
        update_strategy_stats(db, "s3", rating=None)
        row = db.execute(
            "SELECT * FROM strategy_stats WHERE strategy_id = ?", ("s3",)
        ).fetchone()
        assert row is None


class TestAdjustForFeedback:
    @pytest.fixture
    def db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                rating INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )"""
        )
        yield conn
        conn.close()

    def test_no_adjustment_when_no_feedback(self, db):
        """No feedback rows → no tuning."""
        searcher = AdaptiveSearcher()
        strategy = SearchStrategy()
        result, action, reason = searcher.adjust_for_feedback(
            "retrieval_miss", strategy, db
        )
        assert result is None

    def test_no_adjustment_when_not_enough_downvotes(self, db):
        """Only 1 thumbs-down → below threshold of 2."""
        db.execute(
            "INSERT INTO feedback (rating) VALUES (?), (?), (?)",
            (-1, 1, 1),
        )
        searcher = AdaptiveSearcher()
        strategy = SearchStrategy()
        result, action, reason = searcher.adjust_for_feedback(
            "retrieval_miss", strategy, db
        )
        assert result is None

    def test_adjusts_when_enough_downvotes(self, db):
        """2+ thumbs-down in last 20 → triggers tuning."""
        for _ in range(3):
            db.execute("INSERT INTO feedback (rating) VALUES (-1)")
        db.commit()

        searcher = AdaptiveSearcher()
        strategy = SearchStrategy()
        result, action, reason = searcher.adjust_for_feedback(
            "retrieval_miss", strategy, db
        )
        assert result is not None
        assert len(action) > 0
        assert len(reason) > 0

    def test_adjust_top_k_increases(self, db):
        for _ in range(3):
            db.execute("INSERT INTO feedback (rating) VALUES (-1)")
        db.commit()

        searcher = AdaptiveSearcher()
        strategy = SearchStrategy(top_k=4)
        result, action, reason = searcher.adjust_for_feedback(
            "retrieval_miss", strategy, db
        )
        assert result is not None
        assert result.top_k > 4

    def test_adjust_changes_search_type_to_mmr(self, db):
        for _ in range(3):
            db.execute("INSERT INTO feedback (rating) VALUES (-1)")
        db.commit()

        searcher = AdaptiveSearcher()
        strategy = SearchStrategy(search_type="similarity")
        result, action, reason = searcher.adjust_for_feedback(
            "retrieval_miss", strategy, db
        )
        assert result is not None
        assert result.search_type == "mmr"

    def test_unknown_category_returns_none(self, db):
        searcher = AdaptiveSearcher()
        strategy = SearchStrategy()
        result, action, reason = searcher.adjust_for_feedback(
            "unknown_category", strategy, db
        )
        assert result is None
