# Contributing to rag-kb

Thank you for your interest in contributing to **rag-kb** — a local RAG (Retrieval-Augmented Generation) knowledge base system.

This document outlines the process for reporting bugs, suggesting features, and submitting changes.

---

## Table of Contents

- [How to Report a Bug](#how-to-report-a-bug)
- [How to Suggest a Feature](#how-to-suggest-a-feature)
- [How to Submit a Pull Request](#how-to-submit-a-pull-request)
- [Code Style](#code-style)
- [Development Setup](#development-setup)

---

## How to Report a Bug

If you encounter a bug, please [open a GitHub Issue](https://github.com/khusdasz-cmd/rag-kb/issues/new) and include:

1. **A clear, descriptive title**.
2. **Steps to reproduce** the issue — include the exact commands and input.
3. **Expected behaviour** vs **actual behaviour**.
4. **Environment details**:
   - Python version (`python --version`)
   - OS version
   - Embedding model and backend (e.g., Ollama + `nomic-embed-text`, LM Studio, OpenAI)
   - LLM model and backend used for generation
   - ChromaDB version (if installed separately)
   - `pip list | findstr chroma` / `pip list | grep chroma` output
   - Full error traceback (if applicable)
5. **Relevant configuration** — sanitise any secrets before pasting.
6. **Sample PDF** (if the bug is ingestion-related) — attach a minimal file that triggers the issue.

Before filing, search [existing issues](https://github.com/khusdasz-cmd/rag-kb/issues) to see if the problem has already been reported.

---

## How to Suggest a Feature

We welcome feature suggestions! To propose an enhancement:

1. [Open a Feature Request issue](https://github.com/khusdasz-cmd/rag-kb/issues/new).
2. Use a title that starts with `[Feature]` or `[Enhancement]`.
3. Describe **what you want to achieve** and **why** it would be useful.
4. If you have a concrete design in mind, include a sketch of the API, CLI flags, or configuration options.
5. Tag any relevant area (ingestion, retrieval, proxy, embedding).

For small or experimental ideas, consider opening a discussion thread first.

---

## How to Submit a Pull Request

rag-kb follows a **fork → branch → test → PR** workflow.

### Step 1: Fork and clone

```bash
git clone https://github.com/<your-username>/rag-kb.git
cd rag-kb
```

### Step 2: Create a feature branch

```bash
git checkout -b feat/my-feature
# or: fix/my-bugfix, docs/update-readme, refactor/...
```

Branch names should be short and descriptive, using kebab-case.

### Step 3: Make your changes

Keep changes **minimal and focused** — one pull request should address one concern. See [Code Style](#code-style) below.

### Step 4: Test

```bash
pip install -e ".[dev]"
pytest
```

Make sure all existing tests pass and add new tests for your changes. See [Development Setup](#development-setup) for details.

### Step 5: Commit

Write concise, present-tense commit messages:

```text
Add reranking step to ingestion pipeline
Fix OOM when processing large PDFs
```

### Step 6: Push and open a Pull Request

```bash
git push origin feat/my-feature
```

Then [open a PR](https://github.com/khusdasz-cmd/rag-kb/compare) against the `master` branch. In the PR description:

- Link to any related issues.
- Summarise what the change does.
- Note any breaking changes or migration steps.
- Tag a reviewer if you know who should look at it.

Your PR will be reviewed. Please address review feedback — it may take a few rounds before merging.

---

## Code Style

rag-kb follows standard Python conventions:

- **PEP 8** — Formatting should follow [PEP 8](https://peps.python.org/pep-0008/). Use a formatter like `ruff` or `black` (line length 88) if you like.
- **Type hints** — All public function signatures **must** include type annotations.
- **Docstrings** — Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- **Naming** — `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- **Imports** — Group imports in this order, separated by a blank line:
  1. Standard library
  2. Third-party packages
  3. Local application modules
- **Tests** — All new functionality should be covered by `pytest` tests. Place test files in the `tests/` directory with a `test_` prefix.
- **Lint** — Run `ruff check .` (or your preferred linter) before committing. The CI pipeline runs on every PR.

---

## Development Setup

### Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/)
- (Optional) [Ollama](https://ollama.ai/) or [LM Studio](https://lmstudio.ai/) for local embeddings

### Setup

```bash
# Clone the repository
git clone https://github.com/khusdasz-cmd/rag-kb.git
cd rag-kb

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\Activate.ps1   # Windows PowerShell

# Install rag-kb in editable mode with dev dependencies
pip install -e ".[dev]"

# Copy the environment template and edit as needed
cp .env.example .env
```

### Running tests

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_ingest.py -v
```

To run with coverage:

```bash
pytest --cov=rag_kb
```

### Checking code quality

```bash
ruff check .
ruff format --check .
```

---

## Questions?

Open a [discussion](https://github.com/khusdasz-cmd/rag-kb/discussions) or tag the maintainer in an issue. We're happy to help you get started!
