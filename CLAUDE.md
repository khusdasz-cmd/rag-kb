# rag-kb -- CLAUDE.md

## Project Overview

Local RAG knowledge base system -- ingest PDFs, store in ChromaDB, query via a Chatbox-compatible proxy. Supports Ollama / LM Studio / OpenAI backends.

- **GitHub**: https://github.com/khusdasz-cmd/rag-kb
- **Author**: Lin Haokang (khusdasz@gmail.com)
- **Python**: >= 3.9

## Build & Install

```bash
pip install -e .
```

## Project Structure

```
rag-kb/
├── rag_kb/
│   ├── ingest.py           # PDF ingestion pipeline
│   ├── rag_proxy.py        # FastAPI proxy server
│   └── embedder.py         # Ollama embedding client
├── docs/                   # Place PDFs here
├── .env                    # Configuration (auto-generated)
├── .env.example            # Config template
├── pyproject.toml
└── README.md
```

## Coding Conventions

- **Docstrings**: Google-style
- **Naming**: `snake_case` for functions, `PascalCase` for classes
- **Types**: Use type hints for all public function signatures
- **Imports**: Standard lib -> third-party -> local

## Git Workflow

- **Remote**: `origin` -> https://github.com/khusdasz-cmd/rag-kb.git
- **Branch**: work on feature branches, merge to master
- **Commit style**: Concise English, present tense
