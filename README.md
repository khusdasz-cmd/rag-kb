# LLM-RAG-KB

**本地知识库问答系统** — 支持 Ollama / LM Studio / OpenAI 多后端，配合 Chatbox 使用。

## 功能

- **RAG 检索增强生成**：将 PDF 文档导入向量数据库，提问时自动检索相关内容，让 LLM 基于你的文档回答问题
- **多后端支持**：一个 `.env` 文件切换 Ollama、LM Studio、OpenAI 等任意 OpenAI 兼容 API
- **纯本地可选**：搭配 Ollama 可实现完全离线，数据不出本机
- **零配置 UI**：兼容 Chatbox，设好 API 地址即可使用

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 把 PDF 放入 docs/ 目录，然后导入
python ingest.py

# 启动代理
python rag_proxy.py
```

Chatbox → 设置 → AI 模型提供商 → **OpenAI API Compatible**

| 字段 | 值 |
|---|---|
| API Key | 任意（如 `sk-rag`） |
| API 地址 | `http://localhost:9124/v1` |
| 模型 | 按后端填写 |

## 配置

所有配置在 `.env` 文件中（复制 `.env.example` 为 `.env` 后修改）：

### Ollama 模式（默认）

```env
LLM_TYPE=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_EMBED_MODEL=bge-m3:567m
```

### LM Studio 模式

```env
LLM_TYPE=openai
OPENAI_URL=http://127.0.0.1:1234/v1
OPENAI_KEY=not-needed
OPENAI_MODEL=your-model-name
```

### OpenAI / 任意 API

```env
LLM_TYPE=openai
OPENAI_URL=https://api.openai.com/v1
OPENAI_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
```

## 项目结构

```
├── rag_proxy.py          # RAG 代理服务（核心）
├── ingest.py             # PDF 导入脚本
├── .env                  # 配置文件（已 gitignore）
├── .env.example          # 配置模板
├── requirements.txt      # 依赖
├── chroma_db/            # 向量数据库（自动生成，已 gitignore）
└── docs/                 # 放 PDF 文件（已 gitignore）
```

## 工作原理

```
Chatbox 提问 → RAG Proxy → ChromaDB 检索相关文档
                           → 注入上下文到 prompt
                           → 转发给 LLM (Ollama/LM Studio/OpenAI)
                           → 返回带知识库内容的回答
```

- **LLM 后端**：可切换（Ollama 本地 / LM Studio 本地 / OpenAI 远程）
- **Embedding 始终本地**：检索用 Ollama 做向量化，速度快、不花钱、数据不出本机

## 隐私

- `.env`、`docs/`、`chroma_db/` 默认在 `.gitignore` 中，不会提交到 GitHub
- API Key 等敏感信息只存储在本地 `.env` 文件
