# rag-kb

**本地知识库问答系统** — 下载即用，支持 Ollama / LM Studio / OpenAI 多后端，配合 Chatbox 使用。

## 功能

- **RAG 检索增强生成**：将 PDF 导入向量数据库，提问时自动检索相关内容，让 LLM 基于你的文档回答
- **多后端切换**：一个 `.env` 文件切换 Ollama、LM Studio、OpenAI 等任意 OpenAI 兼容 API
- **纯离线可选**：搭配 Ollama 可实现完全本地运行，数据不出本机
- **Chatbox 兼容**：零配置 UI，设好 API 地址即可使用

## 快速开始

```bash
# 下载
git clone https://github.com/khusdasz-cmd/rag-kb.git
cd rag-kb

# 安装依赖
pip install -e .

# 编辑配置（首次运行会自动从 .env.example 生成）
# 按需修改 LLM_TYPE、API Key、模型名等
vi .env

# 把 PDF 丢进 docs/ 目录，导入向量库
python ingest.py

# 启动
python rag_proxy.py
```

然后打开 Chatbox → 设置 → AI 模型提供商 → **OpenAI API Compatible**

| 字段 | 值 |
|---|---|
| API Key | 任意（如 `sk-rag`） |
| API 地址 | `http://localhost:9124/v1` |
| 模型 | 按后端填写 |

**无需手动创建任何文件夹。** `.env` 首次自动生成，`docs/` 自动创建，`chroma_db/` 导入时自动生成。

## 配置

编辑 `.env` 文件切换后端：

### Ollama 模式（默认，完全本地）

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
OPENAI_MODEL=你的模型名
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
├── rag_kb/                 # 核心 Python 包
│   ├── rag_proxy.py        # RAG 代理服务（FastAPI）
│   ├── ingest.py           # PDF 导入脚本
│   └── embedder.py         # Ollama embedding 客户端
├── ingest.py               # 入口（委托给 rag_kb.ingest）
├── rag_proxy.py            # 入口（委托给 rag_kb.rag_proxy）
├── .env                    # 配置文件（自动生成，已 gitignore）
├── .env.example            # 配置模板
├── pyproject.toml          # 项目元数据与依赖
├── LICENSE                 # MIT 许可证
├── chroma_db/              # 向量数据库（导入时自动生成，已 gitignore）
└── docs/                   # 放 PDF 文件（自动创建，已 gitignore）
```

## 工作原理

```
Chatbox → RAG Proxy → ChromaDB 检索相关文档
                     → 注入上下文到 prompt
                     → 转发给 LLM 后端
                     → 返回带知识库内容的回答
```

- **LLM 后端可切换**：Ollama / LM Studio / OpenAI
- **Embedding 始终走本地 Ollama**：速度快、免费用、数据不出本机

## 隐私

- `.env`、`docs/`（含 PDF）、`chroma_db/` 均在 `.gitignore` 中，不会提交
- API Key 只存在本地 `.env` 文件
