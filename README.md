# 🧠 DocuMind — AI Research Assistant

> Upload documents. Ask anything. Get cited answers powered by RAG + AI agents.

[![CI](https://github.com/MatinMusayev/documind/actions/workflows/ci.yml/badge.svg)](https://github.com/MatinMusayev/documind/actions)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-purple)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DocuMind is a production-ready AI document assistant that lets you upload PDFs and text files, then query them with natural language. It uses **Retrieval-Augmented Generation (RAG)** to ground answers in your documents and a **LangChain ReAct agent** with tool-calling for advanced multi-step reasoning.

---

## ✨ Features

- 📄 **Multi-document ingestion** — PDF, TXT, Markdown support
- 🔍 **Semantic search** — ChromaDB vector store with OpenAI embeddings
- 💬 **Multi-turn chat** — Session-aware conversation memory
- 🤖 **Agent mode** — ReAct agent with DocumentSearch, Summarizer, and optional Tavily web search
- 📎 **Source citations** — Every answer links back to the source document
- ⚡ **REST API** — Full FastAPI backend with Swagger docs
- 🖥️ **Streamlit UI** — Clean chat interface with file upload
- 🐳 **Docker ready** — One command to run everything

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│              Streamlit Chat App (:8501)                  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend (:8000)                 │
│   POST /upload    POST /query    DELETE /session/{id}    │
└──────┬───────────────────────────────────────┬──────────┘
       │                                       │
┌──────▼──────────┐                  ┌─────────▼──────────┐
│   RAG Pipeline  │                  │   LangChain Agent   │
│                 │                  │                     │
│ 1. Chunk text   │                  │  ReAct reasoning    │
│ 2. Embed chunks │                  │  Tool selection     │
│ 3. Store in DB  │                  │  Multi-step tasks   │
│ 4. Retrieve k   │                  └──────────┬──────────┘
│ 5. LLM answer   │                             │
└──────┬──────────┘                  ┌──────────▼──────────┐
       │                             │       Tools          │
┌──────▼──────────┐                  │ - DocumentSearch    │
│   ChromaDB      │◄─────────────────│ - Summarizer        │
│  Vector Store   │                  │ - Tavily Web Search │
└─────────────────┘                  └─────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/documind.git
cd documind
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run

**Terminal 1 — API server:**
```bash
uvicorn app.api.main:app --reload --port 8000
```

**Terminal 2 — UI:**
```bash
streamlit run app/ui/streamlit_app.py
```

Open `http://localhost:8501` → upload a PDF → start asking questions!

---

## 🐳 Docker

```bash
docker compose up --build
```

API: `http://localhost:8000/docs` | UI: `http://localhost:8501`

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a PDF/TXT/MD document |
| `POST` | `/query` | Ask a question (RAG or agent mode) |
| `DELETE` | `/session/{id}` | Clear chat history for a session |
| `GET` | `/health` | Service health check |

**Example:**
```bash
# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@research_paper.pdf"

# Query it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "session_id": "abc123"}'
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-3-small |
| Vector Store | ChromaDB |
| Agent Framework | LangChain ReAct |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF |
| Web Search (optional) | Tavily API |
| CI | GitHub Actions |
| Deployment | Docker |

---

## 📁 Project Structure

```
documind/
├── app/
│   ├── api/main.py          # FastAPI routes
│   ├── core/rag_pipeline.py # Chunking, embedding, retrieval, generation
│   ├── agents/agent.py      # LangChain ReAct agent + tools
│   └── ui/streamlit_app.py  # Chat interface
├── tests/
│   └── test_rag.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔮 Roadmap

- [ ] Multi-document comparison mode
- [ ] RAGAS evaluation metrics (faithfulness, relevance)
- [ ] Support for DOCX and HTML files
- [ ] Hugging Face Spaces deployment
- [ ] Streaming responses via SSE

---

## 📄 License

MIT — see [LICENSE](LICENSE)
