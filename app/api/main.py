import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from app.core.rag_pipeline import ingest_document, rag_query
from app.agents.agent import run_agent

app = FastAPI(
    title="DocuMind API",
    description="AI-powered document research assistant with RAG and agent tools",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_histories = {}  # session_id -> list of messages


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    use_agent: Optional[bool] = False


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = []
    mode: str


@app.get("/")
def root():
    return {"message": "DocuMind API is running", "docs": "/docs"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed}")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "File too large (max 10MB)")

    result = ingest_document(content, file.filename)
    return {"message": f"Ingested {result['chunks']} chunks from '{result['filename']}'", **result}


@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    history = chat_histories.get(req.session_id, [])

    if req.use_agent:
        result = run_agent(req.question)
        return QueryResponse(answer=result["answer"], sources=[], mode="agent")

    result = rag_query(req.question, chat_history=history)

    # Update chat history
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": result["answer"]})
    chat_histories[req.session_id] = history[-12:]  # keep last 6 turns

    return QueryResponse(answer=result["answer"], sources=result["sources"], mode="rag")


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    chat_histories.pop(session_id, None)
    return {"message": f"Session '{session_id}' cleared"}


@app.get("/health")
def health():
    return {"status": "ok", "openai_configured": bool(os.getenv("OPENAI_API_KEY"))}
