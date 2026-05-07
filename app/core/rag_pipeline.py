import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    chunks = chunk_text(text)
    doc_id = hashlib.md5(file_bytes).hexdigest()

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return {"filename": filename, "chunks": len(chunks), "doc_id": doc_id}


def retrieve_context(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    results = collection.query(query_texts=[query], n_results=n_results)
    contexts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        contexts.append({"text": doc, "source": meta.get("filename", "unknown")})
    return contexts


def generate_answer(query: str, contexts: List[Dict[str, Any]], chat_history: List[Dict] = None) -> str:
    context_str = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in contexts
    )
    system_prompt = (
        "You are DocuMind, an expert AI research assistant. "
        "Answer questions using ONLY the provided document context. "
        "Always cite the source filename. If the answer is not in the context, say so clearly."
    )
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history[-6:])  # last 3 turns
    messages.append({
        "role": "user",
        "content": f"Context:\n{context_str}\n\nQuestion: {query}"
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )
    return response.choices[0].message.content


def rag_query(query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
    contexts = retrieve_context(query)
    answer = generate_answer(query, contexts, chat_history)
    return {
        "answer": answer,
        "sources": list({c["source"] for c in contexts}),
        "contexts": contexts
    }
