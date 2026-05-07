import pytest
from unittest.mock import patch, MagicMock
from app.core.rag_pipeline import chunk_text, extract_text_from_pdf


def test_chunk_text_basic():
    text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


def test_chunk_text_small_input():
    text = "hello world"
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_text_overlap():
    words = [f"w{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    # Verify overlap: last words of chunk N appear in chunk N+1
    chunk0_words = set(chunks[0].split())
    chunk1_words = set(chunks[1].split())
    assert len(chunk0_words & chunk1_words) > 0


@patch("app.core.rag_pipeline.collection")
@patch("app.core.rag_pipeline.client")
def test_rag_query_returns_answer(mock_client, mock_collection):
    mock_collection.query.return_value = {
        "documents": [["chunk 1 text", "chunk 2 text"]],
        "metadatas": [[{"filename": "test.pdf"}, {"filename": "test.pdf"}]]
    }
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is the answer."
    mock_client.chat.completions.create.return_value = mock_response

    from app.core.rag_pipeline import rag_query
    result = rag_query("What is this about?")
    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "This is the answer."


@patch("app.core.rag_pipeline.collection")
def test_ingest_document_txt(mock_collection):
    mock_collection.add = MagicMock()
    from app.core.rag_pipeline import ingest_document
    content = b"This is a test document with some content to be chunked and embedded."
    result = ingest_document(content, "test.txt")
    assert result["filename"] == "test.txt"
    assert result["chunks"] >= 1
    mock_collection.add.assert_called_once()
