from pathlib import Path

from meteo_ml.rag import answer_question, chunk_text, retrieve


def test_chunk_text_splits_markdown_table_rows() -> None:
    text = "| Station | Value |\n|---|---:|\n| Kochi | 2.0 |"
    chunks = chunk_text(text)
    assert "| Station | Value |" in chunks
    assert "| Kochi | 2.0 |" in chunks


def test_retrieve_returns_cited_chunks() -> None:
    sources = {"demo.md": "Heavy-rain days summarize rainfall extremes for scenario users."}
    chunks = retrieve("heavy rain scenario users", sources)
    assert chunks
    assert chunks[0].source == "demo.md"


def test_answer_question_uses_project_sources() -> None:
    answer = answer_question("How should users interpret heavy-rain scenario indicators?", Path.cwd())
    assert "Retrieved answer evidence" in answer
    assert "source" not in answer.lower() or "[" in answer
