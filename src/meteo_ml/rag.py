from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    score: float


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def load_text_sources(root: Path) -> dict[str, str]:
    """Load project texts used by the local RAG demo."""
    candidates = [
        root / "README.md",
        root / "docs/workflow.md",
        root / "docs/climate_scenario_extension.md",
        root / "products/extreme_rainfall_scenario_summary.md",
    ]
    return {str(path.relative_to(root)): path.read_text() for path in candidates if path.exists()}


def chunk_text(text: str, max_words: int = 90) -> list[str]:
    chunks: list[str] = []
    in_code_block = False
    clean_parts: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if line.strip():
            current.append(line)
        elif current:
            clean_parts.append("\n".join(current))
            current = []
    if current:
        clean_parts.append("\n".join(current))

    for paragraph in [part.strip() for part in clean_parts if part.strip()]:
        table_lines = [line.strip() for line in paragraph.splitlines() if line.strip().startswith("|")]
        if table_lines:
            chunks.extend(line for line in table_lines if not re.fullmatch(r"[|:\-\s]+", line))
            continue

        words = paragraph.split()
        if len(words) <= max_words:
            chunks.append(paragraph)
            continue
        for start in range(0, len(words), max_words):
            chunks.append(" ".join(words[start : start + max_words]))
    return chunks


def _idf(chunks: list[list[str]]) -> dict[str, float]:
    n_docs = len(chunks)
    vocabulary = {token for chunk in chunks for token in set(chunk)}
    return {
        token: math.log((1 + n_docs) / (1 + sum(token in set(chunk) for chunk in chunks))) + 1.0
        for token in vocabulary
    }


def retrieve(question: str, sources: dict[str, str], top_k: int = 3) -> list[RetrievedChunk]:
    """Retrieve relevant project text chunks with a compact TF-IDF scorer."""
    indexed: list[tuple[str, str, list[str]]] = []
    for source, text in sources.items():
        for chunk in chunk_text(text):
            tokens = tokenize(chunk)
            if tokens:
                indexed.append((source, chunk, tokens))

    if not indexed:
        return []

    chunk_tokens = [tokens for _, _, tokens in indexed]
    idf = _idf(chunk_tokens)
    query_tokens = tokenize(question)
    query_weights = {token: query_tokens.count(token) * idf.get(token, 0.0) for token in set(query_tokens)}

    scored: list[RetrievedChunk] = []
    for source, chunk, tokens in indexed:
        token_set = set(tokens)
        score = sum(query_weights.get(token, 0.0) * tokens.count(token) * idf.get(token, 0.0) for token in token_set)
        if score > 0.0:
            scored.append(RetrievedChunk(source=source, text=chunk, score=round(score, 3)))

    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def answer_question(question: str, root: Path) -> str:
    """Return a cited, extractive answer from project documentation."""
    sources = load_text_sources(root)
    chunks = retrieve(question, sources)
    if not chunks:
        return "I could not find a relevant passage in the project documentation."

    lines = [f"Question: {question}", "", "Retrieved answer evidence:"]
    for idx, chunk in enumerate(chunks, start=1):
        compact = " ".join(chunk.text.split())
        lines.append(f"{idx}. {compact} [{chunk.source}, score={chunk.score}]")
    lines.extend(
        [
            "",
            "Note: This is a local RAG-style demo. It retrieves evidence from project files; a production version could add embeddings, access control, evaluation, and an LLM response layer.",
        ]
    )
    return "\n".join(lines)
