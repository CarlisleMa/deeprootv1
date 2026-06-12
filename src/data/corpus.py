"""Historical corpus loader and chunker.

Loads historical medical text files, segments them into passages,
and provides them to the Extraction Agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


CORPUS_DIR = Path("data/historical_corpus")


def load_corpus(corpus_dir: Path | str = CORPUS_DIR) -> list[dict]:
    """Load all text files from the corpus directory.

    Returns list of dicts:
        {"text": str, "source_document": str, "file_path": str}
    """
    corpus_dir = Path(corpus_dir)
    documents = []
    for fp in sorted(corpus_dir.glob("*.txt")):
        text = fp.read_text(encoding="utf-8")
        documents.append({
            "text": text,
            "source_document": fp.stem,
            "file_path": str(fp),
        })
    return documents


def chunk_document(
    document: dict, max_tokens: int = 2000, overlap: int = 200
) -> list[dict]:
    """Split a document into overlapping chunks for LLM processing.

    Args:
        document: Dict with "text" and "source_document" keys.
        max_tokens: Approximate max tokens per chunk (using ~4 chars/token).
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of passage dicts ready for the Extraction Agent.
    """
    text = document["text"]
    max_chars = max_tokens * 4
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + max_chars
        # Try to break at paragraph or sentence boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", ", "]:
                boundary = text.rfind(sep, start, end)
                if boundary > start:
                    end = boundary + len(sep)
                    break

        chunks.append({
            "text": text[start:end].strip(),
            "source_document": document["source_document"],
            "chunk_id": f"{document['source_document']}_chunk_{chunk_id}",
        })
        start = end - overlap
        chunk_id += 1

    return chunks
