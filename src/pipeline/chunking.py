"""Sentence-window chunking for customer reviews.

Reviews in ``samples.bakehouse`` are short (usually one paragraph), so most
become a single chunk. Longer reviews are split on sentence boundaries into
windows of at most ``max_words`` words, with ``overlap_sentences`` sentences
repeated between neighbouring chunks so a complaint that spans two sentences
is not cut in half. Splitting on sentences (rather than a fixed token count)
keeps each chunk readable when it is shown to the user as a citation.

This module is pure Python so it can be unit tested without Spark; the ingest
notebook wraps ``chunk_text`` in a PySpark UDF.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Collapse whitespace and strip; ``None`` becomes an empty string."""
    return _WHITESPACE.sub(" ", text or "").strip()


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(clean_text(text)) if s]


def chunk_text(text: str | None, max_words: int = 120, overlap_sentences: int = 1) -> list[str]:
    """Split ``text`` into sentence windows of at most ``max_words`` words.

    A single sentence longer than ``max_words`` is split on word boundaries.
    """
    if max_words < 1:
        raise ValueError("max_words must be positive")
    sentences: list[str] = []
    for sentence in split_sentences(text or ""):
        words = sentence.split(" ")
        if len(words) <= max_words:
            sentences.append(sentence)
        else:
            sentences.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
    if not sentences:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        end, words = start, 0
        while end < len(sentences):
            n = len(sentences[end].split(" "))
            if words and words + n > max_words:
                break
            words += n
            end += 1
        chunks.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break
        # Step forward, keeping up to ``overlap_sentences`` sentences of context,
        # but always make progress.
        start = max(end - overlap_sentences, start + 1)
    return chunks
