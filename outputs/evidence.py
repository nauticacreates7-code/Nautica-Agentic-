"""Deterministic evidence extraction for a chunk/source comparison."""

import re
from difflib import SequenceMatcher

import numpy as np

from similarity import _MODEL


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "may", "might", "more", "most", "must", "no", "not", "of", "on", "or", "our",
    "ours", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "too", "under",
    "up", "us", "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your",
}
BOILERPLATE = (
    "in this paper we",
    "state of the art",
    "has been shown to",
    "this paper presents",
    "the results show that",
    "we propose a novel",
)


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]


def _citations(chunk_text: str) -> list[str]:
    patterns = (
        r"\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]",
        r"\([A-Z][\w'’-]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}[a-z]?\)",
        r"\bet\s+al\.?(?:\s*,?\s*(?:19|20)\d{2}[a-z]?)?",
    )
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, chunk_text, re.IGNORECASE):
            value = match.group(0).strip()
            if value not in found:
                found.append(value)
    return found


def _alignment(chunk_text: str, source_text: str) -> list[dict]:
    chunk_sentences = _sentences(chunk_text)
    source_sentences = _sentences(source_text)
    if not chunk_sentences or not source_sentences:
        return []

    embeddings = _MODEL.encode(
        chunk_sentences + source_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    chunk_vectors = embeddings[: len(chunk_sentences)]
    source_vectors = embeddings[len(chunk_sentences) :]
    pairs = []
    for sentence, vector in zip(chunk_sentences, chunk_vectors):
        similarities = np.asarray(source_vectors @ vector, dtype=float)
        best_index = int(np.argmax(similarities))
        chunk_words = set(_words(sentence))
        source_words = set(_words(source_sentences[best_index]))
        overlap_pct = 100 * len(chunk_words & source_words) / len(chunk_words) if chunk_words else 0
        pairs.append(
            {
                "chunk_sentence": sentence,
                "source_sentence": source_sentences[best_index],
                "cosine": round(max(0.0, min(1.0, float(similarities[best_index]))), 4),
                "word_overlap_pct": round(overlap_pct, 1),
            }
        )
    return sorted(pairs, key=lambda pair: pair["cosine"], reverse=True)[:3]


def _is_generic(block_text: str) -> bool:
    tokens = _words(block_text)
    if tokens and sum(token in STOP_WORDS for token in tokens) / len(tokens) >= 0.6:
        return True
    normalized = " ".join(tokens)
    return any(phrase in normalized for phrase in BOILERPLATE)


def build_evidence(
    chunk_text: str,
    source_text: str,
    source_title: str,
    source_url: str,
    embedding_score: int,
) -> dict:
    """Return reproducible lexical, sentence-alignment, and citation evidence."""
    chunk_words = _words(chunk_text)
    source_words = _words(source_text)
    chunk_5grams = {tuple(chunk_words[i : i + 5]) for i in range(max(0, len(chunk_words) - 4))}
    source_5grams = {tuple(source_words[i : i + 5]) for i in range(max(0, len(source_words) - 4))}
    verbatim_ngram_count = len(chunk_5grams & source_5grams)

    matcher = SequenceMatcher(None, chunk_words, source_words, autojunk=False)
    matched_blocks = []
    matched_positions = set()
    for block in matcher.get_matching_blocks():
        if block.size < 4:
            continue
        matched_blocks.append(
            {
                "text": " ".join(chunk_words[block.a : block.a + block.size]),
                "words": block.size,
            }
        )
        matched_positions.update(range(block.a, block.a + block.size))

    verbatim_word_pct = round(100 * len(matched_positions) / len(chunk_words), 1) if chunk_words else 0.0
    alignment = _alignment(chunk_text, source_text)
    best_pair = alignment[0] if alignment else None
    best_cosine = float(best_pair["cosine"]) if best_pair else 0.0
    best_overlap = float(best_pair["word_overlap_pct"]) if best_pair else 0.0
    paraphrase_signal = best_cosine >= 0.70 and best_overlap <= 40

    citations = _citations(chunk_text)
    generic_count = sum(1 for block in matched_blocks if _is_generic(block["text"]))
    generic_phrase_flag = bool(matched_blocks) and generic_count > len(matched_blocks) / 2
    has_citation = bool(citations)

    if verbatim_word_pct >= 25:
        category = "verbatim_copy"
    elif paraphrase_signal:
        category = "paraphrase"
    elif generic_phrase_flag or (best_cosine < 0.50 and verbatim_word_pct < 10):
        category = "common_or_generic"
    elif has_citation and (verbatim_ngram_count > 0 or matched_blocks):
        category = "cited_overlap"
    else:
        category = "unrelated"

    chunk_ngram_total = len(chunk_5grams)
    ngram_score = round(100 * verbatim_ngram_count / chunk_ngram_total) if chunk_ngram_total else 0
    embedding = max(0, min(100, int(embedding_score)))
    return {
        "verbatim_ngram_count": verbatim_ngram_count,
        "matched_blocks": matched_blocks,
        "longest_match_words": max((block["words"] for block in matched_blocks), default=0),
        "verbatim_word_pct": verbatim_word_pct,
        "sentence_alignment": alignment,
        "paraphrase_signal": paraphrase_signal,
        "has_citation": has_citation,
        "citation_strings": citations,
        "generic_phrase_flag": generic_phrase_flag,
        "category": category,
        "score_breakdown": {
            "embedding": embedding,
            "ngram": ngram_score,
            "final": max(embedding, ngram_score),
        },
    }
