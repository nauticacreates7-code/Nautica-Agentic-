"""Embedding and lexical similarity scoring for candidate source passages."""

import re
from typing import Any

from sentence_transformers import SentenceTransformer

from llm import ask_llm


# Instantiate once and reuse for all chunks.
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def _word_5grams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\b\w+\b", text.lower())
    return {tuple(words[i : i + 5]) for i in range(max(0, len(words) - 4))}


def _cosine_similarity(a: Any, b: Any) -> float:
    # SentenceTransformer returns normalized vectors when requested, so their
    # dot product is cosine similarity.
    return max(0.0, min(1.0, float(a @ b)))


def score_chunk(chunk_text: str, source_text: str) -> dict[str, int | str]:
    """Compare a paper chunk with a source passage and explain its score."""
    embeddings = _MODEL.encode(
        [chunk_text, source_text], convert_to_numpy=True, normalize_embeddings=True
    )
    embedding_similarity = _cosine_similarity(embeddings[0], embeddings[1])

    chunk_grams = _word_5grams(chunk_text)
    source_grams = _word_5grams(source_text)
    overlap = len(chunk_grams & source_grams) / len(chunk_grams) if chunk_grams else 0.0

    embedding_score = int(round(embedding_similarity * 100))
    ngram_score = int(round(overlap * 100))
    score = max(embedding_score, ngram_score)
    if overlap >= 0.35:
        reason = "Near-verbatim overlap"
    elif embedding_similarity >= 0.70:
        reason = "Likely paraphrased"
    else:
        reason = "Low similarity"

    if score > 60:
        prompt = (
            "In one sentence, is passage A copied, paraphrased, or just a common idea "
            "compared to passage B?\n\n"
            f"Passage A:\n{chunk_text}\n\nPassage B:\n{source_text}"
        )
        model_reason = ask_llm(prompt)
        if model_reason:
            reason = model_reason

    return {
        "score": score,
        "reason": reason,
        "embedding_score": embedding_score,
        "ngram_score": ngram_score,
    }


if __name__ == "__main__":
    copied = (
        "Neural networks learn representations by adjusting weights through "
        "backpropagation and gradient descent."
    )
    paraphrased = (
        "During training, a network changes its parameters using gradients "
        "propagated backward so it can form useful internal representations."
    )
    print("Copied sample:", score_chunk(copied, copied))
    print("Paraphrased sample:", score_chunk(copied, paraphrased))
