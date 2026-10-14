"""PDF extraction and scholarly source search used by the checker."""

import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from io import BytesIO
from typing import Any
from uuid import uuid4

import requests
from pypdf import PdfReader

from llm import ask_llm


SESSIONS: dict[str, dict[str, Any]] = {}


def store_session(overall: int, results: list[dict[str, Any]]) -> str:
    """Keep report data in memory for the explanation chatbot."""
    session_id = str(uuid4())
    SESSIONS[session_id] = {"overall": overall, "results": results}
    return session_id


ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
STOP_WORDS = {
    "about", "after", "also", "among", "been", "being", "between", "both", "could",
    "does", "each", "from", "have", "into", "more", "most", "other", "over", "same",
    "such", "than", "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "under", "using", "very", "was", "were", "what", "when", "where",
    "which", "while", "with", "would", "will", "shall", "should", "have", "has", "had",
    "the", "and", "for", "are", "but", "not", "you", "your", "its", "our", "can", "may",
    "use", "used", "via", "per", "all", "any", "one", "two", "how", "why", "who", "whose",
}


def extract_chunks(pdf_bytes: bytes) -> list[dict[str, int | str]]:
    """Extract PDF text, stop before References, and split into ~150-word chunks."""
    text = _extract_pdf_text(pdf_bytes)
    text, _ = _split_references(text)
    words = text.split()
    return _chunks_from_words(words)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _split_references(text: str) -> tuple[str, str]:
    match = re.search(r"(?im)^\s*(?:references|bibliography)\s*$", text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.end() :]


def _chunks_from_words(words: list[str]) -> list[dict[str, int | str]]:
    chunks = []
    for start in range(0, len(words), 150):
        chunk_text = " ".join(words[start : start + 150]).strip()
        if chunk_text:
            chunks.append({"id": len(chunks) + 1, "text": chunk_text})
    return chunks


def _find_shared_phrases(chunk_text: str, source_text: str) -> list[str]:
    """Find up to three longest exact shared word sequences (minimum five words)."""
    chunk_words = re.findall(r"[\w'-]+", chunk_text)
    source_words = re.findall(r"[\w'-]+", source_text)
    previous = [0] * (len(source_words) + 1)
    candidates = []
    for i, chunk_word in enumerate(chunk_words, start=1):
        current = [0] * (len(source_words) + 1)
        for j, source_word in enumerate(source_words, start=1):
            if chunk_word.casefold() == source_word.casefold():
                current[j] = previous[j - 1] + 1
                length = current[j]
                if length >= 5:
                    candidates.append((length, i - length, i))
        previous = current

    candidates.sort(reverse=True)
    phrases = []
    for _, start, end in candidates:
        phrase = " ".join(chunk_words[start:end])
        normalized = phrase.casefold()
        if not any(normalized in existing.casefold() or existing.casefold() in normalized for existing in phrases):
            phrases.append(phrase)
        if len(phrases) == 3:
            break
    return phrases


def _reference_mentions(title: str, url: str, references: str) -> bool:
    normalized_refs = re.sub(r"[^a-z0-9]+", " ", references.casefold()).strip()
    normalized_title = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
    if normalized_title and normalized_title in normalized_refs:
        return True
    arxiv_id = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9.]+(?:v\d+)?)", url, re.IGNORECASE)
    return bool(arxiv_id and arxiv_id.group(1).casefold() in references.casefold())


def _match_type(reason: str, score: int) -> str:
    reason_lower = reason.casefold()
    if any(word in reason_lower for word in ("verbatim", "copied", "copy", "exact wording")):
        return "Near-verbatim"
    if "paraphras" in reason_lower:
        return "Likely paraphrased"
    if score < 40:
        return "Low similarity"
    return "Possible common idea"


def _fallback_queries(chunk_text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", chunk_text.lower())
    distinctive = [word for word, _ in Counter(words).most_common() if word not in STOP_WORDS]
    if not distinctive:
        distinctive = words
    return [" ".join(distinctive[:8])] if distinctive else []


def _get_queries(chunk_text: str, verification: bool = False) -> list[str]:
    variation = "Use alternative terminology and keywords not used in the first search." if verification else ""
    prompt = (
        "Generate exactly 3 short scholarly search queries (one per line) that could find "
        "the original source of this passage. Return only the queries. "
        f"{variation}\n\nPassage:\n{chunk_text}"
    )
    answer = ask_llm(prompt)
    queries = []
    for line in answer.splitlines():
        query = re.sub(r"^\s*(?:[-*\d.)]+\s*)", "", line).strip().strip('"')
        if query and query.lower() not in {item.lower() for item in queries}:
            queries.append(query[:180])
        if len(queries) == 3:
            break
    return queries or _fallback_queries(chunk_text)


def _search_arxiv(query: str) -> list[dict[str, str]]:
    response = requests.get(
        ARXIV_API,
        params={"search_query": f'all:"{query}"', "start": 0, "max_results": 5},
        timeout=10,
        headers={"User-Agent": "AgenticPlagiarismChecker/1.0"},
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    atom = "{http://www.w3.org/2005/Atom}"
    results = []
    for entry in root.findall(f"{atom}entry"):
        title = " ".join((entry.findtext(f"{atom}title") or "").split())
        abstract = " ".join((entry.findtext(f"{atom}summary") or "").split())
        link = entry.find(f"{atom}id")
        if title and abstract and link is not None and link.text:
            results.append({"title": title, "url": link.text.strip(), "text": abstract})
    return results


def _search_semantic_scholar(query: str) -> list[dict[str, str]]:
    response = requests.get(
        SEMANTIC_SCHOLAR_API,
        params={"query": query, "limit": 5, "fields": "title,url,abstract"},
        timeout=10,
        headers={"User-Agent": "AgenticPlagiarismChecker/1.0"},
    )
    response.raise_for_status()
    results = []
    for paper in response.json().get("data", []):
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        url = paper.get("url") or ""
        if title and abstract and url:
            results.append({"title": title, "url": url, "text": abstract})
    return results


def find_sources(chunk_text: str, verification: bool = False) -> list[dict[str, str]]:
    """Search arXiv and Semantic Scholar, returning up to five unique papers."""
    sources = []
    seen = set()
    queries = _get_queries(chunk_text, verification=verification)
    calls = (_search_arxiv, _search_semantic_scholar)
    for query in queries:
        for search in calls:
            if search is not calls[0] or query != queries[0]:
                time.sleep(1)
            try:
                found = search(query)
            except Exception:
                continue
            for source in found:
                key = source.get("url", "").rstrip("/").lower() or source.get("title", "").lower()
                if key and key not in seen and len(sources) < 5:
                    seen.add(key)
                    sources.append(source)
        if len(sources) >= 5:
            break
    return sources[:5]


def run_check(pdf_bytes: bytes) -> dict[str, Any]:
    """Find and score candidate sources for up to the first 15 paper chunks."""
    from evidence import build_evidence
    from similarity import score_chunk

    full_text = _extract_pdf_text(pdf_bytes)
    body_text, references = _split_references(full_text)
    all_words = body_text.split()
    all_chunks = _chunks_from_words(all_words)
    chunks = all_chunks[:15]
    results = []
    session_results = []
    skipped_chunks = 0
    analyzed_words = 0
    for chunk in chunks:
        text = str(chunk["text"])
        if len(text.split()) < 40:
            skipped_chunks += 1
            continue
        analyzed_words += len(text.split())

        best = {
            "score": 0,
            "reason": "No matching source found",
            "title": "No source found",
            "url": "",
            "text": "",
            "embedding_score": 0,
        }
        for source in find_sources(text):
            comparison = score_chunk(text, source["text"])
            if comparison["score"] > best["score"]:
                best = {
                    "score": comparison["score"],
                    "reason": comparison["reason"],
                    "title": source["title"],
                    "url": source["url"],
                    "text": source["text"],
                    "embedding_score": comparison.get("embedding_score", comparison["score"]),
                }

        if best["score"] > 60:
            for source in find_sources(text, verification=True):
                comparison = score_chunk(text, source["text"])
                if comparison["score"] > best["score"]:
                    best = {
                        "score": comparison["score"],
                        "reason": comparison["reason"],
                        "title": source["title"],
                        "url": source["url"],
                        "text": source["text"],
                        "embedding_score": comparison.get("embedding_score", comparison["score"]),
                    }

        score = int(best["score"])
        source_excerpt = str(best["text"])[:1600]
        has_source = str(best["title"]) != "No source found"
        citation_found = _reference_mentions(str(best["title"]), str(best["url"]), references) if has_source and references.strip() else None
        if not has_source:
            citation_status = "Not checked: no source match"
        elif not references.strip():
            citation_status = "References section not found"
        elif citation_found:
            citation_status = "Likely cited in references"
        else:
            citation_status = "No matching reference found"
        evidence = build_evidence(
            text,
            str(best["text"]),
            str(best["title"]),
            str(best["url"]),
            int(best["embedding_score"]),
        )
        result = {
            "chunk_id": int(chunk["id"]),
            "chunk": text,
            "score": score,
            "source_title": str(best["title"]),
            "source_url": str(best["url"]),
            "reason": str(best["reason"]),
            "match_type": _match_type(str(best["reason"]), score),
            "source_excerpt": source_excerpt,
            "matched_phrases": _find_shared_phrases(text, source_excerpt),
            "citation_found": citation_found,
            "citation_status": citation_status,
            "evidence": evidence,
        }
        results.append(result)
        session_results.append({**result, "source_text": str(best["text"])})

    overall = round(sum(int(item["score"]) for item in results) / len(results)) if results else 0
    high = sum(1 for item in results if int(item["score"]) > 70)
    medium = sum(1 for item in results if 40 <= int(item["score"]) <= 70)
    low = sum(1 for item in results if int(item["score"]) < 40)
    top_result = max(results, key=lambda item: int(item["score"]), default=None)
    coverage = {
        "document_words": len(all_words),
        "document_chunks": len(all_chunks),
        "analyzed_chunks": len(results),
        "skipped_short_chunks": skipped_chunks,
        "analyzed_words": analyzed_words,
        "chunk_limit": 15,
        "truncated": len(all_chunks) > 15,
    }
    summary = {
        "high_matches": high,
        "medium_matches": medium,
        "low_matches": low,
        "flagged_matches": sum(1 for item in results if int(item["score"]) > 40),
        "top_source_title": top_result["source_title"] if top_result else "No source found",
        "top_score": int(top_result["score"]) if top_result else 0,
    }
    session_id = store_session(overall, session_results)
    return {
        "overall": overall,
        "results": results,
        "summary": summary,
        "coverage": coverage,
        "session_id": session_id,
    }
