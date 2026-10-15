"""FastAPI application for the Agentic Plagiarism Checker demo."""

from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Agentic Plagiarism Checker")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    chunk_id: int | None = None


class SuggestionRequest(BaseModel):
    session_id: str
    chunk_id: int


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/mock")
async def mock_report() -> dict:
    results = [
        {
            "chunk_id": 1,
            "chunk": "Transformer models rely on self-attention to weigh relationships between tokens, allowing each position to build a context-sensitive representation of the sequence.",
            "score": 85,
            "source_title": "Attention Is All You Need",
            "source_url": "https://arxiv.org/abs/1706.03762",
            "reason": "Near-verbatim overlap in the description of self-attention and token relationships.",
            "match_type": "Near-verbatim",
            "source_excerpt": "Self-attention weighs relationships between tokens, allowing each position to build a context-sensitive representation of the sequence.",
            "matched_phrases": ["relationships between tokens, allowing each position to build a context-sensitive representation"],
            "citation_found": True,
            "citation_status": "Likely cited in references",
            "evidence": {
                "verbatim_ngram_count": 12,
                "matched_blocks": [{"text": "relationships between tokens allowing each position to build a context sensitive representation", "words": 11}],
                "longest_match_words": 11,
                "verbatim_word_pct": 42.3,
                "sentence_alignment": [{"chunk_sentence": "Transformer models rely on self-attention to weigh relationships between tokens, allowing each position to build a context-sensitive representation of the sequence.", "source_sentence": "Self-attention weighs relationships between tokens, allowing each position to build a context-sensitive representation of the sequence.", "cosine": 0.91, "word_overlap_pct": 72.7}],
                "paraphrase_signal": False,
                "has_citation": False,
                "citation_strings": [],
                "generic_phrase_flag": False,
                "category": "verbatim_copy",
                "score_breakdown": {"embedding": 85, "ngram": 62, "final": 85},
            },
        },
        {
            "chunk_id": 2,
            "chunk": "Contrastive learning brings representations of related examples closer together while pushing representations of unrelated examples farther apart in the embedding space.",
            "score": 72,
            "source_title": "A Simple Framework for Contrastive Learning of Visual Representations",
            "source_url": "https://arxiv.org/abs/2002.05709",
            "reason": "Likely paraphrased: the passage follows the source's core contrastive-learning formulation.",
            "match_type": "Likely paraphrased",
            "source_excerpt": "Contrastive learning brings representations of related examples closer together while pushing representations of unrelated examples farther apart in the embedding space.",
            "matched_phrases": ["representations of related examples closer together while pushing representations of unrelated examples farther apart"],
            "citation_found": True,
            "citation_status": "Likely cited in references",
            "evidence": {
                "verbatim_ngram_count": 2,
                "matched_blocks": [{"text": "representations of related examples", "words": 4}],
                "longest_match_words": 4,
                "verbatim_word_pct": 10.5,
                "sentence_alignment": [{"chunk_sentence": "Contrastive learning brings representations of related examples closer together while pushing representations of unrelated examples farther apart in the embedding space.", "source_sentence": "Contrastive learning encourages representations of related examples to move closer while unrelated examples move farther apart.", "cosine": 0.82, "word_overlap_pct": 31.6}],
                "paraphrase_signal": True,
                "has_citation": True,
                "citation_strings": ["[4]"],
                "generic_phrase_flag": False,
                "category": "paraphrase",
                "score_breakdown": {"embedding": 72, "ngram": 10, "final": 72},
            },
        },
        {
            "chunk_id": 3,
            "chunk": "We evaluate the proposed method on three benchmark datasets and report both accuracy and macro-averaged F1 to account for class imbalance.",
            "score": 55,
            "source_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "source_url": "https://arxiv.org/abs/1810.04805",
            "reason": "Some methodological phrasing is similar, but the wording and evaluation details differ.",
            "match_type": "Possible common idea",
            "source_excerpt": "We evaluate the proposed method on three benchmark datasets and report accuracy and macro-averaged F1 scores to account for class imbalance.",
            "matched_phrases": ["We evaluate the proposed method on three benchmark datasets and report"],
            "citation_found": False,
            "citation_status": "No matching reference found",
            "evidence": {
                "verbatim_ngram_count": 3,
                "matched_blocks": [{"text": "we evaluate the proposed method on three benchmark datasets", "words": 9}],
                "longest_match_words": 9,
                "verbatim_word_pct": 18.0,
                "sentence_alignment": [{"chunk_sentence": "We evaluate the proposed method on three benchmark datasets and report both accuracy and macro-averaged F1 to account for class imbalance.", "source_sentence": "We evaluate the proposed method on three benchmark datasets and report accuracy and macro-averaged F1 scores.", "cosine": 0.76, "word_overlap_pct": 68.4}],
                "paraphrase_signal": False,
                "has_citation": True,
                "citation_strings": ["(Smith, 2021)"],
                "generic_phrase_flag": False,
                "category": "cited_overlap",
                "score_breakdown": {"embedding": 55, "ngram": 18, "final": 55},
            },
        },
        {
            "chunk_id": 4,
            "chunk": "These findings suggest that additional experiments across diverse settings will be useful for understanding when the approach is most effective.",
            "score": 41,
            "source_title": "On the Opportunities and Risks of Foundation Models",
            "source_url": "https://arxiv.org/abs/2108.07258",
            "reason": "Low-level thematic similarity; this is common discussion language rather than a clear copied passage.",
            "match_type": "Possible common idea",
            "source_excerpt": "The broader effects of an approach may depend on the setting in which it is deployed and the communities involved.",
            "matched_phrases": [],
            "citation_found": False,
            "citation_status": "No matching reference found",
            "evidence": {
                "verbatim_ngram_count": 0,
                "matched_blocks": [{"text": "these findings suggest that", "words": 4}],
                "longest_match_words": 4,
                "verbatim_word_pct": 8.0,
                "sentence_alignment": [{"chunk_sentence": "These findings suggest that additional experiments across diverse settings will be useful for understanding when the approach is most effective.", "source_sentence": "The broader effects of an approach may depend on the setting in which it is deployed.", "cosine": 0.43, "word_overlap_pct": 12.5}],
                "paraphrase_signal": False,
                "has_citation": False,
                "citation_strings": [],
                "generic_phrase_flag": True,
                "category": "common_or_generic",
                "score_breakdown": {"embedding": 41, "ngram": 0, "final": 41},
            },
        },
    ]
    overall = round(sum(item["score"] for item in results) / len(results))
    from agent import store_session

    session_id = store_session(
        overall,
        [{**item, "source_text": item.get("source_excerpt", "")} for item in results],
    )
    return {
        "overall": overall,
        "results": results,
        "session_id": session_id,
        "summary": {
            "high_matches": 2,
            "medium_matches": 2,
            "low_matches": 0,
            "flagged_matches": 4,
            "top_source_title": results[0]["source_title"],
            "top_score": 85,
        },
        "coverage": {
            "document_words": 600,
            "document_chunks": 4,
            "analyzed_chunks": 4,
            "skipped_short_chunks": 0,
            "analyzed_words": 600,
            "chunk_limit": 15,
            "truncated": False,
        },
    }


@app.post("/api/chat")
async def explain_report(request: ChatRequest):
    from agent import SESSIONS

    if request.session_id not in SESSIONS:
        return JSONResponse(
            status_code=404,
            content={"error": "Unknown or expired report session. Run a plagiarism check again first."},
        )
    try:
        from chat_agent import chat

        return chat(request.session_id, request.message, request.chunk_id)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "The report explanation could not be generated. Please try again."},
        )


@app.post("/api/suggestions")
async def get_passage_suggestions(request: SuggestionRequest):
    from agent import SESSIONS

    if request.session_id not in SESSIONS:
        return JSONResponse(
            status_code=404,
            content={"error": "Unknown or expired report session. Run a plagiarism check again first."},
        )
    try:
        from chat_agent import suggest_revisions

        result = suggest_revisions(request.session_id, request.chunk_id)
        if result.get("error"):
            return JSONResponse(status_code=404, content=result)
        return result
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "Practical suggestions could not be loaded. Please try again."},
        )


@app.post("/api/check")
async def check_paper(file: UploadFile = File(...)):
    try:
        if file.content_type not in ("application/pdf", "application/octet-stream") and not file.filename.lower().endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": "Please upload a PDF file."})
        pdf_bytes = await file.read()
        if not pdf_bytes:
            return JSONResponse(status_code=400, content={"error": "The uploaded PDF is empty."})
        # Load the optional, heavier analysis stack only for real PDF checks.
        # This keeps the mock endpoint quick to start for the demo.
        from agent import run_check

        return run_check(pdf_bytes)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"We couldn't analyze this PDF. Please try again. ({str(exc)})"},
        )
    finally:
        await file.close()
