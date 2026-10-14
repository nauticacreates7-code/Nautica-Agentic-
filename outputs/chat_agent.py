"""Tool-using report explainer with a deterministic fallback."""

import json
import os
import re
import time
from typing import Any

from agent import SESSIONS


SYSTEM_PROMPT = """You are an academic integrity analyst explaining a plagiarism report. Always call tools to
get evidence before answering; never state a number you did not get from a tool. For any
chunk you discuss, give: (a) Verdict, one of: Copied, Paraphrased (same meaning, different
words), Common knowledge or generic phrasing, Properly cited, Not meaningfully similar;
(b) the evidence: number of verbatim 5-grams, longest matched run in words, % of words
verbatim, best sentence-pair cosine; (c) 2-3 sentences of reasoning on whether the
similarity is MEANINGFUL, i.e. shared specific ideas, structure and claims versus shared
boilerplate or generic terms; (d) one caveat: sources are compared by abstract only, so
absence of a match does not prove originality. Quote at most two short matched phrases. Be
concise. If asked something unrelated to the report, say you only explain this report."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_overview",
            "description": "Get the report overall score, chunk counts by category, and top three chunks.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_flagged",
            "description": "List report chunks at or above the requested score threshold.",
            "parameters": {
                "type": "object",
                "properties": {"min_score": {"type": "integer", "default": 40}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk_evidence",
            "description": "Get all evidence for one report chunk, plus its text and source.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "integer"}},
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_passages",
            "description": "Return chunk/source sentence alignment pairs for a report chunk.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "integer"}},
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_citation",
            "description": "Check whether the chunk contains citation markers and list them.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "integer"}},
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    },
]


def _session(session_id: str) -> dict[str, Any]:
    return SESSIONS[session_id]


def _result_for_chunk(session_id: str, chunk_id: int) -> dict[str, Any] | None:
    results = _session(session_id).get("results", [])
    for result in results:
        if int(result.get("chunk_id", -1)) == chunk_id:
            return result
    # Preserve the existing 1-based API IDs while allowing the documented
    # chunk 0 example to refer to the first result.
    if chunk_id == 0 and results:
        return results[0]
    if 0 <= chunk_id < len(results):
        return results[chunk_id]
    return None


def get_overview(session_id: str) -> dict[str, Any]:
    session = _session(session_id)
    results = session.get("results", [])
    category_counts: dict[str, int] = {}
    for result in results:
        category = result.get("evidence", {}).get("category", "unrelated")
        category_counts[category] = category_counts.get(category, 0) + 1
    top = sorted(results, key=lambda item: int(item.get("score", 0)), reverse=True)[:3]
    return {
        "overall": session.get("overall", 0),
        "number_of_chunks": len(results),
        "count_per_category": category_counts,
        "top_3_chunks": [
            {
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score", 0),
                "category": item.get("evidence", {}).get("category", "unrelated"),
                "source_title": item.get("source_title", "No source found"),
            }
            for item in top
        ],
    }


def list_flagged(session_id: str, min_score: int = 40) -> list[dict[str, Any]]:
    results = _session(session_id).get("results", [])
    return [
        {
            "chunk_id": item.get("chunk_id"),
            "score": item.get("score", 0),
            "category": item.get("evidence", {}).get("category", "unrelated"),
            "source_title": item.get("source_title", "No source found"),
        }
        for item in results
        if int(item.get("score", 0)) >= min_score
    ]


def get_chunk_evidence(session_id: str, chunk_id: int) -> dict[str, Any]:
    result = _result_for_chunk(session_id, chunk_id)
    if result is None:
        return {"error": f"Chunk {chunk_id} was not found in this report."}
    return {
        "chunk_id": result.get("chunk_id"),
        "score": result.get("score", 0),
        "evidence": result.get("evidence", {}),
        "chunk_text": result.get("chunk", ""),
        "source_title": result.get("source_title", "No source found"),
        "source_url": result.get("source_url", ""),
    }


def compare_passages(session_id: str, chunk_id: int) -> dict[str, Any]:
    result = _result_for_chunk(session_id, chunk_id)
    if result is None:
        return {"error": f"Chunk {chunk_id} was not found in this report."}
    evidence = result.get("evidence", {})
    return {"chunk_id": result.get("chunk_id"), "pairs": evidence.get("sentence_alignment", [])}


def check_citation(session_id: str, chunk_id: int) -> dict[str, Any]:
    result = _result_for_chunk(session_id, chunk_id)
    if result is None:
        return {"error": f"Chunk {chunk_id} was not found in this report."}
    evidence = result.get("evidence", {})
    return {
        "chunk_id": result.get("chunk_id"),
        "has_citation": bool(evidence.get("has_citation", False)),
        "citation_strings": evidence.get("citation_strings", []),
    }


def _execute_tool(session_id: str, name: str, args: dict[str, Any]) -> Any:
    if name == "get_overview":
        return get_overview(session_id)
    if name == "list_flagged":
        return list_flagged(session_id, int(args.get("min_score", 40)))
    if name == "get_chunk_evidence":
        return get_chunk_evidence(session_id, int(args["chunk_id"]))
    if name == "compare_passages":
        return compare_passages(session_id, int(args["chunk_id"]))
    if name == "check_citation":
        return check_citation(session_id, int(args["chunk_id"]))
    return {"error": f"Unknown tool: {name}"}


def _tool_summary(name: str, result: Any) -> str:
    """Make the visible tool trace describe the evidence returned."""
    if isinstance(result, dict) and result.get("error"):
        return str(result["error"])
    if name == "get_chunk_evidence" and isinstance(result, dict):
        evidence = result.get("evidence", {})
        return (
            f"{evidence.get('verbatim_ngram_count', 0)} verbatim 5-grams; "
            f"longest match {evidence.get('longest_match_words', 0)} words; "
            f"{evidence.get('verbatim_word_pct', 0)}% verbatim"
        )
    if name == "get_overview" and isinstance(result, dict):
        return f"overall {result.get('overall', 0)}%; {result.get('number_of_chunks', 0)} chunks"
    if name == "list_flagged" and isinstance(result, list):
        return f"found {len(result)} flagged chunks"
    if name == "compare_passages" and isinstance(result, dict):
        return f"returned {len(result.get('pairs', []))} sentence alignment pairs"
    if name == "check_citation" and isinstance(result, dict):
        citations = result.get("citation_strings", [])
        return f"found citations: {', '.join(citations)}" if citations else "no citation marker found"
    return "Returned report evidence."


def _fallback_reply(session_id: str, message: str, chunk_id: int | None) -> tuple[str, list[dict[str, Any]]]:
    report_terms = re.compile(
        r"\b(report|chunk|passage|score|source|citation|similarity|evidence|"
        r"match|matched|flagged|verdict|copied|paraphras|original|overall)\b",
        re.IGNORECASE,
    )
    if not report_terms.search(message):
        # Read report data before answering, consistent with the tool-first policy.
        get_overview(session_id)
        return (
            "I only explain this plagiarism report. Ask about a passage, score, source, citation, or evidence.",
            [{"tool": "get_overview", "args": {}, "summary": "Loaded the report overview."}],
        )
    if chunk_id is None:
        match = re.search(r"\bchunk\s+(\d+)\b", message, re.IGNORECASE)
        if match:
            chunk_id = int(match.group(1))
    if chunk_id is None:
        overview = get_overview(session_id)
        steps = [{"tool": "get_overview", "args": {}, "summary": _tool_summary("get_overview", overview)}]
        best = overview.get("top_3_chunks", [])
        if not best:
            return (
                "Verdict: Not meaningfully similar. The report contains no analyzed chunks to explain. "
                "Caveat: sources are compared by abstract only, so absence of a match does not prove originality.",
                steps,
            )
        chunk_id = int(best[0]["chunk_id"])
    evidence_result = get_chunk_evidence(session_id, chunk_id)
    steps = [{
        "tool": "get_chunk_evidence",
        "args": {"chunk_id": chunk_id},
        "summary": _tool_summary("get_chunk_evidence", evidence_result),
    }]
    if evidence_result.get("error"):
        return (
            f"{evidence_result['error']} I can only explain chunks in this report. "
            "Caveat: sources are compared by abstract only, so absence of a match does not prove originality.",
            steps,
        )

    evidence = evidence_result.get("evidence", {})
    category = evidence.get("category", "unrelated")
    verdicts = {
        "verbatim_copy": "Copied",
        "paraphrase": "Paraphrased (same meaning, different words)",
        "common_or_generic": "Common knowledge or generic phrasing",
        "cited_overlap": "Properly cited",
        "unrelated": "Not meaningfully similar",
    }
    verdict = verdicts.get(category, "Not meaningfully similar")
    pairs = evidence.get("sentence_alignment", [])
    best_cosine = pairs[0].get("cosine", 0) if pairs else 0
    matched = evidence.get("matched_blocks", [])[:2]
    quote_text = " ".join(f'“{block.get("text", "")[:90]}”' for block in matched if block.get("text"))
    if category == "verbatim_copy":
        reasoning = "The report found a substantial continuous wording match, which is more specific than generic academic phrasing. Review the matched block against the linked source and confirm attribution."
    elif category == "paraphrase":
        reasoning = "The strongest sentence pair is semantically close while sharing relatively little wording, which is consistent with paraphrasing. Check whether the source’s specific claims and structure are attributed."
    elif category == "cited_overlap":
        reasoning = "There is overlapping wording and the passage contains a citation marker. The citation is a useful attribution signal, though it does not by itself establish that every reused phrase is properly quoted."
    elif category == "common_or_generic":
        reasoning = "The evidence is dominated by generic language or weak alignment rather than a distinctive copied passage. That makes the match less meaningful, but context still matters."
    else:
        reasoning = "The available evidence shows little exact wording or sentence-level alignment with this candidate source. The search is limited, so this only describes the sources the checker found."
    reply = (
        f"Verdict: {verdict}. Evidence: {evidence.get('verbatim_ngram_count', 0)} shared 5-grams; "
        f"longest matched run {evidence.get('longest_match_words', 0)} words; "
        f"{evidence.get('verbatim_word_pct', 0)}% of chunk words verbatim; "
        f"best sentence-pair cosine {best_cosine}.\n\n{reasoning}"
    )
    if quote_text:
        reply += f" Matched phrase(s): {quote_text}."
    reply += "\n\nCaveat: sources are compared by abstract only, so absence of a match does not prove originality."
    return reply, steps


def chat(session_id: str, message: str, chunk_id: int | None = None) -> dict[str, Any]:
    """Explain a stored report using report tools, falling back without failing."""
    steps = []
    try:
        if session_id not in SESSIONS:
            return {"reply": "That report session is unknown or expired. Run a check again first.", "steps": []}
        user_content = message
        if chunk_id is not None:
            user_content = f"The user is asking about report chunk id {chunk_id}. User message: {message}"
        client = None
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                timeout=20,
            )
            model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            deadline = time.monotonic() + 20
            for iteration in range(5):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                request_client = client.with_options(timeout=remaining)
                response = request_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="required" if iteration == 0 else "auto",
                )
                assistant = response.choices[0].message
                tool_calls = assistant.tool_calls or []
                if not tool_calls:
                    reply = assistant.content or ""
                    if reply.strip() and steps:
                        return {"reply": reply.strip(), "steps": steps}
                    break
                messages.append(assistant.model_dump(exclude_none=True))
                for tool_call in tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments or "{}")
                    result = _execute_tool(session_id, name, args)
                    steps.append(
                        {
                            "tool": name,
                            "args": args,
                            "summary": _tool_summary(name, result),
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
        except Exception:
            pass

        fallback, fallback_steps = _fallback_reply(session_id, message, chunk_id)
        return {"reply": fallback, "steps": steps or fallback_steps}
    except Exception:
        try:
            fallback, fallback_steps = _fallback_reply(session_id, message, chunk_id)
            return {"reply": fallback, "steps": steps or fallback_steps}
        except Exception:
            return {
                "reply": "I couldn't read that report session. Please run the check again, then ask about a result. Caveat: sources are compared by abstract only, so absence of a match does not prove originality.",
                "steps": steps,
            }
