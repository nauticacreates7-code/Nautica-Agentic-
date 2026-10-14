"""Small optional OpenRouter-backed language model helper."""

import os


def ask_llm(prompt: str) -> str:
    """Return a short model response, or an empty string when unavailable."""
    if os.getenv("USE_LLM", "1") == "0":
        return ""

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            timeout=15,
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        return answer.strip() if answer else ""
    except Exception:
        return ""
