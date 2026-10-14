"""Seed a sample report and print one chatbot explanation.

Run from this directory with: python demo_chat.py
If OpenRouter is unavailable, chat_agent uses its deterministic fallback.
"""

import asyncio
import json

from chat_agent import chat
from main import mock_report


if __name__ == "__main__":
    report = asyncio.run(mock_report())
    response = chat(
        report["session_id"],
        "Why did chunk 0 score so high?",
        chunk_id=0,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))
