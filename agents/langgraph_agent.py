"""agents/langgraph_agent.py – Phase 1 agent entrypoint.

In Phase 1 this module wraps a straightforward Ollama LLM call so that bot.py
already uses the correct interface (``await run_agent(user_id, content)``).
Later phases will swap in a full LangGraph graph here – adding per-user memory,
tool planning, RAG retrieval, and consistency reflection – without any changes
to bot.py.

Public API:
    run_agent(user_id: str, message_content: str) -> str
        Asynchronously generates an in-character reply for the given user.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Module logger – logs user_id but never sensitive message content at INFO.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read from environment; defaults match the .env.example file).
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

# Resolve the persona directory relative to this file's location so the
# module works regardless of the working directory the bot is launched from.
_PERSONA_DIR = Path(__file__).parent.parent / "persona"

# ---------------------------------------------------------------------------
# Persona loading (cached at import time for performance).
# ---------------------------------------------------------------------------


def _load_persona() -> str:
    """Build and return the fully-rendered system prompt.

    Reads ``persona/system_prompt.txt`` as a format-string template and
    ``persona/character_card.json`` for substitution values, then appends
    a brief character summary so the LLM always has full persona context.

    Returns:
        The complete system prompt string ready to pass to the LLM.

    Raises:
        FileNotFoundError: If either persona file is missing.
        json.JSONDecodeError: If the character card is malformed JSON.
    """
    system_prompt_path = _PERSONA_DIR / "system_prompt.txt"
    character_card_path = _PERSONA_DIR / "character_card.json"

    with system_prompt_path.open(encoding="utf-8") as fh:
        template = fh.read()

    with character_card_path.open(encoding="utf-8") as fh:
        card: dict = json.load(fh)

    # Only substitute plain string values to avoid format errors on lists/ints.
    string_fields = {k: v for k, v in card.items() if isinstance(v, str)}
    system_prompt = template.format_map(string_fields)

    # Append a concise character summary so key traits are always visible.
    character_summary = (
        "\n\nCharacter summary:\n"
        f"Name: {card.get('name', 'Unknown')}\n"
        f"Archetype: {card.get('archetype', '')}\n"
        f"Speech style: {card.get('speech_style', '')}\n"
        f"Background: {card.get('background', '')}\n"
    )
    return system_prompt + character_summary


# Cache the prompt once so every request pays zero I/O cost.
_SYSTEM_PROMPT: str = _load_persona()

# ---------------------------------------------------------------------------
# LLM client (synchronous; called in an executor to keep asyncio free).
# ---------------------------------------------------------------------------

_llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_agent(user_id: str, message_content: str) -> str:
    """Generate an in-character reply for the given user message.

    This is the single function bot.py calls.  In Phase 1 it performs a
    straightforward LLM inference.  In later phases it will be replaced by a
    LangGraph graph that handles memory retrieval, tool planning, and
    reflection while keeping this exact signature stable.

    Args:
        user_id: The Discord user's ID as a string.  Used for per-user
            isolation; will be the key for memory/RAG stores in future phases.
        message_content: The cleaned text of the user's message (mentions
            already stripped by bot.py).

    Returns:
        The assistant's reply as a plain string, always in persona voice.

    Raises:
        RuntimeError: Re-raised from the LLM layer if inference fails, so
            bot.py can catch it and send a friendly error message.
    """
    logger.info("run_agent called | user_id=%s | content_len=%d", user_id, len(message_content))

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=message_content),
    ]

    # ChatOllama.invoke is synchronous; run it in a thread-pool executor so
    # the Discord event loop is never blocked during (potentially slow) LLM calls.
    loop = asyncio.get_event_loop()
    try:
        ai_message = await loop.run_in_executor(None, _llm.invoke, messages)
    except Exception as exc:
        logger.error("LLM inference failed | user_id=%s | error=%s", user_id, exc)
        raise

    content = ai_message.content

    # Guard against unexpected empty or non-string responses.
    if not isinstance(content, str) or not content.strip():
        logger.warning("Empty or invalid LLM response | user_id=%s | raw=%r", user_id, content)
        return "Hmm, I couldn't think of a reply just now. Try asking me again? 🌸"

    logger.info("run_agent succeeded | user_id=%s | reply_len=%d", user_id, len(content))
    return content.strip()
