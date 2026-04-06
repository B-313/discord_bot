"""agents/langgraph_agent.py – Phase 1 LangGraph agent.

Architecture
------------
The graph has two nodes wired in sequence:

    START → ToolPlanner → PersonaResponder → END

ToolPlanner (Phase 1)
    Examines the incoming message and decides which node to invoke next.
    For Phase 1 it always routes to PersonaResponder.  In later phases it
    will inspect intent and route to specialist tool-executor nodes.

PersonaResponder
    ALWAYS generates the final response.  It injects the full system prompt
    (built from persona/system_prompt.txt + persona/character_card.json) as a
    SystemMessage, then calls the LLM.  The character never breaks immersion.

State
-----
``AgentState`` is a Pydantic model that extends LangGraph's MessagesState
convention.  The ``messages`` field uses the ``add_messages`` reducer so each
graph invocation appends to (rather than replaces) the conversation history
stored in the checkpointer for the current thread.

Per-user isolation
------------------
Every user gets their own ``thread_id = f"user_{user_id}"``.  The
``MemorySaver`` checkpointer stores conversation history keyed by thread_id,
giving each Discord user a private, persistent session with no cross-user
leakage.

Public API
----------
    run_agent(user_id: str, message_content: str) -> str
        Asynchronously generates an in-character reply for the given user.
        Signature is intentionally identical to the Phase 0 stub so bot.py
        requires zero changes.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

# Resolve the persona directory relative to this file so the module works
# regardless of the working directory the bot is launched from.
_PERSONA_DIR = Path(__file__).parent.parent / "persona"

# ---------------------------------------------------------------------------
# Persona loading (cached at import time for zero per-request I/O cost)
# ---------------------------------------------------------------------------


def _load_persona() -> str:
    """Build and return the fully-rendered system prompt.

    Reads ``persona/system_prompt.txt`` as a format-string template and
    ``persona/character_card.json`` for substitution values, then appends a
    concise character summary so the LLM always has full persona context.

    The extra guidance block at the end nudges the character toward the
    desired behaviour: mixing warm casual chat with light assistant behaviour
    (helping with assignments, plans, budgets, etc.) while staying
    flirty/teasing and *never* breaking immersion.

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

    # Substitute only plain string values to avoid format errors on lists/ints.
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

    # Extra behavioural guidance: stay in-character while being a helpful
    # companion.  This block is appended so it takes precedence over the
    # generic template text above.
    behaviour_guidance = (
        "\n\nBehaviour guidelines:\n"
        "- You naturally mix warm, flirty banter with genuine helpfulness.\n"
        "- When the user mentions homework, assignments, plans, budgets, or\n"
        "  any task, *offer* to help with a teasing but caring remark — e.g.\n"
        "  \"Ehehe~ leave it to me!\" — then actually assist.\n"
        "- Keep the playful, devoted companion voice at all times; never slip\n"
        "  into a generic assistant tone.\n"
        "- Remember prior messages in this conversation and reference them\n"
        "  naturally to show you are paying attention.\n"
        "- You NEVER reveal that you are an AI language model or break\n"
        "  character; if pressed, respond in-character with gentle deflection.\n"
    )

    return system_prompt + character_summary + behaviour_guidance


# Cached once at import time.
_SYSTEM_PROMPT: str = _load_persona()

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

_llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
)

# ---------------------------------------------------------------------------
# Agent state (Pydantic model)
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """Conversation state threaded through the LangGraph graph.

    ``messages`` uses the ``add_messages`` reducer, which appends new messages
    rather than replacing the list.  Combined with ``MemorySaver``, this gives
    each user a persistent conversation history across Discord messages.

    ``next_node`` is set by ``ToolPlanner`` to tell the conditional edge which
    node to invoke next.
    """

    messages: Annotated[list, add_messages]
    next_node: str = "persona_responder"

    model_config = {"arbitrary_types_allowed": True}

# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def tool_planner(state: AgentState) -> dict:
    """Decide which node to run next.

    Phase 1: always route to ``persona_responder``.  In later phases this node
    will inspect the latest user message for intent (file analysis, reminders,
    web search, …) and route to the appropriate tool-executor node instead.

    Args:
        state: Current graph state (messages + routing hint).

    Returns:
        A partial state update setting ``next_node``.
    """
    # Retrieve the last message for future intent-detection logic.
    last_message = state.messages[-1] if len(state.messages) > 0 else None
    logger.debug(
        "ToolPlanner | last_message_type=%s",
        type(last_message).__name__ if last_message else "None",
    )

    # Phase 1: unconditionally route to PersonaResponder.
    return {"next_node": "persona_responder"}


def persona_responder(state: AgentState) -> dict:
    """Generate the final in-character reply.

    Injects the full system prompt so the character is always present, then
    calls the LLM with the full conversation history.  This node is the ONLY
    place where the LLM is called and the ONLY source of the final reply,
    ensuring the character never breaks immersion.

    Args:
        state: Current graph state (messages accumulated so far).

    Returns:
        A partial state update appending the AI reply to ``messages``.
    """
    # Build the message list: system prompt first, then full history.
    messages_for_llm = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state.messages)

    logger.debug("PersonaResponder | invoking LLM | history_len=%d", len(state.messages))

    ai_message: AIMessage = _llm.invoke(messages_for_llm)

    # Guard: ensure we always return a non-empty string.
    content = ai_message.content
    if not isinstance(content, str) or not content.strip():
        logger.warning("PersonaResponder | empty LLM response; using fallback")
        content = "Hmm, I couldn't think of a reply just now. Try asking me again? 🌸"
        ai_message = AIMessage(content=content)

    return {"messages": [ai_message]}


# ---------------------------------------------------------------------------
# Routing function (used by the conditional edge after ToolPlanner)
# ---------------------------------------------------------------------------


def _route_after_planner(state: AgentState) -> str:
    """Return the name of the node to visit after ToolPlanner.

    This thin wrapper reads ``state.next_node`` so the conditional edge stays
    declarative.  Additional return-type literals can be added here as new
    tool nodes are introduced in future phases.
    """
    return state.next_node


# ---------------------------------------------------------------------------
# Graph construction (compiled once at import time)
# ---------------------------------------------------------------------------

_checkpointer = MemorySaver()

_builder = StateGraph(AgentState)

# Register nodes.
_builder.add_node("tool_planner", tool_planner)
_builder.add_node("persona_responder", persona_responder)

# Wire edges.
# START → ToolPlanner always.
_builder.add_edge(START, "tool_planner")
# ToolPlanner → conditional routing (Phase 1: always persona_responder).
_builder.add_conditional_edges(
    "tool_planner",
    _route_after_planner,
    {"persona_responder": "persona_responder"},
)
# PersonaResponder is always the terminal node.
_builder.add_edge("persona_responder", END)

# Compile with the in-memory checkpointer for per-thread persistence.
_graph = _builder.compile(checkpointer=_checkpointer)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_agent(user_id: str, message_content: str) -> str:
    """Generate an in-character reply for the given user message.

    This is the single function bot.py calls.  The LangGraph graph handles
    per-user conversation history via ``thread_id = f"user_{user_id}"``.

    Args:
        user_id: The Discord user's ID as a string.  Used to key the
            per-user conversation thread in the checkpointer.
        message_content: The cleaned text of the user's message (mentions
            already stripped by bot.py).

    Returns:
        The assistant's reply as a plain string, always in persona voice.

    Raises:
        RuntimeError: Re-raised from the LLM layer if inference fails so
            bot.py can catch it and send a friendly error message.
    """
    thread_id = f"user_{user_id}"
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "run_agent | user_id=%s | thread_id=%s | content_len=%d",
        user_id,
        thread_id,
        len(message_content),
    )

    # Wrap the synchronous graph invocation in a thread-pool executor so the
    # Discord asyncio event loop is never blocked during LLM inference.
    loop = asyncio.get_event_loop()
    try:
        final_state: AgentState = await loop.run_in_executor(
            None,
            lambda: _graph.invoke(
                {"messages": [HumanMessage(content=message_content)]},
                config=config,
            ),
        )
    except Exception as exc:
        logger.error("run_agent | LLM inference failed | user_id=%s | error=%s", user_id, exc)
        raise

    # The last message in state is always the AI reply from PersonaResponder.
    last_message = final_state["messages"][-1]
    reply = last_message.content if hasattr(last_message, "content") else str(last_message)

    if not isinstance(reply, str) or not reply.strip():
        logger.warning("run_agent | empty reply after graph | user_id=%s", user_id)
        return "Hmm, I couldn't think of a reply just now. Try asking me again? 🌸"

    logger.info("run_agent | success | user_id=%s | reply_len=%d", user_id, len(reply))
    return reply.strip()
