"""bot.py – WaifuAssistant Discord bot entrypoint.

Flow:
    Discord message → extract user_id → show typing → build persona prompt
    → call Ollama via langchain-ollama → send reply.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DISCORD_TOKEN: str = os.environ["DISCORD_TOKEN"]
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

PERSONA_DIR = Path(__file__).parent / "persona"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persona loading
# ---------------------------------------------------------------------------


def load_persona() -> str:
    """Return the fully-rendered system prompt for the bot persona.

    Reads ``persona/system_prompt.txt`` and ``persona/character_card.json``,
    then substitutes the character's ``name`` (and any other top-level string
    fields) into the prompt template using str.format_map so that placeholders
    like ``{name}`` are replaced with the actual character values.
    """
    system_prompt_path = PERSONA_DIR / "system_prompt.txt"
    character_card_path = PERSONA_DIR / "character_card.json"

    with system_prompt_path.open(encoding="utf-8") as fh:
        template = fh.read()

    with character_card_path.open(encoding="utf-8") as fh:
        card: dict = json.load(fh)

    # Only pass plain string values to avoid format errors with complex fields.
    string_fields = {k: v for k, v in card.items() if isinstance(v, str)}
    system_prompt = template.format_map(string_fields)

    # Append a brief character summary so the LLM has full context.
    character_summary = (
        f"\n\nCharacter summary:\n"
        f"Name: {card.get('name', 'Unknown')}\n"
        f"Archetype: {card.get('archetype', '')}\n"
        f"Speech style: {card.get('speech_style', '')}\n"
        f"Background: {card.get('background', '')}\n"
    )
    return system_prompt + character_summary


# Load once at startup so every message reuses the same prompt.
SYSTEM_PROMPT: str = load_persona()

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
)

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True  # Required to read message text.

bot = commands.Bot(
    command_prefix=commands.when_mentioned,  # Respond only when @mentioned.
    intents=intents,
    help_command=None,
)

# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


@bot.event
async def on_ready() -> None:
    """Called when the bot has connected to Discord and is ready."""
    logger.info("Logged in as %s (id=%s)", bot.user, bot.user.id)
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="your messages~",
        )
    )


@bot.event
async def on_message(message: discord.Message) -> None:
    """Handle incoming messages.

    The bot replies only when it is mentioned directly, so it can coexist
    peacefully in busy channels.
    """
    # Ignore messages from bots (including self) to prevent loops.
    if message.author.bot:
        return

    # Only respond when the bot is explicitly @mentioned.
    if bot.user not in message.mentions:
        return

    # Extract the Discord user_id as a string (used for per-user isolation).
    user_id: str = str(message.author.id)

    # Strip the mention from the message text so the LLM sees clean input.
    user_text: str = message.content
    for mention in message.mentions:
        user_text = user_text.replace(mention.mention, "").strip()

    if not user_text:
        await message.reply("Ehehe~ You mentioned me but didn't say anything! What's on your mind? 💬")
        return

    logger.info("Message from user_id=%s: %s", user_id, user_text[:80])

    # Show a "typing…" indicator while we wait for the LLM response.
    async with message.channel.typing():
        try:
            response = await _get_llm_reply(user_id, user_text)
        except Exception:
            logger.exception("LLM call failed for user_id=%s", user_id)
            await message.reply(
                "Ah… something went wrong on my end. Please try again in a moment! 🙏"
            )
            return

    await message.reply(response)

    # Allow the commands extension to process any prefix commands as well.
    await bot.process_commands(message)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


async def _get_llm_reply(user_id: str, user_text: str) -> str:
    """Call Ollama synchronously inside an executor to keep the event loop free.

    langchain-ollama's ChatOllama.invoke is synchronous; running it via
    ``loop.run_in_executor`` avoids blocking the Discord gateway.

    Args:
        user_id: Discord user ID (string). Passed here as the foundation for
            per-user conversation history that will be added in a later module.
        user_text: The clean message text from the user.
    """
    loop = asyncio.get_event_loop()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]
    ai_message = await loop.run_in_executor(None, llm.invoke, messages)
    content = ai_message.content
    if not isinstance(content, str) or not content:
        logger.warning("Unexpected LLM response for user_id=%s: %r", user_id, content)
        return "Hmm, I couldn't think of a reply just now. Try asking me again? 🌸"
    return content.strip()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)
