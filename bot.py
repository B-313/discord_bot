"""bot.py – WaifuAssistant Discord bot entrypoint (Phase 1).

Flow:
    Discord message → check trigger (mention or "!" prefix)
    → extract user_id + username → show typing indicator
    → await run_agent(user_id, message_content)
    → send reply in thread (if available) or same channel.

Design notes:
    - All LLM / persona logic lives in agents/langgraph_agent.py.  This file
      is intentionally thin: Discord I/O only.
    - bot.py is stable across phases; only the agent internals will change.
"""

import logging
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

from agents.langgraph_agent import run_agent

# ---------------------------------------------------------------------------
# Logging – configured before anything else so early errors are visible.
# Includes user_id in log lines via the %(message)s field; never logs raw
# message content above DEBUG to avoid accidental data leaks in production.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – loaded from .env (see .env.example).
# ---------------------------------------------------------------------------

load_dotenv()

# DISCORD_TOKEN is mandatory; fail fast at startup if it is absent.
DISCORD_TOKEN: str = os.environ["DISCORD_TOKEN"]

# Prefix for non-mention invocations (e.g. "!hello").
COMMAND_PREFIX: str = os.getenv("COMMAND_PREFIX", "!")

# ---------------------------------------------------------------------------
# Bot setup – intents declare which Gateway events Discord sends us.
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
# message_content: required to read the text of messages (privileged intent –
#   must be enabled in the Discord Developer Portal).
intents.message_content = True
# members: required to resolve member display names and for future member-event
#   handlers (privileged intent – enable in Developer Portal).
intents.members = True

bot = commands.Bot(
    # Accept both "!" prefix and @mention as command triggers.
    command_prefix=commands.when_mentioned_or(COMMAND_PREFIX),
    intents=intents,
    # Disable the default help command; the persona handles "!help" naturally.
    help_command=None,
)

# ---------------------------------------------------------------------------
# Event: on_ready
# ---------------------------------------------------------------------------


@bot.event
async def on_ready() -> None:
    """Called once the bot has successfully connected to the Discord gateway."""
    logger.info("Bot ready | logged in as %s (id=%s)", bot.user, bot.user.id)
    # Set a visible status so users know the bot is alive and listening.
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="your messages~",
        )
    )


# ---------------------------------------------------------------------------
# Event: on_message – core request handler
# ---------------------------------------------------------------------------


@bot.event
async def on_message(message: discord.Message) -> None:
    """Handle every incoming message and decide whether to reply.

    Triggers:
        1. The bot is directly @mentioned anywhere in the message.
        2. The message starts with the configured prefix (default ``!``).

    Both triggers go through the same pipeline:
        user_id extraction → clean text → typing indicator → agent → reply.
    """
    # ── Guard: never respond to bots (including ourselves) to prevent loops. ──
    if message.author.bot:
        return

    # ── Determine if this message is intended for the bot. ──────────────────
    is_mention = bot.user in message.mentions
    is_prefix = message.content.startswith(COMMAND_PREFIX)

    if not is_mention and not is_prefix:
        # Message is not addressed to the bot – let other handlers run.
        await bot.process_commands(message)
        return

    # ── Extract identifiers (user_id is the stable Discord snowflake). ───────
    user_id: str = str(message.author.id)
    # display_name respects server nicknames; falls back to username.
    username: str = message.author.display_name

    # ── Clean the message text the agent will see. ───────────────────────────
    user_text: str = message.content

    if is_mention:
        # Strip all @mentions so the agent receives plain prose.
        for mentioned_user in message.mentions:
            user_text = user_text.replace(mentioned_user.mention, "")
        user_text = user_text.strip()
    else:
        # Strip the command prefix (e.g. leading "!").
        user_text = user_text[len(COMMAND_PREFIX):].strip()

    # Guard: acknowledge empty messages warmly without calling the agent.
    if not user_text:
        await message.reply(
            f"Ehehe~ {username}! You called but didn't say anything. "
            "What's on your mind? 💬"
        )
        return

    logger.info(
        "Incoming message | user_id=%s | username=%s | trigger=%s | text_len=%d",
        user_id,
        username,
        "mention" if is_mention else "prefix",
        len(user_text),
    )

    # ── Determine the reply destination. ─────────────────────────────────────
    # If the message arrived inside a thread, reply there so the conversation
    # stays organized.  Otherwise reply directly in the channel (which creates
    # a quoted reply bubble in Discord).
    reply_channel = (
        message.channel
        if isinstance(message.channel, discord.Thread)
        else None  # None → use message.reply() which works in any channel.
    )

    # ── Show the "typing…" indicator for the whole duration of agent work. ───
    async with message.channel.typing():
        try:
            # Delegate all LLM / persona / (future) tool logic to the agent.
            # Pass username so the agent can personalize its reply ("Hey, Aria~").
            response: str = await run_agent(
                user_id,
                f"[{username}]: {user_text}",
            )
        except Exception:
            logger.exception(
                "Agent call failed | user_id=%s", user_id
            )
            await message.reply(
                "Ah… something went wrong on my end~ "
                "Please try again in a moment! 🙏"
            )
            return

    # ── Send the response. ───────────────────────────────────────────────────
    if reply_channel is not None:
        # We are inside a thread – send as a plain message to keep thread flow.
        await reply_channel.send(response)
    else:
        # Reply with a quote bubble so Discord links the reply to the original.
        await message.reply(response)

    logger.info("Reply sent | user_id=%s | reply_len=%d", user_id, len(response))

    # Allow the commands extension to process any registered prefix commands.
    await bot.process_commands(message)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # log_handler=None tells discord.py not to install its own log handler
    # because we already configured logging above.
    bot.run(DISCORD_TOKEN, log_handler=None)
