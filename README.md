# WaifuAssistant (discord_bot)

A local, multi-user Discord bot powered by LLMs and agentic AI – featuring a persistent, dynamic waifu/husbando persona. It helps with homework, budgeting, and travel, and always stays in character. All on your hardware: fully private, with per-user memory and tool use.

## ✨ What’s Inside

- Persona-Driven: Affectionate, playful, helpful waifu/husbando AI for each user.
- Fully Local: Runs Ollama LLMs + ChromaDB/FAISS vector stores on your own PC.
- Secure & Private: Each user’s chat history, files, and context are strictly isolated.
- Agentic & Useful: Handles academic, budgeting, file analysis, search, and reminders.
- Modern Stack: discord.py, Ollama, LangGraph, LangChain, ChromaDB.

## 🚀 Features
- Multi-user support: separate memory/RAG per Discord user.
- File uploads (PDFs, CSVs) tied to each user for context-aware help.
- Discord threads for project organization.
- Modular agentic actions: LLM-driven reply, tool planning, and secure tool execution.
- Easy to extend with more tools, voices, or persona modules.

## 🏗️ Architecture Overview

See architecture/simple_diagram.md for a visual flow.

### High-level flow:

1. Discord Event – Message received, with user_id.
2. Agent Pipeline (LangGraph):
    - PersonaResponder (in-character)
    - ToolPlanner (should a tool be invoked?)
    - ToolExecutor (user-scoped: RAG, file analysis, calculator, etc.)
    - Reflector/Consistency checks
3. Memory & Data:
    - Per-user chat history
    - RAG context per user (ChromaDB/FAISS)
    - Per-user file uploads
4. Reply is always in persona voice; never breaks character.

See the repo scaffold for detailed modules.

## ⚡ Getting Started

### 1. Prerequisites
- Python 3.11+
- Ollama (https://ollama.com/) installed and running (ollama serve)
  - Recommended model: glm-5, cydonia, mistral, or any OpenAI-compatible
- Discord bot account & token (see Discord developer docs)

### 2. Clone & Install

```bash
git clone https://github.com/B-313/discord_bot.git
cd discord_bot
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure .env

```bash
cp .env.example .env
# Fill in your DISCORD_TOKEN, OLLAMA_BASE_URL, etc.
```

### 4. Run the Bot

```bash
python bot.py
```
The bot will appear online in your configured Discord server.

## 🗂️ Project Structure
```
discord_bot/
├── bot.py                # Main Discord bot entrypoint
├── persona/              # Persona system prompts and configs
│   ├── system_prompt.txt
│   └── character_card.json
├── rag/                  # RAG (vector store, context retrieval)
│   └── vector_store.py
├── agents/
│   └── langgraph_agent.py  # Agent workflow (LangGraph)
├── tools/
│   └── custom_tools.py     # User-aware tools
├── deployment/
│   └── docker-compose.yml  # Optional: easy deployment
├── architecture/
│   └── simple_diagram.md   # System/data flow diagram
├── requirements.txt
├── .env.example
└── README.md
```

## ✍️ Credits & Inspiration
- Built by B-313
- Powered by discord.py, Ollama, LangChain, LangGraph, ChromaDB
- Persona character guidelines inspired by the AI companion community

---

PRs/feature requests are welcome as long as they respect per-user privacy and persona consistency!