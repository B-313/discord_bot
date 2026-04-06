# WaifuAssistant: System Architecture (2026)

## Flow Overview

```mermaid
flowchart TD
    subgraph Discord
        U1[User#1]
        U2[User#2]
        B1[Discord Bot ('Husbando / WaifuAssistant')]
    end
    U1 -- chat --> B1
    U2 -- chat --> B1
    B1 -- user_id, message --> AGENT[LangGraph Agent]
    AGENT -->|Checks user history| MEMORY[LangChain + ChromaDB RAG]
    AGENT -->|Needs tool?| TOOLS[Custom Tool Nodes]
    TOOLS --> AGENT
    AGENT -->|Reply| B1
    B1 -- reply in-character --> U1
```

## Main Modules

- **bot.py**: Discord event loop, user_id extraction, message passing.
- **/persona**: System prompt, character config, persona injection.
- **/agents/langgraph_agent.py**: LangGraph agent graph (PersonaResponder, ToolPlanner, ToolExecutor, Reflector).
- **/rag/vector_store.py**: ChromaDB wrapper, file embedding, document retrieval (scoped per user).
- **/tools/custom_tools.py**: Safe Python executor, file/CSV/PDF analysis, reminders, user-local access only.
- **/uploads/{user_id}**: Directory for each user's files.

## Data & Privacy

- All data isolated by user_id (from Discord): prevents cross-user leaks.
- Persona dynamically references user’s own history, files, and state.
- RAG context, tool outputs, and files = private per-user.

## Extensibility

- Add more tools by expanding `/tools` and agent graph edges
- Easy swap between different LLMs in Ollama
- Optional: enable TTS, image generation, or web search tools, all per-user

---
