# Travel Planner Agent

A multi-agent travel planning assistant built with **LangGraph**. It takes a single free-text
request — e.g. *"I'm from Bangalore and I want a hall in Chennai for 100 guests, jazz
playlist"* — and returns structured results for flights, venue, and playlist.

Two graphs implement two different takes on the same problem:

- **`agent`** — a coordinator that extracts intent up front, then runs a **fixed** pipeline
  (flights → venue → playlist), skipping any step the user didn't ask for.
- **`dynamic_agent`** — a single **Dynamic Planner Agent** that decides for itself, at runtime,
  which of the three tools are actually relevant and calls only those — e.g. for *"Small party
  in Goa with EDM music"* it skips flights (no origin mentioned) and calls only venue +
  playlist.

## Where the code lives

- [notebook.ipynb](notebook.ipynb) — the annotated, cell-by-cell walkthrough of both
  architectures. Best place to read *why* each piece exists.
- [travel_agent/graph.py](travel_agent/graph.py) — the same logic extracted into a plain Python
  module, importable and servable by `langgraph dev` / the LangGraph API. This is what actually
  runs when you test the graphs interactively (see [Running & testing](#running--testing)
  below). Keep the two in sync if you change one.
- [langgraph.json](langgraph.json) — tells the LangGraph CLI where to find each compiled graph
  (`agent` and `dynamic_agent`) inside `travel_agent/graph.py`.
- [agent-chat-ui/](agent-chat-ui/) — a vendored copy of
  [langchain-ai/agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui), a small Next.js
  chat frontend, pre-wired via `agent-chat-ui/.env` to talk to a local LangGraph server. Useful
  for testing without LangGraph Studio.

## Purpose

Planning a trip or event usually means juggling several unrelated lookups — flights, a venue,
music for the occasion. Both graphs decompose that the same general way:

1. Figure out *what* the user is actually asking for (which of flights/venue/playlist apply)
   and *what parameters* to pass along (origin, destination, guest count, genre).
2. Only run the tools that are relevant — never call flights without an origin, never call
   venue/playlist without a destination/genre.
3. Merge whatever results were produced into one structured response.

They differ in *how* step 2 is decided: a rule-based router walking a fixed order, vs. an agent
making the call itself.

## Architecture — `agent` (fixed pipeline)

```
START
  │
  ▼
extract_intent_node   (LLM reads raw_input → JSON: origin, destination, guest_count, genre, intents)
  │
  ▼
update_state_node     (parses the JSON, merges fields into state, falls back to regex if the LLM's
                        output isn't valid JSON)
  │
  ▼
   route()  ──┬──▶ flights_node   ──┐
              ├──▶ venue_node      ─┼──▶ route() again ──▶ ... ──▶ final_response_node ──▶ END
              └──▶ playlist_node  ─┘
```

- **Coordinator** = `extract_intent_node` + `update_state_node` + the `route()` function.
- **Specialists** = `flights_node`, `venue_node`, `playlist_node` — each just calls its own
  ReAct agent (`create_agent` from `langchain.agents`) bound to a single tool, with a shared
  summarization middleware attached (see below).
- **Router** (`route()`) is one function reused at every branch point. It walks a fixed
  pipeline order (`flights → venue → playlist`) and returns whichever intent was requested by
  the user and hasn't produced a result yet. Once every requested intent has a result, it sends
  the graph to `final_response_node`. This means the graph automatically skips any agent the
  user didn't ask for — if the request only mentions a playlist, `flights_node` and
  `venue_node` never run.
- **State** (`TravelState`, a `TypedDict`) is the single object threaded through every node.
  Nodes return partial dicts and LangGraph merges them into state — nodes never mutate state
  directly.

## Architecture — `dynamic_agent` (dynamic planner)

```
User → Pre-Middleware (extract, validate, default) → Dynamic Planner Agent
     → Tool Executor → Post-Middleware (merge, format) → Final Output
```

- **Pre-Middleware** (`pre_middleware_node`) — extracts origin/destination/guest_count/genre
  via the LLM, defaults `guest_count` to 20 if unspecified, and validates that a destination is
  present (routes straight to an error output otherwise, never invoking the planner).
- **Dynamic Planner Agent** (`dynamic_planner_agent`) — one `create_agent` with all three tools
  (`flights_tool`, `search_venues`, `query_playlist_db`) bound directly. It decides itself which
  tools are relevant to the extracted fields and calls only those.
- **Tool Executor** — there's no separate node for this; it's the planner agent's own ReAct
  tool-calling loop (`planner_node` just invokes the agent and captures the resulting message
  history).
- **Post-Middleware** (`post_middleware_node`) — walks the agent's message history for
  `ToolMessage`s to report exactly which tools ran, merges their raw output, and formats the
  final response. Any tool the planner skipped stays `None` in the output.

No human-in-the-loop step is implemented in this version.

## How LangGraph is used here

- `StateGraph(TravelState)` / `StateGraph(DynamicPlannerState)` define each graph over a typed
  state schema.
- Nodes are plain async functions `(state) -> dict`; LangGraph merges the returned dict into
  the running state after each node.
- `graph.add_conditional_edges(node, route, ROUTE_MAP)` is what makes the fixed pipeline
  *dynamic in ordering* — the same `route()` function decides the next hop after
  `update_state_node` and after each specialist, so the graph's actual path through the nodes
  depends on the user's request. The dynamic planner graph uses a much simpler conditional edge
  (`route_after_pre_middleware`) that only branches on validation failure.
- `graph.compile()` produces a runnable graph object (`app`, `dynamic_app`);
  `await app.ainvoke(initial_state)` runs it end-to-end.
- `create_agent(llm, [tools], system_prompt=..., middleware=[...])` (from `langchain.agents`)
  builds each ReAct agent: the LLM decides when to call its tool(s) and when to answer. This is
  the current, non-deprecated replacement for LangGraph's
  `langgraph.prebuilt.create_react_agent`.
- Both graphs also accept a `messages` list (in addition to `raw_input`) so chat-style clients
  — LangGraph Studio's Chat tab, Agent Chat, agent-chat-ui — work directly: the latest
  `HumanMessage` becomes `raw_input` if not set explicitly, and the final node appends an
  `AIMessage` summarizing `final_output` for the client to render.

## Middleware — summarization & message trimming

Every agent in both graphs (the three specialists, and the dynamic planner) can call its
tool(s) more than once inside its own ReAct loop (e.g. retry after a bad query), and every
call/response is appended to that agent's `messages` list. A shared `SummarizationMiddleware`
(from `langchain.agents.middleware`) is attached to all of them to keep that list bounded:

```python
summarization_middleware = SummarizationMiddleware(
    model=llm,
    trigger=("messages", 12),   # start summarizing once history passes 12 messages
    keep=("messages", 6),       # always keep the 6 most recent messages verbatim
)
```

Once an agent's message history crosses the `trigger` threshold, the middleware asks the LLM to
compress everything except the most recent `keep` messages into a single summary message. The
original older messages are then **deleted** from state (via LangGraph's `RemoveMessage`) and
replaced by that summary, so context stays bounded no matter how many tool-call round-trips an
agent takes — instead of growing unboundedly with every retry.

## Tools & integrations

| Tool | Backing service | Offline fallback |
|---|---|---|
| `search_flights` | Kiwi MCP server (`https://mcp.kiwi.com`, via `langchain-mcp-adapters`) | Stub response if the MCP server is unreachable or doesn't expose `search_flights`; the MCP tool is fetched lazily on first call so module import never blocks on network I/O |
| `search_venues` | Tavily web search | Stub message if `TAVILY_API_KEY` is missing |
| `query_playlist_db` | Local SQLite (`travel_agent.db`, table `songs`, seeded with sample tracks) | N/A — always available, no external dependency |

The LLM itself is **Groq** (`ChatGroq`, model `llama-3.3-70b-versatile`), shared by every
extraction step and every agent in both graphs.

**LangSmith** tracing is wired up via `LANGSMITH_TRACING` / `LANGSMITH_ENDPOINT` /
`LANGSMITH_PROJECT` env vars — optional, disabled by default, useful for inspecting each node's
LLM calls and tool calls in the LangSmith UI when enabled.

Because every tool has a graceful fallback, both graphs can be exercised end-to-end with no API
keys at all — you'll just get stubbed output instead of live flight/venue data.

## Setup

```bash
uv sync            # or: pip install -e .
```

Fill in `.env` (copy from `.env.example`) with real values:

```
GROQ_API_KEY=...       # required for any LLM call to succeed
TAVILY_API_KEY=...     # optional — enables live venue search
LANGSMITH_API_KEY=...  # optional — enables LangSmith tracing
```

## Running & testing

**Notebook** — open [notebook.ipynb](notebook.ipynb) and run all cells. Section 11 runs the
fixed pipeline end-to-end; section 12 runs the dynamic planner.

**LangGraph dev server** — serves both graphs over the LangGraph API:

```bash
langgraph dev --no-browser
```

This starts an in-memory API at `http://127.0.0.1:2024` (auto-reloads on changes to
`travel_agent/graph.py`). From here you can:

- Open **LangGraph Studio** at the printed `Studio UI` link. Browsers block secure pages like
  `smith.langchain.com` from calling `http://localhost` directly, so if the connection fails,
  restart with `langgraph dev --no-browser --tunnel` and use the printed `https://*.trycloudflare.com`
  URL as the Studio "Base URL" instead. In Studio, pick `agent` or `dynamic_agent` from the
  graph/assistant picker, expand the **Input** panel, and fill in **Raw Input** with your test
  message.
- Point any LangGraph-API-compatible chat client (e.g.
  [Agent Chat](https://agentchat.vercel.app)) at `http://127.0.0.1:2024` with assistant ID
  `agent` or `dynamic_agent`.

**Local chat UI** (`agent-chat-ui/`) — a self-hosted alternative to the two options above, with
no manual URL/ID entry:

```bash
cd agent-chat-ui
npm install
npm run dev
```
