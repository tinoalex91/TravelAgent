# Travel Planner Agent

A multi-agent travel planning assistant built with **LangGraph**. It takes a single free-text
request — e.g. *"I'm from Bangalore and I want a hall in Chennai for 100 guests, jazz
playlist"* — and returns structured results for flights, venue, and playlist by fanning the
request out to specialized sub-agents.

The whole implementation lives in [notebook.ipynb](notebook.ipynb).

## Purpose

Planning a trip or event usually means juggling several unrelated lookups — flights, a venue,
music for the occasion. This agent shows how to decompose that into a **coordinator +
specialist** pattern:

1. A **coordinator** reads the raw request once and figures out *what* the user is actually
   asking for (which of flights/venue/playlist apply) and *what parameters* to pass along
   (origin, destination, guest count, genre).
2. Each **specialist agent** is scoped to exactly one tool and one job, so it can't do
   anything else and is easy to reason about, test, and swap out independently.
3. A shared **graph state** carries the extracted parameters and each specialist's result
   until a final node assembles them into one structured response.

## Architecture

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

## How LangGraph is used here

- `StateGraph(TravelState)` defines the graph over a typed state schema.
- Nodes are plain async functions `(state) -> dict`; LangGraph merges the returned dict into
  the running state after each node.
- `graph.add_conditional_edges(node, route, ROUTE_MAP)` is what makes this a *dynamic* pipeline
  instead of a fixed sequence — the same `route()` function decides the next hop after
  `update_state_node` and after each specialist, so the graph's actual path through the nodes
  depends on the user's request.
- `graph.compile()` produces a runnable `app`; `await app.ainvoke(initial_state)` runs it
  end-to-end.
- `create_agent(llm, [tool], system_prompt=..., middleware=[...])` (from `langchain.agents`)
  builds each specialist as a minimal ReAct loop: the LLM decides when to call its one tool and
  when to answer. This is the current, non-deprecated replacement for LangGraph's
  `langgraph.prebuilt.create_react_agent`.

## Middleware — summarization & message trimming

Each specialist agent can call its tool more than once inside its own ReAct loop (e.g. retry
after a bad query), and every call/response is appended to that agent's `messages` list. A
shared `SummarizationMiddleware` (from `langchain.agents.middleware`) is attached to all three
specialist agents to keep that list bounded:

```python
summarization_middleware = SummarizationMiddleware(
    model=llm,
    trigger=("messages", 12),   # start summarizing once history passes 12 messages
    keep=("messages", 6),       # always keep the 6 most recent messages verbatim
)
```

Once a specialist's message history crosses the `trigger` threshold, the middleware asks the
LLM to compress everything except the most recent `keep` messages into a single summary
message. The original older messages are then **deleted** from state (via LangGraph's
`RemoveMessage`) and replaced by that summary, so context stays bounded no matter how many
tool-call round-trips a specialist takes — instead of growing unboundedly with every retry.

## Tools & integrations

| Agent | Tool | Backing service | Offline fallback |
|---|---|---|---|
| Flights | `search_flights` | Kiwi MCP server (`https://mcp.kiwi.com`, via `langchain-mcp-adapters`) | Stub tool returning a canned string if the MCP server is unreachable or doesn't expose `search_flights` |
| Venue | `search_venues` | Tavily web search | Stub message if `TAVILY_API_KEY` is missing |
| Playlist | `query_playlist_db` | Local SQLite (`travel_agent.db`, table `songs`, seeded with sample tracks) | N/A — always available, no external dependency |

The LLM itself is **Groq** (`ChatGroq`, model `llama-3.3-70b-versatile`), shared by the
coordinator's extraction step and all three specialist agents.

**LangSmith** tracing is wired up via `LANGSMITH_TRACING` / `LANGSMITH_ENDPOINT` /
`LANGSMITH_PROJECT` env vars — optional, disabled by default, useful for inspecting each
node's LLM calls and tool calls in the LangSmith UI when enabled.

Because every tool has a graceful fallback, the full graph can be exercised end-to-end with no
API keys at all — you'll just get stubbed output instead of live flight/venue data.

## Setup

```bash
pip install -r requirements.txt
```

Fill in `.env` with real values:

```
GROQ_API_KEY=...       # required for any LLM call to succeed
TAVILY_API_KEY=...     # optional — enables live venue search
LANGSMITH_API_KEY=...  # optional — enables LangSmith tracing
```

Then open [notebook.ipynb](notebook.ipynb) and run all cells. Section 11 ("Test execution")
shows an example end-to-end run.

## Notes / known issues

- The Kiwi MCP server currently doesn't expose a `search_flights` tool, so the flights agent
  runs on the stub tool by default (visible as a `WARNING` in the section 4a output).
- Field extraction relies on the LLM returning strict JSON; `update_state_node` has a
  regex/keyword-based fallback (`_fallback_extraction`) for when it doesn't.
