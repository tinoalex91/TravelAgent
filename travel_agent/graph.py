"""LangGraph multi-agent Travel Planner.

Mirrors notebook.ipynb, extracted into an importable module so it can be served by
`langgraph dev` / the LangGraph API. Exposes two compiled graphs:

- `app` — Coordinator (fixed flights -> venue -> playlist order, skips unrequested intents)
- `dynamic_app` — Pre-Middleware -> Dynamic Planner Agent -> Post-Middleware
"""

import json
import os
import re
import sqlite3
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tavily import TavilyClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "false"))
os.environ.setdefault("LANGSMITH_ENDPOINT", os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))
os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "travel-agent"))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY,
)

# --- Flights tool — Kiwi MCP server, loaded lazily so import stays sync ---------------------

KIWI_MCP_URL = "https://mcp.kiwi.com"

mcp_client = MultiServerMCPClient(
    {
        "kiwi": {
            "url": KIWI_MCP_URL,
            "transport": "streamable_http",
        }
    }
)

_mcp_flights_tool_cache = {"tool": None, "attempted": False}


async def _get_mcp_flights_tool():
    if not _mcp_flights_tool_cache["attempted"]:
        _mcp_flights_tool_cache["attempted"] = True
        try:
            tools = await mcp_client.get_tools()
            for t in tools:
                if t.name == "search_flights":
                    _mcp_flights_tool_cache["tool"] = t
                    break
        except Exception as exc:
            print(f"WARNING: falling back to stub flights tool ({exc})")
    return _mcp_flights_tool_cache["tool"]


@tool
async def search_flights(origin: str, destination: str) -> str:
    """Search for flights between `origin` and `destination` via the Kiwi MCP server.

    Falls back to a stub response if the MCP server is unreachable.
    """
    mcp_tool = await _get_mcp_flights_tool()
    if mcp_tool is None:
        return (
            f"[stub] No live Kiwi MCP connection available. "
            f"Would search flights from {origin} to {destination}."
        )
    return await mcp_tool.ainvoke({"origin": origin, "destination": destination})


flights_tool = search_flights

# --- Venue tool — Tavily web search -----------------------------------------------------------

tavily_client = (
    TavilyClient(api_key=TAVILY_API_KEY)
    if TAVILY_API_KEY and TAVILY_API_KEY != "your_key_here"
    else None
)


@tool
def search_venues(destination: str, guest_count: int) -> str:
    """Search the web for wedding venues in `destination` that can host `guest_count` guests."""
    query = f"wedding venues in {destination} for {guest_count}"

    if tavily_client is None:
        return f"[stub] No TAVILY_API_KEY configured. Would search: '{query}'"

    try:
        response = tavily_client.search(query=query, max_results=5)
        results = response.get("results", [])
        if not results:
            return f"No venue results found for '{query}'."
        lines = [f"- {r['title']}: {r['url']}" for r in results]
        return f"Top venues for '{query}':\n" + "\n".join(lines)
    except Exception as exc:
        return f"Venue search failed: {exc}"


# --- Playlist tool — SQLite database ----------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "travel_agent.db")


def init_playlist_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            genre TEXT NOT NULL
        )
        """
    )
    cur.execute("SELECT COUNT(*) FROM songs")
    if cur.fetchone()[0] == 0:
        sample_songs = [
            ("Take Five", "Dave Brubeck", "jazz"),
            ("So What", "Miles Davis", "jazz"),
            ("Feeling Good", "Nina Simone", "jazz"),
            ("Fly Me to the Moon", "Frank Sinatra", "jazz"),
            ("Blinding Lights", "The Weeknd", "pop"),
            ("Uptown Funk", "Mark Ronson ft. Bruno Mars", "pop"),
            ("Perfect", "Ed Sheeran", "pop"),
            ("Bohemian Rhapsody", "Queen", "rock"),
            ("Sweet Child O' Mine", "Guns N' Roses", "rock"),
            ("Get Lucky", "Daft Punk", "electronic"),
        ]
        cur.executemany(
            "INSERT INTO songs (title, artist, genre) VALUES (?, ?, ?)", sample_songs
        )
        conn.commit()
    conn.close()


init_playlist_db()


@tool
def query_playlist_db(genre: str) -> str:
    """Query the local `songs` SQLite table for tracks matching `genre`."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT title, artist FROM songs WHERE LOWER(genre) = LOWER(?)", (genre,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return f"No songs found for genre '{genre}'."

    lines = [f"- {title} - {artist}" for title, artist in rows]
    return f"Playlist for genre '{genre}':\n" + "\n".join(lines)


# --- Shared middleware -------------------------------------------------------------------------

summarization_middleware = SummarizationMiddleware(
    model=llm,
    trigger=("messages", 12),
    keep=("messages", 6),
)

# --- Chat I/O helpers ---------------------------------------------------------------------------
#
# Both graphs below accept a `messages` list too (in addition to `raw_input`) so chat-style
# clients like Agent Chat, which only ever send/read `messages`, work against them: the latest
# HumanMessage becomes `raw_input` if `raw_input` wasn't set directly, and the final node appends
# an AIMessage summarizing `final_output` so the client has something to render as the reply.


def _latest_human_text(messages) -> Optional[str]:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return message.content
    return None


def _format_final_output_as_text(final_output: dict) -> str:
    if final_output.get("error"):
        return final_output["error"]

    parts = []
    summary = final_output.get("summary")
    if summary:
        parts.append(summary)
    for key in ("flights", "venue", "playlist"):
        value = final_output.get(key)
        if value:
            parts.append(f"**{key.capitalize()}**\n{value}")
    return "\n\n".join(parts) if parts else "No results."


# =================================================================================================
# Graph 1 — Coordinator with fixed flights -> venue -> playlist order (skips unrequested intents)
# =================================================================================================

FLIGHTS_SYSTEM_PROMPT = (
    "You are a flights specialist agent. Use the search_flights tool to find flight "
    "options between the given origin and destination. Summarize the best options concisely."
)
VENUE_SYSTEM_PROMPT = (
    "You are a venue specialist agent. Use the search_venues tool to find suitable event "
    "venues for the given destination and guest count. Summarize the top options concisely."
)
PLAYLIST_SYSTEM_PROMPT = (
    "You are a playlist specialist agent. Use the query_playlist_db tool to build a short "
    "playlist for the requested music genre."
)

flights_agent = create_agent(
    llm, [flights_tool], system_prompt=FLIGHTS_SYSTEM_PROMPT, middleware=[summarization_middleware]
)
venue_agent = create_agent(
    llm, [search_venues], system_prompt=VENUE_SYSTEM_PROMPT, middleware=[summarization_middleware]
)
playlist_agent = create_agent(
    llm, [query_playlist_db], system_prompt=PLAYLIST_SYSTEM_PROMPT, middleware=[summarization_middleware]
)


class TravelState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_input: str

    origin: Optional[str]
    destination: Optional[str]
    guest_count: Optional[int]
    genre: Optional[str]
    flights_result: Optional[str]
    venue_result: Optional[str]
    playlist_result: Optional[str]

    intents: List[str]
    extracted_raw: Optional[str]
    final_output: Optional[dict]


EXTRACTION_PROMPT = """You are an information-extraction engine for a travel planner.
Given the user's message, extract the following fields and return STRICT JSON only
(no prose, no markdown fences):

{{
  "origin": string or null,
  "destination": string or null,
  "guest_count": integer or null,
  "genre": string or null,
  "intents": array of zero or more of ["flights", "venue", "playlist"]
}}

Infer "intents" from what the user is asking for:
- "flights" if travel/flights between an origin and destination is implied. This includes
  indirect phrasing — e.g. "I'm from X" plus any mention of an event/destination in Y implies
  travel from X to Y, even if the words "flight" or "fly" never appear. If both an origin and a
  destination city can be identified, always include "flights".
- "venue" if an event/venue/wedding location is implied
- "playlist" if music/genre/playlist is implied

User message: {user_input}
"""


async def extract_intent_node(state: TravelState) -> dict:
    raw_input = state.get("raw_input") or _latest_human_text(state.get("messages")) or ""
    prompt = EXTRACTION_PROMPT.format(user_input=raw_input)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"extracted_raw": response.content, "raw_input": raw_input}


def _fallback_extraction(raw_input: str) -> dict:
    text = raw_input.lower()
    guest_match = re.search(r"(\d+)\s*guests?", text)

    intents = []
    if any(k in text for k in ["flight", "fly", "from "]):
        intents.append("flights")
    if any(k in text for k in ["venue", "wedding", "party", "event"]):
        intents.append("venue")
    if any(k in text for k in ["playlist", "music", "genre", "jazz", "song"]):
        intents.append("playlist")

    return {
        "origin": None,
        "destination": None,
        "guest_count": int(guest_match.group(1)) if guest_match else None,
        "genre": None,
        "intents": intents,
    }


async def update_state_node(state: TravelState) -> dict:
    raw = state.get("extracted_raw") or ""
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    try:
        data = json.loads(match.group(0)) if match else json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        data = _fallback_extraction(state["raw_input"])

    return {
        "origin": data.get("origin"),
        "destination": data.get("destination"),
        "guest_count": data.get("guest_count"),
        "genre": data.get("genre"),
        "intents": data.get("intents") or [],
        "flights_result": None,
        "venue_result": None,
        "playlist_result": None,
    }


async def flights_node(state: TravelState) -> dict:
    try:
        task = f"Find flights from {state['origin']} to {state['destination']}."
        result = await flights_agent.ainvoke({"messages": [HumanMessage(content=task)]})
        content = result["messages"][-1].content
    except Exception as exc:
        content = f"Flights lookup failed: {exc}"
    return {"flights_result": content}


async def venue_node(state: TravelState) -> dict:
    try:
        task = f"Find wedding venues in {state['destination']} for {state['guest_count']} guests."
        result = await venue_agent.ainvoke({"messages": [HumanMessage(content=task)]})
        content = result["messages"][-1].content
    except Exception as exc:
        content = f"Venue lookup failed: {exc}"
    return {"venue_result": content}


async def playlist_node(state: TravelState) -> dict:
    try:
        task = f"Build a short playlist for the genre: {state['genre']}."
        result = await playlist_agent.ainvoke({"messages": [HumanMessage(content=task)]})
        content = result["messages"][-1].content
    except Exception as exc:
        content = f"Playlist lookup failed: {exc}"
    return {"playlist_result": content}


async def final_response_node(state: TravelState) -> dict:
    final_output = {
        "flights": state.get("flights_result"),
        "venue": state.get("venue_result"),
        "playlist": state.get("playlist_result"),
    }
    reply = AIMessage(content=_format_final_output_as_text(final_output))
    return {"final_output": final_output, "messages": [reply]}


PIPELINE_ORDER = ["flights", "venue", "playlist"]

NODE_BY_INTENT = {
    "flights": "flights_node",
    "venue": "venue_node",
    "playlist": "playlist_node",
}

RESULT_FIELD_BY_INTENT = {
    "flights": "flights_result",
    "venue": "venue_result",
    "playlist": "playlist_result",
}


def route(state: TravelState) -> str:
    for intent in PIPELINE_ORDER:
        already_done = state.get(RESULT_FIELD_BY_INTENT[intent]) is not None
        if intent in state.get("intents", []) and not already_done:
            return NODE_BY_INTENT[intent]
    return "final_response_node"


graph = StateGraph(TravelState)

graph.add_node("extract_intent_node", extract_intent_node)
graph.add_node("update_state_node", update_state_node)
graph.add_node("flights_node", flights_node)
graph.add_node("venue_node", venue_node)
graph.add_node("playlist_node", playlist_node)
graph.add_node("final_response_node", final_response_node)

graph.add_edge(START, "extract_intent_node")
graph.add_edge("extract_intent_node", "update_state_node")

ROUTE_MAP = {
    "flights_node": "flights_node",
    "venue_node": "venue_node",
    "playlist_node": "playlist_node",
    "final_response_node": "final_response_node",
}

graph.add_conditional_edges("update_state_node", route, ROUTE_MAP)
graph.add_conditional_edges("flights_node", route, ROUTE_MAP)
graph.add_conditional_edges("venue_node", route, ROUTE_MAP)
graph.add_conditional_edges("playlist_node", route, ROUTE_MAP)

graph.add_edge("final_response_node", END)

app = graph.compile()

# =================================================================================================
# Graph 2 — Pre-Middleware -> Dynamic Planner Agent -> Post-Middleware (no human-in-the-loop)
# =================================================================================================


class DynamicPlannerState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_input: str

    origin: Optional[str]
    destination: Optional[str]
    guest_count: Optional[int]
    genre: Optional[str]

    extracted_raw: Optional[str]
    error: Optional[str]
    agent_messages: Optional[list]
    final_output: Optional[dict]


DYNAMIC_EXTRACTION_PROMPT = """You are an information-extraction engine for a travel/event planner.
Given the user's message, extract the following fields and return STRICT JSON only
(no prose, no markdown fences):

{{
  "origin": string or null,
  "destination": string or null,
  "guest_count": integer or null,
  "genre": string or null
}}

Only set "origin" if the user explicitly mentions traveling from somewhere. Do not infer
travel intent from an event location alone.

User message: {user_input}
"""

DEFAULT_GUEST_COUNT = 20


async def pre_middleware_node(state: DynamicPlannerState) -> dict:
    """Extract fields, apply defaults, and validate the request before planning.

    - Extract: ask the LLM for origin/destination/guest_count/genre as JSON.
    - Validate: a destination is required to do anything useful; reject otherwise.
    - Memory trim: this graph is single-shot, so there's no multi-turn history to trim here —
      trimming instead happens inside the planner agent via `summarization_middleware`, which
      bounds its message history across tool-call round trips.
    """
    raw_input = state.get("raw_input") or _latest_human_text(state.get("messages")) or ""
    if not raw_input.strip():
        return {"error": "No input provided."}

    prompt = DYNAMIC_EXTRACTION_PROMPT.format(user_input=raw_input)
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        data = {}

    destination = data.get("destination")
    if not destination:
        return {
            "raw_input": raw_input,
            "extracted_raw": response.content,
            "error": "Could not identify a destination in the request.",
        }

    return {
        "raw_input": raw_input,
        "extracted_raw": response.content,
        "origin": data.get("origin"),
        "destination": destination,
        "guest_count": data.get("guest_count") or DEFAULT_GUEST_COUNT,
        "genre": data.get("genre"),
        "error": None,
    }


DYNAMIC_PLANNER_SYSTEM_PROMPT = (
    "You are the planning agent for an event/travel assistant. You have three tools "
    "available: one for flights, one for venues, and one for playlists. You decide which "
    "tools are actually needed for the request in front of you — call only the tools "
    "that are relevant, skip the rest, and never call a tool if a field it needs is missing "
    "(e.g. don't search flights without both an origin and a destination). "
    "After calling the tools you need, summarize what you found."
)

dynamic_planner_agent = create_agent(
    llm,
    [flights_tool, search_venues, query_playlist_db],
    system_prompt=DYNAMIC_PLANNER_SYSTEM_PROMPT,
    middleware=[summarization_middleware],
)


def _build_planner_task(state: DynamicPlannerState) -> str:
    parts = [f"Destination: {state.get('destination')}"]
    if state.get("origin"):
        parts.append(f"Origin: {state['origin']}")
    if state.get("guest_count") is not None:
        parts.append(f"Guest count: {state['guest_count']}")
    if state.get("genre"):
        parts.append(f"Music genre: {state['genre']}")
    parts.append(
        "Decide which of the flights/venue/playlist tools apply to this request and call "
        "only those."
    )
    return "\n".join(parts)


async def planner_node(state: DynamicPlannerState) -> dict:
    """Dynamic Planner Agent + Tool Executor: the agent decides which tools to call and
    executes them itself via its own ReAct tool-calling loop."""
    task = _build_planner_task(state)
    result = await dynamic_planner_agent.ainvoke({"messages": [HumanMessage(content=task)]})
    return {"agent_messages": result["messages"]}


TOOL_NAME_TO_KEY = {
    "search_flights": "flights",
    "search_venues": "venue",
    "query_playlist_db": "playlist",
}


def post_middleware_node(state: DynamicPlannerState) -> dict:
    """Post-Middleware: merge each tool's raw result and format the final response."""
    if state.get("error"):
        final_output = {"error": state["error"]}
        reply = AIMessage(content=_format_final_output_as_text(final_output))
        return {"final_output": final_output, "messages": [reply]}

    tool_results = {"flights": None, "venue": None, "playlist": None}
    for message in state.get("agent_messages") or []:
        if isinstance(message, ToolMessage):
            key = TOOL_NAME_TO_KEY.get(message.name)
            if key:
                tool_results[key] = message.content

    summary = ""
    for message in reversed(state.get("agent_messages") or []):
        if isinstance(message, AIMessage) and message.content:
            summary = message.content
            break

    final_output = {
        "fields": {
            "origin": state.get("origin"),
            "destination": state.get("destination"),
            "guest_count": state.get("guest_count"),
            "genre": state.get("genre"),
        },
        "tools_called": [k for k, v in tool_results.items() if v is not None],
        **tool_results,
        "summary": summary,
    }
    reply = AIMessage(content=_format_final_output_as_text(final_output))
    return {"final_output": final_output, "messages": [reply]}


def route_after_pre_middleware(state: DynamicPlannerState) -> str:
    return "post_middleware_node" if state.get("error") else "planner_node"


dynamic_graph = StateGraph(DynamicPlannerState)

dynamic_graph.add_node("pre_middleware_node", pre_middleware_node)
dynamic_graph.add_node("planner_node", planner_node)
dynamic_graph.add_node("post_middleware_node", post_middleware_node)

dynamic_graph.add_edge(START, "pre_middleware_node")
dynamic_graph.add_conditional_edges(
    "pre_middleware_node",
    route_after_pre_middleware,
    {"planner_node": "planner_node", "post_middleware_node": "post_middleware_node"},
)
dynamic_graph.add_edge("planner_node", "post_middleware_node")
dynamic_graph.add_edge("post_middleware_node", END)

dynamic_app = dynamic_graph.compile()
