from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# ─── State ────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# ─── Tools ────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for a given query. Returns top 3 results with titles, snippets and URLs."""
    print(f"\n[TOOL] web_search(query='{query}')")
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}")
        output = "\n\n".join(results)
        print(f"[TOOL] Found {len(results)} results")
        return output
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def fetch_page(url: str) -> str:
    """Fetch the content of a webpage and return cleaned text. Use this to read full articles."""
    print(f"\n[TOOL] fetch_page(url='{url}')")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines[:100])  # first 100 lines only
        print(f"[TOOL] Fetched {len(cleaned)} characters")
        return cleaned
    except Exception as e:
        return f"Failed to fetch page: {str(e)}"

tools = [web_search, fetch_page]

# ─── LLM ──────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o",
    api_key="github token",
    base_url="https://models.inference.ai.azure.com",
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# ─── System prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are a research assistant. When given a topic:
1. Use web_search to find relevant results
2. Use fetch_page on the most relevant URL to get full content
3. Synthesise findings into a structured summary

Always structure your final answer as:
SUMMARY: [2-3 sentence overview]
KEY FINDINGS:
- finding 1 (source: url)
- finding 2 (source: url)
- finding 3 (source: url)
SOURCES:
- url1
- url2

Never answer from your own knowledge. Always search first."""

# ─── Agent node ───────────────────────────────────────────────
def agent_node(state: AgentState):
    print("\n[AGENT] Thinking...")
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ─── Routing ──────────────────────────────────────────────────
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tools"
    return END

# ─── Build graph ──────────────────────────────────────────────
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tools": "tools",
        END: END
    }
)
graph.add_edge("tools", "agent")
app = graph.compile()

# ─── Run ──────────────────────────────────────────────────────
def research(topic: str):
    print(f"\n{'='*60}")
    print(f"Researching: {topic}")
    print('='*60)

    result = app.invoke({
        "messages": [HumanMessage(content=f"Research this topic and give me a structured summary: {topic}")]
    })

    # print final answer
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            msg_type = type(msg).__name__
            if msg_type == "AIMessage":
                print(f"\n{msg.content}")

# ─── Test ─────────────────────────────────────────────────────
research("Model Context Protocol MCP Anthropic")