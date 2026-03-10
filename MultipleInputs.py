from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM and tools
llm = ChatOpenAI(model="gpt-4o")
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

class AgentState(TypedDict):
    values: List[int]
    name: str
    result: str
    messages: Annotated[list[BaseMessage], add_messages]
    search_query: str

def process_values(state: AgentState) -> dict:
    """This function handles multiple different inputs"""
    result = f"Hi there {state['name']}! Your sum = {sum(state['values'])}"
    state["result"] = result
    print(f"Processed values: {result}")
    return {"result": result}

def web_search_node(state: AgentState) -> dict:
    """Node that performs web search using the search tool"""
    if not state.get("search_query"):
        return {"messages": [AIMessage(content="No search query provided")]}
    
    search_results = search_tool.invoke({"query": state["search_query"]})
    search_response = f"Search results for '{state['search_query']}':\n{search_results}"
    
    return {"messages": [AIMessage(content=search_response)]}

def should_search(state: AgentState) -> str:
    """Determine if web search is needed based on messages"""
    if state.get("search_query"):
        return "web_search"
    return "processor"

# Build graph with both data processing and web search
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("processor", process_values)
graph.add_node("web_search", web_search_node)

# Add edges
graph.add_edge(START, "processor")
graph.add_conditional_edges(
    "processor",
    should_search,
    {
        "web_search": "web_search",
        "processor": "processor"
    }
)
graph.add_edge("web_search", END)
graph.set_finish_point("processor")
graph.set_finish_point("web_search")

app = graph.compile()

try:
    from IPython.display import Image, display 
    display(Image(app.get_graph().draw_mermaid_png()))
except ImportError:
    print("IPython not available for graph visualization")

# Example 1: Process values
answers = app.invoke({
    "values": [1, 2, 3, 4], 
    "name": "Steve",
    "messages": [],
    "search_query": ""
})
print(f"Result 1: {answers['result']}")

# Example 2: With web search
search_answers = app.invoke({
    "values": [10, 20, 30], 
    "name": "Alice",
    "messages": [],
    "search_query": "latest AI developments 2026"
})
print(f"\nSearch results: {search_answers['messages'][-1].content}")

