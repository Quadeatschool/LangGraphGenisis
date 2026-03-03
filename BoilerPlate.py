"""
LangGraph Boilerplate
=====================
A minimal, extensible starting point for building stateful LLM agents with LangGraph.

─────────────────────────────────────────────
SETUP: Virtual Environment (recommended)
─────────────────────────────────────────────
Run these commands once before starting:

    # Create the virtual environment
    python -m venv .venv

    # Activate it
    # macOS/Linux:
    source .venv/bin/activate
    # Windows:
    .venv\\Scripts\\activate

    # Install dependencies
    pip install langgraph langchain-openai langchain-core

    # Set your API key (or use python-dotenv with a .env file)
    export OPENAI_API_KEY=your_key_here   # macOS/Linux
    set OPENAI_API_KEY=your_key_here      # Windows

    # Run the script
    python langgraph_boilerplate.py

    # Deactivate when done
    deactivate

─────────────────────────────────────────────
FILE SAFETY NOTES
─────────────────────────────────────────────
Your files are safe by default. LangGraph agents can ONLY access your
file system if you explicitly give them a file-related tool (e.g. read_file,
write_file, shell). Without such tools wired in, agents are sandboxed to
the LLM call itself.

Rules of thumb:
  ✅ Safe     — agents with only LLM + structured output tools
  ⚠️  Caution — agents with read_file tools (scope to a specific directory)
  🚫 Dangerous — agents with shell/exec tools (avoid unless sandboxed)
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# 1. Define State
#    The state is shared across all nodes in the graph.
# ─────────────────────────────────────────────
class State(TypedDict):
    # `add_messages` is a reducer that appends messages instead of overwriting
    messages: Annotated[list[BaseMessage], add_messages]
    # Add any other fields you need here, e.g.:
    # context: str
    # step_count: int


# ─────────────────────────────────────────────
# 2. Initialize LLM
# ─────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o")


# ─────────────────────────────────────────────
# 3. Define Nodes
#    Each node is a function: State -> dict (partial state update)
# ─────────────────────────────────────────────
def call_llm(state: State) -> dict:
    """Primary LLM node — sends messages and gets a response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def preprocess(state: State) -> dict:
    """Optional preprocessing node — transform/validate input before LLM."""
    # Example: trim or add context to the last message
    return {}  # Return empty dict if no state changes needed


def postprocess(state: State) -> dict:
    """Optional postprocessing node — parse or act on the LLM's response."""
    last_message = state["messages"][-1]
    print(f"[Postprocess] Last response: {last_message.content[:100]}...")
    return {}


# ─────────────────────────────────────────────
# 4. Define Conditional Routing (optional)
#    Return the name of the next node as a string.
# ─────────────────────────────────────────────
def should_continue(state: State) -> str:
    """Route to 'postprocess' or END based on some condition."""
    last_message = state["messages"][-1]
    # Example: if the LLM says "DONE", end the loop
    if "DONE" in last_message.content:
        return "end"
    return "postprocess"


# ─────────────────────────────────────────────
# 5. Build the Graph
# ─────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("preprocess", preprocess)
    graph.add_node("call_llm", call_llm)
    graph.add_node("postprocess", postprocess)

    # Add edges
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "call_llm")

    # Conditional edge from call_llm
    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "postprocess": "postprocess",
            "end": END,
        },
    )

    graph.add_edge("postprocess", END)

    return graph


# ─────────────────────────────────────────────
# 6. Compile & Run
# ─────────────────────────────────────────────
def main():
    graph = build_graph()
    app = graph.compile()

    # Optional: visualize the graph (requires graphviz)
    # print(app.get_graph().draw_mermaid())

    # Run the graph
    initial_state = {
        "messages": [HumanMessage(content="Hello! What can you help me with?")]
    }

    result = app.invoke(initial_state)

    print("\n=== Final Messages ===")
    for msg in result["messages"]:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"[{role}]: {msg.content}\n")


if __name__ == "__main__":
    main()