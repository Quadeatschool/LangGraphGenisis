from typing import Dict, TypedDict
from langgraph.graph import StateGraph

class AgentState (TypedDict):
    message : str 

def personal_node(state: AgentState) -> AgentState:
    """An ai that is personal lol"""
    state['message'] = "Hey + state['message'] +, love you munches"

    return state


graph = StateGraph(AgentState)

graph.add_node("receptrionist", personal_node)

graph.set_entry_point("receptrionist")
graph.set_finish_point("receptrionist")

app = graph.compile()

from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png))

result = app.invoke({"message": "Toaster"})

result["message"]




