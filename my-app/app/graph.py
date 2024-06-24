import operator
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from app.nodes import date_node, embedding_retrieve_node, tool_node


"""
    Router
"""
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    """
        This is the router
        Either agent can decide to end
    """
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    return "continue"


"""
    Graph
"""
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

workflow = StateGraph(AgentState)

workflow.add_node("date_finder", date_node)
workflow.add_node("embedding_retriever", embedding_retrieve_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "date_finder",
    router,
    {"continue": "embedding_retriever", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "embedding_retriever",
    router,
    {"continue": END, "call_tool": "call_tool", "__end__": END},
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: "continue",
    {
        "continue": END,
    },
)

workflow.set_entry_point("date_finder")
graph = workflow.compile()