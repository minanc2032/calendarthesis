import argparse
import os
from datetime import datetime
import pytz
from langchain_core.messages import SystemMessage, HumanMessage
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Any

# Import calendar tools
from calendar_tools import (
    create_calendar_event,
    delete_calendar_event,
    get_calendar_events,
    get_calendar_event,
    get_current_time
)

def get_langfuse_handler() -> CallbackHandler:
    """Create a Langfuse callback handler from environment variables."""
    return CallbackHandler(
        public_key="pk-lf-4301db05-d17f-4fd4-9cf7-d04314a1690e",
        secret_key="sk-lf-d4b2f997-698f-4283-932e-6d6e7110f28e",
        host="https://cloud.langfuse.com"
    )

os.environ["OPENAI_API_KEY"] = "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"

class State(TypedDict):
    messages: Annotated[list, add_messages]

def build_graph() -> Any:
    """Build and compile the LangGraph chatbot graph."""
    graph_builder = StateGraph(State)
    tools = [create_calendar_event, delete_calendar_event, get_calendar_events, get_calendar_event]
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State) -> Dict[str, List[Any]]:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    return graph_builder.compile()

def main():
    parser = argparse.ArgumentParser(description="Run LangGraph chatbot with a custom message.")
    parser.add_argument('--message', type=str, default="Schedule meeting", help='Message to send to the chatbot')
    args = parser.parse_args()

    system_time = get_current_time()
    initial_messages = [
        SystemMessage(content=f"The current date and time is {system_time} (Europe/Amsterdam)."),
        HumanMessage(content=args.message)
    ]

    graph = build_graph()
    langfuse_handler = get_langfuse_handler()
    events = graph.stream({"messages": initial_messages}, config={"callbacks": [langfuse_handler]})
    for event in events:
        print(event)

if __name__ == "__main__":
    main()



