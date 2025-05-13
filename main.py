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

# Environment variables for OpenRouter
# Ensure OPENROUTER_API_KEY is set in your environment if using OpenRouter models
# e.g., export OPENROUTER_API_KEY='your_openrouter_api_key'
#OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://openrouter.ai/api/v1") # Changed default
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "CalendarThesisApp") # Example default

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

def _get_llm_with_tools(model_identifier: str, tools: list) -> Any:
    """
    Creates and configures an LLM instance based on the model identifier
    and binds the provided tools to it.
    Uses ChatOpenAI for both OpenAI and OpenRouter (by setting api_base).
    """
    llm_instance: ChatOpenAI
    
    if model_identifier == "gpt-4o":
        # Assumes OPENAI_API_KEY is set in the environment (either by script or shell)
        print(f"Initializing LLM: OpenAI model '{model_identifier}'")
        llm_instance = ChatOpenAI(model=model_identifier, temperature=0)
    else: # Assume it's an OpenRouter model (e.g., "mistralai/mistral-7b-instruct", "meta-llama/llama-3.1-8b-instruct:free")
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Please set it to use OpenRouter models (e.g., 'export OPENROUTER_API_KEY=your_key')."
            )
        
        openrouter_model_name = model_identifier # The identifier is the model name for OpenRouter
            
        print(f"Initializing LLM: OpenRouter model '{openrouter_model_name}' with site URL '{OPENROUTER_SITE_URL}' and site name '{OPENROUTER_SITE_NAME}'")
        llm_instance = ChatOpenAI(
            model_name=openrouter_model_name,
            temperature=0, # Good default for tool use
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=OPENROUTER_API_KEY,
        )
        # If model_identifier is invalid for OpenRouter, the API call will fail,
        # which is the desired behavior. No need for a specific "Unsupported model" error here.
        
    return llm_instance.bind_tools(tools)

def build_graph(model_identifier: str) -> Any:
    """Build and compile the LangGraph chatbot graph using the specified LLM."""
    graph_builder = StateGraph(State)
    tools = [create_calendar_event, delete_calendar_event, get_calendar_events, get_calendar_event]
    
    llm_with_tools = _get_llm_with_tools(model_identifier, tools)

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
    parser.add_argument(
        '--model',
        type=str,
        default="qwen/qwen3-32b",
        help='LLM to use. Examples: "gpt-4o", or an OpenRouter model like "mistralai/mistral-7b-instruct", "meta-llama/llama-3.1-8b-instruct:free"'
    )
    args = parser.parse_args()

    # Warning for OpenRouter models if API key is missing
    if args.model != "gpt-4o" and not OPENROUTER_API_KEY:
        print(f"Warning: Attempting to use OpenRouter model '{args.model}' but OPENROUTER_API_KEY environment variable is not set.")
        print("The application will likely fail if this model requires an API key and it's not found by other means.")

    system_time = get_current_time()
    
    # Enhanced system prompt for better tool usage with the Llama model
    if "llama-3.2-3b" in args.model:
        system_prompt = (
            f"The current date and time is {system_time} (Europe/Amsterdam). "
            f"IMPORTANT: When you use a tool, you MUST provide complete and valid JSON arguments. "
            f"FORMAT REQUIREMENTS:\n"
            f"1. For any date/time fields, ALWAYS use ISO format like '2024-07-17T10:00:00' (not natural language like 'tomorrow' or 'next Monday').\n"
            f"2. For timezone fields, ALWAYS use the complete string 'Europe/Amsterdam'.\n"
            f"3. All string values in JSON must be properly quoted and all JSON objects must have closing braces.\n"
            f"4. Argument example for create_calendar_event: {{\"summary\": \"Meeting\", \"start_datetime\": \"2024-07-17T10:00:00\", \"end_datetime\": \"2024-07-17T11:00:00\", \"timezone\": \"Europe/Amsterdam\"}}\n"
            f"5. Argument example for get_calendar_events: {{\"start_datetime\": \"2024-07-17T00:00:00\", \"end_datetime\": \"2024-07-17T23:59:59\", \"time_zone\": \"Europe/Amsterdam\"}}"
        )
    else:
        system_prompt = f"The current date and time is {system_time} (Europe/Amsterdam)."
    
    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=args.message)
    ]

    graph = build_graph(args.model)
    langfuse_handler = get_langfuse_handler()
    events = graph.stream({"messages": initial_messages}, config={"callbacks": [langfuse_handler]})
    for event in events:
        print(event)

if __name__ == "__main__":
    main()



