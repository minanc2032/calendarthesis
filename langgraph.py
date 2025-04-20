import argparse
import os
from datetime import datetime
import pytz

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.create_event import CalendarCreateEvent
from langchain_google_community.calendar.search_events import CalendarSearchEvents
from langchain_google_community.calendar.get_calendars_info import GetCalendarsInfo

# tool = GetCalendarsInfo()
# output = tool.invoke({})
# print(output)



toolkit = CalendarToolkit()
tools = toolkit.get_tools()
print(tools)

# Get current time in Amsterdam timezone
amsterdam_tz = pytz.timezone('Europe/Amsterdam')
current_time = datetime.now(amsterdam_tz).strftime('%Y-%m-%d %H:%M:%S')
SYSTEM_PROMPT = f"System: The current date and time is {current_time} (Europe/Amsterdam)."


os.environ["OPENAI_API_KEY"] = "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"
@tool
def create_calendar_event(
    start_datetime: str,
    end_datetime: str, 
    summary: str,
    timezone: str = "Europe/Amsterdam"
) -> str:
    """Create a calendar event.
    
    Args:
        start_datetime: The start date and time of the event in YYYY-MM-DD HH:MM:SS format
        end_datetime: Upper bound (exclusive) for an event's end date/time, in 'YYYY-MM-DD HH:MM:SS' format.
        timezone: The timezone for the event, defaults to Europe/Amsterdam
        summary: The reason for the event if not provided, it will be 'No summary provided' 
        
    Returns:
        str: Confirmation message that event was scheduled
    """
    tool = CalendarCreateEvent()
    tool.invoke(
        {
            "summary": summary,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "timezone": timezone,
            "location": "UAM Cuajimalpa",
            "description": "Event created from the LangChain toolkit",
            "reminders": [{"method": "popup", "minutes": 60}],
            "conference_data": True,
            "color_id": "5",
        }
    )
    return "Event scheduled"
@tool
def delete_calendar_event():
    """Deletes a calendar event."""
    return f"Event deleted"

@tool
def get_calendar_events(
    start_datetime: None,
    end_datetime: None,
    calendar_id: str = "primary",
    max_results: int = 10,
    order_by: str = "startTime",
    time_zone: str = "Europe/Amsterdam",
) -> str:
    """Retrieve a list of events from a Google Calendar.

    Args:
        start_datetime: Lower bound (inclusive) for an event's start date/time, in 'YYYY-MM-DD HH:MM:SS' format.
        end_datetime: Upper bound (exclusive) for an event's end date/time, in 'YYYY-MM-DD HH:MM:SS' format.
        calendar_id: The calendar to fetch from (e.g. 'primary').
        max_results: Maximum number of events to return (default: 10).
        order_by: Sort field, either 'startTime' or 'updated' (default: 'startTime').
        time_zone: Time zone to use for interpreting the start/end filters (default: Europe/Amsterdam).
    Returns:
        str: list of events matching the query.
    """
    get_calendar_info = GetCalendarsInfo()
    calendars_info = get_calendar_info.invoke({})
    tool = CalendarSearchEvents()
    events = tool.invoke(
        {
            "calendars_info": calendars_info,
            "max_datetime": end_datetime,
            "max_results": 10,
            "min_datetime": start_datetime,
            "order_by": 'startTime',
        }
    )
    print(events)
    return "Events retrieved"

@tool
def get_calendar_event(
    event_id: str,
    calendar_id: str = "primary",
    timezone: str = "Europe/Amsterdam"
) -> str:
    """Retrieve a single calendar event by its ID.

    Args:
        event_id: The unique identifier of the event to fetch.
        calendar_id: The calendar to retrieve it from (e.g. 'primary'; default: primary).
        timezone: Time zone to use when interpreting the event's date/time fields, in 'Region/City' format (default: Europe/Amsterdam).

    Returns:
        str: A representation of the retrieved event.
    """
    return "Event retrieved"

from typing import Annotated

from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tools = [create_calendar_event, delete_calendar_event, get_calendar_events, get_calendar_event ]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[create_calendar_event, delete_calendar_event, get_calendar_events, get_calendar_event])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangGraph chatbot with a custom message and OpenAI API key.")
    parser.add_argument('--message', type=str, default="Schedule meeting", help='Message to send to the chatbot')
    args = parser.parse_args()

    system_time = current_time
    initial_messages = [
        SystemMessage(content=f"The current date and time is {system_time}."),
        HumanMessage(content=args.message)
    ]


    events = graph.stream({"messages": initial_messages})
    for event in events:
        print(event)


