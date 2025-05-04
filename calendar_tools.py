import os
from datetime import datetime, timedelta
from typing import Optional
import pytz
from langchain_core.tools import tool

# Timezone config
AMSTERDAM_TZ = pytz.timezone('Europe/Amsterdam')


def get_current_time() -> str:
    """Get the current time in Amsterdam timezone as a string."""
    return datetime.now(AMSTERDAM_TZ).strftime('%Y-%m-%d %H:%M:%S')


@tool
def create_calendar_event(
    start_datetime: str,
    end_datetime: str,
    summary: str,
    timezone: str = "Europe/Amsterdam"
) -> str:
    """Create a calendar event in Google Calendar."""
    
    return "Event scheduled"


@tool
def delete_calendar_event(event_id: str) -> str:
    """Deletes a specific calendar event using its unique event_id."""
    # In a real implementation, use event_id to delete the event via API
    print(f"Attempting to delete event with ID: {event_id}") # Add print for stub visibility
    return f"Event deleted (ID: {event_id})"


@tool
def get_calendar_events(
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    summary: Optional[str] = None,
    calendar_id: str = "primary",
    max_results: int = 10,
    order_by: str = "startTime",
    time_zone: str = "Europe/Amsterdam",
) -> str:
    """Retrieve a list of events from a Google Calendar between start_datetime and end_datetime. 
    Can optionally filter by event summary.
    If start_datetime is not provided, defaults to the current time.
    If end_datetime is not provided, defaults to 7 days from the current time.
    """
    if start_datetime is None:
        start_datetime = get_current_time()
    if end_datetime is None:
        end_datetime = (datetime.now(AMSTERDAM_TZ) + timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

    # Add print for stub visibility
    print(f"Searching events: Start='{start_datetime}', End='{end_datetime}', Summary='{summary}', MaxResults={max_results}")
    return "Events retrieved"


@tool
def get_calendar_event(
    event_id: str,
    calendar_id: str = "primary",
    timezone: str = "Europe/Amsterdam"
) -> str:
    """Retrieves the details for a single calendar event based on its unique event_id."""
    return "Event retrieved" 