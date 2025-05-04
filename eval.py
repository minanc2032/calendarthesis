import argparse
import os
import pandas as pd
import pytz
import time
import json
from datetime import datetime, timezone, timedelta, time as dt_time
from dateutil import parser as dateutil_parser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langfuse.callback import CallbackHandler
from typing import Optional, List, Dict, Any
from main import build_graph, get_langfuse_handler

# Define a fixed time for evaluation consistency
FIXED_EVAL_TIME = "2024-07-16 09:00:00"
AMSTERDAM_TZ_INFO = "Europe/Amsterdam"

# --- Define Tool Default Arguments ---
TOOL_DEFAULTS = {
    "create_calendar_event": {
        "timezone": "Europe/Amsterdam"
    },
    "get_calendar_events": {
        "calendar_id": "primary",
        "max_results": 10,
        "order_by": "startTime",
        "time_zone": "Europe/Amsterdam", # Note: Tool uses time_zone, not timezone
    },
    "get_calendar_event": {
        "calendar_id": "primary",
        "timezone": "Europe/Amsterdam"
    },
    "delete_calendar_event": {}
}
# ---

def evaluate(input_csv: str, output_csv: str) -> None:
    """
    Evaluate the chatbot across test inputs in a CSV, recording latency,
    token usage, and comparing actual tool calls against expected ones.
    Args:
        input_csv: Path to CSV file with columns "input", "expected_tool_name",
                   and "expected_tool_args" (as JSON string).
        output_csv: Path where the results CSV will be written.
    """
    df = pd.read_csv(input_csv, engine='python')
    results = []
    # Use the fixed time for the system message
    system_time = f"{FIXED_EVAL_TIME} ({AMSTERDAM_TZ_INFO})"
    graph = build_graph()
    langfuse_handler = get_langfuse_handler()

    # Initialize counters
    correct_count = 0
    incorrect_count = 0
    total_count = 0

    for index, row in df.iterrows():
        total_count += 1
        user_input = row.get('input')
        expected_tool_name = row.get('expected_tool_name')
        expected_tool_args_str = row.get('expected_tool_args')

        if pd.isna(user_input) or pd.isna(expected_tool_name) or pd.isna(expected_tool_args_str):
            print(f"Skipping row {index+2}: Missing required data.")
            continue

        # Parse expected args
        try:
            expected_tool_args = json.loads(expected_tool_args_str)
        except json.JSONDecodeError as e:
            print(f"Skipping row {index+2}: Invalid JSON in expected_tool_args: {e}")
            continue

        start_time = time.time()
        # Define time interpretations
        time_definitions = (
            "Interpret time ranges as follows unless specified otherwise: \n"
            "- 'all day': 00:00:00 to 23:59:59 of the specified day. \n"
            "- 'afternoon': 12:00:00 to 17:00:00. \n"
            "- 'evening': 17:00:00 to 21:00:00. \n"
            "- 'night': 21:00:00 to 06:00:00 the following day."
        )

        system_prompt = (
            f"You are a helpful assistant. The current date and time for ALL calculations is EXACTLY {system_time}. "
            f"Do NOT use any other time reference. Interpret all relative dates (like 'tomorrow', 'this Friday', "
            f"'next week') based *only* on this provided date and time. \n"
            f"{time_definitions}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=str(user_input))
        ]

        response_chunks = []
        token_usage = None
        actual_tool_calls: List[Dict[str, Any]] = [] # Store actual tool calls here
        full_response = ""
        error = None
        success = True # Represents successful execution, not evaluation pass/fail

        try:
            events = graph.stream({"messages": messages}, config={"callbacks": [langfuse_handler]})
            # Iterate through events robustly
            for event in events:
                if isinstance(event, dict):
                     # Process each key-value pair within the dictionary event
                     for event_key, event_data in event.items():
                        # --- Tool Call Interception ---
                        # Check if the event is from the 'chatbot' node and contains an AIMessage with tool_calls
                        if event_key == "chatbot" and isinstance(event_data, dict):
                            messages_in_event = event_data.get("messages", [])
                            if messages_in_event:
                                last_message = messages_in_event[-1]
                                if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                    for tool_call in last_message.tool_calls:
                                        # Ensure args is a dict, might be None or other types if parsing fails
                                        args_dict = tool_call.get('args') if isinstance(tool_call.get('args'), dict) else {}
                                        actual_tool_calls.append({
                                            "name": tool_call.get('name'),
                                            "args": args_dict
                                        })
                        # --- Existing Logic for response and metadata ---
                        if isinstance(event_data, dict) and 'content' in event_data:
                             # This might capture intermediate content, let's accumulate from AIMessage instead later
                             pass # response_chunks.append(event_data['content']) # Original line commented out
                        elif hasattr(event_data, 'content'):
                             pass # response_chunks.append(event_data.content) # Original line commented out

                        # Extract final response from the AIMessage if it's the final response
                        if event_key == "chatbot" and isinstance(event_data, dict):
                            messages_in_event = event_data.get("messages", [])
                            if messages_in_event:
                                last_message = messages_in_event[-1]
                                # Check if it's the final AI response (no tool calls)
                                if isinstance(last_message, AIMessage) and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                                     full_response = last_message.content.strip()


                        metadata = None
                        if isinstance(event_data, dict) and 'response_metadata' in event_data:
                            metadata = event_data['response_metadata']
                        elif hasattr(event_data, 'response_metadata'):
                            metadata = event_data.response_metadata

                        if metadata is not None:
                            # Handle potential nested structure for token usage more robustly
                            if isinstance(metadata, dict):
                                usage_info = metadata.get('token_usage') or metadata.get('usage_metadata')
                                if isinstance(usage_info, dict):
                                    token_usage = usage_info # Store the whole dict
                                elif hasattr(metadata, 'token_usage'): # Fallback for direct attribute
                                    token_usage = metadata.token_usage
                            elif hasattr(metadata, 'token_usage'):
                                 token_usage = metadata.token_usage


        except Exception as e:
            print(f"Error during graph execution for input: {user_input[:50]}... - {e}")
            full_response = ""
            success = False
            error = str(e)

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # --- Evaluation Logic ---
        evaluation_result = "FAIL"
        matched_tool_call = None
        for actual_call in actual_tool_calls:
            actual_args = actual_call.get("args", {})
            actual_name = actual_call.get("name")

            # 1. Compare tool name
            if actual_name != expected_tool_name:
                continue # Try next actual call

            # 2. Compare arguments flexibly
            args_match = True
            processed_keys = set()

            # Compare keys present in expected_args
            for key, expected_value in expected_tool_args.items():
                processed_keys.add(key)
                actual_value = actual_args.get(key)

                # Check if the actual argument is missing
                if actual_value is None and key in actual_args:
                    # Handle case where arg exists but is explicitly None
                    # Decide if None is acceptable based on tool/context, for now treat as mismatch if expected is not None
                    if expected_value is not None:
                         args_match = False
                         break
                    else:
                         continue # Both are None, OK.
                
                if key not in actual_args:
                    # Actual argument is missing. Check against defaults.
                    tool_defaults_for_func = TOOL_DEFAULTS.get(actual_name, {})
                    if key in tool_defaults_for_func:
                        # Parameter has a default value defined
                        default_value = tool_defaults_for_func[key]
                        # Compare expected value with the tool's default value
                        # Use flexible comparison if needed (e.g., for default dates/times if they existed)
                        if key == "summary": # Example if default summary existed
                             if str(expected_value).lower() != str(default_value).lower():
                                  args_match = False
                                  break # Expected non-default value, but agent omitted
                        elif expected_value != default_value:
                             # Add a fallback for potential type differences (e.g., int vs str)
                             if str(expected_value) != str(default_value):
                                 args_match = False
                                 break # Expected non-default value, but agent omitted
                        # If expected_value == default_value (or str representations match), it's OK, continue loop.
                    else:
                         # Parameter has no default, but was expected.
                         args_match = False
                         break # Agent omitted required parameter
                    continue # Move to next key after handling missing arg
                
                # --- Argument exists in actual_args, perform comparison ---
                if key in ["start_datetime", "end_datetime"]:
                    # Flexible comparison logic
                    try:
                        # Attempt to parse both as datetimes
                        dt_expected_raw = datetime.fromisoformat(str(expected_value).replace('Z', '+00:00'))
                        dt_actual_raw = datetime.fromisoformat(str(actual_value).replace('Z', '+00:00'))

                        # Make timezone naive for comparison
                        dt_expected = dt_expected_raw.replace(tzinfo=None)
                        dt_actual = dt_actual_raw.replace(tzinfo=None)

                        # Assume match unless proven otherwise by checks below
                        arg_matches = True 

                        # 1. Exact match check (applies to all datetimes)
                        if dt_expected == dt_actual:
                            arg_matches = True # It's a match
                        # 2. End-of-day equivalence check (only for end_datetime)
                        elif key == "end_datetime":
                            t_expected = dt_expected.time()
                            t_actual = dt_actual.time()
                            d_expected = dt_expected.date()
                            d_actual = dt_actual.date()

                            is_end_of_day_equivalent = False
                            # Case 1: Expected is 23:59:59, Actual is 00:00:00 next day
                            if t_expected == dt_time(23, 59, 59) and t_actual == dt_time(0, 0, 0) and d_actual == d_expected + timedelta(days=1):
                                is_end_of_day_equivalent = True
                            # Case 2: Actual is 23:59:59, Expected is 00:00:00 next day
                            elif t_actual == dt_time(23, 59, 59) and t_expected == dt_time(0, 0, 0) and d_expected == d_actual + timedelta(days=1):
                                is_end_of_day_equivalent = True

                            if is_end_of_day_equivalent:
                                arg_matches = True # Consider it a match
                            else:
                                arg_matches = False # Not exact, not equivalent end-of-day
                        # 3. Not end_datetime and not an exact match
                        else: # Must be start_datetime and not an exact match
                            arg_matches = False 

                        # If this specific datetime arg didn't match any criteria
                        if not arg_matches:
                            args_match = False
                            break

                    except ValueError:
                        # Fallback to string comparison if parsing fails
                        if str(expected_value) != str(actual_value):
                            args_match = False
                            break
                elif key == "summary":
                    # Compare summary case-insensitively
                    if str(expected_value).lower() != str(actual_value).lower():
                        args_match = False
                        break
                else:
                    # Direct comparison for other keys (handle type differences if needed, e.g., int vs str)
                    # Basic comparison might fail if types differ (e.g., expected 10, actual '10')
                    # Add type casting if necessary: e.g. str(expected_value) != str(actual_value)
                    if type(expected_value) != type(actual_value):
                         # Attempt common type casting for comparison (e.g., int/float/str)
                         try:
                              if type(expected_value) is int and float(expected_value) != float(actual_value):
                                   args_match = False; break
                              elif type(expected_value) is float and float(expected_value) != float(actual_value):
                                   args_match = False; break
                              # Add other casting rules if needed (e.g., bool('True') vs True)
                              elif str(expected_value) != str(actual_value): # General fallback
                                   args_match = False; break
                         except (ValueError, TypeError):
                              args_match = False; break # Cannot cast/compare types
                    elif expected_value != actual_value:
                         args_match = False
                         break

            if not args_match:
                continue # Argument mismatch for this call, try next actual call

            # 3. (Optional Check) Check if actual_args has extra keys not in expected_args
            # Since we updated expected_args to include defaults, this check might not be needed
            # If needed, uncomment the following:
            # for key in actual_args:
            #     if key not in processed_keys:
            #         # Decide how to handle extra keys (e.g., ignore if default, fail otherwise)
            #         # For now, we assume if expected keys match, it's a pass
            #         pass 

            # If we reach here, all expected args matched this actual call
            evaluation_result = "PASS"
            matched_tool_call = actual_call
            break # Stop searching once a match is found

        # Increment counters based on evaluation result
        if evaluation_result == "PASS":
            correct_count += 1
        else:
            incorrect_count += 1

        result = {
            'input': user_input,
            'expected_tool_name': expected_tool_name,
            'expected_tool_args': expected_tool_args_str, # Store original string
            'actual_tool_calls': json.dumps(actual_tool_calls), # Store actual calls as JSON string
            'matched_tool_call': json.dumps(matched_tool_call) if matched_tool_call else None, # Store matched call as JSON string
            'evaluation_result': evaluation_result,
            'output': full_response, # Store final agent textual response
            'latency_ms': latency_ms,
            'token_usage': json.dumps(token_usage) if token_usage else None, # Store usage as JSON string
            'execution_success': success # Renamed to avoid confusion
        }
        if not success:
            result['error'] = error

        print(result, "\n\n")

        results.append(result)

    out_df = pd.DataFrame(results)
    # Define column order for clarity
    column_order = [
        'input', 'expected_tool_name', 'expected_tool_args', 'actual_tool_calls',
        'matched_tool_call', 'evaluation_result', 'output', 'latency_ms',
        'token_usage', 'execution_success', 'error'
    ]
    # Ensure all columns exist, fill missing ones like 'error' with NaN if needed
    out_df = out_df.reindex(columns=column_order)
    out_df.to_csv(output_csv, index=False)

    # --- Print Summary Statistics ---
    print("\n--- Evaluation Summary ---")
    print(f"Total Test Cases: {total_count}")
    print(f"Correct (PASS):   {correct_count}")
    print(f"Incorrect (FAIL): {incorrect_count}")
    print(f"Accuracy:         {correct_count / total_count:.2%}")
    print(f"Results saved to: {output_csv}")
    # ---

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate chatbot tool usage against expected calls.'
    )
    parser.add_argument('--input_csv', type=str, default='calendarthesis/test_inputs.csv', help='Path to CSV file with inputs and expected tool calls.')
    parser.add_argument('--output_csv', type=str, default='calendarthesis/evaluation_results.csv', help='Path where the evaluation results CSV will be written.')
    args = parser.parse_args()
    evaluate(args.input_csv, args.output_csv)

if __name__ == '__main__':
    main()
