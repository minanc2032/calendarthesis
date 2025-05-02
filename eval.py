import argparse
import os
import pandas as pd
import pytz
import time
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langfuse.callback import CallbackHandler
from typing import Optional
from main import build_graph, get_langfuse_handler, get_current_time

def evaluate(input_csv: str, output_csv: str) -> None:
    """
    Evaluate the chatbot across test inputs in a CSV, recording latency and token usage.
    Args:
        input_csv: Path to CSV file with a column "input" of test messages.
        output_csv: Path where the results CSV will be written.
    """
    df = pd.read_csv(input_csv)
    results = []
    system_time = get_current_time()
    graph = build_graph()
    langfuse_handler = get_langfuse_handler()

    for _, row in df.iterrows():
        user_input = row.get('input')
        if pd.isna(user_input):
            continue
        start_time = time.time()
        messages = [
            SystemMessage(content=f"The current date and time is {system_time} (Europe/Amsterdam)."),
            HumanMessage(content=str(user_input))
        ]
        response_chunks = []
        token_usage = None
        try:
            for event in graph.stream({"messages": messages}, config={"callbacks": [langfuse_handler]}):
                if isinstance(event, dict) and 'content' in event:
                    response_chunks.append(event['content'])
                elif hasattr(event, 'content'):
                    response_chunks.append(event.content)
                metadata = None
                if isinstance(event, dict) and 'response_metadata' in event:
                    metadata = event['response_metadata']
                elif hasattr(event, 'response_metadata'):
                    metadata = event.response_metadata
                if metadata is not None:
                    if hasattr(metadata, 'token_usage'):
                        token_usage = metadata.token_usage
                    elif isinstance(metadata, dict) and 'token_usage' in metadata:
                        token_usage = metadata['token_usage']
            full_response = ''.join(response_chunks).strip()
            success = True
            error = None
        except Exception as e:
            full_response = ""
            success = False
            error = str(e)
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        result = {
            'input': user_input,
            'output': full_response,
            'latency_ms': latency_ms,
            'token_usage': token_usage,
            'success': success
        }
        if not success:
            result['error'] = error
        results.append(result)
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate chatbot across test inputs in a CSV, recording latency and token usage.'
    )
    parser.add_argument('--input_csv', type=str, required=True, help='Path to CSV file with a column "input" of test messages.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path where the results CSV will be written.')
    args = parser.parse_args()
    evaluate(args.input_csv, args.output_csv)

if __name__ == '__main__':
    main()

