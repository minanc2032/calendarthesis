import argparse
import pandas as pd
import pytz
import time
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from mine_langgraph import graph


def evaluate(input_csv: str, output_csv: str):
    # Load test cases
    df = pd.read_csv(input_csv)
    results = []

    # Prepare timezone and a single system timestamp
    tz = pytz.timezone('Europe/Amsterdam')
    system_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

    for _, row in df.iterrows():
        user_input = row.get('input')
        if pd.isna(user_input):
            continue

        # Start latency timer
        start_time = time.time()

        # Build message sequence
        messages = [
            SystemMessage(content=f"The current date and time is {system_time} (Europe/Amsterdam)."),
            HumanMessage(content=str(user_input))
        ]

        response_chunks = []
        token_usage = None
        try:
            # Stream through the graph, collecting response and metadata
            for event in graph.stream({"messages": messages}):
                if isinstance(event, dict) and 'content' in event:
                    response_chunks.append(event['content'])
                elif hasattr(event, 'content'):
                    response_chunks.append(event.content)

                # Capture token_usage if present in metadata
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
        except Exception as e:
            full_response = ""
            success = False
            error = str(e)

        # End latency timer
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # Build result row
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

    # Write results to CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate mine_langgraph chatbot across test inputs in a CSV, recording latency and token usage.'
    )
    parser.add_argument(
        '--input_csv', type=str, required=True,
        help='Path to CSV file with a column "input" of test messages.'
    )
    parser.add_argument(
        '--output_csv', type=str, required=True,
        help='Path where the results CSV will be written.'
    )
    args = parser.parse_args()
    evaluate(args.input_csv, args.output_csv)
