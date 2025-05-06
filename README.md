# CalendarThesis Chatbot

This project implements a chatbot capable of interacting with Google Calendar using various Language Learning Models (LLMs) via Langchain and LangGraph. It supports models from OpenAI (e.g., GPT-4o) and OpenRouter.

## Setup

### Prerequisites
- Python 3.8+
- Pip (Python package installer)

### Installation
1.  Clone the repository (if you haven't already):
    ```bash
    git clone <your-repository-url>
    cd calendarthesis
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt 
    ```
    *(Assuming you have a `requirements.txt` file. If not, you'll need to install the necessary packages like `langchain`, `langchain-openai`, `langgraph`, `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`, `python-dotenv`, `pandas`, `pytz`, `langfuse` etc., manually or generate a `requirements.txt` file.)*

### Environment Variables
Create a `.env` file in the root of the `calendarthesis` directory or set the following environment variables in your shell:

1.  **OpenAI API Key (Required for GPT-4o):**
    ```
    OPENAI_API_KEY="your_openai_api_key"
    ```
    This is used when `--model "gpt-4o"` is specified. The script also has a hardcoded OpenAI key, but using an environment variable is best practice.

2.  **OpenRouter API Key (Required for OpenRouter models):**
    ```
    OPENROUTER_API_KEY="your_openrouter_api_key"
    ```
    This is required if you intend to use models from OpenRouter (e.g., `"mistralai/mistral-7b-instruct"`).
    **Security Note:** While `main.py` currently includes a default fallback key for OpenRouter, it is **strongly recommended** to set your own key via this environment variable and remove the hardcoded default from the script to avoid exposing sensitive credentials.

3.  **OpenRouter Optional Headers (Optional):**
    These are used to identify your site/app to OpenRouter for ranking purposes if you pass them in the `ChatOpenAI` call (currently, these headers are not being explicitly passed in `main.py` after recent changes).
    ```
    OPENROUTER_SITE_URL="your_site_url"
    OPENROUTER_SITE_NAME="your_site_name"
    ```
    If not set, `main.py` defines defaults (`http://localhost:3000` and `CalendarThesisApp` respectively), but these are only used if the logic to send headers is re-instated.

4.  **Langfuse (Optional but recommended for tracing):**
    The script is configured to use Langfuse for tracing. The credentials for Langfuse are currently hardcoded in `get_langfuse_handler()` in both `main.py` and `eval.py`. For better practice, consider moving these to environment variables as well.
    ```
    LANGFUSE_PUBLIC_KEY="pk-lf-..."
    LANGFUSE_SECRET_KEY="sk-lf-..."
    LANGFUSE_HOST="https://cloud.langfuse.com" 
    ```

You will also need to have a `credentials.json` file for Google Calendar API access, and a `token.json` will be generated after the first successful authentication. Ensure `calendar_tools.py` is correctly set up for this.

## Running the Chatbot

### Main Interaction (`main.py`)
This script allows you to send a single message to the chatbot.

**Command Structure:**
```bash
python calendarthesis/main.py --message "Your message to the chatbot" --model "model_identifier"
```

**Arguments:**
-   `--message`: (Optional) The message string to send to the chatbot. Defaults to "Schedule meeting".
-   `--model`: (Optional) The LLM to use. Defaults to "gpt-4o".
    -   For OpenAI: `"gpt-4o"`
    -   For OpenRouter: Use the model identifier directly, e.g., `"mistralai/mistral-7b-instruct"`, `"meta-llama/llama-3.1-8b-instruct:free"`, `"google/gemma-7b-it:free"`.

**Examples:**
1.  Using GPT-4o (default):
    ```bash
    python calendarthesis/main.py --message "What's on my calendar tomorrow?"
    ```
    or explicitly:
    ```bash
    python calendarthesis/main.py --message "What's on my calendar tomorrow?" --model "gpt-4o"
    ```

2.  Using an OpenRouter model (e.g., Mistral 7B Instruct):
    ```bash
    export OPENROUTER_API_KEY="your_openrouter_api_key" # Ensure this is set
    python calendarthesis/main.py --message "Schedule a meeting with Jane for next Monday at 10 AM" --model "mistralai/mistral-7b-instruct"
    ```

### Evaluation (`eval.py`)
This script evaluates the chatbot's tool usage against a set of test inputs provided in a CSV file.

**Command Structure:**
```bash
python calendarthesis/eval.py --input_csv "path/to/your/test_inputs.csv" --output_csv "path/to/your/results.csv" --model "model_identifier"
```

**Arguments:**
-   `--input_csv`: (Optional) Path to the input CSV file containing test cases. Defaults to `calendarthesis/test_inputs.csv`.
    The CSV should have columns: `input`, `expected_tool_name`, `expected_tool_args` (as a JSON string).
-   `--output_csv`: (Optional) Path where the evaluation results CSV will be written. Defaults to `calendarthesis/evaluation_results.csv`.
-   `--model`: (Optional) The LLM to use for the evaluation. Defaults to "gpt-4o". Model identifiers are the same as for `main.py`.

**Examples:**
1.  Evaluating with GPT-4o (default):
    ```bash
    python calendarthesis/eval.py --input_csv "data/my_tests.csv" --output_csv "results/gpt4o_eval.csv"
    ```
    or explicitly:
    ```bash
    python calendarthesis/eval.py --input_csv "data/my_tests.csv" --output_csv "results/gpt4o_eval.csv" --model "gpt-4o"
    ```

2.  Evaluating with an OpenRouter model (e.g., Llama 3.1 8B Instruct):
    ```bash
    export OPENROUTER_API_KEY="your_openrouter_api_key" # Ensure this is set
    python calendarthesis/eval.py --input_csv "data/my_tests.csv" --output_csv "results/llama3_8b_eval.csv" --model "meta-llama/llama-3.1-8b-instruct:free"
    ```

## Supported LLM Models

The scripts support:
-   **OpenAI:**
    -   `gpt-4o` (and potentially other models available via `ChatOpenAI` if you modify the identifier)
-   **OpenRouter:**
    -   A wide range of models available through their API. You provide the model's string identifier as listed on OpenRouter (e.g., `mistralai/mistral-7b-instruct`, `meta-llama/llama-3.1-8b-instruct:free`, `anthropic/claude-3-haiku`, etc.).

Ensure you have the respective API keys configured as environment variables.