import pandas as pd, json

df = pd.read_csv("results_mistral8b.csv")  # Changed to mistral results
tool_name_match = []
arg_syntax_ok   = []

for _, row in df.iterrows():
    expected = row.expected_tool_name
    actual_calls = json.loads(row.actual_tool_calls or "[]")
    if not actual_calls:           # model made no call
        tool_name_match.append(False)
        arg_syntax_ok.append(False)
        continue

    actual = actual_calls[0]       # we evaluate only the first call
    tool_name_match.append(actual["name"] == expected)

    # --- minimal "syntactically valid args" check ---
    try:
        ok = isinstance(actual["args"], dict)          # JSON object
        ok &= all(k in actual["args"] for k in
                  json.loads(row.expected_tool_args).keys())  # required keys present
    except Exception:
        ok = False
    arg_syntax_ok.append(ok)

df["tool_name_ok"] = tool_name_match
df["arg_syntax_ok"] = arg_syntax_ok

# Convert 'evaluation_result' to numeric (1 for PASS, 0 for FAIL)
df["evaluation_result_numeric"] = df["evaluation_result"].apply(lambda x: 1 if x == "PASS" else 0)

summary = df.agg({
    "tool_name_ok":  "mean",
    "arg_syntax_ok": "mean",
    "evaluation_result_numeric": "mean" # Use the new numeric column
}) * 100
print(summary)
