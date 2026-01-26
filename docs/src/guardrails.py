# guardrails.py
def validate_output(text: str) -> str:
    # placeholder checks (extend as needed)
    if "password" in text.lower():
        return "Output blocked by policy."
    return text
