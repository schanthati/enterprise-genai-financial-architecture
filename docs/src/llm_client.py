# LLM client
from typing import Dict
from .config import settings

def generate_stub(prompt: Dict[str, str]) -> str:
    # This is a placeholder: returns a deterministic response based on context presence.
    user = prompt["user"]
    if "NO_CONTEXT" in user:
        return "I do not have enough approved information to answer that."
    return "Based on the approved context, here is a grounded summary with source references."

def generate_openai(prompt: Dict[str, str]) -> str:
    # Optional; requires: pip install openai and OPENAI_API_KEY env var
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def generate(prompt: Dict[str, str]) -> str:
    if settings.LLM_PROVIDER == "openai":
        return generate_openai(prompt)
    return generate_stub(prompt)
