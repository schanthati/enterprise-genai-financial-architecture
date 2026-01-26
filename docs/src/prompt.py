# Prompt assembly

from typing import List, Dict, Any, Tuple

SYSTEM = """You are an enterprise assistant.
Use only the provided approved-source context.
If the answer is not in the context, say you do not have enough approved information.
Return concise, accurate answers and cite source titles when relevant.
"""

def build_prompt(user_query: str, contexts: List[Tuple[float, Dict[str, Any]]]) -> Dict[str, str]:
    ctx_blocks = []
    for score, d in contexts:
        ctx_blocks.append(f"[SOURCE: {d.get('title')} | domain={d.get('domain')} | score={score:.3f}]\n{d.get('text')}\n")

    context_text = "\n---\n".join(ctx_blocks) if ctx_blocks else "NO_CONTEXT"
    user = f"""User question:
{user_query}

Approved-source context:
{context_text}

Answer:"""
    return {"system": SYSTEM, "user": user}
