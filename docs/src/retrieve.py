
# Retrieval with authorization filter

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

def authorize_docs(docs: List[Dict[str, Any]], user_ctx: Dict[str, Any]) -> List[int]:
    """
    Example policy: user can only see docs matching tenant and sensitivity rules.
    """
    allowed = []
    user_tenant = user_ctx.get("tenant", "enterprise")
    user_clearance = user_ctx.get("clearance", "internal")  # public|internal|restricted

    clearance_rank = {"public": 0, "internal": 1, "restricted": 2}

    for i, d in enumerate(docs):
        if d.get("tenant") != user_tenant:
            continue
        if clearance_rank.get(d.get("sensitivity", "internal"), 1) <= clearance_rank.get(user_clearance, 1):
            allowed.append(i)
    return allowed

def embed_query(query: str, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    v = model.encode([query], normalize_embeddings=True)
    return np.array(v, dtype="float32")

def retrieve_topk(index, docs, query: str, model_name: str, user_ctx: Dict[str, Any], k: int = 5):
    allowed_idx = authorize_docs(docs, user_ctx)
    if not allowed_idx:
        return []

    qv = embed_query(query, model_name)

    # Search a larger pool then filter (simple approach)
    D, I = index.search(qv, min(len(docs), max(k * 5, 10)))
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        if idx in allowed_idx:
            hits.append((float(score), docs[idx]))
        if len(hits) >= k:
            break
    return hits
