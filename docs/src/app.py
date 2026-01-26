# End-to-end API (FastAPI)

import os
from fastapi import FastAPI
from pydantic import BaseModel

from .config import settings
from .pii import redact_pii
from .ingest import load_docs, save_docstore_jsonl
from .embed_store import build_embeddings, build_faiss_index, save_faiss, load_faiss, load_docstore
from .retrieve import retrieve_topk
from .prompt import build_prompt
from .llm_client import generate
from .guardrails import validate_output
from .telemetry import log_event

app = FastAPI(title="Enterprise RAG Reference Implementation")

class QueryReq(BaseModel):
    query: str
    tenant: str = "enterprise"
    clearance: str = "internal"  # public|internal|restricted
    redact_pii: bool = True

@app.post("/build-index")
def build_index():
    docs = load_docs(settings.APPROVED_SOURCES_DIR, tenant=settings.DEFAULT_TENANT)
    save_docstore_jsonl(docs, settings.DOCSTORE_PATH)

    texts = [d.text for d in docs]
    vectors = build_embeddings(texts, settings.EMBED_MODEL)
    index = build_faiss_index(vectors)
    save_faiss(index, settings.FAISS_INDEX_PATH)

    return {"status": "ok", "docs_indexed": len(docs), "faiss": settings.FAISS_INDEX_PATH}

@app.post("/ask")
def ask(req: QueryReq):
    # Load index/docstore
    if not (os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.DOCSTORE_PATH)):
        return {"error": "Index not built. Call /build-index first."}

    index = load_faiss(settings.FAISS_INDEX_PATH)
    docs = load_docstore(settings.DOCSTORE_PATH)

    user_ctx = {"tenant": req.tenant, "clearance": req.clearance}

    q = redact_pii(req.query) if (settings.ENABLE_PII_REDACTION and req.redact_pii) else req.query

    hits = retrieve_topk(index, docs, q, settings.EMBED_MODEL, user_ctx, k=5)
    prompt = build_prompt(q, hits)
    raw = generate(prompt)
    final = validate_output(raw)

    log_event({
        "event": "ask",
        "tenant": req.tenant,
        "clearance": req.clearance,
        "query": req.query,
        "query_redacted": q,
        "hits": [{"title": d.get("title"), "domain": d.get("domain"), "score": s} for s, d in hits],
    })

    return {"answer": final, "sources": [d.get("title") for _, d in hits]}
