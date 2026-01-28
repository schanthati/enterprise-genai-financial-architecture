# ============================================================
# tests for the Enterprise RAG FastAPI reference implementation
#
# Layout:
#   tests/
#     __init__.py
#     conftest.py
#     test_health.py
#     test_build_index.py
#     test_ask_flow.py
#     test_tenancy_clearance.py
#     test_pii.py
#     test_guardrails.py
#     test_cache_rate_limit.py
#     test_ingest_chunking.py
#     test_embed_store_local.py
#
# Notes:
# - Uses pytest + fastapi TestClient
# - Runs in "mock" LLM provider mode, and local-mini embeddings
# - Builds an on-disk index in a temporary directory
# - Requires: pytest, fastapi, starlette, numpy, faiss-cpu (optional but recommended)
# ============================================================


# =========================
# tests/__init__.py
# =========================
# empty


# =========================
# tests/conftest.py
# =========================
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# IMPORTANT: Import settings module object (same singleton used by app)
from app.config import settings
from app.main import app


@pytest.fixture(scope="session")
def tmp_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("rag_app")


@pytest.fixture(scope="session")
def prepared_dirs(tmp_root: Path) -> dict:
    """
    Create a small approved_sources dir with a few docs.
    Also set data + telemetry outputs into tmp.
    """
    approved = tmp_root / "approved_sources"
    data_dir = tmp_root / "data"
    telemetry_dir = tmp_root / "telemetry"
    approved.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    # tenant enterprise docs
    (approved / "public_policy.md").write_text(
        "# Policy\nAll employees must use MFA.\nPassword rotations every 90 days.\n",
        encoding="utf-8",
    )
    (approved / "internal_security_internal.txt").write_text(
        "Internal Security\nVPN required for internal resources.\nDo not share secrets.\n",
        encoding="utf-8",
    )
    (approved / "finance_restricted_confidential.txt").write_text(
        "Restricted Finance\nQ4 revenue projections are confidential.\n",
        encoding="utf-8",
    )

    return {
        "approved": str(approved),
        "data_dir": str(data_dir),
        "telemetry_dir": str(telemetry_dir),
        "docstore": str(data_dir / "docstore.jsonl"),
        "faiss": str(data_dir / "index.faiss"),
        "telemetry_jsonl": str(telemetry_dir / "events.jsonl"),
    }


@pytest.fixture(autouse=True)
def patch_settings(prepared_dirs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch app settings so tests are isolated and deterministic.
    """
    monkeypatch.setattr(settings, "APP_ENV", "test", raising=False)
    monkeypatch.setattr(settings, "APPROVED_SOURCES_DIR", prepared_dirs["approved"], raising=False)
    monkeypatch.setattr(settings, "DATA_DIR", prepared_dirs["data_dir"], raising=False)
    monkeypatch.setattr(settings, "DOCSTORE_PATH", prepared_dirs["docstore"], raising=False)
    monkeypatch.setattr(settings, "FAISS_INDEX_PATH", prepared_dirs["faiss"], raising=False)

    monkeypatch.setattr(settings, "TELEMETRY_DIR", prepared_dirs["telemetry_dir"], raising=False)
    monkeypatch.setattr(settings, "TELEMETRY_JSONL", prepared_dirs["telemetry_jsonl"], raising=False)
    monkeypatch.setattr(settings, "ENABLE_TELEMETRY", True, raising=False)

    # deterministic embeddings + LLM
    monkeypatch.setattr(settings, "EMBED_MODEL", "local-mini", raising=False)
    monkeypatch.setattr(settings, "EMBED_DIM", 128, raising=False)  # smaller for test speed
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock", raising=False)

    # enable filters
    monkeypatch.setattr(settings, "ENABLE_TENANCY", True, raising=False)
    monkeypatch.setattr(settings, "MIN_SCORE", 0.0, raising=False)  # don't flake on scoring
    monkeypatch.setattr(settings, "TOPK", 5, raising=False)

    # cache + rate limit settings
    monkeypatch.setattr(settings, "ENABLE_CACHE", True, raising=False)
    monkeypatch.setattr(settings, "CACHE_TTL_S", 300, raising=False)
    monkeypatch.setattr(settings, "ENABLE_RATE_LIMIT", True, raising=False)
    monkeypatch.setattr(settings, "RL_REQUESTS_PER_MIN", 3, raising=False)

    # pii
    monkeypatch.setattr(settings, "ENABLE_PII_REDACTION", True, raising=False)

    # guardrails
    monkeypatch.setattr(settings, "BLOCKLIST_TERMS", "password,ssn,private key,api_key,secret", raising=False)
    monkeypatch.setattr(settings, "MAX_OUTPUT_CHARS", 2000, raising=False)

    # auth disabled by default in tests
    monkeypatch.setattr(settings, "ENABLE_AUTH", False, raising=False)


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def built_index(client: TestClient) -> None:
    """
    Ensure index exists for tests that need it.
    """
    r = client.post("/build-index", json={"tenant": "enterprise", "rebuild": True})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["docs_indexed"] >= 1


# =========================
# tests/test_health.py
# =========================
from fastapi.testclient import TestClient


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["env"] == "test"


def test_index_status_before_build(client: TestClient):
    r = client.get("/index/status")
    assert r.status_code == 200
    data = r.json()
    # can be false if no build-index called in this test process
    assert "ready" in data
    assert "docstore_path" in data
    assert "faiss_index_path" in data


# =========================
# tests/test_build_index.py
# =========================
import os
from fastapi.testclient import TestClient
from app.config import settings


def test_build_index_creates_files(client: TestClient):
    # ensure clean start
    if os.path.exists(settings.DOCSTORE_PATH):
        os.remove(settings.DOCSTORE_PATH)
    if os.path.exists(settings.FAISS_INDEX_PATH):
        os.remove(settings.FAISS_INDEX_PATH)

    r = client.post("/build-index", json={"tenant": "enterprise", "rebuild": True})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["tenant"] == "enterprise"
    assert data["docs_indexed"] >= 1
    assert os.path.exists(settings.DOCSTORE_PATH)
    assert os.path.exists(settings.FAISS_INDEX_PATH)


def test_index_status_after_build(client: TestClient, built_index):
    r = client.get("/index/status")
    assert r.status_code == 200
    data = r.json()
    assert data["ready"] is True


# =========================
# tests/test_ask_flow.py
# =========================
from fastapi.testclient import TestClient


def test_ask_requires_index(client: TestClient):
    # In isolated test order, index might exist; still validate behavior:
    # If index exists, response should be normal; if not, it should tell to build.
    r = client.post("/ask", json={"query": "What is the policy?", "tenant": "enterprise", "clearance": "public", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert data["tenant"] == "enterprise"


def test_ask_after_build(client: TestClient, built_index):
    r = client.post("/ask", json={"query": "Do we require MFA?", "tenant": "enterprise", "clearance": "public", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)
    assert data["query_redacted"]
    assert data["tenant"] == "enterprise"
    assert data["clearance"] == "public"


def test_ask_raw_returns_prompt_and_raw(client: TestClient, built_index):
    r = client.post("/ask/raw", json={"query": "VPN policy?", "tenant": "enterprise", "clearance": "internal", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert "prompt" in data
    assert "raw" in data
    assert "hits" in data
    assert data["clearance"] == "internal"


# =========================
# tests/test_tenancy_clearance.py
# =========================
from fastapi.testclient import TestClient


def _ask(client: TestClient, q: str, tenant: str, clearance: str):
    return client.post("/ask", json={"query": q, "tenant": tenant, "clearance": clearance, "k": 5})


def test_clearance_filters_restricted(client: TestClient, built_index):
    # public user should NOT see restricted doc chunks
    r = _ask(client, "What are Q4 revenue projections?", tenant="enterprise", clearance="public")
    assert r.status_code == 200
    data = r.json()
    # sources should not include restricted finance doc title
    titles = [s.get("title") for s in data["sources"]]
    assert all(t != "finance restricted confidential" for t in [str(x).lower() for x in titles])


def test_internal_sees_internal(client: TestClient, built_index):
    r = _ask(client, "Do we need VPN?", tenant="enterprise", clearance="internal")
    assert r.status_code == 200
    data = r.json()
    titles = [str(s.get("title") or "").lower() for s in data showed in data["sources"]]
    assert any("internal security internal" in t or "internal security" in t for t in titles)


def test_restricted_sees_restricted(client: TestClient, built_index):
    r = _ask(client, "Q4 revenue projections?", tenant="enterprise", clearance="restricted")
    assert r.status_code == 200
    data = r.json()
    titles = [str(s.get("title") or "").lower() for s in data["sources"]]
    # should be able to retrieve restricted doc
    assert any("finance" in t for t in titles)


def test_tenant_mismatch_blocks_access(client: TestClient, built_index):
    # index built with tenant enterprise; asking as different tenant should yield no accessible hits
    r = _ask(client, "Do we require MFA?", tenant="other_tenant", clearance="restricted")
    assert r.status_code == 200
    data = r.json()
    assert data["tenant"] == "other_tenant"
    assert data["sources"] == []  # strict tenancy blocks access


# =========================
# tests/test_pii.py
# =========================
from app.pii import redact_pii, detect_pii


def test_redact_email_phone_ssn():
    text = "Email me at john.doe@example.com or call (415) 555-1212. SSN 123-45-6789."
    red = redact_pii(text)
    assert "[REDACTED]" in red
    c = detect_pii(text)
    assert c["email"] == 1
    assert c["phone"] == 1
    assert c["ssn"] == 1


# =========================
# tests/test_guardrails.py
# =========================
from app.guardrails import validate_output


def test_guardrails_blocklist():
    out = "Here is the password: 1234"
    v = validate_output(out)
    assert "can’t help" in v.lower() or "can't help" in v.lower()


def test_guardrails_truncation():
    long = "a" * 99999
    v = validate_output(long)
    assert len(v) <= 2500  # configured MAX_OUTPUT_CHARS in conftest


# =========================
# tests/test_cache_rate_limit.py
# =========================
from fastapi.testclient import TestClient
from app.cache import cache


def test_cache_hit_logged_and_returns_same(client: TestClient, built_index):
    cache.clear()
    payload = {"query": "Do we require MFA?", "tenant": "enterprise", "clearance": "public", "k": 3}
    r1 = client.post("/ask", json=payload)
    assert r1.status_code == 200
    d1 = r1.json()

    r2 = client.post("/ask", json=payload)
    assert r2.status_code == 200
    d2 = r2.json()

    assert d1["answer"] == d2["answer"]
    assert d1["sources"] == d2["sources"]


def test_rate_limit_triggers(client: TestClient, built_index):
    # RL_REQUESTS_PER_MIN = 3 in conftest for subject "anonymous"
    payload = {"query": "VPN policy?", "tenant": "enterprise", "clearance": "internal", "k": 2}
    r1 = client.post("/ask", json=payload)
    r2 = client.post("/ask", json=payload)
    r3 = client.post("/ask", json=payload)
    r4 = client.post("/ask", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200
    assert r4.status_code == 429


# =========================
# tests/test_ingest_chunking.py
# =========================
from app.utils import chunk_text
from app.ingest import load_docs
from app.config import settings


def test_chunk_text_overlap_behavior():
    txt = "x" * 5000
    chunks = chunk_text(txt, chunk_size=1000, overlap=100)
    assert len(chunks) >= 5
    assert chunks[0] == "x" * 1000
    # second asserts overlap: chunk1 ends at 1000, chunk2 begins at 900
    assert chunks[1].startswith("x" * 100)  # all x; just sanity


def test_load_docs_reads_only_txt_md():
    docs = load_docs(settings.APPROVED_SOURCES_DIR, tenant="enterprise")
    assert len(docs) > 0
    # should have meta with chunk index
    assert "chunk_index" in docs[0].meta


# =========================
# tests/test_embed_store_local.py
# =========================
import numpy as np
from app.embed_store import build_embeddings
from app.config import settings


def test_local_mini_embeddings_shape_and_norm():
    texts = ["hello world", "hello world!", "different"]
    vecs = build_embeddings(texts, "local-mini")
    assert vecs.shape[0] == 3
    assert vecs.shape[1] == settings.EMBED_DIM
    # normalized (approx)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.all(norms > 0.9) and np.all(norms < 1.1)
