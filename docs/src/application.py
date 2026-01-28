# ============================================================
# Enterprise RAG Reference Implementation (FastAPI)
# End-to-end sample project (approx. 1k–2k LOC)
#
# Layout (copy into a package, e.g. `app/`):
#   app/
#     __init__.py
#     main.py
#     config.py
#     schemas.py
#     pii.py
#     ingest.py
#     embed_store.py
#     retrieve.py
#     prompt.py
#     llm_client.py
#     guardrails.py
#     telemetry.py
#     auth.py
#     tenancy.py
#     cache.py
#     rate_limit.py
#     utils.py
#
# Notes:
# - Designed to be readable and extensible.
# - Optional dependencies: faiss-cpu, numpy, httpx, pydantic-settings
# - This is a reference implementation (not production-hardened).
# ============================================================


# =========================
# app/__init__.py
# =========================
__all__ = [
    "main",
    "config",
    "schemas",
    "pii",
    "ingest",
    "embed_store",
    "retrieve",
    "prompt",
    "llm_client",
    "guardrails",
    "telemetry",
    "auth",
    "tenancy",
    "cache",
    "rate_limit",
    "utils",
]


# =========================
# app/config.py
# =========================
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Dict

try:
    # pydantic-settings is preferred for Pydantic v2
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # pragma: no cover
    BaseSettings = object  # type: ignore
    SettingsConfigDict = dict  # type: ignore


class Settings(BaseSettings):
    """
    Central configuration via environment variables.

    Typical env vars:
      APP_ENV=dev
      APP_HOST=0.0.0.0
      APP_PORT=8000
      DEFAULT_TENANT=enterprise
      APPROVED_SOURCES_DIR=./approved_sources
      DOCSTORE_PATH=./data/docstore.jsonl
      FAISS_INDEX_PATH=./data/index.faiss
      EMBED_MODEL=local-mini
      ENABLE_PII_REDACTION=true

      # OpenAI-compatible endpoint (optional)
      LLM_BASE_URL=https://api.openai.com/v1
      LLM_API_KEY=...
      LLM_MODEL=gpt-4o-mini
      EMBED_BASE_URL=https://api.openai.com/v1
      EMBED_API_KEY=...
      EMBED_MODEL_REMOTE=text-embedding-3-small

      # Guardrails
      MAX_OUTPUT_CHARS=8000
      BLOCKLIST_TERMS=...
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = os.getenv("APP_ENV", "dev")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # Tenancy & policy
    DEFAULT_TENANT: str = os.getenv("DEFAULT_TENANT", "enterprise")
    ENABLE_TENANCY: bool = os.getenv("ENABLE_TENANCY", "true").lower() == "true"

    # Source ingestion
    APPROVED_SOURCES_DIR: str = os.getenv("APPROVED_SOURCES_DIR", "./approved_sources")
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    DOCSTORE_PATH: str = os.getenv("DOCSTORE_PATH", "./data/docstore.jsonl")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/index.faiss")

    # Embeddings
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "local-mini")  # local-mini|openai|custom
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))

    # OpenAI-compatible remote embeddings (optional)
    EMBED_BASE_URL: str = os.getenv("EMBED_BASE_URL", "https://api.openai.com/v1")
    EMBED_API_KEY: str = os.getenv("EMBED_API_KEY", "")
    EMBED_MODEL_REMOTE: str = os.getenv("EMBED_MODEL_REMOTE", "text-embedding-3-small")
    EMBED_TIMEOUT_S: float = float(os.getenv("EMBED_TIMEOUT_S", "30"))

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai_compat")  # openai_compat|mock
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT_S: float = float(os.getenv("LLM_TIMEOUT_S", "60"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "700"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # PII
    ENABLE_PII_REDACTION: bool = os.getenv("ENABLE_PII_REDACTION", "true").lower() == "true"

    # Guardrails
    MAX_OUTPUT_CHARS: int = int(os.getenv("MAX_OUTPUT_CHARS", "8000"))
    BLOCKLIST_TERMS: str = os.getenv("BLOCKLIST_TERMS", "password,ssn,private key,api_key,secret")
    ALLOW_MARKDOWN: bool = os.getenv("ALLOW_MARKDOWN", "true").lower() == "true"

    # Retrieval
    TOPK: int = int(os.getenv("TOPK", "5"))
    MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.15"))
    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

    # Telemetry
    TELEMETRY_DIR: str = os.getenv("TELEMETRY_DIR", "./telemetry")
    TELEMETRY_JSONL: str = os.getenv("TELEMETRY_JSONL", "./telemetry/events.jsonl")
    ENABLE_TELEMETRY: bool = os.getenv("ENABLE_TELEMETRY", "true").lower() == "true"

    # Auth
    ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    API_KEYS: str = os.getenv("API_KEYS", "")  # comma-separated list
    AUTH_HEADER: str = os.getenv("AUTH_HEADER", "x-api-key")

    # Rate limiting (very simple)
    ENABLE_RATE_LIMIT: bool = os.getenv("ENABLE_RATE_LIMIT", "true").lower() == "true"
    RL_REQUESTS_PER_MIN: int = int(os.getenv("RL_REQUESTS_PER_MIN", "60"))

    # Caching
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL_S: int = int(os.getenv("CACHE_TTL_S", "300"))

    # Safety: do not let build-index read arbitrary files outside approved directory
    ENFORCE_APPROVED_SOURCES: bool = os.getenv("ENFORCE_APPROVED_SOURCES", "true").lower() == "true"

    # Misc
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()


@dataclass(frozen=True)
class Clearance:
    """
    Clearance levels used for access filtering.
    """
    PUBLIC: str = "public"
    INTERNAL: str = "internal"
    RESTRICTED: str = "restricted"

    ORDER: Dict[str, int] = None  # type: ignore

    @staticmethod
    def order_map() -> Dict[str, int]:
        return {
            "public": 0,
            "internal": 1,
            "restricted": 2,
        }

    @staticmethod
    def allows(user_clearance: str, doc_clearance: str) -> bool:
        m = Clearance.order_map()
        return m.get(user_clearance, 0) >= m.get(doc_clearance, 0)


# =========================
# app/schemas.py
# =========================
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class QueryReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    tenant: str = "enterprise"
    clearance: str = "internal"  # public|internal|restricted
    redact_pii: bool = True
    k: int = 5


class BuildIndexReq(BaseModel):
    tenant: str = "enterprise"
    # optional: rebuild always or incremental
    rebuild: bool = True


class SourceHit(BaseModel):
    title: Optional[str] = None
    domain: Optional[str] = None
    score: float
    doc_id: str
    chunk_id: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class AskResp(BaseModel):
    answer: str
    sources: List[SourceHit]
    query_redacted: str
    tenant: str
    clearance: str


class HealthResp(BaseModel):
    status: str
    env: str


class ErrorResp(BaseModel):
    error: str
    details: Optional[Dict[str, Any]] = None


# =========================
# app/utils.py
# =========================
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ms() -> int:
    return int(time.time() * 1000)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_filename(name: str, max_len: int = 120) -> str:
    cleaned = SAFE_FILENAME_RE.sub("_", name).strip("._-")
    if not cleaned:
        cleaned = "file"
    return cleaned[:max_len]


def clamp(n: int, lo: int, hi: int) -> int:
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]..."


def jsonl_write(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jsonl_read(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple char-based chunker. In production, use token-aware chunking.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunk = text[i:j]
        chunks.append(chunk)
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def is_probably_binary(data: bytes) -> bool:
    """
    Heuristic: detect NUL bytes in the first portion.
    """
    sample = data[:2048]
    return b"\x00" in sample


# =========================
# app/pii.py
# =========================
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Pattern, Tuple

# Minimal PII regex patterns (extend per your needs)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def redact_pii(text: str, replacement: str = "[REDACTED]") -> str:
    """
    Redact common PII patterns. Keep this conservative to reduce false positives.
    """
    if not text:
        return text
    # Order matters a bit; apply strict patterns first.
    text = SSN_RE.sub(replacement, text)
    text = EMAIL_RE.sub(replacement, text)
    text = PHONE_RE.sub(replacement, text)
    # Credit cards can false-positive; keep it last.
    text = CC_RE.sub(replacement, text)
    text = IP_RE.sub(replacement, text)
    return text


def detect_pii(text: str) -> Dict[str, int]:
    """
    Basic detector returning counts by category.
    """
    return {
        "email": len(EMAIL_RE.findall(text)),
        "phone": len(PHONE_RE.findall(text)),
        "ssn": len(SSN_RE.findall(text)),
        "cc": len(CC_RE.findall(text)),
        "ip": len(IP_RE.findall(text)),
    }


# =========================
# app/tenancy.py
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import settings, Clearance


@dataclass
class UserContext:
    tenant: str
    clearance: str
    subject: str = "anonymous"
    roles: tuple[str, ...] = ()


def build_user_context(tenant: str, clearance: str, subject: str = "anonymous", roles: Optional[list[str]] = None) -> UserContext:
    roles_t = tuple(roles or [])
    return UserContext(tenant=tenant, clearance=clearance, subject=subject, roles=roles_t)


def validate_tenant(tenant: str) -> str:
    # You can enforce allowed tenants via config/DB.
    if not tenant or len(tenant) > 80:
        return settings.DEFAULT_TENANT
    return tenant


def validate_clearance(clearance: str) -> str:
    c = clearance.lower().strip()
    if c not in ("public", "internal", "restricted"):
        return "public"
    return c


def can_access_doc(user_ctx: UserContext, doc_meta: Dict[str, Any]) -> bool:
    """
    Enforce:
      - tenant match (if enabled)
      - clearance dominance (public < internal < restricted)
    """
    if settings.ENABLE_TENANCY:
        doc_tenant = (doc_meta.get("tenant") or settings.DEFAULT_TENANT)
        if doc_tenant != user_ctx.tenant:
            return False

    doc_clearance = (doc_meta.get("clearance") or "public").lower()
    return Clearance.allows(user_ctx.clearance, doc_clearance)


# =========================
# app/auth.py
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from fastapi import Header, HTTPException, status

from .config import settings


@dataclass(frozen=True)
class AuthResult:
    subject: str
    roles: Tuple[str, ...]


def _allowed_keys() -> List[str]:
    raw = settings.API_KEYS.strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias=None)) -> AuthResult:
    """
    Very simple API-key auth. For real systems, use OAuth/JWT/mTLS.
    """
    if not settings.ENABLE_AUTH:
        return AuthResult(subject="anonymous", roles=())

    header_name = settings.AUTH_HEADER.lower()
    # FastAPI doesn't dynamically read header alias easily here; accept x-api-key by default.
    key = x_api_key
    if not key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")

    allowed = _allowed_keys()
    if allowed and key not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")

    # Map key->subject if you want; here, just hash prefix.
    subject = f"key_{key[:6]}"
    return AuthResult(subject=subject, roles=("api",))


# =========================
# app/cache.py
# =========================
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .config import settings


@dataclass
class CacheItem:
    value: Any
    expires_at: float


class SimpleTTLCache:
    """
    Small in-memory TTL cache for request-level reuse.
    Not multi-process safe. Use Redis for production.
    """

    def __init__(self, ttl_s: int = 300, max_items: int = 1024):
        self.ttl_s = ttl_s
        self.max_items = max_items
        self._store: Dict[str, CacheItem] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        if item.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return item.value

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None) -> None:
        if not settings.ENABLE_CACHE:
            return
        if len(self._store) >= self.max_items:
            # naive eviction: drop oldest-ish by soonest expiration
            oldest_key = min(self._store.items(), key=lambda kv: kv[1].expires_at)[0]
            self._store.pop(oldest_key, None)
        ttl = ttl_s if ttl_s is not None else self.ttl_s
        self._store[key] = CacheItem(value=value, expires_at=time.time() + ttl)

    def clear(self) -> None:
        self._store.clear()


cache = SimpleTTLCache(ttl_s=settings.CACHE_TTL_S, max_items=2048)


# =========================
# app/rate_limit.py
# =========================
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, status

from .config import settings


@dataclass
class Bucket:
    window_start: float
    count: int


class SimpleRateLimiter:
    """
    Fixed-window rate limiter by subject (e.g., API key subject or IP).
    This is a reference only.
    """

    def __init__(self, requests_per_min: int):
        self.rpm = max(1, requests_per_min)
        self._buckets: Dict[str, Bucket] = {}

    def check(self, subject: str) -> None:
        if not settings.ENABLE_RATE_LIMIT:
            return
        now = time.time()
        window = 60.0
        b = self._buckets.get(subject)
        if not b or (now - b.window_start) >= window:
            self._buckets[subject] = Bucket(window_start=now, count=1)
            return
        b.count += 1
        if b.count > self.rpm:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({self.rpm}/min). Try later.",
            )


rate_limiter = SimpleRateLimiter(settings.RL_REQUESTS_PER_MIN)


# =========================
# app/telemetry.py
# =========================
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

from .config import settings
from .utils import ensure_dir, now_ms


_lock = threading.Lock()


def log_event(evt: Dict[str, Any]) -> None:
    """
    Append-only JSONL logging.
    """
    if not settings.ENABLE_TELEMETRY:
        return
    evt = dict(evt)
    evt.setdefault("ts_ms", now_ms())
    ensure_dir(os.path.dirname(settings.TELEMETRY_JSONL) or ".")
    line = json.dumps(evt, ensure_ascii=False)
    with _lock:
        with open(settings.TELEMETRY_JSONL, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# =========================
# app/ingest.py
# =========================
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import settings
from .utils import chunk_text, jsonl_write, safe_filename, sha256_text, normalize_ws


@dataclass
class DocChunk:
    """
    Stored unit of retrieval. Each chunk is separately embedded and indexed.
    """
    doc_id: str
    chunk_id: str
    title: str
    domain: str
    tenant: str
    clearance: str
    source_path: str
    text: str
    meta: Dict[str, Any]


def _read_text_file(path: str, max_bytes: int = 5_000_000) -> str:
    # Only handle text-like files in this reference. Extend for PDF/HTML as needed.
    with open(path, "rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("utf-8", errors="ignore")


def _infer_domain(path: str) -> str:
    base = os.path.basename(path).lower()
    if "hr" in base or "policy" in base:
        return "policy"
    if "security" in base or "iam" in base:
        return "security"
    if "finance" in base or "billing" in base:
        return "finance"
    return "general"


def _infer_clearance(path: str) -> str:
    base = os.path.basename(path).lower()
    if "restricted" in base or "confidential" in base:
        return "restricted"
    if "internal" in base:
        return "internal"
    return "public"


def _infer_title(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name.replace("_", " ").replace("-", " ").strip()


def _is_approved_path(path: str) -> bool:
    if not settings.ENFORCE_APPROVED_SOURCES:
        return True
    approved_root = os.path.abspath(settings.APPROVED_SOURCES_DIR)
    abs_path = os.path.abspath(path)
    return abs_path.startswith(approved_root + os.sep) or abs_path == approved_root


def load_docs(root_dir: str, tenant: str) -> List[DocChunk]:
    """
    Load and chunk documents from an approved directory.

    Supported extensions: .txt, .md
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir):
        return []

    docs: List[DocChunk] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in (".txt", ".md"):
                continue

            path = os.path.join(dirpath, fn)
            if not _is_approved_path(path):
                continue

            raw = _read_text_file(path)
            raw = normalize_ws(raw)

            title = _infer_title(path)
            domain = _infer_domain(path)
            clearance = _infer_clearance(path)
            doc_id = sha256_text(f"{tenant}::{path}")[:24]

            chunks = chunk_text(raw, chunk_size=1400, overlap=220)
            for i, ch in enumerate(chunks):
                chunk_id = f"{doc_id}:{i:04d}"
                docs.append(
                    DocChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        title=title,
                        domain=domain,
                        tenant=tenant,
                        clearance=clearance,
                        source_path=path,
                        text=ch,
                        meta={
                            "filename": fn,
                            "relpath": os.path.relpath(path, root_dir),
                            "ext": ext,
                            "chunk_index": i,
                            "chunks_total": len(chunks),
                        },
                    )
                )

    return docs


def save_docstore_jsonl(docs: List[DocChunk], out_path: str) -> None:
    rows = []
    for d in docs:
        rows.append(asdict(d))
    jsonl_write(out_path, rows)


# =========================
# app/embed_store.py
# =========================
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import settings
from .utils import jsonl_read, ensure_dir

# Optional faiss
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

# Optional httpx for remote embeddings
try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _local_hash_embed(texts: List[str], dim: int) -> np.ndarray:
    """
    Deterministic cheap embedding for demo/dev without external models.

    WARNING: Not semantically meaningful, just stable and fast.
    Replace with sentence-transformers or real embed service.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        b = t.encode("utf-8", errors="ignore")
        # simple rolling hash into buckets
        h = 2166136261
        for by in b[:20000]:
            h ^= by
            h *= 16777619
            h &= 0xFFFFFFFF
            out[i, h % dim] += 1.0
        # add length signal
        out[i, (len(b) % dim)] += 0.5
    return _l2_normalize(out)


def _remote_openai_compat_embeddings(texts: List[str]) -> np.ndarray:
    """
    Calls OpenAI-compatible embeddings endpoint:
      POST {base_url}/embeddings
      { "model": "...", "input": ["...", "..."] }
    """
    if httpx is None:
        raise RuntimeError("httpx not installed; cannot call remote embeddings")
    if not settings.EMBED_API_KEY:
        raise RuntimeError("EMBED_API_KEY not set")

    url = settings.EMBED_BASE_URL.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {settings.EMBED_API_KEY}"}
    payload = {"model": settings.EMBED_MODEL_REMOTE, "input": texts}

    with httpx.Client(timeout=settings.EMBED_TIMEOUT_S) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    # OpenAI format: data.data[i].embedding
    embs = [row["embedding"] for row in data["data"]]
    mat = np.array(embs, dtype=np.float32)
    return _l2_normalize(mat)


def build_embeddings(texts: List[str], embed_model: str) -> np.ndarray:
    """
    Returns np.ndarray shape (n, dim) float32, L2-normalized.
    """
    if not texts:
        return np.zeros((0, settings.EMBED_DIM), dtype=np.float32)

    if embed_model == "local-mini":
        return _local_hash_embed(texts, settings.EMBED_DIM)
    if embed_model == "openai":
        return _remote_openai_compat_embeddings(texts)

    # fallback
    return _local_hash_embed(texts, settings.EMBED_DIM)


def build_faiss_index(vectors: np.ndarray):
    """
    Build an inner-product index. We L2-normalize vectors so IP ~= cosine similarity.
    """
    if faiss is None:
        raise RuntimeError("faiss not installed. Install faiss-cpu to use FAISS index.")
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_faiss(index, path: str) -> None:
    if faiss is None:
        raise RuntimeError("faiss not installed")
    ensure_dir(os.path.dirname(path) or ".")
    faiss.write_index(index, path)


def load_faiss(path: str):
    if faiss is None:
        raise RuntimeError("faiss not installed")
    return faiss.read_index(path)


def load_docstore(path: str) -> List[Dict[str, Any]]:
    return jsonl_read(path)


# =========================
# app/retrieve.py
# =========================
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import settings
from .embed_store import build_embeddings
from .tenancy import UserContext, can_access_doc
from .utils import clamp


def _faiss_search(index, q_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, idxs) arrays.
    """
    # faiss expects shape (nq, dim)
    scores, idxs = index.search(q_vec.astype(np.float32), k)
    return scores[0], idxs[0]


def retrieve_topk(
    index,
    docs: List[Dict[str, Any]],
    query: str,
    embed_model: str,
    user_ctx: UserContext,
    k: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Retrieve top-k chunks subject to tenancy/clearance filters.

    Returns list of (score, doc_dict).
    """
    k = clamp(int(k), 1, 20)
    if not query.strip():
        return []

    q_vec = build_embeddings([query], embed_model)
    scores, idxs = _faiss_search(index, q_vec, k=min(max(k * 6, 20), 200))

    hits: List[Tuple[float, Dict[str, Any]]] = []
    for score, idx in zip(scores.tolist(), idxs.tolist()):
        if idx < 0 or idx >= len(docs):
            continue
        d = docs[idx]
        if not can_access_doc(user_ctx, d):
            continue
        if score < settings.MIN_SCORE:
            continue
        hits.append((float(score), d))
        if len(hits) >= k:
            break

    return hits


# =========================
# app/prompt.py
# =========================
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config import settings
from .utils import truncate


SYSTEM_PROMPT = """You are an enterprise assistant using Retrieval-Augmented Generation (RAG).
Follow these rules:
- Use the provided context only; if insufficient, say you don't have enough information.
- Do not invent citations. When referring to documents, cite the source titles from the provided context.
- Keep answers concise and actionable.
- If asked for secrets, credentials, or personal data, refuse and explain.
"""


def _format_context(hits: List[Tuple[float, Dict[str, Any]]]) -> str:
    parts: List[str] = []
    for score, d in hits:
        title = d.get("title") or "Untitled"
        domain = d.get("domain") or "general"
        relpath = d.get("meta", {}).get("relpath") if isinstance(d.get("meta"), dict) else None
        header = f"[SOURCE] title={title} domain={domain} score={score:.3f}"
        if relpath:
            header += f" path={relpath}"
        body = (d.get("text") or "").strip()
        parts.append(header + "\n" + body)

    ctx = "\n\n---\n\n".join(parts)
    return truncate(ctx, settings.MAX_CONTEXT_CHARS)


def build_prompt(query: str, hits: List[Tuple[float, Dict[str, Any]]]) -> str:
    ctx = _format_context(hits) if hits else ""
    user_prompt = f"""Question:
{query}

Context:
{ctx}

Answer requirements:
- If context is empty or irrelevant, say: "I don't have enough approved information to answer that."
- Provide bullet points when appropriate.
- At the end, list Sources: <titles> used.
"""
    # For OpenAI-compatible chat-completions, we’ll send messages, but this string is fine for a simple client.
    return SYSTEM_PROMPT + "\n\n" + user_prompt


# =========================
# app/guardrails.py
# =========================
from __future__ import annotations

import re
from typing import List, Tuple

from .config import settings
from .utils import truncate


def _split_blocklist(raw: str) -> List[str]:
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


def validate_output(text: str) -> str:
    """
    Simple output validator:
      - max length
      - blocklist checks
      - optional markdown stripping
    """
    if not text:
        return "I don't have enough approved information to answer that."

    t = text.strip()

    # blocklist
    lower = t.lower()
    for term in _split_blocklist(settings.BLOCKLIST_TERMS):
        if term and term in lower:
            return "I can’t help with that request."

    # length
    t = truncate(t, settings.MAX_OUTPUT_CHARS)

    # markdown control (optional)
    if not settings.ALLOW_MARKDOWN:
        t = re.sub(r"[`*_>#-]{1,3}", "", t)

    return t


# =========================
# app/llm_client.py
# =========================
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

from .config import settings


def _mock_generate(prompt: str) -> str:
    # Deterministic placeholder (useful for tests/dev).
    # It echoes a short response based on prompt length.
    n = len(prompt)
    return (
        "I don't have enough approved information to answer that."
        if n < 200
        else "Based on the provided context, here are the key points:\n- (mock) The context suggests ...\n\nSources: (mock)"
    )


def _openai_compat_chat(prompt: str) -> str:
    """
    Calls OpenAI-compatible chat completions endpoint:
      POST {base_url}/chat/completions
      { "model": "...", "messages": [...], "temperature": ..., "max_tokens": ... }
    """
    if httpx is None:
        raise RuntimeError("httpx not installed; cannot call LLM endpoint")
    if not settings.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY not set")

    url = settings.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}

    messages = [
        {"role": "system", "content": "You are a helpful enterprise assistant."},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
    }

    with httpx.Client(timeout=settings.LLM_TIMEOUT_S) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    # OpenAI format: choices[0].message.content
    return data["choices"][0]["message"]["content"]


def generate(prompt: str) -> str:
    """
    Provider switch.
    """
    prov = settings.LLM_PROVIDER.lower().strip()
    if prov == "mock":
        return _mock_generate(prompt)
    # default
    return _openai_compat_chat(prompt)


# =========================
# app/main.py
# =========================
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import QueryReq, BuildIndexReq, AskResp, HealthResp, ErrorResp, SourceHit
from .pii import redact_pii, detect_pii
from .ingest import load_docs, save_docstore_jsonl
from .embed_store import (
    build_embeddings,
    build_faiss_index,
    save_faiss,
    load_faiss,
    load_docstore,
)
from .retrieve import retrieve_topk
from .prompt import build_prompt
from .llm_client import generate
from .guardrails import validate_output
from .telemetry import log_event
from .auth import require_api_key, AuthResult
from .tenancy import build_user_context, validate_tenant, validate_clearance
from .cache import cache
from .rate_limit import rate_limiter
from .utils import ensure_dir, now_ms, sha256_text, clamp


app = FastAPI(title="Enterprise RAG Reference Implementation", version="1.0.0")

# CORS (configure in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _index_ready() -> bool:
    return os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.DOCSTORE_PATH)


def _build_cache_key(req: QueryReq, subject: str) -> str:
    # include tenant + clearance in cache key to avoid leakage
    return sha256_text(f"{subject}::{req.tenant}::{req.clearance}::{req.redact_pii}::{req.k}::{req.query}")[:32]


@app.get("/health", response_model=HealthResp)
def health() -> HealthResp:
    return HealthResp(status="ok", env=settings.APP_ENV)


@app.get("/index/status")
def index_status() -> Dict[str, Any]:
    return {
        "ready": _index_ready(),
        "docstore_path": settings.DOCSTORE_PATH,
        "faiss_index_path": settings.FAISS_INDEX_PATH,
    }


@app.post("/build-index")
def build_index(req: BuildIndexReq = BuildIndexReq()) -> Dict[str, Any]:
    """
    Build (or rebuild) the docstore + FAISS index from approved sources directory.
    """
    ensure_dir(settings.DATA_DIR)

    tenant = validate_tenant(req.tenant)

    docs = load_docs(settings.APPROVED_SOURCES_DIR, tenant=tenant)
    save_docstore_jsonl(docs, settings.DOCSTORE_PATH)

    texts = [d.text for d in docs]
    vectors = build_embeddings(texts, settings.EMBED_MODEL)
    index = build_faiss_index(vectors)
    save_faiss(index, settings.FAISS_INDEX_PATH)

    log_event(
        {
            "event": "build_index",
            "tenant": tenant,
            "docs_indexed": len(docs),
            "docstore_path": settings.DOCSTORE_PATH,
            "faiss_index_path": settings.FAISS_INDEX_PATH,
        }
    )

    return {"status": "ok", "tenant": tenant, "docs_indexed": len(docs), "faiss": settings.FAISS_INDEX_PATH}


@app.post("/ask", response_model=AskResp, responses={400: {"model": ErrorResp}, 429: {"model": ErrorResp}})
def ask(req: QueryReq, auth: AuthResult = Depends(require_api_key)) -> AskResp:
    """
    Main RAG endpoint:
      - optional auth + rate limit
      - optional PII redaction
      - retrieve top-k allowed chunks
      - build prompt
      - LLM generate
      - validate output
      - telemetry log
    """
    start = now_ms()

    subject = auth.subject or "anonymous"
    rate_limiter.check(subject)

    tenant = validate_tenant(req.tenant)
    clearance = validate_clearance(req.clearance)
    k = clamp(req.k or req.k, 1, 20)

    if not _index_ready():
        return AskResp(
            answer="Index not built. Call /build-index first.",
            sources=[],
            query_redacted=req.query,
            tenant=tenant,
            clearance=clearance,
        )

    # Cache
    cache_key = _build_cache_key(req, subject)
    cached = cache.get(cache_key)
    if cached is not None:
        log_event(
            {
                "event": "ask_cache_hit",
                "tenant": tenant,
                "clearance": clearance,
                "subject": subject,
                "latency_ms": now_ms() - start,
            }
        )
        return cached

    # Load index/docstore (in production, keep these warm / singleton)
    index = load_faiss(settings.FAISS_INDEX_PATH)
    docs = load_docstore(settings.DOCSTORE_PATH)

    user_ctx = build_user_context(tenant=tenant, clearance=clearance, subject=subject, roles=list(auth.roles))

    # PII
    q_in = req.query
    q_redacted = redact_pii(q_in) if (settings.ENABLE_PII_REDACTION and req.redact_pii) else q_in

    hits = retrieve_topk(index, docs, q_redacted, settings.EMBED_MODEL, user_ctx, k=k)

    prompt = build_prompt(q_redacted, hits)
    raw = generate(prompt)
    final = validate_output(raw)

    # Build response sources
    sources: List[SourceHit] = []
    for score, d in hits:
        sources.append(
            SourceHit(
                title=d.get("title"),
                domain=d.get("domain"),
                score=float(score),
                doc_id=d.get("doc_id", ""),
                chunk_id=d.get("chunk_id", ""),
                meta=d.get("meta") if isinstance(d.get("meta"), dict) else {},
            )
        )

    resp = AskResp(
        answer=final,
        sources=sources,
        query_redacted=q_redacted,
        tenant=tenant,
        clearance=clearance,
    )

    # Telemetry
    log_event(
        {
            "event": "ask",
            "tenant": tenant,
            "clearance": clearance,
            "subject": subject,
            "query": q_in,
            "query_redacted": q_redacted,
            "pii_counts": detect_pii(q_in) if settings.ENABLE_PII_REDACTION else {},
            "hits": [{"title": h.title, "domain": h.domain, "score": h.score, "chunk_id": h.chunk_id} for h in sources],
            "latency_ms": now_ms() - start,
        }
    )

    cache.set(cache_key, resp)
    return resp


@app.post("/ask/raw")
def ask_raw(req: QueryReq, auth: AuthResult = Depends(require_api_key)) -> Dict[str, Any]:
    """
    Debug endpoint: returns raw prompt and raw model output.
    """
    subject = auth.subject or "anonymous"
    rate_limiter.check(subject)

    tenant = validate_tenant(req.tenant)
    clearance = validate_clearance(req.clearance)
    k = clamp(req.k or req.k, 1, 20)

    if not _index_ready():
        return {"error": "Index not built. Call /build-index first."}

    index = load_faiss(settings.FAISS_INDEX_PATH)
    docs = load_docstore(settings.DOCSTORE_PATH)

    user_ctx = build_user_context(tenant=tenant, clearance=clearance, subject=subject, roles=list(auth.roles))

    q_in = req.query
    q_redacted = redact_pii(q_in) if (settings.ENABLE_PII_REDACTION and req.redact_pii) else q_in
    hits = retrieve_topk(index, docs, q_redacted, settings.EMBED_MODEL, user_ctx, k=k)

    prompt = build_prompt(q_redacted, hits)
    raw = generate(prompt)

    return {
        "tenant": tenant,
        "clearance": clearance,
        "query": q_in,
        "query_redacted": q_redacted,
        "prompt": prompt,
        "raw": raw,
        "hits": [{"title": d.get("title"), "domain": d.get("domain"), "score": s} for s, d in hits],
    }


@app.post("/cache/clear")
def clear_cache() -> Dict[str, Any]:
    cache.clear()
    return {"status": "ok", "cache": "cleared"}


# Optional: add a startup check
@app.on_event("startup")
def on_startup() -> None:
    ensure_dir(settings.DATA_DIR)
    ensure_dir(os.path.dirname(settings.TELEMETRY_JSONL) or ".")
    log_event({"event": "startup", "env": settings.APP_ENV})


# Optional: add a shutdown hook
@app.on_event("shutdown")
def on_shutdown() -> None:
    log_event({"event": "shutdown"})


# =========================
# (Optional) runner snippet
# =========================
# Run with:
#   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
#
# Then:
#   POST /build-index
#   POST /ask  { "query": "....", "tenant": "enterprise", "clearance": "internal" }
#
# Put .txt/.md into ./approved_sources
# ============================================================
