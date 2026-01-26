from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    # Approved sources root (filesystem demo)
    APPROVED_SOURCES_DIR: str = os.getenv("APPROVED_SOURCES_DIR", "data/approved_sources")

    # Index storage
    INDEX_DIR: str = os.getenv("INDEX_DIR", "index")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "index/faiss.index")
    DOCSTORE_PATH: str = os.getenv("DOCSTORE_PATH", "index/docs.jsonl")

    # Embeddings
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Governance / access
    DEFAULT_TENANT: str = os.getenv("DEFAULT_TENANT", "enterprise")
    ENABLE_PII_REDACTION: bool = os.getenv("ENABLE_PII_REDACTION", "true").lower() == "true"

    # LLM (optional)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "stub")  # stub | openai
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

settings = Settings()
