# Ingestion + metadata

import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    source_path: str
    domain: str
    owner: str
    effective_date: str
    tenant: str
    sensitivity: str  # public | internal | restricted

def iter_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith((".txt", ".md")):
                yield os.path.join(dirpath, fn)

def infer_domain(path: str) -> str:
    p = path.replace("\\", "/").lower()
    if "/policies/" in p:
        return "policy"
    if "/runbooks/" in p:
        return "runbook"
    return "general"

def load_docs(root_dir: str, tenant: str = "enterprise"):
    docs = []
    for path in iter_files(root_dir):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        doc_id = os.path.relpath(path, root_dir).replace("\\", "/")
        title = os.path.basename(path)
        docs.append(Doc(
            doc_id=doc_id,
            title=title,
            text=text,
            source_path=path,
            domain=infer_domain(path),
            owner="knowledge-owner",  # set per org
            effective_date=datetime.utcnow().date().isoformat(),
            tenant=tenant,
            sensitivity="internal",
        ))
    return docs

def save_docstore_jsonl(docs, jsonl_path: str):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
