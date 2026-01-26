# telemetry.py
import json
import os
from datetime import datetime

def log_event(event: dict, path: str = "index/telemetry.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    event["ts"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
