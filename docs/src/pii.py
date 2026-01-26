# PII redaction

import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?){1}\d{3}[-.\s]?\d{4}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

def redact_pii(text: str) -> str:
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    text = PHONE.sub("[REDACTED_PHONE]", text)
    text = SSN.sub("[REDACTED_SSN]", text)
    return text
