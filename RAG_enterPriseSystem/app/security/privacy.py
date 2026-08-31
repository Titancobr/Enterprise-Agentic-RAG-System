"""
DPDP-aligned privacy helpers for IP-SAKTI Sahayak.
Prototype scope:
- Redacts common personal identifiers before audit logging.
- Produces minimal audit records for accountability.
"""

import re
from datetime import datetime
from typing import Any, Dict

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")


def redact_personal_data(text: str) -> str:
    if not text:
        return text
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = AADHAAR_RE.sub("[REDACTED_AADHAAR]", text)
    text = PAN_RE.sub("[REDACTED_PAN]", text)
    return text


def build_audit_record(
    *,
    thread_id: str,
    action: str,
    query: str,
    intent: str | None = None,
    jurisdiction: str | None = None,
    status: str = "success",
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "thread_id": thread_id,
        "action": action,
        "query_redacted": redact_personal_data(query)[:500],
        "intent": intent,
        "jurisdiction": jurisdiction,
        "status": status,
    }
