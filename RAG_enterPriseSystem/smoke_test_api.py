"""
Offline API smoke test for IP-SAKTI Sahayak.

This verifies the FastAPI contract and degraded/offline fallbacks without
requiring Groq, Portkey, Qdrant Cloud, Gemini, Bhashini, or Logfire credentials.
"""

import os

os.environ["LOGFIRE_TOKEN"] = ""
os.environ["IP_SAKTI_DISABLE_LOGFIRE"] = "1"
os.environ["PORTKEY_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["QDRANT_CLUSTER_ENDPOINT"] = ""
os.environ["QDRANT_API_KEY"] = ""
os.environ["BHASHINI_API_URL"] = ""
os.environ["BHASHINI_API_KEY"] = ""

from fastapi.testclient import TestClient

from app.main import app


def main() -> int:
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "q": "Compare India and international rules for an Ashwagandha patent and ABS compliance.",
            "thread_id": "smoke",
            "language": "en",
            "jurisdiction": "BOTH",
        },
    )
    print("POST /query:", response.status_code)
    if response.status_code != 200:
        print(response.text)
        return 1

    payload = response.json()
    required_keys = [
        "answer",
        "jurisdiction",
        "citations",
        "confidence_score",
        "abs_helper",
        "tkdl_prior_art_pointer",
        "escalation",
        "disclaimer",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        print("Missing response keys:", missing)
        return 1

    if payload["jurisdiction"] != "BOTH":
        print("Expected jurisdiction BOTH, got:", payload["jurisdiction"])
        return 1

    if not isinstance(payload["abs_helper"], dict):
        print("abs_helper should be an object")
        return 1

    if not isinstance(payload["tkdl_prior_art_pointer"], dict):
        print("tkdl_prior_art_pointer should be an object")
        return 1

    if not isinstance(payload["escalation"], dict):
        print("escalation should be an object")
        return 1

    classify = client.post(
        "/classify",
        json={
            "description": "A new Ashwagandha and Guduchi tablet for immunity, not from a classical text.",
            "ingredients": "Ashwagandha, Guduchi",
            "intended_use": "Therapeutic",
            "language": "en",
        },
    )
    print("POST /classify:", classify.status_code)
    if classify.status_code != 200:
        print(classify.text)
        return 1

    class_payload = classify.json()
    for key in ["formulation_type", "abs_required", "confidence_score", "explanation"]:
        if key not in class_payload:
            print("Missing classification key:", key)
            return 1

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
