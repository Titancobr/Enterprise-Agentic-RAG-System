import importlib.util
import sys
from pathlib import Path

from dotenv import dotenv_values


packages = {
    "qdrant_client": "qdrant-client",
    "logfire": "logfire",
    "nemoguardrails": "nemoguardrails",
    "langgraph": "langgraph",
    "fastapi": "fastapi",
    "streamlit": "streamlit",
    "rank_bm25": "rank-bm25",
    "flashrank": "flashrank",
}

required_cloud = {
    "GROQ_API_KEY": "Live LLM answers and guardrail classification",
    "QDRANT_CLUSTER_ENDPOINT": "Cloud vector store",
    "QDRANT_API_KEY": "Cloud vector store authentication",
    "GEMINI_API_KEY": "Gemini embeddings for ingestion/search",
}

optional_cloud = {
    "PORTKEY_API_KEY": "LLM gateway, routing, cache, observability",
    "LOGFIRE_TOKEN": "Trace dashboard",
    "BHASHINI_API_URL": "Bhashini translation endpoint",
    "BHASHINI_ENDPOINT": "Alternative Bhashini endpoint variable",
    "BHASHINI_API_KEY": "Bhashini translation authentication",
    "LANGSMITH_API_KEY": "LangSmith tracing",
    "API_KEY": "Optional API protection for deployed endpoints",
}


def configured(value: str | None) -> bool:
    if not value:
        return False
    value = value.strip()
    return bool(value and not value.startswith("your_"))


def masked_status(values: dict, key: str) -> str:
    value = values.get(key)
    if configured(value):
        return f"SET ({len(value.strip())} chars)"
    if value:
        return "MISSING/PLACEHOLDER"
    return "MISSING"


print("Python executable:", sys.executable)
print("Python version:", sys.version)

print("\nPackage check:")
missing = []
for import_name, pip_name in packages.items():
    found = importlib.util.find_spec(import_name) is not None
    print(f"- {pip_name}: {'OK' if found else 'MISSING'}")
    if not found:
        missing.append(pip_name)

env_path = Path(".env")
values = dotenv_values(env_path) if env_path.exists() else {}

print("\n.env check:")
print(f"- .env file: {'FOUND' if env_path.exists() else 'MISSING'}")

print("\nRequired cloud credentials:")
missing_required = []
for key, purpose in required_cloud.items():
    status = masked_status(values, key)
    print(f"- {key}: {status} - {purpose}")
    if not configured(values.get(key)):
        missing_required.append(key)

print("\nOptional cloud credentials:")
for key, purpose in optional_cloud.items():
    print(f"- {key}: {masked_status(values, key)} - {purpose}")

bhashini_endpoint_ok = configured(values.get("BHASHINI_API_URL")) or configured(values.get("BHASHINI_ENDPOINT"))
bhashini_key_ok = configured(values.get("BHASHINI_API_KEY"))
print("\nBhashini readiness:")
if bhashini_endpoint_ok and bhashini_key_ok:
    print("- READY: endpoint and key are configured.")
else:
    print("- NOT READY: set BHASHINI_API_URL or BHASHINI_ENDPOINT plus BHASHINI_API_KEY.")

if missing:
    print("\nInstall missing packages into THIS Python using:")
    print(f"{sys.executable} -m pip install " + " ".join(missing))
elif missing_required:
    print("\nCore package environment is ready, but live cloud mode is missing:")
    print("- " + "\n- ".join(missing_required))
else:
    print("\nEnvironment looks ready for live core RAG mode.")
