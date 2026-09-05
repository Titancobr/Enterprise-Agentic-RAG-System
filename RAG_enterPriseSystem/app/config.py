import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = (os.getenv("QDRANT_CLUSTER_ENDPOINT") or "").strip()
    QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or "").strip()
    QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION") or "enterprise_rag").strip()

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_KEYS_RAW = os.getenv("GROQ_API_KEYS", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b").strip()
    GROQ_GUARDRAIL_MODEL = os.getenv("GROQ_GUARDRAIL_MODEL", "openai/gpt-oss-20b").strip()

    @property
    def groq_key_list(self) -> list[str]:
        keys = []
        if self.GROQ_API_KEYS_RAW:
            keys.extend([k.strip() for k in self.GROQ_API_KEYS_RAW.split(",") if k.strip()])
        if self.GROQ_API_KEY:
            for k in self.GROQ_API_KEY.split(","):
                k_clean = k.strip()
                if k_clean and k_clean not in keys:
                    keys.append(k_clean)
        fb_key = os.getenv("GROQ_FALLBACK_API_KEY")
        if fb_key and fb_key.strip() and fb_key.strip() not in keys:
            keys.append(fb_key.strip())
        for i in range(1, 10):
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if k and k.strip() and k.strip() not in keys:
                keys.append(k.strip())
        # Filter out placeholder keys
        valid_keys = [k for k in keys if not k.startswith("your_")]
        return valid_keys

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = os.getenv("GROQ_SLUG", "rag")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2", "brag")

    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ip-sakti-sahayak")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    # --- LOGFIRE ---
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    LOGFIRE_IGNORE_NO_CONFIG = os.getenv("LOGFIRE_IGNORE_NO_CONFIG", "1")

    # --- MULTILINGUAL / BHASHINI ---
    BHASHINI_API_URL = os.getenv("BHASHINI_API_URL") or os.getenv("BHASHINI_ENDPOINT")
    BHASHINI_API_KEY = os.getenv("BHASHINI_API_KEY")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ip-sakti-sahayak")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = os.getenv("LOGFIRE_IGNORE_NO_CONFIG", "1")

settings = Settings()
