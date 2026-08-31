from app.observability.logfire_compat import logfire

from app.config import settings

try:
    from langchain_groq import ChatGroq
except ModuleNotFoundError:
    ChatGroq = None

try:
    from groq import Groq
except ModuleNotFoundError:
    Groq = None

try:
    from portkey_ai import Portkey
except ModuleNotFoundError:
    Portkey = None


class _OfflineMessage:
    def __init__(self, content: str):
        self.content = content


class _OfflineChoice:
    def __init__(self, content: str):
        self.message = _OfflineMessage(content)


class _OfflineResponse:
    def __init__(self, content: str):
        self.choices = [_OfflineChoice(content)]


class _OfflineCompletions:
    def create(self, *args, **kwargs):
        return _OfflineResponse(
            "I cannot call the configured LLM gateway in this offline environment. "
            "I will rely on the system's grounded template fallback where available."
        )


class _OfflineChat:
    completions = _OfflineCompletions()


class OfflineChatClient:
    chat = _OfflineChat()


class OfflineLangChainLLM:
    def invoke(self, prompt: str):
        lowered = prompt.lower()
        if "classify the jurisdiction" in lowered:
            if any(term in lowered for term in ("wipo", "pct", "trips", "nagoya", "international", "foreign")):
                content = "INTERNATIONAL"
            elif "compare" in lowered:
                content = "BOTH"
            else:
                content = "INDIA"
            return _OfflineMessage(content)
        if "strict json" in lowered and "formulation_type" in lowered:
            return _OfflineMessage('{"formulation_type":"INSUFFICIENT_INFO","abs_required":null,"confidence_score":0.45,"missing_questions":["Please provide ingredients, intended use, claims, and classical text reference."]}')
        return _OfflineMessage("SAFE")


def _build_chat_client():
    """
    Prefer Portkey for the live gateway path; fall back to direct Groq for
    local development when Portkey is not configured.
    """
    if settings.PORTKEY_API_KEY and not settings.PORTKEY_API_KEY.startswith("your_") and Portkey:
        logfire.info("Using Portkey gateway for chat completions.")
        return Portkey(
            api_key=settings.PORTKEY_API_KEY,
            virtual_key=settings.GROQ_SLUG,
            provider="groq",
            request_timeout=20,
            max_retries=0,
        )
    if settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("your_") and Groq:
        logfire.warning("PORTKEY_API_KEY not set; using direct Groq chat client.")
        return Groq(api_key=settings.GROQ_API_KEY, timeout=20, max_retries=0)
    logfire.warning("No LLM gateway credentials/client available; using offline fallback client.")
    return OfflineChatClient()


portkey_client = _build_chat_client()


def get_langchain_llm(feature: str = "rag") -> ChatGroq:
    """
    Returns a direct Groq-backed ChatGroq.
    """
    if not ChatGroq or not settings.GROQ_API_KEY or settings.GROQ_API_KEY.startswith("your_"):
        logfire.warning(f"LangChain Groq unavailable for {feature}; using offline deterministic fallback.")
        return OfflineLangChainLLM()
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
        request_timeout=20,
        max_retries=0,
    )

def extract_cache_status(response) -> str:
    """
    Return dummy 'MISS' since direct Groq is used without Portkey caching.
    """
    return "MISS"
