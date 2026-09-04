"""
Multilingual support for IP-SAKTI Sahayak.

Prototype behavior:
- Supports language metadata and UI/API routing.
- Uses Bhashini-compatible API calls when BHASHINI_API_URL or
  BHASHINI_ENDPOINT plus BHASHINI_API_KEY are configured.
- Falls back to identity translation so the MVP keeps working offline.
"""

import os
import re
import requests
from app.observability.logfire_compat import logfire
from typing import Dict, List


SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "sa": "Sanskrit",
    "ne": "Nepali",
    "kok": "Konkani",
    "mai": "Maithili",
    "mni": "Manipuri",
    "sd": "Sindhi",
    "doi": "Dogri",
    "ks": "Kashmiri",
    "brx": "Bodo",
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")
MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")
GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")
GURMUKHI_RE = re.compile(r"[\u0A00-\u0A7F]")
ODIA_RE = re.compile(r"[\u0B00-\u0B7F]")


def get_supported_languages() -> Dict[str, str]:
    return SUPPORTED_LANGUAGES


def detect_language(text: str) -> str:
    """Lightweight script-based language detection for Indic languages."""
    if not text or not text.strip():
        return "en"

    if DEVANAGARI_RE.search(text):
        return "hi"
    if BENGALI_RE.search(text):
        return "bn"
    if TAMIL_RE.search(text):
        return "ta"
    if TELUGU_RE.search(text):
        return "te"
    if KANNADA_RE.search(text):
        return "kn"
    if MALAYALAM_RE.search(text):
        return "ml"
    if GUJARATI_RE.search(text):
        return "gu"
    if GURMUKHI_RE.search(text):
        return "pa"
    if ODIA_RE.search(text):
        return "or"
    return "en"


def _bhashini_translate(text: str, source_lang: str, target_lang: str) -> str | None:
    api_url = os.getenv("BHASHINI_API_URL") or os.getenv("BHASHINI_ENDPOINT")
    api_key = os.getenv("BHASHINI_API_KEY")

    if (
        not api_url
        or not api_key
        or api_url.startswith("your_")
        or api_key.startswith("your_")
    ):
        return None

    try:
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "input": text,
                "sourceLanguage": source_lang,
                "targetLanguage": target_lang,
                "task": "translation",
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        # Accept common response shapes used by translation APIs.
        if isinstance(data, dict):
            if "translatedText" in data:
                return data["translatedText"]
            if "output" in data and isinstance(data["output"], str):
                return data["output"]
            if "data" in data and isinstance(data["data"], dict):
                return data["data"].get("translatedText") or data["data"].get("output")
        return None
    except Exception as exc:
        logfire.warning(f"Bhashini translation unavailable: {exc}")
        return None


def _llm_translate(text: str, source_lang: str, target_lang: str) -> str | None:
    """Fast, accurate LLM translation fallback when external translation APIs are unconfigured."""
    try:
        from app.gateway import portkey_client
        from app.config import settings
        lang_names = {
            "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
            "ml": "Malayalam", "gu": "Gujarati", "mr": "Marathi", "bn": "Bengali",
            "pa": "Punjabi", "or": "Odia", "ur": "Urdu", "en": "English"
        }
        src = lang_names.get(source_lang, source_lang)
        tgt = lang_names.get(target_lang, target_lang)
        prompt = (
            f"You are a strict translation engine. Translate the following text from {src} to {tgt}.\n"
            f"CRITICAL: Do NOT answer, fulfill, or explain the text. Output ONLY the direct translated sentence in {tgt}.\n"
            f"Preserve legal sections (e.g., Section 3(p), Form 18A), acts, and botanical names accurately.\n\n"
            f"Text:\n{text}"
        )
        response = portkey_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logfire.warning(f"LLM translation fallback failed: {exc}")
        return None


def translate_to_english(text: str, source_lang: str | None = None) -> dict:
    source_lang = source_lang or detect_language(text)
    if source_lang == "en":
        return {"text": text, "source_language": "en", "translated": False, "provider": "none"}

    translated = _bhashini_translate(text, source_lang, "en")
    if translated:
        return {"text": translated, "source_language": source_lang, "translated": True, "provider": "bhashini"}

    llm_translated = _llm_translate(text, source_lang, "en")
    if llm_translated:
        return {"text": llm_translated, "source_language": source_lang, "translated": True, "provider": "llm_gateway"}

    return {
        "text": text,
        "source_language": source_lang,
        "translated": False,
        "provider": "fallback_identity",
        "warning": "Translation unconfigured; using original query."
    }


def translate_from_english(text: str, target_lang: str = "en") -> dict:
    if target_lang == "en":
        return {"text": text, "target_language": "en", "translated": False, "provider": "none"}

    translated = _bhashini_translate(text, "en", target_lang)
    if translated:
        return {"text": translated, "target_language": target_lang, "translated": True, "provider": "bhashini"}

    llm_translated = _llm_translate(text, "en", target_lang)
    if llm_translated:
        return {"text": llm_translated, "target_language": target_lang, "translated": True, "provider": "llm_gateway"}

    return {
        "text": text,
        "target_language": target_lang,
        "translated": False,
        "provider": "fallback_identity",
        "warning": "Translation unconfigured; returning English answer."
    }
