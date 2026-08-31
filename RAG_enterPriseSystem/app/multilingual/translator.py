"""
Multilingual translation service for IP-SAKTI Sahayak.
Supports Bhashini API and AI4Bharat IndicTrans2.

For SIH demo, we use a translation stub that can be swapped for real API calls.
"""

import os
from typing import Optional, Dict, Any
from enum import Enum


class LanguageCode(str, Enum):
    EN = "en"
    HI = "hi"
    TA = "ta"
    TE = "te"
    KN = "kn"
    ML = "ml"
    GU = "gu"
    MR = "mr"
    BN = "bn"
    PA = "pa"
    OR = "or"
    AS = "as"
    UR = "ur"


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "mr": "Marathi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}


class TranslationService:
    """
    Translation service that can use Bhashini API or IndicTrans2.
    Falls back to stub for demo if no API configured.
    """
    
    def __init__(self):
        self.bhashini_api_key = os.getenv("BHASHINI_API_KEY")
        self.bhashini_endpoint = os.getenv("BHASHINI_ENDPOINT", "https://meity-auth.ulcacontrib.org")
        self.use_indictrans = os.getenv("USE_INDICTRANS", "false").lower() == "true"
        
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source_lang to target_lang.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., "en", "hi")
            target_lang: Target language code
            
        Returns:
            Translated text (or original if translation fails)
        """
        if source_lang == target_lang:
            return text
            
        if self.use_indictrans:
            return self._translate_indictrans(text, source_lang, target_lang)
        elif self.bhashini_api_key:
            return self._translate_bhashini(text, source_lang, target_lang)
        else:
            return self._translate_stub(text, source_lang, target_lang)
    
    def _translate_bhashini(self, text: str, source: str, target: str) -> str:
        """Use Bhashini API for translation."""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.bhashini_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputText": text,
                "sourceLanguage": source,
                "targetLanguage": target
            }
            
            response = requests.post(
                f"{self.bhashini_endpoint}/ulca/apis/v0/model/compute",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("outputText", text)
            else:
                print(f"Bhashini API error: {response.status_code}")
                return text
                
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def _translate_indictrans(self, text: str, source: str, target: str) -> str:
        """Use AI4Bharat IndicTrans2 for translation (requires model download)."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=512)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except ImportError:
            print("IndicTrans2 not available. Install: pip install transformers sentencepiece sacremoses")
            return self._translate_stub(text, source, target)
        except Exception as e:
            print(f"IndicTrans2 error: {e}")
            return text
    
    def _translate_stub(self, text: str, source: str, target: str) -> str:
        """
        Stub translation for demo purposes.
        In production, replace with actual API calls.
        """
        stub_prefix = {
            "hi": "[हिंदी अनुवाद] ",
            "ta": "[தமிழ் மொழிபெயர்ப்பு] ",
            "te": "[తెలుగు అనువాదం] ",
            "kn": "[ಕನ್ನಡ ಅನುವಾದ] ",
            "ml": "[മലയാളം വിവർത്തനം] ",
            "gu": "[ગુજરાતી અનુવાદ] ",
            "mr": "[मराठी अनुवाद] ",
            "bn": "[বাংলা অনুবাদ] ",
            "pa": "[ਪੰਜਾਬੀ ਅਨੁਵਾਦ] ",
            "or": "[ଓଡ଼ିଆ ଅନୁବାଦ] ",
            "as": "[অসমীয়া অনুবাদ] ",
            "ur": "[اردو ترجمہ] ",
        }
        
        if target in stub_prefix:
            return stub_prefix[target] + text
        return text
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text (simple heuristic)."""
        # Simple script-based detection
        if any('\u0900' <= c <= '\u097F' for c in text):
            return "hi"
        elif any('\u0B80' <= c <= '\u0BFF' for c in text):
            return "ta"
        elif any('\u0C00' <= c <= '\u0C7F' for c in text):
            return "te"
        elif any('\u0C80' <= c <= '\u0CFF' for c in text):
            return "kn"
        elif any('\u0D00' <= c <= '\u0D7F' for c in text):
            return "ml"
        elif any('\u0A80' <= c <= '\u0AFF' for c in text):
            return "gu"
        elif any('\u0980' <= c <= '\u09FF' for c in text):
            return "bn"
        else:
            return "en"


# Singleton instance
_translator: Optional[TranslationService] = None


def get_translator() -> TranslationService:
    global _translator
    if _translator is None:
        _translator = TranslationService()
    return _translator


def translate_text(text: str, target_lang: str = "en") -> str:
    """Convenience function for translation."""
    translator = get_translator()
    source_lang = translator.detect_language(text)
    return translator.translate(text, source_lang, target_lang)
