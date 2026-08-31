from app.agents.state import AgentState
from app.gateway import get_langchain_llm
from app.observability.logfire_compat import logfire

llm = get_langchain_llm(feature="formulation_classifier")

FORMULATION_TYPES = {
    "CLASSICAL_AYURVEDIC": "Classical Ayurvedic medicine based on authoritative texts",
    "PROPRIETARY_AYURVEDIC": "Patent/proprietary Ayurvedic medicine with modified or new composition",
    "PHYTOPHARMACEUTICAL": "Plant-based drug under phytopharmaceutical pathway",
    "FOOD_AYURVEDA_AAHAR": "Food/nutraceutical route such as Ayurveda Aahar",
    "COSMETIC": "Cosmetic/personal care product route",
    "INSUFFICIENT_INFO": "Needs more details before classification"
}


def formulation_classifier_node(state: AgentState):
    """
    Asks minimal classification questions and estimates the likely product pathway.
    This is guidance, not a legal determination.
    """
    query = state["current_query"]

    prompt = f"""
    You are classifying an Ayurveda-related product for regulatory/IP guidance.

    User description:
    "{query}"

    Classify into EXACTLY ONE:
    - CLASSICAL_AYURVEDIC
    - PROPRIETARY_AYURVEDIC
    - PHYTOPHARMACEUTICAL
    - FOOD_AYURVEDA_AAHAR
    - COSMETIC
    - INSUFFICIENT_INFO

    Also decide if ABS/biodiversity compliance may be required when biological resources,
    commercial utilization, research, or foreign access are involved.

    Return STRICT JSON only:
    {{
      "formulation_type": "...",
      "abs_required": true/false,
      "confidence_score": 0.0-1.0,
      "missing_questions": ["question 1", "question 2"]
    }}
    """

    with logfire.span("🧪 Formulation Classifier"):
        try:
            raw = llm.invoke(prompt).content.strip()
            logfire.info(f"Classifier raw output: {raw[:300]}")
        except Exception as e:
            logfire.warning(f"Classifier LLM unavailable; using fallback: {e}")
            raw = ""

    formulation_type = "INSUFFICIENT_INFO"
    abs_required = None
    confidence_score = 0.5

    try:
        import json
        parsed = json.loads(raw)
        formulation_type = parsed.get("formulation_type", formulation_type)
        abs_required = parsed.get("abs_required", abs_required)
        confidence_score = float(parsed.get("confidence_score", confidence_score))
    except Exception:
        logfire.warning("Could not parse classifier JSON; falling back to insufficient info.")
        formulation_type, abs_required, confidence_score = _fallback_classification(query)

    if formulation_type not in FORMULATION_TYPES:
        formulation_type = "INSUFFICIENT_INFO"

    return {
        "formulation_type": formulation_type,
        "abs_required": abs_required,
        "confidence_score": confidence_score,
        "plan": state["plan"] + [f"Formulation: {formulation_type}", f"ABS required: {abs_required}"],
        "status": f"Likely pathway: {FORMULATION_TYPES[formulation_type]}"
    }


def _fallback_classification(query: str) -> tuple[str, bool | None, float]:
    text = query.lower()
    abs_required = any(term in text for term in (
        "biological", "plant", "herb", "herbal", "forest", "traditional knowledge",
        "foreign", "export", "commercial", "source", "origin", "ashwagandha",
        "awasheagandha", "shallaki", "guduchi", "guggulu", "turmeric", "neem"
    ))
    if any(term in text for term in ("cosmetic", "skin", "hair", "soap", "cream", "personal care")):
        return "COSMETIC", abs_required, 0.65
    if any(term in text for term in ("therapeutic", "treating disease", "joint pain", "pain", "disease", "tablet", "capsule", "syrup")):
        if any(term in text for term in ("charaka", "sushruta", "ashtanga", "classical text", "first schedule")):
            return "CLASSICAL_AYURVEDIC", abs_required, 0.7
        return "PROPRIETARY_AYURVEDIC", abs_required, 0.65
    if any(term in text for term in ("food", "nutraceutical", "supplement", "ayurveda aahar")):
        return "FOOD_AYURVEDA_AAHAR", abs_required, 0.65
    if "phytopharmaceutical" in text:
        return "PHYTOPHARMACEUTICAL", abs_required, 0.7
    if any(term in text for term in ("charaka", "sushruta", "ashtanga", "classical text", "first schedule")):
        return "CLASSICAL_AYURVEDIC", abs_required, 0.7
    if any(term in text for term in ("proprietary", "new combination", "modified", "tablet", "capsule", "syrup", "formulation")):
        return "PROPRIETARY_AYURVEDIC", abs_required, 0.6
    return "INSUFFICIENT_INFO", None, 0.45
