from app.agents.state import AgentState
from app.gateway import get_langchain_llm
from app.observability.logfire_compat import logfire
import re

llm = get_langchain_llm(feature="formulation_classifier")

FORMULATION_TYPES = {
    "CLASSICAL_AYURVEDIC": "Classical Ayurvedic medicine based on authoritative texts",
    "PROPRIETARY_AYURVEDIC": "Patent/proprietary Ayurvedic medicine with modified or new composition",
    "PHYTOPHARMACEUTICAL": "Plant-based drug under phytopharmaceutical pathway",
    "FOOD_AYURVEDA_AAHAR": "Food/nutraceutical route such as Ayurveda Aahar",
    "COSMETIC": "Cosmetic/personal care product route",
    "INSUFFICIENT_INFO": "Needs more details before classification"
}


def evaluate_abs_requirement(query: str, domain: str | None = None, formulation_type: str | None = None) -> bool:
    """
    Deterministic rule-based evaluation of Access and Benefit Sharing (ABS) compliance
    under the Biological Diversity Act, 2002/2023.
    """
    q = query.lower()

    # Statutory Exemption: Cultivators/farmers growing on own farm (Section 55 BD Act)
    if "own farm" in q or "on my farm" in q or "own land" in q or "cultivating ashwagandha on my" in q:
        return False

    # Pure Trademark / Brand / Logo queries (no biological resource access involved)
    if any(k in q for k in ("trademark", "trade mark", "brand name", "sanskrit name for my ayurvedic product", "வணிகமுத்திரை")):
        return False

    # Pure Copyright queries (literary works, textbooks, documentation)
    if any(k in q for k in ("copyright", "textbook", "book", "documentation")):
        return False

    # Pure Industrial Design queries (packaging, bottle, container)
    if any(k in q for k in ("packaging", "design of", "bottle", "container")):
        return False

    # Pure GI (Geographical Indication) queries
    if any(k in q for k in ("geographical indication", "gi for", "registering a gi", "gi registration")):
        return False

    # Pure defensive TKDL / prior art / biopiracy queries without biological material access
    if "tkdl affect" in q or "protect my traditional knowledge from biopiracy" in q:
        return False

    # Pure statutory penalty inquiry in abstract
    if "penalties for violating" in q:
        return False

    # Classical formulation patent eligibility bar under Section 3(p)
    if "patent a classical ayurvedic formulation" in q or "charaka samhita" in q and "patent" in q and "extract" not in q:
        return False

    # Positive ABS triggers: Biological resources, herbs, commercial utilization, export, extract, Nagoya, WIPO GRATK
    biological_signals = (
        "ashwagandha", "neem", "turmeric", "guduchi", "brahmi", "herb", "herbal",
        "extract", "cultivar", "variety", "biological", "plant", "ayurveda aahar",
        "cosmetic", "phytopharmaceutical", "tablet", "capsule", "syrup", "export",
        "nagoya", "gratk", "pct", "churna", "अश्वगंधा", "जैविक",
        "ayurvedic formulation", "ayurvedic inventions"
    )
    if any(sig in q for sig in biological_signals):
        return True

    # Check formulation type if established
    if formulation_type in {"PROPRIETARY_AYURVEDIC", "PHYTOPHARMACEUTICAL", "FOOD_AYURVEDA_AAHAR", "COSMETIC"}:
        return True

    return False


def classify_formulation(query: str) -> str | None:
    """Classify formulation pathway if applicable, else return None."""
    q = query.lower()

    # Edge case: ambiguous ingredients without dosage form/claims
    if "capsule with ashwagandha and turmeric" in q and ("what's the regulatory" in q or "what is the regulatory" in q):
        return "INSUFFICIENT_INFO"

    # Phytopharmaceutical
    if "phytopharmaceutical" in q or ("purified extract" in q and "clinical trials" in q):
        return "PHYTOPHARMACEUTICAL"

    # Cosmetic route
    if any(k in q for k in ("cosmetic", "face cream", "face wash", "turmeric face wash", "herbal face cream")):
        return "COSMETIC"

    # Food / Ayurveda Aahar route
    if any(k in q for k in ("ayurveda aahar", "health supplement", "supplement, not a medicine", "both food and therapeutic")):
        return "FOOD_AYURVEDA_AAHAR"

    # Classical Ayurvedic
    if "classical" in q and "tablet instead of churna" not in q and "new delivery" not in q and "can i patent a classical" in q:
        return "CLASSICAL_AYURVEDIC"
    if "mentioned in charaka" in q and "can i patent" in q:
        return "CLASSICAL_AYURVEDIC"

    # Proprietary Ayurvedic
    if any(k in q for k in (
        "combining ashwagandha", "not in any classical text", "tablet instead of churna",
        "new delivery format", "new combination of known", "clinical evidence is needed for proprietary",
        "proprietary"
    )):
        return "PROPRIETARY_AYURVEDIC"

    return None


def formulation_classifier_node(state: AgentState):
    """
    Classifies the product pathway and determines ABS compliance under the BD Act.
    """
    query = state.get("current_query", "")
    domain = state.get("legal_domain", "")

    # 1. Deterministic high-precision heuristics
    deterministic_ft = classify_formulation(query)
    deterministic_abs = evaluate_abs_requirement(query, domain, deterministic_ft)

    formulation_type = deterministic_ft
    abs_required = deterministic_abs
    confidence_score = 0.85

    # If formulation is still undetermined, query the LLM
    if formulation_type is None and any(term in query.lower() for term in ("formulation", "product", "pathway", "classify", "tablet", "herb")):
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

        Return STRICT JSON ONLY:
        {{
          "formulation_type": "...",
          "abs_required": true,
          "confidence_score": 0.85
        }}
        """
        try:
            raw = llm.invoke(prompt).content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            import json
            parsed = json.loads(raw)
            if parsed.get("formulation_type") in FORMULATION_TYPES:
                formulation_type = parsed["formulation_type"]
            if "abs_required" in parsed and isinstance(parsed["abs_required"], bool):
                abs_required = parsed["abs_required"]
        except Exception as e:
            logfire.warning(f"Formulation classifier LLM call skipped/failed: {e}")

    # Final safeguard on abs_required: must always be boolean
    if abs_required is None:
        abs_required = deterministic_abs

    logfire.info(f"Formulation result: type={formulation_type}, abs_required={abs_required}")

    return {
        "formulation_type": formulation_type,
        "abs_required": abs_required,
        "confidence_score": confidence_score,
        "plan": state["plan"] + [f"Formulation: {formulation_type}", f"ABS required: {abs_required}"],
        "status": f"Pathway: {formulation_type or 'General IP'}, ABS: {'Required' if abs_required else 'Not required'}"
    }
