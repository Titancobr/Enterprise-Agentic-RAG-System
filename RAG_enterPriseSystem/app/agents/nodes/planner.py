from app.agents.state import AgentState
from app.gateway import get_langchain_llm
from app.observability.logfire_compat import logfire
import re

llm = get_langchain_llm(feature="planner")

LEGAL_DOMAINS = {
    "PATENT_ELIGIBILITY",
    "TRADEMARK",
    "GI_PROTECTION",
    "ABS_COMPLIANCE",
    "FSSAI_FOOD",
    "COSMETIC",
    "COPYRIGHT",
    "PLANT_VARIETY",
    "DESIGN",
    "PHYTOPHARMACEUTICAL",
    "INTERNATIONAL_IP",
    "FORMULATION_CLASSIFICATION",
    "GENERAL_IP"
}


def classify_legal_domain(query: str) -> str:
    """Classify the query into a specific Ayurveda IP / regulatory domain."""
    q = query.lower()
    if any(k in q for k in ("trademark", "trade mark", "brand name", "logo", "sanskrit name", "வணிகமுத்திரை")):
        return "TRADEMARK"
    if any(k in q for k in ("copyright", "textbook", "book", "documentation", "literary")):
        return "COPYRIGHT"
    if any(k in q for k in ("plant variety", "cultivar", "ppvfr", "farmers' rights", "breeder")):
        return "PLANT_VARIETY"
    if any(k in q for k in ("packaging", "bottle", "container", "design of my ayurvedic", "designs act")):
        return "DESIGN"
    if any(k in q for k in ("gi", "geographical indication", "kerala ayurvedic oil")):
        return "GI_PROTECTION"
    if any(k in q for k in ("abs approval", "abs compliance", "biodiversity act", "nba", "sbb", "benefit sharing", "own farm")):
        return "ABS_COMPLIANCE"
    if any(k in q for k in ("food", "supplement", "ayurveda aahar", "fssai", "health supplement", "food vs drug")):
        return "FSSAI_FOOD"
    if any(k in q for k in ("cosmetic", "face cream", "face wash", "skin", "hair wash")):
        return "COSMETIC"
    if any(k in q for k in ("phytopharmaceutical", "botanical extract with clinical")):
        return "PHYTOPHARMACEUTICAL"
    if any(k in q for k in ("pct", "wipo", "gratk", "nagoya", "us patent", "japan", "europe", "differences between indian and us")):
        return "INTERNATIONAL_IP"
    if any(k in q for k in ("patent", "patentability", "3(p)", "extraction method", "prior art", "tkdl", "novelty", "inventive")):
        return "PATENT_ELIGIBILITY"
    if any(k in q for k in ("classify", "classification", "which pathway", "classical or proprietary")):
        return "FORMULATION_CLASSIFICATION"
    return "GENERAL_IP"


def planner_node(state: AgentState):
    """
    IP-SAKTI Sahayak Planner:
    Classifies user intent and identifies the specific legal domain.
    """
    history = ""
    for msg in state.get("messages", [])[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = (state.get("messages")[-1]["content"] if state.get("messages") else "") or state.get("current_query", "") or state.get("query", "")
    preclassified = _preclassify_simple_message(user_message)
    legal_domain = classify_legal_domain(user_message)
    
    if preclassified:
        decision = preclassified
        logfire.info(f"Intent identified by deterministic precheck: {decision} (domain={legal_domain})")
    else:
        decision = None
    
    prompt = f"""
    You are IP-SAKTI Sahayak, an AI assistant for Ayurveda IP and regulatory guidance.
    
    Classify the user's query into EXACTLY ONE of these categories:
    
    1. COMPREHENSIVE - Questions that ask BOTH to classify a product/formulation AND ask about regulatory/IP matters (patents, GI, trademarks, ABS, etc.)
    2. REGULATORY_IP - Questions about patents, GI, trademarks, ABS, biodiversity, drug licensing, FSSAI classification, or Indian/international IP frameworks
    3. FORMULATION_CLASSIFICATION - Questions ONLY asking to classify an Ayurvedic product or formulation (Classical vs Proprietary vs Phytopharmaceutical vs Food vs Cosmetic)
    4. CONVERSATIONAL - Greetings, follow-ups using conversation history, general chat
    5. OFF_TOPIC - Queries completely unrelated to Ayurveda, IP, or regulatory matters
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Output ONLY the category name (COMPREHENSIVE, REGULATORY_IP, FORMULATION_CLASSIFICATION, CONVERSATIONAL, or OFF_TOPIC).
    """
    
    if decision is None:
        with logfire.span("🧠 Planner Decision"):
            try:
                decision = llm.invoke(prompt).content.strip().upper()
            except Exception as e:
                logfire.warning(f"Planner LLM unavailable; using deterministic fallback: {e}")
                decision = _fallback_intent(user_message)
            if decision not in {"COMPREHENSIVE", "REGULATORY_IP", "FORMULATION_CLASSIFICATION", "CONVERSATIONAL", "OFF_TOPIC"}:
                decision = _fallback_intent(user_message)
            logfire.info(f"Intent identified: {decision} (domain={legal_domain})")
    
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "intent": "CONVERSATIONAL",
            "legal_domain": legal_domain,
            "status": "Handling conversationally...",
            "plan": ["Intent: Conversational", "Retrieval: Skipped"]
        }
    
    if decision == "OFF_TOPIC":
        refusal = "I am IP-SAKTI Sahayak, focused on Ayurveda IP, regulatory pathways, ABS, biodiversity, and traditional knowledge guidance. I cannot assist with off-topic queries, but you can ask me about Ayurveda-related compliance, patents, trademarks, or IP protection."
        return {
            "current_query": "OFF_TOPIC",
            "intent": "OFF_TOPIC",
            "legal_domain": "OFF_TOPIC",
            "status": "Off-topic query blocked.",
            "plan": ["Intent: Off-topic", "Action: Refuse"],
            "refusal_reason": refusal,
            "final_answer": refusal,
        }
    
    if decision == "FORMULATION_CLASSIFICATION":
        return {
            "current_query": user_message,
            "intent": "FORMULATION_CLASSIFICATION",
            "legal_domain": legal_domain,
            "status": f"Formulation classification required (domain: {legal_domain})...",
            "plan": ["Intent: Formulation Classification", f"Domain: {legal_domain}", "Action: Classify product"]
        }
    
    if decision == "COMPREHENSIVE":
        return {
            "current_query": user_message,
            "intent": "COMPREHENSIVE",
            "legal_domain": legal_domain,
            "status": f"Comprehensive analysis required (domain: {legal_domain})...",
            "plan": ["Intent: Comprehensive", f"Domain: {legal_domain}", "Action: Classify product", "Action: RAG Research"]
        }

    return {
        "current_query": user_message,
        "intent": "REGULATORY_IP",
        "legal_domain": legal_domain,
        "status": f"Regulatory/IP research needed: {user_message[:50]}... (domain: {legal_domain})",
        "plan": ["Intent: Regulatory/IP", f"Domain: {legal_domain}", f"Query: {user_message}"]
    }


def _fallback_intent(message: str) -> str:
    text = message.lower()
    simple_intent = _preclassify_simple_message(message)
    if simple_intent:
        return simple_intent
    is_class = _looks_like_formulation_classification(text)
    is_reg = _looks_like_regulatory_ip(text)
    
    if is_class and is_reg:
        return "COMPREHENSIVE"
    if is_class:
        return "FORMULATION_CLASSIFICATION"
    if is_reg:
        return "REGULATORY_IP"
    if re.search(r"\b(thanks|thank you|who are you|what can you do|help)\b", text):
        return "CONVERSATIONAL"
    return "OFF_TOPIC"


def _preclassify_simple_message(message: str) -> str | None:
    text = message.strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", "", text)
    if normalized in {"hi", "hello", "hey", "hii", "hiii", "namaste", "thanks", "thank you"}:
        return "CONVERSATIONAL"
    if re.fullmatch(r"(hi|hello|hey|namaste)\s+(there|sakti|ip sakti|assistant)", normalized):
        return "CONVERSATIONAL"
    is_class = _looks_like_formulation_classification(normalized)
    is_reg = _looks_like_regulatory_ip(normalized)
    
    if is_class and is_reg:
        return "COMPREHENSIVE"
    if is_class:
        return "FORMULATION_CLASSIFICATION"
    if is_reg:
        return "REGULATORY_IP"
    return None


def _looks_like_regulatory_ip(text: str) -> bool:
    regulatory_terms = (
        "patent", "patentability", "patents act", "section", "ipc",
        "gi", "geographical indication", "trademark", "trade mark", "abs", "biodiversity",
        "traditional knowledge", "tkdl", "fssai", "ayush", "drug licence",
        "drug license", "cosmetics act", "wipo", "trips", "nagoya", "pct",
        "biological source", "biological origin", "disclosure", "copyright",
        "plant variety", "cultivar", "ppvfr", "packaging", "export",
        "अश्वगंधा", "வணிகமுத்திரை"
    )
    return any(term in text for term in regulatory_terms)


def _looks_like_formulation_classification(text: str) -> bool:
    classification_terms = (
        "classify", "classification", "which pathway", "what pathway",
        "which category", "what category", "classify it as",
        "classical or proprietary", "food or drug", "food or cosmetic",
        "phytopharmaceutical", "ayurveda aahar", "cosmetic product",
    )
    if any(term in text for term in classification_terms):
        return True
    product_terms = (
        "tablet", "capsule", "syrup", "cream", "oil", "supplement",
        "ingredients", "intended use", "formulation", "product",
        "herbal product", "ayurvedic product", "launch",
    )
    pathway_terms = ("classical", "proprietary", "cosmetic", "food", "drug")
    return (
        any(term in text for term in product_terms)
        and any(term in text for term in pathway_terms)
        and "patent" not in text
    )
