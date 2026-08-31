from app.agents.state import AgentState
from app.gateway import get_langchain_llm
from app.observability.logfire_compat import logfire
import re

llm = get_langchain_llm(feature="planner")

def planner_node(state: AgentState):
    """
    IP-SAKTI Sahayak Planner:
    Classifies user intent into REGULATORY_IP, FORMULATION_CLASSIFICATION, 
    CONVERSATIONAL, or OFF_TOPIC.
    """
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    preclassified = _preclassify_simple_message(user_message)
    if preclassified:
        decision = preclassified
        logfire.info(f"Intent identified by deterministic precheck: {decision}")
    else:
        decision = None
    
    prompt = f"""
    You are IP-SAKTI Sahayak, an AI assistant for Ayurveda IP and regulatory guidance.
    
    Classify the user's query into EXACTLY ONE of these categories:
    
    1. REGULATORY_IP - Questions about patents, GI, trademarks, ABS, biodiversity, 
       drug licensing, FSSAI classification, or Indian/international IP frameworks
    2. FORMULATION_CLASSIFICATION - Questions asking to classify an Ayurvedic product 
       or formulation (Classical vs Proprietary vs Phytopharmaceutical vs Food vs Cosmetic)
    3. CONVERSATIONAL - Greetings, follow-ups using conversation history, general chat
    4. OFF_TOPIC - Queries unrelated to Ayurveda, IP, or regulatory matters
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Output ONLY the category name (REGULATORY_IP, FORMULATION_CLASSIFICATION, 
    CONVERSATIONAL, or OFF_TOPIC).
    """
    
    if decision is None:
        with logfire.span("🧠 Planner Decision"):
            try:
                decision = llm.invoke(prompt).content.strip().upper()
            except Exception as e:
                logfire.warning(f"Planner LLM unavailable; using deterministic fallback: {e}")
                decision = _fallback_intent(user_message)
            if decision not in {"REGULATORY_IP", "FORMULATION_CLASSIFICATION", "CONVERSATIONAL", "OFF_TOPIC"}:
                decision = _fallback_intent(user_message)
            logfire.info(f"Intent identified: {decision}")
    
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "intent": "CONVERSATIONAL",
            "status": "Handling conversationally...",
            "plan": ["Intent: Conversational", "Retrieval: Skipped"]
        }
    
    if decision == "OFF_TOPIC":
        refusal = "I'm IP-SAKTI Sahayak, focused on Ayurveda IP, regulatory pathways, ABS, biodiversity, and traditional-knowledge guidance. I can't help with that topic, but you can ask me about Ayurveda-related compliance or IP protection."
        return {
            "current_query": "OFF_TOPIC",
            "intent": "OFF_TOPIC",
            "status": "Off-topic query blocked.",
            "plan": ["Intent: Off-topic", "Action: Refuse"],
            "refusal_reason": refusal,
            "final_answer": refusal,
        }
    
    if decision == "FORMULATION_CLASSIFICATION":
        return {
            "current_query": user_message,
            "intent": "FORMULATION_CLASSIFICATION",
            "status": "Formulation classification required...",
            "plan": ["Intent: Formulation Classification", "Action: Classify product"]
        }
    
    return {
        "current_query": user_message,
        "intent": "REGULATORY_IP",
        "status": f"Regulatory/IP research needed: {user_message[:50]}...",
        "plan": ["Intent: Regulatory/IP", f"Query: {user_message}"]
    }


def _fallback_intent(message: str) -> str:
    text = message.lower()
    simple_intent = _preclassify_simple_message(message)
    if simple_intent:
        return simple_intent
    if _looks_like_regulatory_ip(text):
        return "REGULATORY_IP"
    if re.search(r"\b(thanks|thank you|who are you|what can you do)\b", text):
        return "CONVERSATIONAL"
    if _looks_like_formulation_classification(text):
        return "FORMULATION_CLASSIFICATION"
    return "OFF_TOPIC"


def _preclassify_simple_message(message: str) -> str | None:
    text = message.strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", "", text)
    if normalized in {"hi", "hello", "hey", "hii", "hiii", "namaste", "thanks", "thank you"}:
        return "CONVERSATIONAL"
    if re.fullmatch(r"(hi|hello|hey|namaste)\s+(there|sakti|ip sakti|assistant)", normalized):
        return "CONVERSATIONAL"
    if _looks_like_regulatory_ip(normalized):
        return "REGULATORY_IP"
    if _looks_like_formulation_classification(normalized):
        return "FORMULATION_CLASSIFICATION"
    return None


def _looks_like_regulatory_ip(text: str) -> bool:
    regulatory_terms = (
        "patent", "patentability", "patents act", "section", "ipc",
        "gi", "geographical indication", "trademark", "abs", "biodiversity",
        "traditional knowledge", "tkdl", "fssai", "ayush", "drug licence",
        "drug license", "cosmetics act", "wipo", "trips", "nagoya", "pct",
        "biological source", "biological origin", "disclosure"
    )
    return any(term in text for term in regulatory_terms)


def _looks_like_formulation_classification(text: str) -> bool:
    classification_terms = (
        "classify", "classification", "which pathway", "what pathway",
        "category", "classical or proprietary", "food or drug",
        "phytopharmaceutical", "ayurveda aahar", "cosmetic product"
    )
    product_terms = (
        "tablet", "capsule", "syrup", "cream", "oil", "supplement",
        "ingredients", "intended use", "formulation", "product"
    )
    return any(term in text for term in classification_terms) or (
        any(term in text for term in product_terms)
        and any(term in text for term in ("classical", "proprietary", "cosmetic", "food", "drug"))
        and "patent" not in text
    )
