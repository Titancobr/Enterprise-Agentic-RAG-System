from app.agents.state import AgentState
from app.gateway import get_langchain_llm
from app.observability.logfire_compat import logfire

llm = get_langchain_llm(feature="jurisdiction_router")


def jurisdiction_router_node(state: AgentState):
    """
    Separates Indian domestic law from international IP/ABS frameworks.
    Prevents jurisdiction mixing in final answers.
    """
    query = state["current_query"]
    requested_jurisdiction = (state.get("requested_jurisdiction") or "").upper()
    if requested_jurisdiction in {"INDIA", "INTERNATIONAL", "BOTH"}:
        return {
            "jurisdiction": requested_jurisdiction,
            "plan": state["plan"] + [f"Jurisdiction: {requested_jurisdiction} (user selected)"],
            "status": f"Jurisdiction routed to {requested_jurisdiction}."
        }

    prompt = f"""
    Classify the jurisdiction needed for this Ayurveda IP/regulatory query.

    Query: "{query}"

    Choose EXACTLY ONE:
    - INDIA: Indian patents, GI, trademarks, Drugs & Cosmetics Act, FSSAI, BD Act, NBA/SBB/BMC, AYUSH licensing
    - INTERNATIONAL: WIPO, PCT, TRIPS, Nagoya Protocol, foreign patent/trademark filing, international biopiracy
    - BOTH: Query explicitly asks to compare India and international frameworks
    - UNKNOWN: Not enough information

    Output ONLY: INDIA, INTERNATIONAL, BOTH, or UNKNOWN.
    """

    with logfire.span("⚖️ Jurisdiction Router"):
        try:
            jurisdiction = llm.invoke(prompt).content.strip().upper()
        except Exception as e:
            logfire.warning(f"Jurisdiction router LLM unavailable; using fallback: {e}")
            jurisdiction = _fallback_jurisdiction(query)
        if jurisdiction not in {"INDIA", "INTERNATIONAL", "BOTH", "UNKNOWN"}:
            jurisdiction = "UNKNOWN"
        logfire.info(f"Jurisdiction: {jurisdiction}")

    return {
        "jurisdiction": jurisdiction,
        "plan": state["plan"] + [f"Jurisdiction: {jurisdiction}"],
        "status": f"Jurisdiction routed to {jurisdiction}."
    }


def _fallback_jurisdiction(query: str) -> str:
    text = query.lower()
    india_terms = (
        "india", "indian", "patents act", "section", "ayush", "fssai",
        "drugs and cosmetics", "biodiversity act", "nba", "sbb", "bmc",
        "gi registry", "chennai", "ipc"
    )
    international_terms = (
        "wipo", "pct", "trips", "nagoya", "international", "foreign",
        "global", "export", "outside india"
    )
    has_india = any(term in text for term in india_terms)
    has_international = any(term in text for term in international_terms)
    if has_india and has_international:
        return "BOTH"
    if has_international:
        return "INTERNATIONAL"
    if has_india or any(term in text for term in ("ayurveda", "patent", "abs", "biodiversity", "traditional knowledge")):
        return "INDIA"
    return "UNKNOWN"
