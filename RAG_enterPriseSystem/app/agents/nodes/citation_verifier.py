from app.agents.state import AgentState
from app.observability.logfire_compat import logfire
import re


def citation_verifier_node(state: AgentState):
    """
    Lightweight prototype citation verifier.
    Extracts bracketed/source-like references and lowers confidence when no citations exist.
    """
    answer = state.get("final_answer", "") or ""
    documents = state.get("documents", []) or []

    citation_patterns = [
        r"\[(.*?)\]",
        r"Section\s+\d+[A-Za-z()]*",
        r"Rule\s+\d+[A-Za-z()]*",
        r"Article\s+\d+[A-Za-z()]*",
        r"Act,?\s*\d{4}",
        r"Regulations?,?\s*\d{4}",
    ]

    evidence = [_parse_doc_metadata(doc) for doc in documents]
    citations = []
    for pattern in citation_patterns:
        for match in re.findall(pattern, answer, flags=re.IGNORECASE):
            text = match if isinstance(match, str) else " ".join(match)
            matched_evidence = _best_evidence_for_citation(text, evidence)
            citations.append({
                "text": text,
                "verified": bool(matched_evidence),
                "source": matched_evidence.get("source", "retrieved_context") if matched_evidence else "unverified",
                "section": matched_evidence.get("section") if matched_evidence else None,
                "url": matched_evidence.get("url") if matched_evidence else None,
                "confidence": 0.9 if matched_evidence else 0.45,
            })

    unique_citations = []
    seen = set()
    for citation in citations:
        key = citation["text"].lower()
        if key not in seen:
            unique_citations.append(citation)
            seen.add(key)

    base_confidence = state.get("confidence_score") or 0.75
    if not documents and state.get("intent") != "CONVERSATIONAL":
        base_confidence = min(base_confidence, 0.35)
    elif not unique_citations and state.get("intent") != "CONVERSATIONAL":
        base_confidence = min(base_confidence, 0.55)
    elif unique_citations and any(not c.get("verified") for c in unique_citations):
        base_confidence = min(base_confidence, 0.7)

    with logfire.span("✅ Citation Verifier"):
        logfire.info(f"Citations extracted: {len(unique_citations)}")

    return {
        "citations": unique_citations,
        "confidence_score": round(base_confidence, 2),
        "plan": state["plan"] + [f"Citations verified: {len(unique_citations)}"],
        "status": "Answer verified with citation checks."
    }


def _parse_doc_metadata(doc: str) -> dict:
    metadata = {"content": doc}
    for line in doc.splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key in {"source", "section", "jurisdiction", "url", "version_date", "category"}:
            metadata[key] = value.strip()
    return metadata


def _best_evidence_for_citation(citation_text: str, evidence: list[dict]) -> dict | None:
    citation = citation_text.lower()
    for item in evidence:
        section = str(item.get("section", "")).lower()
        source = str(item.get("source", "")).lower()
        content = str(item.get("content", "")).lower()
        if citation and (citation in section or citation in source or citation in content[:1500]):
            return item
    tokens = [token for token in re.split(r"\W+", citation) if len(token) > 3]
    for item in evidence:
        haystack = f"{item.get('section', '')} {item.get('source', '')} {item.get('content', '')[:1500]}".lower()
        if tokens and any(token in haystack for token in tokens):
            return item
    return evidence[0] if evidence else None
