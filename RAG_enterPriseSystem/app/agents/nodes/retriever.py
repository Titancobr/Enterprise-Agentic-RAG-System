from app.observability.logfire_compat import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import hybrid_search
from app.services.retrieval.ranking_service import rerank_documents


AUTHORIZED_SOURCE_TERMS = (
    "patents act",
    "patents (amendment) rules",
    "drugs & cosmetics",
    "drugs and cosmetics",
    "biological diversity",
    "fssai",
    "geographical indications",
    "gi act",
    "trips",
    "wipo",
    "nagoya",
    "trade marks",
    "trademark",
    "copyright",
    "plant varieties",
    "ppvfr",
    "designs act",
    "cosmetic rules",
    "cosmetics rules",
    "tkdl",
    "pct",
)


def retrieve_node(state: AgentState):
    """
    Performs hybrid search (dense vector + BM25 keyword) followed by
    semantic reranking for regulatory/IP queries.
    Returns empty list on failure to allow graceful degradation.
    """
    query = state["current_query"]
    jurisdiction = state.get("jurisdiction", "INDIA")
    
    try:
        with logfire.span("🔍 Knowledge Retrieval", query=query, jurisdiction=jurisdiction):
            logfire.info(f"Searching Qdrant + BM25 for: {query}")
            raw_results = hybrid_search(query, limit=15)
            logfire.info(f"Retrieved {len(raw_results)} candidates from Hybrid Search")
            raw_results = [doc for doc in raw_results if _is_authorized_evidence(doc)]
            raw_results = [doc for doc in raw_results if _matches_jurisdiction(doc, jurisdiction)]
            logfire.info(f"Evidence filter kept {len(raw_results)} authorized candidates")
            
            if not raw_results:
                logfire.warning("No retrieval results - returning empty context")
                return {
                    "documents": [],
                    "status": "No relevant documents found.",
                    "plan": state["plan"] + ["Context Retrieved: 0 chunks"]
                }
            
            doc_contents = [doc['content'] for doc in raw_results]
            metadata_by_content = {doc["content"]: doc for doc in raw_results}
            
            with logfire.span("⚖️ Semantic Reranking"):
                reranked_contents = rerank_documents(query, doc_contents, top_n=5)
                logfire.info("Reranking complete. Kept top 5 most relevant chunks.")
            
            formatted_docs = [
                _format_doc(metadata_by_content.get(doc, {}), doc)
                for doc in reranked_contents
            ]
        
        return {
            "documents": formatted_docs,
            "status": f"Found {len(formatted_docs)} relevant context chunks.",
            "plan": state["plan"] + [f"Context Retrieved: {len(formatted_docs)} chunks"]
        }
    
    except Exception as e:
        logfire.error(f"Retrieval failed: {e}")
        # Graceful degradation - return empty docs so responder can still answer
        return {
            "documents": [],
            "status": "Retrieval unavailable - answering from general knowledge.",
            "plan": state["plan"] + ["Context Retrieved: ERROR - degraded mode"]
        }


def _is_authorized_evidence(doc: dict) -> bool:
    source = str(doc.get("source", "")).lower()
    content = str(doc.get("content", "")).lower()
    if "noisy_data" in source or "/noisy/" in source:
        return False
    if "legal_corpus" in source or any(term in source for term in AUTHORIZED_SOURCE_TERMS):
        return True
    return any(term in content[:1000] for term in AUTHORIZED_SOURCE_TERMS)


def _matches_jurisdiction(doc: dict, jurisdiction: str) -> bool:
    if jurisdiction in {"", "UNKNOWN", "BOTH", None}:
        return True
    doc_jurisdiction = str(doc.get("jurisdiction") or "").upper()
    if not doc_jurisdiction or doc_jurisdiction == "UNKNOWN":
        source = str(doc.get("source", "")).lower()
        if jurisdiction == "INDIA":
            return not any(term in source for term in ("trips", "wipo", "nagoya", "pct", "treaties"))
        if jurisdiction == "INTERNATIONAL":
            return any(term in source for term in ("trips", "wipo", "nagoya", "pct", "treaties"))
    return doc_jurisdiction == jurisdiction


def _format_doc(metadata: dict, content: str) -> str:
    return "\n".join([
        f"SOURCE: {metadata.get('source', 'authorized_source')}",
        f"SECTION: {metadata.get('section', '')}",
        f"JURISDICTION: {metadata.get('jurisdiction', 'UNKNOWN')}",
        f"URL: {metadata.get('url', '')}",
        f"VERSION_DATE: {metadata.get('version_date', '')}",
        f"CATEGORY: {metadata.get('category', '')}",
        f"CONTENT: {content}",
    ])
