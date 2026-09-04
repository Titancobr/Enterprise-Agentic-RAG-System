"""
BM25 Sparse Keyword Index for Hybrid Retrieval.

Maintains an in-memory BM25Okapi index over all document chunks stored in
Qdrant.  At query time, returns ranked results by keyword relevance which
are then fused with dense-vector results via Reciprocal Rank Fusion (RRF).
"""

import re
from app.observability.logfire_compat import logfire

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:
    BM25Okapi = None

# ── Module state ───────────────────────────────────────────────────────────────
_bm25_index = None
_corpus_docs: list[dict] = []          # [{content, source, score}]
_tokenized_corpus: list[list[str]] = []


# ── Tokeniser ──────────────────────────────────────────────────────────────────

_LEGAL_SPLIT = re.compile(r"[^a-z0-9()§/]+")   # keep parens & section signs


def _tokenize(text: str) -> list[str]:
    """
    Lowercase whitespace split with light legal-aware normalisation.
    Keeps tokens like '3(p)', 'section', '158b', 'form18a' intact.
    """
    return [t for t in _LEGAL_SPLIT.split(text.lower()) if len(t) >= 2]


def _expand_query_tokens(query: str) -> list[str]:
    """Expand user query with statutory synonyms to maximize legal recall."""
    tokens = _tokenize(query)
    q_lower = query.lower()
    expansions = []
    
    if "trademark" in q_lower or "brand" in q_lower or "name" in q_lower:
        expansions.extend(["trade", "marks", "act", "1999", "section", "distinctive", "generic", "madrid"])
    if "copyright" in q_lower or "textbook" in q_lower or "book" in q_lower:
        expansions.extend(["copyright", "act", "1957", "literary", "expression", "authoritative", "public", "domain"])
    if "plant" in q_lower or "variety" in q_lower or "cultivar" in q_lower or "cultivat" in q_lower:
        expansions.extend(["ppvfr", "2001", "dus", "farmers", "rights", "breeders"])
    if "design" in q_lower or "packaging" in q_lower or "bottle" in q_lower or "container" in q_lower:
        expansions.extend(["designs", "act", "2000", "shape", "configuration", "ornament"])
    if "patent" in q_lower:
        expansions.extend(["patents", "act", "1970", "novelty", "inventive", "section", "3(p)", "form18a", "tkdl"])
    if "abs" in q_lower or "biodiversity" in q_lower or "nba" in q_lower or "sbb" in q_lower or "export" in q_lower:
        expansions.extend(["biological", "diversity", "act", "2023", "section", "nba", "sbb", "intimation", "benefit", "sharing"])
    if "food" in q_lower or "supplement" in q_lower or "aahar" in q_lower:
        expansions.extend(["fssai", "ayurveda", "aahar", "regulations", "2022", "section", "therapeutic", "claims"])
    if "cosmetic" in q_lower or "cream" in q_lower or "wash" in q_lower:
        expansions.extend(["cosmetics", "rules", "2020", "beautifying", "therapeutic", "claims"])
    if "proprietary" in q_lower or "tablet" in q_lower or "extract" in q_lower or "combination" in q_lower:
        expansions.extend(["rule", "158b", "safety", "efficacy", "clinical", "proprietary"])
    if "gi" in q_lower or "geographical" in q_lower:
        expansions.extend(["geographical", "indications", "goods", "act", "1999", "chennai", "registry", "section"])
    if "nagoya" in q_lower or "gratk" in q_lower or "pct" in q_lower:
        expansions.extend(["nagoya", "protocol", "pic", "mat", "wipo", "gratk", "pct"])

    return list(dict.fromkeys(tokens + expansions))



# ── Public API ─────────────────────────────────────────────────────────────────

def build_bm25_index(documents: list[dict]) -> None:
    """
    Build (or rebuild) the BM25 index from a list of document dicts.
    Each dict must have at least a 'content' key.

    Parameters
    ----------
    documents : list[dict]
        Dicts with keys: content, source, score (score can be 0.0).
    """
    global _bm25_index, _corpus_docs, _tokenized_corpus

    _corpus_docs = documents
    _tokenized_corpus = [_tokenize(doc["content"]) for doc in documents]
    _bm25_index = BM25Okapi(_tokenized_corpus) if BM25Okapi else "keyword_fallback"

    logfire.info(
        f"📖 BM25 index built — {len(documents)} documents, "
        f"avg tokens/doc: {sum(len(t) for t in _tokenized_corpus) / max(len(_tokenized_corpus), 1):.0f}"
    )


def bm25_search(query: str, limit: int = 15) -> list[dict]:
    """
    Search the BM25 index and return the top-*limit* documents ranked by
    keyword relevance.

    Returns
    -------
    list[dict]
        Each dict has keys: content, source, bm25_score.
    """
    if _bm25_index is None or not _corpus_docs:
        logfire.warning("BM25 index not initialised — returning empty results.")
        return []

    query_tokens = _expand_query_tokens(query)
    if _bm25_index == "keyword_fallback":
        query_set = set(query_tokens)
        scores = [
            len(query_set.intersection(tokens)) / max(len(query_set), 1)
            for tokens in _tokenized_corpus
        ]
    else:
        scores = _bm25_index.get_scores(query_tokens)

    # Pair each doc with its score and sort descending
    scored = sorted(
        zip(scores, _corpus_docs),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, doc in scored[:limit]:
        results.append({
            "content": doc["content"],
            "source": doc.get("source", "Unknown"),
            "jurisdiction": doc.get("jurisdiction", "UNKNOWN"),
            "section": doc.get("section", ""),
            "url": doc.get("url", ""),
            "version_date": doc.get("version_date", ""),
            "category": doc.get("category", ""),
            "bm25_score": float(score),
        })

    return results


def is_index_ready() -> bool:
    """Check whether the BM25 index has been built."""
    return _bm25_index is not None and len(_corpus_docs) > 0
