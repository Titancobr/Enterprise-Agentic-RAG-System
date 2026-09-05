from app.observability.logfire_compat import logfire
from pathlib import Path
from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.bm25_service import bm25_search, build_bm25_index, is_index_ready

try:
    from qdrant_client import QdrantClient
except ModuleNotFoundError:
    QdrantClient = None

_qdrant_url = (settings.QDRANT_URL or "").strip()
_qdrant_key = (settings.QDRANT_API_KEY or "").strip()

client = QdrantClient(
    url=_qdrant_url,
    api_key=_qdrant_key,
    check_compatibility=False
) if QdrantClient and _qdrant_url and not _qdrant_url.startswith("your_") else None

def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        if client is None:
            return []
        query_vector = embed_query(query)

        # Using query_points - the modern standard for Qdrant
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True # JSON
        )

        results = []
        for res in response.points:
            payload = res.payload or {}
            results.append({
                "content": payload.get("text", ""),
                "source": payload.get("source", "Unknown"),
                "score": res.score,
                "jurisdiction": payload.get("jurisdiction", "UNKNOWN"),
                "section": payload.get("section", ""),
                "url": payload.get("url", ""),
                "version_date": payload.get("version_date", ""),
                "category": payload.get("category", ""),
            })
        
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []


def get_all_documents() -> list[dict]:
    """
    Scroll through the entire Qdrant collection and return all document
    chunks.  Used to build the in-memory BM25 index at startup.
    """
    all_docs: list[dict] = []
    try:
        if not QdrantClient:
            raise RuntimeError("qdrant-client package is not installed")
        if client is None:
            raise RuntimeError("Qdrant cloud credentials not configured in environment (QDRANT_CLUSTER_ENDPOINT / QDRANT_API_KEY)")
        offset = None
        while True:
            response = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = response
            for pt in points:
                payload = pt.payload or {}
                all_docs.append({
                    "content": payload.get("text", ""),
                    "source": payload.get("source", "Unknown"),
                    "score": 0.0,
                    "jurisdiction": payload.get("jurisdiction", "UNKNOWN"),
                    "section": payload.get("section", ""),
                    "url": payload.get("url", ""),
                    "version_date": payload.get("version_date", ""),
                    "category": payload.get("category", ""),
                })
            if next_offset is None:
                break
            offset = next_offset

        logfire.info(f"📚 Loaded {len(all_docs)} documents from Qdrant for BM25 index.")
    except Exception as e:
        logfire.info(f"ℹ️ Qdrant cloud sync skipped: {e} — loading local authorized legal corpus.")
    if not all_docs:
        all_docs = load_local_legal_documents()
    return all_docs


def load_local_legal_documents() -> list[dict]:
    """
    Load bundled authorized legal source snippets for offline/degraded MVP use.
    These files are intentionally separate from DATA/noisy_data.
    """
    docs: list[dict] = []
    corpus_dir = Path(__file__).resolve().parents[3] / "DATA" / "legal_corpus"
    for path in sorted(corpus_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = path.read_text(errors="ignore").strip()
        except Exception as e:
            logfire.warning(f"Could not read local legal source {path}: {e}")
            continue
        if not text:
            continue
        metadata, body = _parse_local_legal_metadata(text)
        docs.append({
            "content": body or text,
            "source": str(path.relative_to(corpus_dir)),
            "score": 0.0,
            "jurisdiction": metadata.get("jurisdiction", "UNKNOWN").upper(),
            "section": metadata.get("section", metadata.get("sections", "")),
            "url": metadata.get("url", ""),
            "version_date": metadata.get("version_date", ""),
            "category": metadata.get("category", ""),
        })
    if docs:
        logfire.info(f"Loaded {len(docs)} local authorized legal documents.")
    return docs


def _parse_local_legal_metadata(text: str) -> tuple[dict[str, str], str]:
    metadata: dict[str, str] = {}
    body_start = 0
    lines = text.splitlines()
    allowed = {"source", "section", "sections", "url", "version_date", "jurisdiction", "category", "language"}
    for i, line in enumerate(lines):
        if not line.strip():
            body_start = i + 1
            break
        if ":" not in line:
            body_start = i
            break
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key not in allowed:
            body_start = i
            break
        metadata[key] = value.strip()
    return metadata, "\n".join(lines[body_start:]).strip()


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

RRF_K = 60          # standard smoothing constant
ALPHA  = 0.5        # weight: 0.5 = equal dense + sparse


def hybrid_search(query: str, limit: int = 15) -> list[dict]:
    """
    Hybrid retrieval: runs dense-vector (Qdrant) and sparse-keyword (BM25)
    searches in parallel, then merges via Reciprocal Rank Fusion.

    Parameters
    ----------
    query : str
        The user's search query.
    limit : int
        Number of candidates to retrieve from *each* retriever before fusion.

    Returns
    -------
    list[dict]
        Fused results sorted by RRF score (descending).  Each dict has keys:
        content, source, rrf_score, dense_rank, bm25_rank.
    """
    with logfire.span("🔀 Hybrid Search (Dense + BM25)", query=query[:80]):
        if not is_index_ready():
            local_docs = load_local_legal_documents()
            if local_docs:
                build_bm25_index(local_docs)

        # 1. Dense vector search
        dense_results = search_enterprise_knowledge(query, limit=limit) if client else []
        logfire.info(f"Dense vector returned {len(dense_results)} results")

        # 2. BM25 sparse keyword search
        bm25_results = bm25_search(query, limit=limit)
        logfire.info(f"BM25 keyword returned {len(bm25_results)} results")

        # 3. Build rank maps (content → 1-indexed rank)
        dense_rank: dict[str, int] = {}
        for i, doc in enumerate(dense_results):
            key = doc["content"]
            if key not in dense_rank:
                dense_rank[key] = i + 1

        bm25_rank: dict[str, int] = {}
        for i, doc in enumerate(bm25_results):
            key = doc["content"]
            if key not in bm25_rank:
                bm25_rank[key] = i + 1

        # 4. Collect all unique documents
        all_contents: dict[str, dict] = {}
        for doc in dense_results + bm25_results:
            key = doc["content"]
            if key not in all_contents:
                all_contents[key] = {
                    "content": doc["content"],
                    "source": doc.get("source", "Unknown"),
                    "jurisdiction": doc.get("jurisdiction", "UNKNOWN"),
                    "section": doc.get("section", ""),
                    "url": doc.get("url", ""),
                    "version_date": doc.get("version_date", ""),
                    "category": doc.get("category", ""),
                }

        # 5. Compute RRF score for each document
        miss_rank = limit + 1  # penalty rank for docs missing from one list
        fused = []
        for content_key, doc in all_contents.items():
            dr = dense_rank.get(content_key, miss_rank)
            br = bm25_rank.get(content_key, miss_rank)
            rrf = ALPHA * (1.0 / (RRF_K + dr)) + (1 - ALPHA) * (1.0 / (RRF_K + br))
            fused.append({
                **doc,
                "score": rrf,
                "dense_rank": dr,
                "bm25_rank": br,
            })

        # 6. Sort by fused RRF score (highest first)
        fused.sort(key=lambda x: x["score"], reverse=True)

        logfire.info(
            f"✅ Hybrid fusion complete — {len(fused)} unique docs, "
            f"top RRF score: {fused[0]['score']:.6f}" if fused else "no results"
        )

        return fused
