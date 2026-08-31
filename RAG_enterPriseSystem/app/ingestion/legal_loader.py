"""
Legal corpus ingestion for IP-SAKTI Sahayak.
Parses statute text files with metadata headers and indexes into Qdrant.
"""

import os
import uuid
from app.observability.logfire_compat import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim


qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

LEGAL_CORPUS_DIR = "DATA/legal_corpus"


def parse_legal_file(file_path: str) -> dict:
    """
    Parses a legal corpus file with metadata headers.
    Expected format:
    SOURCE: ...
    SECTION: ...
    URL: ...
    VERSION_DATE: ...
    JURISDICTION: ...
    CATEGORY: ...
    LANGUAGE: ...
    
    [Content follows after a blank line]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    metadata = {}
    content_start = 0
    
    for i, line in enumerate(lines):
        if ':' in line and not line.startswith('[') and not line.startswith(' '):
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            if key in ['source', 'section', 'sections', 'url', 'version_date', 'jurisdiction', 'category', 'language']:
                metadata[key] = value
        elif line.strip() == '' and content_start == 0:
            content_start = i + 1
        elif content_start > 0:
            break
    
    text_content = '\n'.join(lines[content_start:]).strip()
    
    return {
        'metadata': metadata,
        'content': text_content
    }


def chunk_legal_text(text: str, max_chars: int = 1500) -> list:
    """
    Chunk legal text by paragraphs/sections, preserving structure.
    """
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def process_legal_file(file_path: str, relative_path: str):
    """Process a single legal corpus file."""
    with logfire.span("Processing Legal File", file=relative_path):
        try:
            parsed = parse_legal_file(file_path)
            metadata = parsed['metadata']
            content = parsed['content']
            
            if not content:
                logfire.warning(f"Empty content in {relative_path}")
                return
            
            chunks = chunk_legal_text(content)
            if not chunks:
                return
            
            # Prepare payloads
            payloads = []
            for chunk in chunks:
                payload = {
                    "text": chunk,
                    "source": relative_path,
                    "source_type": "legal_corpus",
                    "source_id": metadata.get('source', 'unknown'),
                    "jurisdiction": metadata.get('jurisdiction', 'UNKNOWN').upper(),
                    "category": metadata.get('category', 'general'),
                    "section": metadata.get('section', metadata.get('sections', '')),
                    "version_date": metadata.get('version_date', ''),
                    "url": metadata.get('url', ''),
                    "language": metadata.get('language', 'en')
                }
                payloads.append(payload)
            
            # Embed
            embeddings = embed_texts(chunks)
            
            # Index
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
                for payload, vector in zip(payloads, embeddings)
            ]
            
            qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points,
            )
            logfire.info(f"Indexed {len(points)} chunks from {relative_path}")
            
        except Exception as e:
            logfire.error(f"Failed to process {relative_path}: {e}")


def ingest_legal_corpus(base_dir: str = LEGAL_CORPUS_DIR, wipe: bool = False):
    """Ingest all legal corpus files."""
    with logfire.span("Legal Corpus Ingestion", directory=base_dir):
        
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")
        
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(f"Created collection '{settings.QDRANT_COLLECTION}' ({dim}-dim, Cosine).")
        
        # Walk directory
        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                if filename.endswith('.txt'):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, base_dir)
                    process_legal_file(file_path, relative_path)
        
        logfire.info("Legal corpus ingestion completed.")


if __name__ == "__main__":
    import sys
    wipe = "--wipe" in sys.argv
    ingest_legal_corpus(wipe=wipe)
