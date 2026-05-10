"""
embedder.py
-----------
Embeds legal judgment chunks and stores them in ChromaDB.

Legal-specific additions over the generic embedder:
  - Stores is_held flag as metadata (for retrieval boosting)
  - Stores court name (for conflict detection — finding disagreements
    between different High Courts on the same legal point)
  - Stores acts cited (for Act-level filtering)
  - Uses 'multi-qa-MiniLM-L6-cos-v1' instead of 'all-MiniLM-L6-v2'
    because it was trained on QA pairs — better for question → judgment retrieval
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Optional

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"   # QA-optimised, still only 80MB
DB_PATH    = "./legal_chroma_db"
COLLECTION = "indian_judgments"

_model: Optional[SentenceTransformer] = None
_client = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[embedder] Model ready.")
    return _model


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_and_store(chunks: list[dict]) -> int:
    """
    Embeds and stores legal judgment chunks.

    Metadata stored per chunk (enables filtered retrieval):
        judgment_id, case_title, court, date, url,
        chunk_type, section_name, is_held,
        acts (comma-separated), citations (comma-separated)
    """

    if not chunks:
        return 0

    model      = _get_model()
    collection = _get_collection()

    texts      = [c["text"] for c in chunks]
    print(f"[embedder] Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    ids       = [c["chunk_id"] for c in chunks]
    documents = texts
    metadatas = []

    for c in chunks:
        metadatas.append({
            "judgment_id":  c["judgment_id"],
            "case_title":   c["case_title"][:200],
            "court":        c["court"],
            "date":         c["date"],
            "url":          c["url"],
            "chunk_type":   c["chunk_type"],
            "section_name": c.get("section_name", ""),
            # ChromaDB only supports str/int/float metadata
            "is_held":      str(c.get("is_held", False)),
            "acts":         ", ".join(c.get("acts", [])),
            "citations":    ", ".join(c.get("citations", [])),
            "petitioner":   c.get("petitioner", "")[:100],
            "respondent":   c.get("respondent", "")[:100],
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=documents,
    )

    count = collection.count()
    print(f"[embedder] Done. Total chunks in DB: {count}")
    return len(chunks)


def get_stats() -> dict:
    """Stats about what's in the legal vector store."""
    collection = _get_collection()
    total = collection.count()

    if total == 0:
        return {"total_chunks": 0, "total_judgments": 0, "courts": []}

    results   = collection.get(include=["metadatas"])
    metas     = results["metadatas"]
    judgments = {m["judgment_id"] for m in metas}
    courts    = list({m["court"] for m in metas if m.get("court")})

    return {
        "total_chunks":    total,
        "total_judgments": len(judgments),
        "courts":          courts,
    }


def judgment_exists(judgment_id: str) -> bool:
    """Check if a judgment is already in the store."""
    collection = _get_collection()
    try:
        results = collection.get(
            where={"judgment_id": judgment_id},
            limit=1,
            include=["metadatas"],
        )
        return len(results["ids"]) > 0
    except Exception:
        return False


if __name__ == "__main__":
    print("Stats:", get_stats())