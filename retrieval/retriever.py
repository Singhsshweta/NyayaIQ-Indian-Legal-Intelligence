"""
retriever.py
------------
Hybrid retrieval: ChromaDB semantic search + BM25 re-ranking.

When ChromaDB has data (after /ingest), uses full semantic + BM25 hybrid.
Falls back to BM25-only over the offline dataset if DB is empty.
"""

import re
import math
from collections import Counter
from typing import Optional

# ── Constants (defined ONCE) ──────────────────────────────────────────────────
TOP_K      = 8
BM25_K1    = 1.5   # term frequency saturation
BM25_B     = 0.75  # length normalisation
HELD_BOOST = 1.3   # score multiplier for HELD/ORDER section chunks
SEMANTIC_POOL = 20


def retrieve(query: str, top_k: int = TOP_K, court_filter: str = None) -> list[dict]:
    """
    Retrieve the most relevant judgment chunks for a legal query.

    Tries ChromaDB semantic search first; falls back to BM25 over
    the offline curated dataset if the vector store is empty.
    """

    candidates = _semantic_retrieve(query, court_filter)

    if not candidates:
        print("[retriever] ChromaDB empty — falling back to offline BM25 retrieval")
        candidates = _offline_retrieve(query, court_filter)

    if not candidates:
        return []

    candidates = _bm25_rerank(query, candidates)

    for c in candidates:
        if c.get("is_held"):
            c["score"] *= HELD_BOOST

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # At most 2 chunks per judgment so one long case doesn't dominate
    seen: Counter = Counter()
    final = []
    for c in candidates:
        jid = c["judgment_id"]
        if seen[jid] < 2:
            final.append(c)
            seen[jid] += 1
        if len(final) >= top_k:
            break

    print(f"[retriever] Returning {len(final)} chunks for: '{query}'")
    return final


# ── Semantic retrieval (ChromaDB) ─────────────────────────────────────────────

def _semantic_retrieve(query: str, court_filter: Optional[str]) -> list[dict]:
    try:
        from ingestion.embedder import _get_model, _get_collection
        model      = _get_model()
        collection = _get_collection()

        if collection.count() == 0:
            return []

        query_embedding = model.encode([query])[0].tolist()
        where_filter    = {"court": court_filter} if court_filter else None

        n = min(SEMANTIC_POOL, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        candidates = []
        for i, chunk_id in enumerate(results["ids"][0]):
            doc            = results["documents"][0][i]
            meta           = results["metadatas"][0][i]
            distance       = results["distances"][0][i]
            semantic_score = 1 - (distance / 2)

            candidates.append({
                "chunk_id":      chunk_id,
                "text":          doc,
                "case_title":    meta.get("case_title", ""),
                "court":         meta.get("court", ""),
                "date":          meta.get("date", ""),
                "url":           meta.get("url", ""),
                "judgment_id":   meta.get("judgment_id", ""),
                "section_name":  meta.get("section_name", ""),
                "is_held":       meta.get("is_held", "False") == "True",
                "acts":          [a.strip() for a in meta.get("acts", "").split(",") if a.strip()],
                "citations":     [c.strip() for c in meta.get("citations", "").split(",") if c.strip()],
                "petitioner":    meta.get("petitioner", ""),
                "respondent":    meta.get("respondent", ""),
                "semantic_score": semantic_score,
                "score":         semantic_score,
            })

        return candidates

    except Exception as e:
        print(f"[retriever] Semantic retrieval error: {e}")
        return []


# ── Offline BM25 fallback ─────────────────────────────────────────────────────

def _offline_retrieve(query: str, court_filter: Optional[str]) -> list[dict]:
    from ingestion.ecourts_fetcher import fetch_judgments

    judgments = fetch_judgments(query, max_results=20)
    if court_filter:
        judgments = [j for j in judgments
                     if court_filter.lower() in j.get("court", "").lower()]

    candidates = []
    for j in judgments:
        text = j.get("full_text") or j.get("summary") or ""
        if not text:
            continue
        candidates.append({
            "chunk_id":      f"{j.get('id', 'unk')}_summary",
            "judgment_id":   j.get("id", "unk"),
            "case_title":    j.get("title", "Unknown"),
            "court":         j.get("court", "Unknown Court"),
            "date":          j.get("date", ""),
            "url":           j.get("url", ""),
            "section_name":  "summary",
            "is_held":       False,
            "acts":          j.get("acts", []),
            "citations":     j.get("citations", []),
            "petitioner":    j.get("petitioner", ""),
            "respondent":    j.get("respondent", ""),
            "text":          text[:1200],
            "semantic_score": 0.0,
            "score":         0.0,
        })

    return candidates


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _bm25_rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return candidates

    query_terms = _tokenize(query)
    corpus      = [_tokenize(c["text"]) for c in candidates]
    avg_dl      = sum(len(d) for d in corpus) / len(corpus)

    df = Counter()
    for doc in corpus:
        for term in set(doc):
            df[term] += 1

    N            = len(candidates)
    bm25_scores  = []

    for doc_tokens in corpus:
        tf    = Counter(doc_tokens)
        dl    = len(doc_tokens)
        score = 0.0

        for term in query_terms:
            if term not in tf:
                continue
            idf     = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf_norm = (tf[term] * (BM25_K1 + 1)) / (
                tf[term] + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl)
            )
            score += idf * tf_norm

        bm25_scores.append(score)

    max_bm25  = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_norm = [s / max_bm25 for s in bm25_scores]

    for i, c in enumerate(candidates):
        c["bm25_score"] = bm25_norm[i]
        c["score"]      = 0.6 * c["semantic_score"] + 0.4 * bm25_norm[i]

    return candidates


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [t for t in text.split() if len(t) > 2]


if __name__ == "__main__":
    results = retrieve("bail conditions under NDPS Act", top_k=3)
    for r in results:
        print(f"\n{'='*60}")
        print(f"Case:  {r['case_title']}")
        print(f"Court: {r['court']}  |  is_held: {r['is_held']}")
        print(f"Score: {r['score']:.3f}")
        print(f"Text:  {r['text'][:200]}...")