"""
main.py
--------
FastAPI wrapper for the Indian Case Law Intelligence Engine (NyayaIQ).

DEPLOYMENT:
  Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000
  Docs at:  http://localhost:8000/docs
"""

import sys
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ingestion.ecourts_fetcher import fetch_judgments
from ingestion.chunker import chunk_judgment
from ingestion.embedder import embed_and_store, get_stats, judgment_exists
from ingestion.pdf_ingester import ingest_pdf
from retrieval.retriever import retrieve
from retrieval.conflict_detector import detect_conflicts
from generation.generator import generate_answer

app = FastAPI(
    title="NyayaIQ — Indian Case Law Intelligence Engine",
    description="RAG-powered legal research engine over Indian court data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    court_filter: Optional[str] = None

class IngestRequest(BaseModel):
    query: str
    max_results: Optional[int] = 15

class AskRequest(BaseModel):
    question: str
    max_context: Optional[int] = 8

class AskResponse(BaseModel):
    answer: str
    cited_cases: List[Dict[str, Any]]
    confidence: str
    caveat: str
    conflicts: Optional[List[Dict[str, Any]]] = None
    raw_context: str

class HealthResponse(BaseModel):
    status: str
    version: str
    db_stats: Dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest_judgments(request: IngestRequest):
    """
    Fetch judgments from Indian Kanoon / offline dataset and embed them
    into ChromaDB. Call this once per topic before asking questions.
    """
    try:
        judgments = fetch_judgments(request.query, request.max_results or 15)
        if not judgments:
            raise HTTPException(status_code=404, detail="No judgments found for this query")

        new_count = 0
        total_chunks = 0

        for j in judgments:
            if judgment_exists(j["id"]):
                continue  # already indexed — skip
            chunks = chunk_judgment(j)
            if chunks:
                embed_and_store(chunks)
                total_chunks += len(chunks)
                new_count += 1

        return {
            "judgments_found":   len(judgments),
            "new_judgments":     new_count,
            "chunks_embedded":   total_chunks,
            "message":           f"Indexed {new_count} new judgments ({total_chunks} chunks)",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/search")
async def search_judgments(request: SearchRequest):
    """Search for relevant judgments (returns raw judgment metadata, no embedding needed)."""
    try:
        judgments = fetch_judgments(request.query, request.max_results or 10)
        if request.court_filter:
            judgments = [j for j in judgments
                         if request.court_filter.lower() in j["court"].lower()]
        return judgments
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_legal_question(request: AskRequest):
    """
    Full RAG pipeline: retrieve → generate → detect conflicts.
    Make sure you've called /ingest first so the vector store has data.
    """
    try:
        # 1. Retrieve
        retrieved_chunks = retrieve(request.question, top_k=request.max_context or 8)

        if not retrieved_chunks:
            return AskResponse(
                answer="No relevant judgments found. Please call /ingest with a related topic first.",
                cited_cases=[],
                confidence="low",
                caveat=_caveat(),
                conflicts=None,
                raw_context="",
            )

        # 2. Generate grounded answer
        answer_result = generate_answer(request.question, retrieved_chunks)

        # 3. Detect conflicts — FIXED: correct function signature (no mode param)
        conflicts = None
        courts = {c["court"] for c in retrieved_chunks}
        if len(courts) > 1:
            conflicts = detect_conflicts(retrieved_chunks)

        return AskResponse(
            answer=answer_result["answer"],
            cited_cases=answer_result["cited_cases"],
            confidence=answer_result["confidence"],
            caveat=answer_result["caveat"],
            conflicts=conflicts or [],
            raw_context=answer_result["raw_context"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.post("/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    """Upload a judgment PDF, ingest it into ChromaDB."""
    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        judgment = ingest_pdf(temp_path)
        os.remove(temp_path)

        if not judgment:
            raise HTTPException(status_code=400, detail="Failed to parse PDF")

        # Embed immediately
        if not judgment_exists(judgment["id"]):
            chunks = chunk_judgment(judgment)
            embed_and_store(chunks)

        return {
            "status":  "success",
            "id":      judgment["id"],
            "title":   judgment["title"],
            "court":   judgment["court"],
            "date":    judgment["date"],
            "chunks":  len(chunk_judgment(judgment)),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — also shows DB stats."""
    try:
        stats = get_stats()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            db_stats=stats,
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            db_stats={"error": str(e)},
        )


def _caveat() -> str:
    return (
        "⚠️ AI-generated legal research summary for informational purposes only. "
        "Does not constitute legal advice. Verify all citations independently and "
        "consult a qualified advocate before relying on this in legal proceedings."
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)