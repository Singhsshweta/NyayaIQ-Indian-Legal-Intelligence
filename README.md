---
title: NyayaIQ - Indian Legal Intelligence
emoji: ⚖️
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
short_description: RAG-powered Indian case law research with conflict detection
---

<div align="center">

# ⚖️ NyayaIQ
### Indian Case Law Intelligence Engine

*RAG-powered legal research over real Indian court data — with grounded citations and automatic conflict detection*

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-purple)](https://huggingface.co/spaces/YOUR_HF_USERNAME/NyayaIQ)
[![GitHub](https://img.shields.io/badge/GitHub-NyayaIQ-black?logo=github)](https://github.com/Singhsshweta/NyayaIQ-Indian-Legal-Intelligence)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## What is NyayaIQ?

NyayaIQ is an AI-powered legal research engine built specifically for Indian court data. A lawyer or researcher types a legal question in plain English — the system retrieves the most relevant court judgments, generates a grounded answer where every claim is backed by a real case citation, and automatically flags when different courts have ruled differently on the same legal point.

**The problem it solves:** Legal research in India is slow and fragmented. Lawyers manually search Indian Kanoon, cross-reference multiple judgments, and often miss circuit splits between High Courts. NyayaIQ automates all of this in seconds.

---

## Demo

> **Try this question:** *"Can a court grant bail in a commercial quantity NDPS case if the accused has been in custody for over 2 years?"*

This triggers retrieval across Supreme Court and High Court judgments, generates a grounded answer, and surfaces the **Bombay HC vs Delhi HC conflict** on NDPS Section 37 automatically.

---

## Key Features

**Hybrid Retrieval**
Combines BM25 keyword search with semantic vector search. BM25 catches exact legal terms like "Section 37 NDPS" or specific CNR numbers. Semantic search handles paraphrase — "released on bail" matches "bail granted". HELD/ORDER section chunks get a 1.3× score boost since that's what lawyers actually cite.

**Grounded Generation**
Every answer is generated with a strict prompt that forces the LLM to cite every claim with a case name and citation. The model cannot use general knowledge — only what is in the retrieved judgments. Uncited claims are not allowed.

**Court Conflict Detection**
Automatically detects when different courts have ruled differently on the same legal issue. Identifies High Court vs High Court splits, Supreme Court vs High Court divergence, and potentially overruled judgments — the equivalent of a circuit split detector for Indian law.

**PDF Ingestion Pipeline**
Upload any judgment PDF directly. PyMuPDF extracts text with legal-structure awareness — removes headers/footers, preserves section boundaries (FACTS, HELD, ORDER), and extracts metadata from the judgment header automatically.

**Production REST API**
FastAPI wrapper exposes `/ingest`, `/search`, `/ask`, and `/upload-pdf` endpoints with automatic Swagger documentation. Designed for integration into legal platforms.

**Zero Cost Stack**
Runs entirely on free infrastructure — no paid APIs, no cloud GPU required.

---

## How It Works

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│         Retrieval Layer         │
│   BM25 keyword + Semantic       │
│   ChromaDB vector store         │
│   HELD section chunks → 1.3×    │
└──────────────┬──────────────────┘
               │  Top 8 chunks
               ▼
┌─────────────────────────────────┐
│       Generation Layer          │
│   Mistral 7B (Ollama local)     │
│   Zephyr-7B (HF free fallback)  │
│   Strict citation enforcement   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│      Conflict Detection         │
│   Compare judgments by court    │
│   Surface HC splits + SC        │
│   overrulings automatically     │
└──────────────┬──────────────────┘
               │
               ▼
    Grounded answer + citations
    + conflict alerts (if any)
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Embeddings | `multi-qa-MiniLM-L6-cos-v1` | QA-optimised, 80MB, runs locally |
| Vector store | ChromaDB | Local, persistent, no server needed |
| LLM (local) | Mistral 7B via Ollama | Free, no API key |
| LLM (hosted) | Zephyr-7B-beta via HF Inference API | Free tier fallback for deployment |
| Retrieval | Hybrid BM25 + semantic | Exact terms + paraphrase matching |
| PDF parsing | PyMuPDF | Best reading order for legal PDFs |
| API | FastAPI + Uvicorn | Production-ready, auto Swagger docs |
| UI | Streamlit | Demo interface |
| Data | Indian Kanoon · eCourts India | Real Indian court judgments |

---

## Data Sources

NyayaIQ pulls from three sources in order of preference:

1. **Indian Kanoon API** (free tier) — full judgment text, covers all Indian courts
2. **eCourts India public API** — case metadata, CNR numbers, court-level data
3. **Curated offline dataset** — 8 real landmark judgments embedded at startup

The offline dataset covers: bail law (Satender Kumar Antil), NDPS Section 37 (Thamisharasi, Rohit Tandon, Mohd. Salim), death penalty (Bachan Singh), contracts/force majeure (Energy Watchdog), arbitration (Cox and Kings), and cheque dishonour jurisdiction (Dashrath Rupsingh Rathod).

---

## Project Structure

```
NyayaIQ/
├── app.py                        # Streamlit UI (entry point)
├── api/
│   └── main.py                   # FastAPI REST API
├── ingestion/
│   ├── ecourts_fetcher.py        # Indian Kanoon + eCourts + offline dataset
│   ├── pdf_ingester.py           # PyMuPDF PDF parsing pipeline
│   ├── chunker.py                # Legal-structure-aware chunking
│   └── embedder.py               # Sentence-transformers + ChromaDB
├── retrieval/
│   ├── retriever.py              # Hybrid BM25 + semantic search
│   └── conflict_detector.py     # Court disagreement detection
├── generation/
│   └── generator.py              # Grounded LLM answer generation
└── requirements.txt
```

---

## Local Setup

```bash
# Clone
git clone https://github.com/Singhsshweta/NyayaIQ-Indian-Legal-Intelligence.git
cd NyayaIQ-Indian-Legal-Intelligence

# Install
pip install -r requirements.txt

# Optional: local LLM via Ollama (https://ollama.ai)
ollama pull mistral

# Run
streamlit run app.py
```

App opens at `http://localhost:8501`

To run the API:
```bash
uvicorn api.main:app --reload
# Swagger docs at http://localhost:8000/docs
```

---

## Sample Questions

- *"What are the twin conditions for bail under NDPS Act Section 37?"*
- *"When does frustration of contract apply under Section 56 Indian Contract Act?"*
- *"What is the rarest of rare doctrine for death penalty cases?"*
- *"Which court has jurisdiction for cheque dishonour cases under NI Act?"*
- *"Can a non-signatory be bound by an arbitration agreement in India?"*

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Fetch and embed judgments for a topic |
| `POST` | `/search` | Search judgments by topic |
| `POST` | `/ask` | Full RAG — question to grounded answer |
| `POST` | `/upload-pdf` | Upload and index a judgment PDF |
| `GET` | `/health` | Health check + DB stats |

Full docs at `/docs` when running locally.

---

## Design Decisions

**Why hybrid retrieval?**
Legal text has exact statutory references that must match precisely — "Section 37 NDPS Act", "Article 21". Pure semantic search retrieves thematically similar but legally wrong sections. BM25 handles exact terms; semantic handles paraphrase. Combined they outperform either alone.

**Why boost HELD section chunks?**
The HELD or ORDER section is the ratio decidendi — what lawyers actually cite. A 1.3× score boost for these chunks reflects domain knowledge about what matters in a judgment.

**Why legal-structure-aware chunking?**
Generic fixed-size chunking separates citations from the claims they support. Our chunker keeps citations with their claims and uses section headers (FACTS, HELD, ORDER) as natural chunk boundaries.

**Why heuristic conflict detection?**
LLM-based conflict detection is more accurate but adds 10-15 seconds of latency. Keyword heuristics catch the most important conflicts — bail granted vs denied, relief allowed vs dismissed — instantly and reliably.

---

## Limitations

- Offline dataset covers 8 cases — index more topics via the sidebar for broader coverage
- Indian Kanoon free tier: 500 API calls/month
- HuggingFace Inference API cold-starts in ~30 seconds on first query
- Conflict detection uses heuristics — may miss subtle doctrinal disagreements

---

## Roadmap

- [ ] Indian Kanoon API token support for higher rate limits
- [ ] Retrieval evaluation metrics (precision@k, answer grounding score)
- [ ] Multi-hop retrieval for complex legal questions
- [ ] Citation graph — visualise how cases cite each other
- [ ] District court data via eCourts bulk download

---

## Legal Disclaimer

NyayaIQ is an AI research tool for informational purposes only. It does not constitute legal advice. All outputs must be independently verified. Consult a qualified advocate before relying on any information in legal proceedings.

---

<div align="center">

Built with Python · sentence-transformers · ChromaDB · FastAPI · Streamlit

*Star the repo if this was useful ⭐*

</div>
