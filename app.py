"""
app.py
------
Streamlit UI for NyayaIQ — Indian Case Law Intelligence Engine.

IMPORTANT ARCHITECTURE NOTE:
  This file calls Python modules DIRECTLY — it does NOT go through the FastAPI server.
  Why? HF Spaces runs one process. We can't have both uvicorn + streamlit running.

  The FastAPI server (api/main.py) is for external API consumers (other apps).
  This Streamlit app is the demo UI — same logic, different entry point.

DEPLOYMENT:
  Local:     streamlit run app.py
  HF Spaces: set main file to app.py in Space settings
"""

import sys
import os
from pathlib import Path

import streamlit as st
import time

# Make sure project root is importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="NyayaIQ — Indian Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports with user-friendly error handling ───────────────────────────
# We import inside functions so Streamlit can render the page even if
# a heavy dependency (torch, chromadb) takes time to load.

@st.cache_resource(show_spinner="Loading AI models (first run only)...")
def load_retriever():
    from retrieval.retriever import retrieve
    return retrieve

@st.cache_resource(show_spinner="Loading generator...")
def load_generator():
    from generation.generator import generate_answer
    return generate_answer

@st.cache_resource
def load_conflict_detector():
    from retrieval.conflict_detector import detect_conflicts
    return detect_conflicts

@st.cache_resource
def load_ingestion():
    from ingestion.ecourts_fetcher import fetch_judgments
    from ingestion.chunker import chunk_judgment
    from ingestion.embedder import embed_and_store, get_stats, judgment_exists
    return fetch_judgments, chunk_judgment, embed_and_store, get_stats, judgment_exists

# ── Helper: run full ingest pipeline ─────────────────────────────────────────

def run_ingest(topic: str, max_results: int = 15) -> dict:
    fetch_judgments, chunk_judgment, embed_and_store, get_stats, judgment_exists = load_ingestion()

    judgments = fetch_judgments(topic, max_results=max_results)
    if not judgments:
        return {"error": "No judgments found for this topic"}

    new_count    = 0
    total_chunks = 0
    for j in judgments:
        if judgment_exists(j["id"]):
            continue
        chunks = chunk_judgment(j)
        if chunks:
            embed_and_store(chunks)
            total_chunks += len(chunks)
            new_count    += 1

    return {
        "found":   len(judgments),
        "new":     new_count,
        "chunks":  total_chunks,
    }


# ── UI ────────────────────────────────────────────────────────────────────────

def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("⚖️ NyayaIQ")
    st.caption("Indian Case Law Intelligence Engine · RAG-powered · Grounded citations · Conflict detection")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📥 Index Judgments")
        st.markdown("*Index a topic first, then ask questions about it.*")

        ingest_topic = st.text_input(
            "Topic to index:",
            placeholder="e.g. bail conditions NDPS Act",
            key="ingest_topic",
        )
        ingest_n = st.slider("Max judgments to fetch", 5, 20, 12, key="ingest_n")

        if st.button("🔄 Index Topic", type="primary", use_container_width=True):
            if not ingest_topic.strip():
                st.warning("Enter a topic first")
            else:
                with st.spinner(f"Fetching and indexing '{ingest_topic}'..."):
                    result = run_ingest(ingest_topic, max_results=ingest_n)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(
                        f"✅ Indexed {result['new']} new judgments "
                        f"({result['chunks']} chunks) from {result['found']} found"
                    )

        st.divider()

        # DB stats
        st.header("📊 Index Stats")
        try:
            _, _, _, get_stats, _ = load_ingestion()
            stats = get_stats()
            st.metric("Judgments indexed", stats.get("total_judgments", 0))
            st.metric("Chunks in DB",      stats.get("total_chunks", 0))
            courts = stats.get("courts", [])
            if courts:
                st.markdown("**Courts:**")
                for c in courts:
                    st.markdown(f"  · {c}")
        except Exception:
            st.info("Index stats unavailable until first indexing run")

        st.divider()

        # PDF upload
        st.header("📄 Upload Judgment PDF")
        uploaded = st.file_uploader("Upload a judgment PDF", type=["pdf"])
        if uploaded and st.button("⬆️ Process PDF", use_container_width=True):
            from ingestion.pdf_ingester import ingest_pdf
            from ingestion.chunker import chunk_judgment
            from ingestion.embedder import embed_and_store, judgment_exists

            temp = f"/tmp/{uploaded.name}"
            with open(temp, "wb") as f:
                f.write(uploaded.read())

            with st.spinner("Parsing and indexing PDF..."):
                judgment = ingest_pdf(temp)
                os.remove(temp)

            if judgment:
                if not judgment_exists(judgment["id"]):
                    chunks = chunk_judgment(judgment)
                    embed_and_store(chunks)
                st.success(f"✅ Indexed: {judgment['title']}")
            else:
                st.error("Failed to parse PDF")

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_ask, tab_search, tab_about = st.tabs(["💬 Ask", "🔍 Browse", "ℹ️ About"])

    # ── ASK tab ───────────────────────────────────────────────────────────────
    with tab_ask:
        st.header("Ask a Legal Question")
        st.markdown(
            "Get a grounded answer with **case citations** from indexed Indian court judgments. "
            "Index a topic in the sidebar first."
        )

        question = st.text_area(
            "Your legal question:",
            placeholder="e.g. What are the twin conditions for bail under NDPS Act Section 37?",
            height=90,
            key="question",
        )

        col_btn, col_filter = st.columns([2, 3])
        with col_btn:
            ask = st.button("🤖 Get Answer", type="primary", use_container_width=True)
        with col_filter:
            court_filter = st.selectbox(
                "Filter by court (optional):",
                ["All courts", "Supreme Court of India", "Delhi High Court",
                 "Bombay High Court", "Madras High Court", "Allahabad High Court"],
                key="court_filter",
            )

        if ask:
            if not question.strip():
                st.warning("Please enter a question")
            else:
                retrieve      = load_retriever()
                generate      = load_generator()
                detect        = load_conflict_detector()

                filter_val = None if court_filter == "All courts" else court_filter

                with st.spinner("Searching judgments and generating answer..."):
                    t0      = time.time()
                    chunks  = retrieve(question, top_k=8, court_filter=filter_val)
                    elapsed = time.time() - t0

                if not chunks:
                    st.error(
                        "No relevant judgments found. "
                        "Please index a related topic first using the sidebar."
                    )
                else:
                    with st.spinner("Generating grounded answer..."):
                        result = generate(question, chunks)

                    # ── Answer ────────────────────────────────────────────────
                    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                        result["confidence"], "⚪"
                    )
                    st.subheader(f"{conf_icon} Answer — confidence: {result['confidence']}")
                    st.write(result["answer"])

                    # ── Citations ─────────────────────────────────────────────
                    if result["cited_cases"]:
                        st.subheader("📚 Cited Cases")
                        for case in result["cited_cases"]:
                            with st.expander(f"📖 {case['case_title']}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Court:** {case['court']}")
                                    st.write(f"**Date:** {case['date']}")
                                with col2:
                                    if case.get("citations"):
                                        st.write(f"**Citation:** {case['citations'][0]}")
                                if case.get("url"):
                                    st.markdown(f"[🔗 View judgment]({case['url']})")

                    # ── Conflicts ─────────────────────────────────────────────
                    courts_in_result = {c["court"] for c in chunks}
                    if len(courts_in_result) > 1:
                        conflicts = detect(chunks)
                        if conflicts:
                            st.subheader("⚠️ Court Conflicts Detected")
                            for conflict in conflicts:
                                severity_style = {
                                    "high":   st.error,
                                    "medium": st.warning,
                                    "low":    st.info,
                                }.get(conflict["severity"], st.info)

                                severity_style(
                                    f"**[{conflict['type'].upper().replace('_', ' ')}]** "
                                    f"{conflict['description']}"
                                )

                                col1, col2 = st.columns(2)
                                ja = conflict["judgment_a"]
                                jb = conflict["judgment_b"]

                                with col1:
                                    st.markdown(
                                        f"**{ja['case_title']}**  \n"
                                        f"{ja['court']} · {ja['date']}  \n"
                                        f"Outcome: `{ja['outcome']}`"
                                    )
                                with col2:
                                    st.markdown(
                                        f"**{jb['case_title']}**  \n"
                                        f"{jb['court']} · {jb['date']}  \n"
                                        f"Outcome: `{jb['outcome']}`"
                                    )

                    # ── Caveat ────────────────────────────────────────────────
                    st.info(result["caveat"])

                    # ── Debug ─────────────────────────────────────────────────
                    with st.expander(f"🔍 Debug — retrieved {len(chunks)} chunks in {elapsed:.1f}s"):
                        for i, c in enumerate(chunks, 1):
                            st.markdown(
                                f"**{i}. {c['case_title']}** ({c['court']})  "
                                f"score: `{c['score']:.3f}` · is_held: `{c['is_held']}`"
                            )
                            st.caption(c["text"][:250])

    # ── BROWSE tab ────────────────────────────────────────────────────────────
    with tab_search:
        st.header("Browse Judgments")

        browse_query = st.text_input(
            "Search topic:",
            placeholder="e.g. cheque dishonour NI Act",
            key="browse_query",
        )

        if st.button("🔎 Search", key="browse_btn"):
            if not browse_query.strip():
                st.warning("Enter a topic")
            else:
                fetch_judgments, *_ = load_ingestion()
                with st.spinner("Searching..."):
                    judgments = fetch_judgments(browse_query, max_results=15)

                if not judgments:
                    st.error("No results found")
                else:
                    st.write(f"Found **{len(judgments)}** judgments:")
                    for i, j in enumerate(judgments, 1):
                        with st.expander(f"{i}. {j['title']}"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**Court:** {j['court']}  |  **Date:** {j['date']}")
                                if j.get("petitioner"):
                                    st.write(f"**Petitioner:** {j['petitioner']}")
                                if j.get("respondent"):
                                    st.write(f"**Respondent:** {j['respondent']}")
                                st.write(f"**Summary:** {j['summary']}")
                            with col2:
                                if j.get("acts"):
                                    st.markdown("**Acts cited:**")
                                    for act in j["acts"][:6]:
                                        st.markdown(f"  · `{act}`")
                                if j.get("citations"):
                                    st.markdown("**Citations:**")
                                    st.caption(j["citations"][0] if j["citations"] else "")
                            if j.get("url"):
                                st.markdown(f"[🔗 View source]({j['url']})")

    # ── ABOUT tab ─────────────────────────────────────────────────────────────
    with tab_about:
        st.header("About NyayaIQ")
        st.markdown("""
**NyayaIQ** is an AI-powered Indian legal research engine built to demonstrate
production-grade RAG (Retrieval-Augmented Generation) over real court data.

---

### How it works

1. **Index** — fetches judgments from Indian Kanoon / eCourts APIs, chunks them
   with legal-structure awareness (HELD section gets priority), embeds with
   `sentence-transformers`, stores in ChromaDB on disk.

2. **Retrieve** — hybrid search: semantic vector similarity + BM25 keyword
   re-ranking. HELD/ORDER chunks get a 1.3× score boost since that's what
   lawyers actually cite.

3. **Generate** — retrieval context is passed to a local Mistral 7B (via Ollama)
   or Zephyr-7B-beta (HuggingFace free tier). Strict grounded-generation prompt
   forces citation of every claim.

4. **Conflict detection** — compares outcome keywords across judgments from
   different courts. Surfaces High Court vs High Court splits and potential
   SC overrulings automatically.

---

### Tech stack (100% free, no API keys)

| Component | Tool |
|---|---|
| Embeddings | `multi-qa-MiniLM-L6-cos-v1` (sentence-transformers) |
| Vector store | ChromaDB (local, persistent) |
| LLM (local) | Mistral 7B via Ollama |
| LLM (hosted) | Zephyr-7B-beta via HF Inference API |
| Data | Indian Kanoon · eCourts India · curated offline dataset |
| UI | Streamlit |
| API | FastAPI + Uvicorn |

---

### Resume talking points
- Hybrid BM25 + semantic retrieval with domain-specific re-ranking
- Legal-structure-aware chunking (section boundary detection, HELD boost)
- Conflict detection across courts (circuit-split equivalent for Indian law)
- PDF ingestion pipeline with metadata extraction from judgment headers
- Grounded generation with strict citation enforcement
- Dual deployment: FastAPI REST API + Streamlit demo

---

⚠️ *For research purposes only. Not legal advice.*
        """)


if __name__ == "__main__":
    main()