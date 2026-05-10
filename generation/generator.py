"""
generator.py
------------
Generates grounded legal answers using a local or hosted LLM.

TWO MODES — automatically selected:
  LOCAL (dev):  Ollama running Mistral 7B on your machine (no API key)
  HOSTED (HF):  HuggingFace Inference API with Zephyr-7B-beta (free tier)

WHY strict citation grounding?
  In legal AI, an uncited claim is worthless — lawyers need to verify everything.
  Our prompt FORCES the model to:
    1. Only state things supported by the retrieved chunks
    2. Cite EVERY claim with the case name and citation
    3. Explicitly state if the retrieved cases don't answer the question
    4. Flag if the law may have changed (date awareness)

PROMPT DESIGN:
  We use a structured prompt that separates:
    - The legal question
    - The retrieved context (labeled by case + court)
    - Strict instructions on citation format
  This is called "grounded generation" — the model is a summariser,
  not a knowledge source. It can only say what the cases say.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# HuggingFace Inference API (free, no key for public models)
HF_API_URL  = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN    = os.environ.get("HF_TOKEN", "")   # optional, increases rate limit


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate a grounded legal answer from retrieved chunks.

    Args:
        query:            the user's legal question
        retrieved_chunks: output of retriever.retrieve()

    Returns dict with:
        answer          - the grounded answer text with inline citations
        cited_cases     - list of cases actually cited in the answer
        confidence      - "high" | "medium" | "low" based on retrieval quality
        caveat          - standard legal disclaimer
        raw_context     - the context passed to the LLM (useful for debugging)
    """

    if not retrieved_chunks:
        return {
            "answer":      "No relevant judgments found in the database for this query.",
            "cited_cases": [],
            "confidence":  "low",
            "caveat":      _standard_caveat(),
            "raw_context": "",
        }

    # ── Build context string ───────────────────────────────────────────────
    context = _build_context(retrieved_chunks)

    # ── Build prompt ───────────────────────────────────────────────────────
    prompt = _build_prompt(query, context)

    # ── Call LLM ──────────────────────────────────────────────────────────
    llm_response = _call_llm(prompt)

    # ── Parse response ─────────────────────────────────────────────────────
    cited_cases = _extract_cited_cases(llm_response, retrieved_chunks)
    confidence  = _assess_confidence(retrieved_chunks, cited_cases)

    return {
        "answer":      llm_response,
        "cited_cases": cited_cases,
        "confidence":  confidence,
        "caveat":      _standard_caveat(),
        "raw_context": context[:500] + "..." if len(context) > 500 else context,
    }


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a labeled context block.
    Each chunk is labeled with case name + court so the model can cite properly.
    """

    parts = []
    seen_judgments = set()

    for i, chunk in enumerate(chunks):
        jid   = chunk["judgment_id"]
        label = f"[SOURCE {i+1}]"

        # Add case header once per judgment
        if jid not in seen_judgments:
            seen_judgments.add(jid)
            header = (
                f"{label} {chunk['case_title']}\n"
                f"Court: {chunk['court']} | Date: {chunk['date']}\n"
            )
            if chunk.get("citations"):
                header += f"Citation: {chunk['citations'][0]}\n"
        else:
            header = f"{label} (continued from {chunk['case_title']})\n"

        text_block = chunk["text"][:600]   # cap at 600 chars per chunk

        parts.append(f"{header}{text_block}")

    return "\n\n---\n\n".join(parts)


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(query: str, context: str) -> str:
    return f"""You are a precise Indian legal research assistant. Your job is to answer the legal question below using ONLY the provided court judgments.

STRICT RULES:
1. Cite every legal claim using the format: [Case Name, Court, Year]
2. If the provided cases do not clearly answer the question, say so explicitly
3. Do NOT use your general knowledge — only use what is in the provided sources
4. Note if courts have disagreed on this point
5. Keep the answer structured: first state the general legal position, then exceptions, then practical implications
6. Maximum 4 paragraphs

LEGAL QUESTION:
{query}

RETRIEVED COURT JUDGMENTS:
{context}

ANSWER (cite every claim):"""


# ── LLM callers ───────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Try Ollama first, fall back to HuggingFace Inference API."""

    # Try Ollama (local)
    response = _call_ollama(prompt)
    if response:
        return response

    # Try HuggingFace Inference API
    print("[generator] Ollama not available, trying HuggingFace API...")
    response = _call_huggingface(prompt)
    if response:
        return response

    # Last resort: template-based answer from context
    print("[generator] LLM unavailable — using template answer")
    return _template_answer(prompt)


def _call_ollama(prompt: str) -> Optional[str]:
    """Call local Ollama instance."""
    try:
        payload = json.dumps({
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # low temperature = more factual
                "num_predict": 600,
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()

    except Exception as e:
        print(f"[generator] Ollama error: {e}")
        return None


def _call_huggingface(prompt: str) -> Optional[str]:
    """Call HuggingFace Inference API (free tier)."""
    try:
        headers = {"Content-Type": "application/json"}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        payload = json.dumps({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens":  600,
                "temperature":     0.1,
                "return_full_text": False,
                "stop": ["QUESTION:", "LEGAL QUESTION:"],
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            HF_API_URL,
            data=payload,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

            # HF returns a list of generated texts
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
            elif isinstance(data, dict):
                return data.get("generated_text", "").strip()

    except urllib.error.HTTPError as e:
        if e.code == 503:
            print("[generator] HuggingFace model loading (cold start) — try again in 30s")
        else:
            print(f"[generator] HuggingFace API error: {e.code}")
    except Exception as e:
        print(f"[generator] HuggingFace error: {e}")

    return None


def _template_answer(prompt: str) -> str:
    """
    Fallback when no LLM is available.
    Generates a structured answer directly from the context.
    Used in demo mode or when HF cold-starts.
    """
    # Extract context section from prompt
    context_start = prompt.find("RETRIEVED COURT JUDGMENTS:") + len("RETRIEVED COURT JUDGMENTS:")
    context_end   = prompt.find("ANSWER (cite every claim):")
    context       = prompt[context_start:context_end].strip()

    # Pull out source labels and first sentence of each
    import re
    sources = re.findall(r'\[SOURCE \d+\](.*?)(?=\[SOURCE|\Z)', context, re.DOTALL)

    answer_parts = [
        "Based on the retrieved Indian court judgments, the following legal position emerges:\n"
    ]

    for i, source in enumerate(sources[:3], 1):
        lines = [l.strip() for l in source.strip().split("\n") if l.strip()]
        if lines:
            case_line  = lines[0]
            court_line = lines[1] if len(lines) > 1 else ""
            content    = " ".join(lines[2:])[:300] if len(lines) > 2 else ""
            answer_parts.append(f"[{case_line}] — {content}")

    answer_parts.append(
        "\nNote: LLM generation unavailable. Please install Ollama with Mistral 7B "
        "for full natural language answers. Retrieved cases shown above are relevant to your query."
    )

    return "\n\n".join(answer_parts)


# ── Post-processing ───────────────────────────────────────────────────────────

def _extract_cited_cases(answer: str, chunks: list[dict]) -> list[dict]:
    """Find which retrieved cases are actually mentioned in the answer."""
    cited = []
    seen  = set()

    for chunk in chunks:
        title = chunk["case_title"]
        jid   = chunk["judgment_id"]

        if jid in seen:
            continue

        # Check if case name or citation appears in answer
        # Use the most distinctive part of the case name (first party)
        first_party = title.split(" v.")[0].split("v.")[0].strip()

        if first_party.lower() in answer.lower() or any(
            cite.lower() in answer.lower() for cite in chunk.get("citations", [])
        ):
            cited.append({
                "case_title":  title,
                "court":       chunk["court"],
                "date":        chunk["date"],
                "url":         chunk["url"],
                "citations":   chunk.get("citations", []),
            })
            seen.add(jid)

    return cited


def _assess_confidence(chunks: list[dict], cited_cases: list[dict]) -> str:
    """Assess confidence based on retrieval quality."""
    if not chunks:
        return "low"

    avg_score   = sum(c.get("score", 0) for c in chunks) / len(chunks)
    has_held    = any(c.get("is_held") for c in chunks)
    cited_count = len(cited_cases)

    if avg_score > 0.7 and has_held and cited_count >= 2:
        return "high"
    elif avg_score > 0.4 or cited_count >= 1:
        return "medium"
    else:
        return "low"


def _standard_caveat() -> str:
    return (
        "⚠️ This is an AI-generated legal research summary for informational purposes only. "
        "It does not constitute legal advice. Always verify citations independently and "
        "consult a qualified advocate before relying on this information in legal proceedings."
    )


if __name__ == "__main__":
    # Test with mock chunks
    mock_chunks = [
        {
            "judgment_id": "SC_2021_BAIL_001",
            "case_title":  "Satender Kumar Antil v. CBI",
            "court":       "Supreme Court of India",
            "date":        "2021-07-11",
            "url":         "https://indiankanoon.org/doc/168768371/",
            "text":        "Bail is the rule and jail is the exception. The Supreme Court established four categories for bail.",
            "is_held":     True,
            "acts":        ["CrPC 437", "CrPC 439"],
            "citations":   ["(2022) 10 SCC 51"],
            "score":       0.85,
        }
    ]

    result = generate_answer("What are the bail conditions under CrPC?", mock_chunks)
    print(f"Confidence: {result['confidence']}")
    print(f"Answer:\n{result['answer']}")
    print(f"Cited: {[c['case_title'] for c in result['cited_cases']]}")