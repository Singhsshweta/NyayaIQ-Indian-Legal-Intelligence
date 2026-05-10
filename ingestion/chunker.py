"""
chunker.py
----------
Legal-aware chunking for Indian court judgments.

WHY legal chunking is different from generic chunking:
  - A judgment has a clear structure: Facts → Issues → Arguments → Held
  - The "HELD" section is the most important — it's what gets cited
  - Citations in a chunk must stay WITH the claim they support
    (splitting "The court held X. See AIR 1980 SC 898." breaks retrieval)
  - Case names are proper nouns that must never be split mid-name

OUR STRATEGY:
  1. Abstract chunk  → case metadata summary (always chunk 0)
  2. Section chunks  → split on section headers (HELD, FACTS, etc.)
  3. Paragraph chunks → within sections, split by paragraph with
                        overlap to preserve citation context
  4. Every chunk gets: court, date, acts cited, case name
     so retrieval results are self-contained for citation generation
"""

import re
from typing import Optional

# Target chunk size in words
CHUNK_SIZE_WORDS   = 250
OVERLAP_WORDS      = 50   # words of overlap between consecutive chunks

# Section headers in Indian judgments — these are chunk boundaries
SECTION_HEADERS = [
    "JUDGMENT", "ORDER", "HELD", "HEADNOTES", "HEADNOTE",
    "FACTS", "BACKGROUND", "ISSUES", "ARGUMENTS", "SUBMISSIONS",
    "REASONING", "ANALYSIS", "CONCLUSION", "DIRECTIONS",
    "PER CURIAM", "DISSENT", "CONCURRING OPINION",
]

SECTION_RE = re.compile(
    r'^\s*(' + '|'.join(re.escape(h) for h in SECTION_HEADERS) + r')[:\s]*$',
    re.MULTILINE | re.IGNORECASE
)


def chunk_judgment(judgment: dict) -> list[dict]:
    """
    Takes a judgment dict and returns a list of chunk dicts.

    Each chunk dict:
        chunk_id      - unique ID e.g. "SC_2021_BAIL_001_held_0"
        judgment_id   - source judgment ID
        case_title    - full case name
        court         - court name
        date          - judgment date
        url           - source URL
        acts          - list of Acts cited in this chunk
        citations     - list of case citations in this chunk
        chunk_type    - "summary" | "section" | "body"
        section_name  - which section this came from (e.g. "HELD")
        text          - the actual text to embed
        is_held       - bool, True if this is from the HELD/ORDER section
                        (these chunks get retrieval priority boost)
    """

    chunks = []

    base_meta = {
        "judgment_id": judgment["id"],
        "case_title":  judgment["title"],
        "court":       judgment["court"],
        "date":        judgment["date"],
        "url":         judgment["url"],
        "petitioner":  judgment.get("petitioner", ""),
        "respondent":  judgment.get("respondent", ""),
        "all_acts":    judgment.get("acts", []),
    }

    # ── 1. Summary chunk (always present) ───────────────────────────────────
    # Combines case name + court + summary into one searchable chunk
    summary_text = _build_summary_text(judgment)
    if summary_text:
        chunks.append({
            **base_meta,
            "chunk_id":    f"{judgment['id']}_summary",
            "chunk_type":  "summary",
            "section_name": "summary",
            "text":        summary_text,
            "is_held":     False,
            "acts":        judgment.get("acts", []),
            "citations":   judgment.get("citations", []),
        })

    # ── 2. Full text chunks ─────────────────────────────────────────────────
    full_text = judgment.get("full_text", "")
    if not full_text:
        return chunks  # abstract-only mode (arXiv style)

    # Split into sections first
    sections = _split_into_sections(full_text)

    for section_name, section_text in sections:
        if not section_text.strip():
            continue

        is_held = section_name.upper() in {"HELD", "ORDER", "JUDGMENT", "CONCLUSION", "DIRECTIONS"}

        # Split each section into paragraph-based chunks
        para_chunks = _split_section(section_text, target_words=CHUNK_SIZE_WORDS)

        for i, chunk_text in enumerate(para_chunks):
            if len(chunk_text.split()) < 20:
                continue  # skip very short fragments

            chunk_citations = _extract_inline_citations(chunk_text)
            chunk_acts      = _extract_inline_acts(chunk_text)

            chunks.append({
                **base_meta,
                "chunk_id":    f"{judgment['id']}_{_slugify(section_name)}_{i}",
                "chunk_type":  "section",
                "section_name": section_name,
                "text":        chunk_text.strip(),
                "is_held":     is_held,
                "acts":        chunk_acts,
                "citations":   chunk_citations,
            })

    return chunks


def _build_summary_text(judgment: dict) -> str:
    """Build a rich summary chunk that captures case identity."""
    parts = [
        f"Case: {judgment['title']}",
        f"Court: {judgment['court']}",
        f"Date: {judgment['date']}",
    ]

    if judgment.get("petitioner"):
        parts.append(f"Petitioner: {judgment['petitioner']}")
    if judgment.get("respondent"):
        parts.append(f"Respondent: {judgment['respondent']}")
    if judgment.get("acts"):
        parts.append(f"Acts cited: {', '.join(judgment['acts'])}")
    if judgment.get("citations"):
        parts.append(f"Citations: {', '.join(judgment['citations'][:3])}")
    if judgment.get("summary"):
        parts.append(f"\nSummary: {judgment['summary']}")

    return "\n".join(parts)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split judgment text into (section_name, section_text) pairs.
    Falls back to treating the whole text as one "body" section.
    """

    # Find all section header positions
    matches = list(SECTION_RE.finditer(text))

    if not matches:
        # No section headers found — treat as one block
        return [("body", text)]

    sections = []

    # Text before first header → preamble
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append(("preamble", preamble))

    # Each section = from this header to next header
    for i, match in enumerate(matches):
        section_name = match.group(1).strip().lower()
        start        = match.end()
        end          = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append((section_name, section_text))

    return sections


def _split_section(text: str, target_words: int = CHUNK_SIZE_WORDS) -> list[str]:
    """
    Split a section into overlapping chunks, preferring paragraph boundaries.
    """

    # First split by paragraphs (double newline or numbered paragraph)
    paragraphs = re.split(r'\n{2,}|\n(?=\d+\.\s)', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    chunks   = []
    current  = []
    cur_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If adding this paragraph would exceed target, flush current chunk
        if cur_words + para_words > target_words and current:
            chunks.append(" ".join(current))

            # Overlap: keep last ~OVERLAP_WORDS words for context continuity
            overlap_text = " ".join(current)
            overlap_words = overlap_text.split()[-OVERLAP_WORDS:]
            current  = [" ".join(overlap_words)]
            cur_words = len(overlap_words)

        current.append(para)
        cur_words += para_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def _extract_inline_citations(text: str) -> list[str]:
    patterns = [
        r'\(\d{4}\)\s+\d+\s+SCC\s+\d+',
        r'AIR\s+\d{4}\s+SC\s+\d+',
        r'\d{4}\s+SCC\s+OnLine\s+\w+\s+\d+',
    ]
    hits = []
    for p in patterns:
        hits.extend(re.findall(p, text))
    return list(set(hits))


def _extract_inline_acts(text: str) -> list[str]:
    patterns = [
        r'(?:Section|S\.)\s*\d+[A-Z]?\s+(?:of\s+)?(?:the\s+)?(?:IPC|CrPC|CPC|NDPS|NI Act)',
        r'(?:IPC|CrPC|NDPS Act|NI Act)\s+(?:Section\s+)?\d+[A-Z]?',
        r'Article\s+\d+[A-Z]?(?:\(\d+\))?',
    ]
    hits = []
    for p in patterns:
        hits.extend(re.findall(p, text))
    return list(set(h.strip() for h in hits))[:8]


def _slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


if __name__ == "__main__":
    # Test with a sample judgment
    sample = {
        "id":          "TEST_001",
        "title":       "Test v. State",
        "court":       "Supreme Court of India",
        "date":        "2023-01-01",
        "url":         "https://example.com",
        "petitioner":  "Test",
        "respondent":  "State",
        "acts":        ["IPC 302", "CrPC 439"],
        "citations":   ["(2020) 5 SCC 100"],
        "summary":     "Test case summary for chunker validation.",
        "full_text":   (
            "FACTS\n\nThe accused was charged under IPC 302. "
            "The trial court convicted the accused.\n\n"
            "HELD\n\nThe court held that the conviction is maintained. "
            "Bail is denied. See AIR 1980 SC 898 for the rarest of rare doctrine."
        ),
    }

    chunks = chunk_judgment(sample)
    for c in chunks:
        print(f"\n[{c['chunk_type']}:{c['section_name']}] {c['chunk_id']}")
        print(f"  is_held: {c['is_held']}")
        print(f"  acts: {c['acts']}")
        print(f"  text: {c['text'][:150]}...")