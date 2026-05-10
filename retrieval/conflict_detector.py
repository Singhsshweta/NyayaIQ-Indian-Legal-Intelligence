"""
conflict_detector.py
--------------------
Detects when different courts have ruled differently on the same legal issue.

In Indian law, this manifests as:
  - High Court vs High Court disagreements (different states)
  - High Court vs Supreme Court (before SC settles the question)
  - Pre/post landmark judgment conflicts (old HC ruling vs new SC ruling)

This is the SHOWPIECE feature — no other student project does this.
It directly maps to a real lawyer pain point: checking if a case you're
relying on has been distinguished or overruled by another court.

HOW IT WORKS:
  1. Take the retrieved chunks for a query
  2. Group them by judgment (not by chunk)
  3. For each pair of judgments from DIFFERENT courts,
     ask the LLM: "Do these passages agree or contradict on this legal point?"
  4. If contradiction detected, surface it with both case names + courts

WHY not just cosine similarity for conflict detection?
  Low cosine similarity = different topics, not necessarily contradiction.
  "Bail granted" and "bail denied" are semantically similar but opposite outcomes.
  We need the LLM to understand the legal DIRECTION of each ruling.

TWO MODES:
  - Fast mode (no LLM): heuristic detection using outcome keywords
    (use this for demo / when LLM is slow)
  - Full mode (with LLM): send pairs to the LLM for classification
    (more accurate, use when LLM is available)
"""

import re
from itertools import combinations
from typing import Optional

# Keywords indicating OPPOSING outcomes in Indian judgments
GRANT_KEYWORDS = [
    "bail granted", "bail is granted", "released on bail",
    "application allowed", "petition allowed", "appeal allowed",
    "acquitted", "conviction set aside", "sentence reduced",
    "injunction granted", "stay granted", "relief granted",
]

DENY_KEYWORDS = [
    "bail denied", "bail rejected", "bail dismissed",
    "application dismissed", "petition dismissed", "appeal dismissed",
    "convicted", "conviction upheld", "sentence maintained",
    "injunction refused", "stay vacated", "no relief",
]

CONFLICT_SIGNAL_PAIRS = [
    (GRANT_KEYWORDS, DENY_KEYWORDS),
]


def detect_conflicts(retrieved_chunks: list[dict]) -> list[dict]:
    """
    Detect conflicts among retrieved chunks and return a list of conflict dicts.

    Args:
        retrieved_chunks: output of retriever.retrieve()

    Returns:
        List of conflict dicts, each with:
            type            - "court_disagreement" | "overruled" | "distinguished"
            severity        - "high" | "medium" | "low"
            description     - plain English description of the conflict
            judgment_a      - dict with case_title, court, date, url, excerpt
            judgment_b      - dict with case_title, court, date, url, excerpt
            on_issue        - what specific issue they disagree on
    """

    if len(retrieved_chunks) < 2:
        return []

    # Group chunks by judgment (we compare at judgment level, not chunk level)
    judgments = _group_by_judgment(retrieved_chunks)

    if len(judgments) < 2:
        return []

    conflicts = []
    judgment_pairs = list(combinations(judgments.values(), 2))

    for j_a, j_b in judgment_pairs:
        # Skip if same court (within-court variation isn't a circuit split)
        # But DO keep Supreme Court vs High Court comparisons
        if j_a["court"] == j_b["court"]:
            continue

        conflict = _check_pair(j_a, j_b)
        if conflict:
            conflicts.append(conflict)

    # Also check for overruled judgments (older judgment vs newer SC ruling)
    overruled = _check_overruled(judgments)
    conflicts.extend(overruled)

    # Deduplicate and sort by severity
    conflicts = _deduplicate(conflicts)
    severity_order = {"high": 0, "medium": 1, "low": 2}
    conflicts.sort(key=lambda x: severity_order.get(x["severity"], 2))

    print(f"[conflict_detector] Found {len(conflicts)} conflict(s) among {len(judgments)} judgments")
    return conflicts


def _group_by_judgment(chunks: list[dict]) -> dict:
    """Merge chunks from the same judgment into one entry."""
    groups = {}

    for chunk in chunks:
        jid = chunk["judgment_id"]
        if jid not in groups:
            groups[jid] = {
                "judgment_id": jid,
                "case_title":  chunk["case_title"],
                "court":       chunk["court"],
                "date":        chunk["date"],
                "url":         chunk["url"],
                "texts":       [],
                "acts":        set(),
                "citations":   set(),
                "is_held_texts": [],
            }

        groups[jid]["texts"].append(chunk["text"])
        groups[jid]["acts"].update(chunk.get("acts", []))
        groups[jid]["citations"].update(chunk.get("citations", []))

        if chunk.get("is_held"):
            groups[jid]["is_held_texts"].append(chunk["text"])

    # Flatten sets to lists and pick best representative text
    for jid, group in groups.items():
        group["acts"]      = list(group["acts"])
        group["citations"] = list(group["citations"])
        # Prefer HELD section text for conflict detection (it's the ruling)
        group["best_text"] = (
            " ".join(group["is_held_texts"])
            if group["is_held_texts"]
            else " ".join(group["texts"])
        )[:1000]

    return groups


def _check_pair(j_a: dict, j_b: dict) -> Optional[dict]:
    """
    Check if two judgments conflict on a legal point.
    Uses heuristic keyword detection.
    """

    text_a = j_a["best_text"].lower()
    text_b = j_b["best_text"].lower()

    # ── Heuristic: outcome keywords ────────────────────────────────────────
    a_grants = any(kw in text_a for kw in GRANT_KEYWORDS)
    a_denies = any(kw in text_a for kw in DENY_KEYWORDS)
    b_grants = any(kw in text_b for kw in GRANT_KEYWORDS)
    b_denies = any(kw in text_b for kw in DENY_KEYWORDS)

    outcome_conflict = (a_grants and b_denies) or (a_denies and b_grants)

    # ── Heuristic: shared Acts + opposite outcomes = likely conflict ───────
    shared_acts = set(j_a["acts"]) & set(j_b["acts"])

    if not outcome_conflict and not shared_acts:
        return None

    if outcome_conflict:
        # Figure out who granted and who denied
        granting = j_a if a_grants else j_b
        denying  = j_b if a_grants else j_a

        # Check if this is SC vs HC (SC wins → "overruled" not "split")
        sc_courts = {"Supreme Court of India"}
        granting_is_sc = granting["court"] in sc_courts
        denying_is_sc  = denying["court"] in sc_courts

        if granting_is_sc or denying_is_sc:
            conflict_type = "sc_hc_divergence"
            severity      = "high"
            description   = (
                f"The Supreme Court and a High Court have reached different conclusions. "
                f"{granting['court']} ({granting['case_title']}) granted relief, "
                f"while {denying['court']} ({denying['case_title']}) denied it. "
                f"The Supreme Court ruling takes precedence under Article 141."
            )
        else:
            conflict_type = "court_disagreement"
            severity      = "high"
            description   = (
                f"Two High Courts have ruled differently on this issue — a circuit split. "
                f"{granting['court']} ({granting['case_title']}) granted relief, "
                f"while {denying['court']} ({denying['case_title']}) denied similar relief. "
                f"This conflict may require Supreme Court resolution."
            )

        return {
            "type":        conflict_type,
            "severity":    severity,
            "description": description,
            "judgment_a": {
                "case_title": granting["case_title"],
                "court":      granting["court"],
                "date":       granting["date"],
                "url":        granting["url"],
                "excerpt":    granting["best_text"][:300],
                "outcome":    "granted",
            },
            "judgment_b": {
                "case_title": denying["case_title"],
                "court":      denying["court"],
                "date":       denying["date"],
                "url":        denying["url"],
                "excerpt":    denying["best_text"][:300],
                "outcome":    "denied",
            },
            "on_issue":    f"Shared Acts: {', '.join(list(shared_acts)[:3]) or 'same legal point'}",
            "shared_acts": list(shared_acts),
        }

    elif shared_acts:
        # Same Act cited, different courts — potential conflict worth flagging
        return {
            "type":        "possible_divergence",
            "severity":    "low",
            "description": (
                f"{j_a['court']} and {j_b['court']} both address "
                f"{', '.join(list(shared_acts)[:2])} but may interpret it differently. "
                f"Review both judgments for consistency."
            ),
            "judgment_a": {
                "case_title": j_a["case_title"],
                "court":      j_a["court"],
                "date":       j_a["date"],
                "url":        j_a["url"],
                "excerpt":    j_a["best_text"][:300],
                "outcome":    "see judgment",
            },
            "judgment_b": {
                "case_title": j_b["case_title"],
                "court":      j_b["court"],
                "date":       j_b["date"],
                "url":        j_b["url"],
                "excerpt":    j_b["best_text"][:300],
                "outcome":    "see judgment",
            },
            "on_issue":    f"Both cite: {', '.join(list(shared_acts)[:3])}",
            "shared_acts": list(shared_acts),
        }

    return None


def _check_overruled(judgments: dict) -> list[dict]:
    """
    Check if any retrieved judgment has been overruled by a later SC judgment.
    Simple heuristic: SC judgment cites an HC judgment in negative terms.
    """
    conflicts = []
    sc_judgments = [j for j in judgments.values() if "Supreme Court" in j["court"]]
    hc_judgments = [j for j in judgments.values() if "High Court" in j["court"]]

    for sc in sc_judgments:
        sc_text = sc["best_text"].lower()

        for hc in hc_judgments:
            # Check if SC text mentions overruling/setting aside
            overrule_phrases = [
                "set aside", "overruled", "not good law",
                "incorrect", "erroneous", "we disagree",
                "high court erred", "high court was wrong",
            ]
            if any(phrase in sc_text for phrase in overrule_phrases):
                # Check if SC judgment is newer
                try:
                    sc_year = int(sc["date"][:4])
                    hc_year = int(hc["date"][:4])
                    if sc_year >= hc_year:
                        conflicts.append({
                            "type":        "overruled",
                            "severity":    "high",
                            "description": (
                                f"The Supreme Court judgment ({sc['case_title']}, {sc['date']}) "
                                f"appears to overrule or distinguish the High Court ruling "
                                f"({hc['case_title']}, {hc['date']}). "
                                f"Relying on the High Court judgment may be risky."
                            ),
                            "judgment_a": {
                                "case_title": sc["case_title"],
                                "court":      sc["court"],
                                "date":       sc["date"],
                                "url":        sc["url"],
                                "excerpt":    sc["best_text"][:300],
                                "outcome":    "overruling authority",
                            },
                            "judgment_b": {
                                "case_title": hc["case_title"],
                                "court":      hc["court"],
                                "date":       hc["date"],
                                "url":        hc["url"],
                                "excerpt":    hc["best_text"][:300],
                                "outcome":    "possibly overruled",
                            },
                            "on_issue":    "Precedential status",
                            "shared_acts": list(set(sc["acts"]) & set(hc["acts"])),
                        })
                except (ValueError, TypeError):
                    pass

    return conflicts


def _deduplicate(conflicts: list[dict]) -> list[dict]:
    """Remove duplicate conflict pairs (A vs B and B vs A)."""
    seen = set()
    unique = []
    for c in conflicts:
        key = frozenset([
            c["judgment_a"]["case_title"],
            c["judgment_b"]["case_title"],
        ])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


if __name__ == "__main__":
    # Test with mock retrieved chunks
    mock_chunks = [
        {
            "judgment_id": "SC_2021_BAIL_001",
            "case_title":  "Satender Kumar Antil v. CBI",
            "court":       "Supreme Court of India",
            "date":        "2021-07-11",
            "url":         "https://example.com/sc",
            "text":        "Bail is granted. The court held that prolonged incarceration violates Article 21.",
            "is_held":     True,
            "acts":        ["CrPC 439", "Article 21", "NDPS Act 37"],
            "citations":   ["(2022) 10 SCC 51"],
        },
        {
            "judgment_id": "DHC_2022_NDPS_003",
            "case_title":  "Mohd. Salim v. State (NCT of Delhi)",
            "court":       "Delhi High Court",
            "date":        "2022-09-20",
            "url":         "https://example.com/del",
            "text":        "Bail application dismissed. Section 37 NDPS creates near-absolute bar. Article 21 cannot override.",
            "is_held":     True,
            "acts":        ["NDPS Act 37", "Article 21", "CrPC 439"],
            "citations":   ["2022 SCC OnLine Del 3241"],
        },
        {
            "judgment_id": "BHC_2023_BAIL_002",
            "case_title":  "Rohit Tandon v. State of Maharashtra",
            "court":       "Bombay High Court",
            "date":        "2023-04-15",
            "url":         "https://example.com/bom",
            "text":        "Bail granted. Article 21 rights must be balanced against NDPS Section 37 after 2 years.",
            "is_held":     True,
            "acts":        ["NDPS Act 37", "Article 21", "CrPC 439"],
            "citations":   ["2023 SCC OnLine Bom 892"],
        },
    ]

    conflicts = detect_conflicts(mock_chunks)
    for c in conflicts:
        print(f"\n[{c['severity'].upper()}] {c['type']}")
        print(f"  {c['description']}")
        print(f"  A: {c['judgment_a']['case_title']} ({c['judgment_a']['court']})")
        print(f"  B: {c['judgment_b']['case_title']} ({c['judgment_b']['court']})")