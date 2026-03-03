"""
Post-processing utilities for clause selection.

- Pure score-based top-k can miss the disposition/holding, which is critical for legal summaries.
- Ranked lists are not in narrative order; for summarization usually want to re-order the selected clauses in original document sequence.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Sequence, Tuple

# Common decision / disposition cues
STRONG_DISPOSITION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bfor the foregoing reasons\b", re.IGNORECASE),
    re.compile(
        r"\b(for these reasons|for these grounds|for the above reasons|for the aforementioned reasons)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(i proceed to (?:dismiss|allow|grant))\b", re.IGNORECASE),
    # Appeal/application disposition
    re.compile(
        r"\b(?:the )?appeal(?:s)? (?:of [^.]*? )?(?:is|are|must be|should|should therefore stand) (?:allowed|dismissed|rejected)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bappeal(?:s)? (?:allowed|dismissed|rejected)\b", re.IGNORECASE),
    re.compile(
        r"\bappeal(?:s)? (?:is|are) (?:hereby|accordingly) (?:dismissed|allowed|rejected)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(application|petition|revision application|motion) (?:is|are|must be) (?:allowed|dismissed|rejected)\b",
        re.IGNORECASE,
    ),
    # "Application dismissed"
    re.compile(
        r"\b(application|petition|motion) (?:dismissed|allowed|rejected)\b",
        re.IGNORECASE,
    ),
    # "The application is, accordingly dismissed"
    re.compile(
        r"\b(?:the )?(?:application|petition|motion) is,? (?:accordingly|therefore|hereby) (?:dismissed|allowed|rejected)\b",
        re.IGNORECASE,
    ),
    # Order-making phrases
    re.compile(
        r"\b(i (?:therefore|accordingly|hereby) (?:make|would make|hereby make) order)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(i|is|are) (?:set aside|affirmed|quashed|varied)", re.IGNORECASE),
    re.compile(
        r"\bjudgment(?:s)? (?:is|are) (?:set aside|affirmed|quashed|varied)\b",
        re.IGNORECASE,
    ),
    # "I set aside... and allow"
    re.compile(
        r"\bi set aside.*?and (?:allow|dismiss|restore)", re.IGNORECASE | re.DOTALL
    ),
    # "Stand rejected/dismissed"
    re.compile(
        r"\b(?:should|must) (?:stand|be) (?:rejected|dismissed|allowed)\b",
        re.IGNORECASE,
    ),
    # "I hold that" + disposition
    re.compile(r"\bi (?:therefore|accordingly)? hold that", re.IGNORECASE),
    re.compile(
        r"\bi hold that (?:there is no|the|this) (?:lawful )?(?:appeal|application|petition)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi (?:have no hesitation in )?(?:uphold|dismiss|allow|reject)", re.IGNORECASE
    ),
    # "I declare"
    re.compile(
        r"\bi (?:therefore|accordingly|further)? declare that.*(?:violates|illegal|null|void|infringed)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\bi (?:therefore|accordingly|further)? declare that", re.IGNORECASE),
    # "I answer" + questions of law
    re.compile(
        r"\bi (?:proceed to )?answer (?:both|the|all|three) (?:questions? of law|issues? of law)",
        re.IGNORECASE,
    ),
    # "Re-hearing ordered"
    re.compile(r"\b(?:re-hearing|rehearing) (?:ordered|directed)\b", re.IGNORECASE),
    # "I am in agreement" + final stance
    re.compile(r"\bi am (?:in )?agreement (?:with|that)", re.IGNORECASE),
    # "Order must remain unaltered"
    re.compile(
        r"\border(?:s)? (?:made|issued).*must (?:remain|stand) (?:unaltered|unchanged)",
        re.IGNORECASE,
    ),
    # "That date will stand"
    re.compile(r"\bthat (?:date|order|judgment) will stand\b", re.IGNORECASE),
    # "Accordingly, the motion... is rejected"
    re.compile(
        r"\baccordingly,? (?:the )?(?:application|petition|motion|order).*?is (?:rejected|dismissed|allowed)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # "I order no costs" or "No costs ordered"
    re.compile(r"\b(?:i )?(?:order|ordered) (?:no )?costs\b", re.IGNORECASE),
    re.compile(r"\bno costs (?:ordered|awarded)\b", re.IGNORECASE),
    # "I direct" + final orders
    re.compile(
        r"\bi (?:further )?direct (?:the|that|respondents?|registrar)", re.IGNORECASE
    ),
    # "I therefore order" + compensation/costs
    re.compile(r"\bi therefore order.*(?:pay|compensation|costs)", re.IGNORECASE),
    # "The State shall pay"
    re.compile(
        r"\b(?:the )?state (?:shall|will) pay.*(?:costs|compensation)", re.IGNORECASE
    ),
    # "is forthwith directed"
    re.compile(r"\bis forthwith directed (?:to|that)", re.IGNORECASE),
    # "is entitled to" + final relief
    re.compile(
        r"\b(?:the )?(?:petitioner|appellant|respondent) is entitled to (?:support (?:his|her|their) application for leave to proceed|proceed|receive)",
        re.IGNORECASE,
    ),
]

WEAK_DISPOSITION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b with costs\b", re.IGNORECASE),
]

# When multiple disposition clauses exist, prefer the one that states the main outcome over costs-only
DISPOSITION_PRIORITY_1_MAIN_OUTCOME: List[re.Pattern] = [
    re.compile(
        r"\b(?:the )?(?:application|petition|appeal|motion)(?:s)?\s+(?:is|are|was|were)?\s*,?\s*(?:accordingly|therefore|hereby)?\s*(?:dismissed|allowed|rejected)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bappeal(?:s)?\s+(?:allowed|dismissed|rejected)\b", re.IGNORECASE),
    re.compile(r"\bi proceed to (?:dismiss|allow|grant)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:is|are|judgment)\s+(?:set aside|affirmed|quashed|varied)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi set aside.*?and (?:allow|dismiss|restore)", re.IGNORECASE | re.DOTALL
    ),
    re.compile(
        r"\bfor the foregoing reasons.*?(?:set aside|allowed|dismissed|rejected)",
        re.IGNORECASE | re.DOTALL,
    ),
]
DISPOSITION_PRIORITY_2_ORDER_OR_DIRECT: List[re.Pattern] = [
    re.compile(r"\bi (?:therefore|accordingly)? (?:hold|declare) that", re.IGNORECASE),
    re.compile(r"\bi (?:further )?direct (?:the|that|respondents?)", re.IGNORECASE),
    re.compile(
        r"\bi (?:therefore|accordingly) order.*(?:pay|compensation)", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:the )?state (?:shall|will) pay.*(?:costs|compensation)", re.IGNORECASE
    ),
    re.compile(r"\bis forthwith directed", re.IGNORECASE),
]
# Tier 3: costs-only or procedural tail (lowest priority)
DISPOSITION_PRIORITY_3_COSTS_ONLY: List[re.Pattern] = [
    re.compile(r"\bno costs (?:ordered|awarded)\b", re.IGNORECASE),
    re.compile(r"\bi (?:make |order )?no order for costs\b", re.IGNORECASE),
    re.compile(r"\b(?:i )?(?:order|ordered) (?:no )?costs\b", re.IGNORECASE),
]


def _disposition_priority(text: str) -> int:
    """Return 1 (best), 2, or 3 (worst). Prefer main outcome over order/direct over costs-only."""
    if not (text or "").strip():
        return 3
    t = text.strip()
    for p in DISPOSITION_PRIORITY_1_MAIN_OUTCOME:
        if p.search(t):
            return 1
    for p in DISPOSITION_PRIORITY_2_ORDER_OR_DIRECT:
        if p.search(t):
            return 2
    for p in DISPOSITION_PRIORITY_3_COSTS_ONLY:
        if p.search(t):
            return 3
    return 2


def _build_id_to_position(nodes: Sequence[Dict]) -> Dict[int, int]:
    """Map node id -> its position in the nodes list (document order)."""
    id_to_pos: Dict[int, int] = {}
    for idx, n in enumerate(nodes):
        try:
            node_id = int(n["id"])
        except Exception:
            continue
        id_to_pos[node_id] = idx
    return id_to_pos


def find_disposition_clause_ids(
    nodes: Sequence[Dict],
) -> Tuple[List[int], List[int]]:
    """
    Return (strong_ids, weak_ids) for disposition/holding candidates.
    - strong_ids: clauses that very likely state the final holding/disposition
    - weak_ids: fallback cues
    """
    strong: List[int] = []
    weak: List[int] = []

    for n in nodes:
        text = (n.get("text") or "").strip()
        if not text:
            continue

        try:
            cid = int(n["id"])
        except Exception:
            continue

        if any(p.search(text) for p in STRONG_DISPOSITION_PATTERNS):
            strong.append(cid)
            continue
        if any(p.search(text) for p in WEAK_DISPOSITION_PATTERNS):
            weak.append(cid)

    return strong, weak


def ensure_disposition_in_topk(
    ranked: Sequence[Tuple[int, float, Dict]],
    nodes: Sequence[Dict],
    top_k: int,
) -> Tuple[List[Tuple[int, float, Dict]], List[int]]:
    """
    Ensure at least one disposition clause is present in the selected top_k.

    Strategy:
    - Detect disposition candidates from node texts.
    - If none of the selected top_k contain a disposition clause, inject the best
      disposition candidate from the overall ranked list.
    - Keep output length == top_k by replacing the last element if needed.
    """
    if top_k <= 0:
        return list(ranked), []

    strong_ids, weak_ids = find_disposition_clause_ids(nodes)
    strong_set = set(strong_ids)
    weak_set = set(weak_ids)

    selected = list(ranked[:top_k])
    selected_ids = {cid for cid, _, _ in selected}

    # Select the disposition clause with the highest priority
    required_set: set[int]
    if strong_set:
        required_set = strong_set
    elif weak_set:
        required_set = weak_set
    else:
        return selected, []

    if selected_ids & required_set:
        return selected, sorted(selected_ids & required_set)

    # Prefer the latest disposition clause (in doc order)
    id_to_pos = _build_id_to_position(nodes)
    best_required_id = max(
        required_set,
        key=lambda cid: id_to_pos.get(int(cid), -1),
    )

    best_disp: Optional[Tuple[int, float, Dict]] = None
    for item in ranked:
        if int(item[0]) == int(best_required_id):
            best_disp = item
            break

    # Fallback if the required clause id isn't present in ranked
    if best_disp is None:
        for item in ranked:
            if int(item[0]) in required_set:
                best_disp = item
                break

    if best_disp is None:
        return selected, []

    if len(selected) < top_k:
        selected.append(best_disp)
    else:
        selected[-1] = best_disp

    return selected, [best_disp[0]]


def ensure_disposition_in_selection(
    selected: Sequence[Tuple[int, float, Dict]],
    ranked_full: Sequence[Tuple[int, float, Dict]],
    nodes: Sequence[Dict],
    top_k: int,
) -> Tuple[List[Tuple[int, float, Dict]], List[int]]:
    """
    Ensure at least one disposition clause is present. If missing, append it as an
    additional clause beyond top_k .

    Strategy:
    - Take the top_k selected clauses as-is
    - If a disposition clause is already in the selection, return as-is
    - If missing, append the best disposition clause from the document

    Args:
        selected: Already-selected clauses
        ranked_full: Full ranked list for lookup
        nodes: All nodes from graph
        top_k: Target number of clauses (disposition is added beyond this)

    Returns:
        (final_selected_list, forced_disposition_ids)
    """
    if top_k <= 0:
        return list(selected), []

    strong_ids, weak_ids = find_disposition_clause_ids(nodes)
    strong_set = set(strong_ids)
    weak_set = set(weak_ids)

    # Take exactly top_k clauses
    out = list(selected[:top_k])
    selected_ids = {cid for cid, _, _ in out}

    # Determine which disposition clauses we need to check for
    required_set: set[int]
    if strong_set:
        required_set = strong_set
    elif weak_set:
        required_set = weak_set
    else:
        # No disposition clauses found in document
        return out, []

    if selected_ids & required_set:
        return out, sorted(selected_ids & required_set)

    # Disposition is missing - find the best one and append it
    id_to_pos = _build_id_to_position(nodes)
    id_to_text: Dict[int, str] = {}
    for n in nodes:
        try:
            id_to_text[int(n.get("id"))] = (n.get("text") or "").strip()
        except Exception:
            pass
    best_required_id = max(
        required_set,
        key=lambda cid: (
            -_disposition_priority(id_to_text.get(int(cid), "")),
            id_to_pos.get(int(cid), -1),
        ),
    )

    # Lookup best clause tuple in ranked_full
    best_tuple: Optional[Tuple[int, float, Dict]] = None
    for cid, score, info in ranked_full:
        if int(cid) == int(best_required_id):
            best_tuple = (cid, score, info)
            break

    if best_tuple is None:
        node_lookup = {int(n.get("id")): n for n in nodes if "id" in n}
        info = node_lookup.get(int(best_required_id), {"id": int(best_required_id)})
        best_tuple = (int(best_required_id), 0.0, info)

    # Append the disposition clause
    out.append(best_tuple)

    return out, [best_tuple[0]]


def order_selected_by_document(
    selected: Sequence[Tuple[int, float, Dict]],
    nodes: Sequence[Dict],
) -> List[Tuple[int, float, Dict]]:
    """Reorder selected clauses in original document order using nodes list positions."""
    id_to_pos = _build_id_to_position(nodes)
    # Unknown ids go to the end but keep stable order
    return sorted(
        list(selected),
        key=lambda t: (id_to_pos.get(int(t[0]), 10**9)),
    )
