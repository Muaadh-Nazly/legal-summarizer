"""
Hybrid Clause Ranking System

Combines multiple signals to rank clauses for summarization:
- Graph centrality
- Rhetorical roles
- Citation importance
- Embedding similarity
- Support/opposition degree
- Positional signals
"""

import json
import re
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

from graph_centrality import compute_all_centrality_features
from rhetorical_scoring import compute_all_rhetorical_features
from citation_importance import (
    compute_all_citation_features,
)
from embedding_importance import compute_all_embedding_features


# Default weights for combining signals
DEFAULT_WEIGHTS = {
    "pagerank": 0.25,  # Graph centrality
    "betweenness": 0.15,  # Bridge importance
    "role": 0.20,  # Rhetorical role
    "citation": 0.15,  # Precedent importance
    "embedding": 0.10,  # Semantic centrality
    "support_degree": 0.10,  # Support network
    "position": 0.05,  # Document position
    "case_type": 0.00,  # Case type (optional)
}


def compute_position_score(nodes: List[Dict]) -> Dict[int, float]:
    """
    Compute position-based importance score.
    Legal judgments often place key reasoning + the final holding toward the end.
    So we bias toward later clauses and mildly down-weight the opening background.

    Args:
        nodes: List of node dicts

    Returns:
        Dict mapping clause_id -> position score
    """
    position_scores = {}
    total_nodes = len(nodes)

    for idx, node in enumerate(nodes):
        node_id = node["id"]
        # Normalized position in [0, 1].
        if total_nodes <= 1:
            position = 0.5
        else:
            position = idx / (total_nodes - 1)

        # Piecewise scoring:
        # - first 20%: background / procedural history (lower)
        # - middle: reasoning (medium)
        # - last 15%: conclusions / disposition (high)
        if position < 0.20:
            score = 0.35
        elif position < 0.60:
            score = 0.55
        elif position < 0.85:
            score = 0.70
        else:
            # Ramp 0.95 -> 1.0 in the final segment
            tail = (position - 0.85) / 0.15  # 0..1
            score = 0.95 + 0.05 * max(0.0, min(1.0, tail))

        position_scores[node_id] = score

    return position_scores


def compute_clause_length_score(nodes: List[Dict]) -> Dict[int, float]:
    """
    Compute length-based score (penalize very short or very long clauses).

    Args:
        nodes: List of node dicts with 'text'

    Returns:
        Dict mapping clause_id -> length score
    """
    length_scores = {}

    for node in nodes:
        node_id = node["id"]
        text = node.get("text", "")
        word_count = len(text.split())

        # Optimal length
        if word_count < 10:
            # Too short - penalty
            length_scores[node_id] = 0.5
        elif word_count > 100:
            # Too long - slight penalty
            length_scores[node_id] = 0.8
        else:
            # Good length
            length_scores[node_id] = 1.0

    return length_scores


# Quality refinements: down-rank statutory fragments and procedure-heavy clauses
def _is_likely_statutory_fragment(text: str) -> bool:
    """
    Heuristic: clause looks like a statutory sub-paragraph
    """
    if not text or len(text.strip()) < 20:
        return False
    t = text.strip()[:120]
    # Opens with parenthesized letter/number
    if t.startswith("(") and ")" in t[:15]:
        return True
    # Starts with (i) or (ii) or (a) or (b)
    if re.match(r"^\s*\([i]+\)", t, re.IGNORECASE) or re.match(
        r"^\s*\([a-d]\)", t, re.IGNORECASE
    ):
        return True
    return False


def _has_procedure_keywords(text: str) -> bool:
    """True if clause looks procedure-heavy"""
    if not text:
        return False
    t = text.lower()
    keywords = (
        "section",
        "shall",
        "lodge",
        "notice of appeal",
        "form and manner",
        "procedure",
        "subsection",
        "hereinbefore",
    )
    return sum(1 for k in keywords if k in t) >= 2


def apply_semantic_deduplication_penalty(
    scores: Dict[int, float],
    nodes: List[Dict],
    embeddings: Optional[torch.Tensor],
    similarity_threshold: float = 0.92,
    penalty_multiplier: float = 0.93,
) -> Dict[int, float]:
    """
    Slightly reduce score of clauses that are very similar to a higher-scoring clause

    Args:
        scores: Dict mapping clause_id -> score
        nodes: List of node dicts
        embeddings: Optional HGT embeddings tensor [num_clauses, embedding_dim]
        similarity_threshold: Threshold for similarity
        penalty_multiplier: Multiplier for penalty

    Returns:
        Dict mapping clause_id -> score
    """
    if embeddings is None or not nodes or not scores:
        return dict(scores)
    node_ids = [n["id"] for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Embeddings: shape [num_nodes, dim]
    if embeddings.dim() == 3:
        embeddings = embeddings.squeeze(0)
    emb = embeddings.detach().cpu().numpy()
    if emb.shape[0] != len(node_ids):
        return dict(scores)
    # Sort by score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    out = dict(scores)

    for i, cid in enumerate(sorted_ids):
        idx = id_to_idx.get(cid)
        if idx is None:
            continue
        vec = emb[idx]
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            continue
        # Check similarity to any higher-scoring clause
        for j in range(i):
            other_id = sorted_ids[j]
            if out[other_id] <= 0:
                continue
            oidx = id_to_idx.get(other_id)
            if oidx is None:
                continue
            other_vec = emb[oidx]
            onorm = np.linalg.norm(other_vec)
            if onorm < 1e-9:
                continue
            sim = float(np.dot(vec, other_vec) / (norm * onorm))
            if sim >= similarity_threshold:
                out[cid] = out[cid] * penalty_multiplier
                break
    return out


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Normalize scores to [0, 1] range using min-max normalization.
    Args:
        scores: Dict mapping clause_id -> score

    Returns:
        Normalized scores
    """
    if not scores:
        return scores

    min_score = min(scores.values())
    max_score = max(scores.values())

    if max_score == min_score:
        return {k: 0.5 for k in scores.keys()}

    return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}


def compute_hybrid_ranking_score(
    nodes: List[Dict],
    edges: List[Dict],
    embeddings: Optional[torch.Tensor] = None,
    precedent_importance: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    selected_clauses: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Compute hybrid ranking score combining all signals.

    Args:
        nodes: List of node dicts
        edges: List of edge dicts
        embeddings: Optional HGT embeddings tensor [num_clauses, embedding_dim]
        precedent_importance: Optional dict of document_id -> importance
        weights: Optional custom weights (defaults to DEFAULT_WEIGHTS)
        selected_clauses: Optional list of already-selected clause IDs

    Returns:
        Dict mapping clause_id -> final ranking score
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # 1. Graph centrality features
    centrality_features = compute_all_centrality_features(nodes, edges)
    pagerank_scores = {
        node_id: feat["pagerank"] for node_id, feat in centrality_features.items()
    }

    # 2. Rhetorical role features
    rhetorical_features = compute_all_rhetorical_features(nodes, edges, pagerank_scores)

    # 3. Citation features
    citation_features = compute_all_citation_features(
        nodes, edges, precedent_importance
    )

    # 4. Embedding features
    embedding_features = {}
    if embeddings is not None:
        node_ids = [node["id"] for node in nodes]
        embedding_features = compute_all_embedding_features(
            embeddings, node_ids, selected_clauses
        )

    # 5. Position features
    position_scores = compute_position_score(nodes)
    length_scores = compute_clause_length_score(nodes)

    # Combine all signals
    final_scores = {}

    for node in nodes:
        node_id = node["id"]
        score = 0.0

        # Graph centrality
        if node_id in centrality_features:
            cf = centrality_features[node_id]
            score += weights["pagerank"] * cf["pagerank"]
            score += weights["betweenness"] * cf["betweenness"]
            score += weights["support_degree"] * (cf["support_in"] / 10.0)

        # Rhetorical role
        if node_id in rhetorical_features:
            rf = rhetorical_features[node_id]
            score += weights["role"] * rf["role_boosted"]

        # Citation
        if node_id in citation_features:
            citf = citation_features[node_id]
            score += weights["citation"] * citf["citation_importance"]

        # Embedding
        if node_id in embedding_features:
            ef = embedding_features[node_id]
            score += weights["embedding"] * ef["centroid_similarity"]

        # Position
        score += weights["position"] * position_scores.get(node_id, 0.5)

        # Length
        length_mult = length_scores.get(node_id, 1.0)
        score *= 0.9 + 0.1 * length_mult

        # Down-rank statutory fragments
        text = (node.get("text") or "").strip()
        if _is_likely_statutory_fragment(text):
            score *= 0.92

        final_scores[node_id] = score

    # Slightly down-rank clauses that are mostly procedure keywords
    procedure_heavy_count = sum(
        1 for n in nodes if _has_procedure_keywords((n.get("text") or "").strip())
    )
    if procedure_heavy_count > 10:
        for node in nodes:
            node_id = node["id"]
            if _has_procedure_keywords((node.get("text") or "").strip()):
                final_scores[node_id] = final_scores.get(node_id, 0) * 0.96

    # Normalize final scores
    final_scores = normalize_scores(final_scores)

    return final_scores


def rank_clauses(
    graph_json_path: str,
    embeddings: Optional[torch.Tensor] = None,
    precedent_importance: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[int, float, Dict]]:
    """
    Rank clauses in a graph for summarization.

    Args:
        graph_json_path: Path to graph JSON file
        embeddings: Optional HGT embeddings
        precedent_importance: Optional precedent importance scores
        weights: Optional custom weights
        top_k: Optional number of top clauses to return

    Returns:
        List of tuples (clause_id, score, clause_info) sorted by score (descending)
    """
    # Load graph
    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # Compute ranking scores
    scores = compute_hybrid_ranking_score(
        nodes, edges, embeddings, precedent_importance, weights
    )

    # Slight down-rank for clauses very similar to higher-scoring ones
    scores = apply_semantic_deduplication_penalty(scores, nodes, embeddings)

    # Create clause info dict
    clause_info = {node["id"]: node for node in nodes}

    # Sort by score
    ranked = sorted(
        [(node_id, scores[node_id], clause_info[node_id]) for node_id in scores.keys()],
        key=lambda x: x[1],
        reverse=True,
    )

    # Return top_k if specified
    if top_k is not None:
        ranked = ranked[:top_k]

    return ranked


def rank_clauses_with_diversity(
    graph_json_path: str,
    embeddings: Optional[torch.Tensor] = None,
    precedent_importance: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 10,
    diversity_weight: float = 0.3,
) -> List[Tuple[int, float, Dict]]:
    """
    Rank clauses with diversity constraint

    Uses Maximal Marginal Relevance (MMR) approach.
    Args:
        graph_json_path: Path to graph JSON file
        embeddings: HGT embeddings
        precedent_importance: Optional precedent importance
        weights: Optional custom weights
        top_k: Number of clauses to select
        diversity_weight: Weight for diversity (0 = no diversity, 1 = only diversity)

    Returns:
        List of selected clauses with scores
    """
    if embeddings is None:
        # Fallback to regular ranking if embeddings are not provided
        return rank_clauses(graph_json_path, None, precedent_importance, weights, top_k)

    # Load graph
    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = [node["id"] for node in nodes]

    # Compute initial ranking scores
    scores = compute_hybrid_ranking_score(
        nodes, edges, embeddings, precedent_importance, weights
    )

    # Slight down-rank for clauses very similar to higher-scoring ones
    scores = apply_semantic_deduplication_penalty(scores, nodes, embeddings)

    # Get embedding features for diversity
    embedding_features = compute_all_embedding_features(embeddings, node_ids)

    # MMR selection
    selected = []
    remaining = node_ids.copy()

    for _ in range(min(top_k, len(remaining))):
        best_score = -float("inf")
        best_clause_id = None

        for clause_id in remaining:
            relevance = scores[clause_id]

            # Diversity score
            if selected:
                diversity = embedding_features[clause_id]["diversity"]
            else:
                diversity = 1.0

            # MMR score
            mmr_score = (
                1 - diversity_weight
            ) * relevance + diversity_weight * diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best_clause_id = clause_id

        if best_clause_id is not None:
            selected.append(best_clause_id)
            remaining.remove(best_clause_id)

            # Update diversity scores for remaining clauses
            if selected:
                embedding_features = compute_all_embedding_features(
                    embeddings, remaining, selected
                )

    # Return selected clauses with their scores
    clause_info = {node["id"]: node for node in nodes}
    result = [
        (clause_id, scores[clause_id], clause_info[clause_id]) for clause_id in selected
    ]

    return result
