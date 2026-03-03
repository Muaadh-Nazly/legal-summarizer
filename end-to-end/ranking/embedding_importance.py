"""
HGT Embedding-Based Importance Scoring

Computes importance based on:
- Document centroid similarity (clauses similar to document theme)
- Embedding diversity (penalize redundant clauses)
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity


def compute_document_centroid(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute document centroid (mean of all clause embeddings).
    Args:
        embeddings: Tensor of shape [num_clauses, embedding_dim]

    Returns:
        Centroid vector of shape [embedding_dim]
    """
    return embeddings.mean(dim=0)


def compute_centroid_similarity(
    embeddings: torch.Tensor, node_ids: List[int]
) -> Dict[int, float]:
    """
    Compute cosine similarity of each clause to document centroid.
    Args:
        embeddings: Tensor of shape [num_clauses, embedding_dim]
        node_ids: List of clause IDs corresponding to embeddings

    Returns:
        Dict mapping clause_id -> similarity score [0, 1]
    """
    if len(embeddings) == 0:
        return {node_id: 0.0 for node_id in node_ids}

    # Compute centroid
    centroid = compute_document_centroid(embeddings)
    centroid = centroid.unsqueeze(0)

    # Compute cosine similarity
    embeddings_np = embeddings.cpu().numpy()
    centroid_np = centroid.cpu().numpy()

    similarities = cosine_similarity(embeddings_np, centroid_np).flatten()

    # Normalize to [0, 1]
    similarities = (similarities + 1) / 2

    return {node_id: float(sim) for node_id, sim in zip(node_ids, similarities)}


def compute_embedding_diversity(
    embeddings: torch.Tensor,
    node_ids: List[int],
    selected_clauses: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Compute diversity score (penalize clauses too similar to already selected).

    Args:
        embeddings: Tensor of shape [num_clauses, embedding_dim]
        node_ids: List of clause IDs
        selected_clauses: Optional list of already-selected clause IDs

    Returns:
        Dict mapping clause_id -> diversity score [0, 1]
        (higher = more diverse/less similar to selected)
    """
    if selected_clauses is None or len(selected_clauses) == 0:
        return {node_id: 1.0 for node_id in node_ids}

    # Get embeddings for selected clauses
    selected_indices = [
        node_ids.index(cid) for cid in selected_clauses if cid in node_ids
    ]
    if len(selected_indices) == 0:
        return {node_id: 1.0 for node_id in node_ids}

    selected_embeddings = embeddings[selected_indices]
    all_embeddings = embeddings.cpu().numpy()
    selected_embeddings_np = selected_embeddings.cpu().numpy()

    # Compute similarity to selected clauses
    similarities = cosine_similarity(all_embeddings, selected_embeddings_np)
    max_similarity = similarities.max(axis=1)

    diversity_scores = 1.0 - max_similarity

    # Normalize to [0, 1]
    diversity_scores = np.clip(diversity_scores, 0.0, 1.0)

    return {node_id: float(div) for node_id, div in zip(node_ids, diversity_scores)}


def compute_all_embedding_features(
    embeddings: torch.Tensor,
    node_ids: List[int],
    selected_clauses: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Compute all embedding-based features.

    Args:
        embeddings: Tensor of shape [num_clauses, embedding_dim]
        node_ids: List of clause IDs
        selected_clauses: Optional list of selected clause IDs for diversity
    Returns:
        Dict mapping clause_id -> dict of embedding features
    """
    features = {}

    # Centroid similarity
    centroid_sim = compute_centroid_similarity(embeddings, node_ids)

    # Diversity
    diversity = compute_embedding_diversity(embeddings, node_ids, selected_clauses)

    # Combine
    for node_id in node_ids:
        features[node_id] = {
            "centroid_similarity": centroid_sim.get(node_id, 0.0),
            "diversity": diversity.get(node_id, 1.0),
        }

    return features
