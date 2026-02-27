"""
Case Type Embeddings for Clause Ranking

Provides case type-specific embeddings and scoring adjustments.
"""

import torch
import torch.nn as nn
from typing import Dict


# Case type mapping
CASE_TYPES = {"Civil Appeal": 0, "Fundamental Rights": 1}

CASE_TYPE_NAMES = {v: k for k, v in CASE_TYPES.items()}


class CaseTypeEmbedding(nn.Module):
    """
    Learnable case type embeddings.
    """

    def __init__(self, num_case_types: int = 5, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_case_types, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, case_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            case_type_ids: Tensor of shape [batch_size] with case type IDs

        Returns:
            Embeddings of shape [batch_size, embedding_dim]
        """
        return self.embedding(case_type_ids)


def extract_case_type_from_doc_id(doc_id: str) -> str:
    """
    Extract case type from document ID.
    """
    doc_id_upper = doc_id.upper()

    if "_FR_" in doc_id_upper or "FR_" in doc_id_upper:
        return "Fundamental Rights"
    else:
        return "Civil Appeal"


def get_case_type_id(doc_id: str) -> int:
    """
    Get case type ID from document ID.

    Args:
        doc_id: Document ID string

    Returns:
        Case type ID (0-4)
    """
    case_type = extract_case_type_from_doc_id(doc_id)
    return CASE_TYPES.get(case_type, CASE_TYPES["Other"])


def compute_case_type_weights(case_type: str) -> Dict[str, float]:
    """
    Get case type-specific weight adjustments.
    Different case types may benefit from different signal weights.

    Args:
        case_type: Case type name
    Returns:
        Dict of weight adjustments (to be added to base weights)
    """
    # Base weights
    base_adjustments = {
        "pagerank": 0.0,
        "betweenness": 0.0,
        "role": 0.0,
        "citation": 0.0,
        "embedding": 0.0,
        "support_degree": 0.0,
        "position": 0.0,
    }

    # Case type-specific adjustments
    if case_type == "Fundamental Rights":
        # Emphasize role and citation
        return {
            "pagerank": 0.0,
            "betweenness": 0.0,
            "role": 0.05,
            "citation": 0.05,
            "embedding": 0.0,
            "support_degree": -0.05,
            "position": 0.0,
        }
    elif case_type == "Civil Appeal":
        # Emphasize citation and graph structure (precedents matter)
        return {
            "pagerank": 0.05,
            "betweenness": 0.0,
            "role": 0.0,
            "citation": 0.05,
            "embedding": 0.0,
            "support_degree": 0.0,
            "position": -0.05,
        }
    else:
        return base_adjustments


def apply_case_type_weights(
    base_weights: Dict[str, float], case_type: str
) -> Dict[str, float]:
    """
    Apply case type-specific weight adjustments.

    Args:
        base_weights: Base weight dict
        case_type: Case type name

    Returns:
        Adjusted weights (normalized to sum to 1.0)
    """
    adjustments = compute_case_type_weights(case_type)

    # Apply adjustments
    adjusted = {
        key: base_weights.get(key, 0.0) + adjustments.get(key, 0.0)
        for key in set(list(base_weights.keys()) + list(adjustments.keys()))
    }

    # Normalize to ensure weights sum to 1.0
    total = sum(v for k, v in adjusted.items() if k != "case_type")
    if total > 0:
        adjusted = {
            k: v / total if k != "case_type" else v for k, v in adjusted.items()
        }
    return adjusted
