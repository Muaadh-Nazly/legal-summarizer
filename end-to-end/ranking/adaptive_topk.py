"""
Adaptive Top-K Selection

Determines the optimal number of clauses to select for summarization based on:
- Document length (number of clauses)
- Graph complexity (edge density, connectivity)
- Content characteristics
"""

from typing import Dict, Optional
import math


def compute_adaptive_topk(
    num_clauses: int,
    num_edges: int = 0,
    case_type: str = "Other",
    method: str = "hybrid",
    min_clauses: int = 5,
    max_clauses: int = 50,
    **kwargs,
) -> int:
    """
    Compute adaptive top_k based on document characteristics.

    Args:
        num_clauses: Total number of clauses in document
        num_edges: Total number of edges in graph
        case_type: Case type
        method: Method to use ('percentage', 'length', 'complexity', 'case_type', 'hybrid')
        min_clauses: Minimum clauses to select
        max_clauses: Maximum clauses to select
    Returns:
        Optimal top_k value
    """
    if method == "percentage":
        return compute_adaptive_topk_percentage(
            num_clauses, min_clauses, max_clauses, **kwargs
        )
    elif method == "length":
        return compute_adaptive_topk_length_based(
            num_clauses, min_clauses, max_clauses, **kwargs
        )
    elif method == "complexity":
        return compute_adaptive_topk_complexity_based(
            num_clauses, num_edges, min_clauses, max_clauses, **kwargs
        )
    elif method == "case_type":
        return compute_adaptive_topk_case_type_based(
            num_clauses, case_type, min_clauses, max_clauses, **kwargs
        )
    elif method == "hybrid":
        return compute_adaptive_topk_hybrid(
            num_clauses, num_edges, case_type, min_clauses, max_clauses, **kwargs
        )
    else:
        # Default: percentage-based
        return compute_adaptive_topk_percentage(num_clauses, min_clauses, max_clauses)


def compute_adaptive_topk_percentage(
    num_clauses: int,
    min_clauses: int = 5,
    max_clauses: int = 50,
    percentage: float = 0.15,
) -> int:
    """
    Select top_k as a percentage of total clauses.
    Args:
        num_clauses: Total number of clauses
        min_clauses: Minimum to select
        max_clauses: Maximum to select
        percentage: Percentage of clauses to select (default 15%)

    Returns:
        Top_k value
    """
    top_k = max(min_clauses, min(max_clauses, int(num_clauses * percentage)))
    return top_k


def compute_adaptive_topk_length_based(
    num_clauses: int,
    min_clauses: int = 5,
    max_clauses: int = 30,
    short_threshold: int = 100,
    medium_threshold: int = 200,
    short_ratio: float = 0.20,
    medium_ratio: float = 0.15,
    long_ratio: float = 0.10,
) -> int:
    """
    Select top_k based on document length categories.

    Short documents (< 100 clauses): 20% of clauses
    Medium documents (100-200 clauses): 15% of clauses
    Long documents (> 200 clauses): 10% of clauses

    Args:
        num_clauses: Total number of clauses
        min_clauses: Minimum to select
        max_clauses: Maximum to select
        short_threshold: Threshold for short documents
        medium_threshold: Threshold for medium documents
        short_ratio: Ratio for short documents
        medium_ratio: Ratio for medium documents
        long_ratio: Ratio for long documents

    Returns:
        Top_k value
    """
    if num_clauses < short_threshold:
        ratio = short_ratio
    elif num_clauses < medium_threshold:
        ratio = medium_ratio
    else:
        ratio = long_ratio

    top_k = max(min_clauses, min(max_clauses, int(num_clauses * ratio)))
    return top_k


def compute_adaptive_topk_complexity_based(
    num_clauses: int,
    num_edges: int,
    min_clauses: int = 5,
    max_clauses: int = 30,
    base_ratio: float = 0.10,
    complexity_factor: float = 0.05,
) -> int:
    """
    Select top_k based on graph complexity (edge density).

    More complex graphs (higher edge density) may need more clauses to capture
    the argument structure.

    Args:
        num_clauses: Total number of clauses
        num_edges: Total number of edges
        min_clauses: Minimum to select
        max_clauses: Maximum to select
        base_ratio: Base percentage of clauses
        complexity_factor: Factor to add based on complexity

    Returns:
        Top_k value
    """
    if num_clauses == 0:
        return min_clauses

    # Compute edge density (edges per clause)
    edge_density = num_edges / num_clauses if num_clauses > 0 else 0

    # Higher density indicates more complex graph
    if edge_density < 2:
        complexity_boost = 0.0
    elif edge_density < 4:
        complexity_boost = complexity_factor
    else:
        complexity_boost = complexity_factor * 2

    ratio = base_ratio + complexity_boost
    top_k = max(min_clauses, min(max_clauses, int(num_clauses * ratio)))

    return top_k


def compute_adaptive_topk_case_type_based(
    num_clauses: int,
    case_type: str,
    min_clauses: int = 5,
    max_clauses: int = 30,
    **kwargs,
) -> int:
    """
    Select top_k based on case type.

    Different case types may have different optimal summary lengths:
    - Fundamental Rights: Often more concise, 12-15% of clauses
    - Civil Appeal: More detailed, 15-18% of clauses

    Args:
        num_clauses: Total number of clauses
        case_type: Case type
        min_clauses: Minimum to select
        max_clauses: Maximum to select

    Returns:
        Top_k value
    """
    # Case type specific ratios
    case_type_ratios = {
        "Fundamental Rights": 0.13,
        "Civil Appeal": 0.17,
    }

    ratio = case_type_ratios.get(case_type, 0.15)
    top_k = max(min_clauses, min(max_clauses, int(num_clauses * ratio)))

    return top_k


def compute_adaptive_topk_hybrid(
    num_clauses: int,
    num_edges: int = 0,
    case_type: str = "Other",
    min_clauses: int = 5,
    max_clauses: int = 30,
    length_weight: float = 0.4,
    complexity_weight: float = 0.3,
    case_type_weight: float = 0.3,
) -> int:
    """
    Hybrid approach combining length, complexity, and case type.

    Args:
        num_clauses: Total number of clauses
        num_edges: Total number of edges
        case_type: Case type
        min_clauses: Minimum to select
        max_clauses: Maximum to select
        length_weight: Weight for length-based component
        complexity_weight: Weight for complexity-based component
        case_type_weight: Weight for case-type-based component

    Returns:
        Top_k value
    """

    length_k = compute_adaptive_topk_length_based(num_clauses, min_clauses, max_clauses)
    complexity_k = compute_adaptive_topk_complexity_based(
        num_clauses, num_edges, min_clauses, max_clauses
    )
    case_type_k = compute_adaptive_topk_case_type_based(
        num_clauses, case_type, min_clauses, max_clauses
    )

    # Weighted average
    total_weight = length_weight + complexity_weight + case_type_weight
    if total_weight == 0:
        total_weight = 1.0

    hybrid_k = (
        length_weight * length_k
        + complexity_weight * complexity_k
        + case_type_weight * case_type_k
    ) / total_weight

    # Round and clamp
    top_k = max(min_clauses, min(max_clauses, int(round(hybrid_k))))

    return top_k


# Predefined configurations
ADAPTIVE_CONFIGS = {
    "conservative": {
        "method": "percentage",
        "percentage": 0.10,
        "min_clauses": 5,
        "max_clauses": 30,
    },
    "balanced": {
        "method": "hybrid",
        "min_clauses": 8,
        "max_clauses": 30,
        "length_weight": 0.4,
        "complexity_weight": 0.3,
        "case_type_weight": 0.3,
    },
    "comprehensive": {
        "method": "hybrid",
        "min_clauses": 10,
        "max_clauses": 30,
        "length_weight": 0.3,
        "complexity_weight": 0.4,
        "case_type_weight": 0.3,
    },
    "case_type_focused": {"method": "case_type", "min_clauses": 8, "max_clauses": 30},
    "context_rich": {
        "method": "hybrid",
        "min_clauses": 20,
        "max_clauses": 50,
        "length_weight": 0.4,
        "complexity_weight": 0.4,
        "case_type_weight": 0.2,
    },
}
