"""
Rhetorical Role-Based Importance Scoring

Assigns importance scores based on clause rhetorical roles:
- Claim: Highest importance
- Premise: Medium-high importance
- Opposition: Medium importance
"""

from typing import Dict, List


# Base role weights
ROLE_WEIGHTS = {
    "Claim": 1.0,
    "Premise": 0.7,
    "Opposition": 0.6,
}
# Fallback when label is null/missing in JSON
DEFAULT_ROLE_WEIGHT = 0.5


def compute_role_importance(nodes: List[Dict]) -> Dict[int, float]:
    """
    Compute base importance score based on rhetorical role.
    Args:
        nodes: List of node dicts with 'id' and 'label'
    Returns:
        Dict mapping clause_id -> role importance score [0, 1]
    """
    role_scores = {}

    for node in nodes:
        node_id = node["id"]
        label = node.get("label")
        role_scores[node_id] = ROLE_WEIGHTS.get(label, DEFAULT_ROLE_WEIGHT)

    return role_scores


def compute_role_centrality_boost(
    nodes: List[Dict], edges: List[Dict], pagerank_scores: Dict[int, float]
) -> Dict[int, float]:
    """
    Boost role importance for clauses that are central in the graph.
    Args:
        nodes: List of node dicts
        edges: List of edge dicts
        pagerank_scores: PageRank scores from graph_centrality

    Returns:
        Dict mapping clause_id -> boosted role score
    """
    base_role_scores = compute_role_importance(nodes)
    boosted_scores = {}

    for node in nodes:
        node_id = node["id"]
        base_score = base_role_scores[node_id]
        pagerank = pagerank_scores.get(node_id, 0.0)

        # Boost base score by pagerank
        boosted_scores[node_id] = base_score * (1.0 + pagerank * 0.5)  # Max 1.5x boost

    max_score = max(boosted_scores.values()) if boosted_scores.values() else 1.0
    if max_score > 0:
        boosted_scores = {k: v / max_score for k, v in boosted_scores.items()}

    return boosted_scores


def compute_premise_support_importance(
    nodes: List[Dict], edges: List[Dict]
) -> Dict[int, float]:
    """
    Boost importance of premises that support multiple claims.
    Args:
        nodes: List of node dicts
        edges: List of edge dicts
    Returns:
        Dict mapping clause_id -> support importance score
    """
    premise_support_count = {}
    claim_ids = {node["id"] for node in nodes if node.get("label") == "Claim"}

    for edge in edges:
        if edge.get("relation") == "green_support":
            source = edge["source"]
            target = edge["target"]

            # If source is a premise and target is a claim
            source_node = next((n for n in nodes if n["id"] == source), None)
            if (
                source_node
                and source_node.get("label") == "Premise"
                and target in claim_ids
            ):
                premise_support_count[source] = premise_support_count.get(source, 0) + 1

    # Normalize support counts
    max_support = (
        max(premise_support_count.values()) if premise_support_count.values() else 1
    )
    support_importance = {
        node["id"]: (
            premise_support_count.get(node["id"], 0) / max_support
            if max_support > 0
            else 0.0
        )
        for node in nodes
    }

    return support_importance


def compute_all_rhetorical_features(
    nodes: List[Dict], edges: List[Dict], pagerank_scores: Dict[int, float]
) -> Dict[int, Dict[str, float]]:
    """
    Compute all rhetorical role-based features.
    Returns:
        Dict mapping clause_id -> dict of rhetorical features
    """
    features = {}

    role_importance = compute_role_importance(nodes)
    role_boosted = compute_role_centrality_boost(nodes, edges, pagerank_scores)

    premise_support = compute_premise_support_importance(nodes, edges)

    for node in nodes:
        node_id = node["id"]
        features[node_id] = {
            "role_importance": role_importance[node_id],
            "role_boosted": role_boosted[node_id],
            "premise_support": premise_support[node_id],
        }

    return features
