"""
Graph Centrality Features for Clause Ranking

Computes various graph centrality measures on the argument graph:
- PageRank
- Betweenness Centrality
- Degree Centrality (in-degree, out-degree)
"""

import networkx as nx
from typing import Dict, List, Tuple


def build_networkx_graph(nodes: List[Dict], edges: List[Dict]) -> nx.DiGraph:
    """
    Build NetworkX directed graph from JSON graph structure.

    Args:
        nodes: List of node dicts with 'id' and 'label'
        edges: List of edge dicts with 'source', 'target', 'relation', 'score'

    Returns:
        NetworkX DiGraph with weighted edges
    """
    G = nx.DiGraph()

    # Add nodes
    for node in nodes:
        G.add_node(node["id"], label=node.get("label", "None"))

    # Add edges with weights based on relation type
    relation_weights = {
        "green_support": 1.0,
        "red_opposes": 0.8,
        "inter_citation": 1.2,
        "follows": 0.5,
        "semantic_incaselawbert": 0.7,
    }

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        relation = edge.get("relation", "follows")

        # Get weight from relation type
        weight = relation_weights.get(relation, 0.5)
        if "score" in edge:
            weight *= edge["score"]

        G.add_edge(source, target, weight=weight, relation=relation)

    return G


def compute_pagerank_scores(
    nodes: List[Dict], edges: List[Dict], alpha: float = 0.85, max_iter: int = 100
) -> Dict[int, float]:
    """
    Compute PageRank scores for each clause.
    Args:
        nodes: List of node dicts
        edges: List of edge dicts
        alpha: Damping parameter
        max_iter: Maximum number of iterations

    Returns:
        Dict mapping clause_id -> PageRank scores
    """
    G = build_networkx_graph(nodes, edges)

    if len(G.nodes()) == 0:
        return {node["id"]: 0.0 for node in nodes}

    # Compute PageRank with edge weights
    pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, weight="weight")

    # Normalize to [0, 1]
    if pagerank:
        max_score = max(pagerank.values())
        if max_score > 0:
            pagerank = {k: v / max_score for k, v in pagerank.items()}

    # Ensure all nodes have scores
    result = {node["id"]: pagerank.get(node["id"], 0.0) for node in nodes}

    return result


def compute_betweenness_centrality(
    nodes: List[Dict], edges: List[Dict], normalized: bool = True
) -> Dict[int, float]:
    """
    Compute betweenness centrality for each clause.
    Args:
        nodes: List of node dicts
        edges: List of edge dicts
        normalized: Whether to normalize scores
    Returns:
        Dict mapping clause_id -> betweenness score
    """
    G = build_networkx_graph(nodes, edges)

    if len(G.nodes()) == 0 or len(G.edges()) == 0:
        return {node["id"]: 0.0 for node in nodes}

    # Compute betweenness centrality
    betweenness = nx.betweenness_centrality(G, normalized=normalized, weight="weight")

    # Normalize to [0, 1]
    if betweenness:
        max_score = max(betweenness.values())
        if max_score > 0:
            betweenness = {k: v / max_score for k, v in betweenness.items()}

    # Ensure all nodes have scores
    result = {node["id"]: betweenness.get(node["id"], 0.0) for node in nodes}

    return result


def compute_degree_centrality(
    nodes: List[Dict], edges: List[Dict]
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Compute in-degree, out-degree, and total degree centrality.

    Args:
        nodes: List of node dicts
        edges: List of edge dicts

    Returns:
        Tuple of (in_degree, out_degree, total_degree) dicts
    """
    G = build_networkx_graph(nodes, edges)

    # Count degrees
    in_degree = {node["id"]: G.in_degree(node["id"]) for node in nodes}
    out_degree = {node["id"]: G.out_degree(node["id"]) for node in nodes}
    total_degree = {
        node["id"]: in_degree[node["id"]] + out_degree[node["id"]] for node in nodes
    }

    # Normalize by max degree
    max_in = max(in_degree.values()) if in_degree.values() else 1
    max_out = max(out_degree.values()) if out_degree.values() else 1
    max_total = max(total_degree.values()) if total_degree.values() else 1

    in_degree_norm = {
        k: v / max_in if max_in > 0 else 0.0 for k, v in in_degree.items()
    }
    out_degree_norm = {
        k: v / max_out if max_out > 0 else 0.0 for k, v in out_degree.items()
    }
    total_degree_norm = {
        k: v / max_total if max_total > 0 else 0.0 for k, v in total_degree.items()
    }

    return in_degree_norm, out_degree_norm, total_degree_norm


def compute_support_opposition_degree(
    nodes: List[Dict], edges: List[Dict]
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, float]]:
    """
    Compute support and opposition degrees for each clause.
    Args:
        nodes: List of node dicts
        edges: List of edge dicts

    Returns:
        Tuple of (support_in, opposition_in, support_ratio) dicts
        - support_in: Number of clauses supporting this clause
        - opposition_in: Number of clauses opposing this clause
        - support_ratio: support_in / (support_in + opposition_in + 1)
    """
    support_in = {node["id"]: 0 for node in nodes}
    opposition_in = {node["id"]: 0 for node in nodes}

    for edge in edges:
        target = edge["target"]
        relation = edge.get("relation", "")

        if relation == "green_support":
            support_in[target] = support_in.get(target, 0) + 1
        elif relation == "red_opposes":
            opposition_in[target] = opposition_in.get(target, 0) + 1

    # Compute support ratio
    support_ratio = {}
    for node in nodes:
        node_id = node["id"]
        support = support_in[node_id]
        opposition = opposition_in[node_id]
        total = support + opposition
        support_ratio[node_id] = support / (total + 1) if total > 0 else 0.0

    return support_in, opposition_in, support_ratio


def compute_all_centrality_features(
    nodes: List[Dict], edges: List[Dict]
) -> Dict[int, Dict[str, float]]:
    """
    Compute all centrality features for each clause.

    Returns:
        Dict mapping clause_id -> dict of feature_name -> score
    """
    features = {}

    # PageRank
    pagerank = compute_pagerank_scores(nodes, edges)

    # Betweenness
    betweenness = compute_betweenness_centrality(nodes, edges)

    # Degree centrality
    in_degree, out_degree, total_degree = compute_degree_centrality(nodes, edges)

    # Support/opposition
    support_in, opposition_in, support_ratio = compute_support_opposition_degree(
        nodes, edges
    )

    # Combine all features
    for node in nodes:
        node_id = node["id"]
        features[node_id] = {
            "pagerank": pagerank[node_id],
            "betweenness": betweenness[node_id],
            "in_degree": in_degree[node_id],
            "out_degree": out_degree[node_id],
            "total_degree": total_degree[node_id],
            "support_in": float(support_in[node_id]),
            "opposition_in": float(opposition_in[node_id]),
            "support_ratio": support_ratio[node_id],
        }

    return features
