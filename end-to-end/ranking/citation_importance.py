"""
Citation-Based Importance Scoring

Computes importance scores based on:
- Citation count
- Cited precedent importance
- Citation position in document
"""

from typing import Dict, List, Optional
from collections import defaultdict
import networkx as nx
import json
import re


def extract_citations_from_text(text: str) -> List[str]:
    """
    Extract citation patterns from clause text using regex.
    Args:
        text: Clause text
    Returns:
        List of citation strings found in text
    """
    import re

    citations = []

    # Citation patterns
    patterns = [
        r"SC/CA/(\d+)/(\d{4})",
        r"SC/FR/(\d+)/(\d{4})",
        r"SC Appeal No[:\s]*(\d+)/(\d{4})",
        r"SC APPEAL NO[:\s]*(\d+)/(\d{4})",
        r"SC Appeal (\d+)/(\d{4})",
        r"S\.C Appeal No[:\s]*(\d+)/(\d{4})",
        r"S\.C\. Appeal No[:\s]*(\d+)/(\d{4})",
        r"SC FR No[:\s]*(\d+)/(\d{4})",
        r"SC FR (\d+)/(\d{4})",
        r"\([^)]*SC Appeal No[^)]*(\d+)/(\d{4})[^)]*\)",
        r"\[[^\]]*SC Appeal No[^\]]*(\d+)/(\d{4})[^\]]*\]",
        r"SC/(\d+)/(\d{4})",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            number = match.group(1)
            year = (
                match.group(2)
                if len(match.groups()) > 1
                else match.group(2) if match.lastindex else None
            )
            if year:
                # Format citation
                if "FR" in pattern or "FR" in match.group(0):
                    citation = f"SC/FR/{number}/{year}"
                elif "CA" in pattern or "Appeal" in pattern:
                    citation = f"SC/CA/{number}/{year}"
                else:
                    citation = f"SC/{number}/{year}"
                citations.append(citation)

    # Remove duplicates
    return list(set(citations))


def compute_citation_count(
    nodes: List[Dict], edges: List[Dict] = None
) -> Dict[int, int]:
    """
    Count number of citations from each clause.

    Args:
        nodes: List of node dicts
        edges: Optional list of edge dicts (for full graphs)

    Returns:
        Dict mapping clause_id -> citation count
    """
    citation_count = {node["id"]: 0 for node in nodes}

    # Count inter_citation edges
    if edges:
        for edge in edges:
            if edge.get("relation") == "inter_citation":
                source = edge["source"]
                citation_count[source] = citation_count.get(source, 0) + 1

    # Extract citations from text
    for node in nodes:
        node_id = node["id"]
        text = node.get("text", "")

        # Extract citations from text
        citations = extract_citations_from_text(text)
        citation_count[node_id] = len(citations)

    return citation_count


def build_citation_graph(graph_files: List[str]) -> nx.DiGraph:
    """
    Build inter-document citation graph from multiple graph JSON files.
    Args:
        graph_files: List of paths to graph JSON files
    Returns:
        NetworkX DiGraph where nodes are document IDs and edges are citations
    """

    citation_graph = nx.DiGraph()

    for graph_file in graph_files:
        try:
            with open(graph_file, "r", encoding="utf-8") as f:
                graph = json.load(f)

            doc_id = graph.get("doc_id", "")
            if not doc_id:
                continue

            citation_graph.add_node(doc_id)

            # Extract citations from edges
            for edge in graph.get("edges", []):
                if edge.get("relation") == "inter_citation":
                    target_node_id = edge.get("target")
                    if target_node_id:
                        for node in graph.get("nodes", []):
                            if node.get("id") == target_node_id:
                                cited_doc = node.get("cited_doc")
                                if cited_doc:
                                    citation_graph.add_edge(doc_id, cited_doc)
        except Exception as e:
            print(f"Warning: Error processing {graph_file}: {e}")
            continue

    return citation_graph


def compute_precedent_importance(
    citation_graph: nx.DiGraph, alpha: float = 0.85
) -> Dict[str, float]:
    """
    Compute importance of precedents using PageRank on citation graph.
    Args:
        citation_graph: NetworkX DiGraph of document citations
        alpha: PageRank damping parameter
    Returns:
        Dict mapping document_id -> precedent importance score
    """
    if len(citation_graph.nodes()) == 0:
        return {}

    # Compute PageRank on citation graph
    precedent_pagerank = nx.pagerank(citation_graph, alpha=alpha)

    # Normalize to [0, 1] scores
    if precedent_pagerank:
        max_score = max(precedent_pagerank.values())
        if max_score > 0:
            precedent_pagerank = {
                k: v / max_score for k, v in precedent_pagerank.items()
            }

    return precedent_pagerank


def extract_cited_doc_ids_from_text(text: str) -> List[str]:
    """
    Extract cited document IDs from clause text.
    Converts citation format to doc_id format.
    """
    citations = extract_citations_from_text(text)
    cited_doc_ids = []

    for citation in citations:
        # Convert citation format to doc_id format
        match = re.search(r"SC/(?:CA|FR)?/(\d+)/(\d{4})", citation)
        if match:
            number = match.group(1)
            year = match.group(2)
            case_type = (
                "CA"
                if "CA" in citation or "Appeal" in citation
                else "FR" if "FR" in citation else "CA"
            )
            doc_id = f"SC_{case_type}_{number}_{year}"
            cited_doc_ids.append(doc_id)

    return cited_doc_ids


def compute_citation_importance_scores(
    nodes: List[Dict],
    edges: List[Dict] = None,
    precedent_importance: Optional[Dict[str, float]] = None,
) -> Dict[int, float]:
    """
    Compute citation-based importance for each clause.
    Args:
        nodes: List of node dicts
        edges: Optional list of edge dicts
        precedent_importance: Optional dict of document_id -> importance

    Returns:
        Dict mapping clause_id -> citation importance score
    """
    citation_scores = {}

    # Get citation counts
    citation_counts = compute_citation_count(nodes, edges)
    max_citations = max(citation_counts.values()) if citation_counts.values() else 1

    # Build map of clause
    clause_citations = defaultdict(set)

    # From inter_citation edges
    if edges:
        for edge in edges:
            if edge.get("relation") == "inter_citation":
                source = edge["source"]
                target = edge["target"]

                # Find cited document from target node
                target_node = next((n for n in nodes if n.get("id") == target), None)
                if target_node:
                    cited_doc = target_node.get("cited_doc")
                    if cited_doc:
                        clause_citations[source].add(cited_doc)

    # Extract from text
    for node in nodes:
        node_id = node["id"]
        text = node.get("text", "")

        # Extract cited document IDs from text
        cited_doc_ids = extract_cited_doc_ids_from_text(text)
        clause_citations[node_id].update(cited_doc_ids)

    # Compute scores
    for node in nodes:
        node_id = node["id"]

        # Base score from citation count
        citation_count = citation_counts.get(node_id, 0)
        base_score = citation_count / max_citations if max_citations > 0 else 0.0

        # Boost if cited precedents are important
        boost = 0.0
        if precedent_importance:
            cited_docs = clause_citations.get(node_id, set())
            if cited_docs:
                avg_precedent_importance = sum(
                    precedent_importance.get(doc, 0.0) for doc in cited_docs
                ) / len(cited_docs)
                boost = avg_precedent_importance * 0.5

        citation_scores[node_id] = min(1.0, base_score + boost)

    return citation_scores


def compute_citation_position_score(
    nodes: List[Dict], edges: List[Dict] = None
) -> Dict[int, float]:
    """
    Compute position-based citation importance.
    Citations middle in document get boost.
    Args:
        nodes: List of node dicts (should be in document order)
        edges: Optional list of edge dicts

    Returns:
        Dict mapping clause_id -> position score
    """
    # Get clause IDs that have citations
    citing_clauses = set()

    # From inter_citation edges
    if edges:
        citing_clauses.update(
            edge["source"] for edge in edges if edge.get("relation") == "inter_citation"
        )

    #  Extract from text
    for node in nodes:
        text = node.get("text", "")
        citations = extract_citations_from_text(text)
        if citations:
            citing_clauses.add(node["id"])

    # Compute position scores
    position_scores = {}
    total_nodes = len(nodes)

    for idx, node in enumerate(nodes):
        node_id = node["id"]

        if node_id in citing_clauses:
            # Position in document (0 = start, 1 = end)
            position = idx / total_nodes if total_nodes > 0 else 0.5

            # Boost for middle citations
            if 0.2 < position < 0.8:
                position_scores[node_id] = 1.0 - position * 0.5
            else:
                position_scores[node_id] = 0.5
        else:
            position_scores[node_id] = 0.0

    return position_scores


def compute_all_citation_features(
    nodes: List[Dict],
    edges: List[Dict] = None,
    precedent_importance: Optional[Dict[str, float]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Compute all citation-based features.

    Returns:
        Dict mapping clause_id -> dict of citation features
    """
    features = {}

    # Citation count
    citation_counts = compute_citation_count(nodes, edges)
    max_citations = max(citation_counts.values()) if citation_counts.values() else 1
    citation_count_norm = {
        node_id: count / max_citations if max_citations > 0 else 0.0
        for node_id, count in citation_counts.items()
    }

    # Citation importance
    citation_importance = compute_citation_importance_scores(
        nodes, edges, precedent_importance
    )

    # Citation position
    citation_position = compute_citation_position_score(nodes, edges)

    # Combine
    for node in nodes:
        node_id = node["id"]
        features[node_id] = {
            "citation_count": citation_count_norm.get(node_id, 0.0),
            "citation_importance": citation_importance.get(node_id, 0.0),
            "citation_position": citation_position.get(node_id, 0.0),
        }

    return features
