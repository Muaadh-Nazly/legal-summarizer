import json
from pathlib import Path
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
from tqdm import tqdm
from typing import List, Dict, Optional


# CONFIGURATION
class GraphConfig:
    # Similarity thresholds
    SIM_THRESHOLD = 0.75
    SIM_WINDOW = 10
    SEMANTIC_TOP_K = 3
    SEMANTIC_MIN_SCORE = 0.75

    # Label edges
    LABEL_WINDOW = 5

    # Processing
    BATCH_SIZE = 32
    MAX_LENGTH = 512

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INPUT_DIR = Path("/kaggle/input/annotated-extended")
    OUT_DIR = Path("/kaggle/working/graphs")

    @classmethod
    def setup_directories(cls):
        cls.OUT_DIR.mkdir(parents=True, exist_ok=True)


config = GraphConfig()
# tokenizer and model loaded inside function
tokenizer = None
model = None


# EMBEDDING FUNCTIONS
def get_embeddings_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Process embeddings in batches for efficiency.
    Returns numpy array of shape [num_texts, embedding_dim]
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding=True,
        ).to(config.DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            token_emb = outputs.last_hidden_state  # [batch_size, L, D]
            mask = (
                inputs["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            )
            mean_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(mean_emb.cpu().numpy())

    return np.vstack(all_embeddings)


# GRAPH BUILDING FUNCTIONS
def build_graph_from_doc(doc: Dict) -> Optional[nx.MultiDiGraph]:
    """
    Build base graph from document with nodes and sequential edges.
    """
    try:
        clauses = [
            c for c in doc.get("clauses", []) if c.get("label", "None") != "None"
        ]

        if not clauses:
            print(f"⚠️ Warning: {doc.get('doc_id', 'unknown')} has no labeled clauses")
            return None

        N = len(clauses)
        G = nx.MultiDiGraph()

        # Add nodes
        for i, c in enumerate(clauses):
            cid = c.get("clause_id", i)
            G.add_node(cid, text=c["text"], label=c["label"], idx=i)

        # Sequential edges (follows)
        for i in range(N - 1):
            a, b = clauses[i]["clause_id"], clauses[i + 1]["clause_id"]
            G.add_edge(a, b, relation="follows", score=1.0)

        return G

    except KeyError as e:
        print(f"❌ Error processing {doc.get('doc_id', 'unknown')}: Missing key {e}")
        return None
    except Exception as e:
        print(f"❌ Error processing {doc.get('doc_id', 'unknown')}: {e}")
        return None


def add_semantic_edges_incaselawbert(
    G: nx.MultiDiGraph, threshold: float = 0.7, window: int = 10, batch_size: int = 32
) -> nx.MultiDiGraph:
    """
    Add semantic similarity edges using InCaseLawBert embeddings.
    """
    nodes = sorted(G.nodes(data=True), key=lambda x: x[1]["idx"])
    texts = [d["text"] for _, d in nodes]
    ids = [n for n, _ in nodes]
    labels = [d["label"] for _, d in nodes]

    if not texts:
        return G

    print(
        f"⚙️ Computing InCaseLawBert embeddings for {len(texts)} clauses (batch size: {batch_size})..."
    )

    embeddings = get_embeddings_batch(texts, batch_size=batch_size)
    sim_matrix = cosine_similarity(embeddings)

    added = 0
    N = len(ids)

    for i in range(N):
        # Determine window bounds
        start_j = max(0, i - window) if window else 0
        end_j = min(N, i + window + 1) if window else N

        for j in range(start_j, end_j):
            if i == j:
                continue

            sim = sim_matrix[i, j]
            if sim >= threshold:
                G.add_edge(
                    ids[i],
                    ids[j],
                    relation="semantic_incaselawbert",
                    score=float(sim),
                    src_label=labels[i],
                    tgt_label=labels[j],
                )
                added += 1

    print(f"✅ Added {added} semantic edges")
    return G


def filter_semantic_edges(
    G: nx.MultiDiGraph, k: int = 3, min_score: float = 0.75
) -> nx.MultiDiGraph:
    """
    Keep only top-k semantic edges per node in BOTH directions (outgoing and incoming).
    Each node ends up with at most k outgoing and at most k incoming semantic edges.
    """

    def is_semantic(d):
        return "semantic" in d.get("relation", "")

    to_remove = []

    # Cap Outgoing
    for node in G.nodes():
        sem_edges = []
        for _, v, key, d in G.out_edges(node, data=True, keys=True):
            if is_semantic(d):
                sem_edges.append((v, key, d.get("score", 0.0)))
        sem_edges.sort(key=lambda x: x[2], reverse=True)
        keep_out = {(v, key) for v, key, score in sem_edges[:k] if score >= min_score}
        for _, v, key, d in G.out_edges(node, data=True, keys=True):
            if is_semantic(d) and (v, key) not in keep_out:
                to_remove.append((node, v, key))

    for u, v, key in to_remove:
        if G.has_edge(u, v, key):
            G.remove_edge(u, v, key)
    to_remove.clear()

    # Cap Incoming
    for node in G.nodes():
        sem_edges = []
        for u, _, key, d in G.in_edges(node, data=True, keys=True):
            if is_semantic(d):
                sem_edges.append((u, key, d.get("score", 0.0)))
        sem_edges.sort(key=lambda x: x[2], reverse=True)
        keep_in = {(u, key) for u, key, score in sem_edges[:k] if score >= min_score}
        for u, _, key, d in G.in_edges(node, data=True, keys=True):
            if is_semantic(d) and (u, key) not in keep_in:
                to_remove.append((u, node, key))

    for u, v, key in to_remove:
        if G.has_edge(u, v, key):
            G.remove_edge(u, v, key)

    return G


def transform_semantic_edges(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Transform semantic edges to green_support/red_opposes based on node labels.
    """
    for u, v, key, d in list(G.edges(data=True, keys=True)):
        if "semantic" not in d.get("relation", ""):
            continue

        src_label = G.nodes[u]["label"]
        tgt_label = G.nodes[v]["label"]

        # Determine edge type based on labels
        if src_label == "Opposition" or tgt_label == "Opposition":
            edge_type = "red_opposes"
        else:
            edge_type = "green_support"

        # Update edge attributes
        d["relation"] = edge_type
        d["color"] = "green" if "green" in edge_type else "red"
        d["style"] = "dashed"
        d["source"] = "semantic"

    print("✅ Semantic edges transformed into logic-based support/opposes edges")
    return G


def merge_graph(G_master: nx.MultiDiGraph, G_new: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Merge edges from G_new into G_master without adding duplicates
    of the same relation type between the same (u,v).
    """
    for u, v, key, data_new in G_new.edges(data=True, keys=True):
        relation_new = data_new.get("relation")

        # Get all existing edges between u and v
        edge_dict = G_master.get_edge_data(u, v, default={})
        duplicate = False
        for _, data_existing in edge_dict.items():
            if data_existing.get("relation") == relation_new:
                duplicate = True
                break

        # Add edge if no duplicate relation exists
        if not duplicate:
            G_master.add_edge(u, v, **data_new)

    return G_master


def print_graph_stats(G: nx.MultiDiGraph, doc_id: str):
    """Print detailed graph statistics."""
    print(f"\n📊 Graph Statistics: {doc_id}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Edge counts by relation
    edge_counts = {}
    for _, _, d in G.edges(data=True):
        rel = d.get("relation", "unknown")
        edge_counts[rel] = edge_counts.get(rel, 0) + 1

    print("  Edges by relation:")
    for rel, count in sorted(edge_counts.items()):
        print(f"    {rel}: {count}")

    # Node label distribution
    label_counts = {}
    for _, d in G.nodes(data=True):
        label = d.get("label", "None")
        label_counts[label] = label_counts.get(label, 0) + 1

    print("  Node labels:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")


# PROCESS DOCUMENTS - BUILD BASE GRAPHS
def process_document(
    doc: Dict, add_semantic: bool = False
) -> Optional[nx.MultiDiGraph]:
    """
    Process a single document into a graph.
    """
    # Build base graph
    G = build_graph_from_doc(doc)
    if G is None:
        return None

    # Add semantic edges
    if add_semantic:
        G = add_semantic_edges_incaselawbert(
            G,
            threshold=config.SIM_THRESHOLD,
            window=config.SIM_WINDOW,
            batch_size=config.BATCH_SIZE,
        )
        G = filter_semantic_edges(
            G, k=config.SEMANTIC_TOP_K, min_score=config.SEMANTIC_MIN_SCORE
        )
        G = transform_semantic_edges(G)

    return G


def run_construct_graph(
    input_dir,
    output_dir,
    bert_model_path=None,
):
    """Run graph construction: base graphs → semantic edges.
    Semantic graphs are used for clause ranking.
    bert_model_path: optional path to InCaseLawBERT; if None, uses HuggingFace 'law-ai/InCaseLawBERT'.
    """
    global tokenizer, model
    config.INPUT_DIR = Path(input_dir)
    config.OUT_DIR = Path(output_dir)
    config.setup_directories()

    bert_src = str(bert_model_path) if bert_model_path else "law-ai/InCaseLawBERT"
    import logging

    logging.getLogger("transformers").setLevel(logging.WARNING)
    tokenizer = AutoTokenizer.from_pretrained(bert_src)
    model = AutoModel.from_pretrained(bert_src).to(config.DEVICE)
    model.eval()
    print("Models loading done.")

    print("\n" + "=" * 60)
    print("PROCESSING DOCUMENTS - BASE GRAPHS")
    print("=" * 60)
    json_files = list(config.INPUT_DIR.rglob("*.annotated.json"))
    base_graphs = {}
    if not json_files:
        print(f"⚠️ No JSON files found in {config.INPUT_DIR}")
    else:
        for p in tqdm(json_files, desc="Processing documents"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                G = process_document(doc, add_semantic=False)
                if G is not None:
                    doc_id_raw = doc["doc_id"]
                    base_id = (
                        doc_id_raw.removesuffix(".clauses")
                        if doc_id_raw.endswith(".clauses")
                        else doc_id_raw
                    )
                    base_graphs[base_id] = G
                    print(
                        f"✅ {base_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                    )
            except Exception as e:
                print(f"❌ Error processing {p}: {e}")

    print("\n" + "=" * 60)
    print("ADDING SEMANTIC EDGES (no pickle)")
    print(f"Threshold: {config.SIM_THRESHOLD}, Window: {config.SIM_WINDOW}")
    print("=" * 60)
    for doc_id, G_orig in tqdm(base_graphs.items(), desc="Semantic edges"):
        try:
            G_sem = add_semantic_edges_incaselawbert(
                G_orig.copy(),
                threshold=config.SIM_THRESHOLD,
                window=config.SIM_WINDOW,
                batch_size=config.BATCH_SIZE,
            )
            G_sem = filter_semantic_edges(
                G_sem, k=config.SEMANTIC_TOP_K, min_score=config.SEMANTIC_MIN_SCORE
            )
            G_sem = transform_semantic_edges(G_sem)
            G_final = merge_graph(G_sem, G_orig)
            nodes_data = [
                {"id": n, "text": d.get("text", ""), "label": d.get("label")}
                for n, d in G_final.nodes(data=True)
            ]
            edges_data = [
                {
                    "source": u,
                    "target": v,
                    "relation": d.get("relation"),
                    "score": d.get("score", None),
                }
                for u, v, key, d in G_final.edges(data=True, keys=True)
            ]
            base_id = (
                doc_id.removesuffix(".clauses")
                if doc_id.endswith(".clauses")
                else doc_id
            )
            out_json = config.OUT_DIR / f"{base_id}.semantic.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(
                    {"doc_id": base_id, "nodes": nodes_data, "edges": edges_data},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print_graph_stats(G_final, base_id)
        except Exception as e:
            print(f"❌ Error processing {doc_id}: {e}")

    print("\n" + "=" * 60)
    print("SEMANTIC EDGE ADDITION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent
    _input = _root.parent / "pipeline_workspace" / "09_predicted"
    _output = _root.parent / "pipeline_workspace" / "10_graphs"
    run_construct_graph(_input, _output)
