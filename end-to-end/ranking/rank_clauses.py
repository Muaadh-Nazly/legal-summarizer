"""
Main entry point for clause ranking.
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


from .hybrid_ranker import rank_clauses, rank_clauses_with_diversity, DEFAULT_WEIGHTS
from .citation_importance import build_citation_graph, compute_precedent_importance
from .case_type_embeddings import (
    apply_case_type_weights,
    extract_case_type_from_doc_id,
)
from .adaptive_topk import compute_adaptive_topk, ADAPTIVE_CONFIGS
from .gnn_inference import load_model, graph_json_to_pyg_inference
from .postprocessing import ensure_disposition_in_selection, order_selected_by_document

# Configuration
CONFIG = {
    # Directories
    "graph_dir": "/kaggle/input/semantic-graphs",
    "output_dir": "/kaggle/working/ranked_clauses",
    # GNN Model
    "gnn_model_path": "/kaggle/input/gnn-trained/hgt_encoder_final.pt",
    # Top-K Selection
    "top_k": None,  # Adaptive (overrides adaptive if set)
    "adaptive_topk": True,  # Use adaptive top_k based on document characteristics
    "adaptive_method": "hybrid",  # "percentage", "length", "complexity", "case_type", "hybrid"
    "adaptive_config": "context_rich",  # "conservative", "balanced", "comprehensive", "case_type_focused", "context_rich"
    # Diversity Ranking
    "use_diversity": True,
    "diversity_weight": 0.3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_gnn_model_and_tokenizer(
    model_path: str, device: str = "cuda", bert_model_path: Optional[str] = None
):
    """
    Load trained GNN model and InCaseLawBert tokenizer/model.
    Args:
        model_path: Path to trained GNN model (.pt)
        bert_model_path: Optional path to InCaseLawBERT; if None, uses HuggingFace 'law-ai/InCaseLawBERT'.
    Returns:
        Tuple of (gnn_model, tokenizer, bert_model)
    """
    inference_module = load_model

    bert_src = bert_model_path or "law-ai/InCaseLawBERT"
    import logging

    _log = logging.getLogger("transformers")
    _old = _log.level
    _log.setLevel(logging.WARNING)
    try:
        tokenizer = AutoTokenizer.from_pretrained(bert_src)
        bert_model = AutoModel.from_pretrained(bert_src)
    finally:
        _log.setLevel(_old)
    bert_model = bert_model.to(device)
    bert_model.eval()

    # Load GNN model
    gnn_model = inference_module(model_path, in_dim=768, device=device)
    print("Models loading done.")
    return gnn_model, tokenizer, bert_model


def extract_embeddings_from_graph(
    graph_json_path: str, gnn_model, tokenizer, bert_model, device: str = "cuda"
) -> Optional[torch.Tensor]:
    """
    Extract GNN embeddings for clauses in a graph.
    Args:
        graph_json_path: Path to graph JSON
        gnn_model: Trained GNN model
        tokenizer: InCaseLawBert tokenizer
        bert_model: InCaseLawBert model

    Returns:
        Tensor of shape [num_clauses, embedding_dim] or None if error
    """
    try:
        graph_to_pyg = graph_json_to_pyg_inference
        # Convert graph to PyG format
        data = graph_to_pyg(graph_json_path, tokenizer, bert_model, device)
        if data is None:
            return None

        data = data.to(device)

        # Extract embeddings using GNN model
        gnn_model.eval()
        with torch.no_grad():
            # Try to get embeddings
            try:
                out = gnn_model(data, return_embeddings=True)
                embeddings = out[1] if isinstance(out, tuple) else out
            except TypeError:
                embeddings = data.x

        return embeddings.cpu()
    except Exception as e:
        print(f"⚠️  Error extracting embeddings from {graph_json_path}: {e}")
        import traceback

        traceback.print_exc()
        return None


def process_all_graphs(
    graph_dir: str,
    output_dir: str,
    gnn_model_path: Optional[str] = None,
    bert_model_path: Optional[str] = None,
    top_k: Optional[int] = None,
    adaptive_topk: bool = True,
    adaptive_method: str = "hybrid",
    adaptive_config: Optional[str] = None,
    use_diversity: bool = False,
    diversity_weight: float = 0.3,
    device: str = "cuda",
):
    """
    Process all graphs and rank clauses.
    Args:
        graph_dir: Directory containing graph JSON files
        output_dir: Directory to save ranked clauses
        gnn_model_path: Optional path to GNN model (for embeddings)
        bert_model_path: Optional path to InCaseLawBERT (for diversity embeddings)
        top_k: Number of top clauses to select
        use_diversity: Whether to use diversity-aware ranking
        diversity_weight: Weight for diversity (if use_diversity=True)
    """
    graph_dir = Path(graph_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all graph files (prefer inter-doc graphs with citation edges)
    graph_files = sorted(graph_dir.glob("*.inter-doc.json"))
    if not graph_files:
        graph_files = sorted(graph_dir.glob("*.json"))
    print(f"📊 Found {len(graph_files)} graph files")

    # Load models if embeddings are needed
    gnn_model = None
    tokenizer = None
    bert_model = None
    if gnn_model_path and use_diversity:
        gnn_model, tokenizer, bert_model = load_gnn_model_and_tokenizer(
            gnn_model_path, device, bert_model_path=bert_model_path
        )

    # Build citation graph for precedent importance
    print("🔗 Building citation graph...")
    citation_graph = build_citation_graph([str(f) for f in graph_files])
    precedent_importance = compute_precedent_importance(citation_graph)
    print(f"✅ Citation graph built with {len(precedent_importance)} documents")

    # Process each graph
    all_results = []

    for graph_file in tqdm(graph_files, desc="Ranking clauses"):
        try:
            # Load graph
            with open(graph_file, "r", encoding="utf-8") as f:
                graph = json.load(f)

            doc_id = graph.get("doc_id", graph_file.stem)
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])

            # Extract embeddings if using diversity
            embeddings = None
            if use_diversity and gnn_model:
                embeddings = extract_embeddings_from_graph(
                    str(graph_file), gnn_model, tokenizer, bert_model, device
                )

            # Get case type and apply case type weights
            case_type = extract_case_type_from_doc_id(doc_id)
            weights = apply_case_type_weights(DEFAULT_WEIGHTS.copy(), case_type)

            # Compute adaptive top-k if enabled
            num_clauses = len(nodes)
            num_edges = len(edges)

            if adaptive_topk:
                if adaptive_config and adaptive_config in ADAPTIVE_CONFIGS:
                    config = ADAPTIVE_CONFIGS[adaptive_config]
                    doc_top_k = compute_adaptive_topk(
                        num_clauses,
                        num_edges,
                        case_type,
                        method=config.get("method", "hybrid"),
                        min_clauses=config.get("min_clauses", 8),
                        max_clauses=config.get("max_clauses", 30),
                        **{
                            k: v
                            for k, v in config.items()
                            if k not in ["method", "min_clauses", "max_clauses"]
                        },
                    )
                else:
                    doc_top_k = compute_adaptive_topk(
                        num_clauses, num_edges, case_type, method=adaptive_method
                    )
            else:
                doc_top_k = top_k if top_k is not None else 20

            # Rank clauses
            ranked_full = rank_clauses(
                str(graph_file),
                embeddings,
                precedent_importance,
                weights,
                top_k=None,
            )

            # Select top-k either by score or MMR diversity
            if use_diversity and embeddings is not None:
                ranked_selected = rank_clauses_with_diversity(
                    str(graph_file),
                    embeddings,
                    precedent_importance,
                    weights,
                    top_k=doc_top_k,
                    diversity_weight=diversity_weight,
                )
            else:
                ranked_selected = ranked_full[:doc_top_k]

            # Post-process:
            ranked, forced_disposition_ids = ensure_disposition_in_selection(
                ranked_selected, ranked_full, nodes, top_k=doc_top_k
            )
            # Prepare a coherent, document-ordered version
            ranked_doc_order = order_selected_by_document(ranked, nodes)

            # Save results
            actual_selected_count = len(ranked)
            result = {
                "doc_id": doc_id,
                "case_type": case_type,
                "total_clauses": num_clauses,
                "total_edges": num_edges,
                "selected_clauses": doc_top_k,
                "actual_selected_clauses": actual_selected_count,
                "disposition_added_as_extra": len(forced_disposition_ids) > 0
                and actual_selected_count > doc_top_k,
                "adaptive_topk": adaptive_topk,
                "adaptive_method": adaptive_method if adaptive_topk else None,
                "forced_disposition_clause_ids": forced_disposition_ids,
                "ranked_clauses": [
                    {
                        "clause_id": clause_id,
                        "score": float(score),
                        "text": info.get("text", ""),
                        "label": info.get("label", "None"),
                        "rank": idx + 1,
                    }
                    for idx, (clause_id, score, info) in enumerate(ranked)
                ],
                # Same selected clauses in original document order
                "selected_clauses_in_doc_order": [
                    {
                        "clause_id": clause_id,
                        "score": float(score),
                        "text": info.get("text", ""),
                        "label": info.get("label", "None"),
                        "doc_order": idx + 1,
                    }
                    for idx, (clause_id, score, info) in enumerate(ranked_doc_order)
                ],
            }

            # Save to file
            output_file = output_dir / f"{doc_id}.ranked.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            all_results.append(result)

        except Exception as e:
            print(f"❌ Error processing {graph_file}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save summary
    summary = {
        "total_graphs_processed": len(all_results),
        "total_graphs": len(graph_files),
        "top_k": top_k,
        "use_diversity": use_diversity,
        "results": [
            {
                "doc_id": r["doc_id"],
                "case_type": r["case_type"],
                "total_clauses": r["total_clauses"],
                "selected_clauses": len(r["ranked_clauses"]),
            }
            for r in all_results
        ],
    }

    summary_file = output_dir / "ranking_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Processed {len(all_results)}/{len(graph_files)} graphs")
    print(f"📁 Results saved to {output_dir}")
    print(f"📊 Summary saved to {summary_file}")


def parse_args():
    """Parse command-line arguments; passable options override CONFIG defaults."""
    p = argparse.ArgumentParser(
        description="Clause ranking for legal judgment summarization"
    )
    p.add_argument(
        "--graph-dir",
        type=str,
        default=None,
        help="Directory containing clause graph JSON files (e.g. *_semantic.json). Overrides CONFIG.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write ranked clause JSON files. Overrides CONFIG.",
    )
    p.add_argument(
        "--gnn-model-path",
        type=str,
        default=None,
        help="Path to trained GNN model (.pt). Overrides CONFIG.",
    )
    p.add_argument(
        "--adaptive-config",
        type=str,
        default=None,
        choices=list(ADAPTIVE_CONFIGS.keys()) if ADAPTIVE_CONFIGS else None,
        help="Adaptive top-k config: conservative, balanced, comprehensive, case_type_focused, context_rich. Overrides CONFIG.",
    )
    return p.parse_args()


def main():
    """Main function. Passable args override CONFIG."""
    args = parse_args()

    config = dict(CONFIG)
    if args.graph_dir is not None:
        config["graph_dir"] = args.graph_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.gnn_model_path is not None:
        config["gnn_model_path"] = args.gnn_model_path
    if args.adaptive_config is not None:
        config["adaptive_config"] = args.adaptive_config

    print("=" * 80)
    print("CLAUSE RANKING FOR LEGAL JUDGMENT SUMMARIZATION")
    print("=" * 80)
    print(f"Graph directory: {config['graph_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"GNN model: {config['gnn_model_path']}")
    print(f"Top-K: {config['top_k']}")
    print(f"Adaptive Top-K: {config['adaptive_topk']}")
    if config["adaptive_topk"]:
        print(f"  Method: {config['adaptive_method']}")
        if config["adaptive_config"]:
            print(f"  Config: {config['adaptive_config']}")
    print(f"Use diversity: {config['use_diversity']}")
    if config["use_diversity"]:
        print(f"  Diversity weight: {config['diversity_weight']}")
    print(f"Device: {config['device']}")
    print("=" * 80)

    process_all_graphs(
        graph_dir=config["graph_dir"],
        output_dir=config["output_dir"],
        gnn_model_path=config["gnn_model_path"],
        top_k=config["top_k"],
        adaptive_topk=config["adaptive_topk"],
        adaptive_method=config["adaptive_method"],
        adaptive_config=config["adaptive_config"],
        use_diversity=config["use_diversity"],
        diversity_weight=config["diversity_weight"],
        device=config["device"],
    )


if __name__ == "__main__":
    main()
