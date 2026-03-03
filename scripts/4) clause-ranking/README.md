# 4) Clause-ranking

**Hybrid clause ranking** to select the most important clauses per document for summarization. Combines graph centrality, rhetorical role, citation importance, embedding similarity, and optional GNN embeddings; supports diversity (MMR) and adaptive top-k.

## Contents

| File | Description |
|------|-------------|
| **rank_clauses.py** | **Entry point.** Loads graph JSONs (`*.semantic.json`), builds citation graph for precedent importance, loads GNN + BERT if diversity is used. For each doc: optional adaptive top-k, hybrid scoring, diversity ranking, postprocessing (disposition in selection, doc order). Writes `*.ranked.json` and `ranking_summary.json`. |
| **hybrid_ranker.py** | Combines scores: **graph centrality** (PageRank), **rhetorical** (Claim/Premise/Opposition weights), **citation importance** (precedent graph), **embedding importance** (query vs clause similarity). Optional MMR-style diversity. |
| **graph_centrality.py** | Centrality features (PageRank, degree) on the clause graph. |
| **rhetorical_scoring.py** | Score clauses by argument role (Claim greater than Premise greater than Opposition). |
| **citation_importance.py** | Build citation graph across documents; compute precedent importance per doc. |
| **embedding_importance.py** | Embedding-based importance (centroid or query similarity). |
| **case_type_embeddings.py** | Case-type-specific weights (Civil Appeal or Fundamenta Rights). |
| **gnn_inference.py** | Load GNN encoder; compute node embeddings from graph JSON for diversity ranking. |
| **postprocessing.py** | Ensure disposition clause in selection; order selected clauses by document order. |
| **adaptive_topk** | A top-k selection criteria based on the configurations. |

## Data flow

- **Input:** Directory of graph JSONs (`*.semantic.json`) with `nodes`, `edges`, `doc_id`. Optional: citation graph from ranking over same corpus.
- **Output:** One `*.ranked.json` per document (with `ranked_clauses`, `selected_clauses_in_doc_order`, `doc_id`, top-k config) and `ranking_summary.json`.

## Config

- In **rank_clauses.py** `CONFIG`: set `graph_dir`, `output_dir`, `gnn_model_path`, `adaptive_topk`, `use_diversity`, `top_k`, etc. Override via CLI (`--graph-dir`, `--output-dir`, etc.).

Used as a standalone step (Kaggle) or as part of **end-to-end** pipeline.
