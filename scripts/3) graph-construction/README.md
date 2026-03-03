# 3) Graph-construction

Build **clause-level argument graphs** from annotated judgments: extract precedent citations, construct base + semantic graphs, and optionally train a GNN for embeddings.

## Contents

| File | Description |
|------|-------------|
| **extract_citations.py** | Reads `*.annotated.json` and extracts **precedent citations** from clause text (regex/patterns). Maps citations to document IDs. Optional **semantic matching** (InCaseLawBERT) to find target clause in cited doc. Writes **graph_citations.json** for use in inter-document edges. |
| **graph-construction.ipynb** | **Notebook:** Load annotated JSONs, build base graph (nodes = clauses, edges = label-based support/oppose + sequential). Add **semantic edges** (InCaseLawBERT similarity, top-k). Filter and transform edges. Optionally **add inter-document citation edges** from `graph_citations.json`. Output: `*.semantic.json` per document. |
| **gnn-training.ipynb** | **Notebook:** Train a GNN (HGT/RGCN/RGAT) on clause graphs for node embeddings; save encoder (`.pt`) for use in **clause-ranking**. |

## Data flow

- **Input:** Directory of annotated JSONs (`*.annotated.json`) with `doc_id` and `clauses` (each with `clause_id`, `text`, `label`).
- **extract_citations.py output:** `graph_citations.json` (citations by document, optional target clause IDs).
- **graph-construction.ipynb output:** Per-doc graph JSONs (`{doc_id}.semantic.json`) with `nodes`, `edges`, `doc_id`. Naming: `*.semantic.json`.
- **gnn-training.ipynb output:** Trained GNN encoder (`rgcn_encoder.pt`) for ranking.

## Config

- **extract_citations.py:** Set `ANNOTATED_DIRS`, `OUTPUT_FILE`, `USE_SEMANTIC_MATCHING`, `MODEL_NAME`.
- **graph-construction.ipynb:** Set `INPUT_DIR`, `OUT_DIR`, paths to BERT and citation file. Semantic edge threshold and top-k in `GraphConfig`.

Inter-document edges link a clause in one judgment to a cited clause in another; they are added in the notebook using `graph_citations.json` and the same `*.semantic.json` naming.
