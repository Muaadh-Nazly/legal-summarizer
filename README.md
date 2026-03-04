# Precedent-Aware Legal Argument Graphs for Automated Summaries

An automated system that finds and summarizes the key arguments in **Sri Lankan Supreme Court judgments**. It processes judgment PDFs (including multilingual English, Sinhala, and Tamil text), builds clause-level argument graphs with precedent citations, ranks the most important clauses, and produces structured summaries merged back into PDFs.

---

## Overview

The pipeline:

1. **Preprocess** - Extract text, Clean, Translate Sinhala/Tamil → English, Split metadata/body, Split sentences, then clauses.
2. **Annotate** - Manually or automatically label clauses (Claim, Premise, Opposition, None) for argument structure.
3. **Graph** - Build intra-document argument graphs (label + semantic edges), optionally add inter-document citation edges.
4. **Rank** - Score and select top clauses (hybrid ranker with adaptive top-k).
5. **Summarize** - Abstractive summarization over selected clauses; merge summary into original PDF.

The system can be run **stage-by-stage scripts** (notebooks and Python in `scripts/`) for development and training, or use the **end-to-end pipeline** (single Gradio UI) to go from PDF(s) to final summarized PDF(s).

---

## Repository Structure

```
Legal-Summarizer/
├── requirements.txt          # Root Python dependencies (general)
├── models/                   # Saved model assets (InCaseLawBERT, NLLB LoRA)
├── scripts/
│   ├── 1) pre-process/       # Stage 1: PDF → clauses
│   ├── 2) clause-label-trainer/  # Manual annotation UI + training notebooks
│   ├── 3) graph-construction/   # Citation extraction, graph building, GNN training
│   ├── 4) clause-ranking/       # Hybrid ranking (centrality, rhetoric, citations, GNN)
│   ├── 5) finalization/         # Abstractive summarization, PDF merge, SLR evaluation
└── end-to-end/             # Full pipeline (all stages) + Gradio UI
```

---

## Quick Start

- **Full pipeline (one shot):** Use `scripts/end-to-end/`. Install dependencies from `scripts/end-to-end/requirements.txt`, set model paths in `pipeline_config.py`, then run `end-to-end-pipeline.ipynb` (Gradio UI) or call the pipeline stages in code.
- **Stage by stage:** Run the numbered folders in order: pre-process → clause-label-trainer → graph-construction → clause-ranking → finalization. See each folder’s `README.md` for details.

---

## Models

- **InCaseLawBERT** - BERT model for clause embeddings and sequence classification (argument labels). Used in prediction, graph construction, and citation matching.
- **NLLB-1.3B + LoRA** - Translation (Sinhala/Tamil → English); optional fine-tuned adapter in `models/nllb_sinhala2english_lora/`.
- **GNN** — Optional encoder for clause graph embeddings used in ranking (trained in `scripts/3) graph-construction/`).

Place model weights in `models/` or set paths in `pipeline_config.py` (end-to-end) or the relevant script configs.

---

## Requirements

- Python 3.10+ recommended. See `requirements.txt` (root) and `scripts/end-to-end/requirements.txt` for the full pipeline.
- GPU recommended for extraction (Surya), translation, BERT, GNN, and abstractive summarization.

---

## Citation & License

Part of research on precedent-aware legal summarization for Sri Lankan Supreme Court judgments. See individual scripts and notebooks for any embedded references or licenses.
