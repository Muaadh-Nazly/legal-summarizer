# End-to-end pipeline

Single codebase for the **full pipeline**: PDF to summarized PDF (extract, clean, translate, meta/sentence/clause split, clauses→JSON, predict labels, build graph, rank, abstract, merge). Driven by **Gradio UI** in the main notebook or by calling stages in code.

## Main files

- **end-to-end-pipeline.ipynb** — Install dependencies, set workspace, optionally run stages 1–12, launch Gradio (upload PDFs, Run, log, download zip of final PDFs).
- **pipeline_config.py** — Optional paths: `NLLB_PATH`, `INCASELAWBERT_PATH`, `GNN_MODEL_PATH`.
- **requirements.txt** — Full pipeline dependencies (torch, transformers, surya-ocr, peft, torch-geometric, gradio, etc.).

## Directories

- **preprocess/** — Stages 1–6: pdf_extraction, clean_text, translator, meta_splitter, sentence_splitter, clause_splitter. Outputs: `.extracted.txt` → `.cleaned.txt` → `.translated.txt` → `.metasplit.txt` → `.sentences.txt` → `.clauses.txt`.
- **tag_clause/** — clauses_to_json (→ `.clauses.json`), predict_clauses (→ `.annotated.json`).
- **graph/** — construct_graph: annotated JSONs → base + semantic edges → `*.semantic.json`.
- **ranking/** — rank_clauses + hybrid_ranker, adaptive_topk, centrality, rhetoric, citations, GNN, postprocessing → `*.ranked.json`.
- **abstraction_merge/** — abstraction (BART → `*.abstracted.json`), merge (PDF + abstracted → `*.final.pdf`).

## Running

1. Install from **requirements.txt** (restart kernel if on Kaggle).
2. Set workspace and input in the notebook; set NLLB/InCaseLawBERT/GNN paths in **pipeline_config** if needed.
3. Run notebook and use Gradio to upload PDF(s) and run, or run stages in order in code.

Outputs use pipeline naming under the workspace ( 01_extracted … 13_final_pdf). UI provides zip download of final PDFs.
