# Scripts

This directory contains all pipeline stages, runnable **stage-by-stage** (numbered folders) .

## Layout

| Folder | Purpose |
|--------|--------|
| **1) pre-process** | PDF → extracted text → cleaned → translated → meta/body split → sentences → clauses. Jupyter notebooks per stage + NLLB fine-tuning. |
| **2) clause-label-trainer** | Manual annotation UI (Gradio), prediction script, and training notebooks (weighted / upsampled) for argument labels (Claim, Premise, Opposition, None). |
| **3) graph-construction** | Extract precedent citations from annotated JSONs; build clause argument graphs; add semantic edges; optional GNN training. |
| **4) clause-ranking** | Hybrid ranker (centrality, rhetoric, citations, embeddings, optional GNN) to select top-k clauses per document. |
| **5) finalization** | Abstractive summarization (BART), merge summary into PDF, and evaluate against SLR headnotes. |

## Recommended order (stage-by-stage)

1. Run **1) pre-process** notebooks in order (1 → 6) to get clause-split texts.
2. Run **2) clause-label-trainer** to annotate clauses (or use existing `.annotated.json`).
3. Run **3) graph-construction** to get citation JSON and semantic graphs.
4. Run **4) clause-ranking** on the graphs to get `.ranked.json` per document.
5. Run **5) finalization** to abstract and merge into PDFs, and optionally evaluate.


## File naming

- `*.extracted.txt` → `*.cleaned.txt` → `*.translated.txt` → `*.metasplit.txt` → `*.sentences.txt` → `*.clauses.txt` → `*.clauses.json` → `*.annotated.json` → `*.semantic.json` → `*.ranked.json` → `*.abstracted.json` → `*.final.pdf`

See each subfolder’s **README.md** for file lists and usage.
