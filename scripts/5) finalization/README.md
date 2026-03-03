# 5) Finalization

**Abstractive summarization** of ranked clauses and **merge** of the summary into the original PDF. Optional **evaluation** against SLR (headnote) references.

## Contents

| File | Description |
|------|-------------|
| **abstractive-summarizer.ipynb** | **Notebook:** Load `*.ranked.json`; extract clause texts in doc order; run **BART** (`facebook/bart-large-cnn`) per clause (with fallback to original if output is bad); optional third-person conversion. Builds summary with Key reasoning (bullets) + Key holding (disposition). Writes `*.abstracted.json` per document. |
| **pdf_preserve_summarize.py** | **Merge:** For each `*.abstracted.json`, find the matching PDF by stem; detect metadata/body split in PDF (meta_splitter); append summary after metadata (with optional OCR for image-only pages). Writes `*.final.pdf` to an output dir. |
| **evaluate_against_slr.py** | **Evaluation:** Compare system summaries to **headnote** references. Reference = subject + facts + Held + Cases referred to from headnote `.txt`; candidate = Key reasoning from `*.abstracted.json`. Metrics: ROUGE-1/2/L, BLEU-4, BERTScore, compression ratio, precedent coverage. Output: `evaluation_results/report_rouge.json` (and CSV/pairs). |

## Data flow

- **Input (abstractive):** Directory of `*.ranked.json` (from clause-ranking).
- **Output (abstractive):** Directory of `*.abstracted.json` (summary text + structure).
- **Input (merge):** `SUMMARY_JSON_DIR` with `*.abstracted.json`; `PDF_DIR` with source PDFs (stems must match).
- **Output (merge):** Directory of final PDFs (`*.final.pdf`).
- **Input (evaluate):** Headnote `.txt` files and matching `*.abstracted.json`; paths set in script (e.g. `HEADNOTES_DIR`, `ABSTRACTIVE_DIR`).

## Config

- **abstractive-summarizer.ipynb:** Set `RANKED_DIR`, `OUT_DIR`, BART model id, device.
- **pdf_preserve_summarize.py:** Set `INPUT_ROOT`, `PDF_DIR`, `SUMMARY_JSON_DIR`, `OUTPUT_DIR`; ensure meta_splitter is importable or set `META_SPLITTER_PATH`.
- **evaluate_against_slr.py:** Set `HEADNOTES_DIR`, `ABSTRACTIVE_DIR`, `RESULTS_DIR`; headnote stem patterns (SC_CA_, SC_FR_, etc.) for matching abstracted files.
