# 2) Clause-label-trainer

Tools for **argument labeling** of legal clauses: manual annotation (Gradio UI), inference with a trained model, and training notebooks (weighted loss or upsampling).

## Contents

| File | Description |
|------|-------------|
| **tag_clauses_ui.py** | **Gradio UI** for manual clause tagging. Loads clause-split `.txt` (with `=== BODY ===`); displays one clause at a time; labels: Claim, Premise, Opposition, None. Saves per-doc JSON (e.g. `*.annotated.json`). Supports navigation, editing, splitting clauses, and resuming. |
| **predict_clauses.py** | **Inference** with a trained InCaseLawBERT-based classifier. Reads `*.clauses.json` (with prev/next context); writes `*.annotated.json` with predicted labels. Same tokenization as training (`[PREV_1] [PREV] [CURRENT]`). |
| **weighted-label-trainer.ipynb** | **Training** on annotated JSONs: document-level train/val/test split, **class-weighted** cross-entropy to handle label imbalance, transformer classifier. Outputs combined JSONL, splits, and trained model. |
| **upsampled-trainer.ipynb** | **Training** with **upsampled** minority classes so each class appears equally in the training set; document-level split; transformer classifier. |

## Data flow

- **Input (UI):** Clause-split text files (e.g. `*.clauses.txt` or any `.txt` with `=== BODY ===` and one clause per line).
- **Output (UI):** Per-document JSON with `doc_id`, `metadata`, `clauses` (each with `text`, `label`). Naming: `{doc_id}.annotated.json`.
- **Input (predict):** Directory of `*.clauses.json`.
- **Output (predict):** Directory of `*.annotated.json` (one per doc, with predicted `label` per clause).

## Config

- In **tag_clauses_ui.py**: set `DATA_DIR` and `OUTPUT_DIR` for your clause-split files and annotation output.
- In **predict_clauses.py**: set model path (e.g. InCaseLawBERT or fine-tuned), `input_dir`, `output_dir`, `batch_size`.

Training notebooks expect an **annotated folder** of JSONs (same structure as UI/predict output). Adjust `ANNOTATED_FOLDER` and paths in the first cells.
