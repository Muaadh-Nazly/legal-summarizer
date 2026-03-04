# 1) Pre-process

Preprocessing pipeline for Sri Lankan Supreme Court judgment PDFs: **extract → clean → translate → meta/body split → sentence split → clause split**. Each stage is implemented as a Jupyter notebook (and can be run on Kaggle or locally).

## Stages (run in order)

| # | Notebook | Input | Output | Description |
|---|----------|--------|--------|-------------|
| 1 | **1-extract-text-ocr.ipynb** | PDFs | `*.extracted.txt` | Extract text with pdfminer/pdfplumber for clean pages. fallback to **Surya OCR** (English, Sinhala, Tamil) for garbled text. |
| 2 | **2-clean-extracted-texts.ipynb** | `*.extracted.txt` | `*.cleaned.txt` | Remove headers, footers, page numbers, boilerplate. |
| 3 | **3-translation-using-1.3B.ipynb** | `*.cleaned.txt` | `*.translated.txt` | Translate Sinhala/Tamil blocks to English using NLLB  with fine-tuned LoRA. |
| 4 | **4-meta-body-split-joiner.ipynb** | `*.translated.txt` | `*.metasplit.txt` | Split metadata vs body; join body lines for sentence splitting. |
| 5 | **5-sentences-splitter.ipynb** | `*.metasplit.txt` | `*.sentences.txt` | Rule-based sentence splitting (brackets, enumerations, legal phrasing). |
| 6 | **6-clause-split.ipynb** | `*.sentences.txt` | `*.clauses.txt` | Split sentences into clauses at idea markers (one clause per line). |

## Extra

- **fine-tune-nllb 1.3b.ipynb** - Fine-tune NLLB-200-1.3B (e.g. Sinhala → English) and save adapter/tokenizer for use in Stage 3.

## Dependencies

- Stage 1: `surya-ocr`, `pdfplumber`, `pdf2image`, `pdfminer.six`, `Pillow`, `torch`, `transformers`; system `poppler-utils`.
- Stages 2-6: `regex`, `spacy` (e.g. `en_core_web_trf` for clause splitting).
- Stage 3 / fine-tune: `peft`, `accelerate`, `safetensors`, `transformers`.

Outputs follow pipeline naming (e.g. `*.extracted.txt`, `*.cleaned.txt`, …).
