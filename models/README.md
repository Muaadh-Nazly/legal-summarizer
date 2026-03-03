# Models

This directory holds **saved model assets** used by the Legal-Summarizer pipeline. The repo may include config/tokenizer files; full weights are often stored elsewhere (Hugging Face, Kaggle datasets, or local paths) and referenced via **pipeline_config** or script configs.

## Expected contents

| Asset | Purpose |
|-------|--------|
| **InCaseLawBERT/** | Hugging Face–style model dir (e.g. `config.json`, `tokenizer.json`, `vocab.txt`, `special_tokens_map.json`). Used for: clause classification (Claim/Premise/Opposition/None), graph semantic embeddings, citation target matching. Can also be loaded by name: `law-ai/InCaseLawBERT`. |
| **nllb_sinhala2english_lora/** | LoRA adapter (and tokenizer) for NLLB (e.g. 1.3B) fine-tuned for Sinhala → English. Used by the **preprocess/translator** stage. Contains e.g. `adapter_config.json`, `adapter_*.safetensors`, `tokenizer.json`, `training_args.json`. |

## GNN encoder

The **clause-ranking** stage can use a trained GNN encoder (e.g. `.pt` file). That file is typically produced by **scripts/3) graph-construction/gnn-training.ipynb** and path set in **pipeline_config** (`GNN_MODEL_PATH`) or in the ranking script config.

## Usage

- **End-to-end pipeline:** Set `INCASELAWBERT_PATH`, `NLLB_PATH`, and `GNN_MODEL_PATH` in `scripts/end-to-end/pipeline_config.py` if your assets live under `models/` or another path.
- **Stage-by-stage scripts:** Each script or notebook has its own config (e.g. model path, Hugging Face model id); point them to this folder or to your own locations.

If this directory is empty or partial, clone or download the model weights and place them here, or set the corresponding paths in config to where the weights actually reside.
