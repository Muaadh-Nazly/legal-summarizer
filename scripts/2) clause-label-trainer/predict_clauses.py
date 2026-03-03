"""
Load trained InCaseLawBERT and predict labels for clauses.

Input: directory of per-doc JSON files.
Output: one JSON file per document in output_dir (doc_id, clauses with predicted_label).

Tokenization: [PREV_1] [PREV] [CURRENT] (same as training).

"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


MAX_LEN = 512
LABEL2ID = {"Premise": 0, "None": 1, "Opposition": 2, "Claim": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def tokenize_fn(batch: dict, tokenizer) -> dict:
    """Build [PREV_1] [PREV] [CURRENT] and tokenize."""
    texts = []
    n = len(batch["text"])
    for i in range(n):
        text = batch["text"][i]
        prev = batch.get("prev_text", [None] * n)[i]
        prev_1 = batch.get("prev_text_1", [None] * n)[i]
        parts = []
        if prev_1 and prev_1.strip():
            parts.append(f"[PREV_1] {prev_1.strip()}")
        if prev and prev.strip():
            parts.append(f"[PREV] {prev.strip()}")
        parts.append(f"[CURRENT] {text.strip()}")
        texts.append(" ".join(parts))
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )


class UnannotatedDataset(Dataset):
    """Dataset of raw clauses; tokenization done in collate_fn."""

    def __init__(self, clauses: list[dict]):
        self.clauses = clauses

    def __len__(self):
        return len(self.clauses)

    def __getitem__(self, idx):
        return self.clauses[idx]


def collate_and_tokenize(batch: list[dict], tokenizer) -> dict:
    """Turn a list of clause dicts into tokenized tensors."""
    batch_dict = {
        "text": [c["text"] for c in batch],
        "prev_text": [c.get("prev_text") for c in batch],
        "prev_text_1": [c.get("prev_text_1") for c in batch],
    }
    enc = tokenize_fn(batch_dict, tokenizer)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


def predict_one_doc(
    clauses: list[dict],
    model,
    tokenizer,
    device: str,
    batch_size: int,
) -> list[int]:
    """Predict labels for one document's clauses. Returns list of label IDs."""
    if not clauses:
        return []
    dataset = UnannotatedDataset(clauses)

    def collate_fn(batch_list):
        return collate_and_tokenize(batch_list, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy().tolist())
    return preds


def predict(
    model_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: int = 16,
    device: str | None = None,
) -> None:
    """
    For each JSON file in input_dir: load doc -> predict -> write one JSON to output_dir (file-by-file).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    import logging

    logging.getLogger("transformers").setLevel(logging.WARNING)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(LABEL2ID)
    )
    model.to(device)
    model.eval()
    print("Models loading done.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob("*.clauses.json"))
    print(f"Found {len(json_files)} documents in {input_dir}")
    print()

    total_clauses = 0
    label_counts = Counter()

    for doc_file in tqdm(json_files, desc="Predicting", unit="doc"):
        with open(doc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = data["doc_id"]
        base_id = (
            doc_id.removesuffix(".clauses") if doc_id.endswith(".clauses") else doc_id
        )
        clauses = data["clauses"]
        if not clauses:
            out = {"doc_id": base_id, "clauses": []}
        else:
            pred_ids = predict_one_doc(clauses, model, tokenizer, device, batch_size)
            for i, c in enumerate(clauses):
                label = ID2LABEL[pred_ids[i]]
                c["predicted_label"] = label
                label_counts[label] += 1
            total_clauses += len(clauses)
            out = {
                "doc_id": base_id,
                "clauses": [
                    {
                        "clause_id": c["clause_id"],
                        "text": c["text"],
                        "label": c["predicted_label"],
                        "prev_clause": c.get("prev_text"),
                        "next_clause": c.get("next_text"),
                    }
                    for c in clauses
                ],
            }
        out_file = output_path / f"{base_id}.annotated.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    # Summary
    print()
    print("=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Documents: {len(json_files)}")
    print(f"Total clauses: {total_clauses}")
    print()
    for label in ["Premise", "None", "Opposition", "Claim"]:
        count = label_counts.get(label, 0)
        pct = 100 * count / total_clauses if total_clauses else 0
        print(f"  {label:12} : {count:6} ({pct:5.2f}%)")
    print("=" * 60)

    summary = {
        "total_docs": len(json_files),
        "total_clauses": total_clauses,
        "by_label": {
            label: {
                "count": label_counts.get(label, 0),
                "pct": (
                    round(100 * label_counts.get(label, 0) / total_clauses, 2)
                    if total_clauses
                    else 0
                ),
            }
            for label in ["Premise", "None", "Opposition", "Claim"]
        },
    }
    summary_file = output_path / "_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    print(f"Per-doc predictions: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict labels for unannotated clauses"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory of per-doc JSONs from gather_unannotated_clauses.py",
    )
    parser.add_argument(
        "--output_dir",
        default="predicted_by_doc",
        help="Output directory (one JSON per document)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size per document"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PREDICT UNANNOTATED CLAUSES (file-by-file)")
    print("=" * 60)

    predict(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    print()
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
