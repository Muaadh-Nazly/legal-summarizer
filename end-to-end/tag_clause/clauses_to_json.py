"""
Convert .clauses.txt files (one clause per line) to per-document JSON files
Each clause gets prev_text, next_text, prev_text_1 for context.
"""

import json
from pathlib import Path
from typing import List, Dict


def extract_clauses_from_file(filepath: Path) -> List[str]:
    """Read clauses from a .clauses.txt file. Each non-empty line is a clause."""
    text = filepath.read_text(encoding="utf-8")
    if "=== BODY ===" in text:
        body = text.split("=== BODY ===", 1)[1].strip()
    else:
        body = text.strip()
    return [line.strip() for line in body.split("\n") if line.strip()]


def get_doc_id_from_path(filepath: Path) -> str:
    """Base doc_id for naming: doc_id.clauses.txt -> doc_id.
    Output will be {doc_id}.clauses.json."""
    stem = filepath.stem
    return stem


def clauses_dir_to_json_dir(clauses_dir: Path, output_dir: Path) -> int:
    """
    For each .clauses.txt in clauses_dir, build one JSON file in output_dir with
    doc_id and clauses (each with clause_id, text, prev_text, next_text, prev_text_1).
    Returns number of documents written.
    """
    clauses_dir = Path(clauses_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(clauses_dir.rglob("*.clauses.txt")):
        clauses = extract_clauses_from_file(path)
        if not clauses:
            continue
        doc_id = get_doc_id_from_path(path)
        clause_list = []
        for i, text in enumerate(clauses):
            clause_list.append(
                {
                    "clause_id": i + 1,
                    "text": text,
                    "prev_text": clauses[i - 1] if i > 0 else None,
                    "next_text": clauses[i + 1] if i < len(clauses) - 1 else None,
                    "prev_text_1": clauses[i - 2] if i > 1 else None,
                }
            )
        out_file = output_dir / f"{doc_id}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {"doc_id": doc_id, "clauses": clause_list},
                f,
                ensure_ascii=False,
                indent=2,
            )
        count += 1
    return count


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Clauses (.clauses.txt) to per-doc JSON for prediction"
    )
    p.add_argument(
        "clauses_dir", type=Path, help="Directory containing .clauses.txt files"
    )
    p.add_argument("output_dir", type=Path, help="Output directory for JSON files")
    args = p.parse_args()
    n = clauses_dir_to_json_dir(args.clauses_dir, args.output_dir)
    print(f"Wrote {n} JSON file(s) to {args.output_dir}")
