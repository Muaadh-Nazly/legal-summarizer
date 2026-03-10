# LIBRARIES
import re
from pathlib import Path
import os

# CLAUSE SPLIT MARKERS
# Strong idea markers which create independent clauses when split
IDEA_MARKERS = [
    "however,",
    "furthermore,",
    "moreover,",
    "in addition,",
    "additionally,",
    "in contrast,",
    "as a result,",
    "therefore,",
    "nevertheless",
    "nonetheless",
    "hence",
    "thus",
    "but",
    "meanwhile,",
    "similarly,",
    "likewise,",
    "rather,",
    "on the other hand,",
    "on the contrary,",
    "in conclusion,",
    "in summary,",
    "for example,",
    "for instance,",
    "in other words,",
    "provided that",
    "assuming that",
    "given that",
    "whereas",
    "even though",
    "although,",
]


# SPLIT CLAUSE
def clause_split_nlp(text):
    clauses = []
    for marker in IDEA_MARKERS:
        index = text.lower().find(marker)
        if index > 0:
            before_text = text[:index].strip()
            after_text = text[index + len(marker) :].strip()
            # Split only if at least 3 words before and 3 words after the marker
            if len(before_text.split()) >= 3 and len(after_text.split()) >= 3:
                clauses.append(before_text)
                text = text[index:].strip()
    clauses.append(text.strip())
    return clauses


# PROCESS FILES AND DIRECTORY
def process_file(input_path: Path, output_path: Path):
    """Read text, split into clauses, and write to output file."""
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.readlines()

    all_clauses = []
    for line in content:
        line = line.strip()
        if not line:
            continue
        clauses = clause_split_nlp(line)
        all_clauses.extend(clauses)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, clause in enumerate(all_clauses, 1):
            f.write(f"{clause}\n")

    print(f"✅ Processed {len(all_clauses)} clauses written to: {output_path}")


def process_all_files(input_dir: Path, output_dir: Path, file_extension=".txt"):
    """Process all files in a directory and output clauses."""
    input_files = list(input_dir.rglob(f"*{file_extension}"))
    print(f"📂 Found {len(input_files)} files with extension '{file_extension}'...")

    for input_file in input_files:
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path

        output_file = Path(str(output_file).replace(".sentences.txt", ".clauses.txt"))

        process_file(input_file, output_file)

    print("🎉 All files processed.")


if __name__ == "__main__":
    input_dir = Path("/kaggle/input/d/muaadhnazly/sentence-split/Sentence Splitted")
    output_dir = Path("/kaggle/working/CLAUSE SPLIT")
    os.makedirs(output_dir, exist_ok=True)
    process_all_files(input_dir, output_dir)
