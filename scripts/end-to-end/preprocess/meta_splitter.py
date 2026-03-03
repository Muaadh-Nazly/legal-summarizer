# LIBRARIES
import re
from pathlib import Path
import os
import re

# REGEX PATTERNS
DATE_VARIATIONS = [
    r"\d{1,2}(?:st|nd|rd|th)?\s*,?\s*(?:of\s+)?[A-Za-z]+,?\s*\d{4}",
    r"\d{1,2}\s*(?:st|nd|rd|th)?\s*(?:of\s+)?[A-Za-z]+,?\s*\d{4}",
    r"\d{1,2}(?:st|nd|rd|th)?\s*,?\s+[A-Za-z]+,?\s*\d{4}",
    r"\d{1,2}\.\d{1,2}\.\d{4}",
    r"\d{1,2}\s*[-./]\s*\d{1,2}(?:\s*[-./]\s*\d{2,4})?",
    r"\d{4}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{1,2}",
    r"\d{1,2}(?:st|nd|rd|th)?\s*[–\-]?\s*[A-Za-z]+\s*[–\-]?\s*\d{4}",
    r"((?:\d\s*){1,2}\s*\.\s*(?:\d\s*){1,2}\s*\.\s*(?:\d\s*){4})",
    r"\d{4}\s*[-./]\s*\d{1,2}\s*(?:[-./]\s*\d{0,2})?",
    r"\d{1,2}\s*[-./~]\s*\d{1,2}\s*[-./~]\s*\d{2,4}\.?",
    r"\d{4}\s*[–\-]\s*\d{1,2}\s*[–\-]\s*\d{1,2}",
    r"\d{1,2}\.\d{1,2}\.\d{4}",
    r"\b\d{4}\b",
    r"\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{4}\.?",
]

METADATA_PHRASES = [
    r"DECIDED\s*ON",
    r"Decided\s*on",
    r"Judgement\s*on",
    r"Decide\s*On",
    r"DECIDEN\s*ON",
    r"Decided\s*:?",
    r"DECIDEDON",
    r"ARGUED\s*&\s*(?:DECIDED)?(?:\s*ON)?",
    r"ARGUED\s*AND\s*(?:DECIDED)?(?:\s*ON)?",
    r"Judgment\s*(?:delivered\s*on|on)",
    r"Order\s*(?:Delivered\s*on|on)",
    r"Delivered\s*on",
    r"DATE\s*OF\s*JUDGMENT",
    r"Date",
    r"Judgment\s*(?:pronounced\s*on|delivered\s*on|on)",
    r"ARGUED\s*,?\s*DECIDED\s*(?:AND\s*JUDGMENT\s*PRONOUNCED\s*)?ON",
]

# Search only in first N chars after date to avoid matching end-of-doc
JUDGE_SEARCH_LIMIT = 120

# Justice "J" must be uppercase and followed by body start.
JUDGE_PATTERN = re.compile(
    r"(?:(?i:Judgement|Judgment)\s+)?"
    r"([A-Z][A-Za-z0-9\.\s\-',]{1,90})"  # greedy: full name
    r"(?:\s*[,.]?\s*(?:PC|P\.C\.?|PCJ)?\s*[,.]?\s*J\.?(?=[\s,.:]*[A-Z0-9])[\s,.:]*|\s*CJ(?=[\s,.:]*[A-Z0-9])[\s]*)"
)


# Phrases that are too generic must not match body text.
BODY_START_AFTER_PHRASE = re.compile(
    r"^\s*(?:to|and|the|when|by|it|that|which|in|as|for|an?|is|was|were|have|has|had)\s",
    re.IGNORECASE,
)


# SPLIT META AND BODY
def _find_phrase_and_date_end(text: str) -> int | None:
    """Return end index for first matching metadata phrase, else None."""
    for phrase in METADATA_PHRASES:
        pat = re.compile(rf"({phrase}\s*(?::\s*:|[:;\-–—])*\s*)", re.IGNORECASE)
        for m in pat.finditer(text):
            after_phrase = text[m.end() :]
            # Optional date
            for date_pat in DATE_VARIATIONS:
                date_m = re.match(rf"\s*{date_pat}", after_phrase)
                if date_m:
                    return m.end() + date_m.end()
            # No date
            after_strip = after_phrase.strip()
            if after_strip and (
                after_strip[0].islower() or BODY_START_AFTER_PHRASE.match(after_phrase)
            ):
                continue
            return m.end()
    return None


def _find_judge_in_tail(tail: str) -> re.Match | None:
    """First judge-name match in tail."""
    search = tail[:JUDGE_SEARCH_LIMIT]
    return JUDGE_PATTERN.search(search)


def split_metadata_body(text: str) -> tuple[str | None, str, list]:
    """
    Split into (meta, body, candidates).
    Primary: split after delivering judge name (after phrase+date).
    Fallback: split after phrase+date only.
    """
    end = _find_phrase_and_date_end(text)
    if end is None:
        return None, text, []

    tail = text[end:]
    judge_m = _find_judge_in_tail(tail)
    if judge_m:
        split_idx = end + judge_m.end()
        return (
            text[:split_idx].strip(),
            text[split_idx:].strip(),
            [{"split_after": "judge"}],
        )
    # Fallback: split after phrase+date
    return text[:end].strip(), text[end:].strip(), [{"split_after": "date"}]


def clean_body(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\s\.\*\-–—_•.·~]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


# PROCESS DIRECTORY
def process_file(
    input_file: Path,
    output_dir: Path | None = None,
    input_dir: Path | None = None,
) -> bool:
    """
    Process a single file: split meta/body and write one .metasplit.txt.
    Uses split_metadata_body() and clean_body().
    Returns True if metadata was detected, False otherwise.

    - If output_dir and input_dir are both given, output path is
      output_dir / relative_path / {stem}.metasplit.txt
    - Otherwise (single-file use), output is next to input_file with suffix .metasplit.txt.
    """
    text = input_file.read_text(errors="ignore")
    meta, body, _ = split_metadata_body(text)
    body_clean = clean_body(body) if body else ""
    orig_name = (
        input_file.stem.replace(".translated", "")
        if input_file.stem.endswith(".translated")
        else input_file.stem
    )
    if output_dir is not None and input_dir is not None:
        rel_parent = input_file.relative_to(input_dir).parent
        output_file = output_dir / rel_parent / f"{orig_name}.metasplit.txt"
    else:
        output_file = input_file.parent / f"{orig_name}.metasplit.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        if meta:
            f.write("=== METADATA ===\n" + meta + "\n\n")
        f.write("=== BODY ===\n" + body_clean + "\n")
    print(f"✅ {input_file.name}")
    return meta is not None


def process_dir(
    input_dir: Path, output_dir: Path, log_file: str = "unmatched_files_judge.log"
) -> None:
    """Process all .txt files in input_dir"""
    unmatched = []
    for path in input_dir.rglob("*.txt"):
        if not process_file(path, output_dir=output_dir, input_dir=input_dir):
            unmatched.append(str(path))
    if unmatched:
        log_path = output_dir / log_file
        log_path.write_text("\n".join(unmatched), encoding="utf-8")
        print(f"Logged {len(unmatched)} unmatched → {log_path}")


# JOIN BODY INTO SINGLE BLOCK
class BodyLineJoiner:
    def __init__(self):
        pass

    def join_body_lines(self, text: str) -> str:
        # Split at === BODY ===
        parts = re.split(r"(=== BODY ===)", text)
        new_parts = []

        i = 0
        while i < len(parts):
            part = parts[i]
            if part == "=== BODY ===" and i + 1 < len(parts):
                new_parts.append(part)
                body_content = parts[i + 1]
                # Join all lines into a single line
                body_content = " ".join(
                    line.strip() for line in body_content.splitlines() if line.strip()
                )
                new_parts.append(body_content)
                i += 2
            else:
                new_parts.append(part)
                i += 1

        return "\n".join(new_parts)

    def process_file(self, input_file: str, output_file: str):
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
        processed = self.join_body_lines(content)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(processed)
        print(f"Processed {input_file}")

    def process_directory(self, input_dir: str, output_dir: str):
        """Recursively process all .txt files in input_dir using self.process_file()."""
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for root, _, files in os.walk(input_dir):
            for filename in files:
                if not filename.lower().endswith(".txt"):
                    continue

                input_path = os.path.join(root, filename)

                # Preserve relative folder structure
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                self.process_file(input_path, output_path)

        print("🎉 All files processed!")


if __name__ == "__main__":
    input_dir = Path("/kaggle/input/d/muaadhnazly/translated/3 TRANSLATED")
    output_dir = Path("/kaggle/working/Meta Split")
    output_dir.mkdir(parents=True, exist_ok=True)
    process_dir(input_dir, output_dir)
    INPUT_DIR = "/kaggle/working/Meta Split"
    OUTPUT_DIR = "/kaggle/working/Body Joined"
    joiner = BodyLineJoiner()
    joiner.process_directory(INPUT_DIR, OUTPUT_DIR)
