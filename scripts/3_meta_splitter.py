import os
import re

input_dir = 'Cleaned Text'
output_dir = 'Meta Splitted'
os.makedirs(output_dir, exist_ok=True)

unmatched_log_path = os.path.join(output_dir, 'unmatched_files.txt')
unmatched_files = []

# Regex for "Decided On"
date_pattern = re.compile(
    r'(?P<label>DECIDED\s+ON)\s*[:\-–—]?\s*'
    r'(?P<date>'
        r'\d{1,2}(?:st|nd|rd|th)?(?:\s+of)?\s+[A-Za-z]+,?\s*\d{4}'
        r'|\d{1,2}\s*[-./,]\s*\d{1,2}(?:\s*[-./,]\s*\d{2,4})?'
        r'|\d{4}\s*[-./,]\s*\d{1,2}\s*[-./,]\s*\d{1,2}'
    r')',
    flags=re.IGNORECASE
)

# Regex for "Argued & Decided"
argued_decided_pattern = re.compile(
    r'(ARGUED\s*&(?:\s*DECIDED)?(?:\s*ON)?\s*[:\-–—]?\s*)'
    r'('
        r'\d{1,2}(?:st|nd|rd|th)?(?:\s+of)?\s+[A-Za-z]+,?\s*\d{4}'
        r'|\d{1,2}\s*[-./,]\s*\d{1,2}(?:\s*[-./,]\s*\d{2,4})?'
        r'|\d{4}\s*[-./,]\s*\d{1,2}\s*[-./,]\s*\d{1,2}'
    r')',
    flags=re.IGNORECASE
)

for filename in os.listdir(input_dir):
    if not filename.endswith('.txt'):
        continue

    filepath = os.path.join(input_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    match = date_pattern.search(text)
    if not match:
        match = argued_decided_pattern.search(text)

    if match:
        split_idx = match.end()
        metadata = text[:split_idx].strip()
        body = text[split_idx:].strip()

        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write("=== METADATA ===\n")
            f.write(metadata + "\n\n")
            f.write("=== BODY ===\n")
            f.write(body + "\n")
    else:
        unmatched_files.append(filename)
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write("=== BODY ===\n")
            f.write(text)

# Log unmatched
if unmatched_files:
    with open(unmatched_log_path, 'w', encoding='utf-8') as log_file:
        for file in unmatched_files:
            log_file.write(file + '\n')
    print(f"⚠️ {len(unmatched_files)} file(s) did not contain 'Decided On' or 'Argued & Decided' - see {unmatched_log_path}")
else:
    print("🎉 All files matched and split correctly.")

print("✅ Done: Metadata and body split (only first match used).")
