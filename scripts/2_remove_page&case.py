import os
import re

# Input and output paths
input_dir = 'Extracted Text'       # Your raw input
output_dir = 'Cleaned Text'      # Final cleaned output

# Create output dir if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# === Page Number Removal Patterns ===
page_number_patterns = [
    r'^\s*\d+\s*$',
    r'^\s*\d+\s+of\s+\d+\s*$',
    r'^\s*Page\s+\d+\s+of\s+\d+\s*$',
    r'^\s*Page\s+\d+\s*$',
    r'^\s*\d+\s+page\s*$',
]

compiled_page_patterns = [re.compile(p, re.IGNORECASE) for p in page_number_patterns]

def is_page_number_line(line):
    line = line.strip()
    return any(p.match(line) for p in compiled_page_patterns)

inline_page_cleanup = [
    (r'\bPage\s+\d+\s+of\s+\d+\b', ''),
    (r'\bPage\s+\d+\s+of\b', ''),
]

# === Case Info Removal Patterns ===
case_removal_patterns = [
    (r'^\s*(SC\.?\s*Appeal\s+[0-9A-Za-z/& ]+?)(-?\s*JUDGMENT)?\s*$', ''),
    (r'^\s*(SC\.?\s*Appeal\s+[0-9A-Za-z/& ]+?)(-?\s*JUDGMENT)?\s+', ''),
    (r'\s+(SC\.?\s*Appeal\s+[0-9A-Za-z/& ]+?)(-?\s*JUDGMENT)?\s*$', ''),
    (r'^\s*(S\.?C\.?|SC)[/\.]?\s*Appeal(?:\s*No\.?|:)?\s*[\[\(]?\d+[A-Za-z/]*/\d+\]?[-–]?\s*(JUDGMENT|JUDGEMENT)?\s*$', ''),
    (r'^\s*(\[\s*)?(S\.?C\.?|SC)[/\.]?\s*Appeal(?:\s*No\.?|:)?\s*[\[\(]?\d+[A-Za-z/]*/\d+\]?[-–]?\s*(JUDGMENT|JUDGEMENT)?\s*-?\s*', ''),
    (r'\s*(\[\s*)?(S\.?C\.?|SC)[/\.]?\s*Appeal(?:\s*No\.?|:)?\s*[\[\(]?\d+[A-Za-z/]*/\d+\]?[-–]?\s*(JUDGMENT|JUDGEMENT)?\s*$', ''),
    (r'^\s*/\s*20\d{2}\s+', ''),  # floating "/ 2015"
    (r'\s*/\s*20\d{2}\s*$', ''),

    # Extended patterns:
    (r'^\s*SC[/\.]?(?:APPEAL|SPL|SPL/LA)?[/\.]?[A-Z]*[/\.]?\s*\d{1,4}[/\.]?\d{2,4}[-–]?\s*', ''),
    (r'\s*SC[/\.]?(?:APPEAL|SPL|SPL/LA)?[/\.]?[A-Z]*[/\.]?\s*\d{1,4}[/\.]?\d{2,4}[-–]?\s*$', ''),

    (r'^\s*S\s*C\s*/\s*A\s*P\s*P\s*E\s*A\s*L\s*/\s*\d{1,4}\s*/\s*\d{2,4}\s*', ''),
    (r'\s*S\s*C\s*/\s*A\s*P\s*P\s*E\s*A\s*L\s*/\s*\d{1,4}\s*/\s*\d{2,4}\s*$', ''),

    (r'^\s*[\[\(]\s*SC(?:/|\.|\.SPL\.LA\.?|/SPL/LA)?\s*[\w/\.]*\s*\d{1,4}[/\.]?\d{2,4}\s*[\]\)]\s*[-–]?\s*', ''),
    (r'\s*[\[\(]\s*SC(?:/|\.|\.SPL\.LA\.?|/SPL/LA)?\s*[\w/\.]*\s*\d{1,4}[/\.]?\d{2,4}\s*[\]\)]\s*[-–]?\s*$', ''),

    (r'^\s*SC(?:\.SPL\.LA\.?|\.APPEAL)?\.?\s*\d{1,4}[/\.]?\d{2,4}[-–]?\s*', ''),
    (r'\s*SC(?:\.SPL\.LA\.?|\.APPEAL)?\.?\s*\d{1,4}[/\.]?\d{2,4}[-–]?\s*$', ''),
    (r'^\s*S\s*C\s*/\s*A\s*P\s*P\s*E\s*A\s*L\s*/\s*(?:\d\s*){1,4}/\s*(?:\d\s*){2,4}\s*', ''),
]

# === Line Cleaning Function ===
def clean_line(line):
    line = line.strip()

    # Remove case references
    for pattern, repl in case_removal_patterns:
        line = re.sub(pattern, repl, line, flags=re.IGNORECASE)

    # Remove if it's a standalone page number line
    if is_page_number_line(line):
        return '\n'

    # Remove inline "Page X of Y" fragments
    for pattern, repl in inline_page_cleanup:
        line = re.sub(pattern, repl, line, flags=re.IGNORECASE)

    # Remove leading numbers (page line prefix)
    line = re.sub(r'^\s*\d{1,3}(?![\.\d])\s+', '', line)

    # Remove trailing number-only tokens
    line = re.sub(r'\s\d{1,3}$', '', line)

    return line + '\n'

# === File Processing ===
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # Run cleaning twice for safety
        for _ in range(2):
            lines = [clean_line(line) for line in lines]

        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines)

print("✅ Page numbers and case references cleaned (double-pass). Files saved in:", output_dir)

