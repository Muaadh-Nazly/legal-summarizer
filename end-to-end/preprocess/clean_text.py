import os
import re
import regex as re
from collections import Counter
from pathlib import Path

MAX_WORDS = 20


# INTIALIZE REGEX VARIABLES
# Patterns for page numbers and inline fragments
INLINE_PAGE_PATTERNS = [
    r"^\s*Page\s*\d+(\s*of\s*\d+)?\s*$",
    r"^\s*\d+\s+of\s+\d+\s*$",
    r"^\s*Page\s*\d+\s*$",
    r"^\s*\d+\s*page\s*$",
    r"(?<!\w)^\s*\d+\s+(?=$)",
    r"(?<!\w)^\s*\d+\s+(?=\b[A-Z])$",
    r"^\s*of\s+\d{1,3}\s*[-–—:]?\s*$",
    r"\s*[-–—:]?\s*of\s+\d{1,3}\s*$",
    r"^\s*Page\s*\d+(\s*of\s*\d+)?\s+",
]

# Pure page lines
ENTIRE_LINE_PATTERNS = [
    r"^\s*Page\s*\d+(\s*of\s*\d+)?\s*$",
    r"^\s*\d+\s+of\s+\d+\s*$",
    r"^\s*\d{1,3}\s*$",
]

entire_line_regexes = [re.compile(p, re.IGNORECASE) for p in ENTIRE_LINE_PATTERNS]
inline_regexes = [re.compile(p, re.IGNORECASE) for p in INLINE_PAGE_PATTERNS]

# Legal-keywords should be preserved
LEGAL_KEYWORDS = re.compile(
    r"\b(v\.|vs\.|Petitioner|Respondent|Court|Judge|Order|Judgment|Applicant|Respondent|Plaintiff|Defendant)\b",
    re.IGNORECASE,
)

PAGE_FRAGMENT_REGEX = re.compile(
    r"""
    (?:                # non-capturing group for the whole fragment
        (?:[-–—]?\s*)? # optional dash or space before
        (?:Page|Pg|P\.?)\s*\d+(?:\s*(?:of)\s*\d+)?   # "Page 1", "Pg 1 of 6"
        |
        (?:^|\s)\d+\s*(?:of)\s*\d+(?=\s|$)           # "1 of 6" or "2/10"
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

LATEX_HTML_CLEANER = re.compile(
    r"""
    (\\[a-zA-Z]+(\{[^{}]*\})*)        # LaTeX commands like \frac{..}, \mathbf{..}, \mathsf{..}
    |(\{\\[a-zA-Z]+\})                # Wrapped LaTeX macros
    |(\^\{\\[a-zA-Z]+\})              # Superscripted macros like ^{\mathrm{st}}
    |(\^\{[^\}]+\})                   # Superscript content like ^{th}, ^{rd}
    |(<[^>]+>)                        # HTML tags like <b>, <i>, etc.
    |(&[a-zA-Z]+;)                    # HTML entities like &nbsp;
    """,
    re.VERBOSE,
)

LEADING_NUMERIC_EDGE = re.compile(
    r"^\s*(?:\d{1,3}(?![A-Za-z])\s*[-–—:\)]*\s*|\d{1,3}\s+of\s+\d{1,3}\s*[-–—:\)]*\s*)",
    re.IGNORECASE,
)
TRAILING_NUMERIC_EDGE = re.compile(
    r"(?:[-–—:\(]*\s*\d{1,3}\s*|\s*\d{1,3}\s+of\s+\d{1,3}\s*)\s*$", re.IGNORECASE
)
LEADING_OF_N = re.compile(r"^\s*of\s+\d{1,3}\s*[-–—:]?\s*", re.IGNORECASE)
TRAILING_OF_N = re.compile(r"\s*[-–—:]?\s*of\s+\d{1,3}\s*$", re.IGNORECASE)

ENUMERATION_PATTERN = re.compile(r"^\(?\d{1,3}[\.\)]\s")


# REMOVE HTML / LATEX TAGS
# simple html clean for normalization
def strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_latex_html(text: str) -> str:
    """
    Removes LaTeX and HTML formatting commands but keeps meaningful text inside braces.
    Example: '\\textbf{Order}' -> 'Order'
    """
    text = text.replace("\\n", "\n")
    # Remove HTML tags and entities
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)

    # Replace LaTeX-style commands with their inner content
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)

    # Simplify by removing the command names and braces while preserving letters/numbers inside
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}]", " ", text)

    # Superscript/subscript cleanup
    text = re.sub(r"\^\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"_\{([^{}]*)\}", r" \1 ", text)

    # Escape sequences and double slashes
    text = re.sub(r"\\\\", " ", text)
    text = re.sub(r"\\", " ", text)

    # Normalize spaces and quotes
    text = re.sub(r'["“”‘’]+', '"', text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


# REMOVE PAGE NUMBERS
# remove leading/trailing small numeric tokens (1-3 digits) and small "x of y" at edges


def _leading_token(line):
    m = re.match(r"^\s*([^\s]+)(.*)$", line, flags=re.DOTALL)
    if not m:
        return None, line
    return m.group(1), m.group(2)


def _trailing_token(line):
    m = re.match(r"^(.*?)([^\s]+)\s*$", line, flags=re.DOTALL)
    if not m:
        return None, line
    return m.group(2), m.group(1)


def strip_edge_numbers(line):
    """Safely strip page-like numbers from start/end while protecting enumerations and years."""
    if re.match(r"^\s*\d+\/[A-Za-z0-9]", line):
        return line
    if not line:
        return line

    orig = line

    if ENUMERATION_PATTERN.match(line):
        tok, head = _trailing_token(line)
        if tok:
            if len(re.sub(r"\D", "", tok)) >= 4:
                return line.strip()
            ln = re.sub(TRAILING_NUMERIC_EDGE, "", line).strip()
            return ln

    lead_tok, rest = _leading_token(line)
    if lead_tok:
        lead_digits = re.sub(r"\D", "", lead_tok)
        if len(lead_digits) >= 4:
            ln = re.sub(TRAILING_NUMERIC_EDGE, "", line).strip()
            return ln

    trail_tok, head = _trailing_token(line)
    if trail_tok:
        trail_digits = re.sub(r"\D", "", trail_tok)
        if len(trail_digits) >= 4:
            ln = re.sub(LEADING_NUMERIC_EDGE, "", line).strip()
            return ln

    line = re.sub(r"^\s*Page\s*\d+\s*(?:of\s*\d+)?\s*", "", line, flags=re.IGNORECASE)

    ln = re.sub(LEADING_NUMERIC_EDGE, "", line)
    ln = re.sub(TRAILING_NUMERIC_EDGE, "", ln)
    ln = re.sub(LEADING_OF_N, "", ln)
    ln = re.sub(TRAILING_OF_N, "", ln)
    return ln.strip()


# Keep semantically important content
def keep_line_safely(original_line, candidate_line_after_removal):
    if LEGAL_KEYWORDS.search(original_line):
        return original_line.strip()

    if not candidate_line_after_removal.strip():
        if len(re.findall(r"[A-Za-z]", original_line)) >= 3:
            return original_line.strip()
        return ""
    return candidate_line_after_removal.strip()


# REMOVE HEADERS AND FOOTERS
def detect_common_prefix(lines, max_words=20, threshold=0.6):
    """
    Detects a common header prefix based on word frequency.
    Ignores variations of 'Page'.
    Stops as soon as a word fails the threshold.
    """
    split_lines = [l.split()[:max_words] for l in lines if l.strip()]
    if not split_lines:
        return []

    prefix = []
    for i in range(max_words):
        ith_words = [
            words[i]
            for words in split_lines
            if len(words) > i and not re.match(r"(?i)^pages?$", words[i])
        ]
        if not ith_words:
            break

        most_common, count = Counter(ith_words).most_common(1)[0]
        if count / len(split_lines) >= threshold:
            prefix.append(most_common)
        else:
            break

    return prefix


def detect_common_suffix(lines, max_words=20, threshold=0.6):
    """
    Detects a common footer suffix based on word frequency.
    Ignores variations of 'Page'.
    """
    split_lines = [l.split()[-max_words:] for l in lines if l.strip()]
    if not split_lines:
        return []

    suffix_counts = Counter()
    total_lines = len(split_lines)

    for words in split_lines:
        # Filter out 'Page' variants before suffix analysis
        words = [w for w in words if not re.match(r"(?i)^pages?$", w)]
        for i in range(1, len(words) + 1):
            suffix = tuple(words[-i:])
            suffix_counts[suffix] += 1

    candidates = [
        (suffix, count)
        for suffix, count in suffix_counts.items()
        if count / total_lines >= threshold
    ]

    if not candidates:
        return []

    best_suffix = max(candidates, key=lambda x: (len(x[0]), x[1]))[0]
    return list(best_suffix)


# PROCESS FILES AND DIRECTORY
# Main Cleaning pipeline
def clean_file(path_in, path_out):
    """
    main cleaning pipeline which removes headers and footers.
    runs two full passes to ensure headers and footers are removed.
    """
    with open(path_in, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]
    norm_lines = [strip_html(ln) for ln in raw_lines]

    preclean = []
    for orig, ln in zip(raw_lines, norm_lines):
        if any(rx.match(ln.strip()) for rx in entire_line_regexes):
            preclean.append("")
        else:
            ln2 = strip_edge_numbers(ln)
            preclean.append(ln2.strip())

    nonempty = [l for l in preclean if l and l.strip()]
    middle_for_detection = nonempty[1:] if len(nonempty) > 2 else nonempty

    HEADER_THRESHOLD = 0.55
    FOOTER_THRESHOLD = 0.55
    header_words = detect_common_prefix(
        middle_for_detection, MAX_WORDS, HEADER_THRESHOLD
    )
    footer_words = detect_common_suffix(
        middle_for_detection, MAX_WORDS, FOOTER_THRESHOLD
    )
    header_str = " ".join(header_words).strip() if header_words else ""
    footer_str = " ".join(footer_words).strip() if footer_words else ""

    working = raw_lines.copy()

    # Run two full passes: each pass removes header/footer anchors + inline/edge page numbers
    for pass_i in range(2):
        cleaned = []
        for orig in working:
            ln = strip_html(orig)
            ln = strip_latex_html(ln)

            if header_str:
                ln = re.sub(
                    rf"^\s*{re.escape(header_str)}\s*[-–—:]*\s*",
                    "",
                    ln,
                    flags=re.IGNORECASE,
                )
                ln = re.sub(
                    rf"\b{re.escape(header_str)}\b", "", ln, flags=re.IGNORECASE
                )

            if footer_str:
                ln = re.sub(
                    rf"\s*[-–—:]*\s*{re.escape(footer_str)}\s*$",
                    "",
                    ln,
                    flags=re.IGNORECASE,
                )
                ln = re.sub(
                    rf"\b{re.escape(footer_str)}\b", "", ln, flags=re.IGNORECASE
                )

            ln = strip_edge_numbers(ln)
            ln = re.sub(r"\s{2,}", " ", ln).strip()

            if any(rx.match(ln.strip()) for rx in entire_line_regexes):
                kept = keep_line_safely(orig, "")
                if kept:
                    cleaned.append(kept)
                continue

            # if line is empty after removals, maybe keep original if it looks important
            if not ln:
                kept = keep_line_safely(orig, ln)
                if kept:
                    cleaned.append(kept)
                continue

            cleaned.append(ln)
        working = cleaned

    final_out = []
    for ln in working:
        final_out.append(ln)

    with open(path_out, "w", encoding="utf-8") as out:
        out.write("\n".join(final_out))


# PROCESS DIRECTORY
def process_directory_recursive(input_dir, output_dir):
    """Recursively process all .txt files in input_dir using clean_file()."""
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue

            in_p = os.path.join(root, fname)

            rel_path = os.path.relpath(in_p, input_dir)

            base_name, ext = os.path.splitext(os.path.basename(rel_path))
            stem = (
                base_name.replace(".extracted", "")
                if base_name.endswith(".extracted")
                else base_name
            )
            new_name = f"{stem}.cleaned.txt"
            out_p = os.path.join(output_dir, os.path.dirname(rel_path), new_name)

            # Ensure subdirectories exist
            os.makedirs(os.path.dirname(out_p), exist_ok=True)

            clean_file(in_p, out_p)
