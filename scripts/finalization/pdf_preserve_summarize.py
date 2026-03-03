import fitz
import json
import re
import importlib.util
from pathlib import Path
import sys


import pytesseract
from PIL import Image

_OCR_AVAILABLE = True

INPUT_ROOT = Path("/kaggle/input/datasets/muaadhnazly/")
OUTPUT_ROOT = Path("/kaggle/working")

# Meta splitter script
META_SPLITTER_PATH = INPUT_ROOT / "full-pipeline/preprocess/meta_splitter.py"
# Data folders
PDF_DIR = INPUT_ROOT / "supreme-court"
SUMMARY_JSON_DIR = (
    INPUT_ROOT / "context-rich-abstractive-summary/Rich Context Sctractive Summary"
)
OUTPUT_DIR = OUTPUT_ROOT / "Final Pdfs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import splitter
_spec = importlib.util.spec_from_file_location(
    "meta_splitter_judge",
    META_SPLITTER_PATH,
)
_meta_splitter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_meta_splitter)
split_metadata_body = _meta_splitter.split_metadata_body


def find_split_in_pdf(doc):
    """
    Extract full text from PDF and find split index using meta split logic.
    Returns:
        split_index (global char index),
        split_page (1-based),
        page_texts (list[str]),
        meta (str or None) – metadata text up to and including judge name
    """
    page_texts = []
    for page in doc:
        page_texts.append(page.get_text())

    full_text = "\n".join(page_texts)
    meta, body, _ = split_metadata_body(full_text)

    if not meta:
        return None, None, None, None

    split_index = len(meta)
    cumulative = 0
    for i, pt in enumerate(page_texts):
        page_end = cumulative + len(pt) + 1
        if split_index <= page_end:
            return split_index, i + 1, page_texts, meta
        cumulative = page_end

    return split_index, len(page_texts), page_texts, meta


# OCR fallback for image-only PDFs
OCR_DPI = 150


def _page_to_pil(page, dpi=150):
    """Render a fitz page to a PIL Image for OCR."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def _get_page_texts_via_ocr(doc):
    """Extract text from each page using Tesseract OCR. Returns list of strings."""
    if not _OCR_AVAILABLE:
        return []
    page_texts = []
    for page in doc:
        img = _page_to_pil(page, OCR_DPI)
        text = pytesseract.image_to_string(img).strip()
        page_texts.append(text)
    return page_texts


def _y_crop_from_ocr_bbox(page, meta, offset_in_page, dpi=150):
    """
    Use OCR word boxes on the split page to find the y of the end of meta.
    Returns y in PDF coordinates (bottom of the word that contains offset_in_page).
    """
    if not _OCR_AVAILABLE or not meta:
        return None
    img = _page_to_pil(page, dpi)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n = len(data["text"])
    pos = 0
    best_bottom = None
    for i in range(n):
        word = data["text"][i]
        if not word:
            continue
        pos_end = pos + len(word) + (1 if pos else 0)
        if offset_in_page <= pos_end:
            top = data["top"][i]
            height = data["height"][i]
            best_bottom = top + height
            break
        pos = pos_end
    if best_bottom is None and n:
        for i in range(n - 1, -1, -1):
            if data["text"][i]:
                best_bottom = data["top"][i] + data["height"][i]
                break
    if best_bottom is None:
        return None
    # Scale from image y to PDF y
    scale = page.rect.height / img.height
    return best_bottom * scale


def find_split_via_ocr(doc):
    """
    Find metadata split using OCR text.
    Returns (split_index, split_page, page_texts, meta, y_crop) or (None, None, None, None, None).
    """
    if not _OCR_AVAILABLE:
        return None, None, None, None, None
    page_texts = _get_page_texts_via_ocr(doc)
    if not page_texts:
        return None, None, None, None, None
    full_text = "\n".join(page_texts)
    meta, body, _ = split_metadata_body(full_text)
    if not meta:
        return None, None, None, None, None
    split_index = len(meta)
    cumulative = 0
    split_page = None
    for i, pt in enumerate(page_texts):
        page_end = cumulative + len(pt) + 1
        if split_index <= page_end:
            split_page = i + 1
            break
        cumulative = page_end
    if split_page is None:
        split_page = len(page_texts)
    cum = 0
    for j in range(split_page - 1):
        cum += len(page_texts[j]) + 1
    offset_in_page = split_index - cum
    page = doc[split_page - 1]
    y_crop = _y_crop_from_ocr_bbox(page, meta, offset_in_page, OCR_DPI)
    return split_index, split_page, page_texts, meta, y_crop


def char_offset_on_page(page_texts, split_page, split_index):
    """Return character offset inside split page."""
    cumulative = 0
    for i in range(split_page - 1):
        cumulative += len(page_texts[i]) + 1
    return split_index - cumulative


def y_position_from_offset(page, char_offset):
    """Convert character offset → Y coordinate using blocks."""
    blocks = page.get_text("blocks")
    pos = 0
    for block in blocks:
        text = block[4]
        if pos + len(text) >= char_offset:
            return block[3]
        pos += len(text)
    if blocks:
        return blocks[-1][3]
    return None


def load_summary_from_json(path):
    """
    Load summary text from an abstracted JSON file.
    Returns a single string: prefers summary_final, else joined summary_final_bullets.
    Returns None if file missing or no summary field.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        s = data.get("summary_final")
        if s and isinstance(s, str):
            return s
        bullets = data.get("summary_final_bullets")
        if bullets and isinstance(bullets, list):
            return "\n".join(b for b in bullets if b and isinstance(b, str))
        return None
    except Exception:
        return None


def load_bart_summary(pdf_stem):
    """Load summary string from SUMMARY_JSON_DIR by stem."""
    path = SUMMARY_JSON_DIR / f"{pdf_stem}.abstracted.json"
    return load_summary_from_json(path)


# Remove "(compressed from selected clauses)" after Key reasoning
_SUMMARY_CLEAN_PATTERN = re.compile(
    r"\s*\(compressed from selected clauses\)\s*",
    re.IGNORECASE,
)


def prepare_summary_text(text):
    """
    Clean summary before appending to PDF
    """
    if not text or not isinstance(text, str):
        return text or ""
    return _SUMMARY_CLEAN_PATTERN.sub(" ", text.strip()).strip()


def _max_chars_that_fit(
    rect, text, page_width, page_height, fontsize=11, fontname="helv"
):
    """
    Use a temporary page to measure: max n such that text[:n] fits in rect.
    insert_textbox returns >= 0 when text fits. Returns (n, temp_doc to close).
    """
    if not text.strip():
        return 0, None
    temp_doc = fitz.open()
    temp_page = temp_doc.new_page(width=page_width, height=page_height)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        rc = temp_page.insert_textbox(
            rect,
            text[:mid],
            fontsize=fontsize,
            fontname=fontname,
            align=fitz.TEXT_ALIGN_JUSTIFY,
        )
        if rc >= 0:
            lo = mid
        else:
            hi = mid - 1
    temp_doc.close()
    return lo


def _draw_summary_separator(page, page_rect, y_after_meta, margin_h=50):
    """
    Draw a horizontal line to separate metadata from the summary / reasoning section.
    Line is drawn at y_after_meta + 20, extended on both edges (smaller margin_h = longer line).
    """
    y_line = y_after_meta + 20
    if y_line >= page_rect.height - 20:
        return
    p1 = (margin_h, y_line)
    p2 = (page_rect.width - margin_h, y_line)
    page.draw_line(p1, p2, color=(0, 0, 0), width=1.5)


def append_summary_to_page(
    out_doc, page_rect, summary_text, start_y, fontsize=12, fontname="helv"
):
    """
    Insert summary_text into out_doc starting at start_y on the last page.
    If text overflows, add new pages. Uses binary search to find chunk sizes.
    """
    margin_l, margin_r, margin_b = 80, 80, 50
    remaining = summary_text.strip()
    if not remaining:
        return

    current_page = out_doc[-1]
    y_cursor = start_y

    if "\n\n" in remaining:
        first_para, remaining = remaining.split("\n\n", 1)
        first_para = first_para.strip()
        remaining = remaining.strip()
    else:
        first_para = remaining
        remaining = ""

    if first_para:
        intro_height = min(3 * fontsize * 1.3, page_rect.height - margin_b - y_cursor)
        intro_rect = fitz.Rect(
            margin_l,
            y_cursor,
            page_rect.width - margin_r,
            y_cursor + intro_height,
        )
        rc = current_page.insert_textbox(
            intro_rect,
            first_para,
            fontsize=fontsize + 1,
            fontname="helv",
            align=fitz.TEXT_ALIGN_LEFT,
        )
        if rc >= 0:
            y_cursor = intro_rect.y0 + (intro_rect.height - rc)
        else:
            y_cursor = intro_rect.y1
        y_cursor += 12  # gap before rest

    if not remaining:
        return

    summary_rect = fitz.Rect(
        margin_l,
        y_cursor,
        page_rect.width - margin_r,
        page_rect.height - margin_b,
    )
    while remaining:
        n = _max_chars_that_fit(
            summary_rect,
            remaining,
            page_rect.width,
            page_rect.height,
            fontsize=fontsize,
            fontname=fontname,
        )
        if n == 0:
            # Nothing fits on this rect; add a full new page and use full rect
            current_page = out_doc.new_page(
                width=page_rect.width, height=page_rect.height
            )
            summary_rect = fitz.Rect(
                margin_l,
                margin_l,
                page_rect.width - margin_r,
                page_rect.height - margin_b,
            )
            continue
        current_page.insert_textbox(
            summary_rect,
            remaining[:n],
            fontsize=fontsize,
            fontname=fontname,
            align=fitz.TEXT_ALIGN_LEFT,
        )
        remaining = remaining[n:].lstrip()
        if remaining:
            current_page = out_doc.new_page(
                width=page_rect.width, height=page_rect.height
            )
            summary_rect = fitz.Rect(
                margin_l,
                margin_l,
                page_rect.width - margin_r,
                page_rect.height - margin_b,
            )


def crop_pdf_after_metadata_and_append_summary(
    input_pdf, output_pdf, summary_text=None
):
    doc = fitz.open(input_pdf)
    split_index, split_page, page_texts, meta = find_split_in_pdf(doc)
    y_crop_from_ocr = None

    # If no split from normal text, try OCR to get split point
    if not split_index and _OCR_AVAILABLE:
        ocr_result = find_split_via_ocr(doc)
        split_index, split_page, page_texts, meta, y_crop_from_ocr = ocr_result
        if split_index:
            print("    (used OCR for split point)", end=" ")

    if not split_index:
        msg = (
            "❌ No metadata split found (tried text + OCR)."
            if _OCR_AVAILABLE
            else "❌ No metadata split found (tried text)."
        )
        print(msg)
        doc.close()
        return

    out_doc = fitz.open()

    # Copy full pages before split page
    for p in range(split_page - 1):
        out_doc.insert_pdf(doc, from_page=p, to_page=p)

    # Handle split page
    split_page_idx = split_page - 1
    page = doc[split_page_idx]
    rect = page.rect

    if y_crop_from_ocr is not None:
        y_crop = y_crop_from_ocr
    else:
        # Extract last line of metadata and search on page
        meta_lines = meta.strip().split("\n")
        last_meta_line = meta_lines[-1].strip()
        instances = page.search_for(last_meta_line)
        if instances:
            judge_rect = instances[-1]
            y_crop = judge_rect.y1
        else:
            y_crop = None

    if y_crop and y_crop < rect.height:
        new_page = out_doc.new_page(width=rect.width, height=rect.height)
        clip = fitz.Rect(0, 0, rect.width, y_crop + 5)
        new_page.show_pdf_page(
            fitz.Rect(0, 0, rect.width, y_crop + 5), doc, split_page_idx, clip=clip
        )
        redact_rect = fitz.Rect(0, y_crop + 5, rect.width, rect.height)
        new_page.add_redact_annot(redact_rect, fill=(1, 1, 1))
        new_page.apply_redactions()

        # Horizontal line to separate metadata from reasoning
        if summary_text:
            _draw_summary_separator(new_page, rect, y_crop)

        # Append BART summary just below metadata (space after line)
        if summary_text:
            summary_text = prepare_summary_text(summary_text)
            append_summary_to_page(out_doc, rect, summary_text, start_y=y_crop + 42)
    else:
        out_doc.insert_pdf(doc, from_page=split_page_idx, to_page=split_page_idx)
        if summary_text:
            _draw_summary_separator(out_doc[-1], out_doc[-1].rect, 30)
            summary_text = prepare_summary_text(summary_text)
            rect = out_doc[-1].rect
            append_summary_to_page(out_doc, rect, summary_text, start_y=72)

    out_doc.save(output_pdf, garbage=4, deflate=True)
    out_doc.close()
    doc.close()

    print(f"✅ Saved → {output_pdf}")


def process_supreme_court_dir(pdf_dir=None, summary_dir=None, out_dir=None):
    """
    Process each case that has both a summary JSON and a PDF.
    Iterates over *.abstracted.json in summary_dir, finds the
    matching PDF in pdf_dir by stem, then writes meta+summary to out_dir.
    """
    pdf_dir = Path(pdf_dir or PDF_DIR)
    summary_dir = Path(summary_dir or SUMMARY_JSON_DIR)
    out_dir = Path(out_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_dir.is_dir():
        print(f"❌ Summary JSON directory not found: {summary_dir}")
        return

    json_files = sorted(summary_dir.glob("*.abstracted.json"))
    if not json_files:
        print(f"❌ No *.abstracted.json files in {summary_dir}")
        return

    if not pdf_dir.is_dir():
        print(f"❌ PDF directory not found: {pdf_dir}")
        return

    print(
        f"Processing {len(json_files)} case(s): JSONs from {summary_dir}, PDFs from {pdf_dir} → {out_dir}\n"
    )
    for i, json_path in enumerate(json_files, 1):
        stem = json_path.stem.removesuffix(".abstracted")
        pdf_path = pdf_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            pdf_path = next(pdf_dir.rglob(f"{stem}.pdf"), None)
        if not pdf_path or not pdf_path.exists():
            print(f"[{i}/{len(json_files)}] {stem} — ⚠️ No PDF found, skipped")
            continue

        output_path = out_dir / f"{stem}_summarized.pdf"
        print(f"[{i}/{len(json_files)}] {pdf_path.name} ... ", end="", flush=True)
        summary_text = load_summary_from_json(json_path)
        if not summary_text:
            print("(no summary in JSON; meta-only) ", end="")
        try:
            crop_pdf_after_metadata_and_append_summary(
                pdf_path, output_path, summary_text=summary_text
            )
        except Exception as e:
            print(f"❌ {e}")


if __name__ == "__main__":
    process_supreme_court_dir()
