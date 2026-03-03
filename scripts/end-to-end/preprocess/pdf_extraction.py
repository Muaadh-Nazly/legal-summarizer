import os
import re
import json
from io import StringIO
from pathlib import Path
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text_to_fp
from pdfminer.pdfpage import PDFPage
import pdfplumber
from PIL import Image
import warnings
import contextlib

# Surya OCR
os.environ["TQDM_DISABLE"] = "1"
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

try:
    from surya.foundation import FoundationPredictor

    _use_foundation = True
except ModuleNotFoundError:
    _use_foundation = False

# Load Surya models once
print("Loading Surya OCR models...")
if _use_foundation:
    _fp = FoundationPredictor()
    rec = RecognitionPredictor(_fp)
else:
    rec = RecognitionPredictor()
det = DetectionPredictor()
print("Surya ready.")


## Functions
def contains_garbled_text(text: str, max_consonant_seq=5, quote_bracket_seq=4):
    """
    Detects garbled text including multi-line quotes/brackets:
    - Unicode or legacy Sinhala characters
    - Multiple semicolons or commas inside a word
    - Corrupt symbols like <>{}^+@|
    - Long consonant sequences
    """
    unicode_sinhala_pattern = re.compile(r"[\u0D80-\u0DFF]")
    legacy_sinhala_pattern = re.compile(r"[úïñÿ]")
    corrupt_symbols_pattern = re.compile(r"[<>{}^+@\\|]")
    bracket_quote_pattern = re.compile(r'[\(\["“‘](.+?)[\)\]”’"]', re.DOTALL)

    if unicode_sinhala_pattern.search(text) or legacy_sinhala_pattern.search(text):
        return True

    lines = text.splitlines()
    for line in lines:
        words = line.split()
        for word in words:
            if word.count(";") > 1 or word.count(",") > 1:
                return True
            if corrupt_symbols_pattern.search(word):
                return True

    # Check multi-line quoted/bracketed sequences
    for match in bracket_quote_pattern.findall(text):
        clean_match = re.sub(r"^[^\w]+|[^\w]+$", "", match)
        if not clean_match:
            continue
        # long consonant sequences
        if re.search(
            r"[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ;=]{%d,}" % quote_bracket_seq,
            clean_match,
        ):
            return True

    # normal consonant sequences
    for line in lines:
        for word in line.split():
            if not word[0].isupper() and re.search(
                r"[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ;=]{%d,}"
                % max_consonant_seq,
                word,
            ):
                return True
    return False


# CLEAN TEXT PROCESSING
def clean_paragraphs(text):
    """Clean and format text using the improved method from 1_extract_all_pdfs_to_text.py"""
    # Step 1: Fix hyphenation across lines/pages
    text = re.sub(r"-\n", "", text)

    # Step 2: Join broken lines
    # Join lines that end with lowercase letters (most common case)
    text = re.sub(r"([a-z])\n([a-z])", r"\1 \2", text)

    # Join lines that end with commas
    text = re.sub(r",\n([a-z])", r", \1", text)

    # Join lines that don't end with sentence punctuation and next line starts with lowercase
    text = re.sub(r"([^.!?])\n([a-z])", r"\1 \2", text)

    # Join lines that end with words and next line starts with lowercase
    text = re.sub(r"(\w)\n([a-z])", r"\1 \2", text)

    # Step 3: Turn remaining line breaks within paragraphs into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Step 4: Normalize multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Step 5: Preserve real paragraph breaks (2+ newlines)
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()


# OCR PROCESSING
def extract_surya_text_from_page(page_output_dir: str) -> str:
    """Recursively find JSON from Surya output and extract text lines."""
    json_file = None
    for root, _, files in os.walk(page_output_dir):
        for f in files:
            if f.endswith(".json"):
                json_file = os.path.join(root, f)
                break
        if json_file:
            break

    if not json_file:
        print(f"⚠️ No JSON found in {page_output_dir}")
        return ""

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for page_key, samples in data.items():
        for sample_item in samples:
            for line in sample_item.get("text_lines", []):
                line_text = line.get("text", "").strip()
                if line_text:
                    texts.append(line_text)
    return " ".join(texts)


def _extract_text_from_surya_predictions(predictions):
    """Extract text from Surya OCR result (supports both Pydantic OCRResult and dict)."""
    texts = []
    for page_result in predictions:
        lines = getattr(page_result, "text_lines", None) or page_result.get(
            "text_lines", []
        )
        for line in lines:
            t = (getattr(line, "text", None) or line.get("text", "") or "").strip()
            if t:
                texts.append(t)
    return " ".join(texts)


def process_page_with_ocr(pdf_path: Path, page_num: int) -> str:
    """Process a single page with in-process Surya OCR; no per-page progress output."""
    print(f"   🔎 Using OCR for page {page_num}")
    images = convert_from_path(str(pdf_path), first_page=page_num, last_page=page_num)
    img = images[0].convert("RGB")
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            predictions = rec([img], det_predictor=det)
    return _extract_text_from_surya_predictions(predictions)


def extract_text_from_pdf_page_by_page(pdf_path: Path) -> list:
    """Extract text from PDF page by page using pdfplumber for better text extraction."""
    page_texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    page_texts.append(page_text)
                else:
                    page_texts.append("")
    except Exception as e:
        print(f"❌ Error reading {pdf_path.name}: {e}")
        # Fallback to pdfminer if pdfplumber fails
        try:
            with open(pdf_path, "rb") as f:
                for i, page in enumerate(PDFPage.get_pages(f)):
                    output_string = StringIO()
                    extract_text_to_fp(f, output_string, page_numbers=[i])
                    page_texts.append(output_string.getvalue())
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return [""]

    return page_texts


def process_pdf(pdf_path: Path, output_file: Path, ocr_output_dir: Path):
    """Process one PDF using the merged approach."""
    page_texts = extract_text_from_pdf_page_by_page(pdf_path)
    final_text_pages = []

    for i, page_text in enumerate(page_texts):
        page_num = i + 1
        print(f"➡️ {pdf_path.name} - Page {page_num}/{len(page_texts)}")

        if contains_garbled_text(page_text) or not page_text.strip():
            ocr_text = process_page_with_ocr(pdf_path, page_num)
            cleaned = clean_paragraphs(ocr_text)
            final_text_pages.append(cleaned)
        else:
            print(f"   ✅ Using clean text extraction for page {page_num}")
            cleaned = clean_paragraphs(page_text)
            final_text_pages.append(cleaned)

    with open(output_file, "w", encoding="utf-8") as f:
        for line in final_text_pages:
            f.write(line + "\n\n")

    print(f"   📄 Saved {output_file}")


def process_all_pdfs(input_dir: Path, output_dir: Path, ocr_output_dir: Path):
    """Process all PDF files in the input directory recursively, preserving folder structure."""
    pdf_files = list(input_dir.rglob("*.pdf"))
    print(f"📂 Found {len(pdf_files)} PDF files...")

    for pdf_file in pdf_files:
        relative_path = pdf_file.relative_to(input_dir)
        output_file = output_dir / relative_path.with_suffix(".extracted.txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists():
            print(f"➡️ Skipping {pdf_file.name} (already processed)")
            continue

        process_pdf(pdf_file, output_file, ocr_output_dir)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda")

    INPUT_DIR = Path("/kaggle/input/datasets/muaadhnazly/new-files")
    OUTPUT_DIR = Path("/kaggle/working/new-files")
    OCR_OUTPUT_DIR = Path("/kaggle/working/ocr_out")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_all_pdfs(INPUT_DIR, OUTPUT_DIR, OCR_OUTPUT_DIR)
