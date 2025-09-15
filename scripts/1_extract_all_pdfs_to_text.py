import pdfplumber
import re
from pathlib import Path

# === CONFIGURATION ===
INPUT_DIR = Path("Supreme Court/Categorized/criminal")
OUTPUT_DIR = Path("Extracted Text")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_paragraphs(text):
    # Step 1: Fix hyphenation across lines/pages
    text = re.sub(r'-\n', '', text)

    # Step 2: Join broken lines more aggressively
    # Join lines that end with lowercase letters (most common case)
    text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
    
    # Join lines that end with commas
    text = re.sub(r',\n([a-z])', r', \1', text)
    
    # Join lines that don't end with sentence punctuation and next line starts with lowercase
    text = re.sub(r'([^.!?])\n([a-z])', r'\1 \2', text)
    
    # Join lines that end with words and next line starts with lowercase
    text = re.sub(r'(\w)\n([a-z])', r'\1 \2', text)

    # Step 3: Turn remaining line breaks within paragraphs into spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Step 4: Normalize multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Step 5: Preserve real paragraph breaks (2+ newlines)
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()


def extract_text_from_pdf(pdf_path):
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n"
    except Exception as e:
        print(f"❌ Error reading {pdf_path.name}: {e}")
    return full_text

def process_all_pdfs(input_dir, output_dir):
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files...")

    for pdf_file in pdf_files:
        print(f"➡️ Processing: {pdf_file.name}")
        raw_text = extract_text_from_pdf(pdf_file)
        cleaned_text = clean_paragraphs(raw_text)
        output_file = output_dir / (pdf_file.stem + ".txt")
        output_file.write_text(cleaned_text, encoding="utf-8")

    print("✅ All PDFs processed and cleaned.")

# === MAIN ===
if __name__ == "__main__":
    process_all_pdfs(INPUT_DIR, OUTPUT_DIR)
