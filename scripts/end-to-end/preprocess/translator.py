import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file
import pandas as pd
import os
from pathlib import Path

# EXTRACT NON-ENGLISH LANGUAGES
# Language detection patterns
SINHALA_PATTERN = re.compile(r"[\u0D80-\u0DFF]")
TAMIL_PATTERN = re.compile(r"[\u0B80-\u0BFF]")
# Combined pattern for both languages
MULTILANG_PATTERN = re.compile(r"[\u0D80-\u0DFF\u0B80-\u0BFF]")

LANG_CODES = {"sinhala": "sin_Sinh", "tamil": "tam_Taml"}

Q_MARKERS = ["Q:", "Q.", "පු:"]
A_MARKERS = ["A:", "A.", "උ:", "c:", "C:"]
ALL_MARKERS = Q_MARKERS + A_MARKERS


def detect_language(text):
    """
    Detect if text contains Sinhala, Tamil, or both.
    Returns: 'sinhala', 'tamil', 'mixed', or None
    """
    has_sinhala = bool(SINHALA_PATTERN.search(text))
    has_tamil = bool(TAMIL_PATTERN.search(text))

    if has_sinhala and has_tamil:
        return "mixed"
    elif has_sinhala:
        return "sinhala"
    elif has_tamil:
        return "tamil"
    return None


def get_language_pattern(lang):
    """
    Get the appropriate regex pattern for the given language.
    """
    if lang == "sinhala":
        return SINHALA_PATTERN
    elif lang == "tamil":
        return TAMIL_PATTERN
    else:
        return MULTILANG_PATTERN


def clean_marker(text):
    """
    Normalize QA markers:
    - 'පු:' and 'Q.', 'Q:' → 'Q:'
    - 'උ:' and 'A.', 'A:' → 'A:'
    """
    text = text.strip()

    for q_marker in ["පු:", "Q.", "Q:"]:
        if text.startswith(q_marker):
            text = text[len(q_marker) :].strip()
            return "Q:" + text

    for a_marker in ["A:", "A.", "උ:", "c:", "C:"]:
        if text.startswith(a_marker):
            text = text[len(a_marker) :].strip()
            return "A:" + text

    return text


def extract_multilang_block(text, lookahead=10):
    """
    Extract Sinhala or Tamil text block from given text.
    Returns tuple: (extracted_text, detected_language)
    """
    # Try to find first occurrence of either language
    sinhala_match = re.search(SINHALA_PATTERN, text)
    tamil_match = re.search(TAMIL_PATTERN, text)

    # Determine which language appears first
    if sinhala_match and tamil_match:
        if sinhala_match.start() < tamil_match.start():
            match = sinhala_match
            lang = "sinhala"
            pattern = SINHALA_PATTERN
        else:
            match = tamil_match
            lang = "tamil"
            pattern = TAMIL_PATTERN
    elif sinhala_match:
        match = sinhala_match
        lang = "sinhala"
        pattern = SINHALA_PATTERN
    elif tamil_match:
        match = tamil_match
        lang = "tamil"
        pattern = TAMIL_PATTERN
    else:
        return "", None

    text = text[match.start() :]
    result = []
    i = 0
    while i < len(text):
        result.append(text[i])
        if text[i] in [".", "!", "?", "”", '"', "\n"]:
            snippet = text[i + 1 : i + 1 + lookahead]
            # Check if snippet contains the same language
            if not pattern.search(snippet):
                break
        i += 1

    block = "".join(result).rstrip()
    while block and not pattern.search(block[-1]):
        block = block[:-1]
    return block.strip(), lang


def extract_qa_with_signatures(
    text, signature_map=None, sig_prefix="§SIG", start_counter=1
):
    """
    Extract Sinhala/Tamil QA blocks, replace them in text with signatures, and update signature map.

    Args:
        text (str): Original text with English + Sinhala/Tamil QA
        signature_map (dict): Existing signature map - format: {signature: {'text': text, 'lang': lang}}
        sig_prefix (str): Prefix for signatures
        start_counter (int): Starting signature number

    Returns:
        processed_text (str): Text with Sinhala/Tamil replaced by signatures
        signature_map (dict): Updated mapping signature -> {'text': text, 'lang': lang}
        next_sig_counter (int): Next counter value for further signatures
    """
    signature_map = signature_map or {}
    sig_counter = start_counter

    qa_pairs = []

    pattern = r"(" + "|".join(re.escape(m) for m in ALL_MARKERS) + r")"
    markers = [(m.group(), m.start()) for m in re.finditer(pattern, text)]

    if not markers:
        return text, signature_map, sig_counter

    current_q_sig = None
    processed_text = text

    for i, (marker, pos) in enumerate(markers):
        end = markers[i + 1][1] if i + 1 < len(markers) else len(text)
        block = text[pos:end].strip()
        normalized_block = clean_marker(block)
        extracted_text, detected_lang = extract_multilang_block(normalized_block)

        if not extracted_text or not detected_lang:
            continue

        if normalized_block.startswith("Q:") and not extracted_text.endswith("?"):
            extracted_text += "?"
        elif normalized_block.startswith("A:") and not extracted_text.endswith("."):
            extracted_text += "."

        # Assign signature
        sig = f"{sig_prefix}{sig_counter:04d}"
        signature_map[sig] = {"text": extracted_text, "lang": detected_lang}
        sig_counter += 1

        # Replace extracted text in normalized block with signature
        new_block = normalized_block.replace(extracted_text, sig)
        processed_text = processed_text.replace(block, new_block, 1)

        # Track QA pairs using signatures
        if marker in Q_MARKERS:
            current_q_sig = sig
        elif marker in A_MARKERS and current_q_sig:
            qa_pairs.append({"Q": current_q_sig, "A": sig})
            current_q_sig = None

    return processed_text, signature_map, sig_counter


# TRANSLATE EXTRACTED BLOCKS
def translate_multilang_to_english(
    signature_map, legal_dict=None, max_tokens=1024, adapter_dir=None
):
    """
    Translates Sinhala/Tamil texts in a dictionary to English using LoRA-adapted NLLB-200 1.3B,
    with dictionary fallback, token-limit-aware chunking, and safe GPU inference.

    Args:
        signature_map (dict): {signature: {'text': text, 'lang': lang}} where lang is 'sinhala' or 'tamil'
        legal_dict (dict, optional): {phrase: english_translation} to prioritize
        max_tokens (int): maximum token length per chunk
    Returns:
        dict: {signature: english_translation}
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translated_map = {}
    base_model_name = "facebook/nllb-200-1.3B"
    if adapter_dir is None:
        adapter_dir = str(Path(__file__).resolve().parent / "nllb_sinhala2english_lora")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    target_lang = "eng_Latn"

    for sig, data in signature_map.items():
        if isinstance(data, dict):
            text = data["text"]
            lang = data["lang"]
        else:
            # if old format (just text), try to detect language
            text = data
            lang = detect_language(text) or "sinhala"

        source_lang = LANG_CODES.get(lang, "sin_Sinh")
        tokenizer.src_lang = source_lang

        # Tokenize and check length
        tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
        if len(tokens) <= max_tokens:
            chunks = [text]
        else:
            sentences = re.split(r"[।\.]", text)
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if not sentence.strip():
                    continue
                candidate = (current_chunk + " " + sentence).strip()
                candidate_tokens = tokenizer(candidate, return_tensors="pt").input_ids[
                    0
                ]

                if len(candidate_tokens) > max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # Check if sentence is too long
                    sentence_tokens = tokenizer(
                        sentence, return_tensors="pt"
                    ).input_ids[0]
                    if len(sentence_tokens) > max_tokens:
                        # Split sentence into sub-chunks of max_tokens
                        for i in range(0, len(sentence_tokens), max_tokens):
                            sub_tokens = sentence_tokens[i : i + max_tokens]
                            sub_chunk = tokenizer.decode(
                                sub_tokens, skip_special_tokens=True
                            )
                            chunks.append(sub_chunk.strip())
                        current_chunk = ""
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = candidate

            if current_chunk:
                chunks.append(current_chunk.strip())

        # Translate each chunk with dictionary replacement
        translated_chunks = []
        for chunk in chunks:
            # Model inference
            with torch.no_grad():
                inputs = tokenizer(chunk, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
                    max_new_tokens=256,
                    num_beams=5,
                    length_penalty=1.0,
                )
                english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated_chunks.append(english_text)

            torch.cuda.empty_cache()

        # Combine all chunks
        translated_map[sig] = " ".join(translated_chunks).strip()

    return translated_map


def replace_signatures_with_translations(translated_text, translated_map):
    """
    Replaces signature placeholders in a translated text with their English translations.

    Args:
        translated_text (str): Text containing signatures like §SIG0001
        translated_map (dict): {signature: english_translation}

    Returns:
        str: Text with signatures replaced by their translated context
    """

    # Regex to match §SIGxxxx patterns
    pattern = r"§SIG\d{4}"

    def replacer(match):
        sig = match.group(0)
        return translated_map.get(sig, sig)

    return re.sub(pattern, replacer, translated_text)


# PROCESS FILES AND DIRECTORY
def process_text_file(input_path, output_dir, adapter_dir=None):
    """
    Process a Sinhala/Tamil+English text file, extract all Sinhala/Tamil content,
    replace with signatures, translate to English, and save results.
    adapter_dir: optional path to NLLB LoRA adapter; if None, uses default.
    """

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n📄 Processing file: {os.path.basename(input_path)}")
    signature_map = {}
    sig_counter = 1

    # QA extractor
    processed_text, signature_map, sig_counter = extract_qa_with_signatures(
        text, signature_map, "§SIG", sig_counter
    )

    # Remaining Sinhala/Tamil blocks
    text = processed_text
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "§":
            end_sig = text.find(" ", i)
            if end_sig == -1:
                end_sig = n
            i = end_sig
            continue
        # Check for either Sinhala or Tamil
        if MULTILANG_PATTERN.match(text[i]):
            remaining_text = text[i:]
            block, detected_lang = extract_multilang_block(remaining_text)
            if block and detected_lang:
                sig = f"§SIG{sig_counter:04d}"
                signature_map[sig] = {"text": block, "lang": detected_lang}
                text = text[:i] + sig + text[i + len(block) :]
                sig_counter += 1
                i += len(sig)
                n = len(text)
                continue
        i += 1

    processed_text = text

    translated_map = translate_multilang_to_english(
        signature_map, adapter_dir=adapter_dir
    )

    final_translation = replace_signatures_with_translations(
        processed_text, translated_map
    )

    base_name, ext = os.path.splitext(os.path.basename(input_path))
    if base_name.endswith(".cleaned"):
        new_base_name = base_name[:-8] + ".translated"
    else:
        new_base_name = f"{base_name}.translated"
    translated_output = os.path.join(output_dir, f"{new_base_name}{ext}")

    with open(translated_output, "w", encoding="utf-8") as f:
        f.write(final_translation)

    lang_counts = {}
    for sig_data in signature_map.values():
        if isinstance(sig_data, dict):
            lang = sig_data.get("lang", "unknown")
        else:
            lang = "unknown"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print(f"✅ Translated file saved → {translated_output}")
    print(f"🧭 Total blocks: {len(signature_map)}")
    if lang_counts:
        lang_info = ", ".join(
            [f"{lang}: {count}" for lang, count in lang_counts.items()]
        )
        print(f"📊 Language breakdown: {lang_info}")

    return signature_map


def process_directory(input_dir, output_dir, recursive=True, adapter_dir=None):
    """
    Process all text files in a directory through the Sinhala/Tamil translation pipeline.

    Args:
        input_dir (str): Path to input directory containing .txt files
        output_dir (str): Path to output directory where results will be saved
        recursive (bool): If True, process subdirectories as well
        adapter_dir: optional path to NLLB LoRA adapter; if None, uses default.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = (
        list(input_dir.rglob("*.txt")) if recursive else list(input_dir.glob("*.txt"))
    )

    count = 0
    skipped = 0
    for file_path in files:
        rel_path = file_path.relative_to(input_dir).parent
        target_output_dir = output_dir / rel_path
        os.makedirs(target_output_dir, exist_ok=True)

        base_name = file_path.stem
        if base_name.endswith(".cleaned"):
            new_base_name = base_name[:-8] + ".translated"
        else:
            new_base_name = f"{base_name}.translated"
        translated_output_file = target_output_dir / f"{new_base_name}.txt"

        if translated_output_file.exists():
            print(f"⏭️ Skipping already translated file: {file_path.name}")
            skipped += 1
            continue

        try:
            process_text_file(file_path, target_output_dir, adapter_dir=adapter_dir)
            count += 1
        except Exception as e:
            print(f"⚠️ Error processing {file_path.name}: {e}")

    print(f"\n✅ Completed processing {count} file(s) from {input_dir}")
    if skipped > 0:
        print(f"⏭️ Skipped {skipped} file(s) that were already translated.")


if __name__ == "__main__":
    input_dir = "/kaggle/input/cleaned/CLEANED TEXTS"
    output_dir = "/kaggle/working/Translated"
    os.makedirs(output_dir, exist_ok=True)
    process_directory(input_dir, output_dir)
