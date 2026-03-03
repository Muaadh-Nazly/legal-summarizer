# LIBRARIES
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# CONFIGURATIONS
BART_MODEL_ID = "facebook/bart-large-cnn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_THIRD_PERSON_STEP = True

# Lazy-loaded by run_abstraction; used by summarize_one_clause
tokenizer = None
model = None

ABSTRACTIVE_CONFIG = {
    "max_new_tokens": 128,
    "min_new_tokens": 10,
    "min_accepted_len": 15,
}


# INITIAL SETUP FUNCTIONS
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def load_ranked(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_clause_texts_in_doc_order(
    record: Dict[str, Any],
) -> Tuple[List[int], List[str]]:
    items = record.get("selected_clauses_in_doc_order") or []
    items = sorted(items, key=lambda x: x.get("doc_order", 0))
    ids, texts = [], []
    for it in items:
        cid = it.get("clause_id")
        if cid is None:
            continue
        ids.append(int(cid))
        texts.append(normalize_space(str(it.get("text", ""))))
    return ids, texts


def get_primary_disposition_clause_id(record: Dict[str, Any]) -> Optional[int]:
    """Among forced_disposition_clause_ids, return the one that appears last in doc order.
    That clause is the final disposition and is used as Key holding (verbatim); the rest stay in reasoning.
    """
    forced_ids = set(record.get("forced_disposition_clause_ids") or [])
    if not forced_ids:
        return None
    items = record.get("selected_clauses_in_doc_order") or []
    by_order = [
        (it.get("doc_order", 999), it.get("clause_id"))
        for it in items
        if it.get("clause_id") in forced_ids
    ]
    if not by_order:
        return None
    return max(by_order, key=lambda x: (x[0], x[1]))[1]


def get_disposition_text(record: Dict[str, Any]) -> str:
    primary_id = get_primary_disposition_clause_id(record)
    if primary_id is None:
        return ""
    for it in record.get("selected_clauses_in_doc_order") or []:
        if it.get("clause_id") == primary_id:
            return normalize_space(str(it.get("text", "")))
    for it in record.get("ranked_clauses") or []:
        if it.get("clause_id") == primary_id:
            return normalize_space(str(it.get("text", "")))
    return ""


def make_lead_sentence(case_type: Optional[str]) -> str:
    ct = (case_type or "case").strip().lower()
    if not ct:
        return "This case concerns the key legal issue, reasoning, and the court's final order."
    if "fundamental" in ct:
        return "This fundamental rights application concerns the key legal issue, reasoning, and the court's final order."
    return f"This {ct} concerns the key legal issue, reasoning, and the court's final order."


def build_step1_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = record.get("doc_id") or "unknown"
    case_type = record.get("case_type")
    clause_ids, clause_texts = get_clause_texts_in_doc_order(record)
    if not clause_texts:
        return {
            "doc_id": doc_id,
            "case_type": case_type,
            "summary_structured": "",
            "disposition_text": "",
            "num_bullets": 0,
            "error": "No selected clauses in doc order",
        }
    disposition_text = get_disposition_text(record)
    lead = make_lead_sentence(case_type)
    lines = [lead, "", "Key reasoning"]
    for i, text in enumerate(clause_texts, start=1):
        lines.append(f"{i}. {text}")
    if disposition_text:
        lines.extend(["", "Key holding (verbatim clause):", disposition_text])
    summary_structured = "\n".join(lines).strip()
    return {
        "doc_id": doc_id,
        "case_type": case_type,
        "selected_clause_ids": clause_ids,
        "num_bullets": len(clause_texts),
        "disposition_text": disposition_text,
        "summary_structured": summary_structured,
    }


# SUMMARIZATION FUNCTION
def summarize_one_clause(
    text: str, max_new_tokens: int = 128, min_new_tokens: int = 10
) -> str:
    """One clause in → one shortened sentence."""
    if not text or not text.strip():
        return text or ""
    inputs = tokenizer(
        text.strip(), return_tensors="pt", truncation=True, max_length=1024
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def summarize_clauses_raw(clause_texts: List[str], config: dict = None) -> List[str]:
    """One clause in → one out, no safety. For use inside safe pipeline."""
    config = config or ABSTRACTIVE_CONFIG
    out = []
    for t in clause_texts:
        if not (t and str(t).strip()):
            out.append(t or "")
            continue
        out.append(
            summarize_one_clause(
                t,
                max_new_tokens=config.get("max_new_tokens", 128),
                min_new_tokens=config.get("min_new_tokens", 10),
            )
        )
    return out


# ABSTRACTION FUNCTIONS
PRECEDENT_PATTERNS = [
    r"\bv\.\s",
    r"\bNLR\s+\d+",
    r"\bAIR\s+\d{4}",
    r"\b\d+\s+NLR\s+\d+",
    r"it was held that",
    r"was held that",
    r"stated at page",
    r"learned author",
    r"\.C\.J\.",
    r"\bVol\.?\s*\d+",
    r"\bpage\s+\d+",
    r"Butterworths",
]
PRECEDENT_RE = re.compile("|".join(f"(?:{p})" for p in PRECEDENT_PATTERNS), re.I)


def contains_precedent_or_citation(text: str) -> bool:
    if not text or not text.strip():
        return False
    return bool(PRECEDENT_RE.search(text))


# Unwanted phrases in output
UNWANTED_IN_OUTPUT_IF_NOT_IN_ORIGINAL = [
    "Supreme Court of the United Kingdom",
    "European Union",
    "U.S. District Court",
    "Supreme Court of the United States",
    "Court of Appeal of the United States",
    "Constitution of India",
]


def output_has_unwanted_phrase_not_in_original(original: str, summary: str) -> bool:
    orig_lower = (original or "").lower()
    summ = summary or ""
    for phrase in UNWANTED_IN_OUTPUT_IF_NOT_IN_ORIGINAL:
        if phrase in summ and phrase.lower() not in orig_lower:
            return True
    return False


# Truncation / fragment / repetition
def looks_truncated(summary: str, min_len: int = 30) -> bool:
    if not summary or len(summary) < min_len:
        return False
    if summary.rstrip() and summary.rstrip()[-1] in ".?!\"'":
        return False
    last_word = (summary.strip().split() or [""])[-1].lower()
    return last_word in ("the", "a", "an", "of", "and", "to", "in")


def looks_like_fragment(summary: str, min_len: int = 25) -> bool:
    if not summary or len(summary) < min_len:
        return False
    s = summary.strip()
    if s and s[0].islower():
        return True
    for start in ("of the ", "and the ", "but the ", "which ", "that the "):
        if s.lower().startswith(start):
            return True
    return False


def has_repetition_loop(
    summary: str, min_repeats: int = 3, phrase_len_range: Tuple[int, int] = (12, 60)
) -> bool:
    if not summary or len(summary) < phrase_len_range[0] * min_repeats:
        return False
    for L in range(phrase_len_range[1], phrase_len_range[0] - 1, -1):
        for i in range(len(summary) - L * min_repeats + 1):
            phrase = summary[i : i + L]
            if " " not in phrase:
                continue
            pat = (
                re.escape(phrase)
                + r"(?:\s+"
                + re.escape(phrase)
                + r"){"
                + str(min_repeats - 2)
                + r",}"
            )
            if re.search(pat, summary):
                return True
    return False


def has_repeated_sentence(summary: str) -> bool:
    if not summary or len(summary) < 20:
        return False
    raw = re.split(r"(?<=[.!?])\s+", summary)
    sentences = [re.sub(r"\s+", " ", s).strip() for s in raw if s and s.strip()]
    return len(sentences) > 1 and len(sentences) != len(set(sentences))


def has_duplicate_word_introduced(original: str, summary: str) -> bool:
    if not summary or len(summary) < 5:
        return False
    orig = (original or "").strip()
    for m in re.finditer(r"\b(\w+)\s+\1\b", summary):
        if m.group(0) not in orig:
            return True
    return False


def summary_drops_numbered_list(original: str, summary: str) -> bool:
    def count(t: str):
        return len(re.findall(r"\([a-e]\)", t or ""))

    return count(original or "") > count(summary or "")


# Minimal cleanup only for ACCEPTED output
def minimal_cleanup(text: str) -> str:
    if not text:
        return text
    t = re.sub(r"<n>", " ", text, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"Article 11\s+that\s+Article 11", "Article 11", t, flags=re.I)
    t = re.sub(r"arbitrarily\s+arbitrary", "arbitrary", t, flags=re.I)
    t = re.sub(r"\bis\s+entitlement\s+to\b", "is entitled to", t, flags=re.I)
    t = re.sub(r"\bthe\s+1\s+first\s+", "the 1st ", t, flags=re.I)
    t = re.sub(r"\bfirst\s+limbs\b", "first limb", t, flags=re.I)
    t = re.sub(r"\b(\d+)\.\1\.\s+", r"\1. ", t)
    return t


# Accept or Fallback
def decide_final_line(
    original: str,
    model_output: str,
    min_accepted_len: int = 15,
    run_minimal_cleanup: bool = True,
) -> Tuple[str, bool, str]:
    """Returns (final_text, used_model_output, reason)."""
    orig = (original or "").strip()
    raw = (model_output or "").strip()
    if run_minimal_cleanup:
        raw = minimal_cleanup(raw)
    if not raw or len(raw) < min_accepted_len:
        return orig, False, "fallback: empty or too short"
    if output_has_unwanted_phrase_not_in_original(orig, raw):
        return orig, False, "fallback: unwanted phrase"
    if looks_truncated(raw):
        return orig, False, "fallback: truncated"
    if looks_like_fragment(raw):
        return orig, False, "fallback: fragment"
    if has_repetition_loop(raw):
        return orig, False, "fallback: repetition loop"
    if has_repeated_sentence(raw):
        return orig, False, "fallback: repeated sentence"
    if has_duplicate_word_introduced(orig, raw):
        return orig, False, "fallback: duplicate word"
    if summary_drops_numbered_list(orig, raw):
        return orig, False, "fallback: dropped list"
    return raw, True, "accepted"


# Third person convertion
THIRD_PERSON_RULES = [
    (r"\bI\s+therefore\s+hold\b", "The Court therefore held"),
    (r"\bI pointed out\b", "The Court pointed out"),
    (r"\bI\s+set\s+aside\b", "The Court set aside"),
    (r"\bI affirm\b", "The Court affirmed"),
    (r"\bI answer\b", "The Court answers"),
    (r"\bI\s+hold\b", "The Court holds"),
    (r"\bI find\b", "The Court finds"),
    (r"\bI dismiss\b", "The Court dismisses"),
    (r"\bI allow\b", "The Court allows"),
    (r"\bI proceed to dismiss\b", "The Court proceeds to dismiss"),
    (r"\bI am of the view\b", "The Court is of the view"),
    (r"\bI am of the considered opinion\b", "The Court was of the considered opinion"),
    (r"\bI agree\b", "The Court agreed"),
    (r"\bwe are of the opinion\b", "The Court is of the opinion"),
    (r"\bwe find\b", "The Court finds"),
    (r"\bwe hold\b", "The Court holds"),
    (r"\bIn my view\b", "In the Court's view"),
    (r"\bIn our view\b", "In the Court's view"),
    (r"\bmy view\b", "the Court's view"),
    (r"The Court finds myself\b", "The Court finds itself"),
    (r"The Court found myself\b", "The Court found itself"),
    (r"\bI find myself\b", "The Court finds itself"),
    (r"\bI (?:am|was) mindful\b", "The Court is mindful"),
    (r"\bI have to hold\b", "The Court had to hold"),
    (r"\bI direct\b", "The Court directs"),
    (r"enumerated by me\b", "enumerated above"),
    (r"observed by me\b", "observed by the Court"),
    (r"It appears to me\b", "It appears to the Court"),
    (r"\bI took the view\b", "The Court took the view"),
    (r"\bI unhesitatingly reject\b", "The Court unhesitatingly rejected"),
    (r"\bI further declare\b", "The Court further declares"),
    (r"\bI therefore, set aside\b", "The Court therefore set aside"),
    (r"\bI declare\b", "The Court declares"),
    (r"\bI am not inclined to hold\b", "The Court is not inclined to hold"),
    (r"\bI uphold\b", "The Court upholds"),
    (r"\bI order\b", "The Court orders"),
    (r"\bI consider\b", "The Court considers"),
    (r"\bI observe\b", "The Court observes"),
    (r"\bI find\b", "The Court finds"),
    (r"\bWe direct\b", "The Court directs"),
    (r"\bWe are\b", "The Court is"),
    (r"\bI make\b", "The Court makes"),
    (r"\bI overrule\b", "The Court overrules"),
    (r"\bI consider it appropriate\b", "The Court considers it appropriate"),
    (r"\bwe wish to note\b", "The Court wishes to note"),
    (r"\bWe are empowered\b", "The Court is empowered"),
    (r"\bWe observe\b", "The Court observes"),
    (r"\bwe proceed to\b", "The Court proceeds to"),
    (r"\bWe also proceed to\b", "The Court also proceeds to"),
    (r"\bwe are unable to see\b", "The Court is unable to see"),
    (r"\bwe can direct\b", "The Court can direct"),
    (r"\bwe specify\b", "The Court specifies"),
    (r"\bwe have quashed\b", "The Court has quashed"),
    (r"\bwe are mindful\b", "The Court is mindful"),
    (r"\bI am of the opinion\b", "The Court is of the opinion"),
    (r"\bI refuse\b", "The Court refuses"),
]


def convert_to_third_person(text: str) -> str:
    if not text or not text.strip():
        return text
    t = text.strip()
    for pat, repl in THIRD_PERSON_RULES:
        t = re.sub(pat, repl, t, flags=re.I)
    t = re.sub(r"\b(\d+)\.\1\.\s+", r"\1. ", t)
    return t


# Grammetically correct verbs after conjunction
_VERB_AFTER_CONJUNCTION = frozenset(
    (
        "denied",
        "held",
        "found",
        "stated",
        "allowed",
        "rejected",
        "affirmed",
        "dismissed",
        "ordered",
        "rather",
        "concluded",
        "ruled",
        "determined",
        "noted",
        "observed",
        "accepted",
        "refused",
        "reversed",
        "remanded",
        "applied",
        "considered",
        "decided",
    )
)


def fix_bullet_starts_with_conjunction(text: str) -> str:
    if not text or not text.strip():
        return text
    s = text.strip()
    lower = s.lower()
    if lower.startswith("but ") or lower.startswith("but,"):
        rest = s[4:].lstrip() if lower.startswith("but ") else s[4:].lstrip()
        first = (rest.strip().split() or [""])[0].lower()
        if first in _VERB_AFTER_CONJUNCTION:
            if first == "denied":
                return "The Court noted that the Respondent " + (
                    rest[0].lower() + rest[1:] if len(rest) > 1 else rest
                )
            if first == "rather":
                after_rather = rest[6:].lstrip()
                return "The Court noted that the provision applies rather " + (
                    after_rather[0].lower() + after_rather[1:]
                    if len(after_rather) > 1
                    else after_rather
                )
            return "The Court noted that it " + (
                rest[0].lower() + rest[1:] if len(rest) > 1 else rest
            )
        return "The Court noted that " + (
            rest[0].lower() + rest[1:] if len(rest) > 1 else rest
        )
    if lower.startswith("even though "):
        rest = s[12:].lstrip()
        first = (rest.strip().split() or [""])[0].lower()
        if first in _VERB_AFTER_CONJUNCTION:
            return "The Court noted that it " + (
                rest[0].lower() + rest[1:] if len(rest) > 1 else rest
            )
        return "The Court noted that " + (
            rest[0].lower() + rest[1:] if len(rest) > 1 else rest
        )
    if lower.startswith("even "):
        rest = s[5:].lstrip()
        first = (rest.strip().split() or [""])[0].lower()
        if first in _VERB_AFTER_CONJUNCTION:
            return "The Court noted that it " + (
                rest[0].lower() + rest[1:] if len(rest) > 1 else rest
            )
        return "The Court noted that " + (
            rest[0].lower() + rest[1:] if len(rest) > 1 else rest
        )
    if lower.startswith("and the "):
        rest = s[8:].lstrip()
        return "The Court noted that " + (
            rest[0].lower() + rest[1:] if len(rest) > 1 else rest
        )
    return text


# Precedent verbatim, else model + decide_final_line
def summarize_clauses_safe_v2(
    clause_texts: List[str],
    summarize_fn: Callable[[str], str],
    config: dict = None,
    return_diagnostics: bool = False,
):
    config = config or ABSTRACTIVE_CONFIG
    result = []
    fallbacks = []
    for i, text in enumerate(clause_texts):
        orig = (text or "").strip()
        if not orig:
            result.append("")
            continue
        if contains_precedent_or_citation(orig):
            result.append(orig)
            if return_diagnostics:
                fallbacks.append((i, "verbatim: precedent"))
            continue
        model_out = summarize_fn(orig)
        final, used, reason = decide_final_line(
            orig,
            model_out,
            min_accepted_len=config.get("min_accepted_len", 15),
            run_minimal_cleanup=True,
        )
        result.append(final)
        if return_diagnostics and not used:
            fallbacks.append((i, reason))
    assert len(result) == len(
        clause_texts
    ), "bullet count mismatch: result len != clause_texts len"
    if return_diagnostics:
        return result, fallbacks
    return result


def normalize_space_preserve_newlines(text: str) -> str:
    """Collapse spaces per line; keep newlines."""
    if not text:
        return text
    lines = text.split("\n")
    return "\n".join(re.sub(r"\s+", " ", line).strip() for line in lines).strip()


# MAIN ENTRY POINT
def run_abstraction(ranked_dir, output_dir):
    """Run abstractive summarization on ranked clause JSONs."""
    global tokenizer, model
    ranked_dir = Path(ranked_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ranked = sorted(ranked_dir.glob("*.ranked.json"))
    if not all_ranked:
        raise FileNotFoundError(f"No *.ranked.json files in {ranked_dir}")

    if tokenizer is None or model is None:
        import logging

        logging.getLogger("transformers").setLevel(logging.WARNING)
        tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_ID)
        model = model.to(DEVICE)
        model.eval()
        print("Models loading done.")

    summarize_fn = lambda t: summarize_one_clause(
        t,
        max_new_tokens=ABSTRACTIVE_CONFIG["max_new_tokens"],
        min_new_tokens=ABSTRACTIVE_CONFIG["min_new_tokens"],
    )
    results_list = []

    for idx, ranked_path in enumerate(all_ranked):
        record = load_ranked(ranked_path)
        doc_id = record.get("doc_id") or ranked_path.stem.replace(".ranked", "")
        case_type = record.get("case_type")
        clause_ids, clause_texts = get_clause_texts_in_doc_order(record)
        primary_disp_id = get_primary_disposition_clause_id(record)
        disposition_text = get_disposition_text(record)
        reasoning_texts = [
            t
            for cid, t in zip(clause_ids, clause_texts)
            if primary_disp_id is None or cid != primary_disp_id
        ]
        reasoning_clause_ids = [
            cid
            for cid in clause_ids
            if primary_disp_id is None or cid != primary_disp_id
        ]
        if not reasoning_texts:
            print(f"Skip {doc_id}: no reasoning clauses")
            continue
        result, fallbacks = summarize_clauses_safe_v2(
            reasoning_texts,
            summarize_fn,
            config=ABSTRACTIVE_CONFIG,
            return_diagnostics=True,
        )
        if USE_THIRD_PERSON_STEP:
            result = [convert_to_third_person(s) for s in result]
            result = [fix_bullet_starts_with_conjunction(s) for s in result]
        disposition_display = (
            convert_to_third_person(disposition_text) if disposition_text else ""
        )
        lead = make_lead_sentence(case_type)
        lines = [lead, "", "Key reasoning (compressed from selected clauses):"]
        for i, text in enumerate(result, start=1):
            lines.append(f"{i}. {text}")
        if disposition_display:
            lines.extend(["", "Key holding (verbatim clause):", disposition_display])
        summary_final = normalize_space_preserve_newlines("\n".join(lines).strip())
        step1 = build_step1_summary(record)
        out = {
            "doc_id": doc_id,
            "case_type": case_type,
            "selected_clause_ids": clause_ids,
            "reasoning_clause_ids": reasoning_clause_ids,
            "num_bullets": len(result),
            "disposition_text": disposition_text,
            "original_bullets": reasoning_texts,
            "summarized_bullets": result,
            "summary_structured": step1["summary_structured"],
            "summary_final": summary_final,
            "summary_final_bullets": summary_final.split("\n"),
            "fallback_count": len(fallbacks),
        }
        out_path = output_dir / f"{doc_id}.abstracted.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        results_list.append((doc_id, len(result), len(fallbacks)))
        print(f"  {doc_id}: bullets={len(result)}, fallbacks={len(fallbacks)}")

    print(f"\n{len(results_list)} docs saved to {output_dir}.")
    return len(results_list)


if __name__ == "__main__":
    _ranked = Path(
        "/kaggle/input/datasets/muaadhnazly/rich-context-ranked/rich_context_ranked"
    )
    _out = Path("/kaggle/working/abstractive_summarization")
    run_abstraction(_ranked, _out)
