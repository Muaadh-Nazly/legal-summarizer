"""
Evaluate system summaries against headnotes by comparing:
  - Reference: content after the date and before Held (subject + facts) + "Held" (+ Held further, Per X J.) + "Cases referred to" from headnote .txt
  - Candidate: "Key reasoning" bullets only
  - Metrics: ROUGE-1, ROUGE-2, ROUGE-L; BLEU-4; BERTScore (F1); compression ratio (ref_words/cand_words); precedent coverage

Output: evaluation_results/report_rouge.json.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLR_DIR = Path(__file__).resolve().parent
HEADNOTES_DIR = SLR_DIR / "HeadNotes"
ABSTRACTIVE_DIR = PROJECT_ROOT / "Abstractive Summarization/Abstractive Summary"

RESULTS_DIR = SLR_DIR / "evaluation_results"

# Headnote stem patterns: SC_CA_*, SC_FR_*, SC_Spl_*, SC_SPL_*
HEADNOTE_STEM_TO_ABSTRACTED_PATTERN = re.compile(r"^SC_(CA|FR|Spl|SPL)_(.+)$", re.I)


def _candidates_for_headnote_stem(stem: str) -> List[str]:
    """
    Return possible abstracted JSON filenames for this headnote stem.
    Abstractive Summary folder uses full stem (e.g. SC_CA_04_2001.abstracted.json, SC_FR_107_2007.abstracted.json).
    Try that first, then fallbacks for other naming conventions (SC_{suffix}, zero-padded, etc.).
    """
    stem_clean = stem.replace(" ", "").strip()
    candidates: List[str] = []
    # Primary: full stem as used in Abstractive Summary
    candidates.append(f"{stem_clean}.abstracted.json")
    if stem_clean != stem:
        candidates.append(f"{stem}.abstracted.json")
    m = HEADNOTE_STEM_TO_ABSTRACTED_PATTERN.match(stem_clean)
    if m:
        typ, suffix = m.group(1), m.group(2)
        suffix_no_space = suffix.replace(" ", "").strip()
        if typ.upper() == "FR":
            candidates.append(f"SC_FR_{suffix_no_space}.abstracted.json")
        elif typ.upper() == "SPL":
            candidates.append(f"SC_SPL_{suffix_no_space}.abstracted.json")
            candidates.append(f"SC_{suffix_no_space}.abstracted.json")
        else:
            candidates.append(f"SC_{suffix_no_space}.abstracted.json")
            num_part = re.match(r"^(\d+)([A-Za-z]?_?\d.*)$", suffix_no_space)
            if num_part:
                num, rest = num_part.group(1), num_part.group(2)
                if len(num) <= 2:
                    padded = num.zfill(2) + rest
                    cand = f"SC_{padded}.abstracted.json"
                    if cand not in candidates:
                        candidates.append(cand)
    return candidates


def discover_pairs() -> Dict[str, str]:
    """Build headnote_stem -> abstracted_json_name for every headnote .txt that has a matching abstracted file."""
    pairs: Dict[str, str] = {}
    if not HEADNOTES_DIR.exists():
        return pairs
    for txt_path in sorted(HEADNOTES_DIR.glob("*.txt")):
        stem = txt_path.stem
        for json_name in _candidates_for_headnote_stem(stem):
            if (ABSTRACTIVE_DIR / json_name).exists():
                pairs[stem] = json_name
                break
    return pairs


def extract_pre_held_from_slr(txt_path: Path) -> str:
    """
    Extract the section after the header dates and before 'Held': subject/keywords and facts.
    """
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    # Find the last "date" line before "Held"
    month_start = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)"
    date_line_re = re.compile(r"^" + month_start + r".*?(19|20)\d{2}", re.I)
    last_date_idx: Optional[int] = None
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r"^Held\s*:?\s*$", s, re.I) or re.match(r"^Held\s*$", s, re.I):
            break
        if date_line_re.match(s) and len(s) < 80:
            last_date_idx = i

    start_idx = (last_date_idx + 1) if last_date_idx is not None else 0
    collected: List[str] = []
    for i in range(start_idx, len(lines)):
        s = lines[i].strip()
        if re.match(r"^Held\s*:?\s*$", s, re.I) or re.match(r"^Held\s*$", s, re.I):
            break
        if re.match(r"^Held\s+further\s*", s, re.I):
            break
        if s:
            collected.append(s)

    return " ".join(collected)


def extract_held_from_slr(txt_path: Path) -> str:
    """
    Extract Held section from headnote. Stops at 'Cases referred to', 'Cases Referred To', 'APPEAL from', 'Appeal from', or 'An APPLICATION'.
    """
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    started = False
    collected: List[str] = []
    for line in lines:
        s = line.strip()
        # Start of Held section
        if re.match(r"^Held\s*:?\s*$", s, re.I) or re.match(r"^Held\s*$", s, re.I):
            started = True
            continue
        if re.match(r"^Held\s+further\s*", s, re.I) or re.match(
            r"^Held\s+Further\s*", s
        ):
            started = True
            continue
        # "Per X, J." continuation of holdings
        if started and re.match(r"^Per\s+.+J\.", s):
            if s:
                collected.append(s)
            continue
        if not started:
            continue
        # Stop at next section
        if re.match(r"^Cases\s+referred\s+to\s*", s, re.I) or re.match(
            r"^Cases\s+Referred\s+To\s*", s
        ):
            break
        if re.match(r"^APPEAL\s+from\s+", s, re.I) or re.match(r"^Appeal\s+from\s+", s):
            break
        if re.match(r"^An?\s+APPLICATION\s+under", s, re.I):
            break
        if s:
            collected.append(s)

    return " ".join(collected)


def extract_cases_referred_to_from_slr(txt_path: Path) -> Tuple[str, List[str]]:
    """
    Extract 'Cases referred to' section from headnote.
    Returns (full block as text for reference, list of case name snippets for coverage).
    Stops at counsel lines or Cur. adv. vult.
    """
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    started = False
    collected: List[str] = []
    for line in lines:
        s = line.strip()
        if re.match(r"^Cases\s+referred\s+to\s*:?-?", s, re.I) or re.match(
            r"^Cases\s+Referred\s+To\s*", s
        ):
            started = True
            rest = re.sub(
                r"^Cases\s+referred\s+to\s*:?-?\s*", "", s, flags=re.I
            ).strip()
            rest = re.sub(r"^Cases\s+Referred\s+To\s*", "", rest, flags=re.I).strip()
            if rest:
                collected.append(rest)
            continue
        if not started:
            continue
        # stop at counsel
        if re.search(
            r"\s+for\s+(petitioner|respondent|Petitioner|Respondent)", s, re.I
        ):
            break
        if re.match(r"^Cur\.\s+adv\.\s+vult", s, re.I):
            break
        if s and not re.match(r"^[A-Z]\.\s+[A-Z]", s):
            collected.append(s)

    block = " ".join(collected)

    # Build list of case-name snippets for coverage
    case_names: List[str] = []
    for raw in collected:
        raw = re.sub(r"^\d+\.\s*", "", raw).strip()
        parts = re.split(r"\s+-\s+|\s+–\s+", raw, 1)
        name_part = (parts[0] if parts else raw).strip()
        if name_part and len(name_part) > 5:
            case_names.append(re.sub(r"\s+", " ", name_part).strip())

    return block, case_names


def precedent_coverage(
    slr_case_names: List[str], candidate_text: str
) -> Tuple[int, int, List[str]]:
    """
    How many SLR-cited case names appear in the candidate summary.
    Returns (matched_count, total_count, list of matched case name snippets).
    """
    if not slr_case_names:
        return 0, 0, []
    cand_lower = candidate_text.lower()
    matched: List[str] = []
    for name in slr_case_names:
        name_clean = re.sub(r"\s+", " ", name).strip().lower()
        if len(name_clean) < 4:
            continue
        if name_clean in cand_lower:
            matched.append(name)
            continue
        name_short = re.sub(r"\s+and\s+others?\s+", " ", name_clean, flags=re.I)
        if name_short in cand_lower:
            matched.append(name)
            continue
        tokens = [t for t in re.split(r"\W+", name_clean) if len(t) > 2]
        if len(tokens) >= 2 and all(t in cand_lower for t in tokens[:3]):
            matched.append(name)
    return len(matched), len(slr_case_names), matched


def extract_reasoning_from_abstracted(json_path: Path) -> str:
    """
    From abstracted JSON, extract only the Key reasoning bullets.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary_final = data.get("summary_final") or ""
    # Remove lead paragraph
    if "Key reasoning" in summary_final:
        summary_final = summary_final.split("Key reasoning", 1)[-1].strip()
    if "compressed from selected clauses):" in summary_final:
        summary_final = summary_final.split("):", 1)[-1].strip()
    # Remove "Key holding (verbatim clause):" and everything after
    if "Key holding" in summary_final:
        summary_final = summary_final.split("Key holding")[0].strip()
    summary_final = re.sub(r"\s+", " ", summary_final).strip()
    return summary_final


def run_rouge(reference: str, candidate: str) -> Optional[Dict]:
    """Apply ROUGE-1/2/L using rouge_score.RougeScorer class and .score(reference, candidate)."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": {
            "f1": scores["rouge1"].fmeasure,
            "p": scores["rouge1"].precision,
            "r": scores["rouge1"].recall,
        },
        "rouge2": {
            "f1": scores["rouge2"].fmeasure,
            "p": scores["rouge2"].precision,
            "r": scores["rouge2"].recall,
        },
        "rougeL": {
            "f1": scores["rougeL"].fmeasure,
            "p": scores["rougeL"].precision,
            "r": scores["rougeL"].recall,
        },
    }


def run_bleu(reference: str, candidate: str) -> Optional[Dict]:
    """
    Apply BLEU-4 using NLTK's sentence_bleu.
    Returns dict with bleu4 only.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu
    except ImportError:
        return None
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return {"bleu4": 0.0}
    bleu4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    return {"bleu4": bleu4}


def run_bertscore_batch(refs: List[str], cands: List[str]) -> Optional[Tuple]:
    """
    Apply BERTScore using the bert_score library.
    Returns (P, R, F1) as lists of floats, or None if unavailable or on error.
    """
    if not refs or not cands or len(refs) != len(cands):
        return None
    try:
        from bert_score import BERTScorer
    except ImportError:
        return None
    refs = [re.sub(r"\s+", " ", s).strip() for s in refs]
    cands = [re.sub(r"\s+", " ", s).strip() for s in cands]
    try:
        scorer = BERTScorer(lang="en")
        P, R, F1 = scorer.score(cands, refs, verbose=False)
    except (RuntimeError, AttributeError, OSError) as e:
        import warnings

        warnings.warn(f"BERTScore skipped due to: {e}", UserWarning)
        return None
    # Convert tensors to Python floats
    P = [float(p) for p in P]
    R = [float(r) for r in R]
    F1 = [float(f) for f in F1]
    return (P, R, F1)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    extracted_dir = RESULTS_DIR / "extracted"
    extracted_dir.mkdir(exist_ok=True)

    PAIRS = discover_pairs()
    all_headnote_stems = {p.stem for p in HEADNOTES_DIR.glob("*.txt")}
    headnotes_without_abstracted = sorted(all_headnote_stems - set(PAIRS.keys()))
    print(f"Discovered {len(PAIRS)} headnote–abstracted pairs from {HEADNOTES_DIR}")
    if headnotes_without_abstracted:
        print(
            f"  Headnotes with no matching abstracted file ({len(headnotes_without_abstracted)}): {headnotes_without_abstracted}"
        )

    rows: List[Dict] = []
    refs_list: List[str] = []
    cands_list: List[str] = []
    for slr_stem, json_name in PAIRS.items():
        slr_path = HEADNOTES_DIR / f"{slr_stem}.txt"
        json_path = ABSTRACTIVE_DIR / json_name
        if not slr_path.exists():
            print(f"Skip (missing SLR): {slr_stem}.txt")
            continue
        if not json_path.exists():
            print(f"Skip (missing abstracted): {json_name}")
            continue

        pre_held_text = extract_pre_held_from_slr(slr_path)
        held_text = extract_held_from_slr(slr_path)
        cases_block, slr_case_names = extract_cases_referred_to_from_slr(slr_path)
        ref_text = f"{pre_held_text} {held_text} {cases_block}".strip()
        cand_text = extract_reasoning_from_abstracted(json_path)

        # Precedent coverage: how many SLR-cited cases appear in candidate
        prec_matched, prec_total, prec_matched_list = precedent_coverage(
            slr_case_names, cand_text
        )

        (extracted_dir / f"{slr_stem}_reference.txt").write_text(
            ref_text, encoding="utf-8"
        )
        (extracted_dir / f"{slr_stem}_reference_pre_held_only.txt").write_text(
            pre_held_text, encoding="utf-8"
        )
        (extracted_dir / f"{slr_stem}_reference_held_only.txt").write_text(
            held_text, encoding="utf-8"
        )
        (extracted_dir / f"{slr_stem}_reference_cases_only.txt").write_text(
            cases_block, encoding="utf-8"
        )
        (extracted_dir / f"{slr_stem}_candidate.txt").write_text(
            cand_text, encoding="utf-8"
        )

        ref_words = len(ref_text.split())
        cand_words = len(cand_text.split())
        compression_ratio = ref_words / cand_words if cand_words else 0.0

        rouge = run_rouge(ref_text, cand_text)
        bleu = run_bleu(ref_text, cand_text)
        refs_list.append(ref_text)
        cands_list.append(cand_text)
        row = {
            "slr_id": slr_stem,
            "system_json": json_name,
            "ref_words": ref_words,
            "cand_words": cand_words,
            "compression_ratio": round(compression_ratio, 4),
            "rouge": rouge,
            "bleu": bleu,
            "bertscore": None,
            "precedent_coverage": {
                "matched": prec_matched,
                "total": prec_total,
                "recall": prec_matched / prec_total if prec_total else 0.0,
                "matched_cases": prec_matched_list,
            },
        }
        rows.append(row)
        prec_str = f" precedents {prec_matched}/{prec_total}" if prec_total else ""
        r_str = f" R1={rouge['rouge1']['f1']:.3f}" if rouge else ""
        b_str = f" BLEU4={bleu['bleu4']:.3f}" if bleu else ""
        cr_str = f" CR={compression_ratio:.2f}" if cand_words else ""
        print(
            f"  {slr_stem}: ref={ref_words}w cand={cand_words}w{r_str}{b_str}{cr_str}{prec_str}"
        )

    # BERTScore in one batch
    if refs_list and cands_list:
        bert_result = run_bertscore_batch(refs_list, cands_list)
        if bert_result is not None:
            P_list, R_list, F1_list = bert_result
            for i, row in enumerate(rows):
                if i < len(P_list):
                    row["bertscore"] = {
                        "precision": round(P_list[i], 4),
                        "recall": round(R_list[i], 4),
                        "f1": round(F1_list[i], 4),
                    }
            print("  BERTScore computed for all pairs.")
        else:
            print("  BERTScore skipped.")

    # Save mapping for reference
    pairs_path = RESULTS_DIR / "pairs.csv"
    with open(pairs_path, "w", encoding="utf-8") as f:
        f.write("slr_reference,system_json\n")
        for slr_stem, json_name in PAIRS.items():
            f.write(f"{slr_stem}.txt,{json_name}\n")

    # Aggregate summary (mean ± std)
    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        n = len(vals)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n if n else 0
        std = var**0.5
        return round(mean, 4), round(std, 4)

    r1_f1 = [
        r.get("rouge", {}).get("rouge1", {}).get("f1")
        for r in rows
        if (r.get("rouge") or {}).get("rouge1")
    ]
    r2_f1 = [
        r.get("rouge", {}).get("rouge2", {}).get("f1")
        for r in rows
        if (r.get("rouge") or {}).get("rouge2")
    ]
    rl_f1 = [
        r.get("rouge", {}).get("rougeL", {}).get("f1")
        for r in rows
        if (r.get("rouge") or {}).get("rougeL")
    ]
    bleu4_vals = [
        r.get("bleu", {}).get("bleu4")
        for r in rows
        if (r.get("bleu") or {}).get("bleu4") is not None
    ]
    bs_f1_vals = [
        r.get("bertscore", {}).get("f1")
        for r in rows
        if (r.get("bertscore") or {}).get("f1") is not None
    ]
    cr_vals = [
        r.get("compression_ratio")
        for r in rows
        if r.get("compression_ratio") is not None
    ]
    prec_recall = []
    for r in rows:
        pc = r.get("precedent_coverage") or {}
        if pc.get("total", 0) > 0:
            prec_recall.append(pc.get("recall", 0))

    summary = {
        "n": len(rows),
        "ROUGE-1 F1": {"mean": _mean_std(r1_f1)[0], "std": _mean_std(r1_f1)[1]},
        "ROUGE-2 F1": {"mean": _mean_std(r2_f1)[0], "std": _mean_std(r2_f1)[1]},
        "ROUGE-L F1": {"mean": _mean_std(rl_f1)[0], "std": _mean_std(rl_f1)[1]},
        "BLEU-4": {"mean": _mean_std(bleu4_vals)[0], "std": _mean_std(bleu4_vals)[1]},
        "BERTScore F1": (
            {"mean": _mean_std(bs_f1_vals)[0], "std": _mean_std(bs_f1_vals)[1]}
            if bs_f1_vals
            else None
        ),
        "Compression ratio": {
            "mean": _mean_std(cr_vals)[0],
            "std": _mean_std(cr_vals)[1],
        },
        "Precedent recall": {
            "mean": _mean_std(prec_recall)[0],
            "std": _mean_std(prec_recall)[1],
            "n_with_precedents": len(prec_recall),
        },
    }

    report = {
        "num_pairs": len(rows),
        "num_headnotes_total": len(all_headnote_stems),
        "headnotes_without_abstracted": headnotes_without_abstracted,
        "summary": summary,
        "pairs": [p["slr_id"] for p in rows],
        "results": rows,
        "note": "Metrics: ROUGE-1/2/L, BLEU-4, BERTScore (F1), compression_ratio (ref_words/cand_words). Reference = pre-Held + Held + Cases referred to; Candidate = Key reasoning only. Precedent coverage = headnote-cited cases in candidate.",
    }

    report_json = RESULTS_DIR / "report_rouge.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults: {RESULTS_DIR}")
    print("  Summary (for thesis):")
    print(
        f"    ROUGE-1 F1: {summary['ROUGE-1 F1']['mean']:.4f} ± {summary['ROUGE-1 F1']['std']:.4f}"
    )
    print(
        f"    ROUGE-2 F1: {summary['ROUGE-2 F1']['mean']:.4f} ± {summary['ROUGE-2 F1']['std']:.4f}"
    )
    print(
        f"    ROUGE-L F1: {summary['ROUGE-L F1']['mean']:.4f} ± {summary['ROUGE-L F1']['std']:.4f}"
    )
    print(
        f"    BLEU-4:     {summary['BLEU-4']['mean']:.4f} ± {summary['BLEU-4']['std']:.4f}"
    )
    if summary.get("BERTScore F1"):
        print(
            f"    BERTScore F1: {summary['BERTScore F1']['mean']:.4f} ± {summary['BERTScore F1']['std']:.4f}"
        )
    print(
        f"    Compression ratio: {summary['Compression ratio']['mean']:.4f} ± {summary['Compression ratio']['std']:.4f}"
    )
    print(f"  report_rouge.json, pairs.csv, extracted/*.txt")


if __name__ == "__main__":
    main()
