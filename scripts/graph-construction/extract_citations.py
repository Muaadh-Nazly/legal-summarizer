"""
Extract Precedent Citations from Annotated Clause JSONs

Reads only annotated *.clauses.json files. The script:
- Extracts precedent citations from clause text (regex/patterns)
- Maps citations to document IDs
- Optionally finds target clauses in cited documents using InCaseLawBERT similarity
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch


_annotated_dirs = [
    Path("/kaggle/input/datasets/muaadhnazly/final-annotated/final_annotated")
]

ANNOTATED_DIRS = [d for d in _annotated_dirs if d.exists()] or _annotated_dirs

OUTPUT_FILE = "graph_citations.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

USE_SEMANTIC_MATCHING = True
MODEL_NAME = "law-ai/InCaseLawBERT"


def extract_case_number_year(text: str) -> Optional[Tuple[str, str]]:
    """
    Extract case number and year from any document ID or citation format.
    Returns:
        Tuple of (number, year) or None if not found
    """

    patterns = [
        # SC_CA or SC_FR or SC
        r"SC[_\s]?(?:CA|FR|SPL)?[_\s]?(\d+)[_\s]?(\d{4})",
        # SC/CA/ or SC/FR/
        r"SC[/\s]?(?:CA|FR|APPEAL)[/\s]?(\d+)[/\s]?(\d{4})",
        # SC Appeal or SC Appeal No
        r"SC\s+Appeal\s+(?:No[:\s]*)?(\d+)[/\s]?(\d{4})",
        # SC (shortened)
        r"SC[_\s/](\d+)[_\s/](\d{4})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return (match.group(1), match.group(2))

    return None


def normalize_citation_to_doc_id(citation: str, current_doc_id: str) -> Optional[str]:
    """
    Convert citation format to document ID format.
    Detects and filters self-citations.
    Args:
        current_doc_id: Current document ID to exclude self-citations
    Returns:
        Document ID in format "SC_CA_" or "SC_FR_", or None if self-citation
    """

    # Extract current case number and year
    current_case_info = extract_case_number_year(current_doc_id)
    current_number = current_case_info[0] if current_case_info else None
    current_year = current_case_info[1] if current_case_info else None

    # Parse citation to extract case type, number, and year
    citation_case_info = extract_case_number_year(citation)

    if not citation_case_info:
        return None

    citation_number = citation_case_info[0]
    citation_year = citation_case_info[1]

    # Determine case type from citation text
    case_type = "CA"
    if re.search(r"SC[/\s]?FR[/\s]?", citation, re.IGNORECASE):
        case_type = "FR"
    elif re.search(r"SC[/\s]?CA[/\s]?", citation, re.IGNORECASE):
        case_type = "CA"
    elif re.search(r"SC\s+Appeal", citation, re.IGNORECASE):
        case_type = "CA"

    # Self-citation check
    if current_number and current_year and citation_number and citation_year:
        if citation_number == current_number and citation_year == current_year:
            return None

    doc_id = f"SC_{case_type}_{citation_number}_{citation_year}"
    return doc_id


def extract_citations_from_text(text: str, current_doc_id: str) -> List[Dict]:
    """
    Extract all precedent citations from a text string.

    Returns:
        List of citation dicts with format and position info
    """
    citations = []

    # Comprehensive citation patterns
    patterns = [
        # Standard formats
        (r"SC/CA/(\d+)/(\d{4})", "SC/CA/{}/{}/{}"),
        (r"SC/FR/(\d+)/(\d{4})", "SC/FR/{}/{}/{}"),
        (r"SC/APPEAL/(\d+)/(\d{4})", "SC/APPEAL/{}/{}/{}"),
        (r"SC Appeal No[:\s]*(\d+)/(\d{4})", "SC Appeal No {}/{}"),
        (r"SC APPEAL NO[:\s]*(\d+)/(\d{4})", "SC APPEAL NO {}/{}"),
        (r"SC Appeal (\d+)/(\d{4})", "SC Appeal {}/{}"),
        (r"S\.C Appeal No[:\s]*(\d+)/(\d{4})", "S.C Appeal No {}/{}"),
        (r"S\.C\. Appeal No[:\s]*(\d+)/(\d{4})", "S.C. Appeal No {}/{}"),
        (r"SC FR No[:\s]*(\d+)/(\d{4})", "SC FR No {}/{}"),
        (r"SC FR (\d+)/(\d{4})", "SC FR {}/{}"),
        # Parenthetical
        (r"\([^)]*SC Appeal No[^)]*(\d+)/(\d{4})[^)]*\)", "SC Appeal No {}/{}"),
        (r"\[[^\]]*SC Appeal No[^\]]*(\d+)/(\d{4})[^\]]*\]", "SC Appeal No {}/{}"),
        (r"\([^)]*SC/CA/[^)]*(\d+)/(\d{4})[^)]*\)", "SC/CA/{}/{}"),
        (r'\[[^\]]*SC/CA/[^"]*(\d+)/(\d{4})[^\]]*\]', "SC/CA/{}/{}"),
        # Case name + citation
        (
            r"[A-Z][^[]*\[[^\]]*SC Appeal No[^\]]*(\d+)/(\d{4})[^\]]*\]",
            "SC Appeal No {}/{}",
        ),
        (r'[A-Z][^[]*\[[^\]]*SC/CA/[^"]*(\d+)/(\d{4})[^\]]*\]', "SC/CA/{}/{}"),
        # Cross-citations
        (r"cited.*?SC/CA/(\d+)/(\d{4})", "SC/CA/{}/{}"),
        (r"cited.*?SC/FR/(\d+)/(\d{4})", "SC/FR/{}/{}"),
        (r"referred.*?SC/CA/(\d+)/(\d{4})", "SC/CA/{}/{}"),
        (r"referred.*?SC/FR/(\d+)/(\d{4})", "SC/FR/{}/{}"),
        (r"following.*?SC/CA/(\d+)/(\d{4})", "SC/CA/{}/{}"),
        (r"following.*?SC/FR/(\d+)/(\d{4})", "SC/FR/{}/{}"),
        # Shortened format
        (r"SC/(\d+)/(\d{4})", "SC/{}/{}"),
    ]

    for pattern, format_str in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            number = match.group(1)
            year = match.group(2)

            # Determine case type from pattern
            if "FR" in pattern or "FR" in match.group(0):
                case_type = "FR"
            elif "APPEAL" in pattern.upper():
                case_type = "CA"
            else:
                # Check context
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]
                if "FR" in context.upper():
                    case_type = "FR"
                else:
                    case_type = "CA"

            # Format citation
            if "SC/CA" in format_str or "SC/FR" in format_str:
                citation_text = f"SC/{case_type}/{number}/{year}"
            elif "SC Appeal" in format_str or "SC APPEAL" in format_str:
                citation_text = f"SC Appeal No {number}/{year}"
            else:
                citation_text = f"SC/{number}/{year}"

            # Convert to doc_id
            doc_id = normalize_citation_to_doc_id(citation_text, current_doc_id)

            if doc_id:
                citations.append(
                    {
                        "citation_text": citation_text,
                        "cited_doc_id": doc_id,
                        "match_text": match.group(0),
                        "position": (match.start(), match.end()),
                    }
                )

    # Remove duplicates
    seen = set()
    unique_citations = []
    for cit in citations:
        key = (cit["cited_doc_id"], cit["citation_text"])
        if key not in seen:
            seen.add(key)
            unique_citations.append(cit)

    return unique_citations


# SEMANTIC MATCHING to find target clauses
class SemanticMatcher:
    """Helper class for finding target clauses using semantic similarity."""

    def __init__(self):
        if USE_SEMANTIC_MATCHING:
            print(f"Loading {MODEL_NAME} for semantic matching...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
            self.model.eval()
        else:
            self.tokenizer = None
            self.model = None

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        if not self.model:
            return None

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            token_emb = outputs.last_hidden_state
            mask = (
                inputs["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            )
            embedding = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return embedding.cpu().numpy()[0]

    def extract_quoted_text(self, text: str) -> Optional[str]:
        """
        Extract quoted text from source clause.
        Looks for text within quotes after citation patterns.
        """
        # Pattern: citation followed by quoted text
        patterns = [
            r'that\s+["""]([^"""]+)["""]',  # that "text"
            r'["""]([^"""]+)["""]',  # "text" (anywhere)
            r'["""]([^"""]{50,})["""]',  # Long quoted text (50+ chars)
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                quoted = match.group(1).strip()
                if len(quoted) > 30:
                    return quoted
        return None

    def find_target_clause(
        self, source_clause_text: str, cited_doc: Dict, threshold: float = 0.6
    ) -> Optional[Dict]:
        """
        Find the best matching clause in the cited document.
        cited_doc must have "clauses": [{"id", "text"}, ...].
        Tries both full text and extracted quoted text for better matching.

        Returns:
            Dict with target clause info, or None if no good match
        """
        if not self.model:
            return None

        clauses = cited_doc.get("clauses", [])
        if not clauses:
            return None

        # Try to extract quoted text first
        quoted_text = self.extract_quoted_text(source_clause_text)

        # Get embeddings for source text
        source_emb = self.get_embedding(source_clause_text)
        if source_emb is None:
            return None

        quoted_emb = None
        if quoted_text:
            quoted_emb = self.get_embedding(quoted_text)

        # Get embeddings for all clauses in cited document
        best_match = None
        best_score = threshold - 0.01

        for clause in clauses:
            clause_text = clause.get("text", "")
            if not clause_text.strip():
                continue

            clause_emb = self.get_embedding(clause_text)
            if clause_emb is None:
                continue

            # Try similarity with full source text
            similarity_full = cosine_similarity(
                source_emb.reshape(1, -1), clause_emb.reshape(1, -1)
            )[0][0]

            # Try with quoted text if available
            similarity = similarity_full
            if quoted_emb is not None:
                similarity_quoted = cosine_similarity(
                    quoted_emb.reshape(1, -1), clause_emb.reshape(1, -1)
                )[0][0]

                similarity = max(similarity_full, similarity_quoted)

            if similarity > best_score:
                best_score = similarity
                best_match = {
                    "target_clause_id": clause.get("id"),
                    "target_clause_text": clause_text,
                    "similarity_score": float(similarity),
                }

        return best_match


# LOAD DOCUMENT
def parse_annotated_to_clauses(data: Dict, fallback_doc_id: str = "") -> Optional[Dict]:
    """
    Parse annotated JSON into a uniform structure.
    Omits clauses with label "None".
    Returns {"doc_id": str, "clauses": [{"id", "text"}, ...]} or None.
    """
    doc_id = data.get("doc_id") or data.get("document_id") or fallback_doc_id
    raw_clauses = data.get("clauses", [])
    if not raw_clauses:
        return None
    clauses = [
        {"id": c.get("clause_id", i), "text": c.get("text", "")}
        for i, c in enumerate(
            c for c in raw_clauses if c.get("label", "None") != "None"
        )
    ]
    if not clauses:
        return None
    return {"doc_id": doc_id, "clauses": clauses}


def load_doc_for_extraction(file_path: Path) -> Optional[Dict]:
    """
    Load a document from annotated JSON
    Returns {"doc_id": str, "clauses": [{"id", "text"}, ...]} or None.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    doc_id = data.get("doc_id") or data.get("document_id") or file_path.stem
    return parse_annotated_to_clauses(data, doc_id)


# MAIN EXTRACTION FUNCTION
def extract_citations_from_doc(
    doc_path: Path, semantic_matcher: Optional[SemanticMatcher] = None
) -> Optional[Dict]:
    """
    Extract citations from a single annotated JSON file (clauses with clause_id, text, label).
    Returns:
        Dict with doc_id and list of citations, or None if load failed.
    """
    print(f"Processing: {doc_path.name}")

    doc_data = load_doc_for_extraction(doc_path)
    if not doc_data:
        print(f"  ⚠️  Skipped (no 'clauses'): {doc_path.name}")
        return None

    doc_id = doc_data["doc_id"]
    clauses = doc_data["clauses"]

    # Extract citations from each clause
    all_citations = []

    for clause in clauses:
        clause_id = clause.get("id")
        clause_text = clause.get("text", "")

        if not clause_text.strip():
            continue

        # Extract citations from this clause
        citations = extract_citations_from_text(clause_text, doc_id)

        for cit in citations:
            citation_entry = {
                "source_clause_id": clause_id,
                "source_clause_text": clause_text,
                "cited_doc_id": cit["cited_doc_id"],
                "citation_text": cit["citation_text"],
                "match_text": cit["match_text"],
                "target_clause_id": None,
                "target_clause_text": None,
                "similarity_score": None,
            }

            # Try to find target clause if semantic matching is enabled
            if semantic_matcher and USE_SEMANTIC_MATCHING:
                # Try to load cited document's clauses
                cited_doc_id = cit["cited_doc_id"]

                # Candidate file names
                def _candidate_names(doc_id: str):
                    return [
                        f"{doc_id}.clauses.json",
                        f"{doc_id}.json",
                    ]

                possible_names = []
                # Prefer short form first if cited_doc_id has CA/FR
                if "_CA_" in cited_doc_id or "_FR_" in cited_doc_id:
                    match = re.search(
                        r"SC[_\s]?(?:CA|FR)[_\s]?(\d+)[_\s]?(\d{4})",
                        cited_doc_id,
                        re.IGNORECASE,
                    )
                    if match:
                        number, year = match.group(1), match.group(2)
                        alt_doc_id = f"SC_{number}_{year}"
                        possible_names.extend(_candidate_names(alt_doc_id))
                possible_names.extend(_candidate_names(cited_doc_id))

                search_dirs = [d for d in ANNOTATED_DIRS if d.exists()]

                cited_doc = None
                found_file = None
                for name in possible_names:
                    cited_doc_path = None
                    for d in search_dirs:
                        p = d / name
                        if p.exists():
                            cited_doc_path = p
                            break
                    if cited_doc_path is None:
                        continue
                    try:
                        with open(cited_doc_path, "r", encoding="utf-8") as f:
                            raw = json.load(f)
                        cited_doc = parse_annotated_to_clauses(raw, cited_doc_id)
                        if cited_doc is None:
                            continue
                        found_file = name
                        break
                    except Exception as e:
                        print(f"    ⚠️  Error loading {name}: {e}")
                        continue

                if cited_doc:
                    # Find target clause
                    try:
                        match = semantic_matcher.find_target_clause(
                            clause_text, cited_doc, threshold=0.6
                        )
                        if match:
                            citation_entry["target_clause_id"] = match[
                                "target_clause_id"
                            ]
                            citation_entry["target_clause_text"] = match[
                                "target_clause_text"
                            ]
                            citation_entry["similarity_score"] = match[
                                "similarity_score"
                            ]
                            if len(all_citations) % 10 == 0:
                                print(
                                    f"    ✓ Found target clause for {cited_doc_id} (similarity: {match['similarity_score']:.3f})"
                                )
                        else:
                            if len(all_citations) % 20 == 0:
                                print(
                                    f"    ℹ️  No target clause found for {cited_doc_id} (similarity < 0.6)"
                                )
                    except Exception as e:
                        print(
                            f"    ⚠️  Error finding target clause for {cited_doc_id}: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                elif cited_doc_id:
                    print(
                        f"    ⚠️  Cited doc not found: {cited_doc_id} (tried both "
                        "SC_number_year and SC_CA_/SC_FR_ names in ANNOTATED_DIRS)"
                    )
                    print(f"       Search dirs: {[str(d) for d in search_dirs]}")

            all_citations.append(citation_entry)

    return {
        "doc_id": doc_id,
        "source_file": doc_path.name,
        "total_clauses": len(clauses),
        "citations": all_citations,
        "unique_cited_docs": len(set(c["cited_doc_id"] for c in all_citations)),
    }


# MAIN FUNCTION
def main():
    """Extract citations from annotated JSONs."""

    print("=" * 80)
    print("EXTRACTING PRECEDENT CITATIONS (ANNOTATED ONLY)")
    print("=" * 80)

    doc_files = []
    for d in ANNOTATED_DIRS:
        if d.exists():
            doc_files.extend(d.glob("*.clauses.json"))
    doc_files = sorted(set(doc_files))
    if not doc_files:
        print(f"❌ No *.clauses.json found under {ANNOTATED_DIRS}")
        return
    print(f"Annotated dirs: {ANNOTATED_DIRS}")
    print(f"Found {len(doc_files)} file(s)")
    print(f"Using device: {DEVICE}")
    print(f"Semantic matching: {'ENABLED' if USE_SEMANTIC_MATCHING else 'DISABLED'}")
    print()

    semantic_matcher = SemanticMatcher() if USE_SEMANTIC_MATCHING else None

    # Process each annotated JSON
    all_results = []
    stats = {
        "total_docs": 0,
        "docs_with_citations": 0,
        "total_citations": 0,
        "citations_with_targets": 0,
    }

    for doc_file in doc_files:
        try:
            result = extract_citations_from_doc(doc_file, semantic_matcher)
            if result is None:
                continue
            all_results.append(result)

            stats["total_docs"] += 1
            if result["citations"]:
                stats["docs_with_citations"] += 1
                stats["total_citations"] += len(result["citations"])
                stats["citations_with_targets"] += sum(
                    1 for c in result["citations"] if c["target_clause_id"] is not None
                )

                print(
                    f"  ✓ {result['doc_id']}: {len(result['citations'])} citations "
                    f"({result['unique_cited_docs']} unique docs)"
                )
        except Exception as e:
            print(f"  ❌ Error processing {doc_file.name}: {e}")

    # Save results
    output = {
        "metadata": {
            "total_documents_processed": stats["total_docs"],
            "documents_with_citations": stats["docs_with_citations"],
            "total_citations_found": stats["total_citations"],
            "citations_with_target_clauses": stats["citations_with_targets"],
            "semantic_matching_enabled": USE_SEMANTIC_MATCHING,
        },
        "citations_by_document": all_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total documents: {stats['total_docs']}")
    print(f"Documents with citations: {stats['docs_with_citations']}")
    print(f"Total citations: {stats['total_citations']}")
    print(f"Citations with target clauses: {stats['citations_with_targets']}")
    print(f"\n✅ Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
