import re
import os
import regex as re

# -------------
# INLINE RULES
# -------------
INLINE_RULES = [
    
    # BRACKET AND PARENTHESIS HANDLING
    # Move stray dots after brackets to the end of the bracket (e.g., "(text)\n." → "(text).\n")
    {"pattern": r"(\([^\)]*\)|\[[^\]]+\])\s*\n\s*\.\s*", "replacement": r"\1.\n"},

    # Attach bracketed notes on their own line to the previous sentence (e.g., "\n(note)" → " (note) ")
    {"pattern": r"\n\s*(\([^\)]*\)|\[[^\]]+\])\s*", "replacement": r" \1 "},

    # Ensure brackets followed by enumerations end with a dot and newline (e.g., "(note) 2." → "(note).\n")
    {"pattern": r"(\([^\)]*\)|\[[^\]]+\])\s*(?:\.)?\s*(?=(?:\d+\.)|(?:[ivxlcdmIVXLCDM]{1,4}\.))", "replacement": r"\1.\n"},

    # Join numeric enumerations on their own line with following text (e.g., "(2).\nNext" → "(2). Next")
    {"pattern": r"(?m)^[ \t]*(\(\d+\)|\d+)\.\s*\n\s*([A-Z])", "replacement": r"\1. \2"},

    # Join lowercase roman enumerations on their own line with following text (e.g., "i.\nNext" → "i. Next")
    {"pattern": r"(?m)^[ \t]*(i|ii|iii|iv|v|vi|vii|viii|ix|x)\.\s*\n\s*([A-Z])", "replacement": r"\1. \2"},

    # Remove stray lines containing only a single dot (e.g., "\n.\n" → "\n")
    {"pattern": r"(?m)^[ \t]*\.[ \t]*\n", "replacement": r""},

    # Normalize bracketed letters broken by newlines (e.g., "(a)\nto\n(b)" → "(a) to (b)")
    {"pattern": r"\(\s*([a-zA-Z])\s*\)\s*\n\s*to\s*\n\s*\(\s*([a-zA-Z])\s*\)", "replacement": r"(\1) to (\2)"},

    # Join consecutive bracketed letters (e.g., "(a)\n(b)" → "(a) (b)")
    {"pattern": r"\(\s*([a-zA-Z])\s*\)\s*\n\s*\(\s*([a-zA-Z])\s*\)", "replacement": r"(\1) (\2)"},

    # Join bracketed letters with following text (e.g., "(a)\nNext" → "(a) Next")
    {"pattern": r"\(\s*([a-zA-Z])\s*\)\s*\n", "replacement": r"(\1) "},
    

    # SENTENCE SPLITTING RULES
    # Split when closing brackets end sentences followed by capital letters (e.g., ". (note) Next" → ". (note)\nNext")
    {"pattern": r"\.\s*(\((?!\d+\)|[ivxlcdmIVXLCDM]+\)|[a-zA-Z]\))[^)]*\]?\)|\[[^\]]+\])\s+(?=[A-Z])", "replacement": r". \1\n"},

    # Split sentences at punctuation followed by capital letters but not after numbers/abbreviations (e.g., ". Next" → ".\nNext" but not "No. 123")
    {"pattern": r"(?<!\b(?:\d{1,3}|[a-zA-Z]|[ivxlcdmIVXLCDM]))([.?!])\s+(?=[A-Z])", "replacement": r"\1\n"},

    # Split before lowercase Roman numeral enumerations (e.g., "text i." → "text\ni.")
    {"pattern": r"(?<=[.?!;])\s+(?=\b(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\.\s)", "replacement": r"\n"},

    # Split before Roman numeral enumerations followed by capital letters (e.g., "text (i) Next" → "text\n(i) Next")
    {"pattern": r"(?i)(?<=[,;.\)])\s*(?=\b(?:[ivxlcdm]{1,4})\.\s+[A-Z])", "replacement": r"\n"},

    # Split before Roman numerals after conjunctions (e.g., "text and i." → "text and\ni.")
    {"pattern": r"(?<=[.;:])\s+(?:and\s+|or\s+)?(?=(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\.\s)", "replacement": r"\n"},

    # Join Roman numerals on their own line with following text (e.g., "i.\nNext" → "i. Next")
    {"pattern": r"^(?:\s*)(i|ii|iii|iv|v|vi|vii|viii|ix|x)\.\s*\n\s*([A-Z])", "replacement": r"\1. \2"},

    # Split sentences but not after "No." or enumerations like 12(i), (i) (e.g., ". Next" → ".\nNext" but not "No. 123")
    {"pattern": r"(?<!\b[Nn]o)\.(?=\s+(?!\d+\(|\([ivxlcdm]+\)))([A-Z])", "replacement": r".\n\1"},

    # Normalize punctuation with closing quotes/brackets when broken across line (e.g., ". \"" → ".\"")
    {"pattern": r"([.?!])\s*(?:\n\s*)?([\"\"'')])\s*([.,;:])?", "replacement": r"\1\2\3 "},

    # Split sentences before enumerations (e.g., ". (1) Next" → ".\n(1) Next")
    {"pattern": r'''([.?!]["'"\)]*)\s+(\d+\.|\([ivxlcdmIVXLCDM]+\)|\([a-zA-Z]\))\s+(?=[A-Z])''', "replacement": r'\1\n\2 '},

    # Join enumerations with following text (e.g., "41A.\nRelevancy" → "41A. Relevancy")
    {"pattern": r"(\b\d+[A-Z]?\.)\s*\n\s*([A-Z])", "replacement": r"\1 \2"},

    # Split after punctuation with closing quotes (e.g., ".\" Next" → ".\"\nNext")
    {"pattern": r"([.?!][\"\"'')])\s+(?=[A-Z])", "replacement": r"\1\n"},

    # Split before bracketed enumerations (e.g., "text (i)" → "text\n(i)")
    {"pattern": r"(?<!\b[Nn]o)([.?!:;])\s+(?=\(\s*(?:[ivxlcdmIVXLCDM]+|[0-9]+|[A-Za-z])\s*\)|[0-9]+\s*\)|[A-Za-z]\s*\))", "replacement": r"\1\n"},

    # Split after quotes followed by capital letters (e.g., ".\" Next" → ".\"\nNext")
    {"pattern": r'''([.?!:])"\s+(?=[A-Z])''', "replacement": r'\1"\n'},
    
    
    # TIME AND ABBREVIATION HANDLING
    # Join abbreviated words split across lines (e.g., "Dr.\nSmith" → "Dr. Smith")
    {"pattern": r"\b([A-Z][a-z]{1,3})\.\s*\n\s*([A-Z][a-z.]*)", "replacement": r"\1. \2"},

    # Fix time formats split across lines PM (e.g., "2.30 p.\nm." → "2.30 p.m.")
    {"pattern": r"(\d{1,2}\.\d{1,2})\s*p\.\s*\n\s*m\.", "replacement": r"\1 p.m."},

    # Fix time formats split across lines AM (e.g., "2.30 a.\nm." → "2.30 a.m.")
    {"pattern": r"(\d{1,2}\.\d{1,2})\s*a\.\s*\n\s*m\.", "replacement": r"\1 a.m."},

    # Remove periods from company abbreviations to prevent false sentence splits (e.g., "Ltd." → "Ltd")
    {"pattern": r"\b(Ltd|Pvt|Co|Inc|Corp|PLC|LLC)\.", "replacement": r"\1"},
    {"pattern": r"\(Pvt\)\s*\n\s*(Ltd)", "replacement": r"(Pvt) \1"},

    # Normalize sic notations (e.g., ".\n(sic)" → ". (sic)\n")
    {"pattern": r"\.\s*\n\(?sic\)?\s*(?=(?:[0-9]+\.|\([ivxlcdm]+\)))?", "replacement": r". (sic)\n"},

    # Join content split across square brackets (e.g., "[text\nmore]" → "[text more]")
    {"pattern": r"\[\s*([^\[\]]*?)\n\s*([^\[\]]*?)\]", "replacement": r"[\1 \2]"},

    # Join "i.e." and "e.g." broken by newline (e.g., "i.\ne." → "i.e.")
    {"pattern": r"\b([ie])\.\s*\n\s*([eg])\.", "replacement": r"\1.\2."},

    # Normalize "i.e." across line breaks (e.g., "i. e." → "i.e.")
    {"pattern": r"\b(i)\.\s*e\.", "replacement": r"i.e."},

    # Normalize "e.g." across line breaks (e.g., "e. g." → "e.g.")
    {"pattern": r"\b(e)\.\s*g\.", "replacement": r"e.g."},

    # Join "i.e." and "e.g." with following text (e.g., "i.e.\nNext" → "i.e. Next")
    {"pattern": r"(i\.e\.|e\.g\.)\s*\n\s*", "replacement": r"\1 "},

    # Join lines starting with "i.e." or "e.g." to previous line (e.g., "\ni.e." → " i.e.")
    {"pattern": r"(?mi)\n\s*(i\.e\.|e\.g\.)\s*", "replacement": r" \1 "},
    
    
    # CASE NUMBER AND LEGAL REFERENCE HANDLING
    # Fix case numbers split across lines CA, SC, HC, etc. (e.g., "No.\nCA/123" → "No. CA/123")
    {"pattern": r"No\.\s*\n\s*(CA|SC|HC|MC|DC|WRIT|Appeal|Revision)", "replacement": r"No. \1"},

    # Fix "No." with lettered subclauses split across lines (e.g., "No.\n(a)" → "No. (a)")
    {"pattern": r"(?mi)\bNo\.\s*\n\s*(\([a-z]+\))", "replacement": r"No. \1"},

    # Fix "No." with roman numeral subclauses split across lines (e.g., "No.\n(i)" → "No. (i)")
    {"pattern": r"(?mi)\bNo\.\s*\n\s*(\([ivxlcdm]+\))", "replacement": r"No. \1"},

    # Join "No." with following number across newline (e.g., "No.\n123" → "No. 123")
    {"pattern": r"\b[Nn]o\.\s*\n?\s*(?=\d)", "replacement": r"No. "},

    # Handle "No." as answer after questions (e.g., "? No. (" → "? No.\n")
    {"pattern": r"([?])\s*No\.\s+(?=\()", "replacement": r"\1 No.\n"},

    # Handle "No." as answer before capital letters (e.g., "? No. Next" → "? No.\nNext")
    {"pattern": r"([?])\s*No\.\s+(?=[A-Z])", "replacement": r"\1 No.\n"},

    # Fix currency amounts split across lines (e.g., "Rs.\n12345" → "Rs. 12345")
    {"pattern": r"(?mi)\bRs\.\s*\n\s*(\d)", "replacement": r"Rs. \1"},

    # Join lowercase "no." with case IDs containing digits or slashes (e.g., "no.\n123/2023" → "No. 123/2023")
    {"pattern": r"(?mi)\bno\.\s*\n\s*([^\s]*[0-9/][^\s]*)", "replacement": r"No. \1"},

    # Handle explicit 'case no.' variants split across lines (e.g., "case no.\n123" → "case no. 123")
    {"pattern": r"(?mi)(case(?:\s+bearing)?\s+no)\.\s*\n\s*([^\s]*[0-9/][^\s]*)", "replacement": r"\1. \2"},

    # Join "Nos." with following numbers (e.g., "Nos.\n1" → "Nos. 1")
    {"pattern": r"(?mi)\bNos\.\s*\n\s*(\d+)", "replacement": r"Nos. \1"},

    # Join "No." with number/letter enumerations broken by newline (e.g., "No.\n(1)" → "No. (1)")
    {"pattern": r"(?i)\bNo\.\s*\n\s*(\(?[0-9ivxlcdmIVXLCDM]+\)?)", "replacement": r"No. \1"},

    # Join broken "No.\n<number>." and split after if followed by capital letter (e.g., "No.\n123. Next" → "No. 123.\nNext")
    {"pattern": r"(?i)No\.\s*\n?\s*(\d+)\.(\s+)(?=[A-Z])", "replacement": r"No. \1.\n"},
    
    {"pattern": r"(\(\d{4}\)\.)\s*\n\s*(\d+\.\s*[A-Z])", "replacement": r"\1 \2"},
    
    
    # TITLES AND INITIALS HANDLING
    # Join titles with following text (e.g., "Dr.\nSmith" → "Dr. Smith")
    {"pattern": r"(?m)\b(Dr|Mr|Mrs|Ms|Prof|Hon|Rev|Sr|Jr|Gen|Col|Capt|Lt|St)\.\s*\n\s*", "replacement": r"\1. "},

    # Join initials broken across newline (e.g., "J.\nSmith" → "J. Smith")
    {"pattern": r"((?:[A-Z]\.){2,})\s*\n\s*([A-Z][a-z]+)", "replacement": r"\1 \2"},

    # Normalize multiple initials (e.g., "J. A. M." → "J.A.M.")
    {"pattern": r"\b([A-Z])\.\s+([A-Z])\.\s+([A-Z])\.", "replacement": r"\1.\2.\3."},

    # Join case citations with v. (e.g., "Smith\nv." → "Smith v.")
    {"pattern": r"\b(vs?\.?|VS\.?)\s*\n+", "replacement": "v. "},

    # Remove newlines before v. in case citations (e.g., "\nv." → "v.")
    {"pattern": r"(?m)\n(?=(?:[Vv]s?\.?|VS\.?)\s)", "replacement": " "},

    # Join single initials with following text (e.g., "J.\nSmith" → "J. Smith")
    {"pattern": r"\b([A-Z])\.\s*\n+", "replacement": r"\1. "},

    # Join initials after commas (e.g., "Denning,\nL. J" → "Denning, L. J")
    {"pattern": r"([A-Z][a-z]+,)\s*\n\s*((?:[A-Z]\.){1,}\s*[A-Z]?)", "replacement": r"\1 \2"},
    
    
    # SECTION AND REFERENCE HANDLING
    # Join section references split across lines (e.g., "s.\n114" → "s. 114")
    {"pattern": r"(?mi)\bs\.\s*\n\s*([\d][\w.\-]*)", "replacement": r"s. \1"},

    # Remove newlines before section references (e.g., "\ns." → "s.")
    {"pattern": r"(?mi)\n(?=\s*(?:s\.|ss\.|sec\.)(?=\s|$|[^\w]))", "replacement": " "},

    # Join "Sec." with following text (e.g., "Sec.\n123" → "Sec. 123")
    {"pattern": r"(?m)(?<=\b(?:Sec|sec)\.)\s*\n\s*", "replacement": " "},

    # Split after section references (e.g., "Section 123. Next" → "Section 123.\nNext")
    {"pattern": r"(Section\s+\d+[A-Za-z()]*\.)\s+(?=[A-Z])", "replacement": r"\1\n"},

    # Split after "Sec." followed by capital letters (e.g., "Sec. 123. Next" → "Sec. 123.\nNext")
    {"pattern": r"(Sec\.\s*\d+)\.\s+(?=[A-Z])", "replacement": r"\1.\n"},

    # Split after "Case No." followed by capital letters (e.g., "Case No. 123. Next" → "Case No. 123.\nNext")
    {"pattern": r"(Case\s+No\.\s*[^\s]+)\.\s+(?=[A-Z])", "replacement": r"\1.\n"},

    # Join section numbers with subclauses (e.g., "Section 123.\n(1)" → "Section 123. (1)")
    {"pattern": r"(Section\s+\d+)\.\s*\n\s*(\(\d+\))", "replacement": r"\1. \2"},

    # Join "r." with following numbers (e.g., "r.\n123" → "r. 123")
    {"pattern": r"(?mi)\brr?\.\s*\n\s*([\d]+(?:\.\d+)*(?:\([^)]+\))*)", "replacement": r"r. \1"},

    # Join section references with numbers (e.g., "Section\n123" → "Section 123")
    {"pattern": r"(?mi)\b(Section|Sec|s)\.\s*\n\s*(\d+\(?\d*\)?)", "replacement": r"\1. \2"},

    # Join section references without periods (e.g., "Section\n123" → "Section 123")
    {"pattern": r"(?mi)\b(Section|Sec|s)\s*\n\s*(\d+\(?\d*\)?)", "replacement": r"\1 \2"},
    
    
    # PAGE REFERENCE HANDLING
    # Join page references split across lines (e.g., "p.\n546" → "p. 546")
    {"pattern": r"\b(p|pg)\.\s*\n?\s*(\d+)", "replacement": r"\1. \2"},

    # Fix "at p." references in parentheses (e.g., "(at\np.\n123)" → "(at p. 123)")
    {"pattern": r"(?m)\(\s*at\s*p\.\s*\n\s*(\d+)\s*\)", "replacement": r"(at p. \1)"},

    # Fix "at p." references outside parentheses (e.g., "at\np.\n123" → "at p. 123")
    {"pattern": r"(?m)at\s*\n?\s*p\.\s*(\d+)", "replacement": r"at p. \1"},

    # Join "pgs." with following numbers (e.g., "pgs.\n25-30" → "pgs. 25-30")
    {"pattern": r"(?m)pgs\.\s*\n\s*([\d,]+)", "replacement": r"pgs. \1"},

    # Fix page references in parentheses with volume info (e.g., "(\np.\n123)" → "(p. 123)")
    {"pattern": r"\(\s*\n\s*(p\.\s*\d+(?:\s+of\s+Vol\.\s*[I1-3]+(?:\s*Book\s*\d+)?)?)\s*\)", "replacement": r"(\1)"},

    # Fix general parentheses content split across lines (e.g., "(\ncontent)" → "(content)")
    {"pattern": r"\(\s*\n\s*([^)]+)\)", "replacement": r"(\1)"},

    # Join content within parentheses split across lines (e.g., "(text\nmore)" → "(text more)")
    {"pattern": r"(?m)\(\s*([^)]*?)\s*\n\s*([^)]*?)\)", "replacement": r"(\1 \2)"},

    # Fix "at p." references case insensitive (e.g., "at\np.\n123" → "at p. 123")
    {"pattern": r"(?mi)\bat\s*p\.\s*\n\s*(\d+)", "replacement": r"at p. \1"},

    # Fix "at pp." references (e.g., "(at\npp.\n25-30)" → "(at pp. 25-30)")
    {"pattern": r"(?m)\(\s*at\s*pp\.\s*\n\s*([\d,\-– ]+)\s*\)", "replacement": r"(at pp. \1)"},

    # Join volume and page references (e.g., "Vol. I,\np. 123" → "Vol. I, p. 123")
    {"pattern": r"(Vol\.\s+[IVXLCDM]+,)\s*\n\s*(p{1,2}\.\s*\d+)", "replacement": r"\1 \2"},

    # Join "at." with page references (e.g., "at.\np. 123" → "at. p. 123")
    {"pattern": r"(\bat\.)\n+(?=p\.\s*\d+)","replacement": r"\1 "},
    
    
    # CITATION AND QUOTE HANDLING
    # Ensure space after closing parentheses (e.g., ").text" → "). text")
    {"pattern": r"\)\.(?=\S)", "replacement": "). "},

    # Split after closing parentheses followed by text (e.g., "). Next" → ").\nNext")
    {"pattern": r"\)\.\s*(?=[A-Za-z])", "replacement": ").\n"},

    # Join legal citations split across lines (e.g., "(SC\n123)" → "(SC 123)")
    {"pattern": r"(\([^\n]*SC)\s*\n\s*(\d+)\)", "replacement": r"\1 \2)"},

    # Join AIR citations split across lines (e.g., "AIR 2023\nSC 123" → "AIR 2023 SC 123")
    {"pattern": r"(\bAIR\s*(?:\(?[0-9]{4}\)?)?\s*[A-Za-z.]+)\s*\n\s*([0-9]+)\)?", "replacement": r"\1 \2)"},

    # Join Roman numerals with following text (e.g., "(i).\nNext" → "(i). Next")
    {"pattern": r"(\b(?:[ivxlcdm]+)\.)\n+([A-Z])", "replacement": r"\1 \2"},

    # Join legal citations with court abbreviations (e.g., ").\nL.A." → "). L.A.")
    {"pattern": r"(\)\.?)\n+(?=(?:L\.A\.|B\.L\.R\.|S\.C\.))","replacement": r"\1 "},

    # Join case citations with year and court (e.g., "((2008)\nB.L.R." → "((2008) B.L.R.")
    {"pattern": r"(\(\(\d{4}\)\s*)\n+(?=B\.L\.R\.)", "replacement": r"\1"},

    # Join "no." with case references (e.g., "no.\nB123" → "No. B123")
    {"pattern": r"(\b[Nn]o\.)\n+([A-Z0-9])", "replacement": r"\1 \2"},

    # Join year references with page numbers (e.g., "(2023)\np. 123" → "(2023) p. 123")
    {"pattern": r"(\(\d{4}\))\n+(?=p\.\s*\d+)", "replacement": r"\1 "},

    # Join bracketed words with following text (e.g., "(word).\nNext" → "(word). Next")
    {"pattern": r"(\(\w+\))\s*\.\n+([A-Z])", "replacement": r"\1. \2"},

    # Join "et al." with following numbers (e.g., "et al.\n123" → "et al. 123")
    {"pattern": r"(et\. al\.)\n+(\d+)", "replacement": r"\1 \2"},

    # Join numbers with bracketed subclauses (e.g., "1.\n(1)" → "1. (1)")
    {"pattern": r"(\b\d+\.)\n+(\(\d+\))", "replacement": r"\1 \2"},

    # Join "Art." with following numbers (e.g., "Art.\n123" → "Art. 123")
    {"pattern": r"(\bArt\.)\n+(\d+)", "replacement": r"\1 \2"},

    # Join "Vol." with volume references (e.g., "Vol.\n123 at pg. 456" → "Vol. 123 at pg. 456")
    {"pattern": r"(Vol\.)\s*\n+(\d+\s+at\s+pg\.\s*\d+)", "replacement": r"\1 \2"},

    # Join numbers with lettered subclauses (e.g., "1.\n(a)" → "1. (a)")
    {"pattern": r"(\b\d+\.)\n+\(([a-z])\)", "replacement": r"\1 (\2)"},

    # Join Roman numerals with following text (e.g., "I.\nNext" → "I. Next")
    {"pattern": r"\b([IVXLCDM]+)\.\n+([A-Z])", "replacement": r"\1. \2"},

    # Join "CH." with following text (e.g., "CH.\nD123" → "CH.D123")
    {"pattern": r"(\bCH)\.\s*\n\s*(D\s*\d+)", "replacement": r"\1.\2"},

    # Join "SC SPL" with following text (e.g., "SC SPL.\nLA No." → "SC SPL. LA No.")
    {"pattern": r"(?mi)(SC(?:\s+SPL)?)\.\s*\n\s*(LA\s+No\.)", "replacement": r"\1. \2"},
    
    {"pattern": r"”\s+(?=\d+\.\s+[A-Z])", "replacement": r"”\n"},
    
    {"pattern": r"(CA\.)\s*\n\s*(No\.)", "replacement": r"\1 \2"},
    
    {"pattern": r"([A-Za-z])\s*\n\s*(\d+\(?\d*\)?)", "replacement": r"\1 \2"},

    
    # DATE FORMAT HANDLING
    # Split after dates followed by capital letters (e.g., "12/31/2023. Next" → "12/31/2023.\nNext")
    {"pattern": r"(\d{1,2}[./\- ]\d{1,2}[./\- ]\d{2,4})\.(\s+[A-Z])", "replacement": r"\1.\n\2"},

    # Split after month names followed by capital letters (e.g., "Jan 15, 2023. Next" → "Jan 15, 2023.\nNext")
    {"pattern": r"(\d{1,2}\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(\s+\d{2,4})?)\.(\s+[A-Z])", "replacement": r"\1.\n\4"},

    # Split after ordinal dates followed by capital letters (e.g., "15th Jan 2023. Next" → "15th Jan 2023.\nNext")
    {"pattern": r"(\d{1,2}(?:st|nd|rd|th)?\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(\s+\d{2,4})?)\.(\s+[A-Z])", "replacement": r"\1.\n\4"},

    # Join ordinal dates with years split across lines (e.g., "15th Jan.\n2023" → "15th Jan. 2023")
    {"pattern": r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:uary|ruary|ch|il|e|y|e|ober|ember)?\.)\s*\n\s*(\d{2,4})", "replacement": r"\1 \2"},

    # Join dates split across lines (e.g., "12/31\n2023" → "12/31 2023")
    {"pattern": r"(\d{1,2}[./\- ]\d{1,2}[./\- ]\d{2,4})\s*\n\s*", "replacement": r"\1 "},

    # Join date formats (e.g., "12.31.\n2023" → "12.31.2023")
    {"pattern": r"(\d{2}\.\d{2}\.)\s*\n\s*(\d{4})", "replacement": r"\1\2"},

    # Join partial dates (e.g., "1.\n12.2023" → "1.12.2023")
    {"pattern": r"(\d{1,2})\.\s*\n\s*(\d{2}\.\d{4})", "replacement": r"\1.\2"},

    # Join citation sequences (e.g., "(2023) text; (2024) text" → "(2023) text; (2024) text")
    {"pattern": r"(\(\d{4}\)\s[^\n;]+;\s*)\n\s*(\(\d{4}\)\s[^\n]+)", "replacement": r"\1 \2"},
    
    {"pattern": r"(Vs\.)\s*\n\s*(?=[A-Z])", "replacement": r"\1 "},
    
    {"pattern": r"(\d{1,2})\.\s*\n\s*(\d{1,2}\.\s*\d{4})", "replacement": r"\1. \2"},
    
    
    # QUESTION AND ANSWER HANDLING
    # Join questions with Yes/No answers with punctuation (e.g., "?\nYes, Next" → "? Yes,\nNext")
    {"pattern": r"(?i)\?\s*\n\s*(yes|no)([.,])\s+(?=[A-Z])", "replacement": r"? \1\2\n"},

    # Join questions with Yes/No answers without punctuation (e.g., "?\nYes Next" → "? Yes.\nNext")
    {"pattern": r"(?i)\?\s*\n\s*(yes|no)\s+(?=[A-Z])", "replacement": r"? \1.\n"},
    
    
    # MISCELLANEOUS HANDLING
    # Split after "Act" when followed by capital letters (e.g., "Act. Next" → "Act.\nNext")
    {"pattern": r"\b(Act)\.\s+(?=(?!of\b|or\b|on\b|and\b|to\b)[A-Z])", "replacement": r"\1.\n"},

    # Split after specific legal terms (e.g., "Code. Next" → "Code.\nNext")
    {"pattern": r"(Code|Card|Town|Sudu|notes)[.;]\s+(?=[A-Za-z0-9\(])", "replacement": r"\1.\n"},

    # Split after emphasis notations (e.g., "[Emphasis added]. Next" → "[Emphasis added].\nNext")
    {"pattern": r"(?:\n\s*)?(\[?\(?[Ee]mphasis added[\)\]]?)([.?!]?)\s+(?=[A-Z])", "replacement": r"\1\2\n"},
    {"pattern": r"\.\s*\n\s*”\s*(\[[Ee]mphasis added\])", "replacement": r".” \1"},

    # Join quoted text with following capital letters (e.g., "\"text\nNext" → "\"text Next")
    {"pattern": r'''(^\s*[""'\.\-]{1,})\s*\n\s*([A-Z])''', "replacement": r'\1 \2'},

    # Join numeric enumerations in brackets with following text (e.g., "(1).\nNext" → "(1). Next")
    {"pattern": r"(\(\d+\)\.)\s*\n\s*([A-Z])", "replacement": r"\1 \2"},
    {"pattern": r"([a-z])\n([a-z])", "replacement": r"\1 \2"},
]

# -------------------------
# Helpers
# -------------------------
def pre_join_paragraphs(text):
    """Join lines inside paragraphs (paragraphs separated by blank line)."""
    paragraphs = text.replace("\r\n", "\n").split("\n\n")
    joined = []
    for p in paragraphs:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if not lines:
            continue
        joined.append(" ".join(lines))
    return joined  # list of paragraph strings

def apply_inline_rules_to_text(text, rules):
    for r in rules:
        text = re.sub(r["pattern"], r["replacement"], text, flags=re.MULTILINE)
    return text

def merge_bracketed_letters_in_paragraph(par):
    # fix (a) to (e) style if broken
    par = re.sub(r'\(\s*([a-zA-Z])\s*\)\s+to\s+\(\s*([a-zA-Z])\s*\)', r'(\1) to (\2)', par)
    # collapse repeated bracketed letters into single-line forms
    par = re.sub(r'\(\s*([a-zA-Z])\s*\)\s*\(\s*([a-zA-Z])\s*\)', r'(\1) (\2)', par)
    return par

def split_lettered_subclauses_and_merge(paragraph):
    """
    Insert newlines before lowercase lettered markers (a. b. c.) then
    ensure markers join with their text.
    Returns list of clause-lines.
    """
    # Put temporary marker before lowercase lettered markers appearing as subclauses
    text = re.sub(r'(?<!\w)([a-z])\.\s+', r'@@\1.@@', paragraph)
    # Replace marker with newline
    text = text.replace('@@', '\n')
    # Normalize and split
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    # Merge markers that ended up alone with next line
    merged = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'^([a-z]\.)\s*(.*)$', ln)
        if m:
            marker = m.group(1)
            rest = m.group(2)
            if rest:
                merged.append(marker + " " + rest)
                i += 1
            else:
                if i + 1 < len(lines):
                    merged.append(marker + " " + lines[i+1])
                    i += 2
                else:
                    merged.append(marker)
                    i += 1
        else:
            merged.append(ln)
            i += 1
    return merged

SPLIT_RE = re.compile(
    r'(?<!\bNo\.)'                   # do NOT split right after "No."
    r'(?<=[.!?])\s+'                 # sentence end .!? then whitespace
    r'(?=(?!\('                        # don't split before (
        r'(?:[Vv]ide|[Ss]ection(?:s)?|at\s*p\.)\b)'  # include 'at p.' as exception
    r'[A-Z0-9“”"\'‘’(])'             # next chunk starts sensibly
)

def simple_perfect_split(text):
    parts = re.split(SPLIT_RE, text.strip())
    return [p.strip() for p in parts if p.strip()]

def postprocess_sentences(sentences):
    """
    Attach isolated '1.' tokens, attach initials to following names, merge (i) blocks if needed.
    """
    out = []
    i = 0
    n = len(sentences)
    while i < n:
        cur = sentences[i].strip()

        # If exact '1.' or '2.' attach to next
        if re.fullmatch(r'\d+\.', cur) and i + 1 < n:
            out.append(cur + " " + sentences[i+1].strip())
            i += 2
            continue

        # If chunk ends with '; 1.' or ': 1.' then split into left and '1. next'
        m = re.match(r'^(.*?)([;:])\s*(\d+\.)\s*$', cur)
        if m and i + 1 < n:
            left = (m.group(1) + m.group(2)).strip()
            numtok = m.group(3)
            out.append(left)
            out.append(numtok + " " + sentences[i+1].strip())
            i += 2
            continue

        # If cur ends with initials like S.A.M. and next starts with capitalised name -> merge
        if i + 1 < n and re.search(r'((?:[A-Z]\.){2,})\s*$', cur):
            nxt = sentences[i+1].strip()
            if re.match(r'^[A-Z][a-z]', nxt):
                out.append(cur + " " + nxt)
                i += 2
                continue

        # If current starts with (i) or (a) and previous exists and previous doesn't end with terminal punctuation -> attach
        if re.match(r'^\(\s*(?:[ivx]+|[A-Za-z])\s*\)', cur) and out:
            prev = out[-1]
            if not re.search(r'[.!?]$', prev):
                out[-1] = prev + " " + cur
                i += 1
                continue
            
        # default
        out.append(cur)
        i += 1

    return out

def process_directory(input_dir, output_dir):
    """
    Process all .txt files in input_dir and save results to output_dir
    using the existing pipeline.
    Only prints processed filenames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".txt"):
            continue
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        # Read the file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # split metadata and body
        if "=== BODY ===" in text:
            meta, body = text.split("=== BODY ===", 1)
            body = body.strip()
        else:
            meta = ""
            body = text.strip()

        # Run your existing pipeline (without print statements)
        paragraphs = pre_join_paragraphs(body)
        all_sentences = []
        for par in paragraphs:
            par = apply_inline_rules_to_text(par, INLINE_RULES)
            par = merge_bracketed_letters_in_paragraph(par)
            clause_lines = split_lettered_subclauses_and_merge(par)
            joined = "\n".join(clause_lines)
            joined, _ = re.subn(r'(?m)(\b\d+\.|\b[a-z]\.)\s*\n\s*([A-Z])', r'\1 \2', joined)
            clause_lines = [ln.strip() for ln in joined.split("\n") if ln.strip()]
            for cl in clause_lines:
                all_sentences.extend(simple_perfect_split(cl))

        final = postprocess_sentences(all_sentences)
        joined_all = "\n".join(final)
        joined_all, _ = re.subn(r'(?m)(\b\d+\.|\b[a-z]\.)\s*\n\s*([A-Z])', r'\1 \2', joined_all)
        final_lines = [ln.strip() for ln in joined_all.split("\n") if ln.strip()]
        joined_again = "\n".join(final_lines)
        joined_again = apply_inline_rules_to_text(joined_again, INLINE_RULES)
        final_body = "\n".join([ln.strip() for ln in joined_again.split("\n") if ln.strip()])

        # Save processed file
        with open(output_path, "w", encoding="utf-8") as f:
            if meta:
                f.write(meta.strip() + "\n=== BODY ===\n")
            f.write(final_body)

        # Print only the filename to indicate processing
        print(f"processed {fname}")

if __name__ == "__main__":
    INPUT_DIR = "Broken Lines Joined"   
    OUTPUT_DIR = "Sentence Splitted"  
    process_directory(INPUT_DIR, OUTPUT_DIR)