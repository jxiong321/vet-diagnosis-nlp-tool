"""
Functions:
- load_vocab(): load two JSON vocab files, build synonym->canonical map
- build_replacement_regex(): compile longest-first, whole-word, case-insensitive pattern
- normalize_text(): lowercase -> hyphen->space -> phrase replace -> strip punctuation -> collapse whitespace
- tokenize(): regex tokenization used everywhere (docs + queries)
- normalize_and_tokenize_query(): convenience wrapper for queries
"""
import json
import os
import re
import string
from typing import Dict, Tuple, List

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

def load_vocab(symptoms_path:str, conditions_path:str) -> dict:
    #read the two json files
    with open(conditions_path, 'r') as file:
        conditions = json.load(file)
    with open(symptoms_path, 'r') as file:
        symptoms = json.load(file)

    synonym_to_canonical = {}

    def add_mapping(canonical, synonyms, source=""):
        canonical_l = canonical.strip().lower()
        #include canonical
        synonym_to_canonical[canonical_l] = canonical_l
        for s in synonyms:
            s_l = s.strip().lower()
            #prefer conditions if duplicates appear
            if s_l in synonym_to_canonical and source == "symptom":
                continue
            synonym_to_canonical[s_l] = canonical_l

    #do conditions first (should override symptoms)
    for canonical, syns in conditions.items():
        add_mapping(canonical, syns, source="condition")

    # then symptoms
    for canonical, syns in symptoms.items():
        add_mapping(canonical, syns, source="symptom")

    return synonym_to_canonical

#tests to make sure it works:
#syn2can = load_vocab("data/symptoms.json", "data/conditions.json")
#print(syn2can["parvo"])       #canine parvovirus
#syn2can["throwing up"]        #vomiting
#syn2can["bloody diarrhea"]    #bloody diarrhea
#syn2can["cushing's"]          #hyperadrenocorticism
#print(syn2can["lepto"])              #leptospirosis

def build_replacement_regex(synonym_to_canonical: Dict[str, str]) -> re.Pattern:
    """
    Build a compiled regex that matches any synonym/canonical as a whole word,
    with longest-first priority and case-insensitive matching.
    """
    # gather terms (keys) and sort longest-first to prefer multi-word phrases
    terms = sorted(set(synonym_to_canonical.keys()), key=len, reverse=True)
    #escape 4 regex
    escaped = [re.escape(t) for t in terms if t]  # skip empty strings just in case
    if not escaped:
        #pattern that matches nothing
        return re.compile(r"$a")
    pattern = r"\b(?:%s)\b" % "|".join(escaped)
    return re.compile(pattern, flags=re.IGNORECASE)


def normalize_text(
    text: str,
    synonym_to_canonical: Dict[str, str],
    replacement_pattern: re.Pattern,
    *,
    keep_apostrophes: bool = True
) -> str:
    """
    Lowercase -> replace hyphens with spaces -> phrase-level synonym replacement ->
    strip punctuation (optionally keep apostrophes) -> collapse whitespace.
    Returns a space-delimited normalized string.
    """
    if not text:
        return ""

    s = text.lower()

    #make hyphen-joined words match space-joined synonyms
    s = s.replace("-", " ")

    #replace synonyms via callback
    def _repl(m: re.Match) -> str:
        span = m.group(0).lower()
        return synonym_to_canonical.get(span, span)

    s = replacement_pattern.sub(_repl, s)

    #strip punctuation consistently
    if keep_apostrophes:
        punct_to_remove = string.punctuation.replace("'", "")
    else:
        punct_to_remove = string.punctuation
    s = s.translate(str.maketrans("", "", punct_to_remove))

    #collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(text: str) -> List[str]:
    """Tokenize with the same regex you will use throughout (docs + queries)."""
    return _WORD_RE.findall(text)


def normalize_and_tokenize_query(
    query: str,
    synonym_to_canonical: Dict[str, str],
    replacement_pattern: re.Pattern
) -> Tuple[str, List[str]]:
    """Convenience wrapper for query-time processing."""
    qnorm = normalize_text(query, synonym_to_canonical, replacement_pattern)
    q_tokens = tokenize(qnorm)
    return qnorm, q_tokens