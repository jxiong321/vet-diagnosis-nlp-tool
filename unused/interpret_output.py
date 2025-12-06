from typing import List, Dict, Tuple
from normalize import normalize_and_tokenize_query

def format_passage(hit: Dict, qnorm: str) -> str:
    score = f"{hit['score']:.3f}"
    cond  = hit.get("condition") or hit.get("disease") or "Unknown condition"
    url   = hit.get("url", "")
    section = hit.get("section")
    section_str = f" — {section}" if section else ""
    snippet = hit.get("snippet", "").replace("\n", " ").strip()

    STOPWORDS = {
        "the","a","an","of","and","with","to","in","on","for","is","are",
        "as","by","at","from","that","this","it","its","dog","dogs"
    }
    q_tokens = [t for t in qnorm.split() if len(t) >= 3 and t not in STOPWORDS]
    matched = [t for t in q_tokens if t in snippet.lower()]
    match_str = f"Matched: {', '.join(sorted(set(matched)))}" if matched else ""

    return (
        f"{score}  [{cond}]{section_str} {url}\n"
        f"  {snippet}\n"
        f"  {match_str}\n"
    )


def print_results(
    query: str,
    hits: List[Dict],
    condition_scores: List[Tuple[str, float]],
    *,
    top_conditions: int = 10,
    hide_zero_scores: bool = True,
) -> None:
    """Pretty CLI output."""
    from normalize import load_vocab, build_replacement_regex, normalize_text

    print("\n=== Query ===")
    print(query)

    print("\n=== Top Passages ===")
    print()

    # Print each passage
    # (can't recompute qnorm here without the normalizer; simplest is to pass qnorm from caller.)

    qnorm = ""
    for h in hits:
        if "_qnorm" in h:  # caller attached normalized query
            qnorm = h["_qnorm"]
            break

    for h in hits:
        #Attach qnorm for format_passage
        print(format_passage(h, qnorm))

    print("\n=== Top Conditions ===")
    shown = 0
    for cond, s in condition_scores:
        if hide_zero_scores and s <= 0:
            continue
        print(f"{s:.3f}  {cond}")
        shown += 1
        if shown >= top_conditions:
            break
    print()
