#BM25 retriever with vocab normalization, top-k citations, and disease aggregation.
#Depends on normalize.py

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import math
from normalize import (
    load_vocab,
    build_replacement_regex,
    normalize_text,
    tokenize,
    normalize_and_tokenize_query,
)

@dataclass

class Passage:
    paragraph_id: str
    condition: str
    species: str
    text: str
    url: str
    section: Optional[str] = None  #optional, add later
    source: Optional[str] = None
    license: Optional[str] = None


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        #structures
        self.passages: List[Passage] = []
        self.doc_len: List[int] = []
        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        self.N: int = 0
        self.avgdl: float = 0.0

    def _add_document(self, tokens: List[str]):
        """Update per-doc and global stats for a new document token list."""
        i = len(self.doc_len)
        self.doc_len.append(len(tokens))

        #term frequencies for this doc
        tfd: Dict[str, int] = {}
        for t in tokens:
            tfd[t] = tfd.get(t, 0) + 1
        self.tf.append(tfd)

        #ocument frequency
        seen = set(tokens)
        for t in seen:
            self.df[t] = self.df.get(t, 0) + 1

    def build(self, passages: List[Dict[str, Any]], syn2can: Dict[str, str], repl_pat) -> None:
        """Normalize and index the corpus. Passages format: dicts with keys id,disease,url,text,(section?)."""
        self.passages = [Passage(**p) for p in passages]
        self.doc_len.clear()
        self.tf.clear()
        self.df.clear()

        for p in self.passages:
            norm = normalize_text(p.text, syn2can, repl_pat)
            toks = tokenize(norm)
            self._add_document(toks)

        self.N = len(self.passages) if self.passages else 0
        self.avgdl = (sum(self.doc_len) / self.N) if self.N > 0 else 0.0

    # -------- Scoring --------
    def _idf(self, term: str) -> float:
        #Okapi IDF
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0) if self.N > 0 else 0.0

    def score_doc(self, q_terms: List[str], doc_id: int) -> float:
        """BM25 score of document doc_id for the (unique) query terms."""
        if self.N == 0:
            return 0.0
        tfd = self.tf[doc_id]
        dl = self.doc_len[doc_id] or 1  #no div-by-zero
        denom_norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl)) if self.avgdl > 0 else self.k1

        score = 0.0
        for t in q_terms:
            f = tfd.get(t, 0)
            if f == 0:
                continue
            idf = self._idf(t)
            num = f * (self.k1 + 1)
            denom = f + denom_norm
            score += idf * (num / denom)
        return score

class BM25Retriever:
    def __init__(self, symptoms_path: str, conditions_path: str, k1: float = 1.2, b: float = 0.75):
        # vocab + replace
        self.syn2can = load_vocab(symptoms_path, conditions_path)
        self.repl_pat = build_replacement_regex(self.syn2can)
        #index
        self.idx = BM25Index(k1=k1, b=b)

    # ---------- Public API ----------
    def index(self, passages: List[Dict[str, Any]]) -> None:
        """Build the BM25 index from raw passage dicts."""
        self.idx.build(passages, self.syn2can, self.repl_pat)

    def _tiny_boost(self, qnorm: str, p: Passage) -> float:
        """Small, transparent rule boosts. Keep tiny so BM25 dominates."""
        boost = 1.0
        #+10% if normalized query contains exact canonical disease name
        if p.condition and p.condition.lower() in qnorm:
            boost *= 1.10
        #+5% if section mentions 'sign' (when add sections later)
        if p.section and ("sign" in p.section.lower() or "symptom" in p.section.lower()):
            boost *= 1.05
        return boost

    @staticmethod
    def _snippet(text: str, qnorm: str, width: int = 200) -> str:
        """Return a short snippet centered near first query token; fall back to head."""
        #Use the first token from normalized query that is >= 3 chars
        q_tokens = [t for t in qnorm.split() if len(t) >= 3]
        if not q_tokens:
            return text[:width] + ("…" if len(text) > width else "")
        lower = text.lower()
        pos = -1
        for t in q_tokens:
            pos = lower.find(t)
            if pos != -1:
                break
        if pos == -1:
            return text[:width] + ("…" if len(text) > width else "")
        start = max(0, pos - width // 2)
        end = min(len(text), start + width)
        snip = text[start:end]
        if start > 0:
            snip = "…" + snip
        if end < len(text):
            snip = snip + "…"
        return snip

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Top-k passages with citations."""
        qnorm, q_tokens_raw = normalize_and_tokenize_query(query, self.syn2can, self.repl_pat)

        #remove  common, non-informative words from the query before scoring (it was making my stuff bug)
        STOPWORDS = {
            "the", "a", "an", "of", "and", "with", "to", "in",
            "on", "for", "is", "are", "as", "by", "at", "from",
            "that", "this", "it", "its", "dog", "dogs"
        }

        #keep unique query tokens but drop stopwords
        q_terms = [t for t in sorted(set(q_tokens_raw)) if t not in STOPWORDS]


        scores: List[float] = []
        for i, p in enumerate(self.idx.passages):
            s = self.idx.score_doc(q_terms, i)
            s *= self._tiny_boost(qnorm, p)
            scores.append(s)

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for i in order:
            p = self.idx.passages[i]
            results.append({
                "score": float(scores[i]),
                "id": p.paragraph_id,
                "disease": p.condition,
                "section": p.section,   # could be None
                "url": p.url,
                "snippet": self._snippet(p.text, qnorm)
            })
        return results

    def disease_scores(self, query: str, top_m: int = 20, penalize_skew: bool = False) -> List[Tuple[str, float]]:
        """Aggregate passage scores → disease ranking (sum of top-M per disease).
        If penalize_skew=True, divide by log(2 + #passages for disease)."""
        qnorm, q_tokens_raw = normalize_and_tokenize_query(query, self.syn2can, self.repl_pat)

        STOPWORDS = {
            "the", "a", "an", "of", "and", "with", "to", "in",
            "on", "for", "is", "are", "as", "by", "at", "from",
            "that", "this", "it", "its", "dog", "dogs"
        }
        q_terms = [t for t in sorted(set(q_tokens_raw)) if t not in STOPWORDS]


        # score all passages once
        per_passage = []
        for i, p in enumerate(self.idx.passages):
            s = self.idx.score_doc(q_terms, i) * self._tiny_boost(qnorm, p)
            per_passage.append((p.condition, s))

        # group by disease
        by_dis: Dict[str, List[float]] = {}
        for dis, s in per_passage:
            by_dis.setdefault(dis, []).append(s)

        # aggregate
        out: List[Tuple[str, float]] = []
        for dis, arr in by_dis.items():
            arr.sort(reverse=True)
            S = sum(arr[:top_m])
            if penalize_skew:
                S = S / math.log(2 + len(arr))
            out.append((dis, float(S)))

        out.sort(key=lambda x: x[1], reverse=True)
        return out
