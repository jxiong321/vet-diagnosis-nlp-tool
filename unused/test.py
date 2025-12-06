from bm25_retriever import BM25Retriever
from interpret_output import print_results
import json

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

if __name__ == "__main__":
    r = BM25Retriever(
        symptoms_path="data/symptoms.json",
        conditions_path="data/conditions.json"
    )
    passages = load_jsonl("data/passages.jsonl")
    r.index(passages)

    query = "puppy has bloody diarrhea"
    hits = r.retrieve(query, k=3)

    qnorm, _ = r.normalize_and_tokenize_query(query) if hasattr(r, "normalize_and_tokenize_query") else (query.lower(), [])
    for h in hits:
        h["_qnorm"] = qnorm

    cond_scores = r.disease_scores(query) 

    print_results(query, hits, cond_scores, top_conditions=12, hide_zero_scores=True)
