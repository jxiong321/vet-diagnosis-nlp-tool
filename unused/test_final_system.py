"""
test_final_system.py

End-to-end test of the complete blended diagnosis system.
Tests 3 key cases and displays full explanations.
"""

import json
import pickle
from bm25_retriever import BM25Retriever
from blended_system import BlendedDiagnosisSystem
from explanation_generator import ExplanationGenerator
# Need these for unpickling the classifier
from log_regression import MulticlassTargetEncoder, SymptomMultiHot


def load_system():
    """Load all components with optimized weights."""
    print("\n📦 Loading system components...")

    # load BM25
    retriever = BM25Retriever(
        symptoms_path='data/symptoms.json',
        conditions_path='data/conditions.json'
    )

    # load passages
    passages = []
    with open('data/passages.jsonl', 'r') as f:
        for line in f:
            passages.append(json.loads(line))
    retriever.index(passages)
    print(f"✓ Indexed {len(passages)} passages")

    # load classifier
    with open('trained_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    print("✓ Loaded trained classifier")

    # load optimized weights
    try:
        with open('best_weights.json', 'r') as f:
            weights = json.load(f)
        print(f"✓ Loaded optimized weights: {weights}")
    except FileNotFoundError:
        weights = {'retrieval': 1/3, 'classifier': 1/3, 'rules': 1/3}
        print("⚠️  Using equal weights (best_weights.json not found)")

    # create system
    system = BlendedDiagnosisSystem(retriever, classifier, weights=weights)

    # load explainer
    explainer = ExplanationGenerator()

    return system, explainer


def run_test_case(system, explainer, query, expected_disease, case_num):
    """Run a single test case and display results."""
    print("\n" + "="*70)
    print(f"TEST CASE #{case_num}")
    print("="*70)
    print(f"Query: '{query}'")
    print(f"Expected: {expected_disease}")

    # get predictions
    results = system.diagnose(query, top_k=3)
    predictions = results['blended_top']

    print(f"\nTop 3 Predictions:")
    for i, disease in enumerate(predictions, 1):
        score = results['blended_scores'][disease]
        check = "✓" if disease.lower() == expected_disease.lower() else " "
        print(f"  {check} {i}. {disease} (score: {score:.3f})")

    # check if passed
    predicted_diseases_lower = [d.lower() for d in predictions]
    if expected_disease.lower() in predicted_diseases_lower:
        status = "✅ PASS"
    else:
        status = "❌ FAIL"

    print(f"\nResult: {status}")

    # show full explanation for #1 prediction
    if predictions:
        top_disease = predictions[0]
        top_score = results['blended_scores'][top_disease]

        print(f"\n📋 Full Explanation for #{1} Diagnosis ({top_disease}):")
        print("-" * 70)

        # get component scores
        r_score = results['retrieval_scores'].get(top_disease, 0)
        c_score = results['classifier_scores'].get(top_disease, 0)
        ru_score = results['rule_scores'].get(top_disease, 0)

        # generate explanation
        explanation = explainer.generate_diagnosis_explanation(
            disease=top_disease,
            query=query,
            scores={
                'retrieval': r_score,
                'classifier': c_score,
                'rules': ru_score,
                'blended': top_score
            },
            features=results['features']
        )

        print(f"\nSummary:")
        print(f"  {explanation['summary']}")

        print(f"\nComponent Scores:")
        print(f"  Retrieval:  {r_score:.3f} - {explanation['score_interpretation']['retrieval']}")
        print(f"  Classifier: {c_score:.3f} - {explanation['score_interpretation']['classifier']}")
        print(f"  Rules:      {ru_score:.3f} - {explanation['score_interpretation']['rules']}")
        print(f"  BLENDED:    {top_score:.3f}")

        print(f"\nSupporting Evidence:")
        if explanation['evidence_sentences']:
            for i, sent in enumerate(explanation['evidence_sentences'], 1):
                print(f"  {i}. {sent['text'][:150]}...")
                print(f"     Source: {sent['source']}")
        else:
            print("  (no textbook evidence found)")

        print("-" * 70)

    return status


def main():
    print("\n" + "="*70)
    print("END-TO-END SYSTEM TEST")
    print("="*70)

    # load system
    system, explainer = load_system()

    # define test cases
    test_cases = [
        {
            'query': "puppy has bloody diarrhea and vomiting",
            'expected': "Canine Parvovirus",
            'case_num': 1
        },
        {
            'query': "dog has persistent dry cough",
            'expected': "Kennel Cough",
            'case_num': 2
        },
        {
            'query': "young dog with nasal discharge and eye discharge",
            'expected': "Canine Distemper",
            'case_num': 3
        }
    ]

    # run all tests
    results = []
    for test_case in test_cases:
        status = run_test_case(
            system,
            explainer,
            test_case['query'],
            test_case['expected'],
            test_case['case_num']
        )
        results.append(status)

    # summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if "PASS" in r)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    for i, (test_case, status) in enumerate(zip(test_cases, results), 1):
        print(f"  Test {i}: {status}")

    if passed == total:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
