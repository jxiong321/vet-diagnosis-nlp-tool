"""
tune_weights.py

Grid search over weight combinations to optimize blended system performance.
Splits test cases 50/50 into validation and test sets.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
import pickle
from bm25_retriever import BM25Retriever
from blended_system import BlendedDiagnosisSystem, create_sample_test_cases
# need these for unpickling the classifier
from log_regression import MulticlassTargetEncoder, SymptomMultiHot


def load_passages(passages_path: str) -> List[Dict]:
    """Load passages from JSONL file."""
    passages = []
    with open(passages_path, 'r') as f:
        for line in f:
            passages.append(json.loads(line))
    return passages


def evaluate_weights(
    system: BlendedDiagnosisSystem,
    weights: Dict[str, float],
    test_cases: List[Dict]
) -> float:
    """
    Evaluate a weight configuration on test cases.

    Returns:
        Top-3 accuracy (fraction of cases where true disease is in top 3)
    """
    system.set_weights(weights)

    correct = 0
    for test_case in test_cases:
        query = test_case['query']
        true_disease = test_case['true_disease'].lower()

        # Get predictions
        results = system.diagnose(query, top_k=3)
        predictions = [pred.lower() for pred in results['blended_top']]

        # Check if correct
        if true_disease in predictions:
            correct += 1

    accuracy = correct / len(test_cases) if test_cases else 0.0
    return accuracy


def grid_search_weights(
    system: BlendedDiagnosisSystem,
    val_cases: List[Dict],
    granularity: int = 10
) -> Tuple[Dict[str, float], float]:
    """
    Grid search over weight combinations.

    Args:
        system: BlendedDiagnosisSystem instance
        val_cases: Validation test cases
        granularity: Number of steps for each weight (default 10)

    Returns:
        (best_weights, best_accuracy)
    """
    print(f"\nStarting grid search with granularity={granularity}")
    print(f"Total combinations to test: {granularity * (granularity + 1) // 2}\n")

    best_weights = None
    best_accuracy = 0.0

    # Generate all valid weight combinations
    step = 1.0 / granularity

    combinations_tested = 0

    for i in range(granularity + 1):
        w_retrieval = i * step

        for j in range(granularity + 1 - i):
            w_classifier = j * step
            w_rules = 1.0 - w_retrieval - w_classifier

            # ensure weights sum to 1.0
            if abs(w_retrieval + w_classifier + w_rules - 1.0) > 0.001:
                continue

            weights = {
                'retrieval': w_retrieval,
                'classifier': w_classifier,
                'rules': w_rules
            }

            # Evaluate on validation set
            accuracy = evaluate_weights(system, weights, val_cases)

            combinations_tested += 1

            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights
                print(f"✓ New best! Accuracy: {accuracy:.3f} | "
                      f"Weights: R={w_retrieval:.2f}, C={w_classifier:.2f}, Ru={w_rules:.2f}")

    print(f"\nTested {combinations_tested} weight combinations")
    return best_weights, best_accuracy


def main():
    print("\n" + "="*70)
    print("WEIGHT OPTIMIZATION FOR BLENDED DIAGNOSIS SYSTEM")
    print("="*70)

    # Load BM25 retriever
    retriever = BM25Retriever(
        symptoms_path='data/symptoms.json',
        conditions_path='data/conditions.json'
    )
    passages = load_passages('data/passages.jsonl')
    retriever.index(passages)
    print(f"✓ Indexed {len(passages)} passages")

    # Load classifier
    with open('trained_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    print("✓ Loaded trained classifier")

    # create blended system with equal weights (baseline)
    baseline_weights = {
        'retrieval': 1/3,
        'classifier': 1/3,
        'rules': 1/3
    }
    system = BlendedDiagnosisSystem(retriever, classifier, weights=baseline_weights)

    # Get test cases
    all_test_cases = create_sample_test_cases()
    print(f"✓ Loaded {len(all_test_cases)} test cases")

    # Split 50/50 into validation and test
    np.random.seed(42)
    indices = np.random.permutation(len(all_test_cases))
    split_point = len(all_test_cases) // 2

    val_indices = indices[:split_point]
    test_indices = indices[split_point:]

    val_cases = [all_test_cases[i] for i in val_indices]
    test_cases = [all_test_cases[i] for i in test_indices]

    print(f"✓ Split: {len(val_cases)} validation, {len(test_cases)} test")

    # Evaluate baseline on validation set
    print("\n" + "="*70)
    print("BASELINE PERFORMANCE (Equal Weights: 1/3 each)")
    print("="*70)
    baseline_val_acc = evaluate_weights(system, baseline_weights, val_cases)
    print(f"\nValidation Accuracy: {baseline_val_acc:.3f} ({int(baseline_val_acc * len(val_cases))}/{len(val_cases)})")

    # Grid search for optimal weights
    print("\n" + "="*70)
    print("GRID SEARCH")
    print("="*70)
    best_weights, best_val_acc = grid_search_weights(system, val_cases, granularity=10)

    # Report results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nBest weights found:")
    print(f"  Retrieval:  {best_weights['retrieval']:.3f}")
    print(f"  Classifier: {best_weights['classifier']:.3f}")
    print(f"  Rules:      {best_weights['rules']:.3f}")
    print(f"\nValidation accuracy: {best_val_acc:.3f}")
    print(f"Baseline accuracy:   {baseline_val_acc:.3f}")
    print(f"Improvement:         {best_val_acc - baseline_val_acc:+.3f} ({(best_val_acc - baseline_val_acc) / baseline_val_acc * 100:+.1f}%)")

    # Evaluate on held-out test set
    print("\n" + "="*70)
    print("HELD-OUT TEST SET EVALUATION")
    print("="*70)

    baseline_test_acc = evaluate_weights(system, baseline_weights, test_cases)
    print(f"\nBaseline test accuracy: {baseline_test_acc:.3f}")

    optimized_test_acc = evaluate_weights(system, best_weights, test_cases)
    print(f"Optimized test accuracy: {optimized_test_acc:.3f}")
    print(f"Improvement: {optimized_test_acc - baseline_test_acc:+.3f}")

    # Save best weights
    with open('best_weights.json', 'w') as f:
        json.dump(best_weights, f, indent=2)
    print(f"\nBest weights saved to best_weights.json")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
