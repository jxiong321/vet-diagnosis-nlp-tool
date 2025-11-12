"""
run_evaluation.py

Complete integration script that runs Week 7 evaluation.
Combines retrieval + classifier + rules and generates comparison table.
"""

import json
import pickle
from bm25_retriever import BM25Retriever
from blended_system import BlendedDiagnosisSystem, SystemEvaluator, create_sample_test_cases
from log_regression import MulticlassTargetEncoder, SymptomMultiHot

with open('trained_classifier.pkl', 'rb') as f:
    pipe = pickle.load(f)  #So pickle can find the classes

def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    print("\n" + "="*70)
    print("🐕 VETERINARY DIAGNOSIS SYSTEM - WEEK 7 EVALUATION")
    print("="*70 + "\n")
    
    # ============ STEP 1: Load Retrieval System ============
    print("📚 Loading retrieval system...")
    retriever = BM25Retriever(
        symptoms_path="data/symptoms.json",
        conditions_path="data/conditions.json"
    )
    
    passages = load_jsonl("data/passages.jsonl")
    retriever.index(passages)
    print(f"✅ Indexed {len(passages)} passages\n")
    
    # ============ STEP 2: Load Trained Classifier ============
    print("🤖 Loading trained classifier...")
    with open('trained_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    print("✅ Loaded classifier from trained_classifier.pkl\n")
    
    # ============ STEP 3: Create Blended System ============
    print("🔧 Creating blended diagnosis system...")
    system = BlendedDiagnosisSystem(
        retriever=retriever,
        classifier=classifier,
        weights={
            'retrieval': 1/3,   #equal weights fr now - change in future
            'classifier': 1/3,
            'rules': 1/3
        }
    )
    print()
    
    # ============ STEP 4: Test Single Query ============
    print("="*70)
    print("🧪 TESTING SINGLE QUERY")
    print("="*70 + "\n")
    
    test_query = "puppy has bloody diarrhea and is vomiting"
    results = system.diagnose(test_query, top_k=3)
    system.print_results(results)
    
    # ============ STEP 5: Run Full Evaluation ============
    print("="*70)
    print("📊 RUNNING FULL EVALUATION")
    print("="*70 + "\n")
    
    # Load test cases
    print("Using sample test cases")
    test_cases = create_sample_test_cases()
    
    print(f"Loaded {len(test_cases)} test cases\n")
    
    # Run evaluation
    evaluator = SystemEvaluator(system)
    eval_results = evaluator.evaluate(test_cases)
    
    # Print comparison table
    evaluator.print_comparison_table(eval_results)
    
    # ============ STEP 6: Test Different Weight Combinations ============
    print("\n" + "="*70)
    print("🔬 TESTING DIFFERENT WEIGHT COMBINATIONS")
    print("="*70 + "\n")
    
    weight_configs = [
        {'retrieval': 0.5, 'classifier': 0.3, 'rules': 0.2},
        {'retrieval': 0.4, 'classifier': 0.4, 'rules': 0.2},
        {'retrieval': 0.3, 'classifier': 0.5, 'rules': 0.2},
    ]
    
    print("Testing alternative weight configurations:\n")
    
    for i, weights in enumerate(weight_configs, 1):
        print(f"Config {i}: Retrieval={weights['retrieval']:.1f}, "
              f"Classifier={weights['classifier']:.1f}, "
              f"Rules={weights['rules']:.1f}")
        
        #Create system with new weights
        system_alt = BlendedDiagnosisSystem(retriever, classifier, weights=weights)
        evaluator_alt = SystemEvaluator(system_alt)
        
        # Evaluate
        results_alt = evaluator_alt.evaluate(test_cases)
        blended_acc = results_alt['Blended (Equal Weights)']['top3_accuracy']
        
        print(f"   → Top-3 Accuracy: {blended_acc:.3f}\n")
    
    print("="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()