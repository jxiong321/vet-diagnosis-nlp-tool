"""
blended_system.py

Combines retrieval + classifier + rules into a unified diagnosis system.
Evaluates all components individually and blended to compare results
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from symptom_extractor import SymptomExtractor
from rule_based_scorer import RuleBasedScorer


class BlendedDiagnosisSystem:
    """
    Integrates BM25 retrieval, ML classifier, and rule-based validation
    """
    
    def __init__(self, retriever, classifier, weights=None):
        """
        Args:
            retriever: bm25_retriever.py
            classifier: log_regression.py
            weights: Dict with keys 'retrieval', 'classifier', 'rules' (currently at default equal weight)
        """
        self.retriever = retriever
        self.classifier = classifier
        self.symptom_extractor = SymptomExtractor()
        self.rule_scorer = RuleBasedScorer()
        
        # Default: equal weights (adjust in future)
        self.weights = weights or {
            'retrieval': 1/3,
            'classifier': 1/3,
            'rules': 1/3
        }
        
        print(f"✅ Blended system initialized with weights: {self.weights}")
    
    def diagnose(self, query: str, top_k: int = 3) -> Dict:
        """
        Run all 3 systems and blend their scores.
        """
        # STEP 1: Get retrieval scores (returns list of tuples)
        retrieval_result = self.retriever.disease_scores(query)
        
        # convert list of tuples to dict
        if isinstance(retrieval_result, list):
            retrieval_scores = {disease: score for disease, score in retrieval_result}
        elif isinstance(retrieval_result, dict):
            retrieval_scores = retrieval_result
        else:
            retrieval_scores = {}
        
        retrieval_scores = self._normalize_scores(retrieval_scores)
        
        # STEP 2: Extract structured features from query
        features = self.symptom_extractor.extract_features(query)
        
        #STEP 3: Get classifier predictions
        classifier_scores = self._get_classifier_scores(features)
        
        # STEP 4: Get rule valida=tion scores
        all_diseases = set(retrieval_scores.keys()) | set(classifier_scores.keys())
        rule_scores = self.rule_scorer.score_all_diseases(features, list(all_diseases))
        
        # STEP 5: Blend all three
        blended_scores = self._blend_scores(retrieval_scores, classifier_scores, rule_scores)
        
        # STEP 6: Get top-k for each system
        results = {
            'query': query,
            'retrieval_top': self._get_top_k(retrieval_scores, top_k),
            'classifier_top': self._get_top_k(classifier_scores, top_k),
            'rules_top': self._get_top_k(rule_scores, top_k),
            'blended_top': self._get_top_k(blended_scores, top_k),
            'retrieval_scores': retrieval_scores,
            'classifier_scores': classifier_scores,
            'rule_scores': rule_scores,
            'blended_scores': blended_scores,
            'features': features
        }
        
        return results
    
    def _get_classifier_scores(self, features: Dict) -> Dict[str, float]:
        """
        Get disease probabilities from classifier.
        """
        # Convert features dict to DataFrame
        X = pd.DataFrame([features])
        
        # Get predictions
        try:
            probs = self.classifier.predict_proba(X)[0]
            classes = self.classifier.classes_
            return dict(zip(classes, probs))
        except Exception as e:
            print(f"⚠️  Classifier error: {e}")
            return {}
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to 0-1 range using min-max scaling."""
        if not scores:
            return {}
        
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 0.5 for k in scores}
        
        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }
    
    def _blend_scores(self, retrieval: Dict, classifier: Dict, rules: Dict) -> Dict[str, float]:
        """
        Weighted average of all 3 systems.
        """
        all_diseases = set(retrieval.keys()) | set(classifier.keys()) | set(rules.keys())
        
        blended = {}
        for disease in all_diseases:
            r_score = retrieval.get(disease, 0)
            c_score = classifier.get(disease, 0)
            ru_score = rules.get(disease, 0)
            
            #If the rules block it (score=0), the entire blended score becomes 0
            if ru_score == 0:
                blended[disease] = 0.0
            else:
                # Weighted average
                blended[disease] = (
                    self.weights['retrieval'] * r_score +
                    self.weights['classifier'] * c_score +
                    self.weights['rules'] * ru_score
                )
        
        return blended
    
    def _get_top_k(self, scores: Dict[str, float], k: int) -> List[str]:
        """Get top-k diseases from score dict."""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [disease for disease, score in sorted_items[:k]]
    
    def print_results(self, results: Dict):
        """Pretty print diagnosis results."""
        print(f"\n{'='*70}")
        print(f"DIAGNOSIS: '{results['query']}'")
        print(f"{'='*70}\n")
        
        print("📊 Top-3 Predictions by System:\n")
        
        systems = [
            ('Retrieval Only', results['retrieval_top']),
            ('Classifier Only', results['classifier_top']),
            ('Rules Only', results['rules_top']),
            ('Blended', results['blended_top'])
        ]
        
        for system_name, top_diseases in systems:
            print(f"{system_name:20s} → {', '.join(top_diseases)}")
        
        print(f"\n{'='*70}\n")


class SystemEvaluator:
    """
    Evaluates all systems on test cases and generates comparison table.
    """
    
    def __init__(self, blended_system: BlendedDiagnosisSystem):
        self.system = blended_system
    
    def evaluate(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate all systems on test cases.
        
        test_cases format:
        [
            {'query': "puppy has bloody diarrhea", 'true_disease': "Canine Parvovirus"},
            {'query': "dog coughing", 'true_disease': "Kennel Cough"},
            ...
        ]
        
        Returns:
            Dict with accuracy metrics for each system
        """
        print(f"\n{'='*70}")
        print(f"Evaluating on {len(test_cases)} test cases...")
        print(f"{'='*70}\n")
        
        # Tracks correct predictions for each system
        retrieval_correct = []
        classifier_correct = []
        rules_correct = []
        blended_correct = []
        
        for i, test_case in enumerate(test_cases):
            query = test_case['query']
            true_disease = test_case['true_disease']
            
            # Get predictiodns from all systems
            results = self.system.diagnose(query, top_k=3)
            
            # Check if true disease are in top-3 for each system (edit: case insensitive)
            true_disease_lower = true_disease.lower()

            retrieval_correct.append(
                any(pred.lower() == true_disease_lower for pred in results['retrieval_top'])
            )
            classifier_correct.append(
                any(pred.lower() == true_disease_lower for pred in results['classifier_top'])
            )
            rules_correct.append(
                any(pred.lower() == true_disease_lower for pred in results['rules_top'])
            )
            blended_correct.append(
                any(pred.lower() == true_disease_lower for pred in results['blended_top'])
            )
            
            # # Print progress
            # if (i + 1) % 10 == 0:
            #     print(f"✓ Processed {i+1}/{len(test_cases)} cases")
        
        # Calculate accuracies
        results = {
            'Retrieval Only': {
                'top3_accuracy': np.mean(retrieval_correct),
                'correct_count': sum(retrieval_correct),
                'total': len(test_cases)
            },
            'Classifier Only': {
                'top3_accuracy': np.mean(classifier_correct),
                'correct_count': sum(classifier_correct),
                'total': len(test_cases)
            },
            'Rules Only': {
                'top3_accuracy': np.mean(rules_correct),
                'correct_count': sum(rules_correct),
                'total': len(test_cases)
            },
            'Blended (Equal Weights)': {
                'top3_accuracy': np.mean(blended_correct),
                'correct_count': sum(blended_correct),
                'total': len(test_cases)
            }
        }
        
        return results
    
    def print_comparison_table(self, results: Dict):
        """Print formatted comparison table for your report."""
        print(f"\n{'='*70}")
        print("📊 SYSTEM COMPARISON TABLE (Week 7 Results)")
        print(f"{'='*70}\n")
        
        print(f"{'System':<30} {'Top-3 Accuracy':>15} {'Correct/Total':>15}")
        print("-" * 70)
        
        baseline_acc = results['Retrieval Only']['top3_accuracy']
        
        for system_name, metrics in results.items():
            acc = metrics['top3_accuracy']
            correct = metrics['correct_count']
            total = metrics['total']
            
            # Get improvement over baseline
            if system_name == 'Retrieval Only':
                improvement = ""
            else:
                diff = acc - baseline_acc
                improvement = f"({diff:+.3f})"
            
            print(f"{system_name:<30} {acc:>15.3f} {correct:>6}/{total:<6} {improvement}")
        
        print("=" * 70)
        
        # Determine winner
        winner = max(results.items(), key=lambda x: x[1]['top3_accuracy'])
        print(f"\n🏆 Best System: {winner[0]}")
        print(f"   Top-3 Accuracy: {winner[1]['top3_accuracy']:.3f}")

        print()


def create_sample_test_cases() -> List[Dict]:
    """
    Manually created test cases for evaluation.
    Based on the 5 diseases in the training data <- can add more in future
    """
    return [
        # Canine Parvovirus - bloody diarrhea, vomiting, dehydration, puppies
        {'query': "puppy has bloody diarrhea and vomiting", 'true_disease': "Canine Parvovirus"},
        {'query': "young dog with severe diarrhea and dehydration", 'true_disease': "Canine Parvovirus"},
        {'query': "6 month old puppy vomiting and not eating", 'true_disease': "Canine Parvovirus"},
        {'query': "puppy with bloody stool", 'true_disease': "Canine Parvovirus"},
        
        # Kennel Cough - dry cough, honking cough, respiratory
        {'query': "dog has dry cough", 'true_disease': "Kennel Cough"},
        {'query': "honking cough in dog", 'true_disease': "Kennel Cough"},
        {'query': "dog coughing for 3 days", 'true_disease': "Kennel Cough"},
        {'query': "dog with persistent cough and labored breathing", 'true_disease': "Kennel Cough"},
        
        # Canine Distemper - nasal discharge, eye discharge, fever
        {'query': "puppy with nasal discharge and eye discharge", 'true_disease': "Canine Distemper"},
        {'query': "young dog has runny nose and eyes", 'true_disease': "Canine Distemper"},
        {'query': "dog with goopy eyes and nasal discharge", 'true_disease': "Canine Distemper"},
        {'query': "puppy has fever and discharge from nose", 'true_disease': "Canine Distemper"},
        
        # Gastroenteritis - vomiting, upset stomach, diarrhea
        {'query': "dog vomiting after eating", 'true_disease': "Gastroenteritis"},
        {'query': "adult dog with upset stomach and diarrhea", 'true_disease': "Gastroenteritis"},
        {'query': "dog has been vomiting for 2 days", 'true_disease': "Gastroenteritis"},
        {'query': "dog with stomach problems and loss of appetite", 'true_disease': "Gastroenteritis"},
        
        # Canine Leptospirosis - vomiting, loss of appetite, lethargy
        {'query': "dog not eating and vomiting", 'true_disease': "Canine Leptospirosis"},
        {'query': "dog lethargic with loss of appetite", 'true_disease': "Canine Leptospirosis"},
        {'query': "dog vomiting and very tired", 'true_disease': "Canine Leptospirosis"},
        {'query': "adult dog not eating and has fever", 'true_disease': "Canine Leptospirosis"},
    ]

# # Example usage
# if __name__ == "__main__":
#     print("\n" + "="*70)
#     print("BLENDED DIAGNOSIS SYSTEM - Example Usage")
#     print("="*70)
    
#     print("\nTo use this system, you need:")
#     print("1. Your BM25 retriever (already working)")
#     print("2. Your trained classifier (from log_regression.py)")
#     print("3. Test cases")
    
#     print("\nExample code:")
#     print("""
# from bm25_retriever import BM25Retriever
# from blended_system import BlendedDiagnosisSystem, SystemEvaluator
# import pickle

# # Load your components
# retriever = BM25Retriever(...)
# retriever.index(passages)

# with open('trained_classifier.pkl', 'rb') as f:
#     classifier = pickle.load(f)

# # Create blended system
# system = BlendedDiagnosisSystem(retriever, classifier)

# # Single diagnosis
# results = system.diagnose("puppy has bloody diarrhea", top_k=3)
# system.print_results(results)

# # Full evaluation
# test_cases = [...]  # Your test data
# evaluator = SystemEvaluator(system)
# results = evaluator.evaluate(test_cases)
# evaluator.print_comparison_table(results)
#     """)