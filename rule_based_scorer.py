"""
rule_scorer.py

Minimal rule-based scorer with hard constraints.
"""

from typing import Dict


class RuleBasedScorer:
    """
    Rule-based validation system.
    Currently implements one rule: species must match disease type.
    """
    
    def score_disease(self, features: Dict, disease: str) -> float:
        """
        Validate a single disease against patient features.
        
        Args:
            features: Patient features dict (from SymptomExtractor)
            disease: Disease name (e.g., "Canine Parvovirus")
        
        Returns:
            1.0 if disease is possible, 0.0 if impossible
        """
        # Rule 1: Species must match disease type
        species = features.get('species', 'dog').lower()
        
        # Canine diseases only affect dogs
        if 'canine' in disease.lower() and species != 'dog':
            return 0.0
        
        # Feline diseases only affect cats
        if 'feline' in disease.lower() and species != 'cat':
            return 0.0
        
        # Disease is possible so return neutral pass score
        return 1.0
    
    def score_all_diseases(self, features: Dict, diseases: list) -> Dict[str, float]:
        """
        Score all candidate diseases.
        
        Args:
            features: Patient features dict
            diseases: List of disease names to evaluate
        
        Returns:
            Dict mapping disease -> score (0.0 or 1.0)
        """
        scores = {}
        for disease in diseases:
            scores[disease] = self.score_disease(features, disease)
        return scores


# Test the scorer
if __name__ == "__main__":
    from symptom_extractor import SymptomExtractor
    
    scorer = RuleBasedScorer()
    extractor = SymptomExtractor()
    
    # Test cases
    test_cases = [
        ("puppy has diarrhea", "Canine Parvovirus"),
        ("cat is sneezing", "Canine Distemper"),  # sghoud fail
        ("dog is coughing", "Feline Leukemia"),   # shoudl fail
    ]
    
    print("\n🧪 Testing Rule-Based Scorer\n")
    
    for query, disease in test_cases:
        features = extractor.extract_features(query)
        score = scorer.score_disease(features, disease)
        
        status = "✅ PASS" if score == 1.0 else "❌ BLOCKED"
        print(f"{status} | '{query}' → {disease} (score: {score})")