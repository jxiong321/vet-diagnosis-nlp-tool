"""
rule_scorer.py

Rule-based scorer with hard constraints and soft symptom overlap scoring.
"""

from typing import Dict, Set


class RuleBasedScorer:
    """
    Rule-based validation system with:
    1. hard constraint: Species must match disease type (returns 0.0 if mismatch)
    2. soft scoring: Symptom overlap calculation (0.5 + overlap_ratio * 0.5)
    """

    def __init__(self):
        # hardcode disease-symptom mappings
        self.disease_symptoms = {
            'Canine Parvovirus': {'Vomiting', 'Diarrhea', 'Appetite_Loss'},
            'Kennel Cough': {'Coughing', 'Labored_Breathing'},
            'Canine Distemper': {'Nasal_Discharge', 'Eye_Discharge', 'Coughing'},
            'Gastroenteritis': {'Vomiting', 'Diarrhea', 'Appetite_Loss'},
            'Canine Leptospirosis': {'Vomiting', 'Appetite_Loss'},
            'Hemorrhagic Gastroenteritis': {'Vomiting', 'Diarrhea'},
            'Diabetes in Dogs': {'Appetite_Loss'},
            'Canine Pancreatitis': {'Vomiting', 'Appetite_Loss'},
            'Tracheal Collapse': {'Coughing', 'Labored_Breathing'},
            'Leptospirosis': {'Vomiting', 'Appetite_Loss'},
            'Threadworm Infections of Dogs': {'Diarrhea'},
            'Panosteitis': {'Lameness'},
            'Babesiosis of Dogs': {'Appetite_Loss'},
            'Progressive Retinal Atrophy': set(), 
            'Pseudorabies': set(),
            'Hypersensitivity': {'Skin_Lesions'},
            'Lymphoma in Animals': {'Appetite_Loss'},
        }

    def score_disease(self, features: Dict, disease: str) -> float:
        """
        Validate a single disease against patient features.

        Args:
            features: Patient features dict (from SymptomExtractor)
            disease: Disease name (e.g., "Canine Parvovirus")

        Returns:
            0.0 if species mismatch (HARD constraint) or no symptom overlap
            overlap_ratio (0.0-1.0) based on symptom matching (SOFT scoring)
        """
        # Species must match disease type
        species = features.get('species', 'dog').lower()

        # Canine diseases only affect dogs
        if 'canine' in disease.lower() and species != 'dog':
            return 0.0

        if 'feline' in disease.lower() and species != 'cat':
            return 0.0

        # Calculate symptom overlap using F1-score
        expected_symptoms = self.disease_symptoms.get(disease.title(), set())
        if not expected_symptoms:
            # Unknown disease return neutral score
            return 0.5

        # Extract present symptoms from features
        present_symptoms = self._get_present_symptoms(features)

        if not present_symptoms:
            # No symptoms detected can't match
            return 0.0

        # Calculate overlap
        overlap_count = len(expected_symptoms & present_symptoms)

        if overlap_count == 0:
            return 0.0

        # Precision: what fraction of patient symptoms match this disease?
        precision = overlap_count / len(present_symptoms)

        # Recall: what fraction of disease symptoms are present in patient?
        recall = overlap_count / len(expected_symptoms)

        # F1-score:penalizes diseases that either have symptoms the patient doesn't have (low precision)
        # or don't explain all the patient's symptoms (low recall)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score

    def _get_present_symptoms(self, features: Dict) -> Set[str]:
        """
        Extract which symptoms are present from feature dict.
        """
        symptom_cols = [
            'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
            'Labored_Breathing', 'Lameness', 'Skin_Lesions',
            'Nasal_Discharge', 'Eye_Discharge'
        ]

        present = set()
        for symptom in symptom_cols:
            if features.get(symptom, 0) == 1:
                present.add(symptom)

        return present
    
    def score_all_diseases(self, features: Dict, diseases: list) -> Dict[str, float]:
        """
        Score all candidate diseases.
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
    
    print("\nTesting Rule-Based Scorer\n")
    
    for query, disease in test_cases:
        features = extractor.extract_features(query)
        score = scorer.score_disease(features, disease)
        
        status = "PASS" if score == 1.0 else "BLOCKED"
        print(f"{status} | '{query}' → {disease} (score: {score})")