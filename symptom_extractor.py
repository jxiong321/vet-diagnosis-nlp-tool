"""
symptom_extractor.py

Converts natural language queries into structured features 
that match your logistic regression classifier's expected input format.
"""

import json
import re
from typing import Dict, List, Any


class SymptomExtractor:
    def __init__(self, symptoms_path="data/symptoms.json"):
        """
        Load the controlled vocabulary for symptoms
        """
        with open(symptoms_path, 'r') as f:
            self.symptoms_vocab = json.load(f)
        
        #Build reverse mapping: any alias → canonical name
        #eg: "puking" → "vomiting", "emesis" → "vomiting"
        self.alias_to_canonical = {}
        
        for canonical_name, aliases in self.symptoms_vocab.items():
            # The canonical name itself
            self.alias_to_canonical[canonical_name.lower()] = canonical_name
            
            # All the aliases
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical_name
        
        print(f"✅ Loaded {len(self.symptoms_vocab)} symptoms with {len(self.alias_to_canonical)} total terms")

    def _initialize_features(self) -> Dict[str, Any]:
        """
        Create empty feature dictionary with ALL columns your classifier expects.
        These are the EXACT column names from your CSV/training data.
        """
        return {
            # Demographics (it tries to extract, otherwise defaults)
            'species': 'dog',           # Not in classifier, but useful for rules
            'Breed': 'Mixed',           # Default breed
            'Gender': 'Male',           # Default gender
            'Age': 5.0,                 # Default adult (5 years)
            
            #Vitals (defaults: but i think this could be helpful as an input for vets in the future)
            #it would be great if i could get the dataset for this
            'Weight': 25.0,
            'Heart_Rate': 100.0,
            'Body_Temperature_C': 38.5,
            'duration_days': 2.0,
            
            # Binary symptoms (initialize at 0 and add 1 if present)
            'Appetite_Loss': 0,
            'Vomiting': 0,
            'Diarrhea': 0,
            'Coughing': 0,
            'Labored_Breathing': 0,
            'Lameness': 0,
            'Skin_Lesions': 0,
            'Nasal_Discharge': 0,
            'Eye_Discharge': 0,
            
            # Text symptom columns (it fills them one by one but starts empty)
            'Symptom_1': '',
            'Symptom_2': '',
            'Symptom_3': '',
            'Symptom_4': ''
        }
    
    def _extract_species(self, text: str) -> str:
        """
        Extract animal species from text
        
        Examples:
            "puppy has diarrhea" → "dog"
            "kitten is sneezing" → "cat"
            "my dog is vomiting" → "dog"
        """
        text_lower = text.lower()
        
        # Dog keywords
        dog_words = ['dog', 'puppy', 'canine', 'pup', 'doggo']
        if any(word in text_lower for word in dog_words):
            return 'dog'
        
        # Cat keywords
        cat_words = ['cat', 'kitten', 'feline', 'kitty']
        if any(word in text_lower for word in cat_words):
            return 'cat'
        
        # Default to dog if not specified
        return 'dog'

    def _extract_age(self, text: str) -> float:
        """
        Extract age from text, return in years
        
        Examples:
            "puppy" → 0.5 (6 months)
            "2 year old dog" → 2.0
            "senior dog" → 10.0
            "6 month old puppy" → 0.5
        """
        text_lower = text.lower()
        
        # CHECK FOR EXPLICIT NUMBERS FIRST
        
        # Years: "2 year old", "3 years"
        year_match = re.search(r'(\d+)\s*(?:year|yr)s?\s*old|(\d+)\s*(?:year|yr)s?', text_lower)
        if year_match:
            age = year_match.group(1) or year_match.group(2)
            return float(age)
        
        # Months: "6 months old", "6 month old"
        month_match = re.search(r'(\d+)\s*(?:month|mo)s?\s*old|(\d+)\s*(?:month|mo)s?', text_lower)
        if month_match:
            months = month_match.group(1) or month_match.group(2)
            return float(months) / 12.0
        
        # Weeks: "3 weeks old"
        week_match = re.search(r'(\d+)\s*weeks?\s*old|(\d+)\s*weeks?', text_lower)
        if week_match:
            weeks = week_match.group(1) or week_match.group(2)
            return float(weeks) / 52.0
        
        #now check for age descriptors
        if any(word in text_lower for word in ['puppy', 'young', 'newborn', 'baby']):
            return 0.5  # Young puppy
        
        if any(word in text_lower for word in ['senior', 'elderly', 'aged', 'geriatric']):
            return 10.0  # Senior dog
        
        # Be careful with "old" - only if it's standalone, not part of "year old"
        if re.search(r'\bold\b', text_lower) and not re.search(r'year\s*old|month\s*old|week\s*old', text_lower):
            return 10.0  # Old dog
        
        # Default: adult dog (5 years)
        return 5.0

    def _extract_gender(self, text: str) -> str:
        """
        Extract gender if mentioned
        
        Examples:
            "male dog" → "Male"
            "she is vomiting" → "Female"
        """
        text_lower = text.lower()
        
        # Male indicators
        if any(word in text_lower for word in ['male', 'boy', 'him', 'his', 'he']):
            return 'Male'
        
        # Female indicators
        if any(word in text_lower for word in ['female', 'girl', 'her', 'she']):
            return 'Female'
        
        # Default
        return 'Male'

    def _extract_symptoms(self, text: str) -> List[str]:
        """
        Extract all symptoms from text using controlled vocabulary
        
        Example:
            "puppy has bloody diarrhea and is puking"
            → finds "diarrhea" and "vomiting" (maps "puking" to "vomiting")
        """
        text_lower = text.lower()
        symptoms_found = []
        
        # Check every term in vocab list
        for term, canonical_name in self.alias_to_canonical.items():
            if term in text_lower:
                # Avoid duplicates
                if canonical_name not in symptoms_found:
                    symptoms_found.append(canonical_name)
        
        return symptoms_found
    
    def _symptom_to_feature_name(self, canonical_symptom: str) -> str:
        """
        Map symptom vocabulary names to classifier column names
        
        Your symptoms.json has: "vomiting", "diarrhea", "cough"
        Your classifier expects: "Vomiting", "Diarrhea", "Coughing"
        """
        mapping = {
            'vomiting': 'Vomiting',
            'diarrhea': 'Diarrhea',
            'bloody diarrhea': 'Diarrhea',  # Still maps to Diarrhea
            'anorexia': 'Appetite_Loss',
            'cough': 'Coughing',
            'nasal discharge': 'Nasal_Discharge',
            'ocular discharge': 'Eye_Discharge',
            'dyspnea': 'Labored_Breathing',
            'tachypnea': 'Labored_Breathing',
            'lameness': 'Lameness',
            'pruritus': 'Skin_Lesions',
            'alopecia': 'Skin_Lesions',
        }
        
        return mapping.get(canonical_symptom, None)

    def extract_features(self, query_text: str) -> Dict[str, Any]:
        """
        MAIN METHOD: Convert text query to structured features
        
        Input: "puppy has bloody diarrhea and is vomiting"
        Output: {Age: 0.5, Diarrhea: 1, Vomiting: 1, ...}
        """
        # Start with defaults
        features = self._initialize_features()
        
        # Extract demographics
        features['species'] = self._extract_species(query_text)
        features['Age'] = self._extract_age(query_text)
        features['Gender'] = self._extract_gender(query_text)
        
        # Extract symptoms
        symptoms_found = self._extract_symptoms(query_text)
        
        # Map each symptom to the classifier's feature columns
        for symptom in symptoms_found:
            feature_name = self._symptom_to_feature_name(symptom)
            if feature_name and feature_name in features:
                features[feature_name] = 1  # Mark as present
        
        return features
    
    def print_extraction(self, query_text: str):
        """
        Pretty print what was extracted  for debug
        """
        features = self.extract_features(query_text)
        
        print(f"\n{'='*60}")
        print(f"Query: '{query_text}'")
        print(f"{'='*60}")
        
        print("\n📋 Extracted Demographics:")
        print(f"  Species: {features['species']}")
        print(f"  Age: {features['Age']} years")
        print(f"  Gender: {features['Gender']}")
        
        print("\n🩺 Extracted Symptoms:")
        symptom_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
                       'Labored_Breathing', 'Lameness', 'Skin_Lesions',
                       'Nasal_Discharge', 'Eye_Discharge']
        
        found_any = False
        for symptom in symptom_cols:
            if features.get(symptom, 0) == 1:
                print(f"  ✓ {symptom}")
                found_any = True
        
        if not found_any:
            print("  (none detected)")
        
        print(f"{'='*60}\n")


# Test] extractor
if __name__ == "__main__":
    extractor = SymptomExtractor()
    
    # Test cases
    test_queries = [
        "puppy has bloody diarrhea",
        "2 year old dog is vomiting and coughing",
        "senior dog with runny nose",
        "my dog has been limping for 3 days"
    ]
    
    print("\n🧪 Testing Symptom Extractor\n")
    for query in test_queries:
        extractor.print_extraction(query)