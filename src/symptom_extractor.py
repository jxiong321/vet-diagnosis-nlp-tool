"""
Converts natural language queries into structured features 
that match logistic regression classifier expected input format
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
            self.alias_to_canonical[canonical_name.lower()] = canonical_name

            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical_name
        
        print(f"Loaded {len(self.symptoms_vocab)} symptoms with {len(self.alias_to_canonical)} total terms")

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
            
            # Text symptom columns
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
        
        dog_words = ['dog', 'puppy', 'canine', 'pup', 'doggo']
        if any(word in text_lower for word in dog_words):
            return 'dog'
        cat_words = ['cat', 'kitten', 'feline', 'kitty']
        if any(word in text_lower for word in cat_words):
            return 'cat'

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
        
        # Years: "2 year old", "3 years"
        year_match = re.search(r'(\d+)\s*(?:year|yr)s?\s*old|(\d+)\s*(?:year|yr)s?', text_lower)
        if year_match:
            age = year_match.group(1) or year_match.group(2)
            return float(age)
        
        # nonths: "6 months old", "6 month old"
        month_match = re.search(r'(\d+)\s*(?:month|mo)s?\s*old|(\d+)\s*(?:month|mo)s?', text_lower)
        if month_match:
            months = month_match.group(1) or month_match.group(2)
            return float(months) / 12.0
        
        # Weks: "3 weeks old"
        week_match = re.search(r'(\d+)\s*weeks?\s*old|(\d+)\s*weeks?', text_lower)
        if week_match:
            weeks = week_match.group(1) or week_match.group(2)
            return float(weeks) / 52.0
        
        #now check for age descriptors
        if any(word in text_lower for word in ['puppy', 'young', 'newborn', 'baby']):
            return 0.5  # Young puppy
        
        if any(word in text_lower for word in ['senior', 'elderly', 'aged', 'geriatric']):
            return 10.0  # Senior dog
        
        # need to be careful with word 'old' 
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
        if any(word in text_lower for word in ['male', 'boy', 'him', 'his', 'he']):
            return 'Male'

        if any(word in text_lower for word in ['female', 'girl', 'her', 'she']):
            return 'Female'

        return 'Male'

    def _extract_vitals(self, text: str) -> Dict[str, float]:
        """
        Extract vital signs from text
        Examples:
            "weight 30 kg" → Weight: 30.0
            "heart rate 120 bpm" → Heart_Rate: 120.0
            "temp 39.5 C" → Body_Temperature_C: 39.5
            "duration 3 days" → duration_days: 3.0
        """
        vitals = {}

        # Weight: "30 kg", "weight 30kg", "30kg", "30 pounds"
        weight_match = re.search(r'(?:weight\s*)?(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)', text.lower())
        if weight_match:
            vitals['Weight'] = float(weight_match.group(1))

        # Heart rate: "120 bpm", "heart rate 120", "HR 120"
        hr_match = re.search(r'(?:heart\s*rate|hr)\s*(\d+(?:\.\d+)?)\s*(?:bpm)?', text.lower())
        if hr_match:
            vitals['Heart_Rate'] = float(hr_match.group(1))

        # Temperature: "39.5 C", "temp 39.5", "temperature 39.5°C"
        temp_match = re.search(r'(?:temp(?:erature)?|body\s*temp)\s*(\d+(?:\.\d+)?)\s*(?:°?c|celsius)?', text.lower())
        if temp_match:
            vitals['Body_Temperature_C'] = float(temp_match.group(1))

        # Duration: "3 days", "for 2 days", "duration 5 days"
        duration_match = re.search(r'(?:for|duration)\s*(\d+(?:\.\d+)?)\s*days?', text.lower())
        if duration_match:
            vitals['duration_days'] = float(duration_match.group(1))

        return vitals

    def _extract_symptoms(self, text: str) -> List[str]:
        """
        Extract all symptos from text using controlled vocabulary

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
        """
        mapping = {
            'vomiting': 'Vomiting',
            'diarrhea': 'Diarrhea',
            'bloody diarrhea': 'Diarrhea',
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
        features = self._initialize_features()

        # Extract demographics
        features['species'] = self._extract_species(query_text)
        features['Age'] = self._extract_age(query_text)
        features['Gender'] = self._extract_gender(query_text)

        # Extract vitals (override defaults if found in query)
        vitals = self._extract_vitals(query_text)
        features.update(vitals)

        # Extract symptoms
        symptoms_found = self._extract_symptoms(query_text)

        # Map each symptom to the classifier's feature columns
        for symptom in symptoms_found:
            feature_name = self._symptom_to_feature_name(symptom)
            if feature_name and feature_name in features:
                features[feature_name] = 1  # Mark as present

            # if fever detected, set high body temp (only if not already specified)
            if symptom == 'fever' and 'Body_Temperature_C' not in vitals:
                features['Body_Temperature_C'] = 40.5  # high fever temp

        return features
    
    def print_extraction(self, query_text: str):
        """
        Pretty print what was extracted  for debug
        """
        features = self.extract_features(query_text)

        print(f"\n{'='*60}")
        print(f"Query: '{query_text}'")
        print(f"{'='*60}")

        print("\nExtracted Demographics:")
        print(f"  Species: {features['species']}")
        print(f"  Age: {features['Age']} years")
        print(f"  Gender: {features['Gender']}")

        print("\n Extracted Vitals:")
        print(f"  Weight: {features['Weight']} kg")
        print(f"  Heart Rate: {features['Heart_Rate']} bpm")
        print(f"  Body Temperature: {features['Body_Temperature_C']} °C")
        print(f"  Duration: {features['duration_days']} days")

        print("\n Extracted Symptoms:")
        symptom_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
                       'Labored_Breathing', 'Lameness', 'Skin_Lesions',
                       'Nasal_Discharge', 'Eye_Discharge']

        found_any = False
        for symptom in symptom_cols:
            if features.get(symptom, 0) == 1:
                print(f" {symptom}")
                found_any = True

        if not found_any:
            print("(none detected)")

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
    
    print("\nTesting Symptom Extractor\n")
    for query in test_queries:
        extractor.print_extraction(query)