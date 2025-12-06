# generates human-readable explanations for diagnosis predictions
# just templated text and evidence extraction

import json
from typing import Dict, List, Tuple


class ExplanationGenerator:
    # creates explanations using templated text + evidence sentences from passages

    def __init__(self, passages_path='data/passages.jsonl'):
        self.passages = []
        with open(passages_path, 'r') as f:
            for line in f:
                p = json.loads(line)
                self.passages.append(p)

        print(f"Loaded {len(self.passages)} passages for evidence extraction")

    def _score_sentence(self, sentence: str, query: str) -> int:
        """
        Score a sentence by simple word overlap with query.
        Higher = more relevant.
        """
        import re

        # normalize both and remove numbers to avoid false matches
        query_clean = re.sub(r'\d+', '', query.lower())
        sentence_clean = re.sub(r'\d+', '', sentence.lower())

        query_words = set(query_clean.split())
        sentence_words = set(sentence_clean.split())

        # count overlap
        overlap = len(query_words & sentence_words)
        return overlap

    def _extract_top_sentences(self, disease: str, query: str, top_n: int = 2) -> List[Dict]:
        # find passages about this disease
        disease_passages = [p for p in self.passages if p['condition'].lower() == disease.lower()]

        if not disease_passages:
            return []

        # split into sentences and score them
        sentence_scores = []
        for passage in disease_passages:
            text = passage['text']
            # simple sentence split (good enough)
            sentences = text.replace('!', '.').replace('?', '.').split('.')

            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:  # skip short fragments
                    continue

                score = self._score_sentence(sent, query)
                if score > 0:  # only keep sentences with some relevance
                    sentence_scores.append({
                        'text': sent,
                        'source': passage.get('url', 'Unknown'),
                        'score': score
                    })

        # sort by score and take top N
        sentence_scores.sort(key=lambda x: x['score'], reverse=True)
        return sentence_scores[:top_n]

    def _interpret_retrieval_score(self, score: float) -> str:
        """Convert retrieval score to human readable text."""
        if score >= 0.7:
            return "High textbook evidence - strong keyword match with veterinary literature"
        elif score >= 0.4:
            return "Moderate textbook evidence - some keyword overlap with documented cases"
        else:
            return "Limited textbook evidence - weak match with veterinary literature"

    def _interpret_classifier_score(self, score: float) -> str:
        """Convert classifier probability to human readable text."""
        if score >= 0.6:
            return "High statistical likelihood - strong pattern match from trained model"
        elif score >= 0.3:
            return "Moderate statistical likelihood - partial pattern match"
        else:
            return "Low statistical likelihood - weak pattern match"

    def _interpret_rules_score(self, score: float) -> str:
        """Convert rules score to human readable text."""
        if score == 0.0:
            return "Species mismatch - BLOCKED (wrong animal type for this disease)"
        elif score >= 0.9:
            return "Strong symptom match - excellent clinical alignment (F1 > 0.9)"
        elif score >= 0.7:
            return "Good symptom match - strong clinical alignment (F1 > 0.7)"
        elif score > 0.0:
            return "Partial symptom match - some clinical alignment"
        else:
            return "No symptom overlap - symptoms don't match disease profile"

    def _get_confidence_level(self, blended_score: float) -> Tuple[str, float]:
        """
        Map blended score to confidence level and percentage.

        Returns:
            (level_str, percentage)
        """
        percentage = blended_score * 100

        if blended_score >= 0.7:
            return ("high", percentage)
        elif blended_score >= 0.4:
            return ("moderate", percentage)
        else:
            return ("low", percentage)

    def _explain_symptoms(self, features: Dict, disease: str) -> str:
        """
        Explain which symptoms match this disease.
        """
        # get present symptoms
        symptom_cols = [
            'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
            'Labored_Breathing', 'Lameness', 'Skin_Lesions',
            'Nasal_Discharge', 'Eye_Discharge'
        ]

        present_symptoms = []
        for symptom in symptom_cols:
            if features.get(symptom, 0) == 1:
                # make it readable (remove underscores)
                readable = symptom.replace('_', ' ').lower()
                present_symptoms.append(readable)

        if not present_symptoms:
            return "No specific symptoms detected in query."

        # format nicely
        if len(present_symptoms) == 1:
            return f"Patient presents with {present_symptoms[0]}."
        elif len(present_symptoms) == 2:
            return f"Patient presents with {present_symptoms[0]} and {present_symptoms[1]}."
        else:
            all_but_last = ', '.join(present_symptoms[:-1])
            return f"Patient presents with {all_but_last}, and {present_symptoms[-1]}."

    def generate_diagnosis_explanation(
        self,
        disease: str,
        query: str,
        scores: Dict[str, float],
        features: Dict
    ) -> Dict:
        """
        Generate full explanation for a diagnosis.

        Args:
            disease: Disease name
            query: Original query text
            scores: Dict with keys 'retrieval', 'classifier', 'rules', 'blended'
            features: Extracted patient features

        Returns:
            Dict with explanation components
        """
        # get confidence level
        confidence_level, confidence_pct = self._get_confidence_level(scores['blended'])

        # extract evidence
        evidence_sentences = self._extract_top_sentences(disease, query, top_n=2)

        # explain symptoms
        symptom_explanation = self._explain_symptoms(features, disease)

        # build summary using template
        summary = (
            f"Based on the patient presentation, {disease} has a {confidence_level} "
            f"likelihood ({confidence_pct:.1f}% confidence). {symptom_explanation}"
        )

        # interpret component scores
        score_interpretation = {
            'retrieval': self._interpret_retrieval_score(scores['retrieval']),
            'classifier': self._interpret_classifier_score(scores['classifier']),
            'rules': self._interpret_rules_score(scores['rules'])
        }

        return {
            'summary': summary,
            'evidence_sentences': evidence_sentences,
            'score_interpretation': score_interpretation,
            'symptom_explanation': symptom_explanation,
            'scores': scores
        }


# test it
if __name__ == "__main__":
    from symptom_extractor import SymptomExtractor

    print("\n🧪 Testing Explanation Generator\n")

    generator = ExplanationGenerator()
    extractor = SymptomExtractor()

    # test query
    query = "puppy has bloody diarrhea and vomiting"
    features = extractor.extract_features(query)

    # fake scores for testing
    test_scores = {
        'retrieval': 0.85,
        'classifier': 0.72,
        'rules': 1.0,
        'blended': 0.86
    }

    explanation = generator.generate_diagnosis_explanation(
        disease="Canine Parvovirus",
        query=query,
        scores=test_scores,
        features=features
    )

    print("="*70)
    print("EXPLANATION TEST")
    print("="*70)
    print(f"\nQuery: {query}")
    print(f"\nSummary:\n  {explanation['summary']}")
    print(f"\nScore Interpretations:")
    for component, interp in explanation['score_interpretation'].items():
        print(f"  {component.capitalize()}: {interp}")
    print(f"\nEvidence Sentences:")
    for i, sent in enumerate(explanation['evidence_sentences'], 1):
        print(f"  {i}. {sent['text'][:100]}...")
        print(f"     Source: {sent['source']}")
    print("="*70)
