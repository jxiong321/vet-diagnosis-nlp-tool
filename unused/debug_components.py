"""Debug script to analyze individual component performance."""

from blended_system import BlendedDiagnosisSystem, create_sample_test_cases
from bm25_retriever import BM25Retriever
from log_regression import MulticlassTargetEncoder, SymptomMultiHot
import json
import pickle

# Load components
retriever = BM25Retriever(
    symptoms_path='data/symptoms.json',
    conditions_path='data/conditions.json'
)
passages = []
with open('data/passages.jsonl', 'r') as f:
    for line in f:
        passages.append(json.loads(line))
retriever.index(passages)

with open('trained_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

system = BlendedDiagnosisSystem(retriever, classifier, weights={'retrieval': 1/3, 'classifier': 1/3, 'rules': 1/3})

# Test first 3 cases and see individual component scores
test_cases = create_sample_test_cases()

for idx in range(3):
    query = test_cases[idx]['query']
    true_disease = test_cases[idx]['true_disease']

    print(f'\n{"="*70}')
    print(f'TEST CASE {idx + 1}')
    print(f'{"="*70}')
    print(f'Query: "{query}"')
    print(f'True disease: {true_disease}\n')

    results = system.diagnose(query, top_k=5)

    print('Component Rankings:')
    print('-' * 70)

    print('\nRetrieval Top 5:')
    for i, disease in enumerate(results['retrieval_top'][:5], 1):
        score = results['retrieval_scores'].get(disease, 0)
        check = '✓' if disease.lower() == true_disease.lower() else ' '
        print(f'{check} {i}. {disease}: {score:.3f}')

    print('\nClassifier Top 5:')
    for i, disease in enumerate(results['classifier_top'][:5], 1):
        score = results['classifier_scores'].get(disease, 0)
        check = '✓' if disease.lower() == true_disease.lower() else ' '
        print(f'{check} {i}. {disease}: {score:.3f}')

    print('\nRules Top 5:')
    for i, disease in enumerate(results['rules_top'][:5], 1):
        score = results['rule_scores'].get(disease, 0)
        check = '✓' if disease.lower() == true_disease.lower() else ' '
        print(f'{check} {i}. {disease}: {score:.3f}')

    print('\nBlended Top 5 (Equal Weights):')
    for i, disease in enumerate(results['blended_top'][:5], 1):
        score = results['blended_scores'].get(disease, 0)
        check = '✓' if disease.lower() == true_disease.lower() else ' '
        print(f'{check} {i}. {disease}: {score:.3f}')
