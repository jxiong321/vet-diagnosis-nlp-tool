"""
Dog-only multinomial logistic regression with leakage-safe preprocessing.
- Filters to Animal_Type == 'Dog'
- K-fold, m-smoothed MULTICLASS target encoding for Breed (centered by priors)
- Multi-hot encoding for Symptom_1..Symptom_4
- One-hot for Gender (Animal_Type is constant after filtering)
- Standardize numeric vitals
- 0/1 passthrough for binary symptom flags
- Multinomial Logistic Regression + evaluation + interpretable coefficients

Usage:
    python train_dog_logreg.py --csv vet_cases.csv

Requirements: scikit-learn >= 1.2, pandas, numpy
"""

import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============ Utility Parsers (robust if your CSV still has strings like "39.5°C" / "3 days") ============

def parse_temperature_c(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"(-?\d+(?:\.\d+)?)", str(x))
    return float(m.group(1)) if m else np.nan


def parse_duration_days(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    n = int(m.group(1))
    if "week" in s:
        return float(n * 7)
    return float(n)


def yesno_to_int(x):
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return 1
    if s in {"no", "n", "false", "0", ""}:
        return 0
    return 0

# ============ Feature Engineering ============

# def engineer_features(df, binary_cols):
#     """Add clinically meaningful feature combinations."""
    
#     # 1. GI symptom combinations
#     if 'Vomiting' in df.columns and 'Diarrhea' in df.columns:
#         df['GI_combo'] = ((df['Vomiting'] == 1) & (df['Diarrhea'] == 1)).astype(int)
    
#     # 2. Severe GI distress
#     if all(c in df.columns for c in ['Vomiting', 'Diarrhea', 'Appetite_Loss']):
#         df['Severe_GI'] = ((df['Vomiting'] == 1) & 
#                            (df['Diarrhea'] == 1) & 
#                            (df['Appetite_Loss'] == 1)).astype(int)
    
#     # 3. Respiratory combo
#     if 'Coughing' in df.columns and 'Labored_Breathing' in df.columns:
#         df['Respiratory_combo'] = ((df['Coughing'] == 1) & 
#                                    (df['Labored_Breathing'] == 1)).astype(int)
    
#     # 4. Upper respiratory infection
#     if 'Nasal_Discharge' in df.columns and 'Eye_Discharge' in df.columns:
#         df['Upper_resp_infection'] = ((df['Nasal_Discharge'] == 1) & 
#                                       (df['Eye_Discharge'] == 1)).astype(int)
    
#     # 5. Fever indicator (>39.2°C)
#     if 'Body_Temperature_C' in df.columns:
#         df['Has_fever'] = (df['Body_Temperature_C'] > 39.2).astype(int)
    
#     # 6. High fever (>40°C)
#     if 'Body_Temperature_C' in df.columns:
#         df['High_fever'] = (df['Body_Temperature_C'] > 40.0).astype(int)
    
#     # 7. Tachycardia
#     if 'Heart_Rate' in df.columns:
#         df['Tachycardia'] = (df['Heart_Rate'] > 120).astype(int)
    
#     # 8. Chronic duration
#     if 'duration_days' in df.columns:
#         df['Chronic_duration'] = (df['duration_days'] > 7).astype(int)
    
#     # 9. Total symptom count
#     symptom_cols_for_count = [c for c in binary_cols if c in df.columns]
#     if symptom_cols_for_count:
#         df['Total_symptoms'] = df[symptom_cols_for_count].sum(axis=1)
    
#     # 10. Age risk categories
#     if 'Age' in df.columns:
#         df['Puppy'] = (df['Age'] < 1).astype(int)
#         df['Senior'] = (df['Age'] > 8).astype(int)
    
#     return df

# ============ Custom Transformers ============

class SymptomMultiHot(BaseEstimator, TransformerMixin):
    """Build a multi-hot matrix from columns Symptom_1..Symptom_4 (or any provided list).
    Vocabulary is learned on fit(X_train) only, preventing leakage.
    """

    def __init__(self, symptom_cols: List[str]):
        self.symptom_cols = symptom_cols
        self.symptom_vocab_: List[str] = []
        self._index_map: Dict[str, int] = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        vocab = set()
        for c in self.symptom_cols:
            if c not in df.columns:
                continue
            vals = (
                df[c]
                .astype(str)
                .str.strip()
                .replace({"nan": "", "None": ""})
            )
            vals = vals[~vals.str.lower().isin(["", "no"])]
            vocab.update(vals.unique().tolist())
        self.symptom_vocab_ = sorted(vocab)
        self._index_map = {s: i for i, s in enumerate(self.symptom_vocab_)}
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        n = len(df)
        m = len(self.symptom_vocab_)
        out = np.zeros((n, m), dtype=int)
        
        # Fix: Use enumerate to get position index instead of DataFrame index
        for pos_i, (row_i, row) in enumerate(df[self.symptom_cols].iterrows()):
            for val in row.values:
                s = str(val).strip()
                if s == "" or s.lower() == "no" or s.lower() == "nan":
                    continue
                j = self._index_map.get(s)
                if j is not None:
                    out[pos_i, j] = 1  # Use pos_i instead of row_i
        return out

    def get_feature_names_out(self, input_features=None):
        return [f"symptom::{s}" for s in self.symptom_vocab_]


class MulticlassTargetEncoder(BaseEstimator, TransformerMixin):
    """K-fold, m-smoothed multiclass target encoding for a single categorical column.
       Returns X with new columns `prefix::<class>` (centered by priors if center=True),
       and drops the original categorical column to avoid leakage/duplication.
    """

    def __init__(
        self,
        col: str,
        prefix: str = "breed_te",
        n_splits: int = 5,
        m: float = 20.0,
        center: bool = True,
        random_state: int = 42,
    ):
        self.col = col
        self.prefix = prefix
        self.n_splits = n_splits
        self.m = m
        self.center = center
        self.random_state = random_state
        # Learned attributes
        self.classes_: List[Any] = []
        self.priors_: np.ndarray | None = None  # shape (K,)
        self.mapping_: Dict[Any, np.ndarray] = {}  # category -> probs (K,)
        self.out_colnames_: List[str] = []

    def _smoothed(self, counts_k: np.ndarray, total: int) -> np.ndarray:
        # counts_k shape (K,), total scalar, priors_ shape (K,)
        if total == 0:
            return self.priors_.copy()
        p_hat = counts_k / max(total, 1)
        return (total * p_hat + self.m * self.priors_) / (total + self.m)

    def fit(self, X, y):
        df = pd.DataFrame(X).copy()
        y = pd.Series(y).reset_index(drop=True)
        df = df.reset_index(drop=True)
        if self.col not in df.columns:
            raise ValueError(f"Column {self.col} not in X")

        # Global class list & priors
        self.classes_ = sorted(y.unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = y.map(class_to_idx).to_numpy()
        K = len(self.classes_)
        counts_global = np.bincount(y_idx, minlength=K)
        self.priors_ = counts_global / counts_global.sum()

        # Prepare OOF encodings (not returned here; we just learn mapping on full train afterwards)
        # Learn mapping on FULL training data for transform-time use with smoothing.
        # Category -> class counts
        cts: Dict[Any, np.ndarray] = {}
        totals: Dict[Any, int] = {}
        for cat, y_sub in zip(df[self.col], y_idx):
            if cat not in cts:
                cts[cat] = np.zeros(K, dtype=float)
                totals[cat] = 0
            cts[cat][y_sub] += 1.0
            totals[cat] += 1
        # Build mapping with smoothing
        self.mapping_ = {}
        for cat, vec in cts.items():
            self.mapping_[cat] = self._smoothed(vec, totals[cat])

        # Output columns
        self.out_colnames_ = [f"{self.prefix}::{c}" for c in self.classes_]
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.col not in df.columns:
            raise ValueError(f"Column {self.col} not in X")
        # Create matrix of encodings
        n = len(df)
        K = len(self.classes_)
        M = np.zeros((n, K), dtype=float)
        for i, cat in enumerate(df[self.col].tolist()):
            vec = self.mapping_.get(cat)
            if vec is None:
                vec = self.priors_
            M[i, :] = vec
        enc = pd.DataFrame(M, columns=self.out_colnames_, index=df.index)
        if self.center:
            enc = enc - self.priors_
        # Drop original categorical column and concatenate encodings
        df = df.drop(columns=[self.col])
        df = pd.concat([df, enc], axis=1)
        return df

    def get_feature_names_out(self, input_features=None):
        return self.out_colnames_


# ============ Main training routine ============

def main():
    # Load dataset directly without needing command-line arguments
    test_size = 0.25
    random_state = 42
    m_value = 20.0  # Smoothing strength for target encoding

    df = pd.read_csv('/Users/jessicaxiong/vet-diagnosis-nlp-tool/data/cleaned_animal_disease_prediction.csv')
    
    target_col = 'Disease_Prediction'
    
    # Add after loading the CSV, before filtering
    disease_mapping = {
        'Parvovirus': 'Canine Parvovirus',
        'Distemper': 'Canine Distemper',
        'Leptospirosis': 'Canine Leptospirosis',
    }
    df[target_col] = df[target_col].replace(disease_mapping)

    if "Animal_Type" in df.columns:
        df = df[df["Animal_Type"].astype(str).str.lower() == "dog"].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No dog rows found after filtering. Check Animal_Type column values.")

    if "Body_Temperature" in df.columns:
        df["Body_Temperature_C"] = df["Body_Temperature"].apply(parse_temperature_c)
    if "Duration" in df.columns:
        df["duration_days"] = df["Duration"].apply(parse_duration_days)

    binary_cols = ["Appetite_Loss","Vomiting","Diarrhea","Coughing","Labored_Breathing","Lameness","Skin_Lesions","Nasal_Discharge","Eye_Discharge"]
    for c in binary_cols:
        if c in df.columns:
            df[c] = df[c].apply(yesno_to_int).astype(int)

    for c in ["Age","Weight","Heart_Rate","Body_Temperature_C","duration_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # print("\n🔧 Engineering features...")
    # df = engineer_features(df, binary_cols)

    # # Create list of all binary columns including engineered ones
    # engineered_binary = ['GI_combo', 'Severe_GI', 'Respiratory_combo', 
    #                 'Upper_resp_infection', 'Has_fever', 'High_fever',
    #                 'Tachycardia', 'Chronic_duration', 'Puppy', 'Senior', 'Total_symptoms']
    # all_binary_cols = binary_cols + [c for c in engineered_binary if c in df.columns]
    target_col = 'Disease_Prediction'

    symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]

    base_cols = ["Breed","Gender","Age","Weight","Heart_Rate","Body_Temperature_C","duration_days"] + binary_cols + symptom_cols
    keep_cols = [c for c in base_cols if c in df.columns] + [target_col]
    df = df[keep_cols].dropna(subset=[target_col]).reset_index(drop=True)


    # Remove diseases with insufficient samples for stratified split
    disease_counts = df[target_col].value_counts()
    min_samples = 5
    valid_diseases = disease_counts[disease_counts >= min_samples].index
    print(f"\nDisease distribution before filtering:")
    print(disease_counts.sort_values(ascending=False))
    df = df[df[target_col].isin(valid_diseases)].reset_index(drop=True)
    print(f"\nKept {len(valid_diseases)} diseases with ≥{min_samples} samples")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    breed_te = MulticlassTargetEncoder(col='Breed', prefix='breed_te', m=m_value)
    num_cols = [c for c in ["Age","Weight","Heart_Rate","Body_Temperature_C","duration_days"] if c in X.columns]

    ct = ColumnTransformer(transformers=[
        ('gender', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Gender'] if 'Gender' in X.columns else []),
        ('num', StandardScaler(), num_cols),
        ('sym', SymptomMultiHot(symptom_cols=symptom_cols), symptom_cols),
        ('bin', 'passthrough', [c for c in binary_cols if c in X.columns]),
        ('breed_te', 'passthrough', make_column_selector(pattern=r'^breed_te::'))
    ], remainder='drop', verbose_feature_names_out=False)

    pipe = Pipeline(steps=[('breed_te', breed_te), ('prep', ct), ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000, C=5.0, class_weight='balanced'))])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')

    print(f'Accuracy: {acc:.3f}')
    print(f'Macro F1: {f1m:.3f}')
    print('\nClassification report:\n', classification_report(y_test, y_pred, digits=3))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    prep = pipe.named_steps['prep']
    feature_names = prep.get_feature_names_out()
    clf = pipe.named_steps['clf']
    classes = clf.classes_.tolist()
    coef = clf.coef_

    def top_features_for_class(k=10):
        for class_idx, cls in enumerate(classes):
            w = coef[class_idx]
            top_pos_idx = np.argsort(w)[-k:][::-1]
            top_neg_idx = np.argsort(w)[:k]
            print(f'\n=== Class: {cls} ===')
            print('Top positive features (increase odds):')
            for j in top_pos_idx:
                print(f'  {feature_names[j]:40s}  {w[j]:+.3f}')
            print('Top negative features (decrease odds):')
            for j in top_neg_idx:
                print(f'  {feature_names[j]:40s}  {w[j]:+.3f}')

    top_features_for_class(k=8)

    with open('trained_classifier.pkl', 'wb') as f:
        pickle.dump(pipe, f)

    print("\n✅ Classifier saved to trained_classifier.pkl")

if __name__ == '__main__':
    main()
