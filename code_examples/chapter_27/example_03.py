"""
Chapter 27 - Example 3
Extracted from Healthcare AI Implementation Guide
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re

\# --- 1. Simulate Patient Data (Conceptual) ---
\# In a real scenario, this would be loaded from EHRs.
\# Features include structured data and unstructured clinical notes.

np.random.seed(42)
num_patients = 1000
data = {
    'age': np.random.randint(20, 90, num_patients),
    'gender': np.random.choice([0, 1], num_patients), \# 0 for female, 1 for male
    'num_diagnoses': np.random.randint(1, 10, num_patients),
    'num_medications': np.random.randint(1, 15, num_patients),
    'length_of_stay': np.random.randint(1, 30, num_patients),
    'clinical_notes': [
        "Patient presented with chest pain, shortness of breath. History of heart failure. Discharged with follow-up plan.",
        "Routine check-up, no significant findings. Healthy patient.",
        "Diabetic patient with poor glycemic control. Admitted for hyperglycemia. Education provided.",
        "Elderly patient with fall. Multiple comorbidities. Requires home health. Risk of readmission high.",
        "Post-surgical recovery, stable. No complications. Discharged.",
        "Patient with COPD exacerbation. Frequent admissions. Social support concerns. High risk.",
        "Mild pneumonia, treated with antibiotics. Good prognosis.",
        "Chronic kidney disease, managed with dialysis. Stable.",
        "Patient with mental health crisis. Follow-up with psychiatrist arranged.",
        "Hypertension, well-controlled. Routine medication refill."
    ] * (num_patients // 10), \# Repeat notes to fill up
    'readmitted_30_days': np.random.choice([0, 1], num_patients, p=[0.7, 0.3]) \# 0 for no readmission, 1 for readmission
}
df_workflow = pd.DataFrame(data)

\# --- 2. Preprocessing and Feature Engineering ---
\# Clean clinical notes (simple example)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) \# Remove non-alphabetic characters
    return text

df_workflow['cleaned_notes'] = df_workflow['clinical_notes'].apply(clean_text)

\# Separate features (X) and target (y)
X = df_workflow.drop(columns=['clinical_notes', 'readmitted_30_days'])
y = df_workflow['readmitted_30_days']

\# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

\# --- 3. Create a Pipeline for Structured and Text Data ---
\# For structured data, we'll scale it
\# For text data, we'll use TF-IDF

\# Define a custom transformer to select columns
class ColumnSelector(object):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

\# Pipeline for structured features
structured_features = ['age', 'gender', 'num_diagnoses', 'num_medications', 'length_of_stay']
structured_pipeline = Pipeline([
    ('selector', ColumnSelector(structured_features)),
    ('scaler', StandardScaler())
])

\# Pipeline for text features
text_pipeline = Pipeline([
    ('selector', ColumnSelector(['cleaned_notes'])),
    ('tfidf', TfidfVectorizer(max_features=1000)) \# Limit features for simplicity
])

\# Combine pipelines (using FeatureUnion or manual concatenation for simplicity here)
\# In a real application, use ColumnTransformer from sklearn.compose

\# For demonstration, we'll process separately and then combine
X_train_structured = structured_pipeline.fit_transform(X_train)
X_test_structured = structured_pipeline.transform(X_test)

X_train_text = text_pipeline.fit_transform(X_train)
X_test_text = text_pipeline.transform(X_test)

X_train_combined = np.hstack((X_train_structured, X_train_text.toarray()))
X_test_combined = np.hstack((X_test_structured, X_test_text.toarray()))

\# --- 4. Model Training (Logistic Regression) ---
model_lr = LogisticRegression(solver='liblinear', random_state=42)
model_lr.fit(X_train_combined, y_train)

\# --- 5. Model Evaluation ---
y_pred = model_lr.predict(X_test_combined)
y_proba = model_lr.predict_proba(X_test_combined)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

\# --- 6. Error Handling (Conceptual) ---
def safe_predict_readmission(model, structured_data, text_data, structured_scaler, text_vectorizer):
    try:
        \# Preprocess structured data
        processed_structured = structured_scaler.transform(structured_data)

        \# Preprocess text data
        cleaned_text_data = text_data.apply(clean_text)
        processed_text = text_vectorizer.transform(cleaned_text_data)

        \# Combine features
        combined_features = np.hstack((processed_structured, processed_text.toarray()))

        \# Predict
        prediction = model.predict(combined_features)
        probability = model.predict_proba(combined_features)[:, 1]
        return prediction, probability
    except Exception as e:
        print(f"Error during readmission prediction: {e}")
        return None, None

\# Example usage of error handling
\# sample_patient_structured = pd.DataFrame([[65, 0, 4, 8, 10]], columns=structured_features)
\# sample_patient_text = pd.Series(["Patient with history of heart failure, recent discharge. Concerns for medication adherence."], name='cleaned_notes')

\# pred, prob = safe_predict_readmission(model_lr, sample_patient_structured, sample_patient_text, structured_pipeline.named_steps['scaler'], text_pipeline.named_steps['tfidf'])
\# if pred is not None:
\#     print(f"\nSample Patient Prediction: {pred<sup>0</sup>}, Probability: {prob<sup>0</sup>:.4f}")
