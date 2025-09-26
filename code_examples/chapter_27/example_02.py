"""
Chapter 27 - Example 2
Extracted from Healthcare AI Implementation Guide
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

\# --- 1. Simulate Patient Data (Conceptual) ---
\# In a real scenario, this would be loaded from EHRs or clinical trial data.
\# Features might include age, gender, comorbidities, genomic markers, treatment history.
\# Outcome: time_to_event (e.g., time to recurrence, survival time), event_observed (1 if event occurred, 0 if censored)

np.random.seed(42)
num_patients = 500
data = {
    'age': np.random.randint(40, 80, num_patients),
    'gender': np.random.choice([0, 1], num_patients), \# 0 for female, 1 for male
    'comorbidity_score': np.random.randint(0, 5, num_patients),
    'treatment_A': np.random.choice([0, 1], num_patients), \# 1 if received treatment A
    'treatment_B': np.random.choice([0, 1], num_patients), \# 1 if received treatment B
    'genomic_marker_1': np.random.rand(num_patients),
    'time_to_event': np.random.exponential(scale=100, size=num_patients).astype(int) + 10, \# Survival time
    'event_observed': np.random.choice([0, 1], num_patients, p=[0.3, 0.7]) \# 0 for censored, 1 for event
}
df = pd.DataFrame(data)

\# Ensure time_to_event is positive
df['time_to_event'] = df['time_to_event'].apply(lambda x: max(x, 1))

\# --- 2. Preprocessing ---
\# Scale numerical features
scaler = StandardScaler()
features = ['age', 'comorbidity_score', 'genomic_marker_1']
df[features] = scaler.fit_transform(df[features])

\# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

\# --- 3. Model Training (Cox Proportional Hazards Model) ---
cph = CoxPHFitter()

\# Fit the model
\# The duration_col is 'time_to_event', and event_col is 'event_observed'
cph.fit(train_df, duration_col='time_to_event', event_col='event_observed', formula='age + gender + comorbidity_score + treatment_A + treatment_B + genomic_marker_1')

cph.print_summary()

\# --- 4. Model Evaluation ---
\# Concordance Index (C-index) is a common metric for survival models
c_index_train = concordance_index(train_df['time_to_event'], -cph.predict_partial_hazard(train_df), train_df['event_observed'])
c_index_test = concordance_index(test_df['time_to_event'], -cph.predict_partial_hazard(test_df), test_df['event_observed'])

print(f"Train C-index: {c_index_train:.4f}")
print(f"Test C-index: {c_index_test:.4f}")

\# --- 5. Personalized Risk Prediction (Conceptual) ---
\# Predict individual patient risk (partial hazard)
\# Higher partial hazard means higher risk of event

\# Example: Predict for a new patient (or a subset of test data)
new_patient_data = test_df.drop(columns=['time_to_event', 'event_observed']).iloc[0:5]
predicted_hazards = cph.predict_partial_hazard(new_patient_data)
print("\nPredicted Partial Hazards for 5 test patients:")
print(predicted_hazards)

\# --- 6. Error Handling (Conceptual) ---
def safe_predict_hazard(model, patient_data_df):
    try:
        \# Ensure all required features are present
        required_features = model.data_tr.columns.drop([model.duration_col, model.event_col])
        if not all(feature in patient_data_df.columns for feature in required_features):
            missing = [f for f in required_features if f not in patient_data_df.columns]
            raise ValueError(f"Missing required features for prediction: {missing}")

        \# Ensure data types are correct (e.g., numerical)
        for col in patient_data_df.columns:
            if col in required_features and not pd.api.types.is_numeric_dtype(patient_data_df[col]):
                raise TypeError(f"Feature '{col}' is not numeric. Please ensure all features are numerical.")

        \# Predict and return
        return model.predict_partial_hazard(patient_data_df)
    except Exception as e:
        print(f"Error during hazard prediction: {e}")
        return None

\# Example usage of error handling
\# result_hazards = safe_predict_hazard(cph, new_patient_data)
\# if result_hazards is not None:
\#     print("\nSafe Predicted Hazards:")
\#     print(result_hazards)
