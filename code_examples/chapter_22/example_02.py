"""
Chapter 22 - Example 2
Extracted from Healthcare AI Implementation Guide
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

\# Simulate a dataset for drug outcome prediction
def generate_simulated_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'gender': np.random.choice([0, 1], num_samples), \# 0 for female, 1 for male
        'bmi': np.random.normal(25, 5, num_samples),
        'creatinine': np.random.normal(1.0, 0.3, num_samples),
        'drug_dose': np.random.normal(100, 20, num_samples),
        'genetic_marker_A': np.random.choice([0, 1], num_samples),
        'genetic_marker_B': np.random.choice([0, 1], num_samples),
        'outcome': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) \# 0 for no response, 1 for response
    }
    df = pd.DataFrame(data)
    
    \# Introduce some correlation for a more realistic scenario
    df['outcome'] = df.apply(lambda row: 1 if (row['age'] > 60 and row['drug_dose'] > 110) or \
                                            (row['genetic_marker_A'] == 1 and row['bmi'] > 30) else row['outcome'], axis=1)
    df['outcome'] = df.apply(lambda row: 0 if (row['age'] < 30 and row['drug_dose'] < 90) else row['outcome'], axis=1)
    
    return df

df = generate_simulated_data()

X = df.drop('outcome', axis=1)
y = df['outcome']

\# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

\# Feature Scaling (important for many ML models, though less critical for tree-based models like XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"Training data shape: {X_train_scaled_df.shape}")
print(f"Testing data shape: {X_test_scaled_df.shape}")