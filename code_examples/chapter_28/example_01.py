"""
Chapter 28 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

\# 1. Simulate Data
np.random.seed(42)
n_samples = 1000

\# Confounders (X): age, severity_score
X = np.random.normal(0, 1, size=(n_samples, 2))
X_df = pd.DataFrame(X, columns=['age', 'severity_score'])

\# Treatment (T): new_drug (binary)
\# Treatment assignment depends on confounders (simulating observational data)
propensity_score = 1 / (1 + np.exp(-(X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.normal(0, 0.5, n_samples))))
T = np.random.binomial(1, propensity_score)

\# Outcome (Y): recovery_time
\# Outcome depends on confounders, treatment, and noise
Y = (X[:, 0] * 2 + X[:, 1] * 1.5 + T * 5 + np.random.normal(0, 1, n_samples))

Y_df = pd.DataFrame(Y, columns=['recovery_time'])
T_df = pd.DataFrame(T, columns=['new_drug'])

\# Combine into a single DataFrame
data = pd.concat([X_df, T_df, Y_df], axis=1)

\# 2. Define Causal Model using EconML (Double Machine Learning)
\# Y: outcome (recovery_time)
\# T: treatment (new_drug)
\# X: confounders (age, severity_score)
\# W: effect modifiers (none in this simple example, but could be included)

\# Initialize the Double Machine Learning estimator
\# We use RandomForestRegressor for both the outcome and treatment models
\# to handle potential non-linearities.
dml = LinearDML(model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42),
                model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=42),
                random_state=42)

\# Fit the model
dml.fit(Y=data['recovery_time'],
        T=data['new_drug'],
        X=data[['age', 'severity_score']])

\# 3. Estimate Causal Effect
\# The ATE is the average of the treatment effects for all individuals
ate_estimate = dml.ate(X=data[['age', 'severity_score']])
print(f"Estimated Average Treatment Effect (ATE): {ate_estimate<sup>0</sup>:.2f} (95% CI: {ate_estimate<sup>1</sup><sup>0</sup>:.2f}, {ate_estimate<sup>1</sup><sup>1</sup>:.2f})")

\# Estimate Conditional Average Treatment Effect (CATE) for specific individuals
\# For example, for a younger patient with low severity vs. an older patient with high severity
X_test = pd.DataFrame([
    {'age': -1, 'severity_score': -1}, \# Younger, less severe
    {'age': 1, 'severity_score': 1}    \# Older, more severe
])

cate_estimates = dml.effect(X_test)
cate_intervals = dml.effect_interval(X_test)

print(f"\nCATE for younger, less severe patient: {cate_estimates<sup>0</sup>:.2f} (95% CI: {cate_intervals<sup>0</sup><sup>0</sup>:.2f}, {cate_intervals<sup>0</sup><sup>1</sup>:.2f})")
print(f"CATE for older, more severe patient: {cate_estimates<sup>1</sup>:.2f} (95% CI: {cate_intervals<sup>1</sup><sup>0</sup>:.2f}, {cate_intervals<sup>1</sup><sup>1</sup>:.2f})")

\# Interpretation:
\# The ATE represents the average causal effect of the new drug on recovery time across the entire population.
\# CATE estimates show how this effect might vary for different patient profiles, enabling personalized treatment decisions.