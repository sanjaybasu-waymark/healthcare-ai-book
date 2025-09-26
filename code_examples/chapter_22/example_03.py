"""
Chapter 22 - Example 3
Extracted from Healthcare AI Implementation Guide
"""

import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

\# Initialize and train the XGBoost Classifier
\# Parameters can be tuned for optimal performance
model = xgb.XGBClassifier(
    objective='binary:logistic', \# For binary classification
    eval_metric='logloss',       \# Evaluation metric during training
    use_label_encoder=False,     \# Suppress warning for label encoding
    n_estimators=100,            \# Number of boosting rounds
    learning_rate=0.1,           \# Step size shrinkage to prevent overfitting
    max_depth=5,                 \# Maximum depth of a tree
    subsample=0.8,               \# Subsample ratio of the training instance
    colsample_bytree=0.8,        \# Subsample ratio of columns when constructing each tree
    random_state=42,
    n_jobs=-1                    \# Use all available CPU cores
)

model.fit(X_train_scaled_df, y_train)

\# Make predictions on the test set
y_pred_proba = model.predict_proba(X_test_scaled_df)[:, 1] \# Probability of positive class
y_pred = model.predict(X_test_scaled_df)

\# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

\# Example of error handling: checking for missing features
try:
    \# Simulate missing a feature in new data
    X_new_data = X_test_scaled_df.drop('bmi', axis=1).head(5)
    model.predict(X_new_data)
except xgb.core.XGBoostError as e:
    print(f"\nError during prediction: {e}")
    print("Ensure that the input data for prediction has the same features as the training data.")

\# Example of saving and loading the model for production
import joblib

model_filename = 'xgboost_drug_outcome_model.joblib'
joblib.dump(model, model_filename)
print(f"\nModel saved to {model_filename}")

loaded_model = joblib.load(model_filename)
loaded_y_pred = loaded_model.predict(X_test_scaled_df)
print(f"Model loaded and predictions made successfully.")
assert np.array_equal(y_pred, loaded_y_pred)