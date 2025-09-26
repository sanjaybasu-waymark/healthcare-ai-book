"""
Chapter 27 - Example 4
Extracted from Healthcare AI Implementation Guide
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

\# --- 1. Simulate Molecular Data (Conceptual) ---
\# In a real scenario, this would be loaded from chemical databases (e.g., PubChem, ChEMBL).
\# We'll use SMILES strings to represent molecules and predict a hypothetical 'activity' score.

data_molecules = {
    'SMILES': [
        'CCO',
        'CCC(=O)O',
        'c1ccccc1',
        'CC(=O)Oc1ccccc1C(=O)O', \# Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', \# Caffeine
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', \# Ibuprofen
        'O=C(Nc1ccc(Cl)cc1)c1ccccc1', \# Example drug-like molecule
        'CC(C)N(C(=O)OC(C)(C)C)Cc1ccccc1', \# Another example
        'C1=CC(=CC=C1C(=O)O)O', \# Salicylic acid
        'CC(=O)NC1=CC=C(C=C1)O' \# Paracetamol
    ],
    'activity_score': [0.7, 0.3, 0.1, 0.85, 0.6, 0.9, 0.75, 0.4, 0.5, 0.8]
}
df_molecules = pd.DataFrame(data_molecules)

\# --- 2. Feature Engineering (Molecular Descriptors using RDKit) ---
def generate_molecular_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    \# Example descriptors: Molecular Weight, LogP, TPSA, NumHDonors, NumHAcceptors
    features = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
    }
    return pd.Series(features)

\# Apply feature generation
molecular_features = df_molecules['SMILES'].apply(generate_molecular_features)
df_molecules = pd.concat([df_molecules, molecular_features], axis=1).dropna()

\# Separate features (X) and target (y)
X_mol = df_molecules.drop(columns=['SMILES', 'activity_score'])
y_mol = df_molecules['activity_score']

\# Split data
X_train_mol, X_test_mol, y_train_mol, y_test_mol = train_test_split(X_mol, y_mol, test_size=0.2, random_state=42)

\# --- 3. Model Training (Random Forest Regressor) ---
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train_mol, y_train_mol)

\# --- 4. Model Evaluation ---
y_pred_mol = model_rf.predict(X_test_mol)

print(f"\nMean Squared Error: {mean_squared_error(y_test_mol, y_pred_mol):.4f}")
print(f"R-squared: {r2_score(y_test_mol, y_pred_mol):.4f}")

\# --- 5. Error Handling (Conceptual) ---
def safe_predict_activity(model, smiles_string):
    try:
        features = generate_molecular_features(smiles_string)
        if features is None:
            raise ValueError("Invalid SMILES string provided.")
        \# Ensure features are in the correct order and format for the model
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)
        return prediction<sup>0</sup>
    except Exception as e:
        print(f"Error during activity prediction: {e}")
        return None

\# Example usage
\# new_smiles = 'CC(=O)O'
\# predicted_activity = safe_predict_activity(model_rf, new_smiles)
\# if predicted_activity is not None:
\#     print(f"\nPredicted activity for {new_smiles}: {predicted_activity:.4f}")
