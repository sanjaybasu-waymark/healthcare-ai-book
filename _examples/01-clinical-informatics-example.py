#!/usr/bin/env python3
"""
Healthcare AI Implementation Guide - Chapter 1 Examples
Clinical Informatics Foundations

This module provides comprehensive examples for clinical informatics concepts
including data processing, quality assessment, and decision support systems.

Author: Healthcare AI Implementation Guide
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalDataGenerator:
    """
    Generates synthetic clinical data for demonstration and testing purposes.
    
    This class creates realistic healthcare data that maintains clinical
    relationships and distributions while ensuring patient privacy.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize the clinical data generator."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_patient_demographics(self, n_patients: int) -> Dict[str, np.ndarray]:
        """Generate patient demographic data."""
        logger.info(f"Generating demographics for {n_patients} patients")
        
        # Patient identifiers
        patient_ids = [f"P{i:06d}" for i in range(1, n_patients + 1)]
        
        # Age distribution (slightly skewed toward older patients)
        ages = np.random.normal(65, 15, n_patients).astype(int)
        ages = np.clip(ages, 18, 100)
        
        # Gender distribution
        genders = np.random.choice(['M', 'F'], n_patients, p=[0.48, 0.52])
        
        # Race/ethnicity distribution (US demographics)
        races = np.random.choice(
            ['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
            n_patients, 
            p=[0.6, 0.2, 0.15, 0.04, 0.01]
        )
        
        return {
            'patient_id': patient_ids,
            'age': ages,
            'gender': genders,
            'race': races
        }
    
    def generate_vital_signs(self, n_patients: int, ages: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate vital signs with age-related correlations."""
        logger.info("Generating vital signs data")
        
        # Systolic BP increases with age
        systolic_bp = 120 + 0.5 * (ages - 40) + np.random.normal(0, 15, n_patients)
        systolic_bp = np.clip(systolic_bp, 90, 200)
        
        # Diastolic BP
        diastolic_bp = 80 + 0.2 * (ages - 40) + np.random.normal(0, 10, n_patients)
        diastolic_bp = np.clip(diastolic_bp, 60, 120)
        
        # Heart rate (slightly lower in older patients)
        heart_rate = 75 - 0.1 * (ages - 40) + np.random.normal(0, 12, n_patients)
        heart_rate = np.clip(heart_rate, 50, 120)
        
        # Temperature (normal distribution)
        temperature = np.random.normal(98.6, 1.2, n_patients)
        temperature = np.clip(temperature, 96, 102)
        
        return {
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'temperature': temperature
        }
    
    def generate_laboratory_values(self, n_patients: int, ages: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate laboratory values with clinical correlations."""
        logger.info("Generating laboratory values")
        
        # Glucose (higher in older patients, log-normal distribution)
        glucose_base = np.log(100 + 0.5 * (ages - 40))
        glucose = np.random.lognormal(glucose_base, 0.3, n_patients)
        glucose = np.clip(glucose, 70, 400)
        
        # Creatinine (increases with age, especially >65)
        creatinine_base = 1.0 + 0.01 * np.maximum(0, ages - 65)
        creatinine = np.random.lognormal(np.log(creatinine_base), 0.3, n_patients)
        creatinine = np.clip(creatinine, 0.5, 5.0)
        
        # Hemoglobin (slightly lower in older patients)
        hemoglobin = 13.5 - 0.02 * (ages - 40) + np.random.normal(0, 1.5, n_patients)
        hemoglobin = np.clip(hemoglobin, 8, 18)
        
        return {
            'glucose': glucose,
            'creatinine': creatinine,
            'hemoglobin': hemoglobin
        }
    
    def generate_comorbidities(self, n_patients: int, ages: np.ndarray, 
                             glucose: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate comorbidity data with realistic prevalence."""
        logger.info("Generating comorbidity data")
        
        # Diabetes probability increases with age and glucose
        diabetes_prob = 0.05 + 0.01 * (ages - 40) + 0.001 * (glucose - 100)
        diabetes_prob = np.clip(diabetes_prob, 0, 0.8)
        diabetes = np.random.binomial(1, diabetes_prob, n_patients)
        
        # Hypertension probability increases with age
        hypertension_prob = 0.1 + 0.015 * (ages - 30)
        hypertension_prob = np.clip(hypertension_prob, 0, 0.9)
        hypertension = np.random.binomial(1, hypertension_prob, n_patients)
        
        return {
            'diabetes': diabetes,
            'hypertension': hypertension
        }
    
    def generate_outcomes(self, n_patients: int, ages: np.ndarray, 
                         diabetes: np.ndarray, hypertension: np.ndarray,
                         systolic_bp: np.ndarray, creatinine: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate outcome variables (e.g., readmission risk)."""
        logger.info("Generating outcome data")
        
        # 30-day readmission risk model
        risk_score = (
            0.02 * ages +
            0.8 * diabetes +
            0.5 * hypertension +
            0.01 * (systolic_bp - 120) +
            0.7 * (creatinine > 1.5).astype(int) +
            np.random.normal(0, 0.5, n_patients)  # Random noise
        )
        
        # Convert to probability using logistic function
        readmission_prob = 1 / (1 + np.exp(-risk_score + 3))
        readmission = np.random.binomial(1, readmission_prob, n_patients)
        
        return {
            'readmission_30d': readmission,
            'risk_score': risk_score
        }
    
    def generate_complete_dataset(self, n_patients: int = 1000) -> pd.DataFrame:
        """Generate a complete synthetic clinical dataset."""
        logger.info(f"Generating complete clinical dataset for {n_patients} patients")
        
        # Generate all components
        demographics = self.generate_patient_demographics(n_patients)
        vitals = self.generate_vital_signs(n_patients, demographics['age'])
        labs = self.generate_laboratory_values(n_patients, demographics['age'])
        comorbidities = self.generate_comorbidities(
            n_patients, demographics['age'], labs['glucose']
        )
        outcomes = self.generate_outcomes(
            n_patients, demographics['age'], 
            comorbidities['diabetes'], comorbidities['hypertension'],
            vitals['systolic_bp'], labs['creatinine']
        )
        
        # Combine all data
        all_data = {**demographics, **vitals, **labs, **comorbidities, **outcomes}
        clinical_df = pd.DataFrame(all_data)
        
        logger.info(f"Dataset generated successfully with {len(clinical_df)} patients")
        logger.info(f"Readmission rate: {clinical_df['readmission_30d'].mean():.1%}")
        
        return clinical_df


class ClinicalDataQualityAssessment:
    """
    Comprehensive data quality assessment framework for clinical data.
    
    This class implements healthcare-specific data quality checks including
    completeness, validity, consistency, and clinical plausibility.
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with clinical dataset."""
        self.data = data
        self.quality_report = {}
        
        # Define clinical reference ranges
        self.clinical_ranges = {
            'age': (0, 120),
            'systolic_bp': (60, 250),
            'diastolic_bp': (30, 150),
            'heart_rate': (30, 200),
            'temperature': (95, 110),
            'glucose': (30, 800),
            'creatinine': (0.1, 15),
            'hemoglobin': (3, 20)
        }
        
    def assess_completeness(self) -> pd.Series:
        """Assess data completeness across all variables."""
        logger.info("Assessing data completeness")
        
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        self.quality_report['completeness'] = {
            'missing_counts': missing_data.to_dict(),
            'missing_percentages': missing_percent.to_dict(),
            'total_records': len(self.data),
            'complete_records': len(self.data.dropna())
        }
        
        return missing_percent
    
    def assess_validity(self) -> Dict[str, Dict]:
        """Assess data validity using clinical reference ranges."""
        logger.info("Assessing data validity")
        
        validity_issues = {}
        
        for column, (min_val, max_val) in self.clinical_ranges.items():
            if column in self.data.columns:
                out_of_range = (
                    (self.data[column] < min_val) | 
                    (self.data[column] > max_val)
                ).sum()
                
                validity_issues[column] = {
                    'out_of_range_count': out_of_range,
                    'out_of_range_percent': (out_of_range / len(self.data)) * 100,
                    'min_value': self.data[column].min(),
                    'max_value': self.data[column].max(),
                    'reference_range': (min_val, max_val)
                }
        
        self.quality_report['validity'] = validity_issues
        return validity_issues
    
    def assess_consistency(self) -> Dict[str, Dict]:
        """Assess internal data consistency."""
        logger.info("Assessing data consistency")
        
        consistency_issues = {}
        
        # Check systolic vs diastolic BP
        if 'systolic_bp' in self.data.columns and 'diastolic_bp' in self.data.columns:
            bp_inconsistent = (self.data['systolic_bp'] <= self.data['diastolic_bp']).sum()
            consistency_issues['bp_inconsistency'] = {
                'count': bp_inconsistent,
                'percent': (bp_inconsistent / len(self.data)) * 100,
                'description': 'Systolic BP <= Diastolic BP'
            }
        
        # Check age vs comorbidity consistency
        if 'age' in self.data.columns and 'diabetes' in self.data.columns:
            young_diabetes = ((self.data['age'] < 30) & (self.data['diabetes'] == 1)).sum()
            consistency_issues['young_diabetes'] = {
                'count': young_diabetes,
                'percent': (young_diabetes / len(self.data)) * 100,
                'description': 'Diabetes in patients under 30'
            }
        
        self.quality_report['consistency'] = consistency_issues
        return consistency_issues
    
    def assess_clinical_plausibility(self) -> Dict[str, Dict]:
        """Assess clinical plausibility of data combinations."""
        logger.info("Assessing clinical plausibility")
        
        plausibility_issues = {}
        
        # Check for extremely high creatinine with normal age
        if 'creatinine' in self.data.columns and 'age' in self.data.columns:
            high_creat_young = (
                (self.data['creatinine'] > 3.0) & (self.data['age'] < 50)
            ).sum()
            plausibility_issues['high_creatinine_young'] = {
                'count': high_creat_young,
                'percent': (high_creat_young / len(self.data)) * 100,
                'description': 'Very high creatinine in young patients'
            }
        
        # Check for very low hemoglobin without obvious cause
        if 'hemoglobin' in self.data.columns:
            severe_anemia = (self.data['hemoglobin'] < 7.0).sum()
            plausibility_issues['severe_anemia'] = {
                'count': severe_anemia,
                'percent': (severe_anemia / len(self.data)) * 100,
                'description': 'Severe anemia (Hgb < 7.0)'
            }
        
        self.quality_report['plausibility'] = plausibility_issues
        return plausibility_issues
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive data quality report."""
        logger.info("Generating comprehensive quality report")
        
        print("Clinical Data Quality Assessment Report")
        print("=" * 60)
        
        # Completeness assessment
        completeness = self.assess_completeness()
        print(f"\n1. DATA COMPLETENESS:")
        print(f"   Total records: {self.quality_report['completeness']['total_records']:,}")
        print(f"   Complete records: {self.quality_report['completeness']['complete_records']:,}")
        
        missing_vars = completeness[completeness > 0]
        if len(missing_vars) > 0:
            print("   Variables with missing data:")
            for var, pct in missing_vars.items():
                print(f"     • {var}: {pct:.1f}% missing")
        else:
            print("   ✓ No missing data found")
        
        # Validity assessment
        validity = self.assess_validity()
        print(f"\n2. DATA VALIDITY:")
        validity_issues_found = False
        for var, issues in validity.items():
            if issues['out_of_range_count'] > 0:
                validity_issues_found = True
                print(f"   • {var}: {issues['out_of_range_count']} values out of range "
                      f"({issues['out_of_range_percent']:.1f}%)")
                print(f"     Range: {issues['min_value']:.2f} - {issues['max_value']:.2f}, "
                      f"Expected: {issues['reference_range']}")
        
        if not validity_issues_found:
            print("   ✓ All values within expected clinical ranges")
        
        # Consistency assessment
        consistency = self.assess_consistency()
        print(f"\n3. DATA CONSISTENCY:")
        consistency_issues_found = False
        for issue, data in consistency.items():
            if data['count'] > 0:
                consistency_issues_found = True
                print(f"   • {data['description']}: {data['count']} cases "
                      f"({data['percent']:.1f}%)")
        
        if not consistency_issues_found:
            print("   ✓ No consistency issues found")
        
        # Clinical plausibility
        plausibility = self.assess_clinical_plausibility()
        print(f"\n4. CLINICAL PLAUSIBILITY:")
        plausibility_issues_found = False
        for issue, data in plausibility.items():
            if data['count'] > 0:
                plausibility_issues_found = True
                print(f"   • {data['description']}: {data['count']} cases "
                      f"({data['percent']:.1f}%)")
        
        if not plausibility_issues_found:
            print("   ✓ No plausibility issues identified")
        
        print("\n" + "=" * 60)
        
        return self.quality_report


class ClinicalDecisionSupportSystem:
    """
    Advanced clinical decision support system for readmission risk prediction.
    
    This system implements machine learning models with clinical validation,
    interpretability features, and actionable recommendations.
    """
    
    def __init__(self):
        """Initialize the clinical decision support system."""
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        self.model_metrics = {}
        
        # Risk thresholds for clinical decision making
        self.risk_thresholds = {
            'low': 0.15,
            'moderate': 0.35,
            'high': 0.55,
            'very_high': 0.75
        }
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer clinical features for modeling."""
        logger.info("Engineering clinical features")
        
        features = data.copy()
        
        # Encode categorical variables
        features['gender_M'] = (features['gender'] == 'M').astype(int)
        features['race_Black'] = (features['race'] == 'Black').astype(int)
        features['race_Hispanic'] = (features['race'] == 'Hispanic').astype(int)
        features['race_Asian'] = (features['race'] == 'Asian').astype(int)
        
        # Create clinically meaningful derived features
        features['pulse_pressure'] = features['systolic_bp'] - features['diastolic_bp']
        features['mean_arterial_pressure'] = (
            features['diastolic_bp'] + (features['pulse_pressure'] / 3)
        )
        
        # Age-based risk factors
        features['elderly'] = (features['age'] >= 75).astype(int)
        features['very_elderly'] = (features['age'] >= 85).astype(int)
        
        # Clinical risk combinations
        features['diabetes_elderly'] = features['diabetes'] * features['elderly']
        features['ckd_stage'] = np.where(features['creatinine'] > 1.5, 1, 0)
        features['anemia'] = np.where(features['hemoglobin'] < 12, 1, 0)
        
        # Metabolic syndrome indicators
        features['metabolic_syndrome_score'] = (
            features['diabetes'] +
            features['hypertension'] +
            (features['glucose'] > 126).astype(int)
        )
        
        # Define final feature set
        self.feature_columns = [
            'age', 'gender_M', 'race_Black', 'race_Hispanic', 'race_Asian',
            'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature',
            'glucose', 'creatinine', 'hemoglobin',
            'diabetes', 'hypertension',
            'pulse_pressure', 'mean_arterial_pressure',
            'elderly', 'very_elderly', 'diabetes_elderly',
            'ckd_stage', 'anemia', 'metabolic_syndrome_score'
        ]
        
        return features[self.feature_columns]
    
    def train_model(self, data: pd.DataFrame, target_column: str = 'readmission_30d') -> Dict:
        """Train the readmission prediction model with comprehensive evaluation."""
        logger.info("Training clinical prediction model")
        
        # Prepare features and target
        X = self.engineer_features(data)
        y = data[target_column]
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model with clinical-appropriate parameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train, y_train)
        
        # Model evaluation
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        self.model_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        self.model_metrics['auc_roc'] = auc(fpr, tpr)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Print results
        print("Clinical Decision Support Model Training Results")
        print("=" * 55)
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Baseline readmission rate: {y.mean():.1%}")
        print("\nModel Performance:")
        print(f"  Accuracy: {self.model_metrics['accuracy']:.3f}")
        print(f"  Precision: {self.model_metrics['precision']:.3f}")
        print(f"  Recall: {self.model_metrics['recall']:.3f}")
        print(f"  F1-Score: {self.model_metrics['f1_score']:.3f}")
        print(f"  AUC-ROC: {self.model_metrics['auc_roc']:.3f}")
        
        print("\nTop 10 Most Important Clinical Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'metrics': self.model_metrics
        }
    
    def predict_patient_risk(self, patient_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict readmission risk for individual patient with clinical recommendations."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Engineer features
        features = self.engineer_features(patient_data)
        
        # Get prediction
        risk_probability = self.model.predict_proba(features)[:, 1][0]
        
        # Determine risk category
        if risk_probability < self.risk_thresholds['low']:
            risk_category = 'Low'
            risk_color = 'green'
        elif risk_probability < self.risk_thresholds['moderate']:
            risk_category = 'Moderate'
            risk_color = 'yellow'
        elif risk_probability < self.risk_thresholds['high']:
            risk_category = 'High'
            risk_color = 'orange'
        else:
            risk_category = 'Very High'
            risk_color = 'red'
        
        # Generate clinical recommendations
        recommendations = self._generate_clinical_recommendations(
            risk_category, patient_data, features
        )
        
        # Get feature contributions (simplified SHAP-like explanation)
        feature_contributions = self._explain_prediction(features)
        
        return {
            'patient_id': patient_data['patient_id'].iloc[0],
            'risk_probability': risk_probability,
            'risk_category': risk_category,
            'risk_color': risk_color,
            'recommendations': recommendations,
            'key_risk_factors': feature_contributions[:5],  # Top 5 contributors
            'model_confidence': self._calculate_prediction_confidence(features)
        }
    
    def _generate_clinical_recommendations(self, risk_category: str, 
                                         patient_data: pd.DataFrame,
                                         features: pd.DataFrame) -> List[str]:
        """Generate evidence-based clinical recommendations."""
        recommendations = []
        patient = patient_data.iloc[0]
        
        # Base recommendations by risk level
        if risk_category in ['High', 'Very High']:
            recommendations.extend([
                "Comprehensive discharge planning with multidisciplinary team",
                "Schedule follow-up appointment within 7 days of discharge",
                "Medication reconciliation and adherence counseling",
                "Consider home health services or transitional care management"
            ])
        elif risk_category == 'Moderate':
            recommendations.extend([
                "Standard discharge planning with care coordination",
                "Schedule follow-up within 14 days",
                "Provide discharge education materials"
            ])
        else:
            recommendations.extend([
                "Standard discharge process",
                "Routine follow-up as clinically indicated"
            ])
        
        # Condition-specific recommendations
        if patient['diabetes'] == 1:
            recommendations.append("Diabetes self-management education and glucose monitoring plan")
            if patient['glucose'] > 180:
                recommendations.append("Endocrinology consultation for glucose management")
        
        if patient['creatinine'] > 1.5:
            recommendations.append("Nephrology consultation and renal function monitoring")
            recommendations.append("Medication dosing adjustments for renal function")
        
        if patient['hypertension'] == 1 and patient['systolic_bp'] > 160:
            recommendations.append("Blood pressure monitoring and antihypertensive optimization")
        
        if patient['age'] >= 75:
            recommendations.append("Geriatric assessment for functional status and fall risk")
        
        if patient['hemoglobin'] < 10:
            recommendations.append("Anemia workup and management")
        
        return recommendations
    
    def _explain_prediction(self, features: pd.DataFrame) -> List[Dict]:
        """Provide simplified feature importance explanation for prediction."""
        # Get feature values
        feature_values = features.iloc[0]
        
        # Calculate approximate contributions (feature_value * importance)
        contributions = []
        for idx, row in self.feature_importance.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            value = feature_values[feature_name]
            
            contribution = {
                'feature': feature_name,
                'value': value,
                'importance': importance,
                'contribution_score': abs(value * importance)
            }
            contributions.append(contribution)
        
        # Sort by contribution score
        contributions.sort(key=lambda x: x['contribution_score'], reverse=True)
        
        return contributions
    
    def _calculate_prediction_confidence(self, features: pd.DataFrame) -> float:
        """Calculate prediction confidence based on model uncertainty."""
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features)[0]
        
        # Confidence is the maximum probability
        confidence = max(probabilities)
        
        return confidence


def create_fhir_resources(patient_data: pd.Series) -> Dict[str, Any]:
    """
    Create FHIR-compliant resources for a patient.
    
    This function demonstrates how to structure clinical data according to
    FHIR (Fast Healthcare Interoperability Resources) standards.
    """
    patient_id = patient_data['patient_id']
    
    # Patient resource
    patient_resource = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [{
            "use": "usual",
            "system": "http://hospital.example.com/patient-ids",
            "value": patient_id
        }],
        "gender": "male" if patient_data['gender'] == 'M' else "female",
        "birthDate": str(datetime.now().year - patient_data['age']) + "-01-01",
        "extension": [{
            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
            "extension": [{
                "url": "text",
                "valueString": patient_data['race']
            }]
        }]
    }
    
    # Observation resources
    observations = []
    
    # Define LOINC codes for common observations
    observation_mappings = {
        'systolic_bp': {'code': '8480-6', 'display': 'Systolic blood pressure', 'unit': 'mmHg'},
        'diastolic_bp': {'code': '8462-4', 'display': 'Diastolic blood pressure', 'unit': 'mmHg'},
        'heart_rate': {'code': '8867-4', 'display': 'Heart rate', 'unit': '/min'},
        'temperature': {'code': '8310-5', 'display': 'Body temperature', 'unit': 'degF'},
        'glucose': {'code': '2345-7', 'display': 'Glucose [Mass/volume] in Serum or Plasma', 'unit': 'mg/dL'},
        'creatinine': {'code': '2160-0', 'display': 'Creatinine [Mass/volume] in Serum or Plasma', 'unit': 'mg/dL'},
        'hemoglobin': {'code': '718-7', 'display': 'Hemoglobin [Mass/volume] in Blood', 'unit': 'g/dL'}
    }
    
    for field, mapping in observation_mappings.items():
        if field in patient_data:
            observation = {
                "resourceType": "Observation",
                "status": "final",
                "subject": {"reference": f"Patient/{patient_id}"},
                "effectiveDateTime": datetime.now().isoformat(),
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": mapping['code'],
                        "display": mapping['display']
                    }]
                },
                "valueQuantity": {
                    "value": float(patient_data[field]),
                    "unit": mapping['unit'],
                    "system": "http://unitsofmeasure.org"
                }
            }
            observations.append(observation)
    
    # Condition resources for comorbidities
    conditions = []
    
    if patient_data.get('diabetes', 0) == 1:
        diabetes_condition = {
            "resourceType": "Condition",
            "subject": {"reference": f"Patient/{patient_id}"},
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "73211009",
                    "display": "Diabetes mellitus"
                }]
            },
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active"
                }]
            }
        }
        conditions.append(diabetes_condition)
    
    if patient_data.get('hypertension', 0) == 1:
        hypertension_condition = {
            "resourceType": "Condition",
            "subject": {"reference": f"Patient/{patient_id}"},
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "38341003",
                    "display": "Hypertensive disorder"
                }]
            },
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active"
                }]
            }
        }
        conditions.append(hypertension_condition)
    
    return {
        "patient": patient_resource,
        "observations": observations,
        "conditions": conditions
    }


def main():
    """
    Main function demonstrating the complete clinical informatics workflow.
    """
    print("Healthcare AI Implementation Guide - Chapter 1 Examples")
    print("Clinical Informatics Foundations")
    print("=" * 70)
    
    # 1. Generate synthetic clinical data
    print("\n1. Generating Synthetic Clinical Data...")
    data_generator = ClinicalDataGenerator(random_seed=42)
    clinical_data = data_generator.generate_complete_dataset(n_patients=2000)
    
    print(f"Generated dataset with {len(clinical_data)} patients")
    print(f"Readmission rate: {clinical_data['readmission_30d'].mean():.1%}")
    
    # 2. Perform data quality assessment
    print("\n2. Performing Data Quality Assessment...")
    qa_system = ClinicalDataQualityAssessment(clinical_data)
    quality_report = qa_system.generate_comprehensive_report()
    
    # 3. Train clinical decision support system
    print("\n3. Training Clinical Decision Support System...")
    cds_system = ClinicalDecisionSupportSystem()
    training_results = cds_system.train_model(clinical_data)
    
    # 4. Demonstrate individual patient risk assessment
    print("\n4. Individual Patient Risk Assessments...")
    print("-" * 50)
    
    # Select example patients
    example_patients = clinical_data.sample(3, random_state=42)
    
    for idx, (_, patient) in enumerate(example_patients.iterrows()):
        patient_df = pd.DataFrame([patient])
        risk_assessment = cds_system.predict_patient_risk(patient_df)
        
        print(f"\nPatient {idx + 1} (ID: {patient['patient_id']}):")
        print(f"  Demographics: {patient['age']}y {patient['gender']} {patient['race']}")
        print(f"  Comorbidities: DM={patient['diabetes']}, HTN={patient['hypertension']}")
        print(f"  Key Labs: Glucose={patient['glucose']:.0f}, Creatinine={patient['creatinine']:.2f}")
        print(f"  \n  Risk Assessment:")
        print(f"    Probability: {risk_assessment['risk_probability']:.1%}")
        print(f"    Category: {risk_assessment['risk_category']}")
        print(f"    Confidence: {risk_assessment['model_confidence']:.1%}")
        print(f"    Actual Outcome: {'Readmitted' if patient['readmission_30d'] else 'No Readmission'}")
        
        print(f"  \n  Top Risk Factors:")
        for factor in risk_assessment['key_risk_factors'][:3]:
            print(f"    • {factor['feature']}: {factor['value']:.2f} (importance: {factor['importance']:.3f})")
        
        print(f"  \n  Clinical Recommendations:")
        for rec in risk_assessment['recommendations'][:3]:
            print(f"    • {rec}")
        
        print("-" * 50)
    
    # 5. Create FHIR resources example
    print("\n5. FHIR Resource Creation Example...")
    sample_patient = clinical_data.iloc[0]
    fhir_resources = create_fhir_resources(sample_patient)
    
    print(f"Created FHIR resources for patient {sample_patient['patient_id']}:")
    print(f"  • Patient resource")
    print(f"  • {len(fhir_resources['observations'])} Observation resources")
    print(f"  • {len(fhir_resources['conditions'])} Condition resources")
    
    # 6. Save results
    print("\n6. Saving Results...")
    clinical_data.to_csv('clinical_data_example.csv', index=False)
    
    # Save model metrics
    with open('model_metrics.json', 'w') as f:
        json.dump(cds_system.model_metrics, f, indent=2)
    
    print("✓ Clinical data saved as 'clinical_data_example.csv'")
    print("✓ Model metrics saved as 'model_metrics.json'")
    
    print(f"\nChapter 1 examples completed successfully!")
    print("Ready to proceed to Chapter 2: Mathematical Foundations")


if __name__ == "__main__":
    main()
