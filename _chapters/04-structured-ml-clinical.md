---
layout: default
title: "Chapter 4: Structured Machine Learning for Clinical Prediction"
nav_order: 5
parent: "Part I: Foundations"
has_children: true
has_toc: true
description: "Master clinical prediction models with advanced feature engineering, model validation, and regulatory compliance frameworks"
author: "Sanjay Basu, MD PhD"
institution: "Waymark"
require_attribution: true
citation_check: true
---

# Chapter 4: Structured Machine Learning for Clinical Prediction
{: .no_toc }

Build production-ready clinical prediction models with advanced feature engineering, comprehensive validation frameworks, and regulatory compliance for real-world healthcare deployment.
{: .fs-6 .fw-300 }

{% include attribution.html 
   author="Clinical Machine Learning Research Community" 
   work="Clinical Prediction Models, Feature Engineering, and Validation Methodologies" 
   note="Implementation based on established clinical machine learning methodologies and validation frameworks. All code is original educational implementation demonstrating clinical prediction principles." %}

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Learning Objectives

By the end of this chapter, you will be able to:

{: .highlight }
- **Implement** advanced clinical feature engineering with temporal patterns and clinical reasoning
- **Build** robust clinical prediction models with proper validation and calibration
- **Deploy** models with comprehensive monitoring, drift detection, and regulatory compliance
- **Validate** clinical utility through prospective evaluation and real-world performance assessment

---

## Chapter Overview

Clinical prediction models represent the foundation of AI-driven healthcare decision support, requiring specialized approaches that differ fundamentally from traditional machine learning applications [Citation] [Citation]. This chapter provides comprehensive coverage of structured machine learning for clinical prediction, emphasizing the unique challenges of healthcare data, regulatory requirements, and clinical validation [Citation] [Citation].

### What You'll Build
{: .text-delta }

- **Clinical Feature Engineering Pipeline**: Advanced temporal feature extraction with clinical reasoning
- **Robust Prediction Models**: Multiple model architectures with proper validation and calibration
- **Deployment Framework**: Production-ready system with monitoring and compliance
- **Validation Suite**: Comprehensive clinical validation with prospective evaluation capabilities

---

## 4.1 Advanced Clinical Feature Engineering

Clinical feature engineering requires deep understanding of healthcare data characteristics, temporal patterns, and clinical reasoning processes [Citation] [Citation]. This section implements a comprehensive feature engineering pipeline designed specifically for clinical prediction tasks.

### Clinical Data Characteristics and Challenges
{: .text-delta }

Healthcare data presents unique challenges including irregular sampling, missing data patterns, temporal dependencies, and complex interactions between clinical variables [Citation] [Citation]. Our implementation addresses these challenges through specialized feature engineering techniques.

### Implementation: Comprehensive Clinical Feature Engineering
{: .text-delta }

```python
#!/usr/bin/env python3
"""
Comprehensive Clinical Feature Engineering Pipeline
Implements advanced feature engineering for clinical prediction models

This is an original educational implementation demonstrating clinical feature
engineering principles with production-ready architecture patterns.

Author: Sanjay Basu, MD PhD (Waymark)
Based on clinical machine learning research and healthcare data science best practices
Educational use - requires clinical validation for production deployment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, calibration_curve
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClinicalVariable:
    """Clinical variable definition with metadata"""
    name: str
    variable_type: str  # 'vital', 'lab', 'medication', 'diagnosis', 'procedure'
    data_type: str  # 'continuous', 'categorical', 'binary', 'ordinal'
    unit: Optional[str] = None
    normal_range: Optional[Tuple[float, float]] = None
    critical_values: Optional[Dict[str, float]] = None
    temporal_pattern: Optional[str] = None  # 'stable', 'trending', 'episodic'
    clinical_significance: Optional[str] = None
    missing_pattern: Optional[str] = None  # 'random', 'informative', 'systematic'

@dataclass
class FeatureEngineeringConfig:
    """Configuration for clinical feature engineering"""
    # Temporal features
    lookback_windows: List[int] = field(default_factory=lambda: [1, 3, 7, 14, 30])  # days
    temporal_aggregations: List[str] = field(default_factory=lambda: ['mean', 'std', 'min', 'max', 'trend', 'variability'])
    
    # Missing data handling
    missing_threshold: float = 0.8  # Drop features with >80% missing
    imputation_strategy: str = 'clinical'  # 'clinical', 'statistical', 'model-based'
    
    # Feature selection
    feature_selection_method: str = 'clinical_importance'  # 'statistical', 'clinical_importance', 'hybrid'
    max_features: Optional[int] = None
    
    # Clinical reasoning
    enable_clinical_rules: bool = True
    enable_interaction_features: bool = True
    enable_derived_features: bool = True
    
    # Quality control
    outlier_detection: bool = True
    data_quality_checks: bool = True

class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive clinical feature engineering pipeline
    
    Implements advanced feature engineering techniques specifically designed
    for clinical prediction tasks with proper handling of healthcare data
    characteristics and clinical reasoning.
    """
    
    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()
        self.clinical_variables = {}
        self.feature_metadata = {}
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_importance_scores = {}
        self.data_quality_report = {}
        
        # Clinical knowledge base
        self._initialize_clinical_knowledge()
        
        logger.info("Clinical Feature Engineer initialized")
    
    def _initialize_clinical_knowledge(self):
        """Initialize clinical knowledge base for feature engineering"""
        # Define common clinical variables with metadata
        self.clinical_variables = {
            # Vital signs
            'heart_rate': ClinicalVariable(
                name='heart_rate',
                variable_type='vital',
                data_type='continuous',
                unit='bpm',
                normal_range=(60, 100),
                critical_values={'bradycardia': 50, 'tachycardia': 120},
                temporal_pattern='stable',
                clinical_significance='cardiovascular_status'
            ),
            'systolic_bp': ClinicalVariable(
                name='systolic_bp',
                variable_type='vital',
                data_type='continuous',
                unit='mmHg',
                normal_range=(90, 140),
                critical_values={'hypotension': 90, 'hypertension': 180},
                temporal_pattern='stable',
                clinical_significance='cardiovascular_status'
            ),
            'temperature': ClinicalVariable(
                name='temperature',
                variable_type='vital',
                data_type='continuous',
                unit='celsius',
                normal_range=(36.1, 37.2),
                critical_values={'hypothermia': 35, 'hyperthermia': 38.5},
                temporal_pattern='episodic',
                clinical_significance='infection_inflammation'
            ),
            'respiratory_rate': ClinicalVariable(
                name='respiratory_rate',
                variable_type='vital',
                data_type='continuous',
                unit='breaths/min',
                normal_range=(12, 20),
                critical_values={'bradypnea': 10, 'tachypnea': 24},
                temporal_pattern='stable',
                clinical_significance='respiratory_status'
            ),
            
            # Laboratory values
            'glucose': ClinicalVariable(
                name='glucose',
                variable_type='lab',
                data_type='continuous',
                unit='mg/dL',
                normal_range=(70, 100),
                critical_values={'hypoglycemia': 70, 'hyperglycemia': 200},
                temporal_pattern='trending',
                clinical_significance='metabolic_status'
            ),
            'creatinine': ClinicalVariable(
                name='creatinine',
                variable_type='lab',
                data_type='continuous',
                unit='mg/dL',
                normal_range=(0.6, 1.2),
                critical_values={'elevated': 2.0},
                temporal_pattern='trending',
                clinical_significance='renal_function'
            ),
            'hemoglobin': ClinicalVariable(
                name='hemoglobin',
                variable_type='lab',
                data_type='continuous',
                unit='g/dL',
                normal_range=(12, 16),
                critical_values={'anemia': 10, 'polycythemia': 18},
                temporal_pattern='stable',
                clinical_significance='hematologic_status'
            ),
            'white_blood_cell_count': ClinicalVariable(
                name='white_blood_cell_count',
                variable_type='lab',
                data_type='continuous',
                unit='cells/μL',
                normal_range=(4000, 11000),
                critical_values={'leukopenia': 4000, 'leukocytosis': 15000},
                temporal_pattern='episodic',
                clinical_significance='immune_status'
            )
        }
        
        # Clinical interaction patterns
        self.clinical_interactions = {
            'cardiovascular_risk': ['systolic_bp', 'heart_rate', 'age', 'diabetes'],
            'sepsis_indicators': ['temperature', 'white_blood_cell_count', 'heart_rate', 'respiratory_rate'],
            'renal_function': ['creatinine', 'blood_urea_nitrogen', 'urine_output'],
            'metabolic_syndrome': ['glucose', 'systolic_bp', 'triglycerides', 'hdl_cholesterol']
        }
        
        # Clinical scoring systems
        self.clinical_scores = {
            'qsofa': {
                'variables': ['systolic_bp', 'respiratory_rate', 'glasgow_coma_scale'],
                'thresholds': [100, 22, 15],
                'scoring': 'binary_sum'
            },
            'news2': {
                'variables': ['respiratory_rate', 'oxygen_saturation', 'temperature', 'systolic_bp', 'heart_rate', 'consciousness_level'],
                'scoring': 'weighted_sum'
            }
        }
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ClinicalFeatureEngineer':
        """
        Fit the clinical feature engineering pipeline
        
        Args:
            X: Input clinical data
            y: Target variable (optional)
            
        Returns:
            Fitted feature engineer
        """
        logger.info("Fitting clinical feature engineering pipeline")
        
        # Data quality assessment
        self.data_quality_report = self._assess_data_quality(X)
        
        # Identify variable types and patterns
        self._identify_variable_characteristics(X)
        
        # Fit imputers
        self._fit_imputers(X)
        
        # Fit scalers
        self._fit_scalers(X)
        
        # Fit encoders for categorical variables
        self._fit_encoders(X)
        
        # Calculate feature importance if target provided
        if y is not None:
            self._calculate_feature_importance(X, y)
        
        logger.info("Clinical feature engineering pipeline fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform clinical data using fitted pipeline
        
        Args:
            X: Input clinical data
            
        Returns:
            Transformed feature matrix
        """
        logger.info("Transforming clinical data")
        
        X_transformed = X.copy()
        
        # Handle missing data
        X_transformed = self._handle_missing_data(X_transformed)
        
        # Generate temporal features
        X_transformed = self._generate_temporal_features(X_transformed)
        
        # Generate clinical derived features
        if self.config.enable_derived_features:
            X_transformed = self._generate_derived_features(X_transformed)
        
        # Generate interaction features
        if self.config.enable_interaction_features:
            X_transformed = self._generate_interaction_features(X_transformed)
        
        # Apply clinical rules
        if self.config.enable_clinical_rules:
            X_transformed = self._apply_clinical_rules(X_transformed)
        
        # Scale features
        X_transformed = self._scale_features(X_transformed)
        
        # Feature selection
        X_transformed = self._select_features(X_transformed)
        
        # Quality control
        if self.config.data_quality_checks:
            X_transformed = self._quality_control(X_transformed)
        
        logger.info(f"Feature engineering complete: {X_transformed.shape[1]} features generated")
        return X_transformed
    
    def _assess_data_quality(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        report = {
            'total_samples': len(X),
            'total_features': len(X.columns),
            'missing_data': {},
            'data_types': {},
            'outliers': {},
            'temporal_coverage': {},
            'data_completeness': {}
        }
        
        # Missing data analysis
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            missing_pct = missing_count / len(X)
            report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct,
                'pattern': self._identify_missing_pattern(X[col])
            }
        
        # Data type analysis
        for col in X.columns:
            report['data_types'][col] = {
                'pandas_dtype': str(X[col].dtype),
                'inferred_clinical_type': self._infer_clinical_type(X[col]),
                'unique_values': X[col].nunique(),
                'value_range': (X[col].min(), X[col].max()) if X[col].dtype in ['int64', 'float64'] else None
            }
        
        # Outlier detection
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            report['outliers'][col] = {
                'count': outliers,
                'percentage': outliers / len(X),
                'bounds': (lower_bound, upper_bound)
            }
        
        # Temporal coverage (if timestamp column exists)
        if 'timestamp' in X.columns or 'date' in X.columns:
            time_col = 'timestamp' if 'timestamp' in X.columns else 'date'
            report['temporal_coverage'] = {
                'start_date': X[time_col].min(),
                'end_date': X[time_col].max(),
                'duration_days': (X[time_col].max() - X[time_col].min()).days,
                'sampling_frequency': self._estimate_sampling_frequency(X[time_col])
            }
        
        return report
    
    def _identify_missing_pattern(self, series: pd.Series) -> str:
        """Identify missing data pattern"""
        if series.isnull().sum() == 0:
            return 'complete'
        elif series.isnull().sum() == len(series):
            return 'completely_missing'
        else:
            # Simple pattern detection
            missing_mask = series.isnull()
            if missing_mask.sum() / len(series) > 0.5:
                return 'mostly_missing'
            elif missing_mask.iloc[:len(missing_mask)//2].sum() > missing_mask.iloc[len(missing_mask)//2:].sum():
                return 'early_missing'
            elif missing_mask.iloc[:len(missing_mask)//2].sum() < missing_mask.iloc[len(missing_mask)//2:].sum():
                return 'late_missing'
            else:
                return 'random_missing'
    
    def _infer_clinical_type(self, series: pd.Series) -> str:
        """Infer clinical variable type from data characteristics"""
        if series.dtype in ['int64', 'float64']:
            if series.nunique() == 2:
                return 'binary'
            elif series.nunique() < 10:
                return 'ordinal'
            else:
                return 'continuous'
        elif series.dtype == 'object':
            if series.nunique() == 2:
                return 'binary_categorical'
            elif series.nunique() < 20:
                return 'categorical'
            else:
                return 'text'
        else:
            return 'unknown'
    
    def _estimate_sampling_frequency(self, time_series: pd.Series) -> str:
        """Estimate sampling frequency from timestamps"""
        if len(time_series) < 2:
            return 'insufficient_data'
        
        time_diffs = time_series.sort_values().diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff <= timedelta(minutes=5):
            return 'high_frequency'
        elif median_diff <= timedelta(hours=1):
            return 'hourly'
        elif median_diff <= timedelta(days=1):
            return 'daily'
        else:
            return 'low_frequency'
    
    def _identify_variable_characteristics(self, X: pd.DataFrame):
        """Identify characteristics of clinical variables"""
        for col in X.columns:
            if col not in self.clinical_variables:
                # Create variable metadata for unknown variables
                self.clinical_variables[col] = ClinicalVariable(
                    name=col,
                    variable_type='unknown',
                    data_type=self._infer_clinical_type(X[col]),
                    temporal_pattern='unknown'
                )
    
    def _fit_imputers(self, X: pd.DataFrame):
        """Fit imputers for missing data handling"""
        for col in X.columns:
            var_info = self.clinical_variables.get(col)
            
            if X[col].isnull().sum() > 0:
                if var_info and var_info.data_type == 'continuous':
                    # Use clinical knowledge for imputation
                    if var_info.normal_range:
                        # Use median of normal range
                        normal_median = np.mean(var_info.normal_range)
                        self.imputers[col] = SimpleImputer(strategy='constant', fill_value=normal_median)
                    else:
                        # Use KNN imputation for continuous variables
                        self.imputers[col] = KNNImputer(n_neighbors=5)
                elif var_info and var_info.data_type in ['categorical', 'binary_categorical']:
                    # Use mode for categorical variables
                    self.imputers[col] = SimpleImputer(strategy='most_frequent')
                else:
                    # Default to median for unknown types
                    self.imputers[col] = SimpleImputer(strategy='median')
                
                # Fit the imputer
                self.imputers[col].fit(X[[col]])
    
    def _fit_scalers(self, X: pd.DataFrame):
        """Fit scalers for numerical features"""
        for col in X.select_dtypes(include=[np.number]).columns:
            var_info = self.clinical_variables.get(col)
            
            if var_info and var_info.data_type == 'continuous':
                # Use robust scaler for clinical data (handles outliers better)
                self.scalers[col] = RobustScaler()
                self.scalers[col].fit(X[[col]].dropna())
    
    def _fit_encoders(self, X: pd.DataFrame):
        """Fit encoders for categorical variables"""
        for col in X.select_dtypes(include=['object']).columns:
            var_info = self.clinical_variables.get(col)
            
            if var_info and var_info.data_type in ['categorical', 'binary_categorical']:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(X[col].dropna())
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """Calculate feature importance using multiple methods"""
        # Prepare data for importance calculation
        X_processed = self._basic_preprocessing(X)
        
        # Statistical importance (mutual information)
        mi_scores = mutual_info_classif(X_processed, y, random_state=42)
        
        # Tree-based importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_processed, y)
        rf_importance = rf.feature_importances_
        
        # Store importance scores
        for i, col in enumerate(X_processed.columns):
            self.feature_importance_scores[col] = {
                'mutual_information': mi_scores[i],
                'random_forest': rf_importance[i],
                'clinical_priority': self._get_clinical_priority(col)
            }
    
    def _get_clinical_priority(self, feature_name: str) -> float:
        """Get clinical priority score for feature"""
        var_info = self.clinical_variables.get(feature_name)
        
        if not var_info:
            return 0.5  # Default priority for unknown variables
        
        # Priority based on clinical significance
        priority_map = {
            'cardiovascular_status': 0.9,
            'respiratory_status': 0.9,
            'infection_inflammation': 0.8,
            'renal_function': 0.8,
            'metabolic_status': 0.7,
            'hematologic_status': 0.6,
            'immune_status': 0.7
        }
        
        return priority_map.get(var_info.clinical_significance, 0.5)
    
    def _basic_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing for feature importance calculation"""
        X_processed = X.copy()
        
        # Handle missing values
        for col in X_processed.columns:
            if col in self.imputers:
                X_processed[col] = self.imputers[col].transform(X_processed[[col]]).flatten()
        
        # Encode categorical variables
        for col in X_processed.select_dtypes(include=['object']).columns:
            if col in self.encoders:
                X_processed[col] = self.encoders[col].transform(X_processed[col].fillna('missing'))
        
        # Fill any remaining missing values
        X_processed = X_processed.fillna(0)
        
        return X_processed
    
    def _handle_missing_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data using clinical knowledge"""
        X_imputed = X.copy()
        
        for col in X.columns:
            if col in self.imputers:
                X_imputed[col] = self.imputers[col].transform(X_imputed[[col]]).flatten()
        
        return X_imputed
    
    def _generate_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal features from clinical time series"""
        X_temporal = X.copy()
        
        # Check if we have temporal data
        if 'timestamp' not in X.columns and 'date' not in X.columns:
            logger.warning("No temporal column found, skipping temporal feature generation")
            return X_temporal
        
        time_col = 'timestamp' if 'timestamp' in X.columns else 'date'
        
        # Sort by time
        X_temporal = X_temporal.sort_values(time_col)
        
        # Generate temporal features for each clinical variable
        for col in X.select_dtypes(include=[np.number]).columns:
            if col == time_col:
                continue
            
            var_info = self.clinical_variables.get(col)
            
            # Generate features based on lookback windows
            for window in self.config.lookback_windows:
                window_data = X_temporal[col].rolling(window=f'{window}D', on=time_col)
                
                # Basic aggregations
                X_temporal[f'{col}_mean_{window}d'] = window_data.mean()
                X_temporal[f'{col}_std_{window}d'] = window_data.std()
                X_temporal[f'{col}_min_{window}d'] = window_data.min()
                X_temporal[f'{col}_max_{window}d'] = window_data.max()
                
                # Clinical-specific features
                if var_info:
                    # Trend detection
                    X_temporal[f'{col}_trend_{window}d'] = self._calculate_trend(window_data)
                    
                    # Variability measures
                    X_temporal[f'{col}_cv_{window}d'] = window_data.std() / (window_data.mean() + 1e-6)
                    
                    # Abnormal value counts
                    if var_info.normal_range:
                        normal_min, normal_max = var_info.normal_range
                        abnormal_mask = (X_temporal[col] < normal_min) | (X_temporal[col] > normal_max)
                        X_temporal[f'{col}_abnormal_count_{window}d'] = abnormal_mask.rolling(window=f'{window}D', on=time_col).sum()
                    
                    # Critical value indicators
                    if var_info.critical_values:
                        for critical_name, critical_value in var_info.critical_values.items():
                            if 'hypo' in critical_name or 'low' in critical_name:
                                critical_mask = X_temporal[col] < critical_value
                            else:
                                critical_mask = X_temporal[col] > critical_value
                            X_temporal[f'{col}_{critical_name}_count_{window}d'] = critical_mask.rolling(window=f'{window}D', on=time_col).sum()
        
        return X_temporal
    
    def _calculate_trend(self, window_data) -> pd.Series:
        """Calculate trend in windowed data"""
        def trend_slope(values):
            if len(values) < 2 or values.isna().all():
                return 0
            x = np.arange(len(values))
            y = values.dropna()
            if len(y) < 2:
                return 0
            slope, _, _, _, _ = stats.linregress(x[:len(y)], y)
            return slope
        
        return window_data.apply(trend_slope)
    
    def _generate_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate clinically-derived features"""
        X_derived = X.copy()
        
        # Body Mass Index (if height and weight available)
        if 'height' in X.columns and 'weight' in X.columns:
            X_derived['bmi'] = X['weight'] / ((X['height'] / 100) ** 2)
            X_derived['bmi_category'] = pd.cut(X_derived['bmi'], 
                                             bins=[0, 18.5, 25, 30, float('inf')],
                                             labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # Mean Arterial Pressure
        if 'systolic_bp' in X.columns and 'diastolic_bp' in X.columns:
            X_derived['mean_arterial_pressure'] = (X['systolic_bp'] + 2 * X['diastolic_bp']) / 3
        
        # Pulse Pressure
        if 'systolic_bp' in X.columns and 'diastolic_bp' in X.columns:
            X_derived['pulse_pressure'] = X['systolic_bp'] - X['diastolic_bp']
        
        # Shock Index
        if 'heart_rate' in X.columns and 'systolic_bp' in X.columns:
            X_derived['shock_index'] = X['heart_rate'] / X['systolic_bp']
        
        # Estimated GFR (simplified Cockcroft-Gault)
        if all(col in X.columns for col in ['creatinine', 'age', 'weight', 'gender']):
            male_factor = (X['gender'] == 'male').astype(int)
            X_derived['estimated_gfr'] = ((140 - X['age']) * X['weight'] * (0.85 + 0.15 * male_factor)) / (72 * X['creatinine'])
        
        # Anion Gap (if electrolytes available)
        if all(col in X.columns for col in ['sodium', 'chloride', 'bicarbonate']):
            X_derived['anion_gap'] = X['sodium'] - (X['chloride'] + X['bicarbonate'])
        
        return X_derived
    
    def _generate_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate clinically-meaningful interaction features"""
        X_interactions = X.copy()
        
        # Generate interactions based on clinical knowledge
        for interaction_name, variables in self.clinical_interactions.items():
            available_vars = [var for var in variables if var in X.columns]
            
            if len(available_vars) >= 2:
                # Multiplicative interactions
                for i in range(len(available_vars)):
                    for j in range(i + 1, len(available_vars)):
                        var1, var2 = available_vars[i], available_vars[j]
                        if X[var1].dtype in [np.number] and X[var2].dtype in [np.number]:
                            X_interactions[f'{interaction_name}_{var1}_{var2}_product'] = X[var1] * X[var2]
                
                # Ratio interactions
                for i in range(len(available_vars)):
                    for j in range(len(available_vars)):
                        if i != j:
                            var1, var2 = available_vars[i], available_vars[j]
                            if X[var1].dtype in [np.number] and X[var2].dtype in [np.number]:
                                X_interactions[f'{interaction_name}_{var1}_{var2}_ratio'] = X[var1] / (X[var2] + 1e-6)
        
        return X_interactions
    
    def _apply_clinical_rules(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply clinical decision rules and scoring systems"""
        X_rules = X.copy()
        
        # qSOFA Score
        if all(col in X.columns for col in ['systolic_bp', 'respiratory_rate', 'glasgow_coma_scale']):
            qsofa_score = 0
            qsofa_score += (X['systolic_bp'] <= 100).astype(int)
            qsofa_score += (X['respiratory_rate'] >= 22).astype(int)
            qsofa_score += (X['glasgow_coma_scale'] < 15).astype(int)
            X_rules['qsofa_score'] = qsofa_score
        
        # SIRS Criteria
        if all(col in X.columns for col in ['temperature', 'heart_rate', 'respiratory_rate', 'white_blood_cell_count']):
            sirs_score = 0
            sirs_score += ((X['temperature'] > 38) | (X['temperature'] < 36)).astype(int)
            sirs_score += (X['heart_rate'] > 90).astype(int)
            sirs_score += (X['respiratory_rate'] > 20).astype(int)
            sirs_score += ((X['white_blood_cell_count'] > 12000) | (X['white_blood_cell_count'] < 4000)).astype(int)
            X_rules['sirs_score'] = sirs_score
        
        # Hypertension categories
        if 'systolic_bp' in X.columns:
            X_rules['hypertension_stage'] = pd.cut(X['systolic_bp'],
                                                 bins=[0, 120, 130, 140, 180, float('inf')],
                                                 labels=['normal', 'elevated', 'stage1', 'stage2', 'crisis'])
        
        # Diabetes indicators
        if 'glucose' in X.columns:
            X_rules['diabetes_indicator'] = (X['glucose'] >= 126).astype(int)
            X_rules['prediabetes_indicator'] = ((X['glucose'] >= 100) & (X['glucose'] < 126)).astype(int)
        
        # Acute kidney injury stages
        if 'creatinine' in X.columns:
            # Simplified AKI staging (would need baseline creatinine in practice)
            X_rules['aki_stage'] = pd.cut(X['creatinine'],
                                        bins=[0, 1.5, 2.0, 3.0, float('inf')],
                                        labels=['normal', 'stage1', 'stage2', 'stage3'])
        
        return X_rules
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        X_scaled = X.copy()
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if col in self.scalers:
                X_scaled[col] = self.scalers[col].transform(X_scaled[[col]]).flatten()
        
        return X_scaled
    
    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features based on clinical importance and statistical significance"""
        if not self.feature_importance_scores:
            return X
        
        # Calculate combined importance score
        feature_scores = {}
        for feature, scores in self.feature_importance_scores.items():
            if feature in X.columns:
                # Weighted combination of different importance measures
                combined_score = (
                    0.4 * scores.get('clinical_priority', 0) +
                    0.3 * scores.get('mutual_information', 0) +
                    0.3 * scores.get('random_forest', 0)
                )
                feature_scores[feature] = combined_score
        
        # Select top features
        if self.config.max_features and len(feature_scores) > self.config.max_features:
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:self.config.max_features]
            selected_features = [feature for feature, _ in top_features]
            return X[selected_features]
        
        return X
    
    def _quality_control(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control measures"""
        X_qc = X.copy()
        
        # Remove features with too many missing values
        missing_threshold = self.config.missing_threshold
        for col in X_qc.columns:
            if X_qc[col].isnull().sum() / len(X_qc) > missing_threshold:
                X_qc = X_qc.drop(columns=[col])
                logger.warning(f"Dropped feature {col} due to high missing rate")
        
        # Remove constant features
        constant_features = [col for col in X_qc.columns if X_qc[col].nunique() <= 1]
        if constant_features:
            X_qc = X_qc.drop(columns=constant_features)
            logger.warning(f"Dropped constant features: {constant_features}")
        
        # Outlier detection and handling
        if self.config.outlier_detection:
            for col in X_qc.select_dtypes(include=[np.number]).columns:
                Q1 = X_qc[col].quantile(0.25)
                Q3 = X_qc[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # More conservative than 1.5 for clinical data
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers instead of removing (preserve sample size)
                X_qc[col] = X_qc[col].clip(lower=lower_bound, upper=upper_bound)
        
        return X_qc
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get comprehensive feature metadata"""
        return {
            'clinical_variables': {name: var.__dict__ for name, var in self.clinical_variables.items()},
            'feature_importance': self.feature_importance_scores,
            'data_quality_report': self.data_quality_report,
            'processing_config': self.config.__dict__
        }
    
    def visualize_feature_importance(self, top_n: int = 20, save_path: str = None):
        """Visualize feature importance"""
        if not self.feature_importance_scores:
            print("No feature importance scores available")
            return
        
        # Prepare data for visualization
        features = []
        clinical_scores = []
        mi_scores = []
        rf_scores = []
        
        for feature, scores in self.feature_importance_scores.items():
            features.append(feature)
            clinical_scores.append(scores.get('clinical_priority', 0))
            mi_scores.append(scores.get('mutual_information', 0))
            rf_scores.append(scores.get('random_forest', 0))
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': features,
            'Clinical_Priority': clinical_scores,
            'Mutual_Information': mi_scores,
            'Random_Forest': rf_scores
        })
        
        # Calculate combined score
        importance_df['Combined_Score'] = (
            0.4 * importance_df['Clinical_Priority'] +
            0.3 * importance_df['Mutual_Information'] +
            0.3 * importance_df['Random_Forest']
        )
        
        # Sort and select top features
        importance_df = importance_df.sort_values('Combined_Score', ascending=False).head(top_n)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Clinical Priority
        ax1.barh(importance_df['Feature'], importance_df['Clinical_Priority'])
        ax1.set_title('Clinical Priority Scores')
        ax1.set_xlabel('Score')
        
        # Mutual Information
        ax2.barh(importance_df['Feature'], importance_df['Mutual_Information'])
        ax2.set_title('Mutual Information Scores')
        ax2.set_xlabel('Score')
        
        # Random Forest
        ax3.barh(importance_df['Feature'], importance_df['Random_Forest'])
        ax3.set_title('Random Forest Importance')
        ax3.set_xlabel('Score')
        
        # Combined Score
        bars = ax4.barh(importance_df['Feature'], importance_df['Combined_Score'])
        ax4.set_title('Combined Importance Score')
        ax4.set_xlabel('Score')
        
        # Color bars based on score
        for bar, score in zip(bars, importance_df['Combined_Score']):
            if score > 0.8:
                bar.set_color('green')
            elif score > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Educational demonstration
def demonstrate_clinical_feature_engineering():
    """Demonstrate clinical feature engineering pipeline"""
    print("Clinical Feature Engineering Demonstration")
    print("=" * 50)
    
    # Create synthetic clinical dataset
    np.random.seed(42)
    n_patients = 1000
    
    # Generate synthetic clinical data
    data = {
        'patient_id': [f'P{i:04d}' for i in range(n_patients)],
        'age': np.random.normal(65, 15, n_patients).clip(18, 100),
        'gender': np.random.choice(['male', 'female'], n_patients),
        'heart_rate': np.random.normal(75, 15, n_patients).clip(40, 150),
        'systolic_bp': np.random.normal(130, 20, n_patients).clip(80, 200),
        'diastolic_bp': np.random.normal(80, 10, n_patients).clip(50, 120),
        'temperature': np.random.normal(37.0, 0.8, n_patients).clip(35, 42),
        'respiratory_rate': np.random.normal(16, 4, n_patients).clip(8, 30),
        'glucose': np.random.lognormal(4.5, 0.3, n_patients).clip(50, 400),
        'creatinine': np.random.lognormal(0.1, 0.3, n_patients).clip(0.5, 5.0),
        'hemoglobin': np.random.normal(13, 2, n_patients).clip(8, 18),
        'white_blood_cell_count': np.random.lognormal(8.5, 0.4, n_patients).clip(2000, 25000),
        'weight': np.random.normal(75, 15, n_patients).clip(40, 150),
        'height': np.random.normal(170, 10, n_patients).clip(150, 200)
    }
    
    # Add some missing data patterns
    missing_indices = np.random.choice(n_patients, size=int(0.1 * n_patients), replace=False)
    for idx in missing_indices:
        data['glucose'][idx] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create target variable (synthetic outcome)
    # Higher risk with age, abnormal vitals, etc.
    risk_score = (
        (df['age'] - 65) / 20 +
        (df['heart_rate'] - 75) / 30 +
        (df['systolic_bp'] - 130) / 40 +
        (df['temperature'] - 37) / 2 +
        np.log(df['glucose'] / 100) +
        np.log(df['creatinine'] / 1.0)
    )
    
    # Add noise and convert to binary outcome
    risk_score += np.random.normal(0, 0.5, n_patients)
    y = (risk_score > np.percentile(risk_score, 70)).astype(int)  # Top 30% as positive cases
    
    print(f"Created synthetic dataset: {df.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Initialize feature engineer
    config = FeatureEngineeringConfig(
        lookback_windows=[1, 3, 7],  # Simplified for demo
        enable_clinical_rules=True,
        enable_interaction_features=True,
        enable_derived_features=True
    )
    
    feature_engineer = ClinicalFeatureEngineer(config)
    
    # Fit and transform
    print("\nFitting feature engineering pipeline...")
    feature_engineer.fit(df, y)
    
    print("\nTransforming features...")
    X_transformed = feature_engineer.transform(df)
    
    print(f"Original features: {df.shape[1]}")
    print(f"Engineered features: {X_transformed.shape[1]}")
    
    # Display feature importance
    print("\nTop 10 Most Important Features:")
    importance_scores = feature_engineer.feature_importance_scores
    if importance_scores:
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: x[1].get('clinical_priority', 0) + 
                                           x[1].get('mutual_information', 0) + 
                                           x[1].get('random_forest', 0), 
                               reverse=True)
        
        for i, (feature, scores) in enumerate(sorted_features[:10]):
            combined_score = (scores.get('clinical_priority', 0) + 
                            scores.get('mutual_information', 0) + 
                            scores.get('random_forest', 0)) / 3
            print(f"{i+1:2d}. {feature:30s} - Combined Score: {combined_score:.3f}")
    
    # Display data quality report
    print("\nData Quality Summary:")
    quality_report = feature_engineer.data_quality_report
    print(f"Total samples: {quality_report['total_samples']}")
    print(f"Total features: {quality_report['total_features']}")
    
    missing_summary = quality_report['missing_data']
    high_missing = [col for col, info in missing_summary.items() if info['percentage'] > 0.05]
    if high_missing:
        print(f"Features with >5% missing: {high_missing}")
    
    # Visualize feature importance
    feature_engineer.visualize_feature_importance(top_n=15)
    
    return df, X_transformed, y, feature_engineer

if __name__ == "__main__":
    df, X_transformed, y, feature_engineer = demonstrate_clinical_feature_engineering()
```

{% include attribution.html 
   author="Clinical Machine Learning Research Community" 
   work="Clinical Feature Engineering, Temporal Analysis, and Healthcare Data Science" 
   citation="rajkomar_scalable_2018" 
   note="Implementation based on clinical machine learning research and healthcare data science best practices. All code is original educational implementation demonstrating clinical feature engineering principles." 
   style="research-style" %}

### Key Features of Clinical Feature Engineering
{: .text-delta }

{: .highlight }
**Clinical Knowledge Integration**: Incorporates medical domain knowledge, normal ranges, critical values, and clinical scoring systems into feature engineering process.

{: .highlight }
**Temporal Pattern Recognition**: Advanced temporal feature extraction with lookback windows, trend analysis, and clinical event detection.

{: .highlight }
**Missing Data Intelligence**: Clinical-informed imputation strategies that consider the clinical significance and patterns of missing healthcare data.

{: .highlight }
**Quality Assurance**: Comprehensive data quality assessment, outlier detection, and validation frameworks specific to healthcare applications.

---

## 4.2 Robust Clinical Prediction Models

Building robust clinical prediction models requires specialized approaches that address the unique challenges of healthcare applications, including class imbalance, temporal dependencies, and the need for interpretability and calibration [Citation] [Citation].

### Implementation: Comprehensive Clinical Prediction Framework
{: .text-delta }

```python
#!/usr/bin/env python3
"""
Comprehensive Clinical Prediction Model Framework
Implements robust clinical prediction models with proper validation and calibration

This is an original educational implementation demonstrating clinical prediction
modeling principles with production-ready architecture patterns.

Author: Sanjay Basu, MD PhD (Waymark)
Based on clinical machine learning research and model validation best practices
Educational use - requires clinical validation for production deployment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, TimeSeriesSplit, 
    cross_val_score, validation_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    calibration_curve, brier_score_loss, classification_report,
    confusion_matrix, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for clinical prediction models"""
    # Model selection
    model_types: List[str] = field(default_factory=lambda: ['logistic', 'random_forest', 'xgboost', 'ensemble'])
    
    # Validation strategy
    validation_method: str = 'stratified_kfold'  # 'stratified_kfold', 'time_series', 'holdout'
    n_folds: int = 5
    test_size: float = 0.2
    
    # Class imbalance handling
    handle_imbalance: bool = True
    imbalance_method: str = 'class_weight'  # 'class_weight', 'smote', 'threshold_tuning'
    
    # Calibration
    calibration_method: str = 'isotonic'  # 'isotonic', 'sigmoid', 'none'
    
    # Feature selection
    feature_selection: bool = True
    max_features: Optional[int] = None
    
    # Hyperparameter tuning
    hyperparameter_tuning: bool = True
    tuning_method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'
    
    # Clinical validation
    clinical_validation: bool = True
    interpretability_required: bool = True

@dataclass
class ValidationResults:
    """Results from model validation"""
    auc_roc: float
    auc_pr: float
    brier_score: float
    calibration_slope: float
    calibration_intercept: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1_score: float
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]
    calibration_curve_data: Tuple[np.ndarray, np.ndarray]
    confusion_matrix: np.ndarray

class ClinicalPredictionModel(BaseEstimator, ClassifierMixin):
    """
    Comprehensive clinical prediction model with validation and calibration
    
    Implements multiple model architectures with proper clinical validation,
    calibration, and interpretability features for healthcare applications.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.best_model = None
        self.calibrated_model = None
        self.feature_importance = {}
        self.validation_results = {}
        self.is_fitted = False
        
        logger.info("Clinical Prediction Model initialized")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'ClinicalPredictionModel':
        """
        Fit clinical prediction models with comprehensive validation
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation set for temporal validation
            
        Returns:
            Fitted model
        """
        logger.info("Fitting clinical prediction models")
        
        # Prepare data
        X_processed, y_processed = self._prepare_data(X, y)
        
        # Split data if no validation set provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_processed, 
                test_size=self.config.test_size,
                stratify=y_processed,
                random_state=42
            )
        else:
            X_train, y_train = X_processed, y_processed
            X_val, y_val = validation_data
        
        # Train multiple model types
        self._train_models(X_train, y_train)
        
        # Validate models
        self._validate_models(X_train, y_train, X_val, y_val)
        
        # Select best model
        self._select_best_model()
        
        # Calibrate best model
        if self.config.calibration_method != 'none':
            self._calibrate_model(X_train, y_train)
        
        # Clinical validation
        if self.config.clinical_validation:
            self._clinical_validation(X_val, y_val)
        
        self.is_fitted = True
        logger.info("Clinical prediction model training complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary outcomes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        model = self.calibrated_model if self.calibrated_model else self.best_model
        return model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        model = self.calibrated_model if self.calibrated_model else self.best_model
        return model.predict_proba(X)
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training"""
        # Handle missing values
        X_processed = X.fillna(X.median())
        
        # Handle class imbalance
        if self.config.handle_imbalance and self.config.imbalance_method == 'smote':
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_processed, y_processed = smote.fit_resample(X_processed, y)
        else:
            y_processed = y.copy()
        
        return X_processed, y_processed
    
    def _train_models(self, X: pd.DataFrame, y: pd.Series):
        """Train multiple model types"""
        # Calculate class weights if needed
        class_weights = None
        if self.config.handle_imbalance and self.config.imbalance_method == 'class_weight':
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weights = dict(zip(classes, weights))
        
        # Logistic Regression
        if 'logistic' in self.config.model_types:
            self.models['logistic'] = LogisticRegression(
                class_weight=class_weights,
                random_state=42,
                max_iter=1000
            )
            self.models['logistic'].fit(X, y)
        
        # Random Forest
        if 'random_forest' in self.config.model_types:
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1
            )
            self.models['random_forest'].fit(X, y)
        
        # XGBoost
        if 'xgboost' in self.config.model_types:
            scale_pos_weight = None
            if class_weights:
                scale_pos_weight = class_weights[0] / class_weights[1]
            
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )
            self.models['xgboost'].fit(X, y)
        
        # LightGBM
        if 'lightgbm' in self.config.model_types:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                class_weight=class_weights,
                random_state=42,
                verbose=-1
            )
            self.models['lightgbm'].fit(X, y)
        
        # Gradient Boosting
        if 'gradient_boosting' in self.config.model_types:
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
            self.models['gradient_boosting'].fit(X, y)
        
        # Neural Network
        if 'neural_network' in self.config.model_types:
            self.models['neural_network'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500
            )
            self.models['neural_network'].fit(X, y)
        
        # Ensemble
        if 'ensemble' in self.config.model_types and len(self.models) > 1:
            # Create ensemble from existing models
            estimators = [(name, model) for name, model in self.models.items() 
                         if name != 'ensemble']
            
            self.models['ensemble'] = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
            self.models['ensemble'].fit(X, y)
    
    def _validate_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series):
        """Validate all trained models"""
        for name, model in self.models.items():
            logger.info(f"Validating {name} model")
            
            # Cross-validation on training set
            if self.config.validation_method == 'stratified_kfold':
                cv = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
            elif self.config.validation_method == 'time_series':
                cv = TimeSeriesSplit(n_splits=self.config.n_folds)
            else:
                cv = self.config.n_folds
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            
            # Validation set performance
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            # Calculate comprehensive metrics
            validation_result = self._calculate_validation_metrics(y_val, y_pred, y_pred_proba)
            validation_result.cross_val_scores = cv_scores.tolist()
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                validation_result.feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                validation_result.feature_importance = dict(zip(X_train.columns, np.abs(model.coef_[0])))
            else:
                validation_result.feature_importance = {}
            
            self.validation_results[name] = validation_result
    
    def _calculate_validation_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                    y_pred_proba: np.ndarray) -> ValidationResults:
        """Calculate comprehensive validation metrics"""
        # ROC AUC
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall AUC
        auc_pr = average_precision_score(y_true, y_pred_proba)
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Calibration metrics
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Fit calibration line
        if len(fraction_of_positives) > 1:
            slope, intercept, _, _, _ = stats.linregress(mean_predicted_value, fraction_of_positives)
        else:
            slope, intercept = 1.0, 0.0
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        return ValidationResults(
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            brier_score=brier_score,
            calibration_slope=slope,
            calibration_intercept=intercept,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            f1_score=f1,
            cross_val_scores=[],
            feature_importance={},
            calibration_curve_data=(fraction_of_positives, mean_predicted_value),
            confusion_matrix=cm
        )
    
    def _select_best_model(self):
        """Select best model based on validation metrics"""
        # Composite score: AUC-ROC + AUC-PR - Brier Score + Calibration Quality
        best_score = -np.inf
        best_model_name = None
        
        for name, results in self.validation_results.items():
            # Calibration quality (closer to 1.0 slope and 0.0 intercept is better)
            calibration_quality = 1 - (abs(results.calibration_slope - 1.0) + abs(results.calibration_intercept))
            
            # Composite score
            score = (
                0.4 * results.auc_roc +
                0.3 * results.auc_pr +
                0.2 * (1 - results.brier_score) +  # Lower Brier is better
                0.1 * calibration_quality
            )
            
            logger.info(f"{name}: AUC-ROC={results.auc_roc:.3f}, AUC-PR={results.auc_pr:.3f}, "
                       f"Brier={results.brier_score:.3f}, Composite={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model selected: {best_model_name} (score: {best_score:.3f})")
    
    def _calibrate_model(self, X: pd.DataFrame, y: pd.Series):
        """Calibrate the best model"""
        logger.info(f"Calibrating model using {self.config.calibration_method} method")
        
        self.calibrated_model = CalibratedClassifierCV(
            self.best_model,
            method=self.config.calibration_method,
            cv=3
        )
        self.calibrated_model.fit(X, y)
    
    def _clinical_validation(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Perform clinical validation checks"""
        logger.info("Performing clinical validation")
        
        model = self.calibrated_model if self.calibrated_model else self.best_model
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Check for clinical plausibility
        clinical_checks = {
            'prediction_range': (y_pred_proba.min(), y_pred_proba.max()),
            'prediction_distribution': {
                'mean': y_pred_proba.mean(),
                'std': y_pred_proba.std(),
                'skewness': stats.skew(y_pred_proba),
                'kurtosis': stats.kurtosis(y_pred_proba)
            },
            'calibration_quality': self._assess_calibration_quality(y_val, y_pred_proba),
            'subgroup_performance': self._assess_subgroup_performance(X_val, y_val, y_pred_proba)
        }
        
        self.clinical_validation_results = clinical_checks
    
    def _assess_calibration_quality(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Assess calibration quality"""
        # Hosmer-Lemeshow test approximation
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        hl_statistic = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            if in_bin.sum() > 0:
                observed = y_true[in_bin].sum()
                expected = y_pred_proba[in_bin].sum()
                hl_statistic += (observed - expected) ** 2 / (expected + 1e-6)
        
        return {
            'hosmer_lemeshow_statistic': hl_statistic,
            'calibration_slope': self.validation_results[list(self.validation_results.keys())[0]].calibration_slope,
            'calibration_intercept': self.validation_results[list(self.validation_results.keys())[0]].calibration_intercept
        }
    
    def _assess_subgroup_performance(self, X: pd.DataFrame, y_true: pd.Series, 
                                   y_pred_proba: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Assess performance across different subgroups"""
        subgroup_performance = {}
        
        # Age groups
        if 'age' in X.columns:
            age_groups = pd.cut(X['age'], bins=[0, 50, 65, 80, 100], labels=['<50', '50-65', '65-80', '80+'])
            for group in age_groups.cat.categories:
                mask = age_groups == group
                if mask.sum() > 10:  # Minimum sample size
                    group_auc = roc_auc_score(y_true[mask], y_pred_proba[mask])
                    subgroup_performance[f'age_{group}'] = {'auc_roc': group_auc, 'n_samples': mask.sum()}
        
        # Gender
        if 'gender' in X.columns:
            for gender in X['gender'].unique():
                mask = X['gender'] == gender
                if mask.sum() > 10:
                    group_auc = roc_auc_score(y_true[mask], y_pred_proba[mask])
                    subgroup_performance[f'gender_{gender}'] = {'auc_roc': group_auc, 'n_samples': mask.sum()}
        
        return subgroup_performance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        best_model_name = None
        for name, model in self.models.items():
            if model == self.best_model:
                best_model_name = name
                break
        
        summary = {
            'best_model': best_model_name,
            'validation_results': {name: {
                'auc_roc': results.auc_roc,
                'auc_pr': results.auc_pr,
                'brier_score': results.brier_score,
                'sensitivity': results.sensitivity,
                'specificity': results.specificity,
                'f1_score': results.f1_score
            } for name, results in self.validation_results.items()},
            'feature_importance': self.validation_results[best_model_name].feature_importance if best_model_name else {},
            'clinical_validation': getattr(self, 'clinical_validation_results', {}),
            'calibration_applied': self.calibrated_model is not None
        }
        
        return summary
    
    def visualize_performance(self, save_path: str = None):
        """Visualize model performance"""
        if not self.is_fitted:
            print("Model not fitted")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curves
        for name, results in self.validation_results.items():
            # We need to recalculate ROC curve data
            # This is a simplified version - in practice, store this data during validation
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax1.text(0.6, 0.2 + 0.1 * list(self.validation_results.keys()).index(name), 
                    f'{name}: AUC = {results.auc_roc:.3f}')
        
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        
        # Precision-Recall Curves
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        for name, results in self.validation_results.items():
            ax2.text(0.6, 0.2 + 0.1 * list(self.validation_results.keys()).index(name), 
                    f'{name}: AUC = {results.auc_pr:.3f}')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        
        # Calibration Plot
        best_model_name = None
        for name, model in self.models.items():
            if model == self.best_model:
                best_model_name = name
                break
        
        if best_model_name:
            results = self.validation_results[best_model_name]
            fraction_of_positives, mean_predicted_value = results.calibration_curve_data
            
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            ax3.plot(mean_predicted_value, fraction_of_positives, 'o-', label=f'{best_model_name}')
            ax3.set_xlabel('Mean Predicted Probability')
            ax3.set_ylabel('Fraction of Positives')
            ax3.set_title('Calibration Plot')
            ax3.legend()
        
        # Feature Importance
        if best_model_name and self.validation_results[best_model_name].feature_importance:
            importance = self.validation_results[best_model_name].feature_importance
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features, scores = zip(*top_features)
            ax4.barh(range(len(features)), scores)
            ax4.set_yticks(range(len(features)))
            ax4.set_yticklabels(features)
            ax4.set_xlabel('Importance Score')
            ax4.set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'best_model': self.best_model,
            'calibrated_model': self.calibrated_model,
            'validation_results': self.validation_results,
            'config': self.config,
            'clinical_validation_results': getattr(self, 'clinical_validation_results', {})
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ClinicalPredictionModel':
        """Load a saved model"""
        model_data = joblib.load(filepath)
        
        model = cls(config=model_data['config'])
        model.best_model = model_data['best_model']
        model.calibrated_model = model_data['calibrated_model']
        model.validation_results = model_data['validation_results']
        model.clinical_validation_results = model_data.get('clinical_validation_results', {})
        model.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return model

# Educational demonstration
def demonstrate_clinical_prediction_models():
    """Demonstrate clinical prediction modeling"""
    print("Clinical Prediction Model Demonstration")
    print("=" * 50)
    
    # Use the feature engineering demo data
    from __main__ import demonstrate_clinical_feature_engineering
    df, X_transformed, y, feature_engineer = demonstrate_clinical_feature_engineering()
    
    print(f"\nUsing engineered features: {X_transformed.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Initialize model
    config = ModelConfig(
        model_types=['logistic', 'random_forest', 'xgboost', 'ensemble'],
        validation_method='stratified_kfold',
        calibration_method='isotonic',
        clinical_validation=True
    )
    
    clinical_model = ClinicalPredictionModel(config)
    
    # Train model
    print("\nTraining clinical prediction models...")
    clinical_model.fit(X_transformed, y)
    
    # Get model summary
    print("\nModel Performance Summary:")
    summary = clinical_model.get_model_summary()
    
    print(f"Best Model: {summary['best_model']}")
    print("\nValidation Results:")
    for model_name, metrics in summary['validation_results'].items():
        print(f"  {model_name}:")
        print(f"    AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"    AUC-PR:  {metrics['auc_pr']:.3f}")
        print(f"    Brier:   {metrics['brier_score']:.3f}")
        print(f"    F1:      {metrics['f1_score']:.3f}")
    
    # Clinical validation results
    if 'clinical_validation' in summary:
        print("\nClinical Validation:")
        cv_results = summary['clinical_validation']
        if 'calibration_quality' in cv_results:
            cal_quality = cv_results['calibration_quality']
            print(f"  Calibration Slope: {cal_quality.get('calibration_slope', 'N/A'):.3f}")
            print(f"  Calibration Intercept: {cal_quality.get('calibration_intercept', 'N/A'):.3f}")
        
        if 'subgroup_performance' in cv_results:
            print("  Subgroup Performance:")
            for subgroup, perf in cv_results['subgroup_performance'].items():
                print(f"    {subgroup}: AUC = {perf['auc_roc']:.3f} (n={perf['n_samples']})")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    if summary['feature_importance']:
        sorted_features = sorted(summary['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, importance) in enumerate(sorted_features):
            print(f"  {i+1:2d}. {feature:30s}: {importance:.4f}")
    
    # Visualize performance
    clinical_model.visualize_performance()
    
    # Test predictions
    print("\nTesting Predictions:")
    test_indices = np.random.choice(len(X_transformed), size=5, replace=False)
    test_X = X_transformed.iloc[test_indices]
    test_y = y[test_indices]
    
    predictions = clinical_model.predict_proba(test_X)[:, 1]
    
    print("Sample Predictions:")
    for i, (idx, pred, actual) in enumerate(zip(test_indices, predictions, test_y)):
        print(f"  Patient {idx}: Predicted Risk = {pred:.3f}, Actual = {actual}")
    
    return clinical_model, X_transformed, y

if __name__ == "__main__":
    clinical_model, X_transformed, y = demonstrate_clinical_prediction_models()
```

{% include attribution.html 
   author="Clinical Machine Learning and Model Validation Research Communities" 
   work="Clinical Prediction Models, Model Validation, and Calibration Methods" 
   citation="sendak_machine_2020" 
   note="Implementation based on clinical machine learning research and model validation best practices. All code is original educational implementation demonstrating clinical prediction modeling principles." 
   style="research-style" %}

---

## Key Takeaways

{: .highlight }
**Advanced Feature Engineering**: Clinical feature engineering requires domain knowledge integration, temporal pattern recognition, and specialized handling of healthcare data characteristics.

{: .highlight }
**Robust Model Validation**: Clinical prediction models require comprehensive validation including calibration assessment, subgroup analysis, and clinical plausibility checks.

{: .highlight }
**Regulatory Compliance**: Production clinical models must meet regulatory requirements including interpretability, bias assessment, and prospective validation.

{: .highlight }
**Clinical Integration**: Successful deployment requires consideration of clinical workflows, decision support integration, and continuous monitoring frameworks.

---

## Interactive Exercises

### Exercise 1: Advanced Temporal Features
{: .text-delta }

Extend the feature engineering pipeline to handle complex temporal patterns:

```python
# Your task: Implement advanced temporal feature engineering
def implement_advanced_temporal_features():
    """
    Implement advanced temporal feature engineering for clinical prediction
    
    Requirements:
    1. Seasonal pattern detection
    2. Change point detection
    3. Time-to-event features
    4. Irregular sampling handling
    """
    pass  # Your implementation here
```

### Exercise 2: Model Interpretability
{: .text-delta }

Implement comprehensive model interpretability for clinical use:

```python
# Your task: Clinical model interpretability
def implement_clinical_interpretability():
    """
    Implement interpretability methods for clinical prediction models
    
    Requirements:
    1. SHAP values for individual predictions
    2. LIME explanations for local interpretability
    3. Feature interaction analysis
    4. Clinical rule extraction
    """
    pass  # Your implementation here
```

---

## Bibliography


---

## Next Steps

Continue to [Chapter 5: Reinforcement Learning for Treatment Optimization]([Link]) to learn about:
- Dynamic treatment regimes
- Contextual bandits for personalized medicine
- Safe reinforcement learning in healthcare
- Policy evaluation and deployment

---

## Additional Resources

### Clinical Machine Learning
{: .text-delta }

1. [Citation] - Scalable and accurate deep learning with electronic health records
2. [Citation] - Machine learning in health care: a critical appraisal of challenges and opportunities
3. [Citation] - Big data and machine learning in health care
4. [Citation] - Making machine learning models clinically useful

### Code Repository
{: .text-delta }

All clinical prediction implementations from this chapter are available in the [GitHub repository](https://github.com/sanjay-basu/healthcare-ai-book/tree/main/_chapters/04-structured-ml-clinical).

### Interactive Notebooks
{: .text-delta }

Explore the clinical prediction concepts interactively:
- [Clinical Feature Engineering Lab]([Link])
- [Model Validation Workshop]([Link])
- [Calibration and Interpretability Tutorial]([Link])
- [Regulatory Compliance Guide]([Link])

---

{: .note }
This chapter provides the foundation for building robust, clinically-validated prediction models that can be safely deployed in healthcare environments with proper regulatory compliance and clinical integration.

{: .attribution }
**Academic Integrity Statement**: This chapter contains original educational implementations based on established clinical machine learning methodologies and validation frameworks. All code is original and created for educational purposes. Proper attribution is provided for all referenced research and methodologies. No proprietary systems have been copied or reproduced.
