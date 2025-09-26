"""
Chapter 4 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

"""
Comprehensive Clinical Temporal Feature Engineering System

This implementation provides advanced temporal feature engineering capabilities
specifically designed for clinical machine learning applications, including
trend analysis, pattern recognition, and clinical knowledge integration.

Author: Sanjay Basu MD PhD
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
import logging

\# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClinicalNormalRanges:
    """Clinical normal ranges for common laboratory and vital sign values"""
    
    \# Laboratory values (standard units)
    hemoglobin: Tuple[float, float] = (12.0, 16.0)  \# g/dL
    hematocrit: Tuple[float, float] = (36.0, 48.0)  \# %
    white_blood_cell_count: Tuple[float, float] = (4.0, 11.0)  \# K/uL
    platelet_count: Tuple[float, float] = (150, 450)  \# K/uL
    sodium: Tuple[float, float] = (136, 145)  \# mEq/L
    potassium: Tuple[float, float] = (3.5, 5.0)  \# mEq/L
    chloride: Tuple[float, float] = (98, 107)  \# mEq/L
    co2: Tuple[float, float] = (22, 28)  \# mEq/L
    bun: Tuple[float, float] = (7, 20)  \# mg/dL
    creatinine: Tuple[float, float] = (0.6, 1.2)  \# mg/dL
    glucose: Tuple[float, float] = (70, 100)  \# mg/dL
    albumin: Tuple[float, float] = (3.5, 5.0)  \# g/dL
    total_bilirubin: Tuple[float, float] = (0.2, 1.2)  \# mg/dL
    alt: Tuple[float, float] = (7, 56)  \# U/L
    ast: Tuple[float, float] = (10, 40)  \# U/L
    
    \# Vital signs
    systolic_bp: Tuple[float, float] = (90, 140)  \# mmHg
    diastolic_bp: Tuple[float, float] = (60, 90)  \# mmHg
    heart_rate: Tuple[float, float] = (60, 100)  \# bpm
    temperature: Tuple[float, float] = (97.0, 99.5)  \# F
    respiratory_rate: Tuple[float, float] = (12, 20)  \# breaths/min
    oxygen_saturation: Tuple[float, float] = (95, 100)  \# %
    
    def get_normal_range(self, variable_name: str) -> Optional[Tuple[float, float]]:
        """Get normal range for a clinical variable"""
        return getattr(self, variable_name.lower().replace(' ', '_'), None)
    
    def is_abnormal(self, variable_name: str, value: float) -> Optional[str]:
        """Determine if a value is abnormal and return direction"""
        normal_range = self.get_normal_range(variable_name)
        if normal_range is None:
            return None
        
        if value < normal_range<sup>0</sup>:
            return 'low'
        elif value > normal_range<sup>1</sup>:
            return 'high'
        else:
            return 'normal'

class ClinicalTemporalFeatureEngineer:
    """
    Comprehensive temporal feature engineering for clinical data.
    
    This class implements state-of-the-art techniques for processing
    time-series clinical data, including trend analysis, statistical
    aggregations, temporal pattern recognition, and clinical knowledge integration.
    
    The implementation is based on established clinical informatics principles
    and incorporates domain knowledge about clinical data patterns and
    temporal relationships in healthcare settings.
    """
    
    def __init__(self, 
                 window_hours: int = 24,
                 min_measurements: int = 3,
                 clinical_ranges: Optional[ClinicalNormalRanges] = None):
        """
        Initialize the clinical temporal feature engineer.
        
        Args:
            window_hours: Time window for feature extraction (hours)
            min_measurements: Minimum number of measurements required
            clinical_ranges: Clinical normal ranges for abnormality detection
        """
        self.window_hours = window_hours
        self.min_measurements = min_measurements
        self.clinical_ranges = clinical_ranges or ClinicalNormalRanges()
        self.scalers = {}
        self.feature_names = []
        
        logger.info(f"Initialized clinical temporal feature engineer with {window_hours}h window")
    
    def extract_temporal_features(self, 
                                df: pd.DataFrame,
                                patient_id_col: str = 'patient_id',
                                timestamp_col: str = 'timestamp',
                                value_col: str = 'value',
                                variable_col: str = 'variable') -> pd.DataFrame:
        """
        Extract comprehensive temporal features from clinical time series data.
        
        This method processes clinical time series data to extract a rich set of
        temporal features including statistical aggregations, trend analysis,
        pattern recognition, and clinical knowledge-based features.
        
        Args:
            df: DataFrame with columns [patient_id, timestamp, variable, value]
            patient_id_col: Name of patient identifier column
            timestamp_col: Name of timestamp column
            value_col: Name of value column
            variable_col: Name of variable/measurement type column
            
        Returns:
            DataFrame with engineered temporal features for each patient
        """
        
        \# Validate input data
        required_cols = [patient_id_col, timestamp_col, value_col, variable_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        \# Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        \# Sort by patient and timestamp
        df = df.sort_values([patient_id_col, timestamp_col])
        
        logger.info(f"Processing {len(df)} measurements for {df[patient_id_col].nunique()} patients")
        
        features_list = []
        
        for patient_id in df[patient_id_col].unique():
            patient_data = df[df[patient_id_col] == patient_id].copy()
            
            try:
                patient_features = self._extract_patient_features(
                    patient_data, timestamp_col, value_col, variable_col
                )
                patient_features[patient_id_col] = patient_id
                features_list.append(patient_features)
                
            except Exception as e:
                logger.warning(f"Error processing patient {patient_id}: {e}")
                continue
        
        result_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(result_df.columns)} features for {len(result_df)} patients")
        
        return result_df
    
    def _extract_patient_features(self, 
                                patient_data: pd.DataFrame,
                                timestamp_col: str,
                                value_col: str,
                                variable_col: str) -> Dict[str, Any]:
        """Extract comprehensive features for a single patient."""
        
        features = {}
        
        \# Group by variable type
        for variable in patient_data[variable_col].unique():
            var_data = patient_data[patient_data[variable_col] == variable].copy()
            
            if len(var_data) < self.min_measurements:
                continue
            
            \# Extract values and timestamps
            values = var_data[value_col].values
            timestamps = var_data[timestamp_col].values
            
            \# Skip if all values are NaN
            if np.all(np.isnan(values)):
                continue
            
            \# Remove NaN values for calculations
            valid_mask = ~np.isnan(values)
            valid_values = values[valid_mask]
            valid_timestamps = timestamps[valid_mask]
            
            if len(valid_values) < self.min_measurements:
                continue
            
            \# Extract comprehensive feature set
            var_features = {}
            
            \# Basic statistical features
            var_features.update(self._extract_statistical_features(variable, valid_values))
            
            \# Temporal trend features
            var_features.update(self._extract_trend_features(variable, valid_values, valid_timestamps))
            
            \# Clinical knowledge-based features
            var_features.update(self._extract_clinical_features(variable, valid_values))
            
            \# Pattern recognition features
            var_features.update(self._extract_pattern_features(variable, valid_values, valid_timestamps))
            
            \# Time-based features
            var_features.update(self._extract_time_features(variable, valid_timestamps))
            
            \# Add to overall features
            features.update(var_features)
        
        return features
    
    def _extract_statistical_features(self, variable: str, values: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features."""
        
        features = {}
        
        \# Central tendency
        features[f'{variable}_mean'] = np.mean(values)
        features[f'{variable}_median'] = np.median(values)
        features[f'{variable}_mode'] = stats.mode(values, keepdims=True)<sup>0</sup><sup>0</sup> if len(values) > 1 else values<sup>0</sup>
        
        \# Dispersion
        features[f'{variable}_std'] = np.std(values)
        features[f'{variable}_var'] = np.var(values)
        features[f'{variable}_range'] = np.max(values) - np.min(values)
        features[f'{variable}_iqr'] = np.percentile(values, 75) - np.percentile(values, 25)
        features[f'{variable}_cv'] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        
        \# Extremes
        features[f'{variable}_min'] = np.min(values)
        features[f'{variable}_max'] = np.max(values)
        features[f'{variable}_q25'] = np.percentile(values, 25)
        features[f'{variable}_q75'] = np.percentile(values, 75)
        features[f'{variable}_q10'] = np.percentile(values, 10)
        features[f'{variable}_q90'] = np.percentile(values, 90)
        
        \# Distribution shape
        features[f'{variable}_skewness'] = stats.skew(values)
        features[f'{variable}_kurtosis'] = stats.kurtosis(values)
        
        \# Count features
        features[f'{variable}_count'] = len(values)
        features[f'{variable}_unique_count'] = len(np.unique(values))
        
        return features
    
    def _extract_trend_features(self, variable: str, values: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
        """Extract temporal trend features."""
        
        features = {}
        
        if len(values) < 3:
            return features
        
        \# Convert timestamps to numeric (hours from first measurement)
        time_numeric = np.array([(t - timestamps<sup>0</sup>).total_seconds() / 3600 for t in timestamps])
        
        \# Linear trend analysis
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
            features[f'{variable}_trend_slope'] = slope
            features[f'{variable}_trend_intercept'] = intercept
            features[f'{variable}_trend_r2'] = r_value ** 2
            features[f'{variable}_trend_p_value'] = p_value
            features[f'{variable}_trend_std_err'] = std_err
            
            \# Trend direction and significance
            features[f'{variable}_trend_direction'] = 1 if slope > 0 else -1 if slope < 0 else 0
            features[f'{variable}_trend_significant'] = 1 if p_value < 0.05 else 0
            
        except Exception as e:
            logger.warning(f"Error calculating trend for {variable}: {e}")
        
        \# Rate of change features
        if len(values) >= 2:
            changes = np.diff(values)
            time_diffs = np.diff(time_numeric)
            rates = changes / time_diffs
            
            features[f'{variable}_mean_change'] = np.mean(changes)
            features[f'{variable}_std_change'] = np.std(changes)
            features[f'{variable}_max_increase'] = np.max(changes)
            features[f'{variable}_max_decrease'] = np.min(changes)
            features[f'{variable}_mean_rate'] = np.mean(rates)
            features[f'{variable}_std_rate'] = np.std(rates)
            features[f'{variable}_max_rate_increase'] = np.max(rates)
            features[f'{variable}_max_rate_decrease'] = np.min(rates)
            
            \# Volatility measures
            features[f'{variable}_volatility'] = np.std(changes) / np.mean(np.abs(values)) if np.mean(np.abs(values)) > 0 else 0
            features[f'{variable}_rate_volatility'] = np.std(rates) / np.mean(np.abs(rates)) if np.mean(np.abs(rates)) > 0 else 0
        
        \# Polynomial trend features (quadratic)
        if len(values) >= 4:
            try:
                poly_coeffs = np.polyfit(time_numeric, values, 2)
                features[f'{variable}_quadratic_a'] = poly_coeffs<sup>0</sup>
                features[f'{variable}_quadratic_b'] = poly_coeffs<sup>1</sup>
                features[f'{variable}_quadratic_c'] = poly_coeffs<sup>2</sup>
                
                \# Curvature indicator
                features[f'{variable}_curvature'] = 2 * poly_coeffs<sup>0</sup>
                
            except Exception as e:
                logger.warning(f"Error calculating polynomial trend for {variable}: {e}")
        
        return features
    
    def _extract_clinical_features(self, variable: str, values: np.ndarray) -> Dict[str, float]:
        """Extract clinical knowledge-based features."""
        
        features = {}
        
        \# Clinical abnormality features
        normal_range = self.clinical_ranges.get_normal_range(variable)
        if normal_range:
            low_threshold, high_threshold = normal_range
            
            \# Abnormality counts and proportions
            low_count = np.sum(values < low_threshold)
            high_count = np.sum(values > high_threshold)
            normal_count = len(values) - low_count - high_count
            
            features[f'{variable}_low_count'] = low_count
            features[f'{variable}_high_count'] = high_count
            features[f'{variable}_normal_count'] = normal_count
            features[f'{variable}_low_proportion'] = low_count / len(values)
            features[f'{variable}_high_proportion'] = high_count / len(values)
            features[f'{variable}_normal_proportion'] = normal_count / len(values)
            
            \# Severity of abnormality
            if low_count > 0:
                low_values = values[values < low_threshold]
                features[f'{variable}_low_severity_mean'] = np.mean(low_threshold - low_values)
                features[f'{variable}_low_severity_max'] = np.max(low_threshold - low_values)
            
            if high_count > 0:
                high_values = values[values > high_threshold]
                features[f'{variable}_high_severity_mean'] = np.mean(high_values - high_threshold)
                features[f'{variable}_high_severity_max'] = np.max(high_values - high_threshold)
            
            \# Distance from normal range
            distances = np.minimum(np.abs(values - low_threshold), np.abs(values - high_threshold))
            features[f'{variable}_mean_distance_from_normal'] = np.mean

(distances)
            features[f'{variable}_max_distance_from_normal'] = np.max(distances)
        
        \# Clinical stability features
        if len(values) >= 3:
            \# Consecutive abnormal values
            if normal_range:
                abnormal_mask = (values < normal_range<sup>0</sup>) | (values > normal_range<sup>1</sup>)
                consecutive_abnormal = self._find_consecutive_runs(abnormal_mask)
                features[f'{variable}_max_consecutive_abnormal'] = max(consecutive_abnormal) if consecutive_abnormal else 0
                features[f'{variable}_total_abnormal_runs'] = len(consecutive_abnormal)
        
        return features
    
    def _extract_pattern_features(self, variable: str, values: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
        """Extract pattern recognition features."""
        
        features = {}
        
        if len(values) < 4:
            return features
        
        \# Peak and valley detection
        try:
            peaks, peak_properties = find_peaks(values, height=np.mean(values))
            valleys, valley_properties = find_peaks(-values, height=-np.mean(values))
            
            features[f'{variable}_peak_count'] = len(peaks)
            features[f'{variable}_valley_count'] = len(valleys)
            
            if len(peaks) > 0:
                features[f'{variable}_mean_peak_height'] = np.mean(peak_properties['peak_heights'])
                features[f'{variable}_max_peak_height'] = np.max(peak_properties['peak_heights'])
            
            if len(valleys) > 0:
                features[f'{variable}_mean_valley_depth'] = np.mean(valley_properties['peak_heights'])
                features[f'{variable}_max_valley_depth'] = np.max(valley_properties['peak_heights'])
                
        except Exception as e:
            logger.warning(f"Error in peak detection for {variable}: {e}")
        
        \# Oscillation patterns
        if len(values) >= 6:
            \# Simple oscillation detection using zero crossings of detrended signal
            detrended = values - np.mean(values)
            zero_crossings = np.where(np.diff(np.signbit(detrended)))<sup>0</sup>
            features[f'{variable}_oscillation_frequency'] = len(zero_crossings) / len(values)
        
        \# Stability patterns
        if len(values) >= 5:
            \# Rolling stability (coefficient of variation in windows)
            window_size = min(5, len(values) // 2)
            rolling_cv = []
            for i in range(len(values) - window_size + 1):
                window_values = values[i:i + window_size]
                cv = np.std(window_values) / np.mean(window_values) if np.mean(window_values) != 0 else 0
                rolling_cv.append(cv)
            
            features[f'{variable}_mean_rolling_cv'] = np.mean(rolling_cv)
            features[f'{variable}_std_rolling_cv'] = np.std(rolling_cv)
            features[f'{variable}_max_rolling_cv'] = np.max(rolling_cv)
            features[f'{variable}_min_rolling_cv'] = np.min(rolling_cv)
        
        return features
    
    def _extract_time_features(self, variable: str, timestamps: np.ndarray) -> Dict[str, float]:
        """Extract time-based features."""
        
        features = {}
        
        if len(timestamps) < 2:
            return features
        
        \# Time interval analysis
        time_diffs = np.diff(timestamps)
        time_diffs_hours = np.array([td.total_seconds() / 3600 for td in time_diffs])
        
        features[f'{variable}_mean_interval'] = np.mean(time_diffs_hours)
        features[f'{variable}_std_interval'] = np.std(time_diffs_hours)
        features[f'{variable}_min_interval'] = np.min(time_diffs_hours)
        features[f'{variable}_max_interval'] = np.max(time_diffs_hours)
        features[f'{variable}_median_interval'] = np.median(time_diffs_hours)
        
        \# Regularity of measurements
        features[f'{variable}_interval_cv'] = np.std(time_diffs_hours) / np.mean(time_diffs_hours) if np.mean(time_diffs_hours) > 0 else 0
        
        \# Time span features
        total_time_hours = (timestamps[-1] - timestamps<sup>0</sup>).total_seconds() / 3600
        features[f'{variable}_total_time_span'] = total_time_hours
        features[f'{variable}_measurement_density'] = len(timestamps) / total_time_hours if total_time_hours > 0 else 0
        
        return features
    
    def _find_consecutive_runs(self, boolean_array: np.ndarray) -> List[int]:
        """Find lengths of consecutive True values in boolean array."""
        
        runs = []
        current_run = 0
        
        for value in boolean_array:
            if value:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0
        
        \# Don't forget the last run
        if current_run > 0:
            runs.append(current_run)
        
        return runs

class ClinicalMissingDataHandler:
    """
    Advanced missing data handling for clinical datasets.
    
    Implements multiple imputation strategies appropriate for clinical data,
    including clinical knowledge-based imputation, uncertainty quantification,
    and missing data pattern analysis that accounts for the clinical significance
    of missing values in healthcare settings.
    """
    
    def __init__(self, strategy: str = 'clinical_aware', uncertainty_quantification: bool = True):
        """
        Initialize the missing data handler.
        
        Args:
            strategy: Imputation strategy ('clinical_aware', 'knn', 'forward_fill', 'multiple')
            uncertainty_quantification: Whether to quantify imputation uncertainty
        """
        self.strategy = strategy
        self.uncertainty_quantification = uncertainty_quantification
        self.imputers = {}
        self.missing_indicators = {}
        self.imputation_uncertainty = {}
        self.clinical_ranges = ClinicalNormalRanges()
        
        logger.info(f"Initialized missing data handler with strategy: {strategy}")
    
    def fit_transform(self, X: pd.DataFrame, 
                     clinical_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
        """
        Fit imputation models and transform data.
        
        Args:
            X: Feature matrix with missing values
            clinical_ranges: Dict of normal clinical ranges for each feature
            
        Returns:
            Imputed feature matrix with missing indicators and uncertainty estimates
        """
        
        X_imputed = X.copy()
        missing_mask = X.isnull()
        
        \# Analyze missing data patterns
        missing_patterns = self._analyze_missing_patterns(X)
        logger.info(f"Identified {len(missing_patterns)} missing data patterns")
        
        \# Create missing indicators for features with missing data
        for col in X.columns:
            if missing_mask[col].any():
                X_imputed[f'{col}_missing'] = missing_mask[col].astype(int)
                self.missing_indicators[col] = f'{col}_missing'
                
                \# Calculate missing data statistics
                missing_rate = missing_mask[col].mean()
                X_imputed[f'{col}_missing_rate'] = missing_rate
        
        \# Apply imputation strategy
        if self.strategy == 'clinical_aware':
            X_imputed = self._clinical_aware_imputation(X_imputed, clinical_ranges)
        elif self.strategy == 'knn':
            X_imputed = self._knn_imputation(X_imputed)
        elif self.strategy == 'forward_fill':
            X_imputed = self._forward_fill_imputation(X_imputed)
        elif self.strategy == 'multiple':
            X_imputed = self._multiple_imputation(X_imputed, clinical_ranges)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.strategy}")
        
        \# Add uncertainty quantification if requested
        if self.uncertainty_quantification:
            X_imputed = self._add_uncertainty_features(X_imputed, missing_mask)
        
        return X_imputed
    
    def _analyze_missing_patterns(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data."""
        
        missing_mask = X.isnull()
        patterns = {}
        
        \# Overall missing statistics
        patterns['overall_missing_rate'] = missing_mask.sum().sum() / (len(X) * len(X.columns))
        patterns['features_with_missing'] = missing_mask.any().sum()
        patterns['complete_cases'] = (~missing_mask.any(axis=1)).sum()
        
        \# Per-feature missing rates
        patterns['feature_missing_rates'] = missing_mask.mean().to_dict()
        
        \# Missing data correlations
        if len(X.columns) > 1:
            missing_corr = missing_mask.astype(int).corr()
            patterns['missing_correlations'] = missing_corr.to_dict()
        
        \# Identify common missing patterns
        pattern_counts = missing_mask.value_counts()
        patterns['common_patterns'] = pattern_counts.head(10).to_dict()
        
        return patterns
    
    def _clinical_aware_imputation(self, X: pd.DataFrame, 
                                 clinical_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
        """Impute using clinical knowledge and normal ranges."""
        
        X_imputed = X.copy()
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if col.endswith('_missing') or col.endswith('_missing_rate'):
                continue
            
            missing_mask = X[col].isnull()
            
            if missing_mask.any():
                \# Determine imputation value based on clinical knowledge
                if clinical_ranges and col in clinical_ranges:
                    \# Use clinical normal range midpoint
                    normal_range = clinical_ranges[col]
                    impute_value = (normal_range<sup>0</sup> + normal_range<sup>1</sup>) / 2
                    
                elif hasattr(self.clinical_ranges, col.lower().replace(' ', '_')):
                    \# Use built-in clinical ranges
                    normal_range = self.clinical_ranges.get_normal_range(col)
                    if normal_range:
                        impute_value = (normal_range<sup>0</sup> + normal_range<sup>1</sup>) / 2
                    else:
                        impute_value = X[col].median()
                else:
                    \# Use median of observed values
                    impute_value = X[col].median()
                
                \# Apply imputation
                X_imputed.loc[missing_mask, col] = impute_value
                
                \# Store imputation information
                self.imputers[col] = {
                    'method': 'clinical_aware',
                    'value': impute_value,
                    'missing_count': missing_mask.sum()
                }
        
        return X_imputed
    
    def _knn_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """KNN-based imputation for numerical features."""
        
        X_imputed = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols 
                         if not col.endswith('_missing') and not col.endswith('_missing_rate')]
        
        if len(numerical_cols) > 0:
            \# Use KNN imputation with clinical-appropriate number of neighbors
            imputer = KNNImputer(n_neighbors=min(5, len(X) // 10), weights='distance')
            X_imputed[numerical_cols] = imputer.fit_transform(X[numerical_cols])
            self.imputers['knn'] = imputer
            
            logger.info(f"Applied KNN imputation to {len(numerical_cols)} numerical features")
        
        return X_imputed
    
    def _multiple_imputation(self, X: pd.DataFrame, 
                           clinical_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
        """Multiple imputation with uncertainty quantification."""
        
        \# For demonstration, implement a simplified version
        \# In production, would use more sophisticated multiple imputation
        
        X_imputed = X.copy()
        n_imputations = 5
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if col.endswith('_missing') or col.endswith('_missing_rate'):
                continue
            
            missing_mask = X[col].isnull()
            
            if missing_mask.any():
                \# Generate multiple imputations
                imputations = []
                
                for i in range(n_imputations):
                    \# Add noise to clinical-aware imputation
                    if clinical_ranges and col in clinical_ranges:
                        normal_range = clinical_ranges[col]
                        base_value = (normal_range<sup>0</sup> + normal_range<sup>1</sup>) / 2
                        noise_std = (normal_range<sup>1</sup> - normal_range<sup>0</sup>) / 6  \# Assume 99.7% within range
                    else:
                        base_value = X[col].median()
                        noise_std = X[col].std() * 0.1  \# 10% of standard deviation
                    
                    imputed_values = np.random.normal(base_value, noise_std, missing_mask.sum())
                    imputations.append(imputed_values)
                
                \# Use mean of imputations
                final_imputation = np.mean(imputations, axis=0)
                X_imputed.loc[missing_mask, col] = final_imputation
                
                \# Store uncertainty information
                imputation_std = np.std(imputations, axis=0)
                self.imputation_uncertainty[col] = {
                    'mean_uncertainty': np.mean(imputation_std),
                    'max_uncertainty': np.max(imputation_std)
                }
        
        return X_imputed
    
    def _add_uncertainty_features(self, X: pd.DataFrame, missing_mask: pd.DataFrame) -> pd.DataFrame:
        """Add features quantifying imputation uncertainty."""
        
        X_with_uncertainty = X.copy()
        
        \# Add overall uncertainty score for each row
        uncertainty_scores = []
        
        for idx in X.index:
            row_uncertainty = 0
            missing_count = 0
            
            for col in missing_mask.columns:
                if missing_mask.loc[idx, col]:
                    missing_count += 1
                    if col in self.imputation_uncertainty:
                        row_uncertainty += self.imputation_uncertainty[col]['mean_uncertainty']
            
            \# Normalize by number of missing values
            if missing_count > 0:
                uncertainty_scores.append(row_uncertainty / missing_count)
            else:
                uncertainty_scores.append(0)
        
        X_with_uncertainty['imputation_uncertainty_score'] = uncertainty_scores
        X_with_uncertainty['total_missing_features'] = missing_mask.sum(axis=1)
        X_with_uncertainty['missing_proportion'] = missing_mask.sum(axis=1) / len(missing_mask.columns)
        
        return X_with_uncertainty

class ClinicalFeatureSelector:
    """
    Advanced feature selection for clinical machine learning.
    
    Combines statistical methods with clinical domain knowledge
    to select optimal feature sets for clinical prediction tasks.
    This implementation prioritizes clinically relevant features
    while maintaining statistical rigor and model performance.
    """
    
    def __init__(self, 
                 max_features: int = 50,
                 clinical_priority_features: Optional[List[str]] = None,
                 stability_threshold: float = 0.8):
        """
        Initialize the clinical feature selector.
        
        Args:
            max_features: Maximum number of features to select
            clinical_priority_features: List of clinically important features to prioritize
            stability_threshold: Minimum stability score for feature selection
        """
        self.max_features = max_features
        self.clinical_priority_features = clinical_priority_features or []
        self.stability_threshold = stability_threshold
        self.selected_features = []
        self.feature_scores = {}
        self.selection_history = []
        
        logger.info(f"Initialized clinical feature selector with max {max_features} features")
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'combined',
                       cv_folds: int = 5) -> Tuple[List[str], Dict[str, float]]:
        """
        Select optimal features using multiple selection strategies.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('statistical', 'clinical', 'combined')
            cv_folds: Number of cross-validation folds for stability assessment
            
        Returns:
            Tuple of (selected feature names, feature importance scores)
        """
        
        if method == 'statistical':
            return self._statistical_selection(X, y, cv_folds)
        elif method == 'clinical':
            return self._clinical_selection(X, y)
        elif method == 'combined':
            return self._combined_selection(X, y, cv_folds)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _combined_selection(self, X: pd.DataFrame, y: pd.Series, 
                          cv_folds: int) -> Tuple[List[str], Dict[str, float]]:
        """Combined statistical and clinical feature selection."""
        
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LassoCV
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        \# Step 1: Statistical feature selection
        statistical_features, statistical_scores = self._statistical_selection(X, y, cv_folds)
        
        \# Step 2: Clinical priority features
        clinical_features = [f for f in self.clinical_priority_features if f in X.columns]
        
        \# Step 3: Combine and rank features
        all_candidate_features = list(set(statistical_features + clinical_features))
        
        \# Step 4: Stability-based selection
        stable_features = self._assess_feature_stability(X[all_candidate_features], y, cv_folds)
        
        \# Step 5: Final selection with clinical prioritization
        final_features = []
        final_scores = {}
        
        \# Always include stable clinical priority features
        for feature in clinical_features:
            if feature in stable_features and len(final_features) < self.max_features:
                final_features.append(feature)
                final_scores[feature] = statistical_scores.get(feature, 1.0) * 1.5  \# Boost clinical features
        
        \# Add remaining stable statistical features
        remaining_statistical = [f for f in stable_features if f not in final_features]
        remaining_statistical.sort(key=lambda x: statistical_scores.get(x, 0), reverse=True)
        
        for feature in remaining_statistical:
            if len(final_features) < self.max_features:
                final_features.append(feature)
                final_scores[feature] = statistical_scores.get(feature, 0)
        
        self.selected_features = final_features
        self.feature_scores = final_scores
        
        logger.info(f"Selected {len(final_features)} features using combined method")
        logger.info(f"Clinical priority features included: {len([f for f in final_features if f in clinical_features])}")
        
        return final_features, final_scores
    
    def _statistical_selection(self, X: pd.DataFrame, y: pd.Series, 
                             cv_folds: int) -> Tuple[List[str], Dict[str, float]]:
        """Statistical feature selection using multiple criteria."""
        
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LassoCV
        
        feature_scores = {}
        
        \# Univariate statistical tests (F-test)
        try:
            f_selector = SelectKBest(f_classif, k=min(self.max_features * 2, X.shape<sup>1</sup>))
            f_selector.fit(X, y)
            f_scores = f_selector.scores_
            f_selected_features = X.columns[f_selector.get_support()].tolist()
            
            for i, feature in enumerate(X.columns):
                if feature in f_selected_features:
                    feature_scores[feature] = feature_scores.get(feature, 0) + f_scores[i] / np.max(f_scores)
                    
        except Exception as e:
            logger.warning(f"F-test selection failed: {e}")
        
        \# Mutual information
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_ranking = np.argsort(mi_scores)[::-1]
            mi_selected_features = X.columns[mi_ranking[:self.max_features * 2]].tolist()
            
            for i, feature in enumerate(X.columns):
                if feature in mi_selected_features:
                    feature_scores[feature] = feature_scores.get(feature, 0) + mi_scores[i] / np.max(mi_scores)
                    
        except Exception as e:
            logger.warning(f"Mutual information selection failed: {e}")
        
        \# L1 regularization (Lasso)
        try:
            lasso = LassoCV(cv=cv_folds, random_state=42, max_iter=1000)
            lasso.fit(X, y)
            lasso_selected_features = X.columns[lasso.coef_ != 0].tolist()
            
            for feature in lasso_selected_features:
                coef_magnitude = abs(lasso.coef_[X.columns.get_loc(feature)])
                feature_scores[feature] = feature_scores.get(feature, 0) + coef_magnitude / np.max(np.abs(lasso.coef_))
                
        except Exception as e:
            logger.warning(f"Lasso selection failed: {e}")
        
        \# Random Forest feature importance
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            rf_importances = rf.feature_importances_
            rf_ranking = np.argsort(rf_importances)[::-1]
            rf_selected_features = X.columns[rf_ranking[:self.max_features * 2]].tolist()
            
            for i, feature in enumerate(X.columns):
                if feature in rf_selected_features:
                    feature_scores[feature] = feature_scores.get(feature, 0) + rf_importances[i] / np.max(rf_importances)
                    
        except Exception as e:
            logger.warning(f"Random Forest selection failed: {e}")
        
        \# Rank features by combined scores
        sorted_features = sorted(feature_scores.items(), key=lambda x: x<sup>1</sup>, reverse=True)
        selected_features = [f<sup>0</sup> for f in sorted_features[:self.max_features]]
        
        return selected_features, feature_scores
    
    def _clinical_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Clinical knowledge-based feature selection."""
        
        clinical_scores = {}
        
        \# Prioritize clinical priority features
        for feature in self.clinical_priority_features:
            if feature in X.columns:
                clinical_scores[feature] = 1.0
        
        \# Add features based on clinical naming patterns
        clinical_patterns = [
            'hemoglobin', 'hematocrit', 'white_blood_cell', 'platelet',
            'sodium', 'potassium', 'chloride', 'co2', 'bun', 'creatinine',
            'glucose', 'albumin', 'bilirubin', 'alt', 'ast',
            'systolic', 'diastolic', 'heart_rate', 'temperature',
            'respiratory_rate', 'oxygen_saturation'
        ]
        
        for feature in X.columns:
            feature_lower = feature.lower()
            for pattern in clinical_patterns:
                if pattern in feature_lower:
                    clinical_scores[feature] = clinical_scores.get(feature, 0) + 0.8
        
        \# Sort by clinical relevance
        sorted_features = sorted(clinical_scores.items(), key=lambda x: x<sup>1</sup>, reverse=True)
        selected_features = [f<sup>0</sup> for f in sorted_features[:self.max_features]]
        
        return selected_features, clinical_scores
    
    def _assess_feature_stability(self, X: pd.DataFrame, y: pd.Series, 
                                cv_folds: int) -> List[str]:
        """Assess feature selection stability across cross-validation folds."""
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.feature_selection import SelectKBest, f_classif
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        feature_selection_counts = {feature: 0 for feature in X.columns}
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            
            \# Perform feature selection on this fold
            try:
                selector = SelectKBest(f_classif, k=min(self.max_features, X_train.shape<sup>1</sup>))
                selector.fit(X_train, y_train)
                selected_features = X_train.columns[selector.get_support()].tolist()
                
                for feature in selected_features:
                    feature_selection_counts[feature] += 1
                    
            except Exception as e:
                logger.warning(f"Feature stability assessment failed for fold: {e}")
                continue
        
        \# Select features that appear in at least stability_threshold of folds
        min_appearances = int(cv_folds * self.stability_threshold)
        stable_features = [
            feature for feature, count in feature_selection_counts.items()
            if count >= min_appearances
        ]
        
        logger.info(f"Found {len(stable_features)} stable features (threshold: {self.stability_threshold})")
        
        return stable_features

\#\# 4.3 Advanced Clinical Machine Learning Models

\#\#\# 4.3.1 Ensemble Methods for Clinical Prediction

Clinical prediction tasks benefit significantly from ensemble methods that combine multiple models to improve prediction accuracy, robustness, and uncertainty quantification. The following implementation demonstrates a comprehensive ensemble framework designed specifically for clinical applications:
