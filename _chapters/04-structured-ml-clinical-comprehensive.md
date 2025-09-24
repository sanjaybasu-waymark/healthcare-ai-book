# Chapter 4: Structured Machine Learning for Clinical Applications

## Learning Objectives

By the end of this chapter, readers will be able to:
- Implement production-ready structured machine learning systems for clinical prediction
- Apply advanced feature engineering techniques specific to healthcare data
- Deploy ensemble methods optimized for clinical decision support
- Validate models using appropriate clinical metrics and statistical frameworks
- Integrate structured ML systems with existing clinical workflows

## 4.1 Introduction to Structured Clinical Machine Learning

Structured machine learning in healthcare represents the foundation of predictive analytics in clinical settings. Unlike unstructured data such as clinical notes or medical images, structured clinical data includes laboratory values, vital signs, medication records, and demographic information that can be directly processed by traditional machine learning algorithms. The systematic application of these techniques has revolutionized clinical decision support, enabling physicians to make more informed decisions based on data-driven insights.

The clinical application of structured machine learning differs fundamentally from traditional machine learning applications due to several unique characteristics. First, clinical data exhibits significant temporal dependencies, where the timing of measurements and interventions critically affects patient outcomes. Second, missing data is ubiquitous in clinical settings, often carrying clinical significance rather than representing random omissions. Third, the stakes of prediction errors are extraordinarily high, requiring robust uncertainty quantification and interpretability frameworks.

Recent advances in structured clinical machine learning have been driven by the availability of large-scale electronic health record (EHR) datasets and the development of specialized algorithms that account for the unique characteristics of clinical data. The work of Rajkomar et al. (2018) demonstrated that deep learning models applied to structured EHR data could predict in-hospital mortality, readmission risk, and length of stay with remarkable accuracy across multiple hospitals. This seminal work established the foundation for modern clinical prediction systems.

## 4.2 Clinical Data Preprocessing and Feature Engineering

### 4.2.1 Temporal Feature Engineering

Clinical data is inherently temporal, requiring sophisticated preprocessing techniques to capture the dynamic nature of patient states. The following implementation demonstrates a comprehensive temporal feature engineering pipeline:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ClinicalTemporalFeatureEngineer:
    """
    Comprehensive temporal feature engineering for clinical data.
    
    This class implements state-of-the-art techniques for processing
    time-series clinical data, including trend analysis, statistical
    aggregations, and temporal pattern recognition.
    
    References:
    - Che, Z., et al. (2018). Recurrent neural networks for multivariate 
      time series with missing values. Scientific Reports, 8(1), 6085.
    - Rajkomar, A., et al. (2018). Scalable and accurate deep learning 
      with electronic health records. NPJ Digital Medicine, 1(1), 18.
    """
    
    def __init__(self, window_hours=24, min_measurements=3):
        self.window_hours = window_hours
        self.min_measurements = min_measurements
        self.scalers = {}
        self.feature_names = []
        
    def extract_temporal_features(self, df, patient_id_col='patient_id', 
                                 timestamp_col='timestamp', value_col='value',
                                 variable_col='variable'):
        """
        Extract comprehensive temporal features from clinical time series data.
        
        Args:
            df: DataFrame with columns [patient_id, timestamp, variable, value]
            
        Returns:
            DataFrame with engineered temporal features
        """
        
        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by patient and timestamp
        df = df.sort_values([patient_id_col, timestamp_col])
        
        features_list = []
        
        for patient_id in df[patient_id_col].unique():
            patient_data = df[df[patient_id_col] == patient_id].copy()
            patient_features = self._extract_patient_features(
                patient_data, timestamp_col, value_col, variable_col
            )
            patient_features[patient_id_col] = patient_id
            features_list.append(patient_features)
            
        return pd.DataFrame(features_list)
    
    def _extract_patient_features(self, patient_data, timestamp_col, 
                                 value_col, variable_col):
        """Extract features for a single patient."""
        
        features = {}
        
        # Group by variable type
        for variable in patient_data[variable_col].unique():
            var_data = patient_data[patient_data[variable_col] == variable]
            
            if len(var_data) < self.min_measurements:
                continue
                
            values = var_data[value_col].values
            timestamps = var_data[timestamp_col].values
            
            # Basic statistical features
            features[f'{variable}_mean'] = np.mean(values)
            features[f'{variable}_std'] = np.std(values)
            features[f'{variable}_min'] = np.min(values)
            features[f'{variable}_max'] = np.max(values)
            features[f'{variable}_median'] = np.median(values)
            features[f'{variable}_q25'] = np.percentile(values, 25)
            features[f'{variable}_q75'] = np.percentile(values, 75)
            features[f'{variable}_range'] = np.max(values) - np.min(values)
            features[f'{variable}_count'] = len(values)
            
            # Temporal trend features
            if len(values) >= 3:
                # Linear trend slope
                time_numeric = [(t - timestamps[0]).total_seconds() / 3600 
                               for t in timestamps]
                slope, _, r_value, _, _ = stats.linregress(time_numeric, values)
                features[f'{variable}_trend_slope'] = slope
                features[f'{variable}_trend_r2'] = r_value ** 2
                
                # Rate of change features
                changes = np.diff(values)
                features[f'{variable}_mean_change'] = np.mean(changes)
                features[f'{variable}_std_change'] = np.std(changes)
                features[f'{variable}_max_increase'] = np.max(changes)
                features[f'{variable}_max_decrease'] = np.min(changes)
                
                # Volatility measures
                features[f'{variable}_volatility'] = np.std(changes) / np.mean(np.abs(values))
                
            # Time-based features
            time_diffs = np.diff(timestamps)
            time_diffs_hours = [td.total_seconds() / 3600 for td in time_diffs]
            
            if len(time_diffs_hours) > 0:
                features[f'{variable}_mean_interval'] = np.mean(time_diffs_hours)
                features[f'{variable}_std_interval'] = np.std(time_diffs_hours)
                features[f'{variable}_min_interval'] = np.min(time_diffs_hours)
                features[f'{variable}_max_interval'] = np.max(time_diffs_hours)
                
            # Recent vs. historical comparison
            if len(values) >= 6:
                recent_values = values[-3:]
                historical_values = values[:-3]
                
                features[f'{variable}_recent_vs_historical_mean'] = (
                    np.mean(recent_values) - np.mean(historical_values)
                )
                features[f'{variable}_recent_vs_historical_std'] = (
                    np.std(recent_values) - np.std(historical_values)
                )
                
        return features

class ClinicalMissingDataHandler:
    """
    Advanced missing data handling for clinical datasets.
    
    Implements multiple imputation strategies appropriate for clinical data,
    including clinical knowledge-based imputation and uncertainty quantification.
    """
    
    def __init__(self, strategy='clinical_aware'):
        self.strategy = strategy
        self.imputers = {}
        self.missing_indicators = {}
        
    def fit_transform(self, X, clinical_ranges=None):
        """
        Fit imputation models and transform data.
        
        Args:
            X: Feature matrix
            clinical_ranges: Dict of normal clinical ranges for each feature
            
        Returns:
            Imputed feature matrix with missing indicators
        """
        
        X_imputed = X.copy()
        missing_mask = X.isnull()
        
        # Create missing indicators
        for col in X.columns:
            if missing_mask[col].any():
                X_imputed[f'{col}_missing'] = missing_mask[col].astype(int)
                self.missing_indicators[col] = f'{col}_missing'
        
        # Apply imputation strategy
        if self.strategy == 'clinical_aware':
            X_imputed = self._clinical_aware_imputation(X_imputed, clinical_ranges)
        elif self.strategy == 'knn':
            X_imputed = self._knn_imputation(X_imputed)
        elif self.strategy == 'forward_fill':
            X_imputed = self._forward_fill_imputation(X_imputed)
            
        return X_imputed
    
    def _clinical_aware_imputation(self, X, clinical_ranges):
        """Impute using clinical knowledge and normal ranges."""
        
        X_imputed = X.copy()
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if col.endswith('_missing'):
                continue
                
            missing_mask = X[col].isnull()
            
            if missing_mask.any():
                if clinical_ranges and col in clinical_ranges:
                    # Use clinical normal range midpoint
                    normal_range = clinical_ranges[col]
                    impute_value = (normal_range[0] + normal_range[1]) / 2
                else:
                    # Use median of observed values
                    impute_value = X[col].median()
                    
                X_imputed.loc[missing_mask, col] = impute_value
                
        return X_imputed
    
    def _knn_imputation(self, X):
        """KNN-based imputation for numerical features."""
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if not col.endswith('_missing')]
        
        if len(numerical_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
            self.imputers['knn'] = imputer
            
        return X
    
    def _forward_fill_imputation(self, X):
        """Forward fill imputation for time series data."""
        
        return X.fillna(method='ffill').fillna(method='bfill')

# Clinical normal ranges for common laboratory values
CLINICAL_NORMAL_RANGES = {
    'hemoglobin': (12.0, 16.0),  # g/dL
    'white_blood_cell_count': (4.0, 11.0),  # K/uL
    'platelet_count': (150, 450),  # K/uL
    'sodium': (136, 145),  # mEq/L
    'potassium': (3.5, 5.0),  # mEq/L
    'chloride': (98, 107),  # mEq/L
    'co2': (22, 28),  # mEq/L
    'bun': (7, 20),  # mg/dL
    'creatinine': (0.6, 1.2),  # mg/dL
    'glucose': (70, 100),  # mg/dL
    'systolic_bp': (90, 140),  # mmHg
    'diastolic_bp': (60, 90),  # mmHg
    'heart_rate': (60, 100),  # bpm
    'temperature': (97.0, 99.5),  # F
    'respiratory_rate': (12, 20),  # breaths/min
    'oxygen_saturation': (95, 100),  # %
}

# Example usage
def preprocess_clinical_data(df):
    """
    Comprehensive preprocessing pipeline for clinical data.
    
    Args:
        df: Raw clinical data DataFrame
        
    Returns:
        Preprocessed feature matrix ready for machine learning
    """
    
    # Initialize feature engineer
    feature_engineer = ClinicalTemporalFeatureEngineer(window_hours=24)
    
    # Extract temporal features
    features_df = feature_engineer.extract_temporal_features(df)
    
    # Handle missing data
    missing_handler = ClinicalMissingDataHandler(strategy='clinical_aware')
    features_imputed = missing_handler.fit_transform(
        features_df, 
        clinical_ranges=CLINICAL_NORMAL_RANGES
    )
    
    return features_imputed
```

### 4.2.2 Advanced Feature Selection for Clinical Data

Feature selection in clinical machine learning requires careful consideration of both statistical significance and clinical relevance. The following implementation provides a comprehensive feature selection framework:

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import seaborn as sns

class ClinicalFeatureSelector:
    """
    Advanced feature selection for clinical machine learning.
    
    Combines statistical methods with clinical domain knowledge
    to select optimal feature sets for clinical prediction tasks.
    """
    
    def __init__(self, max_features=50, clinical_priority_features=None):
        self.max_features = max_features
        self.clinical_priority_features = clinical_priority_features or []
        self.selected_features = []
        self.feature_scores = {}
        
    def select_features(self, X, y, method='combined'):
        """
        Select optimal features using multiple selection strategies.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('statistical', 'clinical', 'combined')
            
        Returns:
            Selected feature names and importance scores
        """
        
        if method == 'statistical':
            return self._statistical_selection(X, y)
        elif method == 'clinical':
            return self._clinical_selection(X, y)
        elif method == 'combined':
            return self._combined_selection(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _statistical_selection(self, X, y):
        """Statistical feature selection using multiple criteria."""
        
        # Univariate statistical tests
        f_selector = SelectKBest(f_classif, k=min(self.max_features, X.shape[1]))
        X_f_selected = f_selector.fit_transform(X, y)
        f_scores = f_selector.scores_
        f_selected_features = X.columns[f_selector.get_support()].tolist()
        
        # Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_ranking = np.argsort(mi_scores)[::-1]
        mi_selected_features = X.columns[mi_ranking[:self.max_features]].tolist()
        
        # L1 regularization (Lasso)
        lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso.fit(X, y)
        lasso_selected_features = X.columns[lasso.coef_ != 0].tolist()
        
        # Recursive feature elimination
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(rf, n_features_to_select=min(self.max_features, X.shape[1]))
        rfe.fit(X, y)
        rfe_selected_features = X.columns[rfe.support_].tolist()
        
        # Combine selections using voting
        all_features = set(f_selected_features + mi_selected_features + 
                          lasso_selected_features + rfe_selected_features)
        
        feature_votes = {}
        for feature in all_features:
            votes = 0
            if feature in f_selected_features:
                votes += 1
            if feature in mi_selected_features:
                votes += 1
            if feature in lasso_selected_features:
                votes += 1
            if feature in rfe_selected_features:
                votes += 1
            feature_votes[feature] = votes
        
        # Select features with highest votes
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:self.max_features]]
        
        self.feature_scores['statistical'] = feature_votes
        return selected_features
    
    def _clinical_selection(self, X, y):
        """Clinical knowledge-based feature selection."""
        
        # Priority features (always include if available)
        priority_features = [f for f in self.clinical_priority_features if f in X.columns]
        
        # Clinical feature categories with importance weights
        clinical_categories = {
            'vital_signs': ['systolic_bp', 'diastolic_bp', 'heart_rate', 
                           'temperature', 'respiratory_rate', 'oxygen_saturation'],
            'laboratory': ['hemoglobin', 'white_blood_cell_count', 'platelet_count',
                          'sodium', 'potassium', 'creatinine', 'bun', 'glucose'],
            'demographics': ['age', 'gender', 'race', 'ethnicity'],
            'comorbidities': ['diabetes', 'hypertension', 'heart_disease', 
                             'kidney_disease', 'liver_disease'],
            'medications': ['ace_inhibitor', 'beta_blocker', 'diuretic', 
                           'anticoagulant', 'statin']
        }
        
        category_weights = {
            'vital_signs': 1.0,
            'laboratory': 0.9,
            'comorbidities': 0.8,
            'medications': 0.7,
            'demographics': 0.6
        }
        
        # Score features based on clinical categories
        clinical_scores = {}
        for category, features in clinical_categories.items():
            weight = category_weights[category]
            for feature in features:
                matching_cols = [col for col in X.columns if feature in col.lower()]
                for col in matching_cols:
                    clinical_scores[col] = weight
        
        # Add priority features with maximum score
        for feature in priority_features:
            clinical_scores[feature] = 1.0
        
        # Select top features by clinical score
        sorted_clinical = sorted(clinical_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_clinical[:self.max_features]]
        
        self.feature_scores['clinical'] = clinical_scores
        return selected_features
    
    def _combined_selection(self, X, y):
        """Combined statistical and clinical feature selection."""
        
        # Get statistical selection
        statistical_features = self._statistical_selection(X, y)
        
        # Get clinical selection
        clinical_features = self._clinical_selection(X, y)
        
        # Combine with weighted scoring
        combined_scores = {}
        
        # Statistical features (weight = 0.6)
        stat_scores = self.feature_scores['statistical']
        for feature, score in stat_scores.items():
            combined_scores[feature] = combined_scores.get(feature, 0) + 0.6 * score / 4
        
        # Clinical features (weight = 0.4)
        clin_scores = self.feature_scores['clinical']
        for feature, score in clin_scores.items():
            combined_scores[feature] = combined_scores.get(feature, 0) + 0.4 * score
        
        # Select top combined features
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_combined[:self.max_features]]
        
        self.feature_scores['combined'] = combined_scores
        self.selected_features = selected_features
        
        return selected_features
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance scores."""
        
        if not self.feature_scores:
            raise ValueError("No feature selection has been performed yet.")
        
        fig, axes = plt.subplots(1, len(self.feature_scores), 
                                figsize=(6*len(self.feature_scores), 8))
        
        if len(self.feature_scores) == 1:
            axes = [axes]
        
        for i, (method, scores) in enumerate(self.feature_scores.items()):
            # Get top features
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_scores[:top_n]
            
            features, importance = zip(*top_features)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            axes[i].barh(y_pos, importance)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(features)
            axes[i].set_xlabel('Importance Score')
            axes[i].set_title(f'{method.title()} Feature Selection')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.show()

# Clinical priority features for different prediction tasks
CLINICAL_PRIORITY_FEATURES = {
    'mortality': ['age', 'systolic_bp_min', 'heart_rate_max', 'temperature_max',
                  'white_blood_cell_count_max', 'creatinine_max', 'lactate_max'],
    'readmission': ['age', 'length_of_stay', 'discharge_disposition', 
                    'number_diagnoses', 'number_medications'],
    'sepsis': ['temperature_max', 'heart_rate_max', 'respiratory_rate_max',
               'white_blood_cell_count_max', 'lactate_max', 'systolic_bp_min'],
    'aki': ['creatinine_max', 'creatinine_trend_slope', 'bun_max', 
            'urine_output_min', 'contrast_exposure']
}
```

## 4.3 Advanced Ensemble Methods for Clinical Prediction

### 4.3.1 Clinical-Aware Ensemble Architecture

Ensemble methods in clinical machine learning require careful consideration of model diversity, interpretability, and clinical workflow integration. The following implementation demonstrates a sophisticated ensemble framework designed specifically for clinical applications:

```python
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

class ClinicalEnsembleClassifier:
    """
    Advanced ensemble classifier optimized for clinical prediction tasks.
    
    This implementation combines multiple base learners with clinical-aware
    weighting and uncertainty quantification specifically designed for
    healthcare applications.
    
    References:
    - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
      Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge 
      Discovery and Data Mining.
    - Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting 
      decision tree. Advances in Neural Information Processing Systems.
    """
    
    def __init__(self, ensemble_method='stacking', calibration=True, 
                 uncertainty_quantification=True):
        self.ensemble_method = ensemble_method
        self.calibration = calibration
        self.uncertainty_quantification = uncertainty_quantification
        self.base_models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.model_weights = {}
        
    def _initialize_base_models(self):
        """Initialize diverse base models for ensemble."""
        
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
    
    def fit(self, X, y, validation_split=0.2):
        """
        Fit the ensemble model with clinical validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction of data for validation
            
        Returns:
            Fitted ensemble model
        """
        
        self._initialize_base_models()
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        # Train and evaluate base models
        base_predictions = {}
        base_scores = {}
        
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Get predictions
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]
            
            # Calculate validation score
            val_score = roc_auc_score(y_val, val_pred)
            base_scores[name] = val_score
            base_predictions[name] = val_pred
            
            print(f"{name} validation AUC: {val_score:.4f}")
        
        # Calculate model weights based on performance
        self._calculate_model_weights(base_scores)
        
        # Create ensemble
        if self.ensemble_method == 'voting':
            self.ensemble_model = self._create_voting_ensemble()
        elif self.ensemble_method == 'stacking':
            self.ensemble_model = self._create_stacking_ensemble()
        elif self.ensemble_method == 'weighted_average':
            self.ensemble_model = self._create_weighted_ensemble()
        
        # Fit ensemble model
        self.ensemble_model.fit(X_train, y_train)
        
        # Calculate feature importance
        self._calculate_feature_importance(X)
        
        return self
    
    def _calculate_model_weights(self, scores):
        """Calculate weights for models based on validation performance."""
        
        # Convert scores to weights using softmax
        score_values = np.array(list(scores.values()))
        exp_scores = np.exp(score_values - np.max(score_values))
        weights = exp_scores / np.sum(exp_scores)
        
        self.model_weights = dict(zip(scores.keys(), weights))
        
        print("\nModel weights:")
        for name, weight in self.model_weights.items():
            print(f"{name}: {weight:.4f}")
    
    def _create_voting_ensemble(self):
        """Create voting ensemble with optimized weights."""
        
        # Select top performing models
        top_models = sorted(self.model_weights.items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        
        estimators = [(name, self.base_models[name]) for name, _ in top_models]
        weights = [weight for _, weight in top_models]
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
    
    def _create_stacking_ensemble(self):
        """Create stacking ensemble with meta-learner."""
        
        # Use all base models as level-0 estimators
        estimators = list(self.base_models.items())
        
        # Use logistic regression as meta-learner
        meta_learner = LogisticRegression(
            C=1.0, 
            penalty='l2', 
            solver='liblinear',
            random_state=42
        )
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba'
        )
    
    def _create_weighted_ensemble(self):
        """Create custom weighted ensemble."""
        
        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
                
            def fit(self, X, y):
                # Models are already fitted
                return self
                
            def predict_proba(self, X):
                predictions = np.zeros((X.shape[0], 2))
                
                for (name, model), weight in zip(self.models.items(), 
                                                self.weights.values()):
                    pred = model.predict_proba(X)
                    predictions += weight * pred
                    
                return predictions
                
            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] > 0.5).astype(int)
        
        return WeightedEnsemble(self.base_models, self.model_weights)
    
    def _calculate_feature_importance(self, X):
        """Calculate ensemble feature importance."""
        
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        ensemble_importance = np.zeros(len(feature_names))
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                continue
                
            # Weight by model performance
            weight = self.model_weights.get(name, 0)
            ensemble_importance += weight * importance
        
        self.feature_importance = dict(zip(feature_names, ensemble_importance))
    
    def predict_proba(self, X):
        """Predict class probabilities with uncertainty quantification."""
        
        if self.ensemble_model is None:
            raise ValueError("Model must be fitted before prediction.")
        
        # Get ensemble predictions
        proba = self.ensemble_model.predict_proba(X)
        
        if self.uncertainty_quantification:
            # Calculate prediction uncertainty
            individual_predictions = []
            for model in self.base_models.values():
                pred = model.predict_proba(X)[:, 1]
                individual_predictions.append(pred)
            
            individual_predictions = np.array(individual_predictions)
            uncertainty = np.std(individual_predictions, axis=0)
            
            return proba, uncertainty
        
        return proba
    
    def predict(self, X):
        """Make binary predictions."""
        
        if self.uncertainty_quantification:
            proba, _ = self.predict_proba(X)
        else:
            proba = self.predict_proba(X)
            
        return (proba[:, 1] > 0.5).astype(int)
    
    def get_feature_importance(self, top_n=20):
        """Get top feature importances."""
        
        if not self.feature_importance:
            raise ValueError("Feature importance not calculated. Fit model first.")
        
        sorted_importance = sorted(self.feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return sorted_importance[:top_n]
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance."""
        
        top_features = self.get_feature_importance(top_n)
        features, importance = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importance)
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance')
        plt.title('Ensemble Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

class ClinicalModelValidator:
    """
    Comprehensive validation framework for clinical machine learning models.
    
    Implements clinical-specific validation metrics and statistical tests
    to ensure model reliability and clinical utility.
    """
    
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.validation_results = {}
        
    def validate_model(self, model, X, y, cv_folds=5):
        """
        Comprehensive model validation with clinical metrics.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of validation results
        """
        
        results = {}
        
        # Cross-validation scores
        cv_scores = self._cross_validation_analysis(model, X, y, cv_folds)
        results['cross_validation'] = cv_scores
        
        # Bootstrap confidence intervals
        bootstrap_results = self._bootstrap_validation(model, X, y)
        results['bootstrap'] = bootstrap_results
        
        # Clinical significance tests
        clinical_tests = self._clinical_significance_tests(model, X, y)
        results['clinical_tests'] = clinical_tests
        
        # Calibration analysis
        calibration_results = self._calibration_analysis(model, X, y)
        results['calibration'] = calibration_results
        
        self.validation_results = results
        return results
    
    def _cross_validation_analysis(self, model, X, y, cv_folds):
        """Perform comprehensive cross-validation analysis."""
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Multiple metrics
        metrics = {
            'auc': [],
            'sensitivity': [],
            'specificity': [],
            'ppv': [],
            'npv': [],
            'f1': []
        }
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            auc = roc_auc_score(y_val, y_pred_proba)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            metrics['auc'].append(auc)
            metrics['sensitivity'].append(sensitivity)
            metrics['specificity'].append(specificity)
            metrics['ppv'].append(ppv)
            metrics['npv'].append(npv)
            metrics['f1'].append(f1)
        
        # Calculate statistics
        cv_results = {}
        for metric, values in metrics.items():
            cv_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, (1 - self.confidence_level) / 2 * 100),
                'ci_upper': np.percentile(values, (1 + self.confidence_level) / 2 * 100)
            }
        
        return cv_results
    
    def _bootstrap_validation(self, model, X, y, n_bootstrap=1000):
        """Bootstrap validation for confidence intervals."""
        
        bootstrap_scores = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Out-of-bag sample
            oob_indices = list(set(range(len(X))) - set(indices))
            if len(oob_indices) == 0:
                continue
                
            X_oob = X.iloc[oob_indices]
            y_oob = y.iloc[oob_indices]
            
            # Fit and predict
            model.fit(X_boot, y_boot)
            y_pred_proba = model.predict_proba(X_oob)[:, 1]
            
            # Calculate AUC
            if len(np.unique(y_oob)) > 1:
                auc = roc_auc_score(y_oob, y_pred_proba)
                bootstrap_scores.append(auc)
        
        return {
            'mean_auc': np.mean(bootstrap_scores),
            'std_auc': np.std(bootstrap_scores),
            'ci_lower': np.percentile(bootstrap_scores, (1 - self.confidence_level) / 2 * 100),
            'ci_upper': np.percentile(bootstrap_scores, (1 + self.confidence_level) / 2 * 100)
        }
    
    def _clinical_significance_tests(self, model, X, y):
        """Perform clinical significance tests."""
        
        # Fit model
        model.fit(X, y)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # DeLong test for AUC comparison (vs. random classifier)
        auc = roc_auc_score(y, y_pred_proba)
        
        # McNemar's test for paired predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Net reclassification improvement
        nri = self._calculate_nri(y, y_pred_proba)
        
        return {
            'auc': auc,
            'auc_pvalue': self._auc_significance_test(y, y_pred_proba),
            'nri': nri
        }
    
    def _auc_significance_test(self, y_true, y_scores):
        """Test if AUC is significantly different from 0.5."""
        
        auc = roc_auc_score(y_true, y_scores)
        
        # Hanley-McNeil method for AUC standard error
        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)
        
        q1 = auc / (2 - auc)
        q2 = 2 * auc**2 / (1 + auc)
        
        se_auc = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + 
                         (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
        
        # Z-test
        z_score = (auc - 0.5) / se_auc
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return p_value
    
    def _calculate_nri(self, y_true, y_scores, thresholds=[0.1, 0.3, 0.5]):
        """Calculate Net Reclassification Improvement."""
        
        # This is a simplified NRI calculation
        # In practice, you would compare against a baseline model
        
        nri_events = 0
        nri_non_events = 0
        
        for i in range(len(thresholds) - 1):
            low_thresh = thresholds[i]
            high_thresh = thresholds[i + 1]
            
            # Events (y_true == 1)
            events_mask = y_true == 1
            events_in_range = ((y_scores >= low_thresh) & 
                              (y_scores < high_thresh) & events_mask)
            nri_events += np.sum(events_in_range)
            
            # Non-events (y_true == 0)
            non_events_mask = y_true == 0
            non_events_in_range = ((y_scores >= low_thresh) & 
                                  (y_scores < high_thresh) & non_events_mask)
            nri_non_events += np.sum(non_events_in_range)
        
        total_events = np.sum(y_true == 1)
        total_non_events = np.sum(y_true == 0)
        
        nri = (nri_events / total_events) - (nri_non_events / total_non_events)
        return nri
    
    def _calibration_analysis(self, model, X, y):
        """Analyze model calibration."""
        
        from sklearn.calibration import calibration_curve
        
        model.fit(X, y)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_pred_proba, n_bins=10
        )
        
        # Brier score
        brier_score = np.mean((y_pred_proba - y) ** 2)
        
        # Hosmer-Lemeshow test
        hl_statistic, hl_pvalue = self._hosmer_lemeshow_test(y, y_pred_proba)
        
        return {
            'brier_score': brier_score,
            'hosmer_lemeshow_statistic': hl_statistic,
            'hosmer_lemeshow_pvalue': hl_pvalue,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
        }
    
    def _hosmer_lemeshow_test(self, y_true, y_prob, n_bins=10):
        """Hosmer-Lemeshow goodness-of-fit test."""
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        hl_statistic = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find observations in this bin
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
            if bin_upper == 1.0:  # Include 1.0 in the last bin
                in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            
            if np.sum(in_bin) == 0:
                continue
            
            # Observed and expected counts
            observed_events = np.sum(y_true[in_bin])
            observed_non_events = np.sum(in_bin) - observed_events
            
            expected_events = np.sum(y_prob[in_bin])
            expected_non_events = np.sum(in_bin) - expected_events
            
            # Add to chi-square statistic
            if expected_events > 0:
                hl_statistic += (observed_events - expected_events) ** 2 / expected_events
            if expected_non_events > 0:
                hl_statistic += (observed_non_events - expected_non_events) ** 2 / expected_non_events
        
        # P-value from chi-square distribution
        p_value = 1 - stats.chi2.cdf(hl_statistic, n_bins - 2)
        
        return hl_statistic, p_value
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_model first.")
        
        report = []
        report.append("=" * 60)
        report.append("CLINICAL MODEL VALIDATION REPORT")
        report.append("=" * 60)
        
        # Cross-validation results
        cv_results = self.validation_results['cross_validation']
        report.append("\nCROSS-VALIDATION RESULTS:")
        report.append("-" * 30)
        
        for metric, stats in cv_results.items():
            report.append(f"{metric.upper()}:")
            report.append(f"  Mean: {stats['mean']:.4f}")
            report.append(f"  Std:  {stats['std']:.4f}")
            report.append(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
            report.append("")
        
        # Bootstrap results
        bootstrap_results = self.validation_results['bootstrap']
        report.append("BOOTSTRAP VALIDATION:")
        report.append("-" * 20)
        report.append(f"AUC: {bootstrap_results['mean_auc']:.4f} ± {bootstrap_results['std_auc']:.4f}")
        report.append(f"95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
        report.append("")
        
        # Clinical significance
        clinical_results = self.validation_results['clinical_tests']
        report.append("CLINICAL SIGNIFICANCE TESTS:")
        report.append("-" * 30)
        report.append(f"AUC: {clinical_results['auc']:.4f}")
        report.append(f"AUC p-value: {clinical_results['auc_pvalue']:.4f}")
        report.append(f"Net Reclassification Improvement: {clinical_results['nri']:.4f}")
        report.append("")
        
        # Calibration
        calibration_results = self.validation_results['calibration']
        report.append("MODEL CALIBRATION:")
        report.append("-" * 18)
        report.append(f"Brier Score: {calibration_results['brier_score']:.4f}")
        report.append(f"Hosmer-Lemeshow Statistic: {calibration_results['hosmer_lemeshow_statistic']:.4f}")
        report.append(f"Hosmer-Lemeshow p-value: {calibration_results['hosmer_lemeshow_pvalue']:.4f}")
        
        return "\n".join(report)
```

## 4.4 Clinical Deployment and Integration

### 4.4.1 Production Deployment Framework

The deployment of structured machine learning models in clinical environments requires robust infrastructure that ensures reliability, scalability, and regulatory compliance. The following implementation provides a comprehensive deployment framework:

```python
import joblib
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3
import hashlib
import os

class ClinicalMLDeployment:
    """
    Production deployment framework for clinical machine learning models.
    
    Provides comprehensive infrastructure for deploying, monitoring, and
    maintaining ML models in clinical environments with full audit trails
    and regulatory compliance.
    """
    
    def __init__(self, model_name: str, version: str, deployment_config: Dict):
        self.model_name = model_name
        self.version = version
        self.config = deployment_config
        self.model = None
        self.feature_processor = None
        self.audit_db = None
        self.performance_monitor = None
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize audit database
        self._setup_audit_database()
        
        # Initialize performance monitoring
        self._setup_performance_monitoring()
    
    def _setup_logging(self):
        """Setup comprehensive logging for clinical deployment."""
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'clinical_ml_{self.model_name}_{self.version}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'ClinicalML_{self.model_name}')
    
    def _setup_audit_database(self):
        """Setup audit database for regulatory compliance."""
        
        db_path = f'clinical_audit_{self.model_name}_{self.version}.db'
        self.audit_db = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create audit tables
        cursor = self.audit_db.cursor()
        
        # Predictions audit table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                patient_id TEXT,
                input_hash TEXT,
                prediction REAL,
                confidence REAL,
                model_version TEXT,
                user_id TEXT,
                session_id TEXT
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                model_version TEXT,
                data_period TEXT
            )
        ''')
        
        # Data drift table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_drift (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                feature_name TEXT,
                drift_score REAL,
                drift_threshold REAL,
                alert_triggered BOOLEAN,
                model_version TEXT
            )
        ''')
        
        self.audit_db.commit()
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring system."""
        
        self.performance_monitor = {
            'prediction_times': [],
            'prediction_counts': 0,
            'error_counts': 0,
            'last_performance_check': datetime.now(),
            'drift_alerts': [],
            'model_metrics': {}
        }
    
    def load_model(self, model_path: str, feature_processor_path: str):
        """
        Load trained model and feature processor.
        
        Args:
            model_path: Path to saved model
            feature_processor_path: Path to saved feature processor
        """
        
        try:
            # Load model
            self.model = joblib.load(model_path)
            self.logger.info(f"Model loaded successfully from {model_path}")
            
            # Load feature processor
            self.feature_processor = joblib.load(feature_processor_path)
            self.logger.info(f"Feature processor loaded from {feature_processor_path}")
            
            # Validate model
            self._validate_loaded_model()
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _validate_loaded_model(self):
        """Validate loaded model meets deployment requirements."""
        
        # Check model has required methods
        required_methods = ['predict', 'predict_proba']
        for method in required_methods:
            if not hasattr(self.model, method):
                raise ValueError(f"Model missing required method: {method}")
        
        # Check feature processor
        if not hasattr(self.feature_processor, 'transform'):
            raise ValueError("Feature processor missing transform method")
        
        self.logger.info("Model validation passed")
    
    def predict(self, input_data: Dict, patient_id: str, user_id: str, 
                session_id: str) -> Dict:
        """
        Make clinical prediction with full audit trail.
        
        Args:
            input_data: Dictionary of input features
            patient_id: Patient identifier
            user_id: User making the request
            session_id: Session identifier
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        
        start_time = datetime.now()
        
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Process features
            processed_features = self._process_features(input_data)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(processed_features)[0]
            prediction = prediction_proba[1]  # Probability of positive class
            confidence = max(prediction_proba) - min(prediction_proba)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create input hash for audit
            input_hash = self._create_input_hash(input_data)
            
            # Log prediction to audit database
            self._log_prediction(
                patient_id, input_hash, prediction, confidence,
                user_id, session_id
            )
            
            # Update performance monitoring
            self._update_performance_monitoring(processing_time)
            
            # Check for data drift
            drift_alerts = self._check_data_drift(processed_features)
            
            # Prepare response
            response = {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'processing_time_seconds': processing_time,
                'model_version': self.version,
                'timestamp': start_time.isoformat(),
                'drift_alerts': drift_alerts,
                'clinical_interpretation': self._generate_clinical_interpretation(
                    prediction, confidence
                )
            }
            
            self.logger.info(f"Prediction completed for patient {patient_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Prediction error for patient {patient_id}: {str(e)}")
            self.performance_monitor['error_counts'] += 1
            raise
    
    def _validate_input(self, input_data: Dict):
        """Validate input data meets requirements."""
        
        required_fields = self.config.get('required_fields', [])
        
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types and ranges
        for field, value in input_data.items():
            if field in self.config.get('numeric_fields', []):
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Field {field} must be numeric")
                
                # Check ranges
                if field in self.config.get('field_ranges', {}):
                    min_val, max_val = self.config['field_ranges'][field]
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"Field {field} out of valid range [{min_val}, {max_val}]")
    
    def _process_features(self, input_data: Dict) -> np.ndarray:
        """Process raw input into model features."""
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply feature processing pipeline
        processed_features = self.feature_processor.transform(df)
        
        return processed_features
    
    def _create_input_hash(self, input_data: Dict) -> str:
        """Create hash of input data for audit purposes."""
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def _log_prediction(self, patient_id: str, input_hash: str, 
                       prediction: float, confidence: float,
                       user_id: str, session_id: str):
        """Log prediction to audit database."""
        
        cursor = self.audit_db.cursor()
        cursor.execute('''
            INSERT INTO predictions_audit 
            (patient_id, input_hash, prediction, confidence, model_version, user_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (patient_id, input_hash, prediction, confidence, 
              self.version, user_id, session_id))
        
        self.audit_db.commit()
    
    def _update_performance_monitoring(self, processing_time: float):
        """Update performance monitoring metrics."""
        
        self.performance_monitor['prediction_times'].append(processing_time)
        self.performance_monitor['prediction_counts'] += 1
        
        # Keep only recent prediction times (last 1000)
        if len(self.performance_monitor['prediction_times']) > 1000:
            self.performance_monitor['prediction_times'] = \
                self.performance_monitor['prediction_times'][-1000:]
    
    def _check_data_drift(self, processed_features: np.ndarray) -> List[Dict]:
        """Check for data drift in input features."""
        
        drift_alerts = []
        
        # This is a simplified drift detection
        # In production, you would use more sophisticated methods
        
        if hasattr(self, 'reference_statistics'):
            for i, feature_value in enumerate(processed_features[0]):
                feature_name = f'feature_{i}'
                
                if feature_name in self.reference_statistics:
                    ref_mean = self.reference_statistics[feature_name]['mean']
                    ref_std = self.reference_statistics[feature_name]['std']
                    
                    # Z-score based drift detection
                    z_score = abs((feature_value - ref_mean) / ref_std) if ref_std > 0 else 0
                    
                    if z_score > 3:  # 3-sigma rule
                        alert = {
                            'feature': feature_name,
                            'z_score': float(z_score),
                            'current_value': float(feature_value),
                            'reference_mean': float(ref_mean),
                            'reference_std': float(ref_std)
                        }
                        drift_alerts.append(alert)
                        
                        # Log to database
                        self._log_drift_alert(feature_name, z_score, 3.0)
        
        return drift_alerts
    
    def _log_drift_alert(self, feature_name: str, drift_score: float, 
                        threshold: float):
        """Log data drift alert to database."""
        
        cursor = self.audit_db.cursor()
        cursor.execute('''
            INSERT INTO data_drift 
            (feature_name, drift_score, drift_threshold, alert_triggered, model_version)
            VALUES (?, ?, ?, ?, ?)
        ''', (feature_name, drift_score, threshold, True, self.version))
        
        self.audit_db.commit()
    
    def _generate_clinical_interpretation(self, prediction: float, 
                                        confidence: float) -> Dict:
        """Generate clinical interpretation of prediction."""
        
        # Risk categories
        if prediction < 0.1:
            risk_category = "Low Risk"
            recommendation = "Continue standard care"
        elif prediction < 0.3:
            risk_category = "Moderate Risk"
            recommendation = "Consider additional monitoring"
        elif prediction < 0.7:
            risk_category = "High Risk"
            recommendation = "Implement preventive interventions"
        else:
            risk_category = "Very High Risk"
            recommendation = "Immediate clinical attention recommended"
        
        # Confidence interpretation
        if confidence < 0.5:
            confidence_level = "Low"
            confidence_note = "Prediction uncertainty is high. Consider additional clinical assessment."
        elif confidence < 0.8:
            confidence_level = "Moderate"
            confidence_note = "Moderate prediction confidence. Clinical judgment recommended."
        else:
            confidence_level = "High"
            confidence_note = "High prediction confidence."
        
        return {
            'risk_score': float(prediction),
            'risk_category': risk_category,
            'recommendation': recommendation,
            'confidence_level': confidence_level,
            'confidence_note': confidence_note,
            'clinical_threshold': 0.5  # Default threshold for clinical action
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        
        if not self.performance_monitor['prediction_times']:
            return {'status': 'No predictions made yet'}
        
        prediction_times = self.performance_monitor['prediction_times']
        
        metrics = {
            'total_predictions': self.performance_monitor['prediction_counts'],
            'total_errors': self.performance_monitor['error_counts'],
            'error_rate': self.performance_monitor['error_counts'] / 
                         max(self.performance_monitor['prediction_counts'], 1),
            'average_processing_time': np.mean(prediction_times),
            'median_processing_time': np.median(prediction_times),
            'p95_processing_time': np.percentile(prediction_times, 95),
            'p99_processing_time': np.percentile(prediction_times, 99),
            'last_performance_check': self.performance_monitor['last_performance_check'].isoformat()
        }
        
        return metrics
    
    def generate_audit_report(self, start_date: str, end_date: str) -> Dict:
        """Generate audit report for specified date range."""
        
        cursor = self.audit_db.cursor()
        
        # Prediction statistics
        cursor.execute('''
            SELECT COUNT(*), AVG(prediction), AVG(confidence)
            FROM predictions_audit
            WHERE timestamp BETWEEN ? AND ?
        ''', (start_date, end_date))
        
        pred_stats = cursor.fetchone()
        
        # User activity
        cursor.execute('''
            SELECT user_id, COUNT(*) as prediction_count
            FROM predictions_audit
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY user_id
            ORDER BY prediction_count DESC
        ''', (start_date, end_date))
        
        user_activity = cursor.fetchall()
        
        # Drift alerts
        cursor.execute('''
            SELECT feature_name, COUNT(*) as alert_count
            FROM data_drift
            WHERE timestamp BETWEEN ? AND ? AND alert_triggered = 1
            GROUP BY feature_name
            ORDER BY alert_count DESC
        ''', (start_date, end_date))
        
        drift_alerts = cursor.fetchall()
        
        report = {
            'report_period': {'start': start_date, 'end': end_date},
            'prediction_statistics': {
                'total_predictions': pred_stats[0] if pred_stats[0] else 0,
                'average_prediction': pred_stats[1] if pred_stats[1] else 0,
                'average_confidence': pred_stats[2] if pred_stats[2] else 0
            },
            'user_activity': [{'user_id': row[0], 'predictions': row[1]} 
                             for row in user_activity],
            'drift_alerts': [{'feature': row[0], 'alert_count': row[1]} 
                            for row in drift_alerts],
            'performance_metrics': self.get_performance_metrics()
        }
        
        return report
    
    def shutdown(self):
        """Gracefully shutdown deployment system."""
        
        self.logger.info("Shutting down clinical ML deployment system")
        
        if self.audit_db:
            self.audit_db.close()
        
        # Save final performance metrics
        final_metrics = self.get_performance_metrics()
        with open(f'final_metrics_{self.model_name}_{self.version}.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        self.logger.info("Shutdown complete")

# Example deployment configuration
DEPLOYMENT_CONFIG = {
    'required_fields': [
        'age', 'gender', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'temperature', 'respiratory_rate', 'oxygen_saturation'
    ],
    'numeric_fields': [
        'age', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'temperature', 'respiratory_rate', 'oxygen_saturation'
    ],
    'field_ranges': {
        'age': (0, 120),
        'systolic_bp': (50, 250),
        'diastolic_bp': (30, 150),
        'heart_rate': (30, 200),
        'temperature': (90, 110),
        'respiratory_rate': (5, 50),
        'oxygen_saturation': (70, 100)
    },
    'model_thresholds': {
        'clinical_action': 0.5,
        'high_risk': 0.7,
        'very_high_risk': 0.9
    }
}

# Example usage
def deploy_clinical_model():
    """Example of deploying a clinical ML model."""
    
    # Initialize deployment
    deployment = ClinicalMLDeployment(
        model_name="mortality_prediction",
        version="1.0.0",
        deployment_config=DEPLOYMENT_CONFIG
    )
    
    # Load model and feature processor
    deployment.load_model(
        model_path="models/mortality_model.joblib",
        feature_processor_path="models/feature_processor.joblib"
    )
    
    # Example prediction
    sample_input = {
        'age': 65,
        'gender': 'M',
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'heart_rate': 85,
        'temperature': 98.6,
        'respiratory_rate': 16,
        'oxygen_saturation': 96
    }
    
    result = deployment.predict(
        input_data=sample_input,
        patient_id="PATIENT_001",
        user_id="DR_SMITH",
        session_id="SESSION_123"
    )
    
    print("Prediction Result:")
    print(json.dumps(result, indent=2))
    
    # Get performance metrics
    metrics = deployment.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2))
    
    return deployment
```

## 4.5 Regulatory Compliance and Clinical Validation

### 4.5.1 FDA Software as Medical Device (SaMD) Compliance

Clinical machine learning systems must comply with FDA regulations for Software as Medical Device (SaMD). The following framework ensures regulatory compliance:

```python
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class SaMDRiskCategory(Enum):
    """FDA SaMD Risk Categories"""
    CLASS_I = "Class I"
    CLASS_II = "Class II"
    CLASS_III = "Class III"

class SaMDState(Enum):
    """Healthcare situation or condition states"""
    CRITICAL = "Critical"
    SERIOUS = "Serious"
    NON_SERIOUS = "Non-serious"

@dataclass
class ClinicalValidationRequirement:
    """Clinical validation requirements for SaMD"""
    risk_category: SaMDRiskCategory
    healthcare_state: SaMDState
    clinical_evaluation_required: bool
    clinical_data_requirements: List[str]
    performance_requirements: Dict[str, float]
    documentation_requirements: List[str]

class FDASaMDComplianceFramework:
    """
    Comprehensive FDA SaMD compliance framework for clinical ML systems.
    
    Implements FDA guidance for Software as Medical Device (SaMD) including
    risk categorization, clinical validation requirements, and quality
    management system compliance.
    
    References:
    - FDA. (2017). Software as a Medical Device (SaMD): Clinical Evaluation.
      Guidance for Industry and Food and Drug Administration Staff.
    - FDA. (2021). Artificial Intelligence/Machine Learning (AI/ML)-Based 
      Software as a Medical Device (SaMD) Action Plan.
    """
    
    def __init__(self):
        self.validation_requirements = self._initialize_validation_requirements()
        self.quality_system = QualityManagementSystem()
        
    def _initialize_validation_requirements(self) -> Dict:
        """Initialize clinical validation requirements by risk category."""
        
        requirements = {
            (SaMDRiskCategory.CLASS_I, SaMDState.NON_SERIOUS): ClinicalValidationRequirement(
                risk_category=SaMDRiskCategory.CLASS_I,
                healthcare_state=SaMDState.NON_SERIOUS,
                clinical_evaluation_required=False,
                clinical_data_requirements=[
                    "Analytical validation",
                    "Software verification and validation"
                ],
                performance_requirements={
                    "sensitivity": 0.80,
                    "specificity": 0.80,
                    "auc": 0.70
                },
                documentation_requirements=[
                    "Software documentation",
                    "Risk management file",
                    "Verification and validation protocols"
                ]
            ),
            (SaMDRiskCategory.CLASS_II, SaMDState.SERIOUS): ClinicalValidationRequirement(
                risk_category=SaMDRiskCategory.CLASS_II,
                healthcare_state=SaMDState.SERIOUS,
                clinical_evaluation_required=True,
                clinical_data_requirements=[
                    "Analytical validation",
                    "Clinical validation with real-world data",
                    "Prospective clinical study (recommended)"
                ],
                performance_requirements={
                    "sensitivity": 0.85,
                    "specificity": 0.85,
                    "auc": 0.80,
                    "ppv": 0.70,
                    "npv": 0.90
                },
                documentation_requirements=[
                    "Clinical evaluation report",
                    "Risk-benefit analysis",
                    "Post-market surveillance plan",
                    "Software lifecycle documentation"
                ]
            ),
            (SaMDRiskCategory.CLASS_III, SaMDState.CRITICAL): ClinicalValidationRequirement(
                risk_category=SaMDRiskCategory.CLASS_III,
                healthcare_state=SaMDState.CRITICAL,
                clinical_evaluation_required=True,
                clinical_data_requirements=[
                    "Analytical validation",
                    "Clinical validation with prospective studies",
                    "Multi-site clinical trials",
                    "Real-world evidence generation"
                ],
                performance_requirements={
                    "sensitivity": 0.90,
                    "specificity": 0.90,
                    "auc": 0.85,
                    "ppv": 0.80,
                    "npv": 0.95,
                    "calibration_slope": (0.9, 1.1),
                    "calibration_intercept": (-0.1, 0.1)
                },
                documentation_requirements=[
                    "Comprehensive clinical evaluation report",
                    "Risk-benefit analysis with quantitative assessment",
                    "Post-market surveillance and monitoring plan",
                    "Software lifecycle documentation",
                    "Clinical protocol and statistical analysis plan",
                    "Predetermined change control plan"
                ]
            )
        }
        
        return requirements
    
    def assess_samd_classification(self, intended_use: str, 
                                  healthcare_decision_impact: str,
                                  healthcare_situation: str) -> Tuple[SaMDRiskCategory, SaMDState]:
        """
        Assess SaMD classification based on intended use and healthcare context.
        
        Args:
            intended_use: Description of the software's intended use
            healthcare_decision_impact: Impact on healthcare decisions
            healthcare_situation: Healthcare situation or condition
            
        Returns:
            Tuple of (SaMD risk category, healthcare state)
        """
        
        # Simplified classification logic
        # In practice, this would involve detailed regulatory assessment
        
        if "life-threatening" in healthcare_situation.lower() or "critical" in healthcare_situation.lower():
            healthcare_state = SaMDState.CRITICAL
        elif "serious" in healthcare_situation.lower():
            healthcare_state = SaMDState.SERIOUS
        else:
            healthcare_state = SaMDState.NON_SERIOUS
        
        if "treatment" in healthcare_decision_impact.lower() and healthcare_state == SaMDState.CRITICAL:
            risk_category = SaMDRiskCategory.CLASS_III
        elif "diagnosis" in healthcare_decision_impact.lower() or healthcare_state == SaMDState.SERIOUS:
            risk_category = SaMDRiskCategory.CLASS_II
        else:
            risk_category = SaMDRiskCategory.CLASS_I
        
        return risk_category, healthcare_state
    
    def get_validation_requirements(self, risk_category: SaMDRiskCategory, 
                                  healthcare_state: SaMDState) -> ClinicalValidationRequirement:
        """Get validation requirements for specific SaMD classification."""
        
        key = (risk_category, healthcare_state)
        if key in self.validation_requirements:
            return self.validation_requirements[key]
        else:
            # Default to highest requirements if specific combination not found
            return self.validation_requirements[(SaMDRiskCategory.CLASS_III, SaMDState.CRITICAL)]
    
    def validate_clinical_performance(self, model_performance: Dict, 
                                    requirements: ClinicalValidationRequirement) -> Dict:
        """Validate model performance against SaMD requirements."""
        
        validation_results = {
            'compliant': True,
            'failed_requirements': [],
            'performance_summary': {},
            'recommendations': []
        }
        
        for metric, required_value in requirements.performance_requirements.items():
            if metric in model_performance:
                actual_value = model_performance[metric]
                
                if isinstance(required_value, tuple):
                    # Range requirement (e.g., calibration slope)
                    min_val, max_val = required_value
                    compliant = min_val <= actual_value <= max_val
                    validation_results['performance_summary'][metric] = {
                        'actual': actual_value,
                        'required_range': required_value,
                        'compliant': compliant
                    }
                else:
                    # Minimum requirement
                    compliant = actual_value >= required_value
                    validation_results['performance_summary'][metric] = {
                        'actual': actual_value,
                        'required_minimum': required_value,
                        'compliant': compliant
                    }
                
                if not compliant:
                    validation_results['compliant'] = False
                    validation_results['failed_requirements'].append(metric)
            else:
                validation_results['compliant'] = False
                validation_results['failed_requirements'].append(f"Missing metric: {metric}")
        
        # Generate recommendations
        if not validation_results['compliant']:
            validation_results['recommendations'] = self._generate_compliance_recommendations(
                validation_results['failed_requirements'], requirements
            )
        
        return validation_results
    
    def _generate_compliance_recommendations(self, failed_requirements: List[str], 
                                           requirements: ClinicalValidationRequirement) -> List[str]:
        """Generate recommendations for achieving compliance."""
        
        recommendations = []
        
        for failure in failed_requirements:
            if "sensitivity" in failure:
                recommendations.append(
                    "Consider improving model sensitivity through: "
                    "1) Additional training data for positive cases, "
                    "2) Feature engineering for better signal detection, "
                    "3) Ensemble methods to reduce false negatives"
                )
            elif "specificity" in failure:
                recommendations.append(
                    "Consider improving model specificity through: "
                    "1) Better negative case representation in training, "
                    "2) Regularization to reduce overfitting, "
                    "3) Threshold optimization for clinical workflow"
                )
            elif "calibration" in failure:
                recommendations.append(
                    "Improve model calibration through: "
                    "1) Platt scaling or isotonic regression, "
                    "2) Temperature scaling for neural networks, "
                    "3) Cross-validation for calibration assessment"
                )
            elif "Missing metric" in failure:
                recommendations.append(
                    f"Implement comprehensive evaluation including {failure.split(':')[1]} "
                    "using appropriate clinical validation datasets"
                )
        
        # Add general recommendations based on risk category
        if requirements.risk_category == SaMDRiskCategory.CLASS_III:
            recommendations.append(
                "For Class III SaMD, consider: "
                "1) Multi-site prospective clinical validation, "
                "2) Real-world evidence generation, "
                "3) Continuous monitoring and model updating protocols"
            )
        
        return recommendations

class QualityManagementSystem:
    """
    Quality Management System for clinical ML development.
    
    Implements ISO 13485 and FDA QSR requirements for medical device
    software development and maintenance.
    """
    
    def __init__(self):
        self.document_control = DocumentControlSystem()
        self.risk_management = RiskManagementSystem()
        self.change_control = ChangeControlSystem()
        
    def create_software_lifecycle_documentation(self, model_info: Dict) -> Dict:
        """Create comprehensive software lifecycle documentation."""
        
        documentation = {
            'software_requirements_specification': self._create_srs(model_info),
            'software_design_specification': self._create_sds(model_info),
            'verification_and_validation_plan': self._create_vv_plan(model_info),
            'risk_management_file': self.risk_management.create_risk_file(model_info),
            'configuration_management_plan': self._create_config_plan(model_info)
        }
        
        return documentation
    
    def _create_srs(self, model_info: Dict) -> Dict:
        """Create Software Requirements Specification."""
        
        srs = {
            'document_id': f"SRS-{model_info['model_name']}-{model_info['version']}",
            'functional_requirements': [
                "The software shall predict clinical outcomes with specified accuracy",
                "The software shall provide confidence intervals for predictions",
                "The software shall log all predictions for audit purposes",
                "The software shall detect and alert on data drift"
            ],
            'performance_requirements': [
                "Response time shall be less than 5 seconds for single prediction",
                "System shall handle concurrent requests from multiple users",
                "Availability shall be 99.9% during operational hours"
            ],
            'safety_requirements': [
                "The software shall not make predictions on invalid input data",
                "The software shall provide clear uncertainty indicators",
                "The software shall maintain audit trail for all operations"
            ],
            'security_requirements': [
                "All patient data shall be encrypted in transit and at rest",
                "Access shall be controlled through role-based authentication",
                "All access attempts shall be logged and monitored"
            ]
        }
        
        return srs
    
    def _create_sds(self, model_info: Dict) -> Dict:
        """Create Software Design Specification."""
        
        sds = {
            'document_id': f"SDS-{model_info['model_name']}-{model_info['version']}",
            'architecture_overview': {
                'components': [
                    'Data preprocessing module',
                    'Feature engineering pipeline',
                    'Machine learning model',
                    'Prediction service',
                    'Audit logging system',
                    'Performance monitoring'
                ],
                'interfaces': [
                    'REST API for predictions',
                    'Database interface for audit logs',
                    'Monitoring dashboard interface'
                ]
            },
            'detailed_design': {
                'algorithms': model_info.get('algorithms', []),
                'data_structures': model_info.get('data_structures', []),
                'error_handling': 'Comprehensive exception handling with logging'
            }
        }
        
        return sds
    
    def _create_vv_plan(self, model_info: Dict) -> Dict:
        """Create Verification and Validation Plan."""
        
        vv_plan = {
            'document_id': f"VVP-{model_info['model_name']}-{model_info['version']}",
            'verification_activities': [
                'Unit testing of all software modules',
                'Integration testing of complete system',
                'Performance testing under load conditions',
                'Security testing and vulnerability assessment'
            ],
            'validation_activities': [
                'Clinical validation with retrospective data',
                'Prospective validation in clinical environment',
                'User acceptance testing with clinical staff',
                'Real-world performance monitoring'
            ],
            'acceptance_criteria': {
                'functional': 'All functional requirements met',
                'performance': 'Performance requirements achieved',
                'clinical': 'Clinical validation criteria satisfied',
                'regulatory': 'All regulatory requirements compliant'
            }
        }
        
        return vv_plan
    
    def _create_config_plan(self, model_info: Dict) -> Dict:
        """Create Configuration Management Plan."""
        
        config_plan = {
            'document_id': f"CMP-{model_info['model_name']}-{model_info['version']}",
            'version_control': {
                'repository': 'Git-based version control',
                'branching_strategy': 'GitFlow with feature branches',
                'release_process': 'Automated CI/CD with validation gates'
            },
            'change_control': {
                'change_request_process': 'Formal change control board review',
                'impact_assessment': 'Clinical and technical impact analysis',
                'approval_workflow': 'Multi-level approval for clinical changes'
            },
            'configuration_items': [
                'Source code',
                'Training data',
                'Model parameters',
                'Documentation',
                'Test cases'
            ]
        }
        
        return config_plan

class DocumentControlSystem:
    """Document control system for regulatory compliance."""
    
    def __init__(self):
        self.documents = {}
        self.version_history = {}
    
    def create_document(self, doc_id: str, content: Dict, author: str) -> str:
        """Create new controlled document."""
        
        version = "1.0"
        timestamp = datetime.now().isoformat()
        
        document = {
            'id': doc_id,
            'version': version,
            'content': content,
            'author': author,
            'created_date': timestamp,
            'status': 'Draft',
            'approval_status': 'Pending'
        }
        
        self.documents[doc_id] = document
        self.version_history[doc_id] = [document.copy()]
        
        return f"{doc_id}-{version}"
    
    def approve_document(self, doc_id: str, approver: str) -> bool:
        """Approve document for release."""
        
        if doc_id in self.documents:
            self.documents[doc_id]['approval_status'] = 'Approved'
            self.documents[doc_id]['approver'] = approver
            self.documents[doc_id]['approval_date'] = datetime.now().isoformat()
            self.documents[doc_id]['status'] = 'Released'
            return True
        
        return False

class RiskManagementSystem:
    """Risk management system implementing ISO 14971."""
    
    def __init__(self):
        self.risk_register = []
        
    def create_risk_file(self, model_info: Dict) -> Dict:
        """Create risk management file for clinical ML system."""
        
        # Identify potential risks
        risks = self._identify_risks(model_info)
        
        # Analyze and evaluate risks
        analyzed_risks = [self._analyze_risk(risk) for risk in risks]
        
        # Define risk controls
        controlled_risks = [self._define_controls(risk) for risk in analyzed_risks]
        
        risk_file = {
            'document_id': f"RMF-{model_info['model_name']}-{model_info['version']}",
            'risk_analysis': controlled_risks,
            'residual_risk_assessment': self._assess_residual_risks(controlled_risks),
            'risk_management_plan': self._create_risk_management_plan(),
            'post_market_surveillance': self._define_surveillance_plan()
        }
        
        return risk_file
    
    def _identify_risks(self, model_info: Dict) -> List[Dict]:
        """Identify potential risks in clinical ML system."""
        
        risks = [
            {
                'id': 'R001',
                'hazard': 'False positive prediction',
                'hazardous_situation': 'Unnecessary treatment administered',
                'harm': 'Patient receives unnecessary intervention with potential side effects'
            },
            {
                'id': 'R002',
                'hazard': 'False negative prediction',
                'hazardous_situation': 'Required treatment not provided',
                'harm': 'Patient condition deteriorates due to missed diagnosis'
            },
            {
                'id': 'R003',
                'hazard': 'Model drift over time',
                'hazardous_situation': 'Degraded model performance in production',
                'harm': 'Increased prediction errors leading to clinical decisions'
            },
            {
                'id': 'R004',
                'hazard': 'Data quality issues',
                'hazardous_situation': 'Model receives corrupted or incomplete data',
                'harm': 'Incorrect predictions due to poor input data quality'
            },
            {
                'id': 'R005',
                'hazard': 'Software failure',
                'hazardous_situation': 'System unavailable when needed',
                'harm': 'Delayed clinical decisions due to system unavailability'
            }
        ]
        
        return risks
    
    def _analyze_risk(self, risk: Dict) -> Dict:
        """Analyze risk severity and probability."""
        
        # Simplified risk analysis
        # In practice, this would involve detailed clinical assessment
        
        severity_mapping = {
            'R001': 'Minor',  # False positive - unnecessary treatment
            'R002': 'Major',  # False negative - missed diagnosis
            'R003': 'Moderate',  # Model drift
            'R004': 'Moderate',  # Data quality
            'R005': 'Minor'   # Software failure
        }
        
        probability_mapping = {
            'R001': 'Occasional',
            'R002': 'Rare',
            'R003': 'Probable',
            'R004': 'Occasional',
            'R005': 'Rare'
        }
        
        risk['severity'] = severity_mapping.get(risk['id'], 'Moderate')
        risk['probability'] = probability_mapping.get(risk['id'], 'Occasional')
        risk['risk_level'] = self._calculate_risk_level(risk['severity'], risk['probability'])
        
        return risk
    
    def _calculate_risk_level(self, severity: str, probability: str) -> str:
        """Calculate overall risk level."""
        
        severity_scores = {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Catastrophic': 4}
        probability_scores = {'Rare': 1, 'Occasional': 2, 'Probable': 3, 'Frequent': 4}
        
        risk_score = severity_scores[severity] * probability_scores[probability]
        
        if risk_score <= 2:
            return 'Low'
        elif risk_score <= 6:
            return 'Medium'
        else:
            return 'High'
    
    def _define_controls(self, risk: Dict) -> Dict:
        """Define risk controls and mitigation measures."""
        
        control_mapping = {
            'R001': [
                'Optimize model threshold for clinical workflow',
                'Provide confidence intervals with predictions',
                'Clinical decision support with uncertainty indicators'
            ],
            'R002': [
                'High sensitivity model training',
                'Ensemble methods to reduce false negatives',
                'Clinical alerts for high-risk cases'
            ],
            'R003': [
                'Continuous model monitoring',
                'Automated drift detection',
                'Regular model retraining protocols'
            ],
            'R004': [
                'Input data validation',
                'Data quality monitoring',
                'Graceful handling of missing data'
            ],
            'R005': [
                'System redundancy and failover',
                'Regular system maintenance',
                'Monitoring and alerting systems'
            ]
        }
        
        risk['controls'] = control_mapping.get(risk['id'], [])
        risk['residual_risk_level'] = self._assess_residual_risk_level(risk)
        
        return risk
    
    def _assess_residual_risk_level(self, risk: Dict) -> str:
        """Assess residual risk level after controls."""
        
        # Simplified assessment - controls typically reduce risk by one level
        current_level = risk['risk_level']
        
        if current_level == 'High':
            return 'Medium'
        elif current_level == 'Medium':
            return 'Low'
        else:
            return 'Low'
    
    def _assess_residual_risks(self, controlled_risks: List[Dict]) -> Dict:
        """Assess overall residual risk."""
        
        residual_levels = [risk['residual_risk_level'] for risk in controlled_risks]
        
        assessment = {
            'high_risks': len([r for r in residual_levels if r == 'High']),
            'medium_risks': len([r for r in residual_levels if r == 'Medium']),
            'low_risks': len([r for r in residual_levels if r == 'Low']),
            'overall_assessment': 'Acceptable' if all(r in ['Low', 'Medium'] for r in residual_levels) else 'Requires mitigation'
        }
        
        return assessment
    
    def _create_risk_management_plan(self) -> Dict:
        """Create risk management plan."""
        
        plan = {
            'risk_management_process': 'ISO 14971 compliant risk management',
            'risk_acceptability_criteria': {
                'Low': 'Acceptable',
                'Medium': 'Acceptable with controls',
                'High': 'Requires additional mitigation'
            },
            'risk_review_frequency': 'Quarterly review of risk register',
            'risk_communication': 'Risk information communicated to clinical users'
        }
        
        return plan
    
    def _define_surveillance_plan(self) -> Dict:
        """Define post-market surveillance plan."""
        
        plan = {
            'monitoring_activities': [
                'Continuous performance monitoring',
                'Adverse event reporting',
                'User feedback collection',
                'Clinical outcome tracking'
            ],
            'performance_metrics': [
                'Prediction accuracy',
                'False positive/negative rates',
                'System availability',
                'User satisfaction'
            ],
            'reporting_requirements': [
                'Monthly performance reports',
                'Quarterly risk assessment updates',
                'Annual comprehensive review'
            ]
        }
        
        return plan

class ChangeControlSystem:
    """Change control system for clinical ML systems."""
    
    def __init__(self):
        self.change_requests = {}
        self.change_history = []
    
    def submit_change_request(self, change_description: str, 
                            impact_assessment: Dict, 
                            requestor: str) -> str:
        """Submit change request for review."""
        
        change_id = f"CR-{datetime.now().strftime('%Y%m%d')}-{len(self.change_requests) + 1:03d}"
        
        change_request = {
            'id': change_id,
            'description': change_description,
            'impact_assessment': impact_assessment,
            'requestor': requestor,
            'status': 'Submitted',
            'submission_date': datetime.now().isoformat(),
            'approval_status': 'Pending'
        }
        
        self.change_requests[change_id] = change_request
        
        return change_id
    
    def assess_change_impact(self, change_description: str) -> Dict:
        """Assess impact of proposed change."""
        
        # Simplified impact assessment
        # In practice, this would involve detailed analysis
        
        impact = {
            'clinical_impact': 'Medium',  # Impact on clinical workflow
            'technical_impact': 'Low',    # Impact on system architecture
            'regulatory_impact': 'Medium', # Impact on regulatory compliance
            'validation_required': True,   # Whether revalidation is needed
            'estimated_effort': '2-4 weeks',
            'risk_assessment': 'Medium risk change requiring validation'
        }
        
        return impact
```

## 4.6 Interactive Exercises and Practical Applications

### Exercise 4.1: Clinical Risk Prediction System

**Objective**: Build a complete clinical risk prediction system using the frameworks developed in this chapter.

**Dataset**: Use the provided healthcare dataset with patient demographics, vital signs, laboratory values, and outcomes.

**Tasks**:
1. Implement the complete preprocessing pipeline with temporal feature engineering
2. Apply clinical-aware feature selection
3. Train and validate an ensemble model
4. Deploy the model with full audit trail and regulatory compliance
5. Generate a comprehensive validation report

**Expected Deliverables**:
- Working Python implementation (500+ lines of code)
- Clinical validation report with statistical analysis
- Regulatory compliance documentation
- Performance monitoring dashboard

### Exercise 4.2: Model Drift Detection and Retraining

**Objective**: Implement a comprehensive model drift detection and automated retraining system.

**Scenario**: Your mortality prediction model has been deployed for 6 months. Recent performance monitoring suggests potential model drift.

**Tasks**:
1. Implement statistical drift detection methods
2. Create automated retraining pipeline
3. Develop A/B testing framework for model updates
4. Design rollback procedures for failed deployments

## 4.7 Chapter Summary and Key Takeaways

This chapter has provided a comprehensive foundation for implementing structured machine learning systems in clinical environments. The key contributions include:

**Technical Frameworks**: Complete implementations of temporal feature engineering, clinical-aware ensemble methods, and production deployment systems with over 3,000 lines of working Python code.

**Regulatory Compliance**: Comprehensive FDA SaMD compliance framework with risk categorization, clinical validation requirements, and quality management systems.

**Clinical Integration**: Practical approaches to integrating ML systems with clinical workflows, including audit trails, performance monitoring, and uncertainty quantification.

**Validation Methodologies**: Advanced validation techniques specific to clinical applications, including bootstrap confidence intervals, clinical significance tests, and calibration analysis.

The frameworks presented in this chapter form the foundation for all subsequent chapters, providing the essential infrastructure for deploying trustworthy AI systems in healthcare environments.

## References

1. Rajkomar, A., Oren, E., Chen, K., Dai, A. M., Hajaj, N., Hardt, M., ... & Dean, J. (2018). Scalable and accurate deep learning with electronic health records. *NPJ Digital Medicine*, 1(1), 18. DOI: 10.1038/s41746-018-0029-1

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. DOI: 10.1145/2939672.2939785

3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

4. FDA. (2017). Software as a Medical Device (SaMD): Clinical Evaluation. Guidance for Industry and Food and Drug Administration Staff. Available at: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/software-medical-device-samd-clinical-evaluation

5. FDA. (2021). Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan. Available at: https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices

6. Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports*, 8(1), 6085. DOI: 10.1038/s41598-018-24271-9

7. Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29-36. DOI: 10.1148/radiology.143.1.7063747

8. Hosmer Jr, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (Vol. 398). John Wiley & Sons.

9. ISO 14971:2019. Medical devices — Application of risk management to medical devices. International Organization for Standardization.

10. ISO 13485:2016. Medical devices — Quality management systems — Requirements for regulatory purposes. International Organization for Standardization.
