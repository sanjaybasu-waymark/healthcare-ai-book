# Chapter 4: Advanced Risk Prediction in Healthcare - Enhanced Edition

## Learning Objectives

By the end of this chapter, readers will be able to:
- Implement state-of-the-art risk prediction models for healthcare applications
- Apply proper train/validation/test splits with cross-validation for clinical data
- Develop heterogeneous treatment effect estimation models for personalized medicine
- Address equity concerns in risk prediction following Obermeyer's framework
- Build time-to-event and acuity prediction models
- Bridge traditional actuarial methods with modern machine learning approaches

## Introduction

Risk prediction forms the cornerstone of modern healthcare AI, enabling clinicians to identify high-risk patients, optimize treatment strategies, and improve population health outcomes. This chapter provides comprehensive coverage of advanced risk prediction methodologies, drawing from both traditional actuarial science and cutting-edge machine learning approaches.

The integration of machine learning into healthcare risk prediction has revolutionized our ability to process complex, high-dimensional data while maintaining clinical interpretability. However, this advancement comes with significant responsibilities regarding equity, fairness, and clinical validity that must be carefully addressed.

## Mathematical Foundations

### Risk Prediction Framework

Let $X \in \mathbb{R}^d$ represent patient features, $Y \in \{0,1\}$ represent binary outcomes (e.g., mortality, readmission), and $T \in \mathbb{R}^+$ represent time-to-event outcomes. Our goal is to learn a function $f: X \rightarrow [0,1]$ that accurately predicts risk while maintaining fairness across protected attributes $A$.

The fundamental risk prediction equation can be expressed as:

$$P(Y = 1 | X = x) = f(x; \theta)$$

where $\theta$ represents model parameters learned from training data.

### Evaluation Metrics

For binary classification tasks, we employ multiple complementary metrics:

**Discrimination Metrics:**
- Area Under the ROC Curve (AUC-ROC): $\text{AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$
- Area Under the Precision-Recall Curve (AUC-PR): $\text{AUC-PR} = \int_0^1 \text{Precision}(r) \, dr$
- C-statistic for time-to-event models

**Calibration Metrics:**
- Brier Score: $\text{BS} = \frac{1}{n} \sum_{i=1}^n (p_i - y_i)^2$
- Hosmer-Lemeshow test statistic
- Calibration slope and intercept

## Complete Implementation: Advanced Risk Prediction System

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, 
    TimeSeriesSplit, cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import (
    LogisticRegression, ElasticNet, Ridge, Lasso
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    calibration_curve, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskPredictor:
    """
    Comprehensive risk prediction system for healthcare applications.
    
    Implements best practices for:
    - Proper data splitting with temporal considerations
    - Cross-validation strategies for healthcare data
    - Multiple model architectures with ensemble methods
    - Comprehensive evaluation and calibration
    - Fairness assessment across demographic groups
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}
        self.fairness_results = {}
        
    def prepare_data(self, df, target_col, protected_attrs=None, 
                    time_col=None, event_col=None):
        """
        Prepare healthcare data with proper handling of temporal structure.
        
        Args:
            df: Input dataframe
            target_col: Target variable column name
            protected_attrs: List of protected attribute columns
            time_col: Time column for temporal splitting
            event_col: Event indicator for survival analysis
        """
        self.target_col = target_col
        self.protected_attrs = protected_attrs or []
        self.time_col = time_col
        self.event_col = event_col
        
        # Handle missing values with clinical considerations
        df_processed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Handle missing values
        for col in numeric_cols:
            if col not in [target_col, time_col, event_col]:
                # Use median imputation for clinical variables
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
        
        for col in categorical_cols:
            if col not in [target_col]:
                # Use mode imputation for categorical variables
                mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
                df_processed[col].fillna(mode_val, inplace=True)
        
        # Encode categorical variables
        self.label_encoders = {}
        for col in categorical_cols:
            if col != target_col:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        self.processed_data = df_processed
        return df_processed
    
    def temporal_split(self, df, test_size=0.2, val_size=0.2):
        """
        Implement temporal splitting for healthcare data to prevent data leakage.
        """
        if self.time_col is None:
            # Standard random split if no temporal information
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state,
                stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size),
                random_state=self.random_state, stratify=y_temp
            )
        else:
            # Temporal split based on time column
            df_sorted = df.sort_values(self.time_col)
            n = len(df_sorted)
            
            train_end = int(n * (1 - test_size - val_size))
            val_end = int(n * (1 - test_size))
            
            train_data = df_sorted.iloc[:train_end]
            val_data = df_sorted.iloc[train_end:val_end]
            test_data = df_sorted.iloc[val_end:]
            
            X_train = train_data.drop(columns=[self.target_col])
            y_train = train_data[self.target_col]
            X_val = val_data.drop(columns=[self.target_col])
            y_val = val_data[self.target_col]
            X_test = test_data.drop(columns=[self.target_col])
            y_test = test_data[self.target_col]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def cross_validation_strategy(self, X, y, cv_type='stratified', n_splits=5):
        """
        Implement appropriate cross-validation strategy for healthcare data.
        """
        if cv_type == 'stratified':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                               random_state=self.random_state)
        elif cv_type == 'temporal' and self.time_col is not None:
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=self.random_state)
        
        return cv
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple model architectures with proper regularization.
        """
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['standard'] = scaler
        
        models_to_train = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000,
                penalty='elasticnet', solver='saga', l1_ratio=0.5
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state,
                max_depth=10, min_samples_split=20, min_samples_leaf=10
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state,
                max_depth=6, learning_rate=0.1, subsample=0.8
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, random_state=self.random_state,
                max_depth=6, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100, random_state=self.random_state,
                max_depth=6, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, verbose=-1
            )
        }
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                # Calibrate the model
                calibrated_model = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
                calibrated_model.fit(X_train_scaled, y_train)
                self.models[name] = calibrated_model
            else:
                model.fit(X_train, y_train)
                # Calibrate the model
                calibrated_model = CalibratedClassifierCV(
                    model, method='isotonic', cv=3
                )
                calibrated_model.fit(X_train, y_train)
                self.models[name] = calibrated_model
    
    def evaluate_models(self, X_test, y_test):
        """
        Comprehensive model evaluation with clinical metrics.
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Get predictions
            if name == 'logistic_regression':
                X_test_eval = self.scalers['standard'].transform(X_test)
            else:
                X_test_eval = X_test
            
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
            y_pred = model.predict(X_test_eval)
            
            # Calculate metrics
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            auc_pr = average_precision_score(y_test, y_pred_proba)
            brier_score = brier_score_loss(y_test, y_pred_proba)
            
            # Calibration assessment
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_pred_proba, n_bins=10
            )
            
            # Calculate calibration slope and intercept
            cal_slope, cal_intercept, _, _, _ = stats.linregress(
                y_pred_proba, y_test
            )
            
            results[name] = {
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'brier_score': brier_score,
                'calibration_slope': cal_slope,
                'calibration_intercept': cal_intercept,
                'predictions': y_pred_proba,
                'binary_predictions': y_pred
            }
        
        self.evaluation_results = results
        return results
    
    def assess_fairness(self, X_test, y_test):
        """
        Assess fairness across protected attributes following Obermeyer's framework.
        """
        if not self.protected_attrs:
            print("No protected attributes specified for fairness assessment.")
            return {}
        
        fairness_results = {}
        
        for attr in self.protected_attrs:
            if attr not in X_test.columns:
                continue
                
            attr_values = X_test[attr].unique()
            fairness_results[attr] = {}
            
            for model_name, model in self.models.items():
                if model_name == 'logistic_regression':
                    X_test_eval = self.scalers['standard'].transform(X_test)
                else:
                    X_test_eval = X_test
                
                y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
                
                group_metrics = {}
                for value in attr_values:
                    mask = X_test[attr] == value
                    if mask.sum() > 10:  # Minimum group size
                        group_auc = roc_auc_score(y_test[mask], y_pred_proba[mask])
                        group_metrics[f'group_{value}'] = {
                            'auc': group_auc,
                            'mean_prediction': y_pred_proba[mask].mean(),
                            'positive_rate': y_test[mask].mean(),
                            'size': mask.sum()
                        }
                
                # Calculate fairness metrics
                aucs = [metrics['auc'] for metrics in group_metrics.values()]
                if len(aucs) > 1:
                    auc_difference = max(aucs) - min(aucs)
                    auc_ratio = min(aucs) / max(aucs) if max(aucs) > 0 else 0
                else:
                    auc_difference = 0
                    auc_ratio = 1
                
                fairness_results[attr][model_name] = {
                    'group_metrics': group_metrics,
                    'auc_difference': auc_difference,
                    'auc_ratio': auc_ratio
                }
        
        self.fairness_results = fairness_results
        return fairness_results
    
    def plot_evaluation_results(self):
        """
        Create comprehensive evaluation plots.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # ROC Curves
        ax = axes[0, 0]
        for name, results in self.evaluation_results.items():
            y_test = list(self.evaluation_results.values())[0].get('y_test', [])
            if len(y_test) == 0:
                continue
            fpr, tpr, _ = roc_curve(y_test, results['predictions'])
            ax.plot(fpr, tpr, label=f"{name} (AUC: {results['auc_roc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        ax = axes[0, 1]
        for name, results in self.evaluation_results.items():
            if len(y_test) == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_test, results['predictions'])
            ax.plot(recall, precision, label=f"{name} (AUC: {results['auc_pr']:.3f})")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calibration Plot
        ax = axes[0, 2]
        for name, results in self.evaluation_results.items():
            if len(y_test) == 0:
                continue
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, results['predictions'], n_bins=10
            )
            ax.plot(mean_predicted_value, fraction_of_positives, 'o-',
                   label=f"{name}")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Model Performance Comparison
        ax = axes[1, 0]
        metrics = ['auc_roc', 'auc_pr', 'brier_score']
        model_names = list(self.evaluation_results.keys())
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [self.evaluation_results[name][metric] for name in model_names]
            ax.bar(x + i*width, values, width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Feature Importance (for tree-based models)
        if 'random_forest' in self.models:
            ax = axes[1, 1]
            model = self.models['random_forest'].base_estimator
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
                indices = np.argsort(importances)[::-1][:10]
                
                ax.bar(range(len(indices)), importances[indices])
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_title('Top 10 Feature Importances (Random Forest)')
                ax.set_xticks(range(len(indices)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
        
        # Fairness Assessment
        if self.fairness_results:
            ax = axes[1, 2]
            attr = list(self.fairness_results.keys())[0]
            model_names = list(self.fairness_results[attr].keys())
            auc_diffs = [self.fairness_results[attr][name]['auc_difference'] 
                        for name in model_names]
            
            ax.bar(model_names, auc_diffs)
            ax.set_xlabel('Models')
            ax.set_ylabel('AUC Difference')
            ax.set_title(f'Fairness Assessment ({attr})')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """
        Generate comprehensive evaluation report.
        """
        report = []
        report.append("# Healthcare Risk Prediction Model Evaluation Report\n")
        
        # Model Performance Summary
        report.append("## Model Performance Summary\n")
        report.append("| Model | AUC-ROC | AUC-PR | Brier Score | Cal. Slope |")
        report.append("|-------|---------|--------|-------------|------------|")
        
        for name, results in self.evaluation_results.items():
            report.append(f"| {name} | {results['auc_roc']:.3f} | "
                         f"{results['auc_pr']:.3f} | {results['brier_score']:.3f} | "
                         f"{results['calibration_slope']:.3f} |")
        
        # Fairness Assessment
        if self.fairness_results:
            report.append("\n## Fairness Assessment\n")
            for attr, attr_results in self.fairness_results.items():
                report.append(f"### Protected Attribute: {attr}\n")
                report.append("| Model | AUC Difference | AUC Ratio |")
                report.append("|-------|----------------|-----------|")
                
                for model_name, metrics in attr_results.items():
                    report.append(f"| {model_name} | {metrics['auc_difference']:.3f} | "
                                 f"{metrics['auc_ratio']:.3f} |")
        
        # Clinical Recommendations
        report.append("\n## Clinical Recommendations\n")
        
        best_model = max(self.evaluation_results.items(), 
                        key=lambda x: x[1]['auc_roc'])
        report.append(f"**Recommended Model:** {best_model[0]} "
                     f"(AUC-ROC: {best_model[1]['auc_roc']:.3f})\n")
        
        report.append("**Key Considerations:**")
        report.append("- Monitor model performance over time for drift")
        report.append("- Regularly assess fairness across demographic groups")
        report.append("- Validate predictions with clinical expertise")
        report.append("- Consider model interpretability for clinical decision-making")
        
        return "\n".join(report)

# Heterogeneous Treatment Effect Estimation
class HeterogeneousTreatmentEffects:
    """
    Implementation of heterogeneous treatment effect estimation methods
    for personalized medicine applications.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
    
    def s_learner(self, X, T, Y):
        """
        S-Learner: Single model approach including treatment as a feature.
        """
        # Combine treatment with features
        X_combined = np.column_stack([X, T])
        
        # Train single model
        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        model.fit(X_combined, Y)
        
        # Estimate CATE by predicting with T=1 and T=0
        X_treated = np.column_stack([X, np.ones(X.shape[0])])
        X_control = np.column_stack([X, np.zeros(X.shape[0])])
        
        tau_hat = model.predict(X_treated) - model.predict(X_control)
        
        self.models['s_learner'] = model
        return tau_hat
    
    def t_learner(self, X, T, Y):
        """
        T-Learner: Separate models for treatment and control groups.
        """
        # Split data by treatment
        X_treated = X[T == 1]
        Y_treated = Y[T == 1]
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        
        # Train separate models
        model_treated = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        model_control = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        
        model_treated.fit(X_treated, Y_treated)
        model_control.fit(X_control, Y_control)
        
        # Estimate CATE
        tau_hat = model_treated.predict(X) - model_control.predict(X)
        
        self.models['t_learner'] = {'treated': model_treated, 'control': model_control}
        return tau_hat
    
    def x_learner(self, X, T, Y):
        """
        X-Learner: Cross-learning approach with propensity score weighting.
        """
        # Step 1: Initial outcome models
        X_treated = X[T == 1]
        Y_treated = Y[T == 1]
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        
        mu_0 = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        mu_1 = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        
        mu_0.fit(X_control, Y_control)
        mu_1.fit(X_treated, Y_treated)
        
        # Step 2: Imputed treatment effects
        D_treated = Y_treated - mu_0.predict(X_treated)
        D_control = mu_1.predict(X_control) - Y_control
        
        # Step 3: Treatment effect models
        tau_0 = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        tau_1 = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        
        tau_0.fit(X_control, D_control)
        tau_1.fit(X_treated, D_treated)
        
        # Step 4: Propensity score model
        propensity_model = LogisticRegression(random_state=self.random_state)
        propensity_model.fit(X, T)
        e_hat = propensity_model.predict_proba(X)[:, 1]
        
        # Step 5: Weighted combination
        tau_hat = e_hat * tau_0.predict(X) + (1 - e_hat) * tau_1.predict(X)
        
        self.models['x_learner'] = {
            'mu_0': mu_0, 'mu_1': mu_1,
            'tau_0': tau_0, 'tau_1': tau_1,
            'propensity': propensity_model
        }
        
        return tau_hat
    
    def doubly_robust_learner(self, X, T, Y):
        """
        Doubly robust estimation combining outcome regression and propensity scores.
        """
        # Propensity score model
        propensity_model = LogisticRegression(random_state=self.random_state)
        propensity_model.fit(X, T)
        e_hat = propensity_model.predict_proba(X)[:, 1]
        
        # Outcome regression models
        X_treated = X[T == 1]
        Y_treated = Y[T == 1]
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        
        mu_0 = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        mu_1 = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        
        mu_0.fit(X_control, Y_control)
        mu_1.fit(X_treated, Y_treated)
        
        # Doubly robust estimator
        mu_0_hat = mu_0.predict(X)
        mu_1_hat = mu_1.predict(X)
        
        # Clip propensity scores to avoid extreme weights
        e_hat_clipped = np.clip(e_hat, 0.01, 0.99)
        
        tau_hat = (mu_1_hat - mu_0_hat + 
                  T * (Y - mu_1_hat) / e_hat_clipped - 
                  (1 - T) * (Y - mu_0_hat) / (1 - e_hat_clipped))
        
        self.models['doubly_robust'] = {
            'mu_0': mu_0, 'mu_1': mu_1,
            'propensity': propensity_model
        }
        
        return tau_hat

# Time-to-Event Prediction
class SurvivalAnalysis:
    """
    Comprehensive survival analysis implementation for healthcare applications.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
    
    def random_survival_forest(self, X, y):
        """
        Random Survival Forest implementation.
        """
        # Convert to structured array format required by scikit-survival
        y_structured = np.array([(bool(event), time) for event, time in y],
                               dtype=[('event', bool), ('time', float)])
        
        # Train Random Survival Forest
        rsf = RandomSurvivalForest(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.random_state
        )
        
        rsf.fit(X, y_structured)
        
        # Calculate concordance index
        c_index = rsf.score(X, y_structured)
        
        self.models['random_survival_forest'] = rsf
        
        return {
            'model': rsf,
            'c_index': c_index,
            'feature_importances': rsf.feature_importances_
        }
    
    def cox_proportional_hazards(self, df, duration_col, event_col, feature_cols):
        """
        Cox Proportional Hazards model implementation.
        """
        # Prepare data for lifelines
        df_cox = df[feature_cols + [duration_col, event_col]].copy()
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col=duration_col, event_col=event_col)
        
        # Calculate concordance index
        c_index = concordance_index(df_cox[duration_col], 
                                   -cph.predict_partial_hazard(df_cox),
                                   df_cox[event_col])
        
        self.models['cox_ph'] = cph
        
        return {
            'model': cph,
            'c_index': c_index,
            'summary': cph.summary,
            'hazard_ratios': np.exp(cph.params_)
        }

# Example usage and demonstration
def demonstrate_risk_prediction():
    """
    Comprehensive demonstration of risk prediction methodologies.
    """
    print("Healthcare Risk Prediction Demonstration")
    print("=" * 50)
    
    # Generate synthetic healthcare data
    np.random.seed(42)
    n_patients = 5000
    
    # Patient features
    age = np.random.normal(65, 15, n_patients)
    bmi = np.random.normal(28, 5, n_patients)
    systolic_bp = np.random.normal(140, 20, n_patients)
    diabetes = np.random.binomial(1, 0.3, n_patients)
    smoking = np.random.binomial(1, 0.2, n_patients)
    gender = np.random.binomial(1, 0.5, n_patients)  # Protected attribute
    race = np.random.choice([0, 1, 2], n_patients, p=[0.6, 0.3, 0.1])  # Protected attribute
    
    # Create outcome with realistic relationships
    risk_score = (0.05 * age + 0.02 * bmi + 0.01 * systolic_bp + 
                 0.5 * diabetes + 0.3 * smoking + 
                 0.1 * gender + 0.2 * (race == 1) + 0.3 * (race == 2))
    
    # Add noise and convert to probability
    risk_prob = 1 / (1 + np.exp(-(risk_score - 5 + np.random.normal(0, 0.5, n_patients))))
    outcome = np.random.binomial(1, risk_prob, n_patients)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'diabetes': diabetes,
        'smoking': smoking,
        'gender': gender,
        'race': race,
        'outcome': outcome,
        'time': np.random.uniform(0, 365, n_patients)  # Days of follow-up
    })
    
    # Initialize risk predictor
    predictor = AdvancedRiskPredictor(random_state=42)
    
    # Prepare data
    df_processed = predictor.prepare_data(
        df, 
        target_col='outcome',
        protected_attrs=['gender', 'race'],
        time_col='time'
    )
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.temporal_split(df_processed)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    predictor.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test)
    
    # Assess fairness
    fairness_results = predictor.assess_fairness(X_test, y_test)
    
    # Generate report
    report = predictor.generate_report()
    print("\n" + report)
    
    # Plot results
    predictor.plot_evaluation_results()
    
    # Demonstrate heterogeneous treatment effects
    print("\nHeterogeneous Treatment Effect Estimation")
    print("-" * 40)
    
    # Generate treatment data
    treatment = np.random.binomial(1, 0.5, len(X_train))
    
    # Create treatment effect (heterogeneous based on age)
    true_effect = 0.1 * (X_train['age'] - 65) / 15  # Older patients benefit more
    
    # Generate outcome with treatment effect
    y_potential_0 = risk_prob[:len(X_train)]
    y_potential_1 = risk_prob[:len(X_train)] + true_effect
    y_observed = treatment * y_potential_1 + (1 - treatment) * y_potential_0
    y_observed = np.random.binomial(1, np.clip(y_observed, 0, 1))
    
    # Estimate treatment effects
    hte = HeterogeneousTreatmentEffects(random_state=42)
    
    X_train_array = X_train.drop(columns=['time']).values
    
    tau_s = hte.s_learner(X_train_array, treatment, y_observed)
    tau_t = hte.t_learner(X_train_array, treatment, y_observed)
    tau_x = hte.x_learner(X_train_array, treatment, y_observed)
    tau_dr = hte.doubly_robust_learner(X_train_array, treatment, y_observed)
    
    print(f"S-Learner CATE mean: {tau_s.mean():.3f} ± {tau_s.std():.3f}")
    print(f"T-Learner CATE mean: {tau_t.mean():.3f} ± {tau_t.std():.3f}")
    print(f"X-Learner CATE mean: {tau_x.mean():.3f} ± {tau_x.std():.3f}")
    print(f"Doubly Robust CATE mean: {tau_dr.mean():.3f} ± {tau_dr.std():.3f}")
    print(f"True CATE mean: {true_effect.mean():.3f} ± {true_effect.std():.3f}")
    
    return predictor, results, fairness_results

if __name__ == "__main__":
    predictor, results, fairness_results = demonstrate_risk_prediction()
```

## Actuarial Science Integration

Traditional actuarial methods have long been used for risk assessment in insurance and healthcare. Modern machine learning approaches can significantly enhance these traditional methods while maintaining their interpretability and regulatory compliance.

### Traditional vs. Modern Approaches

**Traditional Actuarial Methods:**
- Generalized Linear Models (GLMs)
- Life tables and survival analysis
- Experience rating and credibility theory
- Risk classification based on demographic factors

**Modern Machine Learning Enhancements:**
- Ensemble methods for improved accuracy
- Non-linear relationship modeling
- High-dimensional feature spaces
- Automated feature engineering
- Real-time risk updating

### Implementation Example: Enhanced Actuarial Model

```python
class EnhancedActuarialModel:
    """
    Bridge between traditional actuarial methods and modern ML approaches.
    """
    
    def __init__(self):
        self.traditional_model = None
        self.ml_model = None
        self.ensemble_weights = None
    
    def fit_traditional_glm(self, X, y, family='binomial'):
        """
        Fit traditional GLM following actuarial practices.
        """
        from statsmodels.genmod.families import Binomial, Poisson, Gamma
        import statsmodels.api as sm
        
        if family == 'binomial':
            family_obj = Binomial()
        elif family == 'poisson':
            family_obj = Poisson()
        elif family == 'gamma':
            family_obj = Gamma()
        
        # Add intercept
        X_with_intercept = sm.add_constant(X)
        
        # Fit GLM
        self.traditional_model = sm.GLM(y, X_with_intercept, family=family_obj)
        self.traditional_results = self.traditional_model.fit()
        
        return self.traditional_results
    
    def fit_ml_model(self, X, y):
        """
        Fit modern ML model for comparison and ensemble.
        """
        self.ml_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1
        )
        self.ml_model.fit(X, y)
        
        return self.ml_model
    
    def create_ensemble(self, X_val, y_val):
        """
        Create ensemble combining traditional and ML approaches.
        """
        # Get predictions from both models
        X_val_with_intercept = sm.add_constant(X_val)
        trad_pred = self.traditional_results.predict(X_val_with_intercept)
        ml_pred = self.ml_model.predict_proba(X_val)[:, 1]
        
        # Optimize ensemble weights
        from scipy.optimize import minimize
        
        def ensemble_loss(weights):
            ensemble_pred = weights[0] * trad_pred + weights[1] * ml_pred
            return brier_score_loss(y_val, ensemble_pred)
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1), (0, 1)]
        
        result = minimize(ensemble_loss, [0.5, 0.5], 
                         bounds=bounds, constraints=constraints)
        
        self.ensemble_weights = result.x
        
        return self.ensemble_weights
```

## Clinical Case Studies

### Case Study 1: Acute Care Event Prediction

This case study demonstrates the implementation of a comprehensive acute care event prediction system, incorporating best practices for temporal validation and equity assessment.

**Clinical Context:** Predicting 30-day readmission risk for heart failure patients using electronic health record data.

**Implementation Highlights:**
- Temporal data splitting to prevent information leakage
- Comprehensive feature engineering including clinical trajectories
- Multi-model ensemble with calibration
- Fairness assessment across demographic groups
- Clinical interpretability through SHAP explanations

### Case Study 2: Population Health Risk Stratification

**Clinical Context:** Identifying high-risk patients for diabetes complications in a large health system.

**Key Components:**
- Integration of claims, clinical, and social determinant data
- Causal inference methods for intervention targeting
- Health equity assessment following Obermeyer's framework
- Real-time risk score updating with new clinical data

## Regulatory and Ethical Considerations

The implementation of risk prediction models in healthcare requires careful attention to regulatory requirements and ethical considerations:

### FDA Software as Medical Device (SaMD) Framework

Risk prediction models may fall under FDA regulation depending on their intended use and risk classification. Key considerations include:

- **Risk categorization** based on healthcare situation and SaMD state
- **Clinical evaluation** requirements for different risk levels
- **Quality management system** implementation
- **Post-market surveillance** and performance monitoring

### Bias and Fairness Assessment

Following the framework established by Obermeyer et al. (2019), healthcare AI systems must be rigorously evaluated for bias across demographic groups:

1. **Algorithmic bias detection** using statistical parity and equalized odds
2. **Historical bias** assessment in training data
3. **Representation bias** evaluation across patient populations
4. **Evaluation bias** in outcome measurement and labeling

## Future Directions

The field of healthcare risk prediction continues to evolve with several promising directions:

### Foundation Models for Healthcare

Large-scale foundation models trained on diverse healthcare data show promise for:
- Transfer learning across clinical domains
- Few-shot learning for rare conditions
- Multimodal integration of clinical data types

### Causal Machine Learning

Integration of causal inference with machine learning enables:
- Robust prediction under distribution shift
- Identification of causal risk factors
- Optimal treatment assignment policies

### Federated Learning

Collaborative model development across institutions while preserving privacy:
- Improved model generalization
- Reduced data sharing requirements
- Enhanced representation of diverse populations

## Summary

This chapter has provided comprehensive coverage of advanced risk prediction methodologies for healthcare applications. Key takeaways include:

1. **Methodological rigor** in data splitting, cross-validation, and evaluation
2. **Equity considerations** throughout the model development lifecycle
3. **Integration** of traditional actuarial methods with modern ML approaches
4. **Clinical validation** frameworks for real-world deployment
5. **Regulatory compliance** considerations for healthcare AI systems

The implementations provided serve as production-ready templates that can be adapted for specific clinical applications while maintaining the highest standards of scientific rigor and ethical responsibility.

## References

1. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453. DOI: 10.1126/science.aax2342

2. Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Chin, M. H. (2018). Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12), 866-872. DOI: 10.7326/M18-1990

3. Chen, J. H., & Asch, S. M. (2017). Machine learning and prediction in medicine—beyond the peak of inflated expectations. *New England Journal of Medicine*, 376(26), 2507-2509. DOI: 10.1056/NEJMp1702071

4. Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228-1242. DOI: 10.1080/01621459.2017.1319839

5. Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156-4165. DOI: 10.1073/pnas.1804597116

6. Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008). Random survival forests. *The Annals of Applied Statistics*, 2(3), 841-860. DOI: 10.1214/08-AOAS169

7. Austin, P. C., Lee, D. S., & Fine, J. P. (2016). Introduction to the analysis of survival data in the presence of competing risks. *Circulation*, 133(6), 601-609. DOI: 10.1161/CIRCULATIONAHA.115.017719

8. Steyerberg, E. W., & Vergouwe, Y. (2014). Towards better clinical prediction models: seven steps for development and an ABCD for validation. *European Heart Journal*, 35(29), 1925-1931. DOI: 10.1093/eurheartj/ehu207

9. Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G. (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. *BMJ*, 350, g7594. DOI: 10.1136/bmj.g7594

10. Liu, X., Faes, L., Kale, A. U., Wagner, S. K., Fu, D. J., Bruynseels, A., ... & Denniston, A. K. (2019). A comparison of deep learning performance against health-care professionals in detecting diseases from medical imaging: a systematic review and meta-analysis. *The Lancet Digital Health*, 1(6), e271-e297. DOI: 10.1016/S2589-7500(19)30123-2
