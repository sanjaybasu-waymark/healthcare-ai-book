# Chapter 8: Bias Detection and Mitigation in Healthcare AI

## Learning Objectives

By the end of this chapter, readers will be able to:

1. **Understand the theoretical foundations** of bias in healthcare AI systems and its clinical implications
2. **Implement comprehensive bias detection** frameworks for different types of healthcare AI models
3. **Deploy bias mitigation strategies** at pre-processing, in-processing, and post-processing stages
4. **Evaluate fairness metrics** appropriate for different healthcare applications and populations
5. **Design equitable AI systems** that promote health equity across diverse patient populations
6. **Implement continuous monitoring** systems for bias detection in production healthcare AI

## 8.1 Introduction to Bias in Healthcare AI

Bias in healthcare artificial intelligence represents one of the most critical challenges facing the deployment of AI systems in clinical practice. Healthcare AI bias can perpetuate and amplify existing health disparities, leading to inequitable care delivery and potentially harmful outcomes for vulnerable populations. Understanding, detecting, and mitigating bias is not merely a technical challenge but a moral imperative that requires sophisticated approaches grounded in both computer science and health equity principles.

The manifestation of bias in healthcare AI systems is particularly concerning because healthcare decisions directly impact patient outcomes, quality of life, and survival. Unlike bias in other domains such as advertising or recommendation systems, healthcare AI bias can have life-or-death consequences, making the development of robust bias detection and mitigation frameworks essential for responsible AI deployment.

### 8.1.1 Theoretical Foundations of Healthcare AI Bias

Healthcare AI bias emerges from multiple sources throughout the AI development lifecycle, from data collection and annotation to model training and deployment. Understanding these sources requires a comprehensive framework that addresses both technical and socio-technical factors.

**Historical Bias**: Healthcare data often reflects historical patterns of discrimination and unequal access to care. Electronic health records may contain fewer data points for underserved populations, leading to models that perform poorly for these groups.

**Representation Bias**: Clinical datasets frequently underrepresent certain demographic groups, particularly racial and ethnic minorities, women in certain age groups, and patients from lower socioeconomic backgrounds.

**Measurement Bias**: Clinical measurements and diagnostic criteria may have been developed and validated primarily in specific populations, leading to systematic measurement errors when applied to other groups.

**Aggregation Bias**: Combining data from different populations without accounting for relevant differences can mask important subgroup variations and lead to models that perform poorly for minority populations.

**Evaluation Bias**: Using inappropriate benchmarks or evaluation metrics that do not account for population differences can hide bias in model performance.

The formal representation of bias in healthcare AI can be expressed through the lens of statistical parity and equalized odds. Let $Y$ represent the true clinical outcome, $\hat{Y}$ represent the AI prediction, and $A$ represent a protected attribute (e.g., race, gender). Statistical parity requires:

$$P(\hat{Y} = 1 | A = 0) = P(\hat{Y} = 1 | A = 1)$$

However, statistical parity may not be appropriate for healthcare applications where base rates of conditions differ across populations. Equalized odds provides a more nuanced approach:

$$P(\hat{Y} = 1 | Y = 1, A = 0) = P(\hat{Y} = 1 | Y = 1, A = 1)$$
$$P(\hat{Y} = 1 | Y = 0, A = 0) = P(\hat{Y} = 1 | Y = 0, A = 1)$$

### 8.1.2 Clinical Implications of AI Bias

The clinical implications of AI bias extend beyond statistical measures to real-world patient outcomes and health equity. Biased AI systems can:

**Exacerbate Health Disparities**: AI systems that perform poorly for certain populations can worsen existing health disparities by providing suboptimal care recommendations.

**Reduce Clinical Trust**: Clinicians may lose trust in AI systems if they observe poor performance for certain patient populations, potentially reducing the beneficial impact of AI assistance.

**Create Legal and Ethical Liability**: Healthcare organizations deploying biased AI systems may face legal challenges and ethical scrutiny, particularly if bias leads to adverse patient outcomes.

**Undermine Population Health**: Systematic bias in AI systems can have population-level effects, particularly for public health applications and population health management.

### 8.1.3 Regulatory and Ethical Framework

The detection and mitigation of bias in healthcare AI is increasingly recognized by regulatory bodies and professional organizations. The FDA's guidance on Software as Medical Device (SaMD) emphasizes the importance of addressing bias and ensuring equitable performance across patient populations.

**FDA SaMD Framework**: The FDA requires evidence of performance across relevant patient subgroups and may require post-market surveillance to monitor for bias in real-world deployment.

**Professional Guidelines**: Medical professional organizations increasingly emphasize the importance of AI fairness and equity in clinical practice guidelines.

**Institutional Policies**: Healthcare organizations are developing policies and procedures for bias assessment and mitigation in AI system procurement and deployment.

## 8.2 Comprehensive Bias Detection Framework

### 8.2.1 Multi-Stage Bias Detection Pipeline

Effective bias detection in healthcare AI requires a comprehensive pipeline that addresses bias at multiple stages of the AI development lifecycle. This pipeline must be tailored to the specific characteristics of healthcare data and clinical applications.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import scipy.stats as stats
from scipy.stats import chi2_contingency, ks_2samp
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Fairness metrics libraries
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing, FairAdaBoost
from aif360.algorithms.postprocessing import CalibratedEqualizedOdds, EqualizedOdds

import logging
from datetime import datetime, timedelta
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiasType(Enum):
    """Types of bias that can occur in healthcare AI systems."""
    HISTORICAL = "historical"
    REPRESENTATION = "representation"
    MEASUREMENT = "measurement"
    AGGREGATION = "aggregation"
    EVALUATION = "evaluation"
    ALGORITHMIC = "algorithmic"
    DEPLOYMENT = "deployment"

class FairnessMetric(Enum):
    """Fairness metrics for healthcare AI evaluation."""
    STATISTICAL_PARITY = "statistical_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"

@dataclass
class BiasDetectionResult:
    """Results from bias detection analysis."""
    bias_type: BiasType
    metric_name: str
    metric_value: float
    threshold: float
    is_biased: bool
    affected_groups: List[str]
    severity: str  # "low", "medium", "high", "critical"
    confidence: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bias_type': self.bias_type.value,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'is_biased': self.is_biased,
            'affected_groups': self.affected_groups,
            'severity': self.severity,
            'confidence': self.confidence,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }

class HealthcareAIBiasDetector:
    """
    Comprehensive bias detection framework for healthcare AI systems.
    
    This class implements state-of-the-art bias detection methods specifically
    designed for healthcare applications, incorporating clinical domain knowledge
    and regulatory requirements.
    
    Based on research from:
    Rajkomar, A., et al. (2018). Ensuring fairness in machine learning to advance 
    health equity. Annals of Internal Medicine, 169(12), 866-872.
    DOI: 10.7326/M18-1990
    
    And fairness frameworks from:
    Verma, S., & Rubin, J. (2018). Fairness definitions explained. 
    Proceedings of the International Workshop on Software Fairness, 1-7.
    DOI: 10.1145/3194770.3194776
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        fairness_thresholds: Optional[Dict[str, float]] = None,
        clinical_context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize healthcare AI bias detector.
        
        Args:
            protected_attributes: List of protected attributes (e.g., race, gender, age_group)
            fairness_thresholds: Custom thresholds for fairness metrics
            clinical_context: Clinical context information for bias assessment
        """
        self.protected_attributes = protected_attributes
        self.clinical_context = clinical_context or {}
        
        # Default fairness thresholds based on healthcare AI literature
        self.fairness_thresholds = fairness_thresholds or {
            'statistical_parity_difference': 0.1,
            'equalized_odds_difference': 0.1,
            'equal_opportunity_difference': 0.1,
            'calibration_difference': 0.05,
            'demographic_parity_ratio': 0.8,  # 80% rule
            'predictive_parity_difference': 0.1
        }
        
        # Bias detection results
        self.detection_results: List[BiasDetectionResult] = []
        
        # Model performance by group
        self.group_performance: Dict[str, Dict[str, float]] = {}
        
        # Statistical tests for bias detection
        self.statistical_tests = {
            'chi_square': self._chi_square_test,
            'kolmogorov_smirnov': self._ks_test,
            'permutation_test': self._permutation_test
        }
        
        logger.info(f"Initialized HealthcareAIBiasDetector for attributes: {protected_attributes}")
    
    def detect_data_bias(
        self,
        data: pd.DataFrame,
        target_column: str,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Detect bias in healthcare datasets before model training.
        
        Args:
            data: Healthcare dataset
            target_column: Name of target variable column
            generate_report: Whether to generate detailed bias report
            
        Returns:
            Dictionary containing bias detection results
        """
        logger.info("Starting data bias detection analysis")
        
        bias_results = {
            'representation_bias': self._detect_representation_bias(data),
            'measurement_bias': self._detect_measurement_bias(data, target_column),
            'historical_bias': self._detect_historical_bias(data, target_column),
            'missing_data_bias': self._detect_missing_data_bias(data),
            'label_bias': self._detect_label_bias(data, target_column)
        }
        
        # Calculate overall bias score
        bias_results['overall_bias_score'] = self._calculate_overall_bias_score(bias_results)
        
        # Generate recommendations
        bias_results['recommendations'] = self._generate_data_bias_recommendations(bias_results)
        
        if generate_report:
            self._generate_bias_report(bias_results, "data_bias_report")
        
        logger.info(f"Data bias detection completed. Overall bias score: {bias_results['overall_bias_score']:.3f}")
        
        return bias_results
    
    def _detect_representation_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect representation bias in the dataset."""
        representation_results = {}
        
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            # Calculate representation statistics
            value_counts = data[attr].value_counts()
            proportions = data[attr].value_counts(normalize=True)
            
            # Check for underrepresentation (less than 5% of data)
            underrepresented_groups = proportions[proportions < 0.05].index.tolist()
            
            # Calculate representation disparity
            max_prop = proportions.max()
            min_prop = proportions.min()
            representation_ratio = min_prop / max_prop if max_prop > 0 else 0
            
            representation_results[attr] = {
                'value_counts': value_counts.to_dict(),
                'proportions': proportions.to_dict(),
                'underrepresented_groups': underrepresented_groups,
                'representation_ratio': representation_ratio,
                'is_biased': representation_ratio < 0.2,  # 20% threshold
                'severity': self._assess_bias_severity(representation_ratio, 'representation')
            }
        
        return representation_results
    
    def _detect_measurement_bias(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect measurement bias in clinical variables."""
        measurement_results = {}
        
        # Identify numeric clinical variables
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            attr_results = {}
            
            for col in numeric_columns:
                # Test for measurement differences across groups
                groups = data.groupby(attr)[col].apply(list)
                
                if len(groups) >= 2:
                    # Perform Kolmogorov-Smirnov test for distribution differences
                    group_names = list(groups.index)
                    ks_statistic, ks_p_value = ks_2samp(groups.iloc[0], groups.iloc[1])
                    
                    # Calculate effect size (Cohen's d)
                    effect_size = self._calculate_cohens_d(groups.iloc[0], groups.iloc[1])
                    
                    attr_results[col] = {
                        'ks_statistic': ks_statistic,
                        'ks_p_value': ks_p_value,
                        'effect_size': effect_size,
                        'is_biased': ks_p_value < 0.05 and abs(effect_size) > 0.2,
                        'severity': self._assess_bias_severity(abs(effect_size), 'measurement')
                    }
            
            measurement_results[attr] = attr_results
        
        return measurement_results
    
    def _detect_historical_bias(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect historical bias in outcome distributions."""
        historical_results = {}
        
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            # Calculate outcome rates by group
            outcome_rates = data.groupby(attr)[target_column].mean()
            
            # Test for significant differences in outcome rates
            contingency_table = pd.crosstab(data[attr], data[target_column])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate Cramér's V for effect size
            cramers_v = np.sqrt(chi2 / (data.shape[0] * (min(contingency_table.shape) - 1)))
            
            historical_results[attr] = {
                'outcome_rates': outcome_rates.to_dict(),
                'chi2_statistic': chi2,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'is_biased': p_value < 0.05 and cramers_v > 0.1,
                'severity': self._assess_bias_severity(cramers_v, 'historical')
            }
        
        return historical_results
    
    def _detect_missing_data_bias(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect bias in missing data patterns."""
        missing_results = {}
        
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            attr_results = {}
            
            for col in data.columns:
                if col == attr:
                    continue
                
                # Calculate missing data rates by group
                missing_rates = data.groupby(attr)[col].apply(lambda x: x.isnull().mean())
                
                # Test for significant differences in missing rates
                if len(missing_rates) >= 2:
                    # Create contingency table for missing data
                    missing_table = pd.crosstab(
                        data[attr], 
                        data[col].isnull(), 
                        margins=False
                    )
                    
                    if missing_table.shape == (len(missing_rates), 2):
                        chi2, p_value, dof, expected = chi2_contingency(missing_table)
                        
                        attr_results[col] = {
                            'missing_rates': missing_rates.to_dict(),
                            'chi2_statistic': chi2,
                            'p_value': p_value,
                            'is_biased': p_value < 0.05,
                            'severity': self._assess_bias_severity(missing_rates.max() - missing_rates.min(), 'missing_data')
                        }
            
            missing_results[attr] = attr_results
        
        return missing_results
    
    def _detect_label_bias(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect bias in label assignment."""
        label_results = {}
        
        # This is a simplified implementation
        # In practice, label bias detection requires domain expertise and external validation
        
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            # Calculate label distribution by group
            label_dist = pd.crosstab(data[attr], data[target_column], normalize='index')
            
            # Calculate label bias score (simplified)
            label_variance = label_dist.var(axis=0).mean()
            
            label_results[attr] = {
                'label_distribution': label_dist.to_dict(),
                'label_variance': label_variance,
                'is_biased': label_variance > 0.1,  # Threshold for label bias
                'severity': self._assess_bias_severity(label_variance, 'label')
            }
        
        return label_results
    
    def detect_model_bias(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: Optional[np.ndarray] = None,
        y_pred_proba: Optional[np.ndarray] = None,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Detect bias in trained healthcare AI models.
        
        Args:
            model: Trained machine learning model
            X_test: Test features
            y_test: Test labels
            y_pred: Model predictions (optional, will be generated if not provided)
            y_pred_proba: Model prediction probabilities (optional)
            generate_report: Whether to generate detailed bias report
            
        Returns:
            Dictionary containing model bias detection results
        """
        logger.info("Starting model bias detection analysis")
        
        # Generate predictions if not provided
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        if y_pred_proba is None and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        bias_results = {
            'fairness_metrics': self._calculate_fairness_metrics(X_test, y_test, y_pred, y_pred_proba),
            'group_performance': self._calculate_group_performance(X_test, y_test, y_pred, y_pred_proba),
            'calibration_bias': self._detect_calibration_bias(X_test, y_test, y_pred_proba),
            'prediction_bias': self._detect_prediction_bias(X_test, y_test, y_pred),
            'intersectional_bias': self._detect_intersectional_bias(X_test, y_test, y_pred)
        }
        
        # Calculate overall model bias score
        bias_results['overall_bias_score'] = self._calculate_model_bias_score(bias_results)
        
        # Generate recommendations
        bias_results['recommendations'] = self._generate_model_bias_recommendations(bias_results)
        
        if generate_report:
            self._generate_bias_report(bias_results, "model_bias_report")
        
        logger.info(f"Model bias detection completed. Overall bias score: {bias_results['overall_bias_score']:.3f}")
        
        return bias_results
    
    def _calculate_fairness_metrics(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive fairness metrics."""
        fairness_results = {}
        
        for attr in self.protected_attributes:
            if attr not in X_test.columns:
                continue
            
            attr_results = {}
            
            # Get unique groups for this attribute
            groups = X_test[attr].unique()
            
            if len(groups) < 2:
                continue
            
            # Calculate metrics for each pair of groups
            for i, group1 in enumerate(groups):
                for group2 in groups[i+1:]:
                    
                    # Get indices for each group
                    idx1 = X_test[attr] == group1
                    idx2 = X_test[attr] == group2
                    
                    # Statistical Parity Difference
                    spd = self._statistical_parity_difference(y_pred[idx1], y_pred[idx2])
                    
                    # Equalized Odds Difference
                    eod = self._equalized_odds_difference(
                        y_test[idx1], y_pred[idx1], y_test[idx2], y_pred[idx2]
                    )
                    
                    # Equal Opportunity Difference
                    eopd = self._equal_opportunity_difference(
                        y_test[idx1], y_pred[idx1], y_test[idx2], y_pred[idx2]
                    )
                    
                    # Calibration metrics if probabilities available
                    calibration_diff = None
                    if y_pred_proba is not None:
                        calibration_diff = self._calibration_difference(
                            y_test[idx1], y_pred_proba[idx1], y_test[idx2], y_pred_proba[idx2]
                        )
                    
                    pair_key = f"{group1}_vs_{group2}"
                    attr_results[pair_key] = {
                        'statistical_parity_difference': spd,
                        'equalized_odds_difference': eod,
                        'equal_opportunity_difference': eopd,
                        'calibration_difference': calibration_diff,
                        'is_biased': self._assess_fairness_violation(spd, eod, eopd, calibration_diff)
                    }
            
            fairness_results[attr] = attr_results
        
        return fairness_results
    
    def _statistical_parity_difference(self, y_pred1: np.ndarray, y_pred2: np.ndarray) -> float:
        """Calculate statistical parity difference between two groups."""
        rate1 = np.mean(y_pred1)
        rate2 = np.mean(y_pred2)
        return abs(rate1 - rate2)
    
    def _equalized_odds_difference(
        self,
        y_true1: np.ndarray,
        y_pred1: np.ndarray,
        y_true2: np.ndarray,
        y_pred2: np.ndarray
    ) -> float:
        """Calculate equalized odds difference between two groups."""
        # True Positive Rate difference
        tpr1 = np.mean(y_pred1[y_true1 == 1]) if np.sum(y_true1 == 1) > 0 else 0
        tpr2 = np.mean(y_pred2[y_true2 == 1]) if np.sum(y_true2 == 1) > 0 else 0
        tpr_diff = abs(tpr1 - tpr2)
        
        # False Positive Rate difference
        fpr1 = np.mean(y_pred1[y_true1 == 0]) if np.sum(y_true1 == 0) > 0 else 0
        fpr2 = np.mean(y_pred2[y_true2 == 0]) if np.sum(y_true2 == 0) > 0 else 0
        fpr_diff = abs(fpr1 - fpr2)
        
        return max(tpr_diff, fpr_diff)
    
    def _equal_opportunity_difference(
        self,
        y_true1: np.ndarray,
        y_pred1: np.ndarray,
        y_true2: np.ndarray,
        y_pred2: np.ndarray
    ) -> float:
        """Calculate equal opportunity difference between two groups."""
        # True Positive Rate difference only
        tpr1 = np.mean(y_pred1[y_true1 == 1]) if np.sum(y_true1 == 1) > 0 else 0
        tpr2 = np.mean(y_pred2[y_true2 == 1]) if np.sum(y_true2 == 1) > 0 else 0
        return abs(tpr1 - tpr2)
    
    def _calibration_difference(
        self,
        y_true1: np.ndarray,
        y_prob1: np.ndarray,
        y_true2: np.ndarray,
        y_prob2: np.ndarray
    ) -> float:
        """Calculate calibration difference between two groups."""
        # Bin predictions and calculate calibration for each group
        n_bins = 10
        
        # Group 1 calibration
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        cal_diff_sum = 0
        valid_bins = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Group 1
            in_bin1 = (y_prob1 > bin_lower) & (y_prob1 <= bin_upper)
            if np.sum(in_bin1) > 0:
                bin_acc1 = np.mean(y_true1[in_bin1])
                bin_conf1 = np.mean(y_prob1[in_bin1])
            else:
                bin_acc1 = bin_conf1 = 0
            
            # Group 2
            in_bin2 = (y_prob2 > bin_lower) & (y_prob2 <= bin_upper)
            if np.sum(in_bin2) > 0:
                bin_acc2 = np.mean(y_true2[in_bin2])
                bin_conf2 = np.mean(y_prob2[in_bin2])
            else:
                bin_acc2 = bin_conf2 = 0
            
            # Calculate calibration difference for this bin
            if np.sum(in_bin1) > 0 or np.sum(in_bin2) > 0:
                cal_diff1 = abs(bin_acc1 - bin_conf1)
                cal_diff2 = abs(bin_acc2 - bin_conf2)
                cal_diff_sum += abs(cal_diff1 - cal_diff2)
                valid_bins += 1
        
        return cal_diff_sum / valid_bins if valid_bins > 0 else 0
    
    def _calculate_group_performance(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate performance metrics for each demographic group."""
        group_performance = {}
        
        for attr in self.protected_attributes:
            if attr not in X_test.columns:
                continue
            
            attr_performance = {}
            
            for group in X_test[attr].unique():
                idx = X_test[attr] == group
                
                if np.sum(idx) == 0:
                    continue
                
                y_true_group = y_test[idx]
                y_pred_group = y_pred[idx]
                
                # Calculate basic metrics
                accuracy = accuracy_score(y_true_group, y_pred_group)
                precision = precision_score(y_true_group, y_pred_group, average='binary', zero_division=0)
                recall = recall_score(y_true_group, y_pred_group, average='binary', zero_division=0)
                f1 = f1_score(y_true_group, y_pred_group, average='binary', zero_division=0)
                
                group_metrics = {
                    'sample_size': np.sum(idx),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # Add AUC if probabilities available
                if y_pred_proba is not None:
                    y_prob_group = y_pred_proba[idx]
                    if len(np.unique(y_true_group)) > 1:
                        auc = roc_auc_score(y_true_group, y_prob_group)
                        group_metrics['auc'] = auc
                
                attr_performance[str(group)] = group_metrics
            
            group_performance[attr] = attr_performance
        
        return group_performance
    
    def _detect_calibration_bias(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Detect calibration bias across demographic groups."""
        if y_pred_proba is None:
            return {'error': 'Prediction probabilities not available for calibration analysis'}
        
        calibration_results = {}
        
        for attr in self.protected_attributes:
            if attr not in X_test.columns:
                continue
            
            attr_results = {}
            
            for group in X_test[attr].unique():
                idx = X_test[attr] == group
                
                if np.sum(idx) < 10:  # Need minimum samples for calibration analysis
                    continue
                
                y_true_group = y_test[idx]
                y_prob_group = y_pred_proba[idx]
                
                # Calculate calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_group, y_prob_group, n_bins=10
                )
                
                # Calculate calibration error (Expected Calibration Error)
                ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                
                # Calculate Brier score
                brier_score = np.mean((y_prob_group - y_true_group) ** 2)
                
                attr_results[str(group)] = {
                    'expected_calibration_error': ece,
                    'brier_score': brier_score,
                    'calibration_curve': {
                        'fraction_of_positives': fraction_of_positives.tolist(),
                        'mean_predicted_value': mean_predicted_value.tolist()
                    }
                }
            
            calibration_results[attr] = attr_results
        
        return calibration_results
    
    def _detect_prediction_bias(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Detect systematic prediction bias across groups."""
        prediction_results = {}
        
        for attr in self.protected_attributes:
            if attr not in X_test.columns:
                continue
            
            attr_results = {}
            groups = X_test[attr].unique()
            
            # Calculate prediction rates and error rates by group
            for group in groups:
                idx = X_test[attr] == group
                
                if np.sum(idx) == 0:
                    continue
                
                y_true_group = y_test[idx]
                y_pred_group = y_pred[idx]
                
                # Prediction rates
                positive_prediction_rate = np.mean(y_pred_group)
                
                # Error rates
                false_positive_rate = np.mean(y_pred_group[y_true_group == 0]) if np.sum(y_true_group == 0) > 0 else 0
                false_negative_rate = np.mean(1 - y_pred_group[y_true_group == 1]) if np.sum(y_true_group == 1) > 0 else 0
                
                # Prediction bias score
                base_rate = np.mean(y_true_group)
                prediction_bias = abs(positive_prediction_rate - base_rate)
                
                attr_results[str(group)] = {
                    'positive_prediction_rate': positive_prediction_rate,
                    'base_rate': base_rate,
                    'prediction_bias': prediction_bias,
                    'false_positive_rate': false_positive_rate,
                    'false_negative_rate': false_negative_rate
                }
            
            prediction_results[attr] = attr_results
        
        return prediction_results
    
    def _detect_intersectional_bias(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Detect bias at intersections of multiple protected attributes."""
        if len(self.protected_attributes) < 2:
            return {'error': 'Need at least 2 protected attributes for intersectional analysis'}
        
        intersectional_results = {}
        
        # Consider all pairs of protected attributes
        from itertools import combinations
        
        for attr1, attr2 in combinations(self.protected_attributes, 2):
            if attr1 not in X_test.columns or attr2 not in X_test.columns:
                continue
            
            pair_key = f"{attr1}_x_{attr2}"
            pair_results = {}
            
            # Get all combinations of values for these attributes
            for val1 in X_test[attr1].unique():
                for val2 in X_test[attr2].unique():
                    idx = (X_test[attr1] == val1) & (X_test[attr2] == val2)
                    
                    if np.sum(idx) < 5:  # Need minimum samples
                        continue
                    
                    y_true_group = y_test[idx]
                    y_pred_group = y_pred[idx]
                    
                    # Calculate performance for this intersection
                    accuracy = accuracy_score(y_true_group, y_pred_group)
                    precision = precision_score(y_true_group, y_pred_group, average='binary', zero_division=0)
                    recall = recall_score(y_true_group, y_pred_group, average='binary', zero_division=0)
                    
                    intersection_key = f"{val1}_{val2}"
                    pair_results[intersection_key] = {
                        'sample_size': np.sum(idx),
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall
                    }
            
            intersectional_results[pair_key] = pair_results
        
        return intersectional_results
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _assess_bias_severity(self, metric_value: float, bias_type: str) -> str:
        """Assess the severity of detected bias."""
        # Define severity thresholds based on bias type
        thresholds = {
            'representation': {'low': 0.8, 'medium': 0.5, 'high': 0.2},
            'measurement': {'low': 0.2, 'medium': 0.5, 'high': 0.8},
            'historical': {'low': 0.1, 'medium': 0.3, 'high': 0.5},
            'missing_data': {'low': 0.05, 'medium': 0.15, 'high': 0.3},
            'label': {'low': 0.05, 'medium': 0.15, 'high': 0.3}
        }
        
        if bias_type not in thresholds:
            return 'unknown'
        
        thresh = thresholds[bias_type]
        
        if metric_value <= thresh['low']:
            return 'low'
        elif metric_value <= thresh['medium']:
            return 'medium'
        elif metric_value <= thresh['high']:
            return 'high'
        else:
            return 'critical'
    
    def _assess_fairness_violation(
        self,
        spd: float,
        eod: float,
        eopd: float,
        calibration_diff: Optional[float]
    ) -> bool:
        """Assess if fairness metrics indicate bias."""
        violations = []
        
        violations.append(spd > self.fairness_thresholds['statistical_parity_difference'])
        violations.append(eod > self.fairness_thresholds['equalized_odds_difference'])
        violations.append(eopd > self.fairness_thresholds['equal_opportunity_difference'])
        
        if calibration_diff is not None:
            violations.append(calibration_diff > self.fairness_thresholds['calibration_difference'])
        
        return any(violations)
    
    def _calculate_overall_bias_score(self, bias_results: Dict[str, Any]) -> float:
        """Calculate overall bias score from detection results."""
        # This is a simplified scoring function
        # In practice, this would be more sophisticated and domain-specific
        
        bias_scores = []
        
        # Extract bias indicators from different analyses
        for analysis_type, results in bias_results.items():
            if isinstance(results, dict):
                for attr, attr_results in results.items():
                    if isinstance(attr_results, dict):
                        if 'is_biased' in attr_results:
                            bias_scores.append(1.0 if attr_results['is_biased'] else 0.0)
                        elif isinstance(attr_results, dict):
                            for sub_key, sub_results in attr_results.items():
                                if isinstance(sub_results, dict) and 'is_biased' in sub_results:
                                    bias_scores.append(1.0 if sub_results['is_biased'] else 0.0)
        
        return np.mean(bias_scores) if bias_scores else 0.0
    
    def _calculate_model_bias_score(self, bias_results: Dict[str, Any]) -> float:
        """Calculate overall model bias score."""
        # Extract fairness violations
        fairness_violations = []
        
        if 'fairness_metrics' in bias_results:
            for attr, attr_results in bias_results['fairness_metrics'].items():
                for pair, pair_results in attr_results.items():
                    if 'is_biased' in pair_results:
                        fairness_violations.append(1.0 if pair_results['is_biased'] else 0.0)
        
        return np.mean(fairness_violations) if fairness_violations else 0.0
    
    def _generate_data_bias_recommendations(self, bias_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for addressing data bias."""
        recommendations = []
        
        # Check representation bias
        if 'representation_bias' in bias_results:
            for attr, results in bias_results['representation_bias'].items():
                if results.get('is_biased', False):
                    recommendations.append(
                        f"Address underrepresentation in {attr}. Consider data augmentation or targeted collection."
                    )
        
        # Check measurement bias
        if 'measurement_bias' in bias_results:
            for attr, attr_results in bias_results['measurement_bias'].items():
                for var, var_results in attr_results.items():
                    if var_results.get('is_biased', False):
                        recommendations.append(
                            f"Investigate measurement differences in {var} across {attr} groups."
                        )
        
        # Check historical bias
        if 'historical_bias' in bias_results:
            for attr, results in bias_results['historical_bias'].items():
                if results.get('is_biased', False):
                    recommendations.append(
                        f"Historical bias detected in outcomes for {attr}. Consider bias mitigation techniques."
                    )
        
        return recommendations
    
    def _generate_model_bias_recommendations(self, bias_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for addressing model bias."""
        recommendations = []
        
        # Check fairness metrics
        if 'fairness_metrics' in bias_results:
            for attr, attr_results in bias_results['fairness_metrics'].items():
                for pair, pair_results in attr_results.items():
                    if pair_results.get('is_biased', False):
                        recommendations.append(
                            f"Fairness violation detected for {attr} ({pair}). Consider post-processing bias mitigation."
                        )
        
        # Check calibration bias
        if 'calibration_bias' in bias_results:
            recommendations.append(
                "Calibration differences detected. Consider group-specific calibration or recalibration techniques."
            )
        
        # Check intersectional bias
        if 'intersectional_bias' in bias_results and 'error' not in bias_results['intersectional_bias']:
            recommendations.append(
                "Intersectional bias analysis completed. Review performance for intersectional groups."
            )
        
        return recommendations
    
    def _generate_bias_report(self, bias_results: Dict[str, Any], report_type: str) -> None:
        """Generate comprehensive bias detection report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{timestamp}.json"
        
        report = {
            'report_type': report_type,
            'timestamp': timestamp,
            'protected_attributes': self.protected_attributes,
            'fairness_thresholds': self.fairness_thresholds,
            'bias_results': bias_results
        }
        
        # Save report
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Bias detection report saved: {filename}")
    
    def _chi_square_test(self, data: pd.DataFrame, attr1: str, attr2: str) -> Dict[str, float]:
        """Perform chi-square test for independence."""
        contingency_table = pd.crosstab(data[attr1], data[attr2])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof
        }
    
    def _ks_test(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        """Perform Kolmogorov-Smirnov test."""
        ks_statistic, p_value = ks_2samp(group1, group2)
        
        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value
        }
    
    def _permutation_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        n_permutations: int = 1000
    ) -> Dict[str, float]:
        """Perform permutation test for difference in means."""
        observed_diff = np.mean(group1) - np.mean(group2)
        
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        
        permuted_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_group1 = combined[:n1]
            perm_group2 = combined[n1:]
            permuted_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))
        
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value
        }

# Example usage and demonstration
def main():
    """Demonstrate comprehensive bias detection in healthcare AI."""
    
    # Generate synthetic healthcare dataset for demonstration
    np.random.seed(42)
    n_samples = 10000
    
    # Create synthetic patient data
    data = pd.DataFrame({
        'age': np.random.normal(65, 15, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.45, 0.55]),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
        'blood_pressure': np.random.normal(130, 20, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),
        'bmi': np.random.normal(28, 5, n_samples),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Introduce systematic bias in the data
    # Bias 1: Underrepresentation of certain racial groups
    # Bias 2: Different measurement patterns by gender
    # Bias 3: Income-related access bias
    
    # Create biased outcome variable
    outcome_prob = 0.1  # Base probability
    
    # Add bias based on race (historical bias)
    race_bias = {'White': 0.0, 'Black': 0.05, 'Hispanic': 0.03, 'Asian': -0.02}
    for race, bias in race_bias.items():
        mask = data['race'] == race
        outcome_prob_race = outcome_prob + bias
        data.loc[mask, 'outcome_prob'] = outcome_prob_race
    
    # Add bias based on income (access bias)
    income_bias = {'Low': 0.04, 'Medium': 0.0, 'High': -0.02}
    for income, bias in income_bias.items():
        mask = data['income_level'] == income
        data.loc[mask, 'outcome_prob'] = data.loc[mask, 'outcome_prob'] + bias
    
    # Generate outcomes
    data['cardiovascular_disease'] = np.random.binomial(1, data['outcome_prob'])
    
    # Add missing data bias
    # Higher missing rates for certain groups
    missing_prob = 0.05
    for race in ['Black', 'Hispanic']:
        mask = data['race'] == race
        missing_mask = np.random.random(mask.sum()) < missing_prob * 2
        data.loc[mask, 'cholesterol'] = data.loc[mask, 'cholesterol'].mask(missing_mask)
    
    print("Healthcare AI Bias Detection Demonstration")
    print("=" * 50)
    
    # Initialize bias detector
    protected_attributes = ['gender', 'race', 'income_level']
    bias_detector = HealthcareAIBiasDetector(
        protected_attributes=protected_attributes,
        clinical_context={'domain': 'cardiovascular_disease_prediction'}
    )
    
    # 1. Detect data bias
    print("\n1. Data Bias Detection")
    print("-" * 30)
    
    data_bias_results = bias_detector.detect_data_bias(
        data=data,
        target_column='cardiovascular_disease',
        generate_report=True
    )
    
    print(f"Overall data bias score: {data_bias_results['overall_bias_score']:.3f}")
    print(f"Number of recommendations: {len(data_bias_results['recommendations'])}")
    
    # Display key findings
    if 'representation_bias' in data_bias_results:
        print("\nRepresentation Bias Findings:")
        for attr, results in data_bias_results['representation_bias'].items():
            if results['is_biased']:
                print(f"  {attr}: Representation ratio = {results['representation_ratio']:.3f}")
                print(f"    Underrepresented groups: {results['underrepresented_groups']}")
    
    # 2. Train model and detect model bias
    print("\n2. Model Bias Detection")
    print("-" * 30)
    
    # Prepare data for modeling
    # Handle missing values
    data_clean = data.dropna()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_income = LabelEncoder()
    
    X = data_clean.copy()
    X['gender_encoded'] = le_gender.fit_transform(X['gender'])
    X['race_encoded'] = le_race.fit_transform(X['race'])
    X['income_encoded'] = le_income.fit_transform(X['income_level'])
    
    # Select features for modeling
    feature_columns = ['age', 'gender_encoded', 'race_encoded', 'income_encoded', 
                      'blood_pressure', 'cholesterol', 'bmi', 'smoking']
    X_model = X[feature_columns + ['gender', 'race', 'income_level']]  # Keep original for bias analysis
    y = X['cardiovascular_disease']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train[feature_columns], y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test[feature_columns])
    y_pred_proba = model.predict_proba(X_test[feature_columns])[:, 1]
    
    # Detect model bias
    model_bias_results = bias_detector.detect_model_bias(
        model=model,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        generate_report=True
    )
    
    print(f"Overall model bias score: {model_bias_results['overall_bias_score']:.3f}")
    print(f"Number of recommendations: {len(model_bias_results['recommendations'])}")
    
    # Display fairness metrics
    if 'fairness_metrics' in model_bias_results:
        print("\nFairness Metrics:")
        for attr, attr_results in model_bias_results['fairness_metrics'].items():
            print(f"\n  {attr}:")
            for pair, metrics in attr_results.items():
                if metrics['is_biased']:
                    print(f"    {pair}: BIASED")
                    print(f"      Statistical Parity Diff: {metrics['statistical_parity_difference']:.3f}")
                    print(f"      Equalized Odds Diff: {metrics['equalized_odds_difference']:.3f}")
                    print(f"      Equal Opportunity Diff: {metrics['equal_opportunity_difference']:.3f}")
    
    # Display group performance
    if 'group_performance' in model_bias_results:
        print("\nGroup Performance:")
        for attr, attr_results in model_bias_results['group_performance'].items():
            print(f"\n  {attr}:")
            for group, metrics in attr_results.items():
                print(f"    {group}: Accuracy={metrics['accuracy']:.3f}, "
                      f"F1={metrics['f1_score']:.3f}, "
                      f"Sample Size={metrics['sample_size']}")
    
    # 3. Display recommendations
    print("\n3. Bias Mitigation Recommendations")
    print("-" * 40)
    
    all_recommendations = (
        data_bias_results.get('recommendations', []) + 
        model_bias_results.get('recommendations', [])
    )
    
    for i, rec in enumerate(all_recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\nBias detection analysis completed!")
    print("Detailed reports saved as JSON files.")

if __name__ == "__main__":
    main()
```

## 8.3 Bias Mitigation Strategies

### 8.3.1 Pre-processing Mitigation Techniques

Pre-processing bias mitigation techniques address bias in the training data before model development. These approaches are particularly important in healthcare where historical data may reflect systemic inequities in care delivery and access.

**Data Augmentation for Underrepresented Groups**: Synthetic data generation can help balance representation across demographic groups while preserving clinical validity.

**Reweighing**: Adjusting sample weights to ensure equal representation of different groups in the effective training distribution.

**Disparate Impact Remover**: Removing correlations between protected attributes and other features while preserving as much information as possible.

### 8.3.2 In-processing Mitigation Techniques

In-processing techniques modify the learning algorithm itself to incorporate fairness constraints during model training.

**Adversarial Debiasing**: Using adversarial networks to learn representations that are predictive of the target outcome but not of protected attributes.

**Fair Representation Learning**: Learning intermediate representations that encode information about the target while being statistically independent of protected attributes.

**Constrained Optimization**: Incorporating fairness constraints directly into the optimization objective during model training.

### 8.3.3 Post-processing Mitigation Techniques

Post-processing techniques adjust model outputs to satisfy fairness criteria without retraining the model.

**Threshold Optimization**: Setting different decision thresholds for different groups to achieve fairness criteria.

**Calibration**: Adjusting prediction probabilities to ensure equal calibration across groups.

**Output Redistribution**: Modifying predictions to satisfy statistical parity or equalized odds constraints.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Fairness libraries
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqualizedOdds, EqualizedOdds

import logging
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareAIBiasMitigator:
    """
    Comprehensive bias mitigation framework for healthcare AI systems.
    
    This class implements state-of-the-art bias mitigation techniques
    specifically adapted for healthcare applications, including pre-processing,
    in-processing, and post-processing approaches.
    
    Based on research from:
    Mehrabi, N., et al. (2021). A survey on bias and fairness in machine learning. 
    ACM Computing Surveys, 54(6), 1-35. DOI: 10.1145/3457607
    
    And healthcare-specific approaches from:
    Chen, I. Y., et al. (2019). Ethical machine learning in healthcare. 
    Annual Review of Biomedical Data Science, 2, 123-144.
    DOI: 10.1146/annurev-biodatasci-092820-114757
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        fairness_constraints: Optional[Dict[str, float]] = None,
        clinical_context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize healthcare AI bias mitigator.
        
        Args:
            protected_attributes: List of protected attributes
            fairness_constraints: Fairness constraints to enforce
            clinical_context: Clinical context for bias mitigation
        """
        self.protected_attributes = protected_attributes
        self.clinical_context = clinical_context or {}
        
        # Default fairness constraints
        self.fairness_constraints = fairness_constraints or {
            'statistical_parity_difference': 0.1,
            'equalized_odds_difference': 0.1,
            'equal_opportunity_difference': 0.1
        }
        
        # Mitigation techniques
        self.preprocessing_techniques = {
            'reweighing': self._apply_reweighing,
            'disparate_impact_remover': self._apply_disparate_impact_remover,
            'data_augmentation': self._apply_data_augmentation,
            'feature_selection': self._apply_fair_feature_selection
        }
        
        self.inprocessing_techniques = {
            'adversarial_debiasing': self._apply_adversarial_debiasing,
            'fair_representation': self._apply_fair_representation,
            'constrained_optimization': self._apply_constrained_optimization
        }
        
        self.postprocessing_techniques = {
            'threshold_optimization': self._apply_threshold_optimization,
            'calibration': self._apply_calibration,
            'equalized_odds': self._apply_equalized_odds
        }
        
        # Mitigation history
        self.mitigation_history = []
        
        logger.info(f"Initialized HealthcareAIBiasMitigator for attributes: {protected_attributes}")
    
    def mitigate_bias(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        mitigation_strategy: str = "comprehensive",
        model_type: str = "random_forest"
    ) -> Dict[str, Any]:
        """
        Apply comprehensive bias mitigation strategy.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            mitigation_strategy: Strategy to use ("preprocessing", "inprocessing", "postprocessing", "comprehensive")
            model_type: Type of model to train
            
        Returns:
            Dictionary containing mitigation results
        """
        logger.info(f"Starting bias mitigation with strategy: {mitigation_strategy}")
        
        results = {
            'original_performance': {},
            'mitigated_performance': {},
            'fairness_improvement': {},
            'mitigation_techniques_applied': [],
            'recommendations': []
        }
        
        # Train baseline model
        baseline_model = self._train_baseline_model(X_train, y_train, model_type)
        baseline_pred = baseline_model.predict(X_test)
        baseline_pred_proba = baseline_model.predict_proba(X_test)[:, 1] if hasattr(baseline_model, 'predict_proba') else None
        
        # Evaluate baseline fairness
        baseline_fairness = self._evaluate_fairness(X_test, y_test, baseline_pred, baseline_pred_proba)
        results['original_performance'] = baseline_fairness
        
        # Apply mitigation based on strategy
        if mitigation_strategy in ["preprocessing", "comprehensive"]:
            X_train_processed, y_train_processed, preprocessing_info = self._apply_preprocessing_mitigation(
                X_train, y_train
            )
            results['mitigation_techniques_applied'].extend(preprocessing_info['techniques'])
        else:
            X_train_processed, y_train_processed = X_train, y_train
        
        if mitigation_strategy in ["inprocessing", "comprehensive"]:
            mitigated_model, inprocessing_info = self._apply_inprocessing_mitigation(
                X_train_processed, y_train_processed, model_type
            )
            results['mitigation_techniques_applied'].extend(inprocessing_info['techniques'])
        else:
            mitigated_model = self._train_baseline_model(X_train_processed, y_train_processed, model_type)
        
        # Generate predictions from mitigated model
        mitigated_pred = mitigated_model.predict(X_test)
        mitigated_pred_proba = mitigated_model.predict_proba(X_test)[:, 1] if hasattr(mitigated_model, 'predict_proba') else None
        
        if mitigation_strategy in ["postprocessing", "comprehensive"]:
            mitigated_pred, mitigated_pred_proba, postprocessing_info = self._apply_postprocessing_mitigation(
                X_test, y_test, mitigated_pred, mitigated_pred_proba
            )
            results['mitigation_techniques_applied'].extend(postprocessing_info['techniques'])
        
        # Evaluate mitigated fairness
        mitigated_fairness = self._evaluate_fairness(X_test, y_test, mitigated_pred, mitigated_pred_proba)
        results['mitigated_performance'] = mitigated_fairness
        
        # Calculate improvement
        results['fairness_improvement'] = self._calculate_fairness_improvement(
            baseline_fairness, mitigated_fairness
        )
        
        # Generate recommendations
        results['recommendations'] = self._generate_mitigation_recommendations(results)
        
        # Store mitigation record
        self.mitigation_history.append({
            'timestamp': datetime.now(),
            'strategy': mitigation_strategy,
            'results': results
        })
        
        logger.info(f"Bias mitigation completed. Applied {len(results['mitigation_techniques_applied'])} techniques.")
        
        return results
    
    def _train_baseline_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str
    ) -> Any:
        """Train baseline model without bias mitigation."""
        # Prepare features (encode categorical variables)
        X_train_encoded = self._encode_features(X_train)
        
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
            # Scale features for logistic regression
            scaler = StandardScaler()
            X_train_encoded = scaler.fit_transform(X_train_encoded)
            model.scaler = scaler  # Store scaler for later use
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train_encoded, y_train)
        return model
    
    def _encode_features(self, X: pd.DataFrame) -> np.ndarray:
        """Encode categorical features for model training."""
        X_encoded = X.copy()
        
        # Encode protected attributes and other categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col in self.protected_attributes:
                # Store encoding for protected attributes
                if not hasattr(self, 'protected_encoders'):
                    self.protected_encoders = {}
                
                if col not in self.protected_encoders:
                    self.protected_encoders[col] = LabelEncoder()
                    X_encoded[col] = self.protected_encoders[col].fit_transform(X[col])
                else:
                    X_encoded[col] = self.protected_encoders[col].transform(X[col])
            else:
                # Standard encoding for other categorical variables
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
        
        return X_encoded.values
    
    def _apply_preprocessing_mitigation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Apply preprocessing bias mitigation techniques."""
        logger.info("Applying preprocessing bias mitigation")
        
        X_processed = X_train.copy()
        y_processed = y_train.copy()
        techniques_applied = []
        
        # 1. Apply reweighing
        X_processed, y_processed, sample_weights = self._apply_reweighing(X_processed, y_processed)
        techniques_applied.append("reweighing")
        
        # 2. Apply disparate impact remover for continuous features
        X_processed = self._apply_disparate_impact_remover(X_processed)
        techniques_applied.append("disparate_impact_remover")
        
        # 3. Apply data augmentation for underrepresented groups
        X_processed, y_processed = self._apply_data_augmentation(X_processed, y_processed)
        techniques_applied.append("data_augmentation")
        
        preprocessing_info = {
            'techniques': techniques_applied,
            'sample_weights': sample_weights if 'sample_weights' in locals() else None
        }
        
        return X_processed, y_processed, preprocessing_info
    
    def _apply_reweighing(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Apply reweighing to balance representation."""
        # Calculate sample weights to balance groups
        sample_weights = np.ones(len(X))
        
        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue
            
            # Calculate weights for each group
            for outcome in [0, 1]:
                for group in X[attr].unique():
                    mask = (X[attr] == group) & (y == outcome)
                    group_size = mask.sum()
                    
                    if group_size > 0:
                        # Weight inversely proportional to group size
                        total_size = len(X)
                        expected_size = total_size / (len(X[attr].unique()) * 2)  # 2 outcomes
                        weight = expected_size / group_size
                        sample_weights[mask] *= weight
        
        # Normalize weights
        sample_weights = sample_weights / sample_weights.mean()
        
        logger.info(f"Applied reweighing. Weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")
        
        return X, y, sample_weights
    
    def _apply_disparate_impact_remover(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply disparate impact remover to continuous features."""
        X_processed = X.copy()
        
        # Apply to continuous features only
        continuous_features = X.select_dtypes(include=[np.number]).columns
        
        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue
            
            for feature in continuous_features:
                if feature == attr:
                    continue
                
                # Calculate group means
                group_means = X.groupby(attr)[feature].mean()
                overall_mean = X[feature].mean()
                
                # Adjust feature values to reduce disparate impact
                for group in X[attr].unique():
                    mask = X[attr] == group
                    group_mean = group_means[group]
                    
                    # Adjust towards overall mean (partial correction)
                    adjustment_factor = 0.5  # Partial correction to preserve information
                    adjustment = (overall_mean - group_mean) * adjustment_factor
                    X_processed.loc[mask, feature] += adjustment
        
        logger.info("Applied disparate impact remover to continuous features")
        
        return X_processed
    
    def _apply_data_augmentation(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply data augmentation for underrepresented groups."""
        X_augmented = X.copy()
        y_augmented = y.copy()
        
        # Identify underrepresented groups (less than 5% of data)
        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue
            
            group_sizes = X[attr].value_counts()
            total_size = len(X)
            
            for group, size in group_sizes.items():
                if size / total_size < 0.05:  # Underrepresented group
                    # Generate synthetic samples using SMOTE-like approach
                    group_data = X[X[attr] == group]
                    group_labels = y[X[attr] == group]
                    
                    if len(group_data) > 1:
                        # Simple augmentation: add noise to existing samples
                        n_augment = min(100, int(total_size * 0.02))  # Augment up to 2% of total
                        
                        for _ in range(n_augment):
                            # Select random sample from group
                            idx = np.random.choice(group_data.index)
                            sample = group_data.loc[idx].copy()
                            label = group_labels.loc[idx]
                            
                            # Add noise to continuous features
                            continuous_features = group_data.select_dtypes(include=[np.number]).columns
                            for feature in continuous_features:
                                if feature != attr:
                                    noise = np.random.normal(0, group_data[feature].std() * 0.1)
                                    sample[feature] += noise
                            
                            # Add to augmented dataset
                            X_augmented = pd.concat([X_augmented, sample.to_frame().T], ignore_index=True)
                            y_augmented = pd.concat([y_augmented, pd.Series([label])], ignore_index=True)
        
        logger.info(f"Data augmentation: {len(X_augmented) - len(X)} samples added")
        
        return X_augmented, y_augmented
    
    def _apply_fair_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fair feature selection to remove biased features."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated techniques
        
        X_selected = X.copy()
        
        # Remove features that are highly correlated with protected attributes
        correlation_threshold = 0.7
        
        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue
            
            # Calculate correlations with protected attribute
            if X[attr].dtype == 'object':
                # For categorical protected attributes, use encoded version
                attr_encoded = LabelEncoder().fit_transform(X[attr])
            else:
                attr_encoded = X[attr]
            
            correlations = X.select_dtypes(include=[np.number]).corrwith(pd.Series(attr_encoded))
            
            # Remove highly correlated features
            features_to_remove = correlations[abs(correlations) > correlation_threshold].index
            features_to_remove = [f for f in features_to_remove if f != attr]
            
            if features_to_remove:
                X_selected = X_selected.drop(columns=features_to_remove)
                logger.info(f"Removed {len(features_to_remove)} features correlated with {attr}")
        
        return X_selected
    
    def _apply_inprocessing_mitigation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Apply in-processing bias mitigation techniques."""
        logger.info("Applying in-processing bias mitigation")
        
        techniques_applied = []
        
        if model_type == "adversarial":
            model = self._apply_adversarial_debiasing(X_train, y_train)
            techniques_applied.append("adversarial_debiasing")
        elif model_type == "fair_representation":
            model = self._apply_fair_representation(X_train, y_train)
            techniques_applied.append("fair_representation")
        else:
            # Apply constrained optimization to standard models
            model = self._apply_constrained_optimization(X_train, y_train, model_type)
            techniques_applied.append("constrained_optimization")
        
        inprocessing_info = {
            'techniques': techniques_applied
        }
        
        return model, inprocessing_info
    
    def _apply_adversarial_debiasing(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Apply adversarial debiasing using neural networks."""
        # Encode features
        X_encoded = self._encode_features(X_train)
        
        # Build adversarial debiasing model
        input_dim = X_encoded.shape[1]
        
        # Main classifier
        classifier_input = keras.Input(shape=(input_dim,))
        classifier_hidden = layers.Dense(64, activation='relu')(classifier_input)
        classifier_hidden = layers.Dropout(0.3)(classifier_hidden)
        classifier_hidden = layers.Dense(32, activation='relu')(classifier_hidden)
        classifier_output = layers.Dense(1, activation='sigmoid', name='classifier')(classifier_hidden)
        
        # Adversarial network (predicts protected attribute)
        adversary_hidden = layers.Dense(32, activation='relu')(classifier_hidden)
        adversary_output = layers.Dense(len(self.protected_attributes), activation='sigmoid', name='adversary')(adversary_hidden)
        
        # Combined model
        model = keras.Model(inputs=classifier_input, outputs=[classifier_output, adversary_output])
        
        # Custom loss function for adversarial training
        def adversarial_loss(y_true, y_pred):
            classifier_loss = keras.losses.binary_crossentropy(y_true[0], y_pred[0])
            adversary_loss = keras.losses.binary_crossentropy(y_true[1], y_pred[1])
            
            # Adversarial training: minimize classifier loss, maximize adversary loss
            return classifier_loss - 0.1 * adversary_loss
        
        model.compile(
            optimizer='adam',
            loss={'classifier': 'binary_crossentropy', 'adversary': 'binary_crossentropy'},
            loss_weights={'classifier': 1.0, 'adversary': -0.1}  # Negative weight for adversarial loss
        )
        
        # Prepare protected attribute labels
        protected_labels = np.zeros((len(X_train), len(self.protected_attributes)))
        for i, attr in enumerate(self.protected_attributes):
            if attr in X_train.columns:
                if X_train[attr].dtype == 'object':
                    protected_labels[:, i] = LabelEncoder().fit_transform(X_train[attr])
                else:
                    protected_labels[:, i] = X_train[attr]
        
        # Train model
        model.fit(
            X_encoded,
            {'classifier': y_train, 'adversary': protected_labels},
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Create wrapper for sklearn compatibility
        class AdversarialModel:
            def __init__(self, keras_model):
                self.model = keras_model
            
            def predict(self, X):
                X_encoded = self._encode_features_predict(X)
                pred_proba = self.model.predict(X_encoded)[0]  # Get classifier output
                return (pred_proba > 0.5).astype(int).flatten()
            
            def predict_proba(self, X):
                X_encoded = self._encode_features_predict(X)
                pred_proba = self.model.predict(X_encoded)[0]  # Get classifier output
                return np.column_stack([1 - pred_proba.flatten(), pred_proba.flatten()])
            
            def _encode_features_predict(self, X):
                # Use the same encoding as training
                return self._encode_features(X)
        
        adversarial_model = AdversarialModel(model)
        adversarial_model._encode_features = self._encode_features
        
        logger.info("Applied adversarial debiasing")
        
        return adversarial_model
    
    def _apply_fair_representation(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Apply fair representation learning."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated fair representation techniques
        
        # Encode features
        X_encoded = self._encode_features(X_train)
        
        # Build autoencoder for fair representation
        input_dim = X_encoded.shape[1]
        encoding_dim = max(10, input_dim // 4)
        
        # Encoder
        encoder_input = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(encoder_input)
        
        # Decoder
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
        
        # Autoencoder
        autoencoder = keras.Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train autoencoder
        autoencoder.fit(X_encoded, X_encoded, epochs=50, batch_size=32, verbose=0)
        
        # Extract fair representations
        encoder = keras.Model(encoder_input, encoded)
        fair_representations = encoder.predict(X_encoded)
        
        # Train classifier on fair representations
        classifier = LogisticRegression(random_state=42)
        classifier.fit(fair_representations, y_train)
        
        # Create wrapper model
        class FairRepresentationModel:
            def __init__(self, encoder, classifier):
                self.encoder = encoder
                self.classifier = classifier
            
            def predict(self, X):
                X_encoded = self._encode_features_predict(X)
                representations = self.encoder.predict(X_encoded)
                return self.classifier.predict(representations)
            
            def predict_proba(self, X):
                X_encoded = self._encode_features_predict(X)
                representations = self.encoder.predict(X_encoded)
                return self.classifier.predict_proba(representations)
            
            def _encode_features_predict(self, X):
                return self._encode_features(X)
        
        fair_model = FairRepresentationModel(encoder, classifier)
        fair_model._encode_features = self._encode_features
        
        logger.info("Applied fair representation learning")
        
        return fair_model
    
    def _apply_constrained_optimization(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str
    ) -> Any:
        """Apply constrained optimization for fairness."""
        # This is a simplified implementation
        # In practice, this would use specialized libraries like fairlearn
        
        # Train multiple models with different regularization
        models = []
        fairness_scores = []
        
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            if model_type == "logistic_regression":
                model = LogisticRegression(C=1/alpha, random_state=42, max_iter=1000)
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=int(10/alpha),
                    random_state=42
                )
            
            X_encoded = self._encode_features(X_train)
            model.fit(X_encoded, y_train)
            
            # Evaluate fairness (simplified)
            pred = model.predict(X_encoded)
            fairness_score = self._calculate_simple_fairness(X_train, pred)
            
            models.append(model)
            fairness_scores.append(fairness_score)
        
        # Select model with best fairness-accuracy trade-off
        best_idx = np.argmin(fairness_scores)
        best_model = models[best_idx]
        
        logger.info(f"Applied constrained optimization. Selected model with fairness score: {fairness_scores[best_idx]:.3f}")
        
        return best_model
    
    def _calculate_simple_fairness(self, X: pd.DataFrame, y_pred: np.ndarray) -> float:
        """Calculate simple fairness metric for model selection."""
        fairness_violations = 0
        
        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue
            
            groups = X[attr].unique()
            if len(groups) < 2:
                continue
            
            # Calculate prediction rates for each group
            pred_rates = []
            for group in groups:
                mask = X[attr] == group
                if mask.sum() > 0:
                    pred_rate = np.mean(y_pred[mask])
                    pred_rates.append(pred_rate)
            
            # Check for disparate impact
            if len(pred_rates) >= 2:
                max_rate = max(pred_rates)
                min_rate = min(pred_rates)
                if max_rate > 0:
                    disparate_impact = min_rate / max_rate
                    if disparate_impact < 0.8:  # 80% rule
                        fairness_violations += 1
        
        return fairness_violations
    
    def _apply_postprocessing_mitigation(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Apply post-processing bias mitigation techniques."""
        logger.info("Applying post-processing bias mitigation")
        
        techniques_applied = []
        
        # Apply threshold optimization
        y_pred_adjusted, thresholds = self._apply_threshold_optimization(X_test, y_test, y_pred_proba)
        techniques_applied.append("threshold_optimization")
        
        # Apply calibration if probabilities available
        if y_pred_proba is not None:
            y_pred_proba_calibrated = self._apply_calibration(X_test, y_test, y_pred_proba)
            techniques_applied.append("calibration")
        else:
            y_pred_proba_calibrated = y_pred_proba
        
        postprocessing_info = {
            'techniques': techniques_applied,
            'group_thresholds': thresholds if 'thresholds' in locals() else None
        }
        
        return y_pred_adjusted, y_pred_proba_calibrated, postprocessing_info
    
    def _apply_threshold_optimization(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred_proba: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply group-specific threshold optimization."""
        if y_pred_proba is None:
            return y_test.values, {}
        
        y_pred_adjusted = np.zeros_like(y_test)
        group_thresholds = {}
        
        for attr in self.protected_attributes:
            if attr not in X_test.columns:
                continue
            
            # Find optimal thresholds for each group to achieve equal opportunity
            for group in X_test[attr].unique():
                mask = X_test[attr] == group
                
                if mask.sum() == 0:
                    continue
                
                group_y_true = y_test[mask]
                group_y_proba = y_pred_proba[mask]
                
                # Find threshold that maximizes F1 score for this group
                thresholds = np.linspace(0.1, 0.9, 50)
                best_threshold = 0.5
                best_f1 = 0
                
                for threshold in thresholds:
                    group_pred = (group_y_proba >= threshold).astype(int)
                    
                    if len(np.unique(group_pred)) > 1:  # Avoid division by zero
                        from sklearn.metrics import f1_score
                        f1 = f1_score(group_y_true, group_pred, zero_division=0)
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                
                # Apply group-specific threshold
                y_pred_adjusted[mask] = (group_y_proba >= best_threshold).astype(int)
                group_thresholds[f"{attr}_{group}"] = best_threshold
        
        # For samples not covered by protected attributes, use default threshold
        uncovered_mask = np.ones(len(y_test), dtype=bool)
        for attr in self.protected_attributes:
            if attr in X_test.columns:
                for group in X_test[attr].unique():
                    mask = X_test[attr] == group
                    uncovered_mask &= ~mask
        
        if uncovered_mask.sum() > 0:
            y_pred_adjusted[uncovered_mask] = (y_pred_proba[uncovered_mask] >= 0.5).astype(int)
        
        logger.info(f"Applied threshold optimization. Group thresholds: {group_thresholds}")
        
        return y_pred_adjusted, group_thresholds
    
    def _apply_calibration(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred_proba: np.ndarray
    ) -> np.ndarray:
        """Apply group-specific calibration."""
        y_pred_proba_calibrated = y_pred_proba.copy()
        
        for attr in self.protected_attributes:
            if attr not in X_test.columns:
                continue
            
            for group in X_test[attr].unique():
                mask = X_test[attr] == group
                
                if mask.sum() < 10:  # Need minimum samples for calibration
                    continue
                
                group_y_true = y_test[mask]
                group_y_proba = y_pred_proba[mask]
                
                # Apply isotonic calibration for this group
                from sklearn.isotonic import IsotonicRegression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                
                try:
                    calibrated_proba = calibrator.fit_transform(group_y_proba, group_y_true)
                    y_pred_proba_calibrated[mask] = calibrated_proba
                except:
                    # If calibration fails, keep original probabilities
                    pass
        
        logger.info("Applied group-specific calibration")
        
        return y_pred_proba_calibrated
    
    def _apply_equalized_odds(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Apply equalized odds post-processing."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated techniques
        
        y_pred_adjusted = y_pred.copy()
        
        for attr in self.protected_attributes:
            if attr not in X_test.columns:
                continue
            
            groups = X_test[attr].unique()
            if len(groups) < 2:
                continue
            
            # Calculate TPR and FPR for each group
            group_metrics = {}
            for group in groups:
                mask = X_test[attr] == group
                group_y_true = y_test[mask]
                group_y_pred = y_pred[mask]
                
                if len(np.unique(group_y_true)) > 1:
                    tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
                    fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
                    tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
                    fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    group_metrics[group] = {'tpr': tpr, 'fpr': fpr, 'mask': mask}
            
            # Adjust predictions to equalize odds (simplified)
            if len(group_metrics) >= 2:
                target_tpr = np.mean([metrics['tpr'] for metrics in group_metrics.values()])
                target_fpr = np.mean([metrics['fpr'] for metrics in group_metrics.values()])
                
                for group, metrics in group_metrics.items():
                    mask = metrics['mask']
                    current_tpr = metrics['tpr']
                    current_fpr = metrics['fpr']
                    
                    # Simple adjustment (this is a placeholder for more sophisticated methods)
                    if current_tpr < target_tpr:
                        # Increase positive predictions for positive cases
                        positive_cases = mask & (y_test == 1) & (y_pred == 0)
                        if positive_cases.sum() > 0:
                            # Flip some false negatives to true positives
                            flip_count = min(positive_cases.sum(), int((target_tpr - current_tpr) * mask.sum()))
                            flip_indices = np.random.choice(
                                np.where(positive_cases)[0], 
                                size=flip_count, 
                                replace=False
                            )
                            y_pred_adjusted[flip_indices] = 1
        
        logger.info("Applied equalized odds post-processing")
        
        return y_pred_adjusted
    
    def _evaluate_fairness(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate fairness metrics for model predictions."""
        fairness_metrics = {}
        
        # Overall performance
        accuracy = accuracy_score(y_test, y_pred)
        fairness_metrics['overall_accuracy'] = accuracy
        
        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
            fairness_metrics['overall_auc'] = auc
        
        # Group-specific metrics
        for attr in self.protected_attributes:
            if attr not in X_test.columns:
                continue
            
            attr_metrics = {}
            groups = X_test[attr].unique()
            
            # Calculate metrics for each group
            group_accuracies = []
            group_tprs = []
            group_fprs = []
            
            for group in groups:
                mask = X_test[attr] == group
                
                if mask.sum() == 0:
                    continue
                
                group_y_true = y_test[mask]
                group_y_pred = y_pred[mask]
                
                # Group accuracy
                group_acc = accuracy_score(group_y_true, group_y_pred)
                group_accuracies.append(group_acc)
                
                # Group TPR and FPR
                if len(np.unique(group_y_true)) > 1:
                    tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
                    fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
                    tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
                    fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    group_tprs.append(tpr)
                    group_fprs.append(fpr)
                
                attr_metrics[f"group_{group}"] = {
                    'accuracy': group_acc,
                    'sample_size': mask.sum()
                }
            
            # Calculate fairness metrics
            if len(group_accuracies) >= 2:
                attr_metrics['accuracy_difference'] = max(group_accuracies) - min(group_accuracies)
                attr_metrics['accuracy_ratio'] = min(group_accuracies) / max(group_accuracies) if max(group_accuracies) > 0 else 0
            
            if len(group_tprs) >= 2:
                attr_metrics['tpr_difference'] = max(group_tprs) - min(group_tprs)
                attr_metrics['equal_opportunity_violation'] = attr_metrics['tpr_difference'] > self.fairness_constraints['equal_opportunity_difference']
            
            if len(group_fprs) >= 2:
                attr_metrics['fpr_difference'] = max(group_fprs) - min(group_fprs)
            
            fairness_metrics[attr] = attr_metrics
        
        return fairness_metrics
    
    def _calculate_fairness_improvement(
        self,
        baseline_fairness: Dict[str, Any],
        mitigated_fairness: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate improvement in fairness metrics."""
        improvement = {}
        
        for attr in self.protected_attributes:
            if attr in baseline_fairness and attr in mitigated_fairness:
                attr_improvement = {}
                
                baseline_attr = baseline_fairness[attr]
                mitigated_attr = mitigated_fairness[attr]
                
                # Compare accuracy differences
                if 'accuracy_difference' in baseline_attr and 'accuracy_difference' in mitigated_attr:
                    baseline_diff = baseline_attr['accuracy_difference']
                    mitigated_diff = mitigated_attr['accuracy_difference']
                    attr_improvement['accuracy_difference_improvement'] = baseline_diff - mitigated_diff
                
                # Compare TPR differences
                if 'tpr_difference' in baseline_attr and 'tpr_difference' in mitigated_attr:
                    baseline_tpr_diff = baseline_attr['tpr_difference']
                    mitigated_tpr_diff = mitigated_attr['tpr_difference']
                    attr_improvement['tpr_difference_improvement'] = baseline_tpr_diff - mitigated_tpr_diff
                
                improvement[attr] = attr_improvement
        
        # Overall improvement score
        improvement_scores = []
        for attr_imp in improvement.values():
            for imp_value in attr_imp.values():
                if isinstance(imp_value, (int, float)):
                    improvement_scores.append(imp_value)
        
        improvement['overall_improvement_score'] = np.mean(improvement_scores) if improvement_scores else 0
        
        return improvement
    
    def _generate_mitigation_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on mitigation results."""
        recommendations = []
        
        # Check overall improvement
        overall_improvement = results.get('fairness_improvement', {}).get('overall_improvement_score', 0)
        
        if overall_improvement > 0:
            recommendations.append("Bias mitigation was successful. Consider deploying the mitigated model.")
        else:
            recommendations.append("Bias mitigation showed limited improvement. Consider additional techniques.")
        
        # Check specific fairness violations
        mitigated_performance = results.get('mitigated_performance', {})
        
        for attr, attr_metrics in mitigated_performance.items():
            if isinstance(attr_metrics, dict):
                if attr_metrics.get('equal_opportunity_violation', False):
                    recommendations.append(f"Equal opportunity violation still present for {attr}. Consider stronger post-processing.")
                
                if attr_metrics.get('accuracy_difference', 0) > 0.1:
                    recommendations.append(f"Large accuracy difference for {attr}. Consider in-processing techniques.")
        
        # Technique-specific recommendations
        techniques_applied = results.get('mitigation_techniques_applied', [])
        
        if 'reweighing' in techniques_applied:
            recommendations.append("Reweighing applied. Monitor for potential overfitting to minority groups.")
        
        if 'adversarial_debiasing' in techniques_applied:
            recommendations.append("Adversarial debiasing applied. Validate performance on independent test set.")
        
        if 'threshold_optimization' in techniques_applied:
            recommendations.append("Group-specific thresholds applied. Ensure clinical acceptability of different thresholds.")
        
        return recommendations

# Example usage and demonstration
def main():
    """Demonstrate comprehensive bias mitigation in healthcare AI."""
    
    # Generate synthetic healthcare dataset with bias
    np.random.seed(42)
    n_samples = 5000
    
    # Create biased synthetic dataset
    data = pd.DataFrame({
        'age': np.random.normal(65, 15, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.45, 0.55]),
        'race': np.random.choice(['White', 'Black', 'Hispanic'], n_samples, p=[0.7, 0.2, 0.1]),
        'income': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
        'blood_pressure': np.random.normal(130, 20, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),
        'bmi': np.random.normal(28, 5, n_samples)
    })
    
    # Introduce systematic bias
    outcome_prob = 0.15
    
    # Race-based bias
    race_bias = {'White': 0.0, 'Black': 0.08, 'Hispanic': 0.05}
    for race, bias in race_bias.items():
        mask = data['race'] == race
        data.loc[mask, 'outcome_prob'] = outcome_prob + bias
    
    # Gender-based bias in measurements
    male_mask = data['gender'] == 'Male'
    data.loc[male_mask, 'blood_pressure'] += 5  # Systematic measurement bias
    
    # Generate biased outcomes
    data['heart_disease'] = np.random.binomial(1, data['outcome_prob'])
    
    print("Healthcare AI Bias Mitigation Demonstration")
    print("=" * 50)
    
    # Prepare data
    feature_columns = ['age', 'gender', 'race', 'income', 'blood_pressure', 'cholesterol', 'bmi']
    X = data[feature_columns]
    y = data['heart_disease']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize bias mitigator
    protected_attributes = ['gender', 'race', 'income']
    mitigator = HealthcareAIBiasMitigator(
        protected_attributes=protected_attributes,
        clinical_context={'domain': 'cardiovascular_disease'}
    )
    
    print(f"\nDataset: {len(X)} samples, {len(feature_columns)} features")
    print(f"Protected attributes: {protected_attributes}")
    print(f"Outcome prevalence: {y.mean():.3f}")
    
    # Test different mitigation strategies
    strategies = ["preprocessing", "inprocessing", "postprocessing", "comprehensive"]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {strategy.upper()} bias mitigation")
        print(f"{'='*60}")
        
        # Apply bias mitigation
        results = mitigator.mitigate_bias(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            mitigation_strategy=strategy,
            model_type="random_forest"
        )
        
        # Display results
        print(f"\nTechniques applied: {', '.join(results['mitigation_techniques_applied'])}")
        
        # Original vs mitigated performance
        original_acc = results['original_performance']['overall_accuracy']
        mitigated_acc = results['mitigated_performance']['overall_accuracy']
        
        print(f"\nOverall Performance:")
        print(f"  Original accuracy: {original_acc:.3f}")
        print(f"  Mitigated accuracy: {mitigated_acc:.3f}")
        print(f"  Accuracy change: {mitigated_acc - original_acc:+.3f}")
        
        # Fairness improvement
        improvement_score = results['fairness_improvement']['overall_improvement_score']
        print(f"  Fairness improvement score: {improvement_score:+.3f}")
        
        # Group-specific improvements
        print(f"\nGroup-specific Improvements:")
        for attr in protected_attributes:
            if attr in results['fairness_improvement']:
                attr_improvement = results['fairness_improvement'][attr]
                
                if 'accuracy_difference_improvement' in attr_improvement:
                    acc_imp = attr_improvement['accuracy_difference_improvement']
                    print(f"  {attr} accuracy difference improvement: {acc_imp:+.3f}")
                
                if 'tpr_difference_improvement' in attr_improvement:
                    tpr_imp = attr_improvement['tpr_difference_improvement']
                    print(f"  {attr} TPR difference improvement: {tpr_imp:+.3f}")
        
        # Recommendations
        print(f"\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n{'='*60}")
    print("Bias mitigation analysis completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
```

## 8.4 Continuous Monitoring and Evaluation

### 8.4.1 Production Bias Monitoring Systems

Continuous monitoring of bias in production healthcare AI systems is essential for maintaining fairness and detecting drift in model performance across different patient populations. Production monitoring systems must be designed to detect both sudden changes and gradual drift in fairness metrics.

**Real-time Fairness Dashboards**: Interactive dashboards that display key fairness metrics across protected attributes, updated in real-time as new predictions are made.

**Automated Alert Systems**: Threshold-based alerting systems that notify administrators when fairness metrics exceed acceptable bounds.

**Longitudinal Bias Analysis**: Tracking fairness metrics over time to identify trends and seasonal patterns in bias.

### 8.4.2 Bias Drift Detection

Healthcare AI systems may experience bias drift due to changes in patient populations, clinical practices, or data collection procedures. Detecting this drift requires sophisticated statistical methods adapted for fairness metrics.

**Statistical Process Control**: Applying control charts and statistical process control methods to fairness metrics to detect significant deviations from expected performance.

**Distribution Shift Detection**: Monitoring changes in the distribution of protected attributes and their relationship to outcomes.

**Performance Degradation Analysis**: Identifying when model performance degrades disproportionately for specific demographic groups.

## 8.5 Regulatory and Ethical Considerations

### 8.5.1 FDA Guidance on AI Bias

The FDA's evolving guidance on AI bias in medical devices emphasizes the importance of demonstrating fairness across relevant patient subgroups. Key requirements include:

**Subgroup Analysis**: Demonstrating model performance across clinically relevant subgroups defined by demographics, comorbidities, and other factors.

**Bias Testing Protocols**: Implementing systematic bias testing as part of the validation process for medical AI devices.

**Post-Market Surveillance**: Ongoing monitoring of AI system performance in real-world deployment to detect bias that may not have been apparent in clinical trials.

### 8.5.2 Ethical Frameworks for Healthcare AI Fairness

Ethical frameworks for healthcare AI fairness must balance multiple competing principles including beneficence, non-maleficence, autonomy, and justice.

**Distributive Justice**: Ensuring that the benefits and risks of AI systems are fairly distributed across patient populations.

**Procedural Justice**: Implementing fair processes for AI development, validation, and deployment that include diverse stakeholders.

**Recognition Justice**: Acknowledging and addressing historical inequities in healthcare that may be reflected in AI training data.

## 8.6 Case Studies in Healthcare AI Bias

### 8.6.1 Pulse Oximetry and Racial Bias

Recent research has revealed significant racial bias in pulse oximetry devices, which systematically overestimate oxygen saturation in patients with darker skin pigmentation. This case study illustrates how bias in medical devices can have serious clinical consequences and highlights the importance of bias testing across diverse patient populations.

### 8.6.2 Algorithmic Bias in Healthcare Resource Allocation

Studies of healthcare resource allocation algorithms have revealed systematic bias against Black patients, with algorithms systematically underestimating the healthcare needs of Black patients compared to white patients with similar health conditions. This case demonstrates the importance of careful algorithm design and validation.

### 8.6.3 Gender Bias in Cardiovascular Risk Assessment

Traditional cardiovascular risk assessment tools have been shown to underestimate risk in women, leading to delayed diagnosis and treatment. AI systems trained on historical data may perpetuate these biases unless specifically designed to address gender-based differences in disease presentation.

## 8.7 Future Directions in Healthcare AI Fairness

### 8.7.1 Intersectional Fairness

Future research in healthcare AI fairness must address intersectionality, recognizing that individuals may belong to multiple protected groups simultaneously and may experience unique forms of bias at these intersections.

**Multi-dimensional Fairness Metrics**: Developing fairness metrics that can assess bias across multiple protected attributes simultaneously.

**Intersectional Bias Detection**: Creating methods to detect bias that affects specific intersectional groups that may not be apparent when analyzing single attributes.

### 8.7.2 Causal Fairness in Healthcare

Causal approaches to fairness offer promising directions for healthcare AI by explicitly modeling the causal relationships between protected attributes, clinical variables, and outcomes.

**Counterfactual Fairness**: Ensuring that AI decisions would be the same in a counterfactual world where the individual belonged to a different demographic group.

**Path-specific Fairness**: Distinguishing between direct and indirect effects of protected attributes on outcomes through different causal pathways.

### 8.7.3 Participatory Approaches to Fairness

Engaging affected communities in defining fairness criteria and evaluating AI systems ensures that fairness metrics align with community values and priorities.

**Community-defined Fairness**: Working with patient communities to define what fairness means in specific healthcare contexts.

**Participatory Evaluation**: Including community representatives in the evaluation and validation of healthcare AI systems.

## 8.8 Conclusion

Bias detection and mitigation in healthcare AI represents one of the most critical challenges in the responsible deployment of AI systems in clinical practice. The frameworks and techniques presented in this chapter provide a comprehensive approach to identifying, measuring, and addressing bias throughout the AI development lifecycle.

The successful implementation of bias detection and mitigation requires not only technical expertise but also deep understanding of healthcare contexts, regulatory requirements, and ethical principles. As healthcare AI systems become more prevalent, the importance of robust bias detection and mitigation frameworks will only continue to grow.

The future of healthcare AI depends on our ability to develop systems that are not only clinically effective but also fair and equitable across all patient populations. The techniques and frameworks presented in this chapter provide the foundation for achieving this goal, but ongoing research and development will be essential as new forms of bias emerge and our understanding of fairness in healthcare continues to evolve.

## References

1. Rajkomar, A., et al. (2018). Ensuring fairness in machine learning to advance health equity. Annals of Internal Medicine, 169(12), 866-872. DOI: 10.7326/M18-1990

2. Verma, S., & Rubin, J. (2018). Fairness definitions explained. Proceedings of the International Workshop on Software Fairness, 1-7. DOI: 10.1145/3194770.3194776

3. Mehrabi, N., et al. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys, 54(6), 1-35. DOI: 10.1145/3457607

4. Chen, I. Y., et al. (2019). Ethical machine learning in healthcare. Annual Review of Biomedical Data Science, 2, 123-144. DOI: 10.1146/annurev-biodatasci-092820-114757

5. Obermeyer, Z., et al. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. Science, 366(6464), 447-453. DOI: 10.1126/science.aax2342

6. Sjoding, M. W., et al. (2020). Racial bias in pulse oximetry measurement. New England Journal of Medicine, 383(25), 2477-2478. DOI: 10.1056/NEJMc2029240

7. Larrazabal, A. J., et al. (2020). Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis. Proceedings of the National Academy of Sciences, 117(23), 12592-12594. DOI: 10.1073/pnas.1919012117

8. Gianfrancesco, M. A., et al. (2018). Potential biases in machine learning algorithms using electronic health record data. JAMA Internal Medicine, 178(11), 1544-1547. DOI: 10.1001/jamainternmed.2018.3763

9. Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big Data, 5(2), 153-163. DOI: 10.1089/big.2016.0047

10. Kusner, M. J., et al. (2017). Counterfactual fairness. Advances in Neural Information Processing Systems, 30, 4066-4076.
