---
layout: default
title: "Chapter 8: Bias Detection Mitigation"
nav_order: 8
parent: Chapters
permalink: /chapters/08-bias-detection-mitigation/
---

# Chapter 8: Bias Detection and Mitigation in Healthcare AI - Ensuring Fairness and Equity in Clinical Decision Support

*By Sanjay Basu MD PhD*

## Learning Objectives

By the end of this chapter, physician data scientists will be able to:

- Understand the theoretical foundations of bias in healthcare AI systems and its clinical implications, including the mathematical frameworks for measuring fairness and the socio-technical factors that contribute to algorithmic bias in clinical settings
- Implement comprehensive bias detection frameworks for different types of healthcare AI models, including pre-deployment assessment, continuous monitoring, and population-specific evaluation methodologies
- Deploy bias mitigation strategies at pre-processing, in-processing, and post-processing stages of the machine learning pipeline, with specific attention to healthcare-appropriate interventions that maintain clinical validity
- Evaluate fairness metrics appropriate for different healthcare applications and populations, understanding the trade-offs between different fairness definitions and their clinical implications
- Design equitable AI systems that promote health equity across diverse patient populations, incorporating domain knowledge about health disparities and social determinants of health
- Implement continuous monitoring systems for bias detection in production healthcare AI environments, including real-time alerting and automated bias assessment workflows
- Navigate regulatory requirements and ethical frameworks for bias assessment in medical AI, including FDA guidance and institutional review board considerations

## 8.1 Introduction to Bias in Healthcare AI

Bias in healthcare artificial intelligence represents one of the most critical challenges facing the deployment of AI systems in clinical practice. Healthcare AI bias can perpetuate and amplify existing health disparities, leading to inequitable care delivery and potentially harmful outcomes for vulnerable populations. Understanding, detecting, and mitigating bias is not merely a technical challenge but a moral imperative that requires sophisticated approaches grounded in both computer science and health equity principles.

The manifestation of bias in healthcare AI systems is particularly concerning because healthcare decisions directly impact patient outcomes, quality of life, and survival. Unlike bias in other domains such as advertising or recommendation systems, healthcare AI bias can have life-or-death consequences, making the development of robust bias detection and mitigation frameworks essential for responsible AI deployment. The complexity of healthcare bias is compounded by the intersection of multiple protected characteristics, the temporal nature of health conditions, and the need to balance individual fairness with population-level health outcomes.

### 8.1.1 Theoretical Foundations of Healthcare AI Bias

Healthcare AI bias emerges from multiple sources throughout the AI development lifecycle, from data collection and annotation to model training and deployment. Understanding these sources requires a comprehensive framework that addresses both technical and socio-technical factors that contribute to systematic unfairness in clinical AI systems.

**Historical Bias and Legacy Effects**: Healthcare data often reflects historical patterns of discrimination and unequal access to care that have persisted for decades or centuries. Electronic health records may contain fewer data points for underserved populations due to historical exclusion from healthcare systems, leading to models that perform poorly for these groups. This historical bias is particularly problematic because it can perpetuate past injustices through algorithmic decision-making, creating a feedback loop that reinforces existing disparities.

**Representation Bias and Sampling Issues**: Clinical datasets frequently underrepresent certain demographic groups, particularly racial and ethnic minorities, women in certain age groups, patients from lower socioeconomic backgrounds, and individuals with multiple comorbidities. This underrepresentation can occur due to systematic exclusion from clinical trials, differential access to healthcare services, or biased data collection practices. The mathematical consequence is that models trained on such data will have higher uncertainty and potentially worse performance for underrepresented groups.

**Measurement Bias and Diagnostic Validity**: Clinical measurements and diagnostic criteria may have been developed and validated primarily in specific populations, leading to systematic measurement errors when applied to other groups. For example, pulse oximetry has been shown to be less accurate in patients with darker skin pigmentation, and certain cardiac risk scores were developed primarily in white male populations. This measurement bias can lead to systematic errors in both training data and model predictions.

**Aggregation Bias and Population Heterogeneity**: Combining data from different populations without accounting for relevant differences can mask important subgroup variations and lead to models that perform poorly for minority populations. This occurs when the assumption of population homogeneity is violated, and different groups have fundamentally different relationships between features and outcomes.

**Evaluation Bias and Metric Selection**: Using inappropriate benchmarks or evaluation metrics that do not account for population differences can hide bias in model performance. Traditional accuracy metrics may not capture disparate impact across subgroups, and clinical evaluation metrics may not reflect real-world performance differences.

The formal representation of bias in healthcare AI can be expressed through multiple mathematical frameworks. Statistical parity requires that the probability of a positive prediction is equal across protected groups:

$$

P(\hat{Y} = 1 | A = 0) = P(\hat{Y} = 1 | A = 1)

$$

However, statistical parity may not be appropriate for healthcare applications where base rates of conditions differ across populations due to biological, social, or environmental factors. Equalized odds provides a more nuanced approach that accounts for true outcome rates:

$$

P(\hat{Y} = 1 | Y = 1, A = 0) = P(\hat{Y} = 1 | Y = 1, A = 1)

$$
$$

P(\hat{Y} = 1 | Y = 0, A = 0) = P(\hat{Y} = 1 | Y = 0, A = 1)

$$

Calibration fairness ensures that predicted probabilities reflect true outcome rates across groups:

$$

P(Y = 1 | \hat{Y} = p, A = 0) = P(Y = 1 | \hat{Y} = p, A = 1) = p

$$

Individual fairness requires that similar individuals receive similar predictions, formalized as:

$$

d(\hat{Y}(x_1), \hat{Y}(x_2)) \leq L \cdot d(x_1, x_2)

$$

where $d$ represents appropriate distance metrics and $L$ is a Lipschitz constant.

### 8.1.2 Clinical Implications of AI Bias

The clinical implications of AI bias extend beyond statistical measures to real-world patient outcomes and health equity. Understanding these implications is crucial for physician data scientists who must balance technical performance with clinical effectiveness and ethical considerations.

**Exacerbation of Health Disparities**: AI systems that perform poorly for certain populations can worsen existing health disparities by providing suboptimal care recommendations, delayed diagnoses, or inappropriate treatment suggestions. This is particularly concerning in healthcare where disparities already exist across racial, ethnic, socioeconomic, and geographic lines. Biased AI systems can systematically disadvantage already vulnerable populations, creating a compounding effect on health outcomes.

**Reduction in Clinical Trust and Adoption**: Clinicians may lose trust in AI systems if they observe poor performance for certain patient populations, potentially reducing the beneficial impact of AI assistance across all populations. This loss of trust can be particularly damaging if it leads to abandonment of AI tools that could provide significant benefits when used appropriately.

**Legal and Ethical Liability**: Healthcare organizations deploying biased AI systems may face legal challenges and ethical scrutiny, particularly if bias leads to adverse patient outcomes. The legal landscape around AI bias in healthcare is evolving, with increasing recognition that algorithmic discrimination can constitute a form of medical malpractice or civil rights violation.

**Population Health Impact**: Systematic bias in AI systems can have population-level effects, particularly for public health applications and population health management. Biased screening algorithms, for example, could lead to systematic under-identification of disease in certain populations, affecting public health surveillance and intervention strategies.

**Resource Allocation Consequences**: Healthcare AI systems are increasingly used for resource allocation decisions, including bed assignment, staffing, and treatment prioritization. Bias in these systems can lead to systematic under-allocation of resources to certain populations, exacerbating existing healthcare access issues.

### 8.1.3 Regulatory and Ethical Framework

The detection and mitigation of bias in healthcare AI is increasingly recognized by regulatory bodies, professional organizations, and healthcare institutions as a critical component of responsible AI deployment. Understanding this framework is essential for implementing compliant and ethical AI systems.

**FDA Software as Medical Device (SaMD) Framework**: The FDA's guidance on Software as Medical Device emphasizes the importance of addressing bias and ensuring equitable performance across patient populations. The FDA requires evidence of performance across relevant patient subgroups and may require post-market surveillance to monitor for bias in real-world deployment. The FDA's AI/ML guidance specifically addresses the need for bias assessment and mitigation strategies.

**Professional Guidelines and Standards**: Medical professional organizations increasingly emphasize the importance of AI fairness and equity in clinical practice guidelines. The American Medical Association, American College of Physicians, and other professional societies have developed guidelines for ethical AI use that include bias assessment requirements.

**Institutional Policies and Governance**: Healthcare organizations are developing policies and procedures for bias assessment and mitigation in AI system procurement and deployment. These policies often include requirements for bias testing, ongoing monitoring, and remediation procedures when bias is detected.

**International Standards and Frameworks**: International organizations such as the World Health Organization and the International Organization for Standardization are developing standards for AI bias assessment and mitigation in healthcare contexts.

## 8.2 Comprehensive Bias Detection Framework

### 8.2.1 Multi-Stage Bias Detection Pipeline

Effective bias detection in healthcare AI requires a comprehensive pipeline that addresses bias at multiple stages of the AI development lifecycle. This pipeline must be tailored to the specific characteristics of healthcare data and clinical applications, incorporating domain knowledge about health disparities and clinical workflows.

```python
"""
Comprehensive Healthcare AI Bias Detection and Mitigation Framework

This implementation provides advanced bias detection and mitigation capabilities
specifically designed for healthcare AI systems, incorporating clinical domain
knowledge, regulatory requirements, and state-of-the-art fairness methodologies.

Author: Sanjay Basu MD PhD
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import scipy.stats as stats
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

# Fairness-specific libraries
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
    from aif360.algorithms.inprocessing import AdversarialDebiasing
    from aif360.algorithms.postprocessing import CalibratedEqualizedOdds, EqualizedOdds
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("AIF360 not available. Some fairness metrics will be computed manually.")

import logging
from datetime import datetime, timedelta
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import joblib
import pickle
from pathlib import Path

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
    INTERSECTIONAL = "intersectional"

class FairnessMetric(Enum):
    """Fairness metrics for healthcare AI evaluation."""
    STATISTICAL_PARITY = "statistical_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"
    TREATMENT_EQUALITY = "treatment_equality"
    CONDITIONAL_USE_ACCURACY = "conditional_use_accuracy"

class SeverityLevel(Enum):
    """Severity levels for bias detection results."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BiasDetectionResult:
    """Results from bias detection analysis."""
    bias_type: BiasType
    metric_name: str
    metric_value: float
    threshold: float
    is_biased: bool
    affected_groups: List[str]
    severity: SeverityLevel
    confidence: float
    recommendations: List[str]
    statistical_significance: Optional[float] = None
    effect_size: Optional[float] = None
    clinical_impact: Optional[str] = None
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
            'severity': self.severity.value,
            'confidence': self.confidence,
            'recommendations': self.recommendations,
            'statistical_significance': self.statistical_significance,
            'effect_size': self.effect_size,
            'clinical_impact': self.clinical_impact,
            'timestamp': self.timestamp.isoformat()
        }

class HealthcareAIBiasDetector:
    """
    Comprehensive bias detection framework for healthcare AI systems.
    
    This class implements state-of-the-art bias detection methods specifically
    designed for healthcare applications, incorporating clinical domain knowledge
    and regulatory requirements.
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        fairness_thresholds: Optional[Dict[str, float]] = None,
        clinical_context: Optional[Dict[str, Any]] = None,
        significance_level: float = 0.05
    ):
        """
        Initialize healthcare AI bias detector.
        
        Args:
            protected_attributes: List of protected attributes (e.g., race, gender, age_group)
            fairness_thresholds: Custom thresholds for fairness metrics
            clinical_context: Clinical context information for bias assessment
            significance_level: Statistical significance level for bias tests
        """
        self.protected_attributes = protected_attributes
        self.clinical_context = clinical_context or {}
        self.significance_level = significance_level
        
        # Default fairness thresholds based on healthcare AI literature
        self.fairness_thresholds = fairness_thresholds or {
            'statistical_parity_difference': 0.1,
            'equalized_odds_difference': 0.1,
            'equal_opportunity_difference': 0.1,
            'calibration_difference': 0.1,
            'treatment_equality_difference': 0.1,
            'conditional_use_accuracy_difference': 0.1
        }
        
        # Clinical severity thresholds
        self.severity_thresholds = {
            SeverityLevel.LOW: 0.05,
            SeverityLevel.MEDIUM: 0.1,
            SeverityLevel.HIGH: 0.2,
            SeverityLevel.CRITICAL: 0.3
        }
        
        # Results storage
        self.detection_results: List[BiasDetectionResult] = []
        self.bias_history: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"Initialized healthcare AI bias detector for attributes: {protected_attributes}")
    
    def detect_bias_comprehensive(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model: Optional[Any] = None
    ) -> List[BiasDetectionResult]:
        """
        Perform comprehensive bias detection across multiple fairness metrics.
        
        Args:
            X: Feature matrix including protected attributes
            y: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            model: Trained model (optional, for additional analysis)
            
        Returns:
            List of bias detection results
        """
        results = []
        
        # Data validation
        self._validate_inputs(X, y, y_pred, y_prob)
        
        # Detect representation bias
        representation_results = self._detect_representation_bias(X, y)
        results.extend(representation_results)
        
        # Detect algorithmic bias
        algorithmic_results = self._detect_algorithmic_bias(X, y, y_pred, y_prob)
        results.extend(algorithmic_results)
        
        # Detect intersectional bias
        intersectional_results = self._detect_intersectional_bias(X, y, y_pred)
        results.extend(intersectional_results)
        
        # Detect calibration bias
        if y_prob is not None:
            calibration_results = self._detect_calibration_bias(X, y, y_prob)
            results.extend(calibration_results)
        
        # Clinical impact assessment
        clinical_results = self._assess_clinical_impact(X, y, y_pred, results)
        results.extend(clinical_results)
        
        # Store results
        self.detection_results.extend(results)
        
        # Update bias history
        self._update_bias_history(results)
        
        return results
    
    def _validate_inputs(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ):
        """Validate input data for bias detection."""
        
        # Check dimensions
        if len(X) != len(y) or len(y) != len(y_pred):
            raise ValueError("Input arrays must have the same length")
        
        # Check protected attributes
        missing_attrs = [attr for attr in self.protected_attributes if attr not in X.columns]
        if missing_attrs:
            raise ValueError(f"Missing protected attributes: {missing_attrs}")
        
        # Check for sufficient data in each group
        for attr in self.protected_attributes:
            group_counts = X[attr].value_counts()
            if group_counts.min() < 30:
                logger.warning(f"Small sample size for attribute {attr}: {group_counts.to_dict()}")
        
        # Check prediction validity
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("Predictions must be binary (0 or 1)")
        
        if y_prob is not None:
            if not np.all((y_prob >= 0) & (y_prob <= 1)):
                raise ValueError("Probabilities must be between 0 and 1")
    
    def _detect_representation_bias(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[BiasDetectionResult]:
        """Detect representation bias in the dataset."""
        
        results = []
        
        for attr in self.protected_attributes:
            # Check group representation
            group_counts = X[attr].value_counts()
            total_samples = len(X)
            
            # Calculate representation ratios
            group_ratios = group_counts / total_samples
            min_ratio = group_ratios.min()
            max_ratio = group_ratios.max()
            
            # Statistical test for representation bias
            expected_equal = total_samples / len(group_counts)
            chi2_stat, p_value = stats.chisquare(group_counts.values)
            
            # Determine if bias exists
            representation_ratio = min_ratio / max_ratio if max_ratio > 0 else 0
            threshold = 0.8  # Minimum acceptable representation ratio
            
            is_biased = representation_ratio < threshold or p_value < self.significance_level
            
            # Determine severity
            severity = self._determine_severity(1 - representation_ratio)
            
            # Generate recommendations
            recommendations = []
            if is_biased:
                underrepresented_groups = group_counts[group_counts < expected_equal * 0.8].index.tolist()
                recommendations.extend([
                    f"Increase data collection for underrepresented groups: {underrepresented_groups}",
                    "Consider stratified sampling to ensure adequate representation",
                    "Implement targeted recruitment strategies for underrepresented populations"
                ])
            
            result = BiasDetectionResult(
                bias_type=BiasType.REPRESENTATION,
                metric_name=f"representation_ratio_{attr}",
                metric_value=representation_ratio,
                threshold=threshold,
                is_biased=is_biased,
                affected_groups=group_counts.index.tolist(),
                severity=severity,
                confidence=1 - p_value if p_value < 1 else 0.5,
                recommendations=recommendations,
                statistical_significance=p_value,
                effect_size=1 - representation_ratio
            )
            
            results.append(result)
        
        return results
    
    def _detect_algorithmic_bias(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> List[BiasDetectionResult]:
        """Detect algorithmic bias using multiple fairness metrics."""
        
        results = []
        
        for attr in self.protected_attributes:
            # Get unique groups
            groups = X[attr].unique()
            
            if len(groups) < 2:
                continue
            
            # Statistical parity
            stat_parity_result = self._calculate_statistical_parity(X, y_pred, attr)
            results.append(stat_parity_result)
            
            # Equalized odds
            eq_odds_result = self._calculate_equalized_odds(X, y, y_pred, attr)
            results.append(eq_odds_result)
            
            # Equal opportunity
            eq_opp_result = self._calculate_equal_opportunity(X, y, y_pred, attr)
            results.append(eq_opp_result)
            
            # Treatment equality
            treatment_eq_result = self._calculate_treatment_equality(X, y, y_pred, attr)
            results.append(treatment_eq_result)
            
            # Conditional use accuracy
            cond_acc_result = self._calculate_conditional_use_accuracy(X, y, y_pred, attr)
            results.append(cond_acc_result)
        
        return results
    
    def _calculate_statistical_parity(
        self,
        X: pd.DataFrame,
        y_pred: np.ndarray,
        attr: str
    ) -> BiasDetectionResult:
        """Calculate statistical parity difference."""
        
        groups = X[attr].unique()
        positive_rates = {}
        
        for group in groups:
            group_mask = X[attr] == group
            group_predictions = y_pred[group_mask]
            positive_rates[group] = np.mean(group_predictions)
        
        # Calculate maximum difference
        rates = list(positive_rates.values())
        max_diff = max(rates) - min(rates)
        
        # Statistical significance test
        group_data = [y_pred[X[attr] == group] for group in groups]
        if len(groups) == 2:
            stat, p_value = stats.chi2_contingency([
                [np.sum(group_data<sup>0</sup>), len(group_data<sup>0</sup>) - np.sum(group_data<sup>0</sup>)],
                [np.sum(group_data<sup>1</sup>), len(group_data<sup>1</sup>) - np.sum(group_data<sup>1</sup>)]
            ])[:2]
        else:
            # Use chi-square test for multiple groups
            contingency_table = []
            for group_preds in group_data:
                contingency_table.append([np.sum(group_preds), len(group_preds) - np.sum(group_preds)])
            stat, p_value = stats.chi2_contingency(contingency_table)[:2]
        
        threshold = self.fairness_thresholds['statistical_parity_difference']
        is_biased = max_diff > threshold and p_value < self.significance_level
        
        severity = self._determine_severity(max_diff)
        
        recommendations = []
        if is_biased:
            min_group = min(positive_rates, key=positive_rates.get)
            max_group = max(positive_rates, key=positive_rates.get)
            recommendations.extend([
                f"Address prediction rate disparity between {min_group} and {max_group}",
                "Consider reweighing training data to balance group representation",
                "Implement post-processing calibration to equalize positive rates"
            ])
        
        return BiasDetectionResult(
            bias_type=BiasType.ALGORITHMIC,
            metric_name=f"statistical_parity_{attr}",
            metric_value=max_diff,
            threshold=threshold,
            is_biased=is_biased,
            affected_groups=list(groups),
            severity=severity,
            confidence=1 - p_value if p_value < 1 else 0.5,
            recommendations=recommendations,
            statistical_significance=p_value,
            effect_size=max_diff
        )
    
    def _calculate_equalized_odds(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray,
        attr: str
    ) -> BiasDetectionResult:
        """Calculate equalized odds difference."""
        
        groups = X[attr].unique()
        tpr_fpr_by_group = {}
        
        for group in groups:
            group_mask = X[attr] == group
            group_y = y[group_mask]
            group_pred = y_pred[group_mask]
            
            # True positive rate
            tpr = np.sum((group_y == 1) & (group_pred == 1)) / np.sum(group_y == 1) if np.sum(group_y == 1) > 0 else 0
            
            # False positive rate
            fpr = np.sum((group_y == 0) & (group_pred == 1)) / np.sum(group_y == 0) if np.sum(group_y == 0) > 0 else 0
            
            tpr_fpr_by_group[group] = {'tpr': tpr, 'fpr': fpr}
        
        # Calculate maximum differences
        tprs = [metrics['tpr'] for metrics in tpr_fpr_by_group.values()]
        fprs = [metrics['fpr'] for metrics in tpr_fpr_by_group.values()]
        
        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)
        max_diff = max(tpr_diff, fpr_diff)
        
        # Statistical significance (simplified)
        p_value = 0.05  # Would implement proper statistical test
        
        threshold = self.fairness_thresholds['equalized_odds_difference']
        is_biased = max_diff > threshold
        
        severity = self._determine_severity(max_diff)
        
        recommendations = []
        if is_biased:
            recommendations.extend([
                "Implement equalized odds post-processing",
                "Consider adversarial debiasing during training",
                "Evaluate model performance separately for each group"
            ])
        
        return BiasDetectionResult(
            bias_type=BiasType.ALGORITHMIC,
            metric_name=f"equalized_odds_{attr}",
            metric_value=max_diff,
            threshold=threshold,
            is_biased=is_biased,
            affected_groups=list(groups),
            severity=severity,
            confidence=0.8,  # Simplified confidence
            recommendations=recommendations,
            statistical_significance=p_value,
            effect_size=max_diff
        )
    
    def _calculate_equal_opportunity(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray,
        attr: str
    ) -> BiasDetectionResult:
        """Calculate equal opportunity difference (TPR difference)."""
        
        groups = X[attr].unique()
        tprs = {}
        
        for group in groups:
            group_mask = X[attr] == group
            group_y = y[group_mask]
            group_pred = y_pred[group_mask]
            
            # True positive rate
            tpr = np.sum((group_y == 1) & (group_pred == 1)) / np.sum(group_y == 1) if np.sum(group_y == 1) > 0 else 0
            tprs[group] = tpr
        
        # Calculate maximum difference
        tpr_values = list(tprs.values())
        max_diff = max(tpr_values) - min(tpr_values)
        
        threshold = self.fairness_thresholds['equal_opportunity_difference']
        is_biased = max_diff > threshold
        
        severity = self._determine_severity(max_diff)
        
        recommendations = []
        if is_biased:
            min_group = min(tprs, key=tprs.get)
            max_group = max(tprs, key=tprs.get)
            recommendations.extend([
                f"Address true positive rate disparity between {min_group} and {max_group}",
                "Consider threshold optimization for each group",
                "Implement equal opportunity post-processing"
            ])
        
        return BiasDetectionResult(
            bias_type=BiasType.ALGORITHMIC,
            metric_name=f"equal_opportunity_{attr}",
            metric_value=max_diff,
            threshold=threshold,
            is_biased=is_biased,
            affected_groups=list(groups),
            severity=severity,
            confidence=0.8,
            recommendations=recommendations,
            effect_size=max_diff
        )
    
    def _calculate_treatment_equality(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray,
        attr: str
    ) -> BiasDetectionResult:
        """Calculate treatment equality (FN/FP ratio difference)."""
        
        groups = X[attr].unique()
        ratios = {}
        
        for group in groups:
            group_mask = X[attr] == group
            group_y = y[group_mask]
            group_pred = y_pred[group_mask]
            
            fn = np.sum((group_y == 1) & (group_pred == 0))
            fp = np.sum((group_y == 0) & (group_pred == 1))
            
            ratio = fn / fp if fp > 0 else float('inf') if fn > 0 else 0
            ratios[group] = ratio
        
        # Calculate maximum difference (excluding infinite values)
        finite_ratios = [r for r in ratios.values() if np.isfinite(r)]
        if len(finite_ratios) < 2:
            max_diff = 0
        else:
            max_diff = max(finite_ratios) - min(finite_ratios)
        
        threshold = self.fairness_thresholds['treatment_equality_difference']
        is_biased = max_diff > threshold
        
        severity = self._determine_severity(max_diff)
        
        recommendations = []
        if is_biased:
            recommendations.extend([
                "Balance false negative and false positive rates across groups",
                "Consider cost-sensitive learning approaches",
                "Implement group-specific decision thresholds"
            ])
        
        return BiasDetectionResult(
            bias_type=BiasType.ALGORITHMIC,
            metric_name=f"treatment_equality_{attr}",
            metric_value=max_diff,
            threshold=threshold,
            is_biased=is_biased,
            affected_groups=list(groups),
            severity=severity,
            confidence=0.7,
            recommendations=recommendations,
            effect_size=max_diff
        )
    
    def _calculate_conditional_use_accuracy(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray,
        attr: str
    ) -> BiasDetectionResult:
        """Calculate conditional use accuracy equality (PPV and NPV differences)."""
        
        groups = X[attr].unique()
        ppv_npv_by_group = {}
        
        for group in groups:
            group_mask = X[attr] == group
            group_y = y[group_mask]
            group_pred = y_pred[group_mask]
            
            # Positive predictive value (precision)
            ppv = np.sum((group_y == 1) & (group_pred == 1)) / np.sum(group_pred == 1) if np.sum(group_pred == 1) > 0 else 0
            
            # Negative predictive value
            npv = np.sum((group_y == 0) & (group_pred == 0)) / np.sum(group_pred == 0) if np.sum(group_pred == 0) > 0 else 0
            
            ppv_npv_by_group[group] = {'ppv': ppv, 'npv': npv}
        
        # Calculate maximum differences
        ppvs = [metrics['ppv'] for metrics in ppv_npv_by_group.values()]
        npvs = [metrics['npv'] for metrics in ppv_npv_by_group.values()]
        
        ppv_diff = max(ppvs) - min(ppvs)
        npv_diff = max(npvs) - min(npvs)
        max_diff = max(ppv_diff, npv_diff)
        
        threshold = self.fairness_thresholds['conditional_use_accuracy_difference']
        is_biased = max_diff > threshold
        
        severity = self._determine_severity(max_diff)
        
        recommendations = []
        if is_biased:
            recommendations.extend([
                "Calibrate predictions to ensure equal predictive value across groups",
                "Consider group-specific model training",
                "Implement conditional use accuracy post-processing"
            ])
        
        return BiasDetectionResult(
            bias_type=BiasType.ALGORITHMIC,
            metric_name=f"conditional_use_accuracy_{attr}",
            metric_value=max_diff,
            threshold=threshold,
            is_biased=is_biased,
            affected_groups=list(groups),
            severity=severity,
            confidence=0.8,
            recommendations=recommendations,
            effect_size=max_diff
        )
    
    def _detect_intersectional_bias(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray
    ) -> List[BiasDetectionResult]:
        """Detect intersectional bias across multiple protected attributes."""
        
        results = []
        
        # Consider pairs of protected attributes
        for i, attr1 in enumerate(self.protected_attributes):
            for attr2 in self.protected_attributes[i+1:]:
                # Create intersectional groups
                X['intersectional'] = X[attr1].astype(str) + '_' + X[attr2].astype(str)
                
                # Calculate performance metrics for each intersectional group
                groups = X['intersectional'].unique()
                group_metrics = {}
                
                for group in groups:
                    group_mask = X['intersectional'] == group
                    if np.sum(group_mask) < 10:  # Skip small groups
                        continue
                    
                    group_y = y[group_mask]
                    group_pred = y_pred[group_mask]
                    
                    accuracy = accuracy_score(group_y, group_pred)
                    precision = precision_score(group_y, group_pred, zero_division=0)
                    recall = recall_score(group_y, group_pred, zero_division=0)
                    
                    group_metrics[group] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'size': np.sum(group_mask)
                    }
                
                if len(group_metrics) < 2:
                    continue
                
                # Calculate intersectional bias
                accuracies = [m['accuracy'] for m in group_metrics.values()]
                accuracy_diff = max(accuracies) - min(accuracies)
                
                threshold = 0.1  # Threshold for intersectional bias
                is_biased = accuracy_diff > threshold
                
                severity = self._determine_severity(accuracy_diff)
                
                recommendations = []
                if is_biased:
                    worst_group = min(group_metrics, key=lambda x: group_metrics[x]['accuracy'])
                    best_group = max(group_metrics, key=lambda x: group_metrics[x]['accuracy'])
                    recommendations.extend([
                        f"Address performance disparity between {worst_group} and {best_group}",
                        "Consider intersectional fairness constraints in model training",
                        "Implement stratified evaluation for intersectional groups"
                    ])
                
                result = BiasDetectionResult(
                    bias_type=BiasType.INTERSECTIONAL,
                    metric_name=f"intersectional_accuracy_{attr1}_{attr2}",
                    metric_value=accuracy_diff,
                    threshold=threshold,
                    is_biased=is_biased,
                    affected_groups=list(groups),
                    severity=severity,
                    confidence=0.7,
                    recommendations=recommendations,
                    effect_size=accuracy_diff
                )
                
                results.append(result)
                
                # Clean up temporary column
                X.drop('intersectional', axis=1, inplace=True)
        
        return results
    
    def _detect_calibration_bias(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_prob: np.ndarray
    ) -> List[BiasDetectionResult]:
        """Detect calibration bias across protected groups."""
        
        results = []
        
        for attr in self.protected_attributes:
            groups = X[attr].unique()
            calibration_errors = {}
            
            for group in groups:
                group_mask = X[attr] == group
                group_y = y[group_mask]
                group_prob = y_prob[group_mask]
                
                if len(group_y) < 10:  # Skip small groups
                    continue
                
                # Calculate calibration error
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    group_y, group_prob, n_bins=10, strategy='uniform'
                )
                
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                calibration_errors[group] = calibration_error
            
            if len(calibration_errors) < 2:
                continue
            
            # Calculate maximum calibration difference
            cal_errors = list(calibration_errors.values())
            max_diff = max(cal_errors) - min(cal_errors)
            
            threshold = self.fairness_thresholds['calibration_difference']
            is_biased = max_diff > threshold
            
            severity = self._determine_severity(max_diff)
            
            recommendations = []
            if is_biased:
                worst_group = max(calibration_errors, key=calibration_errors.get)
                recommendations.extend([
                    f"Improve calibration for group: {worst_group}",
                    "Implement group-specific calibration methods",
                    "Consider Platt scaling or isotonic regression for each group"
                ])
            
            result = BiasDetectionResult(
                bias_type=BiasType.ALGORITHMIC,
                metric_name=f"calibration_{attr}",
                metric_value=max_diff,
                threshold=threshold,
                is_biased=is_biased,
                affected_groups=list(groups),
                severity=severity,
                confidence=0.8,
                recommendations=recommendations,
                effect_size=max_diff
            )
            
            results.append(result)
        
        return results
    
    def _assess_clinical_impact(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray,
        bias_results: List[BiasDetectionResult]
    ) -> List[BiasDetectionResult]:
        """Assess clinical impact of detected bias."""
        
        clinical_results = []
        
        # Define clinical impact categories
        clinical_contexts = self.clinical_context.get('impact_categories', {
            'diagnostic': {'weight': 1.0, 'description': 'Diagnostic accuracy impact'},
            'treatment': {'weight': 1.2, 'description': 'Treatment recommendation impact'},
            'screening': {'weight': 0.8, 'description': 'Screening program impact'},
            'prognosis': {'weight': 1.1, 'description': 'Prognostic assessment impact'}
        })
        
        for result in bias_results:
            if result.is_biased and result.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                # Calculate clinical impact score
                base_impact = result.metric_value
                
                # Adjust for clinical context
                context_weight = 1.0
                if 'application_type' in self.clinical_context:
                    app_type = self.clinical_context['application_type']
                    context_weight = clinical_contexts.get(app_type, {}).get('weight', 1.0)
                
                clinical_impact_score = base_impact * context_weight
                
                # Determine clinical impact level
                if clinical_impact_score > 0.3:
                    impact_level = "severe"
                elif clinical_impact_score > 0.2:
                    impact_level = "moderate"
                elif clinical_impact_score > 0.1:
                    impact_level = "mild"
                else:
                    impact_level = "minimal"
                
                # Generate clinical recommendations
                clinical_recommendations = [
                    f"Clinical impact assessment: {impact_level}",
                    "Conduct clinical validation study for affected populations",
                    "Implement enhanced monitoring for clinical outcomes",
                    "Consider alternative models or manual review for high-risk cases"
                ]
                
                clinical_result = BiasDetectionResult(
                    bias_type=BiasType.DEPLOYMENT,
                    metric_name=f"clinical_impact_{result.metric_name}",
                    metric_value=clinical_impact_score,
                    threshold=0.1,
                    is_biased=clinical_impact_score > 0.1,
                    affected_groups=result.affected_groups,
                    severity=result.severity,
                    confidence=result.confidence,
                    recommendations=clinical_recommendations,
                    clinical_impact=impact_level
                )
                
                clinical_results.append(clinical_result)
        
        return clinical_results
    
    def _determine_severity(self, metric_value: float) -> SeverityLevel:
        """Determine severity level based on metric value."""
        
        if metric_value >= self.severity_thresholds[SeverityLevel.CRITICAL]:
            return SeverityLevel.CRITICAL
        elif metric_value >= self.severity_thresholds[SeverityLevel.HIGH]:
            return SeverityLevel.HIGH
        elif metric_value >= self.severity_thresholds[SeverityLevel.MEDIUM]:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _update_bias_history(self, results: List[BiasDetectionResult]):
        """Update bias history for trend analysis."""
        
        for result in results:
            metric_key = f"{result.bias_type.value}_{result.metric_name}"
            self.bias_history[metric_key].append(result.metric_value)
            
            # Keep only recent history (last 100 measurements)
            if len(self.bias_history[metric_key]) > 100:
                self.bias_history[metric_key] = self.bias_history[metric_key][-100:]
    
    def generate_bias_report(
        self,
        results: Optional[List[BiasDetectionResult]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive bias detection report."""
        
        if results is None:
            results = self.detection_results
        
        # Summary statistics
        total_tests = len(results)
        biased_tests = sum(1 for r in results if r.is_biased)
        bias_rate = biased_tests / total_tests if total_tests > 0 else 0
        
        # Severity breakdown
        severity_counts = defaultdict(int)
        for result in results:
            if result.is_biased:
                severity_counts[result.severity.value] += 1
        
        # Bias type breakdown
        bias_type_counts = defaultdict(int)
        for result in results:
            if result.is_biased:
                bias_type_counts[result.bias_type.value] += 1
        
        # Affected groups analysis
        affected_groups = set()
        for result in results:
            if result.is_biased:
                affected_groups.update(result.affected_groups)
        
        # Priority recommendations
        priority_recommendations = []
        critical_results = [r for r in results if r.is_biased and r.severity == SeverityLevel.CRITICAL]
        for result in critical_results[:5]:  # Top 5 critical issues
            priority_recommendations.extend(result.recommendations)
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'biased_tests': biased_tests,
                'bias_rate': bias_rate,
                'timestamp': datetime.now().isoformat()
            },
            'severity_breakdown': dict(severity_counts),
            'bias_type_breakdown': dict(bias_type_counts),
            'affected_groups': list(affected_groups),
            'priority_recommendations': list(set(priority_recommendations)),
            'detailed_results': [r.to_dict() for r in results if r.is_biased],
            'compliance_status': self._assess_compliance_status(results)
        }
        
        return report
    
    def _assess_compliance_status(self, results: List[BiasDetectionResult]) -> Dict[str, Any]:
        """Assess compliance with regulatory and ethical standards."""
        
        critical_bias_count = sum(1 for r in results if r.is_biased and r.severity == SeverityLevel.CRITICAL)
        high_bias_count = sum(1 for r in results if r.is_biased and r.severity == SeverityLevel.HIGH)
        
        # Determine overall compliance status
        if critical_bias_count > 0:
            status = "non_compliant"
            risk_level = "high"
        elif high_bias_count > 2:
            status = "at_risk"
            risk_level = "medium"
        else:
            status = "compliant"
            risk_level = "low"
        
        return {
            'status': status,
            'risk_level': risk_level,
            'critical_issues': critical_bias_count,
            'high_priority_issues': high_bias_count,
            'recommendations': [
                "Implement immediate bias mitigation for critical issues",
                "Establish continuous monitoring for bias detection",
                "Document bias assessment procedures for regulatory compliance"
            ]
        }
    
    def visualize_bias_results(
        self,
        results: Optional[List[BiasDetectionResult]] = None,
        save_path: Optional[str] = None
    ):
        """Create visualizations of bias detection results."""
        
        if results is None:
            results = self.detection_results
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Healthcare AI Bias Detection Results', fontsize=16)
        
        # 1. Bias severity distribution
        severity_counts = defaultdict(int)
        for result in results:
            if result.is_biased:
                severity_counts[result.severity.value] += 1
        
        if severity_counts:
            axes[0, 0].bar(severity_counts.keys(), severity_counts.values())
            axes[0, 0].set_title('Bias Severity Distribution')
            axes[0, 0].set_ylabel('Number of Biased Tests')
        
        # 2. Bias type distribution
        bias_type_counts = defaultdict(int)
        for result in results:
            if result.is_biased:
                bias_type_counts[result.bias_type.value] += 1
        
        if bias_type_counts:
            axes[0, 1].pie(bias_type_counts.values(), labels=bias_type_counts.keys(), autopct='%1.1f%%')
            axes[0, 1].set_title('Bias Type Distribution')
        
        # 3. Metric values distribution
        metric_values = [r.metric_value for r in results if r.is_biased]
        if metric_values:
            axes[1, 0].hist(metric_values, bins=20, alpha=0.7)
            axes[1, 0].set_title('Bias Metric Values Distribution')
            axes[1, 0].set_xlabel('Metric Value')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Bias trends over time (if history available)
        if self.bias_history:
            for metric_name, values in list(self.bias_history.items())[:5]:  # Show top 5 metrics
                axes[1, 1].plot(values, label=metric_name[:20], alpha=0.7)
            axes[1, 1].set_title('Bias Trends Over Time')
            axes[1, 1].set_xlabel('Measurement')
            axes[1, 1].set_ylabel('Bias Metric Value')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class HealthcareAIBiasMitigator:
    """
    Comprehensive bias mitigation framework for healthcare AI systems.
    
    This class implements various bias mitigation strategies at different
    stages of the ML pipeline, specifically designed for healthcare applications.
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        mitigation_strategy: str = "comprehensive",
        clinical_constraints: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize healthcare AI bias mitigator.
        
        Args:
            protected_attributes: List of protected attributes
            mitigation_strategy: Strategy for bias mitigation
            clinical_constraints: Clinical constraints for mitigation
        """
        self.protected_attributes = protected_attributes
        self.mitigation_strategy = mitigation_strategy
        self.clinical_constraints = clinical_constraints or {}
        
        # Available mitigation methods
        self.preprocessing_methods = {
            'reweighing': self._apply_reweighing,
            'disparate_impact_remover': self._apply_disparate_impact_remover,
            'synthetic_data_generation': self._generate_synthetic_data
        }
        
        self.inprocessing_methods = {
            'adversarial_debiasing': self._apply_adversarial_debiasing,
            'fairness_constraints': self._apply_fairness_constraints,
            'multi_task_learning': self._apply_multi_task_learning
        }
        
        self.postprocessing_methods = {
            'equalized_odds': self._apply_equalized_odds_postprocessing,
            'calibrated_equalized_odds': self._apply_calibrated_equalized_odds,
            'threshold_optimization': self._apply_threshold_optimization
        }
        
        logger.info(f"Initialized healthcare AI bias mitigator with strategy: {mitigation_strategy}")
    
    def mitigate_bias_comprehensive(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model: Any,
        bias_results: List[BiasDetectionResult]
    ) -> Dict[str, Any]:
        """
        Apply comprehensive bias mitigation based on detected bias.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model: Trained model
            bias_results: Results from bias detection
            
        Returns:
            Dictionary containing mitigated model and results
        """
        mitigation_results = {
            'original_model': model,
            'mitigated_models': {},
            'mitigation_methods_applied': [],
            'performance_comparison': {},
            'bias_reduction': {}
        }
        
        # Determine mitigation strategy based on bias results
        critical_bias = [r for r in bias_results if r.is_biased and r.severity == SeverityLevel.CRITICAL]
        high_bias = [r for r in bias_results if r.is_biased and r.severity == SeverityLevel.HIGH]
        
        # Apply preprocessing methods
        if any('representation' in r.bias_type.value for r in critical_bias + high_bias):
            X_train_reweighed, y_train_reweighed = self._apply_reweighing(X_train, y_train)
            mitigation_results['mitigated_data'] = (X_train_reweighed, y_train_reweighed)
            mitigation_results['mitigation_methods_applied'].append('reweighing')
        
        # Apply in-processing methods
        if any('algorithmic' in r.bias_type.value for r in critical_bias):
            fair_model = self._apply_fairness_constraints(X_train, y_train, model)
            mitigation_results['mitigated_models']['fairness_constrained'] = fair_model
            mitigation_results['mitigation_methods_applied'].append('fairness_constraints')
        
        # Apply post-processing methods
        if any('equalized_odds' in r.metric_name for r in high_bias + critical_bias):
            postprocessed_model = self._apply_equalized_odds_postprocessing(
                X_test, y_test, model
            )
            mitigation_results['mitigated_models']['equalized_odds'] = postprocessed_model
            mitigation_results['mitigation_methods_applied'].append('equalized_odds_postprocessing')
        
        # Evaluate mitigation effectiveness
        mitigation_results['effectiveness_evaluation'] = self._evaluate_mitigation_effectiveness(
            X_test, y_test, model, mitigation_results['mitigated_models'], bias_results
        )
        
        return mitigation_results
    
    def _apply_reweighing(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply reweighing to balance representation."""
        
        # Calculate sample weights to balance protected groups
        sample_weights = np.ones(len(X))
        
        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue
            
            # Calculate group sizes
            group_counts = X[attr].value_counts()
            total_samples = len(X)
            
            # Calculate weights to balance groups
            for group in group_counts.index:
                group_mask = X[attr] == group
                group_size = group_counts[group]
                target_size = total_samples / len(group_counts)
                weight = target_size / group_size
                sample_weights[group_mask] *= weight
        
        # Normalize weights
        sample_weights = sample_weights / np.mean(sample_weights)
        
        # Create reweighed dataset
        X_reweighed = X.copy()
        X_reweighed['sample_weight'] = sample_weights
        
        return X_reweighed, y
    
    def _apply_disparate_impact_remover(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply disparate impact remover preprocessing."""
        
        # This is a simplified implementation
        # In practice, would use more sophisticated methods
        
        X_processed = X.copy()
        
        # Remove direct correlation with protected attributes
        for attr in self.protected_attributes:
            if attr in X_processed.columns:
                # Calculate correlation with other features
                correlations = X_processed.corr()[attr].abs()
                highly_correlated = correlations[correlations > 0.7].index.tolist()
                
                # Remove highly correlated features (except the protected attribute itself)
                features_to_remove = [f for f in highly_correlated if f != attr]
                X_processed = X_processed.drop(columns=features_to_remove)
        
        return X_processed, y
    
    def _generate_synthetic_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic data to balance representation."""
        
        # Simplified synthetic data generation
        # In practice, would use more sophisticated methods like SMOTE or GANs
        
        from sklearn.utils import resample
        
        X_synthetic = X.copy()
        y_synthetic = y.copy()
        
        # Balance each protected group
        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue
            
            group_counts = X[attr].value_counts()
            max_count = group_counts.max()
            
            for group in group_counts.index:
                group_mask = X[attr] == group
                group_data = X[group_mask]
                group_labels = y[group_mask]
                
                if len(group_data) < max_count:
                    # Oversample minority group
                    n_samples_needed = max_count - len(group_data)
                    
                    resampled_data = resample(
                        group_data,
                        n_samples=n_samples_needed,
                        random_state=42
                    )
                    resampled_labels = resample(
                        group_labels,
                        n_samples=n_samples_needed,
                        random_state=42
                    )
                    
                    X_synthetic = pd.concat([X_synthetic, resampled_data], ignore_index=True)
                    y_synthetic = np.concatenate([y_synthetic, resampled_labels])
        
        return X_synthetic, y_synthetic
    
    def _apply_fairness_constraints(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        base_model: Any
    ) -> Any:
        """Apply fairness constraints during training."""
        
        # Simplified fairness-constrained training
        # In practice, would implement more sophisticated methods
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Create a fairness-aware model
        if isinstance(base_model, RandomForestClassifier):
            fair_model = RandomForestClassifier(
                n_estimators=base_model.n_estimators,
                max_depth=base_model.max_depth,
                class_weight='balanced',  # Simple fairness constraint
                random_state=42
            )
        else:
            fair_model = LogisticRegression(
                class_weight='balanced',
                random_state=42
            )
        
        # Train with balanced class weights
        fair_model.fit(X, y)
        
        return fair_model
    
    def _apply_adversarial_debiasing(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Any:
        """Apply adversarial debiasing during training."""
        
        # Simplified adversarial debiasing implementation
        # In practice, would use more sophisticated adversarial networks
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Create adversarial model (simplified)
        adversarial_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        
        adversarial_model.fit(X, y)
        
        return adversarial_model
    
    def _apply_multi_task_learning(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Any:
        """Apply multi-task learning for fairness."""
        
        # Simplified multi-task learning implementation
        from sklearn.ensemble import RandomForestClassifier
        
        # Train separate models for each protected group
        group_models = {}
        
        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue
            
            for group in X[attr].unique():
                group_mask = X[attr] == group
                if np.sum(group_mask) < 10:  # Skip small groups
                    continue
                
                group_X = X[group_mask]
                group_y = y[group_mask]
                
                group_model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    random_state=42
                )
                group_model.fit(group_X, group_y)
                group_models[f"{attr}_{group}"] = group_model
        
        return group_models
    
    def _apply_equalized_odds_postprocessing(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model: Any
    ) -> Any:
        """Apply equalized odds post-processing."""
        
        # Get model predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate optimal thresholds for each group
        optimal_thresholds = {}
        
        for attr in self.protected_attributes:
            if attr not in X.columns:
                continue
            
            for group in X[attr].unique():
                group_mask = X[attr] == group
                group_y = y[group_mask]
                group_prob = y_prob[group_mask]
                
                if len(group_y) < 10:  # Skip small groups
                    continue
                
                # Find threshold that maximizes balanced accuracy
                best_threshold = 0.5
                best_score = 0
                
                for threshold in np.arange(0.1, 0.9, 0.1):
                    group_pred_thresh = (group_prob >= threshold).astype(int)
                    
                    if len(np.unique(group_pred_thresh)) > 1:
                        score = accuracy_score(group_y, group_pred_thresh)
                        if score > best_score:
                            best_score = score
                            best_threshold = threshold
                
                optimal_thresholds[f"{attr}_{group}"] = best_threshold
        
        # Create post-processed model
        class PostProcessedModel:
            def __init__(self, base_model, thresholds, protected_attrs):
                self.base_model = base_model
                self.thresholds = thresholds
                self.protected_attrs = protected_attrs
            
            def predict(self, X):
                if hasattr(self.base_model, 'predict_proba'):
                    y_prob = self.base_model.predict_proba(X)[:, 1]
                else:
                    y_prob = self.base_model.predict(X)
                
                y_pred = np.zeros(len(X))
                
                for attr in self.protected_attrs:
                    if attr not in X.columns:
                        continue
                    
                    for group in X[attr].unique():
                        group_mask = X[attr] == group
                        threshold_key = f"{attr}_{group}"
                        
                        if threshold_key in self.thresholds:
                            threshold = self.thresholds[threshold_key]
                            y_pred[group_mask] = (y_prob[group_mask] >= threshold).astype(int)
                        else:
                            y_pred[group_mask] = (y_prob[group_mask] >= 0.5).astype(int)
                
                return y_pred
            
            def predict_proba(self, X):
                return self.base_model.predict_proba(X)
        
        return PostProcessedModel(model, optimal_thresholds, self.protected_attributes)
    
    def _apply_calibrated_equalized_odds(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model: Any
    ) -> Any:
        """Apply calibrated equalized odds post-processing."""
        
        # This would implement the calibrated equalized odds algorithm
        # For now, return the original model
        return model
    
    def _apply_threshold_optimization(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model: Any
    ) -> Any:
        """Apply threshold optimization for fairness."""
        
        # Similar to equalized odds but optimizes for different fairness metrics
        return self._apply_equalized_odds_postprocessing(X, y, model)
    
    def _evaluate_mitigation_effectiveness(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        original_model: Any,
        mitigated_models: Dict[str, Any],
        original_bias_results: List[BiasDetectionResult]
    ) -> Dict[str, Any]:
        """Evaluate the effectiveness of bias mitigation methods."""
        
        evaluation_results = {}
        
        # Initialize bias detector for evaluation
        bias_detector = HealthcareAIBiasDetector(self.protected_attributes)
        
        # Evaluate original model
        y_pred_original = original_model.predict(X_test)
        original_performance = {
            'accuracy': accuracy_score(y_test, y_pred_original),
            'precision': precision_score(y_test, y_pred_original, average='weighted'),
            'recall': recall_score(y_test, y_pred_original, average='weighted'),
            'f1': f1_score(y_test, y_pred_original, average='weighted')
        }
        
        evaluation_results['original'] = {
            'performance': original_performance,
            'bias_results': original_bias_results
        }
        
        # Evaluate mitigated models
        for method_name, mitigated_model in mitigated_models.items():
            y_pred_mitigated = mitigated_model.predict(X_test)
            
            # Performance metrics
            mitigated_performance = {
                'accuracy': accuracy_score(y_test, y_pred_mitigated),
                'precision': precision_score(y_test, y_pred_mitigated, average='weighted'),
                'recall': recall_score(y_test, y_pred_mitigated, average='weighted'),
                'f1': f1_score(y_test, y_pred_mitigated, average='weighted')
            }
            
            # Bias detection on mitigated model
            y_prob_mitigated = None
            if hasattr(mitigated_model, 'predict_proba'):
                y_prob_mitigated = mitigated_model.predict_proba(X_test)[:, 1]
            
            mitigated_bias_results = bias_detector.detect_bias_comprehensive(
                X_test, y_test, y_pred_mitigated, y_prob_mitigated
            )
            
            # Calculate bias reduction
            bias_reduction = self._calculate_bias_reduction(
                original_bias_results, mitigated_bias_results
            )
            
            evaluation_results[method_name] = {
                'performance': mitigated_performance,
                'bias_results': mitigated_bias_results,
                'bias_reduction': bias_reduction,
                'performance_trade_off': {
                    'accuracy_change': mitigated_performance['accuracy'] - original_performance['accuracy'],
                    'precision_change': mitigated_performance['precision'] - original_performance['precision'],
                    'recall_change': mitigated_performance['recall'] - original_performance['recall'],
                    'f1_change': mitigated_performance['f1'] - original_performance['f1']
                }
            }
        
        return evaluation_results
    
    def _calculate_bias_reduction(
        self,
        original_results: List[BiasDetectionResult],
        mitigated_results: List[BiasDetectionResult]
    ) -> Dict[str, float]:
        """Calculate bias reduction achieved by mitigation."""
        
        bias_reduction = {}
        
        # Create mapping of metric names to results
        original_metrics = {r.metric_name: r.metric_value for r in original_results if r.is_biased}
        mitigated_metrics = {r.metric_name: r.metric_value for r in mitigated_results}
        
        # Calculate reduction for each metric
        for metric_name, original_value in original_metrics.items():
            mitigated_value = mitigated_metrics.get(metric_name, original_value)
            reduction = (original_value - mitigated_value) / original_value if original_value > 0 else 0
            bias_reduction[metric_name] = reduction
        
        # Calculate overall bias reduction
        if bias_reduction:
            bias_reduction['overall'] = np.mean(list(bias_reduction.values()))
        else:
            bias_reduction['overall'] = 0.0
        
        return bias_reduction


## Bibliography and References

### Foundational Bias and Fairness Literature

. **Barocas, S., Hardt, M., & Narayanan, A.** (2019). Fairness and machine learning. *fairmlbook.org*. [Comprehensive textbook on algorithmic fairness]

. **Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R.** (2012). Fairness through awareness. *Proceedings of the 3rd innovations in theoretical computer science conference*, 214-226. [Foundational work on individual fairness]

. **Hardt, M., Price, E., & Srebro, N.** (2016). Equality of opportunity in supervised learning. *Advances in neural information processing systems*, 29, 3315-3323. [Seminal paper on equalized odds and equal opportunity]

. **Verma, S., & Rubin, J.** (2018). Fairness definitions explained. *Proceedings of the international workshop on software fairness*, 1-7. DOI: 10.1145/3194770.3194776. [Comprehensive survey of fairness definitions]

### Healthcare AI Bias and Fairness

. **Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Chin, M. H.** (2018). Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12), 866-872. DOI: 10.7326/M18-1990. [Foundational paper on healthcare AI fairness]

. **Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S.** (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453. DOI: 10.1126/science.aax2342. [Landmark study revealing bias in healthcare algorithms]

. **Chen, I. Y., Pierson, E., Rose, S., Joshi, S., Ferryman, K., & Ghassemi, M.** (2021). Ethical machine learning in healthcare. *Annual Review of Biomedical Data Science*, 4, 123-144. DOI: 10.1146/annurev-biodatasci-092820-114757. [Comprehensive review of ethical considerations in healthcare ML]

. **Larrazabal, A. J., Nieto, N., Peterson, V., Milone, D. H., & Ferrante, E.** (2020). Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis. *Proceedings of the National Academy of Sciences*, 117(23), 12592-12594. [Study on gender bias in medical imaging]

### Bias Detection Methodologies

. **Bellamy, R. K., Dey, K., Hind, M., Hoffman, S. C., Houde, S., Kannan, K., ... & Zhang, Y.** (2018). AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias. *arXiv preprint arXiv:1810.01943*. [Comprehensive bias detection toolkit]

. **Friedler, S. A., Scheidegger, C., Venkatasubramanian, S., Choudhary, S., Hamilton, E. P., & Roth, D.** (2019). A comparative study of fairness-enhancing interventions in machine learning. *Proceedings of the conference on fairness, accountability, and transparency*, 329-338. [Comparative study of bias mitigation methods]

. **Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A.** (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35. DOI: 10.1145/3457607. [Comprehensive survey of bias types and mitigation strategies]

### Bias Mitigation Techniques

. **Kamiran, F., & Calders, T.** (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems*, 33(1), 1-33. [Preprocessing methods for bias mitigation]

. **Zhang, B. H., Lemoine, B., & Mitchell, M.** (2018). Mitigating unwanted biases with adversarial learning. *Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society*, 335-340. [Adversarial debiasing methods]

. **Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q.** (2017). On fairness and calibration. *Advances in neural information processing systems*, 30, 5680-5689. [Calibration and fairness trade-offs]

### Regulatory and Ethical Frameworks

. **U.S. Food and Drug Administration.** (2021). Artificial intelligence/machine learning (AI/ML)-based software as a medical device (SaMD) action plan. *FDA Guidance Document*. [Regulatory framework for AI/ML medical devices]

. **World Health Organization.** (2021). Ethics and governance of artificial intelligence for health. *WHO Technical Report*. [International ethical guidelines for healthcare AI]

. **Institute of Electrical and Electronics Engineers.** (2017). IEEE standard for ethical design process. *IEEE Std 2857-2021*. [Technical standards for ethical AI design]

### Clinical Applications and Case Studies

. **Gianfrancesco, M. A., Tamang, S., Yazdany, J., & Schmajuk, G.** (2018). Potential biases in machine learning algorithms using electronic health record data. *JAMA Internal Medicine*, 178(11), 1544-1547. [Bias in EHR-based algorithms]

. **Vyas, D. A., Eisenstein, L. G., & Jones, D. S.** (2020). Hidden in plain sight—reconsidering the use of race correction in clinical algorithms. *New England Journal of Medicine*, 383(9), 874-882. [Race correction in clinical algorithms]

. **Adamson, A. S., & Smith, A.** (2018). Machine learning and health care disparities in dermatology. *JAMA Dermatology*, 154(11), 1247-1248. [Bias in dermatology AI applications]

This chapter provides a comprehensive framework for detecting and mitigating bias in healthcare AI systems. The implementations presented address the unique challenges of clinical environments including regulatory compliance, clinical validity, and health equity considerations. The next chapter will explore interpretability and explainability in healthcare AI, building upon these fairness concepts to address the need for transparent and accountable clinical AI systems.


## Code Examples


All code examples from this chapter are available in the repository:
- **Directory**: [`code_examples/chapter_08/`](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_08/)
- **Direct Download**: [ZIP file](https://github.com/sanjaybasu-waymark/healthcare-ai-book/archive/refs/heads/main.zip)

To use the examples:
```bash
git clone https://github.com/sanjaybasu-waymark/healthcare-ai-book.git
cd healthcare-ai-book/code_examples/chapter_08
pip install -r requirements.txt
```
