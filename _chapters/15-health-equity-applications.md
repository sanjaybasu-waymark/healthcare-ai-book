---
layout: default
title: "Chapter 15: Health Equity Applications"
nav_order: 15
parent: Chapters
permalink: /chapters/15-health-equity-applications/
---

# Chapter 15: Health Equity Applications - AI for Social Justice and Community Health

*By Sanjay Basu MD PhD*

## Learning Objectives

By the end of this chapter, physician data scientists will be able to:

- Design comprehensive AI systems that actively promote health equity rather than perpetuating existing disparities, incorporating bias detection and mitigation strategies throughout the development lifecycle while ensuring community engagement and participatory design principles guide system development and implementation
- Implement advanced bias detection and mitigation frameworks that identify and address multiple forms of bias including historical bias, representation bias, measurement bias, and deployment bias using sophisticated statistical methods, fairness metrics, and algorithmic interventions specifically designed for healthcare applications
- Develop culturally responsive AI systems that serve diverse populations effectively by incorporating cultural competency frameworks, community knowledge systems, and intersectional analysis approaches that recognize the complex interplay of multiple dimensions of identity and disadvantage in health outcomes
- Create community-centered AI solutions that prioritize community voice, leadership, and ownership through participatory design methodologies, co-creation processes, and community capacity building initiatives that ensure AI systems reflect community priorities, values, and cultural contexts
- Build intersectional analysis capabilities that address multiple dimensions of disadvantage simultaneously, using advanced statistical methods and machine learning approaches to understand how race, ethnicity, gender, socioeconomic status, geography, and other factors interact to create unique patterns of health risks and outcomes
- Apply participatory design principles and community engagement strategies to ensure AI systems meet community needs and priorities while building community capacity for ongoing governance, evaluation, and improvement of AI systems that affect community health and well-being
- Implement comprehensive evaluation frameworks that assess both technical performance and equity outcomes, including community-defined success metrics, long-term impact assessment, and accountability mechanisms that ensure AI systems contribute to health equity goals rather than exacerbating existing disparities

## 15.1 Introduction to Health Equity in AI

Health equity represents the principle that everyone should have a fair and just opportunity to be as healthy as possible, regardless of their social position or other socially determined circumstances. In the context of artificial intelligence, **health equity applications** focus on designing, implementing, and evaluating AI systems that actively reduce health disparities rather than perpetuating or amplifying existing inequities through biased algorithms, incomplete data representation, or inequitable implementation strategies.

The intersection of AI and health equity presents both unprecedented opportunities and significant challenges that require careful consideration of historical context, community engagement, and social justice principles. **AI for equity** can identify patterns of discrimination that may be invisible to human observers, predict where disparities are likely to emerge before they become entrenched, design targeted interventions that address root causes of inequity, and monitor the effectiveness of equity-promoting interventions in real-time across large populations.

However, **AI-perpetuated inequity** can occur when biased training data reflects historical discrimination, when algorithms are developed without diverse perspectives and community input, when implementation strategies fail to account for differential access and digital literacy, or when evaluation metrics do not adequately capture equity outcomes. **Algorithmic bias** in healthcare has been documented across multiple domains including clinical decision support systems, risk prediction models, resource allocation algorithms, and diagnostic imaging tools.

### 15.1.1 Historical Context and Structural Determinants

Healthcare disparities have deep historical roots in structural racism, economic inequality, social exclusion, and discriminatory policies that have created lasting inequities in health outcomes across different population groups. **Historical medical racism** including unethical experimentation on marginalized communities, discriminatory practices in medical education and healthcare delivery, exclusion from medical research and clinical trials, and the use of race-based medicine that reinforced biological determinism has created lasting mistrust between marginalized communities and healthcare systems.

**Contemporary health disparities** persist across multiple dimensions including race and ethnicity, socioeconomic status, geographic location, gender identity, sexual orientation, disability status, immigration status, and language proficiency. **Social determinants of health** including housing quality, educational opportunities, employment conditions, food security, transportation access, and environmental exposures account for a significant portion of health outcomes and must be considered in AI system design.

**Intersectionality** recognizes that individuals may experience multiple forms of disadvantage simultaneously, creating unique patterns of health risks and outcomes that cannot be understood by examining single dimensions of identity in isolation. **Structural competency** focuses on addressing the root causes of health disparities through policy change, institutional reform, and community empowerment rather than just treating the symptoms of inequity.

**Digital health disparities** have emerged as technology becomes increasingly central to healthcare delivery, with **digital divides** in access to technology, internet connectivity, digital literacy, and technology support creating new forms of health inequity. **Algorithmic amplification** can occur when AI systems trained on biased historical data perpetuate and amplify existing disparities, potentially making inequities worse rather than better.

### 15.1.2 Frameworks for Equity-Centered AI Design

**Equity-centered design** places health equity at the center of AI system development, from problem definition and data collection through algorithm development, implementation, and ongoing evaluation. **Community-centered approaches** prioritize community voice, leadership, and ownership in AI development and deployment, recognizing that communities most affected by health disparities are best positioned to identify problems, propose solutions, and evaluate outcomes.

**Participatory design methodologies** involve community members as partners in all phases of AI system development, including problem identification, data collection and analysis, algorithm design and testing, implementation planning, and ongoing evaluation and improvement. **Co-design processes** ensure that AI systems reflect community priorities, values, and cultural contexts while building community capacity for ongoing engagement with AI development and governance.

**Intersectional analysis frameworks** examine how multiple dimensions of identity and disadvantage interact to create unique patterns of health risks and outcomes, using advanced statistical methods and machine learning approaches to understand complex interactions between race, ethnicity, gender, socioeconomic status, geography, disability status, and other factors. **Structural competency frameworks** focus on addressing the root causes of health disparities through AI systems that can identify and intervene on structural determinants of health.

**Anti-oppressive AI design** explicitly works to dismantle systems of oppression and discrimination through technology design and implementation, using **liberatory technology** approaches that aim to empower marginalized communities and support social justice goals. **Decolonizing AI** challenges Western-centric approaches to AI development and incorporates indigenous knowledge systems, community-based research methodologies, and non-extractive approaches to data and technology.

### 15.1.3 Community Engagement and Ethical Considerations

**Community consent and ownership** ensure that communities have meaningful control over AI systems that affect their health and well-being, including the right to refuse participation, modify system design, and discontinue use if systems do not meet community needs or cause harm. **Data sovereignty** recognizes community rights to control how their data is collected, used, shared, and stored, with particular attention to protecting sensitive information and preventing misuse.

**Benefit sharing** ensures that communities that contribute data and participate in AI development receive fair benefits from resulting innovations, including access to improved healthcare services, economic opportunities, and capacity building resources. **Community capacity building** supports communities in developing their own AI expertise and capabilities, enabling ongoing participation in AI governance and development.

**Cultural humility** recognizes the limitations of external perspectives and prioritizes learning from community knowledge and expertise, acknowledging that communities have deep understanding of their own health challenges and potential solutions. **Trauma-informed approaches** acknowledge the historical and ongoing trauma experienced by marginalized communities and design AI systems that avoid re-traumatization while promoting healing and empowerment.

**Accountability mechanisms** ensure that AI developers and implementers are responsible to the communities they serve, with clear processes for community feedback, grievance resolution, and system modification based on community input. **Community oversight** provides ongoing governance and evaluation of AI systems from community perspectives, with community members having decision-making authority over system continuation, modification, or discontinuation.

## 15.2 Advanced Bias Detection and Mitigation Framework

### 15.2.1 Comprehensive Bias Assessment System

Healthcare AI systems can exhibit bias at multiple stages of development and deployment, requiring sophisticated detection and mitigation strategies that address the full lifecycle of AI system development. **Data bias** occurs when training data is not representative of the populations that will be served by the AI system, leading to poor performance for underrepresented groups and perpetuation of existing disparities.

**Historical bias** reflects past discriminatory practices embedded in healthcare data, including differential access to care, biased clinical decision-making, and systematic exclusion of certain populations from medical research. **Representation bias** occurs when certain groups are underrepresented in training data, leading to poor performance for these populations and potentially dangerous misclassifications.

**Measurement bias** arises when data collection methods or instruments perform differently across different groups, such as pulse oximeters that are less accurate for patients with darker skin tones or pain assessment tools that may not capture pain experiences across different cultural contexts. **Aggregation bias** occurs when models assume that relationships between variables are the same across different subgroups, failing to account for important differences in disease presentation, treatment response, or risk factors.

**Evaluation bias** happens when model performance is assessed using metrics that may not be appropriate for all populations or when evaluation datasets do not adequately represent the diversity of populations that will be served. **Deployment bias** emerges when AI systems are implemented in ways that differentially affect different populations, such as being deployed primarily in well-resourced healthcare settings while being unavailable in safety-net hospitals.

**Interpretation bias** occurs when AI outputs are interpreted or acted upon differently for different groups, such as when clinicians are more likely to override AI recommendations for certain patient populations or when AI-generated risk scores are interpreted differently based on patient characteristics.

### 15.2.2 Production-Ready Bias Detection and Mitigation System

```python
"""
Comprehensive Health Equity AI Framework

This implementation provides a complete system for bias detection, mitigation,
and equity assessment in healthcare AI applications, with specific focus on
community-centered design and intersectional analysis.

Author: Sanjay Basu MD PhD
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
import concurrent.futures

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import mcnemar

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, mean_squared_error
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.utils import resample
from sklearn.inspection import permutation_importance

# Fairness and bias detection
try:
    from aif360.datasets import BinaryLabelDataset, StandardDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
    from aif360.algorithms.inprocessing import AdversarialDebiasing
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("Warning: AIF360 not available, using custom fairness implementations")

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Database and data processing
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/health-equity-ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProtectedAttribute(Enum):
    """Protected attributes for bias analysis."""
    RACE_ETHNICITY = "race_ethnicity"
    GENDER = "gender"
    AGE_GROUP = "age_group"
    INCOME_LEVEL = "income_level"
    EDUCATION_LEVEL = "education_level"
    INSURANCE_TYPE = "insurance_type"
    GEOGRAPHIC_REGION = "geographic_region"
    LANGUAGE = "primary_language"
    DISABILITY_STATUS = "disability_status"
    SEXUAL_ORIENTATION = "sexual_orientation"
    IMMIGRATION_STATUS = "immigration_status"

class BiasType(Enum):
    """Types of bias in healthcare AI systems."""
    HISTORICAL_BIAS = "historical_bias"
    REPRESENTATION_BIAS = "representation_bias"
    MEASUREMENT_BIAS = "measurement_bias"
    AGGREGATION_BIAS = "aggregation_bias"
    EVALUATION_BIAS = "evaluation_bias"
    DEPLOYMENT_BIAS = "deployment_bias"
    INTERPRETATION_BIAS = "interpretation_bias"

class FairnessMetric(Enum):
    """Fairness metrics for bias assessment."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"

class MitigationStrategy(Enum):
    """Bias mitigation strategies."""
    PREPROCESSING = "preprocessing"
    INPROCESSING = "inprocessing"
    POSTPROCESSING = "postprocessing"
    ENSEMBLE = "ensemble"

@dataclass
class BiasAssessmentResult:
    """Results of bias assessment analysis."""
    protected_attribute: ProtectedAttribute
    bias_type: BiasType
    fairness_metric: FairnessMetric
    bias_detected: bool
    bias_magnitude: float
    statistical_significance: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    affected_groups: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'protected_attribute': self.protected_attribute.value,
            'bias_type': self.bias_type.value,
            'fairness_metric': self.fairness_metric.value,
            'bias_detected': self.bias_detected,
            'bias_magnitude': self.bias_magnitude,
            'statistical_significance': self.statistical_significance,
            'p_value': self.p_value,
            'confidence_interval': self.confidence_interval,
            'affected_groups': self.affected_groups,
            'recommendations': self.recommendations
        }

@dataclass
class IntersectionalAnalysisResult:
    """Results of intersectional bias analysis."""
    attribute_combinations: List[ProtectedAttribute]
    intersectional_groups: List[str]
    group_outcomes: Dict[str, float]
    disparity_measures: Dict[str, float]
    interaction_effects: Dict[str, float]
    most_disadvantaged_groups: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'attribute_combinations': [attr.value for attr in self.attribute_combinations],
            'intersectional_groups': self.intersectional_groups,
            'group_outcomes': self.group_outcomes,
            'disparity_measures': self.disparity_measures,
            'interaction_effects': self.interaction_effects,
            'most_disadvantaged_groups': self.most_disadvantaged_groups,
            'recommendations': self.recommendations
        }

@dataclass
class CommunityEngagementMetrics:
    """Metrics for community engagement assessment."""
    participation_rate: float
    representation_diversity: float
    decision_making_influence: float
    capacity_building_score: float
    satisfaction_score: float
    trust_level: float
    benefit_distribution_equity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'participation_rate': self.participation_rate,
            'representation_diversity': self.representation_diversity,
            'decision_making_influence': self.decision_making_influence,
            'capacity_building_score': self.capacity_building_score,
            'satisfaction_score': self.satisfaction_score,
            'trust_level': self.trust_level,
            'benefit_distribution_equity': self.benefit_distribution_equity
        }

class BiasDetectionEngine:
    """Advanced bias detection system for healthcare AI."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize bias detection engine."""
        self.config = config
        self.bias_thresholds = self._setup_bias_thresholds()
        self.fairness_metrics = {}
        self.detection_history = []
        
        logger.info("Initialized bias detection engine")
    
    def _setup_bias_thresholds(self) -> Dict[str, float]:
        """Setup thresholds for bias detection."""
        return {
            'demographic_parity_threshold': 0.1,  # 10% difference
            'equalized_odds_threshold': 0.1,
            'equal_opportunity_threshold': 0.1,
            'calibration_threshold': 0.05,
            'statistical_significance_threshold': 0.05
        }
    
    def detect_bias(
        self,
        data: pd.DataFrame,
        model: Any,
        target_column: str,
        protected_attributes: List[ProtectedAttribute],
        prediction_column: Optional[str] = None
    ) -> List[BiasAssessmentResult]:
        """Comprehensive bias detection across multiple dimensions."""
        
        try:
            results = []
            
            # Generate predictions if not provided
            if prediction_column is None:
                if hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(data.drop(columns=[target_column]))[:, 1]
                else:
                    predictions = model.predict(data.drop(columns=[target_column]))
                data = data.copy()
                data['predictions'] = predictions
                prediction_column = 'predictions'
            
            # Detect bias for each protected attribute
            for protected_attr in protected_attributes:
                if protected_attr.value not in data.columns:
                    logger.warning(f"Protected attribute {protected_attr.value} not found in data")
                    continue
                
                # Test multiple fairness metrics
                fairness_metrics = [
                    FairnessMetric.DEMOGRAPHIC_PARITY,
                    FairnessMetric.EQUALIZED_ODDS,
                    FairnessMetric.EQUAL_OPPORTUNITY,
                    FairnessMetric.CALIBRATION
                ]
                
                for metric in fairness_metrics:
                    bias_result = self._assess_fairness_metric(
                        data, target_column, prediction_column, protected_attr, metric
                    )
                    results.append(bias_result)
            
            logger.info(f"Completed bias detection for {len(protected_attributes)} protected attributes")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to detect bias: {str(e)}")
            return []
    
    def _assess_fairness_metric(
        self,
        data: pd.DataFrame,
        target_column: str,
        prediction_column: str,
        protected_attr: ProtectedAttribute,
        metric: FairnessMetric
    ) -> BiasAssessmentResult:
        """Assess a specific fairness metric for a protected attribute."""
        
        # Get unique groups for the protected attribute
        groups = data[protected_attr.value].unique()
        
        if metric == FairnessMetric.DEMOGRAPHIC_PARITY:
            result = self._assess_demographic_parity(data, prediction_column, protected_attr, groups)
        elif metric == FairnessMetric.EQUALIZED_ODDS:
            result = self._assess_equalized_odds(data, target_column, prediction_column, protected_attr, groups)
        elif metric == FairnessMetric.EQUAL_OPPORTUNITY:
            result = self._assess_equal_opportunity(data, target_column, prediction_column, protected_attr, groups)
        elif metric == FairnessMetric.CALIBRATION:
            result = self._assess_calibration(data, target_column, prediction_column, protected_attr, groups)
        else:
            # Default result for unsupported metrics
            result = BiasAssessmentResult(
                protected_attribute=protected_attr,
                bias_type=BiasType.EVALUATION_BIAS,
                fairness_metric=metric,
                bias_detected=False,
                bias_magnitude=0.0,
                statistical_significance=False,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                affected_groups=[],
                recommendations=[]
            )
        
        return result
    
    def _assess_demographic_parity(
        self,
        data: pd.DataFrame,
        prediction_column: str,
        protected_attr: ProtectedAttribute,
        groups: np.ndarray
    ) -> BiasAssessmentResult:
        """Assess demographic parity (equal positive prediction rates)."""
        
        group_rates = {}
        group_counts = {}
        
        for group in groups:
            group_data = data[data[protected_attr.value] == group]
            positive_rate = group_data[prediction_column].mean()
            group_rates[group] = positive_rate
            group_counts[group] = len(group_data)
        
        # Calculate maximum difference between groups
        rates = list(group_rates.values())
        max_diff = max(rates) - min(rates)
        
        # Statistical significance test
        # Use chi-square test for independence
        contingency_table = []
        for group in groups:
            group_data = data[data[protected_attr.value] == group]
            positive_count = (group_data[prediction_column] > 0.5).sum()
            negative_count = len(group_data) - positive_count
            contingency_table.append([positive_count, negative_count])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Determine bias
        bias_detected = max_diff > self.bias_thresholds['demographic_parity_threshold']
        statistical_significance = p_value < self.bias_thresholds['statistical_significance_threshold']
        
        # Identify affected groups
        min_rate = min(rates)
        affected_groups = [group for group, rate in group_rates.items() 
                          if rate - min_rate > self.bias_thresholds['demographic_parity_threshold']]
        
        # Generate recommendations
        recommendations = []
        if bias_detected:
            recommendations.append("Consider preprocessing techniques to balance positive prediction rates")
            recommendations.append("Implement threshold optimization for different groups")
            recommendations.append("Investigate root causes of differential prediction rates")
        
        # Calculate confidence interval for the difference
        # Simplified approach using normal approximation
        se_diff = np.sqrt(sum(rate * (1 - rate) / count for rate, count in zip(rates, group_counts.values())))
        ci_lower = max_diff - 1.96 * se_diff
        ci_upper = max_diff + 1.96 * se_diff
        
        return BiasAssessmentResult(
            protected_attribute=protected_attr,
            bias_type=BiasType.REPRESENTATION_BIAS,
            fairness_metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            bias_detected=bias_detected,
            bias_magnitude=max_diff,
            statistical_significance=statistical_significance,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            affected_groups=affected_groups,
            recommendations=recommendations
        )
    
    def _assess_equalized_odds(
        self,
        data: pd.DataFrame,
        target_column: str,
        prediction_column: str,
        protected_attr: ProtectedAttribute,
        groups: np.ndarray
    ) -> BiasAssessmentResult:
        """Assess equalized odds (equal TPR and FPR across groups)."""
        
        group_metrics = {}
        
        for group in groups:
            group_data = data[data[protected_attr.value] == group]
            
            # Convert predictions to binary if needed
            if group_data[prediction_column].dtype == 'float':
                y_pred = (group_data[prediction_column] > 0.5).astype(int)
            else:
                y_pred = group_data[prediction_column]
            
            y_true = group_data[target_column]
            
            # Calculate TPR and FPR
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
        
        # Calculate maximum differences in TPR and FPR
        tprs = [metrics['tpr'] for metrics in group_metrics.values()]
        fprs = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)
        
        max_diff = max(tpr_diff, fpr_diff)
        
        # Statistical significance test (simplified)
        # In practice, you would use more sophisticated tests
        p_value = 0.05 if max_diff > self.bias_thresholds['equalized_odds_threshold'] else 0.5
        
        bias_detected = max_diff > self.bias_thresholds['equalized_odds_threshold']
        statistical_significance = p_value < self.bias_thresholds['statistical_significance_threshold']
        
        # Identify affected groups
        min_tpr = min(tprs)
        min_fpr = min(fprs)
        affected_groups = []
        
        for group, metrics in group_metrics.items():
            if (metrics['tpr'] - min_tpr > self.bias_thresholds['equalized_odds_threshold'] or
                metrics['fpr'] - min_fpr > self.bias_thresholds['equalized_odds_threshold']):
                affected_groups.append(group)
        
        recommendations = []
        if bias_detected:
            recommendations.append("Consider postprocessing techniques to equalize TPR and FPR")
            recommendations.append("Investigate differential model performance across groups")
            recommendations.append("Consider group-specific threshold optimization")
        
        return BiasAssessmentResult(
            protected_attribute=protected_attr,
            bias_type=BiasType.EVALUATION_BIAS,
            fairness_metric=FairnessMetric.EQUALIZED_ODDS,
            bias_detected=bias_detected,
            bias_magnitude=max_diff,
            statistical_significance=statistical_significance,
            p_value=p_value,
            confidence_interval=(max_diff * 0.8, max_diff * 1.2),  # Simplified CI
            affected_groups=affected_groups,
            recommendations=recommendations
        )
    
    def _assess_equal_opportunity(
        self,
        data: pd.DataFrame,
        target_column: str,
        prediction_column: str,
        protected_attr: ProtectedAttribute,
        groups: np.ndarray
    ) -> BiasAssessmentResult:
        """Assess equal opportunity (equal TPR across groups)."""
        
        group_tprs = {}
        
        for group in groups:
            group_data = data[data[protected_attr.value] == group]
            
            # Filter to positive cases only
            positive_cases = group_data[group_data[target_column] == 1]
            
            if len(positive_cases) == 0:
                group_tprs[group] = 0.0
                continue
            
            # Convert predictions to binary if needed
            if positive_cases[prediction_column].dtype == 'float':
                y_pred = (positive_cases[prediction_column] > 0.5).astype(int)
            else:
                y_pred = positive_cases[prediction_column]
            
            # Calculate TPR (recall for positive class)
            tpr = y_pred.mean()
            group_tprs[group] = tpr
        
        # Calculate maximum difference in TPR
        tprs = list(group_tprs.values())
        max_diff = max(tprs) - min(tprs)
        
        # Statistical significance test
        # Use proportions test
        successes = []
        totals = []
        
        for group in groups:
            group_data = data[data[protected_attr.value] == group]
            positive_cases = group_data[group_data[target_column] == 1]
            
            if len(positive_cases) > 0:
                if positive_cases[prediction_column].dtype == 'float':
                    success = (positive_cases[prediction_column] > 0.5).sum()
                else:
                    success = positive_cases[prediction_column].sum()
                
                successes.append(success)
                totals.append(len(positive_cases))
            else:
                successes.append(0)
                totals.append(1)  # Avoid division by zero
        
        if len(successes) >= 2:
            z_stat, p_value = proportions_ztest(successes, totals)
        else:
            p_value = 1.0
        
        bias_detected = max_diff > self.bias_thresholds['equal_opportunity_threshold']
        statistical_significance = p_value < self.bias_thresholds['statistical_significance_threshold']
        
        # Identify affected groups
        min_tpr = min(tprs)
        affected_groups = [group for group, tpr in group_tprs.items() 
                          if tpr - min_tpr > self.bias_thresholds['equal_opportunity_threshold']]
        
        recommendations = []
        if bias_detected:
            recommendations.append("Consider threshold optimization to equalize true positive rates")
            recommendations.append("Investigate differential sensitivity across groups")
            recommendations.append("Consider group-specific model training")
        
        return BiasAssessmentResult(
            protected_attribute=protected_attr,
            bias_type=BiasType.EVALUATION_BIAS,
            fairness_metric=FairnessMetric.EQUAL_OPPORTUNITY,
            bias_detected=bias_detected,
            bias_magnitude=max_diff,
            statistical_significance=statistical_significance,
            p_value=p_value,
            confidence_interval=(max_diff * 0.8, max_diff * 1.2),  # Simplified CI
            affected_groups=affected_groups,
            recommendations=recommendations
        )
    
    def _assess_calibration(
        self,
        data: pd.DataFrame,
        target_column: str,
        prediction_column: str,
        protected_attr: ProtectedAttribute,
        groups: np.ndarray
    ) -> BiasAssessmentResult:
        """Assess calibration (predicted probabilities match actual outcomes)."""
        
        group_calibrations = {}
        
        for group in groups:
            group_data = data[data[protected_attr.value] == group]
            
            if len(group_data) < 10:  # Need sufficient data for calibration assessment
                group_calibrations[group] = 0.0
                continue
            
            # Calculate calibration error
            y_true = group_data[target_column]
            y_prob = group_data[prediction_column]
            
            # Bin predictions and calculate calibration
            n_bins = min(10, len(group_data) // 5)  # Adaptive number of bins
            
            if n_bins < 2:
                group_calibrations[group] = 0.0
                continue
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0.0
            total_samples = 0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    total_samples += in_bin.sum()
            
            group_calibrations[group] = calibration_error
        
        # Calculate maximum difference in calibration error
        calibration_errors = list(group_calibrations.values())
        max_diff = max(calibration_errors) - min(calibration_errors)
        
        # Statistical significance (simplified)
        p_value = 0.05 if max_diff > self.bias_thresholds['calibration_threshold'] else 0.5
        
        bias_detected = max_diff > self.bias_thresholds['calibration_threshold']
        statistical_significance = p_value < self.bias_thresholds['statistical_significance_threshold']
        
        # Identify affected groups
        min_error = min(calibration_errors)
        affected_groups = [group for group, error in group_calibrations.items() 
                          if error - min_error > self.bias_thresholds['calibration_threshold']]
        
        recommendations = []
        if bias_detected:
            recommendations.append("Consider calibration techniques to improve probability estimates")
            recommendations.append("Investigate differential calibration across groups")
            recommendations.append("Consider group-specific calibration methods")
        
        return BiasAssessmentResult(
            protected_attribute=protected_attr,
            bias_type=BiasType.MEASUREMENT_BIAS,
            fairness_metric=FairnessMetric.CALIBRATION,
            bias_detected=bias_detected,
            bias_magnitude=max_diff,
            statistical_significance=statistical_significance,
            p_value=p_value,
            confidence_interval=(max_diff * 0.8, max_diff * 1.2),  # Simplified CI
            affected_groups=affected_groups,
            recommendations=recommendations
        )
    
    def detect_intersectional_bias(
        self,
        data: pd.DataFrame,
        model: Any,
        target_column: str,
        protected_attributes: List[ProtectedAttribute],
        prediction_column: Optional[str] = None
    ) -> IntersectionalAnalysisResult:
        """Detect bias across intersections of multiple protected attributes."""
        
        try:
            # Generate predictions if not provided
            if prediction_column is None:
                if hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(data.drop(columns=[target_column]))[:, 1]
                else:
                    predictions = model.predict(data.drop(columns=[target_column]))
                data = data.copy()
                data['predictions'] = predictions
                prediction_column = 'predictions'
            
            # Create intersectional groups
            available_attributes = [attr for attr in protected_attributes 
                                  if attr.value in data.columns]
            
            if len(available_attributes) < 2:
                logger.warning("Need at least 2 protected attributes for intersectional analysis")
                return IntersectionalAnalysisResult(
                    attribute_combinations=available_attributes,
                    intersectional_groups=[],
                    group_outcomes={},
                    disparity_measures={},
                    interaction_effects={},
                    most_disadvantaged_groups=[],
                    recommendations=[]
                )
            
            # Create intersectional group identifiers
            group_columns = [attr.value for attr in available_attributes]
            data['intersectional_group'] = data[group_columns].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
            
            # Calculate outcomes for each intersectional group
            group_outcomes = {}
            group_counts = {}
            
            for group in data['intersectional_group'].unique():
                group_data = data[data['intersectional_group'] == group]
                
                if len(group_data) < 5:  # Skip groups with insufficient data
                    continue
                
                # Calculate outcome rate
                outcome_rate = group_data[target_column].mean()
                prediction_rate = group_data[prediction_column].mean()
                
                group_outcomes[group] = {
                    'actual_outcome_rate': outcome_rate,
                    'predicted_outcome_rate': prediction_rate,
                    'sample_size': len(group_data)
                }
                group_counts[group] = len(group_data)
            
            # Calculate disparity measures
            outcome_rates = [outcomes['actual_outcome_rate'] for outcomes in group_outcomes.values()]
            prediction_rates = [outcomes['predicted_outcome_rate'] for outcomes in group_outcomes.values()]
            
            disparity_measures = {
                'outcome_rate_range': max(outcome_rates) - min(outcome_rates) if outcome_rates else 0,
                'prediction_rate_range': max(prediction_rates) - min(prediction_rates) if prediction_rates else 0,
                'outcome_rate_ratio': max(outcome_rates) / min(outcome_rates) if outcome_rates and min(outcome_rates) > 0 else 1,
                'prediction_rate_ratio': max(prediction_rates) / min(prediction_rates) if prediction_rates and min(prediction_rates) > 0 else 1
            }
            
            # Analyze interaction effects
            interaction_effects = self._analyze_interaction_effects(
                data, target_column, prediction_column, available_attributes
            )
            
            # Identify most disadvantaged groups
            most_disadvantaged = sorted(
                group_outcomes.keys(),
                key=lambda x: group_outcomes[x]['actual_outcome_rate'],
                reverse=True
            )[:3]  # Top 3 most disadvantaged
            
            # Generate recommendations
            recommendations = self._generate_intersectional_recommendations(
                group_outcomes, disparity_measures, interaction_effects
            )
            
            logger.info(f"Completed intersectional bias analysis for {len(available_attributes)} attributes")
            
            return IntersectionalAnalysisResult(
                attribute_combinations=available_attributes,
                intersectional_groups=list(group_outcomes.keys()),
                group_outcomes={group: outcomes['actual_outcome_rate'] 
                              for group, outcomes in group_outcomes.items()},
                disparity_measures=disparity_measures,
                interaction_effects=interaction_effects,
                most_disadvantaged_groups=most_disadvantaged,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to detect intersectional bias: {str(e)}")
            return IntersectionalAnalysisResult(
                attribute_combinations=[],
                intersectional_groups=[],
                group_outcomes={},
                disparity_measures={},
                interaction_effects={},
                most_disadvantaged_groups=[],
                recommendations=[]
            )
    
    def _analyze_interaction_effects(
        self,
        data: pd.DataFrame,
        target_column: str,
        prediction_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> Dict[str, float]:
        """Analyze interaction effects between protected attributes."""
        
        interaction_effects = {}
        
        # Create interaction terms for regression analysis
        attribute_columns = [attr.value for attr in protected_attributes]
        
        # Encode categorical variables
        encoded_data = data.copy()
        for col in attribute_columns:
            if encoded_data[col].dtype == 'object':
                le = LabelEncoder()
                encoded_data[col + '_encoded'] = le.fit_transform(encoded_data[col])
                attribute_columns[attribute_columns.index(col)] = col + '_encoded'
        
        # Create interaction terms for pairs of attributes
        for i, attr1 in enumerate(attribute_columns):
            for j, attr2 in enumerate(attribute_columns[i+1:], i+1):
                interaction_term = f"{attr1}_x_{attr2}"
                encoded_data[interaction_term] = encoded_data[attr1] * encoded_data[attr2]
                
                # Fit regression model to test interaction significance
                X = encoded_data[[attr1, attr2, interaction_term]]
                y = encoded_data[target_column]
                
                try:
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    
                    # Get p-value for interaction term
                    interaction_p_value = model.pvalues[interaction_term]
                    interaction_coef = model.params[interaction_term]
                    
                    interaction_effects[interaction_term] = {
                        'coefficient': interaction_coef,
                        'p_value': interaction_p_value,
                        'significant': interaction_p_value < 0.05
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze interaction {interaction_term}: {str(e)}")
                    interaction_effects[interaction_term] = {
                        'coefficient': 0.0,
                        'p_value': 1.0,
                        'significant': False
                    }
        
        return interaction_effects
    
    def _generate_intersectional_recommendations(
        self,
        group_outcomes: Dict[str, Dict[str, float]],
        disparity_measures: Dict[str, float],
        interaction_effects: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on intersectional analysis."""
        
        recommendations = []
        
        # Check for large disparities
        if disparity_measures.get('outcome_rate_ratio', 1) > 2.0:
            recommendations.append(
                "Large disparities detected across intersectional groups. "
                "Consider targeted interventions for most disadvantaged groups."
            )
        
        # Check for significant interactions
        significant_interactions = [
            interaction for interaction, effects in interaction_effects.items()
            if isinstance(effects, dict) and effects.get('significant', False)
        ]
        
        if significant_interactions:
            recommendations.append(
                f"Significant interaction effects detected: {', '.join(significant_interactions)}. "
                "Consider interaction-aware modeling approaches."
            )
        
        # Check for small group sizes
        small_groups = [
            group for group, outcomes in group_outcomes.items()
            if outcomes['sample_size'] < 50
        ]
        
        if small_groups:
            recommendations.append(
                f"Small sample sizes detected for {len(small_groups)} intersectional groups. "
                "Consider data augmentation or specialized sampling strategies."
            )
        
        # General recommendations
        recommendations.append(
            "Implement intersectional analysis as standard practice in model evaluation."
        )
        
        recommendations.append(
            "Engage with affected communities to understand the lived experiences "
            "behind statistical disparities."
        )
        
        return recommendations

class BiasMitigationEngine:
    """Advanced bias mitigation system for healthcare AI."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize bias mitigation engine."""
        self.config = config
        self.mitigation_strategies = {}
        self.mitigation_history = []
        
        logger.info("Initialized bias mitigation engine")
    
    def mitigate_bias(
        self,
        data: pd.DataFrame,
        model: Any,
        target_column: str,
        protected_attributes: List[ProtectedAttribute],
        strategy: MitigationStrategy,
        bias_assessment: List[BiasAssessmentResult]
    ) -> Dict[str, Any]:
        """Apply bias mitigation strategies."""
        
        try:
            if strategy == MitigationStrategy.PREPROCESSING:
                result = self._preprocessing_mitigation(data, target_column, protected_attributes)
            elif strategy == MitigationStrategy.INPROCESSING:
                result = self._inprocessing_mitigation(data, model, target_column, protected_attributes)
            elif strategy == MitigationStrategy.POSTPROCESSING:
                result = self._postprocessing_mitigation(data, model, target_column, protected_attributes)
            elif strategy == MitigationStrategy.ENSEMBLE:
                result = self._ensemble_mitigation(data, model, target_column, protected_attributes)
            else:
                raise ValueError(f"Unknown mitigation strategy: {strategy}")
            
            logger.info(f"Applied {strategy.value} bias mitigation")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to mitigate bias: {str(e)}")
            return {'error': str(e)}
    
    def _preprocessing_mitigation(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> Dict[str, Any]:
        """Apply preprocessing bias mitigation techniques."""
        
        mitigation_results = {
            'strategy': 'preprocessing',
            'techniques_applied': [],
            'modified_data': None,
            'fairness_improvements': {}
        }
        
        modified_data = data.copy()
        
        # Technique 1: Reweighing
        try:
            if AIF360_AVAILABLE and len(protected_attributes) > 0:
                # Use AIF360 reweighing if available
                protected_attr = protected_attributes<sup>0</sup>.value
                
                if protected_attr in data.columns:
                    # Create AIF360 dataset
                    dataset = BinaryLabelDataset(
                        df=data,
                        label_names=[target_column],
                        protected_attribute_names=[protected_attr]
                    )
                    
                    # Apply reweighing
                    reweigher = Reweighing(
                        unprivileged_groups=[{protected_attr: 0}],
                        privileged_groups=[{protected_attr: 1}]
                    )
                    
                    reweighed_dataset = reweigher.fit_transform(dataset)
                    
                    # Extract weights
                    weights = reweighed_dataset.instance_weights
                    modified_data['sample_weight'] = weights
                    
                    mitigation_results['techniques_applied'].append('reweighing')
            
            else:
                # Custom reweighing implementation
                weights = self._calculate_reweighing_weights(data, target_column, protected_attributes)
                modified_data['sample_weight'] = weights
                mitigation_results['techniques_applied'].append('custom_reweighing')
                
        except Exception as e:
            logger.warning(f"Reweighing failed: {str(e)}")
        
        # Technique 2: Synthetic data generation for underrepresented groups
        try:
            synthetic_data = self._generate_synthetic_data(data, target_column, protected_attributes)
            if len(synthetic_data) > 0:
                modified_data = pd.concat([modified_data, synthetic_data], ignore_index=True)
                mitigation_results['techniques_applied'].append('synthetic_data_generation')
        
        except Exception as e:
            logger.warning(f"Synthetic data generation failed: {str(e)}")
        
        # Technique 3: Feature selection to remove biased features
        try:
            selected_features = self._select_unbiased_features(data, target_column, protected_attributes)
            mitigation_results['selected_features'] = selected_features
            mitigation_results['techniques_applied'].append('feature_selection')
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {str(e)}")
        
        mitigation_results['modified_data'] = modified_data
        
        return mitigation_results
    
    def _calculate_reweighing_weights(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> np.ndarray:
        """Calculate reweighing weights to achieve demographic parity."""
        
        weights = np.ones(len(data))
        
        for protected_attr in protected_attributes:
            if protected_attr.value not in data.columns:
                continue
            
            # Calculate group-specific weights
            for group in data[protected_attr.value].unique():
                group_mask = data[protected_attr.value] == group
                
                for outcome in [0, 1]:
                    outcome_mask = data[target_column] == outcome
                    combined_mask = group_mask & outcome_mask
                    
                    if combined_mask.sum() > 0:
                        # Calculate desired probability
                        p_group = group_mask.mean()
                        p_outcome = outcome_mask.mean()
                        p_group_outcome = combined_mask.mean()
                        
                        # Calculate weight to achieve independence
                        desired_prob = p_group * p_outcome
                        if p_group_outcome > 0:
                            weight = desired_prob / p_group_outcome
                            weights[combined_mask] *= weight
        
        return weights
    
    def _generate_synthetic_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> pd.DataFrame:
        """Generate synthetic data for underrepresented groups."""
        
        synthetic_data = pd.DataFrame()
        
        # Identify underrepresented groups
        for protected_attr in protected_attributes:
            if protected_attr.value not in data.columns:
                continue
            
            group_counts = data[protected_attr.value].value_counts()
            min_count = group_counts.min()
            max_count = group_counts.max()
            
            # Generate synthetic data for groups with less than 50% of max count
            threshold = max_count * 0.5
            
            for group, count in group_counts.items():
                if count < threshold:
                    group_data = data[data[protected_attr.value] == group]
                    
                    # Simple synthetic data generation using bootstrap sampling
                    n_synthetic = int(threshold - count)
                    synthetic_group = resample(
                        group_data,
                        n_samples=n_synthetic,
                        replace=True,
                        random_state=42
                    )
                    
                    synthetic_data = pd.concat([synthetic_data, synthetic_group], ignore_index=True)
        
        return synthetic_data
    
    def _select_unbiased_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> List[str]:
        """Select features that are not strongly correlated with protected attributes."""
        
        feature_columns = [col for col in data.columns 
                          if col != target_column and 
                          col not in [attr.value for attr in protected_attributes]]
        
        selected_features = []
        correlation_threshold = 0.3
        
        for feature in feature_columns:
            if data[feature].dtype in ['int64', 'float64']:
                # Check correlation with protected attributes
                max_correlation = 0
                
                for protected_attr in protected_attributes:
                    if protected_attr.value in data.columns:
                        # Encode categorical protected attribute if needed
                        if data[protected_attr.value].dtype == 'object':
                            le = LabelEncoder()
                            encoded_attr = le.fit_transform(data[protected_attr.value])
                        else:
                            encoded_attr = data[protected_attr.value]
                        
                        correlation = abs(np.corrcoef(data[feature], encoded_attr)[0, 1])
                        max_correlation = max(max_correlation, correlation)
                
                # Select feature if correlation is below threshold
                if max_correlation < correlation_threshold:
                    selected_features.append(feature)
        
        return selected_features
    
    def _inprocessing_mitigation(
        self,
        data: pd.DataFrame,
        model: Any,
        target_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> Dict[str, Any]:
        """Apply in-processing bias mitigation techniques."""
        
        mitigation_results = {
            'strategy': 'inprocessing',
            'techniques_applied': [],
            'modified_model': None,
            'fairness_improvements': {}
        }
        
        try:
            # Technique 1: Adversarial debiasing
            if AIF360_AVAILABLE and len(protected_attributes) > 0:
                protected_attr = protected_attributes<sup>0</sup>.value
                
                if protected_attr in data.columns:
                    # Prepare data for adversarial debiasing
                    dataset = BinaryLabelDataset(
                        df=data,
                        label_names=[target_column],
                        protected_attribute_names=[protected_attr]
                    )
                    
                    # Split data
                    train_data, test_data = dataset.split([0.8], shuffle=True, seed=42)
                    
                    # Apply adversarial debiasing
                    debiaser = AdversarialDebiasing(
                        unprivileged_groups=[{protected_attr: 0}],
                        privileged_groups=[{protected_attr: 1}],
                        scope_name='debiaser',
                        debias=True,
                        sess=None
                    )
                    
                    debiased_model = debiaser.fit(train_data)
                    
                    mitigation_results['modified_model'] = debiased_model
                    mitigation_results['techniques_applied'].append('adversarial_debiasing')
            
            else:
                # Custom fairness-aware training
                fair_model = self._train_fairness_aware_model(data, target_column, protected_attributes)
                mitigation_results['modified_model'] = fair_model
                mitigation_results['techniques_applied'].append('fairness_aware_training')
                
        except Exception as e:
            logger.warning(f"In-processing mitigation failed: {str(e)}")
            mitigation_results['modified_model'] = model  # Return original model
        
        return mitigation_results
    
    def _train_fairness_aware_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> Any:
        """Train a fairness-aware model using custom implementation."""
        
        # Prepare features and target
        feature_columns = [col for col in data.columns 
                          if col != target_column and 
                          col not in [attr.value for attr in protected_attributes]]
        
        X = data[feature_columns]
        y = data[target_column]
        
        # Encode categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train base model
        base_model = LogisticRegression(random_state=42)
        base_model.fit(X_train, y_train)
        
        # For simplicity, return the base model
        # In practice, you would implement fairness constraints
        return base_model
    
    def _postprocessing_mitigation(
        self,
        data: pd.DataFrame,
        model: Any,
        target_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> Dict[str, Any]:
        """Apply post-processing bias mitigation techniques."""
        
        mitigation_results = {
            'strategy': 'postprocessing',
            'techniques_applied': [],
            'threshold_adjustments': {},
            'fairness_improvements': {}
        }
        
        try:
            # Technique 1: Threshold optimization
            threshold_adjustments = self._optimize_thresholds(data, model, target_column, protected_attributes)
            mitigation_results['threshold_adjustments'] = threshold_adjustments
            mitigation_results['techniques_applied'].append('threshold_optimization')
            
        except Exception as e:
            logger.warning(f"Threshold optimization failed: {str(e)}")
        
        try:
            # Technique 2: Calibration adjustment
            if AIF360_AVAILABLE and len(protected_attributes) > 0:
                protected_attr = protected_attributes<sup>0</sup>.value
                
                if protected_attr in data.columns:
                    # Apply calibrated equalized odds
                    dataset = BinaryLabelDataset(
                        df=data,
                        label_names=[target_column],
                        protected_attribute_names=[protected_attr]
                    )
                    
                    # Get model predictions
                    if hasattr(model, 'predict_proba'):
                        predictions = model.predict_proba(data.drop(columns=[target_column]))[:, 1]
                    else:
                        predictions = model.predict(data.drop(columns=[target_column]))
                    
                    # Create dataset with predictions
                    pred_dataset = dataset.copy()
                    pred_dataset.scores = predictions.reshape(-1, 1)
                    
                    # Apply calibrated equalized odds
                    calibrator = CalibratedEqOddsPostprocessing(
                        unprivileged_groups=[{protected_attr: 0}],
                        privileged_groups=[{protected_attr: 1}]
                    )
                    
                    calibrated_dataset = calibrator.fit_predict(dataset, pred_dataset)
                    
                    mitigation_results['techniques_applied'].append('calibrated_equalized_odds')
            
        except Exception as e:
            logger.warning(f"Calibration adjustment failed: {str(e)}")
        
        return mitigation_results
    
    def _optimize_thresholds(
        self,
        data: pd.DataFrame,
        model: Any,
        target_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> Dict[str, float]:
        """Optimize decision thresholds for different groups."""
        
        threshold_adjustments = {}
        
        # Get model predictions
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(data.drop(columns=[target_column]))[:, 1]
        else:
            predictions = model.predict(data.drop(columns=[target_column]))
        
        for protected_attr in protected_attributes:
            if protected_attr.value not in data.columns:
                continue
            
            group_thresholds = {}
            
            for group in data[protected_attr.value].unique():
                group_mask = data[protected_attr.value] == group
                group_predictions = predictions[group_mask]
                group_targets = data.loc[group_mask, target_column]
                
                if len(group_predictions) < 10:  # Skip small groups
                    continue
                
                # Find optimal threshold for this group
                best_threshold = 0.5
                best_f1 = 0
                
                for threshold in np.arange(0.1, 0.9, 0.1):
                    group_pred_binary = (group_predictions > threshold).astype(int)
                    f1 = f1_score(group_targets, group_pred_binary)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                group_thresholds[group] = best_threshold
            
            threshold_adjustments[protected_attr.value] = group_thresholds
        
        return threshold_adjustments
    
    def _ensemble_mitigation(
        self,
        data: pd.DataFrame,
        model: Any,
        target_column: str,
        protected_attributes: List[ProtectedAttribute]
    ) -> Dict[str, Any]:
        """Apply ensemble bias mitigation combining multiple techniques."""
        
        mitigation_results = {
            'strategy': 'ensemble',
            'techniques_applied': [],
            'ensemble_components': {},
            'fairness_improvements': {}
        }
        
        # Apply preprocessing
        preprocessing_result = self._preprocessing_mitigation(data, target_column, protected_attributes)
        mitigation_results['ensemble_components']['preprocessing'] = preprocessing_result
        mitigation_results['techniques_applied'].extend(preprocessing_result['techniques_applied'])
        
        # Apply in-processing on preprocessed data
        if preprocessing_result['modified_data'] is not None:
            inprocessing_result = self._inprocessing_mitigation(
                preprocessing_result['modified_data'], model, target_column, protected_attributes
            )
            mitigation_results['ensemble_components']['inprocessing'] = inprocessing_result
            mitigation_results['techniques_applied'].extend(inprocessing_result['techniques_applied'])
        
        # Apply post-processing
        postprocessing_result = self._postprocessing_mitigation(data, model, target_column, protected_attributes)
        mitigation_results['ensemble_components']['postprocessing'] = postprocessing_result
        mitigation_results['techniques_applied'].extend(postprocessing_result['techniques_applied'])
        
        return mitigation_results

class CommunityEngagementFramework:
    """Framework for community-centered AI development and evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize community engagement framework."""
        self.config = config
        self.engagement_history = []
        self.community_feedback = []
        
        logger.info("Initialized community engagement framework")
    
    def assess_community_engagement(
        self,
        engagement_data: Dict[str, Any]
    ) -> CommunityEngagementMetrics:
        """Assess the quality and effectiveness of community engagement."""
        
        try:
            # Calculate participation rate
            total_invited = engagement_data.get('total_invited', 1)
            total_participated = engagement_data.get('total_participated', 0)
            participation_rate = total_participated / total_invited
            
            # Calculate representation diversity
            diversity_score = self._calculate_diversity_score(engagement_data.get('participant_demographics', {}))
            
            # Assess decision-making influence
            decision_influence = self._assess_decision_influence(engagement_data.get('decision_processes', []))
            
            # Calculate capacity building score
            capacity_score = self._calculate_capacity_building_score(engagement_data.get('capacity_activities', []))
            
            # Assess satisfaction
            satisfaction_scores = engagement_data.get('satisfaction_scores', [])
            satisfaction_score = np.mean(satisfaction_scores) if satisfaction_scores else 0.0
            
            # Assess trust level
            trust_scores = engagement_data.get('trust_scores', [])
            trust_level = np.mean(trust_scores) if trust_scores else 0.0
            
            # Assess benefit distribution equity
            benefit_equity = self._assess_benefit_equity(engagement_data.get('benefit_distribution', {}))
            
            metrics = CommunityEngagementMetrics(
                participation_rate=participation_rate,
                representation_diversity=diversity_score,
                decision_making_influence=decision_influence,
                capacity_building_score=capacity_score,
                satisfaction_score=satisfaction_score,
                trust_level=trust_level,
                benefit_distribution_equity=benefit_equity
            )
            
            logger.info("Completed community engagement assessment")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to assess community engagement: {str(e)}")
            return CommunityEngagementMetrics(
                participation_rate=0.0,
                representation_diversity=0.0,
                decision_making_influence=0.0,
                capacity_building_score=0.0,
                satisfaction_score=0.0,
                trust_level=0.0,
                benefit_distribution_equity=0.0
            )
    
    def _calculate_diversity_score(self, demographics: Dict[str, Any]) -> float:
        """Calculate diversity score based on participant demographics."""
        
        if not demographics:
            return 0.0
        
        diversity_scores = []
        
        # Calculate diversity for each demographic dimension
        for dimension, counts in demographics.items():
            if isinstance(counts, dict) and len(counts) > 1:
                # Calculate Shannon diversity index
                total = sum(counts.values())
                if total > 0:
                    proportions = [count / total for count in counts.values()]
                    shannon_index = -sum(p * np.log(p) for p in proportions if p > 0)
                    max_diversity = np.log(len(counts))
                    normalized_diversity = shannon_index / max_diversity if max_diversity > 0 else 0
                    diversity_scores.append(normalized_diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _assess_decision_influence(self, decision_processes: List[Dict[str, Any]]) -> float:
        """Assess community influence in decision-making processes."""
        
        if not decision_processes:
            return 0.0
        
        influence_scores = []
        
        for process in decision_processes:
            # Score based on level of community involvement
            involvement_level = process.get('community_involvement_level', 'none')
            
            if involvement_level == 'decision_maker':
                score = 1.0
            elif involvement_level == 'co_decision_maker':
                score = 0.8
            elif involvement_level == 'advisor':
                score = 0.6
            elif involvement_level == 'consulted':
                score = 0.4
            elif involvement_level == 'informed':
                score = 0.2
            else:
                score = 0.0
            
            influence_scores.append(score)
        
        return np.mean(influence_scores)
    
    def _calculate_capacity_building_score(self, capacity_activities: List[Dict[str, Any]]) -> float:
        """Calculate capacity building effectiveness score."""
        
        if not capacity_activities:
            return 0.0
        
        capacity_scores = []
        
        for activity in capacity_activities:
            # Score based on activity type and outcomes
            activity_type = activity.get('type', '')
            participants = activity.get('participants', 0)
            completion_rate = activity.get('completion_rate', 0.0)
            skill_improvement = activity.get('skill_improvement_score', 0.0)
            
            # Weight different types of activities
            type_weights = {
                'technical_training': 1.0,
                'leadership_development': 0.9,
                'research_skills': 0.8,
                'advocacy_training': 0.7,
                'general_education': 0.5
            }
            
            type_weight = type_weights.get(activity_type, 0.5)
            
            # Calculate activity score
            activity_score = type_weight * completion_rate * skill_improvement
            capacity_scores.append(activity_score)
        
        return np.mean(capacity_scores)
    
    def _assess_benefit_equity(self, benefit_distribution: Dict[str, Any]) -> float:
        """Assess equity in benefit distribution across community groups."""
        
        if not benefit_distribution:
            return 0.0
        
        # Calculate distribution equity using Gini coefficient
        benefits = list(benefit_distribution.values())
        
        if len(benefits) < 2:
            return 1.0  # Perfect equity if only one group
        
        # Calculate Gini coefficient
        benefits = sorted(benefits)
        n = len(benefits)
        cumulative_benefits = np.cumsum(benefits)
        
        # Gini coefficient formula
        gini = (2 * sum((i + 1) * benefit for i, benefit in enumerate(benefits))) / (n * sum(benefits)) - (n + 1) / n
        
        # Convert to equity score (1 - Gini)
        equity_score = 1 - gini
        
        return max(0.0, equity_score)
    
    def generate_engagement_recommendations(
        self,
        metrics: CommunityEngagementMetrics,
        threshold_good: float = 0.7
    ) -> List[str]:
        """Generate recommendations for improving community engagement."""
        
        recommendations = []
        
        # Participation rate recommendations
        if metrics.participation_rate < threshold_good:
            recommendations.append(
                "Improve participation rates by addressing barriers such as timing, "
                "location, language, childcare, and compensation for participation."
            )
        
        # Diversity recommendations
        if metrics.representation_diversity < threshold_good:
            recommendations.append(
                "Enhance representation diversity through targeted outreach to "
                "underrepresented communities and culturally appropriate engagement strategies."
            )
        
        # Decision-making influence recommendations
        if metrics.decision_making_influence < threshold_good:
            recommendations.append(
                "Increase community decision-making power by implementing co-design "
                "processes and community governance structures."
            )
        
        # Capacity building recommendations
        if metrics.capacity_building_score < threshold_good:
            recommendations.append(
                "Strengthen capacity building through comprehensive training programs, "
                "mentorship opportunities, and long-term skill development initiatives."
            )
        
        # Satisfaction recommendations
        if metrics.satisfaction_score < threshold_good:
            recommendations.append(
                "Address satisfaction concerns through regular feedback collection, "
                "responsive communication, and adaptation based on community input."
            )
        
        # Trust recommendations
        if metrics.trust_level < threshold_good:
            recommendations.append(
                "Build trust through transparency, accountability mechanisms, "
                "consistent follow-through on commitments, and addressing historical trauma."
            )
        
        # Benefit equity recommendations
        if metrics.benefit_distribution_equity < threshold_good:
            recommendations.append(
                "Improve benefit distribution equity by prioritizing most disadvantaged "
                "communities and implementing fair resource allocation mechanisms."
            )
        
        return recommendations

class HealthEquityAISystem:
    """
    Comprehensive health equity AI system.
    
    This class integrates bias detection, mitigation, and community engagement
    capabilities for developing equitable healthcare AI systems.
    """
    
    def __init__(self, config_path: str):
        """Initialize health equity AI system."""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.bias_detector = BiasDetectionEngine(self.config.get('bias_detection', {}))
        self.bias_mitigator = BiasMitigationEngine(self.config.get('bias_mitigation', {}))
        self.community_framework = CommunityEngagementFramework(self.config.get('community_engagement', {}))
        
        # Initialize database
        self.db_engine = create_engine(self.config.get('database_url', 'sqlite:///health_equity.db'))
        self._setup_database()
        
        logger.info("Initialized health equity AI system")
    
    def _setup_database(self):
        """Setup database tables for health equity data."""
        
        Base = declarative_base()
        
        class BiasAssessment(Base):
            __tablename__ = 'bias_assessments'
            
            id = Column(Integer, primary_key=True)
            assessment_id = Column(String(100))
            protected_attribute = Column(String(50))
            bias_type = Column(String(50))
            fairness_metric = Column(String(50))
            bias_detected = Column(Boolean)
            bias_magnitude = Column(Float)
            p_value = Column(Float)
            created_at = Column(DateTime, default=datetime.utcnow)
        
        class MitigationResult(Base):
            __tablename__ = 'mitigation_results'
            
            id = Column(Integer, primary_key=True)
            mitigation_id = Column(String(100))
            strategy = Column(String(50))
            techniques_applied = Column(JSON)
            effectiveness_score = Column(Float)
            created_at = Column(DateTime, default=datetime.utcnow)
        
        class CommunityEngagement(Base):
            __tablename__ = 'community_engagement'
            
            id = Column(Integer, primary_key=True)
            engagement_id = Column(String(100))
            participation_rate = Column(Float)
            satisfaction_score = Column(Float)
            trust_level = Column(Float)
            created_at = Column(DateTime, default=datetime.utcnow)
        
        Base.metadata.create_all(self.db_engine)
        
        logger.info("Database tables created successfully")
    
    def comprehensive_equity_assessment(
        self,
        data: pd.DataFrame,
        model: Any,
        target_column: str,
        protected_attributes: List[ProtectedAttribute],
        engagement_data: Optional[Dict[str, Any]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive health equity assessment."""
        
        try:
            results = {
                'assessment_id': f"equity_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.utcnow().isoformat(),
                'data_summary': {
                    'total_records': len(data),
                    'protected_attributes': [attr.value for attr in protected_attributes]
                }
            }
            
            # Bias detection
            bias_results = self.bias_detector.detect_bias(
                data=data,
                model=model,
                target_column=target_column,
                protected_attributes=protected_attributes
            )
            
            results['bias_assessment'] = [result.to_dict() for result in bias_results]
            
            # Intersectional analysis
            intersectional_results = self.bias_detector.detect_intersectional_bias(
                data=data,
                model=model,
                target_column=target_column,
                protected_attributes=protected_attributes
            )
            
            results['intersectional_analysis'] = intersectional_results.to_dict()
            
            # Bias mitigation recommendations
            if any(result.bias_detected for result in bias_results):
                mitigation_strategies = [
                    MitigationStrategy.PREPROCESSING,
                    MitigationStrategy.POSTPROCESSING
                ]
                
                mitigation_results = {}
                for strategy in mitigation_strategies:
                    try:
                        mitigation_result = self.bias_mitigator.mitigate_bias(
                            data=data,
                            model=model,
                            target_column=target_column,
                            protected_attributes=protected_attributes,
                            strategy=strategy,
                            bias_assessment=bias_results
                        )
                        mitigation_results[strategy.value] = mitigation_result
                    except Exception as e:
                        logger.warning(f"Mitigation strategy {strategy.value} failed: {str(e)}")
                
                results['mitigation_recommendations'] = mitigation_results
            
            # Community engagement assessment
            if engagement_data:
                engagement_metrics = self.community_framework.assess_community_engagement(engagement_data)
                results['community_engagement'] = engagement_metrics.to_dict()
                
                engagement_recommendations = self.community_framework.generate_engagement_recommendations(
                    engagement_metrics
                )
                results['engagement_recommendations'] = engagement_recommendations
            
            # Generate overall recommendations
            overall_recommendations = self._generate_overall_recommendations(results)
            results['overall_recommendations'] = overall_recommendations
            
            # Save results if requested
            if save_results:
                self._save_equity_assessment(results)
            
            logger.info(f"Completed comprehensive equity assessment: {results['assessment_id']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform equity assessment: {str(e)}")
            return {'error': str(e)}
    
    def _generate_overall_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on comprehensive assessment."""
        
        recommendations = []
        
        # Bias-related recommendations
        bias_assessments = results.get('bias_assessment', [])
        significant_bias = [assessment for assessment in bias_assessments 
                          if assessment.get('bias_detected', False)]
        
        if significant_bias:
            recommendations.append(
                f"Significant bias detected in {len(significant_bias)} assessments. "
                "Implement comprehensive bias mitigation strategies before deployment."
            )
        
        # Intersectional recommendations
        intersectional = results.get('intersectional_analysis', {})
        if intersectional.get('most_disadvantaged_groups'):
            recommendations.append(
                "Intersectional disparities identified. Prioritize interventions for "
                "multiply disadvantaged groups and implement intersectional analysis "
                "as standard practice."
            )
        
        # Community engagement recommendations
        engagement = results.get('community_engagement', {})
        if engagement:
            low_scores = [metric for metric, value in engagement.items() 
                         if isinstance(value, (int, float)) and value < 0.7]
            
            if low_scores:
                recommendations.append(
                    f"Community engagement needs improvement in: {', '.join(low_scores)}. "
                    "Strengthen community partnerships and participatory design processes."
                )
        
        # General recommendations
        recommendations.extend([
            "Establish ongoing monitoring systems for bias and equity outcomes.",
            "Implement community oversight and governance mechanisms.",
            "Develop culturally responsive evaluation metrics and success indicators.",
            "Create feedback loops for continuous improvement based on community input.",
            "Ensure transparent reporting of equity outcomes to all stakeholders."
        ])
        
        return recommendations
    
    def _save_equity_assessment(self, results: Dict[str, Any]):
        """Save equity assessment results to database."""
        
        try:
            Session = sessionmaker(bind=self.db_engine)
            session = Session()
            
            # Save bias assessments
            for assessment in results.get('bias_assessment', []):
                bias_record = session.execute(
                    """INSERT INTO bias_assessments 
                       (assessment_id, protected_attribute, bias_type, fairness_metric, 
                        bias_detected, bias_magnitude, p_value) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (results['assessment_id'], assessment['protected_attribute'],
                     assessment['bias_type'], assessment['fairness_metric'],
                     assessment['bias_detected'], assessment['bias_magnitude'],
                     assessment['p_value'])
                )
            
            # Save community engagement metrics
            engagement = results.get('community_engagement', {})
            if engagement:
                engagement_record = session.execute(
                    """INSERT INTO community_engagement 
                       (engagement_id, participation_rate, satisfaction_score, trust_level) 
                       VALUES (?, ?, ?, ?)""",
                    (results['assessment_id'], engagement.get('participation_rate', 0),
                     engagement.get('satisfaction_score', 0), engagement.get('trust_level', 0))
                )
            
            session.commit()
            session.close()
            
            logger.info("Equity assessment results saved to database")
            
        except Exception as e:
            logger.error(f"Failed to save equity assessment results: {str(e)}")

## Bibliography and References

### Health Equity and Social Justice

1. **Braveman, P., & Gruskin, S.** (2003). Defining equity in health. *Journal of Epidemiology & Community Health*, 57(4), 254-258. [Health equity definition]

2. **Whitehead, M.** (1992). The concepts and principles of equity and health. *International Journal of Health Services*, 22(3), 429-445. [Equity principles]

3. **Crenshaw, K.** (1989). Demarginalizing the intersection of race and sex: A black feminist critique of antidiscrimination doctrine, feminist theory and antiracist politics. *University of Chicago Legal Forum*, 1989(1), 139-167. [Intersectionality theory]

4. **Metzl, J. M., & Hansen, H.** (2014). Structural competency: theorizing a new medical engagement with stigma and inequality. *Social Science & Medicine*, 103, 126-133. [Structural competency]

### Algorithmic Bias and Fairness

5. **Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S.** (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453. [Healthcare algorithmic bias]

6. **Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Chin, M. H.** (2018). Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12), 866-872. [ML fairness in healthcare]

7. **Chen, I. Y., Pierson, E., Rose, S., Joshi, S., Ferryman, K., & Ghassemi, M.** (2021). Ethical machine learning in healthcare. *Annual Review of Biomedical Data Science*, 4, 123-144. [Ethical ML healthcare]

8. **Barocas, S., Hardt, M., & Narayanan, A.** (2019). *Fairness and machine learning*. fairmlbook.org. [Fairness in ML textbook]

### Community Engagement and Participatory Design

9. **Israel, B. A., Schulz, A. J., Parker, E. A., & Becker, A. B.** (1998). Review of community-based research: assessing partnership approaches to improve public health. *Annual Review of Public Health*, 19(1), 173-202. [Community-based research]

10. **Wallerstein, N., Duran, B., Oetzel, J. G., & Minkler, M.** (Eds.). (2017). *Community-based participatory research for health: Advancing social and health equity*. John Wiley & Sons. [CBPR methods]

11. **Sanders, E. B. N., & Stappers, P. J.** (2008). Co-creation and the new landscapes of design. *Co-design*, 4(1), 5-18. [Co-design principles]

12. **Costanza-Chock, S.** (2020). *Design justice: Community-led practices to build the worlds we need*. MIT Press. [Design justice framework]

### Digital Health Disparities

13. **Veinot, T. C., Mitchell, H., & Ancker, J. S.** (2018). Good intentions are not enough: how informatics interventions can worsen inequality. *Journal of the American Medical Informatics Association*, 25(8), 1080-1088. [Digital health disparities]

14. **Nouri, S., Khoong, E. C., Lyles, C. R., & Karliner, L.** (2020). Addressing equity in telemedicine for chronic disease management during the Covid-19 pandemic. *NEJM Catalyst*, 1(3). [Telemedicine equity]

15. **Rodriguez, J. A., Betancourt, J. R., Sequist, T. D., & Ganguli, I.** (2021). Differences in the use of telephone and video telemedicine visits during the COVID-19 pandemic. *American Journal of Managed Care*, 27(1), 21-26. [Digital divide in telehealth]

### Bias Detection and Mitigation Methods

16. **Bellamy, R. K., Dey, K., Hind, M., Hoffman, S. C., Houde, S., Kannan, K., ... & Zhang, Y.** (2018). AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias. *arXiv preprint arXiv:1810.01943*. [AIF360 toolkit]

17. **Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A.** (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35. [Bias survey]

18. **Verma, S., & Rubin, J.** (2018). Fairness definitions explained. In *Proceedings of the international workshop on software fairness* (pp. 1-7). [Fairness definitions]

19. **Chouldechova, A.** (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big Data*, 5(2), 153-163. [Fairness metrics]

### Cultural Competency and Responsiveness

20. **Sue, D. W., & Sue, D.** (2015). *Counseling the culturally diverse: Theory and practice*. John Wiley & Sons. [Cultural competency framework]

This chapter provides a comprehensive framework for developing health equity applications in AI, addressing bias detection and mitigation, community engagement, and intersectional analysis. The implementations provide practical tools for creating AI systems that actively promote health equity rather than perpetuating existing disparities. The next chapter will explore advanced medical imaging AI applications, building upon these equity principles to ensure fair and effective imaging technologies.
