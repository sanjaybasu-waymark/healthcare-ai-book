---
layout: default
title: "Chapter 2: Mathematical Foundations"
nav_order: 2
parent: Chapters
permalink: /chapters/02-mathematical-foundations/
---

# Chapter 2: Mathematical Foundations for Healthcare AI

*By Sanjay Basu MD PhD*

## Learning Objectives

By the end of this chapter, physician data scientists will be able to:

- Apply probability theory and Bayesian inference to clinical decision-making with rigorous mathematical foundations
- Implement linear algebra operations for high-dimensional healthcare data analysis and dimensionality reduction
- Design and optimize machine learning models using advanced optimization theory and gradient-based methods
- Utilize information theory and entropy measures for feature selection and uncertainty quantification
- Apply causal inference methods to healthcare data for treatment effect estimation and policy evaluation
- Implement time series analysis techniques for longitudinal patient data and temporal prediction models
- Understand and apply advanced statistical methods including survival analysis and competing risks models

## 2.1 Introduction: Mathematical Rigor in Healthcare AI

The mathematical foundations underlying healthcare artificial intelligence represent a sophisticated synthesis of statistical theory, linear algebra, probability theory, optimization methods, and information theory that have been refined over centuries of mathematical development. This chapter provides a comprehensive exploration of these mathematical principles, demonstrating how they enable the development of robust, clinically validated AI systems that can meaningfully improve patient outcomes while maintaining the highest standards of scientific rigor.

Healthcare AI differs fundamentally from other AI applications due to the unique characteristics of medical data, the high stakes of clinical decision-making, and the complex regulatory environment in which these systems must operate. The mathematical frameworks that support healthcare AI must therefore address challenges such as missing data, measurement uncertainty, temporal dependencies, causal inference, and the need for interpretable models that can be validated by clinical experts.

### 2.1.1 The Clinical Context of Mathematical Modeling

Clinical medicine operates in an environment of fundamental uncertainty, where diagnostic decisions must be made with incomplete information, treatment effects vary across individuals, and outcomes are influenced by complex interactions between biological, social, and environmental factors. Mathematical modeling provides the tools necessary to quantify this uncertainty, make optimal decisions under constraints, and continuously improve clinical practice through evidence-based learning.

The application of mathematical methods to healthcare requires careful consideration of the clinical context in which these methods will be applied. Unlike other domains where mathematical optimization can focus solely on predictive accuracy, healthcare applications must balance multiple competing objectives including patient safety, clinical interpretability, regulatory compliance, and health equity. This multi-objective optimization problem requires sophisticated mathematical frameworks that can accommodate these diverse requirements while maintaining computational efficiency.

### 2.1.2 Foundational Mathematical Principles

The mathematical foundations of healthcare AI rest on several key principles that distinguish medical applications from other domains:

**Uncertainty Quantification**: Medical decisions must be made under uncertainty, requiring probabilistic frameworks that can quantify confidence in predictions and recommendations. This goes beyond simple point estimates to include full uncertainty distributions that can inform clinical decision-making.

**Causal Reasoning**: Healthcare interventions are fundamentally causal in nature, requiring mathematical frameworks that can distinguish between correlation and causation. This is essential for developing AI systems that can support treatment decisions and policy interventions.

**Temporal Modeling**: Patient health states evolve over time, with complex dependencies between past treatments, current conditions, and future outcomes. Mathematical models must capture these temporal relationships while accounting for irregular sampling and missing data.

**Interpretability**: Clinical decisions require explanations that can be understood and validated by healthcare providers. Mathematical models must therefore balance predictive accuracy with interpretability, often requiring specialized techniques for model explanation and validation.

## 2.2 Probability Theory and Statistical Inference in Healthcare

Probability theory forms the cornerstone of healthcare AI, providing the mathematical framework for reasoning under uncertainty that is inherent in medical diagnosis, treatment selection, and outcome prediction. The application of probability theory to healthcare requires careful consideration of the unique characteristics of medical data and the clinical context in which probabilistic reasoning occurs.

### 2.2.1 Bayesian Inference in Clinical Decision Making

Bayesian inference provides a principled approach to updating beliefs about patient conditions based on new evidence, making it particularly well-suited for clinical applications where prior knowledge and new observations must be systematically combined. The fundamental theorem of Bayesian inference, known as Bayes' theorem, can be expressed as:

$$

P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}

$$

Where:
- $P(H|E)$ is the posterior probability of hypothesis $H$ given evidence $E$
- $P(E|H)$ is the likelihood of observing evidence $E$ given hypothesis $H$
- $P(H)$ is the prior probability of hypothesis $H$
- $P(E)$ is the marginal probability of evidence $E$

In the clinical context, this theorem enables the systematic integration of diagnostic test results with prior clinical knowledge to arrive at updated probability estimates for various medical conditions. The power of Bayesian inference lies in its ability to incorporate uncertainty at every level of the analysis while providing interpretable results that can be understood and validated by clinical experts.

The application of Bayesian methods to healthcare AI requires careful consideration of prior specification, likelihood modeling, and computational implementation. The following comprehensive implementation demonstrates how Bayesian inference can be applied to clinical diagnosis:

```python
"""
Comprehensive Bayesian Inference System for Clinical Diagnosis

This implementation demonstrates advanced Bayesian techniques including
hierarchical modeling, MCMC sampling, and clinical validation frameworks
specifically designed for healthcare AI applications.

Author: Sanjay Basu MD PhD
License: MIT
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import logsumexp, gammaln
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import logging
from abc import ABC, abstractmethod
from enum import Enum
import json

# Advanced statistical libraries
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not available. Some advanced Bayesian features will be disabled.")

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of diagnostic tests."""
    LABORATORY = "laboratory"
    IMAGING = "imaging"
    CLINICAL_EXAM = "clinical_exam"
    GENETIC = "genetic"
    BIOMARKER = "biomarker"

class EvidenceLevel(Enum):
    """Levels of clinical evidence."""
    LEVEL_1A = "1a"  # Systematic review of RCTs
    LEVEL_1B = "1b"  # Individual RCT
    LEVEL_2A = "2a"  # Systematic review of cohort studies
    LEVEL_2B = "2b"  # Individual cohort study
    LEVEL_3A = "3a"  # Systematic review of case-control studies
    LEVEL_3B = "3b"  # Individual case-control study
    LEVEL_4 = "4"    # Case series
    LEVEL_5 = "5"    # Expert opinion

@dataclass
class DiagnosticTest:
    """
    Represents a diagnostic test with comprehensive performance characteristics.
    
    This class encapsulates all relevant information about a diagnostic test
    including performance metrics, cost considerations, and clinical context.
    """
    test_name: str
    test_type: TestType
    sensitivity: float  # True positive rate
    specificity: float  # True negative rate
    positive_predictive_value: Optional[float] = None
    negative_predictive_value: Optional[float] = None
    cost: float = 0.0
    risk_level: str = "low"
    turnaround_time_hours: float = 24.0
    reference_standard: str = "gold_standard"
    evidence_level: EvidenceLevel = EvidenceLevel.LEVEL_3B
    sample_size_validation: int = 0
    confidence_interval_sensitivity: Tuple[float, float] = (0.0, 1.0)
    confidence_interval_specificity: Tuple[float, float] = (0.0, 1.0)
    
    def __post_init__(self):
        """Validate test parameters and calculate derived metrics."""
        if not self.validate_performance():
            raise ValueError(f"Invalid performance parameters for test: {self.test_name}")
        
        # Calculate likelihood ratios
        self._positive_lr = self.sensitivity / (1 - self.specificity) if self.specificity < 1.0 else float('inf')
        self._negative_lr = (1 - self.sensitivity) / self.specificity if self.specificity > 0.0 else 0.0
    
    @property
    def positive_likelihood_ratio(self) -> float:
        """Calculate positive likelihood ratio."""
        return self._positive_lr
    
    @property
    def negative_likelihood_ratio(self) -> float:
        """Calculate negative likelihood ratio."""
        return self._negative_lr
    
    @property
    def diagnostic_odds_ratio(self) -> float:
        """Calculate diagnostic odds ratio."""
        if self._negative_lr > 0:
            return self._positive_lr / self._negative_lr
        return float('inf')
    
    def validate_performance(self) -> bool:
        """Validate test performance parameters."""
        return (0 <= self.sensitivity <= 1 and 
<= self.specificity <= 1 and
                self.cost >= 0 and
                self.turnaround_time_hours >= 0)
    
    def calculate_predictive_values(self, prevalence: float) -> Tuple[float, float]:
        """
        Calculate positive and negative predictive values for given prevalence.
        
        Args:
            prevalence: Disease prevalence in the population
            
        Returns:
            Tuple of (positive_predictive_value, negative_predictive_value)
        """
        if not (0 <= prevalence <= 1):
            raise ValueError("Prevalence must be between 0 and 1")
        
        # Calculate using Bayes' theorem
        ppv = (self.sensitivity * prevalence) / (
            self.sensitivity * prevalence + (1 - self.specificity) * (1 - prevalence)
        )
        
        npv = (self.specificity * (1 - prevalence)) / (
            self.specificity * (1 - prevalence) + (1 - self.sensitivity) * prevalence
        )
        
        return ppv, npv
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'positive_lr': self.positive_likelihood_ratio,
            'negative_lr': self.negative_likelihood_ratio,
            'diagnostic_odds_ratio': self.diagnostic_odds_ratio,
            'cost': self.cost,
            'turnaround_time_hours': self.turnaround_time_hours,
            'evidence_level': self.evidence_level.value,
            'sample_size_validation': self.sample_size_validation
        }

@dataclass
class ClinicalCondition:
    """
    Represents a clinical condition with comprehensive epidemiological data.
    
    This class encapsulates all relevant information about a medical condition
    including prevalence, severity, and treatment considerations.
    """
    condition_name: str
    icd_10_code: str
    prevalence: float
    age_adjusted_prevalence: Dict[str, float] = field(default_factory=dict)
    gender_specific_prevalence: Dict[str, float] = field(default_factory=dict)
    severity_score: int = 1  # 1-5 scale
    mortality_risk: float = 0.0
    morbidity_risk: float = 0.0
    treatment_available: bool = True
    treatment_effectiveness: float = 0.8
    natural_history: str = "chronic"
    risk_factors: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    
    def validate_parameters(self) -> bool:
        """Validate condition parameters."""
        return (0 <= self.prevalence <= 1 and
<= self.severity_score <= 5 and
<= self.mortality_risk <= 1 and
<= self.morbidity_risk <= 1 and
<= self.treatment_effectiveness <= 1)
    
    def get_adjusted_prevalence(self, 
                               age: Optional[int] = None,
                               gender: Optional[str] = None) -> float:
        """
        Get prevalence adjusted for patient demographics.
        
        Args:
            age: Patient age in years
            gender: Patient gender ('male', 'female', 'other')
            
        Returns:
            Adjusted prevalence estimate
        """
        adjusted_prevalence = self.prevalence
        
        # Adjust for age if available
        if age is not None and self.age_adjusted_prevalence:
            age_group = self._get_age_group(age)
            if age_group in self.age_adjusted_prevalence:
                adjusted_prevalence *= self.age_adjusted_prevalence[age_group]
        
        # Adjust for gender if available
        if gender is not None and self.gender_specific_prevalence:
            if gender.lower() in self.gender_specific_prevalence:
                adjusted_prevalence *= self.gender_specific_prevalence[gender.lower()]
        
        return min(adjusted_prevalence, 1.0)  # Cap at 1.0
    
    def _get_age_group(self, age: int) -> str:
        """Categorize age into standard age groups."""
        if age < 18:
            return "pediatric"
        elif age < 65:
            return "adult"
        else:
            return "elderly"

class BayesianDiagnosticSystem:
    """
    Advanced Bayesian system for clinical diagnosis with uncertainty quantification.
    
    This system implements state-of-the-art Bayesian methods for combining
    multiple diagnostic tests, incorporating prior clinical knowledge, and
    providing uncertainty-aware diagnostic recommendations.
    """
    
    def __init__(self, 
                 enable_hierarchical_modeling: bool = True,
                 mcmc_samples: int = 2000,
                 random_seed: int = 42):
        """
        Initialize Bayesian diagnostic system.
        
        Args:
            enable_hierarchical_modeling: Whether to use hierarchical Bayesian models
            mcmc_samples: Number of MCMC samples for posterior inference
            random_seed: Random seed for reproducibility
        """
        self.enable_hierarchical_modeling = enable_hierarchical_modeling
        self.mcmc_samples = mcmc_samples
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
        # Data storage
        self.conditions: Dict[str, ClinicalCondition] = {}
        self.tests: Dict[str, DiagnosticTest] = {}
        self.test_correlations: Optional[np.ndarray] = None
        self.population_data: Optional[pd.DataFrame] = None
        
        # Model storage
        self.mcmc_trace = None
        self.posterior_samples: Dict[str, np.ndarray] = {}
        
        # Performance tracking
        self.diagnostic_history: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Dict[str, float]] = {}
        
        logger.info("Bayesian diagnostic system initialized")
    
    def add_condition(self, condition: ClinicalCondition) -> bool:
        """
        Add a clinical condition to the diagnostic system.
        
        Args:
            condition: ClinicalCondition object to add
            
        Returns:
            True if condition added successfully
        """
        try:
            if not condition.validate_parameters():
                raise ValueError(f"Invalid parameters for condition: {condition.condition_name}")
            
            self.conditions[condition.condition_name] = condition
            logger.info(f"Added condition: {condition.condition_name} (ICD-10: {condition.icd_10_code})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding condition {condition.condition_name}: {e}")
            return False
    
    def add_diagnostic_test(self, test: DiagnosticTest) -> bool:
        """
        Add a diagnostic test to the system.
        
        Args:
            test: DiagnosticTest object to add
            
        Returns:
            True if test added successfully
        """
        try:
            if not test.validate_performance():
                raise ValueError(f"Invalid parameters for test: {test.test_name}")
            
            self.tests[test.test_name] = test
            logger.info(f"Added diagnostic test: {test.test_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding test {test.test_name}: {e}")
            return False
    
    def calculate_posterior_probability(
        self, 
        condition_name: str, 
        test_results: Dict[str, bool],
        patient_demographics: Optional[Dict[str, Any]] = None,
        use_test_correlations: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate posterior probability using advanced Bayesian inference.
        
        Args:
            condition_name: Name of the condition to evaluate
            test_results: Dictionary of test names and results (True/False)
            patient_demographics: Optional patient demographic information
            use_test_correlations: Whether to account for test correlations
            
        Returns:
            Dictionary containing posterior probability and uncertainty measures
        """
        if condition_name not in self.conditions:
            raise ValueError(f"Unknown condition: {condition_name}")
        
        condition = self.conditions[condition_name]
        
        # Get adjusted prior probability
        prior_prob = self._get_adjusted_prior(condition, patient_demographics)
        
        # Calculate likelihood with correlation adjustment
        if use_test_correlations and self.test_correlations is not None:
            likelihood_pos, likelihood_neg = self._calculate_correlated_likelihood(
                test_results, condition_name
            )
        else:
            likelihood_pos, likelihood_neg = self._calculate_independent_likelihood(
                test_results
            )
        
        # Bayesian inference using log-space arithmetic for numerical stability
        log_prior_pos = np.log(prior_prob)
        log_prior_neg = np.log(1 - prior_prob)
        
        log_posterior_pos = log_prior_pos + likelihood_pos
        log_posterior_neg = log_prior_neg + likelihood_neg
        
        # Normalize using logsumexp
        log_evidence = logsumexp([log_posterior_pos, log_posterior_neg])
        
        posterior_prob = np.exp(log_posterior_pos - log_evidence)
        
        # Calculate uncertainty measures
        uncertainty_measures = self._calculate_uncertainty_measures(
            posterior_prob, test_results, prior_prob
        )
        
        # Store diagnostic decision for learning
        diagnostic_record = {
            'timestamp': datetime.now(),
            'condition': condition_name,
            'test_results': test_results.copy(),
            'patient_demographics': patient_demographics,
            'prior_probability': prior_prob,
            'posterior_probability': posterior_prob,
            'uncertainty_measures': uncertainty_measures
        }
        self.diagnostic_history.append(diagnostic_record)
        
        return {
            'condition': condition_name,
            'posterior_probability': posterior_prob,
            'prior_probability': prior_prob,
            'likelihood_ratio': np.exp(likelihood_pos - likelihood_neg),
            'uncertainty_measures': uncertainty_measures,
            'test_contributions': self._calculate_test_contributions(test_results),
            'clinical_interpretation': self._generate_clinical_interpretation(
                posterior_prob, uncertainty_measures
            )
        }
    
    def perform_hierarchical_analysis(self, 
                                    patient_data: pd.DataFrame,
                                    condition_name: str) -> Dict[str, Any]:
        """
        Perform hierarchical Bayesian analysis for population-level inference.
        
        Args:
            patient_data: DataFrame with patient data and outcomes
            condition_name: Condition to analyze
            
        Returns:
            Dictionary with hierarchical analysis results
        """
        if not HAS_PYMC:
            raise ImportError("PyMC is required for hierarchical analysis")
        
        if condition_name not in self.conditions:
            raise ValueError(f"Unknown condition: {condition_name}")
        
        try:
            # Prepare data for hierarchical modeling
            y = patient_data[f'has_{condition_name}'].values.astype(int)
            X = patient_data[[col for col in patient_data.columns 
                            if col.startswith('test_')]].values
            
            # Build hierarchical model
            with pm.Model() as hierarchical_model:
                # Hyperpriors for test performance
                alpha_sens = pm.Beta('alpha_sens', alpha=2, beta=2, shape=X.shape<sup>1</sup>)
                beta_spec = pm.Beta('beta_spec', alpha=2, beta=2, shape=X.shape<sup>1</sup>)
                
                # Individual test sensitivities and specificities
                sensitivity = pm.Beta('sensitivity', alpha=alpha_sens * 100, 
                                    beta=(1 - alpha_sens) * 100, shape=X.shape<sup>1</sup>)
                specificity = pm.Beta('specificity', alpha=beta_spec * 100,
                                    beta=(1 - beta_spec) * 100, shape=X.shape<sup>1</sup>)
                
                # Prior probability (prevalence)
                prevalence = pm.Beta('prevalence', alpha=1, beta=1)
                
                # Likelihood for each patient
                for i in range(len(y)):
                    # Calculate likelihood for positive and negative cases
                    likelihood_pos = pm.math.prod(
                        pm.math.where(X[i] == 1, sensitivity, 1 - sensitivity)
                    )
                    likelihood_neg = pm.math.prod(
                        pm.math.where(X[i] == 1, 1 - specificity, specificity)
                    )
                    
                    # Posterior probability
                    posterior_prob = (prevalence * likelihood_pos) / (
                        prevalence * likelihood_pos + (1 - prevalence) * likelihood_neg
                    )
                    
                    # Observed outcome
                    pm.Bernoulli(f'obs_{i}', p=posterior_prob, observed=y[i])
                
                # Sample from posterior
                trace = pm.sample(self.mcmc_samples, random_seed=self.random_seed,
                                return_inferencedata=True)
            
            self.mcmc_trace = trace
            
            # Extract posterior summaries
            summary = az.summary(trace)
            
            # Calculate model diagnostics
            diagnostics = {
                'r_hat': summary['r_hat'].max(),
                'ess_bulk': summary['ess_bulk'].min(),
                'ess_tail': summary['ess_tail'].min(),
                'mcse_mean': summary['mcse_mean'].max(),
                'mcse_sd': summary['mcse_sd'].max()
            }
            
            return {
                'model_summary': summary.to_dict(),
                'diagnostics': diagnostics,
                'posterior_samples': {
                    var: trace.posterior[var].values.flatten() 
                    for var in ['prevalence', 'sensitivity', 'specificity']
                },
                'convergence_achieved': diagnostics['r_hat'] < 1.1
            }
            
        except Exception as e:
            logger.error(f"Error in hierarchical analysis: {e}")
            raise
    
    def validate_diagnostic_performance(self, 
                                      validation_data: pd.DataFrame,
                                      condition_name: str) -> Dict[str, float]:
        """
        Validate diagnostic system performance using held-out data.
        
        Args:
            validation_data: DataFrame with validation data
            condition_name: Condition to validate
            
        Returns:
            Dictionary with validation metrics
        """
        if condition_name not in self.conditions:
            raise ValueError(f"Unknown condition: {condition_name}")
        
        try:
            # Extract true labels and test results
            y_true = validation_data[f'has_{condition_name}'].values
            test_columns = [col for col in validation_data.columns 
                          if col.startswith('test_')]
            
            # Calculate posterior probabilities for all patients
            y_pred_proba = []
            y_pred_binary = []
            
            for idx, row in validation_data.iterrows():
                test_results = {col.replace('test_', ''): bool(row[col]) 
                              for col in test_columns}
                
                patient_demographics = {
                    'age': row.get('age'),
                    'gender': row.get('gender')
                }
                
                result = self.calculate_posterior_probability(
                    condition_name, test_results, patient_demographics
                )
                
                y_pred_proba.append(result['posterior_probability'])
                y_pred_binary.append(result['posterior_probability'] > 0.5)
            
            y_pred_proba = np.array(y_pred_proba)
            y_pred_binary = np.array(y_pred_binary)
            
            # Calculate performance metrics
            auc_roc = roc_auc_score(y_true, y_pred_proba)
            
            # Precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            auc_pr = np.trapz(precision, recall)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred_binary)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate derived metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # F1 score
            f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
            
            # Calibration metrics
            calibration_slope, calibration_intercept = self._calculate_calibration_metrics(
                y_true, y_pred_proba
            )
            
            validation_metrics = {
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'positive_predictive_value': ppv,
                'negative_predictive_value': npv,
                'f1_score': f1_score,
                'calibration_slope': calibration_slope,
                'calibration_intercept': calibration_intercept,
                'n_patients': len(y_true),
                'prevalence': np.mean(y_true)
            }
            
            # Store validation results
            self.validation_results[condition_name] = validation_metrics
            
            logger.info(f"Validation completed for {condition_name}: AUC-ROC = {auc_roc:.3f}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            raise
    
    def _get_adjusted_prior(self, 
                           condition: ClinicalCondition,
                           patient_demographics: Optional[Dict[str, Any]]) -> float:
        """Get prior probability adjusted for patient demographics."""
        if patient_demographics is None:
            return condition.prevalence
        
        age = patient_demographics.get('age')
        gender = patient_demographics.get('gender')
        
        return condition.get_adjusted_prevalence(age, gender)
    
    def _calculate_independent_likelihood(self, 
                                        test_results: Dict[str, bool]) -> Tuple[float, float]:
        """Calculate likelihood assuming test independence."""
        log_likelihood_pos = 0.0
        log_likelihood_neg = 0.0
        
        for test_name, result in test_results.items():
            if test_name not in self.tests:
                logger.warning(f"Unknown test: {test_name}")
                continue
            
            test = self.tests[test_name]
            
            if result:  # Positive test result
                log_likelihood_pos += np.log(test.sensitivity)
                log_likelihood_neg += np.log(1 - test.specificity)
            else:  # Negative test result
                log_likelihood_pos += np.log(1 - test.sensitivity)
                log_likelihood_neg += np.log(test.specificity)
        
        return log_likelihood_pos, log_likelihood_neg
    
    def _calculate_correlated_likelihood(self, 
                                       test_results: Dict[str, bool],
                                       condition_name: str) -> Tuple[float, float]:
        """Calculate likelihood accounting for test correlations."""
        # Simplified implementation - in practice would use copula models
        # or multivariate distributions to model test correlations
        
        # For now, apply a correlation adjustment factor
        log_likelihood_pos, log_likelihood_neg = self._calculate_independent_likelihood(test_results)
        
        # Reduce likelihood magnitude to account for positive correlations
        correlation_adjustment = 0.9  # Assumes moderate positive correlation
        
        return (log_likelihood_pos * correlation_adjustment, 
                log_likelihood_neg * correlation_adjustment)
    
    def _calculate_uncertainty_measures(self, 
                                      posterior_prob: float,
                                      test_results: Dict[str, bool],
                                      prior_prob: float) -> Dict[str, float]:
        """Calculate comprehensive uncertainty measures."""
        # Entropy-based uncertainty
        if posterior_prob == 0 or posterior_prob == 1:
            entropy = 0.0
        else:
            entropy = -(posterior_prob * np.log2(posterior_prob) + 
                       (1 - posterior_prob) * np.log2(1 - posterior_prob))
        
        # Information gain
        if prior_prob == 0 or prior_prob == 1:
            prior_entropy = 0.0
        else:
            prior_entropy = -(prior_prob * np.log2(prior_prob) + 
                            (1 - prior_prob) * np.log2(1 - prior_prob))
        
        information_gain = prior_entropy - entropy
        
        # Confidence based on distance from 0.5
        confidence = abs(posterior_prob - 0.5) * 2
        
        # Number of tests contributing to decision
        n_tests = len(test_results)
        
        return {
            'entropy': entropy,
            'information_gain': information_gain,
            'confidence': confidence,
            'n_tests': n_tests,
            'uncertainty_category': self._categorize_uncertainty(entropy, confidence)
        }
    
    def _categorize_uncertainty(self, entropy: float, confidence: float) -> str:
        """Categorize uncertainty level for clinical interpretation."""
        if entropy < 0.5 and confidence > 0.8:
            return "low_uncertainty"
        elif entropy < 0.8 and confidence > 0.6:
            return "moderate_uncertainty"
        else:
            return "high_uncertainty"
    
    def _calculate_test_contributions(self, test_results: Dict[str, bool]) -> Dict[str, float]:
        """Calculate individual test contributions to the diagnostic decision."""
        contributions = {}
        
        for test_name, result in test_results.items():
            if test_name not in self.tests:
                continue
            
            test = self.tests[test_name]
            
            if result:
                # Positive test - contribution is positive likelihood ratio
                contribution = np.log(test.positive_likelihood_ratio)
            else:
                # Negative test - contribution is negative likelihood ratio
                contribution = np.log(test.negative_likelihood_ratio)
            
            contributions[test_name] = contribution
        
        return contributions
    
    def _generate_clinical_interpretation(self, 
                                        posterior_prob: float,
                                        uncertainty_measures: Dict[str, float]) -> str:
        """Generate clinical interpretation of diagnostic results."""
        prob_percent = posterior_prob * 100
        uncertainty_category = uncertainty_measures['uncertainty_category']
        
        if posterior_prob >= 0.9:
            interpretation = f"High probability ({prob_percent:.1f}%) of condition"
        elif posterior_prob >= 0.7:
            interpretation = f"Moderate-high probability ({prob_percent:.1f}%) of condition"
        elif posterior_prob >= 0.3:
            interpretation = f"Intermediate probability ({prob_percent:.1f}%) of condition"
        elif posterior_prob >= 0.1:
            interpretation = f"Low-moderate probability ({prob_percent:.1f}%) of condition"
        else:
            interpretation = f"Low probability ({prob_percent:.1f}%) of condition"
        
        if uncertainty_category == "high_uncertainty":
            interpretation += " - Consider additional testing"
        elif uncertainty_category == "moderate_uncertainty":
            interpretation += " - Moderate confidence in assessment"
        else:
            interpretation += " - High confidence in assessment"
        
        return interpretation
    
    def _calculate_calibration_metrics(self, 
                                     y_true: np.ndarray,
                                     y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """Calculate calibration slope and intercept."""
        from sklearn.linear_model import LinearRegression
        
        # Logit transformation of predicted probabilities
        y_pred_logit = np.log(y_pred_proba / (1 - y_pred_proba + 1e-10))
        
        # Fit calibration model
        calibration_model = LinearRegression()
        calibration_model.fit(y_pred_logit.reshape(-1, 1), y_true)
        
        slope = calibration_model.coef_<sup>0</sup>
        intercept = calibration_model.intercept_
        
        return slope, intercept
    
    def generate_diagnostic_report(self, 
                                 condition_name: str,
                                 include_validation: bool = True) -> Dict[str, Any]:
        """Generate comprehensive diagnostic system report."""
        if condition_name not in self.conditions:
            raise ValueError(f"Unknown condition: {condition_name}")
        
        condition = self.conditions[condition_name]
        
        # Basic condition information
        report = {
            'condition_name': condition_name,
            'icd_10_code': condition.icd_10_code,
            'prevalence': condition.prevalence,
            'severity_score': condition.severity_score,
            'generated_at': datetime.now().isoformat()
        }
        
        # Available tests
        relevant_tests = {name: test.get_performance_summary() 
                         for name, test in self.tests.items()}
        report['available_tests'] = relevant_tests
        
        # Validation results if available
        if include_validation and condition_name in self.validation_results:
            report['validation_results'] = self.validation_results[condition_name]
        
        # Diagnostic history summary
        condition_history = [record for record in self.diagnostic_history 
                           if record['condition'] == condition_name]
        
        if condition_history:
            report['diagnostic_statistics'] = {
                'total_diagnoses': len(condition_history),
                'average_posterior_probability': np.mean([
                    record['posterior_probability'] for record in condition_history
                ]),
                'high_confidence_diagnoses': len([
                    record for record in condition_history 
                    if record['uncertainty_measures']['confidence'] > 0.8
                ])
            }
        
        return report


# Demonstration and testing functions
def create_example_diagnostic_system() -> BayesianDiagnosticSystem:
    """Create an example diagnostic system for demonstration."""
    system = BayesianDiagnosticSystem()
    
    # Add example condition: Myocardial Infarction
    mi_condition = ClinicalCondition(
        condition_name="Myocardial Infarction",
        icd_10_code="I21.9",
        prevalence=0.05,
        age_adjusted_prevalence={
            "adult": 1.0,
            "elderly": 2.5,
            "pediatric": 0.1
        },
        gender_specific_prevalence={
            "male": 1.5,
            "female": 0.8
        },
        severity_score=5,
        mortality_risk=0.15,
        treatment_available=True,
        treatment_effectiveness=0.85
    )
    system.add_condition(mi_condition)
    
    # Add diagnostic tests
    troponin_test = DiagnosticTest(
        test_name="Troponin I",
        test_type=TestType.LABORATORY,
        sensitivity=0.95,
        specificity=0.90,
        cost=50.0,
        turnaround_time_hours=2.0,
        evidence_level=EvidenceLevel.LEVEL_1A,
        sample_size_validation=5000
    )
    system.add_diagnostic_test(troponin_test)
    
    ecg_test = DiagnosticTest(
        test_name="ECG",
        test_type=TestType.CLINICAL_EXAM,
        sensitivity=0.80,
        specificity=0.85,
        cost=25.0,
        turnaround_time_hours=0.25,
        evidence_level=EvidenceLevel.LEVEL_1B,
        sample_size_validation=10000
    )
    system.add_diagnostic_test(ecg_test)
    
    echo_test = DiagnosticTest(
        test_name="Echocardiogram",
        test_type=TestType.IMAGING,
        sensitivity=0.85,
        specificity=0.95,
        cost=200.0,
        turnaround_time_hours=4.0,
        evidence_level=EvidenceLevel.LEVEL_2A,
        sample_size_validation=2000
    )
    system.add_diagnostic_test(echo_test)
    
    return system


def demonstrate_bayesian_diagnosis():
    """Demonstrate comprehensive Bayesian diagnostic system."""
    print("=== Bayesian Diagnostic System Demonstration ===\n")
    
    # Create example system
    system = create_example_diagnostic_system()
    
    # Example patient scenarios
    scenarios = [
        {
            'name': 'High-risk elderly male',
            'demographics': {'age': 72, 'gender': 'male'},
            'test_results': {'Troponin I': True, 'ECG': True, 'Echocardiogram': False}
        },
        {
            'name': 'Low-risk young female',
            'demographics': {'age': 28, 'gender': 'female'},
            'test_results': {'Troponin I': False, 'ECG': False, 'Echocardiogram': False}
        },
        {
            'name': 'Intermediate-risk middle-aged male',
            'demographics': {'age': 55, 'gender': 'male'},
            'test_results': {'Troponin I': True, 'ECG': False, 'Echocardiogram': True}
        }
    ]
    
    # Analyze each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print("-" * 50)
        
        result = system.calculate_posterior_probability(
            condition_name="Myocardial Infarction",
            test_results=scenario['test_results'],
            patient_demographics=scenario['demographics']
        )
        
        print(f"Prior probability: {result['prior_probability']:.3f}")
        print(f"Posterior probability: {result['posterior_probability']:.3f}")
        print(f"Likelihood ratio: {result['likelihood_ratio']:.2f}")
        print(f"Clinical interpretation: {result['clinical_interpretation']}")
        
        # Show uncertainty measures
        uncertainty = result['uncertainty_measures']
        print(f"Uncertainty category: {uncertainty['uncertainty_category']}")
        print(f"Information gain: {uncertainty['information_gain']:.3f} bits")
        print(f"Confidence: {uncertainty['confidence']:.3f}")
        
        # Show test contributions
        print("\nTest contributions:")
        for test_name, contribution in result['test_contributions'].items():
            print(f"  {test_name}: {contribution:+.3f}")
        
        print("\n")
    
    # Generate diagnostic report
    print("=== Diagnostic System Report ===")
    report = system.generate_diagnostic_report("Myocardial Infarction", include_validation=False)
    
    print(f"Condition: {report['condition_name']} ({report['icd_10_code']})")
    print(f"Population prevalence: {report['prevalence']:.3f}")
    print(f"Severity score: {report['severity_score']}/5")
    
    print(f"\nAvailable tests: {len(report['available_tests'])}")
    for test_name, test_info in report['available_tests'].items():
        print(f"  {test_name}: Sens={test_info['sensitivity']:.2f}, "
              f"Spec={test_info['specificity']:.2f}, "
              f"Cost=${test_info['cost']:.0f}")
    
    if 'diagnostic_statistics' in report:
        stats = report['diagnostic_statistics']
        print(f"\nDiagnostic statistics:")
        print(f"  Total diagnoses: {stats['total_diagnoses']}")
        print(f"  Average posterior probability: {stats['average_posterior_probability']:.3f}")
        print(f"  High confidence diagnoses: {stats['high_confidence_diagnoses']}")


if __name__ == "__main__":
    demonstrate_bayesian_diagnosis()
```

### 2.2.2 Advanced Statistical Inference Methods

Beyond basic Bayesian inference, healthcare AI applications often require more sophisticated statistical methods to handle complex data structures and clinical scenarios. These advanced methods include hierarchical modeling, survival analysis, and causal inference techniques that are essential for developing clinically meaningful AI systems.

#### Hierarchical Bayesian Models

Hierarchical Bayesian models are particularly valuable in healthcare applications where data exhibits natural grouping structures, such as patients within hospitals, repeated measurements within patients, or treatments within therapeutic classes. These models allow for the sharing of information across groups while accounting for group-specific variations.

The mathematical framework for hierarchical models can be expressed as:

**Level 1 (Individual level):**
$$

y_{ij} | \theta_j, \sigma^2 \sim N(\theta_j, \sigma^2)

$$

**Level 2 (Group level):**
$$

\theta_j | \mu, \tau^2 \sim N(\mu, \tau^2)

$$

**Level 3 (Population level):**
$$

\mu \sim N(\mu_0, \sigma_0^2), \quad \tau^2 \sim \mathrm{\1}(\alpha, \beta)

$$

This hierarchical structure enables the model to learn from both individual patient data and population-level patterns, leading to more robust and generalizable predictions.

#### Survival Analysis and Time-to-Event Modeling

Survival analysis provides mathematical frameworks for analyzing time-to-event data, which is ubiquitous in healthcare applications. The fundamental concepts include the survival function, hazard function, and cumulative hazard function:

**Survival Function:**
$$

S(t) = P(T > t) = 1 - F(t)

$$

**Hazard Function:**
$$

h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t | T \geq t)}{\Delta t}

$$

**Cumulative Hazard Function:**
$$

H(t) = \int_0^t h(u) du = -\log S(t)

$$

The Cox proportional hazards model is particularly important for healthcare AI:

$$

h(t|x) = h_0(t) \exp(\beta^T x)

$$

Where $h_0(t)$ is the baseline hazard and $\beta^T x$ represents the linear combination of covariates.

## 2.3 Linear Algebra for Healthcare Data

Linear algebra provides the mathematical foundation for representing and manipulating healthcare data in high-dimensional spaces. The application of linear algebraic methods to healthcare AI enables efficient computation, dimensionality reduction, and the extraction of meaningful patterns from complex medical datasets.

### 2.3.1 Matrix Operations in Clinical Data Analysis

Healthcare data is naturally represented as matrices, where rows correspond to patients and columns represent clinical variables such as laboratory values, vital signs, diagnostic codes, and treatment histories. The mathematical operations on these matrices enable sophisticated analyses that would be computationally intractable using scalar arithmetic.

The fundamental matrix operations in healthcare AI include:

**Matrix Multiplication for Feature Transformation:**
$$

\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}

$$

Where $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the patient data matrix, $\mathbf{W} \in \mathbb{R}^{p \times k}$ is the weight matrix, $\mathbf{b} \in \mathbb{R}^k$ is the bias vector, and $\mathbf{Y} \in \mathbb{R}^{n \times k}$ is the transformed feature matrix.

**Eigenvalue Decomposition for Principal Component Analysis:**
$$

\mathbf{C} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T

$$

Where $\mathbf{C}$ is the covariance matrix of clinical variables, $\mathbf{Q}$ contains the eigenvectors (principal components), and $\mathbf{\Lambda}$ is the diagonal matrix of eigenvalues.

**Singular Value Decomposition for Matrix Factorization:**
$$

\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T

$$

This decomposition is particularly useful for handling missing data and identifying latent clinical phenotypes.

The following implementation demonstrates advanced linear algebra operations for healthcare data analysis:

```python
"""
Advanced Linear Algebra Operations for Healthcare Data Analysis

This module implements sophisticated linear algebraic methods specifically
designed for healthcare AI applications, including dimensionality reduction,
missing data imputation, and clinical phenotype discovery.

Author: Sanjay Basu MD PhD
License: MIT
"""

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClinicalDataMatrix:
    """
    Represents a clinical data matrix with comprehensive metadata.
    
    This class encapsulates clinical data along with variable information,
    patient metadata, and data quality metrics essential for healthcare AI.
    """
    data: np.ndarray
    patient_ids: List[str]
    variable_names: List[str]
    variable_types: Dict[str, str]  # 'continuous', 'categorical', 'binary'
    missing_data_pattern: Optional[np.ndarray] = None
    data_quality_score: float = 0.0
    collection_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate data matrix and calculate quality metrics."""
        self._validate_dimensions()
        self._calculate_missing_pattern()
        self._calculate_quality_score()
    
    def _validate_dimensions(self):
        """Validate matrix dimensions and metadata consistency."""
        n_patients, n_variables = self.data.shape
        
        if len(self.patient_ids) != n_patients:
            raise ValueError(f"Patient ID count ({len(self.patient_ids)}) "
                           f"doesn't match data rows ({n_patients})")
        
        if len(self.variable_names) != n_variables:
            raise ValueError(f"Variable name count ({len(self.variable_names)}) "
                           f"doesn't match data columns ({n_variables})")
    
    def _calculate_missing_pattern(self):
        """Calculate missing data pattern matrix."""
        self.missing_data_pattern = np.isnan(self.data).astype(int)
    
    def _calculate_quality_score(self):
        """Calculate overall data quality score."""
        if self.missing_data_pattern is not None:
            completeness = 1 - np.mean(self.missing_data_pattern)
            self.data_quality_score = completeness
        else:
            self.data_quality_score = 1.0
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get matrix dimensions."""
        return self.data.shape
    
    @property
    def missing_percentage(self) -> float:
        """Get percentage of missing values."""
        if self.missing_data_pattern is not None:
            return np.mean(self.missing_data_pattern) * 100
        return 0.0
    
    def get_variable_summary(self) -> pd.DataFrame:
        """Get summary statistics for all variables."""
        summary_data = []
        
        for i, var_name in enumerate(self.variable_names):
            var_data = self.data[:, i]
            var_type = self.variable_types.get(var_name, 'unknown')
            
            if var_type == 'continuous':
                summary = {
                    'variable': var_name,
                    'type': var_type,
                    'mean': np.nanmean(var_data),
                    'std': np.nanstd(var_data),
                    'min': np.nanmin(var_data),
                    'max': np.nanmax(var_data),
                    'missing_pct': np.mean(np.isnan(var_data)) * 100
                }
            else:
                unique_values = np.unique(var_data[~np.isnan(var_data)])
                summary = {
                    'variable': var_name,
                    'type': var_type,
                    'unique_values': len(unique_values),
                    'most_common': unique_values<sup>0</sup> if len(unique_values) > 0 else None,
                    'missing_pct': np.mean(np.isnan(var_data)) * 100
                }
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)

class AdvancedLinearAlgebra:
    """
    Advanced linear algebra operations for healthcare data analysis.
    
    This class implements sophisticated matrix operations, decompositions,
    and transformations specifically designed for clinical data analysis
    and healthcare AI applications.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize advanced linear algebra processor.
        
        Args:
            random_seed: Random seed for reproducible results
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Storage for computed decompositions
        self.pca_components: Optional[np.ndarray] = None
        self.svd_components: Optional[Dict[str, np.ndarray]] = None
        self.nmf_components: Optional[Dict[str, np.ndarray]] = None
        
        logger.info("Advanced linear algebra processor initialized")
    
    def robust_pca(self, 
                   clinical_data: ClinicalDataMatrix,
                   n_components: Optional[int] = None,
                   handle_missing: str = 'impute') -> Dict[str, Any]:
        """
        Perform robust Principal Component Analysis on clinical data.
        
        Args:
            clinical_data: ClinicalDataMatrix object
            n_components: Number of components to extract (None for all)
            handle_missing: How to handle missing data ('impute', 'exclude', 'iterative')
            
        Returns:
            Dictionary containing PCA results and clinical interpretation
        """
        try:
            # Prepare data
            X = self._prepare_data_for_pca(clinical_data, handle_missing)
            
            # Determine number of components
            if n_components is None:
                n_components = min(X.shape) - 1
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=n_components, random_state=self.random_seed)
            X_transformed = pca.fit_transform(X_scaled)
            
            # Store components
            self.pca_components = pca.components_
            
            # Calculate explained variance ratios
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Identify clinically meaningful components
            component_interpretations = self._interpret_pca_components(
                pca.components_, clinical_data.variable_names, clinical_data.variable_types
            )
            
            # Calculate component stability (using bootstrap if data is large enough)
            if X.shape<sup>0</sup> > 100:
                stability_scores = self._calculate_component_stability(
                    X_scaled, n_components, n_bootstrap=100
                )
            else:
                stability_scores = np.ones(n_components)
            
            results = {
                'transformed_data': X_transformed,
                'components': pca.components_,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance_ratio': cumulative_variance,
                'eigenvalues': pca.explained_variance_,
                'component_interpretations': component_interpretations,
                'stability_scores': stability_scores,
                'scaler': scaler,
                'pca_model': pca,
                'n_components_95_variance': np.argmax(cumulative_variance >= 0.95) + 1,
                'clinical_phenotypes': self._identify_clinical_phenotypes(
                    X_transformed, clinical_data.patient_ids
                )
            }
            
            logger.info(f"PCA completed: {n_components} components explain "
                       f"{cumulative_variance[-1]:.1%} of variance")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in robust PCA: {e}")
            raise
    
    def matrix_completion_svd(self, 
                             clinical_data: ClinicalDataMatrix,
                             rank: Optional[int] = None,
                             max_iterations: int = 100,
                             tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Perform matrix completion using iterative SVD for missing clinical data.
        
        Args:
            clinical_data: ClinicalDataMatrix with missing values
            rank: Target rank for low-rank approximation
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary containing completed matrix and decomposition results
        """
        try:
            X = clinical_data.data.copy()
            missing_mask = np.isnan(X)
            
            if not np.any(missing_mask):
                logger.warning("No missing data found, performing standard SVD")
                return self._standard_svd(X, rank)
            
            # Initialize missing values with column means
            col_means = np.nanmean(X, axis=0)
            for j in range(X.shape<sup>1</sup>):
                X[missing_mask[:, j], j] = col_means[j]
            
            # Determine rank
            if rank is None:
                rank = min(X.shape) // 2
            
            # Iterative SVD completion
            prev_X = X.copy()
            
            for iteration in range(max_iterations):
                # SVD decomposition
                U, s, Vt = la.svd(X, full_matrices=False)
                
                # Truncate to desired rank
                U_r = U[:, :rank]
                s_r = s[:rank]
                Vt_r = Vt[:rank, :]
                
                # Reconstruct matrix
                X_reconstructed = U_r @ np.diag(s_r) @ Vt_r
                
                # Update only missing entries
                X[missing_mask] = X_reconstructed[missing_mask]
                
                # Check convergence
                change = np.linalg.norm(X - prev_X, 'fro')
                if change < tolerance:
                    logger.info(f"Matrix completion converged after {iteration + 1} iterations")
                    break
                
                prev_X = X.copy()
            
            # Final SVD for analysis
            U_final, s_final, Vt_final = la.svd(X, full_matrices=False)
            
            # Calculate completion quality metrics
            completion_metrics = self._calculate_completion_metrics(
                clinical_data.data, X, missing_mask
            )
            
            # Store SVD components
            self.svd_components = {
                'U': U_final[:, :rank],
                'sigma': s_final[:rank],
                'Vt': Vt_final[:rank, :]
            }
            
            results = {
                'completed_matrix': X,
                'U': U_final[:, :rank],
                'singular_values': s_final[:rank],
                'Vt': Vt_final[:rank, :],
                'rank': rank,
                'iterations': iteration + 1,
                'completion_metrics': completion_metrics,
                'missing_data_percentage': np.mean(missing_mask) * 100,
                'latent_factors': self._interpret_svd_factors(
                    U_final[:, :rank], Vt_final[:rank, :], 
                    clinical_data.patient_ids, clinical_data.variable_names
                )
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in matrix completion SVD: {e}")
            raise
    
    def non_negative_matrix_factorization(self, 
                                        clinical_data: ClinicalDataMatrix,
                                        n_components: int,
                                        max_iterations: int = 200) -> Dict[str, Any]:
        """
        Perform Non-negative Matrix Factorization for clinical phenotype discovery.
        
        NMF is particularly useful for identifying clinical phenotypes because
        it produces interpretable, parts-based representations.
        
        Args:
            clinical_data: ClinicalDataMatrix object
            n_components: Number of components (phenotypes) to extract
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary containing NMF results and phenotype interpretations
        """
        try:
            X = clinical_data.data.copy()
            
            # Handle missing data
            if np.any(np.isnan(X)):
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Ensure non-negativity (required for NMF)
            X_min = np.min(X)
            if X_min < 0:
                X = X - X_min  # Shift to make all values non-negative
                logger.info(f"Shifted data by {-X_min} to ensure non-negativity")
            
            # Perform NMF
            nmf = NMF(n_components=n_components, 
                     max_iter=max_iterations,
                     random_state=self.random_seed,
                     init='nndsvd')  # Better initialization
            
            W = nmf.fit_transform(X)  # Patient loadings
            H = nmf.components_       # Feature loadings
            
            # Store NMF components
            self.nmf_components = {'W': W, 'H': H}
            
            # Interpret clinical phenotypes
            phenotype_interpretations = self._interpret_nmf_phenotypes(
                H, clinical_data.variable_names, clinical_data.variable_types
            )
            
            # Assign patients to dominant phenotypes
            patient_phenotypes = self._assign_patient_phenotypes(
                W, clinical_data.patient_ids
            )
            
            # Calculate reconstruction quality
            X_reconstructed = W @ H
            reconstruction_error = np.linalg.norm(X - X_reconstructed, 'fro')
            relative_error = reconstruction_error / np.linalg.norm(X, 'fro')
            
            results = {
                'patient_loadings': W,
                'feature_loadings': H,
                'reconstructed_matrix': X_reconstructed,
                'reconstruction_error': reconstruction_error,
                'relative_error': relative_error,
                'phenotype_interpretations': phenotype_interpretations,
                'patient_phenotypes': patient_phenotypes,
                'nmf_model': nmf,
                'n_components': n_components,
                'phenotype_prevalence': self._calculate_phenotype_prevalence(W)
            }
            
            logger.info(f"NMF completed: {n_components} phenotypes identified "
                       f"with {relative_error:.3f} relative reconstruction error")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in NMF: {e}")
            raise
    
    def tensor_decomposition_clinical(self, 
                                    tensor_data: np.ndarray,
                                    rank: int,
                                    max_iterations: int = 100) -> Dict[str, Any]:
        """
        Perform tensor decomposition for multi-way clinical data analysis.
        
        This method is useful for analyzing data with multiple modes such as
        patients × variables × time or patients × variables × hospitals.
        
        Args:
            tensor_data: 3D numpy array (patients × variables × time/context)
            rank: Rank of the decomposition
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary containing tensor decomposition results
        """
        try:
            if len(tensor_data.shape) != 3:
                raise ValueError("Tensor data must be 3-dimensional")
            
            n_patients, n_variables, n_contexts = tensor_data.shape
            
            # Initialize factor matrices randomly
            A = np.random.rand(n_patients, rank)
            B = np.random.rand(n_variables, rank)
            C = np.random.rand(n_contexts, rank)
            
            # Alternating least squares algorithm
            for iteration in range(max_iterations):
                # Update A (patients)
                for i in range(n_patients):
                    X_i = tensor_data[i, :, :]  # variables × contexts
                    khatri_rao_BC = self._khatri_rao_product(B, C)
                    A[i, :] = la.lstsq(khatri_rao_BC, X_i.flatten())<sup>0</sup>
                
                # Update B (variables)
                for j in range(n_variables):
                    X_j = tensor_data[:, j, :]  # patients × contexts
                    khatri_rao_AC = self._khatri_rao_product(A, C)
                    B[j, :] = la.lstsq(khatri_rao_AC, X_j.flatten())<sup>0</sup>
                
                # Update C (contexts)
                for k in range(n_contexts):
                    X_k = tensor_data[:, :, k]  # patients × variables
                    khatri_rao_AB = self._khatri_rao_product(A, B)
                    C[k, :] = la.lstsq(khatri_rao_AB, X_k.flatten())<sup>0</sup>
                
                # Check convergence (simplified)
                if iteration > 0 and iteration % 10 == 0:
                    reconstructed = self._reconstruct_tensor(A, B, C)
                    error = np.linalg.norm(tensor_data - reconstructed)
                    logger.info(f"Iteration {iteration}: Reconstruction error = {error:.6f}")
            
            # Final reconstruction
            reconstructed_tensor = self._reconstruct_tensor(A, B, C)
            reconstruction_error = np.linalg.norm(tensor_data - reconstructed_tensor)
            
            results = {
                'patient_factors': A,
                'variable_factors': B,
                'context_factors': C,
                'reconstructed_tensor': reconstructed_tensor,
                'reconstruction_error': reconstruction_error,
                'rank': rank,
                'iterations': max_iterations
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in tensor decomposition: {e}")
            raise
    
    def _prepare_data_for_pca(self, 
                             clinical_data: ClinicalDataMatrix,
                             handle_missing: str) -> np.ndarray:
        """Prepare clinical data for PCA analysis."""
        X = clinical_data.data.copy()
        
        if handle_missing == 'impute':
            # Use KNN imputation for better results with clinical data
            imputer = KNNImputer(n_neighbors=5)
            X = imputer.fit_transform(X)
        elif handle_missing == 'exclude':
            # Remove rows with any missing values
            complete_rows = ~np.any(np.isnan(X), axis=1)
            X = X[complete_rows, :]
            logger.info(f"Excluded {np.sum(~complete_rows)} patients with missing data")
        elif handle_missing == 'iterative':
            # Use iterative imputation
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(random_state=self.random_seed)
            X = imputer.fit_transform(X)
        
        return X
    
    def _interpret_pca_components(self, 
                                 components: np.ndarray,
                                 variable_names: List[str],
                                 variable_types: Dict[str, str]) -> List[Dict[str, Any]]:
        """Interpret PCA components in clinical context."""
        interpretations = []
        
        for i, component in enumerate(components):
            # Find variables with highest absolute loadings
            abs_loadings = np.abs(component)
            top_indices = np.argsort(abs_loadings)[-10:][::-1]  # Top 10
            
            top_variables = []
            for idx in top_indices:
                if abs_loadings[idx] > 0.1:  # Threshold for meaningful loading
                    top_variables.append({
                        'variable': variable_names[idx],
                        'loading': component[idx],
                        'type': variable_types.get(variable_names[idx], 'unknown')
                    })
            
            # Generate clinical interpretation
            interpretation = self._generate_component_interpretation(top_variables)
            
            interpretations.append({
                'component_number': i + 1,
                'top_variables': top_variables,
                'clinical_interpretation': interpretation,
                'variance_explained': None  # Will be filled by calling function
            })
        
        return interpretations
    
    def _generate_component_interpretation(self, top_variables: List[Dict[str, Any]]) -> str:
        """Generate clinical interpretation for a PCA component."""
        if not top_variables:
            return "No significant variable loadings"
        
        # Group variables by clinical domain
        lab_vars = [v for v in top_variables if 'lab' in v['variable'].lower()]
        vital_vars = [v for v in top_variables if any(term in v['variable'].lower() 
                     for term in ['bp', 'hr', 'temp', 'resp'])]
        
        interpretation_parts = []
        
        if lab_vars:
            interpretation_parts.append(f"Laboratory profile ({len(lab_vars)} variables)")
        if vital_vars:
            interpretation_parts.append(f"Vital signs pattern ({len(vital_vars)} variables)")
        
        if not interpretation_parts:
            interpretation_parts.append("Mixed clinical variables")
        
        return " + ".join(interpretation_parts)
    
    def _calculate_component_stability(self, 
                                     X: np.ndarray,
                                     n_components: int,
                                     n_bootstrap: int = 100) -> np.ndarray:
        """Calculate stability of PCA components using bootstrap."""
        n_samples = X.shape<sup>0</sup>
        component_similarities = []
        
        # Original PCA
        pca_original = PCA(n_components=n_components, random_state=self.random_seed)
        pca_original.fit(X)
        original_components = pca_original.components_
        
        # Bootstrap iterations
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices, :]
            
            # PCA on bootstrap sample
            pca_bootstrap = PCA(n_components=n_components, random_state=self.random_seed)
            pca_bootstrap.fit(X_bootstrap)
            bootstrap_components = pca_bootstrap.components_
            
            # Calculate component similarities (absolute correlation)
            similarities = []
            for i in range(n_components):
                max_similarity = 0
                for j in range(n_components):
                    similarity = abs(np.corrcoef(original_components[i], 
                                               bootstrap_components[j])[0, 1])
                    max_similarity = max(max_similarity, similarity)
                similarities.append(max_similarity)
            
            component_similarities.append(similarities)
        
        # Average similarities across bootstrap iterations
        stability_scores = np.mean(component_similarities, axis=0)
        
        return stability_scores
    
    def _identify_clinical_phenotypes(self, 
                                    X_transformed: np.ndarray,
                                    patient_ids: List[str]) -> Dict[str, Any]:
        """Identify clinical phenotypes from PCA-transformed data."""
        from sklearn.cluster import KMeans
        
        # Determine optimal number of clusters using elbow method
        max_clusters = min(10, X_transformed.shape<sup>0</sup> // 10)
        inertias = []
        
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_seed)
            kmeans.fit(X_transformed)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        optimal_k = 3  # Default, could implement more sophisticated elbow detection
        
        # Perform final clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=self.random_seed)
        cluster_labels = kmeans_final.fit_predict(X_transformed)
        
        # Analyze phenotypes
        phenotypes = {}
        for k in range(optimal_k):
            cluster_mask = cluster_labels == k
            phenotypes[f'phenotype_{k+1}'] = {
                'patient_count': np.sum(cluster_mask),
                'patient_ids': [patient_ids[i] for i in np.where(cluster_mask)<sup>0</sup>],
                'centroid': kmeans_final.cluster_centers_[k],
                'prevalence': np.mean(cluster_mask)
            }
        
        return {
            'phenotypes': phenotypes,
            'cluster_labels': cluster_labels,
            'optimal_k': optimal_k,
            'clustering_model': kmeans_final
        }
    
    def _standard_svd(self, X: np.ndarray, rank: Optional[int]) -> Dict[str, Any]:
        """Perform standard SVD decomposition."""
        U, s, Vt = la.svd(X, full_matrices=False)
        
        if rank is not None:
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
        
        return {
            'U': U,
            'singular_values': s,
            'Vt': Vt,
            'rank': len(s)
        }
    
    def _calculate_completion_metrics(self, 
                                    original: np.ndarray,
                                    completed: np.ndarray,
                                    missing_mask: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for matrix completion quality."""
        # Only evaluate on originally missing entries
        if not np.any(missing_mask):
            return {'rmse': 0.0, 'mae': 0.0, 'completion_rate': 1.0}
        
        # For missing entries, we can't calculate true error
        # Instead, calculate consistency metrics
        completion_rate = 1.0  # All missing values were filled
        
        # Calculate overall matrix properties
        frobenius_norm_ratio = (np.linalg.norm(completed, 'fro') / 
                               np.linalg.norm(original[~missing_mask], 'fro'))
        
        return {
            'completion_rate': completion_rate,
            'frobenius_norm_ratio': frobenius_norm_ratio,
            'missing_percentage': np.mean(missing_mask) * 100
        }
    
    def _interpret_svd_factors(self, 
                              U: np.ndarray,
                              Vt: np.ndarray,
                              patient_ids: List[str],
                              variable_names: List[str]) -> Dict[str, Any]:
        """Interpret SVD factors in clinical context."""
        n_factors = U.shape<sup>1</sup>
        
        factor_interpretations = []
        for i in range(n_factors):
            # Patient factor interpretation
            patient_loadings = U[:, i]
            top_patient_indices = np.argsort(np.abs(patient_loadings))[-5:][::-1]
            
            # Variable factor interpretation
            variable_loadings = Vt[i, :]
            top_variable_indices = np.argsort(np.abs(variable_loadings))[-10:][::-1]
            
            factor_interpretations.append({
                'factor_number': i + 1,
                'top_patients': [patient_ids[idx] for idx in top_patient_indices],
                'top_variables': [variable_names[idx] for idx in top_variable_indices],
                'patient_loading_range': (np.min(patient_loadings), np.max(patient_loadings)),
                'variable_loading_range': (np.min(variable_loadings), np.max(variable_loadings))
            })
        
        return {
            'factor_interpretations': factor_interpretations,
            'n_factors': n_factors
        }
    
    def _interpret_nmf_phenotypes(self, 
                                 H: np.ndarray,
                                 variable_names: List[str],
                                 variable_types: Dict[str, str]) -> List[Dict[str, Any]]:
        """Interpret NMF phenotypes based on feature loadings."""
        phenotype_interpretations = []
        
        for i in range(H.shape<sup>0</sup>):
            feature_loadings = H[i, :]
            
            # Find top contributing features
            top_indices = np.argsort(feature_loadings)[-10:][::-1]
            
            top_features = []
            for idx in top_indices:
                if feature_loadings[idx] > 0.1:  # Threshold for meaningful contribution
                    top_features.append({
                        'variable': variable_names[idx],
                        'loading': feature_loadings[idx],
                        'type': variable_types.get(variable_names[idx], 'unknown')
                    })
            
            # Generate phenotype description
            phenotype_description = self._generate_phenotype_description(top_features)
            
            phenotype_interpretations.append({
                'phenotype_number': i + 1,
                'top_features': top_features,
                'description': phenotype_description,
                'feature_diversity': len(set(f['type'] for f in top_features))
            })
        
        return phenotype_interpretations
    
    def _generate_phenotype_description(self, top_features: List[Dict[str, Any]]) -> str:
        """Generate clinical description for NMF phenotype."""
        if not top_features:
            return "Undefined phenotype"
        
        # Group features by type
        feature_types = {}
        for feature in top_features:
            ftype = feature['type']
            if ftype not in feature_types:
                feature_types[ftype] = []
            feature_types[ftype].append(feature['variable'])
        
        description_parts = []
        for ftype, variables in feature_types.items():
            if ftype == 'continuous':
                description_parts.append(f"Continuous variables ({len(variables)})")
            elif ftype == 'categorical':
                description_parts.append(f"Categorical features ({len(variables)})")
            elif ftype == 'binary':
                description_parts.append(f"Binary indicators ({len(variables)})")
        
        return " + ".join(description_parts) if description_parts else "Mixed phenotype"
    
    def _assign_patient_phenotypes(self, 
                                  W: np.ndarray,
                                  patient_ids: List[str]) -> Dict[str, Any]:
        """Assign patients to dominant phenotypes."""
        # Find dominant phenotype for each patient
        dominant_phenotypes = np.argmax(W, axis=1)
        
        # Calculate phenotype assignments
        phenotype_assignments = {}
        for i, patient_id in enumerate(patient_ids):
            phenotype_assignments[patient_id] = {
                'dominant_phenotype': int(dominant_phenotypes[i]) + 1,
                'phenotype_weights': W[i, :].tolist(),
                'confidence': np.max(W[i, :]) / np.sum(W[i, :])
            }
        
        return phenotype_assignments
    
    def _calculate_phenotype_prevalence(self, W: np.ndarray) -> Dict[str, float]:
        """Calculate prevalence of each phenotype."""
        dominant_phenotypes = np.argmax(W, axis=1)
        n_patients = W.shape<sup>0</sup>
        n_phenotypes = W.shape<sup>1</sup>
        
        prevalence = {}
        for i in range(n_phenotypes):
            count = np.sum(dominant_phenotypes == i)
            prevalence[f'phenotype_{i+1}'] = count / n_patients
        
        return prevalence
    
    def _khatri_rao_product(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute Khatri-Rao product of two matrices."""
        return np.kron(A, np.ones((1, B.shape<sup>1</sup>))) * np.kron(np.ones((1, A.shape<sup>1</sup>)), B)
    
    def _reconstruct_tensor(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Reconstruct tensor from factor matrices."""
        rank = A.shape<sup>1</sup>
        tensor_shape = (A.shape<sup>0</sup>, B.shape<sup>0</sup>, C.shape<sup>0</sup>)
        reconstructed = np.zeros(tensor_shape)
        
        for r in range(rank):
            reconstructed += np.outer(A[:, r], np.outer(B[:, r], C[:, r]).flatten()).reshape(tensor_shape)
        
        return reconstructed


# Demonstration functions
def create_example_clinical_data() -> ClinicalDataMatrix:
    """Create example clinical data matrix for demonstration."""
    np.random.seed(42)
    
    # Simulate clinical data
    n_patients = 200
    n_variables = 50
    
    # Generate patient IDs
    patient_ids = [f"PATIENT_{i:04d}" for i in range(n_patients)]
    
    # Generate variable names
    variable_names = []
    variable_types = {}
    
    # Laboratory values (continuous)
    lab_vars = ['glucose', 'creatinine', 'hemoglobin', 'wbc_count', 'platelet_count',
                'sodium', 'potassium', 'chloride', 'bun', 'alt', 'ast', 'bilirubin']
    for var in lab_vars:
        variable_names.append(f'lab_{var}')
        variable_types[f'lab_{var}'] = 'continuous'
    
    # Vital signs (continuous)
    vital_vars = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature', 'respiratory_rate']
    for var in vital_vars:
        variable_names.append(f'vital_{var}')
        variable_types[f'vital_{var}'] = 'continuous'
    
    # Demographics (mixed)
    demo_vars = ['age', 'bmi']
    for var in demo_vars:
        variable_names.append(f'demo_{var}')
        variable_types[f'demo_{var}'] = 'continuous'
    
    # Comorbidities (binary)
    comorbidity_vars = ['diabetes', 'hypertension', 'heart_disease', 'copd', 'ckd']
    for var in comorbidity_vars:
        variable_names.append(f'comorbid_{var}')
        variable_types[f'comorbid_{var}'] = 'binary'
    
    # Medications (binary)
    med_vars = ['ace_inhibitor', 'beta_blocker', 'statin', 'metformin', 'insulin']
    for var in med_vars:
        variable_names.append(f'med_{var}')
        variable_types[f'med_{var}'] = 'binary'
    
    # Additional variables to reach 50
    remaining = n_variables - len(variable_names)
    for i in range(remaining):
        var_name = f'additional_var_{i}'
        variable_names.append(var_name)
        variable_types[var_name] = 'continuous'
    
    # Generate correlated clinical data
    data = np.random.randn(n_patients, n_variables)
    
    # Add clinical correlations
    # Diabetes cluster
    diabetes_indices = [i for i, name in enumerate(variable_names) 
                       if any(term in name for term in ['glucose', 'diabetes', 'metformin'])]
    for i in diabetes_indices:
        for j in diabetes_indices:
            if i != j:
                data[:, i] += 0.3 * data[:, j]
    
    # Cardiovascular cluster
    cv_indices = [i for i, name in enumerate(variable_names) 
                 if any(term in name for term in ['bp', 'heart', 'ace_inhibitor', 'beta_blocker'])]
    for i in cv_indices:
        for j in cv_indices:
            if i != j:
                data[:, i] += 0.2 * data[:, j]
    
    # Add missing data pattern
    missing_prob = 0.1
    missing_mask = np.random.random((n_patients, n_variables)) < missing_prob
    data[missing_mask] = np.nan
    
    return ClinicalDataMatrix(
        data=data,
        patient_ids=patient_ids,
        variable_names=variable_names,
        variable_types=variable_types
    )


def demonstrate_advanced_linear_algebra():
    """Demonstrate advanced linear algebra operations on clinical data."""
    print("=== Advanced Linear Algebra for Healthcare Data ===\n")
    
    # Create example clinical data
    clinical_data = create_example_clinical_data()
    
    print(f"Clinical Data Matrix:")
    print(f"  Patients: {clinical_data.shape<sup>0</sup>}")
    print(f"  Variables: {clinical_data.shape<sup>1</sup>}")
    print(f"  Missing data: {clinical_data.missing_percentage:.1f}%")
    print(f"  Data quality score: {clinical_data.data_quality_score:.3f}")
    
    # Initialize linear algebra processor
    la_processor = AdvancedLinearAlgebra(random_seed=42)
    
    # 1. Robust PCA Analysis
    print(f"\n1. Robust PCA Analysis")
    print("-" * 40)
    
    pca_results = la_processor.robust_pca(
        clinical_data=clinical_data,
        n_components=10,
        handle_missing='impute'
    )
    
    print(f"PCA Results:")
    print(f"  Components extracted: {len(pca_results['explained_variance_ratio'])}")
    print(f"  Variance explained by first 5 components: {pca_results['cumulative_variance_ratio']<sup>4</sup>:.1%}")
    print(f"  Components needed for 95% variance: {pca_results['n_components_95_variance']}")
    
    # Show component interpretations
    print(f"\nTop 3 Component Interpretations:")
    for i, interp in enumerate(pca_results['component_interpretations'][:3]):
        print(f"  Component {interp['component_number']}: {interp['clinical_interpretation']}")
        print(f"    Stability score: {pca_results['stability_scores'][i]:.3f}")
    
    # 2. Matrix Completion with SVD
    print(f"\n2. Matrix Completion using SVD")
    print("-" * 40)
    
    svd_results = la_processor.matrix_completion_svd(
        clinical_data=clinical_data,
        rank=15,
        max_iterations=50
    )
    
    print(f"SVD Matrix Completion Results:")
    print(f"  Rank: {svd_results['rank']}")
    print(f"  Iterations: {svd_results['iterations']}")
    print(f"  Missing data: {svd_results['missing_data_percentage']:.1f}%")
    
    completion_metrics = svd_results['completion_metrics']
    print(f"  Completion rate: {completion_metrics['completion_rate']:.1%}")
    print(f"  Frobenius norm ratio: {completion_metrics['frobenius_norm_ratio']:.3f}")
    
    # 3. Non-negative Matrix Factorization
    print(f"\n3. Non-negative Matrix Factorization")
    print("-" * 40)
    
    # Prepare data for NMF (ensure non-negativity)
    nmf_results = la_processor.non_negative_matrix_factorization(
        clinical_data=clinical_data,
        n_components=5,
        max_iterations=200
    )
    
    print(f"NMF Results:")
    print(f"  Phenotypes identified: {nmf_results['n_components']}")
    print(f"  Reconstruction error: {nmf_results['relative_error']:.4f}")
    
    # Show phenotype prevalence
    print(f"\nPhenotype Prevalence:")
    for phenotype, prevalence in nmf_results['phenotype_prevalence'].items():
        print(f"  {phenotype}: {prevalence:.1%}")
    
    # Show phenotype interpretations
    print(f"\nPhenotype Interpretations:")
    for interp in nmf_results['phenotype_interpretations'][:3]:
        print(f"  Phenotype {interp['phenotype_number']}: {interp['description']}")
        print(f"    Feature diversity: {interp['feature_diversity']} types")
    
    # 4. Clinical Phenotype Analysis
    print(f"\n4. Clinical Phenotype Analysis")
    print("-" * 40)
    
    phenotypes = pca_results['clinical_phenotypes']
    print(f"PCA-based Phenotypes:")
    print(f"  Optimal clusters: {phenotypes['optimal_k']}")
    
    for phenotype_name, phenotype_info in phenotypes['phenotypes'].items():
        print(f"  {phenotype_name}: {phenotype_info['patient_count']} patients "
              f"({phenotype_info['prevalence']:.1%})")
    
    # 5. Data Quality Assessment
    print(f"\n5. Data Quality Assessment")
    print("-" * 40)
    
    variable_summary = clinical_data.get_variable_summary()
    
    # Show variables with highest missing rates
    if 'missing_pct' in variable_summary.columns:
        high_missing = variable_summary.nlargest(5, 'missing_pct')
        print(f"Variables with highest missing rates:")
        for _, row in high_missing.iterrows():
            print(f"  {row['variable']}: {row['missing_pct']:.1f}% missing")
    
    # Show continuous variable statistics
    continuous_vars = variable_summary[variable_summary['type'] == 'continuous']
    if not continuous_vars.empty and 'mean' in continuous_vars.columns:
        print(f"\nContinuous variables summary:")
        print(f"  Mean range: [{continuous_vars['mean'].min():.2f}, {continuous_vars['mean'].max():.2f}]")
        print(f"  Std range: [{continuous_vars['std'].min():.2f}, {continuous_vars['std'].max():.2f}]")


if __name__ == "__main__":
    demonstrate_advanced_linear_algebra()
```

## 2.4 Optimization Theory for Model Training

Optimization theory provides the mathematical framework for training AI models by finding parameter values that minimize prediction errors while maintaining model generalizability. In healthcare applications, optimization must balance multiple objectives including predictive accuracy, interpretability, clinical utility, and fairness across different patient populations.

### 2.4.1 Gradient-Based Optimization Methods

The most commonly used optimization methods in healthcare AI are gradient-based approaches that iteratively update model parameters in the direction of steepest descent of the loss function. The fundamental update rule can be expressed as:

$$

\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t)

$$

Where $\theta$ represents the model parameters, $\alpha$ is the learning rate, and $L(\theta)$ is the loss function.

For healthcare applications, the loss function often incorporates multiple clinical considerations:

$$

L(\theta) = L_{prediction}(\theta) + \lambda L_{regularization}(\theta) + \gamma L_{fairness}(\theta) + \delta L_{interpretability}(\theta)

$$

This multi-objective formulation ensures that models not only achieve high predictive accuracy but also maintain fairness across different patient populations, avoid overfitting to training data, and provide interpretable results that can be validated by clinical experts.

### 2.4.2 Constrained Optimization for Clinical Requirements

Healthcare AI systems often must satisfy hard constraints related to clinical safety, regulatory requirements, and operational considerations. These constraints can be incorporated into the optimization problem using Lagrangian methods:

$$

\mathcal{L}(\theta, \lambda, \mu) = L(\theta) + \sum_i \lambda_i g_i(\theta) + \sum_j \mu_j h_j(\theta)

$$

Where $g_i(\theta) \leq 0$ are inequality constraints and $h_j(\theta) = 0$ are equality constraints.

Common clinical constraints include:
- **Safety constraints**: Ensuring that model predictions do not recommend harmful treatments
- **Fairness constraints**: Maintaining equitable performance across different demographic groups
- **Interpretability constraints**: Limiting model complexity to ensure clinical interpretability
- **Resource constraints**: Considering computational and financial costs of model deployment

## 2.5 Information Theory and Entropy in Healthcare

Information theory provides mathematical tools for quantifying uncertainty, measuring information content, and optimizing communication systems. In healthcare AI, information-theoretic measures are used to evaluate model uncertainty, select informative features, and design efficient diagnostic protocols.

### 2.5.1 Entropy and Uncertainty Quantification

The entropy of a clinical variable measures the uncertainty associated with that variable:

$$

H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)

$$

High entropy indicates high uncertainty, while low entropy suggests more predictable outcomes. This measure is particularly useful for:

- **Feature Selection**: Variables with high entropy may provide more information for prediction
- **Uncertainty Quantification**: Model predictions with high entropy require additional clinical validation
- **Diagnostic Test Ordering**: Tests that maximize information gain should be prioritized

### 2.5.2 Mutual Information for Feature Relationships

Mutual information quantifies the amount of information that one variable provides about another:

$$

I(X;Y) = \sum_{x,y} p(x,y) \log_2 \frac{p(x,y)}{p(x)p(y)}

$$

This measure is essential for understanding relationships between clinical variables and identifying redundant measurements.

### 2.5.3 Information-Theoretic Model Selection

Information theory provides principled approaches to model selection through criteria such as the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC):

$$

AIC = 2k - 2\ln(L)

$$
$$

BIC = k\ln(n) - 2\ln(L)

$$

Where $k$ is the number of parameters, $n$ is the sample size, and $L$ is the likelihood.

## 2.6 Causal Inference and Graphical Models

Causal inference provides mathematical frameworks for understanding cause-and-effect relationships in healthcare data, enabling the development of AI systems that can support treatment decisions and policy interventions.

### 2.6.1 Directed Acyclic Graphs (DAGs)

Causal relationships in healthcare can be represented using directed acyclic graphs, where nodes represent variables and edges represent causal relationships. The mathematical framework of DAGs enables:

- **Confounding Control**: Identification of variables that must be controlled to estimate causal effects
- **Mediation Analysis**: Understanding how treatments affect outcomes through intermediate variables
- **Collider Bias Prevention**: Avoiding spurious associations introduced by conditioning on colliders

### 2.6.2 Causal Effect Estimation

The fundamental problem of causal inference is estimating the effect of an intervention on an outcome. This can be formalized using potential outcomes:

$$

\tau = E[Y(1) - Y(0)]

$$

Where $Y(1)$ is the potential outcome under treatment and $Y(0)$ is the potential outcome under control.

Various methods exist for causal effect estimation:

**Propensity Score Methods**: Balance treatment and control groups based on observed covariates
**Instrumental Variables**: Use variables that affect treatment assignment but not outcomes directly
**Regression Discontinuity**: Exploit arbitrary cutoffs in treatment assignment
**Difference-in-Differences**: Compare changes over time between treatment and control groups

## 2.7 Time Series Analysis for Longitudinal Healthcare Data

Healthcare data is inherently temporal, with patient conditions evolving over time and treatments having delayed effects. Time series analysis provides mathematical tools for modeling these temporal dependencies and making predictions about future health states.

### 2.7.1 State Space Models

State space models provide a flexible framework for modeling temporal healthcare data:

**State Equation:**
$$

x_t = F_t x_{t-1} + B_t u_t + w_t

$$

**Observation Equation:**
$$

y_t = H_t x_t + v_t

$$

Where $x_t$ represents the hidden health state, $y_t$ represents observed measurements, $u_t$ represents interventions, and $w_t$, $v_t$ are noise terms.

### 2.7.2 Survival Analysis and Hazard Modeling

Survival analysis extends time series methods to handle censored data and time-to-event outcomes:

**Kaplan-Meier Estimator:**
$$

\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)

$$

**Cox Proportional Hazards Model:**
$$

h(t|x) = h_0(t) \exp(\beta^T x)

$$

These models are essential for analyzing patient outcomes, treatment effectiveness, and disease progression.

## 2.8 Advanced Statistical Methods

Healthcare AI applications often require specialized statistical methods that go beyond standard machine learning approaches. These methods address the unique challenges of medical data including missing values, measurement error, and complex dependency structures.

### 2.8.1 Multiple Imputation for Missing Data

Missing data is ubiquitous in healthcare datasets. Multiple imputation provides a principled approach to handling missing values:

1. **Imputation**: Create multiple complete datasets by imputing missing values
2. **Analysis**: Perform the desired analysis on each complete dataset
3. **Pooling**: Combine results using Rubin's rules

The pooled estimate is:
$$

\bar{Q} = \frac{1}{m} \sum_{i=1}^m Q_i

$$

The total variance is:
$$

T = \bar{U} + \left(1 + \frac{1}{m}\right)B

$$

Where $\bar{U}$ is the within-imputation variance and $B$ is the between-imputation variance.

### 2.8.2 Measurement Error Models

Clinical measurements often contain error, which can bias statistical analyses. Measurement error models account for this uncertainty:

**Classical Measurement Error:**
$$

X^* = X + U

$$

Where $X^*$ is the observed value, $X$ is the true value, and $U$ is the measurement error.

**Berkson Measurement Error:**
$$

X = X^* + U

$$

These models require specialized estimation techniques such as regression calibration or SIMEX (Simulation-Extrapolation).

## 2.9 Computational Considerations

The mathematical methods described in this chapter must be implemented efficiently to handle large healthcare datasets. Key computational considerations include:

### 2.9.1 Numerical Stability

Healthcare data often exhibits extreme values and high dynamic ranges, requiring careful attention to numerical stability:

- **Log-space arithmetic**: Prevent underflow in probability calculations
- **Condition number monitoring**: Detect ill-conditioned matrices
- **Regularization**: Improve numerical stability of optimization problems

### 2.9.2 Scalability

Healthcare datasets can be extremely large, requiring scalable algorithms:

- **Stochastic optimization**: Use mini-batches for large datasets
- **Distributed computing**: Parallelize computations across multiple processors
- **Approximation methods**: Use approximations when exact solutions are computationally intractable

## 2.10 Clinical Validation of Mathematical Models

Mathematical models in healthcare must undergo rigorous clinical validation to ensure safety and effectiveness. This validation process includes:

### 2.10.1 Statistical Validation

- **Cross-validation**: Assess model generalizability
- **Bootstrap confidence intervals**: Quantify parameter uncertainty
- **Goodness-of-fit tests**: Verify model assumptions

### 2.10.2 Clinical Validation

- **Expert review**: Clinical experts evaluate model outputs
- **Prospective studies**: Test models in real clinical settings
- **Outcome validation**: Verify that model use improves patient outcomes

## Bibliography and References

1. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. [Comprehensive coverage of statistical learning methods with mathematical rigor]

2. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. [Foundational text on probabilistic approaches to machine learning]

3. **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B.** (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Authoritative treatment of Bayesian methods]

4. **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. [Seminal work on causal inference and graphical models]

5. **Hernán, M. A., & Robins, J. M.** (2020). *Causal Inference: What If*. CRC Press. [Modern approach to causal inference with healthcare applications]

6. **Kalbfleisch, J. D., & Prentice, R. L.** (2002). *The Statistical Analysis of Failure Time Data* (2nd ed.). Wiley. [Comprehensive treatment of survival analysis methods]

7. **Little, R. J. A., & Rubin, D. B.** (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley. [Authoritative guide to handling missing data]

8. **Strang, G.** (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. [Clear exposition of linear algebra fundamentals]

9. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press. [Comprehensive treatment of optimization theory]

10. **Cover, T. M., & Thomas, J. A.** (2006). *Elements of Information Theory* (2nd ed.). Wiley. [Foundational text on information theory]

### Key Healthcare AI Papers

11. **Rajkomar, A., Dean, J., & Kohane, I.** (2019). Machine learning in medicine. *New England Journal of Medicine*, 380(14), 1347-1358. [Overview of machine learning applications in healthcare]

12. **Topol, E. J.** (2019). High-performance medicine: the convergence of human and artificial intelligence. *Nature Medicine*, 25(1), 44-56. [Vision for AI-augmented healthcare]

13. **Ghassemi, M., Naumann, T., Schulam, P., Beam, A. L., Chen, I. Y., & Ranganath, R.** (2020). A review of challenges and opportunities in machine learning for health. *AMIA Summits on Translational Science Proceedings*, 2020, 191-200. [Comprehensive review of healthcare ML challenges]

14. **Chen, I. Y., Pierson, E., Rose, S., Joshi, S., Ferryman, K., & Ghassemi, M.** (2021). Ethical machine learning in healthcare. *Annual Review of Biomedical Data Science*, 4, 123-144. [Important considerations for ethical AI in healthcare]

15. **Beam, A. L., & Kohane, I. S.** (2018). Big data and machine learning in health care. *JAMA*, 319(13), 1317-1318. [Perspective on big data applications in healthcare]

This chapter provides the mathematical foundation necessary for developing sophisticated healthcare AI systems. The concepts and methods presented here will be applied throughout the remaining chapters as we explore specific applications in clinical prediction, treatment optimization, and population health management.


## Code Examples

All code examples from this chapter are available in the repository:
- **Directory**: [`code_examples/chapter_02/`](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_02/)
- **Direct Download**: [ZIP file](https://github.com/sanjaybasu-waymark/healthcare-ai-book/archive/refs/heads/main.zip)

To use the examples:
```bash
git clone https://github.com/sanjaybasu-waymark/healthcare-ai-book.git
cd healthcare-ai-book/code_examples/chapter_02
pip install -r requirements.txt
```
