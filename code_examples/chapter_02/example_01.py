"""
Chapter 2 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

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

\# Advanced statistical libraries
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
    LEVEL_1A = "1a"  \# Systematic review of RCTs
    LEVEL_1B = "1b"  \# Individual RCT
    LEVEL_2A = "2a"  \# Systematic review of cohort studies
    LEVEL_2B = "2b"  \# Individual cohort study
    LEVEL_3A = "3a"  \# Systematic review of case-control studies
    LEVEL_3B = "3b"  \# Individual case-control study
    LEVEL_4 = "4"    \# Case series
    LEVEL_5 = "5"    \# Expert opinion

@dataclass
class DiagnosticTest:
    """
    Represents a diagnostic test with comprehensive performance characteristics.
    
    This class encapsulates all relevant information about a diagnostic test
    including performance metrics, cost considerations, and clinical context.
    """
    test_name: str
    test_type: TestType
    sensitivity: float  \# True positive rate
    specificity: float  \# True negative rate
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
        
        \# Calculate likelihood ratios
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
        
        \# Calculate using Bayes' theorem
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
    severity_score: int = 1  \# 1-5 scale
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
        
        \# Adjust for age if available
        if age is not None and self.age_adjusted_prevalence:
            age_group = self._get_age_group(age)
            if age_group in self.age_adjusted_prevalence:
                adjusted_prevalence *= self.age_adjusted_prevalence[age_group]
        
        \# Adjust for gender if available
        if gender is not None and self.gender_specific_prevalence:
            if gender.lower() in self.gender_specific_prevalence:
                adjusted_prevalence *= self.gender_specific_prevalence[gender.lower()]
        
        return min(adjusted_prevalence, 1.0)  \# Cap at 1.0
    
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
        
        \# Set random seeds for reproducibility
        np.random.seed(random_seed)
        
        \# Data storage
        self.conditions: Dict[str, ClinicalCondition] = {}
        self.tests: Dict[str, DiagnosticTest] = {}
        self.test_correlations: Optional[np.ndarray] = None
        self.population_data: Optional[pd.DataFrame] = None
        
        \# Model storage
        self.mcmc_trace = None
        self.posterior_samples: Dict[str, np.ndarray] = {}
        
        \# Performance tracking
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
        
        \# Get adjusted prior probability
        prior_prob = self._get_adjusted_prior(condition, patient_demographics)
        
        \# Calculate likelihood with correlation adjustment
        if use_test_correlations and self.test_correlations is not None:
            likelihood_pos, likelihood_neg = self._calculate_correlated_likelihood(
                test_results, condition_name
            )
        else:
            likelihood_pos, likelihood_neg = self._calculate_independent_likelihood(
                test_results
            )
        
        \# Bayesian inference using log-space arithmetic for numerical stability
        log_prior_pos = np.log(prior_prob)
        log_prior_neg = np.log(1 - prior_prob)
        
        log_posterior_pos = log_prior_pos + likelihood_pos
        log_posterior_neg = log_prior_neg + likelihood_neg
        
        \# Normalize using logsumexp
        log_evidence = logsumexp([log_posterior_pos, log_posterior_neg])
        
        posterior_prob = np.exp(log_posterior_pos - log_evidence)
        
        \# Calculate uncertainty measures
        uncertainty_measures = self._calculate_uncertainty_measures(
            posterior_prob, test_results, prior_prob
        )
        
        \# Store diagnostic decision for learning
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
            \# Prepare data for hierarchical modeling
            y = patient_data[f'has_{condition_name}'].values.astype(int)
            X = patient_data[[col for col in patient_data.columns 
                            if col.startswith('test_')]].values
            
            \# Build hierarchical model
            with pm.Model() as hierarchical_model:
                \# Hyperpriors for test performance
                alpha_sens = pm.Beta('alpha_sens', alpha=2, beta=2, shape=X.shape<sup>1</sup>)
                beta_spec = pm.Beta('beta_spec', alpha=2, beta=2, shape=X.shape<sup>1</sup>)
                
                \# Individual test sensitivities and specificities
                sensitivity = pm.Beta('sensitivity', alpha=alpha_sens * 100, 
                                    beta=(1 - alpha_sens) * 100, shape=X.shape<sup>1</sup>)
                specificity = pm.Beta('specificity', alpha=beta_spec * 100,
                                    beta=(1 - beta_spec) * 100, shape=X.shape<sup>1</sup>)
                
                \# Prior probability (prevalence)
                prevalence = pm.Beta('prevalence', alpha=1, beta=1)
                
                \# Likelihood for each patient
                for i in range(len(y)):
                    \# Calculate likelihood for positive and negative cases
                    likelihood_pos = pm.math.prod(
                        pm.math.where(X[i] == 1, sensitivity, 1 - sensitivity)
                    )
                    likelihood_neg = pm.math.prod(
                        pm.math.where(X[i] == 1, 1 - specificity, specificity)
                    )
                    
                    \# Posterior probability
                    posterior_prob = (prevalence * likelihood_pos) / (
                        prevalence * likelihood_pos + (1 - prevalence) * likelihood_neg
                    )
                    
                    \# Observed outcome
                    pm.Bernoulli(f'obs_{i}', p=posterior_prob, observed=y[i])
                
                \# Sample from posterior
                trace = pm.sample(self.mcmc_samples, random_seed=self.random_seed,
                                return_inferencedata=True)
            
            self.mcmc_trace = trace
            
            \# Extract posterior summaries
            summary = az.summary(trace)
            
            \# Calculate model diagnostics
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
            \# Extract true labels and test results
            y_true = validation_data[f'has_{condition_name}'].values
            test_columns = [col for col in validation_data.columns 
                          if col.startswith('test_')]
            
            \# Calculate posterior probabilities for all patients
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
            
            \# Calculate performance metrics
            auc_roc = roc_auc_score(y_true, y_pred_proba)
            
            \# Precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            auc_pr = np.trapz(precision, recall)
            
            \# Confusion matrix
            cm = confusion_matrix(y_true, y_pred_binary)
            tn, fp, fn, tp = cm.ravel()
            
            \# Calculate derived metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            \# F1 score
            f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
            
            \# Calibration metrics
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
            
            \# Store validation results
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
            
            if result:  \# Positive test result
                log_likelihood_pos += np.log(test.sensitivity)
                log_likelihood_neg += np.log(1 - test.specificity)
            else:  \# Negative test result
                log_likelihood_pos += np.log(1 - test.sensitivity)
                log_likelihood_neg += np.log(test.specificity)
        
        return log_likelihood_pos, log_likelihood_neg
    
    def _calculate_correlated_likelihood(self, 
                                       test_results: Dict[str, bool],
                                       condition_name: str) -> Tuple[float, float]:
        """Calculate likelihood accounting for test correlations."""
        \# Simplified implementation - in practice would use copula models
        \# or multivariate distributions to model test correlations
        
        \# For now, apply a correlation adjustment factor
        log_likelihood_pos, log_likelihood_neg = self._calculate_independent_likelihood(test_results)
        
        \# Reduce likelihood magnitude to account for positive correlations
        correlation_adjustment = 0.9  \# Assumes moderate positive correlation
        
        return (log_likelihood_pos * correlation_adjustment, 
                log_likelihood_neg * correlation_adjustment)
    
    def _calculate_uncertainty_measures(self, 
                                      posterior_prob: float,
                                      test_results: Dict[str, bool],
                                      prior_prob: float) -> Dict[str, float]:
        """Calculate comprehensive uncertainty measures."""
        \# Entropy-based uncertainty
        if posterior_prob == 0 or posterior_prob == 1:
            entropy = 0.0
        else:
            entropy = -(posterior_prob * np.log2(posterior_prob) + 
                       (1 - posterior_prob) * np.log2(1 - posterior_prob))
        
        \# Information gain
        if prior_prob == 0 or prior_prob == 1:
            prior_entropy = 0.0
        else:
            prior_entropy = -(prior_prob * np.log2(prior_prob) + 
                            (1 - prior_prob) * np.log2(1 - prior_prob))
        
        information_gain = prior_entropy - entropy
        
        \# Confidence based on distance from 0.5
        confidence = abs(posterior_prob - 0.5) * 2
        
        \# Number of tests contributing to decision
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
                \# Positive test - contribution is positive likelihood ratio
                contribution = np.log(test.positive_likelihood_ratio)
            else:
                \# Negative test - contribution is negative likelihood ratio
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
        
        \# Logit transformation of predicted probabilities
        y_pred_logit = np.log(y_pred_proba / (1 - y_pred_proba + 1e-10))
        
        \# Fit calibration model
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
        
        \# Basic condition information
        report = {
            'condition_name': condition_name,
            'icd_10_code': condition.icd_10_code,
            'prevalence': condition.prevalence,
            'severity_score': condition.severity_score,
            'generated_at': datetime.now().isoformat()
        }
        
        \# Available tests
        relevant_tests = {name: test.get_performance_summary() 
                         for name, test in self.tests.items()}
        report['available_tests'] = relevant_tests
        
        \# Validation results if available
        if include_validation and condition_name in self.validation_results:
            report['validation_results'] = self.validation_results[condition_name]
        
        \# Diagnostic history summary
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


\# Demonstration and testing functions
def create_example_diagnostic_system() -> BayesianDiagnosticSystem:
    """Create an example diagnostic system for demonstration."""
    system = BayesianDiagnosticSystem()
    
    \# Add example condition: Myocardial Infarction
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
    
    \# Add diagnostic tests
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
    
    \# Create example system
    system = create_example_diagnostic_system()
    
    \# Example patient scenarios
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
    
    \# Analyze each scenario
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
        
        \# Show uncertainty measures
        uncertainty = result['uncertainty_measures']
        print(f"Uncertainty category: {uncertainty['uncertainty_category']}")
        print(f"Information gain: {uncertainty['information_gain']:.3f} bits")
        print(f"Confidence: {uncertainty['confidence']:.3f}")
        
        \# Show test contributions
        print("\nTest contributions:")
        for test_name, contribution in result['test_contributions'].items():
            print(f"  {test_name}: {contribution:+.3f}")
        
        print("\n")
    
    \# Generate diagnostic report
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