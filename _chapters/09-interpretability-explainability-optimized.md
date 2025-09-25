---
layout: default
title: "Chapter 9: Interpretability and Explainability in Healthcare AI - Building Trust Through Transparent Clinical Decision Support"
nav_order: 9
parent: Chapters
has_children: false
---

# Chapter 9: Interpretability and Explainability in Healthcare AI - Building Trust Through Transparent Clinical Decision Support

*By Sanjay Basu MD PhD*

## Learning Objectives

By the end of this chapter, physician data scientists will be able to:

- Understand the theoretical foundations of interpretability and explainability in healthcare AI systems, including the distinction between global and local explanations, the cognitive science principles underlying effective explanations, and the regulatory requirements for transparent medical AI
- Implement model-agnostic explanation methods including LIME, SHAP, and counterfactual explanations with specific adaptations for clinical data types, temporal patterns, and multi-modal healthcare information
- Deploy intrinsically interpretable models appropriate for different healthcare applications, including decision trees, linear models, and rule-based systems that maintain clinical validity while providing inherent transparency
- Design clinical explanation interfaces that support physician decision-making workflows, incorporating cognitive load considerations, time constraints, and the need for actionable insights in clinical practice
- Evaluate explanation quality using both technical metrics and clinical validation approaches, including physician user studies, clinical outcome assessments, and regulatory compliance measures
- Navigate regulatory requirements for explainable AI in medical devices, including FDA guidance, EU regulations, and institutional review board considerations for explanation-based clinical studies
- Implement stakeholder-specific explanation systems that provide appropriate levels of detail and clinical context for physicians, nurses, patients, administrators, and regulatory bodies

## 9.1 Introduction to Healthcare AI Interpretability

Interpretability and explainability represent fundamental requirements for the successful deployment of artificial intelligence systems in healthcare environments. Unlike other domains where black-box models may be acceptable for certain applications, healthcare AI systems demand transparency due to the life-critical nature of medical decisions, stringent regulatory requirements, the need to maintain physician trust and clinical workflow integration, and the ethical imperative to provide patients with understandable information about their care.

The distinction between interpretability and explainability, while subtle, is crucial for healthcare applications and affects both technical implementation and clinical adoption. **Interpretability** refers to the degree to which a human can understand the cause of a decision made by an AI system without requiring additional explanation of the decision-making process. **Explainability** refers to the ability to provide post-hoc explanations for decisions made by potentially opaque models, often through separate explanation systems that analyze the original model's behavior. In healthcare contexts, both concepts are essential for different stakeholders, use cases, and regulatory requirements.

### 9.1.1 Theoretical Foundations of Medical AI Interpretability

Healthcare AI interpretability is grounded in several theoretical frameworks that address the unique requirements of medical decision-making, cognitive science principles, and clinical reasoning processes. Understanding these foundations is essential for designing explanation systems that effectively support clinical practice rather than creating additional cognitive burden for healthcare providers.

**Cognitive Load Theory and Clinical Decision-Making**: The cognitive load theory suggests that explanations must be carefully designed to match the cognitive capacity and expertise of the intended user, whether that be a specialist physician with deep domain knowledge, a primary care provider managing multiple conditions, a nurse implementing care protocols, or a patient seeking to understand their diagnosis and treatment options. In healthcare settings, cognitive overload from overly complex explanations can actually harm clinical decision-making rather than improve it, making the design of appropriate explanation complexity a critical consideration.

**Clinical Reasoning Frameworks**: Clinical reasoning frameworks provide the foundation for designing explanations that align with how physicians naturally think about medical problems and make diagnostic and therapeutic decisions. The **hypothetico-deductive model** of clinical reasoning suggests that physicians generate diagnostic hypotheses early in the clinical encounter based on initial patient presentation and then seek evidence to confirm or refute these hypotheses through additional history, physical examination, and diagnostic testing. AI explanations that support this natural reasoning process by highlighting evidence for and against different diagnostic possibilities are more likely to be clinically useful and readily adopted.

The **pattern recognition model** describes how experienced physicians often make rapid diagnoses based on pattern matching with previously encountered cases. AI explanations that provide similar case examples or highlight key diagnostic patterns can support this type of clinical reasoning. The **dual-process theory** of cognition distinguishes between fast, intuitive thinking (System 1) and slow, deliberative thinking (System 2). Healthcare AI explanations must support both types of thinking, providing quick intuitive insights for routine decisions while enabling deep analysis for complex or unusual cases.

**Bayesian Reasoning in Clinical Practice**: Clinical decision-making inherently involves Bayesian reasoning, where physicians update their diagnostic probabilities based on new evidence. AI explanations that explicitly show how different pieces of evidence contribute to diagnostic probability updates align well with this natural clinical reasoning process. The mathematical foundation can be expressed as:

$$P(D|E) = \frac{P(E|D) \cdot P(D)}{P(E)}$$

where $P(D|E)$ is the posterior probability of diagnosis $D$ given evidence $E$, $P(E|D)$ is the likelihood of observing evidence $E$ given diagnosis $D$, $P(D)$ is the prior probability of diagnosis $D$, and $P(E)$ is the marginal probability of evidence $E$.

**Trust Calibration and Appropriate Reliance**: Trust calibration represents a critical challenge in healthcare AI interpretability, as explanations must help physicians develop appropriate trust in AI systems—neither over-relying on AI recommendations nor dismissing potentially valuable insights. This requires explanations that accurately convey both the strengths and limitations of AI predictions, including uncertainty quantification and clear communication of the conditions under which the AI system is most and least reliable.

### 9.1.2 Regulatory and Legal Requirements

The regulatory landscape for explainable AI in healthcare is rapidly evolving, with increasing emphasis on transparency, accountability, and the ability to audit AI-driven medical decisions. Understanding these requirements is essential for developing compliant and deployable healthcare AI systems.

**FDA Software as Medical Device (SaMD) Framework**: The FDA's guidance on Software as Medical Device emphasizes the importance of providing clinicians with sufficient information to understand how AI systems make decisions, particularly for high-risk applications that could significantly impact patient outcomes. The FDA requires that AI-based medical devices provide transparency into their decision-making processes, including information about training data, model limitations, and performance characteristics across different patient populations.

The FDA's AI/ML guidance specifically addresses the need for explainability in adaptive AI systems, requiring that changes in model behavior be transparent and that clinicians understand how and why AI recommendations may change over time. This includes requirements for algorithm change protocols, performance monitoring, and communication of model updates to healthcare providers.

**European Union Medical Device Regulation (MDR) and AI Act**: The EU's Medical Device Regulation requires that AI-based medical devices provide sufficient transparency to enable healthcare providers to use them safely and effectively. The regulation emphasizes the need for clear instructions for use, including information about the AI system's intended purpose, limitations, and appropriate use conditions.

The EU AI Act further strengthens requirements for high-risk AI applications, including many healthcare use cases. The Act requires that high-risk AI systems be designed to enable human oversight and that users receive clear information about the AI system's capabilities and limitations. For healthcare applications, this includes requirements for explainability that enable healthcare providers to understand and validate AI recommendations.

**Clinical Liability and Malpractice Considerations**: Clinical liability considerations make interpretability essential for healthcare AI deployment, as physicians must be able to understand and justify their use of AI recommendations in clinical decision-making, particularly in cases where patient outcomes are suboptimal. Legal precedents increasingly recognize that physicians have a duty to understand the tools they use in patient care, including AI systems.

Malpractice liability may extend to inappropriate reliance on AI recommendations without adequate understanding of the system's limitations or failure to recognize when AI recommendations are inappropriate for a particular clinical situation. Explainable AI systems that clearly communicate their limitations and uncertainty can help protect both patients and physicians by enabling more informed clinical decision-making.

### 9.1.3 Clinical Workflow Integration

Successful healthcare AI interpretability requires deep understanding of clinical workflows, decision-making processes, and the practical constraints of healthcare delivery environments. Explanations that do not integrate well with clinical workflows are unlikely to be adopted, regardless of their technical sophistication.

**Time Constraints and Efficiency Requirements**: Clinical practice involves significant time constraints, with physicians often having only minutes to review patient information and make critical decisions. Explanations must be concise and immediately actionable, providing the most important information first and allowing for progressive disclosure of additional detail when needed. The design of explanation interfaces must consider the cognitive demands of clinical practice and avoid adding unnecessary complexity to already demanding workflows.

**Integration with Electronic Health Records**: Explanations must integrate seamlessly with existing electronic health record systems and clinical decision support tools. This requires careful attention to user interface design, data presentation, and workflow integration. Explanations that require physicians to switch between multiple systems or interfaces are less likely to be used effectively in clinical practice.

**Stakeholder-Specific Requirements**: Different healthcare stakeholders have different explanation needs and capabilities. Physicians require detailed technical information about model performance and limitations, while patients need explanations that are accessible and help them understand their care. Nurses may need explanations that focus on care implementation and monitoring, while administrators may require explanations that address cost-effectiveness and resource utilization.

## 9.2 Model-Agnostic Explanation Methods

Model-agnostic explanation methods provide the flexibility to explain any machine learning model without requiring access to the model's internal structure or training process. This is particularly valuable in healthcare where different types of models may be used for different applications, and where explanation systems must work with both proprietary and open-source models.

### 9.2.1 LIME (Local Interpretable Model-agnostic Explanations)

LIME provides local explanations for individual predictions by learning an interpretable model in the local neighborhood of the instance being explained. For healthcare applications, LIME can provide insights into which clinical features most strongly influence a particular diagnosis, treatment recommendation, or risk assessment for a specific patient.

The mathematical foundation of LIME involves solving the following optimization problem:

$$\xi(x) = \arg\min_{g \in G} L(f, g, \pi_x) + \Omega(g)$$

Where:
- $f$ is the original complex model being explained
- $g$ is the interpretable explanation model (e.g., linear model, decision tree)
- $G$ is the class of interpretable models
- $L(f, g, \pi_x)$ is the locality-aware loss function that measures how well $g$ approximates $f$ in the neighborhood of $x$
- $\pi_x$ defines the neighborhood around instance $x$
- $\Omega(g)$ is a complexity penalty that encourages simpler explanations

For healthcare applications, the neighborhood definition $\pi_x$ must be carefully designed to respect clinical relationships between features. For example, when explaining a cardiac risk prediction, the neighborhood should consider clinically meaningful variations in cardiac risk factors rather than arbitrary perturbations that might not correspond to realistic patient presentations.

### 9.2.2 SHAP (SHapley Additive exPlanations)

SHAP provides a unified framework for feature importance based on cooperative game theory, offering theoretically grounded explanations that satisfy several desirable properties including efficiency, symmetry, dummy feature, and additivity. The Shapley value represents the average marginal contribution of a feature across all possible coalitions of features.

The SHAP value for feature $i$ is defined as:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

Where $F$ is the set of all features and $S$ is a subset of features not including feature $i$. The SHAP framework guarantees that the sum of all feature contributions equals the difference between the model's prediction and the expected model output:

$$f(x) - E[f(X)] = \sum_{i=1}^{|F|} \phi_i$$

For healthcare applications, SHAP values provide clinically interpretable feature attributions that can help physicians understand which patient characteristics most strongly influence AI predictions. Different SHAP variants are appropriate for different types of healthcare models and data.

### 9.2.3 Counterfactual Explanations

Counterfactual explanations answer the question "What would need to change for the AI system to make a different prediction?" This is particularly valuable in healthcare for understanding treatment alternatives, identifying modifiable risk factors, and exploring what-if scenarios for patient care planning.

The mathematical formulation of counterfactual explanation generation can be expressed as an optimization problem:

$$x' = \arg\min_{x'} d(x, x') + \lambda \cdot L(f(x'), y')$$

Where:
- $x'$ is the counterfactual instance
- $d(x, x')$ is a distance function measuring the similarity between the original and counterfactual instances
- $L(f(x'), y')$ is a loss function ensuring the counterfactual achieves the desired prediction $y'$
- $\lambda$ is a regularization parameter balancing similarity and prediction change

For healthcare applications, the distance function $d(x, x')$ must incorporate clinical knowledge about which changes are realistic and actionable. For example, a counterfactual explanation for diabetes risk should focus on modifiable factors like weight, exercise, and diet rather than non-modifiable factors like age or genetic markers.

## 9.3 Comprehensive Healthcare AI Explanation Framework

### 9.3.1 Implementation of Advanced Explanation Methods

```python
"""
Comprehensive Healthcare AI Explainability Framework

This implementation provides advanced explanation capabilities specifically
designed for healthcare AI systems, incorporating clinical domain knowledge,
regulatory requirements, and stakeholder-specific explanation generation.

Author: Sanjay Basu MD PhD
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

# Advanced visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Counterfactual explanation libraries
try:
    from dice_ml import Data, Model, Dice
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    print("DiCE not available. Counterfactual explanations will use simplified implementation.")

# Additional interpretability libraries
try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

import logging
from datetime import datetime, timedelta
import json
import joblib
import pickle
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations for healthcare AI systems."""
    GLOBAL = "global"
    LOCAL = "local"
    COUNTERFACTUAL = "counterfactual"
    EXAMPLE_BASED = "example_based"
    RULE_BASED = "rule_based"
    TEMPORAL = "temporal"
    CAUSAL = "causal"

class StakeholderType(Enum):
    """Types of stakeholders requiring explanations."""
    PHYSICIAN = "physician"
    NURSE = "nurse"
    PATIENT = "patient"
    ADMINISTRATOR = "administrator"
    RESEARCHER = "researcher"
    REGULATOR = "regulator"
    FAMILY_MEMBER = "family_member"

class ExplanationComplexity(Enum):
    """Complexity levels for explanations."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    DETAILED = "detailed"
    TECHNICAL = "technical"

@dataclass
class ClinicalContext:
    """Clinical context information for explanation generation."""
    patient_id: str
    condition: str
    urgency_level: str  # "routine", "urgent", "emergent", "critical"
    care_setting: str  # "outpatient", "inpatient", "emergency", "icu"
    physician_specialty: Optional[str] = None
    patient_complexity: Optional[str] = None  # "simple", "moderate", "complex"
    time_constraints: Optional[int] = None  # minutes available for decision
    
@dataclass
class ExplanationRequest:
    """Request for explanation of AI prediction."""
    request_id: str
    instance_id: str
    prediction: float
    prediction_proba: Optional[np.ndarray] = None
    stakeholder_type: StakeholderType
    explanation_types: List[ExplanationType]
    complexity_level: ExplanationComplexity
    clinical_context: ClinicalContext
    custom_requirements: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ExplanationResult:
    """Result of explanation generation."""
    request_id: str
    explanation_type: ExplanationType
    explanation_data: Dict[str, Any]
    confidence_score: float
    clinical_relevance_score: float
    explanation_text: str
    visualization_data: Optional[Dict[str, Any]] = None
    actionable_insights: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'request_id': self.request_id,
            'explanation_type': self.explanation_type.value,
            'explanation_data': self.explanation_data,
            'confidence_score': self.confidence_score,
            'clinical_relevance_score': self.clinical_relevance_score,
            'explanation_text': self.explanation_text,
            'visualization_data': self.visualization_data,
            'actionable_insights': self.actionable_insights,
            'limitations': self.limitations,
            'timestamp': self.timestamp.isoformat()
        }

class ClinicalKnowledgeBase:
    """Clinical knowledge base for healthcare-aware explanations."""
    
    def __init__(self):
        """Initialize clinical knowledge base."""
        
        # Clinical feature categories
        self.feature_categories = {
            'demographics': ['age', 'gender', 'race', 'ethnicity'],
            'vital_signs': ['heart_rate', 'blood_pressure', 'temperature', 'respiratory_rate', 'oxygen_saturation'],
            'laboratory': ['hemoglobin', 'white_blood_cells', 'platelets', 'creatinine', 'glucose', 'sodium', 'potassium'],
            'medications': ['ace_inhibitor', 'beta_blocker', 'statin', 'insulin', 'anticoagulant'],
            'comorbidities': ['diabetes', 'hypertension', 'heart_failure', 'copd', 'kidney_disease'],
            'social_determinants': ['insurance_status', 'income_level', 'education', 'housing_stability']
        }
        
        # Clinical relationships and constraints
        self.clinical_relationships = {
            'diabetes': {
                'related_labs': ['glucose', 'hba1c', 'creatinine'],
                'related_medications': ['insulin', 'metformin'],
                'modifiable_factors': ['weight', 'diet', 'exercise', 'medication_adherence']
            },
            'heart_failure': {
                'related_labs': ['bnp', 'creatinine', 'sodium'],
                'related_medications': ['ace_inhibitor', 'beta_blocker', 'diuretic'],
                'modifiable_factors': ['fluid_intake', 'sodium_restriction', 'medication_adherence', 'weight']
            },
            'hypertension': {
                'related_measurements': ['systolic_bp', 'diastolic_bp'],
                'related_medications': ['ace_inhibitor', 'beta_blocker', 'diuretic', 'calcium_channel_blocker'],
                'modifiable_factors': ['sodium_intake', 'weight', 'exercise', 'alcohol_consumption']
            }
        }
        
        # Normal ranges and clinical significance
        self.reference_ranges = {
            'glucose': {'normal': (70, 100), 'units': 'mg/dL', 'critical_low': 50, 'critical_high': 400},
            'creatinine': {'normal': (0.6, 1.2), 'units': 'mg/dL', 'critical_high': 3.0},
            'hemoglobin': {'normal': (12.0, 16.0), 'units': 'g/dL', 'critical_low': 7.0},
            'systolic_bp': {'normal': (90, 140), 'units': 'mmHg', 'critical_low': 70, 'critical_high': 200},
            'heart_rate': {'normal': (60, 100), 'units': 'bpm', 'critical_low': 40, 'critical_high': 150}
        }
        
        # Clinical decision thresholds
        self.decision_thresholds = {
            'diabetes_diagnosis': {'hba1c': 6.5, 'fasting_glucose': 126},
            'hypertension_diagnosis': {'systolic_bp': 140, 'diastolic_bp': 90},
            'kidney_disease_stages': {'creatinine': [1.2, 2.0, 3.0, 5.0]}
        }
        
        logger.info("Initialized clinical knowledge base")
    
    def get_feature_category(self, feature_name: str) -> Optional[str]:
        """Get the clinical category for a feature."""
        for category, features in self.feature_categories.items():
            if feature_name.lower() in [f.lower() for f in features]:
                return category
        return None
    
    def get_clinical_significance(self, feature_name: str, value: float) -> Dict[str, Any]:
        """Assess clinical significance of a feature value."""
        
        if feature_name not in self.reference_ranges:
            return {'significance': 'unknown', 'interpretation': 'No reference range available'}
        
        ref_range = self.reference_ranges[feature_name]
        normal_min, normal_max = ref_range['normal']
        
        if value < normal_min:
            if 'critical_low' in ref_range and value < ref_range['critical_low']:
                significance = 'critically_low'
                interpretation = f'Critically low {feature_name}: {value} {ref_range["units"]}'
            else:
                significance = 'low'
                interpretation = f'Below normal {feature_name}: {value} {ref_range["units"]}'
        elif value > normal_max:
            if 'critical_high' in ref_range and value > ref_range['critical_high']:
                significance = 'critically_high'
                interpretation = f'Critically high {feature_name}: {value} {ref_range["units"]}'
            else:
                significance = 'high'
                interpretation = f'Above normal {feature_name}: {value} {ref_range["units"]}'
        else:
            significance = 'normal'
            interpretation = f'Normal {feature_name}: {value} {ref_range["units"]}'
        
        return {
            'significance': significance,
            'interpretation': interpretation,
            'normal_range': f"{normal_min}-{normal_max} {ref_range['units']}"
        }
    
    def get_modifiable_factors(self, condition: str) -> List[str]:
        """Get modifiable factors for a clinical condition."""
        return self.clinical_relationships.get(condition, {}).get('modifiable_factors', [])
    
    def get_related_features(self, condition: str) -> Dict[str, List[str]]:
        """Get features related to a clinical condition."""
        return self.clinical_relationships.get(condition, {})

class HealthcareAIExplainer:
    """
    Comprehensive explainability framework for healthcare AI systems.
    
    This class implements state-of-the-art explanation methods specifically
    adapted for healthcare applications, including clinical workflow integration
    and stakeholder-specific explanation generation.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        feature_types: Dict[str, str],
        training_data: Optional[pd.DataFrame] = None,
        clinical_context: Optional[Dict[str, Any]] = None,
        explanation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize healthcare AI explainer.
        
        Args:
            model: Trained ML model to explain
            feature_names: List of feature names
            feature_types: Dictionary mapping feature names to types ('numerical', 'categorical')
            training_data: Training data for explanation methods that require it
            clinical_context: Clinical context information
            explanation_config: Configuration for explanation methods
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.training_data = training_data
        self.clinical_context = clinical_context or {}
        self.explanation_config = explanation_config or {}
        
        # Initialize clinical knowledge base
        self.clinical_kb = ClinicalKnowledgeBase()
        
        # Initialize explanation methods
        self._initialize_explanation_methods()
        
        # Explanation cache for efficiency
        self.explanation_cache = {}
        
        # Performance tracking
        self.explanation_metrics = {
            'total_explanations': 0,
            'explanation_times': [],
            'stakeholder_feedback': defaultdict(list)
        }
        
        logger.info(f"Initialized healthcare AI explainer for model: {type(model).__name__}")
    
    def _initialize_explanation_methods(self):
        """Initialize various explanation methods."""
        
        # LIME explainer
        if self.training_data is not None:
            categorical_features = [i for i, name in enumerate(self.feature_names) 
                                  if self.feature_types.get(name) == 'categorical']
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.training_data.values,
                feature_names=self.feature_names,
                categorical_features=categorical_features,
                mode='classification' if hasattr(self.model, 'predict_proba') else 'regression',
                discretize_continuous=True
            )
        else:
            self.lime_explainer = None
        
        # SHAP explainer
        try:
            if hasattr(self.model, 'predict_proba'):
                self.shap_explainer = shap.Explainer(self.model.predict_proba, self.training_data)
            else:
                self.shap_explainer = shap.Explainer(self.model.predict, self.training_data)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
        
        # Counterfactual explainer
        if DICE_AVAILABLE and self.training_data is not None:
            try:
                # Prepare data for DiCE
                continuous_features = [name for name in self.feature_names 
                                     if self.feature_types.get(name) == 'numerical']
                
                dice_data = Data(
                    dataframe=self.training_data,
                    continuous_features=continuous_features,
                    outcome_name='target' if 'target' in self.training_data.columns else None
                )
                
                dice_model = Model(model=self.model, backend='sklearn')
                self.dice_explainer = Dice(dice_data, dice_model)
            except Exception as e:
                logger.warning(f"Could not initialize DiCE explainer: {e}")
                self.dice_explainer = None
        else:
            self.dice_explainer = None
    
    def generate_explanation(
        self,
        request: ExplanationRequest,
        instance_data: pd.Series
    ) -> List[ExplanationResult]:
        """
        Generate comprehensive explanation for a prediction.
        
        Args:
            request: Explanation request with stakeholder and context information
            instance_data: Data for the instance to explain
            
        Returns:
            List of explanation results
        """
        start_time = datetime.now()
        results = []
        
        # Check cache first
        cache_key = self._generate_cache_key(request, instance_data)
        if cache_key in self.explanation_cache:
            logger.info(f"Using cached explanation for request {request.request_id}")
            return self.explanation_cache[cache_key]
        
        # Generate explanations based on requested types
        for explanation_type in request.explanation_types:
            try:
                if explanation_type == ExplanationType.LOCAL:
                    result = self._generate_local_explanation(request, instance_data)
                elif explanation_type == ExplanationType.GLOBAL:
                    result = self._generate_global_explanation(request, instance_data)
                elif explanation_type == ExplanationType.COUNTERFACTUAL:
                    result = self._generate_counterfactual_explanation(request, instance_data)
                elif explanation_type == ExplanationType.EXAMPLE_BASED:
                    result = self._generate_example_based_explanation(request, instance_data)
                elif explanation_type == ExplanationType.RULE_BASED:
                    result = self._generate_rule_based_explanation(request, instance_data)
                else:
                    logger.warning(f"Unsupported explanation type: {explanation_type}")
                    continue
                
                if result:
                    # Adapt explanation for stakeholder
                    adapted_result = self._adapt_for_stakeholder(result, request)
                    results.append(adapted_result)
                    
            except Exception as e:
                logger.error(f"Error generating {explanation_type} explanation: {e}")
                continue
        
        # Cache results
        self.explanation_cache[cache_key] = results
        
        # Update metrics
        explanation_time = (datetime.now() - start_time).total_seconds()
        self.explanation_metrics['total_explanations'] += 1
        self.explanation_metrics['explanation_times'].append(explanation_time)
        
        logger.info(f"Generated {len(results)} explanations in {explanation_time:.2f} seconds")
        
        return results
    
    def _generate_local_explanation(
        self,
        request: ExplanationRequest,
        instance_data: pd.Series
    ) -> Optional[ExplanationResult]:
        """Generate local explanation using LIME and SHAP."""
        
        explanation_data = {}
        explanation_text_parts = []
        
        # LIME explanation
        if self.lime_explainer:
            try:
                lime_exp = self.lime_explainer.explain_instance(
                    instance_data.values,
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    num_features=min(10, len(self.feature_names))
                )
                
                lime_features = []
                for feature_idx, importance in lime_exp.as_list():
                    feature_name = self.feature_names[feature_idx] if isinstance(feature_idx, int) else feature_idx
                    lime_features.append({
                        'feature': feature_name,
                        'importance': importance,
                        'value': instance_data[feature_name] if feature_name in instance_data.index else None
                    })
                
                explanation_data['lime'] = lime_features
                
                # Generate clinical interpretation
                top_features = sorted(lime_features, key=lambda x: abs(x['importance']), reverse=True)[:5]
                explanation_text_parts.append("Key factors influencing this prediction:")
                
                for feature_info in top_features:
                    feature_name = feature_info['feature']
                    importance = feature_info['importance']
                    value = feature_info['value']
                    
                    # Get clinical significance
                    if value is not None:
                        clinical_sig = self.clinical_kb.get_clinical_significance(feature_name, value)
                        direction = "increases" if importance > 0 else "decreases"
                        explanation_text_parts.append(
                            f"- {feature_name}: {clinical_sig.get('interpretation', f'{value}')} "
                            f"({direction} prediction by {abs(importance):.3f})"
                        )
                
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")
        
        # SHAP explanation
        if self.shap_explainer:
            try:
                shap_values = self.shap_explainer(instance_data.values.reshape(1, -1))
                
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) > 2:
                        # Multi-class case - use positive class
                        shap_vals = shap_values.values[0, :, 1]
                    else:
                        shap_vals = shap_values.values[0]
                else:
                    shap_vals = shap_values[0]
                
                shap_features = []
                for i, (feature_name, shap_val) in enumerate(zip(self.feature_names, shap_vals)):
                    shap_features.append({
                        'feature': feature_name,
                        'shap_value': float(shap_val),
                        'value': instance_data[feature_name] if feature_name in instance_data.index else None
                    })
                
                explanation_data['shap'] = shap_features
                
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        if not explanation_data:
            return None
        
        # Calculate confidence and clinical relevance
        confidence_score = self._calculate_explanation_confidence(explanation_data)
        clinical_relevance_score = self._calculate_clinical_relevance(explanation_data, request.clinical_context)
        
        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights(explanation_data, request.clinical_context)
        
        return ExplanationResult(
            request_id=request.request_id,
            explanation_type=ExplanationType.LOCAL,
            explanation_data=explanation_data,
            confidence_score=confidence_score,
            clinical_relevance_score=clinical_relevance_score,
            explanation_text="\n".join(explanation_text_parts),
            actionable_insights=actionable_insights
        )
    
    def _generate_global_explanation(
        self,
        request: ExplanationRequest,
        instance_data: pd.Series
    ) -> Optional[ExplanationResult]:
        """Generate global explanation showing overall model behavior."""
        
        explanation_data = {}
        explanation_text_parts = ["Global model behavior analysis:"]
        
        # Feature importance from model
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = list(zip(self.feature_names, self.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            explanation_data['feature_importance'] = [
                {'feature': name, 'importance': float(importance)}
                for name, importance in feature_importance
            ]
            
            explanation_text_parts.append("\nMost important features globally:")
            for name, importance in feature_importance[:5]:
                category = self.clinical_kb.get_feature_category(name)
                explanation_text_parts.append(
                    f"- {name} ({category or 'other'}): {importance:.3f}"
                )
        
        # Permutation importance
        if self.training_data is not None:
            try:
                # Use a subset for efficiency
                sample_size = min(1000, len(self.training_data))
                sample_data = self.training_data.sample(n=sample_size, random_state=42)
                
                if 'target' in sample_data.columns:
                    X_sample = sample_data.drop('target', axis=1)
                    y_sample = sample_data['target']
                    
                    perm_importance = permutation_importance(
                        self.model, X_sample, y_sample, n_repeats=5, random_state=42
                    )
                    
                    perm_features = list(zip(self.feature_names, perm_importance.importances_mean))
                    perm_features.sort(key=lambda x: x[1], reverse=True)
                    
                    explanation_data['permutation_importance'] = [
                        {'feature': name, 'importance': float(importance)}
                        for name, importance in perm_features
                    ]
                    
            except Exception as e:
                logger.warning(f"Permutation importance calculation failed: {e}")
        
        if not explanation_data:
            return None
        
        confidence_score = 0.8  # Global explanations are generally reliable
        clinical_relevance_score = self._calculate_clinical_relevance(explanation_data, request.clinical_context)
        
        return ExplanationResult(
            request_id=request.request_id,
            explanation_type=ExplanationType.GLOBAL,
            explanation_data=explanation_data,
            confidence_score=confidence_score,
            clinical_relevance_score=clinical_relevance_score,
            explanation_text="\n".join(explanation_text_parts)
        )
    
    def _generate_counterfactual_explanation(
        self,
        request: ExplanationRequest,
        instance_data: pd.Series
    ) -> Optional[ExplanationResult]:
        """Generate counterfactual explanation."""
        
        explanation_data = {}
        explanation_text_parts = []
        
        if self.dice_explainer:
            try:
                # Generate counterfactuals
                counterfactuals = self.dice_explainer.generate_counterfactuals(
                    instance_data.to_frame().T,
                    total_CFs=3,
                    desired_class="opposite"
                )
                
                cf_data = []
                for i, cf in enumerate(counterfactuals.cf_examples_list[0].final_cfs_df.iterrows()):
                    cf_instance = cf[1]
                    changes = []
                    
                    for feature in self.feature_names:
                        if feature in instance_data.index and feature in cf_instance.index:
                            original_val = instance_data[feature]
                            cf_val = cf_instance[feature]
                            
                            if abs(original_val - cf_val) > 1e-6:  # Significant change
                                changes.append({
                                    'feature': feature,
                                    'original_value': float(original_val),
                                    'counterfactual_value': float(cf_val),
                                    'change': float(cf_val - original_val)
                                })
                    
                    cf_data.append({
                        'counterfactual_id': i,
                        'changes': changes
                    })
                
                explanation_data['counterfactuals'] = cf_data
                
                # Generate clinical interpretation
                if cf_data:
                    explanation_text_parts.append("To change the prediction, consider modifying:")
                    
                    # Focus on modifiable factors
                    condition = request.clinical_context.condition
                    modifiable_factors = self.clinical_kb.get_modifiable_factors(condition)
                    
                    for cf in cf_data[:2]:  # Show top 2 counterfactuals
                        explanation_text_parts.append(f"\nOption {cf['counterfactual_id'] + 1}:")
                        for change in cf['changes']:
                            feature = change['feature']
                            if feature.lower() in [f.lower() for f in modifiable_factors]:
                                explanation_text_parts.append(
                                    f"- Change {feature} from {change['original_value']:.2f} "
                                    f"to {change['counterfactual_value']:.2f}"
                                )
                
            except Exception as e:
                logger.warning(f"DiCE counterfactual generation failed: {e}")
        
        # Simplified counterfactual generation if DiCE not available
        if not explanation_data and self.training_data is not None:
            explanation_data = self._generate_simple_counterfactuals(instance_data, request)
            explanation_text_parts.append("Simplified counterfactual analysis:")
            explanation_text_parts.append("Consider modifying key risk factors to change the prediction.")
        
        if not explanation_data:
            return None
        
        confidence_score = 0.7  # Counterfactuals have moderate confidence
        clinical_relevance_score = self._calculate_clinical_relevance(explanation_data, request.clinical_context)
        
        # Generate actionable insights
        actionable_insights = self._generate_counterfactual_insights(explanation_data, request.clinical_context)
        
        return ExplanationResult(
            request_id=request.request_id,
            explanation_type=ExplanationType.COUNTERFACTUAL,
            explanation_data=explanation_data,
            confidence_score=confidence_score,
            clinical_relevance_score=clinical_relevance_score,
            explanation_text="\n".join(explanation_text_parts),
            actionable_insights=actionable_insights
        )
    
    def _generate_example_based_explanation(
        self,
        request: ExplanationRequest,
        instance_data: pd.Series
    ) -> Optional[ExplanationResult]:
        """Generate example-based explanation using similar cases."""
        
        if self.training_data is None:
            return None
        
        explanation_data = {}
        explanation_text_parts = ["Similar cases analysis:"]
        
        try:
            # Find similar cases
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Prepare data
            X_train = self.training_data.drop('target', axis=1) if 'target' in self.training_data.columns else self.training_data
            instance_vector = instance_data.values.reshape(1, -1)
            
            # Calculate similarities
            similarities = cosine_similarity(instance_vector, X_train.values)[0]
            
            # Get top similar cases
            top_indices = np.argsort(similarities)[-6:-1][::-1]  # Top 5 similar cases
            
            similar_cases = []
            for idx in top_indices:
                similar_case = {
                    'case_id': int(idx),
                    'similarity': float(similarities[idx]),
                    'features': X_train.iloc[idx].to_dict()
                }
                
                if 'target' in self.training_data.columns:
                    similar_case['outcome'] = self.training_data.iloc[idx]['target']
                
                similar_cases.append(similar_case)
            
            explanation_data['similar_cases'] = similar_cases
            
            # Generate clinical interpretation
            if similar_cases:
                explanation_text_parts.append(f"\nFound {len(similar_cases)} similar cases:")
                
                for i, case in enumerate(similar_cases[:3]):  # Show top 3
                    similarity_pct = case['similarity'] * 100
                    outcome = case.get('outcome', 'unknown')
                    explanation_text_parts.append(
                        f"- Case {i+1}: {similarity_pct:.1f}% similar, outcome: {outcome}"
                    )
                
                # Analyze patterns
                if 'target' in self.training_data.columns:
                    outcomes = [case.get('outcome') for case in similar_cases if 'outcome' in case]
                    if outcomes:
                        positive_rate = sum(outcomes) / len(outcomes)
                        explanation_text_parts.append(
                            f"\nIn similar cases, {positive_rate:.1%} had positive outcomes."
                        )
        
        except Exception as e:
            logger.warning(f"Example-based explanation failed: {e}")
            return None
        
        if not explanation_data:
            return None
        
        confidence_score = 0.8
        clinical_relevance_score = self._calculate_clinical_relevance(explanation_data, request.clinical_context)
        
        return ExplanationResult(
            request_id=request.request_id,
            explanation_type=ExplanationType.EXAMPLE_BASED,
            explanation_data=explanation_data,
            confidence_score=confidence_score,
            clinical_relevance_score=clinical_relevance_score,
            explanation_text="\n".join(explanation_text_parts)
        )
    
    def _generate_rule_based_explanation(
        self,
        request: ExplanationRequest,
        instance_data: pd.Series
    ) -> Optional[ExplanationResult]:
        """Generate rule-based explanation using decision tree surrogate."""
        
        if self.training_data is None:
            return None
        
        explanation_data = {}
        explanation_text_parts = ["Rule-based analysis:"]
        
        try:
            # Train surrogate decision tree
            X_train = self.training_data.drop('target', axis=1) if 'target' in self.training_data.columns else self.training_data
            
            # Get predictions from original model
            y_pred = self.model.predict(X_train)
            
            # Train decision tree surrogate
            surrogate_tree = DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42
            )
            surrogate_tree.fit(X_train, y_pred)
            
            # Extract rules for this instance
            tree_rules = export_text(surrogate_tree, feature_names=self.feature_names)
            
            # Get decision path for this instance
            decision_path = surrogate_tree.decision_path(instance_data.values.reshape(1, -1))
            leaf_id = surrogate_tree.apply(instance_data.values.reshape(1, -1))[0]
            
            # Extract relevant rules
            feature_names_used = []
            thresholds_used = []
            
            for node_id in decision_path.indices:
                if node_id != leaf_id:  # Not a leaf node
                    feature_idx = surrogate_tree.tree_.feature[node_id]
                    if feature_idx >= 0:  # Valid feature
                        feature_name = self.feature_names[feature_idx]
                        threshold = surrogate_tree.tree_.threshold[node_id]
                        feature_names_used.append(feature_name)
                        thresholds_used.append(threshold)
            
            explanation_data['decision_rules'] = {
                'features_used': feature_names_used,
                'thresholds': thresholds_used,
                'tree_rules': tree_rules[:1000]  # Truncate for readability
            }
            
            # Generate clinical interpretation
            if feature_names_used:
                explanation_text_parts.append("\nKey decision rules applied:")
                
                for feature, threshold in zip(feature_names_used[:5], thresholds_used[:5]):
                    current_value = instance_data.get(feature, 'unknown')
                    if current_value != 'unknown':
                        comparison = "≥" if current_value >= threshold else "<"
                        explanation_text_parts.append(
                            f"- {feature}: {current_value} {comparison} {threshold:.2f}"
                        )
        
        except Exception as e:
            logger.warning(f"Rule-based explanation failed: {e}")
            return None
        
        if not explanation_data:
            return None
        
        confidence_score = 0.7
        clinical_relevance_score = self._calculate_clinical_relevance(explanation_data, request.clinical_context)
        
        return ExplanationResult(
            request_id=request.request_id,
            explanation_type=ExplanationType.RULE_BASED,
            explanation_data=explanation_data,
            confidence_score=confidence_score,
            clinical_relevance_score=clinical_relevance_score,
            explanation_text="\n".join(explanation_text_parts)
        )
    
    def _generate_simple_counterfactuals(
        self,
        instance_data: pd.Series,
        request: ExplanationRequest
    ) -> Dict[str, Any]:
        """Generate simplified counterfactuals without DiCE."""
        
        counterfactuals = []
        
        # Focus on modifiable factors
        condition = request.clinical_context.condition
        modifiable_factors = self.clinical_kb.get_modifiable_factors(condition)
        
        for feature in modifiable_factors:
            if feature in instance_data.index:
                current_value = instance_data[feature]
                
                # Generate reasonable alternative values
                if self.feature_types.get(feature) == 'numerical':
                    # Try 10% and 20% changes
                    for change_pct in [0.1, 0.2]:
                        for direction in [-1, 1]:
                            new_value = current_value * (1 + direction * change_pct)
                            counterfactuals.append({
                                'feature': feature,
                                'original_value': float(current_value),
                                'counterfactual_value': float(new_value),
                                'change_percent': direction * change_pct * 100
                            })
        
        return {'simplified_counterfactuals': counterfactuals}
    
    def _adapt_for_stakeholder(
        self,
        result: ExplanationResult,
        request: ExplanationRequest
    ) -> ExplanationResult:
        """Adapt explanation for specific stakeholder type."""
        
        stakeholder = request.stakeholder_type
        complexity = request.complexity_level
        
        # Modify explanation text based on stakeholder
        if stakeholder == StakeholderType.PATIENT:
            # Simplify language for patients
            result.explanation_text = self._simplify_for_patient(result.explanation_text)
        elif stakeholder == StakeholderType.PHYSICIAN:
            # Add clinical context for physicians
            result.explanation_text = self._enhance_for_physician(result.explanation_text, request.clinical_context)
        elif stakeholder == StakeholderType.NURSE:
            # Focus on care implementation for nurses
            result.explanation_text = self._focus_for_nurse(result.explanation_text)
        
        # Adjust complexity
        if complexity == ExplanationComplexity.SIMPLE:
            result.explanation_text = self._simplify_explanation(result.explanation_text)
        elif complexity == ExplanationComplexity.TECHNICAL:
            result.explanation_text = self._add_technical_details(result.explanation_text, result.explanation_data)
        
        return result
    
    def _simplify_for_patient(self, explanation_text: str) -> str:
        """Simplify explanation for patient understanding."""
        
        # Replace medical terms with simpler language
        replacements = {
            'prediction': 'assessment',
            'algorithm': 'computer analysis',
            'feature': 'factor',
            'probability': 'chance',
            'significance': 'importance'
        }
        
        simplified = explanation_text
        for medical_term, simple_term in replacements.items():
            simplified = simplified.replace(medical_term, simple_term)
        
        # Add patient-friendly introduction
        simplified = "Based on your health information, here's what the analysis shows:\n\n" + simplified
        
        return simplified
    
    def _enhance_for_physician(self, explanation_text: str, clinical_context: ClinicalContext) -> str:
        """Enhance explanation with clinical context for physicians."""
        
        enhanced = f"Clinical Decision Support Analysis\n"
        enhanced += f"Patient: {clinical_context.patient_id}\n"
        enhanced += f"Condition: {clinical_context.condition}\n"
        enhanced += f"Care Setting: {clinical_context.care_setting}\n"
        enhanced += f"Urgency: {clinical_context.urgency_level}\n\n"
        enhanced += explanation_text
        
        # Add clinical recommendations
        enhanced += "\n\nClinical Considerations:\n"
        enhanced += "- Validate AI recommendations with clinical judgment\n"
        enhanced += "- Consider patient-specific factors not captured in the model\n"
        enhanced += "- Monitor for changes in patient condition\n"
        
        return enhanced
    
    def _focus_for_nurse(self, explanation_text: str) -> str:
        """Focus explanation on care implementation for nurses."""
        
        focused = "Nursing Care Implications:\n\n" + explanation_text
        focused += "\n\nMonitoring Points:\n"
        focused += "- Watch for changes in key indicators\n"
        focused += "- Document patient response to interventions\n"
        focused += "- Communicate concerns to physician promptly\n"
        
        return focused
    
    def _simplify_explanation(self, explanation_text: str) -> str:
        """Simplify explanation by removing technical details."""
        
        lines = explanation_text.split('\n')
        simplified_lines = []
        
        for line in lines:
            # Remove lines with technical details
            if any(term in line.lower() for term in ['coefficient', 'p-value', 'confidence interval']):
                continue
            simplified_lines.append(line)
        
        return '\n'.join(simplified_lines)
    
    def _add_technical_details(self, explanation_text: str, explanation_data: Dict[str, Any]) -> str:
        """Add technical details for technical stakeholders."""
        
        technical = explanation_text + "\n\nTechnical Details:\n"
        
        # Add model performance metrics if available
        technical += f"- Model type: {type(self.model).__name__}\n"
        
        # Add feature importance details
        if 'lime' in explanation_data:
            technical += "- LIME feature importances included\n"
        if 'shap' in explanation_data:
            technical += "- SHAP values computed\n"
        
        # Add statistical information
        technical += f"- Number of features: {len(self.feature_names)}\n"
        if self.training_data is not None:
            technical += f"- Training data size: {len(self.training_data)}\n"
        
        return technical
    
    def _calculate_explanation_confidence(self, explanation_data: Dict[str, Any]) -> float:
        """Calculate confidence score for explanation."""
        
        confidence_factors = []
        
        # LIME confidence
        if 'lime' in explanation_data:
            lime_features = explanation_data['lime']
            if lime_features:
                # Higher confidence if top features have high importance
                top_importance = max(abs(f['importance']) for f in lime_features)
                confidence_factors.append(min(1.0, top_importance * 2))
        
        # SHAP confidence
        if 'shap' in explanation_data:
            shap_features = explanation_data['shap']
            if shap_features:
                # Higher confidence if SHAP values are consistent
                shap_values = [abs(f['shap_value']) for f in shap_features]
                if shap_values:
                    confidence_factors.append(min(1.0, max(shap_values) * 2))
        
        # Overall confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5  # Default moderate confidence
    
    def _calculate_clinical_relevance(
        self,
        explanation_data: Dict[str, Any],
        clinical_context: ClinicalContext
    ) -> float:
        """Calculate clinical relevance score for explanation."""
        
        relevance_factors = []
        
        # Check if explanation includes clinically relevant features
        relevant_features = set()
        
        if clinical_context.condition:
            related_features = self.clinical_kb.get_related_features(clinical_context.condition)
            for feature_list in related_features.values():
                relevant_features.update(feature_list)
        
        # Analyze explanation features
        explanation_features = set()
        
        if 'lime' in explanation_data:
            explanation_features.update(f['feature'] for f in explanation_data['lime'])
        
        if 'shap' in explanation_data:
            explanation_features.update(f['feature'] for f in explanation_data['shap'])
        
        # Calculate overlap with clinically relevant features
        if relevant_features and explanation_features:
            overlap = len(explanation_features.intersection(relevant_features))
            relevance_factors.append(overlap / len(explanation_features))
        
        # Consider urgency level
        urgency_weights = {
            'routine': 0.7,
            'urgent': 0.8,
            'emergent': 0.9,
            'critical': 1.0
        }
        urgency_weight = urgency_weights.get(clinical_context.urgency_level, 0.7)
        relevance_factors.append(urgency_weight)
        
        # Overall relevance
        if relevance_factors:
            return np.mean(relevance_factors)
        else:
            return 0.6  # Default moderate relevance
    
    def _generate_actionable_insights(
        self,
        explanation_data: Dict[str, Any],
        clinical_context: ClinicalContext
    ) -> List[str]:
        """Generate actionable insights from explanation."""
        
        insights = []
        
        # Extract top influential features
        top_features = []
        
        if 'lime' in explanation_data:
            lime_features = sorted(explanation_data['lime'], key=lambda x: abs(x['importance']), reverse=True)
            top_features.extend(f['feature'] for f in lime_features[:3])
        
        if 'shap' in explanation_data:
            shap_features = sorted(explanation_data['shap'], key=lambda x: abs(x['shap_value']), reverse=True)
            top_features.extend(f['feature'] for f in shap_features[:3])
        
        # Generate insights for modifiable factors
        if clinical_context.condition:
            modifiable_factors = self.clinical_kb.get_modifiable_factors(clinical_context.condition)
            
            for feature in set(top_features):
                if feature.lower() in [f.lower() for f in modifiable_factors]:
                    insights.append(f"Consider interventions targeting {feature}")
        
        # Add general insights
        if clinical_context.urgency_level in ['urgent', 'emergent', 'critical']:
            insights.append("High urgency case - validate AI recommendations with clinical judgment")
        
        insights.append("Monitor patient response to any interventions")
        insights.append("Document decision-making rationale in patient record")
        
        return insights
    
    def _generate_counterfactual_insights(
        self,
        explanation_data: Dict[str, Any],
        clinical_context: ClinicalContext
    ) -> List[str]:
        """Generate actionable insights from counterfactual explanations."""
        
        insights = []
        
        if 'counterfactuals' in explanation_data:
            counterfactuals = explanation_data['counterfactuals']
            
            # Focus on modifiable factors
            modifiable_factors = self.clinical_kb.get_modifiable_factors(clinical_context.condition)
            
            for cf in counterfactuals[:2]:  # Top 2 counterfactuals
                for change in cf['changes']:
                    feature = change['feature']
                    if feature.lower() in [f.lower() for f in modifiable_factors]:
                        direction = "increase" if change['change'] > 0 else "decrease"
                        insights.append(f"Consider interventions to {direction} {feature}")
        
        elif 'simplified_counterfactuals' in explanation_data:
            simplified_cfs = explanation_data['simplified_counterfactuals']
            
            for cf in simplified_cfs[:3]:  # Top 3 changes
                feature = cf['feature']
                change_pct = cf['change_percent']
                direction = "increasing" if change_pct > 0 else "decreasing"
                insights.append(f"Consider {direction} {feature} by approximately {abs(change_pct):.0f}%")
        
        return insights
    
    def _generate_cache_key(self, request: ExplanationRequest, instance_data: pd.Series) -> str:
        """Generate cache key for explanation request."""
        
        # Create hash of key request components
        import hashlib
        
        key_components = [
            request.stakeholder_type.value,
            request.complexity_level.value,
            str(sorted(et.value for et in request.explanation_types)),
            request.clinical_context.condition,
            str(instance_data.values.tolist())
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def generate_explanation_report(
        self,
        explanations: List[ExplanationResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation report."""
        
        if not explanations:
            return {'error': 'No explanations provided'}
        
        # Summary statistics
        explanation_types = [exp.explanation_type.value for exp in explanations]
        avg_confidence = np.mean([exp.confidence_score for exp in explanations])
        avg_clinical_relevance = np.mean([exp.clinical_relevance_score for exp in explanations])
        
        # Collect all actionable insights
        all_insights = []
        for exp in explanations:
            all_insights.extend(exp.actionable_insights)
        
        # Remove duplicates while preserving order
        unique_insights = []
        seen = set()
        for insight in all_insights:
            if insight not in seen:
                unique_insights.append(insight)
                seen.add(insight)
        
        report = {
            'summary': {
                'total_explanations': len(explanations),
                'explanation_types': explanation_types,
                'average_confidence': avg_confidence,
                'average_clinical_relevance': avg_clinical_relevance,
                'timestamp': datetime.now().isoformat()
            },
            'actionable_insights': unique_insights,
            'detailed_explanations': [exp.to_dict() for exp in explanations],
            'recommendations': [
                "Review all explanations in context of clinical judgment",
                "Validate AI recommendations with patient-specific factors",
                "Monitor patient outcomes and adjust as needed",
                "Document explanation review in patient record"
            ]
        }
        
        return report
    
    def visualize_explanations(
        self,
        explanations: List[ExplanationResult],
        save_path: Optional[str] = None
    ):
        """Create visualizations of explanations."""
        
        if not explanations:
            print("No explanations to visualize")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Feature Importance (LIME)', 'Feature Importance (SHAP)', 
                          'Explanation Confidence', 'Clinical Relevance'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # LIME feature importance
        lime_data = []
        for exp in explanations:
            if exp.explanation_type == ExplanationType.LOCAL and 'lime' in exp.explanation_data:
                lime_features = exp.explanation_data['lime']
                for feature_info in lime_features[:5]:  # Top 5 features
                    lime_data.append({
                        'feature': feature_info['feature'],
                        'importance': abs(feature_info['importance'])
                    })
        
        if lime_data:
            lime_df = pd.DataFrame(lime_data)
            lime_grouped = lime_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(x=lime_grouped.index[:10], y=lime_grouped.values[:10], name="LIME"),
                row=1, col=1
            )
        
        # SHAP feature importance
        shap_data = []
        for exp in explanations:
            if exp.explanation_type == ExplanationType.LOCAL and 'shap' in exp.explanation_data:
                shap_features = exp.explanation_data['shap']
                for feature_info in shap_features[:5]:  # Top 5 features
                    shap_data.append({
                        'feature': feature_info['feature'],
                        'importance': abs(feature_info['shap_value'])
                    })
        
        if shap_data:
            shap_df = pd.DataFrame(shap_data)
            shap_grouped = shap_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(x=shap_grouped.index[:10], y=shap_grouped.values[:10], name="SHAP"),
                row=1, col=2
            )
        
        # Explanation confidence
        confidence_scores = [exp.confidence_score for exp in explanations]
        explanation_types = [exp.explanation_type.value for exp in explanations]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(confidence_scores))),
                y=confidence_scores,
                mode='markers+lines',
                text=explanation_types,
                name="Confidence"
            ),
            row=2, col=1
        )
        
        # Clinical relevance
        relevance_scores = [exp.clinical_relevance_score for exp in explanations]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(relevance_scores))),
                y=relevance_scores,
                mode='markers+lines',
                text=explanation_types,
                name="Clinical Relevance"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Healthcare AI Explanation Analysis",
            showlegend=False,
            height=800
        )
        
        # Update axes
        fig.update_xaxes(title_text="Features", row=1, col=1)
        fig.update_xaxes(title_text="Features", row=1, col=2)
        fig.update_xaxes(title_text="Explanation Index", row=2, col=1)
        fig.update_xaxes(title_text="Explanation Index", row=2, col=2)
        
        fig.update_yaxes(title_text="Importance", row=1, col=1)
        fig.update_yaxes(title_text="Importance", row=1, col=2)
        fig.update_yaxes(title_text="Confidence Score", row=2, col=1)
        fig.update_yaxes(title_text="Relevance Score", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()

## Bibliography and References

### Foundational Interpretability and Explainability Literature

1. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144. DOI: 10.1145/2939672.2939778. [Foundational LIME paper]

2. **Lundberg, S. M., & Lee, S. I.** (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774. [Foundational SHAP paper]

3. **Molnar, C.** (2020). Interpretable machine learning: A guide for making black box models explainable. *Christoph Molnar*. [Comprehensive guide to interpretable ML]

4. **Rudin, C.** (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206-215. DOI: 10.1038/s42256-019-0048-x. [Argument for inherently interpretable models]

### Healthcare-Specific Interpretability Research

5. **Holzinger, A., Biemann, C., Pattichis, C. S., & Kell, D. B.** (2017). What do we need to build explainable AI systems for the medical domain? *arXiv preprint arXiv:1712.09923*. [Healthcare-specific XAI requirements]

6. **Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A.** (2019). What clinicians want: contextualizing explainable machine learning for clinical end use. *Proceedings of the 4th Machine Learning for Healthcare Conference*, 359-380. [Clinical user requirements for XAI]

7. **Ghassemi, M., Oakden-Rayner, L., & Beam, A. L.** (2021). The false hope of current approaches to explainable artificial intelligence in health care. *The Lancet Digital Health*, 3(11), e745-e750. DOI: 10.1016/S2589-7500(21)00208-9. [Critical analysis of healthcare XAI]

8. **Amann, J., Blasimme, A., Vayena, E., Frey, D., & Madai, V. I.** (2020). Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. *BMC Medical Informatics and Decision Making*, 20(1), 1-9. [Multidisciplinary perspective on healthcare XAI]

### Counterfactual Explanations and Causal Inference

9. **Wachter, S., Mittelstadt, B., & Russell, C.** (2017). Counterfactual explanations without opening the black box: automated decisions and the GDPR. *Harvard Journal of Law & Technology*, 31, 841. [Legal and technical foundations of counterfactual explanations]

10. **Mothilal, R. K., Sharma, A., & Tan, C.** (2020). Explaining machine learning classifiers through diverse counterfactual explanations. *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607-617. [DiCE framework for counterfactual explanations]

11. **Pearl, J., & Mackenzie, D.** (2018). The book of why: the new science of cause and effect. *Basic Books*. ISBN: 978-0465097609. [Foundational work on causal reasoning]

### Regulatory and Ethical Frameworks

12. **U.S. Food and Drug Administration.** (2021). Artificial intelligence/machine learning (AI/ML)-based software as a medical device (SaMD) action plan. *FDA Guidance Document*. [FDA requirements for AI/ML medical devices]

13. **European Commission.** (2021). Proposal for a regulation laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). *European Commission*. [EU AI Act requirements]

14. **Floridi, L., et al.** (2018). AI4People—an ethical framework for a good AI society: opportunities, risks, principles, and recommendations. *Minds and Machines*, 28(4), 689-707. [Ethical framework for AI systems]

### Clinical Decision-Making and Cognitive Science

15. **Norman, G. R., Young, M., & Brooks, L.** (2007). Non-analytical models of clinical reasoning: the role of experience. *Medical Education*, 41(12), 1140-1145. [Clinical reasoning models]

16. **Croskerry, P.** (2009). A universal model of diagnostic reasoning. *Academic Medicine*, 84(8), 1022-1028. [Dual-process theory in clinical diagnosis]

17. **Eva, K. W.** (2005). What every teacher needs to know about clinical reasoning. *Medical Education*, 39(1), 98-106. [Clinical reasoning for medical education]

### Trust and Human-AI Interaction

18. **Lee, J. D., & See, K. A.** (2004). Trust in automation: designing for appropriate reliance. *Human Factors*, 46(1), 50-80. [Trust calibration in automated systems]

19. **Jacovi, A., & Goldberg, Y.** (2020). Towards faithfulness and consistency for explanations. *arXiv preprint arXiv:2004.10444*. [Faithfulness in explanation methods]

20. **Miller, T.** (2019). Explanation in artificial intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1-38. DOI: 10.1016/j.artint.2018.07.007. [Social science perspectives on explanation]

This chapter provides a comprehensive framework for implementing interpretable and explainable AI systems in healthcare environments. The implementations presented address the unique challenges of clinical settings including regulatory compliance, stakeholder-specific requirements, and clinical workflow integration. The next chapter will explore robustness and security in healthcare AI, building upon these interpretability concepts to address the need for reliable and secure clinical AI systems.
