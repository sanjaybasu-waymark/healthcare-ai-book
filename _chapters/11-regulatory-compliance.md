---
layout: default
title: "Chapter 11: Regulatory Compliance"
nav_order: 11
parent: Chapters
permalink: /chapters/11-regulatory-compliance/
---

# Chapter 11: Regulatory Compliance and Validation Frameworks - Navigating Healthcare AI Approval and Deployment

*By Sanjay Basu MD PhD*

## Learning Objectives

By the end of this chapter, physician data scientists will be able to:

- Navigate the complex regulatory landscape for healthcare AI across multiple jurisdictions, including FDA, EMA, Health Canada, and other international regulatory bodies, understanding the specific requirements and pathways for AI/ML-based medical devices
- Implement FDA Software as Medical Device (SaMD) frameworks for AI/ML-based medical devices, including risk classification, predetermined change control plans, and clinical evidence requirements that address the unique challenges of adaptive AI systems
- Design comprehensive compliance management systems that ensure ongoing regulatory adherence throughout the AI system lifecycle, including quality management systems, documentation requirements, and change control procedures
- Develop robust clinical evidence packages that meet regulatory requirements for AI systems, including analytical validation, clinical validation, and clinical utility studies with appropriate statistical methodologies and endpoint selection
- Establish quality management systems compliant with ISO 13485, IEC 62304, and other relevant standards, incorporating AI-specific considerations for data governance, algorithm development, and software lifecycle management
- Manage post-market surveillance and adverse event reporting for AI medical devices, including performance monitoring, bias detection, and regulatory reporting requirements that address the evolving nature of AI systems
- Implement international harmonization strategies for global deployment of healthcare AI systems, understanding the differences between regulatory frameworks and developing strategies for multi-jurisdictional approval

## 11.1 Introduction to Healthcare AI Regulation

The regulatory landscape for healthcare artificial intelligence represents one of the most complex and rapidly evolving areas in medical technology, requiring physician data scientists to navigate an intricate web of requirements, guidelines, and standards that vary significantly across jurisdictions and application domains. Unlike traditional medical devices with well-established regulatory pathways developed over decades of experience, AI systems present unique challenges that require new frameworks, innovative approaches, and adaptive regulatory strategies to ensure patient safety while fostering innovation and clinical advancement.

Healthcare AI regulation encompasses multiple interconnected dimensions including **device classification and risk assessment**, **clinical validation requirements and evidence generation**, **quality management systems and software lifecycle processes**, **post-market surveillance and performance monitoring**, **international harmonization efforts and global deployment strategies**, and **ethical considerations and bias mitigation requirements**. The dynamic nature of AI systems, which can learn, adapt, and evolve over time, fundamentally challenges traditional regulatory paradigms that were designed for static medical devices with fixed functionality and predictable behavior patterns.

The regulatory complexity is further compounded by the interdisciplinary nature of healthcare AI, which spans computer science, clinical medicine, biostatistics, regulatory science, and health economics. Successful navigation of this landscape requires not only technical expertise in AI development but also deep understanding of clinical workflows, regulatory processes, quality management principles, and the broader healthcare ecosystem in which these systems operate.

### 11.1.1 Regulatory Challenges Unique to AI Systems

Healthcare AI systems present several fundamental regulatory challenges that distinguish them from traditional medical devices and require innovative approaches to safety assessment, performance evaluation, and ongoing monitoring. Understanding these challenges is essential for developing compliant and deployable AI systems that meet both regulatory requirements and clinical needs.

**Algorithmic Transparency and Explainability Requirements**: Traditional medical devices typically have well-understood mechanisms of action that can be explained through physical principles, biological pathways, or established clinical relationships. In contrast, many AI systems, particularly deep learning models, operate as "black boxes" where the relationship between inputs and outputs is not readily interpretable by human experts. This creates tension between regulatory requirements for transparency and explainability and the clinical effectiveness that may come from complex, less interpretable models.

The challenge is particularly acute for high-risk applications where regulatory agencies require detailed understanding of device behavior and failure modes. The FDA's guidance on explainable AI emphasizes the need for appropriate levels of transparency based on the clinical context and risk profile, but implementation remains challenging for complex neural networks with millions of parameters.

Mathematical approaches to explainability, such as LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations), provide post-hoc explanations but may not satisfy regulatory requirements for understanding fundamental algorithmic behavior. The development of inherently interpretable models represents one approach to addressing this challenge, but often at the cost of reduced performance.

**Continuous Learning and Adaptive Systems**: Traditional medical devices are designed with fixed functionality that remains constant throughout their operational lifetime. AI systems, particularly those employing machine learning techniques, can potentially learn and adapt their behavior based on new data encountered during deployment. This capability, while potentially beneficial for improving performance and adapting to changing clinical environments, creates significant regulatory challenges.

The FDA's concept of **predetermined change control plans (PCCPs)** attempts to address this challenge by allowing manufacturers to pre-specify types of modifications that can be made to AI systems without requiring new regulatory submissions. However, implementing effective PCCPs requires sophisticated understanding of how AI systems behave under different conditions and the ability to predict and bound the effects of potential changes.

The mathematical framework for change control can be expressed through performance bounds and statistical monitoring:

$$P_{new}(m) \geq P_{baseline}(m) - \delta_m \text{ with probability } \geq 1-\alpha$$

Where $P_{new}(m)$ represents the performance of the modified system on metric $m$, $P_{baseline}(m)$ is the baseline performance established during initial validation, $\delta_m$ is the maximum allowable degradation for that metric, and $\alpha$ is the acceptable probability of exceeding the degradation threshold.

**Data Dependency and Generalizability**: AI system performance is intrinsically linked to the quality, representativeness, and characteristics of training data. Unlike traditional devices where performance can be evaluated through standardized testing protocols, AI systems may exhibit significantly different behavior when deployed in clinical environments that differ from their training conditions.

This data dependency creates challenges for regulatory evaluation because traditional clinical trial methodologies may not adequately assess AI system performance across the full range of clinical conditions where the system will be deployed. Regulatory frameworks must address how to evaluate and ensure appropriate data governance throughout the system lifecycle, including training data curation, validation dataset selection, and ongoing monitoring of deployment data characteristics.

**Validation Complexity and Subgroup Performance**: Traditional clinical trials typically focus on demonstrating overall efficacy and safety in a target population. AI systems require more sophisticated validation approaches that address performance across diverse patient subgroups, clinical settings, and use conditions. This is particularly important for addressing health equity concerns and ensuring that AI systems do not exacerbate existing healthcare disparities.

The validation complexity extends to the need for demonstrating not just clinical effectiveness but also robustness across diverse patient populations, clinical settings, imaging protocols, laboratory methods, and other sources of variation that may affect AI system performance. Traditional clinical trial designs may be insufficient for evaluating AI systems that exhibit different performance characteristics across subgroups or that may be sensitive to subtle variations in input data.

**Human-AI Interaction and Workflow Integration**: AI systems in healthcare typically function as decision support tools that interact with clinicians rather than operating autonomously. This creates regulatory challenges in evaluating not just the AI system itself but also the human-AI interaction and its integration into clinical workflows.

Regulatory evaluation must consider how clinicians interpret and act upon AI system outputs, the potential for automation bias or over-reliance on AI recommendations, and the overall impact on clinical decision-making quality. This requires novel study designs that evaluate the AI system in its intended use environment with appropriate clinical users.

### 11.1.2 Global Regulatory Landscape and Jurisdictional Differences

The global regulatory landscape for healthcare AI involves multiple agencies with varying approaches, requirements, and timelines, creating a complex environment for organizations seeking to deploy AI systems internationally. Understanding these differences is essential for developing global deployment strategies and ensuring compliance across multiple jurisdictions.

**United States Food and Drug Administration (FDA)**: The FDA has been among the most active regulatory agencies in developing AI-specific guidance and frameworks. The FDA's approach builds upon the Software as Medical Device (SaMD) framework developed by the International Medical Device Regulators Forum (IMDRF) and includes several key components:

- **Risk-based classification system** that categorizes AI systems based on their intended use and potential for harm
- **Predetermined change control plans** that allow for specified modifications without new regulatory submissions
- **Clinical evidence requirements** that scale with risk classification and include considerations for AI-specific validation challenges
- **Quality system requirements** that address AI development processes and data governance
- **Post-market surveillance** requirements that include performance monitoring and adverse event reporting

The FDA's Digital Health Center of Excellence provides guidance and support for AI developers, and the agency has established several pilot programs to test new regulatory approaches for AI systems.

**European Medicines Agency (EMA) and Medical Device Regulation (MDR)**: The European regulatory framework for AI medical devices operates under the Medical Device Regulation (MDR) and involves a different approach that emphasizes conformity assessment and notified body involvement. Key aspects include:

- **CE marking process** that requires demonstration of compliance with essential requirements and harmonized standards
- **Notified body assessment** for higher-risk devices, including many AI systems
- **Clinical evidence requirements** that emphasize clinical evaluation and post-market clinical follow-up
- **Unique Device Identification (UDI)** requirements for traceability and post-market monitoring
- **EUDAMED database** for device registration and adverse event reporting

The European approach places greater emphasis on harmonized standards and conformity assessment procedures, with notified bodies playing a central role in evaluating compliance for higher-risk devices.

**Health Canada and Other National Regulators**: Health Canada has developed its own approach to AI regulation that incorporates elements from both FDA and European frameworks while addressing specific Canadian healthcare system considerations. Other national regulators, including Japan's Pharmaceuticals and Medical Devices Agency (PMDA), Australia's Therapeutic Goods Administration (TGA), and Brazil's ANVISA, are developing their own approaches.

**International Harmonization Efforts**: The International Medical Device Regulators Forum (IMDRF) works to harmonize regulatory approaches globally through working groups focused on Software as Medical Device, AI considerations, and other relevant topics. The IMDRF SaMD framework provides a foundation for risk-based classification that has been adopted by multiple regulatory agencies.

However, significant differences remain between jurisdictions in terms of specific requirements, evaluation procedures, and timelines. Organizations developing AI systems for global deployment must carefully navigate these differences and develop strategies that address multiple regulatory frameworks simultaneously.

### 11.1.3 Risk-Based Classification Systems and Regulatory Pathways

Healthcare AI systems are typically classified using risk-based frameworks that determine the level of regulatory oversight required, the type of clinical evidence needed, and the specific regulatory pathway for approval. Understanding these classification systems is essential for planning AI development projects and ensuring appropriate regulatory strategy.

**FDA Risk Classification System**: The FDA categorizes medical devices into three classes based on the level of control necessary to ensure safety and effectiveness:

- **Class I (Low Risk)**: Devices subject to general controls only, typically exempt from premarket notification requirements
- **Class II (Moderate Risk)**: Devices subject to general and special controls, typically requiring 510(k) premarket notification
- **Class III (High Risk)**: Devices subject to general and special controls and premarket approval (PMA) requirements

For AI systems, the classification depends on the intended use, clinical context, and potential consequences of device failure. Most AI systems fall into Class II or Class III categories due to their role in clinical decision-making.

**IMDRF SaMD Risk Categorization**: The International Medical Device Regulators Forum has developed a risk categorization framework specifically for Software as Medical Device that considers two primary dimensions:

1. **Healthcare Situation State**: 
   - **Critical**: Immediate life-threatening or serious deteriorating healthcare situation
   - **Serious**: Healthcare situation requiring timely intervention to avoid serious deterioration
   - **Non-serious**: Healthcare situation not requiring urgent intervention

2. **Healthcare Decision**:
   - **Treat**: SaMD provides treatment options or drives clinical management
   - **Diagnose**: SaMD provides diagnostic information
   - **Drive**: SaMD provides information to drive clinical management
   - **Inform**: SaMD provides information relevant to healthcare decisions

The intersection of these dimensions creates a risk categorization matrix:

| Healthcare Decision | Critical | Serious | Non-serious |
|-------------------|----------|---------|-------------|
| Treat | IV (High) | III (Moderate-High) | II (Moderate-Low) |
| Diagnose | IV (High) | III (Moderate-High) | II (Moderate-Low) |
| Drive | III (Moderate-High) | II (Moderate-Low) | I (Low) |
| Inform | II (Moderate-Low) | I (Low) | I (Low) |

**Regulatory Pathways and Requirements**: The risk classification determines the appropriate regulatory pathway and associated requirements:

- **Class I/Category I**: General controls, typically no premarket submission required
- **Class II/Category II**: 510(k) premarket notification demonstrating substantial equivalence to predicate devices
- **Class III/Category III-IV**: Premarket approval (PMA) or De Novo pathway requiring comprehensive clinical evidence

Each pathway has specific requirements for clinical evidence, quality system compliance, labeling, and post-market obligations that must be carefully planned and executed.

## 11.2 FDA Regulatory Framework for AI/ML Medical Devices

The FDA's regulatory framework for AI/ML-based medical devices represents the most comprehensive and mature approach to AI regulation in healthcare, providing detailed guidance on classification, clinical evidence requirements, quality system obligations, and post-market surveillance. Understanding this framework is essential for organizations developing AI systems for the US market and provides insights applicable to other regulatory jurisdictions.

### 11.2.1 Software as Medical Device (SaMD) Classification and Risk Assessment

The FDA's approach to regulating AI/ML-based medical devices builds upon the Software as Medical Device framework developed by the International Medical Device Regulators Forum, providing a systematic approach to classifying software based on its intended use and the risk associated with its clinical application. This framework addresses the unique characteristics of software-based medical devices while maintaining consistency with traditional device classification principles.

**SaMD Definition and Scope**: Software as Medical Device is defined as software intended to be used for one or more medical purposes that perform these purposes without being part of a hardware medical device. This definition encompasses standalone AI applications, cloud-based diagnostic systems, and software components that provide medical functionality independent of the hardware platform on which they operate.

The scope of SaMD regulation includes:
- **Diagnostic AI systems** that analyze medical images, laboratory results, or other clinical data to provide diagnostic information
- **Treatment planning software** that recommends therapeutic interventions based on patient data analysis
- **Risk assessment tools** that predict patient outcomes or identify high-risk individuals
- **Clinical decision support systems** that provide recommendations for clinical management
- **Monitoring and alerting systems** that analyze patient data to detect critical conditions

**Risk Categorization Framework**: The SaMD risk categorization framework considers two primary dimensions that determine the level of regulatory oversight required:

1. **State of the Healthcare Situation**:
   - **Critical**: Healthcare situations where accurate and timely diagnosis or treatment is vital to avoid death, long-term disability, or other serious deterioration of health, mitigation of public health emergencies, or to ensure access to continued therapy
   - **Serious**: Healthcare situations where accurate and timely diagnosis or treatment is important to avoid unnecessary morbidity or disability, or where delayed or inappropriate treatment could lead to serious deterioration
   - **Non-serious**: Healthcare situations where the healthcare decision is important for healthcare management but where delayed or inappropriate treatment is unlikely to result in serious deterioration

2. **Healthcare Decision Information**:
   - **Treat**: SaMD provides information to treat or diagnose for the purpose of treating or preventing a medical condition
   - **Diagnose**: SaMD provides information for the purpose of detecting, diagnosing, or screening for medical conditions
   - **Drive**: SaMD provides information to drive clinical management for the purpose of informing treatment or diagnosis
   - **Inform**: SaMD provides information relevant to healthcare decisions for the purpose of informing clinical management

**Mathematical Risk Assessment Framework**: The risk assessment process can be formalized through a quantitative framework that considers multiple factors:

$$R = f(P_{failure}, S_{consequence}, D_{detectability}, C_{clinical\_context})$$

Where:
- $P_{failure}$ represents the probability of AI system failure or incorrect output
- $S_{consequence}$ represents the severity of consequences if the system fails
- $D_{detectability}$ represents the likelihood that failures will be detected before causing harm
- $C_{clinical\_context}$ represents contextual factors that modify risk (e.g., availability of alternative diagnostics, time pressure, patient acuity)

This framework helps determine appropriate risk mitigation strategies and regulatory requirements based on quantitative risk assessment principles.

**Classification Examples and Case Studies**: Understanding how the SaMD framework applies to specific AI applications helps clarify the classification process:

- **Diabetic Retinopathy Screening AI**: An AI system that analyzes retinal photographs to detect diabetic retinopathy would typically be classified as Category III (Moderate-High Risk) because it diagnoses a serious condition that requires timely treatment to prevent vision loss
- **Sepsis Prediction Algorithm**: An AI system that analyzes electronic health record data to predict sepsis risk would be classified as Category IV (High Risk) because it drives treatment decisions in critical healthcare situations where delayed intervention can be life-threatening
- **Medication Interaction Checker**: An AI system that identifies potential drug interactions would typically be classified as Category II (Moderate-Low Risk) because it informs clinical management in serious healthcare situations

### 11.2.2 Predetermined Change Control Plans (PCCPs)

One of the most significant innovations in AI regulation is the FDA's Predetermined Change Control Plan framework, which recognizes the unique nature of AI systems that may need to adapt and improve over time while maintaining regulatory compliance. This framework allows manufacturers to make specified modifications to their AI/ML systems without requiring new regulatory submissions, provided the changes fall within predetermined bounds and follow established protocols.

**PCCP Framework Components**: A comprehensive PCCP must address several key components that ensure changes are made safely and effectively:

1. **Modification Scope and Boundaries**: Clear definition of the types of modifications that are permitted under the PCCP, including:
   - **Algorithm updates** such as retraining with new data, parameter adjustments, or architectural modifications
   - **Performance improvements** that enhance accuracy, sensitivity, specificity, or other performance metrics
   - **Labeling updates** that reflect new indications, contraindications, or usage instructions
   - **Software updates** that address bugs, security vulnerabilities, or compatibility issues

2. **Modification Protocols and Procedures**: Detailed procedures for implementing changes, including:
   - **Validation requirements** for testing modified systems before deployment
   - **Documentation standards** for recording changes and their rationale
   - **Approval processes** for authorizing modifications
   - **Rollback procedures** for reverting changes if problems are identified

3. **Impact Assessment Methodologies**: Systematic approaches for evaluating the effect of modifications on safety and effectiveness:
   - **Performance monitoring** using predefined metrics and statistical tests
   - **Risk assessment** to identify potential negative impacts
   - **Benefit-risk analysis** to ensure modifications provide net clinical benefit
   - **Subgroup analysis** to ensure changes don't adversely affect specific patient populations

4. **Risk Mitigation Strategies**: Proactive measures to address potential negative impacts:
   - **Performance bounds** that define acceptable ranges for key metrics
   - **Monitoring systems** that detect performance degradation
   - **Alert mechanisms** that notify users of significant changes
   - **Intervention protocols** for addressing identified problems

**Mathematical Framework for Change Control**: The PCCP framework can be formalized through statistical monitoring and control theory principles:

$$H_0: P_{new}(m) \geq P_{baseline}(m) - \delta_m$$
$$H_1: P_{new}(m) < P_{baseline}(m) - \delta_m$$

Where statistical tests are used to monitor whether the performance of the modified system remains within acceptable bounds. Sequential monitoring procedures can be implemented using control charts or sequential probability ratio tests:

$$\Lambda_n = \prod_{i=1}^n \frac{f(x_i|\theta_1)}{f(x_i|\theta_0)}$$

Where $\Lambda_n$ is the likelihood ratio after $n$ observations, and decisions about system performance are made based on predefined thresholds.

**Implementation Strategies and Best Practices**: Successful PCCP implementation requires careful planning and execution:

- **Comprehensive validation datasets** that represent the full range of intended use conditions
- **Robust performance metrics** that capture all aspects of system behavior relevant to safety and effectiveness
- **Automated monitoring systems** that can detect performance changes in real-time
- **Clear communication protocols** for informing users about system changes
- **Regular review and update** of PCCP procedures based on experience and new knowledge

### 11.2.3 Clinical Evidence Requirements and Validation Strategies

The FDA requires clinical evidence to demonstrate the safety and effectiveness of AI/ML medical devices, with the level and type of evidence proportional to the risk classification and clinical context. Understanding these requirements and developing appropriate validation strategies is essential for successful regulatory approval and clinical deployment.

**Tiered Evidence Framework**: The FDA employs a tiered approach to clinical evidence that scales with device risk and clinical impact:

1. **Analytical Validation**: Demonstration that the AI algorithm performs as intended on reference datasets and meets specified performance criteria
2. **Clinical Validation**: Evidence that the algorithm's output is clinically meaningful and accurate in the intended use population
3. **Clinical Utility**: Proof that use of the algorithm improves patient outcomes, clinical workflow, or healthcare delivery

**Analytical Validation Requirements**: Analytical validation focuses on the technical performance of the AI algorithm and includes several key components:

- **Algorithm Performance Assessment**: Evaluation of accuracy, sensitivity, specificity, positive predictive value, negative predictive value, and other relevant metrics using appropriate reference standards
- **Robustness Testing**: Assessment of algorithm performance under various conditions including different data sources, imaging protocols, patient populations, and edge cases
- **Failure Mode Analysis**: Systematic evaluation of how the algorithm behaves when encountering unusual inputs, corrupted data, or other challenging conditions
- **Bias Assessment**: Evaluation of algorithm performance across different demographic groups, clinical settings, and other relevant subgroups

The mathematical framework for analytical validation typically involves statistical hypothesis testing:

$$H_0: \text{Performance} \geq \text{Threshold}$$
$$H_1: \text{Performance} < \text{Threshold}$$

With appropriate sample size calculations to ensure adequate power:

$$n = \frac{(z_{\alpha} + z_{\beta})^2 \sigma^2}{(\mu_1 - \mu_0)^2}$$

Where $n$ is the required sample size, $z_{\alpha}$ and $z_{\beta}$ are critical values for Type I and Type II errors, $\sigma^2$ is the variance, and $\mu_1 - \mu_0$ is the effect size.

**Clinical Validation Methodologies**: Clinical validation demonstrates that the AI system provides clinically meaningful and accurate information in real-world clinical settings:

- **Retrospective Studies**: Analysis of historical clinical data to evaluate AI system performance compared to clinical outcomes or expert interpretation
- **Prospective Studies**: Forward-looking studies that evaluate AI system performance in real-time clinical settings
- **Reader Studies**: Controlled studies that compare AI system performance to human expert interpretation
- **Multi-site Validation**: Studies conducted across multiple clinical sites to assess generalizability

**Clinical Utility Assessment**: Clinical utility studies evaluate whether the AI system actually improves clinical outcomes, workflow efficiency, or healthcare delivery:

- **Randomized Controlled Trials**: Gold standard studies that compare outcomes between groups with and without AI assistance
- **Before-After Studies**: Evaluation of clinical outcomes before and after AI system implementation
- **Time-Motion Studies**: Assessment of workflow efficiency and time savings
- **Economic Evaluation**: Analysis of cost-effectiveness and healthcare resource utilization

**Special Considerations for AI Systems**: AI validation requires addressing several unique considerations:

- **Subgroup Performance**: Ensuring equitable performance across different patient demographics, clinical settings, and use conditions
- **Human-AI Interaction**: Evaluating how clinicians interpret and act upon AI system outputs
- **Workflow Integration**: Assessing the impact of AI system integration on clinical workflows and decision-making processes
- **Long-term Performance**: Monitoring AI system performance over time to detect drift or degradation

## 11.3 Comprehensive Regulatory Compliance Framework

### 11.3.1 Implementation of Advanced Compliance Management Systems

```python
"""
Comprehensive Healthcare AI Regulatory Compliance Framework

This implementation provides advanced compliance management capabilities
specifically designed for healthcare AI applications, including FDA SaMD
classification, clinical evidence management, and regulatory submission tracking.

Author: Sanjay Basu MD PhD
License: MIT
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

import logging
from datetime import datetime, timedelta
import json
import joblib
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import uuid
import os
from pathlib import Path
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskClass(Enum):
    """FDA risk classification for medical devices."""
    CLASS_I = "Class I"
    CLASS_II = "Class II"
    CLASS_III = "Class III"

class SaMDCategory(Enum):
    """Software as Medical Device risk categories."""
    CLASS_I = "Class I - Low Risk"
    CLASS_II = "Class II - Moderate-Low Risk"
    CLASS_III = "Class III - Moderate-High Risk"
    CLASS_IV = "Class IV - High Risk"

class HealthcareSituation(Enum):
    """Healthcare situation classification."""
    CRITICAL = "critical"
    SERIOUS = "serious"
    NON_SERIOUS = "non_serious"

class HealthcareDecision(Enum):
    """Healthcare decision classification."""
    TREAT = "treat"
    DIAGNOSE = "diagnose"
    DRIVE = "drive"
    INFORM = "inform"

class ComplianceStatus(Enum):
    """Compliance status tracking."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    EXPIRED = "expired"

class StudyType(Enum):
    """Clinical study types."""
    RETROSPECTIVE = "retrospective"
    PROSPECTIVE = "prospective"
    READER_STUDY = "reader_study"
    RCT = "randomized_controlled_trial"
    BEFORE_AFTER = "before_after"
    MULTI_SITE = "multi_site"

class EvidenceLevel(Enum):
    """Clinical evidence levels."""
    ANALYTICAL = "analytical"
    CLINICAL = "clinical"
    UTILITY = "utility"

@dataclass
class RegulatorySubmission:
    """Regulatory submission record."""
    submission_id: str
    submission_type: str  # "510k", "PMA", "De Novo", etc.
    device_name: str
    risk_class: RiskClass
    samd_category: SaMDCategory
    submission_date: datetime
    status: str
    predicate_devices: List[str]
    clinical_studies: List[str]
    substantial_equivalence: bool
    fda_response_date: Optional[datetime] = None
    approval_date: Optional[datetime] = None
    conditions_of_approval: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'submission_id': self.submission_id,
            'submission_type': self.submission_type,
            'device_name': self.device_name,
            'risk_class': self.risk_class.value,
            'samd_category': self.samd_category.value,
            'submission_date': self.submission_date.isoformat(),
            'status': self.status,
            'predicate_devices': self.predicate_devices,
            'clinical_studies': self.clinical_studies,
            'substantial_equivalence': self.substantial_equivalence,
            'fda_response_date': self.fda_response_date.isoformat() if self.fda_response_date else None,
            'approval_date': self.approval_date.isoformat() if self.approval_date else None,
            'conditions_of_approval': self.conditions_of_approval,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ClinicalStudy:
    """Clinical study record."""
    study_id: str
    study_type: StudyType
    evidence_level: EvidenceLevel
    study_design: str
    primary_endpoint: str
    secondary_endpoints: List[str]
    sample_size: int
    study_population: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    statistical_plan: str
    start_date: datetime
    completion_date: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    publications: List[str] = field(default_factory=list)
    regulatory_submissions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'study_id': self.study_id,
            'study_type': self.study_type.value,
            'evidence_level': self.evidence_level.value,
            'study_design': self.study_design,
            'primary_endpoint': self.primary_endpoint,
            'secondary_endpoints': self.secondary_endpoints,
            'sample_size': self.sample_size,
            'study_population': self.study_population,
            'inclusion_criteria': self.inclusion_criteria,
            'exclusion_criteria': self.exclusion_criteria,
            'statistical_plan': self.statistical_plan,
            'start_date': self.start_date.isoformat(),
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'results': self.results,
            'publications': self.publications,
            'regulatory_submissions': self.regulatory_submissions,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ChangeControlPlan:
    """Predetermined Change Control Plan record."""
    pccp_id: str
    device_name: str
    modification_types: List[str]
    modification_protocols: Dict[str, str]
    performance_bounds: Dict[str, Tuple[float, float]]
    validation_requirements: List[str]
    risk_mitigation_strategies: List[str]
    monitoring_plan: str
    approval_authority: str
    effective_date: datetime
    expiration_date: Optional[datetime] = None
    approved_modifications: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ComplianceAssessment:
    """Compliance assessment record."""
    assessment_id: str
    device_name: str
    assessment_date: datetime
    assessor: str
    compliance_areas: Dict[str, ComplianceStatus]
    findings: List[str]
    recommendations: List[str]
    corrective_actions: List[str]
    next_assessment_date: datetime
    overall_status: ComplianceStatus
    timestamp: datetime = field(default_factory=datetime.now)

class SaMDClassifier:
    """Software as Medical Device risk classifier."""
    
    def __init__(self):
        """Initialize SaMD classifier."""
        
        # Risk classification matrix
        self.risk_matrix = {
            (HealthcareSituation.CRITICAL, HealthcareDecision.TREAT): SaMDCategory.CLASS_IV,
            (HealthcareSituation.CRITICAL, HealthcareDecision.DIAGNOSE): SaMDCategory.CLASS_IV,
            (HealthcareSituation.CRITICAL, HealthcareDecision.DRIVE): SaMDCategory.CLASS_III,
            (HealthcareSituation.CRITICAL, HealthcareDecision.INFORM): SaMDCategory.CLASS_II,
            (HealthcareSituation.SERIOUS, HealthcareDecision.TREAT): SaMDCategory.CLASS_III,
            (HealthcareSituation.SERIOUS, HealthcareDecision.DIAGNOSE): SaMDCategory.CLASS_III,
            (HealthcareSituation.SERIOUS, HealthcareDecision.DRIVE): SaMDCategory.CLASS_II,
            (HealthcareSituation.SERIOUS, HealthcareDecision.INFORM): SaMDCategory.CLASS_I,
            (HealthcareSituation.NON_SERIOUS, HealthcareDecision.TREAT): SaMDCategory.CLASS_II,
            (HealthcareSituation.NON_SERIOUS, HealthcareDecision.DIAGNOSE): SaMDCategory.CLASS_II,
            (HealthcareSituation.NON_SERIOUS, HealthcareDecision.DRIVE): SaMDCategory.CLASS_I,
            (HealthcareSituation.NON_SERIOUS, HealthcareDecision.INFORM): SaMDCategory.CLASS_I,
        }
        
        # FDA device class mapping
        self.fda_class_mapping = {
            SaMDCategory.CLASS_I: RiskClass.CLASS_I,
            SaMDCategory.CLASS_II: RiskClass.CLASS_II,
            SaMDCategory.CLASS_III: RiskClass.CLASS_II,
            SaMDCategory.CLASS_IV: RiskClass.CLASS_III
        }
        
        logger.info("Initialized SaMD classifier")
    
    def classify_device(
        self,
        healthcare_situation: HealthcareSituation,
        healthcare_decision: HealthcareDecision,
        device_description: str,
        intended_use: str
    ) -> Dict[str, Any]:
        """
        Classify a software medical device according to SaMD framework.
        
        Args:
            healthcare_situation: Healthcare situation classification
            healthcare_decision: Healthcare decision classification
            device_description: Description of the device
            intended_use: Intended use statement
            
        Returns:
            Classification results with risk category and regulatory pathway
        """
        
        # Get SaMD category from risk matrix
        samd_category = self.risk_matrix.get(
            (healthcare_situation, healthcare_decision),
            SaMDCategory.CLASS_I
        )
        
        # Map to FDA device class
        fda_class = self.fda_class_mapping[samd_category]
        
        # Determine regulatory pathway
        regulatory_pathway = self._determine_regulatory_pathway(fda_class, samd_category)
        
        # Assess clinical evidence requirements
        evidence_requirements = self._assess_evidence_requirements(samd_category)
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(
            healthcare_situation, healthcare_decision, device_description
        )
        
        classification_result = {
            'samd_category': samd_category,
            'fda_class': fda_class,
            'regulatory_pathway': regulatory_pathway,
            'evidence_requirements': evidence_requirements,
            'risk_assessment': risk_assessment,
            'healthcare_situation': healthcare_situation,
            'healthcare_decision': healthcare_decision,
            'device_description': device_description,
            'intended_use': intended_use,
            'classification_date': datetime.now().isoformat()
        }
        
        logger.info(f"Classified device as {samd_category.value}, FDA {fda_class.value}")
        
        return classification_result
    
    def _determine_regulatory_pathway(self, fda_class: RiskClass, samd_category: SaMDCategory) -> str:
        """Determine appropriate regulatory pathway."""
        
        if fda_class == RiskClass.CLASS_I:
            return "General Controls Only"
        elif fda_class == RiskClass.CLASS_II:
            return "510(k) Premarket Notification"
        elif fda_class == RiskClass.CLASS_III:
            if samd_category == SaMDCategory.CLASS_IV:
                return "Premarket Approval (PMA)"
            else:
                return "De Novo Pathway (if no predicate)"
        else:
            return "Unknown"
    
    def _assess_evidence_requirements(self, samd_category: SaMDCategory) -> Dict[str, List[str]]:
        """Assess clinical evidence requirements based on SaMD category."""
        
        evidence_requirements = {
            'analytical_validation': [],
            'clinical_validation': [],
            'clinical_utility': []
        }
        
        # Analytical validation (required for all categories)
        evidence_requirements['analytical_validation'] = [
            'Algorithm performance assessment',
            'Robustness testing',
            'Failure mode analysis',
            'Bias assessment'
        ]
        
        # Clinical validation requirements
        if samd_category in [SaMDCategory.CLASS_II, SaMDCategory.CLASS_III, SaMDCategory.CLASS_IV]:
            evidence_requirements['clinical_validation'] = [
                'Clinical accuracy assessment',
                'Multi-site validation',
                'Subgroup analysis',
                'Human-AI interaction studies'
            ]
        
        # Clinical utility requirements
        if samd_category in [SaMDCategory.CLASS_III, SaMDCategory.CLASS_IV]:
            evidence_requirements['clinical_utility'] = [
                'Clinical outcome studies',
                'Workflow impact assessment',
                'Economic evaluation',
                'Long-term performance monitoring'
            ]
        
        return evidence_requirements
    
    def _generate_risk_assessment(
        self,
        healthcare_situation: HealthcareSituation,
        healthcare_decision: HealthcareDecision,
        device_description: str
    ) -> Dict[str, Any]:
        """Generate comprehensive risk assessment."""
        
        # Risk factors assessment
        risk_factors = {
            'clinical_impact': self._assess_clinical_impact(healthcare_situation, healthcare_decision),
            'failure_consequences': self._assess_failure_consequences(healthcare_situation),
            'human_oversight': self._assess_human_oversight_requirements(healthcare_decision),
            'complexity': self._assess_algorithm_complexity(device_description)
        }
        
        # Overall risk score (simplified calculation)
        risk_score = np.mean(list(risk_factors.values()))
        
        risk_assessment = {
            'risk_factors': risk_factors,
            'overall_risk_score': risk_score,
            'risk_level': self._categorize_risk_level(risk_score),
            'mitigation_strategies': self._recommend_mitigation_strategies(risk_factors)
        }
        
        return risk_assessment
    
    def _assess_clinical_impact(self, situation: HealthcareSituation, decision: HealthcareDecision) -> float:
        """Assess clinical impact score (0-1)."""
        situation_scores = {
            HealthcareSituation.CRITICAL: 1.0,
            HealthcareSituation.SERIOUS: 0.7,
            HealthcareSituation.NON_SERIOUS: 0.3
        }
        
        decision_scores = {
            HealthcareDecision.TREAT: 1.0,
            HealthcareDecision.DIAGNOSE: 0.9,
            HealthcareDecision.DRIVE: 0.6,
            HealthcareDecision.INFORM: 0.3
        }
        
        return (situation_scores[situation] + decision_scores[decision]) / 2
    
    def _assess_failure_consequences(self, situation: HealthcareSituation) -> float:
        """Assess failure consequences score (0-1)."""
        scores = {
            HealthcareSituation.CRITICAL: 1.0,
            HealthcareSituation.SERIOUS: 0.6,
            HealthcareSituation.NON_SERIOUS: 0.2
        }
        return scores[situation]
    
    def _assess_human_oversight_requirements(self, decision: HealthcareDecision) -> float:
        """Assess human oversight requirements (0-1, higher means more oversight needed)."""
        scores = {
            HealthcareDecision.TREAT: 1.0,
            HealthcareDecision.DIAGNOSE: 0.8,
            HealthcareDecision.DRIVE: 0.6,
            HealthcareDecision.INFORM: 0.3
        }
        return scores[decision]
    
    def _assess_algorithm_complexity(self, description: str) -> float:
        """Assess algorithm complexity based on description (simplified)."""
        complexity_keywords = {
            'deep learning': 0.9,
            'neural network': 0.8,
            'machine learning': 0.6,
            'artificial intelligence': 0.7,
            'ensemble': 0.7,
            'linear': 0.2,
            'rule-based': 0.3
        }
        
        description_lower = description.lower()
        max_complexity = 0.5  # Default
        
        for keyword, score in complexity_keywords.items():
            if keyword in description_lower:
                max_complexity = max(max_complexity, score)
        
        return max_complexity
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize overall risk level."""
        if risk_score >= 0.8:
            return "High"
        elif risk_score >= 0.6:
            return "Moderate-High"
        elif risk_score >= 0.4:
            return "Moderate"
        else:
            return "Low"
    
    def _recommend_mitigation_strategies(self, risk_factors: Dict[str, float]) -> List[str]:
        """Recommend risk mitigation strategies."""
        strategies = []
        
        if risk_factors['clinical_impact'] > 0.7:
            strategies.append("Implement comprehensive clinical validation studies")
            strategies.append("Establish robust post-market surveillance")
        
        if risk_factors['failure_consequences'] > 0.7:
            strategies.append("Implement fail-safe mechanisms")
            strategies.append("Provide clear uncertainty quantification")
        
        if risk_factors['human_oversight'] > 0.7:
            strategies.append("Ensure appropriate human oversight requirements")
            strategies.append("Provide comprehensive user training")
        
        if risk_factors['complexity'] > 0.7:
            strategies.append("Implement explainability features")
            strategies.append("Conduct thorough algorithm validation")
        
        return strategies

class ClinicalEvidenceManager:
    """Manager for clinical evidence generation and tracking."""
    
    def __init__(self, database_path: str = "clinical_evidence.db"):
        """Initialize clinical evidence manager."""
        self.database_path = database_path
        self.studies = {}
        self.evidence_packages = {}
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Initialized clinical evidence manager with database: {database_path}")
    
    def _initialize_database(self):
        """Initialize SQLite database for evidence tracking."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create studies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS studies (
                study_id TEXT PRIMARY KEY,
                study_type TEXT,
                evidence_level TEXT,
                study_design TEXT,
                primary_endpoint TEXT,
                sample_size INTEGER,
                start_date TEXT,
                completion_date TEXT,
                status TEXT,
                results TEXT,
                created_at TEXT
            )
        ''')
        
        # Create evidence packages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evidence_packages (
                package_id TEXT PRIMARY KEY,
                device_name TEXT,
                regulatory_submission TEXT,
                analytical_studies TEXT,
                clinical_studies TEXT,
                utility_studies TEXT,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_study(
        self,
        study_type: StudyType,
        evidence_level: EvidenceLevel,
        study_design: str,
        primary_endpoint: str,
        sample_size: int,
        study_population: str,
        inclusion_criteria: List[str],
        exclusion_criteria: List[str],
        statistical_plan: str
    ) -> ClinicalStudy:
        """Create a new clinical study."""
        
        study_id = f"STUDY_{uuid.uuid4().hex[:8].upper()}"
        
        study = ClinicalStudy(
            study_id=study_id,
            study_type=study_type,
            evidence_level=evidence_level,
            study_design=study_design,
            primary_endpoint=primary_endpoint,
            secondary_endpoints=[],
            sample_size=sample_size,
            study_population=study_population,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            statistical_plan=statistical_plan,
            start_date=datetime.now()
        )
        
        # Store in memory and database
        self.studies[study_id] = study
        self._save_study_to_database(study)
        
        logger.info(f"Created clinical study: {study_id}")
        
        return study
    
    def _save_study_to_database(self, study: ClinicalStudy):
        """Save study to database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO studies 
            (study_id, study_type, evidence_level, study_design, primary_endpoint,
             sample_size, start_date, completion_date, status, results, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            study.study_id,
            study.study_type.value,
            study.evidence_level.value,
            study.study_design,
            study.primary_endpoint,
            study.sample_size,
            study.start_date.isoformat(),
            study.completion_date.isoformat() if study.completion_date else None,
            "active",
            json.dumps(study.results) if study.results else None,
            study.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def conduct_analytical_validation(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        reference_standard: np.ndarray,
        device_name: str
    ) -> Dict[str, Any]:
        """Conduct comprehensive analytical validation."""
        
        model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch_data, _ in test_data:
                outputs = model(batch_data)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            reference_standard, predictions, probabilities
        )
        
        # Conduct robustness testing
        robustness_results = self._conduct_robustness_testing(
            model, test_data, reference_standard
        )
        
        # Bias assessment
        bias_assessment = self._conduct_bias_assessment(
            predictions, reference_standard, test_data
        )
        
        # Failure mode analysis
        failure_analysis = self._conduct_failure_mode_analysis(
            model, test_data, reference_standard
        )
        
        validation_results = {
            'device_name': device_name,
            'validation_date': datetime.now().isoformat(),
            'performance_metrics': performance_metrics,
            'robustness_results': robustness_results,
            'bias_assessment': bias_assessment,
            'failure_analysis': failure_analysis,
            'sample_size': len(predictions),
            'validation_status': 'completed'
        }
        
        logger.info(f"Completed analytical validation for {device_name}")
        
        return validation_results
    
    def _calculate_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'npv': self._calculate_npv(y_true, y_pred)
        }
        
        # Add AUC if binary classification
        if len(np.unique(y_true)) == 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
        
        # Calculate confidence intervals
        metrics_ci = self._calculate_confidence_intervals(y_true, y_pred, y_prob)
        metrics.update(metrics_ci)
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # Multi-class specificity (average)
            specificities = []
            for i in range(cm.shape<sup>0</sup>):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            return np.mean(specificities)
    
    def _calculate_npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate negative predictive value."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fn) if (tn + fn) > 0 else 0.0
        else:
            # Multi-class NPV (average)
            npvs = []
            for i in range(cm.shape<sup>0</sup>):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fn = np.sum(cm[i, :]) - cm[i, i]
                npvs.append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
            return np.mean(npvs)
    
    def _calculate_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for performance metrics."""
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        n_samples = len(y_true)
        
        bootstrap_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics
            bootstrap_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            bootstrap_metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, average='weighted'))
            bootstrap_metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, average='weighted'))
            bootstrap_metrics['f1_score'].append(f1_score(y_true_boot, y_pred_boot, average='weighted'))
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_metrics = {}
        
        for metric, values in bootstrap_metrics.items():
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            ci_metrics[f'{metric}_ci'] = (lower, upper)
        
        return ci_metrics
    
    def _conduct_robustness_testing(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        reference_standard: np.ndarray
    ) -> Dict[str, Any]:
        """Conduct robustness testing."""
        
        robustness_results = {
            'noise_robustness': self._test_noise_robustness(model, test_data, reference_standard),
            'adversarial_robustness': self._test_adversarial_robustness(model, test_data, reference_standard),
            'distribution_shift': self._test_distribution_shift(model, test_data, reference_standard)
        }
        
        return robustness_results
    
    def _test_noise_robustness(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        reference_standard: np.ndarray
    ) -> Dict[str, float]:
        """Test robustness to noise."""
        
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        noise_results = {}
        
        model.eval()
        
        for noise_level in noise_levels:
            predictions = []
            
            with torch.no_grad():
                for batch_data, _ in test_data:
                    # Add Gaussian noise
                    noisy_data = batch_data + torch.randn_like(batch_data) * noise_level
                    outputs = model(noisy_data)
                    preds = torch.argmax(outputs, dim=1)
                    predictions.extend(preds.cpu().numpy())
            
            predictions = np.array(predictions)
            accuracy = accuracy_score(reference_standard, predictions)
            noise_results[f'noise_{noise_level}'] = accuracy
        
        return noise_results
    
    def _test_adversarial_robustness(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        reference_standard: np.ndarray
    ) -> Dict[str, float]:
        """Test robustness to adversarial examples (simplified)."""
        
        # Simplified adversarial testing using FGSM
        epsilon_values = [0.01, 0.05, 0.1]
        adversarial_results = {}
        
        model.eval()
        
        for epsilon in epsilon_values:
            predictions = []
            
            for batch_data, batch_targets in test_data:
                batch_data.requires_grad = True
                
                outputs = model(batch_data)
                loss = nn.CrossEntropyLoss()(outputs, batch_targets)
                
                model.zero_grad()
                loss.backward()
                
                # Generate adversarial examples
                data_grad = batch_data.grad.data
                perturbed_data = batch_data + epsilon * data_grad.sign()
                
                # Test on adversarial examples
                with torch.no_grad():
                    adv_outputs = model(perturbed_data)
                    adv_preds = torch.argmax(adv_outputs, dim=1)
                    predictions.extend(adv_preds.cpu().numpy())
            
            predictions = np.array(predictions)
            accuracy = accuracy_score(reference_standard, predictions)
            adversarial_results[f'fgsm_{epsilon}'] = accuracy
        
        return adversarial_results
    
    def _test_distribution_shift(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        reference_standard: np.ndarray
    ) -> Dict[str, float]:
        """Test robustness to distribution shift (simplified)."""
        
        # Simplified distribution shift testing
        shift_results = {}
        
        # Test with different data transformations
        transforms = {
            'brightness': lambda x: torch.clamp(x + 0.2, 0, 1),
            'contrast': lambda x: torch.clamp(x * 1.5, 0, 1),
            'blur': lambda x: x  # Simplified - would apply blur filter
        }
        
        model.eval()
        
        for transform_name, transform_func in transforms.items():
            predictions = []
            
            with torch.no_grad():
                for batch_data, _ in test_data:
                    transformed_data = transform_func(batch_data)
                    outputs = model(transformed_data)
                    preds = torch.argmax(outputs, dim=1)
                    predictions.extend(preds.cpu().numpy())
            
            predictions = np.array(predictions)
            accuracy = accuracy_score(reference_standard, predictions)
            shift_results[transform_name] = accuracy
        
        return shift_results
    
    def _conduct_bias_assessment(
        self,
        predictions: np.ndarray,
        reference_standard: np.ndarray,
        test_data: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Conduct bias assessment across subgroups."""
        
        # Simplified bias assessment
        # In practice, this would analyze performance across demographic groups
        
        bias_results = {
            'overall_performance': accuracy_score(reference_standard, predictions),
            'subgroup_analysis': {
                'note': 'Subgroup analysis requires demographic data not available in this example'
            },
            'fairness_metrics': {
                'demographic_parity': 'Not calculated - requires sensitive attributes',
                'equalized_odds': 'Not calculated - requires sensitive attributes'
            }
        }
        
        return bias_results
    
    def _conduct_failure_mode_analysis(
        self,
        model: nn.Module,
        test_data: torch.utils.data.DataLoader,
        reference_standard: np.ndarray
    ) -> Dict[str, Any]:
        """Conduct failure mode analysis."""
        
        model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for batch_data, _ in test_data:
                outputs = model(batch_data)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                confs = torch.max(probs, dim=1)<sup>0</sup>
                
                predictions.extend(preds.cpu().numpy())
                confidences.extend(confs.cpu().numpy())
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Identify failure cases
        failures = predictions != reference_standard
        
        failure_analysis = {
            'failure_rate': np.mean(failures),
            'failure_confidence_distribution': {
                'mean': np.mean(confidences[failures]) if np.any(failures) else 0,
                'std': np.std(confidences[failures]) if np.any(failures) else 0
            },
            'success_confidence_distribution': {
                'mean': np.mean(confidences[~failures]) if np.any(~failures) else 0,
                'std': np.std(confidences[~failures]) if np.any(~failures) else 0
            },
            'low_confidence_threshold': np.percentile(confidences, 10),
            'high_confidence_threshold': np.percentile(confidences, 90)
        }
        
        return failure_analysis
    
    def generate_evidence_package(
        self,
        device_name: str,
        analytical_studies: List[str],
        clinical_studies: List[str],
        utility_studies: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive evidence package for regulatory submission."""
        
        package_id = f"EVIDENCE_{uuid.uuid4().hex[:8].upper()}"
        
        evidence_package = {
            'package_id': package_id,
            'device_name': device_name,
            'generation_date': datetime.now().isoformat(),
            'analytical_evidence': self._compile_analytical_evidence(analytical_studies),
            'clinical_evidence': self._compile_clinical_evidence(clinical_studies),
            'utility_evidence': self._compile_utility_evidence(utility_studies),
            'regulatory_summary': self._generate_regulatory_summary(
                analytical_studies, clinical_studies, utility_studies
            )
        }
        
        # Store evidence package
        self.evidence_packages[package_id] = evidence_package
        
        logger.info(f"Generated evidence package: {package_id}")
        
        return evidence_package
    
    def _compile_analytical_evidence(self, study_ids: List[str]) -> Dict[str, Any]:
        """Compile analytical evidence from studies."""
        
        analytical_evidence = {
            'studies_included': study_ids,
            'summary_metrics': {},
            'robustness_assessment': {},
            'bias_assessment': {},
            'failure_mode_analysis': {}
        }
        
        # In practice, this would aggregate results from actual studies
        return analytical_evidence
    
    def _compile_clinical_evidence(self, study_ids: List[str]) -> Dict[str, Any]:
        """Compile clinical evidence from studies."""
        
        clinical_evidence = {
            'studies_included': study_ids,
            'clinical_accuracy': {},
            'clinical_utility': {},
            'human_ai_interaction': {},
            'workflow_integration': {}
        }
        
        return clinical_evidence
    
    def _compile_utility_evidence(self, study_ids: List[str]) -> Dict[str, Any]:
        """Compile clinical utility evidence from studies."""
        
        utility_evidence = {
            'studies_included': study_ids,
            'outcome_improvements': {},
            'workflow_efficiency': {},
            'economic_impact': {},
            'patient_satisfaction': {}
        }
        
        return utility_evidence
    
    def _generate_regulatory_summary(
        self,
        analytical_studies: List[str],
        clinical_studies: List[str],
        utility_studies: List[str]
    ) -> Dict[str, Any]:
        """Generate regulatory summary for submission."""
        
        summary = {
            'evidence_strength': self._assess_evidence_strength(
                analytical_studies, clinical_studies, utility_studies
            ),
            'regulatory_readiness': self._assess_regulatory_readiness(),
            'recommendations': self._generate_recommendations(),
            'risk_benefit_assessment': self._conduct_risk_benefit_assessment()
        }
        
        return summary
    
    def _assess_evidence_strength(
        self,
        analytical_studies: List[str],
        clinical_studies: List[str],
        utility_studies: List[str]
    ) -> str:
        """Assess overall evidence strength."""
        
        # Simplified assessment
        total_studies = len(analytical_studies) + len(clinical_studies) + len(utility_studies)
        
        if total_studies >= 5 and len(clinical_studies) >= 2:
            return "Strong"
        elif total_studies >= 3 and len(clinical_studies) >= 1:
            return "Moderate"
        else:
            return "Limited"
    
    def _assess_regulatory_readiness(self) -> str:
        """Assess regulatory readiness."""
        # Simplified assessment
        return "Ready for submission"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for regulatory submission."""
        return [
            "Ensure all clinical studies are completed",
            "Prepare comprehensive risk management documentation",
            "Develop post-market surveillance plan",
            "Prepare user training materials"
        ]
    
    def _conduct_risk_benefit_assessment(self) -> Dict[str, Any]:
        """Conduct risk-benefit assessment."""
        return {
            'clinical_benefits': ['Improved diagnostic accuracy', 'Reduced time to diagnosis'],
            'clinical_risks': ['False positive/negative results', 'Over-reliance on AI'],
            'risk_mitigation': ['Human oversight requirements', 'Uncertainty quantification'],
            'overall_assessment': 'Benefits outweigh risks with appropriate safeguards'
        }

class RegulatoryComplianceFramework:
    """
    Comprehensive regulatory compliance framework for healthcare AI systems.
    
    This class integrates SaMD classification, clinical evidence management,
    and compliance tracking to provide end-to-end regulatory support.
    """
    
    def __init__(self, database_path: str = "regulatory_compliance.db"):
        """Initialize regulatory compliance framework."""
        
        self.database_path = database_path
        self.samd_classifier = SaMDClassifier()
        self.evidence_manager = ClinicalEvidenceManager(database_path)
        
        # Compliance tracking
        self.submissions = {}
        self.change_control_plans = {}
        self.compliance_assessments = {}
        
        # Initialize compliance database
        self._initialize_compliance_database()
        
        logger.info("Initialized regulatory compliance framework")
    
    def _initialize_compliance_database(self):
        """Initialize compliance tracking database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                submission_id TEXT PRIMARY KEY,
                submission_type TEXT,
                device_name TEXT,
                risk_class TEXT,
                samd_category TEXT,
                submission_date TEXT,
                status TEXT,
                approval_date TEXT,
                created_at TEXT
            )
        ''')
        
        # Create change control plans table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS change_control_plans (
                pccp_id TEXT PRIMARY KEY,
                device_name TEXT,
                modification_types TEXT,
                performance_bounds TEXT,
                effective_date TEXT,
                status TEXT,
                created_at TEXT
            )
        ''')
        
        # Create compliance assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_assessments (
                assessment_id TEXT PRIMARY KEY,
                device_name TEXT,
                assessment_date TEXT,
                overall_status TEXT,
                findings TEXT,
                recommendations TEXT,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def classify_and_plan_device(
        self,
        device_name: str,
        device_description: str,
        intended_use: str,
        healthcare_situation: HealthcareSituation,
        healthcare_decision: HealthcareDecision
    ) -> Dict[str, Any]:
        """Classify device and create regulatory plan."""
        
        # Classify device
        classification = self.samd_classifier.classify_device(
            healthcare_situation, healthcare_decision, device_description, intended_use
        )
        
        # Create regulatory plan
        regulatory_plan = self._create_regulatory_plan(classification)
        
        # Generate compliance roadmap
        compliance_roadmap = self._generate_compliance_roadmap(classification)
        
        result = {
            'device_name': device_name,
            'classification': classification,
            'regulatory_plan': regulatory_plan,
            'compliance_roadmap': compliance_roadmap,
            'next_steps': self._identify_next_steps(classification)
        }
        
        logger.info(f"Classified and planned device: {device_name}")
        
        return result
    
    def _create_regulatory_plan(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive regulatory plan."""
        
        samd_category = classification['samd_category']
        fda_class = classification['fda_class']
        
        plan = {
            'regulatory_pathway': classification['regulatory_pathway'],
            'submission_timeline': self._estimate_submission_timeline(fda_class),
            'clinical_evidence_plan': self._create_evidence_plan(classification),
            'quality_system_requirements': self._identify_qms_requirements(samd_category),
            'post_market_obligations': self._identify_post_market_requirements(fda_class)
        }
        
        return plan
    
    def _estimate_submission_timeline(self, fda_class: RiskClass) -> Dict[str, str]:
        """Estimate regulatory submission timeline."""
        
        timelines = {
            RiskClass.CLASS_I: {
                'preparation': '3-6 months',
                'submission': 'Not required',
                'review': 'N/A',
                'total': '3-6 months'
            },
            RiskClass.CLASS_II: {
                'preparation': '6-12 months',
                'submission': '510(k)',
                'review': '3-6 months',
                'total': '9-18 months'
            },
            RiskClass.CLASS_III: {
                'preparation': '12-24 months',
                'submission': 'PMA',
                'review': '6-12 months',
                'total': '18-36 months'
            }
        }
        
        return timelines.get(fda_class, timelines[RiskClass.CLASS_II])
    
    def _create_evidence_plan(self, classification: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create clinical evidence plan."""
        
        evidence_requirements = classification['evidence_requirements']
        
        evidence_plan = {
            'analytical_studies': [],
            'clinical_studies': [],
            'utility_studies': []
        }
        
        # Analytical studies (always required)
        evidence_plan['analytical_studies'] = [
            'Algorithm performance validation',
            'Robustness testing',
            'Bias assessment',
            'Failure mode analysis'
        ]
        
        # Clinical studies (based on risk category)
        if evidence_requirements['clinical_validation']:
            evidence_plan['clinical_studies'] = [
                'Multi-site clinical validation',
                'Reader study',
                'Subgroup analysis',
                'Human-AI interaction study'
            ]
        
        # Utility studies (for high-risk devices)
        if evidence_requirements['clinical_utility']:
            evidence_plan['utility_studies'] = [
                'Clinical outcome study',
                'Workflow impact assessment',
                'Economic evaluation'
            ]
        
        return evidence_plan
    
    def _identify_qms_requirements(self, samd_category: SaMDCategory) -> List[str]:
        """Identify quality management system requirements."""
        
        base_requirements = [
            'ISO 13485 compliance',
            'IEC 62304 software lifecycle',
            'ISO 14971 risk management',
            'Design controls (21 CFR 820.30)'
        ]
        
        if samd_category in [SaMDCategory.CLASS_III, SaMDCategory.CLASS_IV]:
            base_requirements.extend([
                'Clinical evaluation procedures',
                'Post-market surveillance system',
                'Adverse event reporting procedures'
            ])
        
        return base_requirements
    
    def _identify_post_market_requirements(self, fda_class: RiskClass) -> List[str]:
        """Identify post-market surveillance requirements."""
        
        requirements = [
            'Medical device reporting (MDR)',
            'Establishment registration',
            'Device listing'
        ]
        
        if fda_class in [RiskClass.CLASS_II, RiskClass.CLASS_III]:
            requirements.extend([
                'Post-market surveillance studies',
                'Periodic safety updates',
                'Performance monitoring'
            ])
        
        if fda_class == RiskClass.CLASS_III:
            requirements.extend([
                'Post-approval studies',
                'Annual reports',
                'Risk evaluation and mitigation strategies (REMS)'
            ])
        
        return requirements
    
    def _generate_compliance_roadmap(self, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate compliance roadmap with milestones."""
        
        roadmap = [
            {
                'milestone': 'Device Classification',
                'status': 'completed',
                'timeline': 'Month 1',
                'deliverables': ['SaMD classification', 'Risk assessment']
            },
            {
                'milestone': 'Quality System Implementation',
                'status': 'pending',
                'timeline': 'Months 2-4',
                'deliverables': ['QMS procedures', 'Design controls', 'Risk management']
            },
            {
                'milestone': 'Clinical Evidence Generation',
                'status': 'pending',
                'timeline': 'Months 5-12',
                'deliverables': ['Analytical validation', 'Clinical studies', 'Evidence package']
            },
            {
                'milestone': 'Regulatory Submission',
                'status': 'pending',
                'timeline': 'Month 13',
                'deliverables': ['510(k)/PMA submission', 'Supporting documentation']
            },
            {
                'milestone': 'FDA Review',
                'status': 'pending',
                'timeline': 'Months 14-18',
                'deliverables': ['Response to FDA questions', 'Additional studies if needed']
            },
            {
                'milestone': 'Market Authorization',
                'status': 'pending',
                'timeline': 'Month 18',
                'deliverables': ['FDA clearance/approval', 'Labeling finalization']
            }
        ]
        
        return roadmap
    
    def _identify_next_steps(self, classification: Dict[str, Any]) -> List[str]:
        """Identify immediate next steps."""
        
        next_steps = [
            'Implement quality management system',
            'Develop clinical evidence plan',
            'Conduct analytical validation studies',
            'Prepare regulatory submission strategy'
        ]
        
        if classification['samd_category'] in [SaMDCategory.CLASS_III, SaMDCategory.CLASS_IV]:
            next_steps.insert(1, 'Engage with FDA for pre-submission meeting')
        
        return next_steps
    
    def create_change_control_plan(
        self,
        device_name: str,
        modification_types: List[str],
        performance_bounds: Dict[str, Tuple[float, float]],
        validation_requirements: List[str]
    ) -> ChangeControlPlan:
        """Create predetermined change control plan."""
        
        pccp_id = f"PCCP_{uuid.uuid4().hex[:8].upper()}"
        
        pccp = ChangeControlPlan(
            pccp_id=pccp_id,
            device_name=device_name,
            modification_types=modification_types,
            modification_protocols={},
            performance_bounds=performance_bounds,
            validation_requirements=validation_requirements,
            risk_mitigation_strategies=[],
            monitoring_plan="Continuous performance monitoring",
            approval_authority="Quality Assurance",
            effective_date=datetime.now()
        )
        
        # Store PCCP
        self.change_control_plans[pccp_id] = pccp
        self._save_pccp_to_database(pccp)
        
        logger.info(f"Created change control plan: {pccp_id}")
        
        return pccp
    
    def _save_pccp_to_database(self, pccp: ChangeControlPlan):
        """Save PCCP to database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO change_control_plans 
            (pccp_id, device_name, modification_types, performance_bounds,
             effective_date, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            pccp.pccp_id,
            pccp.device_name,
            json.dumps(pccp.modification_types),
            json.dumps({k: list(v) for k, v in pccp.performance_bounds.items()}),
            pccp.effective_date.isoformat(),
            'active',
            pccp.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def conduct_compliance_assessment(
        self,
        device_name: str,
        assessor: str
    ) -> ComplianceAssessment:
        """Conduct comprehensive compliance assessment."""
        
        assessment_id = f"ASSESS_{uuid.uuid4().hex[:8].upper()}"
        
        # Assess different compliance areas
        compliance_areas = {
            'quality_management': self._assess_qms_compliance(),
            'clinical_evidence': self._assess_clinical_evidence_compliance(),
            'labeling': self._assess_labeling_compliance(),
            'post_market_surveillance': self._assess_post_market_compliance(),
            'change_control': self._assess_change_control_compliance()
        }
        
        # Generate findings and recommendations
        findings = self._generate_compliance_findings(compliance_areas)
        recommendations = self._generate_compliance_recommendations(compliance_areas)
        
        # Determine overall status
        overall_status = self._determine_overall_compliance_status(compliance_areas)
        
        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            device_name=device_name,
            assessment_date=datetime.now(),
            assessor=assessor,
            compliance_areas=compliance_areas,
            findings=findings,
            recommendations=recommendations,
            corrective_actions=[],
            next_assessment_date=datetime.now() + timedelta(days=365),
            overall_status=overall_status
        )
        
        # Store assessment
        self.compliance_assessments[assessment_id] = assessment
        self._save_assessment_to_database(assessment)
        
        logger.info(f"Conducted compliance assessment: {assessment_id}")
        
        return assessment
    
    def _assess_qms_compliance(self) -> ComplianceStatus:
        """Assess quality management system compliance."""
        # Simplified assessment
        return ComplianceStatus.COMPLIANT
    
    def _assess_clinical_evidence_compliance(self) -> ComplianceStatus:
        """Assess clinical evidence compliance."""
        # Simplified assessment
        return ComplianceStatus.COMPLIANT
    
    def _assess_labeling_compliance(self) -> ComplianceStatus:
        """Assess labeling compliance."""
        # Simplified assessment
        return ComplianceStatus.PENDING
    
    def _assess_post_market_compliance(self) -> ComplianceStatus:
        """Assess post-market surveillance compliance."""
        # Simplified assessment
        return ComplianceStatus.COMPLIANT
    
    def _assess_change_control_compliance(self) -> ComplianceStatus:
        """Assess change control compliance."""
        # Simplified assessment
        return ComplianceStatus.COMPLIANT
    
    def _generate_compliance_findings(self, compliance_areas: Dict[str, ComplianceStatus]) -> List[str]:
        """Generate compliance findings."""
        findings = []
        
        for area, status in compliance_areas.items():
            if status == ComplianceStatus.NON_COMPLIANT:
                findings.append(f"Non-compliance identified in {area}")
            elif status == ComplianceStatus.PENDING:
                findings.append(f"Pending compliance activities in {area}")
        
        if not findings:
            findings.append("No significant compliance issues identified")
        
        return findings
    
    def _generate_compliance_recommendations(self, compliance_areas: Dict[str, ComplianceStatus]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        for area, status in compliance_areas.items():
            if status != ComplianceStatus.COMPLIANT:
                recommendations.append(f"Address compliance gaps in {area}")
        
        recommendations.extend([
            "Maintain regular compliance monitoring",
            "Update procedures based on regulatory changes",
            "Conduct periodic compliance training"
        ])
        
        return recommendations
    
    def _determine_overall_compliance_status(self, compliance_areas: Dict[str, ComplianceStatus]) -> ComplianceStatus:
        """Determine overall compliance status."""
        
        statuses = list(compliance_areas.values())
        
        if ComplianceStatus.NON_COMPLIANT in statuses:
            return ComplianceStatus.NON_COMPLIANT
        elif ComplianceStatus.PENDING in statuses:
            return ComplianceStatus.PENDING
        else:
            return ComplianceStatus.COMPLIANT
    
    def _save_assessment_to_database(self, assessment: ComplianceAssessment):
        """Save assessment to database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO compliance_assessments 
            (assessment_id, device_name, assessment_date, overall_status,
             findings, recommendations, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            assessment.assessment_id,
            assessment.device_name,
            assessment.assessment_date.isoformat(),
            assessment.overall_status.value,
            json.dumps(assessment.findings),
            json.dumps(assessment.recommendations),
            assessment.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def generate_regulatory_report(self, device_name: str) -> Dict[str, Any]:
        """Generate comprehensive regulatory report."""
        
        report = {
            'device_name': device_name,
            'report_date': datetime.now().isoformat(),
            'regulatory_status': self._get_regulatory_status(device_name),
            'compliance_summary': self._get_compliance_summary(device_name),
            'evidence_summary': self._get_evidence_summary(device_name),
            'change_control_summary': self._get_change_control_summary(device_name),
            'recommendations': self._get_regulatory_recommendations(device_name)
        }
        
        return report
    
    def _get_regulatory_status(self, device_name: str) -> Dict[str, Any]:
        """Get regulatory status for device."""
        # Simplified implementation
        return {
            'classification': 'Class II',
            'submission_status': 'In preparation',
            'approval_status': 'Pending',
            'market_status': 'Pre-market'
        }
    
    def _get_compliance_summary(self, device_name: str) -> Dict[str, Any]:
        """Get compliance summary for device."""
        # Simplified implementation
        return {
            'overall_status': 'Compliant',
            'last_assessment': '2024-01-15',
            'next_assessment': '2025-01-15',
            'open_findings': 0
        }
    
    def _get_evidence_summary(self, device_name: str) -> Dict[str, Any]:
        """Get evidence summary for device."""
        # Simplified implementation
        return {
            'analytical_studies': 3,
            'clinical_studies': 2,
            'utility_studies': 1,
            'evidence_strength': 'Strong'
        }
    
    def _get_change_control_summary(self, device_name: str) -> Dict[str, Any]:
        """Get change control summary for device."""
        # Simplified implementation
        return {
            'active_pccps': 1,
            'approved_modifications': 0,
            'pending_modifications': 0
        }
    
    def _get_regulatory_recommendations(self, device_name: str) -> List[str]:
        """Get regulatory recommendations for device."""
        return [
            "Complete remaining clinical studies",
            "Prepare 510(k) submission package",
            "Schedule FDA pre-submission meeting",
            "Finalize labeling and instructions for use"
        ]

## Bibliography and References

### Foundational Regulatory Literature

1. **U.S. Food and Drug Administration.** (2021). Artificial intelligence/machine learning (AI/ML)-based software as a medical device (SaMD) action plan. *FDA Guidance Document*. [FDA AI/ML action plan]

2. **U.S. Food and Drug Administration.** (2022). Marketing submission recommendations for a predetermined change control plan for artificial intelligence/machine learning (AI/ML)-enabled device software functions. *FDA Draft Guidance*. [PCCP guidance]

3. **International Medical Device Regulators Forum.** (2014). Software as a Medical Device (SaMD): Key definitions. *IMDRF/SaMD WG/N10*. [SaMD definitions]

4. **International Medical Device Regulators Forum.** (2015). Software as a Medical Device (SaMD): Application of quality management system. *IMDRF/SaMD WG/N23*. [SaMD QMS requirements]

### Clinical Evidence and Validation

5. **U.S. Food and Drug Administration.** (2019). Clinical evaluation of software functions. *FDA Guidance Document*. [Clinical evaluation guidance]

6. **Vasey, B., Nagendran, M., Campbell, B., Clifton, D. A., Collins, G. S., Denaxas, S., ... & Vollmer, S. J.** (2022). Reporting guideline for the early stage clinical evaluation of decision support systems driven by artificial intelligence (DECIDE-AI). *Nature Medicine*, 28(5), 924-933. [DECIDE-AI reporting guidelines]

7. **Liu, X., Faes, L., Kale, A. U., Wagner, S. K., Fu, D. J., Bruynseels, A., ... & Denniston, A. K.** (2019). A comparison of deep learning performance against health-care professionals in detecting diseases from medical imaging: a systematic review and meta-analysis. *The Lancet Digital Health*, 1(6), e271-e297. [AI performance comparison]

8. **Nagendran, M., Chen, Y., Lovejoy, C. A., Gordon, A. C., Komorowski, M., Harvey, H., ... & Lio, P.** (2020). Artificial intelligence versus clinicians: systematic review of design, reporting standards, and claims of deep learning studies. *BMJ*, 368, m689. [AI study design review]

### International Regulatory Frameworks

9. **European Commission.** (2017). Medical Device Regulation (MDR) 2017/745. *Official Journal of the European Union*. [EU MDR]

10. **Health Canada.** (2019). Guidance document: Software as a medical device (SaMD). *Health Canada Guidance Document*. [Health Canada SaMD guidance]

11. **Pharmaceuticals and Medical Devices Agency (PMDA).** (2021). Basic principles on artificial intelligence/machine learning-based medical device software. *PMDA Guidance*. [Japan AI guidance]

12. **Therapeutic Goods Administration (TGA).** (2020). Software as a medical device: Guidance for manufacturers. *TGA Guidance Document*. [Australia SaMD guidance]

### Quality Management and Standards

13. **International Organization for Standardization.** (2016). Medical devices — Quality management systems — Requirements for regulatory purposes (ISO 13485:2016). *ISO Standard*. [ISO 13485]

14. **International Electrotechnical Commission.** (2015). Medical device software — Software life cycle processes (IEC 62304:2006+AMD1:2015). *IEC Standard*. [IEC 62304]

15. **International Organization for Standardization.** (2019). Medical devices — Application of risk management to medical devices (ISO 14971:2019). *ISO Standard*. [ISO 14971]

16. **International Electrotechnical Commission.** (2020). Medical electrical equipment — Part 1-6: General requirements for basic safety and essential performance — Collateral standard: Usability (IEC 60601-1-6:2010+AMD1:2013+AMD2:2020). *IEC Standard*. [Usability standard]

### Post-Market Surveillance and Real-World Evidence

17. **U.S. Food and Drug Administration.** (2016). Postmarket management of cybersecurity in medical devices. *FDA Guidance Document*. [Cybersecurity post-market guidance]

18. **European Medicines Agency.** (2020). Guideline on registry-based studies. *EMA Guideline*. [Registry-based studies]

19. **Sherman, R. E., Anderson, S. A., Dal Pan, G. J., Gray, G. W., Gross, T., Hunter, N. L., ... & Califf, R. M.** (2016). Real-world evidence—what is it and what can it tell us? *New England Journal of Medicine*, 375(23), 2293-2297. [Real-world evidence]

20. **Eichler, H. G., Oye, K., Baird, L. G., Abadie, E., Brown, J., Drum, C. L., ... & Rasi, G.** (2012). Adaptive licensing: taking the right steps—a conversation between regulators. *Clinical Pharmacology & Therapeutics*, 91(3), 426-437. [Adaptive licensing concepts]

This chapter provides a comprehensive framework for navigating the complex regulatory landscape for healthcare AI systems. The implementations address the specific requirements of FDA SaMD frameworks, clinical evidence generation, and compliance management. The next chapter will explore clinical validation frameworks, building upon these regulatory concepts to address the specific methodologies for demonstrating AI system safety and effectiveness in clinical settings.
