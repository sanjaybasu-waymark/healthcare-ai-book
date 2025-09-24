# Chapter 11: Regulatory Compliance for Healthcare AI

## Learning Objectives

By the end of this chapter, readers will be able to:

1. **Navigate the complex regulatory landscape** for healthcare AI across multiple jurisdictions
2. **Implement FDA Software as Medical Device (SaMD) frameworks** for AI/ML-based medical devices
3. **Design compliance management systems** that ensure ongoing regulatory adherence
4. **Develop clinical evidence packages** that meet regulatory requirements for AI systems
5. **Establish quality management systems** compliant with ISO 13485 and other standards
6. **Manage post-market surveillance** and adverse event reporting for AI medical devices

## 11.1 Introduction to Healthcare AI Regulation

The regulatory landscape for healthcare artificial intelligence represents one of the most complex and rapidly evolving areas in medical technology. Unlike traditional medical devices with well-established regulatory pathways, AI systems present unique challenges that require new frameworks, guidelines, and approaches to ensure patient safety while fostering innovation.

Healthcare AI regulation encompasses multiple dimensions including **device classification**, **clinical validation requirements**, **quality management systems**, **post-market surveillance**, and **international harmonization efforts**. The dynamic nature of AI systems, which can learn and adapt over time, challenges traditional regulatory paradigms designed for static medical devices.

### 11.1.1 Regulatory Challenges Unique to AI

Healthcare AI systems present several regulatory challenges that distinguish them from traditional medical devices. **Algorithmic transparency** requirements conflict with the "black box" nature of many machine learning models, particularly deep learning systems. Regulators must balance the need for explainability with the clinical effectiveness that may come from complex, less interpretable models.

**Continuous learning systems** pose particular challenges as they can change behavior after regulatory approval. The FDA's concept of **predetermined change control plans** attempts to address this by allowing pre-approved modifications within specified bounds, but implementation remains complex.

**Data dependency** means that AI system performance is intrinsically linked to training data quality and representativeness. Regulatory frameworks must address how to evaluate and ensure appropriate data governance throughout the system lifecycle.

**Validation complexity** arises from the need to demonstrate not just clinical effectiveness but also robustness across diverse patient populations and clinical settings. Traditional clinical trial designs may be insufficient for evaluating AI systems that exhibit different performance characteristics across subgroups.

### 11.1.2 Global Regulatory Landscape

The global regulatory landscape for healthcare AI involves multiple agencies with varying approaches and requirements. The **United States Food and Drug Administration (FDA)** has been among the most active in developing AI-specific guidance, with frameworks for Software as Medical Device (SaMD) and AI/ML-based medical device software.

The **European Medicines Agency (EMA)** and **Medical Device Regulation (MDR)** in Europe provide a different regulatory framework that emphasizes conformity assessment and notified body involvement. The **CE marking** process for AI medical devices requires demonstration of compliance with essential requirements and harmonized standards.

**Health Canada**, **Japan's Pharmaceuticals and Medical Devices Agency (PMDA)**, and other national regulators are developing their own approaches, creating a complex international landscape that requires careful navigation for global deployment.

The **International Medical Device Regulators Forum (IMDRF)** works to harmonize regulatory approaches globally, with working groups specifically focused on Software as Medical Device and AI considerations.

### 11.1.3 Risk-Based Classification Systems

Healthcare AI systems are typically classified using risk-based frameworks that determine the level of regulatory oversight required. The **FDA's risk classification system** categorizes medical devices into Class I (low risk), Class II (moderate risk), and Class III (high risk) based on the potential for harm if the device fails.

For Software as Medical Device, the **IMDRF SaMD framework** provides a risk categorization based on two factors: the **healthcare decision** (inform, drive, diagnose, treat) and the **healthcare situation** (critical, serious, non-serious). This creates a matrix that determines the appropriate level of clinical evidence and quality management requirements.

**AI-specific risk factors** include the degree of autonomy, the clinical context of use, the availability of human oversight, and the consequences of incorrect outputs. These factors influence both the classification and the specific requirements for clinical validation and post-market monitoring.

## 11.2 FDA Regulatory Framework for AI/ML Medical Devices

### 11.2.1 Software as Medical Device (SaMD) Classification

The FDA's approach to regulating AI/ML-based medical devices builds upon the Software as Medical Device framework developed by the International Medical Device Regulators Forum. This framework provides a systematic approach to classifying software based on its intended use and the risk associated with its clinical application.

The **SaMD risk categorization** considers two primary dimensions: the **state of the healthcare situation** (critical, serious, non-serious) and the **healthcare decision** (treat, diagnose, drive, inform). Critical situations involve immediate life-threatening or serious deteriorating conditions, while serious situations involve conditions requiring timely intervention.

Healthcare decisions range from **informing** clinical management to **treating** patients directly. The intersection of these dimensions determines the risk category and corresponding regulatory requirements:

- **Class IV (High Risk)**: SaMD intended to treat or diagnose in critical healthcare situations
- **Class III (Moderate-High Risk)**: SaMD intended to drive clinical management in critical situations or treat/diagnose in serious situations  
- **Class II (Moderate-Low Risk)**: SaMD intended to inform clinical management in critical situations or drive management in serious situations
- **Class I (Low Risk)**: SaMD intended to inform clinical management in serious or non-serious situations

### 11.2.2 Predetermined Change Control Plans

One of the most significant innovations in AI regulation is the FDA's **Predetermined Change Control Plan (PCCP)** framework, which allows manufacturers to make specified modifications to their AI/ML systems without requiring new regulatory submissions. This approach recognizes the unique nature of AI systems that may need to adapt and improve over time.

The PCCP framework requires manufacturers to define in advance:

- **Specific types of modifications** that may be made (e.g., retraining with new data, algorithm updates)
- **Modification protocols** that describe how changes will be implemented and validated
- **Impact assessment procedures** to evaluate the effect of modifications on safety and effectiveness
- **Risk mitigation strategies** to address potential negative impacts
- **Monitoring and reporting requirements** for tracking the performance of modified systems

The mathematical framework for change control can be expressed through **performance bounds** that define acceptable ranges for key metrics:

$$P_{new}(m) \geq P_{baseline}(m) - \delta_m$$

Where $P_{new}(m)$ represents the performance of the modified system on metric $m$, $P_{baseline}(m)$ is the baseline performance, and $\delta_m$ is the maximum allowable degradation for that metric.

### 11.2.3 Clinical Evidence Requirements

The FDA requires **clinical evidence** to demonstrate the safety and effectiveness of AI/ML medical devices, with the level of evidence proportional to the risk classification. High-risk devices typically require **prospective clinical studies**, while lower-risk devices may rely on **retrospective analyses** or **literature reviews**.

**Clinical validation** for AI systems must address several unique considerations:

- **Performance across subgroups** to ensure equitable outcomes across diverse patient populations
- **Robustness to data variations** including different imaging protocols, laboratory methods, or clinical practices
- **Human-AI interaction** studies to evaluate how clinicians use the AI system in practice
- **Failure mode analysis** to understand how the system behaves when encountering edge cases or adversarial inputs

The **clinical evaluation framework** typically includes:

1. **Analytical validation**: Demonstrating that the algorithm performs as intended on reference datasets
2. **Clinical validation**: Showing that the algorithm's output is clinically meaningful and accurate
3. **Clinical utility**: Proving that use of the algorithm improves patient outcomes or clinical workflow

### 11.2.4 Quality Management System Requirements

AI/ML medical devices must comply with **Quality System Regulation (QSR)** requirements under 21 CFR Part 820, with additional considerations for software development and AI-specific processes. The quality management system must address the entire AI system lifecycle from data collection through post-market monitoring.

**Design controls** for AI systems must include:

- **Data management procedures** ensuring training data quality, representativeness, and traceability
- **Algorithm development processes** with appropriate validation and verification activities
- **Risk management** following ISO 14971 with AI-specific risk analysis
- **Software lifecycle processes** compliant with IEC 62304 for medical device software

**Configuration management** becomes particularly important for AI systems due to their dependence on training data, model parameters, and software environments. The quality system must ensure that all components are properly controlled and that changes are managed through appropriate change control procedures.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

import logging
from datetime import datetime, timedelta
import json
import joblib
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import uuid
import os
import sqlite3
from pathlib import Path

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
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ClinicalStudy:
    """Clinical study record."""
    study_id: str
    study_type: str  # "prospective", "retrospective", "literature"
    study_design: str
    primary_endpoint: str
    secondary_endpoints: List[str]
    sample_size: int
    study_population: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    statistical_plan: str
    results: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AdverseEvent:
    """Adverse event record for post-market surveillance."""
    event_id: str
    device_id: str
    event_type: str
    severity: str  # "minor", "moderate", "severe", "life_threatening"
    description: str
    patient_demographics: Dict[str, Any]
    clinical_context: str
    root_cause: Optional[str] = None
    corrective_actions: List[str] = field(default_factory=list)
    report_date: datetime = field(default_factory=datetime.now)
    regulatory_reporting_required: bool = True

@dataclass
class QualityRecord:
    """Quality management system record."""
    record_id: str
    record_type: str  # "design_control", "risk_management", "validation", etc.
    description: str
    responsible_person: str
    review_date: datetime
    approval_date: Optional[datetime] = None
    status: ComplianceStatus = ComplianceStatus.PENDING
    attachments: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class RegulatoryComplianceFramework:
    """
    Comprehensive regulatory compliance framework for healthcare AI systems.
    
    This class implements FDA and international regulatory requirements for
    AI/ML-based medical devices, including SaMD classification, clinical
    validation, quality management, and post-market surveillance.
    
    Based on FDA guidance documents:
    - Software as Medical Device (SaMD): Clinical Evaluation (2017)
    - Artificial Intelligence/Machine Learning (AI/ML)-Based Medical Device Software (2021)
    - Predetermined Change Control Plans for AI/ML-Based SaMD (2023)
    
    And international standards:
    - ISO 13485: Medical devices - Quality management systems
    - ISO 14971: Medical devices - Application of risk management
    - IEC 62304: Medical device software - Software life cycle processes
    """
    
    def __init__(
        self,
        device_name: str,
        intended_use: str,
        healthcare_situation: HealthcareSituation,
        healthcare_decision: HealthcareDecision,
        compliance_database_path: str = "compliance.db"
    ):
        """
        Initialize regulatory compliance framework.
        
        Args:
            device_name: Name of the AI medical device
            intended_use: Intended use statement for the device
            healthcare_situation: Healthcare situation classification
            healthcare_decision: Healthcare decision classification
            compliance_database_path: Path to compliance database
        """
        self.device_name = device_name
        self.intended_use = intended_use
        self.healthcare_situation = healthcare_situation
        self.healthcare_decision = healthcare_decision
        
        # Determine SaMD classification
        self.samd_category = self._classify_samd()
        self.risk_class = self._determine_risk_class()
        
        # Initialize compliance database
        self.db_path = compliance_database_path
        self._initialize_database()
        
        # Compliance tracking
        self.regulatory_submissions = []
        self.clinical_studies = []
        self.adverse_events = []
        self.quality_records = []
        
        # Performance monitoring
        self.performance_history = []
        self.change_control_log = []
        
        logger.info(f"Regulatory compliance framework initialized for {device_name}")
        logger.info(f"SaMD Category: {self.samd_category.value}")
        logger.info(f"Risk Class: {self.risk_class.value}")
    
    def _classify_samd(self) -> SaMDCategory:
        """Classify Software as Medical Device according to IMDRF framework."""
        # SaMD classification matrix
        classification_matrix = {
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
        
        return classification_matrix[(self.healthcare_situation, self.healthcare_decision)]
    
    def _determine_risk_class(self) -> RiskClass:
        """Determine FDA risk class based on SaMD category."""
        risk_mapping = {
            SaMDCategory.CLASS_I: RiskClass.CLASS_I,
            SaMDCategory.CLASS_II: RiskClass.CLASS_II,
            SaMDCategory.CLASS_III: RiskClass.CLASS_II,
            SaMDCategory.CLASS_IV: RiskClass.CLASS_III
        }
        
        return risk_mapping[self.samd_category]
    
    def _initialize_database(self):
        """Initialize compliance tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for compliance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regulatory_submissions (
                submission_id TEXT PRIMARY KEY,
                submission_type TEXT,
                device_name TEXT,
                risk_class TEXT,
                samd_category TEXT,
                submission_date TEXT,
                status TEXT,
                predicate_devices TEXT,
                clinical_studies TEXT,
                substantial_equivalence BOOLEAN,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clinical_studies (
                study_id TEXT PRIMARY KEY,
                study_type TEXT,
                study_design TEXT,
                primary_endpoint TEXT,
                secondary_endpoints TEXT,
                sample_size INTEGER,
                study_population TEXT,
                inclusion_criteria TEXT,
                exclusion_criteria TEXT,
                statistical_plan TEXT,
                results TEXT,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adverse_events (
                event_id TEXT PRIMARY KEY,
                device_id TEXT,
                event_type TEXT,
                severity TEXT,
                description TEXT,
                patient_demographics TEXT,
                clinical_context TEXT,
                root_cause TEXT,
                corrective_actions TEXT,
                report_date TEXT,
                regulatory_reporting_required BOOLEAN
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_records (
                record_id TEXT PRIMARY KEY,
                record_type TEXT,
                description TEXT,
                responsible_person TEXT,
                review_date TEXT,
                approval_date TEXT,
                status TEXT,
                attachments TEXT,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_monitoring (
                record_id TEXT PRIMARY KEY,
                metric_name TEXT,
                metric_value REAL,
                measurement_date TEXT,
                patient_population TEXT,
                clinical_setting TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_regulatory_submission(
        self,
        submission_type: str,
        predicate_devices: List[str] = None,
        clinical_studies: List[str] = None,
        substantial_equivalence: bool = True
    ) -> RegulatorySubmission:
        """
        Create regulatory submission package.
        
        Args:
            submission_type: Type of submission (510k, PMA, De Novo, etc.)
            predicate_devices: List of predicate device identifiers
            clinical_studies: List of clinical study identifiers
            substantial_equivalence: Whether claiming substantial equivalence
            
        Returns:
            RegulatorySubmission object
        """
        submission_id = str(uuid.uuid4())
        
        submission = RegulatorySubmission(
            submission_id=submission_id,
            submission_type=submission_type,
            device_name=self.device_name,
            risk_class=self.risk_class,
            samd_category=self.samd_category,
            submission_date=datetime.now(),
            status="draft",
            predicate_devices=predicate_devices or [],
            clinical_studies=clinical_studies or [],
            substantial_equivalence=substantial_equivalence
        )
        
        self.regulatory_submissions.append(submission)
        self._save_submission_to_db(submission)
        
        logger.info(f"Created regulatory submission {submission_id} ({submission_type})")
        
        return submission
    
    def _save_submission_to_db(self, submission: RegulatorySubmission):
        """Save regulatory submission to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO regulatory_submissions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            submission.submission_id,
            submission.submission_type,
            submission.device_name,
            submission.risk_class.value,
            submission.samd_category.value,
            submission.submission_date.isoformat(),
            submission.status,
            json.dumps(submission.predicate_devices),
            json.dumps(submission.clinical_studies),
            submission.substantial_equivalence,
            submission.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def design_clinical_study(
        self,
        study_type: str,
        primary_endpoint: str,
        sample_size: int,
        study_population: str,
        inclusion_criteria: List[str],
        exclusion_criteria: List[str]
    ) -> ClinicalStudy:
        """
        Design clinical study for regulatory validation.
        
        Args:
            study_type: Type of study (prospective, retrospective, etc.)
            primary_endpoint: Primary endpoint for the study
            sample_size: Required sample size
            study_population: Description of study population
            inclusion_criteria: List of inclusion criteria
            exclusion_criteria: List of exclusion criteria
            
        Returns:
            ClinicalStudy object
        """
        study_id = str(uuid.uuid4())
        
        # Determine study design based on risk class
        if self.risk_class == RiskClass.CLASS_III:
            study_design = "Prospective, multi-center, randomized controlled trial"
        elif self.risk_class == RiskClass.CLASS_II:
            study_design = "Prospective or retrospective comparative study"
        else:
            study_design = "Retrospective analysis or literature review"
        
        # Generate statistical analysis plan
        statistical_plan = self._generate_statistical_plan(primary_endpoint, sample_size)
        
        study = ClinicalStudy(
            study_id=study_id,
            study_type=study_type,
            study_design=study_design,
            primary_endpoint=primary_endpoint,
            secondary_endpoints=self._generate_secondary_endpoints(),
            sample_size=sample_size,
            study_population=study_population,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            statistical_plan=statistical_plan
        )
        
        self.clinical_studies.append(study)
        self._save_study_to_db(study)
        
        logger.info(f"Designed clinical study {study_id} ({study_type})")
        
        return study
    
    def _generate_statistical_plan(self, primary_endpoint: str, sample_size: int) -> str:
        """Generate statistical analysis plan for clinical study."""
        plan = f"""
        Statistical Analysis Plan:
        
        Primary Endpoint: {primary_endpoint}
        Sample Size: {sample_size}
        
        Primary Analysis:
        - Statistical test: Appropriate for endpoint type (t-test, chi-square, etc.)
        - Significance level: α = 0.05
        - Power: 80% (β = 0.20)
        - Multiple comparison adjustment: Bonferroni correction if applicable
        
        Secondary Analyses:
        - Subgroup analyses by demographics and clinical characteristics
        - Sensitivity analyses for missing data
        - Non-inferiority analysis if applicable
        
        Interim Analysis:
        - Planned interim analysis at 50% enrollment
        - O'Brien-Fleming spending function for alpha adjustment
        
        Missing Data:
        - Multiple imputation for missing primary endpoint data
        - Sensitivity analysis with complete case analysis
        
        Statistical Software: R version 4.0 or later, SAS version 9.4 or later
        """
        
        return plan
    
    def _generate_secondary_endpoints(self) -> List[str]:
        """Generate appropriate secondary endpoints based on device type."""
        base_endpoints = [
            "Clinical utility assessment",
            "User acceptance and usability",
            "Time to diagnosis/decision",
            "Healthcare resource utilization"
        ]
        
        # Add AI-specific endpoints
        ai_endpoints = [
            "Algorithm performance across subgroups",
            "Human-AI interaction effectiveness",
            "False positive/negative rates",
            "Confidence calibration accuracy"
        ]
        
        return base_endpoints + ai_endpoints
    
    def _save_study_to_db(self, study: ClinicalStudy):
        """Save clinical study to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clinical_studies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            study.study_id,
            study.study_type,
            study.study_design,
            study.primary_endpoint,
            json.dumps(study.secondary_endpoints),
            study.sample_size,
            study.study_population,
            json.dumps(study.inclusion_criteria),
            json.dumps(study.exclusion_criteria),
            study.statistical_plan,
            json.dumps(study.results) if study.results else None,
            study.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def conduct_clinical_validation(
        self,
        model: nn.Module,
        validation_data: torch.Tensor,
        validation_labels: torch.Tensor,
        study_id: str,
        clinical_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Conduct clinical validation study for regulatory submission.
        
        Args:
            model: Trained AI model
            validation_data: Validation dataset
            validation_labels: Validation labels
            study_id: Clinical study identifier
            clinical_context: Clinical context information
            
        Returns:
            Clinical validation results
        """
        logger.info(f"Conducting clinical validation for study {study_id}")
        
        model.eval()
        device = next(model.parameters()).device
        
        validation_data = validation_data.to(device)
        validation_labels = validation_labels.to(device)
        
        # Generate predictions
        with torch.no_grad():
            predictions = model(validation_data)
            predicted_probs = F.softmax(predictions, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)
        
        # Calculate performance metrics
        y_true = validation_labels.cpu().numpy()
        y_pred = predicted_classes.cpu().numpy()
        y_prob = predicted_probs.cpu().numpy()
        
        # Primary performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calculate AUC for binary classification
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        
        # Subgroup analysis
        subgroup_results = self._conduct_subgroup_analysis(
            y_true, y_pred, y_prob, clinical_context
        )
        
        # Confidence calibration analysis
        calibration_results = self._analyze_confidence_calibration(y_true, y_prob)
        
        # Clinical utility analysis
        utility_results = self._analyze_clinical_utility(
            y_true, y_pred, y_prob, clinical_context
        )
        
        # Compile results
        results = {
            'study_id': study_id,
            'primary_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            },
            'subgroup_analysis': subgroup_results,
            'calibration_analysis': calibration_results,
            'clinical_utility': utility_results,
            'sample_size': len(y_true),
            'validation_date': datetime.now().isoformat(),
            'regulatory_compliance': self._assess_regulatory_compliance(accuracy, precision, recall)
        }
        
        # Update study record
        self._update_study_results(study_id, results)
        
        logger.info(f"Clinical validation completed: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
        
        return results
    
    def _conduct_subgroup_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        clinical_context: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Conduct subgroup analysis for regulatory compliance."""
        subgroup_results = {}
        
        # Analyze performance by demographic groups if available
        if 'demographics' in clinical_context:
            demographics = clinical_context['demographics']
            
            for group_name, group_indices in demographics.items():
                if len(group_indices) > 10:  # Minimum sample size for analysis
                    group_y_true = y_true[group_indices]
                    group_y_pred = y_pred[group_indices]
                    group_y_prob = y_prob[group_indices]
                    
                    group_accuracy = accuracy_score(group_y_true, group_y_pred)
                    group_precision = precision_score(group_y_true, group_y_pred, average='weighted')
                    group_recall = recall_score(group_y_true, group_y_pred, average='weighted')
                    
                    subgroup_results[group_name] = {
                        'accuracy': group_accuracy,
                        'precision': group_precision,
                        'recall': group_recall,
                        'sample_size': len(group_indices)
                    }
        
        return subgroup_results
    
    def _analyze_confidence_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze confidence calibration for regulatory assessment."""
        # For binary classification
        if y_prob.shape[1] == 2:
            confidences = y_prob[:, 1]
            predictions = (confidences > 0.5).astype(int)
            
            # Calculate calibration metrics
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return {
                'expected_calibration_error': calibration_error,
                'reliability_diagram_data': {
                    'bin_boundaries': bin_boundaries.tolist(),
                    'bin_accuracies': [],
                    'bin_confidences': [],
                    'bin_counts': []
                }
            }
        
        return {'message': 'Calibration analysis not applicable for multi-class with current implementation'}
    
    def _analyze_clinical_utility(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        clinical_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze clinical utility for regulatory assessment."""
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate clinical impact metrics
        if len(np.unique(y_true)) == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Calculate likelihood ratios
            lr_positive = sensitivity / (1 - specificity) if specificity < 1 else float('inf')
            lr_negative = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
            
            utility_metrics = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'positive_predictive_value': ppv,
                'negative_predictive_value': npv,
                'likelihood_ratio_positive': lr_positive,
                'likelihood_ratio_negative': lr_negative,
                'number_needed_to_diagnose': 1 / sensitivity if sensitivity > 0 else float('inf')
            }
        else:
            # Multi-class metrics
            utility_metrics = {
                'per_class_sensitivity': [],
                'per_class_specificity': [],
                'macro_averaged_metrics': {}
            }
        
        return utility_metrics
    
    def _assess_regulatory_compliance(
        self,
        accuracy: float,
        precision: float,
        recall: float
    ) -> Dict[str, bool]:
        """Assess regulatory compliance based on performance metrics."""
        # Define minimum performance thresholds based on risk class
        if self.risk_class == RiskClass.CLASS_III:
            min_accuracy = 0.90
            min_precision = 0.85
            min_recall = 0.85
        elif self.risk_class == RiskClass.CLASS_II:
            min_accuracy = 0.85
            min_precision = 0.80
            min_recall = 0.80
        else:
            min_accuracy = 0.80
            min_precision = 0.75
            min_recall = 0.75
        
        compliance = {
            'accuracy_compliant': accuracy >= min_accuracy,
            'precision_compliant': precision >= min_precision,
            'recall_compliant': recall >= min_recall,
            'overall_compliant': (
                accuracy >= min_accuracy and
                precision >= min_precision and
                recall >= min_recall
            )
        }
        
        return compliance
    
    def _update_study_results(self, study_id: str, results: Dict[str, Any]):
        """Update clinical study with results."""
        # Update in-memory record
        for study in self.clinical_studies:
            if study.study_id == study_id:
                study.results = results
                break
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE clinical_studies SET results = ? WHERE study_id = ?
        ''', (json.dumps(results), study_id))
        
        conn.commit()
        conn.close()
    
    def implement_quality_management_system(self) -> Dict[str, QualityRecord]:
        """
        Implement quality management system compliant with ISO 13485.
        
        Returns:
            Dictionary of quality records
        """
        logger.info("Implementing quality management system")
        
        quality_records = {}
        
        # Design controls
        design_control = QualityRecord(
            record_id=str(uuid.uuid4()),
            record_type="design_control",
            description="Design controls for AI/ML medical device software",
            responsible_person="Quality Manager",
            review_date=datetime.now(),
            status=ComplianceStatus.PENDING
        )
        quality_records['design_control'] = design_control
        
        # Risk management
        risk_management = QualityRecord(
            record_id=str(uuid.uuid4()),
            record_type="risk_management",
            description="Risk management file per ISO 14971",
            responsible_person="Risk Manager",
            review_date=datetime.now(),
            status=ComplianceStatus.PENDING
        )
        quality_records['risk_management'] = risk_management
        
        # Software lifecycle
        software_lifecycle = QualityRecord(
            record_id=str(uuid.uuid4()),
            record_type="software_lifecycle",
            description="Software lifecycle processes per IEC 62304",
            responsible_person="Software Manager",
            review_date=datetime.now(),
            status=ComplianceStatus.PENDING
        )
        quality_records['software_lifecycle'] = software_lifecycle
        
        # Clinical evaluation
        clinical_evaluation = QualityRecord(
            record_id=str(uuid.uuid4()),
            record_type="clinical_evaluation",
            description="Clinical evaluation and post-market clinical follow-up",
            responsible_person="Clinical Affairs Manager",
            review_date=datetime.now(),
            status=ComplianceStatus.PENDING
        )
        quality_records['clinical_evaluation'] = clinical_evaluation
        
        # Save to database and internal records
        for record in quality_records.values():
            self.quality_records.append(record)
            self._save_quality_record_to_db(record)
        
        return quality_records
    
    def _save_quality_record_to_db(self, record: QualityRecord):
        """Save quality record to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.record_id,
            record.record_type,
            record.description,
            record.responsible_person,
            record.review_date.isoformat(),
            record.approval_date.isoformat() if record.approval_date else None,
            record.status.value,
            json.dumps(record.attachments),
            record.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def implement_post_market_surveillance(
        self,
        model: nn.Module,
        monitoring_data: torch.Tensor,
        monitoring_labels: torch.Tensor,
        patient_demographics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implement post-market surveillance system.
        
        Args:
            model: Deployed AI model
            monitoring_data: Real-world monitoring data
            monitoring_labels: Ground truth labels for monitoring
            patient_demographics: Patient demographic information
            
        Returns:
            Post-market surveillance report
        """
        logger.info("Conducting post-market surveillance")
        
        model.eval()
        device = next(model.parameters()).device
        
        monitoring_data = monitoring_data.to(device)
        monitoring_labels = monitoring_labels.to(device)
        
        # Generate predictions
        with torch.no_grad():
            predictions = model(monitoring_data)
            predicted_probs = F.softmax(predictions, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)
        
        # Calculate current performance
        y_true = monitoring_labels.cpu().numpy()
        y_pred = predicted_classes.cpu().numpy()
        y_prob = predicted_probs.cpu().numpy()
        
        current_performance = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Performance drift detection
        drift_analysis = self._detect_performance_drift(current_performance)
        
        # Adverse event detection
        adverse_events = self._detect_adverse_events(
            y_true, y_pred, y_prob, patient_demographics
        )
        
        # Subgroup performance monitoring
        subgroup_monitoring = self._monitor_subgroup_performance(
            y_true, y_pred, patient_demographics
        )
        
        # Generate surveillance report
        surveillance_report = {
            'monitoring_date': datetime.now().isoformat(),
            'sample_size': len(y_true),
            'current_performance': current_performance,
            'performance_drift': drift_analysis,
            'adverse_events': adverse_events,
            'subgroup_performance': subgroup_monitoring,
            'regulatory_actions_required': self._assess_regulatory_actions(
                drift_analysis, adverse_events
            )
        }
        
        # Save performance data
        self._save_performance_monitoring(current_performance, patient_demographics)
        
        # Report adverse events if required
        for event in adverse_events:
            if event.regulatory_reporting_required:
                self._report_adverse_event(event)
        
        return surveillance_report
    
    def _detect_performance_drift(
        self,
        current_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Detect performance drift from baseline."""
        if not self.performance_history:
            # No baseline available
            return {
                'drift_detected': False,
                'message': 'No baseline performance available'
            }
        
        # Calculate baseline performance (average of last 10 measurements)
        recent_history = self.performance_history[-10:]
        baseline_performance = {}
        
        for metric in current_performance.keys():
            baseline_values = [record[metric] for record in recent_history if metric in record]
            if baseline_values:
                baseline_performance[metric] = np.mean(baseline_values)
        
        # Detect significant drift (>5% degradation)
        drift_threshold = 0.05
        drift_detected = False
        drift_details = {}
        
        for metric, current_value in current_performance.items():
            if metric in baseline_performance:
                baseline_value = baseline_performance[metric]
                drift = (baseline_value - current_value) / baseline_value
                
                drift_details[metric] = {
                    'current': current_value,
                    'baseline': baseline_value,
                    'drift_percentage': drift * 100,
                    'significant_drift': drift > drift_threshold
                }
                
                if drift > drift_threshold:
                    drift_detected = True
        
        return {
            'drift_detected': drift_detected,
            'drift_details': drift_details,
            'drift_threshold': drift_threshold * 100
        }
    
    def _detect_adverse_events(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        patient_demographics: Dict[str, Any]
    ) -> List[AdverseEvent]:
        """Detect potential adverse events."""
        adverse_events = []
        
        # Detect high-confidence false positives/negatives
        confidence_threshold = 0.9
        
        for i in range(len(y_true)):
            max_prob = np.max(y_prob[i])
            predicted_class = y_pred[i]
            true_class = y_true[i]
            
            # High-confidence misclassification
            if max_prob > confidence_threshold and predicted_class != true_class:
                event_id = str(uuid.uuid4())
                
                # Determine severity based on clinical context
                severity = self._assess_misclassification_severity(
                    true_class, predicted_class, max_prob
                )
                
                event = AdverseEvent(
                    event_id=event_id,
                    device_id=self.device_name,
                    event_type="high_confidence_misclassification",
                    severity=severity,
                    description=f"High-confidence misclassification: predicted {predicted_class}, actual {true_class}, confidence {max_prob:.3f}",
                    patient_demographics=patient_demographics.get(str(i), {}),
                    clinical_context=f"Healthcare situation: {self.healthcare_situation.value}",
                    regulatory_reporting_required=severity in ["severe", "life_threatening"]
                )
                
                adverse_events.append(event)
        
        return adverse_events
    
    def _assess_misclassification_severity(
        self,
        true_class: int,
        predicted_class: int,
        confidence: float
    ) -> str:
        """Assess severity of misclassification based on clinical context."""
        # This is a simplified assessment - in practice, this would be
        # much more sophisticated and domain-specific
        
        if self.healthcare_situation == HealthcareSituation.CRITICAL:
            if confidence > 0.95:
                return "life_threatening"
            elif confidence > 0.90:
                return "severe"
            else:
                return "moderate"
        elif self.healthcare_situation == HealthcareSituation.SERIOUS:
            if confidence > 0.95:
                return "severe"
            elif confidence > 0.90:
                return "moderate"
            else:
                return "minor"
        else:
            return "minor"
    
    def _monitor_subgroup_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        patient_demographics: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Monitor performance across patient subgroups."""
        subgroup_performance = {}
        
        # Group patients by demographics
        demographic_groups = defaultdict(list)
        
        for patient_id, demographics in patient_demographics.items():
            idx = int(patient_id)
            if idx < len(y_true):
                for demo_key, demo_value in demographics.items():
                    group_key = f"{demo_key}_{demo_value}"
                    demographic_groups[group_key].append(idx)
        
        # Calculate performance for each group
        for group_name, indices in demographic_groups.items():
            if len(indices) >= 10:  # Minimum sample size
                group_y_true = y_true[indices]
                group_y_pred = y_pred[indices]
                
                group_accuracy = accuracy_score(group_y_true, group_y_pred)
                group_precision = precision_score(group_y_true, group_y_pred, average='weighted')
                group_recall = recall_score(group_y_true, group_y_pred, average='weighted')
                
                subgroup_performance[group_name] = {
                    'accuracy': group_accuracy,
                    'precision': group_precision,
                    'recall': group_recall,
                    'sample_size': len(indices)
                }
        
        return subgroup_performance
    
    def _assess_regulatory_actions(
        self,
        drift_analysis: Dict[str, Any],
        adverse_events: List[AdverseEvent]
    ) -> List[str]:
        """Assess required regulatory actions based on surveillance findings."""
        actions = []
        
        # Performance drift actions
        if drift_analysis.get('drift_detected', False):
            actions.append("Performance drift detected - investigate root cause")
            actions.append("Consider model retraining or recalibration")
            actions.append("Notify regulatory authorities if drift exceeds predetermined thresholds")
        
        # Adverse event actions
        severe_events = [e for e in adverse_events if e.severity in ["severe", "life_threatening"]]
        if severe_events:
            actions.append(f"Report {len(severe_events)} severe adverse events to FDA within 24 hours")
            actions.append("Conduct root cause analysis for severe events")
            actions.append("Consider risk mitigation measures")
        
        moderate_events = [e for e in adverse_events if e.severity == "moderate"]
        if len(moderate_events) > 10:
            actions.append("High number of moderate adverse events - investigate patterns")
        
        return actions
    
    def _save_performance_monitoring(
        self,
        performance: Dict[str, float],
        patient_demographics: Dict[str, Any]
    ):
        """Save performance monitoring data."""
        # Add to performance history
        performance_record = performance.copy()
        performance_record['timestamp'] = datetime.now().isoformat()
        self.performance_history.append(performance_record)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in performance.items():
            record_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO performance_monitoring VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_id,
                metric_name,
                metric_value,
                datetime.now().isoformat(),
                "mixed_population",  # Could be more specific
                "real_world_deployment",
                "Post-market surveillance measurement"
            ))
        
        conn.commit()
        conn.close()
    
    def _report_adverse_event(self, event: AdverseEvent):
        """Report adverse event to regulatory authorities."""
        # Add to adverse events list
        self.adverse_events.append(event)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO adverse_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.device_id,
            event.event_type,
            event.severity,
            event.description,
            json.dumps(event.patient_demographics),
            event.clinical_context,
            event.root_cause,
            json.dumps(event.corrective_actions),
            event.report_date.isoformat(),
            event.regulatory_reporting_required
        ))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"Adverse event reported: {event.event_id} ({event.severity})")
    
    def generate_regulatory_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive regulatory compliance report."""
        report = {
            'device_information': {
                'device_name': self.device_name,
                'intended_use': self.intended_use,
                'samd_category': self.samd_category.value,
                'risk_class': self.risk_class.value,
                'healthcare_situation': self.healthcare_situation.value,
                'healthcare_decision': self.healthcare_decision.value
            },
            'regulatory_submissions': len(self.regulatory_submissions),
            'clinical_studies': len(self.clinical_studies),
            'quality_records': len(self.quality_records),
            'adverse_events': len(self.adverse_events),
            'performance_monitoring': len(self.performance_history),
            'compliance_status': self._assess_overall_compliance(),
            'recommendations': self._generate_compliance_recommendations(),
            'report_date': datetime.now().isoformat()
        }
        
        return report
    
    def _assess_overall_compliance(self) -> Dict[str, ComplianceStatus]:
        """Assess overall compliance status."""
        compliance_status = {}
        
        # Regulatory submission compliance
        if self.regulatory_submissions:
            latest_submission = max(self.regulatory_submissions, key=lambda x: x.submission_date)
            if latest_submission.status == "approved":
                compliance_status['regulatory_submission'] = ComplianceStatus.COMPLIANT
            elif latest_submission.status in ["submitted", "under_review"]:
                compliance_status['regulatory_submission'] = ComplianceStatus.UNDER_REVIEW
            else:
                compliance_status['regulatory_submission'] = ComplianceStatus.PENDING
        else:
            compliance_status['regulatory_submission'] = ComplianceStatus.NON_COMPLIANT
        
        # Clinical evidence compliance
        completed_studies = [s for s in self.clinical_studies if s.results is not None]
        if completed_studies:
            # Check if studies meet regulatory requirements
            meets_requirements = all(
                study.results.get('regulatory_compliance', {}).get('overall_compliant', False)
                for study in completed_studies
            )
            compliance_status['clinical_evidence'] = (
                ComplianceStatus.COMPLIANT if meets_requirements else ComplianceStatus.NON_COMPLIANT
            )
        else:
            compliance_status['clinical_evidence'] = ComplianceStatus.PENDING
        
        # Quality management compliance
        approved_quality_records = [
            r for r in self.quality_records 
            if r.status == ComplianceStatus.COMPLIANT
        ]
        if len(approved_quality_records) >= 4:  # Minimum required records
            compliance_status['quality_management'] = ComplianceStatus.COMPLIANT
        else:
            compliance_status['quality_management'] = ComplianceStatus.PENDING
        
        # Post-market surveillance compliance
        if self.performance_history and len(self.performance_history) >= 3:
            compliance_status['post_market_surveillance'] = ComplianceStatus.COMPLIANT
        else:
            compliance_status['post_market_surveillance'] = ComplianceStatus.PENDING
        
        return compliance_status
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        compliance_status = self._assess_overall_compliance()
        
        for area, status in compliance_status.items():
            if status == ComplianceStatus.NON_COMPLIANT:
                if area == 'regulatory_submission':
                    recommendations.append("Submit regulatory application (510(k), PMA, or De Novo)")
                elif area == 'clinical_evidence':
                    recommendations.append("Conduct additional clinical studies to meet performance requirements")
                elif area == 'quality_management':
                    recommendations.append("Complete quality management system implementation")
                elif area == 'post_market_surveillance':
                    recommendations.append("Implement comprehensive post-market surveillance program")
            elif status == ComplianceStatus.PENDING:
                recommendations.append(f"Complete pending activities for {area.replace('_', ' ')}")
        
        # Check for adverse events requiring action
        severe_events = [e for e in self.adverse_events if e.severity in ["severe", "life_threatening"]]
        if severe_events:
            recommendations.append("Address severe adverse events and implement corrective actions")
        
        return recommendations

# Example usage and demonstration
def main():
    """Demonstrate regulatory compliance framework for healthcare AI."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Healthcare AI Regulatory Compliance Demonstration")
    print("=" * 60)
    
    # Initialize compliance framework
    compliance_framework = RegulatoryComplianceFramework(
        device_name="AI Diagnostic Assistant",
        intended_use="AI-powered diagnostic support for radiological image analysis",
        healthcare_situation=HealthcareSituation.SERIOUS,
        healthcare_decision=HealthcareDecision.DIAGNOSE
    )
    
    print(f"Device: {compliance_framework.device_name}")
    print(f"SaMD Category: {compliance_framework.samd_category.value}")
    print(f"Risk Class: {compliance_framework.risk_class.value}")
    
    # Create regulatory submission
    print("\n1. Creating Regulatory Submission")
    print("-" * 40)
    
    submission = compliance_framework.create_regulatory_submission(
        submission_type="510(k)",
        predicate_devices=["K123456", "K789012"],
        substantial_equivalence=True
    )
    
    print(f"Submission ID: {submission.submission_id}")
    print(f"Submission Type: {submission.submission_type}")
    
    # Design clinical study
    print("\n2. Designing Clinical Study")
    print("-" * 35)
    
    study = compliance_framework.design_clinical_study(
        study_type="prospective",
        primary_endpoint="Diagnostic accuracy compared to radiologist consensus",
        sample_size=500,
        study_population="Adult patients with suspected lung nodules",
        inclusion_criteria=[
            "Age 18-80 years",
            "Chest CT scan with suspected lung nodules",
            "Informed consent provided"
        ],
        exclusion_criteria=[
            "Pregnancy",
            "Previous lung surgery",
            "Inability to provide informed consent"
        ]
    )
    
    print(f"Study ID: {study.study_id}")
    print(f"Study Design: {study.study_design}")
    print(f"Sample Size: {study.sample_size}")
    
    # Generate synthetic validation data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 500
    n_features = 100
    
    # Create synthetic medical imaging features
    X_val = torch.randn(n_samples, n_features)
    y_val = (torch.sum(X_val[:, :10], dim=1) > 0).long()
    
    # Create simple model for demonstration
    class DiagnosticModel(nn.Module):
        def __init__(self, input_size: int, num_classes: int = 2):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.dropout = nn.Dropout(0.3)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = DiagnosticModel(n_features).to(device)
    
    # Train model briefly for demonstration
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_val.to(device))
        loss = criterion(outputs, y_val.to(device))
        loss.backward()
        optimizer.step()
    
    # Create clinical context
    clinical_context = {
        'demographics': {
            'age_18_40': list(range(0, 100)),
            'age_41_60': list(range(100, 300)),
            'age_61_80': list(range(300, 500)),
            'male': list(range(0, 250)),
            'female': list(range(250, 500))
        }
    }
    
    # Conduct clinical validation
    print("\n3. Conducting Clinical Validation")
    print("-" * 40)
    
    validation_results = compliance_framework.conduct_clinical_validation(
        model=model,
        validation_data=X_val,
        validation_labels=y_val,
        study_id=study.study_id,
        clinical_context=clinical_context
    )
    
    print("Clinical Validation Results:")
    metrics = validation_results['primary_metrics']
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    compliance = validation_results['regulatory_compliance']
    print(f"\nRegulatory Compliance:")
    for check, status in compliance.items():
        print(f"  {check}: {'✓' if status else '✗'}")
    
    # Implement quality management system
    print("\n4. Implementing Quality Management System")
    print("-" * 50)
    
    quality_records = compliance_framework.implement_quality_management_system()
    
    print("Quality Records Created:")
    for record_type, record in quality_records.items():
        print(f"  {record_type}: {record.record_id}")
    
    # Simulate post-market surveillance
    print("\n5. Post-Market Surveillance")
    print("-" * 35)
    
    # Generate monitoring data
    monitoring_data = torch.randn(100, n_features)
    monitoring_labels = (torch.sum(monitoring_data[:, :10], dim=1) > 0).long()
    
    # Create patient demographics for monitoring
    patient_demographics = {}
    for i in range(100):
        patient_demographics[str(i)] = {
            'age': np.random.choice(['18-40', '41-60', '61-80']),
            'gender': np.random.choice(['male', 'female']),
            'ethnicity': np.random.choice(['caucasian', 'african_american', 'hispanic', 'asian'])
        }
    
    surveillance_report = compliance_framework.implement_post_market_surveillance(
        model=model,
        monitoring_data=monitoring_data,
        monitoring_labels=monitoring_labels,
        patient_demographics=patient_demographics
    )
    
    print("Post-Market Surveillance Results:")
    current_perf = surveillance_report['current_performance']
    for metric, value in current_perf.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"Adverse events detected: {len(surveillance_report['adverse_events'])}")
    print(f"Performance drift detected: {surveillance_report['performance_drift']['drift_detected']}")
    
    # Generate compliance report
    print("\n6. Regulatory Compliance Report")
    print("-" * 40)
    
    compliance_report = compliance_framework.generate_regulatory_compliance_report()
    
    print("Compliance Status:")
    for area, status in compliance_report['compliance_status'].items():
        print(f"  {area}: {status.value}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(compliance_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n{'='*60}")
    print("Regulatory compliance demonstration completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
```

## 11.3 International Regulatory Harmonization

### 11.3.1 European Union Medical Device Regulation (MDR)

The European Union's Medical Device Regulation (MDR) 2017/745, which became fully applicable in May 2021, establishes comprehensive requirements for medical devices including AI-based systems. The MDR emphasizes **clinical evidence**, **post-market surveillance**, and **transparency** through the European Database on Medical Devices (EUDAMED).

**Conformity assessment** under the MDR requires involvement of **Notified Bodies** for most AI medical devices, particularly those classified as Class IIa, IIb, or III. The conformity assessment process includes review of technical documentation, quality management systems, and clinical evidence.

**Clinical evaluation** requirements under the MDR are more stringent than previous directives, requiring **clinical evidence** that demonstrates safety and performance throughout the device lifecycle. For AI systems, this includes evidence of performance across different patient populations and clinical settings.

**Post-market clinical follow-up (PMCF)** is mandatory for most medical devices under the MDR, requiring ongoing collection and evaluation of clinical data to confirm safety and performance in real-world use.

### 11.3.2 Health Canada Regulatory Approach

Health Canada regulates medical devices through the **Medical Devices Regulations** under the Food and Drugs Act. AI-based medical devices are classified using a risk-based approach similar to other jurisdictions, with Class I (low risk) through Class IV (high risk) classifications.

**Software as Medical Device** guidance from Health Canada provides specific requirements for AI/ML systems, including considerations for **algorithm transparency**, **validation data**, and **change control procedures**.

**Quality system requirements** follow ISO 13485 standards, with additional requirements for software development and risk management specific to AI systems.

### 11.3.3 Japan PMDA Framework

Japan's Pharmaceuticals and Medical Devices Agency (PMDA) has developed specific guidance for AI-based medical devices, emphasizing **clinical utility** and **real-world evidence**. The PMDA's approach includes:

**Sakigake designation** for breakthrough medical devices, which can provide expedited review for innovative AI systems that address unmet medical needs.

**Clinical trial consultation** services help developers design appropriate clinical studies for AI medical devices, with consideration for unique validation challenges.

**Post-market study requirements** ensure ongoing monitoring of AI system performance in the Japanese healthcare environment.

### 11.3.4 IMDRF Harmonization Efforts

The International Medical Device Regulators Forum (IMDRF) works to harmonize regulatory approaches globally through working groups and guidance documents. Key IMDRF initiatives for AI include:

**Software as Medical Device Working Group** develops guidance on SaMD classification, quality management, and clinical evaluation that is adopted by multiple regulatory agencies.

**AI Medical Device Working Group** addresses specific challenges related to machine learning systems, including change control, validation, and post-market monitoring.

**Good Regulatory Review Practices** promote consistent regulatory review processes across jurisdictions, reducing regulatory burden for global device manufacturers.

## 11.4 Quality Management Systems for AI

### 11.4.1 ISO 13485 Implementation for AI Systems

ISO 13485 provides the foundation for quality management systems in medical device development, with specific considerations required for AI systems. The standard requires a **process-based approach** that addresses all aspects of the device lifecycle from design through post-market activities.

**Document control** becomes particularly important for AI systems due to the complexity of software, data, and algorithmic components. The quality management system must ensure that all versions of training data, model parameters, and software components are properly controlled and traceable.

**Management responsibility** includes ensuring that quality objectives are established and that adequate resources are provided for AI system development and maintenance. Top management must demonstrate commitment to quality and regulatory compliance.

**Resource management** must address the unique requirements of AI development, including specialized personnel, computational resources, and data management infrastructure.

### 11.4.2 Risk Management per ISO 14971

Risk management for AI medical devices follows ISO 14971 principles but requires additional considerations for AI-specific risks. The risk management process must address:

**Algorithmic risks** including bias, overfitting, and performance degradation over time. Risk analysis must consider how the AI system might fail and the potential consequences of different failure modes.

**Data-related risks** including training data quality, representativeness, and privacy concerns. The risk management file must document how data risks are identified, analyzed, and controlled.

**Human factors risks** related to how clinicians interact with AI systems, including automation bias, over-reliance, and skill degradation.

**Cybersecurity risks** that could compromise system integrity or patient data security.

The **risk management file** must be maintained throughout the device lifecycle and updated based on post-market experience and new risk information.

### 11.4.3 Software Lifecycle Processes per IEC 62304

IEC 62304 defines software lifecycle processes for medical device software, with AI systems requiring careful consideration of how machine learning components fit within the standard framework.

**Software safety classification** determines the level of rigor required for software development processes. AI systems often fall into Class B (non-life-threatening injury) or Class C (death or serious injury) categories due to their clinical impact.

**Software development planning** must address the unique aspects of AI development, including data collection, model training, validation, and deployment processes.

**Software architectural design** must clearly define the boundaries between AI components and other system elements, with appropriate interfaces and error handling.

**Software verification and validation** requires comprehensive testing of AI components, including performance testing, robustness testing, and clinical validation.

### 11.4.4 Change Control for AI Systems

Change control for AI systems presents unique challenges due to the potential for continuous learning and adaptation. The quality management system must define:

**Predetermined change control plans** that specify what types of changes can be made without requiring new regulatory submissions.

**Change impact assessment** procedures that evaluate the potential effects of modifications on safety and effectiveness.

**Validation requirements** for different types of changes, from minor parameter adjustments to major algorithmic modifications.

**Documentation requirements** that ensure all changes are properly recorded and traceable.

## 11.5 Clinical Evidence and Validation

### 11.5.1 Clinical Evidence Requirements by Risk Class

Clinical evidence requirements for AI medical devices vary based on risk classification and intended use. **High-risk devices** (Class III/SaMD Class IV) typically require prospective clinical studies with appropriate controls and statistical power.

**Moderate-risk devices** (Class II/SaMD Class II-III) may rely on retrospective studies, literature reviews, or smaller prospective studies, depending on the specific clinical application and available predicate devices.

**Low-risk devices** (Class I/SaMD Class I) may require only analytical validation and literature support, though clinical evidence may still be necessary for novel applications.

The **clinical evidence** must demonstrate:
- **Safety**: The device does not cause unacceptable harm
- **Effectiveness**: The device performs as intended
- **Clinical utility**: Use of the device improves patient outcomes or clinical workflow

### 11.5.2 Study Design Considerations for AI

Clinical studies for AI systems require careful consideration of study design elements that may differ from traditional medical device studies.

**Comparator selection** must consider whether the AI system is intended to replace, augment, or assist human decision-making. Appropriate comparators might include:
- Standard of care without AI assistance
- Alternative AI systems
- Human experts (radiologists, pathologists, etc.)
- Combination of human experts and AI

**Endpoint selection** should focus on clinically meaningful outcomes rather than purely technical metrics. Appropriate endpoints might include:
- Diagnostic accuracy compared to reference standard
- Time to diagnosis
- Inter-observer agreement
- Patient outcomes (mortality, morbidity, quality of life)
- Healthcare resource utilization

**Statistical considerations** must account for the unique characteristics of AI systems, including:
- Multiple testing corrections for subgroup analyses
- Non-inferiority margins for AI vs. human comparisons
- Clustering effects in multi-site studies
- Interim analysis plans for adaptive studies

### 11.5.3 Real-World Evidence and Post-Market Studies

Real-world evidence (RWE) plays an increasingly important role in AI medical device regulation, particularly for post-market surveillance and label expansion studies.

**Real-world data sources** for AI validation include:
- Electronic health records
- Claims databases
- Patient registries
- Wearable device data
- Patient-reported outcomes

**RWE study designs** must address potential biases and confounding factors inherent in observational data:
- Selection bias in patient populations
- Information bias in data collection
- Confounding by indication
- Temporal trends in clinical practice

**Data quality considerations** are particularly important for RWE studies:
- Completeness of data elements
- Accuracy of coding and classification
- Consistency across sites and time periods
- Representativeness of study population

### 11.5.4 Subgroup Analysis and Health Equity

Regulatory agencies increasingly emphasize the importance of demonstrating AI system performance across diverse patient populations to ensure health equity.

**Subgroup analysis planning** should be specified in advance and include:
- Demographic subgroups (age, sex, race/ethnicity)
- Clinical subgroups (disease severity, comorbidities)
- Healthcare setting subgroups (academic vs. community hospitals)
- Geographic subgroups (different countries or regions)

**Statistical power considerations** for subgroup analyses require larger sample sizes to detect meaningful differences between groups.

**Bias detection and mitigation** strategies should be implemented throughout the clinical validation process:
- Bias testing during algorithm development
- Fairness metrics evaluation
- Disparate impact analysis
- Corrective action plans for identified biases

## 11.6 Post-Market Surveillance and Adverse Event Reporting

### 11.6.1 Post-Market Surveillance Requirements

Post-market surveillance for AI medical devices requires ongoing monitoring of device performance and safety in real-world clinical use. Regulatory requirements include:

**Medical Device Reporting (MDR)** in the US requires manufacturers to report adverse events to the FDA within specified timeframes:
- Death or serious injury: 24 hours (by telephone) and 10 working days (written report)
- Malfunction that could cause death or serious injury: 10 working days

**Periodic safety updates** provide regular summaries of post-market experience and may be required annually or at other specified intervals.

**Post-market clinical follow-up (PMCF)** under EU MDR requires ongoing collection and evaluation of clinical data to confirm device safety and performance.

### 11.6.2 Performance Monitoring and Drift Detection

AI systems require specialized monitoring approaches to detect performance degradation or drift over time.

**Performance metrics monitoring** should track key indicators of system performance:
- Diagnostic accuracy metrics
- Confidence calibration
- Processing time and throughput
- User acceptance and satisfaction

**Statistical process control** methods can be used to detect significant changes in performance:
- Control charts for key metrics
- Cumulative sum (CUSUM) charts for drift detection
- Sequential probability ratio tests

**Drift detection algorithms** can automatically identify when system performance deviates from expected ranges:
- Kolmogorov-Smirnov tests for distribution changes
- Maximum mean discrepancy for feature drift
- Adversarial validation for dataset shift

### 11.6.3 Corrective and Preventive Actions (CAPA)

When post-market surveillance identifies safety or performance issues, manufacturers must implement appropriate corrective and preventive actions.

**Root cause analysis** must identify the underlying causes of identified problems:
- Technical factors (algorithm, data, software)
- Human factors (training, workflow, interface)
- Environmental factors (hardware, network, clinical setting)

**Corrective actions** address immediate safety concerns:
- Software updates or patches
- User training or communication
- Temporary restrictions on use
- Device recall if necessary

**Preventive actions** address systemic issues to prevent recurrence:
- Process improvements in development or manufacturing
- Enhanced testing or validation procedures
- Improved post-market monitoring systems

### 11.6.4 Regulatory Communication and Transparency

Effective communication with regulatory agencies and healthcare stakeholders is essential for maintaining compliance and public trust.

**Proactive communication** with regulators helps build confidence and may facilitate faster resolution of issues:
- Regular meetings or calls with regulatory staff
- Voluntary reporting of potential issues
- Sharing of post-market data and analyses

**Public transparency** through various channels helps maintain stakeholder confidence:
- Publication of clinical study results
- Participation in scientific conferences
- Collaboration with professional societies
- Patient and provider education materials

**Crisis communication** plans should be prepared in advance for serious safety issues:
- Clear communication channels and responsibilities
- Pre-approved messaging templates
- Coordination with regulatory agencies and healthcare partners

## 11.7 Emerging Regulatory Trends and Future Directions

### 11.7.1 Adaptive Regulation and Continuous Learning

Regulatory agencies are exploring new approaches to accommodate the unique characteristics of AI systems, particularly their ability to learn and adapt over time.

**Regulatory sandboxes** provide controlled environments for testing innovative AI systems with relaxed regulatory requirements:
- FDA's Software Precertification Program
- UK's MHRA Software and AI as Medical Device Change Programme
- Singapore's Model AI Governance Framework

**Adaptive licensing** approaches allow for conditional approval with ongoing evidence generation:
- Progressive licensing based on accumulating evidence
- Risk-sharing agreements between manufacturers and payers
- Real-world evidence requirements for label expansion

**Continuous monitoring** frameworks enable ongoing assessment of AI system performance:
- Real-time performance dashboards
- Automated adverse event detection
- Dynamic risk-benefit assessment

### 11.7.2 International Harmonization Initiatives

Efforts to harmonize AI regulation across jurisdictions continue to evolve:

**IMDRF AI Working Group** develops guidance on AI-specific regulatory issues:
- Machine learning-enabled medical device software
- Good machine learning practice for medical device development
- Clinical evaluation of AI medical devices

**ISO/IEC standards development** addresses AI-specific requirements:
- ISO/IEC 23053: Framework for AI systems using ML
- ISO/IEC 23094: Guidance on AI risk management
- ISO/IEC 24028: Overview of trustworthiness in AI

**Bilateral cooperation agreements** facilitate mutual recognition and information sharing:
- US-EU cooperation on AI regulation
- Asia-Pacific regulatory harmonization initiatives
- Global Partnership on AI (GPAI) health working group

### 11.7.3 Ethical and Social Considerations

Regulatory frameworks increasingly incorporate ethical and social considerations beyond traditional safety and effectiveness requirements.

**Algorithmic fairness** requirements address bias and discrimination:
- Fairness metrics and testing requirements
- Diverse representation in training data
- Ongoing monitoring for disparate impact

**Transparency and explainability** requirements vary by risk level and clinical application:
- Algorithm transparency for high-risk applications
- Explainable AI for clinical decision support
- User interface design for appropriate trust calibration

**Privacy and data protection** requirements address the extensive data requirements of AI systems:
- Data minimization principles
- Purpose limitation and consent management
- Cross-border data transfer restrictions

### 11.7.4 Regulatory Science and Evidence Generation

Investment in regulatory science helps develop better methods for evaluating AI medical devices.

**Validation methodologies** continue to evolve:
- Synthetic data for algorithm validation
- Federated learning for multi-site studies
- Digital twins for device testing

**Real-world evidence infrastructure** supports post-market surveillance:
- Distributed research networks
- Common data models and standards
- Privacy-preserving analytics methods

**Regulatory decision tools** help streamline review processes:
- AI-assisted regulatory review
- Automated safety signal detection
- Predictive models for regulatory outcomes

## 11.8 Conclusion

The regulatory landscape for healthcare AI continues to evolve rapidly as agencies worldwide grapple with the unique challenges posed by machine learning systems. Success in this environment requires a comprehensive understanding of regulatory requirements, proactive engagement with regulatory agencies, and robust quality management systems that address the full lifecycle of AI medical devices.

The frameworks and approaches presented in this chapter provide a foundation for navigating the complex regulatory environment while ensuring that AI systems meet the highest standards of safety, effectiveness, and quality. As the field continues to mature, regulatory requirements will undoubtedly continue to evolve, requiring ongoing vigilance and adaptation from healthcare AI developers.

The future of healthcare AI regulation lies in finding the right balance between ensuring patient safety and fostering innovation. The collaborative efforts between regulatory agencies, industry, academia, and healthcare providers will be essential for developing regulatory frameworks that protect patients while enabling the transformative potential of AI in healthcare.

## References

1. FDA. (2021). Artificial Intelligence/Machine Learning (AI/ML)-Based Software as Medical Device (SaMD) Action Plan. U.S. Food and Drug Administration.

2. FDA. (2023). Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence/Machine Learning (AI/ML)-Enabled Device Software Functions. U.S. Food and Drug Administration.

3. IMDRF. (2017). Software as Medical Device (SaMD): Clinical Evaluation. International Medical Device Regulators Forum.

4. European Commission. (2017). Regulation (EU) 2017/745 on medical devices. Official Journal of the European Union.

5. ISO 13485:2016. Medical devices - Quality management systems - Requirements for regulatory purposes. International Organization for Standardization.

6. ISO 14971:2019. Medical devices - Application of risk management to medical devices. International Organization for Standardization.

7. IEC 62304:2006. Medical device software - Software life cycle processes. International Electrotechnical Commission.

8. Health Canada. (2019). Guidance Document: Software as Medical Device (SaMD). Health Canada.

9. PMDA. (2021). Artificial Intelligence-based Medical Device Development Guideline. Pharmaceuticals and Medical Devices Agency, Japan.

10. Muehlematter, U. J., et al. (2021). Approval of artificial intelligence and machine learning-based medical devices in the USA and Europe (2015-20): a comparative analysis. The Lancet Digital Health, 3(3), e195-e203. DOI: 10.1016/S2589-7500(20)30292-2
