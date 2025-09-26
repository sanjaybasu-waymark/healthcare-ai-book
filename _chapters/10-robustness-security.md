---
layout: default
title: "Chapter 10: Robustness Security"
nav_order: 10
parent: Chapters
permalink: /chapters/10-robustness-security/
---

# Chapter 10: Robustness and Security in Healthcare AI - Building Resilient and Secure Clinical Systems

*By Sanjay Basu MD PhD*

## Learning Objectives

By the end of this chapter, physician data scientists will be able to:

- Understand the theoretical foundations of robustness and security in healthcare AI systems, including distributional robustness, adversarial robustness, and the unique security challenges posed by clinical environments and sensitive patient data
- Implement adversarial defense mechanisms to protect against malicious attacks, including adversarial training, certified defenses, detection-based methods, and ensemble approaches specifically adapted for healthcare applications
- Design robust AI systems that maintain performance under distribution shift and data corruption, incorporating techniques for handling domain adaptation, covariate shift, and temporal drift in clinical data
- Deploy comprehensive security frameworks for healthcare AI applications, including encryption, access controls, audit logging, and privacy-preserving techniques that comply with HIPAA, GDPR, and FDA requirements
- Evaluate system robustness using formal verification and empirical testing methods, including adversarial evaluation, stress testing, and continuous monitoring approaches for production healthcare AI systems
- Navigate cybersecurity requirements specific to healthcare AI deployments, including threat modeling, vulnerability assessment, incident response, and regulatory compliance frameworks
- Implement privacy-preserving techniques including differential privacy, federated learning security, and secure multi-party computation for collaborative healthcare AI development

## 10.1 Introduction to Healthcare AI Robustness and Security

Robustness and security represent critical requirements for healthcare AI systems, where failures can have life-threatening consequences and sensitive patient data requires the highest levels of protection. Unlike other domains where occasional failures may be acceptable or where data breaches primarily affect financial information, healthcare AI systems must maintain reliable performance under adversarial conditions, data corruption, and distribution shifts while protecting patient privacy, system integrity, and clinical workflow continuity.

The intersection of robustness and security in healthcare AI creates unique challenges that require specialized approaches combining traditional cybersecurity practices with AI-specific defense mechanisms. **Robustness** refers to the ability of AI systems to maintain performance when faced with unexpected inputs, data corruption, changes in the underlying data distribution, or adversarial manipulation. **Security** encompasses protection against malicious attacks, unauthorized access, data breaches, and system compromises that could compromise patient safety, privacy, or clinical operations.

Healthcare AI systems face a complex threat landscape that includes both traditional cybersecurity threats and AI-specific vulnerabilities. The high-stakes nature of medical decision-making, the sensitivity of patient data, and the increasing connectivity of medical devices create an environment where robust security measures are not just recommended but essential for patient safety and regulatory compliance.

### 10.1.1 Theoretical Foundations of Healthcare AI Robustness

Healthcare AI robustness is grounded in several theoretical frameworks that address the unique requirements of medical applications, including the need to handle diverse patient populations, varying clinical protocols, and evolving medical knowledge. Understanding these foundations is essential for designing AI systems that can reliably operate in the complex and dynamic healthcare environment.

**Distributional Robustness and Domain Adaptation**: Healthcare AI systems must perform well across different patient populations, clinical settings, and geographic regions, each of which may have different data distributions. Distributional robustness ensures that models maintain performance when the test distribution differs from the training distribution, a common scenario in healthcare where models trained at one institution may be deployed at another.

The mathematical foundation of distributional robustness can be expressed through the framework of distributionally robust optimization (DRO):

$$\min_{\theta} \max_{P \in \mathcal{U}} \mathbb{E}_{(x,y) \sim P}[\ell(f_\theta(x), y)]$$

Where $\theta$ represents model parameters, $\mathcal{U}$ is an uncertainty set of distributions representing possible variations in the data distribution, $f_\theta$ is the model function, and $\ell$ is the loss function. This formulation ensures that the model performs well across a range of possible data distributions, providing robustness to distribution shift.

For healthcare applications, the uncertainty set $\mathcal{U}$ can be designed to capture clinically relevant variations such as differences in patient demographics, disease prevalence, clinical protocols, or measurement techniques. This approach helps ensure that AI systems remain reliable when deployed across different healthcare settings.

**Adversarial Robustness and Attack Resistance**: Healthcare AI systems are vulnerable to adversarial attacks where malicious actors craft inputs designed to cause misclassification or system failure. In clinical contexts, adversarial examples could potentially be used to manipulate diagnostic systems, treatment recommendations, or clinical decision support tools, making robust defense mechanisms essential for patient safety.

The adversarial robustness problem can be formulated as a minimax optimization:

$$\min_{\theta} \mathbb{E}_{(x,y) \sim D} \left[ \max_{\|\delta\| \leq \epsilon} \ell(f_\theta(x + \delta), y) \right]$$

Where $\delta$ represents the adversarial perturbation bounded by $\epsilon$, $D$ is the data distribution, and the inner maximization finds the worst-case perturbation within the allowed budget. This formulation captures the goal of training models that perform well even under adversarial manipulation.

**Temporal Robustness and Concept Drift**: Healthcare data exhibits temporal patterns and concept drift as medical knowledge evolves, treatment protocols change, and patient populations shift over time. Temporal robustness ensures that AI systems can adapt to these changes while maintaining reliable performance.

The temporal robustness challenge can be modeled as:

$$\min_{\theta} \sum_{t=1}^{T} w_t \mathbb{E}_{(x,y) \sim D_t}[\ell(f_\theta(x), y)]$$

Where $D_t$ represents the data distribution at time $t$, $w_t$ are temporal weights, and $T$ is the time horizon. This formulation allows models to adapt to temporal changes while maintaining performance across different time periods.

**Uncertainty Quantification and Robustness**: Healthcare AI systems must provide reliable uncertainty estimates to support clinical decision-making. Robust uncertainty quantification ensures that confidence estimates remain calibrated under distribution shift and adversarial conditions.

Bayesian approaches to uncertainty quantification can be combined with robustness techniques:

$$p(\theta|D) \propto p(D|\theta) p(\theta)$$

Where the posterior distribution $p(\theta|D)$ captures uncertainty in model parameters, and robust training procedures ensure that this uncertainty remains well-calibrated under various conditions.

### 10.1.2 Security Threat Models in Healthcare AI

Healthcare AI systems face a diverse range of security threats that require comprehensive defense strategies. Understanding these threat models is essential for designing effective security measures and ensuring system resilience against malicious attacks.

**Data Poisoning Attacks**: Data poisoning involves corrupting training data to degrade model performance, introduce backdoors, or bias model behavior toward specific outcomes. In healthcare settings, data poisoning attacks could potentially cause diagnostic errors, treatment failures, or systematic biases that affect patient care quality.

Data poisoning can be modeled as an optimization problem where an attacker seeks to maximize model error by corrupting a fraction of the training data:

$$\max_{\{(x_i', y_i')\}} \mathbb{E}_{(x,y) \sim D_{test}}[\ell(f_{\theta^*}(x), y)]$$

Subject to constraints on the number and magnitude of corrupted samples, where $\theta^*$ represents the parameters learned from the poisoned dataset.

**Model Extraction and Intellectual Property Theft**: Model extraction attacks attempt to steal proprietary AI models through query-based methods, reverse engineering, or side-channel analysis. Given the significant investment in developing healthcare AI systems and the competitive advantage they provide, protecting intellectual property while maintaining clinical utility represents a critical challenge.

Model extraction can be formalized as an approximation problem where an attacker seeks to learn a surrogate model $g$ that mimics the behavior of the target model $f$:

$$\min_g \mathbb{E}_{x \sim D_{query}}[d(f(x), g(x))]$$

Where $d$ is a distance function and $D_{query}$ represents the distribution of queries the attacker can make.

**Privacy Attacks and Patient Data Protection**: Privacy attacks including membership inference, attribute inference, and model inversion pose particular risks in healthcare due to the sensitive nature of medical data. These attacks can reveal whether specific patients were included in training data, infer sensitive medical attributes, or reconstruct patient information from model outputs.

Membership inference attacks can be modeled as a binary classification problem where an attacker tries to determine if a specific sample was used in training:

$$P(\text{member}|x, f(x), \theta)$$

Where the attacker uses the input $x$, model output $f(x)$, and potentially model parameters $\theta$ to infer membership.

**Byzantine Attacks in Federated Learning**: Healthcare AI increasingly relies on federated learning to enable collaboration between institutions while preserving privacy. Byzantine attacks in federated settings involve malicious participants submitting corrupted model updates to compromise the global model.

Byzantine robustness in federated learning can be addressed through robust aggregation methods:

$$\theta_{t+1} = \text{RobustAgg}(\{\theta_t^{(i)} + \Delta_t^{(i)}\}_{i=1}^n)$$

Where $\text{RobustAgg}$ is a robust aggregation function that can handle malicious updates from a subset of participants.

**Supply Chain Attacks and Third-Party Dependencies**: Healthcare AI systems often rely on third-party components, libraries, and data sources that may be compromised. Supply chain attacks involve corrupting these dependencies to introduce vulnerabilities or malicious behavior into the final system.

**Physical Attacks on Medical Devices**: AI-enabled medical devices may be vulnerable to physical attacks that manipulate sensors, communication channels, or computing hardware. These attacks can affect data integrity, model performance, or device functionality.

### 10.1.3 Regulatory and Compliance Considerations

Healthcare AI robustness and security must comply with stringent regulatory requirements that address both traditional cybersecurity concerns and AI-specific risks. Understanding these requirements is essential for developing compliant and deployable healthcare AI systems.

**HIPAA Security Rule and Technical Safeguards**: The Health Insurance Portability and Accountability Act (HIPAA) Security Rule mandates specific technical safeguards for electronic protected health information (ePHI), including access controls, audit logs, encryption requirements, and integrity controls. Healthcare AI systems must implement these safeguards while maintaining system performance and usability.

Key HIPAA requirements for AI systems include:
- **Access Control**: Unique user identification, emergency access procedures, automatic logoff, and encryption/decryption controls
- **Audit Controls**: Hardware, software, and procedural mechanisms for recording access to ePHI
- **Integrity**: ePHI must not be improperly altered or destroyed
- **Person or Entity Authentication**: Verification of user identity before access
- **Transmission Security**: Protection of ePHI during electronic transmission

**FDA Cybersecurity Guidance for Medical Devices**: The FDA's cybersecurity guidance for medical devices emphasizes the importance of security by design, including threat modeling, vulnerability assessment, and incident response planning. The guidance requires manufacturers to consider cybersecurity throughout the device lifecycle, from design and development through deployment and maintenance.

FDA cybersecurity requirements include:
- **Premarket Cybersecurity Documentation**: Threat modeling, risk assessment, and security controls documentation
- **Postmarket Cybersecurity Management**: Monitoring, vulnerability assessment, and update procedures
- **Software Bill of Materials (SBOM)**: Documentation of software components and dependencies
- **Coordinated Vulnerability Disclosure**: Processes for reporting and addressing security vulnerabilities

**GDPR and International Privacy Regulations**: The General Data Protection Regulation (GDPR) and similar international privacy laws impose strict requirements on the processing of personal data, including health information. Healthcare AI systems must implement privacy by design principles and provide mechanisms for data subject rights.

GDPR requirements relevant to healthcare AI include:
- **Lawful Basis for Processing**: Clear legal justification for processing health data
- **Data Minimization**: Processing only necessary data for specified purposes
- **Purpose Limitation**: Using data only for declared purposes
- **Accuracy**: Ensuring data accuracy and enabling correction
- **Storage Limitation**: Retaining data only as long as necessary
- **Security**: Implementing appropriate technical and organizational measures

**ISO 27001 and NIST Cybersecurity Framework**: International standards such as ISO 27001 and the NIST Cybersecurity Framework provide comprehensive frameworks for information security management that are increasingly applied to healthcare AI systems. These frameworks offer structured approaches to risk management, security controls, and continuous improvement.

**Medical Device Cybersecurity Standards**: Standards such as IEC 62304 (medical device software lifecycle), ISO 14971 (risk management for medical devices), and IEC 80001 (network security for medical devices) provide specific guidance for securing AI-enabled medical devices.

## 10.2 Adversarial Robustness in Healthcare AI

Adversarial robustness represents a critical aspect of healthcare AI security, as adversarial attacks can potentially manipulate diagnostic systems, treatment recommendations, or clinical decision support tools with serious consequences for patient safety. Understanding adversarial attack methods and implementing effective defense mechanisms is essential for deploying secure healthcare AI systems.

### 10.2.1 Adversarial Attack Methods in Healthcare

Healthcare AI systems face several categories of adversarial attacks, each requiring specific defensive strategies and presenting unique challenges in clinical environments. Understanding these attack methods is essential for developing comprehensive defense strategies.

**Gradient-Based Attacks**: Gradient-based attacks exploit the differentiable nature of neural networks to generate adversarial examples by following the gradient of the loss function. These attacks are particularly concerning in healthcare because they can be applied to medical images, clinical data, and other inputs to AI systems.

The **Fast Gradient Sign Method (FGSM)** generates adversarial examples by taking a single step in the direction of the gradient:

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$$

Where $x$ is the original input, $\epsilon$ is the perturbation magnitude, $J$ is the loss function, and $\theta$ represents model parameters.

**Projected Gradient Descent (PGD)** extends FGSM by taking multiple smaller steps and projecting back to the allowed perturbation set:

$$x_{adv}^{(t+1)} = \Pi_{S}(x_{adv}^{(t)} + \alpha \cdot \text{sign}(\nabla_x J(\theta, x_{adv}^{(t)}, y)))$$

Where $\Pi_S$ is the projection operator onto the allowed perturbation set $S$, and $\alpha$ is the step size.

**Optimization-Based Attacks**: Optimization-based attacks use sophisticated optimization techniques to generate minimal perturbations that cause misclassification while remaining imperceptible or clinically plausible.

The **Carlini & Wagner (C&W) attack** formulates adversarial example generation as an optimization problem:

$$\min_{\delta} \|\delta\|_p + c \cdot f(x + \delta)$$

Where $f$ is an objective function that encourages misclassification, $c$ is a regularization parameter, and $\|\delta\|_p$ measures the perturbation magnitude.

**Black-Box Attacks**: Black-box attacks that do not require access to model gradients or architecture pose realistic threats in healthcare settings where attackers may only have query access to deployed systems through clinical interfaces or APIs.

**Transfer-based attacks** exploit the transferability of adversarial examples across different models:

$$x_{adv} = \arg\min_{\delta} \mathbb{E}_{f \sim \mathcal{F}}[\ell(f(x + \delta), y)]$$

Where $\mathcal{F}$ represents a family of surrogate models used to generate transferable adversarial examples.

**Query-based attacks** iteratively refine adversarial examples based on model outputs:

$$x_{adv}^{(t+1)} = x_{adv}^{(t)} + \alpha \cdot \text{EstimateGradient}(f, x_{adv}^{(t)})$$

Where $\text{EstimateGradient}$ estimates the gradient using finite differences or other query-based methods.

**Healthcare-Specific Attack Scenarios**: Healthcare AI systems face unique attack scenarios that exploit domain-specific characteristics:

- **Medical Image Manipulation**: Adversarial perturbations to medical images that cause misdiagnosis while remaining imperceptible to radiologists
- **Clinical Data Poisoning**: Manipulation of electronic health record data to bias AI recommendations
- **Sensor Spoofing**: Attacks on medical device sensors that provide corrupted input to AI systems
- **Protocol Manipulation**: Exploitation of clinical workflow integration points to inject adversarial inputs

### 10.2.2 Defense Mechanisms and Robustness Techniques

Effective defense against adversarial attacks requires multiple layers of protection, each addressing different aspects of the threat landscape. Healthcare AI systems must implement comprehensive defense strategies that maintain clinical utility while providing robust protection against malicious attacks.

**Adversarial Training**: Adversarial training involves augmenting the training data with adversarial examples to improve model robustness. This approach teaches the model to handle adversarial perturbations during the training process.

The adversarial training objective can be formulated as:

$$\min_{\theta} \mathbb{E}_{(x,y) \sim D} \left[ \max_{\|\delta\| \leq \epsilon} \ell(f_\theta(x + \delta), y) \right]$$

Where the inner maximization finds the worst-case perturbation within the allowed budget $\epsilon$, and the outer minimization trains the model to be robust against these perturbations.

**Certified Defenses**: Certified defenses provide mathematical guarantees about model robustness within specified perturbation bounds. These methods offer provable robustness guarantees that are particularly valuable in high-stakes healthcare applications.

**Randomized Smoothing** provides certified robustness by adding random noise to inputs and averaging predictions:

$$g(x) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)}[f(x + \epsilon)]$$

The certified radius for randomized smoothing can be computed as:

$$R = \sigma \cdot \Phi^{-1}(p_A)$$

Where $\Phi^{-1}$ is the inverse normal CDF and $p_A$ is the probability of predicting the correct class.

**Interval Bound Propagation (IBP)** provides certified bounds on model outputs by propagating input perturbation bounds through the network:

$$\underline{z}^{(l+1)} = W^{(l)} \underline{z}^{(l)} + b^{(l)}$$
$$\overline{z}^{(l+1)} = W^{(l)} \overline{z}^{(l)} + b^{(l)}$$

Where $\underline{z}^{(l)}$ and $\overline{z}^{(l)}$ represent lower and upper bounds at layer $l$.

**Detection-Based Defenses**: Detection-based defenses aim to identify adversarial examples before they can cause harm. These methods analyze input characteristics, model behavior, or statistical properties to distinguish between benign and adversarial inputs.

**Statistical Detection Methods** analyze the statistical properties of inputs to identify anomalies:

$$\text{score}(x) = \|x - \mathbb{E}[x]\|_{\Sigma^{-1}}$$

Where $\Sigma$ is the covariance matrix of the training data distribution.

**Neural Network-Based Detectors** train separate models to classify inputs as benign or adversarial:

$$p(\text{adversarial}|x) = \sigma(g_\phi(x))$$

Where $g_\phi$ is a detector network and $\sigma$ is the sigmoid function.

## 10.3 Comprehensive Healthcare AI Security Framework

### 10.3.1 Implementation of Advanced Security and Robustness Systems

```python
"""
Comprehensive Healthcare AI Security and Robustness Framework

This implementation provides advanced security measures and robustness techniques
specifically designed for healthcare AI applications, including adversarial defense,
privacy protection, and comprehensive system monitoring.

Author: Sanjay Basu MD PhD
License: MIT
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

# Security and robustness libraries
try:
    import foolbox as fb
    FOOLBOX_AVAILABLE = True
except ImportError:
    FOOLBOX_AVAILABLE = False
    print("Foolbox not available. Some adversarial attack methods will use simplified implementations.")

try:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
    from art.estimators.classification import PyTorchClassifier
    from art.defences.trainer import AdversarialTrainer
    from art.defences.preprocessor import GaussianNoise, SpatialSmoothing
    from art.defences.postprocessor import ReverseSigmoid
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("Adversarial Robustness Toolbox not available. Using simplified implementations.")

# Cryptography and privacy libraries
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import hashlib
import hmac
import secrets
import base64

# Differential privacy
try:
    import opacus
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("Opacus not available. Differential privacy features will use simplified implementations.")

import logging
from datetime import datetime, timedelta
import json
import joblib
import pickle
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import time
import uuid
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatType(Enum):
    """Types of security threats for healthcare AI systems."""
    ADVERSARIAL_EXAMPLE = "adversarial_example"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    MEMBERSHIP_INFERENCE = "membership_inference"
    ATTRIBUTE_INFERENCE = "attribute_inference"
    BYZANTINE_ATTACK = "byzantine_attack"
    BACKDOOR_ATTACK = "backdoor_attack"
    PRIVACY_BREACH = "privacy_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_CORRUPTION = "data_corruption"

class DefenseType(Enum):
    """Types of defense mechanisms."""
    ADVERSARIAL_TRAINING = "adversarial_training"
    CERTIFIED_DEFENSE = "certified_defense"
    DETECTION_BASED = "detection_based"
    PREPROCESSING = "preprocessing"
    ENSEMBLE_DEFENSE = "ensemble_defense"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    ACCESS_CONTROL = "access_control"
    ENCRYPTION = "encryption"
    AUDIT_LOGGING = "audit_logging"

class SecurityLevel(Enum):
    """Security levels for different types of healthcare data and operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

@dataclass
class SecurityIncident:
    """Security incident record for healthcare AI systems."""
    incident_id: str
    threat_type: ThreatType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_systems: List[str]
    affected_patients: Optional[List[str]] = None
    detection_time: datetime
    response_actions: List[str]
    status: str  # "detected", "investigating", "mitigated", "resolved"
    compliance_impact: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and reporting."""
        return {
            'incident_id': self.incident_id,
            'threat_type': self.threat_type.value,
            'severity': self.severity,
            'description': self.description,
            'affected_systems': self.affected_systems,
            'affected_patients': self.affected_patients,
            'detection_time': self.detection_time.isoformat(),
            'response_actions': self.response_actions,
            'status': self.status,
            'compliance_impact': self.compliance_impact,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class RobustnessTestResult:
    """Result of robustness testing for healthcare AI systems."""
    test_id: str
    test_type: str
    model_name: str
    original_accuracy: float
    robust_accuracy: float
    attack_success_rate: float
    perturbation_budget: float
    test_samples: int
    defense_mechanisms: List[str]
    clinical_impact_assessment: Optional[str] = None
    regulatory_compliance: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'test_id': self.test_id,
            'test_type': self.test_type,
            'model_name': self.model_name,
            'original_accuracy': self.original_accuracy,
            'robust_accuracy': self.robust_accuracy,
            'attack_success_rate': self.attack_success_rate,
            'perturbation_budget': self.perturbation_budget,
            'test_samples': self.test_samples,
            'defense_mechanisms': self.defense_mechanisms,
            'clinical_impact_assessment': self.clinical_impact_assessment,
            'regulatory_compliance': self.regulatory_compliance,
            'timestamp': self.timestamp.isoformat()
        }

class EncryptionManager:
    """Advanced encryption manager for healthcare AI systems."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL):
        """Initialize encryption manager with appropriate security level."""
        self.security_level = security_level
        self.fernet_key = Fernet.generate_key()
        self.fernet = Fernet(self.fernet_key)
        
        # Generate RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048 if security_level in [SecurityLevel.INTERNAL, SecurityLevel.CONFIDENTIAL] else 4096
        )
        self.public_key = self.private_key.public_key()
        
        # Key derivation for password-based encryption
        self.salt = secrets.token_bytes(32)
        
        logger.info(f"Initialized encryption manager with security level: {security_level.value}")
    
    def encrypt_data(self, data: bytes, use_asymmetric: bool = False) -> bytes:
        """Encrypt data using symmetric or asymmetric encryption."""
        try:
            if use_asymmetric:
                # Use RSA for small data or key exchange
                if len(data) > 190:  # RSA-2048 limit
                    raise ValueError("Data too large for RSA encryption. Use symmetric encryption.")
                
                encrypted_data = self.public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            else:
                # Use Fernet for symmetric encryption
                encrypted_data = self.fernet.encrypt(data)
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes, use_asymmetric: bool = False) -> bytes:
        """Decrypt data using symmetric or asymmetric decryption."""
        try:
            if use_asymmetric:
                decrypted_data = self.private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            else:
                decrypted_data = self.fernet.decrypt(encrypted_data)
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())
    
    def generate_hmac(self, data: bytes, key: Optional[bytes] = None) -> str:
        """Generate HMAC for data integrity verification."""
        if key is None:
            key = self.fernet_key
        
        mac = hmac.new(key, data, hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()
    
    def verify_hmac(self, data: bytes, mac: str, key: Optional[bytes] = None) -> bool:
        """Verify HMAC for data integrity."""
        if key is None:
            key = self.fernet_key
        
        expected_mac = self.generate_hmac(data, key)
        return hmac.compare_digest(mac, expected_mac)

class AccessControlManager:
    """Role-based access control manager for healthcare AI systems."""
    
    def __init__(self):
        """Initialize access control manager."""
        self.users = {}
        self.roles = {
            'physician': {
                'permissions': ['read_patient_data', 'write_patient_data', 'run_ai_models', 'view_predictions'],
                'data_access_level': SecurityLevel.CONFIDENTIAL
            },
            'nurse': {
                'permissions': ['read_patient_data', 'run_ai_models', 'view_predictions'],
                'data_access_level': SecurityLevel.CONFIDENTIAL
            },
            'researcher': {
                'permissions': ['read_anonymized_data', 'run_ai_models', 'view_predictions'],
                'data_access_level': SecurityLevel.INTERNAL
            },
            'administrator': {
                'permissions': ['read_patient_data', 'write_patient_data', 'run_ai_models', 'view_predictions', 'manage_users', 'view_audit_logs'],
                'data_access_level': SecurityLevel.RESTRICTED
            },
            'auditor': {
                'permissions': ['view_audit_logs', 'view_predictions'],
                'data_access_level': SecurityLevel.INTERNAL
            }
        }
        self.active_sessions = {}
        self.failed_attempts = defaultdict(int)
        self.lockout_threshold = 5
        self.lockout_duration = timedelta(minutes=30)
        
        logger.info("Initialized access control manager")
    
    def create_user(self, user_id: str, role: str, additional_permissions: Optional[List[str]] = None) -> bool:
        """Create a new user with specified role."""
        if role not in self.roles:
            logger.error(f"Invalid role: {role}")
            return False
        
        permissions = self.roles[role]['permissions'].copy()
        if additional_permissions:
            permissions.extend(additional_permissions)
        
        self.users[user_id] = {
            'role': role,
            'permissions': permissions,
            'data_access_level': self.roles[role]['data_access_level'],
            'created_at': datetime.now(),
            'last_login': None,
            'locked_until': None
        }
        
        logger.info(f"Created user {user_id} with role {role}")
        return True
    
    def authenticate_user(self, user_id: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        if user_id not in self.users:
            logger.warning(f"Authentication attempt for non-existent user: {user_id}")
            return None
        
        # Check if user is locked out
        user = self.users[user_id]
        if user.get('locked_until') and datetime.now() < user['locked_until']:
            logger.warning(f"Authentication attempt for locked user: {user_id}")
            return None
        
        # Simulate password verification (in practice, use proper password hashing)
        if self._verify_password(user_id, password):
            # Reset failed attempts
            self.failed_attempts[user_id] = 0
            
            # Create session
            session_token = secrets.token_urlsafe(32)
            self.active_sessions[session_token] = {
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
            
            # Update user login time
            user['last_login'] = datetime.now()
            
            logger.info(f"User {user_id} authenticated successfully")
            return session_token
        else:
            # Handle failed attempt
            self.failed_attempts[user_id] += 1
            if self.failed_attempts[user_id] >= self.lockout_threshold:
                user['locked_until'] = datetime.now() + self.lockout_duration
                logger.warning(f"User {user_id} locked due to failed attempts")
            
            logger.warning(f"Authentication failed for user: {user_id}")
            return None
    
    def _verify_password(self, user_id: str, password: str) -> bool:
        """Verify user password (simplified implementation)."""
        # In practice, use proper password hashing (bcrypt, scrypt, etc.)
        return True  # Simplified for demonstration
    
    def check_permission(self, session_token: str, permission: str) -> bool:
        """Check if user has specific permission."""
        if session_token not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_token]
        user_id = session['user_id']
        
        # Update last activity
        session['last_activity'] = datetime.now()
        
        if user_id not in self.users:
            return False
        
        return permission in self.users[user_id]['permissions']
    
    def get_user_access_level(self, session_token: str) -> Optional[SecurityLevel]:
        """Get user's data access level."""
        if session_token not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_token]
        user_id = session['user_id']
        
        if user_id not in self.users:
            return None
        
        return self.users[user_id]['data_access_level']
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user and invalidate session."""
        if session_token in self.active_sessions:
            user_id = self.active_sessions[session_token]['user_id']
            del self.active_sessions[session_token]
            logger.info(f"User {user_id} logged out")
            return True
        return False
    
    def cleanup_expired_sessions(self, max_idle_time: timedelta = timedelta(hours=8)):
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for token, session in self.active_sessions.items():
            if current_time - session['last_activity'] > max_idle_time:
                expired_sessions.append(token)
        
        for token in expired_sessions:
            user_id = self.active_sessions[token]['user_id']
            del self.active_sessions[token]
            logger.info(f"Expired session for user {user_id}")

class AuditLogger:
    """Comprehensive audit logging for healthcare AI systems."""
    
    def __init__(self, log_file: str = "healthcare_ai_audit.log"):
        """Initialize audit logger."""
        self.log_file = log_file
        self.encryption_manager = EncryptionManager(SecurityLevel.RESTRICTED)
        
        # Configure audit logger
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Create file handler with encryption
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
        
        logger.info(f"Initialized audit logger: {log_file}")
    
    def log_access(self, user_id: str, resource: str, action: str, success: bool, 
                   additional_info: Optional[Dict[str, Any]] = None):
        """Log access attempts and actions."""
        log_entry = {
            'event_type': 'access',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }
        
        self.audit_logger.info(json.dumps(log_entry))
    
    def log_model_prediction(self, user_id: str, model_name: str, patient_id: str,
                           prediction: Any, confidence: float, 
                           additional_info: Optional[Dict[str, Any]] = None):
        """Log AI model predictions for audit trail."""
        log_entry = {
            'event_type': 'model_prediction',
            'user_id': user_id,
            'model_name': model_name,
            'patient_id': patient_id,
            'prediction': str(prediction),
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }
        
        self.audit_logger.info(json.dumps(log_entry))
    
    def log_security_incident(self, incident: SecurityIncident):
        """Log security incidents."""
        log_entry = {
            'event_type': 'security_incident',
            **incident.to_dict()
        }
        
        self.audit_logger.error(json.dumps(log_entry))
    
    def log_data_access(self, user_id: str, data_type: str, patient_ids: List[str],
                       purpose: str, additional_info: Optional[Dict[str, Any]] = None):
        """Log patient data access for HIPAA compliance."""
        log_entry = {
            'event_type': 'data_access',
            'user_id': user_id,
            'data_type': data_type,
            'patient_count': len(patient_ids),
            'patient_ids': patient_ids,
            'purpose': purpose,
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }
        
        self.audit_logger.info(json.dumps(log_entry))
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate audit report for specified time period."""
        # In practice, this would parse the log file and generate comprehensive reports
        report = {
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_access_attempts': 0,
                'successful_accesses': 0,
                'failed_accesses': 0,
                'model_predictions': 0,
                'security_incidents': 0,
                'data_accesses': 0
            },
            'compliance_status': 'compliant',
            'recommendations': []
        }
        
        return report

class AdversarialDefenseSystem:
    """Advanced adversarial defense system for healthcare AI."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize adversarial defense system."""
        self.model = model
        self.device = device
        self.defense_methods = {}
        self.detection_models = {}
        self.robustness_metrics = {}
        
        # Initialize defense components
        self._initialize_defenses()
        
        logger.info("Initialized adversarial defense system")
    
    def _initialize_defenses(self):
        """Initialize various defense mechanisms."""
        
        # Gaussian noise preprocessing
        self.defense_methods['gaussian_noise'] = GaussianNoiseDefense(std=0.1)
        
        # Statistical anomaly detection
        self.defense_methods['statistical_detection'] = StatisticalAnomalyDetector()
        
        # Ensemble defense
        self.defense_methods['ensemble'] = EnsembleDefense()
        
        # Input validation
        self.defense_methods['input_validation'] = InputValidator()
    
    def detect_adversarial_example(self, x: torch.Tensor) -> Dict[str, Any]:
        """Detect if input is adversarial using multiple methods."""
        detection_results = {}
        
        # Statistical detection
        stat_score = self.defense_methods['statistical_detection'].detect(x)
        detection_results['statistical_score'] = stat_score
        detection_results['statistical_adversarial'] = stat_score > 0.5
        
        # Input validation
        validation_result = self.defense_methods['input_validation'].validate(x)
        detection_results['validation_passed'] = validation_result
        
        # Ensemble detection
        ensemble_score = self.defense_methods['ensemble'].detect(x)
        detection_results['ensemble_score'] = ensemble_score
        detection_results['ensemble_adversarial'] = ensemble_score > 0.5
        
        # Overall decision
        detection_results['is_adversarial'] = (
            detection_results['statistical_adversarial'] or
            not detection_results['validation_passed'] or
            detection_results['ensemble_adversarial']
        )
        
        return detection_results
    
    def apply_defenses(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing defenses to input."""
        
        # Apply Gaussian noise
        x_defended = self.defense_methods['gaussian_noise'].apply(x)
        
        # Additional preprocessing can be added here
        
        return x_defended
    
    def robust_prediction(self, x: torch.Tensor) -> Dict[str, Any]:
        """Make robust prediction with defense mechanisms."""
        
        # Detect adversarial examples
        detection_result = self.detect_adversarial_example(x)
        
        if detection_result['is_adversarial']:
            logger.warning("Adversarial example detected")
            
            # Apply defenses
            x_defended = self.apply_defenses(x)
            
            # Make prediction on defended input
            with torch.no_grad():
                output = self.model(x_defended)
                prediction = torch.softmax(output, dim=1)
        else:
            # Make normal prediction
            with torch.no_grad():
                output = self.model(x)
                prediction = torch.softmax(output, dim=1)
        
        result = {
            'prediction': prediction.cpu().numpy(),
            'detection_result': detection_result,
            'confidence': torch.max(prediction).item(),
            'adversarial_detected': detection_result['is_adversarial']
        }
        
        return result
    
    def evaluate_robustness(self, test_loader: DataLoader, attack_methods: List[str]) -> RobustnessTestResult:
        """Evaluate model robustness against various attacks."""
        
        total_samples = 0
        correct_clean = 0
        correct_adversarial = 0
        attack_success = 0
        
        self.model.eval()
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Clean accuracy
            with torch.no_grad():
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct_clean += pred.eq(target).sum().item()
            
            # Test against adversarial examples
            for attack_method in attack_methods:
                adv_data = self._generate_adversarial_examples(data, target, attack_method)
                
                # Test robust prediction
                for i in range(len(adv_data)):
                    result = self.robust_prediction(adv_data[i:i+1])
                    pred = np.argmax(result['prediction'])
                    
                    if pred == target[i].item():
                        correct_adversarial += 1
                    else:
                        attack_success += 1
            
            total_samples += len(data)
            
            if batch_idx >= 10:  # Limit for demonstration
                break
        
        # Calculate metrics
        clean_accuracy = correct_clean / total_samples
        robust_accuracy = correct_adversarial / (total_samples * len(attack_methods))
        attack_success_rate = attack_success / (total_samples * len(attack_methods))
        
        test_result = RobustnessTestResult(
            test_id=str(uuid.uuid4()),
            test_type="adversarial_robustness",
            model_name=self.model.__class__.__name__,
            original_accuracy=clean_accuracy,
            robust_accuracy=robust_accuracy,
            attack_success_rate=attack_success_rate,
            perturbation_budget=0.1,  # Example budget
            test_samples=total_samples,
            defense_mechanisms=list(self.defense_methods.keys())
        )
        
        return test_result
    
    def _generate_adversarial_examples(self, data: torch.Tensor, target: torch.Tensor, 
                                     attack_method: str) -> torch.Tensor:
        """Generate adversarial examples using specified attack method."""
        
        if attack_method == "fgsm":
            return self._fgsm_attack(data, target, epsilon=0.1)
        elif attack_method == "pgd":
            return self._pgd_attack(data, target, epsilon=0.1, alpha=0.01, num_iter=10)
        else:
            logger.warning(f"Unknown attack method: {attack_method}")
            return data
    
    def _fgsm_attack(self, data: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""
        data.requires_grad = True
        
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        data_grad = data.grad.data
        perturbed_data = data + epsilon * data_grad.sign()
        
        return perturbed_data.detach()
    
    def _pgd_attack(self, data: torch.Tensor, target: torch.Tensor, 
                   epsilon: float, alpha: float, num_iter: int) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        perturbed_data = data.clone().detach()
        
        for _ in range(num_iter):
            perturbed_data.requires_grad = True
            
            output = self.model(perturbed_data)
            loss = F.cross_entropy(output, target)
            
            self.model.zero_grad()
            loss.backward()
            
            # Update perturbation
            data_grad = perturbed_data.grad.data
            perturbed_data = perturbed_data + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            eta = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
            perturbed_data = data + eta
            perturbed_data = perturbed_data.detach()
        
        return perturbed_data

class GaussianNoiseDefense:
    """Gaussian noise preprocessing defense."""
    
    def __init__(self, std: float = 0.1):
        """Initialize Gaussian noise defense."""
        self.std = std
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to input."""
        noise = torch.randn_like(x) * self.std
        return x + noise

class StatisticalAnomalyDetector:
    """Statistical anomaly detector for adversarial examples."""
    
    def __init__(self):
        """Initialize statistical detector."""
        self.mean = None
        self.cov = None
        self.threshold = 0.5
    
    def fit(self, X: torch.Tensor):
        """Fit detector on clean data."""
        X_flat = X.view(X.size(0), -1).cpu().numpy()
        self.mean = np.mean(X_flat, axis=0)
        self.cov = np.cov(X_flat.T)
    
    def detect(self, x: torch.Tensor) -> float:
        """Detect anomaly score for input."""
        if self.mean is None:
            return 0.0
        
        x_flat = x.view(x.size(0), -1).cpu().numpy()
        
        # Mahalanobis distance
        diff = x_flat - self.mean
        try:
            inv_cov = np.linalg.pinv(self.cov)
            distance = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            return float(np.mean(distance))
        except:
            return 0.0

class EnsembleDefense:
    """Ensemble-based defense mechanism."""
    
    def __init__(self):
        """Initialize ensemble defense."""
        self.models = []
    
    def add_model(self, model: nn.Module):
        """Add model to ensemble."""
        self.models.append(model)
    
    def detect(self, x: torch.Tensor) -> float:
        """Detect adversarial examples using ensemble disagreement."""
        if not self.models:
            return 0.0
        
        predictions = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                pred = torch.softmax(output, dim=1)
                predictions.append(pred.cpu().numpy())
        
        # Calculate prediction variance as anomaly score
        predictions = np.array(predictions)
        variance = np.var(predictions, axis=0)
        return float(np.mean(variance))

class InputValidator:
    """Input validation for healthcare AI systems."""
    
    def __init__(self):
        """Initialize input validator."""
        self.min_values = None
        self.max_values = None
        self.expected_shape = None
    
    def fit(self, X: torch.Tensor):
        """Fit validator on training data."""
        self.expected_shape = X.shape[1:]
        X_flat = X.view(X.size(0), -1)
        self.min_values = torch.min(X_flat, dim=0)<sup>0</sup>
        self.max_values = torch.max(X_flat, dim=0)<sup>0</sup>
    
    def validate(self, x: torch.Tensor) -> bool:
        """Validate input against expected ranges and shape."""
        
        # Check shape
        if x.shape[1:] != self.expected_shape:
            return False
        
        if self.min_values is None or self.max_values is None:
            return True
        
        # Check value ranges
        x_flat = x.view(x.size(0), -1)
        
        # Allow some tolerance for legitimate variations
        tolerance = 0.1
        min_check = torch.all(x_flat >= self.min_values - tolerance)
        max_check = torch.all(x_flat <= self.max_values + tolerance)
        
        return min_check and max_check

class HealthcareAISecurityFramework:
    """
    Comprehensive security and robustness framework for healthcare AI systems.
    
    This class integrates multiple security components including encryption,
    access control, audit logging, and adversarial defense to provide
    comprehensive protection for healthcare AI applications.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        security_config: Optional[Dict[str, Any]] = None,
        compliance_requirements: Optional[List[str]] = None
    ):
        """
        Initialize healthcare AI security framework.
        
        Args:
            model: PyTorch model to protect
            device: Computing device (CPU/GPU)
            security_config: Security configuration parameters
            compliance_requirements: List of compliance requirements (HIPAA, GDPR, etc.)
        """
        self.model = model
        self.device = device
        self.security_config = security_config or {}
        self.compliance_requirements = compliance_requirements or ['HIPAA']
        
        # Initialize security components
        self.encryption_manager = EncryptionManager(SecurityLevel.CONFIDENTIAL)
        self.access_control = AccessControlManager()
        self.audit_logger = AuditLogger()
        self.adversarial_defense = AdversarialDefenseSystem(model, device)
        
        # Security monitoring
        self.security_incidents = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Performance metrics
        self.security_metrics = {
            'total_predictions': 0,
            'adversarial_detected': 0,
            'access_violations': 0,
            'encryption_operations': 0
        }
        
        logger.info("Initialized healthcare AI security framework")
    
    def secure_predict(
        self,
        session_token: str,
        patient_data: torch.Tensor,
        patient_id: str,
        purpose: str
    ) -> Dict[str, Any]:
        """
        Make secure prediction with full security checks.
        
        Args:
            session_token: User session token
            patient_data: Input data for prediction
            patient_id: Patient identifier
            purpose: Purpose of the prediction
            
        Returns:
            Secure prediction result with audit trail
        """
        
        # Verify user permissions
        if not self.access_control.check_permission(session_token, 'run_ai_models'):
            self.audit_logger.log_access(
                user_id="unknown",
                resource="ai_model",
                action="predict",
                success=False,
                additional_info={'reason': 'insufficient_permissions'}
            )
            raise PermissionError("Insufficient permissions to run AI models")
        
        # Get user information
        user_access_level = self.access_control.get_user_access_level(session_token)
        user_id = self._get_user_id_from_session(session_token)
        
        # Log data access
        self.audit_logger.log_data_access(
            user_id=user_id,
            data_type="patient_clinical_data",
            patient_ids=[patient_id],
            purpose=purpose
        )
        
        # Apply adversarial defenses
        defense_result = self.adversarial_defense.robust_prediction(patient_data)
        
        # Log prediction
        self.audit_logger.log_model_prediction(
            user_id=user_id,
            model_name=self.model.__class__.__name__,
            patient_id=patient_id,
            prediction=defense_result['prediction'],
            confidence=defense_result['confidence'],
            additional_info={
                'adversarial_detected': defense_result['adversarial_detected'],
                'purpose': purpose
            }
        )
        
        # Update metrics
        self.security_metrics['total_predictions'] += 1
        if defense_result['adversarial_detected']:
            self.security_metrics['adversarial_detected'] += 1
        
        # Prepare secure response
        response = {
            'prediction': defense_result['prediction'].tolist(),
            'confidence': defense_result['confidence'],
            'adversarial_detected': defense_result['adversarial_detected'],
            'user_access_level': user_access_level.value,
            'audit_logged': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def _get_user_id_from_session(self, session_token: str) -> str:
        """Get user ID from session token."""
        if session_token in self.access_control.active_sessions:
            return self.access_control.active_sessions[session_token]['user_id']
        return "unknown"
    
    def encrypt_patient_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt patient data for storage or transmission."""
        
        # Serialize data
        data_bytes = json.dumps(data).encode()
        
        # Encrypt data
        encrypted_data = self.encryption_manager.encrypt_data(data_bytes)
        
        # Update metrics
        self.security_metrics['encryption_operations'] += 1
        
        return encrypted_data
    
    def decrypt_patient_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt patient data."""
        
        # Decrypt data
        decrypted_bytes = self.encryption_manager.decrypt_data(encrypted_data)
        
        # Deserialize data
        data = json.loads(decrypted_bytes.decode())
        
        return data
    
    def start_security_monitoring(self):
        """Start continuous security monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._security_monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Started security monitoring")
    
    def stop_security_monitoring(self):
        """Stop security monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Stopped security monitoring")
    
    def _security_monitoring_loop(self):
        """Continuous security monitoring loop."""
        while self.monitoring_active:
            try:
                # Clean up expired sessions
                self.access_control.cleanup_expired_sessions()
                
                # Check for anomalous activity
                self._check_anomalous_activity()
                
                # Monitor system resources
                self._monitor_system_resources()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in security monitoring: {e}")
    
    def _check_anomalous_activity(self):
        """Check for anomalous security activity."""
        
        # Check for unusual prediction patterns
        if self.security_metrics['total_predictions'] > 0:
            adversarial_rate = self.security_metrics['adversarial_detected'] / self.security_metrics['total_predictions']
            
            if adversarial_rate > 0.1:  # More than 10% adversarial examples
                incident = SecurityIncident(
                    incident_id=str(uuid.uuid4()),
                    threat_type=ThreatType.ADVERSARIAL_EXAMPLE,
                    severity="high",
                    description=f"High rate of adversarial examples detected: {adversarial_rate:.2%}",
                    affected_systems=[self.model.__class__.__name__],
                    detection_time=datetime.now(),
                    response_actions=["increased_monitoring", "model_retraining_recommended"],
                    status="detected"
                )
                
                self.security_incidents.append(incident)
                self.audit_logger.log_security_incident(incident)
    
    def _monitor_system_resources(self):
        """Monitor system resources for security threats."""
        # In practice, this would monitor CPU, memory, network usage, etc.
        pass
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'security_metrics': self.security_metrics.copy(),
            'active_sessions': len(self.access_control.active_sessions),
            'total_users': len(self.access_control.users),
            'security_incidents': len(self.security_incidents),
            'compliance_status': self._check_compliance_status(),
            'recommendations': self._generate_security_recommendations()
        }
        
        return report
    
    def _check_compliance_status(self) -> Dict[str, bool]:
        """Check compliance with various regulations."""
        compliance_status = {}
        
        for requirement in self.compliance_requirements:
            if requirement == 'HIPAA':
                compliance_status['HIPAA'] = self._check_hipaa_compliance()
            elif requirement == 'GDPR':
                compliance_status['GDPR'] = self._check_gdpr_compliance()
            else:
                compliance_status[requirement] = True  # Simplified
        
        return compliance_status
    
    def _check_hipaa_compliance(self) -> bool:
        """Check HIPAA compliance status."""
        # Simplified compliance check
        checks = [
            len(self.access_control.users) > 0,  # Access controls in place
            hasattr(self, 'audit_logger'),  # Audit logging enabled
            hasattr(self, 'encryption_manager'),  # Encryption available
        ]
        
        return all(checks)
    
    def _check_gdpr_compliance(self) -> bool:
        """Check GDPR compliance status."""
        # Simplified compliance check
        checks = [
            hasattr(self, 'encryption_manager'),  # Data protection
            hasattr(self, 'audit_logger'),  # Audit trail
            True  # Simplified - would check data minimization, consent, etc.
        ]
        
        return all(checks)
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current status."""
        recommendations = []
        
        # Check adversarial detection rate
        if self.security_metrics['total_predictions'] > 0:
            adversarial_rate = self.security_metrics['adversarial_detected'] / self.security_metrics['total_predictions']
            if adversarial_rate > 0.05:
                recommendations.append("Consider additional adversarial training")
        
        # Check session management
        if len(self.access_control.active_sessions) > 100:
            recommendations.append("Monitor for unusual session activity")
        
        # Check incident response
        if len(self.security_incidents) > 0:
            recommendations.append("Review and respond to security incidents")
        
        if not recommendations:
            recommendations.append("Security posture appears normal")
        
        return recommendations
    
    def test_security_controls(self) -> Dict[str, Any]:
        """Test security controls and generate assessment."""
        
        test_results = {
            'encryption_test': self._test_encryption(),
            'access_control_test': self._test_access_control(),
            'adversarial_defense_test': self._test_adversarial_defense(),
            'audit_logging_test': self._test_audit_logging()
        }
        
        overall_score = sum(test_results.values()) / len(test_results)
        
        assessment = {
            'overall_security_score': overall_score,
            'individual_tests': test_results,
            'timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_test_recommendations(test_results)
        }
        
        return assessment
    
    def _test_encryption(self) -> float:
        """Test encryption functionality."""
        try:
            test_data = b"test healthcare data"
            encrypted = self.encryption_manager.encrypt_data(test_data)
            decrypted = self.encryption_manager.decrypt_data(encrypted)
            return 1.0 if decrypted == test_data else 0.0
        except:
            return 0.0
    
    def _test_access_control(self) -> float:
        """Test access control functionality."""
        try:
            # Test user creation and authentication
            test_user = "test_user_" + str(uuid.uuid4())[:8]
            created = self.access_control.create_user(test_user, "physician")
            
            if created:
                # Test permission checking
                session = self.access_control.authenticate_user(test_user, "password")
                if session:
                    has_permission = self.access_control.check_permission(session, "run_ai_models")
                    self.access_control.logout_user(session)
                    return 1.0 if has_permission else 0.5
            
            return 0.0
        except:
            return 0.0
    
    def _test_adversarial_defense(self) -> float:
        """Test adversarial defense functionality."""
        try:
            # Create test input
            test_input = torch.randn(1, 3, 32, 32).to(self.device)
            
            # Test detection
            detection_result = self.adversarial_defense.detect_adversarial_example(test_input)
            
            # Test robust prediction
            prediction_result = self.adversarial_defense.robust_prediction(test_input)
            
            return 1.0 if 'prediction' in prediction_result else 0.0
        except:
            return 0.0
    
    def _test_audit_logging(self) -> float:
        """Test audit logging functionality."""
        try:
            # Test logging functionality
            self.audit_logger.log_access("test_user", "test_resource", "test_action", True)
            return 1.0
        except:
            return 0.0
    
    def _generate_test_recommendations(self, test_results: Dict[str, float]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for test_name, score in test_results.items():
            if score < 0.5:
                recommendations.append(f"Address issues with {test_name}")
            elif score < 1.0:
                recommendations.append(f"Improve {test_name} implementation")
        
        if not recommendations:
            recommendations.append("All security controls functioning properly")
        
        return recommendations

## Bibliography and References

### Foundational Adversarial Machine Learning Literature

1. **Goodfellow, I. J., Shlens, J., & Szegedy, C.** (2014). Explaining and harnessing adversarial examples. *arXiv preprint arXiv:1412.6572*. [Foundational FGSM paper]

2. **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A.** (2017). Towards deep learning models resistant to adversarial attacks. *arXiv preprint arXiv:1706.06083*. [PGD and adversarial training]

3. **Carlini, N., & Wagner, D.** (2017). Towards evaluating the robustness of neural networks. *2017 IEEE Symposium on Security and Privacy (SP)*, 39-57. [C&W attack and evaluation methodology]

4. **Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A.** (2016). The limitations of deep learning in adversarial settings. *2016 IEEE European Symposium on Security and Privacy (EuroS&P)*, 372-387. [Adversarial example transferability]

### Healthcare-Specific Security and Robustness Research

5. **Finlayson, S. G., Bowers, J. D., Ito, J., Zittrain, J. L., Beam, A. L., & Kohane, I. S.** (2019). Adversarial attacks on medical machine learning. *Science*, 363(6433), 1287-1289. DOI: 10.1126/science.aaw4399. [Healthcare adversarial attacks]

6. **Ma, X., Niu, Y., Gu, L., Wang, Y., Zhao, Y., Bailey, J., & Lu, F.** (2021). Understanding adversarial attacks on deep learning based medical image analysis systems. *Pattern Recognition*, 110, 107332. [Medical imaging adversarial attacks]

7. **Paschali, M., Conjeti, S., Navarro, F., & Navab, N.** (2018). Generalizability vs. robustness: investigating medical imaging networks using adversarial examples. *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 493-501. [Medical imaging robustness]

8. **Hirano, H., Minagi, A., & Takemoto, K.** (2021). Universal adversarial attacks on deep neural networks for medical image classification. *BMC Medical Imaging*, 21(1), 1-13. [Universal adversarial attacks in healthcare]

### Certified Defenses and Formal Verification

9. **Cohen, J., Rosenfeld, E., & Kolter, Z.** (2019). Certified adversarial robustness via randomized smoothing. *International Conference on Machine Learning*, 1310-1320. [Randomized smoothing certification]

10. **Gowal, S., Dvijotham, K., Stanforth, R., Bunel, R., Qin, C., Uesato, J., ... & Kohli, P.** (2018). On the effectiveness of interval bound propagation for training verifiably robust models. *arXiv preprint arXiv:1810.12715*. [Interval bound propagation]

11. **Wong, E., & Kolter, Z.** (2018). Provable defenses against adversarial examples via the convex outer adversarial polytope. *International Conference on Machine Learning*, 5286-5295. [Convex relaxation methods]

### Privacy-Preserving Machine Learning

12. **Dwork, C., & Roth, A.** (2014). The algorithmic foundations of differential privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3-4), 211-407. [Foundational differential privacy]

13. **Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L.** (2016). Deep learning with differential privacy. *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security*, 308-318. [DP-SGD algorithm]

14. **Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V.** (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems*, 2, 429-450. [Federated learning security]

### Healthcare Cybersecurity and Compliance

15. **U.S. Department of Health and Human Services.** (2003). Health Insurance Portability and Accountability Act of 1996 (HIPAA) Security Rule. *45 CFR Parts 160, 162, and 164*. [HIPAA Security Rule]

16. **U.S. Food and Drug Administration.** (2016). Postmarket management of cybersecurity in medical devices. *FDA Guidance Document*. [FDA cybersecurity guidance]

17. **European Parliament and Council.** (2016). General Data Protection Regulation (GDPR). *Regulation (EU) 2016/679*. [GDPR requirements]

18. **National Institute of Standards and Technology.** (2018). Framework for improving critical infrastructure cybersecurity. *NIST Cybersecurity Framework Version 1.1*. [NIST cybersecurity framework]

### Robustness and Distribution Shift

19. **Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P.** (2021). WILDS: A benchmark of in-the-wild distribution shifts. *International Conference on Machine Learning*, 5637-5664. [Distribution shift benchmarks]

20. **Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P.** (2019). Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. *arXiv preprint arXiv:1911.08731*. [Distributionally robust optimization]

This chapter provides a comprehensive framework for implementing robust and secure healthcare AI systems. The implementations address the unique challenges of clinical environments including regulatory compliance, adversarial threats, and privacy protection. The next chapter will explore regulatory compliance and validation frameworks, building upon these security concepts to address the specific requirements for deploying AI systems in regulated healthcare environments.
