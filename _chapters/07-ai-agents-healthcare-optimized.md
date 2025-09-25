---
layout: default
title: "Chapter 7: AI Agents in Healthcare - Autonomous Systems for Clinical Decision Support and Care Coordination"
nav_order: 7
parent: Chapters
has_children: false
---

# Chapter 7: AI Agents in Healthcare - Autonomous Systems for Clinical Decision Support and Care Coordination

*By Sanjay Basu MD PhD*

## Learning Objectives

By the end of this chapter, physician data scientists will be able to:

- Understand the theoretical foundations of AI agents and their specific applications in healthcare environments, including the Belief-Desire-Intention (BDI) architecture and multi-agent coordination mechanisms
- Design and implement autonomous AI agents for clinical decision support, care coordination, and resource management with comprehensive safety frameworks and human oversight integration
- Deploy multi-agent systems for complex healthcare workflows including care pathway optimization, resource allocation, and cross-departmental coordination
- Evaluate agent performance using appropriate clinical metrics, safety frameworks, and real-world effectiveness measures that account for patient outcomes and system efficiency
- Address ethical considerations in autonomous healthcare AI systems including patient autonomy, beneficence, non-maleficence, and justice principles
- Implement robust human-AI collaboration frameworks for clinical practice that enhance rather than replace clinical judgment and maintain appropriate oversight and accountability
- Develop agent communication protocols and coordination mechanisms that ensure seamless integration with existing clinical workflows and electronic health record systems

## 7.1 Introduction to AI Agents in Healthcare

Artificial Intelligence agents represent a paradigm shift from traditional AI tools to autonomous systems capable of perceiving their environment, making decisions, and taking actions to achieve specific clinical goals. In healthcare, AI agents are emerging as powerful solutions for complex challenges including care coordination, resource allocation, clinical decision support, patient monitoring, and population health management. These systems move beyond reactive AI models to proactive agents that can anticipate clinical needs, coordinate care activities, and adapt their behavior based on changing patient conditions and healthcare system dynamics.

The concept of AI agents in healthcare builds upon decades of research in artificial intelligence, multi-agent systems, clinical informatics, and healthcare operations research. Unlike passive AI models that respond to queries or analyze data when prompted, healthcare AI agents actively monitor patient conditions, anticipate clinical needs, coordinate care activities across multiple providers and departments, and adapt their behavior based on changing clinical circumstances and emerging evidence. This autonomous capability enables healthcare systems to provide more proactive, coordinated, and efficient care while reducing the cognitive burden on healthcare providers.

### 7.1.1 Theoretical Foundations of Healthcare AI Agents

Healthcare AI agents are grounded in several key theoretical frameworks that distinguish them from traditional AI systems and provide the mathematical and conceptual foundation for their operation in clinical environments. Understanding these foundations is essential for physician data scientists who need to design, implement, and validate agent-based systems for healthcare applications.

**Agent Architecture and the BDI Model**: Healthcare AI agents typically follow the Belief-Desire-Intention (BDI) architecture, which provides a structured approach to autonomous decision-making that aligns well with clinical reasoning processes. In this framework, beliefs represent the agent's understanding of the current clinical situation based on available data and observations, desires represent the clinical goals and outcomes the agent aims to achieve, and intentions represent the specific actions and plans the agent commits to executing.

The formal representation of a healthcare AI agent can be expressed as:

$$Agent = \langle B, D, I, \pi, \rho, \sigma \rangle$$

where:
- $B$ is the set of beliefs about patient state and clinical environment
- $D$ is the set of desires representing clinical goals and outcomes
- $I$ is the set of intentions representing committed plans and actions
- $\pi$ is the planning function that generates action sequences: $\pi: B \times D \rightarrow I$
- $\rho$ is the revision function that updates beliefs based on new observations: $\rho: B \times O \rightarrow B$
- $\sigma$ is the selection function that chooses which desires to pursue: $\sigma: B \times D \rightarrow D'$

**Rational Agency in Clinical Contexts**: Healthcare AI agents must exhibit rational behavior, making decisions that maximize expected clinical utility while minimizing risks and adhering to clinical guidelines. The rationality principle in healthcare contexts requires agents to optimize a multi-objective utility function that balances patient outcomes, resource utilization, safety considerations, and ethical constraints.

The utility function for a healthcare agent can be expressed as:

$$U(a, s) = \sum_{i} w_i \cdot u_i(a, s)$$

where $a$ represents an action, $s$ represents the current state, $w_i$ are weights for different objectives, and $u_i$ are utility functions for:
- Patient clinical outcomes and quality of life
- Safety and risk minimization
- Resource efficiency and cost-effectiveness
- Adherence to clinical guidelines and evidence-based practices
- Patient satisfaction and experience

**Multi-Agent Coordination and Communication**: Healthcare delivery inherently involves multiple stakeholders, making multi-agent coordination essential for effective care delivery. The coordination mechanisms must handle complex interactions between different types of agents representing various clinical roles, departments, and healthcare systems.

Key coordination mechanisms include:
- **Contract Net Protocol** for dynamic task allocation among clinical agents
- **Consensus Algorithms** for collaborative decision-making in multidisciplinary care
- **Auction Mechanisms** for efficient resource allocation and scheduling
- **Negotiation Protocols** for resolving conflicts between competing clinical priorities

### 7.1.2 Healthcare-Specific Agent Characteristics

Healthcare AI agents must possess unique characteristics that distinguish them from general-purpose AI agents and ensure their safe and effective operation in clinical environments. These characteristics address the high-stakes nature of healthcare decisions, regulatory requirements, and the complex social and ethical dimensions of medical care.

**Clinical Safety and Risk Management**: Healthcare agents must incorporate multiple layers of safety mechanisms to prevent harm and ensure appropriate clinical decision-making. These include fail-safe behaviors that default to conservative actions when uncertainty is high, mandatory human oversight integration for critical decisions, comprehensive uncertainty quantification to communicate confidence levels to clinicians, and detailed audit trails for accountability and continuous learning.

The safety framework for healthcare agents can be formalized as a constraint satisfaction problem where actions must satisfy:

$$\forall a \in A: \text{Safety}(a, s) \geq \theta_{safety} \land \text{Confidence}(a, s) \geq \theta_{confidence}$$

where $A$ is the set of possible actions, $s$ is the current state, and $\theta_{safety}$ and $\theta_{confidence}$ are minimum thresholds for safety and confidence respectively.

**Regulatory Compliance and Standards Adherence**: Healthcare agents must comply with complex regulatory frameworks including HIPAA privacy requirements for patient data protection, FDA guidelines for medical device software, clinical practice standards for evidence-based care, and institutional policies for AI system deployment. This requires built-in compliance checking mechanisms and documentation systems that ensure all agent actions meet regulatory requirements.

**Ethical Decision-Making Framework**: Healthcare agents must embody fundamental ethical principles including beneficence (acting in the patient's best interest), non-maleficence (avoiding harm to patients), autonomy (respecting patient preferences and choices), and justice (ensuring fair and equitable treatment). These principles must be operationalized in the agent's decision-making algorithms and conflict resolution mechanisms.

**Interoperability and Integration**: Healthcare agents must seamlessly integrate with existing clinical workflows, electronic health record systems, and healthcare information technology infrastructure. This requires standardized communication protocols, data format compatibility, and workflow integration capabilities that minimize disruption to clinical practice.

### 7.1.3 Applications Landscape and Use Cases

Healthcare AI agents are being deployed across multiple domains, each with specific requirements, challenges, and opportunities for improving patient care and healthcare system efficiency. Understanding this landscape is essential for identifying appropriate use cases and implementing effective agent-based solutions.

**Clinical Decision Support Agents**: Autonomous systems that continuously monitor patient data, identify clinical patterns and trends, and provide real-time recommendations to clinicians. These agents can detect early warning signs of clinical deterioration, suggest diagnostic workups, recommend treatment modifications, and alert providers to potential drug interactions or contraindications.

**Care Coordination Agents**: Systems that manage complex care pathways, coordinate appointments and referrals, ensure continuity of care across multiple providers, and facilitate communication between different members of the healthcare team. These agents can optimize care transitions, reduce delays in treatment, and ensure that all necessary care components are delivered in a timely and coordinated manner.

**Resource Management Agents**: Agents that optimize hospital resources including bed allocation, staff scheduling, equipment utilization, and supply chain management. These systems can predict resource needs, optimize allocation algorithms, and coordinate resource sharing across different departments and facilities.

**Patient Monitoring and Engagement Agents**: Continuous monitoring systems that track patient vital signs, medication adherence, symptom progression, and lifestyle factors. These agents can provide personalized health coaching, medication reminders, and early intervention for health issues.

**Population Health Management Agents**: Systems that analyze population-level health data, identify health trends and disparities, and coordinate public health interventions. These agents can support disease surveillance, outbreak response, and preventive care initiatives.

## 7.2 Architecture and Implementation of Healthcare AI Agents

### 7.2.1 Core Agent Architecture

The implementation of healthcare AI agents requires a sophisticated architecture that balances autonomy with safety, efficiency with explainability, and individual patient care with population health considerations. The architecture must support real-time decision-making while maintaining comprehensive audit trails and enabling human oversight at appropriate points in the decision-making process.

```python
"""
Comprehensive Healthcare AI Agent Framework

This implementation provides advanced autonomous agent capabilities specifically
designed for healthcare environments, including clinical decision support,
care coordination, and resource management with comprehensive safety frameworks
and regulatory compliance.

Author: Sanjay Basu MD PhD
License: MIT
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from collections import defaultdict, deque
import threading
import queue
import time
import uuid
import pickle
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Enumeration of possible agent states in healthcare environments."""
    IDLE = "idle"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMMUNICATING = "communicating"
    WAITING_APPROVAL = "waiting_approval"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"

class Priority(Enum):
    """Priority levels for agent tasks and communications."""
    CRITICAL = 1    # Life-threatening situations
    URGENT = 2      # Time-sensitive clinical issues
    HIGH = 3        # Important but not urgent
    MEDIUM = 4      # Routine clinical tasks
    LOW = 5         # Administrative or background tasks

class ActionType(Enum):
    """Types of actions that healthcare agents can perform."""
    MONITOR = "monitor"
    ALERT = "alert"
    RECOMMEND = "recommend"
    COORDINATE = "coordinate"
    SCHEDULE = "schedule"
    DOCUMENT = "document"
    ANALYZE = "analyze"
    COMMUNICATE = "communicate"

@dataclass
class ClinicalBelief:
    """Represents an agent's belief about the clinical environment."""
    belief_id: str
    content: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    source: str
    evidence: List[str] = field(default_factory=list)
    clinical_relevance: float = 1.0
    expiry_time: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if belief is expired based on expiry time or age."""
        if self.expiry_time:
            return datetime.now() > self.expiry_time
        
        # Default expiry based on content type
        max_age_hours = {
            'vital_signs': 1,
            'lab_results': 24,
            'medication_status': 8,
            'patient_location': 0.5,
            'clinical_assessment': 12
        }.get(self.content.get('type', 'general'), 24)
        
        return (datetime.now() - self.timestamp).total_seconds() > max_age_hours * 3600
    
    def update_confidence(self, new_evidence: str, confidence_delta: float):
        """Update belief confidence based on new evidence."""
        self.evidence.append(new_evidence)
        self.confidence = max(0.0, min(1.0, self.confidence + confidence_delta))
        self.timestamp = datetime.now()

@dataclass
class ClinicalDesire:
    """Represents an agent's desire (goal) in the clinical context."""
    goal_id: str
    description: str
    target_outcome: Dict[str, Any]
    priority: Priority
    patient_id: Optional[str] = None
    deadline: Optional[datetime] = None
    constraints: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    clinical_rationale: str = ""
    
    def is_overdue(self) -> bool:
        """Check if desire is overdue."""
        return self.deadline and datetime.now() > self.deadline
    
    def calculate_urgency_score(self) -> float:
        """Calculate urgency score based on priority and deadline."""
        base_score = {
            Priority.CRITICAL: 1.0,
            Priority.URGENT: 0.8,
            Priority.HIGH: 0.6,
            Priority.MEDIUM: 0.4,
            Priority.LOW: 0.2
        }[self.priority]
        
        if self.deadline:
            time_remaining = (self.deadline - datetime.now()).total_seconds()
            if time_remaining <= 0:
                return 1.0  # Overdue
            elif time_remaining <= 3600:  # Less than 1 hour
                return min(1.0, base_score + 0.3)
        
        return base_score

@dataclass
class ClinicalIntention:
    """Represents an agent's intention (committed plan) for clinical action."""
    plan_id: str
    goal_id: str
    actions: List[Dict[str, Any]]
    current_step: int = 0
    status: str = "planned"
    estimated_completion: Optional[datetime] = None
    safety_checks: List[str] = field(default_factory=list)
    approval_required: bool = False
    approved_by: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if intention is complete."""
        return self.current_step >= len(self.actions)
    
    def get_next_action(self) -> Optional[Dict[str, Any]]:
        """Get the next action to execute."""
        if self.is_complete():
            return None
        return self.actions[self.current_step]
    
    def advance_step(self):
        """Advance to the next step in the plan."""
        if not self.is_complete():
            self.current_step += 1
    
    def requires_approval(self) -> bool:
        """Check if current action requires human approval."""
        if self.is_complete():
            return False
        
        current_action = self.get_next_action()
        return (self.approval_required or 
                current_action.get('requires_approval', False) or
                current_action.get('action_type') in ['medication_order', 'procedure_order'])

@dataclass
class AgentMessage:
    """Represents a message between healthcare agents."""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    priority: Priority
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requires_response: bool = False
    clinical_context: Optional[str] = None
    
    def __lt__(self, other):
        """Enable priority queue ordering."""
        return self.priority.value < other.priority.value

class ClinicalKnowledgeBase:
    """Clinical knowledge base for healthcare agents."""
    
    def __init__(self):
        """Initialize clinical knowledge base."""
        
        # Clinical guidelines and protocols
        self.guidelines = self._load_clinical_guidelines()
        
        # Drug information and interactions
        self.drug_database = self._load_drug_database()
        
        # Clinical decision rules
        self.decision_rules = self._load_decision_rules()
        
        # Normal ranges and reference values
        self.reference_ranges = self._load_reference_ranges()
        
        # Clinical pathways
        self.care_pathways = self._load_care_pathways()
        
        logger.info("Initialized clinical knowledge base")
    
    def _load_clinical_guidelines(self) -> Dict[str, Dict]:
        """Load clinical practice guidelines."""
        
        return {
            'sepsis_management': {
                'criteria': {
                    'qsofa_score': '>=2',
                    'lactate': '>2.0',
                    'systolic_bp': '<100'
                },
                'interventions': [
                    'blood_cultures',
                    'broad_spectrum_antibiotics',
                    'fluid_resuscitation',
                    'vasopressors_if_needed'
                ],
                'timeframes': {
                    'antibiotics': '1_hour',
                    'cultures': '1_hour',
                    'fluids': '3_hours'
                }
            },
            'heart_failure_management': {
                'criteria': {
                    'ejection_fraction': '<40%',
                    'bnp': '>400',
                    'symptoms': 'dyspnea_edema'
                },
                'interventions': [
                    'ace_inhibitor',
                    'beta_blocker',
                    'diuretic',
                    'aldosterone_antagonist'
                ],
                'monitoring': [
                    'daily_weights',
                    'electrolytes',
                    'kidney_function'
                ]
            },
            'diabetes_management': {
                'targets': {
                    'hba1c': '<7%',
                    'fasting_glucose': '80-130',
                    'postprandial_glucose': '<180'
                },
                'first_line': 'metformin',
                'monitoring': 'hba1c_every_3_months'
            }
        }
    
    def _load_drug_database(self) -> Dict[str, Dict]:
        """Load drug information database."""
        
        return {
            'metformin': {
                'class': 'biguanide',
                'indications': ['type_2_diabetes'],
                'contraindications': ['kidney_disease', 'liver_disease'],
                'interactions': ['contrast_media', 'alcohol'],
                'monitoring': ['kidney_function', 'b12_levels'],
                'dosing': {
                    'initial': '500mg_twice_daily',
                    'maximum': '2000mg_daily'
                }
            },
            'lisinopril': {
                'class': 'ace_inhibitor',
                'indications': ['hypertension', 'heart_failure'],
                'contraindications': ['pregnancy', 'hyperkalemia'],
                'interactions': ['potassium_supplements', 'nsaids'],
                'monitoring': ['kidney_function', 'potassium', 'blood_pressure'],
                'dosing': {
                    'initial': '5mg_daily',
                    'maximum': '40mg_daily'
                }
            },
            'warfarin': {
                'class': 'anticoagulant',
                'indications': ['atrial_fibrillation', 'dvt', 'pe'],
                'contraindications': ['active_bleeding', 'pregnancy'],
                'interactions': ['aspirin', 'antibiotics', 'vitamin_k'],
                'monitoring': ['inr', 'bleeding_signs'],
                'target_inr': {
                    'atrial_fibrillation': '2.0-3.0',
                    'mechanical_valve': '2.5-3.5'
                }
            }
        }
    
    def _load_decision_rules(self) -> Dict[str, Dict]:
        """Load clinical decision rules."""
        
        return {
            'chads2_vasc': {
                'purpose': 'stroke_risk_in_afib',
                'factors': {
                    'chf': 1,
                    'hypertension': 1,
                    'age_75_plus': 2,
                    'diabetes': 1,
                    'stroke_history': 2,
                    'vascular_disease': 1,
                    'age_65_74': 1,
                    'female': 1
                },
                'interpretation': {
                    0: 'low_risk',
                    1: 'moderate_risk',
                    '>=2': 'high_risk'
                }
            },
            'wells_score_pe': {
                'purpose': 'pulmonary_embolism_probability',
                'factors': {
                    'clinical_signs_dvt': 3,
                    'pe_likely_diagnosis': 3,
                    'heart_rate_gt_100': 1.5,
                    'immobilization_surgery': 1.5,
                    'previous_pe_dvt': 1.5,
                    'hemoptysis': 1,
                    'malignancy': 1
                },
                'interpretation': {
                    '<2': 'low_probability',
                    '2-6': 'moderate_probability',
                    '>6': 'high_probability'
                }
            }
        }
    
    def _load_reference_ranges(self) -> Dict[str, Dict]:
        """Load clinical reference ranges."""
        
        return {
            'vital_signs': {
                'heart_rate': {'min': 60, 'max': 100, 'units': 'bpm'},
                'systolic_bp': {'min': 90, 'max': 140, 'units': 'mmHg'},
                'diastolic_bp': {'min': 60, 'max': 90, 'units': 'mmHg'},
                'respiratory_rate': {'min': 12, 'max': 20, 'units': '/min'},
                'temperature': {'min': 97.0, 'max': 99.5, 'units': 'F'},
                'oxygen_saturation': {'min': 95, 'max': 100, 'units': '%'}
            },
            'laboratory': {
                'hemoglobin': {'min': 12.0, 'max': 16.0, 'units': 'g/dL'},
                'white_blood_cells': {'min': 4.0, 'max': 11.0, 'units': 'K/uL'},
                'platelets': {'min': 150, 'max': 450, 'units': 'K/uL'},
                'sodium': {'min': 136, 'max': 145, 'units': 'mEq/L'},
                'potassium': {'min': 3.5, 'max': 5.0, 'units': 'mEq/L'},
                'creatinine': {'min': 0.6, 'max': 1.2, 'units': 'mg/dL'},
                'glucose': {'min': 70, 'max': 100, 'units': 'mg/dL'},
                'lactate': {'min': 0.5, 'max': 2.0, 'units': 'mmol/L'}
            }
        }
    
    def _load_care_pathways(self) -> Dict[str, List[Dict]]:
        """Load clinical care pathways."""
        
        return {
            'chest_pain_evaluation': [
                {'step': 1, 'action': 'obtain_ecg', 'timeframe': '10_minutes'},
                {'step': 2, 'action': 'cardiac_enzymes', 'timeframe': '30_minutes'},
                {'step': 3, 'action': 'chest_xray', 'timeframe': '1_hour'},
                {'step': 4, 'action': 'risk_stratification', 'timeframe': '2_hours'},
                {'step': 5, 'action': 'disposition_decision', 'timeframe': '4_hours'}
            ],
            'stroke_evaluation': [
                {'step': 1, 'action': 'nihss_assessment', 'timeframe': '10_minutes'},
                {'step': 2, 'action': 'ct_head', 'timeframe': '25_minutes'},
                {'step': 3, 'action': 'lab_studies', 'timeframe': '45_minutes'},
                {'step': 4, 'action': 'thrombolytic_decision', 'timeframe': '60_minutes'},
                {'step': 5, 'action': 'neurology_consult', 'timeframe': '2_hours'}
            ]
        }
    
    def get_clinical_guideline(self, condition: str) -> Optional[Dict]:
        """Retrieve clinical guideline for a condition."""
        return self.guidelines.get(condition.lower())
    
    def check_drug_interactions(self, medications: List[str]) -> List[Dict]:
        """Check for drug interactions among medications."""
        interactions = []
        
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                drug1_info = self.drug_database.get(med1.lower(), {})
                drug2_info = self.drug_database.get(med2.lower(), {})
                
                if med2.lower() in drug1_info.get('interactions', []):
                    interactions.append({
                        'drug1': med1,
                        'drug2': med2,
                        'severity': 'moderate',  # Would be determined by clinical rules
                        'description': f'Interaction between {med1} and {med2}'
                    })
        
        return interactions
    
    def evaluate_clinical_rule(self, rule_name: str, patient_data: Dict) -> Dict:
        """Evaluate a clinical decision rule."""
        rule = self.decision_rules.get(rule_name)
        if not rule:
            return {'error': f'Rule {rule_name} not found'}
        
        score = 0
        applied_factors = []
        
        for factor, points in rule['factors'].items():
            if patient_data.get(factor, False):
                score += points
                applied_factors.append(factor)
        
        # Determine interpretation
        interpretation = 'unknown'
        for threshold, meaning in rule['interpretation'].items():
            if isinstance(threshold, str) and threshold.startswith('>='):
                if score >= int(threshold[2:]):
                    interpretation = meaning
            elif isinstance(threshold, str) and threshold.startswith('<'):
                if score < float(threshold[1:]):
                    interpretation = meaning
            elif isinstance(threshold, int) and score == threshold:
                interpretation = meaning
        
        return {
            'rule': rule_name,
            'score': score,
            'interpretation': interpretation,
            'applied_factors': applied_factors,
            'purpose': rule['purpose']
        }

class AgentSafetyMonitor:
    """Safety monitoring system for healthcare agents."""
    
    def __init__(self):
        """Initialize safety monitor."""
        
        self.safety_rules = self._load_safety_rules()
        self.violation_log = []
        self.safety_thresholds = {
            'medication_dose_change': 0.5,  # Max 50% dose change
            'critical_value_alert': 0.95,   # 95% confidence for critical alerts
            'intervention_delay': 3600,     # Max 1 hour delay for urgent interventions
        }
        
        logger.info("Initialized agent safety monitor")
    
    def _load_safety_rules(self) -> List[Dict]:
        """Load safety rules for agent actions."""
        
        return [
            {
                'rule_id': 'medication_safety',
                'description': 'Prevent unsafe medication orders',
                'conditions': [
                    'check_allergies',
                    'check_interactions',
                    'verify_dosing',
                    'check_contraindications'
                ]
            },
            {
                'rule_id': 'critical_value_handling',
                'description': 'Ensure critical values are handled appropriately',
                'conditions': [
                    'immediate_notification',
                    'verify_result',
                    'document_action'
                ]
            },
            {
                'rule_id': 'patient_identification',
                'description': 'Ensure correct patient identification',
                'conditions': [
                    'verify_patient_id',
                    'check_demographics',
                    'confirm_location'
                ]
            }
        ]
    
    def check_action_safety(self, action: Dict, context: Dict) -> Tuple[bool, List[str]]:
        """Check if an action is safe to execute."""
        
        safety_issues = []
        
        # Check medication safety
        if action.get('action_type') == 'medication_order':
            medication_issues = self._check_medication_safety(action, context)
            safety_issues.extend(medication_issues)
        
        # Check critical value handling
        if action.get('action_type') == 'alert' and action.get('priority') == Priority.CRITICAL:
            critical_issues = self._check_critical_value_safety(action, context)
            safety_issues.extend(critical_issues)
        
        # Check patient identification
        patient_issues = self._check_patient_identification(action, context)
        safety_issues.extend(patient_issues)
        
        is_safe = len(safety_issues) == 0
        
        if not is_safe:
            self.violation_log.append({
                'timestamp': datetime.now(),
                'action': action,
                'context': context,
                'issues': safety_issues
            })
        
        return is_safe, safety_issues
    
    def _check_medication_safety(self, action: Dict, context: Dict) -> List[str]:
        """Check medication-related safety issues."""
        issues = []
        
        medication = action.get('medication')
        dose = action.get('dose')
        patient_allergies = context.get('allergies', [])
        current_medications = context.get('medications', [])
        
        # Check allergies
        if medication and any(allergy.lower() in medication.lower() for allergy in patient_allergies):
            issues.append(f"Patient allergic to {medication}")
        
        # Check dose safety (simplified)
        if dose and isinstance(dose, (int, float)):
            if dose <= 0:
                issues.append("Invalid dose: must be positive")
            elif dose > 1000:  # Simplified check
                issues.append("Dose appears excessive")
        
        return issues
    
    def _check_critical_value_safety(self, action: Dict, context: Dict) -> List[str]:
        """Check critical value handling safety."""
        issues = []
        
        # Ensure critical alerts have proper notification
        if not action.get('notification_method'):
            issues.append("Critical alert missing notification method")
        
        # Ensure proper documentation
        if not action.get('documentation'):
            issues.append("Critical alert missing documentation")
        
        return issues
    
    def _check_patient_identification(self, action: Dict, context: Dict) -> List[str]:
        """Check patient identification safety."""
        issues = []
        
        patient_id = action.get('patient_id') or context.get('patient_id')
        
        if not patient_id:
            issues.append("Missing patient identification")
        
        # Additional patient safety checks would go here
        
        return issues

class HealthcareAIAgent(ABC):
    """
    Abstract base class for healthcare AI agents.
    
    This class provides the foundational architecture for healthcare AI agents
    following the BDI (Belief-Desire-Intention) model with healthcare-specific
    adaptations for safety, compliance, and clinical effectiveness.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        max_beliefs: int = 1000,
        max_desires: int = 100,
        max_intentions: int = 50
    ):
        """
        Initialize healthcare AI agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of healthcare agent
            capabilities: List of agent capabilities
            max_beliefs: Maximum number of beliefs to maintain
            max_desires: Maximum number of desires to maintain
            max_intentions: Maximum number of intentions to maintain
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.state = AgentState.IDLE
        
        # BDI components
        self.beliefs: Dict[str, ClinicalBelief] = {}
        self.desires: Dict[str, ClinicalDesire] = {}
        self.intentions: Dict[str, ClinicalIntention] = {}
        
        # Capacity limits
        self.max_beliefs = max_beliefs
        self.max_desires = max_desires
        self.max_intentions = max_intentions
        
        # Communication
        self.message_queue = queue.PriorityQueue()
        self.communication_partners: Dict[str, 'HealthcareAIAgent'] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_response_time': 0.0,
            'safety_violations': 0,
            'last_activity': datetime.now(),
            'patient_outcomes': [],
            'clinical_accuracy': 0.0
        }
        
        # Safety and compliance
        self.safety_constraints = []
        self.compliance_rules = []
        self.audit_log = []
        
        # Clinical knowledge integration
        self.clinical_knowledge_base = ClinicalKnowledgeBase()
        self.safety_monitor = AgentSafetyMonitor()
        
        # Threading for concurrent operations
        self.running = False
        self.agent_thread = None
        self.lock = threading.Lock()
        
        logger.info(f"Initialized healthcare AI agent: {agent_id} ({agent_type})")
    
    def start(self):
        """Start the agent's main execution loop."""
        if self.running:
            logger.warning(f"Agent {self.agent_id} is already running")
            return
        
        self.running = True
        self.agent_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.agent_thread.start()
        logger.info(f"Started agent {self.agent_id}")
    
    def stop(self):
        """Stop the agent's execution."""
        self.running = False
        self.state = AgentState.SHUTDOWN
        if self.agent_thread:
            self.agent_thread.join(timeout=5.0)
        logger.info(f"Stopped agent {self.agent_id}")
    
    def _main_loop(self):
        """Main execution loop for the agent."""
        while self.running:
            try:
                # Process messages
                self._process_messages()
                
                # Update beliefs
                self._update_beliefs()
                
                # Generate desires
                self._generate_desires()
                
                # Plan intentions
                self._plan_intentions()
                
                # Execute actions
                self._execute_actions()
                
                # Cleanup expired beliefs and completed intentions
                self._cleanup()
                
                # Brief sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id} main loop: {e}")
                self.state = AgentState.ERROR
                time.sleep(1.0)
    
    def _process_messages(self):
        """Process incoming messages from other agents."""
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
    
    def _handle_message(self, message: AgentMessage):
        """Handle a specific message."""
        logger.info(f"Agent {self.agent_id} received message: {message.message_type}")
        
        # Update beliefs based on message content
        if message.message_type == "patient_update":
            self._update_patient_beliefs(message.content)
        elif message.message_type == "clinical_alert":
            self._handle_clinical_alert(message.content)
        elif message.message_type == "coordination_request":
            self._handle_coordination_request(message.content)
        
        # Log message for audit
        self.audit_log.append({
            'timestamp': datetime.now(),
            'event_type': 'message_received',
            'message_id': message.message_id,
            'sender': message.sender_id,
            'message_type': message.message_type
        })
    
    def _update_patient_beliefs(self, patient_data: Dict):
        """Update beliefs about patient state."""
        patient_id = patient_data.get('patient_id')
        if not patient_id:
            return
        
        belief_id = f"patient_state_{patient_id}"
        
        belief = ClinicalBelief(
            belief_id=belief_id,
            content=patient_data,
            confidence=0.9,
            timestamp=datetime.now(),
            source="patient_monitor",
            clinical_relevance=1.0
        )
        
        with self.lock:
            self.beliefs[belief_id] = belief
    
    def _handle_clinical_alert(self, alert_data: Dict):
        """Handle clinical alert messages."""
        alert_type = alert_data.get('alert_type')
        patient_id = alert_data.get('patient_id')
        severity = alert_data.get('severity', 'medium')
        
        # Create high-priority desire to address the alert
        desire_id = f"address_alert_{uuid.uuid4().hex[:8]}"
        
        priority_map = {
            'critical': Priority.CRITICAL,
            'high': Priority.URGENT,
            'medium': Priority.HIGH,
            'low': Priority.MEDIUM
        }
        
        desire = ClinicalDesire(
            goal_id=desire_id,
            description=f"Address {alert_type} alert for patient {patient_id}",
            target_outcome={'alert_resolved': True},
            priority=priority_map.get(severity, Priority.MEDIUM),
            patient_id=patient_id,
            clinical_rationale=f"Clinical alert requires immediate attention: {alert_type}"
        )
        
        with self.lock:
            self.desires[desire_id] = desire
    
    def _handle_coordination_request(self, request_data: Dict):
        """Handle coordination requests from other agents."""
        request_type = request_data.get('request_type')
        
        if request_type == "resource_allocation":
            self._handle_resource_request(request_data)
        elif request_type == "care_coordination":
            self._handle_care_coordination(request_data)
    
    @abstractmethod
    def _update_beliefs(self):
        """Update agent beliefs based on current observations."""
        pass
    
    @abstractmethod
    def _generate_desires(self):
        """Generate new desires based on current beliefs."""
        pass
    
    @abstractmethod
    def _plan_intentions(self):
        """Plan intentions to achieve current desires."""
        pass
    
    def _execute_actions(self):
        """Execute planned actions."""
        self.state = AgentState.EXECUTING
        
        with self.lock:
            intentions_to_execute = [
                intention for intention in self.intentions.values()
                if intention.status == "planned" and not intention.requires_approval()
            ]
        
        for intention in intentions_to_execute:
            try:
                self._execute_intention(intention)
            except Exception as e:
                logger.error(f"Error executing intention {intention.plan_id}: {e}")
                intention.status = "failed"
    
    def _execute_intention(self, intention: ClinicalIntention):
        """Execute a specific intention."""
        action = intention.get_next_action()
        if not action:
            intention.status = "completed"
            return
        
        # Safety check
        context = self._get_execution_context(intention)
        is_safe, safety_issues = self.safety_monitor.check_action_safety(action, context)
        
        if not is_safe:
            logger.warning(f"Safety issues detected for action: {safety_issues}")
            intention.status = "safety_hold"
            return
        
        # Execute the action
        success = self._perform_action(action, context)
        
        if success:
            intention.advance_step()
            if intention.is_complete():
                intention.status = "completed"
                self.performance_metrics['tasks_completed'] += 1
        else:
            intention.status = "failed"
            self.performance_metrics['tasks_failed'] += 1
        
        # Log action for audit
        self.audit_log.append({
            'timestamp': datetime.now(),
            'event_type': 'action_executed',
            'intention_id': intention.plan_id,
            'action': action,
            'success': success,
            'safety_issues': safety_issues if not is_safe else []
        })
    
    def _get_execution_context(self, intention: ClinicalIntention) -> Dict:
        """Get context for action execution."""
        # Find relevant beliefs for this intention
        relevant_beliefs = {}
        
        if intention.goal_id in self.desires:
            desire = self.desires[intention.goal_id]
            patient_id = desire.patient_id
            
            if patient_id:
                # Get patient-related beliefs
                for belief_id, belief in self.beliefs.items():
                    if patient_id in belief.content.get('patient_id', ''):
                        relevant_beliefs[belief_id] = belief.content
        
        return {
            'beliefs': relevant_beliefs,
            'agent_capabilities': self.capabilities,
            'timestamp': datetime.now()
        }
    
    @abstractmethod
    def _perform_action(self, action: Dict, context: Dict) -> bool:
        """Perform a specific action."""
        pass
    
    def _cleanup(self):
        """Clean up expired beliefs and completed intentions."""
        current_time = datetime.now()
        
        with self.lock:
            # Remove expired beliefs
            expired_beliefs = [
                belief_id for belief_id, belief in self.beliefs.items()
                if belief.is_expired()
            ]
            
            for belief_id in expired_beliefs:
                del self.beliefs[belief_id]
            
            # Remove completed intentions
            completed_intentions = [
                intention_id for intention_id, intention in self.intentions.items()
                if intention.status in ["completed", "failed"]
            ]
            
            for intention_id in completed_intentions:
                del self.intentions[intention_id]
            
            # Limit collection sizes
            if len(self.beliefs) > self.max_beliefs:
                # Remove oldest beliefs
                sorted_beliefs = sorted(
                    self.beliefs.items(),
                    key=lambda x: x[1].timestamp
                )
                
                beliefs_to_remove = len(self.beliefs) - self.max_beliefs
                for i in range(beliefs_to_remove):
                    del self.beliefs[sorted_beliefs[i][0]]
    
    def send_message(self, receiver_id: str, message_type: str, content: Dict, priority: Priority = Priority.MEDIUM):
        """Send a message to another agent."""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        if receiver_id in self.communication_partners:
            self.communication_partners[receiver_id].receive_message(message)
        else:
            logger.warning(f"No communication partner found for {receiver_id}")
    
    def receive_message(self, message: AgentMessage):
        """Receive a message from another agent."""
        self.message_queue.put(message)
    
    def add_communication_partner(self, agent: 'HealthcareAIAgent'):
        """Add another agent as a communication partner."""
        self.communication_partners[agent.agent_id] = agent
        agent.communication_partners[self.agent_id] = self
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def get_audit_log(self) -> List[Dict]:
        """Get audit log for compliance reporting."""
        return self.audit_log.copy()

class ClinicalDecisionSupportAgent(HealthcareAIAgent):
    """
    Specialized agent for clinical decision support.
    
    This agent monitors patient data, identifies clinical patterns,
    and provides evidence-based recommendations to healthcare providers.
    """
    
    def __init__(self, agent_id: str):
        """Initialize clinical decision support agent."""
        
        capabilities = [
            "patient_monitoring",
            "clinical_assessment",
            "guideline_adherence",
            "drug_interaction_checking",
            "alert_generation"
        ]
        
        super().__init__(agent_id, "clinical_decision_support", capabilities)
        
        # Clinical decision models
        self.risk_models = self._initialize_risk_models()
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        logger.info(f"Initialized clinical decision support agent: {agent_id}")
    
    def _initialize_risk_models(self) -> Dict[str, Any]:
        """Initialize clinical risk prediction models."""
        
        # In production, these would be trained models
        # For demonstration, we use simple rule-based models
        
        return {
            'sepsis_risk': {
                'type': 'rule_based',
                'rules': [
                    {'condition': 'temperature > 38.3 or temperature < 36', 'score': 1},
                    {'condition': 'heart_rate > 90', 'score': 1},
                    {'condition': 'respiratory_rate > 20', 'score': 1},
                    {'condition': 'wbc > 12000 or wbc < 4000', 'score': 1}
                ],
                'threshold': 2
            },
            'deterioration_risk': {
                'type': 'early_warning_score',
                'parameters': {
                    'respiratory_rate': {'ranges': [(8, 11, 3), (12, 20, 0), (21, 24, 2), (25, 100, 3)]},
                    'oxygen_saturation': {'ranges': [(91, 93, 3), (94, 95, 2), (96, 100, 0)]},
                    'temperature': {'ranges': [(35.0, 36.0, 3), (36.1, 38.0, 0), (38.1, 39.0, 1), (39.1, 50.0, 2)]},
                    'systolic_bp': {'ranges': [(70, 90, 3), (91, 100, 2), (101, 110, 1), (111, 219, 0), (220, 300, 3)]},
                    'heart_rate': {'ranges': [(40, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, 200, 3)]},
                    'consciousness': {'alert': 0, 'voice': 3, 'pain': 3, 'unresponsive': 3}
                },
                'threshold': 7
            }
        }
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict]:
        """Initialize alert thresholds for clinical parameters."""
        
        return {
            'critical_values': {
                'hemoglobin': {'low': 7.0, 'high': 20.0},
                'platelets': {'low': 50, 'high': 1000},
                'potassium': {'low': 2.5, 'high': 6.0},
                'creatinine': {'low': 0.3, 'high': 5.0},
                'glucose': {'low': 40, 'high': 400},
                'lactate': {'low': 0.1, 'high': 4.0}
            },
            'vital_signs': {
                'heart_rate': {'low': 40, 'high': 150},
                'systolic_bp': {'low': 70, 'high': 200},
                'respiratory_rate': {'low': 8, 'high': 30},
                'temperature': {'low': 95.0, 'high': 104.0},
                'oxygen_saturation': {'low': 88, 'high': 100}
            }
        }
    
    def _update_beliefs(self):
        """Update beliefs based on patient monitoring data."""
        self.state = AgentState.MONITORING
        
        # In production, this would interface with real patient monitoring systems
        # For demonstration, we simulate belief updates
        
        # Check for expired beliefs and update confidence
        current_time = datetime.now()
        
        with self.lock:
            for belief in self.beliefs.values():
                if belief.content.get('type') == 'vital_signs':
                    # Decrease confidence over time for vital signs
                    age_hours = (current_time - belief.timestamp).total_seconds() / 3600
                    confidence_decay = max(0.1, 1.0 - (age_hours * 0.1))
                    belief.confidence *= confidence_decay
    
    def _generate_desires(self):
        """Generate desires based on current clinical situation."""
        self.state = AgentState.ANALYZING
        
        # Analyze current beliefs for clinical concerns
        clinical_concerns = self._identify_clinical_concerns()
        
        for concern in clinical_concerns:
            desire_id = f"address_concern_{uuid.uuid4().hex[:8]}"
            
            desire = ClinicalDesire(
                goal_id=desire_id,
                description=f"Address clinical concern: {concern['type']}",
                target_outcome={'concern_resolved': True},
                priority=concern['priority'],
                patient_id=concern.get('patient_id'),
                clinical_rationale=concern['rationale']
            )
            
            with self.lock:
                if desire_id not in self.desires:
                    self.desires[desire_id] = desire
    
    def _identify_clinical_concerns(self) -> List[Dict]:
        """Identify clinical concerns from current beliefs."""
        concerns = []
        
        # Group beliefs by patient
        patient_beliefs = defaultdict(list)
        for belief in self.beliefs.values():
            patient_id = belief.content.get('patient_id')
            if patient_id:
                patient_beliefs[patient_id].append(belief)
        
        # Analyze each patient's data
        for patient_id, beliefs in patient_beliefs.items():
            patient_data = {}
            
            # Consolidate patient data
            for belief in beliefs:
                patient_data.update(belief.content)
            
            # Check for critical values
            critical_concerns = self._check_critical_values(patient_id, patient_data)
            concerns.extend(critical_concerns)
            
            # Check risk scores
            risk_concerns = self._check_risk_scores(patient_id, patient_data)
            concerns.extend(risk_concerns)
            
            # Check drug interactions
            drug_concerns = self._check_drug_interactions(patient_id, patient_data)
            concerns.extend(drug_concerns)
        
        return concerns
    
    def _check_critical_values(self, patient_id: str, patient_data: Dict) -> List[Dict]:
        """Check for critical laboratory and vital sign values."""
        concerns = []
        
        # Check laboratory values
        for param, thresholds in self.alert_thresholds['critical_values'].items():
            value = patient_data.get(param)
            if value is not None:
                if value < thresholds['low'] or value > thresholds['high']:
                    concerns.append({
                        'type': 'critical_value',
                        'parameter': param,
                        'value': value,
                        'patient_id': patient_id,
                        'priority': Priority.CRITICAL,
                        'rationale': f'Critical {param} value: {value}'
                    })
        
        # Check vital signs
        for param, thresholds in self.alert_thresholds['vital_signs'].items():
            value = patient_data.get(param)
            if value is not None:
                if value < thresholds['low'] or value > thresholds['high']:
                    priority = Priority.URGENT if param in ['heart_rate', 'systolic_bp'] else Priority.HIGH
                    concerns.append({
                        'type': 'abnormal_vital',
                        'parameter': param,
                        'value': value,
                        'patient_id': patient_id,
                        'priority': priority,
                        'rationale': f'Abnormal {param}: {value}'
                    })
        
        return concerns
    
    def _check_risk_scores(self, patient_id: str, patient_data: Dict) -> List[Dict]:
        """Check clinical risk scores."""
        concerns = []
        
        # Check sepsis risk
        sepsis_score = self._calculate_sepsis_risk(patient_data)
        if sepsis_score >= self.risk_models['sepsis_risk']['threshold']:
            concerns.append({
                'type': 'sepsis_risk',
                'score': sepsis_score,
                'patient_id': patient_id,
                'priority': Priority.URGENT,
                'rationale': f'Elevated sepsis risk score: {sepsis_score}'
            })
        
        # Check deterioration risk
        ews_score = self._calculate_early_warning_score(patient_data)
        if ews_score >= self.risk_models['deterioration_risk']['threshold']:
            concerns.append({
                'type': 'deterioration_risk',
                'score': ews_score,
                'patient_id': patient_id,
                'priority': Priority.HIGH,
                'rationale': f'Elevated early warning score: {ews_score}'
            })
        
        return concerns
    
    def _calculate_sepsis_risk(self, patient_data: Dict) -> int:
        """Calculate sepsis risk score."""
        score = 0
        
        # Temperature
        temp = patient_data.get('temperature', 98.6)
        if temp > 100.4 or temp < 96.8:
            score += 1
        
        # Heart rate
        hr = patient_data.get('heart_rate', 70)
        if hr > 90:
            score += 1
        
        # Respiratory rate
        rr = patient_data.get('respiratory_rate', 16)
        if rr > 20:
            score += 1
        
        # White blood cell count
        wbc = patient_data.get('white_blood_cells', 7.0)
        if wbc > 12.0 or wbc < 4.0:
            score += 1
        
        return score
    
    def _calculate_early_warning_score(self, patient_data: Dict) -> int:
        """Calculate early warning score (NEWS)."""
        score = 0
        
        ews_params = self.risk_models['deterioration_risk']['parameters']
        
        # Respiratory rate
        rr = patient_data.get('respiratory_rate', 16)
        for low, high, points in ews_params['respiratory_rate']['ranges']:
            if low <= rr <= high:
                score += points
                break
        
        # Oxygen saturation
        o2_sat = patient_data.get('oxygen_saturation', 98)
        for low, high, points in ews_params['oxygen_saturation']['ranges']:
            if low <= o2_sat <= high:
                score += points
                break
        
        # Temperature
        temp = patient_data.get('temperature', 98.6)
        for low, high, points in ews_params['temperature']['ranges']:
            if low <= temp <= high:
                score += points
                break
        
        # Systolic blood pressure
        sbp = patient_data.get('systolic_bp', 120)
        for low, high, points in ews_params['systolic_bp']['ranges']:
            if low <= sbp <= high:
                score += points
                break
        
        # Heart rate
        hr = patient_data.get('heart_rate', 70)
        for low, high, points in ews_params['heart_rate']['ranges']:
            if low <= hr <= high:
                score += points
                break
        
        # Consciousness level
        consciousness = patient_data.get('consciousness', 'alert')
        score += ews_params['consciousness'].get(consciousness, 0)
        
        return score
    
    def _check_drug_interactions(self, patient_id: str, patient_data: Dict) -> List[Dict]:
        """Check for drug interactions."""
        concerns = []
        
        medications = patient_data.get('medications', [])
        if len(medications) > 1:
            interactions = self.clinical_knowledge_base.check_drug_interactions(medications)
            
            for interaction in interactions:
                concerns.append({
                    'type': 'drug_interaction',
                    'interaction': interaction,
                    'patient_id': patient_id,
                    'priority': Priority.HIGH,
                    'rationale': f"Drug interaction: {interaction['description']}"
                })
        
        return concerns
    
    def _plan_intentions(self):
        """Plan intentions to address clinical concerns."""
        self.state = AgentState.PLANNING
        
        with self.lock:
            unaddressed_desires = [
                desire for desire in self.desires.values()
                if desire.goal_id not in self.intentions
            ]
        
        for desire in unaddressed_desires:
            intention = self._create_intention_for_desire(desire)
            if intention:
                with self.lock:
                    self.intentions[intention.plan_id] = intention
    
    def _create_intention_for_desire(self, desire: ClinicalDesire) -> Optional[ClinicalIntention]:
        """Create an intention to fulfill a desire."""
        
        actions = []
        
        if 'critical_value' in desire.description:
            actions = [
                {
                    'action_type': 'alert',
                    'priority': Priority.CRITICAL,
                    'message': f"Critical value alert: {desire.description}",
                    'patient_id': desire.patient_id,
                    'notification_method': 'immediate',
                    'documentation': True
                },
                {
                    'action_type': 'recommend',
                    'recommendation': 'Verify result and assess patient',
                    'patient_id': desire.patient_id
                }
            ]
        
        elif 'sepsis_risk' in desire.description:
            actions = [
                {
                    'action_type': 'alert',
                    'priority': Priority.URGENT,
                    'message': f"Elevated sepsis risk: {desire.description}",
                    'patient_id': desire.patient_id
                },
                {
                    'action_type': 'recommend',
                    'recommendation': 'Consider sepsis workup and early intervention',
                    'patient_id': desire.patient_id
                }
            ]
        
        elif 'drug_interaction' in desire.description:
            actions = [
                {
                    'action_type': 'alert',
                    'priority': Priority.HIGH,
                    'message': f"Drug interaction detected: {desire.description}",
                    'patient_id': desire.patient_id
                },
                {
                    'action_type': 'recommend',
                    'recommendation': 'Review medication regimen for interactions',
                    'patient_id': desire.patient_id
                }
            ]
        
        if not actions:
            return None
        
        intention = ClinicalIntention(
            plan_id=f"intention_{uuid.uuid4().hex[:8]}",
            goal_id=desire.goal_id,
            actions=actions,
            approval_required=desire.priority in [Priority.CRITICAL, Priority.URGENT]
        )
        
        return intention
    
    def _perform_action(self, action: Dict, context: Dict) -> bool:
        """Perform a clinical decision support action."""
        
        action_type = action.get('action_type')
        
        try:
            if action_type == 'alert':
                return self._send_clinical_alert(action, context)
            elif action_type == 'recommend':
                return self._send_recommendation(action, context)
            elif action_type == 'document':
                return self._document_finding(action, context)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
        
        except Exception as e:
            logger.error(f"Error performing action {action_type}: {e}")
            return False
    
    def _send_clinical_alert(self, action: Dict, context: Dict) -> bool:
        """Send a clinical alert to appropriate recipients."""
        
        alert_data = {
            'alert_type': 'clinical_decision_support',
            'message': action.get('message'),
            'priority': action.get('priority'),
            'patient_id': action.get('patient_id'),
            'timestamp': datetime.now(),
            'source_agent': self.agent_id
        }
        
        # In production, this would interface with clinical alerting systems
        logger.info(f"Clinical alert sent: {alert_data['message']}")
        
        return True
    
    def _send_recommendation(self, action: Dict, context: Dict) -> bool:
        """Send a clinical recommendation."""
        
        recommendation_data = {
            'recommendation': action.get('recommendation'),
            'patient_id': action.get('patient_id'),
            'evidence': action.get('evidence', []),
            'timestamp': datetime.now(),
            'source_agent': self.agent_id
        }
        
        # In production, this would interface with clinical decision support systems
        logger.info(f"Clinical recommendation sent: {recommendation_data['recommendation']}")
        
        return True
    
    def _document_finding(self, action: Dict, context: Dict) -> bool:
        """Document a clinical finding."""
        
        documentation = {
            'finding': action.get('finding'),
            'patient_id': action.get('patient_id'),
            'timestamp': datetime.now(),
            'source_agent': self.agent_id
        }
        
        # In production, this would interface with electronic health records
        logger.info(f"Clinical finding documented: {documentation['finding']}")
        
        return True

## Bibliography and References

### Foundational AI Agents and Multi-Agent Systems

1. **Wooldridge, M.** (2009). An introduction to multiagent systems. *John Wiley & Sons*. ISBN: 978-0470519462. [Comprehensive foundation for multi-agent systems theory and implementation]

2. **Russell, S., & Norvig, P.** (2020). Artificial intelligence: a modern approach. *Pearson*. ISBN: 978-0134610993. [Standard AI textbook with extensive coverage of intelligent agents]

3. **Rao, A. S., & Georgeff, M. P.** (1995). BDI agents: from theory to practice. *Proceedings of the first international conference on multi-agent systems*, 312-319. [Foundational BDI architecture paper]

4. **Bratman, M.** (1987). Intention, plans, and practical reason. *Harvard University Press*. ISBN: 978-0674458185. [Philosophical foundations of intention and planning in rational agents]

### Healthcare AI Agents and Clinical Applications

5. **Isern, D., & Moreno, A.** (2016). A systematic literature review of agents applied in healthcare. *Journal of Medical Systems*, 40(2), 1-14. DOI: 10.1007/s10916-015-0376-2. [Comprehensive review of healthcare agent applications]

6. **Anand, V., Carroll, A. E., & Downs, S. M.** (2012). Automated primary care screening in pediatric waiting rooms. *Pediatrics*, 129(5), e1275-e1281. [Clinical implementation of automated screening agents]

7. **Peleg, M., Boxwala, A. A., Bernstam, E., Tu, S., Greenes, R. A., & Shortliffe, E. H.** (2001). Sharable representation of clinical guidelines in GLIF: relationship to the Arden Syntax. *Journal of Biomedical Informatics*, 34(3), 170-181. [Clinical guideline representation for agent systems]

### Clinical Decision Support and Autonomous Systems

8. **Berner, E. S.** (2007). Clinical decision support systems: theory and practice. *Springer*. ISBN: 978-0387381558. [Comprehensive coverage of clinical decision support systems]

9. **Shortliffe, E. H., & Cimino, J. J.** (2013). Biomedical informatics: computer applications in health care and biomedicine. *Springer*. ISBN: 978-1447144731. [Standard biomedical informatics textbook]

10. **Sim, I., Gorman, P., Greenes, R. A., Haynes, R. B., Kaplan, B., Lehmann, H., & Tang, P. C.** (2001). Clinical decision support systems for the practice of evidence-based medicine. *Journal of the American Medical Informatics Association*, 8(6), 527-534. [Evidence-based clinical decision support]

### Safety and Regulatory Considerations

11. **Sittig, D. F., & Singh, H.** (2010). A new sociotechnical model for studying health information technology in complex adaptive healthcare systems. *Quality and Safety in Health Care*, 19(Suppl 3), i68-i74. [Sociotechnical framework for healthcare IT safety]

12. **Coiera, E.** (2003). Guide to health informatics. *CRC Press*. ISBN: 978-0340763384. [Comprehensive guide to health informatics including safety considerations]

13. **U.S. Food and Drug Administration.** (2021). Artificial intelligence/machine learning (AI/ML)-based software as a medical device (SaMD) action plan. *FDA Guidance Document*. [Regulatory framework for AI/ML medical devices]

### Multi-Agent Coordination and Communication

14. **Stone, P., & Veloso, M.** (2000). Multiagent systems: A survey from a machine learning perspective. *Autonomous Robots*, 8(3), 345-383. [Machine learning perspectives on multi-agent systems]

15. **Tambe, M.** (1997). Towards flexible teamwork. *Journal of Artificial Intelligence Research*, 7, 83-124. [Flexible teamwork in multi-agent systems]

16. **Jennings, N. R.** (2000). On agent-based software engineering. *Artificial Intelligence*, 117(2), 277-296. [Software engineering approaches for agent systems]

### Healthcare Workflow and Care Coordination

17. **Unertl, K. M., Weinger, M. B., Johnson, K. B., & Lorenzi, N. M.** (2009). Describing and modeling workflow and information flow in chronic disease care. *Journal of the American Medical Informatics Association*, 16(6), 826-836. [Healthcare workflow modeling]

18. **Holden, R. J., & Karsh, B. T.** (2010). The technology acceptance model: its past and its future in health care. *Journal of Biomedical Informatics*, 43(1), 159-172. [Technology acceptance in healthcare]

19. **Berg, M.** (1999). Patient care information systems and health care work: a sociotechnical approach. *Acta Informatica Medica*, 7(2), 79-85. [Sociotechnical approach to healthcare information systems]

### Performance Evaluation and Metrics

20. **Kawamoto, K., Houlihan, C. A., Balas, E. A., & Lobach, D. F.** (2005). Improving clinical practice using clinical decision support systems: a systematic review of trials to identify features critical to success. *BMJ*, 330(7494), 765. [Critical success factors for clinical decision support]

This chapter provides a comprehensive foundation for implementing AI agent systems in healthcare environments. The implementations presented address the unique challenges of clinical settings including safety frameworks, regulatory compliance, and human-AI collaboration. The next chapter will explore bias detection and mitigation in healthcare AI, building upon these agent concepts to address fairness and equity in clinical AI systems.
