# Chapter 7: AI Agents in Healthcare - From Autonomous Systems to Clinical Teammates

## Learning Objectives

By the end of this chapter, readers will be able to:

1. **Understand the theoretical foundations** of AI agents and their applications in healthcare environments
2. **Design and implement** autonomous AI agents for clinical decision support and care coordination
3. **Deploy multi-agent systems** for complex healthcare workflows and resource optimization
4. **Evaluate agent performance** using clinical metrics and safety frameworks
5. **Address ethical considerations** in autonomous healthcare AI systems
6. **Implement human-AI collaboration** frameworks for clinical practice

## 7.1 Introduction to AI Agents in Healthcare

Artificial Intelligence agents represent a paradigm shift from traditional AI tools to autonomous systems capable of perceiving their environment, making decisions, and taking actions to achieve specific goals. In healthcare, AI agents are emerging as powerful solutions for complex challenges including care coordination, resource allocation, clinical decision support, and patient monitoring.

The concept of AI agents in healthcare builds upon decades of research in artificial intelligence, multi-agent systems, and clinical informatics. Unlike passive AI models that respond to queries, healthcare AI agents actively monitor patient conditions, anticipate needs, coordinate care activities, and adapt their behavior based on changing clinical circumstances.

### 7.1.1 Theoretical Foundations of Healthcare AI Agents

Healthcare AI agents are grounded in several key theoretical frameworks that distinguish them from traditional AI systems:

**Agent Architecture**: Healthcare AI agents typically follow the Belief-Desire-Intention (BDI) architecture, where:
- **Beliefs** represent the agent's understanding of the current clinical situation
- **Desires** represent the clinical goals and outcomes the agent aims to achieve
- **Intentions** represent the specific actions and plans the agent commits to executing

The formal representation of a healthcare AI agent can be expressed as:

$$Agent = \langle B, D, I, \pi, \rho \rangle$$

where:
- $B$ is the set of beliefs about patient state and clinical environment
- $D$ is the set of desires representing clinical goals
- $I$ is the set of intentions representing committed plans
- $\pi$ is the planning function that generates action sequences
- $\rho$ is the revision function that updates beliefs based on new observations

**Rational Agency**: Healthcare AI agents must exhibit rational behavior, making decisions that maximize expected clinical utility while minimizing risks. The rationality principle in healthcare contexts requires agents to:

1. **Maximize patient benefit** while minimizing harm
2. **Respect clinical guidelines** and evidence-based practices
3. **Adapt to changing conditions** in real-time
4. **Coordinate effectively** with human clinicians and other agents

**Multi-Agent Coordination**: Healthcare delivery inherently involves multiple stakeholders, making multi-agent coordination essential. The coordination mechanisms include:

- **Contract Net Protocol** for task allocation among clinical agents
- **Consensus Algorithms** for collaborative decision-making
- **Auction Mechanisms** for resource allocation
- **Negotiation Protocols** for resolving conflicts between agents

### 7.1.2 Healthcare-Specific Agent Characteristics

Healthcare AI agents must possess unique characteristics that distinguish them from general-purpose AI agents:

**Clinical Safety**: Healthcare agents must incorporate multiple safety mechanisms:
- **Fail-safe behaviors** that default to conservative actions
- **Human oversight integration** for critical decisions
- **Uncertainty quantification** to communicate confidence levels
- **Audit trails** for accountability and learning

**Regulatory Compliance**: Healthcare agents must comply with regulations including:
- **HIPAA privacy requirements** for patient data protection
- **FDA guidelines** for medical device software
- **Clinical practice standards** for evidence-based care
- **Institutional policies** for AI system deployment

**Ethical Considerations**: Healthcare agents must embody ethical principles:
- **Beneficence**: Acting in the patient's best interest
- **Non-maleficence**: Avoiding harm to patients
- **Autonomy**: Respecting patient preferences and choices
- **Justice**: Ensuring fair and equitable treatment

### 7.1.3 Applications Landscape

Healthcare AI agents are being deployed across multiple domains:

**Clinical Decision Support Agents**: Autonomous systems that monitor patient data, identify clinical patterns, and provide real-time recommendations to clinicians.

**Care Coordination Agents**: Systems that manage complex care pathways, coordinate appointments, and ensure continuity of care across multiple providers.

**Resource Management Agents**: Agents that optimize hospital resources including bed allocation, staff scheduling, and equipment utilization.

**Patient Monitoring Agents**: Continuous monitoring systems that track patient vital signs, medication adherence, and symptom progression.

**Drug Discovery Agents**: Autonomous systems that design experiments, analyze results, and propose new therapeutic compounds.

## 7.2 Architecture and Implementation of Healthcare AI Agents

### 7.2.1 Core Agent Architecture

The implementation of healthcare AI agents requires a sophisticated architecture that balances autonomy with safety, efficiency with explainability, and individual patient care with population health considerations.

```python
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from collections import defaultdict, deque
import threading
import queue
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Enumeration of possible agent states."""
    IDLE = "idle"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMMUNICATING = "communicating"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class Priority(Enum):
    """Priority levels for agent tasks and communications."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class Belief:
    """Represents an agent's belief about the clinical environment."""
    content: Dict[str, Any]
    confidence: float
    timestamp: datetime
    source: str
    evidence: List[str] = field(default_factory=list)
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if belief is expired based on age."""
        return (datetime.now() - self.timestamp).total_seconds() > max_age_hours * 3600

@dataclass
class Desire:
    """Represents an agent's desire (goal) in the clinical context."""
    goal_id: str
    description: str
    target_outcome: Dict[str, Any]
    priority: Priority
    deadline: Optional[datetime] = None
    constraints: List[str] = field(default_factory=list)
    
    def is_overdue(self) -> bool:
        """Check if desire is overdue."""
        return self.deadline and datetime.now() > self.deadline

@dataclass
class Intention:
    """Represents an agent's intention (committed plan)."""
    plan_id: str
    goal_id: str
    actions: List[Dict[str, Any]]
    current_step: int = 0
    status: str = "planned"
    estimated_completion: Optional[datetime] = None
    
    def is_complete(self) -> bool:
        """Check if intention is complete."""
        return self.current_step >= len(self.actions)
    
    def get_next_action(self) -> Optional[Dict[str, Any]]:
        """Get the next action to execute."""
        if self.is_complete():
            return None
        return self.actions[self.current_step]

@dataclass
class Message:
    """Represents a message between agents."""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    priority: Priority
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class HealthcareAIAgent(ABC):
    """
    Abstract base class for healthcare AI agents.
    
    This class provides the foundational architecture for healthcare AI agents
    following the BDI (Belief-Desire-Intention) model with healthcare-specific
    adaptations for safety, compliance, and clinical effectiveness.
    
    Based on the agent architecture described in:
    Wooldridge, M. (2009). An introduction to multiagent systems. 
    John Wiley & Sons. ISBN: 978-0470519462
    
    Adapted for healthcare applications following:
    Isern, D., & Moreno, A. (2016). A systematic literature review of agents 
    applied in healthcare. Journal of Medical Systems, 40(2), 1-14.
    DOI: 10.1007/s10916-015-0376-2
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
            agent_type: Type of healthcare agent (e.g., "clinical_decision_support")
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
        self.beliefs: Dict[str, Belief] = {}
        self.desires: Dict[str, Desire] = {}
        self.intentions: Dict[str, Intention] = {}
        
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
            'last_activity': datetime.now()
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
                # Update agent state
                self._update_state()
                
                # Process incoming messages
                self._process_messages()
                
                # Update beliefs based on observations
                self._update_beliefs()
                
                # Generate new desires based on current situation
                self._generate_desires()
                
                # Plan actions to achieve desires
                self._plan_actions()
                
                # Execute current intentions
                self._execute_intentions()
                
                # Perform safety checks
                self._safety_check()
                
                # Update performance metrics
                self._update_metrics()
                
                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id} main loop: {e}")
                self.state = AgentState.ERROR
                self.performance_metrics['tasks_failed'] += 1
    
    def _update_state(self):
        """Update agent state based on current conditions."""
        if not self.running:
            self.state = AgentState.SHUTDOWN
            return
        
        # Determine state based on current activities
        if self.intentions and any(i.status == "executing" for i in self.intentions.values()):
            self.state = AgentState.EXECUTING
        elif self.intentions and any(i.status == "planned" for i in self.intentions.values()):
            self.state = AgentState.PLANNING
        elif self.desires:
            self.state = AgentState.ANALYZING
        elif self._has_monitoring_tasks():
            self.state = AgentState.MONITORING
        else:
            self.state = AgentState.IDLE
    
    def _has_monitoring_tasks(self) -> bool:
        """Check if agent has active monitoring tasks."""
        # Override in subclasses to define monitoring conditions
        return False
    
    def _process_messages(self):
        """Process incoming messages from other agents."""
        while not self.message_queue.empty():
            try:
                priority, message = self.message_queue.get_nowait()
                self._handle_message(message)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing message in agent {self.agent_id}: {e}")
    
    def _handle_message(self, message: Message):
        """Handle a specific message."""
        self.audit_log.append({
            'timestamp': datetime.now(),
            'action': 'message_received',
            'details': {
                'sender': message.sender_id,
                'type': message.message_type,
                'priority': message.priority.name
            }
        })
        
        # Route message based on type
        if message.message_type == "belief_update":
            self._handle_belief_update(message)
        elif message.message_type == "task_request":
            self._handle_task_request(message)
        elif message.message_type == "coordination_request":
            self._handle_coordination_request(message)
        elif message.message_type == "safety_alert":
            self._handle_safety_alert(message)
        else:
            logger.warning(f"Unknown message type: {message.message_type}")
    
    def _handle_belief_update(self, message: Message):
        """Handle belief update message."""
        belief_data = message.content
        belief = Belief(
            content=belief_data['content'],
            confidence=belief_data['confidence'],
            timestamp=datetime.now(),
            source=message.sender_id,
            evidence=belief_data.get('evidence', [])
        )
        
        belief_key = belief_data['key']
        self.beliefs[belief_key] = belief
        
        logger.info(f"Agent {self.agent_id} updated belief: {belief_key}")
    
    def _handle_task_request(self, message: Message):
        """Handle task request message."""
        task_data = message.content
        
        # Create desire for the requested task
        desire = Desire(
            goal_id=str(uuid.uuid4()),
            description=task_data['description'],
            target_outcome=task_data['target_outcome'],
            priority=Priority(task_data.get('priority', 3)),
            deadline=datetime.fromisoformat(task_data['deadline']) if 'deadline' in task_data else None,
            constraints=task_data.get('constraints', [])
        )
        
        self.desires[desire.goal_id] = desire
        
        logger.info(f"Agent {self.agent_id} received task request: {desire.description}")
    
    def _handle_coordination_request(self, message: Message):
        """Handle coordination request from another agent."""
        # Override in subclasses for specific coordination protocols
        pass
    
    def _handle_safety_alert(self, message: Message):
        """Handle safety alert message."""
        alert_data = message.content
        
        # Log safety alert
        self.audit_log.append({
            'timestamp': datetime.now(),
            'action': 'safety_alert_received',
            'details': alert_data
        })
        
        # Take appropriate safety actions
        if alert_data.get('severity') == 'critical':
            self._emergency_stop()
        
        logger.warning(f"Agent {self.agent_id} received safety alert: {alert_data}")
    
    def _emergency_stop(self):
        """Emergency stop procedure for safety."""
        # Halt all current intentions
        for intention in self.intentions.values():
            intention.status = "halted"
        
        # Clear non-critical desires
        critical_desires = {
            k: v for k, v in self.desires.items() 
            if v.priority == Priority.CRITICAL
        }
        self.desires = critical_desires
        
        self.state = AgentState.ERROR
        
        logger.critical(f"Agent {self.agent_id} executed emergency stop")
    
    @abstractmethod
    def _update_beliefs(self):
        """Update beliefs based on observations. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _generate_desires(self):
        """Generate new desires based on current situation. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _plan_actions(self):
        """Plan actions to achieve desires. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _execute_intentions(self):
        """Execute current intentions. Must be implemented by subclasses."""
        pass
    
    def _safety_check(self):
        """Perform safety checks on agent behavior."""
        safety_violations = self.safety_monitor.check_agent_safety(self)
        
        if safety_violations:
            self.performance_metrics['safety_violations'] += len(safety_violations)
            
            for violation in safety_violations:
                self.audit_log.append({
                    'timestamp': datetime.now(),
                    'action': 'safety_violation',
                    'details': violation
                })
                
                if violation['severity'] == 'critical':
                    self._emergency_stop()
                    break
    
    def _update_metrics(self):
        """Update performance metrics."""
        self.performance_metrics['last_activity'] = datetime.now()
        
        # Clean up expired beliefs
        expired_beliefs = [
            k for k, v in self.beliefs.items() 
            if v.is_expired()
        ]
        for k in expired_beliefs:
            del self.beliefs[k]
        
        # Clean up overdue desires
        overdue_desires = [
            k for k, v in self.desires.items() 
            if v.is_overdue()
        ]
        for k in overdue_desires:
            del self.desires[k]
            self.performance_metrics['tasks_failed'] += 1
    
    def send_message(self, receiver_id: str, message_type: str, content: Dict[str, Any], priority: Priority = Priority.MEDIUM):
        """Send message to another agent."""
        if receiver_id not in self.communication_partners:
            logger.error(f"Agent {receiver_id} not found in communication partners")
            return
        
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        receiver_agent = self.communication_partners[receiver_id]
        receiver_agent.receive_message(message)
        
        self.audit_log.append({
            'timestamp': datetime.now(),
            'action': 'message_sent',
            'details': {
                'receiver': receiver_id,
                'type': message_type,
                'priority': priority.name
            }
        })
    
    def receive_message(self, message: Message):
        """Receive message from another agent."""
        self.message_queue.put((message.priority.value, message))
    
    def add_communication_partner(self, agent: 'HealthcareAIAgent'):
        """Add another agent as a communication partner."""
        self.communication_partners[agent.agent_id] = agent
        agent.communication_partners[self.agent_id] = self
        
        logger.info(f"Established communication between {self.agent_id} and {agent.agent_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'beliefs_count': len(self.beliefs),
            'desires_count': len(self.desires),
            'intentions_count': len(self.intentions),
            'performance_metrics': self.performance_metrics.copy(),
            'last_activity': self.performance_metrics['last_activity'].isoformat()
        }

class ClinicalDecisionSupportAgent(HealthcareAIAgent):
    """
    Specialized agent for clinical decision support.
    
    This agent monitors patient data, analyzes clinical patterns,
    and provides evidence-based recommendations to healthcare providers.
    """
    
    def __init__(
        self,
        agent_id: str,
        patient_ids: List[str],
        clinical_guidelines: Dict[str, Any],
        alert_thresholds: Dict[str, float]
    ):
        """
        Initialize clinical decision support agent.
        
        Args:
            agent_id: Unique identifier for the agent
            patient_ids: List of patients to monitor
            clinical_guidelines: Clinical practice guidelines
            alert_thresholds: Thresholds for generating alerts
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="clinical_decision_support",
            capabilities=[
                "patient_monitoring",
                "risk_assessment",
                "guideline_compliance",
                "alert_generation"
            ]
        )
        
        self.patient_ids = patient_ids
        self.clinical_guidelines = clinical_guidelines
        self.alert_thresholds = alert_thresholds
        
        # Clinical models
        self.risk_assessment_model = self._initialize_risk_model()
        self.guideline_engine = GuidelineEngine(clinical_guidelines)
        
        # Patient data cache
        self.patient_data_cache = {}
        
        # Alert history
        self.alert_history = defaultdict(list)
        
        logger.info(f"Initialized ClinicalDecisionSupportAgent for {len(patient_ids)} patients")
    
    def _initialize_risk_model(self) -> RandomForestClassifier:
        """Initialize risk assessment model."""
        # In production, this would load a pre-trained model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create dummy training data for demonstration
        X_dummy = np.random.random((1000, 10))
        y_dummy = np.random.binomial(1, 0.3, 1000)
        model.fit(X_dummy, y_dummy)
        
        return model
    
    def _has_monitoring_tasks(self) -> bool:
        """Check if agent has active monitoring tasks."""
        return len(self.patient_ids) > 0
    
    def _update_beliefs(self):
        """Update beliefs based on patient data observations."""
        for patient_id in self.patient_ids:
            # Simulate patient data retrieval
            patient_data = self._get_patient_data(patient_id)
            
            if patient_data:
                # Update belief about patient condition
                risk_score = self._assess_patient_risk(patient_data)
                
                belief = Belief(
                    content={
                        'patient_id': patient_id,
                        'risk_score': risk_score,
                        'vital_signs': patient_data.get('vital_signs', {}),
                        'lab_results': patient_data.get('lab_results', {}),
                        'medications': patient_data.get('medications', [])
                    },
                    confidence=0.9,
                    timestamp=datetime.now(),
                    source="patient_monitoring_system",
                    evidence=[f"patient_data_{patient_id}"]
                )
                
                self.beliefs[f"patient_condition_{patient_id}"] = belief
    
    def _get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient data from EHR system."""
        # Simulate patient data retrieval
        # In production, this would connect to actual EHR systems
        
        if patient_id not in self.patient_data_cache:
            # Simulate patient data
            self.patient_data_cache[patient_id] = {
                'patient_id': patient_id,
                'vital_signs': {
                    'heart_rate': np.random.normal(75, 15),
                    'blood_pressure_systolic': np.random.normal(120, 20),
                    'blood_pressure_diastolic': np.random.normal(80, 10),
                    'temperature': np.random.normal(98.6, 1.0),
                    'respiratory_rate': np.random.normal(16, 4)
                },
                'lab_results': {
                    'glucose': np.random.normal(100, 30),
                    'creatinine': np.random.normal(1.0, 0.3),
                    'hemoglobin': np.random.normal(13.5, 2.0)
                },
                'medications': ['metformin', 'lisinopril'],
                'last_updated': datetime.now()
            }
        
        return self.patient_data_cache[patient_id]
    
    def _assess_patient_risk(self, patient_data: Dict[str, Any]) -> float:
        """Assess patient risk using clinical model."""
        # Extract features for risk assessment
        features = []
        
        # Vital signs features
        vital_signs = patient_data.get('vital_signs', {})
        features.extend([
            vital_signs.get('heart_rate', 75),
            vital_signs.get('blood_pressure_systolic', 120),
            vital_signs.get('blood_pressure_diastolic', 80),
            vital_signs.get('temperature', 98.6),
            vital_signs.get('respiratory_rate', 16)
        ])
        
        # Lab results features
        lab_results = patient_data.get('lab_results', {})
        features.extend([
            lab_results.get('glucose', 100),
            lab_results.get('creatinine', 1.0),
            lab_results.get('hemoglobin', 13.5)
        ])
        
        # Medication count
        medications = patient_data.get('medications', [])
        features.extend([
            len(medications),
            1 if 'insulin' in medications else 0  # Diabetes indicator
        ])
        
        # Ensure we have exactly 10 features
        while len(features) < 10:
            features.append(0.0)
        
        features = np.array(features[:10]).reshape(1, -1)
        
        # Get risk probability
        risk_probability = self.risk_assessment_model.predict_proba(features)[0][1]
        
        return float(risk_probability)
    
    def _generate_desires(self):
        """Generate new desires based on patient conditions."""
        for patient_id in self.patient_ids:
            belief_key = f"patient_condition_{patient_id}"
            
            if belief_key in self.beliefs:
                belief = self.beliefs[belief_key]
                risk_score = belief.content['risk_score']
                
                # Generate alert desire if risk is high
                if risk_score > self.alert_thresholds.get('high_risk', 0.8):
                    desire_id = f"alert_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    if desire_id not in self.desires:
                        desire = Desire(
                            goal_id=desire_id,
                            description=f"Generate high-risk alert for patient {patient_id}",
                            target_outcome={
                                'alert_generated': True,
                                'clinician_notified': True,
                                'patient_id': patient_id,
                                'risk_score': risk_score
                            },
                            priority=Priority.HIGH,
                            deadline=datetime.now() + timedelta(minutes=15)
                        )
                        
                        self.desires[desire_id] = desire
                
                # Generate guideline compliance check desire
                compliance_desire_id = f"compliance_check_{patient_id}"
                
                if compliance_desire_id not in self.desires:
                    desire = Desire(
                        goal_id=compliance_desire_id,
                        description=f"Check guideline compliance for patient {patient_id}",
                        target_outcome={
                            'compliance_checked': True,
                            'recommendations_generated': True,
                            'patient_id': patient_id
                        },
                        priority=Priority.MEDIUM,
                        deadline=datetime.now() + timedelta(hours=1)
                    )
                    
                    self.desires[compliance_desire_id] = desire
    
    def _plan_actions(self):
        """Plan actions to achieve desires."""
        for desire in self.desires.values():
            if desire.goal_id not in self.intentions:
                actions = []
                
                if "alert" in desire.description:
                    actions = [
                        {
                            'action_type': 'generate_alert',
                            'parameters': {
                                'patient_id': desire.target_outcome['patient_id'],
                                'risk_score': desire.target_outcome['risk_score'],
                                'alert_type': 'high_risk'
                            }
                        },
                        {
                            'action_type': 'notify_clinician',
                            'parameters': {
                                'patient_id': desire.target_outcome['patient_id'],
                                'message_type': 'urgent_alert'
                            }
                        }
                    ]
                
                elif "compliance_check" in desire.description:
                    actions = [
                        {
                            'action_type': 'check_guidelines',
                            'parameters': {
                                'patient_id': desire.target_outcome['patient_id']
                            }
                        },
                        {
                            'action_type': 'generate_recommendations',
                            'parameters': {
                                'patient_id': desire.target_outcome['patient_id']
                            }
                        }
                    ]
                
                if actions:
                    intention = Intention(
                        plan_id=str(uuid.uuid4()),
                        goal_id=desire.goal_id,
                        actions=actions,
                        estimated_completion=datetime.now() + timedelta(minutes=5)
                    )
                    
                    self.intentions[desire.goal_id] = intention
    
    def _execute_intentions(self):
        """Execute current intentions."""
        completed_intentions = []
        
        for intention in self.intentions.values():
            if intention.status == "planned":
                intention.status = "executing"
            
            if intention.status == "executing":
                next_action = intention.get_next_action()
                
                if next_action:
                    success = self._execute_action(next_action)
                    
                    if success:
                        intention.current_step += 1
                        
                        if intention.is_complete():
                            intention.status = "completed"
                            completed_intentions.append(intention.goal_id)
                            self.performance_metrics['tasks_completed'] += 1
                    else:
                        intention.status = "failed"
                        completed_intentions.append(intention.goal_id)
                        self.performance_metrics['tasks_failed'] += 1
        
        # Clean up completed intentions
        for goal_id in completed_intentions:
            if goal_id in self.intentions:
                del self.intentions[goal_id]
            if goal_id in self.desires:
                del self.desires[goal_id]
    
    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute a specific action."""
        action_type = action['action_type']
        parameters = action['parameters']
        
        try:
            if action_type == 'generate_alert':
                return self._generate_alert(parameters)
            elif action_type == 'notify_clinician':
                return self._notify_clinician(parameters)
            elif action_type == 'check_guidelines':
                return self._check_guidelines(parameters)
            elif action_type == 'generate_recommendations':
                return self._generate_recommendations(parameters)
            else:
                logger.error(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            return False
    
    def _generate_alert(self, parameters: Dict[str, Any]) -> bool:
        """Generate clinical alert."""
        patient_id = parameters['patient_id']
        risk_score = parameters['risk_score']
        alert_type = parameters['alert_type']
        
        alert = {
            'alert_id': str(uuid.uuid4()),
            'patient_id': patient_id,
            'alert_type': alert_type,
            'risk_score': risk_score,
            'timestamp': datetime.now(),
            'agent_id': self.agent_id,
            'status': 'active'
        }
        
        self.alert_history[patient_id].append(alert)
        
        logger.info(f"Generated {alert_type} alert for patient {patient_id} (risk: {risk_score:.3f})")
        
        return True
    
    def _notify_clinician(self, parameters: Dict[str, Any]) -> bool:
        """Notify clinician about patient condition."""
        patient_id = parameters['patient_id']
        message_type = parameters['message_type']
        
        # In production, this would send actual notifications
        # For demonstration, we log the notification
        
        notification = {
            'notification_id': str(uuid.uuid4()),
            'patient_id': patient_id,
            'message_type': message_type,
            'timestamp': datetime.now(),
            'agent_id': self.agent_id,
            'delivered': True
        }
        
        logger.info(f"Notified clinician about patient {patient_id} ({message_type})")
        
        return True
    
    def _check_guidelines(self, parameters: Dict[str, Any]) -> bool:
        """Check guideline compliance for patient."""
        patient_id = parameters['patient_id']
        
        # Get patient data
        patient_data = self._get_patient_data(patient_id)
        
        if not patient_data:
            return False
        
        # Check compliance using guideline engine
        compliance_results = self.guideline_engine.check_compliance(patient_data)
        
        # Store compliance results as belief
        belief = Belief(
            content={
                'patient_id': patient_id,
                'compliance_results': compliance_results,
                'checked_guidelines': list(self.clinical_guidelines.keys())
            },
            confidence=0.95,
            timestamp=datetime.now(),
            source="guideline_engine",
            evidence=[f"guideline_check_{patient_id}"]
        )
        
        self.beliefs[f"guideline_compliance_{patient_id}"] = belief
        
        logger.info(f"Checked guideline compliance for patient {patient_id}")
        
        return True
    
    def _generate_recommendations(self, parameters: Dict[str, Any]) -> bool:
        """Generate clinical recommendations for patient."""
        patient_id = parameters['patient_id']
        
        # Get compliance results
        compliance_belief_key = f"guideline_compliance_{patient_id}"
        
        if compliance_belief_key not in self.beliefs:
            return False
        
        compliance_belief = self.beliefs[compliance_belief_key]
        compliance_results = compliance_belief.content['compliance_results']
        
        # Generate recommendations based on compliance gaps
        recommendations = []
        
        for guideline, compliance_status in compliance_results.items():
            if not compliance_status['compliant']:
                recommendations.append({
                    'guideline': guideline,
                    'recommendation': compliance_status['recommendation'],
                    'priority': compliance_status['priority'],
                    'evidence_level': compliance_status['evidence_level']
                })
        
        # Store recommendations as belief
        belief = Belief(
            content={
                'patient_id': patient_id,
                'recommendations': recommendations,
                'generated_by': self.agent_id
            },
            confidence=0.9,
            timestamp=datetime.now(),
            source="recommendation_engine",
            evidence=[f"compliance_check_{patient_id}"]
        )
        
        self.beliefs[f"recommendations_{patient_id}"] = belief
        
        logger.info(f"Generated {len(recommendations)} recommendations for patient {patient_id}")
        
        return True

class CareCoordinationAgent(HealthcareAIAgent):
    """
    Specialized agent for care coordination across multiple providers.
    
    This agent manages care pathways, coordinates appointments,
    and ensures continuity of care for complex patients.
    """
    
    def __init__(
        self,
        agent_id: str,
        care_pathways: Dict[str, Any],
        provider_network: Dict[str, Any]
    ):
        """
        Initialize care coordination agent.
        
        Args:
            agent_id: Unique identifier for the agent
            care_pathways: Defined care pathways for different conditions
            provider_network: Network of healthcare providers
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="care_coordination",
            capabilities=[
                "pathway_management",
                "appointment_scheduling",
                "provider_coordination",
                "care_gap_identification"
            ]
        )
        
        self.care_pathways = care_pathways
        self.provider_network = provider_network
        
        # Care coordination state
        self.active_care_plans = {}
        self.appointment_schedule = {}
        self.care_gaps = defaultdict(list)
        
        # Coordination algorithms
        self.pathway_engine = CarePathwayEngine(care_pathways)
        self.scheduling_optimizer = AppointmentSchedulingOptimizer()
        
        logger.info(f"Initialized CareCoordinationAgent with {len(care_pathways)} pathways")
    
    def _update_beliefs(self):
        """Update beliefs about care coordination status."""
        # Monitor active care plans
        for plan_id, care_plan in self.active_care_plans.items():
            # Check care plan progress
            progress = self._assess_care_plan_progress(care_plan)
            
            belief = Belief(
                content={
                    'care_plan_id': plan_id,
                    'patient_id': care_plan['patient_id'],
                    'progress': progress,
                    'next_milestones': care_plan.get('next_milestones', []),
                    'providers_involved': care_plan.get('providers', [])
                },
                confidence=0.9,
                timestamp=datetime.now(),
                source="care_plan_monitoring",
                evidence=[f"care_plan_{plan_id}"]
            )
            
            self.beliefs[f"care_plan_progress_{plan_id}"] = belief
    
    def _assess_care_plan_progress(self, care_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Assess progress of a care plan."""
        # Simulate care plan progress assessment
        total_milestones = len(care_plan.get('milestones', []))
        completed_milestones = len([
            m for m in care_plan.get('milestones', []) 
            if m.get('status') == 'completed'
        ])
        
        progress = {
            'completion_percentage': (completed_milestones / total_milestones * 100) if total_milestones > 0 else 0,
            'milestones_completed': completed_milestones,
            'milestones_total': total_milestones,
            'on_schedule': True,  # Simplified for demonstration
            'care_gaps_identified': len(self.care_gaps.get(care_plan['patient_id'], []))
        }
        
        return progress
    
    def _generate_desires(self):
        """Generate desires for care coordination tasks."""
        # Generate desires for care gap resolution
        for patient_id, gaps in self.care_gaps.items():
            if gaps:
                desire_id = f"resolve_care_gaps_{patient_id}"
                
                if desire_id not in self.desires:
                    desire = Desire(
                        goal_id=desire_id,
                        description=f"Resolve care gaps for patient {patient_id}",
                        target_outcome={
                            'care_gaps_resolved': True,
                            'patient_id': patient_id,
                            'gaps_count': len(gaps)
                        },
                        priority=Priority.HIGH,
                        deadline=datetime.now() + timedelta(days=1)
                    )
                    
                    self.desires[desire_id] = desire
        
        # Generate desires for appointment optimization
        if self.appointment_schedule:
            desire_id = "optimize_appointments"
            
            if desire_id not in self.desires:
                desire = Desire(
                    goal_id=desire_id,
                    description="Optimize appointment scheduling",
                    target_outcome={
                        'appointments_optimized': True,
                        'efficiency_improved': True
                    },
                    priority=Priority.MEDIUM,
                    deadline=datetime.now() + timedelta(hours=4)
                )
                
                self.desires[desire_id] = desire
    
    def _plan_actions(self):
        """Plan actions for care coordination."""
        for desire in self.desires.values():
            if desire.goal_id not in self.intentions:
                actions = []
                
                if "care_gaps" in desire.description:
                    actions = [
                        {
                            'action_type': 'identify_care_gaps',
                            'parameters': {
                                'patient_id': desire.target_outcome['patient_id']
                            }
                        },
                        {
                            'action_type': 'coordinate_care_resolution',
                            'parameters': {
                                'patient_id': desire.target_outcome['patient_id']
                            }
                        }
                    ]
                
                elif "optimize_appointments" in desire.description:
                    actions = [
                        {
                            'action_type': 'analyze_appointment_schedule',
                            'parameters': {}
                        },
                        {
                            'action_type': 'optimize_scheduling',
                            'parameters': {}
                        }
                    ]
                
                if actions:
                    intention = Intention(
                        plan_id=str(uuid.uuid4()),
                        goal_id=desire.goal_id,
                        actions=actions,
                        estimated_completion=datetime.now() + timedelta(minutes=30)
                    )
                    
                    self.intentions[desire.goal_id] = intention
    
    def _execute_intentions(self):
        """Execute care coordination intentions."""
        completed_intentions = []
        
        for intention in self.intentions.values():
            if intention.status == "planned":
                intention.status = "executing"
            
            if intention.status == "executing":
                next_action = intention.get_next_action()
                
                if next_action:
                    success = self._execute_coordination_action(next_action)
                    
                    if success:
                        intention.current_step += 1
                        
                        if intention.is_complete():
                            intention.status = "completed"
                            completed_intentions.append(intention.goal_id)
                            self.performance_metrics['tasks_completed'] += 1
                    else:
                        intention.status = "failed"
                        completed_intentions.append(intention.goal_id)
                        self.performance_metrics['tasks_failed'] += 1
        
        # Clean up completed intentions
        for goal_id in completed_intentions:
            if goal_id in self.intentions:
                del self.intentions[goal_id]
            if goal_id in self.desires:
                del self.desires[goal_id]
    
    def _execute_coordination_action(self, action: Dict[str, Any]) -> bool:
        """Execute a care coordination action."""
        action_type = action['action_type']
        parameters = action['parameters']
        
        try:
            if action_type == 'identify_care_gaps':
                return self._identify_care_gaps(parameters)
            elif action_type == 'coordinate_care_resolution':
                return self._coordinate_care_resolution(parameters)
            elif action_type == 'analyze_appointment_schedule':
                return self._analyze_appointment_schedule(parameters)
            elif action_type == 'optimize_scheduling':
                return self._optimize_scheduling(parameters)
            else:
                logger.error(f"Unknown coordination action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing coordination action {action_type}: {e}")
            return False
    
    def _identify_care_gaps(self, parameters: Dict[str, Any]) -> bool:
        """Identify care gaps for a patient."""
        patient_id = parameters['patient_id']
        
        # Simulate care gap identification
        # In production, this would analyze care plans, guidelines, and patient data
        
        identified_gaps = [
            {
                'gap_type': 'missing_specialist_referral',
                'description': 'Patient needs cardiology consultation',
                'priority': 'high',
                'recommended_action': 'Schedule cardiology appointment within 2 weeks'
            },
            {
                'gap_type': 'overdue_lab_work',
                'description': 'HbA1c test overdue by 3 months',
                'priority': 'medium',
                'recommended_action': 'Order HbA1c test'
            }
        ]
        
        self.care_gaps[patient_id] = identified_gaps
        
        logger.info(f"Identified {len(identified_gaps)} care gaps for patient {patient_id}")
        
        return True
    
    def _coordinate_care_resolution(self, parameters: Dict[str, Any]) -> bool:
        """Coordinate resolution of care gaps."""
        patient_id = parameters['patient_id']
        gaps = self.care_gaps.get(patient_id, [])
        
        resolved_gaps = 0
        
        for gap in gaps:
            # Simulate gap resolution coordination
            if gap['gap_type'] == 'missing_specialist_referral':
                # Find available specialist
                specialist = self._find_available_specialist('cardiology')
                if specialist:
                    # Schedule appointment
                    appointment_scheduled = self._schedule_appointment(patient_id, specialist)
                    if appointment_scheduled:
                        gap['status'] = 'resolved'
                        resolved_gaps += 1
            
            elif gap['gap_type'] == 'overdue_lab_work':
                # Order lab work
                lab_ordered = self._order_lab_work(patient_id, gap['description'])
                if lab_ordered:
                    gap['status'] = 'resolved'
                    resolved_gaps += 1
        
        logger.info(f"Resolved {resolved_gaps}/{len(gaps)} care gaps for patient {patient_id}")
        
        return resolved_gaps > 0
    
    def _find_available_specialist(self, specialty: str) -> Optional[Dict[str, Any]]:
        """Find available specialist in provider network."""
        # Simulate specialist search
        specialists = [
            provider for provider in self.provider_network.values()
            if provider.get('specialty') == specialty and provider.get('available', True)
        ]
        
        return specialists[0] if specialists else None
    
    def _schedule_appointment(self, patient_id: str, provider: Dict[str, Any]) -> bool:
        """Schedule appointment with provider."""
        appointment_id = str(uuid.uuid4())
        
        appointment = {
            'appointment_id': appointment_id,
            'patient_id': patient_id,
            'provider_id': provider['provider_id'],
            'appointment_type': 'consultation',
            'scheduled_time': datetime.now() + timedelta(days=7),  # Schedule for next week
            'status': 'scheduled'
        }
        
        self.appointment_schedule[appointment_id] = appointment
        
        logger.info(f"Scheduled appointment {appointment_id} for patient {patient_id}")
        
        return True
    
    def _order_lab_work(self, patient_id: str, lab_description: str) -> bool:
        """Order lab work for patient."""
        # Simulate lab work ordering
        order_id = str(uuid.uuid4())
        
        lab_order = {
            'order_id': order_id,
            'patient_id': patient_id,
            'description': lab_description,
            'ordered_time': datetime.now(),
            'status': 'ordered'
        }
        
        logger.info(f"Ordered lab work {order_id} for patient {patient_id}: {lab_description}")
        
        return True
    
    def _analyze_appointment_schedule(self, parameters: Dict[str, Any]) -> bool:
        """Analyze current appointment schedule for optimization opportunities."""
        # Simulate schedule analysis
        total_appointments = len(self.appointment_schedule)
        
        if total_appointments == 0:
            return True
        
        # Analyze scheduling efficiency
        efficiency_metrics = {
            'total_appointments': total_appointments,
            'average_wait_time': 7.5,  # days
            'provider_utilization': 0.75,
            'patient_satisfaction': 0.8
        }
        
        # Store analysis results
        belief = Belief(
            content={
                'schedule_analysis': efficiency_metrics,
                'optimization_opportunities': [
                    'Reduce average wait time',
                    'Improve provider utilization',
                    'Better appointment clustering'
                ]
            },
            confidence=0.9,
            timestamp=datetime.now(),
            source="schedule_analyzer",
            evidence=['appointment_schedule_data']
        )
        
        self.beliefs['schedule_analysis'] = belief
        
        logger.info(f"Analyzed appointment schedule: {total_appointments} appointments")
        
        return True
    
    def _optimize_scheduling(self, parameters: Dict[str, Any]) -> bool:
        """Optimize appointment scheduling."""
        # Use scheduling optimizer
        optimized_schedule = self.scheduling_optimizer.optimize(self.appointment_schedule)
        
        if optimized_schedule:
            self.appointment_schedule = optimized_schedule
            
            logger.info("Optimized appointment schedule")
            return True
        
        return False

# Supporting classes for healthcare AI agents

class ClinicalKnowledgeBase:
    """Clinical knowledge base for healthcare AI agents."""
    
    def __init__(self):
        self.guidelines = {}
        self.drug_interactions = {}
        self.clinical_protocols = {}
    
    def get_guideline(self, condition: str) -> Optional[Dict[str, Any]]:
        """Get clinical guideline for a condition."""
        return self.guidelines.get(condition)
    
    def check_drug_interaction(self, drug1: str, drug2: str) -> Optional[Dict[str, Any]]:
        """Check for drug interactions."""
        interaction_key = tuple(sorted([drug1, drug2]))
        return self.drug_interactions.get(interaction_key)

class AgentSafetyMonitor:
    """Safety monitoring system for healthcare AI agents."""
    
    def __init__(self):
        self.safety_rules = [
            self._check_patient_safety,
            self._check_data_privacy,
            self._check_clinical_appropriateness
        ]
    
    def check_agent_safety(self, agent: HealthcareAIAgent) -> List[Dict[str, Any]]:
        """Check agent safety and return any violations."""
        violations = []
        
        for rule in self.safety_rules:
            violation = rule(agent)
            if violation:
                violations.append(violation)
        
        return violations
    
    def _check_patient_safety(self, agent: HealthcareAIAgent) -> Optional[Dict[str, Any]]:
        """Check for patient safety violations."""
        # Implement patient safety checks
        return None
    
    def _check_data_privacy(self, agent: HealthcareAIAgent) -> Optional[Dict[str, Any]]:
        """Check for data privacy violations."""
        # Implement privacy checks
        return None
    
    def _check_clinical_appropriateness(self, agent: HealthcareAIAgent) -> Optional[Dict[str, Any]]:
        """Check for clinical appropriateness violations."""
        # Implement clinical appropriateness checks
        return None

class GuidelineEngine:
    """Engine for checking clinical guideline compliance."""
    
    def __init__(self, guidelines: Dict[str, Any]):
        self.guidelines = guidelines
    
    def check_compliance(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check patient data against clinical guidelines."""
        compliance_results = {}
        
        for guideline_name, guideline in self.guidelines.items():
            compliance_results[guideline_name] = self._check_single_guideline(
                patient_data, guideline
            )
        
        return compliance_results
    
    def _check_single_guideline(
        self, 
        patient_data: Dict[str, Any], 
        guideline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check compliance with a single guideline."""
        # Simplified guideline checking
        return {
            'compliant': True,  # Simplified for demonstration
            'recommendation': 'Continue current treatment',
            'priority': 'medium',
            'evidence_level': 'A'
        }

class CarePathwayEngine:
    """Engine for managing care pathways."""
    
    def __init__(self, pathways: Dict[str, Any]):
        self.pathways = pathways
    
    def get_pathway(self, condition: str) -> Optional[Dict[str, Any]]:
        """Get care pathway for a condition."""
        return self.pathways.get(condition)
    
    def assess_pathway_progress(
        self, 
        pathway: Dict[str, Any], 
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess progress along a care pathway."""
        # Implement pathway progress assessment
        return {
            'current_stage': 'diagnosis',
            'completion_percentage': 25.0,
            'next_milestones': ['specialist_consultation', 'treatment_initiation']
        }

class AppointmentSchedulingOptimizer:
    """Optimizer for appointment scheduling."""
    
    def optimize(self, current_schedule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize appointment schedule."""
        # Implement scheduling optimization algorithm
        # For demonstration, return the current schedule
        return current_schedule

# Example usage and demonstration
def main():
    """Demonstrate healthcare AI agents."""
    
    # Create clinical decision support agent
    clinical_guidelines = {
        'diabetes_management': {
            'hba1c_target': 7.0,
            'blood_pressure_target': '130/80',
            'medication_first_line': 'metformin'
        },
        'hypertension_management': {
            'blood_pressure_target': '130/80',
            'medication_first_line': 'ace_inhibitor'
        }
    }
    
    alert_thresholds = {
        'high_risk': 0.8,
        'medium_risk': 0.6,
        'low_risk': 0.3
    }
    
    cds_agent = ClinicalDecisionSupportAgent(
        agent_id="cds_agent_001",
        patient_ids=["patient_001", "patient_002", "patient_003"],
        clinical_guidelines=clinical_guidelines,
        alert_thresholds=alert_thresholds
    )
    
    # Create care coordination agent
    care_pathways = {
        'diabetes_care': {
            'stages': ['screening', 'diagnosis', 'treatment_initiation', 'monitoring'],
            'duration_weeks': 12,
            'providers': ['primary_care', 'endocrinologist', 'nutritionist']
        },
        'cardiac_care': {
            'stages': ['assessment', 'intervention', 'rehabilitation', 'follow_up'],
            'duration_weeks': 24,
            'providers': ['cardiologist', 'cardiac_surgeon', 'physical_therapist']
        }
    }
    
    provider_network = {
        'provider_001': {
            'provider_id': 'provider_001',
            'name': 'Dr. Smith',
            'specialty': 'cardiology',
            'available': True
        },
        'provider_002': {
            'provider_id': 'provider_002',
            'name': 'Dr. Johnson',
            'specialty': 'endocrinology',
            'available': True
        }
    }
    
    coordination_agent = CareCoordinationAgent(
        agent_id="coordination_agent_001",
        care_pathways=care_pathways,
        provider_network=provider_network
    )
    
    # Establish communication between agents
    cds_agent.add_communication_partner(coordination_agent)
    
    # Start agents
    print("Starting healthcare AI agents...")
    cds_agent.start()
    coordination_agent.start()
    
    # Let agents run for a short time
    time.sleep(5)
    
    # Check agent status
    print("\nAgent Status:")
    print("=" * 50)
    
    cds_status = cds_agent.get_status()
    print(f"Clinical Decision Support Agent:")
    print(f"  State: {cds_status['state']}")
    print(f"  Beliefs: {cds_status['beliefs_count']}")
    print(f"  Desires: {cds_status['desires_count']}")
    print(f"  Intentions: {cds_status['intentions_count']}")
    print(f"  Tasks Completed: {cds_status['performance_metrics']['tasks_completed']}")
    
    coordination_status = coordination_agent.get_status()
    print(f"\nCare Coordination Agent:")
    print(f"  State: {coordination_status['state']}")
    print(f"  Beliefs: {coordination_status['beliefs_count']}")
    print(f"  Desires: {coordination_status['desires_count']}")
    print(f"  Intentions: {coordination_status['intentions_count']}")
    print(f"  Tasks Completed: {coordination_status['performance_metrics']['tasks_completed']}")
    
    # Demonstrate agent communication
    print("\nDemonstrating agent communication...")
    cds_agent.send_message(
        receiver_id=coordination_agent.agent_id,
        message_type="belief_update",
        content={
            'key': 'high_risk_patient',
            'content': {
                'patient_id': 'patient_001',
                'risk_level': 'high',
                'recommended_action': 'urgent_consultation'
            },
            'confidence': 0.9,
            'evidence': ['risk_assessment_model']
        },
        priority=Priority.HIGH
    )
    
    # Let agents process the message
    time.sleep(2)
    
    # Stop agents
    print("\nStopping agents...")
    cds_agent.stop()
    coordination_agent.stop()
    
    print("Healthcare AI agents demonstration completed!")

if __name__ == "__main__":
    main()
```

## 7.3 Multi-Agent Systems for Healthcare

### 7.3.1 Coordination Mechanisms

Multi-agent systems in healthcare require sophisticated coordination mechanisms to ensure effective collaboration while maintaining patient safety and care quality. The complexity of healthcare delivery, with its multiple stakeholders, competing priorities, and time-sensitive decisions, necessitates robust coordination protocols.

**Contract Net Protocol**: This protocol enables dynamic task allocation among healthcare agents. When a clinical task arises, an agent can announce the task and invite bids from other agents. The announcing agent then selects the most appropriate agent based on capabilities, availability, and cost.

The formal representation of the Contract Net Protocol in healthcare contexts:

$$CNP = \langle A, T, B, S, E \rangle$$

where:
- $A$ is the set of participating healthcare agents
- $T$ is the set of clinical tasks to be allocated
- $B$ is the bidding function that maps agent capabilities to task requirements
- $S$ is the selection function that chooses the optimal agent for each task
- $E$ is the evaluation function that assesses task completion quality

**Consensus Algorithms**: For critical clinical decisions requiring agreement among multiple agents, consensus algorithms ensure that all participating agents reach agreement on the appropriate course of action.

**Auction Mechanisms**: Healthcare resource allocation can be optimized using auction mechanisms where agents bid for scarce resources such as operating room time, specialist consultations, or diagnostic equipment.

### 7.3.2 Implementation of Multi-Agent Healthcare Systems

```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import uuid
import queue
import time
from collections import defaultdict, deque
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Status of tasks in the multi-agent system."""
    ANNOUNCED = "announced"
    BIDDING = "bidding"
    ALLOCATED = "allocated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ResourceType(Enum):
    """Types of healthcare resources."""
    OPERATING_ROOM = "operating_room"
    ICU_BED = "icu_bed"
    SPECIALIST_TIME = "specialist_time"
    DIAGNOSTIC_EQUIPMENT = "diagnostic_equipment"
    NURSING_STAFF = "nursing_staff"
    MEDICATION = "medication"

@dataclass
class Task:
    """Represents a clinical task in the multi-agent system."""
    task_id: str
    task_type: str
    description: str
    requirements: Dict[str, Any]
    priority: int  # 1 = highest, 5 = lowest
    deadline: datetime
    patient_id: str
    estimated_duration: timedelta
    required_capabilities: List[str]
    resource_requirements: Dict[ResourceType, int]
    status: TaskStatus = TaskStatus.ANNOUNCED
    assigned_agent: Optional[str] = None
    created_time: datetime = field(default_factory=datetime.now)
    
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        return datetime.now() > self.deadline
    
    def get_urgency_score(self) -> float:
        """Calculate urgency score based on priority and time remaining."""
        time_remaining = (self.deadline - datetime.now()).total_seconds()
        max_time = 24 * 3600  # 24 hours in seconds
        
        time_factor = max(0, min(1, time_remaining / max_time))
        priority_factor = (6 - self.priority) / 5  # Invert priority (1=high becomes 1.0)
        
        return priority_factor * 0.7 + (1 - time_factor) * 0.3

@dataclass
class Bid:
    """Represents a bid for a task."""
    bid_id: str
    task_id: str
    agent_id: str
    cost: float
    estimated_completion_time: datetime
    quality_score: float
    confidence: float
    capabilities_match: float
    bid_time: datetime = field(default_factory=datetime.now)
    
    def get_bid_score(self) -> float:
        """Calculate overall bid score."""
        # Normalize factors (lower cost and time are better)
        cost_factor = 1.0 / (1.0 + self.cost)
        time_factor = 1.0 / (1.0 + (self.estimated_completion_time - datetime.now()).total_seconds() / 3600)
        
        # Combine factors
        score = (
            cost_factor * 0.2 +
            time_factor * 0.3 +
            self.quality_score * 0.2 +
            self.confidence * 0.15 +
            self.capabilities_match * 0.15
        )
        
        return score

@dataclass
class Resource:
    """Represents a healthcare resource."""
    resource_id: str
    resource_type: ResourceType
    capacity: int
    available_capacity: int
    location: str
    capabilities: List[str]
    cost_per_hour: float
    maintenance_schedule: List[Tuple[datetime, datetime]]
    
    def is_available(self, start_time: datetime, duration: timedelta) -> bool:
        """Check if resource is available for the specified time period."""
        end_time = start_time + duration
        
        # Check capacity
        if self.available_capacity <= 0:
            return False
        
        # Check maintenance schedule
        for maint_start, maint_end in self.maintenance_schedule:
            if not (end_time <= maint_start or start_time >= maint_end):
                return False
        
        return True
    
    def reserve(self, start_time: datetime, duration: timedelta) -> bool:
        """Reserve the resource for the specified time period."""
        if self.is_available(start_time, duration):
            self.available_capacity -= 1
            return True
        return False
    
    def release(self) -> None:
        """Release the resource."""
        if self.available_capacity < self.capacity:
            self.available_capacity += 1

class MultiAgentHealthcareSystem:
    """
    Comprehensive multi-agent system for healthcare coordination.
    
    This system implements advanced coordination mechanisms including
    contract net protocol, consensus algorithms, and auction mechanisms
    for optimal healthcare resource allocation and task coordination.
    
    Based on research from:
    Isern, D., & Moreno, A. (2016). A systematic literature review of agents 
    applied in healthcare. Journal of Medical Systems, 40(2), 1-14.
    DOI: 10.1007/s10916-015-0376-2
    
    And coordination mechanisms from:
    Smith, R. G. (1980). The contract net protocol: High-level communication 
    and control in a distributed problem solver. IEEE Transactions on 
    Computers, 29(12), 1104-1113. DOI: 10.1109/TC.1980.1675516
    """
    
    def __init__(
        self,
        system_id: str,
        max_agents: int = 100,
        coordination_timeout: int = 300  # seconds
    ):
        """
        Initialize multi-agent healthcare system.
        
        Args:
            system_id: Unique identifier for the system
            max_agents: Maximum number of agents in the system
            coordination_timeout: Timeout for coordination operations
        """
        self.system_id = system_id
        self.max_agents = max_agents
        self.coordination_timeout = coordination_timeout
        
        # System components
        self.agents: Dict[str, HealthcareAIAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.bids: Dict[str, List[Bid]] = defaultdict(list)
        self.resources: Dict[str, Resource] = {}
        
        # Coordination mechanisms
        self.contract_net = ContractNetProtocol(self)
        self.consensus_engine = ConsensusEngine(self)
        self.auction_system = AuctionSystem(self)
        self.resource_manager = ResourceManager(self)
        
        # System state
        self.running = False
        self.coordination_thread = None
        self.message_broker = MessageBroker()
        
        # Performance monitoring
        self.system_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_task_completion_time': 0.0,
            'resource_utilization': 0.0,
            'agent_efficiency': 0.0,
            'coordination_overhead': 0.0
        }
        
        # Communication network
        self.communication_network = nx.Graph()
        
        logger.info(f"Initialized MultiAgentHealthcareSystem: {system_id}")
    
    def add_agent(self, agent: HealthcareAIAgent) -> bool:
        """Add an agent to the system."""
        if len(self.agents) >= self.max_agents:
            logger.error(f"Cannot add agent {agent.agent_id}: system at capacity")
            return False
        
        if agent.agent_id in self.agents:
            logger.error(f"Agent {agent.agent_id} already exists in system")
            return False
        
        self.agents[agent.agent_id] = agent
        self.communication_network.add_node(agent.agent_id, agent_type=agent.agent_type)
        
        # Connect to message broker
        self.message_broker.register_agent(agent)
        
        logger.info(f"Added agent {agent.agent_id} to system")
        return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the system."""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found in system")
            return False
        
        # Cancel any tasks assigned to this agent
        for task in self.tasks.values():
            if task.assigned_agent == agent_id:
                task.status = TaskStatus.CANCELLED
        
        # Remove from system
        del self.agents[agent_id]
        self.communication_network.remove_node(agent_id)
        self.message_broker.unregister_agent(agent_id)
        
        logger.info(f"Removed agent {agent_id} from system")
        return True
    
    def add_resource(self, resource: Resource) -> bool:
        """Add a resource to the system."""
        if resource.resource_id in self.resources:
            logger.error(f"Resource {resource.resource_id} already exists")
            return False
        
        self.resources[resource.resource_id] = resource
        logger.info(f"Added resource {resource.resource_id} to system")
        return True
    
    def submit_task(self, task: Task) -> bool:
        """Submit a task to the system for allocation."""
        if task.task_id in self.tasks:
            logger.error(f"Task {task.task_id} already exists")
            return False
        
        self.tasks[task.task_id] = task
        
        # Initiate task allocation based on task type
        if task.task_type in ['emergency', 'critical']:
            # Use contract net protocol for urgent tasks
            self.contract_net.announce_task(task)
        elif task.task_type in ['resource_intensive']:
            # Use auction system for resource-intensive tasks
            self.auction_system.create_auction(task)
        else:
            # Use standard allocation for routine tasks
            self._allocate_routine_task(task)
        
        logger.info(f"Submitted task {task.task_id} for allocation")
        return True
    
    def _allocate_routine_task(self, task: Task) -> None:
        """Allocate routine task using simple matching."""
        suitable_agents = self._find_suitable_agents(task)
        
        if suitable_agents:
            # Select agent with highest capability match and lowest current load
            best_agent = min(
                suitable_agents,
                key=lambda a: (
                    -self._calculate_capability_match(a, task),
                    len([t for t in self.tasks.values() if t.assigned_agent == a.agent_id])
                )
            )
            
            task.assigned_agent = best_agent.agent_id
            task.status = TaskStatus.ALLOCATED
            
            # Send task to agent
            self.message_broker.send_message(
                sender_id="system",
                receiver_id=best_agent.agent_id,
                message_type="task_assignment",
                content={'task': task.__dict__}
            )
    
    def _find_suitable_agents(self, task: Task) -> List[HealthcareAIAgent]:
        """Find agents suitable for a task."""
        suitable_agents = []
        
        for agent in self.agents.values():
            # Check if agent has required capabilities
            if all(cap in agent.capabilities for cap in task.required_capabilities):
                # Check if agent is available
                if agent.state not in [AgentState.ERROR, AgentState.SHUTDOWN]:
                    suitable_agents.append(agent)
        
        return suitable_agents
    
    def _calculate_capability_match(self, agent: HealthcareAIAgent, task: Task) -> float:
        """Calculate how well an agent's capabilities match a task."""
        if not task.required_capabilities:
            return 1.0
        
        matched_capabilities = sum(
            1 for cap in task.required_capabilities 
            if cap in agent.capabilities
        )
        
        return matched_capabilities / len(task.required_capabilities)
    
    def start_system(self) -> None:
        """Start the multi-agent system."""
        if self.running:
            logger.warning("System is already running")
            return
        
        self.running = True
        
        # Start all agents
        for agent in self.agents.values():
            agent.start()
        
        # Start coordination thread
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self.coordination_thread.start()
        
        # Start message broker
        self.message_broker.start()
        
        logger.info("Multi-agent healthcare system started")
    
    def stop_system(self) -> None:
        """Stop the multi-agent system."""
        if not self.running:
            logger.warning("System is not running")
            return
        
        self.running = False
        
        # Stop all agents
        for agent in self.agents.values():
            agent.stop()
        
        # Stop message broker
        self.message_broker.stop()
        
        # Wait for coordination thread to finish
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5.0)
        
        logger.info("Multi-agent healthcare system stopped")
    
    def _coordination_loop(self) -> None:
        """Main coordination loop for the system."""
        while self.running:
            try:
                # Process pending tasks
                self._process_pending_tasks()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Perform resource optimization
                self._optimize_resources()
                
                # Handle failed tasks
                self._handle_failed_tasks()
                
                # Clean up completed tasks
                self._cleanup_completed_tasks()
                
                # Sleep briefly
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
    
    def _process_pending_tasks(self) -> None:
        """Process tasks that are pending allocation."""
        pending_tasks = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.ANNOUNCED
        ]
        
        # Sort by urgency
        pending_tasks.sort(key=lambda t: t.get_urgency_score(), reverse=True)
        
        for task in pending_tasks:
            if task.is_overdue():
                task.status = TaskStatus.FAILED
                self.system_metrics['tasks_failed'] += 1
                logger.warning(f"Task {task.task_id} failed due to deadline")
            else:
                # Try to allocate task
                self._allocate_routine_task(task)
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        
        self.system_metrics['tasks_completed'] = completed_tasks
        self.system_metrics['tasks_failed'] = failed_tasks
        
        # Calculate resource utilization
        if self.resources:
            total_capacity = sum(r.capacity for r in self.resources.values())
            used_capacity = sum(r.capacity - r.available_capacity for r in self.resources.values())
            self.system_metrics['resource_utilization'] = used_capacity / total_capacity if total_capacity > 0 else 0
        
        # Calculate agent efficiency
        if self.agents:
            total_efficiency = sum(
                agent.performance_metrics.get('tasks_completed', 0) /
                max(1, agent.performance_metrics.get('tasks_completed', 0) + 
                    agent.performance_metrics.get('tasks_failed', 0))
                for agent in self.agents.values()
            )
            self.system_metrics['agent_efficiency'] = total_efficiency / len(self.agents)
    
    def _optimize_resources(self) -> None:
        """Optimize resource allocation across the system."""
        # Use resource manager to optimize allocation
        self.resource_manager.optimize_allocation()
    
    def _handle_failed_tasks(self) -> None:
        """Handle tasks that have failed."""
        failed_tasks = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.FAILED
        ]
        
        for task in failed_tasks:
            # Try to reassign critical tasks
            if task.priority <= 2:  # High priority tasks
                task.status = TaskStatus.ANNOUNCED
                task.deadline = datetime.now() + timedelta(hours=1)  # Extend deadline
                logger.info(f"Reassigning failed critical task {task.task_id}")
    
    def _cleanup_completed_tasks(self) -> None:
        """Clean up old completed tasks."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        completed_task_ids = [
            task_id for task_id, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED and task.created_time < cutoff_time
        ]
        
        for task_id in completed_task_ids:
            del self.tasks[task_id]
            if task_id in self.bids:
                del self.bids[task_id]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'system_id': self.system_id,
            'running': self.running,
            'agents_count': len(self.agents),
            'tasks_count': len(self.tasks),
            'resources_count': len(self.resources),
            'system_metrics': self.system_metrics.copy(),
            'task_status_distribution': self._get_task_status_distribution(),
            'agent_status_distribution': self._get_agent_status_distribution(),
            'resource_utilization': self._get_resource_utilization()
        }
    
    def _get_task_status_distribution(self) -> Dict[str, int]:
        """Get distribution of task statuses."""
        distribution = defaultdict(int)
        for task in self.tasks.values():
            distribution[task.status.value] += 1
        return dict(distribution)
    
    def _get_agent_status_distribution(self) -> Dict[str, int]:
        """Get distribution of agent statuses."""
        distribution = defaultdict(int)
        for agent in self.agents.values():
            distribution[agent.state.value] += 1
        return dict(distribution)
    
    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization by type."""
        utilization = {}
        
        for resource_type in ResourceType:
            type_resources = [
                r for r in self.resources.values() 
                if r.resource_type == resource_type
            ]
            
            if type_resources:
                total_capacity = sum(r.capacity for r in type_resources)
                used_capacity = sum(
                    r.capacity - r.available_capacity 
                    for r in type_resources
                )
                utilization[resource_type.value] = used_capacity / total_capacity
            else:
                utilization[resource_type.value] = 0.0
        
        return utilization

class ContractNetProtocol:
    """Implementation of Contract Net Protocol for task allocation."""
    
    def __init__(self, system: MultiAgentHealthcareSystem):
        self.system = system
        self.active_announcements: Dict[str, datetime] = {}
        self.bidding_timeout = 60  # seconds
    
    def announce_task(self, task: Task) -> None:
        """Announce a task for bidding."""
        self.active_announcements[task.task_id] = datetime.now()
        
        # Send announcement to all suitable agents
        suitable_agents = self.system._find_suitable_agents(task)
        
        for agent in suitable_agents:
            self.system.message_broker.send_message(
                sender_id="contract_net",
                receiver_id=agent.agent_id,
                message_type="task_announcement",
                content={
                    'task': task.__dict__,
                    'bidding_deadline': (datetime.now() + timedelta(seconds=self.bidding_timeout)).isoformat()
                }
            )
        
        # Schedule bid evaluation
        threading.Timer(
            self.bidding_timeout,
            self._evaluate_bids,
            args=[task.task_id]
        ).start()
        
        logger.info(f"Announced task {task.task_id} for bidding")
    
    def receive_bid(self, bid: Bid) -> None:
        """Receive a bid for a task."""
        if bid.task_id in self.active_announcements:
            self.system.bids[bid.task_id].append(bid)
            logger.info(f"Received bid {bid.bid_id} for task {bid.task_id}")
    
    def _evaluate_bids(self, task_id: str) -> None:
        """Evaluate bids and select winner."""
        if task_id not in self.system.tasks:
            return
        
        task = self.system.tasks[task_id]
        bids = self.system.bids.get(task_id, [])
        
        if not bids:
            logger.warning(f"No bids received for task {task_id}")
            return
        
        # Select best bid
        best_bid = max(bids, key=lambda b: b.get_bid_score())
        
        # Award task to winning agent
        task.assigned_agent = best_bid.agent_id
        task.status = TaskStatus.ALLOCATED
        
        # Notify winning agent
        self.system.message_broker.send_message(
            sender_id="contract_net",
            receiver_id=best_bid.agent_id,
            message_type="task_awarded",
            content={
                'task': task.__dict__,
                'bid': best_bid.__dict__
            }
        )
        
        # Notify losing agents
        for bid in bids:
            if bid.agent_id != best_bid.agent_id:
                self.system.message_broker.send_message(
                    sender_id="contract_net",
                    receiver_id=bid.agent_id,
                    message_type="bid_rejected",
                    content={'task_id': task_id, 'bid_id': bid.bid_id}
                )
        
        # Clean up
        if task_id in self.active_announcements:
            del self.active_announcements[task_id]
        
        logger.info(f"Awarded task {task_id} to agent {best_bid.agent_id}")

class ConsensusEngine:
    """Engine for achieving consensus among agents."""
    
    def __init__(self, system: MultiAgentHealthcareSystem):
        self.system = system
        self.consensus_sessions: Dict[str, Dict[str, Any]] = {}
    
    def initiate_consensus(
        self,
        session_id: str,
        participants: List[str],
        decision_topic: str,
        options: List[Dict[str, Any]],
        timeout_minutes: int = 10
    ) -> None:
        """Initiate a consensus session."""
        session = {
            'session_id': session_id,
            'participants': participants,
            'decision_topic': decision_topic,
            'options': options,
            'votes': {},
            'status': 'active',
            'created_time': datetime.now(),
            'timeout': datetime.now() + timedelta(minutes=timeout_minutes)
        }
        
        self.consensus_sessions[session_id] = session
        
        # Send consensus request to participants
        for participant_id in participants:
            if participant_id in self.system.agents:
                self.system.message_broker.send_message(
                    sender_id="consensus_engine",
                    receiver_id=participant_id,
                    message_type="consensus_request",
                    content=session
                )
        
        # Schedule timeout
        threading.Timer(
            timeout_minutes * 60,
            self._handle_consensus_timeout,
            args=[session_id]
        ).start()
        
        logger.info(f"Initiated consensus session {session_id}")
    
    def receive_vote(
        self,
        session_id: str,
        agent_id: str,
        vote: Dict[str, Any]
    ) -> None:
        """Receive a vote from an agent."""
        if session_id not in self.consensus_sessions:
            logger.error(f"Consensus session {session_id} not found")
            return
        
        session = self.consensus_sessions[session_id]
        
        if agent_id not in session['participants']:
            logger.error(f"Agent {agent_id} not a participant in session {session_id}")
            return
        
        session['votes'][agent_id] = vote
        
        # Check if consensus is reached
        if len(session['votes']) == len(session['participants']):
            self._evaluate_consensus(session_id)
        
        logger.info(f"Received vote from {agent_id} for session {session_id}")
    
    def _evaluate_consensus(self, session_id: str) -> None:
        """Evaluate if consensus has been reached."""
        session = self.consensus_sessions[session_id]
        votes = session['votes']
        
        # Simple majority consensus
        vote_counts = defaultdict(int)
        for vote in votes.values():
            option_id = vote.get('option_id')
            if option_id:
                vote_counts[option_id] += 1
        
        if vote_counts:
            winning_option = max(vote_counts.items(), key=lambda x: x[1])
            consensus_threshold = len(session['participants']) * 0.6  # 60% majority
            
            if winning_option[1] >= consensus_threshold:
                session['status'] = 'consensus_reached'
                session['decision'] = winning_option[0]
                
                # Notify participants of consensus
                for participant_id in session['participants']:
                    self.system.message_broker.send_message(
                        sender_id="consensus_engine",
                        receiver_id=participant_id,
                        message_type="consensus_reached",
                        content={
                            'session_id': session_id,
                            'decision': winning_option[0],
                            'vote_counts': dict(vote_counts)
                        }
                    )
                
                logger.info(f"Consensus reached for session {session_id}: {winning_option[0]}")
            else:
                session['status'] = 'no_consensus'
                logger.info(f"No consensus reached for session {session_id}")
    
    def _handle_consensus_timeout(self, session_id: str) -> None:
        """Handle consensus timeout."""
        if session_id in self.consensus_sessions:
            session = self.consensus_sessions[session_id]
            if session['status'] == 'active':
                session['status'] = 'timeout'
                logger.warning(f"Consensus session {session_id} timed out")

class AuctionSystem:
    """Auction system for resource allocation."""
    
    def __init__(self, system: MultiAgentHealthcareSystem):
        self.system = system
        self.active_auctions: Dict[str, Dict[str, Any]] = {}
    
    def create_auction(
        self,
        task: Task,
        auction_type: str = "sealed_bid",
        duration_minutes: int = 5
    ) -> None:
        """Create an auction for a task."""
        auction_id = f"auction_{task.task_id}"
        
        auction = {
            'auction_id': auction_id,
            'task': task,
            'auction_type': auction_type,
            'bids': [],
            'status': 'active',
            'created_time': datetime.now(),
            'end_time': datetime.now() + timedelta(minutes=duration_minutes)
        }
        
        self.active_auctions[auction_id] = auction
        
        # Announce auction to suitable agents
        suitable_agents = self.system._find_suitable_agents(task)
        
        for agent in suitable_agents:
            self.system.message_broker.send_message(
                sender_id="auction_system",
                receiver_id=agent.agent_id,
                message_type="auction_announcement",
                content={
                    'auction': auction,
                    'bidding_deadline': auction['end_time'].isoformat()
                }
            )
        
        # Schedule auction end
        threading.Timer(
            duration_minutes * 60,
            self._end_auction,
            args=[auction_id]
        ).start()
        
        logger.info(f"Created auction {auction_id} for task {task.task_id}")
    
    def receive_auction_bid(self, auction_id: str, bid: Bid) -> None:
        """Receive a bid for an auction."""
        if auction_id not in self.active_auctions:
            logger.error(f"Auction {auction_id} not found")
            return
        
        auction = self.active_auctions[auction_id]
        
        if auction['status'] != 'active':
            logger.error(f"Auction {auction_id} is not active")
            return
        
        auction['bids'].append(bid)
        logger.info(f"Received auction bid {bid.bid_id} for auction {auction_id}")
    
    def _end_auction(self, auction_id: str) -> None:
        """End an auction and determine winner."""
        if auction_id not in self.active_auctions:
            return
        
        auction = self.active_auctions[auction_id]
        auction['status'] = 'ended'
        
        bids = auction['bids']
        
        if not bids:
            logger.warning(f"No bids received for auction {auction_id}")
            return
        
        # Determine winner based on auction type
        if auction['auction_type'] == 'sealed_bid':
            winner = max(bids, key=lambda b: b.get_bid_score())
        else:
            winner = min(bids, key=lambda b: b.cost)  # Lowest cost wins
        
        # Award task
        task = auction['task']
        task.assigned_agent = winner.agent_id
        task.status = TaskStatus.ALLOCATED
        
        # Notify winner
        self.system.message_broker.send_message(
            sender_id="auction_system",
            receiver_id=winner.agent_id,
            message_type="auction_won",
            content={
                'auction_id': auction_id,
                'task': task.__dict__,
                'winning_bid': winner.__dict__
            }
        )
        
        logger.info(f"Auction {auction_id} won by agent {winner.agent_id}")

class ResourceManager:
    """Manager for healthcare resources."""
    
    def __init__(self, system: MultiAgentHealthcareSystem):
        self.system = system
        self.allocation_history: List[Dict[str, Any]] = []
    
    def optimize_allocation(self) -> None:
        """Optimize resource allocation across the system."""
        # Identify resource bottlenecks
        bottlenecks = self._identify_bottlenecks()
        
        # Reallocate resources if needed
        for resource_type, utilization in bottlenecks.items():
            if utilization > 0.9:  # 90% utilization threshold
                self._reallocate_resource_type(resource_type)
    
    def _identify_bottlenecks(self) -> Dict[ResourceType, float]:
        """Identify resource bottlenecks."""
        utilization = {}
        
        for resource_type in ResourceType:
            type_resources = [
                r for r in self.system.resources.values()
                if r.resource_type == resource_type
            ]
            
            if type_resources:
                total_capacity = sum(r.capacity for r in type_resources)
                used_capacity = sum(
                    r.capacity - r.available_capacity
                    for r in type_resources
                )
                utilization[resource_type] = used_capacity / total_capacity
            else:
                utilization[resource_type] = 0.0
        
        return utilization
    
    def _reallocate_resource_type(self, resource_type: ResourceType) -> None:
        """Reallocate resources of a specific type."""
        # Simple reallocation strategy: find underutilized resources
        type_resources = [
            r for r in self.system.resources.values()
            if r.resource_type == resource_type
        ]
        
        # Sort by utilization
        type_resources.sort(
            key=lambda r: (r.capacity - r.available_capacity) / r.capacity
        )
        
        # Try to balance utilization
        if len(type_resources) >= 2:
            underutilized = type_resources[0]
            overutilized = type_resources[-1]
            
            if underutilized.available_capacity > 0 and overutilized.available_capacity == 0:
                # Transfer some capacity (simplified)
                logger.info(f"Rebalancing {resource_type.value} resources")

class MessageBroker:
    """Message broker for agent communication."""
    
    def __init__(self):
        self.agents: Dict[str, HealthcareAIAgent] = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.broker_thread = None
    
    def register_agent(self, agent: HealthcareAIAgent) -> None:
        """Register an agent with the message broker."""
        self.agents[agent.agent_id] = agent
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the message broker."""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        content: Dict[str, Any]
    ) -> None:
        """Send a message between agents."""
        message = {
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'message_type': message_type,
            'content': content,
            'timestamp': datetime.now()
        }
        
        self.message_queue.put(message)
    
    def start(self) -> None:
        """Start the message broker."""
        if self.running:
            return
        
        self.running = True
        self.broker_thread = threading.Thread(
            target=self._message_loop,
            daemon=True
        )
        self.broker_thread.start()
    
    def stop(self) -> None:
        """Stop the message broker."""
        self.running = False
        if self.broker_thread:
            self.broker_thread.join(timeout=5.0)
    
    def _message_loop(self) -> None:
        """Main message processing loop."""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1.0)
                self._deliver_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in message broker: {e}")
    
    def _deliver_message(self, message: Dict[str, Any]) -> None:
        """Deliver a message to the target agent."""
        receiver_id = message['receiver_id']
        
        if receiver_id in self.agents:
            agent = self.agents[receiver_id]
            
            # Convert to agent message format
            agent_message = Message(
                sender_id=message['sender_id'],
                receiver_id=receiver_id,
                message_type=message['message_type'],
                content=message['content'],
                priority=Priority.MEDIUM
            )
            
            agent.receive_message(agent_message)
        else:
            logger.warning(f"Agent {receiver_id} not found for message delivery")

# Example usage and demonstration
def main():
    """Demonstrate multi-agent healthcare system."""
    
    # Create multi-agent system
    mas = MultiAgentHealthcareSystem(
        system_id="hospital_mas_001",
        max_agents=50
    )
    
    # Create healthcare agents
    agents = []
    
    # Clinical decision support agents
    for i in range(3):
        agent = ClinicalDecisionSupportAgent(
            agent_id=f"cds_agent_{i:03d}",
            patient_ids=[f"patient_{j:03d}" for j in range(i*10, (i+1)*10)],
            clinical_guidelines={
                'diabetes': {'hba1c_target': 7.0},
                'hypertension': {'bp_target': '130/80'}
            },
            alert_thresholds={'high_risk': 0.8}
        )
        agents.append(agent)
        mas.add_agent(agent)
    
    # Care coordination agents
    for i in range(2):
        agent = CareCoordinationAgent(
            agent_id=f"coord_agent_{i:03d}",
            care_pathways={
                'diabetes_care': {
                    'stages': ['screening', 'diagnosis', 'treatment'],
                    'duration_weeks': 12
                }
            },
            provider_network={
                f'provider_{j:03d}': {
                    'provider_id': f'provider_{j:03d}',
                    'specialty': 'endocrinology',
                    'available': True
                } for j in range(5)
            }
        )
        agents.append(agent)
        mas.add_agent(agent)
    
    # Add healthcare resources
    resources = [
        Resource(
            resource_id="or_001",
            resource_type=ResourceType.OPERATING_ROOM,
            capacity=1,
            available_capacity=1,
            location="Floor 3",
            capabilities=["surgery", "anesthesia"],
            cost_per_hour=1000.0,
            maintenance_schedule=[]
        ),
        Resource(
            resource_id="icu_bed_001",
            resource_type=ResourceType.ICU_BED,
            capacity=1,
            available_capacity=1,
            location="ICU",
            capabilities=["critical_care", "monitoring"],
            cost_per_hour=500.0,
            maintenance_schedule=[]
        )
    ]
    
    for resource in resources:
        mas.add_resource(resource)
    
    # Start the system
    print("Starting multi-agent healthcare system...")
    mas.start_system()
    
    # Submit some tasks
    tasks = [
        Task(
            task_id="task_001",
            task_type="emergency",
            description="Emergency surgery required",
            requirements={"urgency": "high"},
            priority=1,
            deadline=datetime.now() + timedelta(hours=2),
            patient_id="patient_001",
            estimated_duration=timedelta(hours=3),
            required_capabilities=["surgery", "anesthesia"],
            resource_requirements={ResourceType.OPERATING_ROOM: 1}
        ),
        Task(
            task_id="task_002",
            task_type="routine",
            description="Diabetes management consultation",
            requirements={"specialty": "endocrinology"},
            priority=3,
            deadline=datetime.now() + timedelta(days=1),
            patient_id="patient_002",
            estimated_duration=timedelta(minutes=30),
            required_capabilities=["diabetes_management"],
            resource_requirements={ResourceType.SPECIALIST_TIME: 1}
        )
    ]
    
    for task in tasks:
        mas.submit_task(task)
    
    # Let the system run for a while
    time.sleep(10)
    
    # Check system status
    status = mas.get_system_status()
    
    print("\nSystem Status:")
    print("=" * 50)
    print(f"System ID: {status['system_id']}")
    print(f"Running: {status['running']}")
    print(f"Agents: {status['agents_count']}")
    print(f"Tasks: {status['tasks_count']}")
    print(f"Resources: {status['resources_count']}")
    
    print("\nSystem Metrics:")
    for metric, value in status['system_metrics'].items():
        print(f"  {metric}: {value}")
    
    print("\nTask Status Distribution:")
    for status_name, count in status['task_status_distribution'].items():
        print(f"  {status_name}: {count}")
    
    print("\nResource Utilization:")
    for resource_type, utilization in status['resource_utilization'].items():
        print(f"  {resource_type}: {utilization:.1%}")
    
    # Stop the system
    print("\nStopping multi-agent healthcare system...")
    mas.stop_system()
    
    print("Multi-agent healthcare system demonstration completed!")

if __name__ == "__main__":
    main()
```

## 7.4 Human-AI Collaboration in Clinical Practice

### 7.4.1 Collaborative Decision-Making Frameworks

The integration of AI agents into clinical practice requires sophisticated frameworks for human-AI collaboration that preserve clinical autonomy while leveraging AI capabilities. Effective collaboration frameworks must address trust, transparency, accountability, and the complementary strengths of human clinicians and AI systems.

**Shared Mental Models**: Successful human-AI collaboration requires the development of shared mental models where both human clinicians and AI agents have compatible understanding of clinical situations, goals, and constraints.

**Adaptive Automation**: The level of AI autonomy should adapt based on clinical context, urgency, and clinician expertise. In routine situations, AI agents may operate with high autonomy, while in complex or ambiguous cases, they should defer to human judgment.

**Explainable AI Integration**: AI agents must provide clear, clinically relevant explanations for their recommendations that align with clinical reasoning patterns familiar to healthcare providers.

### 7.4.2 Trust and Transparency Mechanisms

Building and maintaining trust between clinicians and AI agents requires ongoing transparency about AI capabilities, limitations, and decision-making processes. Trust calibration ensures that clinicians neither over-rely on nor under-utilize AI assistance.

**Confidence Intervals**: AI agents should communicate uncertainty through confidence intervals and probability distributions rather than point estimates.

**Evidence Presentation**: Recommendations should be accompanied by relevant evidence from clinical literature, guidelines, and patient-specific data.

**Performance Monitoring**: Continuous monitoring of AI agent performance with feedback to clinicians helps maintain appropriate trust levels.

## 7.5 Ethical Considerations and Governance

### 7.5.1 Autonomous Decision-Making Ethics

The deployment of autonomous AI agents in healthcare raises fundamental ethical questions about responsibility, accountability, and the appropriate scope of machine decision-making in patient care.

**Moral Agency**: While AI agents can make autonomous decisions, they lack moral agency. The ethical framework must clearly delineate when human oversight is required and how responsibility is distributed between human clinicians and AI systems.

**Beneficence and Non-maleficence**: AI agents must be designed with robust safeguards to ensure they act in patients' best interests and avoid harm, even in novel or unexpected situations.

**Patient Autonomy**: AI agents must respect patient preferences and values, incorporating patient-reported outcomes and preferences into their decision-making processes.

### 7.5.2 Governance Frameworks

Effective governance of healthcare AI agents requires multi-stakeholder frameworks that address technical, clinical, ethical, and regulatory considerations.

**Clinical Oversight Committees**: Multidisciplinary committees should oversee AI agent deployment, monitor performance, and address ethical concerns.

**Algorithmic Auditing**: Regular audits of AI agent behavior, decision patterns, and outcomes ensure continued alignment with clinical and ethical standards.

**Patient Rights and Consent**: Clear frameworks for patient consent to AI-assisted care and rights to human-only decision-making when requested.

## 7.6 Future Directions and Research Opportunities

### 7.6.1 Advanced Agent Architectures

Future healthcare AI agents will incorporate more sophisticated architectures including:

**Hierarchical Multi-Agent Systems**: Complex healthcare organizations may deploy hierarchical agent systems with specialized agents for different clinical domains coordinated by higher-level management agents.

**Federated Learning Agents**: Agents that can learn from distributed healthcare data while preserving privacy through federated learning approaches.

**Quantum-Enhanced Agents**: As quantum computing matures, healthcare AI agents may leverage quantum algorithms for optimization and machine learning tasks.

### 7.6.2 Integration with Emerging Technologies

Healthcare AI agents will increasingly integrate with emerging technologies:

**Internet of Medical Things (IoMT)**: Agents will coordinate with smart medical devices, wearables, and environmental sensors for comprehensive patient monitoring.

**Digital Twins**: Patient-specific digital twins will enable AI agents to simulate treatment outcomes and optimize personalized care plans.

**Blockchain Integration**: Blockchain technology may provide secure, auditable frameworks for agent decision-making and inter-agent communication.

## 7.7 Conclusion

AI agents represent a transformative technology for healthcare delivery, offering the potential for autonomous, intelligent systems that can enhance clinical decision-making, optimize resource allocation, and improve patient outcomes. The successful deployment of healthcare AI agents requires careful attention to technical architecture, clinical integration, ethical considerations, and governance frameworks.

The implementations presented in this chapter provide a foundation for developing production-ready healthcare AI agent systems. As the technology continues to evolve, ongoing research and development will be essential to realize the full potential of AI agents in transforming healthcare delivery while maintaining the highest standards of patient safety and clinical effectiveness.

The future of healthcare AI agents lies not in replacing human clinicians but in creating intelligent partnerships that leverage the complementary strengths of human expertise and artificial intelligence to deliver better, more efficient, and more equitable healthcare for all patients.

## References

1. Wooldridge, M. (2009). An introduction to multiagent systems. John Wiley & Sons. ISBN: 978-0470519462

2. Isern, D., & Moreno, A. (2016). A systematic literature review of agents applied in healthcare. Journal of Medical Systems, 40(2), 1-14. DOI: 10.1007/s10916-015-0376-2

3. Smith, R. G. (1980). The contract net protocol: High-level communication and control in a distributed problem solver. IEEE Transactions on Computers, 29(12), 1104-1113. DOI: 10.1109/TC.1980.1675516

4. Bates, D. W., et al. (2023). The potential of artificial intelligence to improve patient safety: a scoping review. NPJ Digital Medicine, 6(1), 1-11. DOI: 10.1038/s41746-023-00766-5

5. Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Pearson. ISBN: 978-0134610993

6. Shortliffe, E. H. (1976). Computer-based medical consultations: MYCIN. Elsevier. ISBN: 978-0444002181

7. Rao, A. S., & Georgeff, M. P. (1995). BDI agents: From theory to practice. Proceedings of the First International Conference on Multi-Agent Systems, 312-319.

8. Stone, P., & Veloso, M. (2000). Multiagent systems: A survey from a machine learning perspective. Autonomous Robots, 8(3), 345-383. DOI: 10.1023/A:1008942012299

9. Jennings, N. R. (2000). On agent-based software engineering. Artificial Intelligence, 117(2), 277-296. DOI: 10.1016/S0004-3702(99)00107-1

10. Weiss, G. (Ed.). (2013). Multiagent systems: a modern approach to distributed artificial intelligence. MIT Press. ISBN: 978-0262731317
