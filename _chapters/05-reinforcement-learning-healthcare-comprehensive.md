# Chapter 5: Reinforcement Learning for Healthcare Applications

## Learning Objectives

By the end of this chapter, readers will be able to:
- Implement production-ready reinforcement learning systems for clinical decision support
- Apply advanced RL algorithms to optimize treatment protocols and resource allocation
- Design safe and interpretable RL agents for healthcare environments
- Validate RL systems using appropriate clinical metrics and safety frameworks
- Deploy RL-based clinical decision support systems with regulatory compliance

## 5.1 Introduction to Healthcare Reinforcement Learning

Reinforcement Learning (RL) represents a paradigm shift in healthcare AI, moving beyond predictive modeling to active decision-making and treatment optimization. Unlike supervised learning approaches that predict outcomes based on historical data, RL systems learn optimal policies through interaction with clinical environments, continuously improving treatment strategies based on patient responses and outcomes.

The application of reinforcement learning in healthcare addresses fundamental challenges in clinical decision-making, including the optimization of treatment sequences, personalized therapy selection, and resource allocation under uncertainty. The dynamic nature of patient conditions, the complexity of treatment interactions, and the need for personalized care make healthcare an ideal domain for RL applications.

Recent breakthroughs in healthcare RL have demonstrated significant potential across multiple clinical domains. The work of Komorowski et al. (2018) on sepsis treatment optimization using deep reinforcement learning showed that RL-derived policies could potentially reduce mortality rates by suggesting optimal fluid and vasopressor administration strategies. Similarly, Yu et al. (2019) demonstrated the application of RL to glycemic control in intensive care units, achieving better glucose management than existing protocols.

The unique characteristics of healthcare environments present both opportunities and challenges for RL implementation. Healthcare environments are partially observable, as clinicians cannot directly observe all aspects of patient physiology. They are also non-stationary, as patient conditions evolve over time and treatment responses may change. Additionally, the high stakes of clinical decisions require RL systems to incorporate safety constraints and uncertainty quantification to ensure patient welfare.

## 5.2 Mathematical Foundations of Healthcare RL

### 5.2.1 Markov Decision Processes in Clinical Settings

Healthcare decision-making can be formalized as a Markov Decision Process (MDP), where clinical states, actions, and outcomes are modeled probabilistically. The clinical MDP is defined as a tuple (S, A, P, R, γ), where:

- **S** represents the state space of possible patient conditions
- **A** represents the action space of available treatments and interventions
- **P** represents the transition probabilities between states given actions
- **R** represents the reward function encoding clinical objectives
- **γ** represents the discount factor for future rewards

The state space in healthcare RL typically includes patient demographics, vital signs, laboratory values, medical history, and current treatments. The challenge lies in designing state representations that capture clinically relevant information while remaining computationally tractable.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import json

class ClinicalMDP:
    """
    Markov Decision Process formulation for clinical decision-making.
    
    This class provides a comprehensive framework for modeling clinical
    environments as MDPs, including state representation, action spaces,
    transition dynamics, and reward functions.
    
    References:
    - Komorowski, M., et al. (2018). The artificial intelligence clinician 
      learns optimal treatment strategies for sepsis in intensive care. 
      Nature Medicine, 24(11), 1716-1720.
    - Yu, C., et al. (2019). Reinforcement learning in healthcare: A survey. 
      ACM Computing Surveys, 52(5), 1-36.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 clinical_constraints: Dict = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clinical_constraints = clinical_constraints or {}
        
        # Initialize state and action spaces
        self.state_space = self._define_state_space()
        self.action_space = self._define_action_space()
        
        # Clinical safety constraints
        self.safety_constraints = self._define_safety_constraints()
        
        # Reward function components
        self.reward_components = self._define_reward_components()
        
    def _define_state_space(self) -> Dict:
        """Define comprehensive clinical state space."""
        
        state_space = {
            'demographics': {
                'age': {'type': 'continuous', 'range': (0, 120), 'units': 'years'},
                'gender': {'type': 'categorical', 'values': ['M', 'F', 'Other']},
                'weight': {'type': 'continuous', 'range': (1, 300), 'units': 'kg'},
                'height': {'type': 'continuous', 'range': (50, 250), 'units': 'cm'}
            },
            'vital_signs': {
                'heart_rate': {'type': 'continuous', 'range': (30, 200), 'units': 'bpm'},
                'systolic_bp': {'type': 'continuous', 'range': (50, 250), 'units': 'mmHg'},
                'diastolic_bp': {'type': 'continuous', 'range': (30, 150), 'units': 'mmHg'},
                'respiratory_rate': {'type': 'continuous', 'range': (5, 50), 'units': '/min'},
                'temperature': {'type': 'continuous', 'range': (90, 110), 'units': 'F'},
                'oxygen_saturation': {'type': 'continuous', 'range': (70, 100), 'units': '%'}
            },
            'laboratory': {
                'hemoglobin': {'type': 'continuous', 'range': (5, 20), 'units': 'g/dL'},
                'white_blood_cells': {'type': 'continuous', 'range': (1, 50), 'units': 'K/uL'},
                'platelets': {'type': 'continuous', 'range': (10, 1000), 'units': 'K/uL'},
                'sodium': {'type': 'continuous', 'range': (120, 160), 'units': 'mEq/L'},
                'potassium': {'type': 'continuous', 'range': (2, 8), 'units': 'mEq/L'},
                'creatinine': {'type': 'continuous', 'range': (0.3, 10), 'units': 'mg/dL'},
                'bun': {'type': 'continuous', 'range': (5, 100), 'units': 'mg/dL'},
                'lactate': {'type': 'continuous', 'range': (0.5, 20), 'units': 'mmol/L'}
            },
            'treatments': {
                'mechanical_ventilation': {'type': 'binary', 'values': [0, 1]},
                'vasopressors': {'type': 'continuous', 'range': (0, 1), 'units': 'normalized'},
                'fluid_balance': {'type': 'continuous', 'range': (-5000, 5000), 'units': 'mL'},
                'antibiotics': {'type': 'categorical', 'values': ['none', 'broad', 'targeted']},
                'sedation_level': {'type': 'ordinal', 'values': [0, 1, 2, 3, 4]}
            },
            'clinical_scores': {
                'sofa_score': {'type': 'ordinal', 'range': (0, 24), 'units': 'points'},
                'apache_score': {'type': 'ordinal', 'range': (0, 71), 'units': 'points'},
                'glasgow_coma_scale': {'type': 'ordinal', 'range': (3, 15), 'units': 'points'}
            }
        }
        
        return state_space
    
    def _define_action_space(self) -> Dict:
        """Define clinical action space for treatment decisions."""
        
        action_space = {
            'fluid_management': {
                'type': 'continuous',
                'range': (-2000, 3000),  # mL fluid balance change
                'units': 'mL',
                'description': 'Fluid administration or removal'
            },
            'vasopressor_dose': {
                'type': 'continuous',
                'range': (0, 1),  # Normalized dose
                'units': 'normalized',
                'description': 'Vasopressor dose adjustment'
            },
            'ventilator_settings': {
                'peep': {'type': 'continuous', 'range': (0, 20), 'units': 'cmH2O'},
                'fio2': {'type': 'continuous', 'range': (0.21, 1.0), 'units': 'fraction'},
                'tidal_volume': {'type': 'continuous', 'range': (200, 800), 'units': 'mL'}
            },
            'medication_adjustments': {
                'sedation_change': {'type': 'ordinal', 'range': (-2, 2), 'units': 'levels'},
                'antibiotic_escalation': {'type': 'binary', 'values': [0, 1]},
                'steroid_administration': {'type': 'binary', 'values': [0, 1]}
            },
            'monitoring_frequency': {
                'lab_frequency': {'type': 'ordinal', 'values': [6, 12, 24], 'units': 'hours'},
                'vital_frequency': {'type': 'ordinal', 'values': [15, 30, 60], 'units': 'minutes'}
            }
        }
        
        return action_space
    
    def _define_safety_constraints(self) -> Dict:
        """Define clinical safety constraints for actions."""
        
        constraints = {
            'fluid_limits': {
                'max_positive_balance': 3000,  # mL per day
                'max_negative_balance': -2000,  # mL per day
                'cumulative_limit': 10000  # mL total
            },
            'vasopressor_limits': {
                'max_dose_rate': 0.2,  # Maximum dose change per step
                'max_total_dose': 1.0,  # Maximum normalized dose
                'tapering_constraint': True  # Must taper gradually
            },
            'ventilator_limits': {
                'max_peep': 15,  # cmH2O for most patients
                'min_tidal_volume': 4,  # mL/kg predicted body weight
                'max_tidal_volume': 8,  # mL/kg predicted body weight
                'plateau_pressure_limit': 30  # cmH2O
            },
            'physiological_limits': {
                'map_threshold': 65,  # mmHg minimum mean arterial pressure
                'lactate_threshold': 4,  # mmol/L maximum acceptable lactate
                'urine_output_threshold': 0.5  # mL/kg/hr minimum
            }
        }
        
        return constraints
    
    def _define_reward_components(self) -> Dict:
        """Define multi-objective reward function components."""
        
        components = {
            'survival': {
                'weight': 1.0,
                'description': 'Primary outcome - patient survival',
                'calculation': 'binary_survival_outcome'
            },
            'length_of_stay': {
                'weight': -0.1,
                'description': 'Minimize ICU length of stay',
                'calculation': 'negative_los_penalty'
            },
            'organ_function': {
                'weight': 0.3,
                'description': 'Preserve organ function (SOFA score)',
                'calculation': 'sofa_improvement_reward'
            },
            'treatment_burden': {
                'weight': -0.05,
                'description': 'Minimize treatment intensity',
                'calculation': 'treatment_complexity_penalty'
            },
            'safety_violations': {
                'weight': -10.0,
                'description': 'Severe penalty for safety violations',
                'calculation': 'safety_constraint_penalty'
            },
            'physiological_stability': {
                'weight': 0.2,
                'description': 'Reward physiological stability',
                'calculation': 'stability_reward'
            }
        }
        
        return components
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, 
                        next_state: np.ndarray, done: bool, 
                        clinical_outcome: Dict) -> float:
        """
        Calculate comprehensive clinical reward.
        
        Args:
            state: Current patient state
            action: Action taken
            next_state: Resulting patient state
            done: Episode termination flag
            clinical_outcome: Clinical outcome information
            
        Returns:
            Calculated reward value
        """
        
        total_reward = 0.0
        reward_breakdown = {}
        
        # Survival reward (primary outcome)
        if done:
            survival_reward = 1.0 if clinical_outcome.get('survived', False) else -1.0
            total_reward += self.reward_components['survival']['weight'] * survival_reward
            reward_breakdown['survival'] = survival_reward
        
        # Length of stay penalty
        los_penalty = -0.01  # Small penalty per time step
        total_reward += self.reward_components['length_of_stay']['weight'] * los_penalty
        reward_breakdown['length_of_stay'] = los_penalty
        
        # Organ function improvement
        sofa_improvement = self._calculate_sofa_improvement(state, next_state)
        total_reward += self.reward_components['organ_function']['weight'] * sofa_improvement
        reward_breakdown['organ_function'] = sofa_improvement
        
        # Treatment burden penalty
        treatment_burden = self._calculate_treatment_burden(action)
        total_reward += self.reward_components['treatment_burden']['weight'] * treatment_burden
        reward_breakdown['treatment_burden'] = treatment_burden
        
        # Safety constraint penalties
        safety_penalty = self._calculate_safety_penalty(state, action, next_state)
        total_reward += self.reward_components['safety_violations']['weight'] * safety_penalty
        reward_breakdown['safety_violations'] = safety_penalty
        
        # Physiological stability reward
        stability_reward = self._calculate_stability_reward(state, next_state)
        total_reward += self.reward_components['physiological_stability']['weight'] * stability_reward
        reward_breakdown['physiological_stability'] = stability_reward
        
        return total_reward, reward_breakdown
    
    def _calculate_sofa_improvement(self, state: np.ndarray, 
                                   next_state: np.ndarray) -> float:
        """Calculate SOFA score improvement reward."""
        
        # Extract SOFA scores from state vectors
        # This is a simplified calculation - in practice, SOFA calculation
        # would be more complex and based on multiple physiological parameters
        
        current_sofa = self._extract_sofa_score(state)
        next_sofa = self._extract_sofa_score(next_state)
        
        # Reward improvement, penalize deterioration
        sofa_change = current_sofa - next_sofa  # Positive is improvement
        
        # Non-linear reward to emphasize significant improvements
        if sofa_change > 0:
            return np.tanh(sofa_change / 2.0)  # Improvement reward
        else:
            return -np.tanh(abs(sofa_change) / 2.0)  # Deterioration penalty
    
    def _extract_sofa_score(self, state: np.ndarray) -> float:
        """Extract or calculate SOFA score from state vector."""
        
        # Simplified SOFA calculation based on key parameters
        # In practice, this would use the full SOFA scoring algorithm
        
        # Assume state vector has specific indices for SOFA components
        # This is a placeholder implementation
        sofa_components = {
            'respiratory': min(4, max(0, (100 - state[5]) / 20)),  # Based on O2 sat
            'cardiovascular': min(4, max(0, (140 - state[1]) / 20)),  # Based on SBP
            'renal': min(4, max(0, (state[15] - 1.2) * 2)),  # Based on creatinine
            'hepatic': 0,  # Would need bilirubin
            'coagulation': min(4, max(0, (150 - state[12]) / 30)),  # Based on platelets
            'neurological': 0  # Would need GCS
        }
        
        return sum(sofa_components.values())
    
    def _calculate_treatment_burden(self, action: np.ndarray) -> float:
        """Calculate treatment burden penalty."""
        
        # Penalize intensive treatments
        burden = 0.0
        
        # Fluid administration burden
        fluid_burden = abs(action[0]) / 1000.0  # Normalize by 1L
        burden += fluid_burden
        
        # Vasopressor burden
        vasopressor_burden = action[1] ** 2  # Quadratic penalty for high doses
        burden += vasopressor_burden
        
        # Ventilator burden (if applicable)
        if len(action) > 2:
            vent_burden = (action[2] / 20.0) ** 2  # PEEP burden
            burden += vent_burden
        
        return -burden  # Negative because it's a penalty
    
    def _calculate_safety_penalty(self, state: np.ndarray, action: np.ndarray, 
                                 next_state: np.ndarray) -> float:
        """Calculate safety constraint violation penalties."""
        
        penalty = 0.0
        
        # Check physiological safety limits
        map_estimate = (next_state[1] + 2 * next_state[2]) / 3  # Rough MAP calculation
        if map_estimate < self.safety_constraints['physiological_limits']['map_threshold']:
            penalty += 1.0  # Severe penalty for hypotension
        
        # Check for extreme vital signs
        if next_state[0] > 150 or next_state[0] < 40:  # Heart rate
            penalty += 0.5
        
        if next_state[4] > 104 or next_state[4] < 95:  # Temperature
            penalty += 0.3
        
        # Check treatment safety limits
        if abs(action[0]) > self.safety_constraints['fluid_limits']['max_positive_balance']:
            penalty += 0.5  # Fluid overload risk
        
        if action[1] > self.safety_constraints['vasopressor_limits']['max_total_dose']:
            penalty += 0.8  # Excessive vasopressor dose
        
        return -penalty if penalty > 0 else 0.0
    
    def _calculate_stability_reward(self, state: np.ndarray, 
                                   next_state: np.ndarray) -> float:
        """Calculate physiological stability reward."""
        
        # Reward stable vital signs
        stability_metrics = []
        
        # Heart rate stability
        hr_change = abs(next_state[0] - state[0])
        hr_stability = np.exp(-hr_change / 10.0)  # Exponential decay with change
        stability_metrics.append(hr_stability)
        
        # Blood pressure stability
        sbp_change = abs(next_state[1] - state[1])
        bp_stability = np.exp(-sbp_change / 15.0)
        stability_metrics.append(bp_stability)
        
        # Temperature stability
        temp_change = abs(next_state[4] - state[4])
        temp_stability = np.exp(-temp_change / 2.0)
        stability_metrics.append(temp_stability)
        
        # Overall stability score
        overall_stability = np.mean(stability_metrics)
        
        return overall_stability - 0.5  # Center around 0
    
    def check_action_validity(self, state: np.ndarray, action: np.ndarray) -> Tuple[bool, List[str]]:
        """Check if action is clinically valid given current state."""
        
        violations = []
        
        # Check fluid limits
        if abs(action[0]) > self.safety_constraints['fluid_limits']['max_positive_balance']:
            violations.append(f"Fluid change {action[0]} exceeds safety limit")
        
        # Check vasopressor limits
        if action[1] > self.safety_constraints['vasopressor_limits']['max_total_dose']:
            violations.append(f"Vasopressor dose {action[1]} exceeds maximum")
        
        # Check for contraindications based on patient state
        # Example: High fluid administration with signs of fluid overload
        if action[0] > 1000 and state[0] > 120:  # High fluid + tachycardia
            violations.append("High fluid administration with tachycardia")
        
        # Check for drug interactions or contraindications
        if action[1] > 0.5 and state[15] > 3.0:  # High vasopressor + high creatinine
            violations.append("High vasopressor dose with renal dysfunction")
        
        return len(violations) == 0, violations
    
    def get_state_description(self, state: np.ndarray) -> Dict:
        """Get human-readable description of patient state."""
        
        description = {
            'vital_signs': {
                'heart_rate': f"{state[0]:.1f} bpm",
                'blood_pressure': f"{state[1]:.0f}/{state[2]:.0f} mmHg",
                'respiratory_rate': f"{state[3]:.1f} /min",
                'temperature': f"{state[4]:.1f} F",
                'oxygen_saturation': f"{state[5]:.1f} %"
            },
            'laboratory': {
                'hemoglobin': f"{state[6]:.1f} g/dL",
                'white_blood_cells': f"{state[7]:.1f} K/uL",
                'platelets': f"{state[8]:.0f} K/uL",
                'sodium': f"{state[9]:.1f} mEq/L",
                'potassium': f"{state[10]:.1f} mEq/L",
                'creatinine': f"{state[11]:.2f} mg/dL",
                'lactate': f"{state[12]:.1f} mmol/L"
            },
            'clinical_assessment': {
                'estimated_sofa': f"{self._extract_sofa_score(state):.1f}",
                'severity': self._assess_severity(state)
            }
        }
        
        return description
    
    def _assess_severity(self, state: np.ndarray) -> str:
        """Assess patient severity based on state."""
        
        sofa_score = self._extract_sofa_score(state)
        
        if sofa_score < 6:
            return "Mild"
        elif sofa_score < 12:
            return "Moderate"
        elif sofa_score < 18:
            return "Severe"
        else:
            return "Critical"

class ClinicalEnvironment:
    """
    Clinical environment for reinforcement learning training and evaluation.
    
    Simulates patient responses to treatments and provides realistic
    clinical scenarios for RL agent training.
    """
    
    def __init__(self, mdp: ClinicalMDP, patient_population: str = "icu_sepsis"):
        self.mdp = mdp
        self.patient_population = patient_population
        self.current_state = None
        self.episode_length = 0
        self.max_episode_length = 168  # 7 days in hours
        self.patient_trajectory = []
        
        # Load patient population parameters
        self.population_params = self._load_population_parameters()
        
        # Initialize transition model
        self.transition_model = ClinicalTransitionModel()
        
    def _load_population_parameters(self) -> Dict:
        """Load parameters for specific patient population."""
        
        if self.patient_population == "icu_sepsis":
            return {
                'age_distribution': {'mean': 65, 'std': 15, 'min': 18, 'max': 95},
                'severity_distribution': {'mild': 0.3, 'moderate': 0.4, 'severe': 0.3},
                'comorbidity_prevalence': {
                    'diabetes': 0.25,
                    'hypertension': 0.45,
                    'heart_disease': 0.35,
                    'kidney_disease': 0.20,
                    'liver_disease': 0.15
                },
                'mortality_risk_factors': {
                    'age_coefficient': 0.02,
                    'sofa_coefficient': 0.15,
                    'lactate_coefficient': 0.1
                }
            }
        else:
            # Default parameters
            return {
                'age_distribution': {'mean': 60, 'std': 20, 'min': 18, 'max': 90},
                'severity_distribution': {'mild': 0.4, 'moderate': 0.4, 'severe': 0.2}
            }
    
    def reset(self, patient_id: Optional[str] = None) -> np.ndarray:
        """Reset environment with new patient."""
        
        self.episode_length = 0
        self.patient_trajectory = []
        
        # Generate initial patient state
        self.current_state = self._generate_initial_state()
        
        # Store initial state
        self.patient_trajectory.append({
            'timestep': 0,
            'state': self.current_state.copy(),
            'action': None,
            'reward': 0,
            'clinical_info': self.mdp.get_state_description(self.current_state)
        })
        
        return self.current_state.copy()
    
    def _generate_initial_state(self) -> np.ndarray:
        """Generate realistic initial patient state."""
        
        # Sample patient demographics
        age = np.clip(np.random.normal(
            self.population_params['age_distribution']['mean'],
            self.population_params['age_distribution']['std']
        ), self.population_params['age_distribution']['min'],
           self.population_params['age_distribution']['max'])
        
        # Sample severity level
        severity_probs = list(self.population_params['severity_distribution'].values())
        severity_level = np.random.choice(['mild', 'moderate', 'severe'], p=severity_probs)
        
        # Generate vital signs based on severity
        if severity_level == 'mild':
            hr_base, hr_std = 85, 10
            sbp_base, sbp_std = 120, 15
            temp_base, temp_std = 99, 1
            lactate_base, lactate_std = 1.5, 0.5
        elif severity_level == 'moderate':
            hr_base, hr_std = 105, 15
            sbp_base, sbp_std = 100, 20
            temp_base, temp_std = 101, 2
            lactate_base, lactate_std = 2.5, 1.0
        else:  # severe
            hr_base, hr_std = 125, 20
            sbp_base, sbp_std = 85, 25
            temp_base, temp_std = 103, 2
            lactate_base, lactate_std = 4.0, 1.5
        
        # Generate state vector
        state = np.array([
            np.clip(np.random.normal(hr_base, hr_std), 60, 180),  # Heart rate
            np.clip(np.random.normal(sbp_base, sbp_std), 70, 200),  # Systolic BP
            np.clip(np.random.normal(sbp_base - 40, 10), 40, 120),  # Diastolic BP
            np.clip(np.random.normal(18, 4), 10, 35),  # Respiratory rate
            np.clip(np.random.normal(temp_base, temp_std), 96, 106),  # Temperature
            np.clip(np.random.normal(96, 3), 85, 100),  # O2 saturation
            np.clip(np.random.normal(12, 2), 8, 18),  # Hemoglobin
            np.clip(np.random.normal(12, 5), 4, 30),  # WBC
            np.clip(np.random.normal(250, 100), 50, 500),  # Platelets
            np.clip(np.random.normal(140, 5), 130, 150),  # Sodium
            np.clip(np.random.normal(4.0, 0.5), 3.0, 6.0),  # Potassium
            np.clip(np.random.normal(1.2, 0.5), 0.5, 4.0),  # Creatinine
            np.clip(np.random.normal(lactate_base, lactate_std), 0.5, 10),  # Lactate
            age,  # Age
            np.random.choice([0, 1]),  # Gender (0=F, 1=M)
            0,  # Current vasopressor dose
            0,  # Current fluid balance
            0   # Time since admission
        ])
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        
        # Validate action
        valid, violations = self.mdp.check_action_validity(self.current_state, action)
        
        if not valid:
            # Return penalty for invalid action
            reward = -5.0
            info = {'violations': violations, 'valid_action': False}
            return self.current_state.copy(), reward, False, info
        
        # Apply transition model
        next_state = self.transition_model.predict_next_state(
            self.current_state, action, self.episode_length
        )
        
        # Check for episode termination
        done, termination_reason = self._check_termination(next_state)
        
        # Calculate clinical outcome
        clinical_outcome = self._assess_clinical_outcome(next_state, done, termination_reason)
        
        # Calculate reward
        reward, reward_breakdown = self.mdp.calculate_reward(
            self.current_state, action, next_state, done, clinical_outcome
        )
        
        # Update state
        self.current_state = next_state
        self.episode_length += 1
        
        # Store trajectory
        self.patient_trajectory.append({
            'timestep': self.episode_length,
            'state': self.current_state.copy(),
            'action': action.copy(),
            'reward': reward,
            'reward_breakdown': reward_breakdown,
            'clinical_info': self.mdp.get_state_description(self.current_state),
            'done': done,
            'termination_reason': termination_reason if done else None
        })
        
        # Prepare info dictionary
        info = {
            'valid_action': True,
            'clinical_outcome': clinical_outcome,
            'reward_breakdown': reward_breakdown,
            'episode_length': self.episode_length,
            'patient_trajectory': self.patient_trajectory[-1]
        }
        
        return self.current_state.copy(), reward, done, info
    
    def _check_termination(self, state: np.ndarray) -> Tuple[bool, str]:
        """Check if episode should terminate."""
        
        # Maximum episode length reached
        if self.episode_length >= self.max_episode_length:
            return True, "max_length"
        
        # Patient death (simplified criteria)
        map_estimate = (state[1] + 2 * state[2]) / 3
        if map_estimate < 50:  # Severe hypotension
            return True, "cardiovascular_collapse"
        
        if state[12] > 8:  # Severe lactic acidosis
            return True, "metabolic_failure"
        
        if state[5] < 80:  # Severe hypoxemia
            return True, "respiratory_failure"
        
        # Patient recovery (simplified criteria)
        sofa_score = self.mdp._extract_sofa_score(state)
        if sofa_score < 2 and self.episode_length > 24:  # Stable for at least 24 hours
            return True, "recovery"
        
        return False, None
    
    def _assess_clinical_outcome(self, state: np.ndarray, done: bool, 
                                termination_reason: str) -> Dict:
        """Assess clinical outcome for reward calculation."""
        
        outcome = {
            'survived': True,
            'length_of_stay': self.episode_length,
            'final_sofa': self.mdp._extract_sofa_score(state),
            'termination_reason': termination_reason
        }
        
        if done:
            if termination_reason in ['cardiovascular_collapse', 'metabolic_failure', 'respiratory_failure']:
                outcome['survived'] = False
            elif termination_reason == 'recovery':
                outcome['survived'] = True
            elif termination_reason == 'max_length':
                # Assess survival probability based on final state
                survival_prob = self._calculate_survival_probability(state)
                outcome['survived'] = survival_prob > 0.5
                outcome['survival_probability'] = survival_prob
        
        return outcome
    
    def _calculate_survival_probability(self, state: np.ndarray) -> float:
        """Calculate survival probability based on current state."""
        
        # Simplified survival model based on key parameters
        age = state[13]
        sofa_score = self.mdp._extract_sofa_score(state)
        lactate = state[12]
        
        # Logistic regression-like model
        logit = (
            -0.02 * age +
            -0.15 * sofa_score +
            -0.1 * lactate +
            2.0  # Intercept
        )
        
        survival_prob = 1 / (1 + np.exp(-logit))
        return np.clip(survival_prob, 0.01, 0.99)
    
    def get_trajectory_summary(self) -> Dict:
        """Get summary of patient trajectory."""
        
        if not self.patient_trajectory:
            return {}
        
        rewards = [step['reward'] for step in self.patient_trajectory[1:]]  # Skip initial state
        
        summary = {
            'total_reward': sum(rewards),
            'episode_length': self.episode_length,
            'final_outcome': self.patient_trajectory[-1].get('clinical_info', {}),
            'reward_statistics': {
                'mean': np.mean(rewards) if rewards else 0,
                'std': np.std(rewards) if rewards else 0,
                'min': min(rewards) if rewards else 0,
                'max': max(rewards) if rewards else 0
            }
        }
        
        return summary

class ClinicalTransitionModel:
    """
    Clinical transition model for predicting patient state evolution.
    
    Models the complex dynamics of patient physiology in response
    to treatments and natural disease progression.
    """
    
    def __init__(self):
        self.physiological_models = self._initialize_physiological_models()
        self.treatment_effects = self._initialize_treatment_effects()
        self.noise_parameters = self._initialize_noise_parameters()
        
    def _initialize_physiological_models(self) -> Dict:
        """Initialize physiological response models."""
        
        models = {
            'cardiovascular': {
                'heart_rate_response': self._heart_rate_model,
                'blood_pressure_response': self._blood_pressure_model,
                'cardiac_output_response': self._cardiac_output_model
            },
            'respiratory': {
                'oxygen_saturation_response': self._oxygen_saturation_model,
                'respiratory_rate_response': self._respiratory_rate_model
            },
            'metabolic': {
                'lactate_clearance': self._lactate_clearance_model,
                'temperature_response': self._temperature_model
            },
            'renal': {
                'creatinine_evolution': self._creatinine_model,
                'fluid_balance_response': self._fluid_balance_model
            }
        }
        
        return models
    
    def _initialize_treatment_effects(self) -> Dict:
        """Initialize treatment effect parameters."""
        
        effects = {
            'fluid_administration': {
                'blood_pressure_effect': 0.02,  # mmHg per mL
                'heart_rate_effect': -0.001,    # bpm per mL
                'onset_time': 0.5,              # hours
                'duration': 4.0                 # hours
            },
            'vasopressor_administration': {
                'blood_pressure_effect': 50.0,  # mmHg per normalized dose
                'heart_rate_effect': 10.0,      # bpm per normalized dose
                'lactate_effect': 0.5,          # mmol/L per normalized dose
                'onset_time': 0.25,             # hours
                'duration': 2.0                 # hours
            },
            'mechanical_ventilation': {
                'oxygen_saturation_effect': 15.0,  # % improvement
                'respiratory_rate_effect': -5.0,   # breaths/min reduction
                'onset_time': 0.1,                 # hours
                'duration': 24.0                   # hours (while on ventilator)
            }
        }
        
        return effects
    
    def _initialize_noise_parameters(self) -> Dict:
        """Initialize physiological noise parameters."""
        
        noise = {
            'vital_signs': {
                'heart_rate': {'std': 5.0, 'autocorr': 0.8},
                'blood_pressure': {'std': 8.0, 'autocorr': 0.7},
                'respiratory_rate': {'std': 2.0, 'autocorr': 0.6},
                'temperature': {'std': 0.5, 'autocorr': 0.9},
                'oxygen_saturation': {'std': 2.0, 'autocorr': 0.8}
            },
            'laboratory': {
                'lactate': {'std': 0.3, 'autocorr': 0.9},
                'creatinine': {'std': 0.1, 'autocorr': 0.95},
                'hemoglobin': {'std': 0.5, 'autocorr': 0.98}
            }
        }
        
        return noise
    
    def predict_next_state(self, current_state: np.ndarray, action: np.ndarray, 
                          timestep: int) -> np.ndarray:
        """Predict next patient state given current state and action."""
        
        next_state = current_state.copy()
        
        # Apply treatment effects
        next_state = self._apply_fluid_effects(next_state, action[0])
        next_state = self._apply_vasopressor_effects(next_state, action[1])
        
        # Apply physiological evolution
        next_state = self._apply_cardiovascular_evolution(next_state, timestep)
        next_state = self._apply_respiratory_evolution(next_state, timestep)
        next_state = self._apply_metabolic_evolution(next_state, timestep)
        next_state = self._apply_renal_evolution(next_state, timestep)
        
        # Add physiological noise
        next_state = self._add_physiological_noise(next_state)
        
        # Ensure physiological constraints
        next_state = self._enforce_physiological_constraints(next_state)
        
        # Update time
        next_state[17] = timestep + 1  # Time since admission
        
        return next_state
    
    def _apply_fluid_effects(self, state: np.ndarray, fluid_change: float) -> np.ndarray:
        """Apply effects of fluid administration."""
        
        # Update fluid balance
        state[16] += fluid_change
        
        # Blood pressure response
        bp_effect = fluid_change * self.treatment_effects['fluid_administration']['blood_pressure_effect']
        state[1] += bp_effect * 0.7  # Systolic
        state[2] += bp_effect * 0.3  # Diastolic
        
        # Heart rate response (inverse relationship)
        hr_effect = fluid_change * self.treatment_effects['fluid_administration']['heart_rate_effect']
        state[0] += hr_effect
        
        # Hemoglobin dilution effect
        if fluid_change > 0:
            dilution_factor = 1 - (fluid_change / 10000)  # Dilution per 10L
            state[6] *= max(0.8, dilution_factor)
        
        return state
    
    def _apply_vasopressor_effects(self, state: np.ndarray, vasopressor_dose: float) -> np.ndarray:
        """Apply effects of vasopressor administration."""
        
        # Update current vasopressor dose
        state[15] = vasopressor_dose
        
        # Blood pressure response
        bp_effect = vasopressor_dose * self.treatment_effects['vasopressor_administration']['blood_pressure_effect']
        state[1] += bp_effect * 0.8  # Systolic
        state[2] += bp_effect * 0.6  # Diastolic
        
        # Heart rate response
        hr_effect = vasopressor_dose * self.treatment_effects['vasopressor_administration']['heart_rate_effect']
        state[0] += hr_effect
        
        # Lactate response (vasopressors can increase lactate)
        lactate_effect = vasopressor_dose * self.treatment_effects['vasopressor_administration']['lactate_effect']
        state[12] += lactate_effect
        
        return state
    
    def _apply_cardiovascular_evolution(self, state: np.ndarray, timestep: int) -> np.ndarray:
        """Apply cardiovascular system evolution."""
        
        # Heart rate evolution
        state[0] = self._heart_rate_model(state, timestep)
        
        # Blood pressure evolution
        state[1], state[2] = self._blood_pressure_model(state, timestep)
        
        return state
    
    def _heart_rate_model(self, state: np.ndarray, timestep: int) -> float:
        """Model heart rate evolution."""
        
        current_hr = state[0]
        
        # Baseline heart rate based on patient condition
        sofa_score = self._calculate_sofa_from_state(state)
        baseline_hr = 80 + sofa_score * 5  # Higher SOFA -> higher baseline HR
        
        # Trend toward baseline with some persistence
        hr_change = (baseline_hr - current_hr) * 0.1
        
        # Add fever effect
        fever_effect = max(0, state[4] - 98.6) * 2  # 2 bpm per degree F
        
        new_hr = current_hr + hr_change + fever_effect
        
        return np.clip(new_hr, 40, 200)
    
    def _blood_pressure_model(self, state: np.ndarray, timestep: int) -> Tuple[float, float]:
        """Model blood pressure evolution."""
        
        current_sbp, current_dbp = state[1], state[2]
        
        # Baseline blood pressure
        age_factor = (state[13] - 40) * 0.5  # Age effect
        baseline_sbp = 120 + age_factor
        baseline_dbp = 80 + age_factor * 0.3
        
        # Sepsis effect (hypotension)
        lactate_effect = state[12] * -5  # Higher lactate -> lower BP
        
        # Trend toward baseline
        sbp_change = (baseline_sbp - current_sbp) * 0.05 + lactate_effect
        dbp_change = (baseline_dbp - current_dbp) * 0.05 + lactate_effect * 0.5
        
        new_sbp = current_sbp + sbp_change
        new_dbp = current_dbp + dbp_change
        
        # Ensure DBP < SBP
        new_dbp = min(new_dbp, new_sbp - 20)
        
        return np.clip(new_sbp, 60, 250), np.clip(new_dbp, 30, 150)
    
    def _apply_respiratory_evolution(self, state: np.ndarray, timestep: int) -> np.ndarray:
        """Apply respiratory system evolution."""
        
        # Oxygen saturation evolution
        state[5] = self._oxygen_saturation_model(state, timestep)
        
        # Respiratory rate evolution
        state[3] = self._respiratory_rate_model(state, timestep)
        
        return state
    
    def _oxygen_saturation_model(self, state: np.ndarray, timestep: int) -> float:
        """Model oxygen saturation evolution."""
        
        current_spo2 = state[5]
        
        # Baseline based on respiratory status
        baseline_spo2 = 98
        
        # Sepsis effect
        sepsis_effect = -state[12] * 2  # Higher lactate -> lower SpO2
        
        # Trend toward baseline
        spo2_change = (baseline_spo2 - current_spo2) * 0.1 + sepsis_effect
        
        new_spo2 = current_spo2 + spo2_change
        
        return np.clip(new_spo2, 70, 100)
    
    def _respiratory_rate_model(self, state: np.ndarray, timestep: int) -> float:
        """Model respiratory rate evolution."""
        
        current_rr = state[3]
        
        # Baseline respiratory rate
        baseline_rr = 16
        
        # Metabolic acidosis effect (higher lactate -> higher RR)
        acidosis_effect = state[12] * 2
        
        # Fever effect
        fever_effect = max(0, state[4] - 98.6) * 1
        
        # Trend toward baseline with adjustments
        rr_change = (baseline_rr - current_rr) * 0.1 + acidosis_effect + fever_effect
        
        new_rr = current_rr + rr_change
        
        return np.clip(new_rr, 8, 40)
    
    def _apply_metabolic_evolution(self, state: np.ndarray, timestep: int) -> np.ndarray:
        """Apply metabolic system evolution."""
        
        # Lactate clearance
        state[12] = self._lactate_clearance_model(state, timestep)
        
        # Temperature evolution
        state[4] = self._temperature_model(state, timestep)
        
        return state
    
    def _lactate_clearance_model(self, state: np.ndarray, timestep: int) -> float:
        """Model lactate clearance."""
        
        current_lactate = state[12]
        
        # Clearance rate depends on organ function
        liver_function = max(0.1, 1 - state[12] * 0.1)  # Simplified liver function
        kidney_function = max(0.1, 2.0 / state[11])     # Based on creatinine
        
        clearance_rate = 0.1 * liver_function * kidney_function
        
        # Natural lactate production
        production_rate = 0.5
        
        # Net lactate change
        lactate_change = production_rate - current_lactate * clearance_rate
        
        new_lactate = current_lactate + lactate_change
        
        return np.clip(new_lactate, 0.5, 20)
    
    def _temperature_model(self, state: np.ndarray, timestep: int) -> float:
        """Model temperature evolution."""
        
        current_temp = state[4]
        
        # Baseline temperature
        baseline_temp = 98.6
        
        # Infection effect (simplified)
        infection_severity = state[12] / 4.0  # Based on lactate
        fever_effect = infection_severity * 2
        
        # Trend toward baseline with fever
        temp_change = (baseline_temp - current_temp) * 0.2 + fever_effect
        
        new_temp = current_temp + temp_change
        
        return np.clip(new_temp, 95, 108)
    
    def _apply_renal_evolution(self, state: np.ndarray, timestep: int) -> np.ndarray:
        """Apply renal system evolution."""
        
        # Creatinine evolution
        state[11] = self._creatinine_model(state, timestep)
        
        return state
    
    def _creatinine_model(self, state: np.ndarray, timestep: int) -> float:
        """Model creatinine evolution."""
        
        current_creatinine = state[11]
        
        # Baseline creatinine based on age and gender
        age_factor = (state[13] - 40) * 0.005
        gender_factor = 0.2 if state[14] == 0 else 0  # Lower for females
        baseline_creatinine = 1.0 + age_factor - gender_factor
        
        # Acute kidney injury progression
        hypotension_effect = max(0, 90 - (state[1] + 2 * state[2]) / 3) * 0.01
        vasopressor_effect = state[15] * 0.1  # Vasopressors can worsen kidney function
        
        # Net creatinine change
        creatinine_change = (baseline_creatinine - current_creatinine) * 0.05 + hypotension_effect + vasopressor_effect
        
        new_creatinine = current_creatinine + creatinine_change
        
        return np.clip(new_creatinine, 0.3, 10)
    
    def _add_physiological_noise(self, state: np.ndarray) -> np.ndarray:
        """Add realistic physiological noise to state."""
        
        # Add noise to vital signs
        state[0] += np.random.normal(0, self.noise_parameters['vital_signs']['heart_rate']['std'])
        state[1] += np.random.normal(0, self.noise_parameters['vital_signs']['blood_pressure']['std'])
        state[2] += np.random.normal(0, self.noise_parameters['vital_signs']['blood_pressure']['std'] * 0.6)
        state[3] += np.random.normal(0, self.noise_parameters['vital_signs']['respiratory_rate']['std'])
        state[4] += np.random.normal(0, self.noise_parameters['vital_signs']['temperature']['std'])
        state[5] += np.random.normal(0, self.noise_parameters['vital_signs']['oxygen_saturation']['std'])
        
        # Add noise to laboratory values
        state[12] += np.random.normal(0, self.noise_parameters['laboratory']['lactate']['std'])
        state[11] += np.random.normal(0, self.noise_parameters['laboratory']['creatinine']['std'])
        state[6] += np.random.normal(0, self.noise_parameters['laboratory']['hemoglobin']['std'])
        
        return state
    
    def _enforce_physiological_constraints(self, state: np.ndarray) -> np.ndarray:
        """Enforce physiological constraints on state values."""
        
        # Vital signs constraints
        state[0] = np.clip(state[0], 30, 200)    # Heart rate
        state[1] = np.clip(state[1], 50, 250)    # Systolic BP
        state[2] = np.clip(state[2], 30, 150)    # Diastolic BP
        state[3] = np.clip(state[3], 5, 50)      # Respiratory rate
        state[4] = np.clip(state[4], 90, 110)    # Temperature
        state[5] = np.clip(state[5], 70, 100)    # Oxygen saturation
        
        # Laboratory constraints
        state[6] = np.clip(state[6], 5, 20)      # Hemoglobin
        state[7] = np.clip(state[7], 1, 50)      # WBC
        state[8] = np.clip(state[8], 10, 1000)   # Platelets
        state[9] = np.clip(state[9], 120, 160)   # Sodium
        state[10] = np.clip(state[10], 2, 8)     # Potassium
        state[11] = np.clip(state[11], 0.3, 10)  # Creatinine
        state[12] = np.clip(state[12], 0.5, 20)  # Lactate
        
        # Ensure DBP < SBP
        if state[2] >= state[1]:
            state[2] = state[1] - 20
        
        return state
    
    def _calculate_sofa_from_state(self, state: np.ndarray) -> float:
        """Calculate SOFA score from state vector."""
        
        # Simplified SOFA calculation
        sofa_components = {
            'respiratory': min(4, max(0, (100 - state[5]) / 20)),
            'cardiovascular': min(4, max(0, (140 - state[1]) / 20)),
            'renal': min(4, max(0, (state[11] - 1.2) * 2)),
            'coagulation': min(4, max(0, (150 - state[8]) / 30))
        }
        
        return sum(sofa_components.values())
```

### 5.2.2 Deep Q-Network Implementation for Clinical Decision Support

Deep Q-Networks (DQN) provide a powerful framework for learning optimal clinical policies from high-dimensional state spaces. The following implementation demonstrates a comprehensive DQN system designed specifically for healthcare applications:

```python
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import copy

class ClinicalDQN(nn.Module):
    """
    Deep Q-Network for clinical decision-making.
    
    Implements a sophisticated neural network architecture optimized
    for healthcare applications with clinical interpretability features.
    
    References:
    - Mnih, V., et al. (2015). Human-level control through deep reinforcement 
      learning. Nature, 518(7540), 529-533.
    - Gottesman, O., et al. (2019). Guidelines for reinforcement learning in 
      healthcare. Nature Medicine, 25(1), 16-18.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256, 128],
                 dropout_rate: float = 0.1, use_batch_norm: bool = True):
        super(ClinicalDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Clinical attention mechanism for interpretability
        self.attention = ClinicalAttentionLayer(state_dim, hidden_dims[0])
        
        # Uncertainty estimation layers
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state: torch.Tensor, return_attention: bool = False, 
                return_uncertainty: bool = False):
        """
        Forward pass with optional attention and uncertainty outputs.
        
        Args:
            state: Input state tensor
            return_attention: Whether to return attention weights
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Q-values and optional attention/uncertainty information
        """
        
        batch_size = state.size(0)
        
        # Apply attention mechanism
        attended_state, attention_weights = self.attention(state)
        
        # Forward pass through main network
        x = attended_state
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i == len(self.network) - 3:  # Before final layer
                features = x  # Save for uncertainty estimation
        
        q_values = x
        
        # Uncertainty estimation
        if return_uncertainty:
            uncertainty = torch.exp(self.uncertainty_head(features))  # Ensure positive
        
        # Prepare outputs
        outputs = {'q_values': q_values}
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        if return_uncertainty:
            outputs['uncertainty'] = uncertainty
        
        return outputs if (return_attention or return_uncertainty) else q_values

class ClinicalAttentionLayer(nn.Module):
    """
    Attention mechanism for clinical state interpretation.
    
    Provides interpretable attention weights over clinical features
    to understand which aspects of patient state drive decisions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ClinicalAttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention computation layers
        self.attention_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # Feature transformation
        self.feature_transform = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism to clinical state.
        
        Args:
            state: Input clinical state tensor
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        
        # Compute attention weights
        attention_weights = self.attention_fc(state)
        
        # Apply attention to input features
        attended_state = state * attention_weights
        
        # Transform attended features
        attended_features = self.feature_transform(attended_state)
        
        return attended_features, attention_weights

class ClinicalDQNAgent:
    """
    Complete DQN agent for clinical decision-making with safety constraints.
    
    Implements Double DQN, Dueling DQN, and Prioritized Experience Replay
    optimized for healthcare applications.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 1e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000, memory_size: int = 100000,
                 batch_size: int = 32, target_update_freq: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Initialize networks
        self.q_network = ClinicalDQN(state_dim, action_dim).to(device)
        self.target_network = ClinicalDQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        
        # Clinical safety constraints
        self.safety_checker = ClinicalSafetyChecker()
        
        # Logging
        self.logger = logging.getLogger('ClinicalDQNAgent')
        self.writer = SummaryWriter('runs/clinical_dqn')
        
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None,
                     safe_actions_only: bool = True) -> Tuple[int, Dict]:
        """
        Select action using epsilon-greedy policy with safety constraints.
        
        Args:
            state: Current patient state
            epsilon: Exploration rate (if None, uses current training epsilon)
            safe_actions_only: Whether to restrict to clinically safe actions
            
        Returns:
            Tuple of (action_index, action_info)
        """
        
        if epsilon is None:
            epsilon = self._get_current_epsilon()
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values and additional information
        with torch.no_grad():
            outputs = self.q_network(state_tensor, return_attention=True, 
                                   return_uncertainty=True)
            q_values = outputs['q_values']
            attention_weights = outputs['attention_weights']
            uncertainty = outputs['uncertainty']
        
        # Get valid actions based on safety constraints
        if safe_actions_only:
            valid_actions = self.safety_checker.get_valid_actions(state)
        else:
            valid_actions = list(range(self.action_dim))
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Random action from valid actions
            action = np.random.choice(valid_actions)
            selection_method = 'random'
        else:
            # Greedy action from valid actions
            valid_q_values = q_values[0][valid_actions]
            best_valid_idx = torch.argmax(valid_q_values).item()
            action = valid_actions[best_valid_idx]
            selection_method = 'greedy'
        
        # Prepare action information
        action_info = {
            'q_values': q_values[0].cpu().numpy(),
            'selected_q_value': q_values[0][action].item(),
            'attention_weights': attention_weights[0].cpu().numpy(),
            'uncertainty': uncertainty[0][action].item(),
            'valid_actions': valid_actions,
            'selection_method': selection_method,
            'epsilon': epsilon
        }
        
        return action, action_info
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, 
                        clinical_priority: float = 1.0):
        """
        Store experience in replay buffer with clinical priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            clinical_priority: Clinical importance of this experience
        """
        
        # Calculate TD error for prioritization
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            current_q = self.q_network(state_tensor)[0][action]
            
            if done:
                target_q = reward
            else:
                next_q = self.target_network(next_state_tensor).max(1)[0]
                target_q = reward + self.gamma * next_q
            
            td_error = abs(current_q - target_q).item()
        
        # Adjust priority based on clinical importance
        priority = td_error * clinical_priority
        
        self.memory.push(state, action, reward, next_state, done, priority)
    
    def train(self) -> Dict:
        """
        Train the DQN agent using prioritized experience replay.
        
        Returns:
            Dictionary of training metrics
        """
        
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch from memory
        batch, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)
        dones = torch.BoolTensor(batch.done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values using Double DQN
        with torch.no_grad():
            # Use main network to select actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Calculate loss with importance sampling weights
        td_errors = current_q_values - target_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update priorities in replay buffer
        new_priorities = td_errors.abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, new_priorities)
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.training_step += 1
        
        # Log training metrics
        metrics = {
            'loss': loss.item(),
            'mean_q_value': current_q_values.mean().item(),
            'mean_target_q': target_q_values.mean().item(),
            'epsilon': self._get_current_epsilon(),
            'training_step': self.training_step
        }
        
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f'training/{key}', value, self.training_step)
        
        self.losses.append(loss.item())
        
        return metrics
    
    def _get_current_epsilon(self) -> float:
        """Get current epsilon value for exploration."""
        
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 np.exp(-self.training_step / self.epsilon_decay)
        return epsilon
    
    def evaluate_policy(self, env: ClinicalEnvironment, num_episodes: int = 10,
                       render: bool = False) -> Dict:
        """
        Evaluate the current policy on the environment.
        
        Args:
            env: Clinical environment
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Evaluation metrics
        """
        
        self.q_network.eval()
        
        episode_rewards = []
        episode_lengths = []
        survival_rates = []
        clinical_outcomes = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Select action without exploration
                action, action_info = self.select_action(state, epsilon=0.0)
                
                # Take action
                next_state, reward, done, info = env.step(np.array([action]))
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if render:
                    self._render_step(state, action, reward, action_info, info)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Extract clinical outcomes
            clinical_outcome = info.get('clinical_outcome', {})
            survival_rates.append(1.0 if clinical_outcome.get('survived', False) else 0.0)
            clinical_outcomes.append(clinical_outcome)
        
        self.q_network.train()
        
        # Calculate evaluation metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'survival_rate': np.mean(survival_rates),
            'episodes_evaluated': num_episodes
        }
        
        # Log evaluation metrics
        for key, value in metrics.items():
            self.writer.add_scalar(f'evaluation/{key}', value, self.training_step)
        
        return metrics
    
    def _render_step(self, state: np.ndarray, action: int, reward: float,
                    action_info: Dict, env_info: Dict):
        """Render a single step for visualization."""
        
        print(f"Step {env_info.get('episode_length', 0)}:")
        print(f"  Action: {action} (Q-value: {action_info['selected_q_value']:.3f})")
        print(f"  Reward: {reward:.3f}")
        print(f"  Uncertainty: {action_info['uncertainty']:.3f}")
        
        # Show top attention weights
        attention = action_info['attention_weights']
        top_indices = np.argsort(attention)[-3:]
        print(f"  Top attention: {top_indices} ({attention[top_indices]})")
        print()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        
        self.logger.info(f"Model loaded from {filepath}")

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for clinical RL.
    
    Implements prioritized sampling based on TD errors and clinical importance.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Named tuple for experiences
        self.Experience = namedtuple('Experience', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, priority: float):
        """Add experience to buffer with priority."""
        
        max_priority = self.priorities.max() if self.buffer else 1.0
        priority = max(priority, max_priority)
        
        experience = self.Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[namedtuple, np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        
        if self.size == 0:
            raise ValueError("Buffer is empty")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        batch = self.Experience(*zip(*experiences))
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size

class ClinicalSafetyChecker:
    """
    Safety constraint checker for clinical RL actions.
    
    Ensures that RL agent actions comply with clinical safety guidelines
    and physiological constraints.
    """
    
    def __init__(self):
        self.safety_rules = self._initialize_safety_rules()
        self.physiological_limits = self._initialize_physiological_limits()
        
    def _initialize_safety_rules(self) -> Dict:
        """Initialize clinical safety rules."""
        
        rules = {
            'fluid_management': {
                'max_positive_balance_per_hour': 500,  # mL
                'max_negative_balance_per_hour': -300,  # mL
                'contraindications': {
                    'pulmonary_edema': 'no_positive_fluid',
                    'heart_failure': 'limited_fluid',
                    'renal_failure': 'careful_fluid_management'
                }
            },
            'vasopressor_management': {
                'max_dose_increase_per_hour': 0.1,  # Normalized units
                'max_total_dose': 1.0,
                'contraindications': {
                    'severe_arrhythmia': 'avoid_high_dose',
                    'peripheral_ischemia': 'consider_alternatives'
                }
            },
            'general_safety': {
                'minimum_map': 65,  # mmHg
                'maximum_heart_rate': 150,  # bpm
                'minimum_oxygen_saturation': 88,  # %
                'maximum_lactate': 8.0  # mmol/L before intervention
            }
        }
        
        return rules
    
    def _initialize_physiological_limits(self) -> Dict:
        """Initialize physiological safety limits."""
        
        limits = {
            'vital_signs': {
                'heart_rate': {'min': 40, 'max': 180, 'critical_low': 30, 'critical_high': 200},
                'systolic_bp': {'min': 80, 'max': 180, 'critical_low': 60, 'critical_high': 220},
                'diastolic_bp': {'min': 50, 'max': 110, 'critical_low': 30, 'critical_high': 130},
                'temperature': {'min': 96, 'max': 102, 'critical_low': 94, 'critical_high': 106},
                'oxygen_saturation': {'min': 88, 'max': 100, 'critical_low': 80, 'critical_high': 100}
            },
            'laboratory': {
                'lactate': {'max': 4.0, 'critical': 8.0},
                'creatinine': {'max': 2.0, 'critical': 5.0},
                'potassium': {'min': 3.0, 'max': 5.5, 'critical_low': 2.5, 'critical_high': 6.5}
            }
        }
        
        return limits
    
    def get_valid_actions(self, state: np.ndarray) -> List[int]:
        """
        Get list of valid (safe) actions given current patient state.
        
        Args:
            state: Current patient state vector
            
        Returns:
            List of valid action indices
        """
        
        valid_actions = []
        
        # Check each possible action
        for action_idx in range(10):  # Assuming 10 discrete actions
            if self.is_action_safe(state, action_idx):
                valid_actions.append(action_idx)
        
        # Ensure at least one action is always available (conservative action)
        if not valid_actions:
            valid_actions = [0]  # Conservative "no change" action
        
        return valid_actions
    
    def is_action_safe(self, state: np.ndarray, action: int) -> bool:
        """
        Check if a specific action is safe given current state.
        
        Args:
            state: Current patient state
            action: Action index to check
            
        Returns:
            Boolean indicating if action is safe
        """
        
        # Convert action index to action parameters
        action_params = self._decode_action(action)
        
        # Check fluid management safety
        if not self._check_fluid_safety(state, action_params.get('fluid_change', 0)):
            return False
        
        # Check vasopressor safety
        if not self._check_vasopressor_safety(state, action_params.get('vasopressor_change', 0)):
            return False
        
        # Check physiological constraints
        if not self._check_physiological_safety(state):
            return False
        
        return True
    
    def _decode_action(self, action_idx: int) -> Dict:
        """Decode action index to action parameters."""
        
        # Simplified action decoding
        # In practice, this would map to specific clinical actions
        
        action_map = {
            0: {'fluid_change': 0, 'vasopressor_change': 0},      # No change
            1: {'fluid_change': 250, 'vasopressor_change': 0},    # Small fluid bolus
            2: {'fluid_change': 500, 'vasopressor_change': 0},    # Large fluid bolus
            3: {'fluid_change': -200, 'vasopressor_change': 0},   # Fluid removal
            4: {'fluid_change': 0, 'vasopressor_change': 0.1},    # Increase vasopressor
            5: {'fluid_change': 0, 'vasopressor_change': -0.1},   # Decrease vasopressor
            6: {'fluid_change': 250, 'vasopressor_change': 0.05}, # Fluid + vasopressor
            7: {'fluid_change': -100, 'vasopressor_change': 0.05}, # Conservative management
            8: {'fluid_change': 100, 'vasopressor_change': -0.05}, # Fluid sparing
            9: {'fluid_change': 0, 'vasopressor_change': 0.2}     # High vasopressor
        }
        
        return action_map.get(action_idx, action_map[0])
    
    def _check_fluid_safety(self, state: np.ndarray, fluid_change: float) -> bool:
        """Check fluid management safety."""
        
        current_fluid_balance = state[16]  # Assuming index 16 is fluid balance
        
        # Check maximum fluid limits
        if fluid_change > self.safety_rules['fluid_management']['max_positive_balance_per_hour']:
            return False
        
        if fluid_change < self.safety_rules['fluid_management']['max_negative_balance_per_hour']:
            return False
        
        # Check for signs of fluid overload
        if fluid_change > 0:
            # Check for tachycardia (possible fluid overload)
            if state[0] > 120:  # Heart rate > 120
                return False
            
            # Check for low oxygen saturation (possible pulmonary edema)
            if state[5] < 92:  # O2 sat < 92%
                return False
        
        return True
    
    def _check_vasopressor_safety(self, state: np.ndarray, vasopressor_change: float) -> bool:
        """Check vasopressor management safety."""
        
        current_vasopressor = state[15]  # Assuming index 15 is current vasopressor dose
        new_dose = current_vasopressor + vasopressor_change
        
        # Check maximum dose limits
        if new_dose > self.safety_rules['vasopressor_management']['max_total_dose']:
            return False
        
        if new_dose < 0:
            return False
        
        # Check rate of change
        if abs(vasopressor_change) > self.safety_rules['vasopressor_management']['max_dose_increase_per_hour']:
            return False
        
        # Check for contraindications
        if vasopressor_change > 0:
            # Avoid increasing vasopressors with severe tachycardia
            if state[0] > 140:  # Heart rate > 140
                return False
            
            # Avoid with severe renal dysfunction
            if state[11] > 3.0:  # Creatinine > 3.0
                return False
        
        return True
    
    def _check_physiological_safety(self, state: np.ndarray) -> bool:
        """Check general physiological safety constraints."""
        
        # Check vital signs
        hr = state[0]
        sbp = state[1]
        dbp = state[2]
        temp = state[4]
        spo2 = state[5]
        
        # Critical vital sign limits
        if hr < self.physiological_limits['vital_signs']['heart_rate']['critical_low'] or \
           hr > self.physiological_limits['vital_signs']['heart_rate']['critical_high']:
            return False
        
        if sbp < self.physiological_limits['vital_signs']['systolic_bp']['critical_low'] or \
           sbp > self.physiological_limits['vital_signs']['systolic_bp']['critical_high']:
            return False
        
        if spo2 < self.physiological_limits['vital_signs']['oxygen_saturation']['critical_low']:
            return False
        
        # Check laboratory values
        lactate = state[12]
        creatinine = state[11]
        
        if lactate > self.physiological_limits['laboratory']['lactate']['critical']:
            return False
        
        if creatinine > self.physiological_limits['laboratory']['creatinine']['critical']:
            return False
        
        return True
    
    def get_safety_violations(self, state: np.ndarray) -> List[str]:
        """Get list of current safety violations."""
        
        violations = []
        
        # Check vital signs
        hr = state[0]
        sbp = state[1]
        spo2 = state[5]
        
        if hr > 150:
            violations.append(f"Severe tachycardia: HR {hr:.0f}")
        
        if sbp < 80:
            violations.append(f"Severe hypotension: SBP {sbp:.0f}")
        
        if spo2 < 88:
            violations.append(f"Severe hypoxemia: SpO2 {spo2:.1f}%")
        
        # Check laboratory values
        lactate = state[12]
        creatinine = state[11]
        
        if lactate > 4.0:
            violations.append(f"Elevated lactate: {lactate:.1f} mmol/L")
        
        if creatinine > 2.0:
            violations.append(f"Acute kidney injury: Cr {creatinine:.1f} mg/dL")
        
        return violations
```

## 5.3 Advanced RL Algorithms for Healthcare

### 5.3.1 Actor-Critic Methods for Continuous Treatment Optimization

Actor-Critic methods provide superior performance for continuous action spaces common in healthcare, such as drug dosing and fluid management. The following implementation demonstrates a sophisticated Proximal Policy Optimization (PPO) system:

```python
class ClinicalActorCritic(nn.Module):
    """
    Actor-Critic network for continuous clinical decision-making.
    
    Implements separate actor and critic networks with clinical constraints
    and uncertainty quantification for safe policy learning.
    
    References:
    - Schulman, J., et al. (2017). Proximal policy optimization algorithms. 
      arXiv preprint arXiv:1707.06347.
    - Raghu, A., et al. (2017). Continuous state-space models for optimal 
      sepsis treatment: a deep reinforcement learning approach. 
      MLHC 2017.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 action_bounds: Tuple[float, float] = (-1.0, 1.0)):
        super(ClinicalActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, action_dim),
            nn.Tanh()  # Bounded actions
        )
        
        # Actor standard deviation (learnable)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, 1)
        )
        
        # Clinical constraint layers
        self.constraint_checker = ClinicalConstraintNetwork(state_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action distribution and value.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action_mean, action_std, value)
        """
        
        # Shared feature extraction
        features = self.shared_layers(state)
        
        # Actor outputs
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        
        # Scale actions to bounds
        action_mean = action_mean * (self.action_bounds[1] - self.action_bounds[0]) / 2
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action_and_value(self, state: torch.Tensor, 
                           action: Optional[torch.Tensor] = None) -> Dict:
        """
        Get action distribution, sampled action, and value.
        
        Args:
            state: Input state tensor
            action: Optional action tensor for evaluation
            
        Returns:
            Dictionary with action info and value
        """
        
        action_mean, action_std, value = self.forward(state)
        
        # Create action distribution
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            # Sample action
            action = action_dist.sample()
        
        # Calculate log probability
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        # Calculate entropy
        entropy = action_dist.entropy().sum(dim=-1)
        
        # Apply clinical constraints
        constrained_action = self.constraint_checker.apply_constraints(state, action)
        
        return {
            'action': constrained_action,
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value.squeeze(-1),
            'action_mean': action_mean,
            'action_std': action_std
        }

class ClinicalConstraintNetwork(nn.Module):
    """
    Neural network for learning and applying clinical constraints.
    
    Learns to modify actions to satisfy clinical safety constraints
    while minimizing deviation from the original policy.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super(ClinicalConstraintNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Constraint prediction network
        self.constraint_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()  # Constraint satisfaction probability
        )
        
        # Action modification network
        self.modifier_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Bounded modifications
        )
        
    def apply_constraints(self, state: torch.Tensor, 
                         action: torch.Tensor) -> torch.Tensor:
        """
        Apply clinical constraints to actions.
        
        Args:
            state: Current patient state
            action: Proposed action
            
        Returns:
            Constrained action
        """
        
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)
        
        # Predict constraint satisfaction
        constraint_prob = self.constraint_net(state_action)
        
        # Generate action modifications
        modifications = self.modifier_net(state_action)
        
        # Apply modifications where constraints are violated
        constraint_mask = (constraint_prob < 0.5).float()
        constrained_action = action + constraint_mask * modifications * 0.1
        
        # Ensure actions remain in bounds
        constrained_action = torch.clamp(constrained_action, -1.0, 1.0)
        
        return constrained_action

class ClinicalPPOAgent:
    """
    Proximal Policy Optimization agent for clinical decision-making.
    
    Implements PPO with clinical safety constraints, curriculum learning,
    and multi-objective optimization for healthcare applications.
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5, device: str = 'cuda'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Initialize networks
        self.actor_critic = ClinicalActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Experience storage
        self.rollout_buffer = ClinicalRolloutBuffer()
        
        # Clinical evaluation metrics
        self.clinical_evaluator = ClinicalPolicyEvaluator()
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'policy_losses': [],
            'value_losses': [],
            'clinical_scores': []
        }
        
        # Curriculum learning
        self.curriculum = ClinicalCurriculum()
        
        # Logging
        self.logger = logging.getLogger('ClinicalPPOAgent')
        self.writer = SummaryWriter('runs/clinical_ppo')
        
    def collect_rollouts(self, env: ClinicalEnvironment, 
                        num_steps: int = 2048) -> Dict:
        """
        Collect rollouts from the environment.
        
        Args:
            env: Clinical environment
            num_steps: Number of steps to collect
            
        Returns:
            Rollout statistics
        """
        
        self.rollout_buffer.reset()
        
        episode_rewards = []
        episode_lengths = []
        clinical_outcomes = []
        
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_info = self.actor_critic.get_action_and_value(state_tensor)
                
                action = action_info['action'].cpu().numpy()[0]
                log_prob = action_info['log_prob'].cpu().numpy()[0]
                value = action_info['value'].cpu().numpy()[0]
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.rollout_buffer.add(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done
            )
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                # Episode finished
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Extract clinical outcome
                clinical_outcome = info.get('clinical_outcome', {})
                clinical_outcomes.append(clinical_outcome)
                
                # Reset for next episode
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                
                self.training_stats['episodes'] += 1
            else:
                state = next_state
            
            self.training_stats['total_steps'] += 1
        
        # Calculate advantages and returns
        self.rollout_buffer.compute_advantages_and_returns(
            gamma=self.gamma, 
            gae_lambda=self.gae_lambda
        )
        
        # Rollout statistics
        stats = {
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0,
            'num_episodes': len(episode_rewards),
            'survival_rate': np.mean([o.get('survived', False) for o in clinical_outcomes]) if clinical_outcomes else 0
        }
        
        return stats
    
    def update_policy(self, num_epochs: int = 10, batch_size: int = 64) -> Dict:
        """
        Update policy using PPO algorithm.
        
        Args:
            num_epochs: Number of optimization epochs
            batch_size: Batch size for updates
            
        Returns:
            Training statistics
        """
        
        # Get rollout data
        rollout_data = self.rollout_buffer.get_data()
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(rollout_data['states']))
            
            for start_idx in range(0, len(indices), batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Extract batch data
                batch_states = rollout_data['states'][batch_indices]
                batch_actions = rollout_data['actions'][batch_indices]
                batch_old_log_probs = rollout_data['log_probs'][batch_indices]
                batch_advantages = rollout_data['advantages'][batch_indices]
                batch_returns = rollout_data['returns'][batch_indices]
                batch_old_values = rollout_data['values'][batch_indices]
                
                # Get current policy outputs
                action_info = self.actor_critic.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Calculate policy loss
                ratio = torch.exp(action_info['log_prob'] - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_pred = action_info['value']
                value_loss = F.mse_loss(value_pred, batch_returns)
                
                # Calculate entropy loss
                entropy_loss = -action_info['entropy'].mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_loss_coef * value_loss + 
                             self.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 
                                             self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Calculate training statistics
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(policy_losses) + self.value_loss_coef * np.mean(value_losses) + self.entropy_coef * np.mean(entropy_losses)
        }
        
        # Update training statistics
        self.training_stats['policy_losses'].append(stats['policy_loss'])
        self.training_stats['value_losses'].append(stats['value_loss'])
        
        # Log to tensorboard
        for key, value in stats.items():
            self.writer.add_scalar(f'training/{key}', value, self.training_stats['total_steps'])
        
        return stats
    
    def evaluate_clinical_performance(self, env: ClinicalEnvironment, 
                                    num_episodes: int = 50) -> Dict:
        """
        Evaluate clinical performance of the policy.
        
        Args:
            env: Clinical environment
            num_episodes: Number of evaluation episodes
            
        Returns:
            Clinical performance metrics
        """
        
        self.actor_critic.eval()
        
        clinical_metrics = self.clinical_evaluator.evaluate_policy(
            self.actor_critic, env, num_episodes
        )
        
        self.actor_critic.train()
        
        # Log clinical metrics
        for key, value in clinical_metrics.items():
            self.writer.add_scalar(f'clinical/{key}', value, self.training_stats['total_steps'])
        
        self.training_stats['clinical_scores'].append(clinical_metrics)
        
        return clinical_metrics
    
    def train(self, env: ClinicalEnvironment, total_timesteps: int = 1000000,
              eval_freq: int = 10000, save_freq: int = 50000) -> Dict:
        """
        Train the PPO agent.
        
        Args:
            env: Clinical environment
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            save_freq: Model saving frequency
            
        Returns:
            Training history
        """
        
        self.logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        training_history = {
            'timesteps': [],
            'episode_rewards': [],
            'clinical_scores': [],
            'policy_losses': [],
            'value_losses': []
        }
        
        timesteps_collected = 0
        
        while timesteps_collected < total_timesteps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts(env, num_steps=2048)
            timesteps_collected += 2048
            
            # Update policy
            update_stats = self.update_policy()
            
            # Log progress
            self.logger.info(f"Timesteps: {timesteps_collected}, "
                           f"Mean Reward: {rollout_stats['mean_episode_reward']:.2f}, "
                           f"Survival Rate: {rollout_stats['survival_rate']:.2f}, "
                           f"Policy Loss: {update_stats['policy_loss']:.4f}")
            
            # Store training history
            training_history['timesteps'].append(timesteps_collected)
            training_history['episode_rewards'].append(rollout_stats['mean_episode_reward'])
            training_history['policy_losses'].append(update_stats['policy_loss'])
            training_history['value_losses'].append(update_stats['value_loss'])
            
            # Evaluate clinical performance
            if timesteps_collected % eval_freq == 0:
                clinical_metrics = self.evaluate_clinical_performance(env)
                training_history['clinical_scores'].append(clinical_metrics)
                
                self.logger.info(f"Clinical Evaluation - "
                               f"Mortality Rate: {clinical_metrics.get('mortality_rate', 0):.3f}, "
                               f"Mean LOS: {clinical_metrics.get('mean_length_of_stay', 0):.1f}, "
                               f"SOFA Improvement: {clinical_metrics.get('sofa_improvement', 0):.2f}")
            
            # Save model
            if timesteps_collected % save_freq == 0:
                self.save_model(f'clinical_ppo_model_{timesteps_collected}.pt')
            
            # Update curriculum
            self.curriculum.update(rollout_stats, clinical_metrics if timesteps_collected % eval_freq == 0 else None)
        
        self.logger.info("Training completed")
        return training_history
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'curriculum_state': self.curriculum.get_state()
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        self.curriculum.load_state(checkpoint['curriculum_state'])
        
        self.logger.info(f"Model loaded from {filepath}")

class ClinicalRolloutBuffer:
    """
    Rollout buffer for storing and processing PPO experiences.
    
    Handles advantage calculation using Generalized Advantage Estimation (GAE)
    and provides efficient batch sampling for policy updates.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the buffer."""
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            value: float, log_prob: float, done: bool):
        """Add experience to buffer."""
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages_and_returns(self, gamma: float = 0.99, 
                                     gae_lambda: float = 0.95):
        """Compute advantages and returns using GAE."""
        
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Calculate advantages using GAE
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0  # Assuming episode ends
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        
        # Calculate returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get_data(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        
        return {
            'states': torch.FloatTensor(self.states),
            'actions': torch.FloatTensor(self.actions),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'log_probs': torch.FloatTensor(self.log_probs),
            'dones': torch.BoolTensor(self.dones),
            'advantages': torch.FloatTensor(self.advantages),
            'returns': torch.FloatTensor(self.returns)
        }
    
    def __len__(self):
        return len(self.states)

class ClinicalPolicyEvaluator:
    """
    Comprehensive clinical policy evaluation framework.
    
    Evaluates RL policies using clinically relevant metrics including
    mortality rates, length of stay, organ function preservation,
    and treatment appropriateness.
    """
    
    def __init__(self):
        self.clinical_metrics = self._initialize_clinical_metrics()
        
    def _initialize_clinical_metrics(self) -> Dict:
        """Initialize clinical evaluation metrics."""
        
        metrics = {
            'primary_outcomes': {
                'mortality_rate': {'weight': 1.0, 'direction': 'minimize'},
                'survival_rate': {'weight': 1.0, 'direction': 'maximize'},
                'length_of_stay': {'weight': 0.3, 'direction': 'minimize'}
            },
            'secondary_outcomes': {
                'sofa_improvement': {'weight': 0.5, 'direction': 'maximize'},
                'organ_dysfunction_days': {'weight': 0.3, 'direction': 'minimize'},
                'treatment_appropriateness': {'weight': 0.4, 'direction': 'maximize'}
            },
            'safety_metrics': {
                'adverse_events': {'weight': 0.8, 'direction': 'minimize'},
                'protocol_violations': {'weight': 0.6, 'direction': 'minimize'},
                'extreme_interventions': {'weight': 0.5, 'direction': 'minimize'}
            }
        }
        
        return metrics
    
    def evaluate_policy(self, policy: nn.Module, env: ClinicalEnvironment,
                       num_episodes: int = 100) -> Dict:
        """
        Evaluate policy performance on clinical metrics.
        
        Args:
            policy: Trained policy network
            env: Clinical environment
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of clinical performance metrics
        """
        
        policy.eval()
        
        episode_outcomes = []
        
        for episode in range(num_episodes):
            # Run episode
            outcome = self._run_evaluation_episode(policy, env)
            episode_outcomes.append(outcome)
        
        # Calculate aggregate metrics
        clinical_metrics = self._calculate_clinical_metrics(episode_outcomes)
        
        policy.train()
        
        return clinical_metrics
    
    def _run_evaluation_episode(self, policy: nn.Module, 
                               env: ClinicalEnvironment) -> Dict:
        """Run single evaluation episode."""
        
        state = env.reset()
        episode_data = {
            'states': [state.copy()],
            'actions': [],
            'rewards': [],
            'clinical_events': []
        }
        
        done = False
        total_reward = 0
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_info = policy.get_action_and_value(state_tensor)
                action = action_info['action'].cpu().numpy()[0]
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store episode data
            episode_data['actions'].append(action.copy())
            episode_data['rewards'].append(reward)
            episode_data['states'].append(next_state.copy())
            
            # Check for clinical events
            clinical_events = self._detect_clinical_events(state, action, next_state, info)
            episode_data['clinical_events'].extend(clinical_events)
            
            total_reward += reward
            state = next_state
        
        # Calculate episode outcome
        outcome = self._calculate_episode_outcome(episode_data, info)
        outcome['total_reward'] = total_reward
        
        return outcome
    
    def _detect_clinical_events(self, state: np.ndarray, action: np.ndarray,
                               next_state: np.ndarray, info: Dict) -> List[Dict]:
        """Detect significant clinical events during episode."""
        
        events = []
        
        # Detect hypotensive episodes
        map_current = (state[1] + 2 * state[2]) / 3
        map_next = (next_state[1] + 2 * next_state[2]) / 3
        
        if map_next < 65 and map_current >= 65:
            events.append({
                'type': 'hypotensive_episode',
                'severity': 'moderate' if map_next > 55 else 'severe',
                'timestep': info.get('episode_length', 0)
            })
        
        # Detect fluid overload
        if action[0] > 500 and state[0] > 110:  # Large fluid bolus with tachycardia
            events.append({
                'type': 'potential_fluid_overload',
                'severity': 'moderate',
                'timestep': info.get('episode_length', 0)
            })
        
        # Detect excessive vasopressor use
        if len(action) > 1 and action[1] > 0.8:
            events.append({
                'type': 'high_vasopressor_dose',
                'severity': 'moderate',
                'timestep': info.get('episode_length', 0)
            })
        
        # Detect rapid clinical deterioration
        sofa_change = self._calculate_sofa_change(state, next_state)
        if sofa_change > 3:
            events.append({
                'type': 'rapid_deterioration',
                'severity': 'severe',
                'timestep': info.get('episode_length', 0)
            })
        
        return events
    
    def _calculate_sofa_change(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """Calculate change in SOFA score between states."""
        
        # Simplified SOFA calculation
        def sofa_score(s):
            respiratory = min(4, max(0, (100 - s[5]) / 20))
            cardiovascular = min(4, max(0, (140 - s[1]) / 20))
            renal = min(4, max(0, (s[11] - 1.2) * 2))
            coagulation = min(4, max(0, (150 - s[8]) / 30))
            return respiratory + cardiovascular + renal + coagulation
        
        return sofa_score(next_state) - sofa_score(state)
    
    def _calculate_episode_outcome(self, episode_data: Dict, final_info: Dict) -> Dict:
        """Calculate clinical outcome for episode."""
        
        outcome = {
            'survived': final_info.get('clinical_outcome', {}).get('survived', False),
            'length_of_stay': len(episode_data['states']) - 1,
            'final_sofa': self._calculate_sofa_from_state(episode_data['states'][-1]),
            'initial_sofa': self._calculate_sofa_from_state(episode_data['states'][0]),
            'clinical_events': episode_data['clinical_events'],
            'total_fluid_given': sum([max(0, a[0]) for a in episode_data['actions']]),
            'max_vasopressor_dose': max([a[1] if len(a) > 1 else 0 for a in episode_data['actions']]),
            'treatment_intensity': self._calculate_treatment_intensity(episode_data['actions'])
        }
        
        # Calculate derived metrics
        outcome['sofa_improvement'] = outcome['initial_sofa'] - outcome['final_sofa']
        outcome['adverse_events'] = len([e for e in outcome['clinical_events'] 
                                       if e['severity'] in ['moderate', 'severe']])
        
        return outcome
    
    def _calculate_sofa_from_state(self, state: np.ndarray) -> float:
        """Calculate SOFA score from state vector."""
        
        respiratory = min(4, max(0, (100 - state[5]) / 20))
        cardiovascular = min(4, max(0, (140 - state[1]) / 20))
        renal = min(4, max(0, (state[11] - 1.2) * 2))
        coagulation = min(4, max(0, (150 - state[8]) / 30))
        
        return respiratory + cardiovascular + renal + coagulation
    
    def _calculate_treatment_intensity(self, actions: List[np.ndarray]) -> float:
        """Calculate treatment intensity score."""
        
        intensity = 0.0
        
        for action in actions:
            # Fluid intensity
            fluid_intensity = abs(action[0]) / 1000.0  # Normalize by 1L
            intensity += fluid_intensity
            
            # Vasopressor intensity
            if len(action) > 1:
                vasopressor_intensity = action[1] ** 2
                intensity += vasopressor_intensity
        
        return intensity / len(actions) if actions else 0.0
    
    def _calculate_clinical_metrics(self, episode_outcomes: List[Dict]) -> Dict:
        """Calculate aggregate clinical metrics from episode outcomes."""
        
        if not episode_outcomes:
            return {}
        
        metrics = {}
        
        # Primary outcomes
        metrics['mortality_rate'] = 1.0 - np.mean([o['survived'] for o in episode_outcomes])
        metrics['survival_rate'] = np.mean([o['survived'] for o in episode_outcomes])
        metrics['mean_length_of_stay'] = np.mean([o['length_of_stay'] for o in episode_outcomes])
        metrics['median_length_of_stay'] = np.median([o['length_of_stay'] for o in episode_outcomes])
        
        # Secondary outcomes
        metrics['mean_sofa_improvement'] = np.mean([o['sofa_improvement'] for o in episode_outcomes])
        metrics['sofa_improvement_rate'] = np.mean([o['sofa_improvement'] > 0 for o in episode_outcomes])
        
        # Safety metrics
        metrics['mean_adverse_events'] = np.mean([o['adverse_events'] for o in episode_outcomes])
        metrics['adverse_event_rate'] = np.mean([o['adverse_events'] > 0 for o in episode_outcomes])
        
        # Treatment metrics
        metrics['mean_treatment_intensity'] = np.mean([o['treatment_intensity'] for o in episode_outcomes])
        metrics['mean_fluid_volume'] = np.mean([o['total_fluid_given'] for o in episode_outcomes])
        metrics['high_vasopressor_rate'] = np.mean([o['max_vasopressor_dose'] > 0.5 for o in episode_outcomes])
        
        # Calculate composite clinical score
        metrics['composite_clinical_score'] = self._calculate_composite_score(metrics)
        
        return metrics
    
    def _calculate_composite_score(self, metrics: Dict) -> float:
        """Calculate composite clinical performance score."""
        
        # Weighted combination of key metrics
        score = 0.0
        
        # Primary outcomes (60% weight)
        score += 0.4 * metrics['survival_rate']  # Higher is better
        score += 0.2 * max(0, 1 - metrics['mean_length_of_stay'] / 14)  # Shorter LOS is better
        
        # Secondary outcomes (25% weight)
        score += 0.15 * max(0, metrics['mean_sofa_improvement'] / 10)  # SOFA improvement
        score += 0.1 * metrics['sofa_improvement_rate']  # Rate of improvement
        
        # Safety (15% weight)
        score += 0.1 * max(0, 1 - metrics['adverse_event_rate'])  # Fewer adverse events
        score += 0.05 * max(0, 1 - metrics['mean_treatment_intensity'] / 2)  # Lower intensity
        
        return np.clip(score, 0, 1)

class ClinicalCurriculum:
    """
    Curriculum learning framework for clinical RL training.
    
    Gradually increases training difficulty and complexity to improve
    learning efficiency and final performance.
    """
    
    def __init__(self):
        self.current_level = 0
        self.max_level = 5
        self.level_criteria = self._define_level_criteria()
        self.performance_history = []
        
    def _define_level_criteria(self) -> Dict:
        """Define criteria for advancing curriculum levels."""
        
        criteria = {
            0: {  # Basic vital sign management
                'description': 'Simple vital sign stabilization',
                'patient_complexity': 'low',
                'episode_length': 24,  # hours
                'success_threshold': 0.7,
                'required_episodes': 100
            },
            1: {  # Fluid management
                'description': 'Fluid balance optimization',
                'patient_complexity': 'low-medium',
                'episode_length': 48,
                'success_threshold': 0.75,
                'required_episodes': 150
            },
            2: {  # Vasopressor management
                'description': 'Vasopressor titration',
                'patient_complexity': 'medium',
                'episode_length': 72,
                'success_threshold': 0.8,
                'required_episodes': 200
            },
            3: {  # Multi-organ support
                'description': 'Multi-organ dysfunction management',
                'patient_complexity': 'medium-high',
                'episode_length': 120,
                'success_threshold': 0.8,
                'required_episodes': 250
            },
            4: {  # Complex cases
                'description': 'Complex multi-morbid patients',
                'patient_complexity': 'high',
                'episode_length': 168,
                'success_threshold': 0.85,
                'required_episodes': 300
            },
            5: {  # Expert level
                'description': 'Expert-level clinical management',
                'patient_complexity': 'very_high',
                'episode_length': 168,
                'success_threshold': 0.9,
                'required_episodes': float('inf')
            }
        }
        
        return criteria
    
    def update(self, rollout_stats: Dict, clinical_metrics: Optional[Dict] = None):
        """Update curriculum based on performance."""
        
        # Store performance
        performance = {
            'survival_rate': rollout_stats.get('survival_rate', 0),
            'mean_reward': rollout_stats.get('mean_episode_reward', 0),
            'clinical_score': clinical_metrics.get('composite_clinical_score', 0) if clinical_metrics else 0
        }
        
        self.performance_history.append(performance)
        
        # Check for level advancement
        if self.current_level < self.max_level:
            if self._should_advance_level():
                self.current_level += 1
                self.performance_history = []  # Reset for next level
                print(f"Curriculum advanced to level {self.current_level}: {self.level_criteria[self.current_level]['description']}")
    
    def _should_advance_level(self) -> bool:
        """Check if should advance to next curriculum level."""
        
        if len(self.performance_history) < self.level_criteria[self.current_level]['required_episodes']:
            return False
        
        # Check recent performance
        recent_performance = self.performance_history[-50:]  # Last 50 episodes
        avg_survival = np.mean([p['survival_rate'] for p in recent_performance])
        
        threshold = self.level_criteria[self.current_level]['success_threshold']
        
        return avg_survival >= threshold
    
    def get_current_config(self) -> Dict:
        """Get current curriculum configuration."""
        
        return self.level_criteria[self.current_level]
    
    def get_state(self) -> Dict:
        """Get curriculum state for saving."""
        
        return {
            'current_level': self.current_level,
            'performance_history': self.performance_history
        }
    
    def load_state(self, state: Dict):
        """Load curriculum state."""
        
        self.current_level = state['current_level']
        self.performance_history = state['performance_history']
```

## 5.4 Clinical Validation and Safety Framework

### 5.4.1 Offline Reinforcement Learning for Clinical Validation

Offline RL enables training and validation of clinical policies using historical data without direct patient interaction, providing a crucial safety mechanism for healthcare applications:

```python
class ClinicalOfflineRLFramework:
    """
    Comprehensive offline reinforcement learning framework for clinical validation.
    
    Implements Conservative Q-Learning (CQL) and other offline RL algorithms
    specifically adapted for healthcare applications with safety constraints.
    
    References:
    - Kumar, A., et al. (2020). Conservative Q-Learning for offline reinforcement 
      learning. Advances in Neural Information Processing Systems, 33, 1179-1191.
    - Gottesman, O., et al. (2019). Guidelines for reinforcement learning in 
      healthcare. Nature Medicine, 25(1), 16-18.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 dataset_path: str, validation_config: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dataset_path = dataset_path
        self.validation_config = validation_config
        
        # Load and preprocess clinical dataset
        self.dataset = self._load_clinical_dataset()
        
        # Initialize offline RL algorithm
        self.offline_agent = ConservativeQLearning(state_dim, action_dim)
        
        # Clinical validation framework
        self.validator = ClinicalPolicyValidator()
        
        # Safety monitoring
        self.safety_monitor = ClinicalSafetyMonitor()
        
        # Logging
        self.logger = logging.getLogger('ClinicalOfflineRL')
        
    def _load_clinical_dataset(self) -> Dict:
        """Load and preprocess clinical dataset for offline RL."""
        
        # Load raw data
        raw_data = pd.read_csv(self.dataset_path)
        
        # Preprocess data
        processed_data = self._preprocess_clinical_data(raw
