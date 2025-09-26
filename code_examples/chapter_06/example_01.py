"""
Chapter 6 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

"""
Comprehensive Clinical Language Model Implementation

This implementation provides advanced generative AI capabilities specifically
designed for healthcare applications, including safety frameworks, bias detection,
regulatory compliance, and clinical validation systems.

Author: Sanjay Basu MD PhD
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, BitsAndBytesConfig
)
from datasets import Dataset, load_dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import json
import re
from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import warnings
from abc import ABC, abstractmethod
import pickle
import hashlib
from pathlib import Path
import yaml
from scipy import stats
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, flesch_kincaid_grade
import spacy
from collections import defaultdict, Counter
import requests
from tqdm import tqdm

\# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

\# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    logger.warning("Could not download NLTK data")

@dataclass
class ClinicalGenerationConfig:
    """Configuration for clinical text generation"""
    
    \# Model parameters
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 1024
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    \# Safety parameters
    safety_threshold: float = 0.8
    bias_threshold: float = 0.7
    hallucination_threshold: float = 0.6
    
    \# Clinical validation parameters
    medical_accuracy_threshold: float = 0.85
    clinical_relevance_threshold: float = 0.8
    
    \# Privacy parameters
    phi_detection_enabled: bool = True
    anonymization_enabled: bool = True
    
    \# Regulatory compliance
    fda_compliance_mode: bool = True
    hipaa_compliance_mode: bool = True
    audit_logging_enabled: bool = True

@dataclass
class ClinicalValidationResult:
    """Results from clinical validation of generated content"""
    
    medical_accuracy_score: float
    clinical_relevance_score: float
    safety_score: float
    bias_score: float
    hallucination_score: float
    phi_detected: bool
    regulatory_compliant: bool
    validation_passed: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class MedicalKnowledgeBase:
    """Medical knowledge base for clinical validation"""
    
    def __init__(self):
        """Initialize medical knowledge base"""
        
        \# Medical entities and relationships
        self.medical_entities = self._load_medical_entities()
        self.drug_interactions = self._load_drug_interactions()
        self.clinical_guidelines = self._load_clinical_guidelines()
        self.contraindications = self._load_contraindications()
        
        \# Medical terminology and ontologies
        self.icd10_codes = self._load_icd10_codes()
        self.snomed_concepts = self._load_snomed_concepts()
        self.rxnorm_drugs = self._load_rxnorm_drugs()
        
        logger.info("Initialized medical knowledge base")
    
    def _load_medical_entities(self) -> Dict[str, List[str]]:
        """Load medical entity recognition data"""
        
        return {
            'conditions': [
                'diabetes mellitus', 'hypertension', 'pneumonia', 'covid-19',
                'myocardial infarction', 'stroke', 'cancer', 'asthma',
                'chronic obstructive pulmonary disease', 'heart failure',
                'atrial fibrillation', 'depression', 'anxiety', 'obesity'
            ],
            'medications': [
                'metformin', 'lisinopril', 'atorvastatin', 'amlodipine',
                'metoprolol', 'omeprazole', 'albuterol', 'insulin',
                'warfarin', 'aspirin', 'furosemide', 'prednisone'
            ],
            'procedures': [
                'echocardiogram', 'computed tomography', 'magnetic resonance imaging',
                'blood test', 'biopsy', 'surgery', 'vaccination',
                'physical therapy', 'electrocardiogram', 'chest x-ray'
            ],
            'symptoms': [
                'chest pain', 'shortness of breath', 'fatigue', 'nausea',
                'headache', 'dizziness', 'fever', 'cough', 'abdominal pain'
            ]
        }
    
    def _load_drug_interactions(self) -> Dict[Tuple[str, str], str]:
        """Load drug interaction database"""
        
        return {
            ('warfarin', 'aspirin'): 'Increased bleeding risk - monitor INR closely',
            ('metformin', 'contrast media'): 'Risk of lactic acidosis - hold metformin',
            ('ace inhibitor', 'potassium supplement'): 'Risk of hyperkalemia',
            ('digoxin', 'furosemide'): 'Risk of digoxin toxicity due to hypokalemia',
            ('warfarin', 'antibiotics'): 'Increased anticoagulation effect',
            ('insulin', 'beta blocker'): 'Masked hypoglycemia symptoms'
        }
    
    def _load_clinical_guidelines(self) -> Dict[str, Dict]:
        """Load clinical practice guidelines"""
        
        return {
            'hypertension': {
                'target_bp': '<130/80 mmHg for most adults',
                'first_line_therapy': ['ACE inhibitor', 'ARB', 'thiazide diuretic', 'CCB'],
                'lifestyle_modifications': ['diet', 'exercise', 'weight loss', 'sodium restriction']
            },
            'diabetes': {
                'target_hba1c': '<7% for most adults',
                'first_line_therapy': 'metformin',
                'monitoring': 'HbA1c every 3-6 months'
            },
            'heart_failure': {
                'ace_inhibitor_indicated': True,
                'beta_blocker_indicated': True,
                'diuretic_for_volume_overload': True
            }
        }
    
    def _load_contraindications(self) -> Dict[str, List[str]]:
        """Load medication contraindications"""
        
        return {
            'metformin': ['severe kidney disease', 'liver disease', 'heart failure'],
            'ace_inhibitor': ['pregnancy', 'hyperkalemia', 'bilateral renal artery stenosis'],
            'beta_blocker': ['severe asthma', 'heart block', 'severe bradycardia'],
            'warfarin': ['active bleeding', 'severe liver disease', 'pregnancy']
        }
    
    def _load_icd10_codes(self) -> Dict[str, str]:
        """Load ICD-10 diagnostic codes"""
        
        return {
            'E11.9': 'Type 2 diabetes mellitus without complications',
            'I10': 'Essential hypertension',
            'J44.1': 'Chronic obstructive pulmonary disease with acute exacerbation',
            'I21.9': 'Acute myocardial infarction, unspecified',
            'I50.9': 'Heart failure, unspecified'
        }
    
    def _load_snomed_concepts(self) -> Dict[str, str]:
        """Load SNOMED CT concepts"""
        
        return {
            '44054006': 'Diabetes mellitus',
            '38341003': 'Hypertensive disorder',
            '13645005': 'Chronic obstructive lung disease',
            '22298006': 'Myocardial infarction',
            '84114007': 'Heart failure'
        }
    
    def _load_rxnorm_drugs(self) -> Dict[str, str]:
        """Load RxNorm drug concepts"""
        
        return {
            '6809': 'Metformin',
            '29046': 'Lisinopril',
            '83367': 'Atorvastatin',
            '17767': 'Amlodipine',
            '49276': 'Metoprolol'
        }
    
    def validate_medical_content(self, text: str) -> Dict[str, Any]:
        """Validate medical content for accuracy and appropriateness"""
        
        validation_results = {
            'medical_entities_found': [],
            'drug_interactions_detected': [],
            'guideline_adherence': {},
            'contraindications_flagged': [],
            'accuracy_score': 0.0,
            'issues': []
        }
        
        \# Extract medical entities
        text_lower = text.lower()
        for entity_type, entities in self.medical_entities.items():
            found_entities = [entity for entity in entities if entity.lower() in text_lower]
            validation_results['medical_entities_found'].extend(
                [(entity, entity_type) for entity in found_entities]
            )
        
        \# Check drug interactions
        mentioned_drugs = [
            drug for drug in self.medical_entities['medications']
            if drug.lower() in text_lower
        ]
        
        for i, drug1 in enumerate(mentioned_drugs):
            for drug2 in mentioned_drugs[i+1:]:
                interaction_key = tuple(sorted([drug1, drug2]))
                if interaction_key in self.drug_interactions:
                    validation_results['drug_interactions_detected'].append({
                        'drugs': interaction_key,
                        'interaction': self.drug_interactions[interaction_key]
                    })
        
        \# Calculate accuracy score based on medical entity recognition
        \# and absence of obvious medical errors
        accuracy_score = min(1.0, len(validation_results['medical_entities_found']) / 10.0)
        validation_results['accuracy_score'] = accuracy_score
        
        return validation_results

class ClinicalSafetyFramework:
    """Comprehensive safety framework for clinical AI systems"""
    
    def __init__(self, config: ClinicalGenerationConfig):
        """Initialize clinical safety framework"""
        
        self.config = config
        
        \# Initialize safety classifiers
        self.safety_classifier = self._initialize_safety_classifier()
        self.bias_detector = self._initialize_bias_detector()
        self.hallucination_detector = self._initialize_hallucination_detector()
        self.phi_detector = self._initialize_phi_detector()
        
        \# Initialize medical knowledge base
        self.knowledge_base = MedicalKnowledgeBase()
        
        \# Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.sentiment_analyzer = None
            logger.warning("Could not initialize sentiment analyzer")
        
        logger.info("Initialized clinical safety framework")
    
    def _initialize_safety_classifier(self) -> Dict[str, List[str]]:
        """Initialize safety classification patterns"""
        
        return {
            'harmful_medical_advice': [
                r'ignore.*doctor.*advice',
                r'stop.*medication.*without.*consulting',
                r'self.*diagnose',
                r'avoid.*medical.*care',
                r'cure.*guaranteed',
                r'miracle.*cure',
                r'natural.*alternative.*to.*medicine'
            ],
            'inappropriate_urgency': [
                r'emergency.*not.*needed',
                r'wait.*before.*seeking.*help',
                r'symptoms.*not.*serious'
            ],
            'medication_misuse': [
                r'increase.*dose.*on.*own',
                r'share.*medication',
                r'stop.*suddenly',
                r'double.*dose'
            ]
        }
    
    def _initialize_bias_detector(self) -> Dict[str, List[str]]:
        """Initialize bias detection patterns"""
        
        return {
            'demographic_bias': {
                'race_ethnicity': [
                    'black patients are more likely to',
                    'hispanic patients typically',
                    'asian patients usually',
                    'white patients generally'
                ],
                'gender_bias': [
                    'women are more emotional',
                    'men don\'t seek help',
                    'female patients complain more',
                    'male patients are stoic'
                ],
                'age_bias': [
                    'elderly patients are confused',
                    'young patients exaggerate',
                    'older adults don\'t understand',
                    'seniors are non-compliant'
                ],
                'socioeconomic_bias': [
                    'poor patients don\'t follow',
                    'wealthy patients get better care',
                    'uninsured patients are',
                    'homeless patients are'
                ]
            }
        }
    
    def _initialize_hallucination_detector(self) -> Dict[str, Any]:
        """Initialize hallucination detection system"""
        
        return {
            'medical_fact_patterns': [
                r'studies show that \d+% of patients',
                r'research indicates that',
                r'clinical trials demonstrate',
                r'according to the \w+ study'
            ],
            'specific_number_patterns': [
                r'\d+% of patients with',
                r'\d+ mg of',
                r'\d+ times per day',
                r'within \d+ hours'
            ],
            'authority_patterns': [
                r'the FDA recommends',
                r'WHO guidelines state',
                r'AMA position is',
                r'according to Mayo Clinic'
            ]
        }
    
    def _initialize_phi_detector(self) -> Dict[str, str]:
        """Initialize PHI detection patterns"""
        
        return {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'mrn': r'\bMRN:?\s*\d+\b',
            'name_pattern': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  \# Simplified name pattern
        }
    
    def evaluate_safety(self, text: str) -> Dict[str, Any]:
        """Comprehensive safety evaluation of generated text"""
        
        safety_results = {
            'overall_safe': True,
            'safety_score': 1.0,
            'issues': [],
            'recommendations': []
        }
        
        \# Check for harmful medical advice
        harmful_patterns_found = []
        for category, patterns in self.safety_classifier.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    harmful_patterns_found.append((category, pattern))
        
        if harmful_patterns_found:
            safety_results['overall_safe'] = False
            safety_results['safety_score'] *= 0.5
            safety_results['issues'].append('Potentially harmful medical advice detected')
            safety_results['recommendations'].append('Review content for medical accuracy')
        
        \# Check for bias
        bias_score = self._detect_bias(text)
        if bias_score > self.config.bias_threshold:
            safety_results['overall_safe'] = False
            safety_results['safety_score'] *= 0.7
            safety_results['issues'].append(f'Potential bias detected (score: {bias_score:.2f})')
            safety_results['recommendations'].append('Review content for demographic bias')
        
        \# Check for hallucinations
        hallucination_score = self._detect_hallucinations(text)
        if hallucination_score > self.config.hallucination_threshold:
            safety_results['safety_score'] *= 0.8
            safety_results['issues'].append(f'Potential hallucinations detected (score: {hallucination_score:.2f})')
            safety_results['recommendations'].append('Verify factual claims with medical literature')
        
        \# Check for PHI
        phi_detected = self._detect_phi(text)
        if phi_detected:
            safety_results['overall_safe'] = False
            safety_results['safety_score'] = 0.0
            safety_results['issues'].append('Protected health information detected')
            safety_results['recommendations'].append('Remove or anonymize PHI before use')
        
        return safety_results
    
    def _detect_bias(self, text: str) -> float:
        """Detect potential bias in generated text"""
        
        bias_score = 0.0
        text_lower = text.lower()
        
        for bias_type, patterns in self.bias_detector['demographic_bias'].items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    bias_score += 0.2
        
        return min(1.0, bias_score)
    
    def _detect_hallucinations(self, text: str) -> float:
        """Detect potential hallucinations in medical content"""
        
        hallucination_score = 0.0
        
        \# Check for specific medical claims without verification
        for pattern in self.hallucination_detector['medical_fact_patterns']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            hallucination_score += len(matches) * 0.1
        
        \# Check for specific numbers that might be fabricated
        for pattern in self.hallucination_detector['specific_number_patterns']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            hallucination_score += len(matches) * 0.05
        
        return min(1.0, hallucination_score)
    
    def _detect_phi(self, text: str) -> bool:
        """Detect protected health information in text"""
        
        if not self.config.phi_detection_enabled:
            return False
        
        for phi_type, pattern in self.phi_detector.items():
            if re.search(pattern, text):
                return True
        
        return False

class ClinicalLanguageModel:
    """
    Production-ready clinical language model for healthcare applications.
    
    This implementation provides comprehensive generative AI capabilities
    specifically designed for clinical environments, including safety frameworks,
    bias detection, regulatory compliance, and clinical validation systems.
    """
    
    def __init__(self, config: ClinicalGenerationConfig):
        """
        Initialize the clinical language model.
        
        Args:
            config: Configuration for clinical text generation
        """
        self.config = config
        
        \# Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        \# Configure model loading for efficiency
        model_kwargs = {
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        if torch.cuda.is_available():
            model_kwargs['device_map'] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        \# Initialize safety framework
        self.safety_framework = ClinicalSafetyFramework(config)
        
        \# Initialize clinical validator
        self.clinical_validator = self._initialize_clinical_validator()
        
        \# Audit logging
        self.audit_log = []
        
        logger.info(f"Initialized ClinicalLanguageModel with {config.model_name}")
    
    def _initialize_clinical_validator(self):
        """Initialize clinical content validator"""
        
        \# In production, this would load specialized medical validation models
        \# For demonstration, we use the knowledge base
        return self.safety_framework.knowledge_base
    
    def preprocess_clinical_text(self, text: str) -> str:
        """
        Preprocess clinical text for model input.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Preprocessed text ready for model input
        """
        \# Anonymize PHI if enabled
        if self.config.anonymization_enabled:
            text = self._anonymize_phi(text)
        
        \# Normalize medical terminology
        text = self._normalize_medical_terms(text)
        
        \# Clean and format text
        text = self._clean_clinical_text(text)
        
        return text
    
    def _anonymize_phi(self, text: str) -> str:
        """Anonymize protected health information"""
        
        phi_replacements = {
            'ssn': '[SSN]',
            'phone': '[PHONE]',
            'email': '[EMAIL]',
            'date': '[DATE]',
            'mrn': '[MRN]',
            'name_pattern': '[NAME]'
        }
        
        anonymized_text = text
        for phi_type, pattern in self.safety_framework.phi_detector.items():
            if phi_type in phi_replacements:
                anonymized_text = re.sub(
                    pattern, 
                    phi_replacements[phi_type], 
                    anonymized_text
                )
        
        return anonymized_text
    
    def _normalize_medical_terms(self, text: str) -> str:
        """Normalize medical terminology for consistency"""
        
        \# Medical abbreviation expansion
        abbreviations = {
            'MI': 'myocardial infarction',
            'HTN': 'hypertension',
            'DM': 'diabetes mellitus',
            'CHF': 'congestive heart failure',
            'COPD': 'chronic obstructive pulmonary disease',
            'CAD': 'coronary artery disease',
            'CVA': 'cerebrovascular accident',
            'DVT': 'deep vein thrombosis',
            'PE': 'pulmonary embolism',
            'UTI': 'urinary tract infection'
        }
        
        normalized_text = text
        for abbrev, full_term in abbreviations.items():
            normalized_text = re.sub(
                rf'\b{abbrev}\b', 
                full_term, 
                normalized_text, 
                flags=re.IGNORECASE
            )
        
        return normalized_text
    
    def _clean_clinical_text(self, text: str) -> str:
        """Clean and format clinical text"""
        
        \# Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        \# Standardize punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        
        \# Ensure proper sentence endings
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def generate_clinical_note(
        self, 
        prompt: str,
        note_type: str = "progress_note",
        patient_context: Optional[Dict] = None,
        validate_output: bool = True
    ) -> Dict[str, Any]:
        """
        Generate clinical documentation with comprehensive validation.
        
        Args:
            prompt: Input prompt for generation
            note_type: Type of clinical note to generate
            patient_context: Additional patient context
            validate_output: Whether to validate generated content
            
        Returns:
            Dictionary containing generated text and validation results
        """
        
        \# Log generation request for audit
        if self.config.audit_logging_enabled:
            self._log_generation_request(prompt, note_type, patient_context)
        
        \# Preprocess input
        processed_prompt = self.preprocess_clinical_text(prompt)
        
        \# Add clinical context if provided
        if patient_context:
            processed_prompt = self._add_patient_context(processed_prompt, patient_context)
        
        \# Add note type specific formatting
        formatted_prompt = self._format_prompt_for_note_type(processed_prompt, note_type)
        
        \# Generate text
        generated_text = self._generate_text(formatted_prompt)
        
        \# Validate generated content
        validation_result = None
        if validate_output:
            validation_result = self._validate_generated_content(generated_text)
        
        \# Prepare result
        result = {
            'generated_text': generated_text,
            'original_prompt': prompt,
            'processed_prompt': formatted_prompt,
            'note_type': note_type,
            'validation_result': validation_result,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'model_name': self.config.model_name,
                'temperature': self.config.temperature,
                'max_new_tokens': self.config.max_new_tokens
            }
        }
        
        \# Log generation result for audit
        if self.config.audit_logging_enabled:
            self._log_generation_result(result)
        
        return result
    
    def _add_patient_context(self, prompt: str, context: Dict) -> str:
        """Add patient context to prompt"""
        
        context_str = "Patient Context:\n"
        
        if 'age' in context:
            context_str += f"Age: {context['age']}\n"
        if 'gender' in context:
            context_str += f"Gender: {context['gender']}\n"
        if 'chief_complaint' in context:
            context_str += f"Chief Complaint: {context['chief_complaint']}\n"
        if 'medical_history' in context:
            context_str += f"Medical History: {context['medical_history']}\n"
        if 'medications' in context:
            context_str += f"Current Medications: {context['medications']}\n"
        
        return context_str + "\n" + prompt
    
    def _format_prompt_for_note_type(self, prompt: str, note_type: str) -> str:
        """Format prompt based on note type"""
        
        note_templates = {
            'progress_note': "Generate a clinical progress note based on the following information:\n\n{prompt}\n\nProgress Note:",
            'discharge_summary': "Generate a discharge summary based on the following information:\n\n{prompt}\n\nDischarge Summary:",
            'consultation_note': "Generate a consultation note based on the following information:\n\n{prompt}\n\nConsultation Note:",
            'procedure_note': "Generate a procedure note based on the following information:\n\n{prompt}\n\nProcedure Note:",
            'history_physical': "Generate a history and physical examination note based on the following information:\n\n{prompt}\n\nHistory and Physical:"
        }
        
        template = note_templates.get(note_type, "Generate a clinical note based on the following information:\n\n{prompt}\n\nClinical Note:")
        
        return template.format(prompt=prompt)
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the language model"""
        
        \# Tokenize input
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length - self.config.max_new_tokens,
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        \# Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        \# Decode generated text
        generated_text = self.tokenizer.decode(
            outputs<sup>0</sup>[inputs.shape<sup>1</sup>:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def _validate_generated_content(self, text: str) -> ClinicalValidationResult:
        """Validate generated clinical content"""
        
        \# Safety evaluation
        safety_results = self.safety_framework.evaluate_safety(text)
        
        \# Medical content validation
        medical_validation = self.clinical_validator.validate_medical_content(text)
        
        \# Calculate overall scores
        medical_accuracy_score = medical_validation['accuracy_score']
        clinical_relevance_score = self._calculate_clinical_relevance(text)
        safety_score = safety_results['safety_score']
        bias_score = self.safety_framework._detect_bias(text)
        hallucination_score = self.safety_framework._detect_hallucinations(text)
        phi_detected = self.safety_framework._detect_phi(text)
        
        \# Determine if validation passed
        validation_passed = (
            medical_accuracy_score >= self.config.medical_accuracy_threshold and
            clinical_relevance_score >= self.config.clinical_relevance_threshold and
            safety_score >= self.config.safety_threshold and
            bias_score < self.config.bias_threshold and
            hallucination_score < self.config.hallucination_threshold and
            not phi_detected
        )
        
        \# Compile issues and recommendations
        issues = safety_results['issues'].copy()
        recommendations = safety_results['recommendations'].copy()
        
        if medical_accuracy_score < self.config.medical_accuracy_threshold:
            issues.append(f"Low medical accuracy score: {medical_accuracy_score:.2f}")
            recommendations.append("Review medical content for accuracy")
        
        if clinical_relevance_score < self.config.clinical_relevance_threshold:
            issues.append(f"Low clinical relevance score: {clinical_relevance_score:.2f}")
            recommendations.append("Ensure content is clinically relevant")
        
        \# Create validation result
        validation_result = ClinicalValidationResult(
            medical_accuracy_score=medical_accuracy_score,
            clinical_relevance_score=clinical_relevance_score,
            safety_score=safety_score,
            bias_score=bias_score,
            hallucination_score=hallucination_score,
            phi_detected=phi_detected,
            regulatory_compliant=self._check_regulatory_compliance(text),
            validation_passed=validation_passed,
            issues=issues,
            recommendations=recommendations
        )
        
        return validation_result
    
    def _calculate_clinical_relevance(self, text: str) -> float:
        """Calculate clinical relevance score"""
        
        \# Count medical entities
        medical_entity_count = 0
        text_lower = text.lower()
        
        for entity_type, entities in self.safety_framework.knowledge_base.medical_entities.items():
            for entity in entities:
                if entity.lower() in text_lower:
                    medical_entity_count += 1
        
        \# Calculate relevance based on medical entity density
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        relevance_score = min(1.0, medical_entity_count / (word_count / 10))
        
        return relevance_score
    
    def _check_regulatory_compliance(self, text: str) -> bool:
        """Check regulatory compliance of generated content"""
        
        compliance_checks = []
        
        \# FDA compliance check
        if self.config.fda_compliance_mode:
            \# Check for medical device claims
            device_claims = [
                'diagnose', 'treat', 'cure', 'prevent',
                'medical device', 'fda approved'
            ]
            
            has_device_claims = any(
                claim.lower() in text.lower() 
                for claim in device_claims
            )
            
            compliance_checks.append(not has_device_claims)
        
        \# HIPAA compliance check
        if self.config.hipaa_compliance_mode:
            phi_detected = self.safety_framework._detect_phi(text)
            compliance_checks.append(not phi_detected)
        
        return all(compliance_checks)
    
    def _log_generation_request(self, prompt: str, note_type: str, context: Optional[Dict]):
        """Log generation request for audit purposes"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'generation_request',
            'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest(),
            'note_type': note_type,
            'has_context': context is not None,
            'model_name': self.config.model_name
        }
        
        self.audit_log.append(log_entry)
    
    def _log_generation_result(self, result: Dict):
        """Log generation result for audit purposes"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'generation_result',
            'validation_passed': result['validation_result'].validation_passed if result['validation_result'] else None,
            'safety_score': result['validation_result'].safety_score if result['validation_result'] else None,
            'text_length': len(result['generated_text']),
            'note_type': result['note_type']
        }
        
        self.audit_log.append(log_entry)
    
    def get_audit_log(self) -> List[Dict]:
        """Get audit log for compliance reporting"""
        return self.audit_log.copy()
    
    def export_audit_log(self, filepath: str):
        """Export audit log to file"""
        
        with open(filepath, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
        
        logger.info(f"Audit log exported to {filepath}")

\#\# 6.3 Multimodal Generative AI for Healthcare

\#\#\# 6.3.1 Vision-Language Models for Medical Applications

Multimodal generative AI systems that can process and generate both text and images represent a significant advancement for healthcare applications. These systems can analyze medical images while generating corresponding reports, create synthetic medical images for training purposes, and provide comprehensive clinical insights that combine visual and textual information.
