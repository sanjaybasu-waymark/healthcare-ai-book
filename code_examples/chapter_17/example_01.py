"""
Chapter 17 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

"""
Comprehensive Clinical NLP System with Transformers

This implementation provides a complete framework for clinical natural language
processing using state-of-the-art transformer architectures, named entity
recognition, question answering, and scalable processing pipelines.

Author: Sanjay Basu MD PhD
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForTokenClassification,
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification,
    BertTokenizer, BertModel, BertForTokenClassification,
    RobertaTokenizer, RobertaModel, RobertaForTokenClassification,
    pipeline, Trainer, TrainingArguments, DataCollatorForTokenClassification,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.model_selection import train_test_split
import re
import json
import sqlite3
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import aiohttp
import concurrent.futures
from collections import defaultdict, Counter
import spacy
import scispacy
from scispacy.linking import EntityLinker
import medspacy
from medspacy.ner import TargetRule
from medspacy.context import ConTextRule
import warnings
import hashlib
import pickle
from functools import lru_cache
import redis
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import faiss
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import cryptography
from cryptography.fernet import Fernet

warnings.filterwarnings('ignore')

\# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/clinical-nlp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClinicalNLPTask(Enum):
    """Clinical NLP task types."""
    NAMED_ENTITY_RECOGNITION = "ner"
    RELATION_EXTRACTION = "relation_extraction"
    QUESTION_ANSWERING = "question_answering"
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    DE_IDENTIFICATION = "de_identification"
    CLINICAL_CODING = "clinical_coding"
    ADVERSE_EVENT_DETECTION = "adverse_event_detection"

class DocumentType(Enum):
    """Clinical document types."""
    PHYSICIAN_NOTE = "physician_note"
    RADIOLOGY_REPORT = "radiology_report"
    PATHOLOGY_REPORT = "pathology_report"
    NURSING_NOTE = "nursing_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    OPERATIVE_NOTE = "operative_note"
    CONSULTATION_NOTE = "consultation_note"
    EMERGENCY_NOTE = "emergency_note"

class EntityType(Enum):
    """Clinical entity types."""
    DISEASE = "DISEASE"
    MEDICATION = "MEDICATION"
    PROCEDURE = "PROCEDURE"
    ANATOMY = "ANATOMY"
    SYMPTOM = "SYMPTOM"
    LAB_VALUE = "LAB_VALUE"
    DOSAGE = "DOSAGE"
    FREQUENCY = "FREQUENCY"
    DURATION = "DURATION"
    PERSON = "PERSON"
    DATE = "DATE"
    LOCATION = "LOCATION"

@dataclass
class ClinicalNLPConfig:
    """Configuration for clinical NLP system."""
    task: ClinicalNLPTask
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    distributed: bool = False
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    \# Privacy and security
    enable_de_identification: bool = True
    encryption_key: Optional[str] = None
    audit_logging: bool = True
    
    \# Scalability
    use_caching: bool = True
    cache_backend: str = "redis"
    use_elasticsearch: bool = True
    
    \# Clinical integration
    integrate_umls: bool = True
    use_clinical_context: bool = True
    enable_uncertainty_quantification: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task': self.task.value,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'device': self.device,
            'distributed': self.distributed,
            'mixed_precision': self.mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'enable_de_identification': self.enable_de_identification,
            'audit_logging': self.audit_logging,
            'use_caching': self.use_caching,
            'cache_backend': self.cache_backend,
            'use_elasticsearch': self.use_elasticsearch,
            'integrate_umls': self.integrate_umls,
            'use_clinical_context': self.use_clinical_context,
            'enable_uncertainty_quantification': self.enable_uncertainty_quantification
        }

class ClinicalTextPreprocessor:
    """Advanced clinical text preprocessing with HIPAA compliance."""
    
    def __init__(self, config: ClinicalNLPConfig):
        """Initialize preprocessor."""
        self.config = config
        self.analyzer = AnalyzerEngine() if config.enable_de_identification else None
        self.anonymizer = AnonymizerEngine() if config.enable_de_identification else None
        self.encryption_key = config.encryption_key
        self.fernet = Fernet(config.encryption_key.encode()) if config.encryption_key else None
        
        \# Load clinical NLP models
        try:
            self.nlp = spacy.load("en_core_sci_sm")
            self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        except OSError:
            logger.warning("SciSpacy models not found. Using basic spaCy model.")
            self.nlp = spacy.load("en_core_web_sm")
        
        \# Clinical context rules
        self.setup_clinical_context()
        
        logger.info("Clinical text preprocessor initialized")
    
    def setup_clinical_context(self):
        """Setup clinical context processing."""
        if self.config.use_clinical_context:
            try:
                \# Add medspacy components
                self.nlp.add_pipe("medspacy_context")
                self.nlp.add_pipe("medspacy_sectionizer")
                
                \# Add custom context rules
                context_rules = [
                    ConTextRule("no evidence of", "NEGATED_EXISTENCE", direction="forward"),
                    ConTextRule("denies", "NEGATED_EXISTENCE", direction="forward"),
                    ConTextRule("rule out", "POSSIBLE_EXISTENCE", direction="forward"),
                    ConTextRule("possible", "POSSIBLE_EXISTENCE", direction="forward"),
                    ConTextRule("history of", "HISTORICAL", direction="forward"),
                    ConTextRule("family history", "FAMILY_HISTORY", direction="forward")
                ]
                
                context = self.nlp.get_pipe("medspacy_context")
                context.add(context_rules)
                
            except Exception as e:
                logger.warning(f"Could not setup clinical context: {e}")
    
    def de_identify_text(self, text: str) -> Tuple[str, List[Dict]]:
        """De-identify clinical text using Presidio."""
        if not self.analyzer or not self.anonymizer:
            return text, []
        
        try:
            \# Analyze text for PII
            analyzer_results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE_TIME", 
                         "LOCATION", "MEDICAL_LICENSE", "US_SSN"]
            )
            
            \# Anonymize text
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results
            )
            
            return anonymized_result.text, analyzer_results
            
        except Exception as e:
            logger.error(f"De-identification failed: {e}")
            return text, []
    
    def encrypt_text(self, text: str) -> str:
        """Encrypt text for secure storage."""
        if not self.fernet:
            return text
        
        try:
            encrypted_text = self.fernet.encrypt(text.encode())
            return encrypted_text.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return text
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt text for processing."""
        if not self.fernet:
            return encrypted_text
        
        try:
            decrypted_text = self.fernet.decrypt(encrypted_text.encode())
            return decrypted_text.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_text
    
    def clean_clinical_text(self, text: str) -> str:
        """Clean and normalize clinical text."""
        \# Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        \# Normalize common clinical abbreviations
        abbreviation_map = {
            r'\bpt\b': 'patient',
            r'\bhx\b': 'history',
            r'\bdx\b': 'diagnosis',
            r'\btx\b': 'treatment',
            r'\brx\b': 'prescription',
            r'\bs/p\b': 'status post',
            r'\bc/o\b': 'complains of',
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without'
        }
        
        for abbrev, expansion in abbreviation_map.items():
            text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)
        
        \# Remove PHI patterns (additional safety)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)  \# SSN
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)  \# Phone
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)  \# Email
        
        return text.strip()
    
    def extract_clinical_sections(self, text: str) -> Dict[str, str]:
        """Extract clinical sections from text."""
        sections = {}
        
        \# Common clinical section headers
        section_patterns = {
            'chief_complaint': r'(?:chief complaint|cc):\s*(.*?)(?=\n[A-Z]|\n\n|$)',
            'history_present_illness': r'(?:history of present illness|hpi):\s*(.*?)(?=\n[A-Z]|\n\n|$)',
            'past_medical_history': r'(?:past medical history|pmh):\s*(.*?)(?=\n[A-Z]|\n\n|$)',
            'medications': r'(?:medications|meds):\s*(.*?)(?=\n[A-Z]|\n\n|$)',
            'allergies': r'(?:allergies|nkda):\s*(.*?)(?=\n[A-Z]|\n\n|$)',
            'physical_exam': r'(?:physical exam|pe):\s*(.*?)(?=\n[A-Z]|\n\n|$)',
            'assessment_plan': r'(?:assessment and plan|a&p|assessment|plan):\s*(.*?)(?=\n[A-Z]|\n\n|$)'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        return sections
    
    def process_text(self, text: str, document_type: DocumentType = DocumentType.PHYSICIAN_NOTE) -> Dict[str, Any]:
        """Comprehensive text processing pipeline."""
        start_time = datetime.now()
        
        \# De-identification
        if self.config.enable_de_identification:
            text, phi_entities = self.de_identify_text(text)
        else:
            phi_entities = []
        
        \# Clean text
        cleaned_text = self.clean_clinical_text(text)
        
        \# Extract sections
        sections = self.extract_clinical_sections(cleaned_text)
        
        \# Process with spaCy
        doc = self.nlp(cleaned_text)
        
        \# Extract entities and context
        entities = []
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'confidence', 1.0)
            }
            
            \# Add clinical context if available
            if hasattr(ent, '_'):
                entity_info['is_negated'] = ent._.is_negated
                entity_info['is_historical'] = ent._.is_historical
                entity_info['is_hypothetical'] = ent._.is_hypothetical
                entity_info['is_family'] = ent._.is_family
            
            entities.append(entity_info)
        
        \# Extract sentences and tokens
        sentences = [sent.text for sent in doc.sents]
        tokens = [token.text for token in doc]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sections': sections,
            'entities': entities,
            'sentences': sentences,
            'tokens': tokens,
            'phi_entities': phi_entities,
            'document_type': document_type.value,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        \# Audit logging
        if self.config.audit_logging:
            self.log_processing_event(result)
        
        return result
    
    def log_processing_event(self, result: Dict[str, Any]):
        """Log processing event for audit trail."""
        audit_entry = {
            'timestamp': result['timestamp'],
            'document_type': result['document_type'],
            'text_length': len(result['original_text']),
            'entities_found': len(result['entities']),
            'phi_entities_found': len(result['phi_entities']),
            'processing_time': result['processing_time']
        }
        
        logger.info(f"Clinical text processed: {audit_entry}")

class ClinicalNER(nn.Module):
    """Advanced clinical named entity recognition model."""
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_labels: int = 9,  \# BIO tags for 4 entity types + O
        dropout_rate: float = 0.1,
        use_crf: bool = True,
        use_uncertainty: bool = True
    ):
        """Initialize clinical NER model."""
        super().__init__()
        
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.use_uncertainty = use_uncertainty
        
        \# Load pre-trained clinical BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        \# Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        \# CRF layer for sequence labeling
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        
        \# Uncertainty estimation
        if use_uncertainty:
            self.uncertainty_head = nn.Linear(self.bert.config.hidden_size, 1)
        
        \# Clinical knowledge integration
        self.clinical_embeddings = nn.Embedding(1000, self.bert.config.hidden_size)  \# For clinical concepts
        
        logger.info(f"Initialized clinical NER model with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        clinical_concepts: Optional[torch.Tensor] = None
    ):
        """Forward pass."""
        \# BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        \# Integrate clinical concepts if provided
        if clinical_concepts is not None:
            clinical_embeds = self.clinical_embeddings(clinical_concepts)
            sequence_output = sequence_output + clinical_embeds
        
        \# Classification
        logits = self.classifier(sequence_output)
        
        \# Uncertainty estimation
        uncertainty = None
        if self.use_uncertainty:
            uncertainty = torch.sigmoid(self.uncertainty_head(sequence_output))
        
        loss = None
        if labels is not None:
            if self.use_crf:
                \# CRF loss
                loss = -self.crf(logits, labels, mask=attention_mask.bool())
            else:
                \# Standard cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
        
        \# Predictions
        if self.use_crf:
            predictions = self.crf.decode(logits, mask=attention_mask.bool())
        else:
            predictions = torch.argmax(logits, dim=-1)
        
        return {
            'loss': loss,
            'logits': logits,
            'predictions': predictions,
            'uncertainty': uncertainty
        }

class CRF(nn.Module):
    """Conditional Random Field for sequence labeling."""
    
    def __init__(self, num_tags: int, batch_first: bool = False):
        """Initialize CRF."""
        super().__init__()
        
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        \# Transition parameters
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        \# Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor = None):
        """Compute CRF log likelihood."""
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        \# Compute log likelihood
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        
        return numerator - denominator
    
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor = None):
        """Viterbi decoding."""
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        return self._viterbi_decode(emissions, mask)
    
    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor):
        """Compute score for given tag sequence."""
        batch_size, seq_length = tags.shape
        
        \# Start transition score
        score = self.start_transitions[tags[:, 0]]
        
        \# Emission scores
        for i in range(seq_length):
            score += emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1) * mask[:, i]
        
        \# Transition scores
        for i in range(1, seq_length):
            score += self.transitions[tags[:, i-1], tags[:, i]] * mask[:, i]
        
        \# End transition score
        last_tag_indices = mask.sum(1) - 1
        score += self.end_transitions[tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)]
        
        return score
    
    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor):
        """Compute normalizer using forward algorithm."""
        batch_size, seq_length, num_tags = emissions.shape
        
        \# Initialize forward variables
        alpha = self.start_transitions + emissions[:, 0]
        
        \# Forward pass
        for i in range(1, seq_length):
            alpha_t = []
            for tag in range(num_tags):
                emit_score = emissions[:, i, tag]
                trans_score = alpha + self.transitions[:, tag]
                alpha_t.append(torch.logsumexp(trans_score, dim=1) + emit_score)
            
            alpha = torch.stack(alpha_t, dim=1)
            alpha = alpha * mask[:, i].unsqueeze(1) + alpha * (1 - mask[:, i].unsqueeze(1))
        
        \# Add end transitions
        alpha += self.end_transitions
        
        return torch.logsumexp(alpha, dim=1)
    
    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor):
        """Viterbi decoding algorithm."""
        batch_size, seq_length, num_tags = emissions.shape
        
        \# Initialize
        score = self.start_transitions + emissions[:, 0]
        history = []
        
        \# Forward pass
        for i in range(1, seq_length):
            score_t = []
            path_t = []
            
            for tag in range(num_tags):
                broadcast_score = score + self.transitions[:, tag]
                best_score, best_path = torch.max(broadcast_score, dim=1)
                score_t.append(best_score + emissions[:, i, tag])
                path_t.append(best_path)
            
            score = torch.stack(score_t, dim=1)
            history.append(torch.stack(path_t, dim=1))
        
        \# Add end transitions
        score += self.end_transitions
        
        \# Backward pass
        best_score, best_last_tag = torch.max(score, dim=1)
        
        \# Decode best path
        best_path = [best_last_tag]
        for hist in reversed(history):
            best_last_tag = hist.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_path.append(best_last_tag)
        
        best_path.reverse()
        
        return torch.stack(best_path, dim=1)

class ClinicalQuestionAnswering(nn.Module):
    """Clinical question answering system with evidence grading."""
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 512,
        use_evidence_grading: bool = True
    ):
        """Initialize clinical QA model."""
        super().__init__()
        
        self.max_length = max_length
        self.use_evidence_grading = use_evidence_grading
        
        \# Load pre-trained model
        self.bert = AutoModel.from_pretrained(model_name)
        
        \# QA heads
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)  \# Start and end positions
        
        \# Evidence grading
        if use_evidence_grading:
            self.evidence_classifier = nn.Linear(self.bert.config.hidden_size, 4)  \# A, B, C, D grades
        
        \# Clinical relevance scoring
        self.relevance_scorer = nn.Linear(self.bert.config.hidden_size, 1)
        
        logger.info("Initialized clinical question answering model")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None
    ):
        """Forward pass."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        \# QA predictions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        \# Evidence grading
        evidence_logits = None
        if self.use_evidence_grading:
            evidence_logits = self.evidence_classifier(pooled_output)
        
        \# Clinical relevance
        relevance_score = torch.sigmoid(self.relevance_scorer(pooled_output))
        
        \# Compute losses
        total_loss = 0
        
        if start_positions is not None and end_positions is not None:
            \# QA loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2
            total_loss += qa_loss
        
        if evidence_labels is not None and self.use_evidence_grading:
            \# Evidence grading loss
            evidence_loss_fct = nn.CrossEntropyLoss()
            evidence_loss = evidence_loss_fct(evidence_logits, evidence_labels)
            total_loss += evidence_loss
        
        return {
            'loss': total_loss if total_loss > 0 else None,
            'start_logits': start_logits,
            'end_logits': end_logits,
            'evidence_logits': evidence_logits,
            'relevance_score': relevance_score
        }

class ClinicalNLPDataset(Dataset):
    """Dataset class for clinical NLP tasks."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        task: ClinicalNLPTask,
        max_length: int = 512,
        label_map: Optional[Dict[str, int]] = None
    ):
        """Initialize dataset."""
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length
        self.label_map = label_map or {}
    
    def __len__(self):
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """Get dataset item."""
        item = self.data[idx]
        
        if self.task == ClinicalNLPTask.NAMED_ENTITY_RECOGNITION:
            return self._get_ner_item(item)
        elif self.task == ClinicalNLPTask.QUESTION_ANSWERING:
            return self._get_qa_item(item)
        elif self.task == ClinicalNLPTask.TEXT_CLASSIFICATION:
            return self._get_classification_item(item)
        else:
            raise ValueError(f"Unsupported task: {self.task}")
    
    def _get_ner_item(self, item: Dict[str, Any]):
        """Get NER dataset item."""
        text = item['text']
        entities = item.get('entities', [])
        
        \# Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        \# Create labels
        labels = ['O'] * len(encoding['input_ids']<sup>0</sup>)
        
        for entity in entities:
            start_char = entity['start']
            end_char = entity['end']
            entity_type = entity['label']
            
            \# Map character positions to token positions
            start_token = encoding.char_to_token(start_char)
            end_token = encoding.char_to_token(end_char - 1)
            
            if start_token is not None and end_token is not None:
                labels[start_token] = f'B-{entity_type}'
                for i in range(start_token + 1, end_token + 1):
                    labels[i] = f'I-{entity_type}'
        
        \# Convert labels to indices
        label_ids = [self.label_map.get(label, 0) for label in labels]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def _get_qa_item(self, item: Dict[str, Any]):
        """Get QA dataset item."""
        question = item['question']
        context = item['context']
        answer = item.get('answer', {})
        
        \# Tokenize question and context
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        \# Find answer positions
        start_position = 0
        end_position = 0
        
        if answer and 'text' in answer and 'answer_start' in answer:
            answer_start = answer['answer_start']
            answer_text = answer['text']
            
            \# Map character positions to token positions
            start_position = encoding.char_to_token(answer_start)
            end_position = encoding.char_to_token(answer_start + len(answer_text) - 1)
            
            if start_position is None:
                start_position = 0
            if end_position is None:
                end_position = 0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }
    
    def _get_classification_item(self, item: Dict[str, Any]):
        """Get classification dataset item."""
        text = item['text']
        label = item.get('label', 0)
        
        \# Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ClinicalNLPTrainer:
    """Trainer for clinical NLP models."""
    
    def __init__(self, config: ClinicalNLPConfig):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        \# Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        \# Best model tracking
        self.best_val_metric = 0.0
        self.best_model_state = None
        
        logger.info(f"Initialized trainer for {config.task.value}")
    
    def build_model(self):
        """Build model based on task."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        if self.config.task == ClinicalNLPTask.NAMED_ENTITY_RECOGNITION:
            \# Create label map
            entity_types = ['DISEASE', 'MEDICATION', 'PROCEDURE', 'ANATOMY']
            labels = ['O'] + [f'{prefix}-{entity}' for entity in entity_types for prefix in ['B', 'I']]
            self.label_map = {label: idx for idx, label in enumerate(labels)}
            
            self.model = ClinicalNER(
                model_name=self.config.model_name,
                num_labels=len(labels),
                use_uncertainty=self.config.enable_uncertainty_quantification
            )
        
        elif self.config.task == ClinicalNLPTask.QUESTION_ANSWERING:
            self.model = ClinicalQuestionAnswering(
                model_name=self.config.model_name,
                max_length=self.config.max_length
            )
        
        elif self.config.task == ClinicalNLPTask.TEXT_CLASSIFICATION:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=2  \# Binary classification by default
            )
        
        else:
            raise ValueError(f"Unsupported task: {self.config.task}")
        
        self.model = self.model.to(self.device)
        
        \# Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        \# Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.num_epochs * 1000  \# Approximate
        )
        
        logger.info(f"Built {self.config.task.value} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        for batch in train_loader:
            \# Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            \# Collect predictions for metrics
            if self.config.task == ClinicalNLPTask.TEXT_CLASSIFICATION:
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        \# Calculate metrics
        metrics = {'loss': avg_loss}
        if self.config.task == ClinicalNLPTask.TEXT_CLASSIFICATION and len(all_predictions) > 0:
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            metrics.update({'accuracy': accuracy, 'f1': f1})
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                \# Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
                
                \# Collect predictions for metrics
                if self.config.task == ClinicalNLPTask.TEXT_CLASSIFICATION:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        \# Calculate metrics
        metrics = {'loss': avg_loss}
        if self.config.task == ClinicalNLPTask.TEXT_CLASSIFICATION and len(all_predictions) > 0:
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            metrics.update({'accuracy': accuracy, 'f1': f1})
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model."""
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            \# Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            self.train_metrics.append(train_metrics)
            
            \# Validate
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            \# Check for best model
            current_metric = val_metrics.get('f1', val_metrics.get('accuracy', -val_metrics['loss']))
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_model_state = self.model.state_dict().copy()
            
            \# Log progress
            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch}/{self.config.num_epochs}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Metric: {current_metric:.4f}"
                )
        
        \# Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        logger.info(f"Training completed. Best validation metric: {self.best_val_metric:.4f}")
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'config': self.config.to_dict(),
            'label_map': getattr(self, 'label_map', {}),
            'best_val_metric': self.best_val_metric
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        \# Build model if not already built
        if self.model is None:
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        self.label_map = checkpoint.get('label_map', {})
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        
        logger.info(f"Model loaded from {path}")

class ClinicalNLPPipeline:
    """Complete clinical NLP pipeline with scalability and integration."""
    
    def __init__(self, config: ClinicalNLPConfig):
        """Initialize pipeline."""
        self.config = config
        self.preprocessor = ClinicalTextPreprocessor(config)
        self.trainer = ClinicalNLPTrainer(config)
        
        \# Caching
        if config.use_caching:
            self.cache = redis.Redis(host='localhost', port=6379, db=0) if config.cache_backend == 'redis' else {}
        
        \# Elasticsearch for document search
        if config.use_elasticsearch:
            self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        
        \# Sentence transformer for semantic search
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Initialized clinical NLP pipeline")
    
    def process_document(self, text: str, document_type: DocumentType = DocumentType.PHYSICIAN_NOTE) -> Dict[str, Any]:
        """Process a single clinical document."""
        \# Check cache
        if self.config.use_caching:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            cached_result = self.cache.get(cache_key) if isinstance(self.cache, redis.Redis) else self.cache.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result) if isinstance(cached_result, str) else cached_result
        
        \# Process text
        result = self.preprocessor.process_text(text, document_type)
        
        \# Cache result
        if self.config.use_caching:
            if isinstance(self.cache, redis.Redis):
                self.cache.setex(cache_key, 3600, json.dumps(result, default=str))
            else:
                self.cache[cache_key] = result
        
        return result
    
    def batch_process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple documents in batch."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for doc in documents:
                future = executor.submit(
                    self.process_document,
                    doc['text'],
                    DocumentType(doc.get('document_type', 'physician_note'))
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Document processing failed: {e}")
                    results.append({'error': str(e)})
        
        return results
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search clinical documents using semantic similarity."""
        if not self.config.use_elasticsearch:
            logger.warning("Elasticsearch not enabled")
            return []
        
        try:
            \# Encode query
            query_embedding = self.sentence_transformer.encode([query])
            
            \# Search using Elasticsearch
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "sections.*"]
                    }
                },
                "size": top_k
            }
            
            response = self.es.search(index="clinical_documents", body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'document_id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    def extract_clinical_insights(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract clinical insights from processed documents."""
        insights = {
            'total_documents': len(documents),
            'entity_counts': defaultdict(int),
            'common_medications': Counter(),
            'common_procedures': Counter(),
            'common_diagnoses': Counter(),
            'sentiment_distribution': defaultdict(int),
            'processing_stats': {
                'total_processing_time': 0,
                'average_processing_time': 0,
                'total_entities': 0,
                'total_phi_entities': 0
            }
        }
        
        total_processing_time = 0
        total_entities = 0
        total_phi_entities = 0
        
        for doc in documents:
            if 'error' in doc:
                continue
            
            \# Entity statistics
            entities = doc.get('entities', [])
            total_entities += len(entities)
            
            for entity in entities:
                entity_type = entity['label']
                insights['entity_counts'][entity_type] += 1
                
                if entity_type == 'MEDICATION':
                    insights['common_medications'][entity['text']] += 1
                elif entity_type == 'PROCEDURE':
                    insights['common_procedures'][entity['text']] += 1
                elif entity_type == 'DISEASE':
                    insights['common_diagnoses'][entity['text']] += 1
            
            \# PHI statistics
            phi_entities = doc.get('phi_entities', [])
            total_phi_entities += len(phi_entities)
            
            \# Processing time
            processing_time = doc.get('processing_time', 0)
            total_processing_time += processing_time
        
        \# Calculate averages
        insights['processing_stats']['total_processing_time'] = total_processing_time
        insights['processing_stats']['average_processing_time'] = total_processing_time / len(documents) if documents else 0
        insights['processing_stats']['total_entities'] = total_entities
        insights['processing_stats']['total_phi_entities'] = total_phi_entities
        
        \# Convert counters to lists for JSON serialization
        insights['common_medications'] = insights['common_medications'].most_common(10)
        insights['common_procedures'] = insights['common_procedures'].most_common(10)
        insights['common_diagnoses'] = insights['common_diagnoses'].most_common(10)
        
        return insights

\# Example usage and demonstration
def create_sample_clinical_data():
    """Create sample clinical data for demonstration."""
    sample_texts = [
        """
        CHIEF COMPLAINT: Chest pain and shortness of breath.
        
        HISTORY OF PRESENT ILLNESS: The patient is a 65-year-old male with a history of 
        hypertension and diabetes mellitus who presents with acute onset chest pain. 
        The pain started 2 hours ago and is described as crushing and substernal. 
        Associated symptoms include shortness of breath and diaphoresis.
        
        PAST MEDICAL HISTORY: Hypertension, diabetes mellitus type 2, hyperlipidemia.
        
        MEDICATIONS: Metformin 1000mg twice daily, lisinopril 10mg daily, atorvastatin 40mg daily.
        
        PHYSICAL EXAM: Blood pressure 160/90, heart rate 110, respiratory rate 22. 
        Cardiovascular exam reveals tachycardia with no murmurs. Lungs are clear bilaterally.
        
        ASSESSMENT AND PLAN: Acute coronary syndrome. Will obtain EKG, cardiac enzymes, 
        and chest X-ray. Start aspirin and consider cardiac catheterization.
        """,
        """
        RADIOLOGY REPORT
        
        EXAMINATION: Chest CT with contrast
        
        CLINICAL HISTORY: 45-year-old female with persistent cough and weight loss.
        
        FINDINGS: There is a 3.2 cm mass in the right upper lobe with spiculated margins. 
        Multiple enlarged mediastinal lymph nodes are present. No pleural effusion. 
        No evidence of metastatic disease in the visualized portions of the liver and adrenals.
        
        IMPRESSION: Right upper lobe mass suspicious for malignancy with mediastinal 
        lymphadenopathy. Recommend tissue sampling for definitive diagnosis.
        """,
        """
        DISCHARGE SUMMARY
        
        PATIENT: John Smith, DOB: 01/15/1960
        
        ADMISSION DIAGNOSIS: Acute myocardial infarction
        DISCHARGE DIAGNOSIS: ST-elevation myocardial infarction, status post primary PCI
        
        HOSPITAL COURSE: The patient underwent emergent cardiac catheterization which 
        revealed 100% occlusion of the LAD. Successful PCI with drug-eluting stent placement 
        was performed. Post-procedure course was uncomplicated.
        
        DISCHARGE MEDICATIONS:
        1. Aspirin 81mg daily
        2. Clopidogrel 75mg daily
        3. Metoprolol 25mg twice daily
        4. Atorvastatin 80mg daily
        
        FOLLOW-UP: Cardiology in 1 week, primary care in 2 weeks.
        """
    ]
    
    data = []
    for i, text in enumerate(sample_texts):
        data.append({
            'document_id': f'doc_{i+1}',
            'text': text,
            'document_type': ['physician_note', 'radiology_report', 'discharge_summary'][i],
            'patient_id': f'patient_{i+1}'
        })
    
    return data

def demonstrate_clinical_nlp():
    """Demonstrate clinical NLP system."""
    print("Clinical NLP System Demonstration")
    print("=" * 50)
    
    \# Create configuration
    config = ClinicalNLPConfig(
        task=ClinicalNLPTask.NAMED_ENTITY_RECOGNITION,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=512,
        batch_size=4,
        num_epochs=3,  \# Reduced for demo
        enable_de_identification=True,
        use_clinical_context=True,
        enable_uncertainty_quantification=True
    )
    
    print(f"Configuration: {config.task.value}")
    print(f"Model: {config.model_name}")
    print(f"Privacy protection: {config.enable_de_identification}")
    
    \# Create sample data
    documents = create_sample_clinical_data()
    
    print(f"\nProcessing {len(documents)} clinical documents...")
    
    \# Initialize pipeline
    pipeline = ClinicalNLPPipeline(config)
    
    \# Process documents
    results = pipeline.batch_process_documents(documents)
    
    print(f"Processed {len(results)} documents")
    
    \# Extract insights
    insights = pipeline.extract_clinical_insights(results)
    
    print(f"\nClinical Insights:")
    print(f"Total entities found: {insights['processing_stats']['total_entities']}")
    print(f"Average processing time: {insights['processing_stats']['average_processing_time']:.3f}s")
    print(f"Entity type distribution: {dict(insights['entity_counts'])}")
    
    if insights['common_medications']:
        print(f"Common medications: {insights['common_medications'][:3]}")
    
    print("\nNote: This is a demonstration with synthetic data")
    print("In practice, you would:")
    print("1. Load real clinical documents")
    print("2. Implement proper HIPAA compliance")
    print("3. Train with clinical validation")
    print("4. Deploy with EHR integration")
    print("5. Monitor performance continuously")

if __name__ == "__main__":
    demonstrate_clinical_nlp()