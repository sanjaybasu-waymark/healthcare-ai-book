"""
Chapter 18 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

"""
Comprehensive Multimodal AI System for Healthcare

This implementation provides a complete framework for multimodal healthcare AI
including imaging, text, and structured data integration with advanced fusion
techniques, attention mechanisms, and clinical validation capabilities.

Author: Sanjay Basu MD PhD
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cv2
import pydicom
import nibabel as nib
from pathlib import Path
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import asyncio
import concurrent.futures
from collections import defaultdict
import pickle
from functools import lru_cache
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
)
from torchvision import transforms, models
import timm
from PIL import Image
import SimpleITK as sitk
from scipy import stats
from scipy.signal import butter, filtfilt
import h5py

warnings.filterwarnings('ignore')

\# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/multimodal-ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Types of data modalities."""
    IMAGING = "imaging"
    TEXT = "text"
    STRUCTURED = "structured"
    TEMPORAL = "temporal"
    GENOMIC = "genomic"
    PHYSIOLOGICAL = "physiological"

class FusionStrategy(Enum):
    """Multimodal fusion strategies."""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    INTERMEDIATE_FUSION = "intermediate_fusion"
    ATTENTION_FUSION = "attention_fusion"
    CROSS_MODAL_ATTENTION = "cross_modal_attention"

class TaskType(Enum):
    """Multimodal task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"

@dataclass
class MultimodalConfig:
    """Configuration for multimodal AI system."""
    modalities: List[ModalityType]
    fusion_strategy: FusionStrategy
    task_type: TaskType
    
    \# Model architecture
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.1
    
    \# Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    \# Modality-specific parameters
    image_size: Tuple[int, int] = (224, 224)
    max_text_length: int = 512
    num_structured_features: int = 50
    temporal_window: int = 24
    
    \# Advanced features
    use_uncertainty_estimation: bool = True
    use_attention_visualization: bool = True
    use_missing_modality_handling: bool = True
    use_temporal_alignment: bool = True
    
    \# Clinical integration
    enable_clinical_validation: bool = True
    use_clinical_knowledge: bool = True
    enable_interpretability: bool = True
    
    \# Computational
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    distributed_training: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'modalities': [m.value for m in self.modalities],
            'fusion_strategy': self.fusion_strategy.value,
            'task_type': self.task_type.value,
            'hidden_dim': self.hidden_dim,
            'num_attention_heads': self.num_attention_heads,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'gradient_clip_norm': self.gradient_clip_norm,
            'image_size': self.image_size,
            'max_text_length': self.max_text_length,
            'num_structured_features': self.num_structured_features,
            'temporal_window': self.temporal_window,
            'use_uncertainty_estimation': self.use_uncertainty_estimation,
            'use_attention_visualization': self.use_attention_visualization,
            'use_missing_modality_handling': self.use_missing_modality_handling,
            'use_temporal_alignment': self.use_temporal_alignment,
            'enable_clinical_validation': self.enable_clinical_validation,
            'use_clinical_knowledge': self.use_clinical_knowledge,
            'enable_interpretability': self.enable_interpretability,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'distributed_training': self.distributed_training
        }

class ImageEncoder(nn.Module):
    """Advanced image encoder with multiple architecture options."""
    
    def __init__(
        self,
        architecture: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 512,
        dropout_rate: float = 0.1
    ):
        """Initialize image encoder."""
        super().__init__()
        
        self.architecture = architecture
        self.output_dim = output_dim
        
        \# Load backbone
        if architecture.startswith("resnet"):
            self.backbone = models.__dict__[architecture](pretrained=pretrained)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif architecture.startswith("efficientnet"):
            self.backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0)
            backbone_dim = self.backbone.num_features
        elif architecture.startswith("vit"):
            self.backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0)
            backbone_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        \# Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        \# Attention pooling for variable-size inputs
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        logger.info(f"Initialized image encoder with {architecture} backbone")
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Forward pass."""
        batch_size = x.size(0)
        
        \# Extract features
        features = self.backbone(x)
        
        \# Project to common dimension
        projected = self.projection(features)
        
        \# Add sequence dimension for attention
        if len(projected.shape) == 2:
            projected = projected.unsqueeze(1)  \# (batch, 1, dim)
        
        \# Self-attention pooling
        attended, attention_weights = self.attention_pool(
            projected, projected, projected
        )
        
        \# Global pooling
        pooled = attended.mean(dim=1)  \# (batch, dim)
        
        if return_attention:
            return pooled, attention_weights
        return pooled

class TextEncoder(nn.Module):
    """Advanced text encoder with clinical language model support."""
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        output_dim: int = 512,
        max_length: int = 512,
        dropout_rate: float = 0.1,
        use_pooling: str = "attention"
    ):
        """Initialize text encoder."""
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_pooling = use_pooling
        
        \# Load pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        \# Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        \# Attention pooling
        if use_pooling == "attention":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=self.bert.config.hidden_size,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
        
        logger.info(f"Initialized text encoder with {model_name}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_attention: bool = False):
        """Forward pass."""
        \# BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        \# Pooling
        if self.use_pooling == "cls":
            pooled = sequence_output[:, 0]  \# CLS token
            attention_weights = None
        elif self.use_pooling == "mean":
            \# Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            pooled = torch.sum(sequence_output * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
            attention_weights = None
        elif self.use_pooling == "attention":
            \# Attention pooling
            attended, attention_weights = self.attention_pool(
                sequence_output, sequence_output, sequence_output,
                key_padding_mask=~attention_mask.bool()
            )
            pooled = attended.mean(dim=1)
        else:
            pooled = outputs.pooler_output
            attention_weights = None
        
        \# Project to common dimension
        projected = self.projection(pooled)
        
        if return_attention:
            return projected, attention_weights
        return projected

class StructuredDataEncoder(nn.Module):
    """Encoder for structured clinical data."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dims: List[int] = [256, 128],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        """Initialize structured data encoder."""
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        \# Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        \# Output layer
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.LayerNorm(output_dim)
        ])
        
        self.encoder = nn.Sequential(*layers)
        
        \# Residual connection
        if use_residual and input_dim == output_dim:
            self.residual_proj = nn.Identity()
        elif use_residual:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
        
        logger.info(f"Initialized structured data encoder: {input_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        encoded = self.encoder(x)
        
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
            encoded = encoded + residual
        
        return encoded

class TemporalEncoder(nn.Module):
    """Encoder for temporal/time-series data."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        use_attention: bool = True
    ):
        """Initialize temporal encoder."""
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        \# LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  \# Bidirectional
        
        \# Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
        
        \# Projection
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        logger.info(f"Initialized temporal encoder: {input_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False):
        """Forward pass."""
        \# LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        \# Attention pooling
        if self.use_attention:
            key_padding_mask = ~mask.bool() if mask is not None else None
            attended, attention_weights = self.attention(
                lstm_out, lstm_out, lstm_out,
                key_padding_mask=key_padding_mask
            )
            
            \# Masked mean pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand(attended.size()).float()
                pooled = torch.sum(attended * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
            else:
                pooled = attended.mean(dim=1)
        else:
            \# Use final hidden state
            pooled = h_n[-1]  \# Last layer, forward direction
            attention_weights = None
        
        \# Project to output dimension
        projected = self.projection(pooled)
        
        if return_attention:
            return projected, attention_weights
        return projected

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """Initialize cross-modal attention."""
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        \# Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False
    ):
        """Forward pass."""
        \# Cross-attention
        attended, attention_weights = self.attention(
            query, key, value
        )
        
        \# Residual connection and normalization
        query = self.norm1(query + self.dropout(attended))
        
        \# Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_out))
        
        if return_attention:
            return query, attention_weights
        return query

class MultimodalFusionModule(nn.Module):
    """Advanced multimodal fusion module."""
    
    def __init__(
        self,
        config: MultimodalConfig,
        modality_dims: Dict[str, int]
    ):
        """Initialize fusion module."""
        super().__init__()
        
        self.config = config
        self.modality_dims = modality_dims
        self.fusion_strategy = config.fusion_strategy
        self.hidden_dim = config.hidden_dim
        
        \# Modality projections
        self.modality_projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.modality_projections[modality] = nn.Sequential(
                nn.Linear(dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            )
        
        \# Fusion-specific components
        if self.fusion_strategy == FusionStrategy.EARLY_FUSION:
            self._setup_early_fusion()
        elif self.fusion_strategy == FusionStrategy.LATE_FUSION:
            self._setup_late_fusion()
        elif self.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            self._setup_attention_fusion()
        elif self.fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
            self._setup_cross_modal_attention()
        
        logger.info(f"Initialized fusion module with {self.fusion_strategy.value}")
    
    def _setup_early_fusion(self):
        """Setup early fusion components."""
        total_dim = len(self.modality_dims) * self.hidden_dim
        
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _setup_late_fusion(self):
        """Setup late fusion components."""
        self.modality_networks = nn.ModuleDict()
        
        for modality in self.modality_dims.keys():
            self.modality_networks[modality] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_rate)
            )
        
        \# Fusion network
        total_dim = len(self.modality_dims) * self.hidden_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _setup_attention_fusion(self):
        """Setup attention-based fusion."""
        self.attention_weights = nn.ModuleDict()
        
        for modality in self.modality_dims.keys():
            self.attention_weights[modality] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1)
            )
    
    def _setup_cross_modal_attention(self):
        """Setup cross-modal attention."""
        self.cross_attentions = nn.ModuleDict()
        
        modalities = list(self.modality_dims.keys())
        for i, mod_a in enumerate(modalities):
            for j, mod_b in enumerate(modalities):
                if i != j:
                    key = f"{mod_a}_to_{mod_b}"
                    self.cross_attentions[key] = CrossModalAttention(
                        embed_dim=self.hidden_dim,
                        num_heads=self.config.num_attention_heads,
                        dropout=self.config.dropout_rate
                    )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False
    ):
        """Forward pass."""
        \# Project all modalities to common dimension
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.modality_projections:
                projected_features[modality] = self.modality_projections[modality](features)
        
        \# Apply fusion strategy
        if self.fusion_strategy == FusionStrategy.EARLY_FUSION:
            return self._early_fusion(projected_features, return_attention)
        elif self.fusion_strategy == FusionStrategy.LATE_FUSION:
            return self._late_fusion(projected_features, return_attention)
        elif self.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            return self._attention_fusion(projected_features, return_attention)
        elif self.fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
            return self._cross_modal_attention(projected_features, return_attention)
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")
    
    def _early_fusion(self, features: Dict[str, torch.Tensor], return_attention: bool = False):
        """Early fusion implementation."""
        \# Concatenate all features
        concatenated = torch.cat(list(features.values()), dim=-1)
        
        \# Process through fusion network
        fused = self.fusion_network(concatenated)
        
        if return_attention:
            return fused, None
        return fused
    
    def _late_fusion(self, features: Dict[str, torch.Tensor], return_attention: bool = False):
        """Late fusion implementation."""
        \# Process each modality separately
        processed_features = []
        for modality, feat in features.items():
            processed = self.modality_networks[modality](feat)
            processed_features.append(processed)
        
        \# Concatenate and fuse
        concatenated = torch.cat(processed_features, dim=-1)
        fused = self.fusion_network(concatenated)
        
        if return_attention:
            return fused, None
        return fused
    
    def _attention_fusion(self, features: Dict[str, torch.Tensor], return_attention: bool = False):
        """Attention-based fusion implementation."""
        \# Compute attention weights for each modality
        attention_weights = {}
        attention_logits = []
        
        for modality, feat in features.items():
            weight_logit = self.attention_weights[modality](feat)
            attention_logits.append(weight_logit)
            attention_weights[modality] = weight_logit
        
        \# Softmax over modalities
        attention_logits = torch.cat(attention_logits, dim=-1)
        attention_probs = F.softmax(attention_logits, dim=-1)
        
        \# Weighted combination
        fused = torch.zeros_like(list(features.values())<sup>0</sup>)
        for i, (modality, feat) in enumerate(features.items()):
            weight = attention_probs[:, i:i+1]
            fused += weight * feat
        
        if return_attention:
            return fused, attention_weights
        return fused
    
    def _cross_modal_attention(self, features: Dict[str, torch.Tensor], return_attention: bool = False):
        """Cross-modal attention implementation."""
        enhanced_features = {}
        attention_maps = {}
        
        modalities = list(features.keys())
        
        \# Apply cross-modal attention
        for mod_a in modalities:
            enhanced_feat = features[mod_a].unsqueeze(1)  \# Add sequence dimension
            
            for mod_b in modalities:
                if mod_a != mod_b:
                    key = f"{mod_a}_to_{mod_b}"
                    if key in self.cross_attentions:
                        query = enhanced_feat
                        key_val = features[mod_b].unsqueeze(1)
                        
                        attended, attn_weights = self.cross_attentions[key](
                            query, key_val, key_val, return_attention=True
                        )
                        
                        enhanced_feat = enhanced_feat + attended
                        attention_maps[key] = attn_weights
            
            enhanced_features[mod_a] = enhanced_feat.squeeze(1)
        
        \# Final fusion
        fused = torch.stack(list(enhanced_features.values())).mean(dim=0)
        
        if return_attention:
            return fused, attention_maps
        return fused

class MultimodalAI(nn.Module):
    """Complete multimodal AI system."""
    
    def __init__(self, config: MultimodalConfig, num_classes: int = 2):
        """Initialize multimodal AI system."""
        super().__init__()
        
        self.config = config
        self.num_classes = num_classes
        self.modalities = config.modalities
        
        \# Modality encoders
        self.encoders = nn.ModuleDict()
        modality_dims = {}
        
        if ModalityType.IMAGING in self.modalities:
            self.encoders['imaging'] = ImageEncoder(
                architecture="resnet50",
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate
            )
            modality_dims['imaging'] = config.hidden_dim
        
        if ModalityType.TEXT in self.modalities:
            self.encoders['text'] = TextEncoder(
                model_name="emilyalsentzer/Bio_ClinicalBERT",
                output_dim=config.hidden_dim,
                max_length=config.max_text_length,
                dropout_rate=config.dropout_rate
            )
            modality_dims['text'] = config.hidden_dim
        
        if ModalityType.STRUCTURED in self.modalities:
            self.encoders['structured'] = StructuredDataEncoder(
                input_dim=config.num_structured_features,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate
            )
            modality_dims['structured'] = config.hidden_dim
        
        if ModalityType.TEMPORAL in self.modalities:
            self.encoders['temporal'] = TemporalEncoder(
                input_dim=config.num_structured_features,  \# Assuming same as structured
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate
            )
            modality_dims['temporal'] = config.hidden_dim
        
        \# Fusion module
        self.fusion = MultimodalFusionModule(config, modality_dims)
        
        \# Classification/regression head
        if config.task_type == TaskType.CLASSIFICATION:
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dim // 2, num_classes)
            )
        elif config.task_type == TaskType.REGRESSION:
            self.regressor = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dim // 2, 1)
            )
        
        \# Uncertainty estimation
        if config.use_uncertainty_estimation:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 4, 1),
                nn.Sigmoid()
            )
        
        \# Missing modality handling
        if config.use_missing_modality_handling:
            self.modality_presence = nn.ModuleDict()
            for modality in modality_dims.keys():
                self.modality_presence[modality] = nn.Parameter(
                    torch.randn(config.hidden_dim) * 0.1
                )
        
        logger.info(f"Initialized multimodal AI system with {len(self.modalities)} modalities")
    
    def forward(
        self,
        batch: Dict[str, Any],
        return_attention: bool = False,
        return_uncertainty: bool = False
    ):
        """Forward pass."""
        modality_features = {}
        attention_maps = {}
        
        \# Encode each modality
        if 'imaging' in batch and ModalityType.IMAGING in self.modalities:
            if return_attention:
                img_feat, img_attn = self.encoders['imaging'](batch['imaging'], return_attention=True)
                attention_maps['imaging'] = img_attn
            else:
                img_feat = self.encoders['imaging'](batch['imaging'])
            modality_features['imaging'] = img_feat
        
        if 'text' in batch and ModalityType.TEXT in self.modalities:
            if return_attention:
                text_feat, text_attn = self.encoders['text'](
                    batch['text']['input_ids'],
                    batch['text']['attention_mask'],
                    return_attention=True
                )
                attention_maps['text'] = text_attn
            else:
                text_feat = self.encoders['text'](
                    batch['text']['input_ids'],
                    batch['text']['attention_mask']
                )
            modality_features['text'] = text_feat
        
        if 'structured' in batch and ModalityType.STRUCTURED in self.modalities:
            struct_feat = self.encoders['structured'](batch['structured'])
            modality_features['structured'] = struct_feat
        
        if 'temporal' in batch and ModalityType.TEMPORAL in self.modalities:
            temp_mask = batch.get('temporal_mask')
            if return_attention:
                temp_feat, temp_attn = self.encoders['temporal'](
                    batch['temporal'], temp_mask, return_attention=True
                )
                attention_maps['temporal'] = temp_attn
            else:
                temp_feat = self.encoders['temporal'](batch['temporal'], temp_mask)
            modality_features['temporal'] = temp_feat
        
        \# Handle missing modalities
        if self.config.use_missing_modality_handling:
            for modality in self.encoders.keys():
                if modality not in modality_features:
                    \# Use learned missing modality representation
                    batch_size = list(modality_features.values())<sup>0</sup>.size(0)
                    missing_repr = self.modality_presence[modality].unsqueeze(0).expand(
                        batch_size, -1
                    )
                    modality_features[modality] = missing_repr
        
        \# Fusion
        if return_attention:
            fused_features, fusion_attention = self.fusion(
                modality_features, return_attention=True
            )
            if fusion_attention:
                attention_maps['fusion'] = fusion_attention
        else:
            fused_features = self.fusion(modality_features)
        
        \# Task-specific head
        if self.config.task_type == TaskType.CLASSIFICATION:
            logits = self.classifier(fused_features)
            outputs = {'logits': logits}
        elif self.config.task_type == TaskType.REGRESSION:
            predictions = self.regressor(fused_features)
            outputs = {'predictions': predictions}
        else:
            outputs = {'features': fused_features}
        
        \# Uncertainty estimation
        if return_uncertainty and self.config.use_uncertainty_estimation:
            uncertainty = self.uncertainty_head(fused_features)
            outputs['uncertainty'] = uncertainty
        
        \# Attention maps
        if return_attention:
            outputs['attention_maps'] = attention_maps
        
        return outputs

class MultimodalDataset(Dataset):
    """Dataset for multimodal data."""
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        config: MultimodalConfig,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None
    ):
        """Initialize dataset."""
        self.data_df = data_df
        self.config = config
        self.transform = transform
        self.tokenizer = tokenizer
        
        \# Setup transforms
        if self.transform is None and ModalityType.IMAGING in config.modalities:
            self.transform = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        \# Setup tokenizer
        if self.tokenizer is None and ModalityType.TEXT in config.modalities:
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    def __len__(self):
        """Get dataset length."""
        return len(self.data_df)
    
    def __getitem__(self, idx: int):
        """Get dataset item."""
        row = self.data_df.iloc[idx]
        sample = {}
        
        \# Load imaging data
        if ModalityType.IMAGING in self.config.modalities and 'image_path' in row:
            try:
                image = Image.open(row['image_path']).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                sample['imaging'] = image
            except Exception as e:
                logger.warning(f"Failed to load image {row['image_path']}: {e}")
                \# Create dummy image
                sample['imaging'] = torch.zeros(3, *self.config.image_size)
        
        \# Load text data
        if ModalityType.TEXT in self.config.modalities and 'text' in row:
            text = str(row['text'])
            if self.tokenizer:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_text_length,
                    return_tensors='pt'
                )
                sample['text'] = {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                }
        
        \# Load structured data
        if ModalityType.STRUCTURED in self.config.modalities:
            \# Extract structured features (assuming they're in columns)
            structured_cols = [col for col in row.index if col.startswith('feature_')]
            if structured_cols:
                structured_data = torch.tensor(
                    row[structured_cols].values.astype(np.float32),
                    dtype=torch.float32
                )
            else:
                \# Create dummy structured data
                structured_data = torch.randn(self.config.num_structured_features)
            sample['structured'] = structured_data
        
        \# Load temporal data
        if ModalityType.TEMPORAL in self.config.modalities and 'temporal_data' in row:
            try:
                temporal_data = np.load(row['temporal_data'])
                temporal_tensor = torch.tensor(temporal_data, dtype=torch.float32)
                sample['temporal'] = temporal_tensor
                
                \# Create mask for valid time points
                temporal_mask = torch.ones(temporal_tensor.size(0), dtype=torch.bool)
                sample['temporal_mask'] = temporal_mask
            except Exception as e:
                logger.warning(f"Failed to load temporal data: {e}")
                \# Create dummy temporal data
                sample['temporal'] = torch.randn(self.config.temporal_window, self.config.num_structured_features)
                sample['temporal_mask'] = torch.ones(self.config.temporal_window, dtype=torch.bool)
        
        \# Add labels
        if 'label' in row:
            if self.config.task_type == TaskType.CLASSIFICATION:
                sample['label'] = torch.tensor(row['label'], dtype=torch.long)
            elif self.config.task_type == TaskType.REGRESSION:
                sample['label'] = torch.tensor(row['label'], dtype=torch.float32)
        
        \# Add metadata
        sample['patient_id'] = row.get('patient_id', '')
        sample['sample_id'] = idx
        
        return sample

class MultimodalTrainer:
    """Trainer for multimodal AI systems."""
    
    def __init__(self, config: MultimodalConfig, num_classes: int = 2):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device(config.device)
        self.model = MultimodalAI(config, num_classes).to(self.device)
        
        \# Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )
        
        \# Loss function
        if config.task_type == TaskType.CLASSIFICATION:
            self.criterion = nn.CrossEntropyLoss()
        elif config.task_type == TaskType.REGRESSION:
            self.criterion = nn.MSELoss()
        
        \# Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        \# Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        \# Best model tracking
        self.best_val_metric = 0.0 if config.task_type == TaskType.CLASSIFICATION else float('inf')
        self.best_model_state = None
        
        logger.info("Initialized multimodal trainer")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        for batch in train_loader:
            \# Move to device
            batch = self._move_to_device(batch)
            
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        batch,
                        return_uncertainty=self.config.use_uncertainty_estimation
                    )
                    loss = self._compute_loss(outputs, batch)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    batch,
                    return_uncertainty=self.config.use_uncertainty_estimation
                )
                loss = self._compute_loss(outputs, batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            \# Collect predictions for metrics
            self._collect_predictions(outputs, batch, all_predictions, all_labels, all_uncertainties)
        
        avg_loss = total_loss / num_batches
        metrics = self._compute_metrics(all_predictions, all_labels, all_uncertainties)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                \# Move to device
                batch = self._move_to_device(batch)
                
                outputs = self.model(
                    batch,
                    return_uncertainty=self.config.use_uncertainty_estimation
                )
                loss = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
                
                \# Collect predictions for metrics
                self._collect_predictions(outputs, batch, all_predictions, all_labels, all_uncertainties)
        
        avg_loss = total_loss / num_batches
        metrics = self._compute_metrics(all_predictions, all_labels, all_uncertainties)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in value.items()}
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _compute_loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss."""
        if 'label' not in batch:
            return torch.tensor(0.0, device=self.device)
        
        labels = batch['label']
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            loss = self.criterion(outputs['logits'], labels)
        elif self.config.task_type == TaskType.REGRESSION:
            loss = self.criterion(outputs['predictions'].squeeze(), labels)
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        \# Add uncertainty regularization
        if 'uncertainty' in outputs:
            uncertainty_reg = 0.01 * outputs['uncertainty'].mean()
            loss += uncertainty_reg
        
        return loss
    
    def _collect_predictions(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        all_predictions: List,
        all_labels: List,
        all_uncertainties: List
    ):
        """Collect predictions for metrics computation."""
        if 'label' not in batch:
            return
        
        labels = batch['label'].cpu().numpy()
        all_labels.extend(labels)
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            predictions = torch.softmax(outputs['logits'], dim=1).cpu().numpy()
            all_predictions.extend(predictions)
        elif self.config.task_type == TaskType.REGRESSION:
            predictions = outputs['predictions'].cpu().numpy()
            all_predictions.extend(predictions)
        
        if 'uncertainty' in outputs:
            uncertainties = outputs['uncertainty'].cpu().numpy()
            all_uncertainties.extend(uncertainties)
    
    def _compute_metrics(
        self,
        predictions: List,
        labels: List,
        uncertainties: List
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        if not predictions or not labels:
            return {}
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        metrics = {}
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            if predictions.ndim > 1 and predictions.shape<sup>1</sup> > 1:
                \# Multi-class classification
                predicted_classes = np.argmax(predictions, axis=1)
                accuracy = (predicted_classes == labels).mean()
                metrics['accuracy'] = accuracy
                
                if predictions.shape<sup>1</sup> == 2:
                    \# Binary classification
                    auc = roc_auc_score(labels, predictions[:, 1])
                    metrics['auc'] = auc
        
        elif self.config.task_type == TaskType.REGRESSION:
            mse = mean_squared_error(labels, predictions)
            mae = mean_absolute_error(labels, predictions)
            r2 = r2_score(labels, predictions)
            
            metrics.update({
                'mse': mse,
                'mae': mae,
                'r2': r2
            })
        
        if uncertainties:
            uncertainties = np.array(uncertainties)
            metrics['mean_uncertainty'] = uncertainties.mean()
            metrics['std_uncertainty'] = uncertainties.std()
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model."""
        logger.info("Starting multimodal training...")
        
        for epoch in range(self.config.num_epochs):
            \# Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            self.train_metrics.append(train_metrics)
            
            \# Validate
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            \# Update scheduler
            self.scheduler.step()
            
            \# Check for best model
            if self.config.task_type == TaskType.CLASSIFICATION:
                current_metric = val_metrics.get('auc', val_metrics.get('accuracy', 0))
                is_better = current_metric > self.best_val_metric
            else:
                current_metric = val_metrics.get('mse', val_metrics['loss'])
                is_better = current_metric < self.best_val_metric
            
            if is_better:
                self.best_val_metric = current_metric
                self.best_model_state = self.model.state_dict().copy()
            
            \# Log progress
            if epoch % 10 == 0:
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
            'config': self.config.to_dict(),
            'best_val_metric': self.best_val_metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Model loaded from {path}")

\# Example usage and demonstration
def create_sample_multimodal_data():
    """Create sample multimodal data for demonstration."""
    np.random.seed(42)
    
    n_samples = 1000
    data = {
        'patient_id': [f'patient_{i:04d}' for i in range(n_samples)],
        'image_path': [f'/data/images/image_{i:04d}.jpg' for i in range(n_samples)],
        'text': [f'Patient presents with symptoms including chest pain and shortness of breath. Medical history includes hypertension.' for _ in range(n_samples)],
        'label': np.random.randint(0, 2, n_samples)
    }
    
    \# Add structured features
    for i in range(20):
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)

def demonstrate_multimodal_ai():
    """Demonstrate multimodal AI system."""
    print("Multimodal AI System Demonstration")
    print("=" * 50)
    
    \# Create configuration
    config = MultimodalConfig(
        modalities=[ModalityType.IMAGING, ModalityType.TEXT, ModalityType.STRUCTURED],
        fusion_strategy=FusionStrategy.CROSS_MODAL_ATTENTION,
        task_type=TaskType.CLASSIFICATION,
        hidden_dim=256,
        num_attention_heads=8,
        batch_size=8,
        num_epochs=5,  \# Reduced for demo
        use_uncertainty_estimation=True,
        use_attention_visualization=True,
        enable_clinical_validation=True
    )
    
    print(f"Configuration:")
    print(f"  Modalities: {[m.value for m in config.modalities]}")
    print(f"  Fusion strategy: {config.fusion_strategy.value}")
    print(f"  Task type: {config.task_type.value}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    
    \# Create sample data
    data_df = create_sample_multimodal_data()
    
    print(f"\nDataset: {len(data_df)} samples")
    print(f"Features: {[col for col in data_df.columns if col.startswith('feature_')][:5]}...")
    
    \# Initialize trainer
    trainer = MultimodalTrainer(config, num_classes=2)
    
    print(f"\nModel parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")
    
    print("\nMultimodal AI system initialized successfully")
    print("Note: This is a demonstration with synthetic data")
    print("In practice, you would:")
    print("1. Load real multimodal healthcare data")
    print("2. Implement proper preprocessing pipelines")
    print("3. Train with clinical validation")
    print("4. Deploy with EHR integration")
    print("5. Monitor performance across modalities")

if __name__ == "__main__":
    demonstrate_multimodal_ai()