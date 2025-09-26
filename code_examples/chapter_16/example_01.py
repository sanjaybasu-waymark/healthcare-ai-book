"""
Chapter 16 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

"""
Advanced Medical Imaging AI System with Vision Transformers

This implementation provides a comprehensive framework for medical imaging AI
using Vision Transformers, 3D CNNs, and foundation models with clinical
integration capabilities and regulatory compliance features.

Author: Sanjay Basu MD PhD
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
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
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from torchvision.models import resnet50, densenet121
import monai
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandAffined, RandGaussianNoised, RandGaussianSmoothd,
    ToTensord, EnsureChannelFirstd
)
from monai.data import DataLoader as MonaiDataLoader, Dataset as MonaiDataset
from monai.networks.nets import ViT, SwinUNETR, UNETR
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import SimpleITK as sitk

warnings.filterwarnings('ignore')

\# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/medical-imaging-ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImagingModality(Enum):
    """Medical imaging modalities."""
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"
    FUNDUS = "fundus"
    OCT = "oct"
    HISTOPATHOLOGY = "histopathology"
    DERMATOLOGY = "dermatology"

class TaskType(Enum):
    """Medical imaging AI task types."""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    REGRESSION = "regression"
    MULTI_TASK = "multi_task"

class ModelArchitecture(Enum):
    """Model architecture types."""
    VIT = "vision_transformer"
    CNN_3D = "cnn_3d"
    SWIN_TRANSFORMER = "swin_transformer"
    UNETR = "unetr"
    FOUNDATION_MODEL = "foundation_model"

@dataclass
class ImagingConfig:
    """Configuration for medical imaging AI system."""
    modality: ImagingModality
    task_type: TaskType
    architecture: ModelArchitecture
    input_size: Tuple[int, ...]
    num_classes: int
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    uncertainty_estimation: bool = True
    multi_modal: bool = False
    clinical_integration: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'modality': self.modality.value,
            'task_type': self.task_type.value,
            'architecture': self.architecture.value,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'uncertainty_estimation': self.uncertainty_estimation,
            'multi_modal': self.multi_modal,
            'clinical_integration': self.clinical_integration
        }

class MedicalViT(nn.Module):
    """
    Medical Vision Transformer with clinical adaptations.
    
    This implementation extends the standard ViT architecture with features
    specifically designed for medical imaging applications.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        uncertainty_estimation: bool = True,
        clinical_features_dim: int = 0
    ):
        """Initialize Medical Vision Transformer."""
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.uncertainty_estimation = uncertainty_estimation
        self.clinical_features_dim = clinical_features_dim
        
        \# Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        \# Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        \# Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        \# Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        \# Clinical features integration
        if clinical_features_dim > 0:
            self.clinical_proj = nn.Linear(clinical_features_dim, embed_dim)
            self.clinical_norm = nn.LayerNorm(embed_dim)
        
        \# Classification head
        if uncertainty_estimation:
            \# Bayesian classification head
            self.head = BayesianLinear(embed_dim, num_classes)
        else:
            self.head = nn.Linear(embed_dim, num_classes)
        
        \# Attention visualization
        self.attention_maps = []
        
        \# Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        """Initialize individual module weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor, clinical_features: Optional[torch.Tensor] = None):
        """Extract features using transformer encoder."""
        B = x.shape<sup>0</sup>
        
        \# Patch embedding
        x = self.patch_embed(x)  \# (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  \# (B, num_patches, embed_dim)
        
        \# Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        \# Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        \# Integrate clinical features if provided
        if clinical_features is not None and self.clinical_features_dim > 0:
            clinical_embed = self.clinical_proj(clinical_features)
            clinical_embed = self.clinical_norm(clinical_embed)
            clinical_embed = clinical_embed.unsqueeze(1)  \# (B, 1, embed_dim)
            x = torch.cat((x, clinical_embed), dim=1)
        
        \# Apply transformer blocks
        self.attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x, return_attention=True)
            self.attention_maps.append(attn_weights)
        
        x = self.norm(x)
        
        return x[:, 0]  \# Return class token
    
    def forward(self, x: torch.Tensor, clinical_features: Optional[torch.Tensor] = None):
        """Forward pass."""
        features = self.forward_features(x, clinical_features)
        
        if self.uncertainty_estimation:
            \# Bayesian inference
            logits, uncertainty = self.head(features)
            return logits, uncertainty
        else:
            logits = self.head(features)
            return logits
    
    def get_attention_maps(self) -> List[torch.Tensor]:
        """Get attention maps for visualization."""
        return self.attention_maps

class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        """Initialize transformer block."""
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Forward pass."""
        if return_attention:
            attn_output, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attn_output)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn_weights
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        """Initialize multi-head attention."""
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Forward pass."""
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv<sup>0</sup>, qkv<sup>1</sup>, qkv<sup>2</sup>
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_weights = attn.clone() if return_attention else None
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn_weights
        return x

class MLP(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        """Initialize MLP."""
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        """Initialize drop path."""
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape<sup>0</sup>,) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class BayesianLinear(nn.Module):
    """Bayesian linear layer for uncertainty estimation."""
    
    def __init__(self, in_features: int, out_features: int, num_samples: int = 10):
        """Initialize Bayesian linear layer."""
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_samples = num_samples
        
        \# Weight parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        \# Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)
    
    def forward(self, x: torch.Tensor):
        """Forward pass with uncertainty estimation."""
        if self.training:
            \# Sample weights and biases
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
            
            output = F.linear(x, weight, bias)
            
            \# Estimate uncertainty (simplified)
            uncertainty = torch.mean(weight_sigma) + torch.mean(bias_sigma)
            
            return output, uncertainty
        else:
            \# Use mean weights for inference
            output = F.linear(x, self.weight_mu, self.bias_mu)
            
            \# Monte Carlo sampling for uncertainty
            outputs = []
            for _ in range(self.num_samples):
                weight_sigma = torch.log1p(torch.exp(self.weight_rho))
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                
                weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
                bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
                
                sample_output = F.linear(x, weight, bias)
                outputs.append(sample_output)
            
            outputs = torch.stack(outputs)
            uncertainty = torch.std(outputs, dim=0).mean()
            
            return output, uncertainty

class Medical3DCNN(nn.Module):
    """3D CNN for volumetric medical imaging."""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 32,
        dropout_rate: float = 0.2
    ):
        """Initialize 3D CNN."""
        super().__init__()
        
        \# Encoder
        self.conv1 = self._conv_block(in_channels, base_filters)
        self.conv2 = self._conv_block(base_filters, base_filters * 2)
        self.conv3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.conv4 = self._conv_block(base_filters * 4, base_filters * 8)
        
        \# Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        \# Classifier
        self.dropout = nn.Dropout3d(dropout_rate)
        self.classifier = nn.Linear(base_filters * 8, num_classes)
    
    def _conv_block(self, in_channels: int, out_channels: int):
        """Create 3D convolution block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

class MedicalImagingDataset(Dataset):
    """Dataset class for medical imaging data."""
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        modality: ImagingModality = ImagingModality.XRAY,
        task_type: TaskType = TaskType.CLASSIFICATION,
        clinical_features: Optional[List[str]] = None
    ):
        """Initialize dataset."""
        self.data_df = data_df
        self.transform = transform
        self.modality = modality
        self.task_type = task_type
        self.clinical_features = clinical_features or []
    
    def __len__(self):
        """Get dataset length."""
        return len(self.data_df)
    
    def __getitem__(self, idx: int):
        """Get dataset item."""
        row = self.data_df.iloc[idx]
        
        \# Load image
        image_path = row['image_path']
        image = self._load_image(image_path)
        
        \# Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = self.transform(image)
        
        \# Get label
        if self.task_type == TaskType.CLASSIFICATION:
            label = torch.tensor(row['label'], dtype=torch.long)
        elif self.task_type == TaskType.REGRESSION:
            label = torch.tensor(row['label'], dtype=torch.float32)
        else:
            label = torch.tensor(row['label'], dtype=torch.float32)
        
        \# Get clinical features if available
        clinical_data = None
        if self.clinical_features:
            clinical_data = torch.tensor(
                row[self.clinical_features].values.astype(np.float32),
                dtype=torch.float32
            )
        
        sample = {
            'image': image,
            'label': label,
            'patient_id': row.get('patient_id', ''),
            'image_path': image_path
        }
        
        if clinical_data is not None:
            sample['clinical_features'] = clinical_data
        
        return sample
    
    def _load_image(self, image_path: str):
        """Load image based on modality."""
        if self.modality in [ImagingModality.CT, ImagingModality.MRI]:
            \# Load DICOM or NIfTI
            if image_path.endswith('.dcm'):
                ds = pydicom.dcmread(image_path)
                image = ds.pixel_array.astype(np.float32)
            elif image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
                nii = nib.load(image_path)
                image = nii.get_fdata().astype(np.float32)
            else:
                image = np.load(image_path).astype(np.float32)
            
            \# Normalize
            image = (image - image.mean()) / (image.std() + 1e-8)
            
        else:
            \# Load 2D image
            if image_path.endswith('.dcm'):
                ds = pydicom.dcmread(image_path)
                image = ds.pixel_array.astype(np.float32)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = image.astype(np.float32) / 255.0
        
        return image

class MedicalImagingTrainer:
    """Trainer class for medical imaging AI models."""
    
    def __init__(self, config: ImagingConfig):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        \# Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        \# Best model tracking
        self.best_val_metric = 0.0
        self.best_model_state = None
        
        logger.info(f"Initialized trainer for {config.modality.value} {config.task_type.value}")
    
    def build_model(self):
        """Build model based on configuration."""
        if self.config.architecture == ModelArchitecture.VIT:
            self.model = MedicalViT(
                img_size=self.config.input_size<sup>0</sup>,
                patch_size=16,
                in_chans=self.config.input_size<sup>2</sup> if len(self.config.input_size) > 2 else 1,
                num_classes=self.config.num_classes,
                uncertainty_estimation=self.config.uncertainty_estimation
            )
        elif self.config.architecture == ModelArchitecture.CNN_3D:
            self.model = Medical3DCNN(
                in_channels=1,
                num_classes=self.config.num_classes
            )
        elif self.config.architecture == ModelArchitecture.SWIN_TRANSFORMER:
            self.model = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=True,
                num_classes=self.config.num_classes
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.config.architecture}")
        
        self.model = self.model.to(self.device)
        
        \# Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        \# Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )
        
        \# Setup loss function
        if self.config.task_type == TaskType.CLASSIFICATION:
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.task_type == TaskType.REGRESSION:
            self.criterion = nn.MSELoss()
        elif self.config.task_type == TaskType.SEGMENTATION:
            self.criterion = DiceLoss()
        
        logger.info(f"Built {self.config.architecture.value} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        for batch in train_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            clinical_features = batch.get('clinical_features')
            
            if clinical_features is not None:
                clinical_features = clinical_features.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    if self.config.uncertainty_estimation:
                        outputs, uncertainty = self.model(images, clinical_features)
                        loss = self.criterion(outputs, labels)
                        \# Add uncertainty regularization
                        loss += 0.01 * uncertainty.mean()
                    else:
                        outputs = self.model(images, clinical_features)
                        loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.config.uncertainty_estimation:
                    outputs, uncertainty = self.model(images, clinical_features)
                    loss = self.criterion(outputs, labels)
                    loss += 0.01 * uncertainty.mean()
                else:
                    outputs = self.model(images, clinical_features)
                    loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            \# Collect predictions for metrics
            if self.config.task_type == TaskType.CLASSIFICATION:
                predictions = torch.softmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        \# Calculate metrics
        metrics = {}
        if self.config.task_type == TaskType.CLASSIFICATION and len(all_predictions) > 0:
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            if self.config.num_classes == 2:
                auc = roc_auc_score(all_labels, all_predictions[:, 1])
                metrics['auc'] = auc
            
            predicted_classes = np.argmax(all_predictions, axis=1)
            accuracy = (predicted_classes == all_labels).mean()
            metrics['accuracy'] = accuracy
        
        return {'loss': avg_loss, **metrics}
    
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
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                clinical_features = batch.get('clinical_features')
                
                if clinical_features is not None:
                    clinical_features = clinical_features.to(self.device)
                
                if self.config.uncertainty_estimation:
                    outputs, uncertainty = self.model(images, clinical_features)
                    all_uncertainties.extend(uncertainty.cpu().numpy())
                else:
                    outputs = self.model(images, clinical_features)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
                
                \# Collect predictions for metrics
                if self.config.task_type == TaskType.CLASSIFICATION:
                    predictions = torch.softmax(outputs, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        \# Calculate metrics
        metrics = {}
        if self.config.task_type == TaskType.CLASSIFICATION and len(all_predictions) > 0:
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            if self.config.num_classes == 2:
                auc = roc_auc_score(all_labels, all_predictions[:, 1])
                metrics['auc'] = auc
            
            predicted_classes = np.argmax(all_predictions, axis=1)
            accuracy = (predicted_classes == all_labels).mean()
            metrics['accuracy'] = accuracy
            
            if len(all_uncertainties) > 0:
                metrics['mean_uncertainty'] = np.mean(all_uncertainties)
        
        return {'loss': avg_loss, **metrics}
    
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
            
            \# Update scheduler
            self.scheduler.step()
            
            \# Check for best model
            current_metric = val_metrics.get('auc', val_metrics.get('accuracy', -val_metrics['loss']))
            if current_metric > self.best_val_metric:
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
        
        \# Build model if not already built
        if self.model is None:
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Model loaded from {path}")

class MedicalImagingPipeline:
    """Complete medical imaging AI pipeline."""
    
    def __init__(self, config: ImagingConfig):
        """Initialize pipeline."""
        self.config = config
        self.trainer = MedicalImagingTrainer(config)
        self.data_transforms = self._setup_transforms()
        
        logger.info("Initialized medical imaging AI pipeline")
    
    def _setup_transforms(self):
        """Setup data transforms based on modality."""
        if self.config.modality in [ImagingModality.XRAY, ImagingModality.MAMMOGRAPHY]:
            train_transform = A.Compose([
                A.Resize(self.config.input_size<sup>0</sup>, self.config.input_size<sup>1</sup>),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
            
            val_transform = A.Compose([
                A.Resize(self.config.input_size<sup>0</sup>, self.config.input_size<sup>1</sup>),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
        
        elif self.config.modality == ImagingModality.FUNDUS:
            train_transform = A.Compose([
                A.Resize(self.config.input_size<sup>0</sup>, self.config.input_size<sup>1</sup>),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            val_transform = A.Compose([
                A.Resize(self.config.input_size<sup>0</sup>, self.config.input_size<sup>1</sup>),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        else:
            \# Default transforms
            train_transform = A.Compose([
                A.Resize(self.config.input_size<sup>0</sup>, self.config.input_size<sup>1</sup>),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
            
            val_transform = A.Compose([
                A.Resize(self.config.input_size<sup>0</sup>, self.config.input_size<sup>1</sup>),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
        
        return {'train': train_transform, 'val': val_transform}
    
    def prepare_data(self, data_df: pd.DataFrame, clinical_features: Optional[List[str]] = None):
        """Prepare data for training."""
        \# Split data
        train_df, val_df = train_test_split(
            data_df, test_size=0.2, random_state=42, 
            stratify=data_df['label'] if self.config.task_type == TaskType.CLASSIFICATION else None
        )
        
        \# Create datasets
        train_dataset = MedicalImagingDataset(
            train_df, 
            transform=self.data_transforms['train'],
            modality=self.config.modality,
            task_type=self.config.task_type,
            clinical_features=clinical_features
        )
        
        val_dataset = MedicalImagingDataset(
            val_df,
            transform=self.data_transforms['val'],
            modality=self.config.modality,
            task_type=self.config.task_type,
            clinical_features=clinical_features
        )
        
        \# Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Prepared data: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        
        return train_loader, val_loader
    
    def run_training(self, data_df: pd.DataFrame, clinical_features: Optional[List[str]] = None):
        """Run complete training pipeline."""
        \# Prepare data
        train_loader, val_loader = self.prepare_data(data_df, clinical_features)
        
        \# Build model
        self.trainer.build_model()
        
        \# Train model
        self.trainer.train(train_loader, val_loader)
        
        return self.trainer
    
    def evaluate_model(self, test_df: pd.DataFrame, clinical_features: Optional[List[str]] = None):
        """Evaluate trained model."""
        test_dataset = MedicalImagingDataset(
            test_df,
            transform=self.data_transforms['val'],
            modality=self.config.modality,
            task_type=self.config.task_type,
            clinical_features=clinical_features
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        \# Evaluate
        test_metrics = self.trainer.validate_epoch(test_loader)
        
        logger.info(f"Test metrics: {test_metrics}")
        
        return test_metrics

\# Example usage and demonstration
def create_sample_medical_imaging_data():
    """Create sample medical imaging data for demonstration."""
    np.random.seed(42)
    
    \# Create sample data
    n_samples = 1000
    data = {
        'patient_id': [f'patient_{i:04d}' for i in range(n_samples)],
        'image_path': [f'/data/images/image_{i:04d}.jpg' for i in range(n_samples)],
        'label': np.random.randint(0, 2, n_samples),
        'age': np.random.normal(65, 15, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'bmi': np.random.normal(25, 5, n_samples)
    }
    
    return pd.DataFrame(data)

def demonstrate_medical_imaging_ai():
    """Demonstrate medical imaging AI system."""
    print("Medical Imaging AI System Demonstration")
    print("=" * 50)
    
    \# Create configuration
    config = ImagingConfig(
        modality=ImagingModality.XRAY,
        task_type=TaskType.CLASSIFICATION,
        architecture=ModelArchitecture.VIT,
        input_size=(224, 224, 1),
        num_classes=2,
        batch_size=8,
        learning_rate=1e-4,
        num_epochs=5,  \# Reduced for demo
        uncertainty_estimation=True,
        clinical_integration=True
    )
    
    print(f"Configuration: {config.modality.value} {config.task_type.value}")
    print(f"Architecture: {config.architecture.value}")
    print(f"Input size: {config.input_size}")
    
    \# Create sample data
    data_df = create_sample_medical_imaging_data()
    clinical_features = ['age', 'sex', 'bmi']
    
    print(f"Dataset: {len(data_df)} samples")
    print(f"Clinical features: {clinical_features}")
    
    \# Initialize pipeline
    pipeline = MedicalImagingPipeline(config)
    
    print("\nPipeline initialized successfully")
    print("Note: This is a demonstration with synthetic data")
    print("In practice, you would:")
    print("1. Load real medical imaging data")
    print("2. Implement proper preprocessing")
    print("3. Train with appropriate validation")
    print("4. Perform clinical validation")
    print("5. Deploy with regulatory compliance")

if __name__ == "__main__":
    demonstrate_medical_imaging_ai()