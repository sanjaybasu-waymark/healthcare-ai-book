"""
Chapter 19 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

"""
Comprehensive Federated Learning System for Healthcare

This implementation provides a complete framework for federated learning
in healthcare environments with privacy preservation, secure aggregation,
and clinical validation capabilities.

Author: Sanjay Basu MD PhD
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
from collections import defaultdict, OrderedDict
import pickle
import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import threading
import queue
import time
import copy
import random
from pathlib import Path

warnings.filterwarnings('ignore')

\# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/federated-learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FederatedAlgorithm(Enum):
    """Federated learning algorithms."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"
    SCAFFOLD = "scaffold"
    FEDOPT = "fedopt"

class AggregationStrategy(Enum):
    """Aggregation strategies."""
    WEIGHTED_AVERAGE = "weighted_average"
    SIMPLE_AVERAGE = "simple_average"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    BULYAN = "bulyan"

class PrivacyMechanism(Enum):
    """Privacy preservation mechanisms."""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"

@dataclass
class FederatedConfig:
    """Configuration for federated learning system."""
    \# Algorithm settings
    algorithm: FederatedAlgorithm = FederatedAlgorithm.FEDAVG
    aggregation_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE
    privacy_mechanism: PrivacyMechanism = PrivacyMechanism.DIFFERENTIAL_PRIVACY
    
    \# Training parameters
    num_rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 5
    local_batch_size: int = 32
    local_learning_rate: float = 0.01
    global_learning_rate: float = 1.0
    
    \# Algorithm-specific parameters
    fedprox_mu: float = 0.01  \# Proximal term coefficient
    scaffold_lr: float = 1.0  \# SCAFFOLD learning rate
    fednova_momentum: float = 0.0  \# FedNova momentum
    
    \# Privacy parameters
    dp_epsilon: float = 1.0  \# Differential privacy epsilon
    dp_delta: float = 1e-5   \# Differential privacy delta
    dp_clip_norm: float = 1.0  \# Gradient clipping norm
    
    \# Communication and security
    max_communication_rounds: int = 1000
    communication_timeout: int = 300  \# seconds
    use_secure_communication: bool = True
    use_model_compression: bool = True
    compression_ratio: float = 0.1
    
    \# Robustness and fault tolerance
    byzantine_tolerance: bool = True
    max_byzantine_clients: int = 2
    client_dropout_rate: float = 0.1
    
    \# Clinical validation
    enable_clinical_validation: bool = True
    validation_frequency: int = 10  \# rounds
    early_stopping_patience: int = 20
    min_improvement: float = 0.001
    
    \# Compliance and auditing
    enable_audit_logging: bool = True
    require_client_authentication: bool = True
    data_governance_compliance: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm': self.algorithm.value,
            'aggregation_strategy': self.aggregation_strategy.value,
            'privacy_mechanism': self.privacy_mechanism.value,
            'num_rounds': self.num_rounds,
            'clients_per_round': self.clients_per_round,
            'local_epochs': self.local_epochs,
            'local_batch_size': self.local_batch_size,
            'local_learning_rate': self.local_learning_rate,
            'global_learning_rate': self.global_learning_rate,
            'fedprox_mu': self.fedprox_mu,
            'scaffold_lr': self.scaffold_lr,
            'fednova_momentum': self.fednova_momentum,
            'dp_epsilon': self.dp_epsilon,
            'dp_delta': self.dp_delta,
            'dp_clip_norm': self.dp_clip_norm,
            'max_communication_rounds': self.max_communication_rounds,
            'communication_timeout': self.communication_timeout,
            'use_secure_communication': self.use_secure_communication,
            'use_model_compression': self.use_model_compression,
            'compression_ratio': self.compression_ratio,
            'byzantine_tolerance': self.byzantine_tolerance,
            'max_byzantine_clients': self.max_byzantine_clients,
            'client_dropout_rate': self.client_dropout_rate,
            'enable_clinical_validation': self.enable_clinical_validation,
            'validation_frequency': self.validation_frequency,
            'early_stopping_patience': self.early_stopping_patience,
            'min_improvement': self.min_improvement,
            'enable_audit_logging': self.enable_audit_logging,
            'require_client_authentication': self.require_client_authentication,
            'data_governance_compliance': self.data_governance_compliance
        }

class SecureCommunication:
    """Secure communication module for federated learning."""
    
    def __init__(self, use_encryption: bool = True):
        """Initialize secure communication."""
        self.use_encryption = use_encryption
        
        if use_encryption:
            \# Generate RSA key pair for asymmetric encryption
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            
            \# Generate symmetric key for data encryption
            self.symmetric_key = Fernet.generate_key()
            self.cipher = Fernet(self.symmetric_key)
        
        logger.info("Initialized secure communication module")
    
    def encrypt_model_update(self, model_update: Dict[str, torch.Tensor]) -> bytes:
        """Encrypt model update for secure transmission."""
        if not self.use_encryption:
            return pickle.dumps(model_update)
        
        \# Serialize model update
        serialized_update = pickle.dumps(model_update)
        
        \# Encrypt with symmetric key
        encrypted_update = self.cipher.encrypt(serialized_update)
        
        return encrypted_update
    
    def decrypt_model_update(self, encrypted_update: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt received model update."""
        if not self.use_encryption:
            return pickle.loads(encrypted_update)
        
        \# Decrypt with symmetric key
        decrypted_update = self.cipher.decrypt(encrypted_update)
        
        \# Deserialize model update
        model_update = pickle.loads(decrypted_update)
        
        return model_update
    
    def generate_client_token(self, client_id: str) -> str:
        """Generate authentication token for client."""
        timestamp = str(int(time.time()))
        message = f"{client_id}:{timestamp}"
        
        \# Create HMAC signature
        signature = hmac.new(
            self.symmetric_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = f"{message}:{signature}"
        return base64.b64encode(token.encode()).decode()
    
    def verify_client_token(self, token: str, client_id: str) -> bool:
        """Verify client authentication token."""
        try:
            decoded_token = base64.b64decode(token.encode()).decode()
            parts = decoded_token.split(':')
            
            if len(parts) != 3:
                return False
            
            received_client_id, timestamp, signature = parts
            
            \# Verify client ID
            if received_client_id != client_id:
                return False
            
            \# Verify timestamp (token valid for 1 hour)
            current_time = int(time.time())
            token_time = int(timestamp)
            if current_time - token_time > 3600:
                return False
            
            \# Verify signature
            message = f"{received_client_id}:{timestamp}"
            expected_signature = hmac.new(
                self.symmetric_key,
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return False

class DifferentialPrivacy:
    """Differential privacy mechanisms for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, clip_norm: float = 1.0):
        """Initialize differential privacy."""
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        
        logger.info(f"Initialized differential privacy: ε={epsilon}, δ={delta}")
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity."""
        clipped_gradients = {}
        
        \# Calculate total gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        \# Clip if necessary
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))
        
        for name, grad in gradients.items():
            clipped_gradients[name] = grad * clip_coef
        
        return clipped_gradients
    
    def add_noise(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise for differential privacy."""
        noisy_gradients = {}
        
        \# Calculate noise scale
        noise_scale = self.clip_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        for name, grad in gradients.items():
            noise = torch.normal(0, noise_scale, size=grad.shape, device=grad.device)
            noisy_gradients[name] = grad + noise
        
        return noisy_gradients
    
    def privatize_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to gradients."""
        \# Clip gradients
        clipped_gradients = self.clip_gradients(gradients)
        
        \# Add noise
        private_gradients = self.add_noise(clipped_gradients)
        
        return private_gradients

class ModelCompression:
    """Model compression for efficient communication."""
    
    def __init__(self, compression_ratio: float = 0.1):
        """Initialize model compression."""
        self.compression_ratio = compression_ratio
        
        logger.info(f"Initialized model compression: ratio={compression_ratio}")
    
    def compress_model_update(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress model update using top-k sparsification."""
        compressed_update = {}
        
        for name, tensor in model_update.items():
            \# Flatten tensor
            flat_tensor = tensor.flatten()
            
            \# Calculate number of elements to keep
            num_elements = len(flat_tensor)
            num_keep = max(1, int(num_elements * self.compression_ratio))
            
            \# Get top-k elements by magnitude
            _, top_indices = torch.topk(torch.abs(flat_tensor), num_keep)
            
            \# Create sparse representation
            sparse_values = flat_tensor[top_indices]
            
            \# Store compressed representation
            compressed_update[name] = {
                'values': sparse_values,
                'indices': top_indices,
                'shape': tensor.shape,
                'original_size': num_elements
            }
        
        return compressed_update
    
    def decompress_model_update(self, compressed_update: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Decompress model update."""
        decompressed_update = {}
        
        for name, compressed_data in compressed_update.items():
            \# Extract compressed data
            values = compressed_data['values']
            indices = compressed_data['indices']
            shape = compressed_data['shape']
            original_size = compressed_data['original_size']
            
            \# Reconstruct sparse tensor
            flat_tensor = torch.zeros(original_size, device=values.device)
            flat_tensor[indices] = values
            
            \# Reshape to original shape
            decompressed_update[name] = flat_tensor.reshape(shape)
        
        return decompressed_update

class ByzantineRobustness:
    """Byzantine robustness mechanisms."""
    
    def __init__(self, max_byzantine_clients: int = 2):
        """Initialize Byzantine robustness."""
        self.max_byzantine_clients = max_byzantine_clients
        
        logger.info(f"Initialized Byzantine robustness: max_byzantine={max_byzantine_clients}")
    
    def krum_aggregation(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        num_byzantine: int
    ) -> Dict[str, torch.Tensor]:
        """Krum aggregation for Byzantine robustness."""
        num_clients = len(client_updates)
        
        if num_clients <= 2 * num_byzantine:
            logger.warning("Not enough clients for Byzantine robustness")
            return self._simple_average(client_updates)
        
        \# Flatten all updates
        flattened_updates = []
        for update in client_updates:
            flattened = torch.cat([tensor.flatten() for tensor in update.values()])
            flattened_updates.append(flattened)
        
        \# Calculate pairwise distances
        distances = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = torch.norm(flattened_updates[i] - flattened_updates[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        \# Calculate Krum scores
        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            \# Sum of distances to closest n - f - 2 clients
            closest_distances, _ = torch.topk(
                distances[i], 
                num_clients - num_byzantine - 2, 
                largest=False
            )
            scores[i] = closest_distances.sum()
        
        \# Select client with minimum score
        selected_client = torch.argmin(scores).item()
        
        return client_updates[selected_client]
    
    def trimmed_mean_aggregation(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation."""
        if not client_updates:
            return {}
        
        \# Number of clients to trim from each end
        num_trim = int(len(client_updates) * trim_ratio / 2)
        
        aggregated_update = {}
        
        \# Get parameter names from first update
        param_names = list(client_updates<sup>0</sup>.keys())
        
        for param_name in param_names:
            \# Collect parameter values from all clients
            param_values = [update[param_name] for update in client_updates]
            param_tensor = torch.stack(param_values)
            
            \# Sort along client dimension
            sorted_tensor, _ = torch.sort(param_tensor, dim=0)
            
            \# Trim extreme values
            if num_trim > 0:
                trimmed_tensor = sorted_tensor[num_trim:-num_trim]
            else:
                trimmed_tensor = sorted_tensor
            
            \# Calculate mean
            aggregated_update[param_name] = trimmed_tensor.mean(dim=0)
        
        return aggregated_update
    
    def _simple_average(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Simple average aggregation fallback."""
        if not client_updates:
            return {}
        
        aggregated_update = {}
        param_names = list(client_updates<sup>0</sup>.keys())
        
        for param_name in param_names:
            param_values = [update[param_name] for update in client_updates]
            aggregated_update[param_name] = torch.stack(param_values).mean(dim=0)
        
        return aggregated_update

class FederatedClient:
    """Federated learning client."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: FederatedConfig,
        device: str = "cpu"
    ):
        """Initialize federated client."""
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)
        
        \# Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.local_learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        \# Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        \# Privacy mechanisms
        if config.privacy_mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
            self.dp = DifferentialPrivacy(
                epsilon=config.dp_epsilon,
                delta=config.dp_delta,
                clip_norm=config.dp_clip_norm
            )
        else:
            self.dp = None
        
        \# Communication
        self.secure_comm = SecureCommunication(config.use_secure_communication)
        
        \# Compression
        if config.use_model_compression:
            self.compressor = ModelCompression(config.compression_ratio)
        else:
            self.compressor = None
        
        \# SCAFFOLD control variates
        if config.algorithm == FederatedAlgorithm.SCAFFOLD:
            self.control_variate = {
                name: torch.zeros_like(param)
                for name, param in self.model.named_parameters()
            }
        
        \# Training history
        self.training_history = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        logger.info(f"Initialized federated client: {client_id}")
    
    def local_train(
        self,
        global_model_state: Dict[str, torch.Tensor],
        round_num: int
    ) -> Dict[str, Any]:
        """Perform local training."""
        \# Load global model
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
        \# Store initial model for FedProx
        if self.config.algorithm == FederatedAlgorithm.FEDPROX:
            initial_model_state = copy.deepcopy(global_model_state)
        
        \# Training metrics
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        \# Local training loop
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                \# Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                \# Add FedProx regularization
                if self.config.algorithm == FederatedAlgorithm.FEDPROX:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        proximal_term += torch.norm(param - initial_model_state[name]) ** 2
                    loss += (self.config.fedprox_mu / 2) * proximal_term
                
                \# Backward pass
                loss.backward()
                
                \# SCAFFOLD correction
                if self.config.algorithm == FederatedAlgorithm.SCAFFOLD:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param.grad += self.control_variate[name]
                
                self.optimizer.step()
                
                \# Update metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(data)
        
        \# Calculate training metrics
        avg_loss = total_loss / (self.config.local_epochs * len(self.train_loader))
        accuracy = correct_predictions / total_samples
        
        \# Validation
        val_loss, val_accuracy = self._validate()
        
        \# Update training history
        self.training_history['rounds'].append(round_num)
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['train_accuracy'].append(accuracy)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_accuracy'].append(val_accuracy)
        
        \# Prepare model update
        model_update = {}
        for name, param in self.model.named_parameters():
            model_update[name] = param.data - global_model_state[name]
        
        \# Apply differential privacy
        if self.dp:
            model_update = self.dp.privatize_gradients(model_update)
        
        \# Compress model update
        if self.compressor:
            model_update = self.compressor.compress_model_update(model_update)
        
        \# Encrypt model update
        encrypted_update = self.secure_comm.encrypt_model_update(model_update)
        
        \# Update SCAFFOLD control variate
        if self.config.algorithm == FederatedAlgorithm.SCAFFOLD:
            self._update_control_variate(global_model_state)
        
        return {
            'client_id': self.client_id,
            'model_update': encrypted_update,
            'num_samples': len(self.train_loader.dataset),
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'round': round_num
        }
    
    def _validate(self) -> Tuple[float, float]:
        """Validate local model."""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(data)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _update_control_variate(self, global_model_state: Dict[str, torch.Tensor]):
        """Update SCAFFOLD control variate."""
        for name, param in self.model.named_parameters():
            self.control_variate[name] = (
                global_model_state[name] - param.data
            ) / (self.config.local_epochs * self.config.local_learning_rate)

class FederatedServer:
    """Federated learning server."""
    
    def __init__(
        self,
        model: nn.Module,
        config: FederatedConfig,
        device: str = "cpu"
    ):
        """Initialize federated server."""
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        
        \# Client management
        self.registered_clients = {}
        self.client_states = {}
        
        \# Communication
        self.secure_comm = SecureCommunication(config.use_secure_communication)
        
        \# Compression
        if config.use_model_compression:
            self.compressor = ModelCompression(config.compression_ratio)
        else:
            self.compressor = None
        
        \# Byzantine robustness
        if config.byzantine_tolerance:
            self.byzantine_defense = ByzantineRobustness(config.max_byzantine_clients)
        else:
            self.byzantine_defense = None
        
        \# Global control variate for SCAFFOLD
        if config.algorithm == FederatedAlgorithm.SCAFFOLD:
            self.global_control_variate = {
                name: torch.zeros_like(param)
                for name, param in self.model.named_parameters()
            }
        
        \# Training history
        self.training_history = {
            'rounds': [],
            'participating_clients': [],
            'global_loss': [],
            'global_accuracy': [],
            'communication_cost': [],
            'convergence_metrics': []
        }
        
        \# Early stopping
        self.best_global_metric = 0.0
        self.patience_counter = 0
        
        logger.info("Initialized federated server")
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> str:
        """Register a new client."""
        if self.config.require_client_authentication:
            token = self.secure_comm.generate_client_token(client_id)
            self.registered_clients[client_id] = {
                'info': client_info,
                'token': token,
                'registered_at': datetime.now(),
                'last_seen': datetime.now()
            }
            logger.info(f"Registered client: {client_id}")
            return token
        else:
            self.registered_clients[client_id] = {
                'info': client_info,
                'registered_at': datetime.now(),
                'last_seen': datetime.now()
            }
            logger.info(f"Registered client: {client_id}")
            return ""
    
    def select_clients(self, round_num: int) -> List[str]:
        """Select clients for current round."""
        available_clients = list(self.registered_clients.keys())
        
        \# Apply dropout
        if self.config.client_dropout_rate > 0:
            num_dropout = int(len(available_clients) * self.config.client_dropout_rate)
            dropout_clients = random.sample(available_clients, min(num_dropout, len(available_clients)))
            available_clients = [c for c in available_clients if c not in dropout_clients]
        
        \# Select subset for this round
        num_select = min(self.config.clients_per_round, len(available_clients))
        selected_clients = random.sample(available_clients, num_select)
        
        logger.info(f"Round {round_num}: Selected {len(selected_clients)} clients")
        return selected_clients
    
    def aggregate_updates(
        self,
        client_updates: List[Dict[str, Any]],
        round_num: int
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates."""
        \# Decrypt and decompress updates
        processed_updates = []
        client_weights = []
        
        for update in client_updates:
            \# Decrypt
            decrypted_update = self.secure_comm.decrypt_model_update(update['model_update'])
            
            \# Decompress if needed
            if self.compressor:
                decrypted_update = self.compressor.decompress_model_update(decrypted_update)
            
            processed_updates.append(decrypted_update)
            client_weights.append(update['num_samples'])
        
        \# Apply Byzantine robustness
        if self.byzantine_defense and len(processed_updates) > 2 * self.config.max_byzantine_clients:
            if self.config.aggregation_strategy == AggregationStrategy.KRUM:
                aggregated_update = self.byzantine_defense.krum_aggregation(
                    processed_updates, self.config.max_byzantine_clients
                )
            elif self.config.aggregation_strategy == AggregationStrategy.TRIMMED_MEAN:
                aggregated_update = self.byzantine_defense.trimmed_mean_aggregation(
                    processed_updates, trim_ratio=0.2
                )
            else:
                aggregated_update = self._weighted_average_aggregation(
                    processed_updates, client_weights
                )
        else:
            \# Standard aggregation
            if self.config.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
                aggregated_update = self._weighted_average_aggregation(
                    processed_updates, client_weights
                )
            elif self.config.aggregation_strategy == AggregationStrategy.SIMPLE_AVERAGE:
                aggregated_update = self._simple_average_aggregation(processed_updates)
            else:
                aggregated_update = self._weighted_average_aggregation(
                    processed_updates, client_weights
                )
        
        return aggregated_update
    
    def _weighted_average_aggregation(
        self,
        updates: List[Dict[str, torch.Tensor]],
        weights: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Weighted average aggregation."""
        if not updates:
            return {}
        
        total_weight = sum(weights)
        aggregated_update = {}
        
        param_names = list(updates<sup>0</sup>.keys())
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(updates<sup>0</sup>[param_name])
            
            for update, weight in zip(updates, weights):
                weighted_sum += update[param_name] * (weight / total_weight)
            
            aggregated_update[param_name] = weighted_sum
        
        return aggregated_update
    
    def _simple_average_aggregation(
        self,
        updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Simple average aggregation."""
        if not updates:
            return {}
        
        aggregated_update = {}
        param_names = list(updates<sup>0</sup>.keys())
        
        for param_name in param_names:
            param_sum = torch.zeros_like(updates<sup>0</sup>[param_name])
            
            for update in updates:
                param_sum += update[param_name]
            
            aggregated_update[param_name] = param_sum / len(updates)
        
        return aggregated_update
    
    def update_global_model(
        self,
        aggregated_update: Dict[str, torch.Tensor],
        round_num: int
    ):
        """Update global model with aggregated update."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_update:
                    if self.config.algorithm == FederatedAlgorithm.FEDNOVA:
                        \# FedNova uses normalized updates
                        param.data += self.config.global_learning_rate * aggregated_update[name]
                    else:
                        \# Standard update
                        param.data += aggregated_update[name]
        
        logger.info(f"Updated global model for round {round_num}")
    
    def evaluate_global_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate global model on test data."""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(data)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total_samples
        }
    
    def check_convergence(self, current_metric: float, round_num: int) -> bool:
        """Check if training has converged."""
        if current_metric > self.best_global_metric + self.config.min_improvement:
            self.best_global_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at round {round_num}")
                return True
            
            return False
    
    def save_model(self, path: str, round_num: int):
        """Save global model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'round': round_num,
            'training_history': self.training_history,
            'best_metric': self.best_global_metric
        }, path)
        
        logger.info(f"Saved global model at round {round_num}")

class FederatedLearningSystem:
    """Complete federated learning system."""
    
    def __init__(
        self,
        model_class: type,
        model_args: Dict[str, Any],
        config: FederatedConfig,
        device: str = "cpu"
    ):
        """Initialize federated learning system."""
        self.model_class = model_class
        self.model_args = model_args
        self.config = config
        self.device = device
        
        \# Initialize server
        server_model = model_class(**model_args)
        self.server = FederatedServer(server_model, config, device)
        
        \# Client registry
        self.clients = {}
        
        logger.info("Initialized federated learning system")
    
    def add_client(
        self,
        client_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        client_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a client to the federation."""
        \# Create client model
        client_model = self.model_class(**self.model_args)
        
        \# Create client
        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            device=self.device
        )
        
        self.clients[client_id] = client
        
        \# Register with server
        if client_info is None:
            client_info = {
                'num_samples': len(train_loader.dataset),
                'device': self.device
            }
        
        token = self.server.register_client(client_id, client_info)
        
        logger.info(f"Added client {client_id} to federation")
        return token
    
    def train(self, test_loader: Optional[DataLoader] = None):
        """Train federated model."""
        logger.info("Starting federated training...")
        
        for round_num in range(1, self.config.num_rounds + 1):
            logger.info(f"Starting round {round_num}/{self.config.num_rounds}")
            
            \# Select clients for this round
            selected_client_ids = self.server.select_clients(round_num)
            
            if not selected_client_ids:
                logger.warning(f"No clients selected for round {round_num}")
                continue
            
            \# Get current global model state
            global_model_state = self.server.model.state_dict()
            
            \# Collect client updates
            client_updates = []
            
            for client_id in selected_client_ids:
                if client_id in self.clients:
                    try:
                        \# Perform local training
                        update = self.clients[client_id].local_train(
                            global_model_state, round_num
                        )
                        client_updates.append(update)
                        
                        logger.info(
                            f"Client {client_id}: "
                            f"Train Loss: {update['train_loss']:.4f}, "
                            f"Train Acc: {update['train_accuracy']:.4f}, "
                            f"Val Loss: {update['val_loss']:.4f}, "
                            f"Val Acc: {update['val_accuracy']:.4f}"
                        )
                    
                    except Exception as e:
                        logger.error(f"Client {client_id} training failed: {e}")
                        continue
            
            if not client_updates:
                logger.warning(f"No successful client updates in round {round_num}")
                continue
            
            \# Aggregate updates
            aggregated_update = self.server.aggregate_updates(client_updates, round_num)
            
            \# Update global model
            self.server.update_global_model(aggregated_update, round_num)
            
            \# Evaluate global model
            if test_loader and round_num % self.config.validation_frequency == 0:
                global_metrics = self.server.evaluate_global_model(test_loader)
                
                logger.info(
                    f"Round {round_num} Global Metrics: "
                    f"Loss: {global_metrics['loss']:.4f}, "
                    f"Accuracy: {global_metrics['accuracy']:.4f}"
                )
                
                \# Update training history
                self.server.training_history['rounds'].append(round_num)
                self.server.training_history['participating_clients'].append(len(client_updates))
                self.server.training_history['global_loss'].append(global_metrics['loss'])
                self.server.training_history['global_accuracy'].append(global_metrics['accuracy'])
                
                \# Check convergence
                if self.server.check_convergence(global_metrics['accuracy'], round_num):
                    break
        
        logger.info("Federated training completed")
    
    def get_global_model(self) -> nn.Module:
        """Get the trained global model."""
        return self.server.model
    
    def save_system(self, path: str):
        """Save the entire federated learning system."""
        self.server.save_model(path, len(self.server.training_history['rounds']))

\# Example usage and demonstration
def create_sample_federated_data():
    """Create sample federated data for demonstration."""
    from torch.utils.data import TensorDataset
    
    \# Create synthetic data for multiple clients
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_clients = 5
    samples_per_client = 1000
    num_features = 20
    num_classes = 2
    
    client_datasets = {}
    
    for client_id in range(num_clients):
        \# Create non-IID data by varying class distributions
        if client_id < 2:
            \# Clients 0-1: More class 0
            class_0_samples = int(samples_per_client * 0.8)
            class_1_samples = samples_per_client - class_0_samples
        else:
            \# Clients 2-4: More class 1
            class_0_samples = int(samples_per_client * 0.3)
            class_1_samples = samples_per_client - class_0_samples
        
        \# Generate features
        X_0 = torch.randn(class_0_samples, num_features) + torch.tensor([1.0] * num_features)
        X_1 = torch.randn(class_1_samples, num_features) + torch.tensor([-1.0] * num_features)
        
        X = torch.cat([X_0, X_1], dim=0)
        y = torch.cat([torch.zeros(class_0_samples), torch.ones(class_1_samples)], dim=0).long()
        
        \# Shuffle
        indices = torch.randperm(len(X))
        X, y = X[indices], y[indices]
        
        \# Split into train/val
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        client_datasets[f'client_{client_id}'] = {
            'train': train_dataset,
            'val': val_dataset
        }
    
    \# Create test dataset
    X_test = torch.randn(500, num_features)
    y_test = (X_test.sum(dim=1) > 0).long()
    test_dataset = TensorDataset(X_test, y_test)
    
    return client_datasets, test_dataset

class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def demonstrate_federated_learning():
    """Demonstrate federated learning system."""
    print("Federated Learning System Demonstration")
    print("=" * 50)
    
    \# Create configuration
    config = FederatedConfig(
        algorithm=FederatedAlgorithm.FEDAVG,
        aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        privacy_mechanism=PrivacyMechanism.DIFFERENTIAL_PRIVACY,
        num_rounds=20,  \# Reduced for demo
        clients_per_round=3,
        local_epochs=3,
        local_batch_size=32,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        enable_clinical_validation=True,
        byzantine_tolerance=True
    )
    
    print(f"Configuration:")
    print(f"  Algorithm: {config.algorithm.value}")
    print(f"  Privacy mechanism: {config.privacy_mechanism.value}")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Clients per round: {config.clients_per_round}")
    
    \# Create sample data
    client_datasets, test_dataset = create_sample_federated_data()
    
    print(f"\nDataset: {len(client_datasets)} clients")
    for client_id, datasets in client_datasets.items():
        print(f"  {client_id}: {len(datasets['train'])} train, {len(datasets['val'])} val samples")
    
    \# Initialize federated learning system
    model_args = {'input_dim': 20, 'hidden_dim': 64, 'num_classes': 2}
    fl_system = FederatedLearningSystem(
        model_class=SimpleModel,
        model_args=model_args,
        config=config,
        device="cpu"
    )
    
    \# Add clients
    for client_id, datasets in client_datasets.items():
        train_loader = DataLoader(datasets['train'], batch_size=32, shuffle=True)
        val_loader = DataLoader(datasets['val'], batch_size=32, shuffle=False)
        
        token = fl_system.add_client(client_id, train_loader, val_loader)
        print(f"Added {client_id} to federation")
    
    \# Create test loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"\nFederated learning system initialized")
    print(f"Global model parameters: {sum(p.numel() for p in fl_system.server.model.parameters()):,}")
    
    print("\nNote: This is a demonstration with synthetic data")
    print("In practice, you would:")
    print("1. Deploy clients across healthcare institutions")
    print("2. Implement secure communication protocols")
    print("3. Ensure regulatory compliance")
    print("4. Monitor for Byzantine attacks")
    print("5. Validate clinical performance")

if __name__ == "__main__":
    demonstrate_federated_learning()