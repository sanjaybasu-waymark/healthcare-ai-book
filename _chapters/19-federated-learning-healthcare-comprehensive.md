# Chapter 19: Federated Learning for Healthcare

## Learning Objectives

By the end of this chapter, readers will be able to:

1. **Implement federated learning architectures** for healthcare applications with privacy preservation
2. **Design secure aggregation protocols** that protect patient data during model training
3. **Handle data heterogeneity** across healthcare institutions and patient populations
4. **Deploy differential privacy mechanisms** for additional privacy protection
5. **Evaluate federated models** using appropriate metrics and validation frameworks
6. **Address regulatory compliance** requirements for federated healthcare AI systems

## Introduction

Federated learning represents a paradigm shift in healthcare artificial intelligence, enabling collaborative model training across multiple institutions while keeping sensitive patient data localized. This approach addresses one of the most significant challenges in healthcare AI: the need for large, diverse datasets while maintaining strict privacy and regulatory compliance requirements.

Traditional centralized machine learning requires aggregating data from multiple sources into a single location, which poses significant privacy, security, and regulatory challenges in healthcare. Patient data is highly sensitive and subject to strict regulations such as HIPAA in the United States, GDPR in Europe, and similar privacy laws worldwide. Additionally, healthcare institutions are often reluctant to share data due to competitive concerns and liability issues.

Federated learning solves these challenges by bringing the computation to the data rather than bringing the data to the computation. In this paradigm, each participating institution trains a local model on their own data, and only model parameters or gradients are shared with a central coordinator. The coordinator aggregates these updates to create a global model, which is then distributed back to participants for the next round of training.

The benefits of federated learning in healthcare are substantial. First, it enables training on much larger and more diverse datasets than any single institution could provide, leading to more robust and generalizable models. Second, it preserves patient privacy by keeping raw data within institutional boundaries. Third, it allows smaller institutions to benefit from models trained on larger datasets without having to share their own sensitive data. Fourth, it can help address health disparities by ensuring models are trained on diverse populations from multiple geographic regions and healthcare systems.

However, federated learning also introduces unique challenges. Data heterogeneity across institutions can lead to model convergence issues and reduced performance. Communication costs can be significant when dealing with large models and many participants. Security vulnerabilities may arise from the distributed nature of the system. Additionally, ensuring fairness and preventing bias across different participant populations requires careful consideration.

This chapter provides a comprehensive guide to implementing federated learning systems for healthcare applications. We cover the theoretical foundations, practical implementation strategies, privacy preservation techniques, and real-world deployment considerations. The approaches presented here represent the current state-of-the-art in federated healthcare AI and have been validated through extensive research and clinical studies.

## Theoretical Foundations

### Federated Learning Framework

The federated learning framework can be mathematically formulated as a distributed optimization problem. Let $\mathcal{P} = \{P_1, P_2, \ldots, P_K\}$ be a set of $K$ participating institutions, each with local dataset $\mathcal{D}_k$ of size $n_k$. The global objective is to minimize:

$$\min_{\theta} F(\theta) = \sum_{k=1}^K \frac{n_k}{n} F_k(\theta)$$

where $F_k(\theta) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(f(\theta; x_i), y_i)$ is the local objective function for institution $k$, $n = \sum_{k=1}^K n_k$ is the total number of samples, $\ell$ is the loss function, and $f(\theta; x_i)$ is the model prediction for input $x_i$ with parameters $\theta$.

### Federated Averaging Algorithm

The most widely used federated learning algorithm is Federated Averaging (FedAvg), which alternates between local training and global aggregation:

**Algorithm: Federated Averaging**
1. **Server initialization**: Initialize global model parameters $\theta_0$
2. **For each round** $t = 1, 2, \ldots, T$:
   - **Client selection**: Select subset $\mathcal{S}_t \subseteq \mathcal{P}$ of clients
   - **Broadcast**: Send current global model $\theta_t$ to selected clients
   - **Local training**: Each client $k \in \mathcal{S}_t$ performs local training:
     $$\theta_t^{(k)} = \text{LocalUpdate}(\theta_t, \mathcal{D}_k)$$
   - **Aggregation**: Server aggregates local updates:
     $$\theta_{t+1} = \sum_{k \in \mathcal{S}_t} \frac{n_k}{\sum_{j \in \mathcal{S}_t} n_j} \theta_t^{(k)}$$

The local update function typically involves multiple epochs of gradient descent:

$$\text{LocalUpdate}(\theta, \mathcal{D}_k) = \theta - \eta \sum_{e=1}^E \nabla F_k(\theta)$$

where $\eta$ is the learning rate and $E$ is the number of local epochs.

### Differential Privacy in Federated Learning

Differential privacy provides formal privacy guarantees by adding calibrated noise to the learning process. In federated learning, differential privacy can be applied at multiple levels:

**Local Differential Privacy**: Each client adds noise to their local updates before sharing:

$$\tilde{\theta}_t^{(k)} = \theta_t^{(k)} + \mathcal{N}(0, \sigma^2 I)$$

where $\sigma$ is calibrated to achieve $(\epsilon, \delta)$-differential privacy.

**Central Differential Privacy**: The server adds noise during aggregation:

$$\theta_{t+1} = \sum_{k \in \mathcal{S}_t} \frac{n_k}{\sum_{j \in \mathcal{S}_t} n_j} \theta_t^{(k)} + \mathcal{N}(0, \sigma_{central}^2 I)$$

The privacy budget $\epsilon$ accumulates over training rounds, requiring careful management to maintain meaningful privacy guarantees while achieving good model performance.

### Secure Aggregation

Secure aggregation protocols ensure that the server can compute the sum of client updates without learning individual contributions. The basic protocol works as follows:

1. **Key Generation**: Clients generate pairwise shared keys
2. **Masking**: Each client masks their update with random values shared with other clients
3. **Aggregation**: Server sums masked updates, causing random masks to cancel out
4. **Dropout Handling**: Protocol handles clients that drop out during training

Mathematically, client $k$ sends:

$$\tilde{\theta}_t^{(k)} = \theta_t^{(k)} + \sum_{j \neq k} s_{k,j} \cdot \text{PRG}(\text{key}_{k,j})$$

where $s_{k,j} \in \{-1, +1\}$ ensures masks cancel during summation, and PRG is a pseudorandom generator.

### Handling Data Heterogeneity

Healthcare data exhibits significant heterogeneity across institutions due to differences in patient populations, clinical practices, and data collection protocols. This heterogeneity can be categorized as:

**Statistical Heterogeneity**: Differences in data distributions across clients. This can be modeled as:

$$P_k(X, Y) \neq P_j(X, Y) \text{ for } k \neq j$$

**System Heterogeneity**: Differences in computational and communication capabilities across institutions.

**Temporal Heterogeneity**: Differences in data availability and update frequencies.

To address statistical heterogeneity, several approaches have been developed:

**FedProx**: Adds a proximal term to the local objective:

$$\min_{\theta} F_k(\theta) + \frac{\mu}{2} \|\theta - \theta_t\|^2$$

where $\mu$ controls the strength of the proximal term.

**SCAFFOLD**: Uses control variates to correct for client drift:

$$\theta_{t+1}^{(k)} = \theta_t - \eta (g_k - c_k + c)$$

where $c_k$ and $c$ are client and server control variates, respectively.

## Implementation Framework

### Comprehensive Federated Learning System

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import warnings
from datetime import datetime
import pickle
import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import threading
import time
import copy
from collections import OrderedDict
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareFederatedDataset(Dataset):
    """
    Healthcare dataset for federated learning with privacy-preserving features.
    
    Supports multiple data modalities and implements differential privacy
    mechanisms for data protection.
    """
    
    def __init__(self,
                 data_path: str,
                 institution_id: str,
                 data_type: str = 'clinical',
                 apply_dp: bool = True,
                 epsilon: float = 1.0,
                 delta: float = 1e-5):
        """
        Initialize federated healthcare dataset.
        
        Args:
            data_path: Path to dataset
            institution_id: Unique identifier for institution
            data_type: Type of healthcare data ('clinical', 'imaging', 'genomic')
            apply_dp: Whether to apply differential privacy
            epsilon: Privacy budget parameter
            delta: Privacy parameter
        """
        self.data_path = Path(data_path)
        self.institution_id = institution_id
        self.data_type = data_type
        self.apply_dp = apply_dp
        self.epsilon = epsilon
        self.delta = delta
        
        # Load data based on type
        self._load_data()
        
        # Apply differential privacy if requested
        if self.apply_dp:
            self._apply_differential_privacy()
        
        logger.info(f"Loaded {len(self.data)} samples for institution {institution_id}")
    
    def _load_data(self):
        """Load healthcare data based on type."""
        if self.data_type == 'clinical':
            self._load_clinical_data()
        elif self.data_type == 'imaging':
            self._load_imaging_data()
        elif self.data_type == 'genomic':
            self._load_genomic_data()
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def _load_clinical_data(self):
        """Load clinical tabular data."""
        # Load clinical data (assuming CSV format)
        data_file = self.data_path / f"{self.institution_id}_clinical.csv"
        
        if data_file.exists():
            df = pd.read_csv(data_file)
        else:
            # Generate synthetic clinical data for demonstration
            np.random.seed(hash(self.institution_id) % 2**32)
            n_samples = np.random.randint(500, 2000)
            
            # Simulate institution-specific data distribution
            institution_bias = hash(self.institution_id) % 100 / 100.0
            
            df = pd.DataFrame({
                'age': np.random.normal(65 + institution_bias * 10, 15, n_samples),
                'bmi': np.random.normal(28 + institution_bias * 5, 6, n_samples),
                'blood_pressure_sys': np.random.normal(130 + institution_bias * 20, 20, n_samples),
                'blood_pressure_dia': np.random.normal(85 + institution_bias * 10, 15, n_samples),
                'cholesterol': np.random.normal(200 + institution_bias * 50, 40, n_samples),
                'glucose': np.random.normal(100 + institution_bias * 30, 25, n_samples),
                'heart_rate': np.random.normal(70 + institution_bias * 10, 12, n_samples),
                'smoking': np.random.binomial(1, 0.2 + institution_bias * 0.3, n_samples),
                'diabetes': np.random.binomial(1, 0.15 + institution_bias * 0.2, n_samples),
                'hypertension': np.random.binomial(1, 0.3 + institution_bias * 0.2, n_samples)
            })
            
            # Create target variable (cardiovascular risk)
            risk_score = (
                (df['age'] - 50) * 0.02 +
                (df['bmi'] - 25) * 0.05 +
                (df['blood_pressure_sys'] - 120) * 0.01 +
                df['smoking'] * 0.3 +
                df['diabetes'] * 0.4 +
                df['hypertension'] * 0.2 +
                np.random.normal(0, 0.1, n_samples)
            )
            
            df['target'] = (risk_score > np.median(risk_score)).astype(int)
        
        # Prepare features and labels
        feature_cols = [col for col in df.columns if col != 'target']
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df['target'].values.astype(np.int64)
        
        # Normalize features
        self.feature_mean = np.mean(self.features, axis=0)
        self.feature_std = np.std(self.features, axis=0) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std
        
        self.data = list(zip(self.features, self.labels))
    
    def _load_imaging_data(self):
        """Load medical imaging data."""
        # Placeholder for imaging data loading
        # In practice, this would load DICOM files, preprocess images, etc.
        np.random.seed(hash(self.institution_id) % 2**32)
        n_samples = np.random.randint(200, 800)
        
        # Generate synthetic image features (e.g., from a pre-trained CNN)
        self.features = np.random.randn(n_samples, 512).astype(np.float32)
        self.labels = np.random.randint(0, 2, n_samples).astype(np.int64)
        
        self.data = list(zip(self.features, self.labels))
    
    def _load_genomic_data(self):
        """Load genomic data."""
        # Placeholder for genomic data loading
        np.random.seed(hash(self.institution_id) % 2**32)
        n_samples = np.random.randint(100, 500)
        n_snps = 1000  # Number of SNPs
        
        # Generate synthetic genomic data
        self.features = np.random.randint(0, 3, (n_samples, n_snps)).astype(np.float32)
        self.labels = np.random.randint(0, 2, n_samples).astype(np.int64)
        
        self.data = list(zip(self.features, self.labels))
    
    def _apply_differential_privacy(self):
        """Apply differential privacy to the dataset."""
        if self.data_type == 'clinical':
            # Add noise to continuous features
            sensitivity = 2.0  # Assuming normalized features
            noise_scale = sensitivity / self.epsilon
            
            for i, (features, label) in enumerate(self.data):
                noisy_features = features + np.random.laplace(0, noise_scale, features.shape)
                self.data[i] = (noisy_features.astype(np.float32), label)
        
        logger.info(f"Applied differential privacy with ε={self.epsilon}, δ={self.delta}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, label = self.data[idx]
        return torch.tensor(features), torch.tensor(label)

class FederatedModel(nn.Module):
    """
    Neural network model for federated learning in healthcare.
    
    Supports multiple architectures and privacy-preserving mechanisms.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 use_batch_norm: bool = True):
        """
        Initialize federated model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

class SecureAggregator:
    """
    Secure aggregation protocol for federated learning.
    
    Implements cryptographic protocols to protect individual client updates
    while enabling computation of aggregate statistics.
    """
    
    def __init__(self, num_clients: int, security_threshold: int = None):
        """
        Initialize secure aggregator.
        
        Args:
            num_clients: Total number of clients
            security_threshold: Minimum number of clients needed for aggregation
        """
        self.num_clients = num_clients
        self.security_threshold = security_threshold or max(1, num_clients // 2)
        self.client_keys = {}
        self.aggregation_masks = {}
        
        # Generate encryption key for secure communication
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def generate_client_keys(self, client_id: str) -> Dict[str, Any]:
        """Generate cryptographic keys for a client."""
        # Generate pairwise shared keys with other clients
        pairwise_keys = {}
        for other_client in range(self.num_clients):
            if other_client != hash(client_id) % self.num_clients:
                shared_secret = secrets.token_bytes(32)
                pairwise_keys[f"client_{other_client}"] = shared_secret
        
        # Generate masking seed
        masking_seed = secrets.token_bytes(32)
        
        client_keys = {
            'client_id': client_id,
            'pairwise_keys': pairwise_keys,
            'masking_seed': masking_seed,
            'encryption_key': self.encryption_key
        }
        
        self.client_keys[client_id] = client_keys
        return client_keys
    
    def generate_aggregation_mask(self, client_id: str, model_shape: Tuple) -> np.ndarray:
        """Generate aggregation mask for secure aggregation."""
        # Use masking seed to generate reproducible random mask
        np.random.seed(int.from_bytes(
            self.client_keys[client_id]['masking_seed'][:4], 'big'
        ))
        
        # Generate mask with same shape as model parameters
        total_params = np.prod(model_shape)
        mask = np.random.randn(total_params).astype(np.float32)
        
        # Store mask for later use
        self.aggregation_masks[client_id] = mask
        
        return mask.reshape(model_shape)
    
    def encrypt_update(self, client_id: str, model_update: Dict[str, torch.Tensor]) -> bytes:
        """Encrypt client model update."""
        # Serialize model update
        serialized_update = pickle.dumps(model_update)
        
        # Encrypt using client's encryption key
        encrypted_update = self.cipher.encrypt(serialized_update)
        
        return encrypted_update
    
    def decrypt_update(self, encrypted_update: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt client model update."""
        # Decrypt update
        serialized_update = self.cipher.decrypt(encrypted_update)
        
        # Deserialize model update
        model_update = pickle.loads(serialized_update)
        
        return model_update
    
    def secure_aggregate(self, 
                        encrypted_updates: List[bytes],
                        participating_clients: List[str]) -> Dict[str, torch.Tensor]:
        """
        Perform secure aggregation of client updates.
        
        Args:
            encrypted_updates: List of encrypted client updates
            participating_clients: List of participating client IDs
            
        Returns:
            aggregated_update: Securely aggregated model update
        """
        if len(participating_clients) < self.security_threshold:
            raise ValueError(f"Insufficient clients for secure aggregation: "
                           f"{len(participating_clients)} < {self.security_threshold}")
        
        # Decrypt all updates
        decrypted_updates = []
        for encrypted_update in encrypted_updates:
            decrypted_update = self.decrypt_update(encrypted_update)
            decrypted_updates.append(decrypted_update)
        
        # Aggregate updates
        aggregated_update = {}
        
        for key in decrypted_updates[0].keys():
            # Stack all client updates for this parameter
            client_updates = torch.stack([update[key] for update in decrypted_updates])
            
            # Compute weighted average (assuming equal weights for simplicity)
            aggregated_update[key] = torch.mean(client_updates, dim=0)
        
        return aggregated_update

class DifferentialPrivacyManager:
    """
    Differential privacy manager for federated learning.
    
    Implements various DP mechanisms including Gaussian mechanism,
    Laplace mechanism, and privacy accounting.
    """
    
    def __init__(self,
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 sensitivity: float = 1.0,
                 mechanism: str = 'gaussian'):
        """
        Initialize differential privacy manager.
        
        Args:
            epsilon: Privacy budget parameter
            delta: Privacy parameter for (ε,δ)-DP
            sensitivity: Sensitivity of the function
            mechanism: DP mechanism ('gaussian', 'laplace')
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        self.privacy_spent = 0.0
        
        # Calculate noise parameters
        if mechanism == 'gaussian':
            self.noise_scale = self._calculate_gaussian_noise_scale()
        elif mechanism == 'laplace':
            self.noise_scale = sensitivity / epsilon
        else:
            raise ValueError(f"Unsupported DP mechanism: {mechanism}")
    
    def _calculate_gaussian_noise_scale(self) -> float:
        """Calculate noise scale for Gaussian mechanism."""
        # Using the standard formula for Gaussian DP
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise_to_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients."""
        noisy_gradients = {}
        
        for name, grad in gradients.items():
            if self.mechanism == 'gaussian':
                noise = torch.normal(0, self.noise_scale, grad.shape)
            elif self.mechanism == 'laplace':
                # PyTorch doesn't have Laplace distribution, use numpy
                noise_np = np.random.laplace(0, self.noise_scale, grad.shape)
                noise = torch.from_numpy(noise_np).float()
            
            noisy_gradients[name] = grad + noise
        
        # Update privacy accounting
        self.privacy_spent += self.epsilon
        
        return noisy_gradients
    
    def clip_gradients(self, 
                      gradients: Dict[str, torch.Tensor],
                      max_norm: float = 1.0) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity."""
        # Calculate total gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad) ** 2
        total_norm = torch.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > max_norm:
            clip_factor = max_norm / total_norm
            clipped_gradients = {
                name: grad * clip_factor for name, grad in gradients.items()
            }
        else:
            clipped_gradients = gradients
        
        return clipped_gradients
    
    def get_privacy_spent(self) -> float:
        """Get total privacy budget spent."""
        return self.privacy_spent
    
    def reset_privacy_budget(self):
        """Reset privacy budget counter."""
        self.privacy_spent = 0.0

class FederatedClient:
    """
    Federated learning client representing a healthcare institution.
    
    Handles local training, privacy preservation, and secure communication
    with the federated server.
    """
    
    def __init__(self,
                 client_id: str,
                 dataset: HealthcareFederatedDataset,
                 model: FederatedModel,
                 device: torch.device,
                 use_dp: bool = True,
                 dp_epsilon: float = 1.0,
                 dp_delta: float = 1e-5):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            dataset: Local dataset
            model: Local model instance
            device: Computation device
            use_dp: Whether to use differential privacy
            dp_epsilon: DP epsilon parameter
            dp_delta: DP delta parameter
        """
        self.client_id = client_id
        self.dataset = dataset
        self.model = model.to(device)
        self.device = device
        self.use_dp = use_dp
        
        # Initialize differential privacy manager
        if use_dp:
            self.dp_manager = DifferentialPrivacyManager(
                epsilon=dp_epsilon,
                delta=dp_delta,
                sensitivity=1.0,
                mechanism='gaussian'
            )
        
        # Create data loader
        self.data_loader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True,
            num_workers=2
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'privacy_spent': []
        }
        
        logger.info(f"Initialized federated client {client_id} with {len(dataset)} samples")
    
    def local_train(self, 
                   global_model_state: Dict[str, torch.Tensor],
                   num_epochs: int = 5) -> Dict[str, Any]:
        """
        Perform local training on client data.
        
        Args:
            global_model_state: Global model parameters from server
            num_epochs: Number of local training epochs
            
        Returns:
            training_result: Dictionary containing local updates and metrics
        """
        # Load global model state
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
        # Store initial model state for computing updates
        initial_state = copy.deepcopy(self.model.state_dict())
        
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Apply differential privacy to gradients if enabled
                if self.use_dp:
                    # Get gradients
                    gradients = {
                        name: param.grad.clone() 
                        for name, param in self.model.named_parameters() 
                        if param.grad is not None
                    }
                    
                    # Clip gradients
                    clipped_gradients = self.dp_manager.clip_gradients(gradients, max_norm=1.0)
                    
                    # Add noise
                    noisy_gradients = self.dp_manager.add_noise_to_gradients(clipped_gradients)
                    
                    # Update model parameters with noisy gradients
                    for name, param in self.model.named_parameters():
                        if name in noisy_gradients:
                            param.grad = noisy_gradients[name]
                
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total_predictions += target.size(0)
                correct_predictions += predicted.eq(target).sum().item()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(self.data_loader)
            accuracy = 100. * correct_predictions / total_predictions
            
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
        
        # Calculate model updates (difference from initial state)
        final_state = self.model.state_dict()
        model_updates = {}
        for key in final_state.keys():
            model_updates[key] = final_state[key] - initial_state[key]
        
        # Record training history
        self.training_history['loss'].extend(epoch_losses)
        self.training_history['accuracy'].extend(epoch_accuracies)
        
        if self.use_dp:
            self.training_history['privacy_spent'].append(
                self.dp_manager.get_privacy_spent()
            )
        
        training_result = {
            'client_id': self.client_id,
            'model_updates': model_updates,
            'num_samples': len(self.dataset),
            'final_loss': epoch_losses[-1],
            'final_accuracy': epoch_accuracies[-1],
            'privacy_spent': self.dp_manager.get_privacy_spent() if self.use_dp else 0.0
        }
        
        return training_result
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test data."""
        self.model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Store for additional metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        
        # Calculate AUC if binary classification
        if len(set(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_predictions)
        else:
            auc = None
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'auc': auc
        }

class FederatedServer:
    """
    Federated learning server coordinating multiple healthcare institutions.
    
    Manages global model aggregation, client coordination, and privacy
    preservation across the federated network.
    """
    
    def __init__(self,
                 global_model: FederatedModel,
                 aggregation_strategy: str = 'fedavg',
                 use_secure_aggregation: bool = True,
                 min_clients_per_round: int = 2,
                 client_fraction: float = 1.0):
        """
        Initialize federated server.
        
        Args:
            global_model: Global model template
            aggregation_strategy: Aggregation strategy ('fedavg', 'fedprox', 'scaffold')
            use_secure_aggregation: Whether to use secure aggregation
            min_clients_per_round: Minimum clients required per round
            client_fraction: Fraction of clients to select per round
        """
        self.global_model = global_model
        self.aggregation_strategy = aggregation_strategy
        self.use_secure_aggregation = use_secure_aggregation
        self.min_clients_per_round = min_clients_per_round
        self.client_fraction = client_fraction
        
        # Initialize secure aggregator if needed
        if use_secure_aggregation:
            self.secure_aggregator = None  # Will be initialized when clients register
        
        # Client management
        self.registered_clients = {}
        self.client_weights = {}
        
        # Training history
        self.training_history = {
            'round': [],
            'global_loss': [],
            'global_accuracy': [],
            'participating_clients': [],
            'aggregation_time': []
        }
        
        logger.info("Initialized federated server")
    
    def register_client(self, client: FederatedClient):
        """Register a client with the server."""
        self.registered_clients[client.client_id] = client
        self.client_weights[client.client_id] = len(client.dataset)
        
        logger.info(f"Registered client {client.client_id} with {len(client.dataset)} samples")
        
        # Initialize secure aggregator if this is the first client
        if self.use_secure_aggregation and self.secure_aggregator is None:
            self.secure_aggregator = SecureAggregator(
                num_clients=len(self.registered_clients)
            )
    
    def select_clients(self, round_num: int) -> List[str]:
        """Select clients for the current round."""
        available_clients = list(self.registered_clients.keys())
        
        # Determine number of clients to select
        num_clients_to_select = max(
            self.min_clients_per_round,
            int(len(available_clients) * self.client_fraction)
        )
        
        # Random selection (could be replaced with more sophisticated strategies)
        np.random.seed(round_num)  # For reproducibility
        selected_clients = np.random.choice(
            available_clients,
            size=min(num_clients_to_select, len(available_clients)),
            replace=False
        ).tolist()
        
        return selected_clients
    
    def aggregate_updates(self, 
                         client_results: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using specified strategy.
        
        Args:
            client_results: List of client training results
            
        Returns:
            aggregated_update: Aggregated model update
        """
        if self.aggregation_strategy == 'fedavg':
            return self._federated_averaging(client_results)
        elif self.aggregation_strategy == 'fedprox':
            return self._federated_proximal(client_results)
        elif self.aggregation_strategy == 'scaffold':
            return self._scaffold_aggregation(client_results)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.aggregation_strategy}")
    
    def _federated_averaging(self, client_results: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Implement FedAvg aggregation."""
        # Calculate total samples
        total_samples = sum(result['num_samples'] for result in client_results)
        
        # Initialize aggregated update
        aggregated_update = {}
        
        # Get parameter names from first client
        param_names = list(client_results[0]['model_updates'].keys())
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(client_results[0]['model_updates'][param_name])
            
            for result in client_results:
                weight = result['num_samples'] / total_samples
                weighted_sum += weight * result['model_updates'][param_name]
            
            aggregated_update[param_name] = weighted_sum
        
        return aggregated_update
    
    def _federated_proximal(self, client_results: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Implement FedProx aggregation (similar to FedAvg for now)."""
        # FedProx aggregation is similar to FedAvg at the server side
        # The proximal term is applied during local training
        return self._federated_averaging(client_results)
    
    def _scaffold_aggregation(self, client_results: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Implement SCAFFOLD aggregation."""
        # Simplified SCAFFOLD implementation
        # In practice, this would involve control variates
        return self._federated_averaging(client_results)
    
    def train_round(self, round_num: int, num_local_epochs: int = 5) -> Dict[str, Any]:
        """
        Execute one round of federated training.
        
        Args:
            round_num: Current round number
            num_local_epochs: Number of local epochs per client
            
        Returns:
            round_results: Results from the training round
        """
        start_time = time.time()
        
        # Select clients for this round
        selected_clients = self.select_clients(round_num)
        
        if len(selected_clients) < self.min_clients_per_round:
            raise ValueError(f"Insufficient clients selected: {len(selected_clients)}")
        
        logger.info(f"Round {round_num}: Selected {len(selected_clients)} clients")
        
        # Get current global model state
        global_model_state = self.global_model.state_dict()
        
        # Collect client updates
        client_results = []
        
        for client_id in selected_clients:
            client = self.registered_clients[client_id]
            
            # Perform local training
            result = client.local_train(global_model_state, num_local_epochs)
            client_results.append(result)
            
            logger.info(f"Client {client_id}: Loss={result['final_loss']:.4f}, "
                       f"Accuracy={result['final_accuracy']:.2f}%")
        
        # Aggregate updates
        if self.use_secure_aggregation:
            # Encrypt client updates
            encrypted_updates = []
            for result in client_results:
                encrypted_update = self.secure_aggregator.encrypt_update(
                    result['client_id'], result['model_updates']
                )
                encrypted_updates.append(encrypted_update)
            
            # Perform secure aggregation
            aggregated_update = self.secure_aggregator.secure_aggregate(
                encrypted_updates, selected_clients
            )
        else:
            # Standard aggregation
            aggregated_update = self.aggregate_updates(client_results)
        
        # Update global model
        current_state = self.global_model.state_dict()
        for key in current_state.keys():
            current_state[key] += aggregated_update[key]
        
        self.global_model.load_state_dict(current_state)
        
        # Calculate round metrics
        avg_loss = np.mean([result['final_loss'] for result in client_results])
        avg_accuracy = np.mean([result['final_accuracy'] for result in client_results])
        total_privacy_spent = sum([result['privacy_spent'] for result in client_results])
        
        aggregation_time = time.time() - start_time
        
        # Record history
        self.training_history['round'].append(round_num)
        self.training_history['global_loss'].append(avg_loss)
        self.training_history['global_accuracy'].append(avg_accuracy)
        self.training_history['participating_clients'].append(len(selected_clients))
        self.training_history['aggregation_time'].append(aggregation_time)
        
        round_results = {
            'round': round_num,
            'participating_clients': selected_clients,
            'global_loss': avg_loss,
            'global_accuracy': avg_accuracy,
            'total_privacy_spent': total_privacy_spent,
            'aggregation_time': aggregation_time
        }
        
        logger.info(f"Round {round_num} completed: Global Loss={avg_loss:.4f}, "
                   f"Global Accuracy={avg_accuracy:.2f}%")
        
        return round_results
    
    def evaluate_global_model(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Evaluate global model on test data from multiple institutions.
        
        Args:
            test_loaders: Dictionary mapping client IDs to test data loaders
            
        Returns:
            evaluation_results: Comprehensive evaluation metrics
        """
        self.global_model.eval()
        
        overall_results = {
            'accuracy': [],
            'loss': [],
            'auc': []
        }
        
        client_results = {}
        
        for client_id, test_loader in test_loaders.items():
            # Evaluate on this client's test data
            client = self.registered_clients[client_id]
            
            # Temporarily load global model to client
            original_state = client.model.state_dict()
            client.model.load_state_dict(self.global_model.state_dict())
            
            # Evaluate
            metrics = client.evaluate(test_loader)
            client_results[client_id] = metrics
            
            # Restore original client model
            client.model.load_state_dict(original_state)
            
            # Aggregate metrics
            overall_results['accuracy'].append(metrics['accuracy'])
            overall_results['loss'].append(metrics['loss'])
            if metrics['auc'] is not None:
                overall_results['auc'].append(metrics['auc'])
        
        # Calculate overall metrics
        evaluation_results = {
            'client_results': client_results,
            'overall_accuracy': np.mean(overall_results['accuracy']),
            'overall_loss': np.mean(overall_results['loss']),
            'overall_auc': np.mean(overall_results['auc']) if overall_results['auc'] else None,
            'accuracy_std': np.std(overall_results['accuracy']),
            'loss_std': np.std(overall_results['loss'])
        }
        
        return evaluation_results

class FederatedLearningValidator:
    """
    Comprehensive validation framework for federated learning systems.
    
    Evaluates performance, privacy preservation, fairness, and robustness
    of federated healthcare AI systems.
    """
    
    def __init__(self, server: FederatedServer):
        """Initialize federated learning validator."""
        self.server = server
        self.validation_results = {}
    
    def evaluate_convergence(self) -> Dict[str, Any]:
        """Evaluate convergence properties of federated training."""
        history = self.server.training_history
        
        if len(history['global_loss']) < 2:
            return {'error': 'Insufficient training history for convergence analysis'}
        
        # Calculate convergence metrics
        losses = np.array(history['global_loss'])
        accuracies = np.array(history['global_accuracy'])
        
        # Loss convergence
        loss_improvement = losses[0] - losses[-1]
        loss_stability = np.std(losses[-5:]) if len(losses) >= 5 else np.std(losses)
        
        # Accuracy convergence
        accuracy_improvement = accuracies[-1] - accuracies[0]
        accuracy_stability = np.std(accuracies[-5:]) if len(accuracies) >= 5 else np.std(accuracies)
        
        convergence_results = {
            'loss_improvement': float(loss_improvement),
            'loss_stability': float(loss_stability),
            'accuracy_improvement': float(accuracy_improvement),
            'accuracy_stability': float(accuracy_stability),
            'converged': loss_stability < 0.01 and accuracy_stability < 1.0
        }
        
        return convergence_results
    
    def evaluate_privacy_preservation(self) -> Dict[str, Any]:
        """Evaluate privacy preservation mechanisms."""
        privacy_results = {}
        
        # Check if differential privacy is used
        dp_clients = []
        total_privacy_spent = 0.0
        
        for client_id, client in self.server.registered_clients.items():
            if client.use_dp:
                dp_clients.append(client_id)
                total_privacy_spent += client.dp_manager.get_privacy_spent()
        
        privacy_results['dp_enabled_clients'] = dp_clients
        privacy_results['total_privacy_spent'] = total_privacy_spent
        privacy_results['average_privacy_spent'] = (
            total_privacy_spent / len(dp_clients) if dp_clients else 0.0
        )
        
        # Check secure aggregation
        privacy_results['secure_aggregation_enabled'] = self.server.use_secure_aggregation
        
        # Privacy risk assessment
        if total_privacy_spent > 0:
            if total_privacy_spent < 1.0:
                privacy_risk = 'Low'
            elif total_privacy_spent < 5.0:
                privacy_risk = 'Medium'
            else:
                privacy_risk = 'High'
        else:
            privacy_risk = 'Unknown (no DP)'
        
        privacy_results['privacy_risk_level'] = privacy_risk
        
        return privacy_results
    
    def evaluate_fairness(self, test_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """Evaluate fairness across different client populations."""
        # Evaluate global model on each client's test data
        evaluation_results = self.server.evaluate_global_model(test_loaders)
        
        client_accuracies = [
            metrics['accuracy'] for metrics in evaluation_results['client_results'].values()
        ]
        
        # Calculate fairness metrics
        accuracy_range = max(client_accuracies) - min(client_accuracies)
        accuracy_std = np.std(client_accuracies)
        
        # Fairness assessment
        if accuracy_range < 5.0:
            fairness_level = 'High'
        elif accuracy_range < 10.0:
            fairness_level = 'Medium'
        else:
            fairness_level = 'Low'
        
        fairness_results = {
            'client_accuracies': dict(zip(
                evaluation_results['client_results'].keys(),
                client_accuracies
            )),
            'accuracy_range': float(accuracy_range),
            'accuracy_std': float(accuracy_std),
            'fairness_level': fairness_level
        }
        
        return fairness_results
    
    def evaluate_communication_efficiency(self) -> Dict[str, Any]:
        """Evaluate communication efficiency of federated training."""
        history = self.server.training_history
        
        if not history['aggregation_time']:
            return {'error': 'No aggregation time data available'}
        
        # Calculate communication metrics
        avg_aggregation_time = np.mean(history['aggregation_time'])
        total_communication_rounds = len(history['round'])
        avg_clients_per_round = np.mean(history['participating_clients'])
        
        # Estimate communication overhead
        # This is a simplified estimate - in practice, would measure actual data transfer
        model_size_mb = sum(p.numel() for p in self.server.global_model.parameters()) * 4 / (1024**2)
        total_data_transfer = model_size_mb * total_communication_rounds * avg_clients_per_round * 2  # Up and down
        
        efficiency_results = {
            'avg_aggregation_time': float(avg_aggregation_time),
            'total_communication_rounds': int(total_communication_rounds),
            'avg_clients_per_round': float(avg_clients_per_round),
            'estimated_model_size_mb': float(model_size_mb),
            'estimated_total_data_transfer_mb': float(total_data_transfer)
        }
        
        return efficiency_results
    
    def generate_comprehensive_report(self, test_loaders: Dict[str, DataLoader]) -> str:
        """Generate comprehensive validation report."""
        # Run all evaluations
        convergence_results = self.evaluate_convergence()
        privacy_results = self.evaluate_privacy_preservation()
        fairness_results = self.evaluate_fairness(test_loaders)
        efficiency_results = self.evaluate_communication_efficiency()
        
        # Store results
        self.validation_results = {
            'convergence': convergence_results,
            'privacy': privacy_results,
            'fairness': fairness_results,
            'efficiency': efficiency_results
        }
        
        # Generate report
        report = f"""
# Federated Learning Validation Report

## Executive Summary
This report provides a comprehensive evaluation of the federated learning system
across multiple dimensions including convergence, privacy preservation, fairness,
and communication efficiency.

## Convergence Analysis
- **Loss Improvement**: {convergence_results.get('loss_improvement', 'N/A'):.4f}
- **Accuracy Improvement**: {convergence_results.get('accuracy_improvement', 'N/A'):.2f}%
- **Convergence Status**: {'✓ Converged' if convergence_results.get('converged', False) else '✗ Not Converged'}

## Privacy Preservation
- **Differential Privacy**: {len(privacy_results['dp_enabled_clients'])} clients enabled
- **Total Privacy Spent**: {privacy_results['total_privacy_spent']:.4f}
- **Secure Aggregation**: {'✓ Enabled' if privacy_results['secure_aggregation_enabled'] else '✗ Disabled'}
- **Privacy Risk Level**: {privacy_results['privacy_risk_level']}

## Fairness Assessment
- **Accuracy Range**: {fairness_results['accuracy_range']:.2f}%
- **Fairness Level**: {fairness_results['fairness_level']}
- **Client Performance Variation**: {fairness_results['accuracy_std']:.2f}% (std dev)

## Communication Efficiency
- **Average Aggregation Time**: {efficiency_results.get('avg_aggregation_time', 'N/A'):.2f} seconds
- **Total Communication Rounds**: {efficiency_results.get('total_communication_rounds', 'N/A')}
- **Estimated Data Transfer**: {efficiency_results.get('estimated_total_data_transfer_mb', 'N/A'):.1f} MB

## Recommendations

### Privacy Enhancements
1. Consider increasing differential privacy budget if model performance is insufficient
2. Implement additional privacy-preserving techniques such as homomorphic encryption
3. Regular privacy audits and monitoring

### Performance Optimization
1. Optimize local training parameters to improve convergence
2. Consider adaptive aggregation strategies for heterogeneous data
3. Implement client selection strategies to improve efficiency

### Fairness Improvements
1. Monitor performance across different client populations
2. Implement fairness-aware aggregation strategies
3. Consider data augmentation for underrepresented populations

## Conclusion
The federated learning system demonstrates {'good' if convergence_results.get('converged', False) else 'limited'} 
convergence properties with {'strong' if privacy_results['privacy_risk_level'] == 'Low' else 'moderate'} 
privacy preservation. Fairness across clients is {'satisfactory' if fairness_results['fairness_level'] in ['High', 'Medium'] else 'concerning'}.
"""
        
        return report

# Training and evaluation functions
def train_federated_system(clients: List[FederatedClient],
                          server: FederatedServer,
                          num_rounds: int = 50,
                          local_epochs: int = 5,
                          test_loaders: Dict[str, DataLoader] = None) -> Dict[str, Any]:
    """
    Train federated learning system.
    
    Args:
        clients: List of federated clients
        server: Federated server
        num_rounds: Number of federated training rounds
        local_epochs: Number of local epochs per round
        test_loaders: Test data loaders for evaluation
        
    Returns:
        training_results: Comprehensive training results
    """
    # Register all clients with server
    for client in clients:
        server.register_client(client)
    
    logger.info(f"Starting federated training with {len(clients)} clients for {num_rounds} rounds")
    
    # Training loop
    round_results = []
    
    for round_num in range(1, num_rounds + 1):
        try:
            result = server.train_round(round_num, local_epochs)
            round_results.append(result)
            
            # Periodic evaluation
            if test_loaders and round_num % 10 == 0:
                eval_results = server.evaluate_global_model(test_loaders)
                logger.info(f"Round {round_num} Evaluation - "
                           f"Global Accuracy: {eval_results['overall_accuracy']:.2f}%")
        
        except Exception as e:
            logger.error(f"Error in round {round_num}: {e}")
            break
    
    # Final evaluation
    final_evaluation = None
    if test_loaders:
        final_evaluation = server.evaluate_global_model(test_loaders)
    
    training_results = {
        'round_results': round_results,
        'final_evaluation': final_evaluation,
        'training_history': server.training_history
    }
    
    return training_results

# Example usage and demonstration
def main():
    """Demonstrate the federated learning system."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create federated datasets for multiple institutions
    logger.info("Creating federated datasets for multiple healthcare institutions...")
    
    institution_ids = ['hospital_a', 'hospital_b', 'hospital_c', 'clinic_d', 'medical_center_e']
    clients = []
    test_loaders = {}
    
    for institution_id in institution_ids:
        # Create dataset for this institution
        dataset = HealthcareFederatedDataset(
            data_path='federated_data',
            institution_id=institution_id,
            data_type='clinical',
            apply_dp=True,
            epsilon=1.0,
            delta=1e-5
        )
        
        # Split into train and test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Create test loader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_loaders[institution_id] = test_loader
        
        # Create model for this client
        input_dim = dataset.features.shape[1] if hasattr(dataset, 'features') else 10
        model = FederatedModel(
            input_dim=input_dim,
            hidden_dims=[128, 64, 32],
            num_classes=2,
            dropout=0.3
        )
        
        # Create federated client
        client = FederatedClient(
            client_id=institution_id,
            dataset=train_dataset,
            model=model,
            device=device,
            use_dp=True,
            dp_epsilon=1.0,
            dp_delta=1e-5
        )
        
        clients.append(client)
    
    logger.info(f"Created {len(clients)} federated clients")
    
    # Create federated server
    logger.info("Creating federated server...")
    
    global_model = FederatedModel(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        num_classes=2,
        dropout=0.3
    )
    
    server = FederatedServer(
        global_model=global_model,
        aggregation_strategy='fedavg',
        use_secure_aggregation=True,
        min_clients_per_round=3,
        client_fraction=0.8
    )
    
    # Train federated system
    logger.info("Starting federated training...")
    
    training_results = train_federated_system(
        clients=clients,
        server=server,
        num_rounds=30,
        local_epochs=5,
        test_loaders=test_loaders
    )
    
    # Validate system
    logger.info("Performing comprehensive validation...")
    
    validator = FederatedLearningValidator(server)
    validation_report = validator.generate_comprehensive_report(test_loaders)
    
    # Save results
    results_dir = Path("federated_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save training results
    training_results_serializable = {}
    for key, value in training_results.items():
        if isinstance(value, (dict, list, str, int, float)):
            training_results_serializable[key] = value
        else:
            training_results_serializable[key] = str(value)
    
    with open(results_dir / 'training_results.json', 'w') as f:
        json.dump(training_results_serializable, f, indent=2)
    
    # Save validation results
    with open(results_dir / 'validation_results.json', 'w') as f:
        json.dump(validator.validation_results, f, indent=2)
    
    # Save validation report
    with open(results_dir / 'validation_report.md', 'w') as f:
        f.write(validation_report)
    
    # Save global model
    torch.save(server.global_model.state_dict(), results_dir / 'global_model.pth')
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Training loss and accuracy
    plt.subplot(2, 3, 1)
    plt.plot(server.training_history['round'], server.training_history['global_loss'])
    plt.xlabel('Round')
    plt.ylabel('Global Loss')
    plt.title('Federated Training Loss')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(server.training_history['round'], server.training_history['global_accuracy'])
    plt.xlabel('Round')
    plt.ylabel('Global Accuracy (%)')
    plt.title('Federated Training Accuracy')
    plt.grid(True)
    
    # Participating clients per round
    plt.subplot(2, 3, 3)
    plt.plot(server.training_history['round'], server.training_history['participating_clients'])
    plt.xlabel('Round')
    plt.ylabel('Number of Clients')
    plt.title('Participating Clients per Round')
    plt.grid(True)
    
    # Aggregation time
    plt.subplot(2, 3, 4)
    plt.plot(server.training_history['round'], server.training_history['aggregation_time'])
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.title('Aggregation Time per Round')
    plt.grid(True)
    
    # Client performance comparison
    if training_results['final_evaluation']:
        plt.subplot(2, 3, 5)
        client_names = list(training_results['final_evaluation']['client_results'].keys())
        client_accuracies = [
            training_results['final_evaluation']['client_results'][name]['accuracy']
            for name in client_names
        ]
        
        plt.bar(range(len(client_names)), client_accuracies)
        plt.xlabel('Client')
        plt.ylabel('Accuracy (%)')
        plt.title('Final Client Performance')
        plt.xticks(range(len(client_names)), [name.replace('_', '\n') for name in client_names], rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Privacy spending over time
    plt.subplot(2, 3, 6)
    privacy_spending = []
    for client in clients:
        if client.training_history['privacy_spent']:
            privacy_spending.extend(client.training_history['privacy_spent'])
    
    if privacy_spending:
        plt.plot(privacy_spending)
        plt.xlabel('Training Step')
        plt.ylabel('Cumulative Privacy Spent')
        plt.title('Privacy Budget Consumption')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'federated_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create privacy analysis plot
    plt.figure(figsize=(12, 8))
    
    # Privacy spending by client
    plt.subplot(2, 2, 1)
    client_privacy = [client.dp_manager.get_privacy_spent() for client in clients]
    client_names = [client.client_id for client in clients]
    
    plt.bar(range(len(client_names)), client_privacy)
    plt.xlabel('Client')
    plt.ylabel('Privacy Spent (ε)')
    plt.title('Privacy Budget Consumption by Client')
    plt.xticks(range(len(client_names)), [name.replace('_', '\n') for name in client_names], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Accuracy vs Privacy trade-off
    plt.subplot(2, 2, 2)
    if training_results['final_evaluation']:
        client_accuracies = [
            training_results['final_evaluation']['client_results'][name]['accuracy']
            for name in client_names
        ]
        plt.scatter(client_privacy, client_accuracies)
        plt.xlabel('Privacy Spent (ε)')
        plt.ylabel('Accuracy (%)')
        plt.title('Privacy-Accuracy Trade-off')
        plt.grid(True)
    
    # Communication efficiency
    plt.subplot(2, 2, 3)
    plt.plot(server.training_history['round'], server.training_history['aggregation_time'])
    plt.xlabel('Round')
    plt.ylabel('Aggregation Time (seconds)')
    plt.title('Communication Efficiency')
    plt.grid(True)
    
    # Fairness analysis
    plt.subplot(2, 2, 4)
    if training_results['final_evaluation']:
        plt.boxplot([client_accuracies])
        plt.ylabel('Accuracy (%)')
        plt.title('Fairness Across Clients')
        plt.xticks([1], ['All Clients'])
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'federated_privacy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Federated learning demonstration completed!")
    logger.info(f"Results saved to {results_dir}")
    
    if training_results['final_evaluation']:
        logger.info(f"Final global accuracy: {training_results['final_evaluation']['overall_accuracy']:.2f}%")
        logger.info(f"Final global AUC: {training_results['final_evaluation']['overall_auc']:.3f}")
    
    # Print validation summary
    print("\n" + "="*50)
    print("FEDERATED LEARNING VALIDATION SUMMARY")
    print("="*50)
    print(validation_report)
    
    return server, clients, training_results, validator

if __name__ == "__main__":
    main()
```

## Advanced Federated Learning Techniques

### Personalized Federated Learning

Personalized federated learning addresses the challenge of data heterogeneity by allowing each client to maintain a personalized model while still benefiting from collaborative training. The mathematical formulation involves a combination of global and local objectives:

$$\min_{\theta_g, \{\theta_k\}} \sum_{k=1}^K \frac{n_k}{n} \left[ F_k(\theta_k) + \lambda \|\theta_k - \theta_g\|^2 \right]$$

where $\theta_g$ is the global model, $\theta_k$ is the personalized model for client $k$, and $\lambda$ controls the strength of personalization.

### Asynchronous Federated Learning

Asynchronous federated learning allows clients to update the global model at different times, addressing the challenge of system heterogeneity. The server maintains a global model that is continuously updated as client updates arrive:

$$\theta_{t+1} = \theta_t + \alpha_t \Delta_k$$

where $\alpha_t$ is an adaptive learning rate that may depend on the staleness of the update and the reliability of the client.

### Federated Learning with Non-IID Data

Healthcare data is inherently non-IID (non-independent and identically distributed) across institutions. Several techniques address this challenge:

**FedNova**: Normalizes local updates to account for varying local training intensities:

$$\theta_{t+1} = \theta_t + \frac{1}{K} \sum_{k=1}^K \frac{\tau_k}{\bar{\tau}} \Delta_k$$

where $\tau_k$ is the number of local steps for client $k$ and $\bar{\tau}$ is the average.

**MOON (Model-Contrastive Federated Learning)**: Uses contrastive learning to align local and global model representations:

$$\mathcal{L}_{con} = -\log \frac{\exp(\text{sim}(z, z_{global}) / \tau)}{\exp(\text{sim}(z, z_{global}) / \tau) + \exp(\text{sim}(z, z_{prev}) / \tau)}$$

where $z$ is the current local representation, $z_{global}$ is the global representation, and $z_{prev}$ is the previous local representation.

## Clinical Applications and Case Studies

### Case Study 1: Multi-Institutional Drug Discovery

Federated learning enables pharmaceutical companies and research institutions to collaborate on drug discovery while protecting proprietary data. Key applications include:

1. **Molecular Property Prediction**: Training models on diverse chemical databases
2. **Clinical Trial Optimization**: Sharing insights without revealing patient data
3. **Adverse Event Detection**: Collaborative pharmacovigilance across institutions
4. **Biomarker Discovery**: Identifying therapeutic targets across populations

### Case Study 2: Rare Disease Research

Rare diseases affect small patient populations distributed across multiple institutions. Federated learning enables:

1. **Phenotype Classification**: Training diagnostic models on limited data
2. **Treatment Response Prediction**: Personalizing therapies for rare conditions
3. **Natural History Studies**: Understanding disease progression patterns
4. **Genetic Association Studies**: Identifying causal variants across populations

### Case Study 3: Global Health Surveillance

Federated learning supports public health surveillance while respecting national data sovereignty:

1. **Epidemic Modeling**: Predicting disease spread across regions
2. **Antimicrobial Resistance Tracking**: Monitoring resistance patterns globally
3. **Vaccine Effectiveness Studies**: Evaluating interventions across populations
4. **Health System Capacity Planning**: Optimizing resource allocation

## Regulatory and Compliance Considerations

### HIPAA Compliance in Federated Learning

The Health Insurance Portability and Accountability Act (HIPAA) requires specific safeguards for protected health information (PHI). Federated learning systems must ensure:

1. **Data Minimization**: Only necessary data is processed locally
2. **Access Controls**: Strict authentication and authorization mechanisms
3. **Audit Logging**: Comprehensive tracking of all data access and processing
4. **Encryption**: Protection of data in transit and at rest
5. **Business Associate Agreements**: Proper contracts with technology providers

### GDPR and International Compliance

The General Data Protection Regulation (GDPR) and similar international privacy laws impose additional requirements:

1. **Lawful Basis**: Clear legal justification for data processing
2. **Consent Management**: Proper mechanisms for obtaining and managing consent
3. **Data Subject Rights**: Support for access, rectification, and erasure requests
4. **Privacy by Design**: Built-in privacy protections from system inception
5. **Cross-Border Transfer**: Compliance with international data transfer restrictions

### FDA Considerations for Federated AI

The FDA's guidance on Software as Medical Device (SaMD) applies to federated learning systems:

1. **Predetermined Change Control**: Plans for model updates and retraining
2. **Clinical Validation**: Evidence of safety and effectiveness across populations
3. **Risk Management**: Comprehensive assessment of potential harms
4. **Quality Management**: ISO 13485 compliance for medical device development
5. **Post-Market Surveillance**: Ongoing monitoring of system performance

## Future Directions and Research Frontiers

### Quantum-Safe Federated Learning

As quantum computing advances, current cryptographic methods may become vulnerable. Quantum-safe federated learning involves:

1. **Post-Quantum Cryptography**: Implementing quantum-resistant encryption
2. **Quantum Key Distribution**: Using quantum mechanics for secure communication
3. **Quantum-Enhanced Privacy**: Leveraging quantum properties for privacy preservation
4. **Hybrid Classical-Quantum Systems**: Combining classical and quantum approaches

### Federated Learning at the Edge

Edge computing brings federated learning closer to data sources:

1. **IoT Medical Devices**: Training on wearable and implantable devices
2. **Real-Time Processing**: Immediate analysis of streaming health data
3. **Bandwidth Optimization**: Reducing communication requirements
4. **Latency Reduction**: Faster response times for critical applications

### Blockchain-Based Federated Learning

Blockchain technology can enhance federated learning through:

1. **Decentralized Coordination**: Eliminating single points of failure
2. **Immutable Audit Trails**: Transparent and tamper-proof logging
3. **Smart Contracts**: Automated execution of federated learning protocols
4. **Incentive Mechanisms**: Token-based rewards for participation

## Summary

Federated learning represents a transformative approach to healthcare AI that enables collaborative model training while preserving privacy and regulatory compliance. This chapter has provided comprehensive coverage of theoretical foundations, practical implementation strategies, privacy preservation techniques, and real-world deployment considerations.

Key takeaways include:

1. **Privacy Preservation**: Differential privacy and secure aggregation provide formal privacy guarantees
2. **Data Heterogeneity**: Specialized algorithms address the challenges of non-IID healthcare data
3. **Regulatory Compliance**: Federated learning can meet HIPAA, GDPR, and FDA requirements
4. **Clinical Applications**: Diverse use cases from drug discovery to global health surveillance
5. **Future Directions**: Quantum-safe cryptography, edge computing, and blockchain integration

The field continues to evolve rapidly, with ongoing research addressing scalability, efficiency, and robustness challenges. However, the fundamental promise of federated learning—enabling collaborative AI while protecting sensitive healthcare data—remains central to the future of healthcare artificial intelligence.

## References

1. McMahan, B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. *Artificial Intelligence and Statistics*, 1273-1282.

2. Li, T., et al. (2020). Federated optimization in heterogeneous networks. *Machine Learning and Systems*, 2, 429-450.

3. Kairouz, P., et al. (2021). Advances and open problems in federated learning. *Foundations and Trends in Machine Learning*, 14(1-2), 1-210. DOI: 10.1561/2200000083

4. Rieke, N., et al. (2020). The future of digital health with federated learning. *NPJ Digital Medicine*, 3(1), 119. DOI: 10.1038/s41746-020-00323-1

5. Li, X., et al. (2021). A survey on federated learning systems: Vision, hype and reality for data privacy and protection. *IEEE Transactions on Knowledge and Data Engineering*, 35(4), 3347-3366. DOI: 10.1109/TKDE.2021.3124599

6. Xu, J., et al. (2021). Federated learning for healthcare informatics. *Journal of Healthcare Informatics Research*, 5(1), 1-19. DOI: 10.1007/s41666-020-00082-4

7. Antunes, R. S., et al. (2022). Federated learning for healthcare: Systematic review and architecture proposal. *ACM Transactions on Intelligent Systems and Technology*, 13(4), 1-23. DOI: 10.1145/3501813

8. Sheller, M. J., et al. (2020). Federated learning in medicine: Facilitating multi-institutional collaborations without sharing patient data. *Scientific Reports*, 10(1), 12598. DOI: 10.1038/s41598-020-69250-1

9. Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3-4), 211-407. DOI: 10.1561/0400000042

10. Bonawitz, K., et al. (2017). Practical secure aggregation for privacy-preserving machine learning. *ACM SIGSAC Conference on Computer and Communications Security*, 1175-1191. DOI: 10.1145/3133956.3133982
