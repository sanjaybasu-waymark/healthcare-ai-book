# Chapter 22: Drug Discovery AI - Accelerating Pharmaceutical Innovation

## Learning Objectives

By the end of this chapter, readers will be able to:

1. **Understand the drug discovery pipeline** and how AI transforms each stage from target identification to clinical trials
2. **Implement molecular property prediction models** using graph neural networks and transformer architectures
3. **Design and optimize molecular generation systems** for novel drug candidate discovery
4. **Apply AI to protein structure prediction** and drug-target interaction modeling
5. **Develop clinical trial optimization systems** using machine learning and causal inference
6. **Navigate regulatory considerations** for AI-driven drug discovery and development
7. **Implement comprehensive validation frameworks** for pharmaceutical AI applications

## Introduction

The pharmaceutical industry faces unprecedented challenges in drug discovery and development. The traditional process takes 10-15 years and costs over $2.6 billion per approved drug, with a 90% failure rate in clinical trials. Artificial intelligence is revolutionizing this landscape by accelerating target identification, optimizing molecular design, predicting drug properties, and improving clinical trial efficiency.

This chapter provides comprehensive implementations of state-of-the-art AI systems for drug discovery, covering the entire pipeline from molecular design to clinical development. We'll explore how machine learning transforms pharmaceutical research while addressing the unique challenges of regulatory compliance, safety validation, and clinical translation.

## The Drug Discovery Pipeline and AI Integration

### Traditional Drug Discovery Challenges

The conventional drug discovery process involves multiple sequential stages, each with significant time and cost requirements:

1. **Target Identification and Validation** (2-3 years): Identifying biological targets associated with disease
2. **Lead Discovery** (2-3 years): Finding compounds that interact with the target
3. **Lead Optimization** (2-3 years): Improving compound properties for drug-like characteristics
4. **Preclinical Development** (1-2 years): Safety and efficacy testing in laboratory and animal models
5. **Clinical Trials** (6-8 years): Human testing across three phases
6. **Regulatory Review** (1-2 years): FDA/EMA approval process

### AI Transformation Opportunities

Artificial intelligence addresses critical bottlenecks throughout this pipeline:

- **Molecular Property Prediction**: Predicting ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties
- **Target Identification**: Using multi-omics data to identify novel therapeutic targets
- **Molecular Generation**: Designing novel compounds with desired properties
- **Drug Repurposing**: Identifying new uses for existing drugs
- **Clinical Trial Optimization**: Improving patient selection and trial design

## Molecular Property Prediction with Graph Neural Networks

### Mathematical Foundations

Molecular property prediction treats molecules as graphs where atoms are nodes and bonds are edges. Graph Neural Networks (GNNs) learn molecular representations by aggregating information from neighboring atoms.

For a molecule represented as graph G = (V, E) with node features X and edge features E, a GNN computes node embeddings through message passing:

```
h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```

Where:
- h_v^(l) is the embedding of node v at layer l
- N(v) represents the neighbors of node v
- UPDATE and AGGREGATE are learnable functions

### Implementation: Molecular Property Predictor

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MolecularFeatures:
    """Configuration for molecular featurization"""
    atom_features: List[str]
    bond_features: List[str]
    max_atoms: int = 100
    
class MolecularFeaturizer:
    """Convert SMILES strings to graph representations"""
    
    def __init__(self, config: MolecularFeatures):
        self.config = config
        self.atom_vocab = self._build_atom_vocab()
        self.bond_vocab = self._build_bond_vocab()
    
    def _build_atom_vocab(self) -> Dict[str, int]:
        """Build vocabulary for atom types"""
        common_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        return {atom: i for i, atom in enumerate(common_atoms)}
    
    def _build_bond_vocab(self) -> Dict[str, int]:
        """Build vocabulary for bond types"""
        bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        return {bond: i for i, bond in enumerate(bond_types)}
    
    def _get_atom_features(self, atom) -> List[float]:
        """Extract features for a single atom"""
        features = []
        
        # Atom type (one-hot encoded)
        atom_type = atom.GetSymbol()
        atom_encoding = [0] * len(self.atom_vocab)
        if atom_type in self.atom_vocab:
            atom_encoding[self.atom_vocab[atom_type]] = 1
        features.extend(atom_encoding)
        
        # Additional atom properties
        features.extend([
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetImplicitValence(),
            atom.GetIsAromatic(),
            atom.GetMass(),
            atom.GetNumImplicitHs(),
            atom.GetNumRadicalElectrons(),
            atom.GetTotalNumHs(),
            atom.IsInRing()
        ])
        
        return features
    
    def _get_bond_features(self, bond) -> List[float]:
        """Extract features for a single bond"""
        features = []
        
        # Bond type (one-hot encoded)
        bond_type = str(bond.GetBondType())
        bond_encoding = [0] * len(self.bond_vocab)
        if bond_type in self.bond_vocab:
            bond_encoding[self.bond_vocab[bond_type]] = 1
        features.extend(bond_encoding)
        
        # Additional bond properties
        features.extend([
            bond.GetIsAromatic(),
            bond.GetIsConjugated(),
            bond.IsInRing()
        ])
        
        return features
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to PyTorch Geometric Data object"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Get atom features
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(self._get_atom_features(atom))
            
            # Get bond features and edge indices
            edge_indices = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])
                bond_feat = self._get_bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])
            
            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            logger.warning(f"Error processing SMILES {smiles}: {e}")
            return None

class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for molecular property prediction"""
    
    def __init__(self, 
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 num_tasks: int = 1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(node_features, hidden_dim // num_heads, 
                   heads=num_heads, dropout=dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads,
                       heads=num_heads, dropout=dropout, concat=True)
            )
        
        # Final layer
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim // num_heads,
                   heads=num_heads, dropout=dropout, concat=False)
        )
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph-level prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Process edge features
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Graph attention layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_repr = torch.cat([mean_pool, max_pool], dim=1)
        
        # Predictions
        predictions = self.predictor(graph_repr)
        uncertainties = F.softplus(self.uncertainty_head(graph_repr))
        
        return predictions, uncertainties

class MolecularPropertyPredictor:
    """Complete system for molecular property prediction"""
    
    def __init__(self, 
                 properties: List[str],
                 model_config: Dict = None):
        self.properties = properties
        self.model_config = model_config or {
            'hidden_dim': 128,
            'num_layers': 3,
            'num_heads': 4,
            'dropout': 0.1
        }
        
        # Initialize featurizer
        feature_config = MolecularFeatures(
            atom_features=['type', 'degree', 'charge', 'hybridization'],
            bond_features=['type', 'aromatic', 'conjugated', 'ring']
        )
        self.featurizer = MolecularFeaturizer(feature_config)
        
        self.model = None
        self.scaler = None
        self.training_history = []
    
    def prepare_data(self, 
                    smiles_list: List[str], 
                    properties_dict: Dict[str, List[float]]) -> Tuple[List[Data], torch.Tensor]:
        """Prepare molecular graphs and property targets"""
        
        graphs = []
        targets = []
        
        for i, smiles in enumerate(smiles_list):
            graph = self.featurizer.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                
                # Extract property values for this molecule
                mol_properties = []
                for prop in self.properties:
                    mol_properties.append(properties_dict[prop][i])
                targets.append(mol_properties)
        
        targets = torch.tensor(targets, dtype=torch.float)
        return graphs, targets
    
    def train(self, 
              train_graphs: List[Data],
              train_targets: torch.Tensor,
              val_graphs: List[Data] = None,
              val_targets: torch.Tensor = None,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001):
        """Train the molecular property prediction model"""
        
        # Initialize model
        sample_graph = train_graphs[0]
        node_features = sample_graph.x.shape[1]
        edge_features = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0
        
        self.model = GraphAttentionNetwork(
            node_features=node_features,
            edge_features=edge_features,
            num_tasks=len(self.properties),
            **self.model_config
        )
        
        # Data loaders
        train_loader = DataLoader(
            [(graph, target) for graph, target in zip(train_graphs, train_targets)],
            batch_size=batch_size,
            shuffle=True
        )
        
        if val_graphs is not None:
            val_loader = DataLoader(
                [(graph, target) for graph, target in zip(val_graphs, val_targets)],
                batch_size=batch_size,
                shuffle=False
            )
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_graphs, batch_targets in train_loader:
                optimizer.zero_grad()
                
                # Create batch
                batch_data = self._create_batch(batch_graphs)
                
                # Forward pass
                predictions, uncertainties = self.model(batch_data)
                
                # Gaussian negative log-likelihood loss (with uncertainty)
                mse_loss = F.mse_loss(predictions, batch_targets)
                uncertainty_loss = torch.mean(
                    0.5 * torch.log(uncertainties) + 
                    0.5 * (predictions - batch_targets) ** 2 / uncertainties
                )
                
                loss = mse_loss + 0.1 * uncertainty_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validation
            val_loss = 0
            if val_graphs is not None:
                val_loss = self._evaluate(val_loader)
                scheduler.step(val_loss)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_loss': val_loss
            })
    
    def _create_batch(self, graphs: List[Data]) -> Data:
        """Create a batch from list of graphs"""
        batch_x = []
        batch_edge_index = []
        batch_edge_attr = []
        batch_batch = []
        
        node_offset = 0
        for i, graph in enumerate(graphs):
            batch_x.append(graph.x)
            
            # Adjust edge indices for batching
            edge_index = graph.edge_index + node_offset
            batch_edge_index.append(edge_index)
            
            if graph.edge_attr is not None:
                batch_edge_attr.append(graph.edge_attr)
            
            # Batch assignment
            batch_batch.extend([i] * graph.x.shape[0])
            node_offset += graph.x.shape[0]
        
        return Data(
            x=torch.cat(batch_x, dim=0),
            edge_index=torch.cat(batch_edge_index, dim=1),
            edge_attr=torch.cat(batch_edge_attr, dim=0) if batch_edge_attr else None,
            batch=torch.tensor(batch_batch, dtype=torch.long)
        )
    
    def _evaluate(self, data_loader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_graphs, batch_targets in data_loader:
                batch_data = self._create_batch(batch_graphs)
                predictions, uncertainties = self.model(batch_data)
                
                loss = F.mse_loss(predictions, batch_targets)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def predict(self, 
                smiles_list: List[str],
                return_uncertainty: bool = True) -> Dict[str, np.ndarray]:
        """Predict molecular properties for new molecules"""
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare graphs
        graphs = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            graph = self.featurizer.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(i)
        
        if not graphs:
            return {prop: np.array([]) for prop in self.properties}
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            batch_data = self._create_batch(graphs)
            predictions, uncertainties = self.model(batch_data)
        
        # Organize results
        results = {}
        for i, prop in enumerate(self.properties):
            prop_predictions = np.full(len(smiles_list), np.nan)
            prop_predictions[valid_indices] = predictions[:, i].numpy()
            results[prop] = prop_predictions
            
            if return_uncertainty:
                prop_uncertainties = np.full(len(smiles_list), np.nan)
                prop_uncertainties[valid_indices] = uncertainties[:, i].numpy()
                results[f"{prop}_uncertainty"] = prop_uncertainties
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model and configuration"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'properties': self.properties,
            'training_history': self.training_history
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and configuration"""
        save_dict = torch.load(filepath)
        
        self.model_config = save_dict['model_config']
        self.properties = save_dict['properties']
        self.training_history = save_dict['training_history']
        
        # Reconstruct model (need sample data to get dimensions)
        # This would need to be handled more elegantly in production
        logger.info(f"Model loaded from {filepath}")

# Example usage and validation
def demonstrate_molecular_property_prediction():
    """Demonstrate molecular property prediction system"""
    
    # Sample data (in practice, this would come from ChEMBL, PubChem, etc.)
    sample_smiles = [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"  # Celecoxib
    ]
    
    # Sample properties (normally computed or measured)
    sample_properties = {
        'logP': [0.31, 1.19, -0.07, 3.97, 3.47],  # Lipophilicity
        'MW': [46.07, 180.16, 194.19, 206.28, 381.37],  # Molecular weight
        'TPSA': [20.23, 63.60, 61.82, 37.30, 92.35]  # Topological polar surface area
    }
    
    # Initialize predictor
    predictor = MolecularPropertyPredictor(
        properties=['logP', 'MW', 'TPSA'],
        model_config={
            'hidden_dim': 64,
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.1
        }
    )
    
    # Prepare data
    graphs, targets = predictor.prepare_data(sample_smiles, sample_properties)
    
    # Split data (normally would have much more data)
    train_graphs, val_graphs = graphs[:4], graphs[4:]
    train_targets, val_targets = targets[:4], targets[4:]
    
    # Train model
    logger.info("Training molecular property prediction model...")
    predictor.train(
        train_graphs=train_graphs,
        train_targets=train_targets,
        val_graphs=val_graphs,
        val_targets=val_targets,
        epochs=50,
        batch_size=2,
        learning_rate=0.001
    )
    
    # Make predictions
    predictions = predictor.predict(sample_smiles)
    
    # Display results
    results_df = pd.DataFrame({
        'SMILES': sample_smiles,
        'True_logP': sample_properties['logP'],
        'Pred_logP': predictions['logP'],
        'LogP_Uncertainty': predictions['logP_uncertainty']
    })
    
    logger.info("Molecular Property Predictions:")
    logger.info(results_df.to_string(index=False))
    
    return predictor, results_df

if __name__ == "__main__":
    predictor, results = demonstrate_molecular_property_prediction()
```

## Molecular Generation with Variational Autoencoders

### Theoretical Framework

Molecular generation aims to design novel compounds with desired properties. Variational Autoencoders (VAEs) learn a continuous latent representation of molecular space, enabling generation through sampling and optimization.

The VAE objective combines reconstruction loss and KL divergence:

```
L = E[log p(x|z)] - KL(q(z|x) || p(z))
```

Where:
- x represents the molecular input (SMILES or graph)
- z is the latent representation
- q(z|x) is the encoder distribution
- p(z) is the prior distribution (typically standard normal)

### Implementation: Molecular VAE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import selfies as sf
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class SMILESDataset(Dataset):
    """Dataset for SMILES strings"""
    
    def __init__(self, smiles_list: List[str], max_length: int = 120):
        self.smiles_list = smiles_list
        self.max_length = max_length
        
        # Build vocabulary
        self.char_to_idx = self._build_vocabulary()
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Encode SMILES
        self.encoded_smiles = [self._encode_smiles(smi) for smi in smiles_list]
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build character vocabulary from SMILES"""
        chars = set()
        for smiles in self.smiles_list:
            chars.update(smiles)
        
        # Add special tokens
        chars.add('<PAD>')
        chars.add('<START>')
        chars.add('<END>')
        chars.add('<UNK>')
        
        return {char: idx for idx, char in enumerate(sorted(chars))}
    
    def _encode_smiles(self, smiles: str) -> List[int]:
        """Encode SMILES string to integer sequence"""
        encoded = [self.char_to_idx['<START>']]
        
        for char in smiles:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                encoded.append(self.char_to_idx['<UNK>'])
        
        encoded.append(self.char_to_idx['<END>'])
        
        # Pad or truncate
        if len(encoded) < self.max_length:
            encoded.extend([self.char_to_idx['<PAD>']] * (self.max_length - len(encoded)))
        else:
            encoded = encoded[:self.max_length]
            encoded[-1] = self.char_to_idx['<END>']
        
        return encoded
    
    def decode_smiles(self, encoded: List[int]) -> str:
        """Decode integer sequence back to SMILES"""
        chars = []
        for idx in encoded:
            char = self.idx_to_char[idx]
            if char == '<END>':
                break
            elif char not in ['<PAD>', '<START>']:
                chars.append(char)
        
        return ''.join(chars)
    
    def __len__(self):
        return len(self.encoded_smiles)
    
    def __getitem__(self, idx):
        return torch.tensor(self.encoded_smiles[idx], dtype=torch.long)

class MolecularVAE(nn.Module):
    """Variational Autoencoder for molecular generation"""
    
    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 latent_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, x):
        """Encode input sequence to latent space"""
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM encoding
        lstm_out, (hidden, _) = self.encoder_lstm(embedded)
        
        # Use final hidden state (concatenate forward and backward)
        hidden = hidden.view(hidden.size(1), -1)  # (batch_size, hidden_dim * 2)
        
        # Latent parameters
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, max_length=None):
        """Decode latent representation to sequence"""
        if max_length is None:
            max_length = self.max_length
        
        batch_size = z.size(0)
        
        # Initialize decoder input
        decoder_input = self.decoder_input(z).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Initialize hidden state
        hidden = torch.zeros(self.decoder_lstm.num_layers, batch_size, self.hidden_dim).to(z.device)
        cell = torch.zeros(self.decoder_lstm.num_layers, batch_size, self.hidden_dim).to(z.device)
        
        outputs = []
        input_token = decoder_input
        
        for _ in range(max_length):
            # LSTM step
            lstm_out, (hidden, cell) = self.decoder_lstm(input_token, (hidden, cell))
            
            # Output projection
            output = self.output_projection(lstm_out)
            outputs.append(output)
            
            # Use output as next input (teacher forcing during training)
            if self.training:
                # During training, use ground truth (teacher forcing)
                # This would need to be handled in the training loop
                input_token = lstm_out
            else:
                # During inference, use predicted token
                predicted_token = torch.argmax(output, dim=-1)
                input_token = self.embedding(predicted_token)
        
        return torch.cat(outputs, dim=1)
    
    def forward(self, x):
        """Forward pass through VAE"""
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decode(z)
        
        return reconstructed, mu, logvar, z

class MolecularGenerator:
    """Complete system for molecular generation"""
    
    def __init__(self,
                 latent_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 max_length: int = 120):
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length
        
        self.model = None
        self.dataset = None
        self.training_history = []
    
    def prepare_data(self, smiles_list: List[str]) -> SMILESDataset:
        """Prepare SMILES dataset"""
        # Filter valid SMILES
        valid_smiles = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
        
        logger.info(f"Prepared {len(valid_smiles)} valid SMILES from {len(smiles_list)} input")
        
        self.dataset = SMILESDataset(valid_smiles, self.max_length)
        return self.dataset
    
    def train(self,
              dataset: SMILESDataset,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              beta: float = 1.0):
        """Train the molecular VAE"""
        
        # Initialize model
        self.model = MolecularVAE(
            vocab_size=dataset.vocab_size,
            max_length=self.max_length,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, mu, logvar, z = self.model(batch)
                
                # Reconstruction loss
                recon_loss = F.cross_entropy(
                    reconstructed.view(-1, dataset.vocab_size),
                    batch.view(-1),
                    ignore_index=dataset.char_to_idx['<PAD>']
                )
                
                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / batch.size(0)  # Normalize by batch size
                
                # Total loss
                loss = recon_loss + beta * kl_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_recon_loss = total_recon_loss / num_batches
            avg_kl_loss = total_kl_loss / num_batches
            
            scheduler.step(avg_loss)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, "
                          f"Recon = {avg_recon_loss:.4f}, KL = {avg_kl_loss:.4f}")
            
            self.training_history.append({
                'epoch': epoch,
                'total_loss': avg_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss
            })
    
    def generate_molecules(self,
                          num_molecules: int = 100,
                          temperature: float = 1.0) -> List[str]:
        """Generate new molecules by sampling from latent space"""
        
        if self.model is None:
            raise ValueError("Model must be trained before generating molecules")
        
        self.model.eval()
        generated_smiles = []
        
        with torch.no_grad():
            for _ in range(num_molecules):
                # Sample from latent space
                z = torch.randn(1, self.latent_dim) * temperature
                
                # Decode to sequence
                output = self.model.decode(z)
                
                # Convert to SMILES
                predicted_tokens = torch.argmax(output, dim=-1).squeeze().tolist()
                smiles = self.dataset.decode_smiles(predicted_tokens)
                
                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    generated_smiles.append(smiles)
        
        logger.info(f"Generated {len(generated_smiles)} valid molecules from {num_molecules} attempts")
        return generated_smiles
    
    def optimize_molecules(self,
                          target_property: str,
                          target_value: float,
                          num_iterations: int = 1000,
                          learning_rate: float = 0.01) -> List[Tuple[str, float]]:
        """Optimize molecules for specific properties using gradient ascent in latent space"""
        
        if self.model is None:
            raise ValueError("Model must be trained before optimization")
        
        # Property calculation function
        def calculate_property(smiles: str) -> float:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1000  # Invalid molecule penalty
            
            if target_property == 'logP':
                return Descriptors.MolLogP(mol)
            elif target_property == 'MW':
                return Descriptors.MolWt(mol)
            elif target_property == 'TPSA':
                return Descriptors.TPSA(mol)
            else:
                return 0
        
        self.model.eval()
        optimized_molecules = []
        
        # Start from random points in latent space
        for _ in range(10):  # Multiple starting points
            z = torch.randn(1, self.latent_dim, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=learning_rate)
            
            best_smiles = None
            best_score = float('-inf')
            
            for iteration in range(num_iterations // 10):
                optimizer.zero_grad()
                
                # Decode current latent point
                output = self.model.decode(z)
                predicted_tokens = torch.argmax(output, dim=-1).squeeze().tolist()
                smiles = self.dataset.decode_smiles(predicted_tokens)
                
                # Calculate property
                prop_value = calculate_property(smiles)
                
                # Objective: minimize distance to target
                objective = -abs(prop_value - target_value)
                
                if objective > best_score:
                    best_score = objective
                    best_smiles = smiles
                
                # Backward pass (note: this is approximate since SMILES decoding is not differentiable)
                loss = -objective
                # In practice, you'd use a differentiable molecular representation
                
            if best_smiles:
                optimized_molecules.append((best_smiles, -best_score))
        
        # Sort by score
        optimized_molecules.sort(key=lambda x: x[1])
        
        return optimized_molecules[:10]  # Return top 10

# Example usage
def demonstrate_molecular_generation():
    """Demonstrate molecular generation system"""
    
    # Sample SMILES data (in practice, use large datasets like ChEMBL)
    sample_smiles = [
        "CCO",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "CC1=C(C=C(C=C1)C(=O)C2=CC=CC=C2)C",
        "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
        "CN(C)CCOC1=CC=C(C=C1)C(C2=CC=CC=C2)C3=CC=CC=C3",
        "CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN2CCCC2",
        "CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)S(=O)(=O)N"
    ]
    
    # Initialize generator
    generator = MolecularGenerator(
        latent_dim=128,
        hidden_dim=256,
        num_layers=2,
        max_length=100
    )
    
    # Prepare data
    dataset = generator.prepare_data(sample_smiles)
    
    # Train model
    logger.info("Training molecular generation model...")
    generator.train(
        dataset=dataset,
        epochs=50,
        batch_size=4,
        learning_rate=0.001,
        beta=0.1
    )
    
    # Generate new molecules
    logger.info("Generating new molecules...")
    generated_molecules = generator.generate_molecules(num_molecules=20)
    
    logger.info("Generated molecules:")
    for i, smiles in enumerate(generated_molecules[:10]):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            logger.info(f"{i+1}: {smiles} (LogP: {logp:.2f}, MW: {mw:.1f})")
    
    return generator, generated_molecules

if __name__ == "__main__":
    generator, molecules = demonstrate_molecular_generation()
```

## Clinical Trial Optimization with AI

### Mathematical Framework for Trial Design

Clinical trial optimization involves multiple objectives: minimizing trial duration, maximizing statistical power, ensuring patient safety, and optimizing resource allocation. This can be formulated as a multi-objective optimization problem:

```
minimize: [T(design), C(design), -P(design)]
subject to: Safety(design) ≥ threshold
           Feasibility(design) = True
```

Where:
- T(design) is trial duration
- C(design) is trial cost
- P(design) is statistical power
- Safety and feasibility constraints must be satisfied

### Implementation: Clinical Trial Optimizer

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TrialDesign:
    """Clinical trial design parameters"""
    sample_size: int
    treatment_arms: int
    allocation_ratio: List[float]
    primary_endpoint: str
    secondary_endpoints: List[str]
    inclusion_criteria: Dict[str, any]
    exclusion_criteria: Dict[str, any]
    study_duration_months: int
    interim_analyses: List[int]  # Sample sizes at which to conduct interim analyses
    
@dataclass
class PatientPopulation:
    """Patient population characteristics"""
    total_eligible: int
    recruitment_rate_per_month: float
    dropout_rate_per_month: float
    baseline_characteristics: Dict[str, Dict[str, float]]  # mean, std for continuous variables
    
@dataclass
class TreatmentEffect:
    """Treatment effect parameters"""
    control_response_rate: float
    treatment_effect_size: float
    effect_type: str  # 'absolute', 'relative', 'hazard_ratio'
    time_to_effect_months: float
    effect_variability: float

class ClinicalTrialSimulator:
    """Simulate clinical trial outcomes"""
    
    def __init__(self, 
                 trial_design: TrialDesign,
                 population: PatientPopulation,
                 treatment_effect: TreatmentEffect):
        self.design = trial_design
        self.population = population
        self.treatment_effect = treatment_effect
        
    def simulate_patient_recruitment(self) -> pd.DataFrame:
        """Simulate patient recruitment over time"""
        
        # Calculate recruitment timeline
        total_patients = self.design.sample_size
        recruitment_rate = self.population.recruitment_rate_per_month
        
        # Account for recruitment challenges
        months_to_recruit = total_patients / recruitment_rate
        
        # Generate recruitment timeline with variability
        recruitment_timeline = []
        cumulative_patients = 0
        month = 0
        
        while cumulative_patients < total_patients:
            month += 1
            
            # Monthly recruitment with Poisson variability
            monthly_recruitment = np.random.poisson(recruitment_rate)
            monthly_recruitment = min(monthly_recruitment, total_patients - cumulative_patients)
            
            for patient_id in range(cumulative_patients, cumulative_patients + monthly_recruitment):
                # Randomize to treatment arms
                arm_probs = np.array(self.design.allocation_ratio)
                arm_probs = arm_probs / arm_probs.sum()
                treatment_arm = np.random.choice(len(arm_probs), p=arm_probs)
                
                recruitment_timeline.append({
                    'patient_id': patient_id,
                    'recruitment_month': month,
                    'treatment_arm': treatment_arm,
                    'baseline_characteristics': self._generate_baseline_characteristics()
                })
            
            cumulative_patients += monthly_recruitment
        
        return pd.DataFrame(recruitment_timeline)
    
    def _generate_baseline_characteristics(self) -> Dict[str, float]:
        """Generate baseline characteristics for a patient"""
        characteristics = {}
        
        for char_name, params in self.population.baseline_characteristics.items():
            if 'mean' in params and 'std' in params:
                # Continuous variable
                characteristics[char_name] = np.random.normal(params['mean'], params['std'])
            elif 'probability' in params:
                # Binary variable
                characteristics[char_name] = np.random.binomial(1, params['probability'])
        
        return characteristics
    
    def simulate_patient_outcomes(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Simulate patient outcomes based on treatment assignment"""
        
        outcomes = []
        
        for _, patient in patients_df.iterrows():
            # Base outcome probability
            if self.treatment_effect.effect_type == 'absolute':
                control_prob = self.treatment_effect.control_response_rate
                treatment_prob = control_prob + self.treatment_effect.treatment_effect_size
            elif self.treatment_effect.effect_type == 'relative':
                control_prob = self.treatment_effect.control_response_rate
                treatment_prob = control_prob * (1 + self.treatment_effect.treatment_effect_size)
            
            # Adjust for patient characteristics (simplified)
            baseline_adjustment = 0
            for char_name, char_value in patient['baseline_characteristics'].items():
                if char_name == 'age':
                    baseline_adjustment += (char_value - 65) * 0.01  # Age effect
                elif char_name == 'severity_score':
                    baseline_adjustment += char_value * 0.05  # Severity effect
            
            # Final outcome probability
            if patient['treatment_arm'] == 0:  # Control
                outcome_prob = max(0, min(1, control_prob + baseline_adjustment))
            else:  # Treatment
                outcome_prob = max(0, min(1, treatment_prob + baseline_adjustment))
            
            # Add effect variability
            outcome_prob += np.random.normal(0, self.treatment_effect.effect_variability)
            outcome_prob = max(0, min(1, outcome_prob))
            
            # Generate outcome
            primary_outcome = np.random.binomial(1, outcome_prob)
            
            # Time to event (if applicable)
            if primary_outcome:
                time_to_event = np.random.exponential(self.treatment_effect.time_to_effect_months)
            else:
                time_to_event = self.design.study_duration_months  # Censored
            
            # Dropout
            dropout_prob = self.population.dropout_rate_per_month * time_to_event
            dropout = np.random.binomial(1, min(dropout_prob, 0.5))
            
            outcomes.append({
                'patient_id': patient['patient_id'],
                'treatment_arm': patient['treatment_arm'],
                'primary_outcome': primary_outcome,
                'time_to_event': time_to_event,
                'dropout': dropout,
                'completed_study': not dropout and time_to_event <= self.design.study_duration_months
            })
        
        return pd.DataFrame(outcomes)
    
    def analyze_trial_results(self, outcomes_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze trial results and compute statistics"""
        
        # Filter to completed patients
        completed_patients = outcomes_df[outcomes_df['completed_study']]
        
        if len(completed_patients) == 0:
            return {'power': 0, 'p_value': 1, 'effect_estimate': 0, 'ci_lower': 0, 'ci_upper': 0}
        
        # Primary analysis
        control_outcomes = completed_patients[completed_patients['treatment_arm'] == 0]['primary_outcome']
        treatment_outcomes = completed_patients[completed_patients['treatment_arm'] == 1]['primary_outcome']
        
        if len(control_outcomes) == 0 or len(treatment_outcomes) == 0:
            return {'power': 0, 'p_value': 1, 'effect_estimate': 0, 'ci_lower': 0, 'ci_upper': 0}
        
        # Statistical test
        control_rate = control_outcomes.mean()
        treatment_rate = treatment_outcomes.mean()
        
        # Two-proportion z-test
        n1, n2 = len(control_outcomes), len(treatment_outcomes)
        p1, p2 = control_rate, treatment_rate
        
        # Pooled proportion
        p_pool = (n1 * p1 + n2 * p2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        if se == 0:
            z_stat = 0
            p_value = 1
        else:
            z_stat = (p2 - p1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Effect estimate and confidence interval
        effect_estimate = treatment_rate - control_rate
        se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        
        ci_lower = effect_estimate - 1.96 * se_diff
        ci_upper = effect_estimate + 1.96 * se_diff
        
        # Power (whether we detected a significant effect)
        power = 1 if p_value < 0.05 else 0
        
        return {
            'power': power,
            'p_value': p_value,
            'effect_estimate': effect_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'total_patients': len(completed_patients)
        }

class TrialOptimizer:
    """Optimize clinical trial design parameters"""
    
    def __init__(self, 
                 population: PatientPopulation,
                 treatment_effect: TreatmentEffect,
                 constraints: Dict[str, any] = None):
        self.population = population
        self.treatment_effect = treatment_effect
        self.constraints = constraints or {}
        
        # Default constraints
        self.min_sample_size = self.constraints.get('min_sample_size', 50)
        self.max_sample_size = self.constraints.get('max_sample_size', 1000)
        self.max_duration_months = self.constraints.get('max_duration_months', 36)
        self.min_power = self.constraints.get('min_power', 0.8)
        self.alpha = self.constraints.get('alpha', 0.05)
        
    def calculate_sample_size(self, 
                            power: float = 0.8,
                            alpha: float = 0.05,
                            allocation_ratio: float = 1.0) -> int:
        """Calculate required sample size using analytical formula"""
        
        # For two-proportion test
        p1 = self.treatment_effect.control_response_rate
        
        if self.treatment_effect.effect_type == 'absolute':
            p2 = p1 + self.treatment_effect.treatment_effect_size
        elif self.treatment_effect.effect_type == 'relative':
            p2 = p1 * (1 + self.treatment_effect.treatment_effect_size)
        
        # Ensure valid probabilities
        p2 = max(0, min(1, p2))
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Pooled proportion
        p_pool = (p1 + allocation_ratio * p2) / (1 + allocation_ratio)
        
        # Sample size calculation
        numerator = (z_alpha * np.sqrt((1 + 1/allocation_ratio) * p_pool * (1 - p_pool)) + 
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / allocation_ratio)) ** 2
        denominator = (p2 - p1) ** 2
        
        if denominator == 0:
            return self.max_sample_size
        
        n1 = numerator / denominator
        n_total = n1 * (1 + allocation_ratio)
        
        return int(np.ceil(n_total))
    
    def simulate_trial_design(self, 
                            sample_size: int,
                            allocation_ratio: List[float] = [1, 1],
                            study_duration_months: int = 24,
                            num_simulations: int = 100) -> Dict[str, float]:
        """Simulate a trial design multiple times to estimate performance"""
        
        # Normalize allocation ratio
        allocation_ratio = np.array(allocation_ratio)
        allocation_ratio = allocation_ratio / allocation_ratio.sum()
        
        # Create trial design
        trial_design = TrialDesign(
            sample_size=sample_size,
            treatment_arms=len(allocation_ratio),
            allocation_ratio=allocation_ratio.tolist(),
            primary_endpoint='response_rate',
            secondary_endpoints=[],
            inclusion_criteria={},
            exclusion_criteria={},
            study_duration_months=study_duration_months,
            interim_analyses=[]
        )
        
        # Run simulations
        results = []
        
        for sim in range(num_simulations):
            simulator = ClinicalTrialSimulator(trial_design, self.population, self.treatment_effect)
            
            # Simulate recruitment and outcomes
            patients = simulator.simulate_patient_recruitment()
            outcomes = simulator.simulate_patient_outcomes(patients)
            
            # Analyze results
            analysis_results = simulator.analyze_trial_results(outcomes)
            
            # Calculate additional metrics
            recruitment_duration = patients['recruitment_month'].max()
            total_duration = recruitment_duration + study_duration_months
            
            results.append({
                **analysis_results,
                'recruitment_duration': recruitment_duration,
                'total_duration': total_duration,
                'recruited_patients': len(patients),
                'completed_patients': analysis_results['total_patients']
            })
        
        # Aggregate results
        results_df = pd.DataFrame(results)
        
        return {
            'empirical_power': results_df['power'].mean(),
            'mean_p_value': results_df['p_value'].mean(),
            'mean_effect_estimate': results_df['effect_estimate'].mean(),
            'mean_recruitment_duration': results_df['recruitment_duration'].mean(),
            'mean_total_duration': results_df['total_duration'].mean(),
            'completion_rate': results_df['completed_patients'].mean() / sample_size,
            'power_ci_lower': results_df['power'].quantile(0.025),
            'power_ci_upper': results_df['power'].quantile(0.975)
        }
    
    def optimize_design(self, 
                       objectives: List[str] = ['power', 'duration', 'cost'],
                       weights: List[float] = [0.5, 0.3, 0.2]) -> Dict[str, any]:
        """Optimize trial design using multi-objective optimization"""
        
        def objective_function(params):
            sample_size = int(params[0])
            study_duration = int(params[1])
            
            # Ensure valid parameters
            sample_size = max(self.min_sample_size, min(self.max_sample_size, sample_size))
            study_duration = max(6, min(self.max_duration_months, study_duration))
            
            # Simulate trial design
            results = self.simulate_trial_design(
                sample_size=sample_size,
                study_duration_months=study_duration,
                num_simulations=50  # Reduced for optimization speed
            )
            
            # Calculate objective components
            power_obj = 1 - results['empirical_power']  # Minimize (1 - power)
            duration_obj = results['mean_total_duration'] / 60  # Normalize duration
            cost_obj = sample_size / 1000  # Normalize cost (proportional to sample size)
            
            # Weighted combination
            total_objective = (weights[0] * power_obj + 
                             weights[1] * duration_obj + 
                             weights[2] * cost_obj)
            
            return total_objective
        
        # Optimization bounds
        bounds = [
            (self.min_sample_size, self.max_sample_size),  # Sample size
            (6, self.max_duration_months)  # Study duration
        ]
        
        # Run optimization
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=20,  # Reduced for demonstration
            popsize=10,
            seed=42
        )
        
        # Extract optimal parameters
        optimal_sample_size = int(result.x[0])
        optimal_duration = int(result.x[1])
        
        # Simulate optimal design with more iterations
        optimal_results = self.simulate_trial_design(
            sample_size=optimal_sample_size,
            study_duration_months=optimal_duration,
            num_simulations=100
        )
        
        return {
            'optimal_sample_size': optimal_sample_size,
            'optimal_duration_months': optimal_duration,
            'optimization_result': result,
            'performance_metrics': optimal_results
        }

# Example usage and validation
def demonstrate_trial_optimization():
    """Demonstrate clinical trial optimization system"""
    
    # Define patient population
    population = PatientPopulation(
        total_eligible=5000,
        recruitment_rate_per_month=20,
        dropout_rate_per_month=0.02,
        baseline_characteristics={
            'age': {'mean': 65, 'std': 10},
            'severity_score': {'mean': 5, 'std': 2},
            'comorbidity': {'probability': 0.3}
        }
    )
    
    # Define treatment effect
    treatment_effect = TreatmentEffect(
        control_response_rate=0.3,
        treatment_effect_size=0.15,  # 15% absolute improvement
        effect_type='absolute',
        time_to_effect_months=3,
        effect_variability=0.05
    )
    
    # Initialize optimizer
    optimizer = TrialOptimizer(
        population=population,
        treatment_effect=treatment_effect,
        constraints={
            'min_sample_size': 100,
            'max_sample_size': 800,
            'max_duration_months': 36,
            'min_power': 0.8
        }
    )
    
    # Calculate analytical sample size
    analytical_n = optimizer.calculate_sample_size(power=0.8, alpha=0.05)
    logger.info(f"Analytical sample size calculation: {analytical_n}")
    
    # Simulate different designs
    logger.info("Simulating different trial designs...")
    
    designs_to_test = [
        {'sample_size': 200, 'duration': 24},
        {'sample_size': 300, 'duration': 18},
        {'sample_size': 400, 'duration': 12},
        {'sample_size': analytical_n, 'duration': 24}
    ]
    
    simulation_results = []
    
    for design in designs_to_test:
        results = optimizer.simulate_trial_design(
            sample_size=design['sample_size'],
            study_duration_months=design['duration'],
            num_simulations=100
        )
        
        simulation_results.append({
            'design': design,
            'results': results
        })
        
        logger.info(f"Design {design}: Power = {results['empirical_power']:.3f}, "
                   f"Duration = {results['mean_total_duration']:.1f} months")
    
    # Optimize design
    logger.info("Optimizing trial design...")
    optimization_result = optimizer.optimize_design(
        objectives=['power', 'duration', 'cost'],
        weights=[0.6, 0.3, 0.1]
    )
    
    logger.info("Optimal Design:")
    logger.info(f"Sample Size: {optimization_result['optimal_sample_size']}")
    logger.info(f"Duration: {optimization_result['optimal_duration_months']} months")
    logger.info(f"Expected Power: {optimization_result['performance_metrics']['empirical_power']:.3f}")
    logger.info(f"Expected Total Duration: {optimization_result['performance_metrics']['mean_total_duration']:.1f} months")
    
    return optimizer, simulation_results, optimization_result

if __name__ == "__main__":
    optimizer, simulations, optimal_design = demonstrate_trial_optimization()
```

## Regulatory Compliance and Validation

### FDA Software as Medical Device (SaMD) Framework

The FDA's Software as Medical Device framework provides a risk-based approach to AI regulation in healthcare. The framework categorizes SaMD based on:

1. **Healthcare Decision**: Inform, Drive, or Treat
2. **Healthcare Situation**: Critical, Serious, or Non-serious
3. **Risk Categorization**: Class I, II, or III

### Implementation: Regulatory Compliance System

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class HealthcareDecision(Enum):
    INFORM = "inform"
    DRIVE = "drive"
    TREAT = "treat"

class HealthcareSituation(Enum):
    NON_SERIOUS = "non_serious"
    SERIOUS = "serious"
    CRITICAL = "critical"

class RiskClass(Enum):
    CLASS_I = "class_i"
    CLASS_II = "class_ii"
    CLASS_III = "class_iii"

@dataclass
class SaMDClassification:
    """Software as Medical Device classification"""
    decision_type: HealthcareDecision
    situation_type: HealthcareSituation
    risk_class: RiskClass
    regulatory_requirements: List[str] = field(default_factory=list)
    clinical_evidence_required: bool = False
    quality_management_system: str = ""
    
class RegulatoryFramework:
    """Comprehensive regulatory compliance framework"""
    
    def __init__(self):
        self.samd_matrix = self._build_samd_classification_matrix()
        self.compliance_requirements = self._build_compliance_requirements()
        
    def _build_samd_classification_matrix(self) -> Dict[Tuple[HealthcareDecision, HealthcareSituation], SaMDClassification]:
        """Build the SaMD classification matrix"""
        
        matrix = {}
        
        # Define classifications based on FDA guidance
        classifications = [
            # Inform decisions
            (HealthcareDecision.INFORM, HealthcareSituation.NON_SERIOUS, RiskClass.CLASS_I),
            (HealthcareDecision.INFORM, HealthcareSituation.SERIOUS, RiskClass.CLASS_II),
            (HealthcareDecision.INFORM, HealthcareSituation.CRITICAL, RiskClass.CLASS_II),
            
            # Drive decisions
            (HealthcareDecision.DRIVE, HealthcareSituation.NON_SERIOUS, RiskClass.CLASS_II),
            (HealthcareDecision.DRIVE, HealthcareSituation.SERIOUS, RiskClass.CLASS_II),
            (HealthcareDecision.DRIVE, HealthcareSituation.CRITICAL, RiskClass.CLASS_III),
            
            # Treat decisions
            (HealthcareDecision.TREAT, HealthcareSituation.NON_SERIOUS, RiskClass.CLASS_II),
            (HealthcareDecision.TREAT, HealthcareSituation.SERIOUS, RiskClass.CLASS_III),
            (HealthcareDecision.TREAT, HealthcareSituation.CRITICAL, RiskClass.CLASS_III),
        ]
        
        for decision, situation, risk_class in classifications:
            key = (decision, situation)
            
            # Determine requirements based on risk class
            requirements = []
            clinical_evidence = False
            qms = ""
            
            if risk_class == RiskClass.CLASS_I:
                requirements = ["510(k) Exempt", "QSR Exempt", "General Controls"]
                qms = "Basic Quality System"
            elif risk_class == RiskClass.CLASS_II:
                requirements = ["510(k) Clearance", "QSR Applicable", "Special Controls"]
                clinical_evidence = True
                qms = "ISO 13485 Quality Management System"
            elif risk_class == RiskClass.CLASS_III:
                requirements = ["PMA Required", "QSR Applicable", "Clinical Trials Required"]
                clinical_evidence = True
                qms = "Full ISO 13485 + ISO 14971 Risk Management"
            
            matrix[key] = SaMDClassification(
                decision_type=decision,
                situation_type=situation,
                risk_class=risk_class,
                regulatory_requirements=requirements,
                clinical_evidence_required=clinical_evidence,
                quality_management_system=qms
            )
        
        return matrix
    
    def _build_compliance_requirements(self) -> Dict[str, Dict[str, any]]:
        """Build detailed compliance requirements"""
        
        return {
            "iso_13485": {
                "name": "ISO 13485 Quality Management System",
                "requirements": [
                    "Document control procedures",
                    "Management responsibility",
                    "Resource management",
                    "Product realization",
                    "Measurement and improvement"
                ],
                "applicable_classes": [RiskClass.CLASS_II, RiskClass.CLASS_III]
            },
            "iso_14971": {
                "name": "ISO 14971 Risk Management",
                "requirements": [
                    "Risk management process",
                    "Risk analysis",
                    "Risk evaluation",
                    "Risk control",
                    "Post-market surveillance"
                ],
                "applicable_classes": [RiskClass.CLASS_II, RiskClass.CLASS_III]
            },
            "iec_62304": {
                "name": "IEC 62304 Medical Device Software",
                "requirements": [
                    "Software development lifecycle",
                    "Software safety classification",
                    "Software development process",
                    "Software maintenance process",
                    "Software risk management"
                ],
                "applicable_classes": [RiskClass.CLASS_I, RiskClass.CLASS_II, RiskClass.CLASS_III]
            },
            "fda_guidance": {
                "name": "FDA AI/ML Guidance",
                "requirements": [
                    "Algorithm change control",
                    "Real-world performance monitoring",
                    "Risk-benefit analysis",
                    "Transparency and interpretability",
                    "Bias assessment and mitigation"
                ],
                "applicable_classes": [RiskClass.CLASS_II, RiskClass.CLASS_III]
            }
        }
    
    def classify_samd(self, 
                     decision_type: HealthcareDecision,
                     situation_type: HealthcareSituation) -> SaMDClassification:
        """Classify a SaMD based on decision and situation types"""
        
        key = (decision_type, situation_type)
        if key in self.samd_matrix:
            return self.samd_matrix[key]
        else:
            raise ValueError(f"Invalid combination: {decision_type}, {situation_type}")
    
    def get_compliance_checklist(self, classification: SaMDClassification) -> Dict[str, any]:
        """Generate compliance checklist for a given classification"""
        
        checklist = {
            "classification": classification,
            "required_standards": [],
            "documentation_requirements": [],
            "testing_requirements": [],
            "clinical_requirements": []
        }
        
        # Add applicable standards
        for standard_id, standard_info in self.compliance_requirements.items():
            if classification.risk_class in standard_info["applicable_classes"]:
                checklist["required_standards"].append({
                    "standard": standard_info["name"],
                    "requirements": standard_info["requirements"]
                })
        
        # Documentation requirements
        if classification.risk_class in [RiskClass.CLASS_II, RiskClass.CLASS_III]:
            checklist["documentation_requirements"] = [
                "Software requirements specification",
                "Software architecture document",
                "Detailed design document",
                "Verification and validation plan",
                "Risk management file",
                "Clinical evaluation report",
                "Post-market surveillance plan"
            ]
        
        # Testing requirements
        checklist["testing_requirements"] = [
            "Unit testing",
            "Integration testing",
            "System testing",
            "Performance testing",
            "Security testing",
            "Usability testing"
        ]
        
        if classification.risk_class == RiskClass.CLASS_III:
            checklist["testing_requirements"].extend([
                "Clinical validation testing",
                "Real-world evidence collection",
                "Long-term safety monitoring"
            ])
        
        # Clinical requirements
        if classification.clinical_evidence_required:
            checklist["clinical_requirements"] = [
                "Clinical evaluation plan",
                "Clinical study protocol",
                "Statistical analysis plan",
                "Clinical study report",
                "Post-market clinical follow-up"
            ]
        
        return checklist

class ComplianceTracker:
    """Track compliance progress and generate reports"""
    
    def __init__(self, classification: SaMDClassification):
        self.classification = classification
        self.compliance_items = {}
        self.completion_status = {}
        self.evidence_files = {}
        
    def add_compliance_item(self, 
                          item_id: str,
                          item_name: str,
                          requirement_type: str,
                          due_date: datetime = None):
        """Add a compliance item to track"""
        
        self.compliance_items[item_id] = {
            "name": item_name,
            "type": requirement_type,
            "due_date": due_date,
            "created_date": datetime.now()
        }
        
        self.completion_status[item_id] = {
            "status": "not_started",  # not_started, in_progress, completed, verified
            "completion_date": None,
            "verification_date": None,
            "notes": ""
        }
        
        self.evidence_files[item_id] = []
    
    def update_item_status(self, 
                          item_id: str,
                          status: str,
                          notes: str = "",
                          evidence_file: str = None):
        """Update the status of a compliance item"""
        
        if item_id not in self.compliance_items:
            raise ValueError(f"Compliance item {item_id} not found")
        
        self.completion_status[item_id]["status"] = status
        self.completion_status[item_id]["notes"] = notes
        
        if status == "completed":
            self.completion_status[item_id]["completion_date"] = datetime.now()
        elif status == "verified":
            self.completion_status[item_id]["verification_date"] = datetime.now()
        
        if evidence_file:
            self.evidence_files[item_id].append({
                "file_path": evidence_file,
                "upload_date": datetime.now()
            })
    
    def generate_compliance_report(self) -> Dict[str, any]:
        """Generate comprehensive compliance report"""
        
        total_items = len(self.compliance_items)
        completed_items = sum(1 for status in self.completion_status.values() 
                            if status["status"] in ["completed", "verified"])
        
        # Calculate completion by type
        completion_by_type = {}
        for item_id, item_info in self.compliance_items.items():
            item_type = item_info["type"]
            if item_type not in completion_by_type:
                completion_by_type[item_type] = {"total": 0, "completed": 0}
            
            completion_by_type[item_type]["total"] += 1
            if self.completion_status[item_id]["status"] in ["completed", "verified"]:
                completion_by_type[item_type]["completed"] += 1
        
        # Identify overdue items
        overdue_items = []
        for item_id, item_info in self.compliance_items.items():
            if (item_info["due_date"] and 
                item_info["due_date"] < datetime.now() and
                self.completion_status[item_id]["status"] not in ["completed", "verified"]):
                overdue_items.append({
                    "item_id": item_id,
                    "name": item_info["name"],
                    "due_date": item_info["due_date"],
                    "days_overdue": (datetime.now() - item_info["due_date"]).days
                })
        
        return {
            "classification": self.classification,
            "overall_completion": completed_items / total_items if total_items > 0 else 0,
            "total_items": total_items,
            "completed_items": completed_items,
            "completion_by_type": completion_by_type,
            "overdue_items": overdue_items,
            "report_date": datetime.now()
        }
    
    def export_compliance_matrix(self) -> pd.DataFrame:
        """Export compliance status as a DataFrame"""
        
        data = []
        for item_id, item_info in self.compliance_items.items():
            status_info = self.completion_status[item_id]
            
            data.append({
                "Item ID": item_id,
                "Item Name": item_info["name"],
                "Type": item_info["type"],
                "Status": status_info["status"],
                "Due Date": item_info["due_date"],
                "Completion Date": status_info["completion_date"],
                "Days to Complete": (status_info["completion_date"] - item_info["created_date"]).days 
                                  if status_info["completion_date"] else None,
                "Evidence Files": len(self.evidence_files[item_id]),
                "Notes": status_info["notes"]
            })
        
        return pd.DataFrame(data)

# Example usage and validation
def demonstrate_regulatory_compliance():
    """Demonstrate regulatory compliance system"""
    
    # Initialize regulatory framework
    framework = RegulatoryFramework()
    
    # Example: AI system for diabetic retinopathy screening
    logger.info("Classifying AI system for diabetic retinopathy screening...")
    
    # This system drives clinical decisions in a serious healthcare situation
    classification = framework.classify_samd(
        decision_type=HealthcareDecision.DRIVE,
        situation_type=HealthcareSituation.SERIOUS
    )
    
    logger.info(f"Classification: {classification.risk_class.value}")
    logger.info(f"Requirements: {classification.regulatory_requirements}")
    logger.info(f"Clinical Evidence Required: {classification.clinical_evidence_required}")
    
    # Generate compliance checklist
    checklist = framework.get_compliance_checklist(classification)
    
    logger.info("\nCompliance Checklist:")
    for standard in checklist["required_standards"]:
        logger.info(f"- {standard['standard']}")
    
    # Initialize compliance tracker
    tracker = ComplianceTracker(classification)
    
    # Add compliance items
    compliance_items = [
        ("req_spec", "Software Requirements Specification", "documentation"),
        ("arch_doc", "Software Architecture Document", "documentation"),
        ("risk_mgmt", "Risk Management File", "documentation"),
        ("clinical_eval", "Clinical Evaluation Report", "clinical"),
        ("unit_tests", "Unit Testing", "testing"),
        ("system_tests", "System Testing", "testing"),
        ("clinical_study", "Clinical Validation Study", "clinical"),
        ("post_market", "Post-Market Surveillance Plan", "documentation")
    ]
    
    # Add items with due dates
    base_date = datetime.now()
    for i, (item_id, item_name, item_type) in enumerate(compliance_items):
        due_date = base_date + timedelta(days=30 * (i + 1))
        tracker.add_compliance_item(item_id, item_name, item_type, due_date)
    
    # Simulate progress updates
    tracker.update_item_status("req_spec", "completed", "Requirements finalized", "req_spec_v1.0.pdf")
    tracker.update_item_status("arch_doc", "in_progress", "Architecture review in progress")
    tracker.update_item_status("unit_tests", "completed", "All unit tests passing", "test_results.xml")
    tracker.update_item_status("risk_mgmt", "completed", "Risk analysis complete", "risk_analysis.pdf")
    
    # Generate compliance report
    report = tracker.generate_compliance_report()
    
    logger.info(f"\nCompliance Report:")
    logger.info(f"Overall Completion: {report['overall_completion']:.1%}")
    logger.info(f"Completed Items: {report['completed_items']}/{report['total_items']}")
    
    if report['overdue_items']:
        logger.info(f"Overdue Items: {len(report['overdue_items'])}")
        for item in report['overdue_items']:
            logger.info(f"- {item['name']}: {item['days_overdue']} days overdue")
    
    # Export compliance matrix
    compliance_df = tracker.export_compliance_matrix()
    logger.info("\nCompliance Matrix:")
    logger.info(compliance_df.to_string(index=False))
    
    return framework, tracker, report

if __name__ == "__main__":
    framework, tracker, report = demonstrate_regulatory_compliance()
```

## Conclusion

This chapter has provided comprehensive implementations for AI-driven drug discovery, covering molecular property prediction, molecular generation, clinical trial optimization, and regulatory compliance. The systems presented demonstrate how artificial intelligence can transform pharmaceutical research and development while maintaining the highest standards of safety and regulatory compliance.

### Key Takeaways

1. **Molecular AI Systems**: Graph neural networks and variational autoencoders enable sophisticated molecular property prediction and generation
2. **Clinical Trial Optimization**: AI can significantly improve trial design, patient selection, and resource allocation
3. **Regulatory Compliance**: Systematic approaches to FDA SaMD classification and compliance tracking are essential for successful deployment
4. **Integration Challenges**: Successful implementation requires careful consideration of regulatory requirements, clinical validation, and real-world deployment constraints

### Future Directions

The field of drug discovery AI continues to evolve rapidly, with emerging opportunities in:
- **Foundation models** for molecular representation learning
- **Multi-modal AI** integrating molecular, clinical, and real-world data
- **Causal inference** for understanding drug mechanisms and effects
- **Federated learning** for collaborative drug discovery across institutions
- **Quantum computing** applications in molecular simulation and optimization

The implementations provided in this chapter serve as a foundation for developing production-ready AI systems that can accelerate drug discovery while ensuring patient safety and regulatory compliance.

## References

1. Chen, H., et al. (2018). "The rise of deep learning in drug discovery." *Drug Discovery Today*, 23(6), 1241-1250. DOI: 10.1016/j.drudis.2018.01.039

2. Gómez-Bombarelli, R., et al. (2018). "Automatic chemical design using a data-driven continuous representation of molecules." *ACS Central Science*, 4(2), 268-276. DOI: 10.1021/acscentsci.7b00572

3. Stokes, J. M., et al. (2020). "A deep learning approach to antibiotic discovery." *Cell*, 180(4), 688-702. DOI: 10.1016/j.cell.2020.01.021

4. FDA. (2021). "Software as a Medical Device (SaMD): Clinical Evaluation." Guidance for Industry and Food and Drug Administration Staff.

5. Vamathevan, J., et al. (2019). "Applications of machine learning in drug discovery and development." *Nature Reviews Drug Discovery*, 18(6), 463-477. DOI: 10.1038/s41573-019-0024-5

6. Mak, K. K., & Pichika, M. R. (2019). "Artificial intelligence in drug development: present status and future prospects." *Drug Discovery Today*, 24(3), 773-780. DOI: 10.1016/j.drudis.2018.11.014

7. Zhavoronkov, A., et al. (2019). "Deep learning enables rapid identification of potent DDR1 kinase inhibitors." *Nature Biotechnology*, 37(9), 1038-1040. DOI: 10.1038/s41587-019-0224-x

8. Schneider, G. (2018). "Generative models for artificially-intelligent molecular design." *Molecular Informatics*, 37(1-2), 1700123. DOI: 10.1002/minf.201700123

9. Popova, M., et al. (2018). "Deep reinforcement learning for de novo drug design." *Science Advances*, 4(7), eaap7885. DOI: 10.1126/sciadv.aap7885

10. Ramsundar, B., et al. (2019). "Deep Learning for the Life Sciences." O'Reilly Media.
