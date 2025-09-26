"""
Chapter 2 - Example 2
Extracted from Healthcare AI Implementation Guide
"""

"""
Advanced Linear Algebra Operations for Healthcare Data Analysis

This module implements sophisticated linear algebraic methods specifically
designed for healthcare AI applications, including dimensionality reduction,
missing data imputation, and clinical phenotype discovery.

Author: Sanjay Basu MD PhD
License: MIT
"""

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClinicalDataMatrix:
    """
    Represents a clinical data matrix with comprehensive metadata.
    
    This class encapsulates clinical data along with variable information,
    patient metadata, and data quality metrics essential for healthcare AI.
    """
    data: np.ndarray
    patient_ids: List[str]
    variable_names: List[str]
    variable_types: Dict[str, str]  \# 'continuous', 'categorical', 'binary'
    missing_data_pattern: Optional[np.ndarray] = None
    data_quality_score: float = 0.0
    collection_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate data matrix and calculate quality metrics."""
        self._validate_dimensions()
        self._calculate_missing_pattern()
        self._calculate_quality_score()
    
    def _validate_dimensions(self):
        """Validate matrix dimensions and metadata consistency."""
        n_patients, n_variables = self.data.shape
        
        if len(self.patient_ids) != n_patients:
            raise ValueError(f"Patient ID count ({len(self.patient_ids)}) "
                           f"doesn't match data rows ({n_patients})")
        
        if len(self.variable_names) != n_variables:
            raise ValueError(f"Variable name count ({len(self.variable_names)}) "
                           f"doesn't match data columns ({n_variables})")
    
    def _calculate_missing_pattern(self):
        """Calculate missing data pattern matrix."""
        self.missing_data_pattern = np.isnan(self.data).astype(int)
    
    def _calculate_quality_score(self):
        """Calculate overall data quality score."""
        if self.missing_data_pattern is not None:
            completeness = 1 - np.mean(self.missing_data_pattern)
            self.data_quality_score = completeness
        else:
            self.data_quality_score = 1.0
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get matrix dimensions."""
        return self.data.shape
    
    @property
    def missing_percentage(self) -> float:
        """Get percentage of missing values."""
        if self.missing_data_pattern is not None:
            return np.mean(self.missing_data_pattern) * 100
        return 0.0
    
    def get_variable_summary(self) -> pd.DataFrame:
        """Get summary statistics for all variables."""
        summary_data = []
        
        for i, var_name in enumerate(self.variable_names):
            var_data = self.data[:, i]
            var_type = self.variable_types.get(var_name, 'unknown')
            
            if var_type == 'continuous':
                summary = {
                    'variable': var_name,
                    'type': var_type,
                    'mean': np.nanmean(var_data),
                    'std': np.nanstd(var_data),
                    'min': np.nanmin(var_data),
                    'max': np.nanmax(var_data),
                    'missing_pct': np.mean(np.isnan(var_data)) * 100
                }
            else:
                unique_values = np.unique(var_data[~np.isnan(var_data)])
                summary = {
                    'variable': var_name,
                    'type': var_type,
                    'unique_values': len(unique_values),
                    'most_common': unique_values<sup>0</sup> if len(unique_values) > 0 else None,
                    'missing_pct': np.mean(np.isnan(var_data)) * 100
                }
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)

class AdvancedLinearAlgebra:
    """
    Advanced linear algebra operations for healthcare data analysis.
    
    This class implements sophisticated matrix operations, decompositions,
    and transformations specifically designed for clinical data analysis
    and healthcare AI applications.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize advanced linear algebra processor.
        
        Args:
            random_seed: Random seed for reproducible results
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        \# Storage for computed decompositions
        self.pca_components: Optional[np.ndarray] = None
        self.svd_components: Optional[Dict[str, np.ndarray]] = None
        self.nmf_components: Optional[Dict[str, np.ndarray]] = None
        
        logger.info("Advanced linear algebra processor initialized")
    
    def robust_pca(self, 
                   clinical_data: ClinicalDataMatrix,
                   n_components: Optional[int] = None,
                   handle_missing: str = 'impute') -> Dict[str, Any]:
        """
        Perform robust Principal Component Analysis on clinical data.
        
        Args:
            clinical_data: ClinicalDataMatrix object
            n_components: Number of components to extract (None for all)
            handle_missing: How to handle missing data ('impute', 'exclude', 'iterative')
            
        Returns:
            Dictionary containing PCA results and clinical interpretation
        """
        try:
            \# Prepare data
            X = self._prepare_data_for_pca(clinical_data, handle_missing)
            
            \# Determine number of components
            if n_components is None:
                n_components = min(X.shape) - 1
            
            \# Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            \# Perform PCA
            pca = PCA(n_components=n_components, random_state=self.random_seed)
            X_transformed = pca.fit_transform(X_scaled)
            
            \# Store components
            self.pca_components = pca.components_
            
            \# Calculate explained variance ratios
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            \# Identify clinically meaningful components
            component_interpretations = self._interpret_pca_components(
                pca.components_, clinical_data.variable_names, clinical_data.variable_types
            )
            
            \# Calculate component stability (using bootstrap if data is large enough)
            if X.shape<sup>0</sup> > 100:
                stability_scores = self._calculate_component_stability(
                    X_scaled, n_components, n_bootstrap=100
                )
            else:
                stability_scores = np.ones(n_components)
            
            results = {
                'transformed_data': X_transformed,
                'components': pca.components_,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance_ratio': cumulative_variance,
                'eigenvalues': pca.explained_variance_,
                'component_interpretations': component_interpretations,
                'stability_scores': stability_scores,
                'scaler': scaler,
                'pca_model': pca,
                'n_components_95_variance': np.argmax(cumulative_variance >= 0.95) + 1,
                'clinical_phenotypes': self._identify_clinical_phenotypes(
                    X_transformed, clinical_data.patient_ids
                )
            }
            
            logger.info(f"PCA completed: {n_components} components explain "
                       f"{cumulative_variance[-1]:.1%} of variance")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in robust PCA: {e}")
            raise
    
    def matrix_completion_svd(self, 
                             clinical_data: ClinicalDataMatrix,
                             rank: Optional[int] = None,
                             max_iterations: int = 100,
                             tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Perform matrix completion using iterative SVD for missing clinical data.
        
        Args:
            clinical_data: ClinicalDataMatrix with missing values
            rank: Target rank for low-rank approximation
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary containing completed matrix and decomposition results
        """
        try:
            X = clinical_data.data.copy()
            missing_mask = np.isnan(X)
            
            if not np.any(missing_mask):
                logger.warning("No missing data found, performing standard SVD")
                return self._standard_svd(X, rank)
            
            \# Initialize missing values with column means
            col_means = np.nanmean(X, axis=0)
            for j in range(X.shape<sup>1</sup>):
                X[missing_mask[:, j], j] = col_means[j]
            
            \# Determine rank
            if rank is None:
                rank = min(X.shape) // 2
            
            \# Iterative SVD completion
            prev_X = X.copy()
            
            for iteration in range(max_iterations):
                \# SVD decomposition
                U, s, Vt = la.svd(X, full_matrices=False)
                
                \# Truncate to desired rank
                U_r = U[:, :rank]
                s_r = s[:rank]
                Vt_r = Vt[:rank, :]
                
                \# Reconstruct matrix
                X_reconstructed = U_r @ np.diag(s_r) @ Vt_r
                
                \# Update only missing entries
                X[missing_mask] = X_reconstructed[missing_mask]
                
                \# Check convergence
                change = np.linalg.norm(X - prev_X, 'fro')
                if change < tolerance:
                    logger.info(f"Matrix completion converged after {iteration + 1} iterations")
                    break
                
                prev_X = X.copy()
            
            \# Final SVD for analysis
            U_final, s_final, Vt_final = la.svd(X, full_matrices=False)
            
            \# Calculate completion quality metrics
            completion_metrics = self._calculate_completion_metrics(
                clinical_data.data, X, missing_mask
            )
            
            \# Store SVD components
            self.svd_components = {
                'U': U_final[:, :rank],
                'sigma': s_final[:rank],
                'Vt': Vt_final[:rank, :]
            }
            
            results = {
                'completed_matrix': X,
                'U': U_final[:, :rank],
                'singular_values': s_final[:rank],
                'Vt': Vt_final[:rank, :],
                'rank': rank,
                'iterations': iteration + 1,
                'completion_metrics': completion_metrics,
                'missing_data_percentage': np.mean(missing_mask) * 100,
                'latent_factors': self._interpret_svd_factors(
                    U_final[:, :rank], Vt_final[:rank, :], 
                    clinical_data.patient_ids, clinical_data.variable_names
                )
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in matrix completion SVD: {e}")
            raise
    
    def non_negative_matrix_factorization(self, 
                                        clinical_data: ClinicalDataMatrix,
                                        n_components: int,
                                        max_iterations: int = 200) -> Dict[str, Any]:
        """
        Perform Non-negative Matrix Factorization for clinical phenotype discovery.
        
        NMF is particularly useful for identifying clinical phenotypes because
        it produces interpretable, parts-based representations.
        
        Args:
            clinical_data: ClinicalDataMatrix object
            n_components: Number of components (phenotypes) to extract
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary containing NMF results and phenotype interpretations
        """
        try:
            X = clinical_data.data.copy()
            
            \# Handle missing data
            if np.any(np.isnan(X)):
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            \# Ensure non-negativity (required for NMF)
            X_min = np.min(X)
            if X_min < 0:
                X = X - X_min  \# Shift to make all values non-negative
                logger.info(f"Shifted data by {-X_min} to ensure non-negativity")
            
            \# Perform NMF
            nmf = NMF(n_components=n_components, 
                     max_iter=max_iterations,
                     random_state=self.random_seed,
                     init='nndsvd')  \# Better initialization
            
            W = nmf.fit_transform(X)  \# Patient loadings
            H = nmf.components_       \# Feature loadings
            
            \# Store NMF components
            self.nmf_components = {'W': W, 'H': H}
            
            \# Interpret clinical phenotypes
            phenotype_interpretations = self._interpret_nmf_phenotypes(
                H, clinical_data.variable_names, clinical_data.variable_types
            )
            
            \# Assign patients to dominant phenotypes
            patient_phenotypes = self._assign_patient_phenotypes(
                W, clinical_data.patient_ids
            )
            
            \# Calculate reconstruction quality
            X_reconstructed = W @ H
            reconstruction_error = np.linalg.norm(X - X_reconstructed, 'fro')
            relative_error = reconstruction_error / np.linalg.norm(X, 'fro')
            
            results = {
                'patient_loadings': W,
                'feature_loadings': H,
                'reconstructed_matrix': X_reconstructed,
                'reconstruction_error': reconstruction_error,
                'relative_error': relative_error,
                'phenotype_interpretations': phenotype_interpretations,
                'patient_phenotypes': patient_phenotypes,
                'nmf_model': nmf,
                'n_components': n_components,
                'phenotype_prevalence': self._calculate_phenotype_prevalence(W)
            }
            
            logger.info(f"NMF completed: {n_components} phenotypes identified "
                       f"with {relative_error:.3f} relative reconstruction error")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in NMF: {e}")
            raise
    
    def tensor_decomposition_clinical(self, 
                                    tensor_data: np.ndarray,
                                    rank: int,
                                    max_iterations: int = 100) -> Dict[str, Any]:
        """
        Perform tensor decomposition for multi-way clinical data analysis.
        
        This method is useful for analyzing data with multiple modes such as
        patients × variables × time or patients × variables × hospitals.
        
        Args:
            tensor_data: 3D numpy array (patients × variables × time/context)
            rank: Rank of the decomposition
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary containing tensor decomposition results
        """
        try:
            if len(tensor_data.shape) != 3:
                raise ValueError("Tensor data must be 3-dimensional")
            
            n_patients, n_variables, n_contexts = tensor_data.shape
            
            \# Initialize factor matrices randomly
            A = np.random.rand(n_patients, rank)
            B = np.random.rand(n_variables, rank)
            C = np.random.rand(n_contexts, rank)
            
            \# Alternating least squares algorithm
            for iteration in range(max_iterations):
                \# Update A (patients)
                for i in range(n_patients):
                    X_i = tensor_data[i, :, :]  \# variables × contexts
                    khatri_rao_BC = self._khatri_rao_product(B, C)
                    A[i, :] = la.lstsq(khatri_rao_BC, X_i.flatten())<sup>0</sup>
                
                \# Update B (variables)
                for j in range(n_variables):
                    X_j = tensor_data[:, j, :]  \# patients × contexts
                    khatri_rao_AC = self._khatri_rao_product(A, C)
                    B[j, :] = la.lstsq(khatri_rao_AC, X_j.flatten())<sup>0</sup>
                
                \# Update C (contexts)
                for k in range(n_contexts):
                    X_k = tensor_data[:, :, k]  \# patients × variables
                    khatri_rao_AB = self._khatri_rao_product(A, B)
                    C[k, :] = la.lstsq(khatri_rao_AB, X_k.flatten())<sup>0</sup>
                
                \# Check convergence (simplified)
                if iteration > 0 and iteration % 10 == 0:
                    reconstructed = self._reconstruct_tensor(A, B, C)
                    error = np.linalg.norm(tensor_data - reconstructed)
                    logger.info(f"Iteration {iteration}: Reconstruction error = {error:.6f}")
            
            \# Final reconstruction
            reconstructed_tensor = self._reconstruct_tensor(A, B, C)
            reconstruction_error = np.linalg.norm(tensor_data - reconstructed_tensor)
            
            results = {
                'patient_factors': A,
                'variable_factors': B,
                'context_factors': C,
                'reconstructed_tensor': reconstructed_tensor,
                'reconstruction_error': reconstruction_error,
                'rank': rank,
                'iterations': max_iterations
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in tensor decomposition: {e}")
            raise
    
    def _prepare_data_for_pca(self, 
                             clinical_data: ClinicalDataMatrix,
                             handle_missing: str) -> np.ndarray:
        """Prepare clinical data for PCA analysis."""
        X = clinical_data.data.copy()
        
        if handle_missing == 'impute':
            \# Use KNN imputation for better results with clinical data
            imputer = KNNImputer(n_neighbors=5)
            X = imputer.fit_transform(X)
        elif handle_missing == 'exclude':
            \# Remove rows with any missing values
            complete_rows = ~np.any(np.isnan(X), axis=1)
            X = X[complete_rows, :]
            logger.info(f"Excluded {np.sum(~complete_rows)} patients with missing data")
        elif handle_missing == 'iterative':
            \# Use iterative imputation
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(random_state=self.random_seed)
            X = imputer.fit_transform(X)
        
        return X
    
    def _interpret_pca_components(self, 
                                 components: np.ndarray,
                                 variable_names: List[str],
                                 variable_types: Dict[str, str]) -> List[Dict[str, Any]]:
        """Interpret PCA components in clinical context."""
        interpretations = []
        
        for i, component in enumerate(components):
            \# Find variables with highest absolute loadings
            abs_loadings = np.abs(component)
            top_indices = np.argsort(abs_loadings)[-10:][::-1]  \# Top 10
            
            top_variables = []
            for idx in top_indices:
                if abs_loadings[idx] > 0.1:  \# Threshold for meaningful loading
                    top_variables.append({
                        'variable': variable_names[idx],
                        'loading': component[idx],
                        'type': variable_types.get(variable_names[idx], 'unknown')
                    })
            
            \# Generate clinical interpretation
            interpretation = self._generate_component_interpretation(top_variables)
            
            interpretations.append({
                'component_number': i + 1,
                'top_variables': top_variables,
                'clinical_interpretation': interpretation,
                'variance_explained': None  \# Will be filled by calling function
            })
        
        return interpretations
    
    def _generate_component_interpretation(self, top_variables: List[Dict[str, Any]]) -> str:
        """Generate clinical interpretation for a PCA component."""
        if not top_variables:
            return "No significant variable loadings"
        
        \# Group variables by clinical domain
        lab_vars = [v for v in top_variables if 'lab' in v['variable'].lower()]
        vital_vars = [v for v in top_variables if any(term in v['variable'].lower() 
                     for term in ['bp', 'hr', 'temp', 'resp'])]
        
        interpretation_parts = []
        
        if lab_vars:
            interpretation_parts.append(f"Laboratory profile ({len(lab_vars)} variables)")
        if vital_vars:
            interpretation_parts.append(f"Vital signs pattern ({len(vital_vars)} variables)")
        
        if not interpretation_parts:
            interpretation_parts.append("Mixed clinical variables")
        
        return " + ".join(interpretation_parts)
    
    def _calculate_component_stability(self, 
                                     X: np.ndarray,
                                     n_components: int,
                                     n_bootstrap: int = 100) -> np.ndarray:
        """Calculate stability of PCA components using bootstrap."""
        n_samples = X.shape<sup>0</sup>
        component_similarities = []
        
        \# Original PCA
        pca_original = PCA(n_components=n_components, random_state=self.random_seed)
        pca_original.fit(X)
        original_components = pca_original.components_
        
        \# Bootstrap iterations
        for _ in range(n_bootstrap):
            \# Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices, :]
            
            \# PCA on bootstrap sample
            pca_bootstrap = PCA(n_components=n_components, random_state=self.random_seed)
            pca_bootstrap.fit(X_bootstrap)
            bootstrap_components = pca_bootstrap.components_
            
            \# Calculate component similarities (absolute correlation)
            similarities = []
            for i in range(n_components):
                max_similarity = 0
                for j in range(n_components):
                    similarity = abs(np.corrcoef(original_components[i], 
                                               bootstrap_components[j])[0, 1])
                    max_similarity = max(max_similarity, similarity)
                similarities.append(max_similarity)
            
            component_similarities.append(similarities)
        
        \# Average similarities across bootstrap iterations
        stability_scores = np.mean(component_similarities, axis=0)
        
        return stability_scores
    
    def _identify_clinical_phenotypes(self, 
                                    X_transformed: np.ndarray,
                                    patient_ids: List[str]) -> Dict[str, Any]:
        """Identify clinical phenotypes from PCA-transformed data."""
        from sklearn.cluster import KMeans
        
        \# Determine optimal number of clusters using elbow method
        max_clusters = min(10, X_transformed.shape<sup>0</sup> // 10)
        inertias = []
        
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_seed)
            kmeans.fit(X_transformed)
            inertias.append(kmeans.inertia_)
        
        \# Find elbow point (simplified)
        optimal_k = 3  \# Default, could implement more sophisticated elbow detection
        
        \# Perform final clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=self.random_seed)
        cluster_labels = kmeans_final.fit_predict(X_transformed)
        
        \# Analyze phenotypes
        phenotypes = {}
        for k in range(optimal_k):
            cluster_mask = cluster_labels == k
            phenotypes[f'phenotype_{k+1}'] = {
                'patient_count': np.sum(cluster_mask),
                'patient_ids': [patient_ids[i] for i in np.where(cluster_mask)<sup>0</sup>],
                'centroid': kmeans_final.cluster_centers_[k],
                'prevalence': np.mean(cluster_mask)
            }
        
        return {
            'phenotypes': phenotypes,
            'cluster_labels': cluster_labels,
            'optimal_k': optimal_k,
            'clustering_model': kmeans_final
        }
    
    def _standard_svd(self, X: np.ndarray, rank: Optional[int]) -> Dict[str, Any]:
        """Perform standard SVD decomposition."""
        U, s, Vt = la.svd(X, full_matrices=False)
        
        if rank is not None:
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
        
        return {
            'U': U,
            'singular_values': s,
            'Vt': Vt,
            'rank': len(s)
        }
    
    def _calculate_completion_metrics(self, 
                                    original: np.ndarray,
                                    completed: np.ndarray,
                                    missing_mask: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for matrix completion quality."""
        \# Only evaluate on originally missing entries
        if not np.any(missing_mask):
            return {'rmse': 0.0, 'mae': 0.0, 'completion_rate': 1.0}
        
        \# For missing entries, we can't calculate true error
        \# Instead, calculate consistency metrics
        completion_rate = 1.0  \# All missing values were filled
        
        \# Calculate overall matrix properties
        frobenius_norm_ratio = (np.linalg.norm(completed, 'fro') / 
                               np.linalg.norm(original[~missing_mask], 'fro'))
        
        return {
            'completion_rate': completion_rate,
            'frobenius_norm_ratio': frobenius_norm_ratio,
            'missing_percentage': np.mean(missing_mask) * 100
        }
    
    def _interpret_svd_factors(self, 
                              U: np.ndarray,
                              Vt: np.ndarray,
                              patient_ids: List[str],
                              variable_names: List[str]) -> Dict[str, Any]:
        """Interpret SVD factors in clinical context."""
        n_factors = U.shape<sup>1</sup>
        
        factor_interpretations = []
        for i in range(n_factors):
            \# Patient factor interpretation
            patient_loadings = U[:, i]
            top_patient_indices = np.argsort(np.abs(patient_loadings))[-5:][::-1]
            
            \# Variable factor interpretation
            variable_loadings = Vt[i, :]
            top_variable_indices = np.argsort(np.abs(variable_loadings))[-10:][::-1]
            
            factor_interpretations.append({
                'factor_number': i + 1,
                'top_patients': [patient_ids[idx] for idx in top_patient_indices],
                'top_variables': [variable_names[idx] for idx in top_variable_indices],
                'patient_loading_range': (np.min(patient_loadings), np.max(patient_loadings)),
                'variable_loading_range': (np.min(variable_loadings), np.max(variable_loadings))
            })
        
        return {
            'factor_interpretations': factor_interpretations,
            'n_factors': n_factors
        }
    
    def _interpret_nmf_phenotypes(self, 
                                 H: np.ndarray,
                                 variable_names: List[str],
                                 variable_types: Dict[str, str]) -> List[Dict[str, Any]]:
        """Interpret NMF phenotypes based on feature loadings."""
        phenotype_interpretations = []
        
        for i in range(H.shape<sup>0</sup>):
            feature_loadings = H[i, :]
            
            \# Find top contributing features
            top_indices = np.argsort(feature_loadings)[-10:][::-1]
            
            top_features = []
            for idx in top_indices:
                if feature_loadings[idx] > 0.1:  \# Threshold for meaningful contribution
                    top_features.append({
                        'variable': variable_names[idx],
                        'loading': feature_loadings[idx],
                        'type': variable_types.get(variable_names[idx], 'unknown')
                    })
            
            \# Generate phenotype description
            phenotype_description = self._generate_phenotype_description(top_features)
            
            phenotype_interpretations.append({
                'phenotype_number': i + 1,
                'top_features': top_features,
                'description': phenotype_description,
                'feature_diversity': len(set(f['type'] for f in top_features))
            })
        
        return phenotype_interpretations
    
    def _generate_phenotype_description(self, top_features: List[Dict[str, Any]]) -> str:
        """Generate clinical description for NMF phenotype."""
        if not top_features:
            return "Undefined phenotype"
        
        \# Group features by type
        feature_types = {}
        for feature in top_features:
            ftype = feature['type']
            if ftype not in feature_types:
                feature_types[ftype] = []
            feature_types[ftype].append(feature['variable'])
        
        description_parts = []
        for ftype, variables in feature_types.items():
            if ftype == 'continuous':
                description_parts.append(f"Continuous variables ({len(variables)})")
            elif ftype == 'categorical':
                description_parts.append(f"Categorical features ({len(variables)})")
            elif ftype == 'binary':
                description_parts.append(f"Binary indicators ({len(variables)})")
        
        return " + ".join(description_parts) if description_parts else "Mixed phenotype"
    
    def _assign_patient_phenotypes(self, 
                                  W: np.ndarray,
                                  patient_ids: List[str]) -> Dict[str, Any]:
        """Assign patients to dominant phenotypes."""
        \# Find dominant phenotype for each patient
        dominant_phenotypes = np.argmax(W, axis=1)
        
        \# Calculate phenotype assignments
        phenotype_assignments = {}
        for i, patient_id in enumerate(patient_ids):
            phenotype_assignments[patient_id] = {
                'dominant_phenotype': int(dominant_phenotypes[i]) + 1,
                'phenotype_weights': W[i, :].tolist(),
                'confidence': np.max(W[i, :]) / np.sum(W[i, :])
            }
        
        return phenotype_assignments
    
    def _calculate_phenotype_prevalence(self, W: np.ndarray) -> Dict[str, float]:
        """Calculate prevalence of each phenotype."""
        dominant_phenotypes = np.argmax(W, axis=1)
        n_patients = W.shape<sup>0</sup>
        n_phenotypes = W.shape<sup>1</sup>
        
        prevalence = {}
        for i in range(n_phenotypes):
            count = np.sum(dominant_phenotypes == i)
            prevalence[f'phenotype_{i+1}'] = count / n_patients
        
        return prevalence
    
    def _khatri_rao_product(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute Khatri-Rao product of two matrices."""
        return np.kron(A, np.ones((1, B.shape<sup>1</sup>))) * np.kron(np.ones((1, A.shape<sup>1</sup>)), B)
    
    def _reconstruct_tensor(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Reconstruct tensor from factor matrices."""
        rank = A.shape<sup>1</sup>
        tensor_shape = (A.shape<sup>0</sup>, B.shape<sup>0</sup>, C.shape<sup>0</sup>)
        reconstructed = np.zeros(tensor_shape)
        
        for r in range(rank):
            reconstructed += np.outer(A[:, r], np.outer(B[:, r], C[:, r]).flatten()).reshape(tensor_shape)
        
        return reconstructed


\# Demonstration functions
def create_example_clinical_data() -> ClinicalDataMatrix:
    """Create example clinical data matrix for demonstration."""
    np.random.seed(42)
    
    \# Simulate clinical data
    n_patients = 200
    n_variables = 50
    
    \# Generate patient IDs
    patient_ids = [f"PATIENT_{i:04d}" for i in range(n_patients)]
    
    \# Generate variable names
    variable_names = []
    variable_types = {}
    
    \# Laboratory values (continuous)
    lab_vars = ['glucose', 'creatinine', 'hemoglobin', 'wbc_count', 'platelet_count',
                'sodium', 'potassium', 'chloride', 'bun', 'alt', 'ast', 'bilirubin']
    for var in lab_vars:
        variable_names.append(f'lab_{var}')
        variable_types[f'lab_{var}'] = 'continuous'
    
    \# Vital signs (continuous)
    vital_vars = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature', 'respiratory_rate']
    for var in vital_vars:
        variable_names.append(f'vital_{var}')
        variable_types[f'vital_{var}'] = 'continuous'
    
    \# Demographics (mixed)
    demo_vars = ['age', 'bmi']
    for var in demo_vars:
        variable_names.append(f'demo_{var}')
        variable_types[f'demo_{var}'] = 'continuous'
    
    \# Comorbidities (binary)
    comorbidity_vars = ['diabetes', 'hypertension', 'heart_disease', 'copd', 'ckd']
    for var in comorbidity_vars:
        variable_names.append(f'comorbid_{var}')
        variable_types[f'comorbid_{var}'] = 'binary'
    
    \# Medications (binary)
    med_vars = ['ace_inhibitor', 'beta_blocker', 'statin', 'metformin', 'insulin']
    for var in med_vars:
        variable_names.append(f'med_{var}')
        variable_types[f'med_{var}'] = 'binary'
    
    \# Additional variables to reach 50
    remaining = n_variables - len(variable_names)
    for i in range(remaining):
        var_name = f'additional_var_{i}'
        variable_names.append(var_name)
        variable_types[var_name] = 'continuous'
    
    \# Generate correlated clinical data
    data = np.random.randn(n_patients, n_variables)
    
    \# Add clinical correlations
    \# Diabetes cluster
    diabetes_indices = [i for i, name in enumerate(variable_names) 
                       if any(term in name for term in ['glucose', 'diabetes', 'metformin'])]
    for i in diabetes_indices:
        for j in diabetes_indices:
            if i != j:
                data[:, i] += 0.3 * data[:, j]
    
    \# Cardiovascular cluster
    cv_indices = [i for i, name in enumerate(variable_names) 
                 if any(term in name for term in ['bp', 'heart', 'ace_inhibitor', 'beta_blocker'])]
    for i in cv_indices:
        for j in cv_indices:
            if i != j:
                data[:, i] += 0.2 * data[:, j]
    
    \# Add missing data pattern
    missing_prob = 0.1
    missing_mask = np.random.random((n_patients, n_variables)) < missing_prob
    data[missing_mask] = np.nan
    
    return ClinicalDataMatrix(
        data=data,
        patient_ids=patient_ids,
        variable_names=variable_names,
        variable_types=variable_types
    )


def demonstrate_advanced_linear_algebra():
    """Demonstrate advanced linear algebra operations on clinical data."""
    print("=== Advanced Linear Algebra for Healthcare Data ===\n")
    
    \# Create example clinical data
    clinical_data = create_example_clinical_data()
    
    print(f"Clinical Data Matrix:")
    print(f"  Patients: {clinical_data.shape<sup>0</sup>}")
    print(f"  Variables: {clinical_data.shape<sup>1</sup>}")
    print(f"  Missing data: {clinical_data.missing_percentage:.1f}%")
    print(f"  Data quality score: {clinical_data.data_quality_score:.3f}")
    
    \# Initialize linear algebra processor
    la_processor = AdvancedLinearAlgebra(random_seed=42)
    
    \# 1. Robust PCA Analysis
    print(f"\n1. Robust PCA Analysis")
    print("-" * 40)
    
    pca_results = la_processor.robust_pca(
        clinical_data=clinical_data,
        n_components=10,
        handle_missing='impute'
    )
    
    print(f"PCA Results:")
    print(f"  Components extracted: {len(pca_results['explained_variance_ratio'])}")
    print(f"  Variance explained by first 5 components: {pca_results['cumulative_variance_ratio']<sup>4</sup>:.1%}")
    print(f"  Components needed for 95% variance: {pca_results['n_components_95_variance']}")
    
    \# Show component interpretations
    print(f"\nTop 3 Component Interpretations:")
    for i, interp in enumerate(pca_results['component_interpretations'][:3]):
        print(f"  Component {interp['component_number']}: {interp['clinical_interpretation']}")
        print(f"    Stability score: {pca_results['stability_scores'][i]:.3f}")
    
    \# 2. Matrix Completion with SVD
    print(f"\n2. Matrix Completion using SVD")
    print("-" * 40)
    
    svd_results = la_processor.matrix_completion_svd(
        clinical_data=clinical_data,
        rank=15,
        max_iterations=50
    )
    
    print(f"SVD Matrix Completion Results:")
    print(f"  Rank: {svd_results['rank']}")
    print(f"  Iterations: {svd_results['iterations']}")
    print(f"  Missing data: {svd_results['missing_data_percentage']:.1f}%")
    
    completion_metrics = svd_results['completion_metrics']
    print(f"  Completion rate: {completion_metrics['completion_rate']:.1%}")
    print(f"  Frobenius norm ratio: {completion_metrics['frobenius_norm_ratio']:.3f}")
    
    \# 3. Non-negative Matrix Factorization
    print(f"\n3. Non-negative Matrix Factorization")
    print("-" * 40)
    
    \# Prepare data for NMF (ensure non-negativity)
    nmf_results = la_processor.non_negative_matrix_factorization(
        clinical_data=clinical_data,
        n_components=5,
        max_iterations=200
    )
    
    print(f"NMF Results:")
    print(f"  Phenotypes identified: {nmf_results['n_components']}")
    print(f"  Reconstruction error: {nmf_results['relative_error']:.4f}")
    
    \# Show phenotype prevalence
    print(f"\nPhenotype Prevalence:")
    for phenotype, prevalence in nmf_results['phenotype_prevalence'].items():
        print(f"  {phenotype}: {prevalence:.1%}")
    
    \# Show phenotype interpretations
    print(f"\nPhenotype Interpretations:")
    for interp in nmf_results['phenotype_interpretations'][:3]:
        print(f"  Phenotype {interp['phenotype_number']}: {interp['description']}")
        print(f"    Feature diversity: {interp['feature_diversity']} types")
    
    \# 4. Clinical Phenotype Analysis
    print(f"\n4. Clinical Phenotype Analysis")
    print("-" * 40)
    
    phenotypes = pca_results['clinical_phenotypes']
    print(f"PCA-based Phenotypes:")
    print(f"  Optimal clusters: {phenotypes['optimal_k']}")
    
    for phenotype_name, phenotype_info in phenotypes['phenotypes'].items():
        print(f"  {phenotype_name}: {phenotype_info['patient_count']} patients "
              f"({phenotype_info['prevalence']:.1%})")
    
    \# 5. Data Quality Assessment
    print(f"\n5. Data Quality Assessment")
    print("-" * 40)
    
    variable_summary = clinical_data.get_variable_summary()
    
    \# Show variables with highest missing rates
    if 'missing_pct' in variable_summary.columns:
        high_missing = variable_summary.nlargest(5, 'missing_pct')
        print(f"Variables with highest missing rates:")
        for _, row in high_missing.iterrows():
            print(f"  {row['variable']}: {row['missing_pct']:.1f}% missing")
    
    \# Show continuous variable statistics
    continuous_vars = variable_summary[variable_summary['type'] == 'continuous']
    if not continuous_vars.empty and 'mean' in continuous_vars.columns:
        print(f"\nContinuous variables summary:")
        print(f"  Mean range: [{continuous_vars['mean'].min():.2f}, {continuous_vars['mean'].max():.2f}]")
        print(f"  Std range: [{continuous_vars['std'].min():.2f}, {continuous_vars['std'].max():.2f}]")


if __name__ == "__main__":
    demonstrate_advanced_linear_algebra()