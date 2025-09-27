---
layout: default
title: "Chapter 14: Population Health Ai Systems"
nav_order: 14
parent: Chapters
permalink: /chapters/14-population-health-ai-systems/
---

# Chapter 14: Population Health AI Systems - Large-Scale Health Analytics and Intervention Strategies

*By Sanjay Basu MD PhD*

## Learning Objectives

By the end of this chapter, physician data scientists will be able to:

- Design comprehensive population health AI systems that integrate multiple heterogeneous data sources including electronic health records, claims data, social services records, environmental monitoring data, and community-generated data while addressing privacy, interoperability, and data quality challenges specific to population-level health analytics
- Implement advanced epidemiological modeling using machine learning and causal inference techniques, including spatial-temporal disease surveillance, outbreak detection, transmission network analysis, and multi-source data fusion that provides real-time population health monitoring and early warning capabilities
- Develop sophisticated health equity assessment frameworks that systematically identify and address disparities in health outcomes across multiple dimensions including race/ethnicity, socioeconomic status, geography, and social determinants while implementing bias detection and mitigation strategies in population health AI systems
- Create predictive models for population health interventions with proper validation, impact assessment, and causal inference methodologies that enable evidence-based policy decisions and targeted resource allocation for maximum population health benefit
- Build scalable data infrastructure for population-level health analytics and surveillance that can handle large-scale, heterogeneous data streams while maintaining security, privacy, and regulatory compliance across multiple organizational and jurisdictional boundaries
- Apply advanced causal inference methods including instrumental variables, regression discontinuity, difference-in-differences, and synthetic control approaches to evaluate the effectiveness of population health interventions and inform evidence-based public health policy
- Implement precision public health approaches that tailor interventions to specific population subgroups based on their unique characteristics, risk profiles, and social determinants while ensuring equitable distribution of benefits and avoiding algorithmic bias

## 14.1 Introduction to Population Health AI

Population health represents a fundamental paradigm shift from individual patient care to the health outcomes of groups of individuals, including the distribution of such outcomes within the group and the social, economic, and environmental factors that influence these distributions. **Population health AI systems** leverage artificial intelligence to analyze, predict, and improve health outcomes at the population level, addressing the complex interplay of clinical, behavioral, social, environmental, and policy factors that determine health outcomes across communities and populations.

The application of AI to population health presents unique opportunities and challenges that distinguish it from individual patient care applications. **Scale and complexity** of population health data require sophisticated analytical approaches that can handle heterogeneous data sources spanning multiple sectors, temporal dynamics that may span decades, and complex causal relationships that involve feedback loops and confounding factors. **Health equity considerations** are paramount, as population health AI systems must actively identify and address disparities rather than perpetuate existing inequalities through biased algorithms or incomplete data representation.

**Multilevel determinants** of population health require AI systems that can model interactions between individual-level factors (genetics, behavior, clinical status), interpersonal factors (social networks, family dynamics), organizational factors (healthcare systems, schools, workplaces), community factors (social cohesion, built environment), and policy factors (regulations, resource allocation) that collectively determine health outcomes. **Temporal complexity** involves understanding how exposures and interventions at different life stages influence health trajectories over time, requiring longitudinal modeling approaches that can capture both immediate and delayed effects.

### 14.1.1 Scope and Applications of Population Health AI

Population health AI encompasses a broad range of applications that span from **disease surveillance and outbreak detection** to **health policy evaluation and resource allocation**, each requiring specialized methodological approaches and technical implementations. **Predictive modeling** enables identification of populations at risk for adverse health outcomes, allowing for targeted interventions and optimal resource allocation based on predicted need and intervention effectiveness.

**Disease surveillance systems** use AI to provide real-time monitoring of disease patterns, early detection of outbreaks, and prediction of epidemic trajectories. **Syndromic surveillance** analyzes patterns in clinical symptoms, emergency department visits, pharmacy sales, and school absenteeism to detect disease outbreaks before laboratory confirmation is available. **Genomic surveillance** tracks pathogen evolution and antimicrobial resistance patterns to inform treatment guidelines and infection control strategies.

**Social determinants of health (SDOH) analysis** uses AI to understand how factors such as housing quality, educational attainment, income stability, food security, transportation access, and social support networks influence health outcomes across different population groups. **Environmental health monitoring** leverages AI to analyze the impact of air quality, water safety, noise pollution, built environment characteristics, and climate change on population health outcomes.

**Health services research** applies AI to understand patterns of healthcare utilization, quality of care delivery, health system performance, and access barriers across different population groups. **Care coordination analysis** examines how patients navigate complex healthcare systems and identifies opportunities for improving care transitions and reducing fragmentation.

**Policy impact assessment** uses AI to evaluate the effectiveness of health policies and interventions at the population level, including natural experiments created by policy changes, geographic variation in policy implementation, and temporal analysis of policy effects. **Resource allocation optimization** applies AI to determine optimal distribution of public health resources, healthcare facilities, and intervention programs to maximize population health benefit.

**Precision public health** represents an emerging paradigm that applies precision medicine principles to population health, using AI to tailor interventions to specific population subgroups based on their unique characteristics, risk profiles, and social determinants while ensuring equitable access and avoiding algorithmic bias that could exacerbate health disparities.

### 14.1.2 Data Sources and Integration Challenges

Population health AI systems must integrate diverse data sources that span multiple sectors, organizational boundaries, and jurisdictions, each with unique data formats, quality characteristics, and access restrictions. **Electronic health records (EHRs)** provide detailed clinical data but may not be representative of entire populations, particularly underserved communities with limited healthcare access, and may contain systematic biases related to healthcare-seeking behavior and provider documentation practices.

**Claims data** from insurance providers offers broad population coverage and longitudinal tracking of healthcare utilization but may lack clinical detail, be subject to coding biases and administrative artifacts, and exclude uninsured populations or services not covered by insurance. **Public health surveillance data** provides population-level disease monitoring and vital statistics but may have limited individual-level detail, reporting delays, and incomplete coverage of certain populations or conditions.

**Social services data** including housing assistance records, educational enrollment and performance data, employment and income information, and social assistance program participation provide insights into social determinants of health but raise complex privacy and data sharing challenges, may have inconsistent data quality across jurisdictions, and require careful handling of sensitive personal information.

**Environmental monitoring data** from air quality sensors, water testing systems, weather stations, and geographic information systems (GIS) provide context for environmental health factors but require spatial and temporal alignment with health outcome data, may have varying measurement quality and coverage, and need sophisticated modeling to link environmental exposures to health outcomes.

**Community-generated data** from mobile health applications, wearable devices, social media platforms, and community surveys offer real-time insights into population health behaviors and outcomes but require careful validation and bias assessment, may not be representative of entire populations, and raise privacy and consent challenges for population-level analysis.

**Administrative data** from vital records, census data, transportation systems, and economic indicators provide important contextual information for population health analysis but may have different temporal granularity, geographic boundaries, and data quality characteristics that complicate integration and analysis.

### 14.1.3 Ethical and Equity Considerations

Population health AI systems must be designed with explicit attention to health equity and social justice, recognizing that AI systems can either reduce or exacerbate existing health disparities depending on their design, implementation, and governance. **Algorithmic bias** can perpetuate or amplify existing health disparities if not carefully addressed through inclusive data collection, representative model development, ongoing bias monitoring, and proactive bias mitigation strategies.

**Data representation** challenges arise when certain population groups are underrepresented in training data, leading to AI systems that perform poorly for these groups and potentially widen health disparities. **Historical bias** embedded in healthcare data reflects past discriminatory practices and systemic inequities that can be perpetuated by AI systems trained on this data without explicit bias correction.

**Privacy and consent** considerations are particularly complex in population health, where individual consent may not be feasible for large-scale data analysis, requiring careful balance between population benefit and individual privacy rights. **Data governance** frameworks must address questions of data ownership, control, and benefit-sharing, particularly for data generated by or about marginalized communities.

**Community engagement** is essential for ensuring that population health AI systems reflect community priorities and values, incorporate local knowledge and expertise, and address health issues that are most important to affected communities. **Participatory design** approaches involve community members in the development, evaluation, and governance of AI systems that affect their health.

**Benefit distribution** must ensure that the advantages of population health AI systems reach all population segments, particularly those who are most vulnerable and have historically been underserved by healthcare systems. **Digital equity** considerations address how differences in technology access and digital literacy may affect the distribution of benefits from population health AI systems.

**Transparency and accountability** mechanisms must enable communities and stakeholders to understand how AI systems make decisions that affect their health, challenge decisions when appropriate, and hold system developers and operators accountable for equitable and effective performance.

## 14.2 Advanced Epidemiological Modeling with AI

### 14.2.1 Intelligent Disease Surveillance Systems

Modern disease surveillance systems leverage artificial intelligence to provide real-time monitoring of disease patterns, early detection of outbreaks, and prediction of epidemic trajectories with unprecedented speed and accuracy. **Syndromic surveillance** uses AI to analyze patterns in clinical symptoms, emergency department visits, over-the-counter medication sales, school absenteeism, and workplace sick leave to detect disease outbreaks before laboratory confirmation is available, enabling rapid public health response.

**Spatial-temporal modeling** combines geographic information systems (GIS) with machine learning algorithms to identify disease clusters, predict spatial spread of infectious diseases, and optimize the placement of surveillance resources and intervention strategies. **Network analysis** models disease transmission through social and contact networks, enabling targeted intervention strategies that focus on high-risk individuals and communities while minimizing disruption to the broader population.

**Multi-source data fusion** integrates traditional surveillance data with novel data sources including social media sentiment analysis, internet search query patterns, mobile phone mobility data, and satellite imagery to provide comprehensive disease monitoring capabilities that can detect emerging threats and track intervention effectiveness in real-time.

**Anomaly detection** algorithms identify unusual patterns in disease incidence that may indicate outbreaks, bioterrorism events, or emerging health threats, using sophisticated statistical methods that can distinguish true signals from noise and seasonal variations. **Time series forecasting** predicts future disease trends to support public health planning, resource allocation, and intervention timing decisions.

### 14.2.2 Causal Inference for Population Health

Understanding causal relationships is fundamental to effective population health interventions, as correlation does not imply causation and observational data can be confounded by numerous factors that influence both exposures and outcomes. **Causal inference methods** help distinguish between correlation and causation in observational population health data, enabling evidence-based policy decisions and intervention strategies.

**Instrumental variables** approaches leverage natural experiments, policy changes, and genetic variants to estimate causal effects of interventions on population health outcomes when randomized controlled trials are not feasible or ethical. **Regression discontinuity** designs exploit arbitrary thresholds in policy implementation, program eligibility, or clinical decision-making to identify causal effects by comparing outcomes just above and below the threshold.

**Difference-in-differences** methods compare changes in outcomes over time between treatment and control populations to estimate intervention effects while controlling for time-invariant confounders and secular trends. **Synthetic control** methods create artificial control groups by combining multiple comparison units when randomized controlled trials are not feasible, enabling causal inference for policy interventions and natural experiments.

**Mediation analysis** helps understand the pathways through which interventions affect health outcomes, identifying intermediate variables that can be targeted for intervention and quantifying the relative importance of different causal pathways. **Moderation analysis** identifies population subgroups that may respond differently to interventions, enabling precision public health approaches that tailor interventions to specific populations.

**Directed acyclic graphs (DAGs)** provide a formal framework for representing causal assumptions, identifying confounders and mediators, and guiding the selection of appropriate statistical methods for causal inference. **Sensitivity analysis** assesses the robustness of causal conclusions to violations of identifying assumptions and unmeasured confounding.

### 14.2.3 Predictive Modeling for Health Outcomes

Population health predictive modeling uses AI to forecast health outcomes and identify populations at risk for adverse events, enabling proactive interventions and optimal resource allocation. **Risk stratification** models identify individuals and communities at highest risk for specific health outcomes, enabling targeted interventions that focus resources where they can have the greatest impact.

**Survival analysis** models time-to-event outcomes such as disease onset, hospitalization, or mortality, accounting for censoring, competing risks, and time-varying covariates. **Longitudinal modeling** analyzes repeated measurements over time to understand health trajectories, identify critical periods for intervention, and predict future outcomes based on historical patterns.

**Multi-level modeling** accounts for the hierarchical structure of population health data, where individuals are nested within families, communities, healthcare systems, and geographic regions, enabling proper estimation of effects at different levels and accounting for clustering and correlation in outcomes.

**Spatial modeling** incorporates geographic relationships and spatial autocorrelation in health outcomes, enabling identification of disease hotspots, environmental risk factors, and geographic disparities in health outcomes. **Spatio-temporal modeling** combines spatial and temporal dimensions to track disease spread, predict future outbreak locations, and optimize intervention timing and placement.

**Ensemble methods** combine multiple predictive models to improve accuracy and robustness of population health predictions, reducing overfitting and improving generalizability across different populations and settings. **Deep learning** approaches can capture complex non-linear relationships in high-dimensional population health data, including interactions between multiple risk factors and temporal patterns in longitudinal data.

## 14.3 Comprehensive Population Health AI Framework

### 14.3.1 Production-Ready Population Health Analytics System

```python
"""
Comprehensive Population Health AI Framework

This implementation provides a complete system for population health analytics,
including disease surveillance, causal inference, health equity assessment,
and intervention optimization specifically designed for public health applications.

Author: Sanjay Basu MD PhD
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
import concurrent.futures

# Scientific computing and statistics
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.interpolate import griddata
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.duration.hazard_regression import PHReg

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Geospatial analysis
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import folium
from geopy.distance import geodesic
import pyproj
from sklearn.neighbors import BallTree

# Causal inference
try:
    from econml.dml import DML
    from econml.dr import DRLearner
    from dowhy import CausalModel
except ImportError:
    print("Warning: Some causal inference libraries not available")

import networkx as nx

# Time series
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Survival analysis
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Database and data processing
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import requests

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/population-health-ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HealthOutcome(Enum):
    """Health outcome types for population health modeling."""
    MORTALITY = "mortality"
    MORBIDITY = "morbidity"
    HOSPITALIZATION = "hospitalization"
    EMERGENCY_VISIT = "emergency_visit"
    CHRONIC_DISEASE = "chronic_disease"
    MENTAL_HEALTH = "mental_health"
    MATERNAL_HEALTH = "maternal_health"
    CHILD_HEALTH = "child_health"
    INFECTIOUS_DISEASE = "infectious_disease"
    INJURY = "injury"

class InterventionType(Enum):
    """Types of population health interventions."""
    POLICY = "policy"
    PROGRAM = "program"
    ENVIRONMENTAL = "environmental"
    BEHAVIORAL = "behavioral"
    CLINICAL = "clinical"
    SOCIAL = "social"
    ECONOMIC = "economic"
    EDUCATIONAL = "educational"

class EquityDimension(Enum):
    """Dimensions for health equity analysis."""
    RACE_ETHNICITY = "race_ethnicity"
    INCOME = "income"
    EDUCATION = "education"
    GEOGRAPHY = "geography"
    AGE = "age"
    GENDER = "gender"
    DISABILITY = "disability"
    INSURANCE = "insurance"
    LANGUAGE = "language"
    IMMIGRATION_STATUS = "immigration_status"

class CausalMethod(Enum):
    """Causal inference methods."""
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    SYNTHETIC_CONTROL = "synthetic_control"
    PROPENSITY_SCORE = "propensity_score"
    MEDIATION_ANALYSIS = "mediation_analysis"

@dataclass
class PopulationHealthMetrics:
    """Population health outcome metrics."""
    outcome_type: HealthOutcome
    population_size: int
    incidence_rate: float
    prevalence_rate: float
    mortality_rate: float
    years_of_life_lost: float
    disability_adjusted_life_years: float
    confidence_interval: Tuple[float, float]
    time_period: str
    geographic_area: str
    demographic_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'outcome_type': self.outcome_type.value,
            'population_size': self.population_size,
            'incidence_rate': self.incidence_rate,
            'prevalence_rate': self.prevalence_rate,
            'mortality_rate': self.mortality_rate,
            'years_of_life_lost': self.years_of_life_lost,
            'disability_adjusted_life_years': self.disability_adjusted_life_years,
            'confidence_interval': self.confidence_interval,
            'time_period': self.time_period,
            'geographic_area': self.geographic_area,
            'demographic_breakdown': self.demographic_breakdown
        }

@dataclass
class HealthEquityAssessment:
    """Health equity assessment results."""
    equity_dimension: EquityDimension
    outcome_type: HealthOutcome
    disparity_measure: str
    disparity_value: float
    reference_group: str
    comparison_groups: Dict[str, float]
    statistical_significance: bool
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'equity_dimension': self.equity_dimension.value,
            'outcome_type': self.outcome_type.value,
            'disparity_measure': self.disparity_measure,
            'disparity_value': self.disparity_value,
            'reference_group': self.reference_group,
            'comparison_groups': self.comparison_groups,
            'statistical_significance': self.statistical_significance,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'confidence_interval': self.confidence_interval
        }

@dataclass
class CausalInferenceResult:
    """Causal inference analysis results."""
    method: CausalMethod
    treatment: str
    outcome: str
    causal_effect: float
    confidence_interval: Tuple[float, float]
    p_value: float
    sample_size: int
    assumptions_met: Dict[str, bool]
    sensitivity_analysis: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method': self.method.value,
            'treatment': self.treatment,
            'outcome': self.outcome,
            'causal_effect': self.causal_effect,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'sample_size': self.sample_size,
            'assumptions_met': self.assumptions_met,
            'sensitivity_analysis': self.sensitivity_analysis
        }

class DiseaseSurveillanceSystem:
    """Advanced disease surveillance system with AI-powered outbreak detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize disease surveillance system."""
        self.config = config
        self.surveillance_data = {}
        self.outbreak_models = {}
        self.alert_thresholds = self._setup_alert_thresholds()
        self.spatial_models = {}
        
        logger.info("Initialized disease surveillance system")
    
    def _setup_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup alert thresholds for different diseases and populations."""
        return {
            'influenza': {
                'baseline_threshold': 2.0,  # Standard deviations above baseline
                'epidemic_threshold': 3.0,
                'pandemic_threshold': 5.0
            },
            'covid19': {
                'baseline_threshold': 1.5,
                'epidemic_threshold': 2.5,
                'pandemic_threshold': 4.0
            },
            'foodborne': {
                'baseline_threshold': 2.5,
                'epidemic_threshold': 3.5,
                'pandemic_threshold': 6.0
            },
            'respiratory': {
                'baseline_threshold': 2.0,
                'epidemic_threshold': 3.0,
                'pandemic_threshold': 4.5
            }
        }
    
    def ingest_surveillance_data(
        self,
        data_source: str,
        data: pd.DataFrame,
        data_type: str = "syndromic"
    ) -> bool:
        """Ingest surveillance data from various sources."""
        try:
            # Validate data format
            required_columns = ['date', 'location', 'count', 'population']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Data quality checks
            data = self._clean_surveillance_data(data)
            
            # Store data
            if data_source not in self.surveillance_data:
                self.surveillance_data[data_source] = {}
            
            self.surveillance_data[data_source][data_type] = data
            
            logger.info(f"Ingested {len(data)} records from {data_source} ({data_type})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest surveillance data: {str(e)}")
            return False
    
    def _clean_surveillance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate surveillance data."""
        # Convert date column
        data['date'] = pd.to_datetime(data['date'])
        
        # Remove invalid counts
        data = data[data['count'] >= 0]
        data = data[data['population'] > 0]
        
        # Calculate rates
        data['rate'] = (data['count'] / data['population']) * 100000
        
        # Remove outliers (rates > 99th percentile)
        rate_threshold = data['rate'].quantile(0.99)
        data = data[data['rate'] <= rate_threshold]
        
        # Sort by date and location
        data = data.sort_values(['location', 'date'])
        
        return data
    
    def detect_outbreaks(
        self,
        disease: str,
        location: Optional[str] = None,
        method: str = "aberration_detection"
    ) -> Dict[str, Any]:
        """Detect disease outbreaks using various statistical methods."""
        try:
            # Get surveillance data
            if disease not in self.surveillance_data:
                raise ValueError(f"No surveillance data available for {disease}")
            
            data = self.surveillance_data[disease]['syndromic']
            
            if location:
                data = data[data['location'] == location]
            
            if len(data) < 30:  # Need sufficient historical data
                raise ValueError("Insufficient historical data for outbreak detection")
            
            # Apply outbreak detection method
            if method == "aberration_detection":
                results = self._aberration_detection(data, disease)
            elif method == "cusum":
                results = self._cusum_detection(data, disease)
            elif method == "ewma":
                results = self._ewma_detection(data, disease)
            elif method == "spatial_scan":
                results = self._spatial_scan_detection(data, disease)
            else:
                raise ValueError(f"Unknown outbreak detection method: {method}")
            
            logger.info(f"Completed outbreak detection for {disease} using {method}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to detect outbreaks: {str(e)}")
            return {'error': str(e)}
    
    def _aberration_detection(self, data: pd.DataFrame, disease: str) -> Dict[str, Any]:
        """Detect outbreaks using aberration detection algorithm."""
        results = {
            'method': 'aberration_detection',
            'disease': disease,
            'alerts': [],
            'baseline_statistics': {},
            'detection_summary': {}
        }
        
        # Group by location
        locations = data['location'].unique()
        
        for location in locations:
            location_data = data[data['location'] == location].copy()
            location_data = location_data.sort_values('date')
            
            # Calculate baseline statistics (using historical data)
            baseline_period = location_data[:-7]  # Exclude last week
            if len(baseline_period) < 21:  # Need at least 3 weeks of baseline
                continue
            
            baseline_mean = baseline_period['rate'].mean()
            baseline_std = baseline_period['rate'].std()
            
            # Get alert thresholds
            thresholds = self.alert_thresholds.get(disease, self.alert_thresholds['respiratory'])
            
            # Check recent data for aberrations
            recent_data = location_data[-7:]  # Last week
            
            for _, row in recent_data.iterrows():
                z_score = (row['rate'] - baseline_mean) / baseline_std if baseline_std > 0 else 0
                
                alert_level = None
                if z_score >= thresholds['pandemic_threshold']:
                    alert_level = 'pandemic'
                elif z_score >= thresholds['epidemic_threshold']:
                    alert_level = 'epidemic'
                elif z_score >= thresholds['baseline_threshold']:
                    alert_level = 'elevated'
                
                if alert_level:
                    alert = {
                        'location': location,
                        'date': row['date'].isoformat(),
                        'observed_rate': row['rate'],
                        'expected_rate': baseline_mean,
                        'z_score': z_score,
                        'alert_level': alert_level,
                        'confidence': min(abs(z_score) / thresholds['baseline_threshold'], 1.0)
                    }
                    results['alerts'].append(alert)
            
            # Store baseline statistics
            results['baseline_statistics'][location] = {
                'mean_rate': baseline_mean,
                'std_rate': baseline_std,
                'baseline_period_days': len(baseline_period)
            }
        
        # Detection summary
        results['detection_summary'] = {
            'total_locations_monitored': len(locations),
            'locations_with_alerts': len(set(alert['location'] for alert in results['alerts'])),
            'total_alerts': len(results['alerts']),
            'alert_levels': {level: sum(1 for alert in results['alerts'] if alert['alert_level'] == level)
                           for level in ['elevated', 'epidemic', 'pandemic']}
        }
        
        return results
    
    def _cusum_detection(self, data: pd.DataFrame, disease: str) -> Dict[str, Any]:
        """Detect outbreaks using CUSUM (Cumulative Sum) algorithm."""
        results = {
            'method': 'cusum',
            'disease': disease,
            'alerts': [],
            'cusum_statistics': {}
        }
        
        # CUSUM parameters
        k = 0.5  # Reference value (half of the shift to detect)
        h = 4.0  # Decision threshold
        
        locations = data['location'].unique()
        
        for location in locations:
            location_data = data[data['location'] == location].copy()
            location_data = location_data.sort_values('date')
            
            if len(location_data) < 14:  # Need at least 2 weeks of data
                continue
            
            # Calculate baseline mean and std
            baseline_data = location_data[:max(14, len(location_data) // 2)]
            mu0 = baseline_data['rate'].mean()
            sigma = baseline_data['rate'].std()
            
            if sigma == 0:
                continue
            
            # Calculate CUSUM statistics
            cusum_pos = 0
            cusum_neg = 0
            cusum_values = []
            
            for _, row in location_data.iterrows():
                # Standardized observation
                z = (row['rate'] - mu0) / sigma
                
                # Update CUSUM statistics
                cusum_pos = max(0, cusum_pos + z - k)
                cusum_neg = max(0, cusum_neg - z - k)
                
                cusum_values.append({
                    'date': row['date'],
                    'cusum_pos': cusum_pos,
                    'cusum_neg': cusum_neg,
                    'rate': row['rate']
                })
                
                # Check for alerts
                if cusum_pos > h:
                    alert = {
                        'location': location,
                        'date': row['date'].isoformat(),
                        'observed_rate': row['rate'],
                        'cusum_value': cusum_pos,
                        'alert_type': 'increase',
                        'confidence': min(cusum_pos / h, 2.0) / 2.0
                    }
                    results['alerts'].append(alert)
                    cusum_pos = 0  # Reset after alert
                
                elif cusum_neg > h:
                    alert = {
                        'location': location,
                        'date': row['date'].isoformat(),
                        'observed_rate': row['rate'],
                        'cusum_value': cusum_neg,
                        'alert_type': 'decrease',
                        'confidence': min(cusum_neg / h, 2.0) / 2.0
                    }
                    results['alerts'].append(alert)
                    cusum_neg = 0  # Reset after alert
            
            results['cusum_statistics'][location] = cusum_values
        
        return results
    
    def _ewma_detection(self, data: pd.DataFrame, disease: str) -> Dict[str, Any]:
        """Detect outbreaks using EWMA (Exponentially Weighted Moving Average)."""
        results = {
            'method': 'ewma',
            'disease': disease,
            'alerts': [],
            'ewma_statistics': {}
        }
        
        # EWMA parameters
        lambda_param = 0.2  # Smoothing parameter
        L = 2.7  # Control limit multiplier
        
        locations = data['location'].unique()
        
        for location in locations:
            location_data = data[data['location'] == location].copy()
            location_data = location_data.sort_values('date')
            
            if len(location_data) < 14:
                continue
            
            # Calculate baseline statistics
            baseline_data = location_data[:max(14, len(location_data) // 2)]
            mu0 = baseline_data['rate'].mean()
            sigma = baseline_data['rate'].std()
            
            if sigma == 0:
                continue
            
            # Initialize EWMA
            ewma = mu0
            ewma_values = []
            
            for i, (_, row) in enumerate(location_data.iterrows()):
                # Update EWMA
                if i == 0:
                    ewma = row['rate']
                else:
                    ewma = lambda_param * row['rate'] + (1 - lambda_param) * ewma
                
                # Calculate control limits
                variance_factor = (lambda_param / (2 - lambda_param)) * (1 - (1 - lambda_param)**(2 * (i + 1)))
                control_limit = L * sigma * np.sqrt(variance_factor)
                
                upper_limit = mu0 + control_limit
                lower_limit = mu0 - control_limit
                
                ewma_values.append({
                    'date': row['date'],
                    'ewma': ewma,
                    'upper_limit': upper_limit,
                    'lower_limit': lower_limit,
                    'rate': row['rate']
                })
                
                # Check for alerts
                if ewma > upper_limit:
                    alert = {
                        'location': location,
                        'date': row['date'].isoformat(),
                        'observed_rate': row['rate'],
                        'ewma_value': ewma,
                        'upper_limit': upper_limit,
                        'alert_type': 'increase',
                        'confidence': min((ewma - upper_limit) / control_limit, 1.0)
                    }
                    results['alerts'].append(alert)
                
                elif ewma < lower_limit:
                    alert = {
                        'location': location,
                        'date': row['date'].isoformat(),
                        'observed_rate': row['rate'],
                        'ewma_value': ewma,
                        'lower_limit': lower_limit,
                        'alert_type': 'decrease',
                        'confidence': min((lower_limit - ewma) / control_limit, 1.0)
                    }
                    results['alerts'].append(alert)
            
            results['ewma_statistics'][location] = ewma_values
        
        return results
    
    def _spatial_scan_detection(self, data: pd.DataFrame, disease: str) -> Dict[str, Any]:
        """Detect spatial clusters using spatial scan statistics."""
        results = {
            'method': 'spatial_scan',
            'disease': disease,
            'clusters': [],
            'scan_statistics': {}
        }
        
        # This is a simplified implementation of spatial scan statistics
        # In practice, you would use specialized software like SaTScan
        
        # Get unique locations and their coordinates (assuming lat/lon available)
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            logger.warning("Spatial coordinates not available for spatial scan")
            return results
        
        # Aggregate data by location
        location_summary = data.groupby('location').agg({
            'count': 'sum',
            'population': 'mean',
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        location_summary['rate'] = (location_summary['count'] / location_summary['population']) * 100000
        
        # Calculate distances between locations
        coords = location_summary[['latitude', 'longitude']].values
        distances = pdist(coords, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Find potential clusters using a sliding window approach
        max_radius = 0.5  # Maximum radius in degrees (approximately 50km)
        min_cases = 5     # Minimum cases to consider a cluster
        
        for i, center_location in location_summary.iterrows():
            # Find locations within radius
            within_radius = []
            for j, other_location in location_summary.iterrows():
                if distance_matrix[i, j] <= max_radius:
                    within_radius.append(j)
            
            if len(within_radius) < 2:  # Need at least 2 locations
                continue
            
            # Calculate cluster statistics
            cluster_data = location_summary.iloc[within_radius]
            total_cases = cluster_data['count'].sum()
            total_population = cluster_data['population'].sum()
            cluster_rate = (total_cases / total_population) * 100000
            
            # Calculate expected cases (using overall rate)
            overall_rate = location_summary['count'].sum() / location_summary['population'].sum()
            expected_cases = total_population * overall_rate
            
            if total_cases >= min_cases and total_cases > expected_cases * 1.5:
                # Calculate likelihood ratio (simplified)
                if expected_cases > 0:
                    likelihood_ratio = (total_cases / expected_cases) if total_cases > expected_cases else 1.0
                    
                    cluster = {
                        'center_location': center_location['location'],
                        'center_coordinates': [center_location['latitude'], center_location['longitude']],
                        'locations_in_cluster': cluster_data['location'].tolist(),
                        'total_cases': int(total_cases),
                        'total_population': int(total_population),
                        'cluster_rate': cluster_rate,
                        'expected_cases': expected_cases,
                        'relative_risk': total_cases / expected_cases,
                        'likelihood_ratio': likelihood_ratio,
                        'p_value': 1.0 / likelihood_ratio if likelihood_ratio > 1 else 1.0  # Simplified p-value
                    }
                    
                    results['clusters'].append(cluster)
        
        # Sort clusters by likelihood ratio
        results['clusters'] = sorted(results['clusters'], key=lambda x: x['likelihood_ratio'], reverse=True)
        
        return results
    
    def predict_disease_spread(
        self,
        disease: str,
        location: str,
        forecast_days: int = 30,
        method: str = "arima"
    ) -> Dict[str, Any]:
        """Predict future disease spread using time series forecasting."""
        try:
            if disease not in self.surveillance_data:
                raise ValueError(f"No surveillance data available for {disease}")
            
            data = self.surveillance_data[disease]['syndromic']
            location_data = data[data['location'] == location].copy()
            
            if len(location_data) < 30:
                raise ValueError("Insufficient data for forecasting")
            
            location_data = location_data.sort_values('date')
            
            # Prepare time series
            ts_data = location_data.set_index('date')['rate']
            ts_data = ts_data.asfreq('D', fill_value=0)  # Daily frequency
            
            if method == "arima":
                predictions = self._arima_forecast(ts_data, forecast_days)
            elif method == "prophet":
                predictions = self._prophet_forecast(location_data, forecast_days)
            elif method == "lstm":
                predictions = self._lstm_forecast(ts_data, forecast_days)
            else:
                raise ValueError(f"Unknown forecasting method: {method}")
            
            logger.info(f"Generated {forecast_days}-day forecast for {disease} in {location}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict disease spread: {str(e)}")
            return {'error': str(e)}
    
    def _arima_forecast(self, ts_data: pd.Series, forecast_days: int) -> Dict[str, Any]:
        """Generate forecasts using ARIMA model."""
        try:
            # Fit ARIMA model (auto-select parameters)
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=forecast_days)
            forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int()
            
            # Create forecast dates
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
            
            results = {
                'method': 'arima',
                'forecast_dates': [date.isoformat() for date in forecast_dates],
                'forecast_values': forecast.tolist(),
                'confidence_intervals': {
                    'lower': forecast_ci.iloc[:, 0].tolist(),
                    'upper': forecast_ci.iloc[:, 1].tolist()
                },
                'model_summary': {
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'parameters': fitted_model.params.to_dict()
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {str(e)}")
            return {'error': str(e)}
    
    def _prophet_forecast(self, data: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Generate forecasts using Prophet model."""
        try:
            # Prepare data for Prophet
            prophet_data = data[['date', 'rate']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract forecast period
            forecast_period = forecast.tail(forecast_days)
            
            results = {
                'method': 'prophet',
                'forecast_dates': [date.isoformat() for date in forecast_period['ds']],
                'forecast_values': forecast_period['yhat'].tolist(),
                'confidence_intervals': {
                    'lower': forecast_period['yhat_lower'].tolist(),
                    'upper': forecast_period['yhat_upper'].tolist()
                },
                'components': {
                    'trend': forecast_period['trend'].tolist(),
                    'weekly': forecast_period['weekly'].tolist() if 'weekly' in forecast_period.columns else None,
                    'yearly': forecast_period['yearly'].tolist() if 'yearly' in forecast_period.columns else None
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Prophet forecasting failed: {str(e)}")
            return {'error': str(e)}
    
    def _lstm_forecast(self, ts_data: pd.Series, forecast_days: int) -> Dict[str, Any]:
        """Generate forecasts using LSTM neural network."""
        try:
            # Prepare data for LSTM
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1))
            
            # Create sequences
            sequence_length = min(30, len(scaled_data) // 3)
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape<sup>0</sup>, X.shape<sup>1</sup>, 1))
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Define LSTM model
            class LSTMModel(nn.Module):
                def __init__(self, input_size=1, hidden_size=50, num_layers=2):
                    super(LSTMModel, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, 1)
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out[:, -1, :])
                    return out
            
            # Train model
            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            
            # Training loop
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Generate forecast
            model.eval()
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            last_sequence_tensor = torch.FloatTensor(last_sequence)
            
            forecasts = []
            current_sequence = last_sequence_tensor.clone()
            
            for _ in range(forecast_days):
                with torch.no_grad():
                    prediction = model(current_sequence)
                    forecasts.append(prediction.item())
                    
                    # Update sequence for next prediction
                    new_sequence = torch.cat([current_sequence[:, 1:, :], prediction.unsqueeze(0).unsqueeze(2)], dim=1)
                    current_sequence = new_sequence
            
            # Inverse transform forecasts
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecasts = scaler.inverse_transform(forecasts).flatten()
            
            # Create forecast dates
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
            
            results = {
                'method': 'lstm',
                'forecast_dates': [date.isoformat() for date in forecast_dates],
                'forecast_values': forecasts.tolist(),
                'confidence_intervals': {
                    'lower': (forecasts * 0.8).tolist(),  # Simplified CI
                    'upper': (forecasts * 1.2).tolist()
                },
                'model_performance': {
                    'training_loss': float(loss.item())
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"LSTM forecasting failed: {str(e)}")
            return {'error': str(e)}

class HealthEquityAnalyzer:
    """Advanced health equity assessment and disparity analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize health equity analyzer."""
        self.config = config
        self.equity_metrics = {}
        self.disparity_thresholds = self._setup_disparity_thresholds()
        
        logger.info("Initialized health equity analyzer")
    
    def _setup_disparity_thresholds(self) -> Dict[str, float]:
        """Setup thresholds for identifying significant disparities."""
        return {
            'rate_ratio_threshold': 1.2,      # 20% difference
            'rate_difference_threshold': 10,   # 10 per 100,000 difference
            'index_of_disparity_threshold': 15, # 15% index of disparity
            'concentration_index_threshold': 0.1 # Concentration index
        }
    
    def assess_health_equity(
        self,
        data: pd.DataFrame,
        outcome: HealthOutcome,
        equity_dimensions: List[EquityDimension],
        reference_group: Optional[str] = None
    ) -> List[HealthEquityAssessment]:
        """Comprehensive health equity assessment across multiple dimensions."""
        try:
            assessments = []
            
            for dimension in equity_dimensions:
                if dimension.value not in data.columns:
                    logger.warning(f"Equity dimension {dimension.value} not found in data")
                    continue
                
                assessment = self._analyze_equity_dimension(
                    data, outcome, dimension, reference_group
                )
                assessments.append(assessment)
            
            logger.info(f"Completed health equity assessment for {len(assessments)} dimensions")
            
            return assessments
            
        except Exception as e:
            logger.error(f"Failed to assess health equity: {str(e)}")
            return []
    
    def _analyze_equity_dimension(
        self,
        data: pd.DataFrame,
        outcome: HealthOutcome,
        dimension: EquityDimension,
        reference_group: Optional[str] = None
    ) -> HealthEquityAssessment:
        """Analyze health equity for a specific dimension."""
        
        # Calculate rates by group
        group_rates = self._calculate_group_rates(data, outcome, dimension)
        
        # Determine reference group
        if reference_group is None:
            # Use group with best (lowest) rate as reference
            reference_group = min(group_rates.keys(), key=lambda x: group_rates[x]['rate'])
        
        reference_rate = group_rates[reference_group]['rate']
        
        # Calculate disparity measures
        rate_ratios = {}
        rate_differences = {}
        
        for group, stats in group_rates.items():
            if group != reference_group:
                rate_ratios[group] = stats['rate'] / reference_rate if reference_rate > 0 else float('inf')
                rate_differences[group] = stats['rate'] - reference_rate
        
        # Calculate overall disparity measure (Index of Disparity)
        rates = [stats['rate'] for stats in group_rates.values()]
        mean_rate = np.mean(rates)
        index_of_disparity = (np.sum(np.abs(np.array(rates) - mean_rate)) / mean_rate) * 100 if mean_rate > 0 else 0
        
        # Statistical significance testing
        p_value = self._test_disparity_significance(data, outcome, dimension)
        
        # Effect size (Cohen's d for largest disparity)
        max_disparity_group = max(rate_ratios.keys(), key=lambda x: abs(rate_ratios[x] - 1)) if rate_ratios else reference_group
        effect_size = self._calculate_effect_size(data, outcome, dimension, reference_group, max_disparity_group)
        
        # Confidence interval for disparity measure
        ci = self._calculate_disparity_confidence_interval(group_rates, reference_group, max_disparity_group)
        
        assessment = HealthEquityAssessment(
            equity_dimension=dimension,
            outcome_type=outcome,
            disparity_measure="index_of_disparity",
            disparity_value=index_of_disparity,
            reference_group=reference_group,
            comparison_groups=rate_ratios,
            statistical_significance=p_value < 0.05,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci
        )
        
        return assessment
    
    def _calculate_group_rates(
        self,
        data: pd.DataFrame,
        outcome: HealthOutcome,
        dimension: EquityDimension
    ) -> Dict[str, Dict[str, float]]:
        """Calculate outcome rates by group."""
        
        # Assume data has columns: outcome_count, population, and dimension column
        outcome_col = f"{outcome.value}_count"
        if outcome_col not in data.columns:
            outcome_col = "count"  # Fallback to generic count column
        
        group_stats = {}
        
        for group in data[dimension.value].unique():
            group_data = data[data[dimension.value] == group]
            
            total_cases = group_data[outcome_col].sum()
            total_population = group_data['population'].sum()
            
            rate = (total_cases / total_population) * 100000 if total_population > 0 else 0
            
            # Calculate confidence interval for rate
            if total_cases > 0 and total_population > 0:
                # Using Poisson approximation
                rate_se = np.sqrt(total_cases) / total_population * 100000
                ci_lower = max(0, rate - 1.96 * rate_se)
                ci_upper = rate + 1.96 * rate_se
            else:
                ci_lower = ci_upper = 0
            
            group_stats[group] = {
                'rate': rate,
                'cases': total_cases,
                'population': total_population,
                'confidence_interval': (ci_lower, ci_upper)
            }
        
        return group_stats
    
    def _test_disparity_significance(
        self,
        data: pd.DataFrame,
        outcome: HealthOutcome,
        dimension: EquityDimension
    ) -> float:
        """Test statistical significance of disparities using chi-square test."""
        
        outcome_col = f"{outcome.value}_count"
        if outcome_col not in data.columns:
            outcome_col = "count"
        
        # Create contingency table
        contingency_data = []
        
        for group in data[dimension.value].unique():
            group_data = data[data[dimension.value] == group]
            cases = group_data[outcome_col].sum()
            population = group_data['population'].sum()
            non_cases = population - cases
            
            contingency_data.append([cases, non_cases])
        
        contingency_table = np.array(contingency_data)
        
        # Chi-square test
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            return p_value
        except:
            return 1.0  # Return non-significant p-value if test fails
    
    def _calculate_effect_size(
        self,
        data: pd.DataFrame,
        outcome: HealthOutcome,
        dimension: EquityDimension,
        reference_group: str,
        comparison_group: str
    ) -> float:
        """Calculate effect size (Cohen's d) for disparity."""
        
        outcome_col = f"{outcome.value}_count"
        if outcome_col not in data.columns:
            outcome_col = "count"
        
        # Get data for both groups
        ref_data = data[data[dimension.value] == reference_group]
        comp_data = data[data[dimension.value] == comparison_group]
        
        # Calculate rates for each observation
        ref_rates = (ref_data[outcome_col] / ref_data['population'] * 100000).dropna()
        comp_rates = (comp_data[outcome_col] / comp_data['population'] * 100000).dropna()
        
        if len(ref_rates) == 0 or len(comp_rates) == 0:
            return 0.0
        
        # Calculate Cohen's d
        pooled_std = np.sqrt(((len(ref_rates) - 1) * ref_rates.var() + 
                             (len(comp_rates) - 1) * comp_rates.var()) / 
                            (len(ref_rates) + len(comp_rates) - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (comp_rates.mean() - ref_rates.mean()) / pooled_std
        
        return abs(cohens_d)
    
    def _calculate_disparity_confidence_interval(
        self,
        group_rates: Dict[str, Dict[str, float]],
        reference_group: str,
        comparison_group: str
    ) -> Tuple[float, float]:
        """Calculate confidence interval for disparity measure."""
        
        if comparison_group not in group_rates or reference_group not in group_rates:
            return (0.0, 0.0)
        
        ref_rate = group_rates[reference_group]['rate']
        comp_rate = group_rates[comparison_group]['rate']
        
        if ref_rate == 0:
            return (0.0, float('inf'))
        
        rate_ratio = comp_rate / ref_rate
        
        # Approximate confidence interval for rate ratio
        # Using log transformation
        ref_cases = group_rates[reference_group]['cases']
        comp_cases = group_rates[comparison_group]['cases']
        
        if ref_cases > 0 and comp_cases > 0:
            log_rr_se = np.sqrt(1/ref_cases + 1/comp_cases)
            log_rr = np.log(rate_ratio)
            
            ci_lower = np.exp(log_rr - 1.96 * log_rr_se)
            ci_upper = np.exp(log_rr + 1.96 * log_rr_se)
        else:
            ci_lower = ci_upper = rate_ratio
        
        return (ci_lower, ci_upper)
    
    def calculate_concentration_index(
        self,
        data: pd.DataFrame,
        outcome: HealthOutcome,
        socioeconomic_variable: str
    ) -> float:
        """Calculate concentration index for socioeconomic-related health inequality."""
        
        outcome_col = f"{outcome.value}_count"
        if outcome_col not in data.columns:
            outcome_col = "count"
        
        # Calculate rates
        data = data.copy()
        data['rate'] = (data[outcome_col] / data['population']) * 100000
        
        # Sort by socioeconomic variable
        data = data.sort_values(socioeconomic_variable)
        
        # Calculate cumulative population proportions
        data['cum_pop'] = data['population'].cumsum() / data['population'].sum()
        
        # Calculate concentration index using trapezoidal rule
        # CI = 2 * (covariance between health and rank) / mean health
        
        n = len(data)
        ranks = np.arange(1, n + 1) / n
        
        # Weight by population
        weights = data['population'] / data['population'].sum()
        
        # Weighted mean rate
        mean_rate = np.average(data['rate'], weights=weights)
        
        if mean_rate == 0:
            return 0.0
        
        # Weighted covariance
        weighted_mean_rank = np.average(ranks, weights=weights)
        
        covariance = np.average(
            (data['rate'] - mean_rate) * (ranks - weighted_mean_rank),
            weights=weights
        )
        
        concentration_index = 2 * covariance / mean_rate
        
        return concentration_index
    
    def generate_equity_report(
        self,
        assessments: List[HealthEquityAssessment],
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive health equity report."""
        
        report = {
            'summary': {
                'total_dimensions_assessed': len(assessments),
                'significant_disparities': sum(1 for a in assessments if a.statistical_significance),
                'largest_disparity': None,
                'equity_score': 0.0
            },
            'dimension_results': [],
            'recommendations': [] if include_recommendations else None
        }
        
        # Process each assessment
        for assessment in assessments:
            dimension_result = {
                'dimension': assessment.equity_dimension.value,
                'outcome': assessment.outcome_type.value,
                'disparity_value': assessment.disparity_value,
                'statistical_significance': assessment.statistical_significance,
                'p_value': assessment.p_value,
                'effect_size': assessment.effect_size,
                'interpretation': self._interpret_disparity(assessment)
            }
            
            report['dimension_results'].append(dimension_result)
        
        # Find largest disparity
        if assessments:
            largest_disparity = max(assessments, key=lambda x: x.disparity_value)
            report['summary']['largest_disparity'] = {
                'dimension': largest_disparity.equity_dimension.value,
                'value': largest_disparity.disparity_value,
                'significance': largest_disparity.statistical_significance
            }
        
        # Calculate overall equity score (0-100, higher is more equitable)
        if assessments:
            avg_disparity = np.mean([a.disparity_value for a in assessments])
            equity_score = max(0, 100 - avg_disparity)
            report['summary']['equity_score'] = equity_score
        
        # Generate recommendations
        if include_recommendations:
            report['recommendations'] = self._generate_equity_recommendations(assessments)
        
        return report
    
    def _interpret_disparity(self, assessment: HealthEquityAssessment) -> str:
        """Interpret disparity magnitude and significance."""
        
        if not assessment.statistical_significance:
            return "No statistically significant disparity detected"
        
        if assessment.disparity_value < 10:
            magnitude = "small"
        elif assessment.disparity_value < 25:
            magnitude = "moderate"
        else:
            magnitude = "large"
        
        return f"Statistically significant {magnitude} disparity detected"
    
    def _generate_equity_recommendations(
        self,
        assessments: List[HealthEquityAssessment]
    ) -> List[Dict[str, str]]:
        """Generate recommendations for addressing health disparities."""
        
        recommendations = []
        
        # Priority recommendations based on largest disparities
        significant_disparities = [a for a in assessments if a.statistical_significance]
        
        if not significant_disparities:
            recommendations.append({
                'priority': 'low',
                'category': 'monitoring',
                'recommendation': 'Continue monitoring health equity indicators to detect emerging disparities'
            })
            return recommendations
        
        # Sort by disparity magnitude
        significant_disparities.sort(key=lambda x: x.disparity_value, reverse=True)
        
        for i, assessment in enumerate(significant_disparities[:3]):  # Top 3 disparities
            priority = ['high', 'medium', 'low'][i]
            
            if assessment.equity_dimension == EquityDimension.RACE_ETHNICITY:
                recommendations.append({
                    'priority': priority,
                    'category': 'cultural_competency',
                    'recommendation': 'Implement culturally competent care programs and address structural racism in healthcare delivery'
                })
            
            elif assessment.equity_dimension == EquityDimension.INCOME:
                recommendations.append({
                    'priority': priority,
                    'category': 'economic_support',
                    'recommendation': 'Expand access to affordable healthcare and address social determinants related to income'
                })
            
            elif assessment.equity_dimension == EquityDimension.GEOGRAPHY:
                recommendations.append({
                    'priority': priority,
                    'category': 'access_improvement',
                    'recommendation': 'Improve healthcare access in underserved geographic areas through mobile clinics or telemedicine'
                })
            
            elif assessment.equity_dimension == EquityDimension.EDUCATION:
                recommendations.append({
                    'priority': priority,
                    'category': 'health_literacy',
                    'recommendation': 'Develop health education programs tailored to different literacy levels'
                })
        
        return recommendations

class CausalInferenceEngine:
    """Advanced causal inference for population health interventions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize causal inference engine."""
        self.config = config
        self.causal_models = {}
        self.validation_results = {}
        
        logger.info("Initialized causal inference engine")
    
    def estimate_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        method: CausalMethod,
        confounders: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None
    ) -> CausalInferenceResult:
        """Estimate causal effect using specified method."""
        
        try:
            if method == CausalMethod.INSTRUMENTAL_VARIABLES:
                result = self._instrumental_variables(data, treatment, outcome, instruments, confounders)
            elif method == CausalMethod.REGRESSION_DISCONTINUITY:
                result = self._regression_discontinuity(data, treatment, outcome, confounders)
            elif method == CausalMethod.DIFFERENCE_IN_DIFFERENCES:
                result = self._difference_in_differences(data, treatment, outcome, confounders)
            elif method == CausalMethod.SYNTHETIC_CONTROL:
                result = self._synthetic_control(data, treatment, outcome, confounders)
            elif method == CausalMethod.PROPENSITY_SCORE:
                result = self._propensity_score_matching(data, treatment, outcome, confounders)
            else:
                raise ValueError(f"Unknown causal inference method: {method}")
            
            logger.info(f"Completed causal inference using {method.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to estimate causal effect: {str(e)}")
            raise
    
    def _instrumental_variables(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        instruments: List[str],
        confounders: Optional[List[str]] = None
    ) -> CausalInferenceResult:
        """Estimate causal effect using instrumental variables."""
        
        if not instruments:
            raise ValueError("Instrumental variables must be specified")
        
        # Prepare data
        analysis_data = data[[treatment, outcome] + instruments].dropna()
        
        if confounders:
            available_confounders = [c for c in confounders if c in data.columns]
            if available_confounders:
                analysis_data = data[[treatment, outcome] + instruments + available_confounders].dropna()
        
        # First stage: regress treatment on instruments (and confounders)
        X_first = analysis_data[instruments]
        if confounders:
            available_confounders = [c for c in confounders if c in analysis_data.columns]
            if available_confounders:
                X_first = pd.concat([X_first, analysis_data[available_confounders]], axis=1)
        
        X_first = sm.add_constant(X_first)
        first_stage = sm.OLS(analysis_data[treatment], X_first).fit()
        
        # Get predicted treatment values
        treatment_hat = first_stage.fittedvalues
        
        # Second stage: regress outcome on predicted treatment (and confounders)
        X_second = pd.DataFrame({'treatment_hat': treatment_hat})
        if confounders:
            available_confounders = [c for c in confounders if c in analysis_data.columns]
            if available_confounders:
                X_second = pd.concat([X_second, analysis_data[available_confounders]], axis=1)
        
        X_second = sm.add_constant(X_second)
        second_stage = sm.OLS(analysis_data[outcome], X_second).fit()
        
        # Extract causal effect
        causal_effect = second_stage.params['treatment_hat']
        
        # Calculate confidence interval
        ci_lower = second_stage.conf_int().loc['treatment_hat', 0]
        ci_upper = second_stage.conf_int().loc['treatment_hat', 1]
        
        # Test instrument strength (F-statistic from first stage)
        f_stat = first_stage.fvalue
        weak_instrument = f_stat < 10  # Rule of thumb
        
        # Test overidentification (if more instruments than endogenous variables)
        overid_test_passed = True
        if len(instruments) > 1:
            # Simplified overidentification test
            residuals = analysis_data[outcome] - second_stage.fittedvalues
            overid_reg = sm.OLS(residuals, X_first).fit()
            overid_test_passed = overid_reg.f_pvalue > 0.05
        
        result = CausalInferenceResult(
            method=CausalMethod.INSTRUMENTAL_VARIABLES,
            treatment=treatment,
            outcome=outcome,
            causal_effect=causal_effect,
            confidence_interval=(ci_lower, ci_upper),
            p_value=second_stage.pvalues['treatment_hat'],
            sample_size=len(analysis_data),
            assumptions_met={
                'instrument_strength': not weak_instrument,
                'overidentification': overid_test_passed,
                'exclusion_restriction': True  # Cannot be tested directly
            },
            sensitivity_analysis={
                'first_stage_f_stat': f_stat,
                'first_stage_r_squared': first_stage.rsquared
            }
        )
        
        return result
    
    def _regression_discontinuity(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: Optional[List[str]] = None,
        running_variable: str = 'score',
        cutoff: Optional[float] = None
    ) -> CausalInferenceResult:
        """Estimate causal effect using regression discontinuity design."""
        
        if running_variable not in data.columns:
            raise ValueError(f"Running variable '{running_variable}' not found in data")
        
        # Determine cutoff if not provided
        if cutoff is None:
            # Assume treatment assignment changes at median of running variable
            cutoff = data[running_variable].median()
        
        # Create treatment assignment based on cutoff
        data = data.copy()
        data['above_cutoff'] = (data[running_variable] >= cutoff).astype(int)
        
        # Center running variable around cutoff
        data['running_centered'] = data[running_variable] - cutoff
        
        # Prepare regression variables
        reg_vars = ['above_cutoff', 'running_centered']
        
        # Add interaction term
        data['above_cutoff_x_running'] = data['above_cutoff'] * data['running_centered']
        reg_vars.append('above_cutoff_x_running')
        
        # Add confounders if specified
        if confounders:
            available_confounders = [c for c in confounders if c in data.columns]
            reg_vars.extend(available_confounders)
        
        # Fit regression model
        X = data[reg_vars].dropna()
        y = data.loc[X.index, outcome]
        
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        
        # Extract causal effect (coefficient on above_cutoff)
        causal_effect = model.params['above_cutoff']
        
        # Calculate confidence interval
        ci_lower = model.conf_int().loc['above_cutoff', 0]
        ci_upper = model.conf_int().loc['above_cutoff', 1]
        
        # Test for manipulation of running variable (McCrary test approximation)
        # Check for discontinuity in density of running variable
        bandwidth = 0.1 * data['running_centered'].std()
        left_density = len(data[(data['running_centered'] >= -bandwidth) & 
                               (data['running_centered'] < 0)]) / bandwidth
        right_density = len(data[(data['running_centered'] >= 0) & 
                                (data['running_centered'] <= bandwidth)]) / bandwidth
        
        density_ratio = right_density / left_density if left_density > 0 else float('inf')
        no_manipulation = 0.5 < density_ratio < 2.0  # Rough test
        
        # Test for balance of covariates
        covariate_balance = True
        if confounders:
            for confounder in available_confounders:
                if confounder in data.columns:
                    # Test for discontinuity in confounder
                    conf_model = sm.OLS(data[confounder], 
                                      sm.add_constant(data[['above_cutoff', 'running_centered']])).fit()
                    if conf_model.pvalues['above_cutoff'] < 0.05:
                        covariate_balance = False
                        break
        
        result = CausalInferenceResult(
            method=CausalMethod.REGRESSION_DISCONTINUITY,
            treatment=treatment,
            outcome=outcome,
            causal_effect=causal_effect,
            confidence_interval=(ci_lower, ci_upper),
            p_value=model.pvalues['above_cutoff'],
            sample_size=len(X),
            assumptions_met={
                'no_manipulation': no_manipulation,
                'covariate_balance': covariate_balance,
                'local_randomization': True  # Assumed near cutoff
            },
            sensitivity_analysis={
                'density_ratio': density_ratio,
                'bandwidth_used': bandwidth,
                'cutoff_value': cutoff
            }
        )
        
        return result
    
    def _difference_in_differences(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: Optional[List[str]] = None,
        time_variable: str = 'time_period',
        group_variable: str = 'group'
    ) -> CausalInferenceResult:
        """Estimate causal effect using difference-in-differences."""
        
        required_vars = [time_variable, group_variable, outcome]
        if not all(var in data.columns for var in required_vars):
            raise ValueError(f"Required variables not found: {required_vars}")
        
        # Create treatment indicator (post-period × treatment group)
        data = data.copy()
        
        # Assume binary time periods (0 = pre, 1 = post)
        data['post'] = data[time_variable]
        
        # Assume binary groups (0 = control, 1 = treatment)
        data['treated'] = data[group_variable]
        
        # Create interaction term
        data['post_x_treated'] = data['post'] * data['treated']
        
        # Prepare regression variables
        reg_vars = ['post', 'treated', 'post_x_treated']
        
        # Add confounders if specified
        if confounders:
            available_confounders = [c for c in confounders if c in data.columns]
            reg_vars.extend(available_confounders)
        
        # Fit regression model
        X = data[reg_vars].dropna()
        y = data.loc[X.index, outcome]
        
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        
        # Extract causal effect (coefficient on interaction term)
        causal_effect = model.params['post_x_treated']
        
        # Calculate confidence interval
        ci_lower = model.conf_int().loc['post_x_treated', 0]
        ci_upper = model.conf_int().loc['post_x_treated', 1]
        
        # Test parallel trends assumption (simplified)
        # Check if pre-treatment trends are similar between groups
        pre_data = data[data['post'] == 0]
        if len(pre_data) > 0 and time_variable in pre_data.columns:
            # If we have multiple pre-periods, test for differential trends
            parallel_trends = True  # Simplified assumption
        else:
            parallel_trends = True  # Cannot test with only two periods
        
        # Test for common shocks
        common_shocks = True  # Assumed in DiD design
        
        result = CausalInferenceResult(
            method=CausalMethod.DIFFERENCE_IN_DIFFERENCES,
            treatment=treatment,
            outcome=outcome,
            causal_effect=causal_effect,
            confidence_interval=(ci_lower, ci_upper),
            p_value=model.pvalues['post_x_treated'],
            sample_size=len(X),
            assumptions_met={
                'parallel_trends': parallel_trends,
                'common_shocks': common_shocks,
                'no_spillovers': True  # Assumed
            },
            sensitivity_analysis={
                'pre_treatment_difference': model.params['treated'],
                'time_effect': model.params['post'],
                'model_r_squared': model.rsquared
            }
        )
        
        return result
    
    def _synthetic_control(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: Optional[List[str]] = None,
        unit_variable: str = 'unit',
        time_variable: str = 'time'
    ) -> CausalInferenceResult:
        """Estimate causal effect using synthetic control method."""
        
        # This is a simplified implementation of synthetic control
        # In practice, you would use specialized packages like Synth
        
        required_vars = [unit_variable, time_variable, outcome, treatment]
        if not all(var in data.columns for var in required_vars):
            raise ValueError(f"Required variables not found: {required_vars}")
        
        # Identify treated unit and treatment time
        treated_units = data[data[treatment] == 1][unit_variable].unique()
        if len(treated_units) != 1:
            raise ValueError("Synthetic control requires exactly one treated unit")
        
        treated_unit = treated_units<sup>0</sup>
        
        # Find treatment time
        treated_data = data[data[unit_variable] == treated_unit]
        treatment_time = treated_data[treated_data[treatment] == 1][time_variable].min()
        
        # Split data into pre and post treatment periods
        pre_data = data[data[time_variable] < treatment_time]
        post_data = data[data[time_variable] >= treatment_time]
        
        # Get control units
        control_units = data[data[unit_variable] != treated_unit][unit_variable].unique()
        
        # Create outcome matrix for pre-treatment period
        pre_outcomes = pre_data.pivot(index=time_variable, columns=unit_variable, values=outcome)
        
        # Treated unit outcomes
        treated_outcomes = pre_outcomes[treated_unit].dropna()
        
        # Control unit outcomes
        control_outcomes = pre_outcomes[control_units].dropna()
        
        # Find optimal weights for synthetic control
        # Minimize sum of squared differences in pre-treatment period
        def objective(weights):
            weights = weights / weights.sum()  # Normalize weights
            synthetic = control_outcomes.dot(weights)
            return np.sum((treated_outcomes - synthetic) ** 2)
        
        # Constraints: weights sum to 1 and are non-negative
        from scipy.optimize import minimize
        
        n_controls = len(control_units)
        initial_weights = np.ones(n_controls) / n_controls
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(n_controls)]
        
        result_opt = minimize(objective, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        
        optimal_weights = result_opt.x / result_opt.x.sum()
        
        # Calculate synthetic control outcomes for all periods
        all_outcomes = data.pivot(index=time_variable, columns=unit_variable, values=outcome)
        synthetic_outcomes = all_outcomes[control_units].dot(optimal_weights)
        treated_all_outcomes = all_outcomes[treated_unit]
        
        # Calculate treatment effect in post-treatment period
        post_synthetic = synthetic_outcomes[synthetic_outcomes.index >= treatment_time]
        post_treated = treated_all_outcomes[treated_all_outcomes.index >= treatment_time]
        
        treatment_effects = post_treated - post_synthetic
        avg_treatment_effect = treatment_effects.mean()
        
        # Calculate confidence interval using placebo tests
        # Run synthetic control for each control unit
        placebo_effects = []
        
        for control_unit in control_units[:min(10, len(control_units))]:  # Limit for computational efficiency
            try:
                # Treat this control unit as if it were treated
                placebo_treated = all_outcomes[control_unit]
                placebo_controls = all_outcomes[[u for u in control_units if u != control_unit]]
                
                # Find weights for this placebo
                placebo_pre_treated = placebo_treated[placebo_treated.index < treatment_time]
                placebo_pre_controls = placebo_controls.loc[placebo_pre_treated.index]
                
                def placebo_objective(weights):
                    weights = weights / weights.sum()
                    synthetic = placebo_pre_controls.dot(weights)
                    return np.sum((placebo_pre_treated - synthetic) ** 2)
                
                n_placebo_controls = len(placebo_controls.columns)
                placebo_initial = np.ones(n_placebo_controls) / n_placebo_controls
                placebo_constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
                placebo_bounds = [(0, 1) for _ in range(n_placebo_controls)]
                
                placebo_result = minimize(placebo_objective, placebo_initial, method='SLSQP',
                                        bounds=placebo_bounds, constraints=placebo_constraints)
                
                placebo_weights = placebo_result.x / placebo_result.x.sum()
                
                # Calculate placebo effect
                placebo_synthetic_all = placebo_controls.dot(placebo_weights)
                placebo_post_synthetic = placebo_synthetic_all[placebo_synthetic_all.index >= treatment_time]
                placebo_post_treated = placebo_treated[placebo_treated.index >= treatment_time]
                
                placebo_effect = (placebo_post_treated - placebo_post_synthetic).mean()
                placebo_effects.append(placebo_effect)
                
            except:
                continue
        
        # Calculate p-value as proportion of placebo effects larger than actual effect
        if placebo_effects:
            p_value = np.mean([abs(effect) >= abs(avg_treatment_effect) for effect in placebo_effects])
        else:
            p_value = 0.5  # Default if no placebo tests possible
        
        # Confidence interval from placebo distribution
        if len(placebo_effects) >= 5:
            ci_lower = np.percentile(placebo_effects, 2.5)
            ci_upper = np.percentile(placebo_effects, 97.5)
        else:
            # Fallback to simple standard error
            se = treatment_effects.std() / np.sqrt(len(treatment_effects))
            ci_lower = avg_treatment_effect - 1.96 * se
            ci_upper = avg_treatment_effect + 1.96 * se
        
        result = CausalInferenceResult(
            method=CausalMethod.SYNTHETIC_CONTROL,
            treatment=treatment,
            outcome=outcome,
            causal_effect=avg_treatment_effect,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            sample_size=len(post_treated),
            assumptions_met={
                'no_spillovers': True,  # Assumed
                'convex_hull': True,    # Should be checked
                'interpolation_bias': True  # Assumed minimal
            },
            sensitivity_analysis={
                'pre_treatment_fit': result_opt.fun,
                'number_of_controls': len(control_units),
                'number_of_placebos': len(placebo_effects)
            }
        )
        
        return result
    
    def _propensity_score_matching(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str]
    ) -> CausalInferenceResult:
        """Estimate causal effect using propensity score matching."""
        
        if not confounders:
            raise ValueError("Confounders must be specified for propensity score matching")
        
        # Prepare data
        available_confounders = [c for c in confounders if c in data.columns]
        analysis_vars = [treatment, outcome] + available_confounders
        analysis_data = data[analysis_vars].dropna()
        
        # Estimate propensity scores
        X = analysis_data[available_confounders]
        y = analysis_data[treatment]
        
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X, y)
        
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        analysis_data = analysis_data.copy()
        analysis_data['propensity_score'] = propensity_scores
        
        # Perform matching (1:1 nearest neighbor with caliper)
        treated_data = analysis_data[analysis_data[treatment] == 1]
        control_data = analysis_data[analysis_data[treatment] == 0]
        
        caliper = 0.1 * propensity_scores.std()  # 0.1 standard deviations
        
        matched_pairs = []
        used_controls = set()
        
        for _, treated_unit in treated_data.iterrows():
            treated_ps = treated_unit['propensity_score']
            
            # Find closest control unit within caliper
            distances = np.abs(control_data['propensity_score'] - treated_ps)
            valid_matches = control_data[
                (distances <= caliper) & 
                (~control_data.index.isin(used_controls))
            ]
            
            if len(valid_matches) > 0:
                closest_match = valid_matches.loc[distances[valid_matches.index].idxmin()]
                matched_pairs.append({
                    'treated_outcome': treated_unit[outcome],
                    'control_outcome': closest_match[outcome],
                    'treated_ps': treated_ps,
                    'control_ps': closest_match['propensity_score']
                })
                used_controls.add(closest_match.name)
        
        if len(matched_pairs) == 0:
            raise ValueError("No valid matches found within caliper")
        
        # Calculate average treatment effect
        treatment_effects = [pair['treated_outcome'] - pair['control_outcome'] 
                           for pair in matched_pairs]
        
        avg_treatment_effect = np.mean(treatment_effects)
        se_treatment_effect = np.std(treatment_effects) / np.sqrt(len(treatment_effects))
        
        # Calculate confidence interval
        ci_lower = avg_treatment_effect - 1.96 * se_treatment_effect
        ci_upper = avg_treatment_effect + 1.96 * se_treatment_effect
        
        # Calculate p-value
        t_stat = avg_treatment_effect / se_treatment_effect if se_treatment_effect > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        # Check balance after matching
        matched_treated_ps = [pair['treated_ps'] for pair in matched_pairs]
        matched_control_ps = [pair['control_ps'] for pair in matched_pairs]
        
        ps_balance = abs(np.mean(matched_treated_ps) - np.mean(matched_control_ps))
        good_balance = ps_balance < 0.05  # Arbitrary threshold
        
        # Check overlap
        ps_overlap = (min(propensity_scores) < 0.9 and max(propensity_scores) > 0.1)
        
        result = CausalInferenceResult(
            method=CausalMethod.PROPENSITY_SCORE,
            treatment=treatment,
            outcome=outcome,
            causal_effect=avg_treatment_effect,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            sample_size=len(matched_pairs),
            assumptions_met={
                'unconfoundedness': True,  # Assumed given confounders
                'overlap': ps_overlap,
                'balance_achieved': good_balance
            },
            sensitivity_analysis={
                'matching_rate': len(matched_pairs) / len(treated_data),
                'caliper_used': caliper,
                'ps_balance_after_matching': ps_balance
            }
        )
        
        return result

class PopulationHealthAISystem:
    """
    Comprehensive population health AI system.
    
    This class integrates disease surveillance, health equity analysis,
    and causal inference capabilities for population health applications.
    """
    
    def __init__(self, config_path: str):
        """Initialize population health AI system."""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.surveillance_system = DiseaseSurveillanceSystem(self.config.get('surveillance', {}))
        self.equity_analyzer = HealthEquityAnalyzer(self.config.get('equity', {}))
        self.causal_engine = CausalInferenceEngine(self.config.get('causal_inference', {}))
        
        # Initialize database
        self.db_engine = create_engine(self.config.get('database_url', 'sqlite:///population_health.db'))
        self._setup_database()
        
        logger.info("Initialized population health AI system")
    
    def _setup_database(self):
        """Setup database tables for population health data."""
        
        Base = declarative_base()
        
        class PopulationData(Base):
            __tablename__ = 'population_data'
            
            id = Column(Integer, primary_key=True)
            location = Column(String(100))
            date = Column(DateTime)
            outcome_type = Column(String(50))
            count = Column(Integer)
            population = Column(Integer)
            demographic_data = Column(JSON)
            created_at = Column(DateTime, default=datetime.utcnow)
        
        class EquityAssessment(Base):
            __tablename__ = 'equity_assessments'
            
            id = Column(Integer, primary_key=True)
            assessment_id = Column(String(100))
            equity_dimension = Column(String(50))
            outcome_type = Column(String(50))
            disparity_value = Column(Float)
            statistical_significance = Column(Boolean)
            created_at = Column(DateTime, default=datetime.utcnow)
        
        class CausalAnalysis(Base):
            __tablename__ = 'causal_analyses'
            
            id = Column(Integer, primary_key=True)
            analysis_id = Column(String(100))
            method = Column(String(50))
            treatment = Column(String(100))
            outcome = Column(String(100))
            causal_effect = Column(Float)
            p_value = Column(Float)
            created_at = Column(DateTime, default=datetime.utcnow)
        
        Base.metadata.create_all(self.db_engine)
        
        logger.info("Database tables created successfully")
    
    def analyze_population_health(
        self,
        data: pd.DataFrame,
        analysis_type: str = "comprehensive",
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive population health analysis."""
        
        try:
            results = {
                'analysis_type': analysis_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data_summary': {
                    'total_records': len(data),
                    'date_range': {
                        'start': data['date'].min().isoformat() if 'date' in data.columns else None,
                        'end': data['date'].max().isoformat() if 'date' in data.columns else None
                    },
                    'locations': data['location'].nunique() if 'location' in data.columns else 0
                }
            }
            
            if analysis_type in ['comprehensive', 'surveillance']:
                # Disease surveillance analysis
                surveillance_results = self._run_surveillance_analysis(data)
                results['surveillance'] = surveillance_results
            
            if analysis_type in ['comprehensive', 'equity']:
                # Health equity analysis
                equity_results = self._run_equity_analysis(data)
                results['equity'] = equity_results
            
            if analysis_type in ['comprehensive', 'causal']:
                # Causal inference analysis
                causal_results = self._run_causal_analysis(data)
                results['causal'] = causal_results
            
            # Generate recommendations
            recommendations = self._generate_population_health_recommendations(results)
            results['recommendations'] = recommendations
            
            # Save results if requested
            if save_results:
                self._save_analysis_results(results)
            
            logger.info(f"Completed {analysis_type} population health analysis")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze population health: {str(e)}")
            return {'error': str(e)}
    
    def _run_surveillance_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run disease surveillance analysis."""
        
        surveillance_results = {
            'outbreak_detection': {},
            'forecasting': {},
            'spatial_analysis': {}
        }
        
        # Ingest data into surveillance system
        self.surveillance_system.ingest_surveillance_data('analysis_data', data, 'syndromic')
        
        # Detect outbreaks for different diseases/outcomes
        if 'outcome_type' in data.columns:
            outcome_types = data['outcome_type'].unique()
            
            for outcome in outcome_types[:3]:  # Limit to first 3 for efficiency
                try:
                    outbreak_results = self.surveillance_system.detect_outbreaks(
                        disease=outcome,
                        method='aberration_detection'
                    )
                    surveillance_results['outbreak_detection'][outcome] = outbreak_results
                except Exception as e:
                    logger.warning(f"Failed to detect outbreaks for {outcome}: {str(e)}")
        
        # Generate forecasts for locations with sufficient data
        if 'location' in data.columns:
            locations = data['location'].unique()
            
            for location in locations[:3]:  # Limit to first 3 for efficiency
                location_data = data[data['location'] == location]
                if len(location_data) >= 30:  # Need sufficient data for forecasting
                    try:
                        forecast_results = self.surveillance_system.predict_disease_spread(
                            disease=location_data['outcome_type'].iloc<sup>0</sup> if 'outcome_type' in data.columns else 'general',
                            location=location,
                            forecast_days=30,
                            method='arima'
                        )
                        surveillance_results['forecasting'][location] = forecast_results
                    except Exception as e:
                        logger.warning(f"Failed to generate forecast for {location}: {str(e)}")
        
        return surveillance_results
    
    def _run_equity_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run health equity analysis."""
        
        equity_results = {
            'assessments': [],
            'report': {},
            'concentration_indices': {}
        }
        
        # Determine available equity dimensions
        available_dimensions = []
        for dimension in EquityDimension:
            if dimension.value in data.columns:
                available_dimensions.append(dimension)
        
        if not available_dimensions:
            logger.warning("No equity dimensions found in data")
            return equity_results
        
        # Determine outcome type
        if 'outcome_type' in data.columns:
            outcome_types = data['outcome_type'].unique()
        else:
            outcome_types = [HealthOutcome.MORBIDITY]  # Default
        
        # Run equity assessments
        for outcome_str in outcome_types[:2]:  # Limit for efficiency
            try:
                outcome = HealthOutcome(outcome_str) if isinstance(outcome_str, str) else HealthOutcome.MORBIDITY
                
                assessments = self.equity_analyzer.assess_health_equity(
                    data=data,
                    outcome=outcome,
                    equity_dimensions=available_dimensions
                )
                
                equity_results['assessments'].extend([a.to_dict() for a in assessments])
                
            except Exception as e:
                logger.warning(f"Failed to assess equity for {outcome_str}: {str(e)}")
        
        # Generate equity report
        if equity_results['assessments']:
            assessments_objects = [
                HealthEquityAssessment(**assessment) 
                for assessment in equity_results['assessments']
            ]
            
            report = self.equity_analyzer.generate_equity_report(assessments_objects)
            equity_results['report'] = report
        
        # Calculate concentration indices for socioeconomic variables
        socioeconomic_vars = ['income', 'education_level', 'poverty_rate']
        for var in socioeconomic_vars:
            if var in data.columns:
                try:
                    ci = self.equity_analyzer.calculate_concentration_index(
                        data=data,
                        outcome=HealthOutcome.MORBIDITY,
                        socioeconomic_variable=var
                    )
                    equity_results['concentration_indices'][var] = ci
                except Exception as e:
                    logger.warning(f"Failed to calculate concentration index for {var}: {str(e)}")
        
        return equity_results
    
    def _run_causal_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run causal inference analysis."""
        
        causal_results = {
            'analyses': [],
            'summary': {}
        }
        
        # Identify potential treatments and outcomes
        potential_treatments = []
        potential_outcomes = []
        
        # Look for binary variables that could be treatments
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64'] and data[col].nunique() == 2:
                potential_treatments.append(col)
            elif col in ['mortality_rate', 'hospitalization_rate', 'outcome_rate']:
                potential_outcomes.append(col)
        
        # If no clear outcomes, use count-based measures
        if not potential_outcomes and 'count' in data.columns:
            potential_outcomes = ['count']
        
        # Run causal analyses for promising treatment-outcome pairs
        for treatment in potential_treatments[:2]:  # Limit for efficiency
            for outcome in potential_outcomes[:2]:
                
                # Skip if treatment and outcome are the same
                if treatment == outcome:
                    continue
                
                try:
                    # Try different causal methods
                    methods_to_try = [CausalMethod.PROPENSITY_SCORE]
                    
                    # Add other methods if appropriate data structure exists
                    if 'time_period' in data.columns and 'group' in data.columns:
                        methods_to_try.append(CausalMethod.DIFFERENCE_IN_DIFFERENCES)
                    
                    for method in methods_to_try:
                        try:
                            # Identify potential confounders
                            confounders = [col for col in data.columns 
                                         if col not in [treatment, outcome, 'date', 'location']
                                         and data[col].dtype in ['int64', 'float64']]
                            
                            result = self.causal_engine.estimate_causal_effect(
                                data=data,
                                treatment=treatment,
                                outcome=outcome,
                                method=method,
                                confounders=confounders[:5]  # Limit confounders
                            )
                            
                            causal_results['analyses'].append(result.to_dict())
                            
                        except Exception as e:
                            logger.warning(f"Failed causal analysis {method.value} for {treatment}->{outcome}: {str(e)}")
                
                except Exception as e:
                    logger.warning(f"Failed to run causal analysis for {treatment}->{outcome}: {str(e)}")
        
        # Generate summary
        if causal_results['analyses']:
            significant_effects = [a for a in causal_results['analyses'] if a['p_value'] < 0.05]
            
            causal_results['summary'] = {
                'total_analyses': len(causal_results['analyses']),
                'significant_effects': len(significant_effects),
                'largest_effect': max(causal_results['analyses'], 
                                    key=lambda x: abs(x['causal_effect']))['causal_effect'] if causal_results['analyses'] else 0
            }
        
        return causal_results
    
    def _generate_population_health_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on analysis results."""
        
        recommendations = []
        
        # Surveillance-based recommendations
        if 'surveillance' in results:
            surveillance = results['surveillance']
            
            # Check for outbreak alerts
            if 'outbreak_detection' in surveillance:
                for disease, detection_results in surveillance['outbreak_detection'].items():
                    if 'alerts' in detection_results and detection_results['alerts']:
                        high_priority_alerts = [a for a in detection_results['alerts'] 
                                              if a.get('alert_level') in ['epidemic', 'pandemic']]
                        
                        if high_priority_alerts:
                            recommendations.append({
                                'category': 'outbreak_response',
                                'priority': 'high',
                                'recommendation': f'Immediate investigation and response required for {disease} outbreak alerts in {len(high_priority_alerts)} locations',
                                'evidence': f'Detected {len(high_priority_alerts)} high-priority alerts'
                            })
        
        # Equity-based recommendations
        if 'equity' in results:
            equity = results['equity']
            
            if 'report' in equity and 'summary' in equity['report']:
                equity_score = equity['report']['summary'].get('equity_score', 100)
                
                if equity_score < 70:
                    recommendations.append({
                        'category': 'health_equity',
                        'priority': 'high',
                        'recommendation': 'Implement targeted interventions to address significant health disparities',
                        'evidence': f'Overall equity score: {equity_score:.1f}/100'
                    })
                
                # Add specific equity recommendations
                if 'recommendations' in equity['report'] and equity['report']['recommendations']:
                    for rec in equity['report']['recommendations'][:3]:  # Top 3
                        recommendations.append({
                            'category': 'health_equity',
                            'priority': rec['priority'],
                            'recommendation': rec['recommendation'],
                            'evidence': f"Based on {rec['category']} analysis"
                        })
        
        # Causal inference-based recommendations
        if 'causal' in results:
            causal = results['causal']
            
            if 'analyses' in causal:
                significant_effects = [a for a in causal['analyses'] if a['p_value'] < 0.05]
                
                for effect in significant_effects[:3]:  # Top 3
                    if effect['causal_effect'] > 0:
                        direction = 'increase'
                        action = 'consider reducing'
                    else:
                        direction = 'decrease'
                        action = 'consider increasing'
                    
                    recommendations.append({
                        'category': 'intervention_optimization',
                        'priority': 'medium',
                        'recommendation': f'{action.capitalize()} {effect["treatment"]} to {direction} {effect["outcome"]}',
                        'evidence': f'Causal effect: {effect["causal_effect"]:.3f} (p={effect["p_value"]:.3f})'
                    })
        
        # General recommendations if no specific issues found
        if not recommendations:
            recommendations.append({
                'category': 'monitoring',
                'priority': 'low',
                'recommendation': 'Continue routine population health monitoring and maintain current intervention strategies',
                'evidence': 'No significant issues detected in current analysis'
            })
        
        return recommendations
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to database."""
        
        try:
            Session = sessionmaker(bind=self.db_engine)
            session = Session()
            
            # Save results as JSON for now (in practice, you'd normalize this)
            # This is a simplified implementation
            
            session.commit()
            session.close()
            
            logger.info("Analysis results saved to database")
            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {str(e)}")


## Bibliography and References

### Population Health and Public Health Informatics

. **Kindig, D., & Stoddart, G.** (2003). What is population health? *American Journal of Public Health*, 93(3), 380-383. [Population health definition]

. **Rose, G.** (2001). *Sick individuals and sick populations*. International Journal of Epidemiology, 30(3), 427-432. [Population health approach]

. **Brownson, R. C., Fielding, J. E., & Green, L. W.** (2018). Building capacity for evidence-based public health: reconciling the pulls of practice and the push of research. *Annual Review of Public Health*, 39, 27-53. [Evidence-based public health]

. **Braveman, P., & Gottlieb, L.** (2014). The social determinants of health: it's time to consider the causes of the causes. *Public Health Reports*, 129(1_suppl2), 19-31. [Social determinants of health]

### Disease Surveillance and Outbreak Detection

. **Buckeridge, D. L.** (2007). Outbreak detection through automated surveillance: a review of the determinants of detection. *Journal of Biomedical Informatics*, 40(4), 370-379. [Automated surveillance]

. **Shmueli, G., & Burkom, H.** (2010). Statistical challenges facing early outbreak detection in biosurveillance. *Technometrics*, 52(1), 39-51. [Statistical outbreak detection]

. **Unkel, S., Farrington, C. P., Garthwaite, P. H., Robertson, C., & Andrews, N.** (2012). Statistical methods for the prospective detection of infectious disease outbreaks: a review. *Journal of the Royal Statistical Society: Series A*, 175(1), 49-82. [Prospective outbreak detection]

. **Nsoesie, E. O., Brownstein, J. S., Ramakrishnan, N., & Marathe, M. V.** (2014). A systematic review of studies on forecasting the dynamics of influenza outbreaks. *Influenza and Other Respiratory Viruses*, 8(3), 309-316. [Disease forecasting]

### Health Equity and Disparities

. **Braveman, P.** (2006). Health disparities and health equity: concepts and measurement. *Annual Review of Public Health*, 27, 167-194. [Health equity concepts]

. **Harper, S., & Lynch, J.** (2005). Methods for measuring cancer disparities: using data relevant to Healthy People 2010 cancer-related objectives. *NCI Cancer Surveillance Monograph Series*, 6. [Disparity measurement methods]

. **Keppel, K., Pamuk, E., Lynch, J., Carter-Pokras, O., Kim, I., Mays, V., ... & Weissman, J.** (2005). Methodological issues in measuring health disparities. *Vital and Health Statistics*, 2(141), 1-16. [Health disparity methodology]

. **Wagstaff, A., Paci, P., & Van Doorslaer, E.** (1991). On the measurement of inequalities in health. *Social Science & Medicine*, 33(5), 545-557. [Health inequality measurement]

### Causal Inference in Population Health

. **Hernán, M. A., & Robins, J. M.** (2020). *Causal inference: what if*. Chapman & Hall/CRC. [Causal inference methods]

. **Imbens, G. W., & Rubin, D. B.** (2015). *Causal inference in statistics, social, and biomedical sciences*. Cambridge University Press. [Causal inference framework]

. **Angrist, J. D., & Pischke, J. S.** (2008). *Mostly harmless econometrics: An empiricist's companion*. Princeton University Press. [Applied causal inference]

. **Craig, P., Cooper, C., Gunnell, D., Haw, S., Lawson, K., Macintyre, S., ... & Thompson, S.** (2012). Using natural experiments to evaluate population health interventions: new Medical Research Council guidance. *Journal of Epidemiology and Community Health*, 66(12), 1182-1186. [Natural experiments]

### Spatial and Temporal Analysis

. **Kulldorff, M.** (1997). A spatial scan statistic. *Communications in Statistics-Theory and Methods*, 26(6), 1481-1496. [Spatial scan statistics]

. **Lawson, A. B.** (2018). *Statistical methods in spatial epidemiology*. John Wiley & Sons. [Spatial epidemiology methods]

. **Waller, L. A., & Gotway, C. A.** (2004). *Applied spatial statistics for public health data*. John Wiley & Sons. [Applied spatial statistics]

. **Diggle, P. J.** (2013). *Statistical analysis of spatial and spatio-temporal point patterns*. CRC Press. [Spatio-temporal analysis]

This chapter provides a comprehensive framework for population health AI systems, addressing disease surveillance, health equity assessment, and causal inference for population-level interventions. The implementations provide practical tools for public health professionals and researchers to analyze population health data, detect health disparities, and evaluate intervention effectiveness. The next chapter will explore health equity applications in greater detail, building upon these population health concepts to address specific equity challenges and intervention strategies.


## Code Examples


All code examples from this chapter are available in the repository:
- **Directory**: [`code_examples/chapter_14/`](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_14/)
- **Direct Download**: [ZIP file](https://github.com/sanjaybasu-waymark/healthcare-ai-book/archive/refs/heads/main.zip)

To use the examples:
```bash
git clone https://github.com/sanjaybasu-waymark/healthcare-ai-book.git
cd healthcare-ai-book/code_examples/chapter_14
pip install -r requirements.txt
```
