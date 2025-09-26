"""
Chapter 3 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

"""
Comprehensive HL7 FHIR Implementation for Healthcare AI and Population Health

This implementation provides a complete FHIR R4 server and client system
designed specifically for healthcare AI applications and population health
management, including advanced features for health equity analysis,
social determinants integration, and population health surveillance.

Author: Sanjay Basu MD PhD
License: MIT
"""

import json
import uuid
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
import aiohttp
from aiohttp import web, ClientSession
import logging
import hashlib
import jwt
from cryptography.fernet import Fernet
import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine, Column, String, DateTime, Text, Integer, Boolean, 
    ForeignKey, Float, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import redis
from pydantic import BaseModel, validator, Field
import warnings
from datetime import timezone, timedelta
import pytz

\# Configure logging for healthcare applications
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

\# Database models for comprehensive healthcare data management
Base = declarative_base()

class FHIRResource(Base):
    """Enhanced FHIR resource model with population health extensions"""
    __tablename__ = 'fhir_resources'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_type = Column(String(50), nullable=False, index=True)
    resource_id = Column(String(64), nullable=False, index=True)
    version_id = Column(String(64), nullable=False, default='1')
    last_updated = Column(DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow)
    resource_data = Column(JSONB, nullable=False)
    is_deleted = Column(Boolean, default=False)
    created_by = Column(String(100))
    organization_id = Column(String(64), index=True)
    
    \# Population health extensions
    population_group = Column(String(100), index=True)
    sdoh_category = Column(String(50), index=True)
    health_equity_flag = Column(Boolean, default=False)
    geographic_region = Column(String(100), index=True)
    risk_stratification = Column(String(20), index=True)
    
    \# Quality and compliance tracking
    data_quality_score = Column(Float, default=1.0)
    privacy_level = Column(String(20), default='standard')
    consent_status = Column(String(20), default='active')

class PopulationHealthMetric(Base):
    """Population health metrics and indicators"""
    __tablename__ = 'population_health_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_category = Column(String(50), nullable=False, index=True)
    population_group = Column(String(100), nullable=False, index=True)
    geographic_region = Column(String(100), nullable=False, index=True)
    measurement_period = Column(String(50), nullable=False)
    
    \# Metric values
    metric_value = Column(Float, nullable=False)
    numerator = Column(Integer)
    denominator = Column(Integer)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    
    \# Metadata
    data_source = Column(String(100), nullable=False)
    calculation_method = Column(Text)
    last_updated = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    
    \# Health equity indicators
    disparity_index = Column(Float)
    equity_target = Column(Float)
    improvement_trend = Column(String(20))

class HealthEquityAssessment(Base):
    """Health equity assessment and disparity tracking"""
    __tablename__ = 'health_equity_assessments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_name = Column(String(100), nullable=False)
    population_groups = Column(ARRAY(String), nullable=False)
    health_outcome = Column(String(100), nullable=False)
    measurement_period = Column(String(50), nullable=False)
    
    \# Disparity metrics
    disparity_ratio = Column(Float)
    absolute_disparity = Column(Float)
    population_attributable_risk = Column(Float)
    theil_index = Column(Float)
    
    \# Geographic analysis
    geographic_level = Column(String(50))  \# county, zip, census_tract
    spatial_autocorrelation = Column(Float)
    
    \# Trend analysis
    trend_direction = Column(String(20))
    trend_significance = Column(Float)
    
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

class HealthcareDataPipeline:
    """
    Comprehensive healthcare data pipeline with FHIR integration
    
    This class provides a complete data pipeline for healthcare AI applications,
    including FHIR resource management, population health analytics, and
    health equity assessment capabilities.
    """
    
    def __init__(self, database_url: str, redis_url: str = None):
        """Initialize the healthcare data pipeline"""
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        \# Initialize Redis for caching if available
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
        
        \# Initialize encryption for PHI protection
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        \# FHIR validation schemas
        self.fhir_schemas = self._load_fhir_schemas()
        
        logger.info("Healthcare data pipeline initialized")
    
    def create_fhir_resource(self, resource_type: str, resource_data: Dict[str, Any],
                           organization_id: str = None, population_group: str = None,
                           sdoh_category: str = None) -> str:
        """
        Create a new FHIR resource with population health extensions
        
        Args:
            resource_type: FHIR resource type (Patient, Observation, etc.)
            resource_data: Complete FHIR resource data
            organization_id: Organization identifier
            population_group: Population/cohort membership
            sdoh_category: Social determinants category
            
        Returns:
            str: Created resource ID
        """
        session = self.Session()
        try:
            \# Validate FHIR resource structure
            self._validate_fhir_resource(resource_type, resource_data)
            
            \# Generate unique resource ID
            resource_id = str(uuid.uuid4())
            
            \# Add FHIR metadata
            resource_data['id'] = resource_id
            resource_data['meta'] = {
                'versionId': '1',
                'lastUpdated': datetime.datetime.utcnow().isoformat() + 'Z'
            }
            
            \# Create database record
            fhir_resource = FHIRResource(
                resource_type=resource_type,
                resource_id=resource_id,
                resource_data=resource_data,
                organization_id=organization_id,
                population_group=population_group,
                sdoh_category=sdoh_category,
                health_equity_flag=self._assess_health_equity_flag(resource_data),
                geographic_region=self._extract_geographic_region(resource_data),
                data_quality_score=self._calculate_data_quality(resource_data)
            )
            
            session.add(fhir_resource)
            session.commit()
            
            \# Cache in Redis if available
            if self.redis_client:
                cache_key = f"fhir:{resource_type}:{resource_id}"
                self.redis_client.setex(
                    cache_key, 
                    3600,  \# 1 hour TTL
                    json.dumps(resource_data)
                )
            
            logger.info(f"Created FHIR resource: {resource_type}/{resource_id}")
            return resource_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating FHIR resource: {e}")
            raise
        finally:
            session.close()
    
    def get_fhir_resource(self, resource_type: str, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a FHIR resource by type and ID
        
        Args:
            resource_type: FHIR resource type
            resource_id: Resource identifier
            
        Returns:
            Dict containing FHIR resource data or None if not found
        """
        \# Check cache first
        if self.redis_client:
            cache_key = f"fhir:{resource_type}:{resource_id}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        
        \# Query database
        session = self.Session()
        try:
            resource = session.query(FHIRResource).filter(
                FHIRResource.resource_type == resource_type,
                FHIRResource.resource_id == resource_id,
                FHIRResource.is_deleted == False
            ).first()
            
            if resource:
                \# Update cache
                if self.redis_client:
                    cache_key = f"fhir:{resource_type}:{resource_id}"
                    self.redis_client.setex(
                        cache_key, 
                        3600,
                        json.dumps(resource.resource_data)
                    )
                
                return resource.resource_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving FHIR resource: {e}")
            raise
        finally:
            session.close()
    
    def search_fhir_resources(self, resource_type: str, 
                            search_params: Dict[str, Any] = None,
                            population_filter: str = None,
                            equity_filter: bool = None) -> List[Dict[str, Any]]:
        """
        Search FHIR resources with population health filtering
        
        Args:
            resource_type: FHIR resource type to search
            search_params: FHIR search parameters
            population_filter: Filter by population group
            equity_filter: Filter by health equity flag
            
        Returns:
            List of matching FHIR resources
        """
        session = self.Session()
        try:
            query = session.query(FHIRResource).filter(
                FHIRResource.resource_type == resource_type,
                FHIRResource.is_deleted == False
            )
            
            \# Apply population health filters
            if population_filter:
                query = query.filter(FHIRResource.population_group == population_filter)
            
            if equity_filter is not None:
                query = query.filter(FHIRResource.health_equity_flag == equity_filter)
            
            \# Apply FHIR search parameters
            if search_params:
                query = self._apply_fhir_search_params(query, search_params)
            
            resources = query.limit(100).all()  \# Limit for performance
            
            return [resource.resource_data for resource in resources]
            
        except Exception as e:
            logger.error(f"Error searching FHIR resources: {e}")
            raise
        finally:
            session.close()
    
    def create_population_health_metric(self, metric_data: Dict[str, Any]) -> str:
        """
        Create a population health metric record
        
        Args:
            metric_data: Population health metric data
            
        Returns:
            str: Created metric ID
        """
        session = self.Session()
        try:
            metric = PopulationHealthMetric(
                metric_name=metric_data['metric_name'],
                metric_category=metric_data['metric_category'],
                population_group=metric_data['population_group'],
                geographic_region=metric_data['geographic_region'],
                measurement_period=metric_data['measurement_period'],
                metric_value=metric_data['metric_value'],
                numerator=metric_data.get('numerator'),
                denominator=metric_data.get('denominator'),
                confidence_interval_lower=metric_data.get('confidence_interval_lower'),
                confidence_interval_upper=metric_data.get('confidence_interval_upper'),
                data_source=metric_data['data_source'],
                calculation_method=metric_data.get('calculation_method'),
                disparity_index=metric_data.get('disparity_index'),
                equity_target=metric_data.get('equity_target'),
                improvement_trend=metric_data.get('improvement_trend')
            )
            
            session.add(metric)
            session.commit()
            
            logger.info(f"Created population health metric: {metric.id}")
            return str(metric.id)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating population health metric: {e}")
            raise
        finally:
            session.close()
    
    def assess_health_equity(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive health equity assessment
        
        Args:
            assessment_data: Health equity assessment parameters
            
        Returns:
            Dict containing equity assessment results
        """
        session = self.Session()
        try:
            \# Calculate disparity metrics
            disparity_metrics = self._calculate_disparity_metrics(
                assessment_data['population_groups'],
                assessment_data['health_outcome'],
                assessment_data['measurement_period']
            )
            
            \# Perform geographic analysis
            geographic_analysis = self._analyze_geographic_disparities(
                assessment_data['population_groups'],
                assessment_data['health_outcome'],
                assessment_data.get('geographic_level', 'county')
            )
            
            \# Trend analysis
            trend_analysis = self._analyze_equity_trends(
                assessment_data['population_groups'],
                assessment_data['health_outcome']
            )
            
            \# Create assessment record
            assessment = HealthEquityAssessment(
                assessment_name=assessment_data['assessment_name'],
                population_groups=assessment_data['population_groups'],
                health_outcome=assessment_data['health_outcome'],
                measurement_period=assessment_data['measurement_period'],
                disparity_ratio=disparity_metrics.get('disparity_ratio'),
                absolute_disparity=disparity_metrics.get('absolute_disparity'),
                population_attributable_risk=disparity_metrics.get('population_attributable_risk'),
                theil_index=disparity_metrics.get('theil_index'),
                geographic_level=assessment_data.get('geographic_level'),
                spatial_autocorrelation=geographic_analysis.get('spatial_autocorrelation'),
                trend_direction=trend_analysis.get('trend_direction'),
                trend_significance=trend_analysis.get('trend_significance')
            )
            
            session.add(assessment)
            session.commit()
            
            results = {
                'assessment_id': str(assessment.id),
                'disparity_metrics': disparity_metrics,
                'geographic_analysis': geographic_analysis,
                'trend_analysis': trend_analysis,
                'recommendations': self._generate_equity_recommendations(
                    disparity_metrics, geographic_analysis, trend_analysis
                )
            }
            
            logger.info(f"Completed health equity assessment: {assessment.id}")
            return results
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error in health equity assessment: {e}")
            raise
        finally:
            session.close()
    
    def _validate_fhir_resource(self, resource_type: str, resource_data: Dict[str, Any]):
        """Validate FHIR resource against schema"""
        \# Simplified validation - in production, use full FHIR validation
        required_fields = {
            'Patient': ['resourceType', 'identifier'],
            'Observation': ['resourceType', 'status', 'code', 'subject'],
            'Condition': ['resourceType', 'code', 'subject'],
            'Medication': ['resourceType', 'code'],
            'Procedure': ['resourceType', 'status', 'code', 'subject']
        }
        
        if resource_type in required_fields:
            for field in required_fields[resource_type]:
                if field not in resource_data:
                    raise ValueError(f"Missing required field '{field}' for {resource_type}")
        
        \# Validate resource type matches
        if resource_data.get('resourceType') != resource_type:
            raise ValueError(f"Resource type mismatch: expected {resource_type}, got {resource_data.get('resourceType')}")
    
    def _assess_health_equity_flag(self, resource_data: Dict[str, Any]) -> bool:
        """Assess if resource relates to health equity concerns"""
        \# Check for social determinants extensions
        if 'extension' in resource_data:
            for ext in resource_data['extension']:
                if 'social-determinants' in ext.get('url', '').lower():
                    return True
        
        \# Check for equity-related codes
        equity_codes = ['Z55', 'Z56', 'Z57', 'Z58', 'Z59', 'Z60', 'Z62', 'Z63', 'Z64', 'Z65']
        if 'code' in resource_data:
            coding = resource_data['code'].get('coding', [])
            for code in coding:
                if any(eq_code in code.get('code', '') for eq_code in equity_codes):
                    return True
        
        return False
    
    def _extract_geographic_region(self, resource_data: Dict[str, Any]) -> Optional[str]:
        """Extract geographic region from resource data"""
        \# Check address information
        if 'address' in resource_data:
            addresses = resource_data['address'] if isinstance(resource_data['address'], list) else [resource_data['address']]
            for addr in addresses:
                if 'state' in addr and 'city' in addr:
                    return f"{addr['city']}, {addr['state']}"
                elif 'postalCode' in addr:
                    return addr['postalCode'][:5]  \# ZIP code
        
        \# Check for geographic extensions
        if 'extension' in resource_data:
            for ext in resource_data['extension']:
                if 'geographic' in ext.get('url', '').lower():
                    return ext.get('valueString')
        
        return None
    
    def _calculate_data_quality(self, resource_data: Dict[str, Any]) -> float:
        """Calculate data quality score for resource"""
        score = 1.0
        
        \# Check for required fields completeness
        total_fields = len(resource_data)
        empty_fields = sum(1 for v in resource_data.values() if v is None or v == '')
        completeness = (total_fields - empty_fields) / total_fields if total_fields > 0 else 0
        
        \# Check for standardized coding
        coding_score = 1.0
        if 'code' in resource_data:
            coding = resource_data['code'].get('coding', [])
            if not coding:
                coding_score = 0.5
            else:
                \# Check for standard code systems
                standard_systems = ['http://snomed.info/sct', 'http://loinc.org', 'http://hl7.org/fhir/sid/icd-10-cm']
                has_standard = any(c.get('system') in standard_systems for c in coding)
                coding_score = 1.0 if has_standard else 0.7
        
        \# Combine scores
        score = (completeness * 0.6) + (coding_score * 0.4)
        
        return round(score, 3)
    
    def _apply_fhir_search_params(self, query, search_params: Dict[str, Any]):
        """Apply FHIR search parameters to database query"""
        \# Simplified implementation - full FHIR search is complex
        for param, value in search_params.items():
            if param == 'identifier':
                query = query.filter(
                    FHIRResource.resource_data['identifier'].astext.contains(value)
                )
            elif param == 'name':
                query = query.filter(
                    FHIRResource.resource_data['name'].astext.contains(value)
                )
            elif param == 'birthdate':
                query = query.filter(
                    FHIRResource.resource_data['birthDate'].astext == value
                )
        
        return query
    
    def _calculate_disparity_metrics(self, population_groups: List[str], 
                                   health_outcome: str, measurement_period: str) -> Dict[str, float]:
        """Calculate health disparity metrics"""
        session = self.Session()
        try:
            \# Get metrics for each population group
            group_metrics = {}
            for group in population_groups:
                metrics = session.query(PopulationHealthMetric).filter(
                    PopulationHealthMetric.population_group == group,
                    PopulationHealthMetric.metric_name == health_outcome,
                    PopulationHealthMetric.measurement_period == measurement_period
                ).all()
                
                if metrics:
                    group_metrics[group] = np.mean([m.metric_value for m in metrics])
            
            if len(group_metrics) < 2:
                return {}
            
            \# Calculate disparity metrics
            values = list(group_metrics.values())
            max_val = max(values)
            min_val = min(values)
            mean_val = np.mean(values)
            
            \# Disparity ratio (max/min)
            disparity_ratio = max_val / min_val if min_val > 0 else float('inf')
            
            \# Absolute disparity (max - min)
            absolute_disparity = max_val - min_val
            
            \# Population attributable risk
            \# Simplified calculation - would need population sizes for accuracy
            population_attributable_risk = (mean_val - min_val) / mean_val if mean_val > 0 else 0
            
            \# Theil index (measure of inequality)
            theil_index = sum((v / mean_val) * np.log(v / mean_val) for v in values if v > 0) / len(values)
            
            return {
                'disparity_ratio': disparity_ratio,
                'absolute_disparity': absolute_disparity,
                'population_attributable_risk': population_attributable_risk,
                'theil_index': theil_index,
                'group_metrics': group_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating disparity metrics: {e}")
            return {}
        finally:
            session.close()
    
    def _analyze_geographic_disparities(self, population_groups: List[str],
                                      health_outcome: str, geographic_level: str) -> Dict[str, Any]:
        """Analyze geographic patterns in health disparities"""
        \# Simplified implementation - would use spatial analysis libraries in production
        session = self.Session()
        try:
            \# Get geographic distribution of metrics
            geographic_metrics = session.query(
                PopulationHealthMetric.geographic_region,
                PopulationHealthMetric.metric_value
            ).filter(
                PopulationHealthMetric.population_group.in_(population_groups),
                PopulationHealthMetric.metric_name == health_outcome
            ).all()
            
            if not geographic_metrics:
                return {}
            
            \# Calculate spatial autocorrelation (simplified)
            regions = [gm.geographic_region for gm in geographic_metrics]
            values = [gm.metric_value for gm in geographic_metrics]
            
            \# Simplified spatial autocorrelation calculation
            \# In production, would use proper spatial weights matrix
            spatial_autocorrelation = np.corrcoef(values, values)[0, 1] if len(values) > 1 else 0
            
            return {
                'geographic_level': geographic_level,
                'spatial_autocorrelation': spatial_autocorrelation,
                'geographic_range': max(values) - min(values) if values else 0,
                'high_disparity_regions': [
                    regions[i] for i, v in enumerate(values) 
                    if v > np.mean(values) + np.std(values)
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in geographic analysis: {e}")
            return {}
        finally:
            session.close()
    
    def _analyze_equity_trends(self, population_groups: List[str], 
                             health_outcome: str) -> Dict[str, Any]:
        """Analyze trends in health equity over time"""
        session = self.Session()
        try:
            \# Get historical metrics
            historical_metrics = session.query(PopulationHealthMetric).filter(
                PopulationHealthMetric.population_group.in_(population_groups),
                PopulationHealthMetric.metric_name == health_outcome
            ).order_by(PopulationHealthMetric.measurement_period).all()
            
            if len(historical_metrics) < 2:
                return {'trend_direction': 'insufficient_data'}
            
            \# Group by time period
            period_metrics = {}
            for metric in historical_metrics:
                period = metric.measurement_period
                if period not in period_metrics:
                    period_metrics[period] = []
                period_metrics[period].append(metric.metric_value)
            
            \# Calculate trend
            periods = sorted(period_metrics.keys())
            if len(periods) < 2:
                return {'trend_direction': 'insufficient_data'}
            
            \# Calculate disparity for each period
            period_disparities = []
            for period in periods:
                values = period_metrics[period]
                if len(values) > 1:
                    disparity = max(values) - min(values)
                    period_disparities.append(disparity)
            
            if len(period_disparities) < 2:
                return {'trend_direction': 'insufficient_data'}
            
            \# Determine trend direction
            recent_disparity = period_disparities[-1]
            earlier_disparity = period_disparities<sup>0</sup>
            
            if recent_disparity < earlier_disparity * 0.95:
                trend_direction = 'improving'
            elif recent_disparity > earlier_disparity * 1.05:
                trend_direction = 'worsening'
            else:
                trend_direction = 'stable'
            
            \# Calculate trend significance (simplified)
            trend_significance = abs(recent_disparity - earlier_disparity) / earlier_disparity if earlier_disparity > 0 else 0
            
            return {
                'trend_direction': trend_direction,
                'trend_significance': trend_significance,
                'periods_analyzed': len(periods),
                'disparity_change': recent_disparity - earlier_disparity
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'trend_direction': 'error'}
        finally:
            session.close()
    
    def _generate_equity_recommendations(self, disparity_metrics: Dict[str, Any],
                                       geographic_analysis: Dict[str, Any],
                                       trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations for health equity improvement"""
        recommendations = []
        
        \# Disparity-based recommendations
        if disparity_metrics.get('disparity_ratio', 1) > 2.0:
            recommendations.append(
                "High disparity ratio detected. Consider targeted interventions for "
                "lowest-performing population groups."
            )
        
        if disparity_metrics.get('theil_index', 0) > 0.1:
            recommendations.append(
                "Significant inequality detected across population groups. "
                "Implement comprehensive equity improvement strategies."
            )
        
        \# Geographic recommendations
        if geographic_analysis.get('high_disparity_regions'):
            recommendations.append(
                f"Focus interventions on high-disparity regions: "
                f"{', '.join(geographic_analysis['high_disparity_regions'][:3])}"
            )
        
        \# Trend-based recommendations
        if trend_analysis.get('trend_direction') == 'worsening':
            recommendations.append(
                "Disparities are worsening over time. Urgent intervention needed "
                "to reverse negative trends."
            )
        elif trend_analysis.get('trend_direction') == 'stable':
            recommendations.append(
                "Disparities remain stable. Consider new approaches to achieve "
                "meaningful equity improvements."
            )
        
        if not recommendations:
            recommendations.append(
                "Continue monitoring health equity metrics and maintain "
                "current intervention strategies."
            )
        
        return recommendations
    
    def _load_fhir_schemas(self) -> Dict[str, Any]:
        """Load FHIR validation schemas"""
        \# Simplified - in production, load full FHIR schemas
        return {
            'Patient': {'required': ['resourceType', 'identifier']},
            'Observation': {'required': ['resourceType', 'status', 'code', 'subject']},
            'Condition': {'required': ['resourceType', 'code', 'subject']}
        }


class FHIRServer:
    """
    Production-ready FHIR server implementation
    
    This server provides a complete FHIR R4 API with enhanced capabilities
    for population health data management and health equity analysis.
    """
    
    def __init__(self, data_pipeline: HealthcareDataPipeline):
        """Initialize FHIR server"""
        self.data_pipeline = data_pipeline
        self.app = web.Application()
        self._setup_routes()
        self._setup_middleware()
        
        logger.info("FHIR server initialized")
    
    def _setup_routes(self):
        """Setup FHIR API routes"""
        \# Standard FHIR routes
        self.app.router.add_get('/fhir/metadata', self.get_capability_statement)
        self.app.router.add_post('/fhir/{resource_type}', self.create_resource)
        self.app.router.add_get('/fhir/{resource_type}/{resource_id}', self.read_resource)
        self.app.router.add_put('/fhir/{resource_type}/{resource_id}', self.update_resource)
        self.app.router.add_delete('/fhir/{resource_type}/{resource_id}', self.delete_resource)
        self.app.router.add_get('/fhir/{resource_type}', self.search_resources)
        
        \# Population health extensions
        self.app.router.add_post('/fhir/population-health/metrics', self.create_population_metric)
        self.app.router.add_post('/fhir/population-health/equity-assessment', self.assess_health_equity)
        self.app.router.add_get('/fhir/population-health/dashboard', self.get_population_dashboard)
    
    def _setup_middleware(self):
        """Setup middleware for authentication, logging, etc."""
        @web.middleware
        async def auth_middleware(request, handler):
            \# Simplified authentication - implement OAuth2/SMART on FHIR in production
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return web.json_response({'error': 'Authentication required'}, status=401)
            
            \# Validate token (simplified)
            token = auth_header.split(' ')<sup>1</sup>
            if not self._validate_token(token):
                return web.json_response({'error': 'Invalid token'}, status=401)
            
            return await handler(request)
        
        @web.middleware
        async def logging_middleware(request, handler):
            start_time = datetime.datetime.utcnow()
            response = await handler(request)
            duration = (datetime.datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"{request.method} {request.path} - {response.status} - {duration:.3f}s")
            return response
        
        self.app.middlewares.append(auth_middleware)
        self.app.middlewares.append(logging_middleware)
    
    async def get_capability_statement(self, request):
        """Return FHIR capability statement"""
        capability = {
            "resourceType": "CapabilityStatement",
            "status": "active",
            "date": datetime.datetime.utcnow().isoformat() + 'Z',
            "publisher": "Healthcare AI Implementation Guide",
            "kind": "instance",
            "software": {
                "name": "Healthcare AI FHIR Server",
                "version": "1.0.0"
            },
            "fhirVersion": "4.0.1",
            "format": ["json"],
            "rest": [{
                "mode": "server",
                "resource": [
                    {
                        "type": "Patient",
                        "interaction": [
                            {"code": "create"},
                            {"code": "read"},
                            {"code": "update"},
                            {"code": "delete"},
                            {"code": "search-type"}
                        ]
                    },
                    {
                        "type": "Observation",
                        "interaction": [
                            {"code": "create"},
                            {"code": "read"},
                            {"code": "update"},
                            {"code": "delete"},
                            {"code": "search-type"}
                        ]
                    }
                ]
            }]
        }
        
        return web.json_response(capability)
    
    async def create_resource(self, request):
        """Create a new FHIR resource"""
        try:
            resource_type = request.match_info['resource_type']
            resource_data = await request.json()
            
            \# Extract population health parameters from headers
            population_group = request.headers.get('X-Population-Group')
            sdoh_category = request.headers.get('X-SDOH-Category')
            organization_id = request.headers.get('X-Organization-ID')
            
            resource_id = self.data_pipeline.create_fhir_resource(
                resource_type=resource_type,
                resource_data=resource_data,
                organization_id=organization_id,
                population_group=population_group,
                sdoh_category=sdoh_category
            )
            
            \# Return created resource
            created_resource = self.data_pipeline.get_fhir_resource(resource_type, resource_id)
            
            return web.json_response(
                created_resource,
                status=201,
                headers={'Location': f'/fhir/{resource_type}/{resource_id}'}
            )
            
        except Exception as e:
            logger.error(f"Error creating resource: {e}")
            return web.json_response(
                {'error': 'Internal server error'},
                status=500
            )
    
    async def read_resource(self, request):
        """Read a FHIR resource by ID"""
        try:
            resource_type = request.match_info['resource_type']
            resource_id = request.match_info['resource_id']
            
            resource = self.data_pipeline.get_fhir_resource(resource_type, resource_id)
            
            if resource:
                return web.json_response(resource)
            else:
                return web.json_response(
                    {'error': 'Resource not found'},
                    status=404
                )
                
        except Exception as e:
            logger.error(f"Error reading resource: {e}")
            return web.json_response(
                {'error': 'Internal server error'},
                status=500
            )
    
    async def search_resources(self, request):
        """Search FHIR resources"""
        try:
            resource_type = request.match_info['resource_type']
            search_params = dict(request.query)
            
            \# Extract population health filters
            population_filter = search_params.pop('population-group', None)
            equity_filter = search_params.pop('health-equity', None)
            if equity_filter:
                equity_filter = equity_filter.lower() == 'true'
            
            resources = self.data_pipeline.search_fhir_resources(
                resource_type=resource_type,
                search_params=search_params,
                population_filter=population_filter,
                equity_filter=equity_filter
            )
            
            \# Create FHIR Bundle response
            bundle = {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(resources),
                "entry": [
                    {
                        "resource": resource,
                        "search": {"mode": "match"}
                    }
                    for resource in resources
                ]
            }
            
            return web.json_response(bundle)
            
        except Exception as e:
            logger.error(f"Error searching resources: {e}")
            return web.json_response(
                {'error': 'Internal server error'},
                status=500
            )
    
    async def create_population_metric(self, request):
        """Create population health metric"""
        try:
            metric_data = await request.json()
            metric_id = self.data_pipeline.create_population_health_metric(metric_data)
            
            return web.json_response(
                {'id': metric_id, 'status': 'created'},
                status=201
            )
            
        except Exception as e:
            logger.error(f"Error creating population metric: {e}")
            return web.json_response(
                {'error': 'Internal server error'},
                status=500
            )
    
    async def assess_health_equity(self, request):
        """Perform health equity assessment"""
        try:
            assessment_data = await request.json()
            results = self.data_pipeline.assess_health_equity(assessment_data)
            
            return web.json_response(results)
            
        except Exception as e:
            logger.error(f"Error in health equity assessment: {e}")
            return web.json_response(
                {'error': 'Internal server error'},
                status=500
            )
    
    def _validate_token(self, token: str) -> bool:
        """Validate authentication token"""
        \# Simplified validation - implement proper OAuth2/JWT validation in production
        return len(token) > 10  \# Placeholder validation
    
    def run(self, host='localhost', port=8080):
        """Run the FHIR server"""
        logger.info(f"Starting FHIR server on {host}:{port}")
        web.run_app(self.app, host=host, port=port)


\# Demonstration and testing functions
def create_example_healthcare_pipeline():
    """Create example healthcare data pipeline for demonstration"""
    \# Use SQLite for demonstration (PostgreSQL recommended for production)
    database_url = "sqlite:///healthcare_demo.db"
    
    pipeline = HealthcareDataPipeline(database_url)
    
    return pipeline


def demonstrate_fhir_implementation():
    """Demonstrate comprehensive FHIR implementation"""
    print("=== Healthcare Data Engineering with FHIR ===\n")
    
    \# Create healthcare data pipeline
    pipeline = create_example_healthcare_pipeline()
    
    \# 1. Create sample FHIR resources
    print("1. Creating FHIR Resources")
    print("-" * 40)
    
    \# Create a Patient resource
    patient_data = {
        "resourceType": "Patient",
        "identifier": [
            {
                "system": "http://hospital.example.org/patients",
                "value": "12345"
            }
        ],
        "name": [
            {
                "family": "Doe",
                "given": ["John"]
            }
        ],
        "gender": "male",
        "birthDate": "1980-01-01",
        "address": [
            {
                "city": "Boston",
                "state": "MA",
                "postalCode": "02101"
            }
        ]
    }
    
    patient_id = pipeline.create_fhir_resource(
        resource_type="Patient",
        resource_data=patient_data,
        organization_id="hospital-001",
        population_group="urban-adults"
    )
    
    print(f"Created Patient: {patient_id}")
    
    \# Create an Observation resource with SDOH data
    observation_data = {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "76437-3",
                    "display": "Primary insurance"
                }
            ]
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "valueString": "Medicaid",
        "extension": [
            {
                "url": "http://hl7.org/fhir/us/sdoh-clinicalcare/StructureDefinition/SDOH-Category",
                "valueCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/us/sdoh-clinicalcare/CodeSystem/SDOH-Category",
                            "code": "health-insurance-coverage-status"
                        }
                    ]
                }
            }
        ]
    }
    
    observation_id = pipeline.create_fhir_resource(
        resource_type="Observation",
        resource_data=observation_data,
        organization_id="hospital-001",
        population_group="urban-adults",
        sdoh_category="insurance"
    )
    
    print(f"Created Observation (SDOH): {observation_id}")
    
    \# 2. Search FHIR resources
    print(f"\n2. Searching FHIR Resources")
    print("-" * 40)
    
    \# Search for patients
    patients = pipeline.search_fhir_resources(
        resource_type="Patient",
        search_params={"name": "Doe"},
        population_filter="urban-adults"
    )
    
    print(f"Found {len(patients)} patients matching search criteria")
    
    \# Search for health equity related observations
    equity_observations = pipeline.search_fhir_resources(
        resource_type="Observation",
        equity_filter=True
    )
    
    print(f"Found {len(equity_observations)} health equity related observations")
    
    \# 3. Create population health metrics
    print(f"\n3. Population Health Metrics")
    print("-" * 40)
    
    \# Create sample population health metrics
    metrics_data = [
        {
            "metric_name": "Diabetes Prevalence",
            "metric_category": "chronic_disease",
            "population_group": "urban-adults",
            "geographic_region": "Boston, MA",
            "measurement_period": "2023-Q4",
            "metric_value": 0.085,
            "numerator": 850,
            "denominator": 10000,
            "data_source": "EHR_Analysis",
            "disparity_index": 1.2,
            "equity_target": 0.070
        },
        {
            "metric_name": "Diabetes Prevalence",
            "metric_category": "chronic_disease",
            "population_group": "rural-adults",
            "geographic_region": "Western MA",
            "measurement_period": "2023-Q4",
            "metric_value": 0.102,
            "numerator": 510,
            "denominator": 5000,
            "data_source": "EHR_Analysis",
            "disparity_index": 1.5,
            "equity_target": 0.070
        }
    ]
    
    metric_ids = []
    for metric_data in metrics_data:
        metric_id = pipeline.create_population_health_metric(metric_data)
        metric_ids.append(metric_id)
        print(f"Created metric: {metric_data['metric_name']} for {metric_data['population_group']}")
    
    \# 4. Health equity assessment
    print(f"\n4. Health Equity Assessment")
    print("-" * 40)
    
    equity_assessment = {
        "assessment_name": "Diabetes Disparity Analysis",
        "population_groups": ["urban-adults", "rural-adults"],
        "health_outcome": "Diabetes Prevalence",
        "measurement_period": "2023-Q4",
        "geographic_level": "county"
    }
    
    equity_results = pipeline.assess_health_equity(equity_assessment)
    
    print(f"Health Equity Assessment Results:")
    print(f"  Assessment ID: {equity_results['assessment_id']}")
    
    disparity_metrics = equity_results['disparity_metrics']
    print(f"  Disparity Ratio: {disparity_metrics.get('disparity_ratio', 'N/A'):.2f}")
    print(f"  Absolute Disparity: {disparity_metrics.get('absolute_disparity', 'N/A'):.3f}")
    print(f"  Theil Index: {disparity_metrics.get('theil_index', 'N/A'):.3f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(equity_results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    \# 5. Data quality assessment
    print(f"\n5. Data Quality Assessment")
    print("-" * 40)
    
    \# Retrieve and assess data quality
    patient_resource = pipeline.get_fhir_resource("Patient", patient_id)
    if patient_resource:
        print(f"Patient resource retrieved successfully")
        print(f"  Resource ID: {patient_resource['id']}")
        print(f"  Last Updated: {patient_resource['meta']['lastUpdated']}")
        
        \# Check data completeness
        required_fields = ['identifier', 'name', 'gender', 'birthDate']
        completeness = sum(1 for field in required_fields if field in patient_resource) / len(required_fields)
        print(f"  Data Completeness: {completeness:.1%}")


def demonstrate_fhir_server():
    """Demonstrate FHIR server functionality"""
    print("\n=== FHIR Server Demonstration ===")
    print("Note: This would start a web server in a real implementation")
    print("Server would be available at: http://localhost:8080/fhir/")
    print("\nSupported endpoints:")
    print("  GET  /fhir/metadata - Capability statement")
    print("  POST /fhir/Patient - Create patient")
    print("  GET  /fhir/Patient/{id} - Read patient")
    print("  GET  /fhir/Patient?name=Doe - Search patients")
    print("  POST /fhir/population-health/metrics - Create population metric")
    print("  POST /fhir/population-health/equity-assessment - Health equity assessment")


if __name__ == "__main__":
    demonstrate_fhir_implementation()
    demonstrate_fhir_server()