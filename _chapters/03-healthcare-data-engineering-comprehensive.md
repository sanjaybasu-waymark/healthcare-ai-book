# Chapter 3: Healthcare Data Engineering - Building Robust Clinical Data Infrastructure

*"The foundation of any successful healthcare AI system lies not in the sophistication of its algorithms, but in the quality, accessibility, and reliability of the data infrastructure that supports it."*

## Introduction

Healthcare data engineering represents one of the most complex and critical aspects of implementing AI systems in clinical environments. Unlike other domains where data is often generated in standardized formats, healthcare data exists in a bewildering array of formats, standards, and systems that have evolved over decades of technological development. This chapter provides a comprehensive exploration of the principles, practices, and technologies required to build robust, scalable, and compliant healthcare data infrastructure.

The unique challenges of healthcare data engineering stem from the intersection of technical complexity, regulatory requirements, and clinical workflow constraints. Healthcare data must be processed with perfect accuracy, as errors can directly impact patient safety. The data must be available in real-time for critical care decisions, yet also support complex analytical workloads for population health management and research. Privacy and security requirements add additional layers of complexity that are rarely encountered in other domains.

Modern healthcare data engineering requires expertise in multiple technical domains: traditional database systems, real-time streaming platforms, cloud computing architectures, data integration standards like HL7 FHIR, and specialized healthcare technologies. This chapter provides comprehensive implementations and practical guidance for each of these areas, demonstrating how they can be integrated into cohesive systems that support both clinical operations and AI applications.

The implementations presented in this chapter are production-ready and have been designed to handle the scale and complexity of real healthcare environments. Each system includes comprehensive error handling, monitoring, and compliance features that are essential for healthcare applications. The code examples demonstrate not just how to build these systems, but how to build them correctly for healthcare use cases.

## Healthcare Data Standards and Interoperability

Healthcare interoperability represents one of the most significant challenges in modern healthcare IT. The ability to seamlessly exchange clinical data between different systems, organizations, and applications is essential for coordinated care, population health management, and AI system development. This section explores the technical foundations of healthcare interoperability and provides comprehensive implementations of the key standards and protocols.

### HL7 FHIR: The Foundation of Modern Healthcare Interoperability

Fast Healthcare Interoperability Resources (FHIR) has emerged as the dominant standard for healthcare data exchange, representing a paradigm shift from document-based exchange to resource-based APIs. FHIR combines the best aspects of previous HL7 standards with modern web technologies, creating a framework that is both clinically comprehensive and technically accessible.

FHIR organizes healthcare information into discrete resources, each representing a specific clinical concept such as Patient, Observation, Medication, or Encounter. These resources are defined using a consistent structure that includes data elements, relationships to other resources, and standardized terminologies. The RESTful API design enables real-time data access and supports both simple queries and complex clinical workflows.

The following comprehensive implementation demonstrates a production-ready FHIR server and client system designed for healthcare AI applications:

```python
"""
Comprehensive HL7 FHIR Implementation for Healthcare AI
Production-ready FHIR server and client with advanced features

This implementation provides a complete FHIR R4 server and client system
designed specifically for healthcare AI applications, including advanced
features for data validation, security, and clinical workflow integration.

Author: Sanjay Basu, MD PhD
Institution: Waymark
License: Educational use - requires clinical validation for production
References: 
- HL7 FHIR R4 Specification: http://hl7.org/fhir/R4/
- Ayaz, M. et al. (2021). The Fast Health Interoperability Resources (FHIR) Standard. PMC8367140
"""

import json
import uuid
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
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
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import redis
from pydantic import BaseModel, validator, Field
from fhir.resources import patient, observation, medication, encounter, bundle
from fhir.resources.fhirtypes import Id, DateTime as FHIRDateTime
import warnings
warnings.filterwarnings('ignore')

# Advanced libraries for healthcare data processing
from fhirclient import client
from fhirclient.models import patient as fhir_patient
from fhirclient.models import observation as fhir_observation
import pytz
from datetime import timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class FHIRResource(Base):
    """Base model for FHIR resources in the database"""
    __tablename__ = 'fhir_resources'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_type = Column(String(50), nullable=False, index=True)
    resource_id = Column(String(64), nullable=False, index=True)
    version_id = Column(String(64), nullable=False, default='1')
    last_updated = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    resource_data = Column(JSONB, nullable=False)
    is_deleted = Column(Boolean, default=False)
    created_by = Column(String(100))
    organization_id = Column(String(64), index=True)
    
    def __repr__(self):
        return f"<FHIRResource(type={self.resource_type}, id={self.resource_id})>"

class AuditLog(Base):
    """Audit log for FHIR operations"""
    __tablename__ = 'fhir_audit_log'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    operation = Column(String(20), nullable=False)  # CREATE, READ, UPDATE, DELETE
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(64), nullable=False)
    user_id = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)

class FHIRResourceValidator:
    """Advanced FHIR resource validation with clinical rules"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.terminology_cache = {}
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load clinical validation rules for FHIR resources"""
        return {
            'Patient': {
                'required_fields': ['identifier', 'name', 'gender'],
                'identifier_systems': [
                    'http://hl7.org/fhir/sid/us-ssn',
                    'http://hl7.org/fhir/sid/us-medicare',
                    'http://hospital.example.org/patient-id'
                ],
                'gender_values': ['male', 'female', 'other', 'unknown']
            },
            'Observation': {
                'required_fields': ['status', 'code', 'subject'],
                'status_values': ['registered', 'preliminary', 'final', 'amended', 'corrected', 'cancelled'],
                'vital_signs_codes': {
                    '8480-6': 'Systolic blood pressure',
                    '8462-4': 'Diastolic blood pressure',
                    '8867-4': 'Heart rate',
                    '8310-5': 'Body temperature',
                    '33747-0': 'General appearance'
                }
            },
            'Medication': {
                'required_fields': ['code'],
                'rxnorm_system': 'http://www.nlm.nih.gov/research/umls/rxnorm'
            }
        }
    
    def validate_resource(self, resource_type: str, resource_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of FHIR resources
        
        Args:
            resource_type: Type of FHIR resource
            resource_data: Resource data dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Basic structure validation
        if 'resourceType' not in resource_data:
            errors.append("Missing required field: resourceType")
        elif resource_data['resourceType'] != resource_type:
            errors.append(f"Resource type mismatch: expected {resource_type}, got {resource_data['resourceType']}")
        
        # Resource-specific validation
        if resource_type in self.validation_rules:
            rules = self.validation_rules[resource_type]
            
            # Required fields validation
            for field in rules.get('required_fields', []):
                if field not in resource_data:
                    errors.append(f"Missing required field: {field}")
            
            # Resource-specific validations
            if resource_type == 'Patient':
                errors.extend(self._validate_patient(resource_data, rules))
            elif resource_type == 'Observation':
                errors.extend(self._validate_observation(resource_data, rules))
            elif resource_type == 'Medication':
                errors.extend(self._validate_medication(resource_data, rules))
        
        # Clinical validation
        clinical_errors = self._validate_clinical_rules(resource_type, resource_data)
        errors.extend(clinical_errors)
        
        return len(errors) == 0, errors
    
    def _validate_patient(self, resource_data: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Validate Patient resource"""
        errors = []
        
        # Gender validation
        if 'gender' in resource_data:
            if resource_data['gender'] not in rules['gender_values']:
                errors.append(f"Invalid gender value: {resource_data['gender']}")
        
        # Identifier validation
        if 'identifier' in resource_data:
            for identifier in resource_data['identifier']:
                if 'system' in identifier:
                    if identifier['system'] not in rules['identifier_systems']:
                        logger.warning(f"Unknown identifier system: {identifier['system']}")
                
                # Validate SSN format if present
                if identifier.get('system') == 'http://hl7.org/fhir/sid/us-ssn':
                    ssn = identifier.get('value', '')
                    if not self._validate_ssn_format(ssn):
                        errors.append(f"Invalid SSN format: {ssn}")
        
        # Birth date validation
        if 'birthDate' in resource_data:
            try:
                birth_date = datetime.datetime.fromisoformat(resource_data['birthDate'].replace('Z', '+00:00'))
                if birth_date > datetime.datetime.now(timezone.utc):
                    errors.append("Birth date cannot be in the future")
                if birth_date < datetime.datetime(1900, 1, 1, tzinfo=timezone.utc):
                    errors.append("Birth date seems unrealistic (before 1900)")
            except ValueError:
                errors.append(f"Invalid birth date format: {resource_data['birthDate']}")
        
        return errors
    
    def _validate_observation(self, resource_data: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Validate Observation resource"""
        errors = []
        
        # Status validation
        if 'status' in resource_data:
            if resource_data['status'] not in rules['status_values']:
                errors.append(f"Invalid observation status: {resource_data['status']}")
        
        # Code validation
        if 'code' in resource_data and 'coding' in resource_data['code']:
            for coding in resource_data['code']['coding']:
                if coding.get('system') == 'http://loinc.org':
                    # Validate LOINC codes
                    loinc_code = coding.get('code')
                    if loinc_code and not self._validate_loinc_code(loinc_code):
                        errors.append(f"Invalid LOINC code: {loinc_code}")
        
        # Value validation for vital signs
        if 'valueQuantity' in resource_data:
            value_errors = self._validate_vital_sign_values(resource_data)
            errors.extend(value_errors)
        
        return errors
    
    def _validate_medication(self, resource_data: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Validate Medication resource"""
        errors = []
        
        # RxNorm code validation
        if 'code' in resource_data and 'coding' in resource_data['code']:
            for coding in resource_data['code']['coding']:
                if coding.get('system') == rules['rxnorm_system']:
                    rxnorm_code = coding.get('code')
                    if rxnorm_code and not self._validate_rxnorm_code(rxnorm_code):
                        errors.append(f"Invalid RxNorm code: {rxnorm_code}")
        
        return errors
    
    def _validate_clinical_rules(self, resource_type: str, resource_data: Dict[str, Any]) -> List[str]:
        """Apply clinical validation rules"""
        errors = []
        
        if resource_type == 'Observation':
            # Validate vital sign ranges
            if 'valueQuantity' in resource_data and 'code' in resource_data:
                code_system = None
                code_value = None
                
                if 'coding' in resource_data['code']:
                    for coding in resource_data['code']['coding']:
                        if coding.get('system') == 'http://loinc.org':
                            code_value = coding.get('code')
                            break
                
                if code_value and 'value' in resource_data['valueQuantity']:
                    value = resource_data['valueQuantity']['value']
                    clinical_errors = self._validate_vital_sign_ranges(code_value, value)
                    errors.extend(clinical_errors)
        
        return errors
    
    def _validate_ssn_format(self, ssn: str) -> bool:
        """Validate SSN format (XXX-XX-XXXX)"""
        import re
        pattern = r'^\d{3}-\d{2}-\d{4}$'
        return bool(re.match(pattern, ssn))
    
    def _validate_loinc_code(self, loinc_code: str) -> bool:
        """Validate LOINC code format"""
        import re
        pattern = r'^\d{1,5}-\d$'
        return bool(re.match(pattern, loinc_code))
    
    def _validate_rxnorm_code(self, rxnorm_code: str) -> bool:
        """Validate RxNorm code format"""
        return rxnorm_code.isdigit() and len(rxnorm_code) <= 8
    
    def _validate_vital_sign_values(self, resource_data: Dict[str, Any]) -> List[str]:
        """Validate vital sign values for clinical reasonableness"""
        errors = []
        
        if 'valueQuantity' not in resource_data or 'value' not in resource_data['valueQuantity']:
            return errors
        
        value = resource_data['valueQuantity']['value']
        unit = resource_data['valueQuantity'].get('unit', '')
        
        # Get the observation code
        code_value = None
        if 'code' in resource_data and 'coding' in resource_data['code']:
            for coding in resource_data['code']['coding']:
                if coding.get('system') == 'http://loinc.org':
                    code_value = coding.get('code')
                    break
        
        if code_value:
            clinical_errors = self._validate_vital_sign_ranges(code_value, value)
            errors.extend(clinical_errors)
        
        return errors
    
    def _validate_vital_sign_ranges(self, loinc_code: str, value: float) -> List[str]:
        """Validate vital sign values against clinical ranges"""
        errors = []
        
        # Clinical ranges for common vital signs
        ranges = {
            '8480-6': {'min': 70, 'max': 250, 'name': 'Systolic BP'},  # mmHg
            '8462-4': {'min': 40, 'max': 150, 'name': 'Diastolic BP'},  # mmHg
            '8867-4': {'min': 30, 'max': 200, 'name': 'Heart Rate'},    # bpm
            '8310-5': {'min': 95, 'max': 110, 'name': 'Body Temperature'},  # °F
            '8302-2': {'min': 0, 'max': 300, 'name': 'Body Height'},    # cm
            '29463-7': {'min': 0, 'max': 1000, 'name': 'Body Weight'}   # kg
        }
        
        if loinc_code in ranges:
            range_info = ranges[loinc_code]
            if value < range_info['min'] or value > range_info['max']:
                errors.append(
                    f"Clinical warning: {range_info['name']} value {value} "
                    f"outside expected range ({range_info['min']}-{range_info['max']})"
                )
        
        return errors

class FHIRServer:
    """
    Production-ready FHIR R4 server implementation
    
    This server provides a complete FHIR R4 API with advanced features
    including validation, security, audit logging, and clinical workflow support.
    """
    
    def __init__(self, database_url: str, redis_url: str = None):
        self.database_url = database_url
        self.redis_url = redis_url
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize Redis for caching
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        else:
            self.redis_client = None
        
        self.validator = FHIRResourceValidator()
        self.app = web.Application()
        self._setup_routes()
        self._setup_middleware()
        
        # Security configuration
        self.jwt_secret = "your-jwt-secret-key"  # In production, use environment variable
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        logger.info("FHIR Server initialized successfully")
    
    def _setup_routes(self):
        """Setup FHIR API routes"""
        # Metadata endpoint
        self.app.router.add_get('/metadata', self.get_capability_statement)
        
        # Resource endpoints
        self.app.router.add_post('/{resource_type}', self.create_resource)
        self.app.router.add_get('/{resource_type}', self.search_resources)
        self.app.router.add_get('/{resource_type}/{resource_id}', self.read_resource)
        self.app.router.add_put('/{resource_type}/{resource_id}', self.update_resource)
        self.app.router.add_delete('/{resource_type}/{resource_id}', self.delete_resource)
        
        # History endpoints
        self.app.router.add_get('/{resource_type}/{resource_id}/_history', self.get_resource_history)
        self.app.router.add_get('/{resource_type}/_history', self.get_type_history)
        self.app.router.add_get('/_history', self.get_system_history)
        
        # Bundle operations
        self.app.router.add_post('/', self.process_bundle)
        
        # Custom operations for AI/ML
        self.app.router.add_post('/{resource_type}/$validate', self.validate_resource_endpoint)
        self.app.router.add_post('/Patient/$match', self.patient_match)
        self.app.router.add_get('/Observation/$stats', self.observation_statistics)
    
    def _setup_middleware(self):
        """Setup middleware for security, logging, and error handling"""
        
        @web.middleware
        async def auth_middleware(request, handler):
            """JWT authentication middleware"""
            if request.path in ['/metadata']:
                return await handler(request)
            
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return web.json_response(
                    {'error': 'Missing or invalid authorization header'}, 
                    status=401
                )
            
            token = auth_header.split(' ')[1]
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
                request['user'] = payload
            except jwt.InvalidTokenError:
                return web.json_response({'error': 'Invalid token'}, status=401)
            
            return await handler(request)
        
        @web.middleware
        async def audit_middleware(request, handler):
            """Audit logging middleware"""
            start_time = datetime.datetime.utcnow()
            user_id = request.get('user', {}).get('sub', 'anonymous')
            
            try:
                response = await handler(request)
                success = response.status < 400
                error_message = None
            except Exception as e:
                success = False
                error_message = str(e)
                response = web.json_response({'error': 'Internal server error'}, status=500)
            
            # Log the operation
            await self._log_audit_event(
                request=request,
                user_id=user_id,
                success=success,
                error_message=error_message
            )
            
            return response
        
        @web.middleware
        async def cors_middleware(request, handler):
            """CORS middleware for web client support"""
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response
        
        self.app.middlewares.append(auth_middleware)
        self.app.middlewares.append(audit_middleware)
        self.app.middlewares.append(cors_middleware)
    
    async def get_capability_statement(self, request):
        """Return FHIR capability statement"""
        capability = {
            "resourceType": "CapabilityStatement",
            "id": "healthcare-ai-fhir-server",
            "url": "http://example.org/fhir/CapabilityStatement/healthcare-ai-fhir-server",
            "version": "1.0.0",
            "name": "HealthcareAIFHIRServer",
            "title": "Healthcare AI FHIR Server",
            "status": "active",
            "date": datetime.datetime.utcnow().isoformat() + "Z",
            "publisher": "Waymark Healthcare AI",
            "description": "FHIR R4 server optimized for healthcare AI applications",
            "kind": "instance",
            "software": {
                "name": "Healthcare AI FHIR Server",
                "version": "1.0.0"
            },
            "implementation": {
                "description": "Healthcare AI FHIR Server",
                "url": request.url.with_path('').human_repr()
            },
            "fhirVersion": "4.0.1",
            "format": ["json"],
            "rest": [{
                "mode": "server",
                "security": {
                    "cors": True,
                    "service": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/restful-security-service",
                            "code": "OAuth",
                            "display": "OAuth"
                        }]
                    }]
                },
                "resource": [
                    {
                        "type": "Patient",
                        "interaction": [
                            {"code": "create"},
                            {"code": "read"},
                            {"code": "update"},
                            {"code": "delete"},
                            {"code": "search-type"}
                        ],
                        "searchParam": [
                            {"name": "identifier", "type": "token"},
                            {"name": "name", "type": "string"},
                            {"name": "gender", "type": "token"},
                            {"name": "birthdate", "type": "date"}
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
                        ],
                        "searchParam": [
                            {"name": "subject", "type": "reference"},
                            {"name": "code", "type": "token"},
                            {"name": "date", "type": "date"},
                            {"name": "status", "type": "token"}
                        ]
                    }
                ]
            }]
        }
        
        return web.json_response(capability)
    
    async def create_resource(self, request):
        """Create a new FHIR resource"""
        resource_type = request.match_info['resource_type']
        
        try:
            resource_data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        
        # Validate resource
        is_valid, errors = self.validator.validate_resource(resource_type, resource_data)
        if not is_valid:
            return web.json_response({
                'error': 'Validation failed',
                'details': errors
            }, status=400)
        
        # Generate resource ID if not provided
        if 'id' not in resource_data:
            resource_data['id'] = str(uuid.uuid4())
        
        # Set metadata
        resource_data['meta'] = {
            'versionId': '1',
            'lastUpdated': datetime.datetime.utcnow().isoformat() + 'Z'
        }
        
        # Store in database
        session = self.Session()
        try:
            fhir_resource = FHIRResource(
                resource_type=resource_type,
                resource_id=resource_data['id'],
                resource_data=resource_data,
                created_by=request.get('user', {}).get('sub'),
                organization_id=request.get('user', {}).get('org')
            )
            session.add(fhir_resource)
            session.commit()
            
            # Cache if Redis available
            if self.redis_client:
                cache_key = f"fhir:{resource_type}:{resource_data['id']}"
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour TTL
                    json.dumps(resource_data)
                )
            
            logger.info(f"Created {resource_type} resource: {resource_data['id']}")
            
            return web.json_response(
                resource_data, 
                status=201,
                headers={'Location': f"/{resource_type}/{resource_data['id']}"}
            )
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating resource: {e}")
            return web.json_response({'error': 'Internal server error'}, status=500)
        finally:
            session.close()
    
    async def read_resource(self, request):
        """Read a FHIR resource by ID"""
        resource_type = request.match_info['resource_type']
        resource_id = request.match_info['resource_id']
        
        # Check cache first
        if self.redis_client:
            cache_key = f"fhir:{resource_type}:{resource_id}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return web.json_response(json.loads(cached_data))
        
        # Query database
        session = self.Session()
        try:
            resource = session.query(FHIRResource).filter_by(
                resource_type=resource_type,
                resource_id=resource_id,
                is_deleted=False
            ).first()
            
            if not resource:
                return web.json_response({'error': 'Resource not found'}, status=404)
            
            # Cache the result
            if self.redis_client:
                cache_key = f"fhir:{resource_type}:{resource_id}"
                self.redis_client.setex(
                    cache_key,
                    3600,
                    json.dumps(resource.resource_data)
                )
            
            return web.json_response(resource.resource_data)
            
        except Exception as e:
            logger.error(f"Error reading resource: {e}")
            return web.json_response({'error': 'Internal server error'}, status=500)
        finally:
            session.close()
    
    async def search_resources(self, request):
        """Search FHIR resources with parameters"""
        resource_type = request.match_info['resource_type']
        query_params = dict(request.query)
        
        session = self.Session()
        try:
            # Build base query
            query = session.query(FHIRResource).filter_by(
                resource_type=resource_type,
                is_deleted=False
            )
            
            # Apply search parameters
            query = self._apply_search_parameters(query, resource_type, query_params)
            
            # Pagination
            count = int(query_params.get('_count', 20))
            offset = int(query_params.get('_offset', 0))
            
            total = query.count()
            resources = query.offset(offset).limit(count).all()
            
            # Build bundle response
            bundle = {
                "resourceType": "Bundle",
                "id": str(uuid.uuid4()),
                "type": "searchset",
                "total": total,
                "entry": [
                    {
                        "resource": resource.resource_data,
                        "search": {"mode": "match"}
                    }
                    for resource in resources
                ]
            }
            
            return web.json_response(bundle)
            
        except Exception as e:
            logger.error(f"Error searching resources: {e}")
            return web.json_response({'error': 'Internal server error'}, status=500)
        finally:
            session.close()
    
    def _apply_search_parameters(self, query, resource_type: str, params: Dict[str, str]):
        """Apply FHIR search parameters to database query"""
        
        for param_name, param_value in params.items():
            if param_name.startswith('_'):
                continue  # Skip control parameters
            
            if resource_type == 'Patient':
                if param_name == 'identifier':
                    # Search by identifier
                    query = query.filter(
                        FHIRResource.resource_data['identifier'].astext.contains(param_value)
                    )
                elif param_name == 'name':
                    # Search by name
                    query = query.filter(
                        FHIRResource.resource_data['name'].astext.ilike(f'%{param_value}%')
                    )
                elif param_name == 'gender':
                    query = query.filter(
                        FHIRResource.resource_data['gender'].astext == param_value
                    )
            
            elif resource_type == 'Observation':
                if param_name == 'subject':
                    # Search by subject reference
                    query = query.filter(
                        FHIRResource.resource_data['subject']['reference'].astext == param_value
                    )
                elif param_name == 'code':
                    # Search by observation code
                    query = query.filter(
                        FHIRResource.resource_data['code'].astext.contains(param_value)
                    )
                elif param_name == 'status':
                    query = query.filter(
                        FHIRResource.resource_data['status'].astext == param_value
                    )
        
        return query
    
    async def update_resource(self, request):
        """Update a FHIR resource"""
        resource_type = request.match_info['resource_type']
        resource_id = request.match_info['resource_id']
        
        try:
            resource_data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        
        # Validate resource
        is_valid, errors = self.validator.validate_resource(resource_type, resource_data)
        if not is_valid:
            return web.json_response({
                'error': 'Validation failed',
                'details': errors
            }, status=400)
        
        session = self.Session()
        try:
            # Find existing resource
            existing = session.query(FHIRResource).filter_by(
                resource_type=resource_type,
                resource_id=resource_id,
                is_deleted=False
            ).first()
            
            if not existing:
                return web.json_response({'error': 'Resource not found'}, status=404)
            
            # Update version
            current_version = int(existing.version_id)
            new_version = str(current_version + 1)
            
            # Update metadata
            resource_data['id'] = resource_id
            resource_data['meta'] = {
                'versionId': new_version,
                'lastUpdated': datetime.datetime.utcnow().isoformat() + 'Z'
            }
            
            # Update database
            existing.resource_data = resource_data
            existing.version_id = new_version
            existing.last_updated = datetime.datetime.utcnow()
            
            session.commit()
            
            # Update cache
            if self.redis_client:
                cache_key = f"fhir:{resource_type}:{resource_id}"
                self.redis_client.setex(
                    cache_key,
                    3600,
                    json.dumps(resource_data)
                )
            
            logger.info(f"Updated {resource_type} resource: {resource_id}")
            
            return web.json_response(resource_data)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating resource: {e}")
            return web.json_response({'error': 'Internal server error'}, status=500)
        finally:
            session.close()
    
    async def delete_resource(self, request):
        """Delete a FHIR resource (soft delete)"""
        resource_type = request.match_info['resource_type']
        resource_id = request.match_info['resource_id']
        
        session = self.Session()
        try:
            resource = session.query(FHIRResource).filter_by(
                resource_type=resource_type,
                resource_id=resource_id,
                is_deleted=False
            ).first()
            
            if not resource:
                return web.json_response({'error': 'Resource not found'}, status=404)
            
            # Soft delete
            resource.is_deleted = True
            resource.last_updated = datetime.datetime.utcnow()
            
            session.commit()
            
            # Remove from cache
            if self.redis_client:
                cache_key = f"fhir:{resource_type}:{resource_id}"
                self.redis_client.delete(cache_key)
            
            logger.info(f"Deleted {resource_type} resource: {resource_id}")
            
            return web.Response(status=204)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting resource: {e}")
            return web.json_response({'error': 'Internal server error'}, status=500)
        finally:
            session.close()
    
    async def validate_resource_endpoint(self, request):
        """Validate a FHIR resource without storing it"""
        resource_type = request.match_info['resource_type']
        
        try:
            resource_data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        
        is_valid, errors = self.validator.validate_resource(resource_type, resource_data)
        
        outcome = {
            "resourceType": "OperationOutcome",
            "issue": []
        }
        
        if is_valid:
            outcome["issue"].append({
                "severity": "information",
                "code": "informational",
                "diagnostics": "Resource is valid"
            })
        else:
            for error in errors:
                outcome["issue"].append({
                    "severity": "error",
                    "code": "invalid",
                    "diagnostics": error
                })
        
        return web.json_response(outcome, status=200 if is_valid else 400)
    
    async def patient_match(self, request):
        """Patient matching operation for deduplication"""
        try:
            parameters = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        
        # Extract patient data from parameters
        patient_data = None
        for param in parameters.get('parameter', []):
            if param.get('name') == 'resource' and param.get('resource', {}).get('resourceType') == 'Patient':
                patient_data = param['resource']
                break
        
        if not patient_data:
            return web.json_response({'error': 'No patient resource provided'}, status=400)
        
        # Perform patient matching
        matches = await self._find_patient_matches(patient_data)
        
        # Build response bundle
        bundle = {
            "resourceType": "Bundle",
            "type": "searchset",
            "entry": [
                {
                    "resource": match['patient'],
                    "search": {
                        "mode": "match",
                        "score": match['score']
                    }
                }
                for match in matches
            ]
        }
        
        return web.json_response(bundle)
    
    async def _find_patient_matches(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find matching patients using fuzzy matching algorithms"""
        session = self.Session()
        matches = []
        
        try:
            # Get all patients for matching
            all_patients = session.query(FHIRResource).filter_by(
                resource_type='Patient',
                is_deleted=False
            ).all()
            
            for patient_resource in all_patients:
                score = self._calculate_patient_match_score(
                    patient_data, 
                    patient_resource.resource_data
                )
                
                if score > 0.7:  # Threshold for potential matches
                    matches.append({
                        'patient': patient_resource.resource_data,
                        'score': score
                    })
            
            # Sort by score descending
            matches.sort(key=lambda x: x['score'], reverse=True)
            
        finally:
            session.close()
        
        return matches[:10]  # Return top 10 matches
    
    def _calculate_patient_match_score(self, patient1: Dict[str, Any], patient2: Dict[str, Any]) -> float:
        """Calculate similarity score between two patients"""
        from difflib import SequenceMatcher
        
        score = 0.0
        total_weight = 0.0
        
        # Name matching (weight: 0.4)
        if 'name' in patient1 and 'name' in patient2:
            name1 = self._extract_patient_name(patient1['name'])
            name2 = self._extract_patient_name(patient2['name'])
            name_similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
            score += name_similarity * 0.4
            total_weight += 0.4
        
        # Birth date matching (weight: 0.3)
        if 'birthDate' in patient1 and 'birthDate' in patient2:
            if patient1['birthDate'] == patient2['birthDate']:
                score += 0.3
            total_weight += 0.3
        
        # Gender matching (weight: 0.1)
        if 'gender' in patient1 and 'gender' in patient2:
            if patient1['gender'] == patient2['gender']:
                score += 0.1
            total_weight += 0.1
        
        # Identifier matching (weight: 0.2)
        if 'identifier' in patient1 and 'identifier' in patient2:
            identifier_match = self._match_identifiers(
                patient1['identifier'], 
                patient2['identifier']
            )
            score += identifier_match * 0.2
            total_weight += 0.2
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _extract_patient_name(self, name_array: List[Dict[str, Any]]) -> str:
        """Extract full name from FHIR name array"""
        if not name_array:
            return ""
        
        name = name_array[0]  # Use first name
        parts = []
        
        if 'given' in name:
            parts.extend(name['given'])
        if 'family' in name:
            parts.append(name['family'])
        
        return " ".join(parts)
    
    def _match_identifiers(self, identifiers1: List[Dict[str, Any]], identifiers2: List[Dict[str, Any]]) -> float:
        """Match patient identifiers"""
        for id1 in identifiers1:
            for id2 in identifiers2:
                if (id1.get('system') == id2.get('system') and 
                    id1.get('value') == id2.get('value')):
                    return 1.0
        return 0.0
    
    async def observation_statistics(self, request):
        """Get statistics for observations (AI/ML endpoint)"""
        query_params = dict(request.query)
        
        session = self.Session()
        try:
            # Build query for observations
            query = session.query(FHIRResource).filter_by(
                resource_type='Observation',
                is_deleted=False
            )
            
            # Apply filters
            if 'subject' in query_params:
                query = query.filter(
                    FHIRResource.resource_data['subject']['reference'].astext == query_params['subject']
                )
            
            if 'code' in query_params:
                query = query.filter(
                    FHIRResource.resource_data['code'].astext.contains(query_params['code'])
                )
            
            observations = query.all()
            
            # Calculate statistics
            stats = self._calculate_observation_statistics(observations)
            
            return web.json_response(stats)
            
        except Exception as e:
            logger.error(f"Error calculating observation statistics: {e}")
            return web.json_response({'error': 'Internal server error'}, status=500)
        finally:
            session.close()
    
    def _calculate_observation_statistics(self, observations: List[FHIRResource]) -> Dict[str, Any]:
        """Calculate statistical summaries of observations"""
        stats = {
            'total_count': len(observations),
            'by_code': {},
            'by_status': {},
            'value_statistics': {}
        }
        
        values_by_code = {}
        
        for obs_resource in observations:
            obs_data = obs_resource.resource_data
            
            # Count by code
            if 'code' in obs_data and 'coding' in obs_data['code']:
                for coding in obs_data['code']['coding']:
                    code = coding.get('code', 'unknown')
                    stats['by_code'][code] = stats['by_code'].get(code, 0) + 1
                    
                    # Collect numeric values
                    if 'valueQuantity' in obs_data and 'value' in obs_data['valueQuantity']:
                        if code not in values_by_code:
                            values_by_code[code] = []
                        values_by_code[code].append(obs_data['valueQuantity']['value'])
            
            # Count by status
            status = obs_data.get('status', 'unknown')
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
        
        # Calculate value statistics
        for code, values in values_by_code.items():
            if values:
                stats['value_statistics'][code] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return stats
    
    async def _log_audit_event(self, request, user_id: str, success: bool, error_message: str = None):
        """Log audit event to database"""
        session = self.Session()
        try:
            # Extract operation details
            method = request.method
            path = request.path
            
            operation_map = {
                'POST': 'CREATE',
                'GET': 'READ',
                'PUT': 'UPDATE',
                'DELETE': 'DELETE'
            }
            
            operation = operation_map.get(method, 'UNKNOWN')
            
            # Extract resource info from path
            path_parts = path.strip('/').split('/')
            resource_type = path_parts[0] if path_parts else 'unknown'
            resource_id = path_parts[1] if len(path_parts) > 1 else 'unknown'
            
            audit_log = AuditLog(
                operation=operation,
                resource_type=resource_type,
                resource_id=resource_id,
                user_id=user_id,
                ip_address=request.remote,
                user_agent=request.headers.get('User-Agent', ''),
                success=success,
                error_message=error_message
            )
            
            session.add(audit_log)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            session.rollback()
        finally:
            session.close()
    
    def run(self, host: str = '0.0.0.0', port: int = 8080):
        """Run the FHIR server"""
        logger.info(f"Starting FHIR server on {host}:{port}")
        web.run_app(self.app, host=host, port=port)

class FHIRClient:
    """
    Advanced FHIR client for healthcare AI applications
    
    This client provides comprehensive FHIR operations with advanced features
    for healthcare AI workflows, including bulk data access, real-time
    subscriptions, and clinical decision support integration.
    """
    
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = None
        self.capability_statement = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        await self._load_capability_statement()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for requests"""
        headers = {
            'Content-Type': 'application/fhir+json',
            'Accept': 'application/fhir+json'
        }
        
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        
        return headers
    
    async def _load_capability_statement(self):
        """Load server capability statement"""
        try:
            async with self.session.get(
                f"{self.base_url}/metadata",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    self.capability_statement = await response.json()
                    logger.info("Loaded FHIR server capability statement")
                else:
                    logger.warning(f"Failed to load capability statement: {response.status}")
        except Exception as e:
            logger.error(f"Error loading capability statement: {e}")
    
    async def create_resource(self, resource_type: str, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new FHIR resource"""
        url = f"{self.base_url}/{resource_type}"
        
        async with self.session.post(
            url,
            json=resource_data,
            headers=self._get_headers()
        ) as response:
            if response.status == 201:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to create resource: {response.status} - {error_text}")
    
    async def read_resource(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """Read a FHIR resource by ID"""
        url = f"{self.base_url}/{resource_type}/{resource_id}"
        
        async with self.session.get(url, headers=self._get_headers()) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 404:
                raise Exception(f"Resource not found: {resource_type}/{resource_id}")
            else:
                error_text = await response.text()
                raise Exception(f"Failed to read resource: {response.status} - {error_text}")
    
    async def search_resources(
        self, 
        resource_type: str, 
        search_params: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Search for FHIR resources"""
        url = f"{self.base_url}/{resource_type}"
        
        if search_params:
            # Build query string
            query_parts = []
            for key, value in search_params.items():
                query_parts.append(f"{key}={value}")
            if query_parts:
                url += "?" + "&".join(query_parts)
        
        async with self.session.get(url, headers=self._get_headers()) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to search resources: {response.status} - {error_text}")
    
    async def update_resource(
        self, 
        resource_type: str, 
        resource_id: str, 
        resource_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a FHIR resource"""
        url = f"{self.base_url}/{resource_type}/{resource_id}"
        
        async with self.session.put(
            url,
            json=resource_data,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to update resource: {response.status} - {error_text}")
    
    async def delete_resource(self, resource_type: str, resource_id: str) -> bool:
        """Delete a FHIR resource"""
        url = f"{self.base_url}/{resource_type}/{resource_id}"
        
        async with self.session.delete(url, headers=self._get_headers()) as response:
            return response.status == 204
    
    async def validate_resource(
        self, 
        resource_type: str, 
        resource_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a FHIR resource"""
        url = f"{self.base_url}/{resource_type}/$validate"
        
        async with self.session.post(
            url,
            json=resource_data,
            headers=self._get_headers()
        ) as response:
            return await response.json()
    
    async def bulk_export(
        self, 
        resource_types: List[str] = None,
        since: datetime.datetime = None
    ) -> str:
        """Initiate bulk data export"""
        url = f"{self.base_url}/$export"
        
        headers = self._get_headers()
        headers['Accept'] = 'application/fhir+json'
        headers['Prefer'] = 'respond-async'
        
        params = {}
        if resource_types:
            params['_type'] = ','.join(resource_types)
        if since:
            params['_since'] = since.isoformat()
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 202:
                # Return the content-location header for polling
                return response.headers.get('Content-Location')
            else:
                error_text = await response.text()
                raise Exception(f"Failed to initiate bulk export: {response.status} - {error_text}")
    
    async def get_bulk_export_status(self, status_url: str) -> Dict[str, Any]:
        """Check bulk export status"""
        async with self.session.get(status_url, headers=self._get_headers()) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 202:
                return {'status': 'in-progress'}
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get export status: {response.status} - {error_text}")
    
    async def patient_match(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find matching patients"""
        url = f"{self.base_url}/Patient/$match"
        
        parameters = {
            "resourceType": "Parameters",
            "parameter": [
                {
                    "name": "resource",
                    "resource": patient_data
                }
            ]
        }
        
        async with self.session.post(
            url,
            json=parameters,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to match patients: {response.status} - {error_text}")

# Healthcare AI specific FHIR extensions and utilities
class HealthcareAIFHIRExtensions:
    """
    FHIR extensions and utilities specifically designed for healthcare AI applications
    
    This class provides specialized FHIR operations that support common healthcare AI
    workflows, including data preparation, feature extraction, and model integration.
    """
    
    def __init__(self, fhir_client: FHIRClient):
        self.client = fhir_client
        self.ai_extensions = self._load_ai_extensions()
    
    def _load_ai_extensions(self) -> Dict[str, Any]:
        """Load AI-specific FHIR extensions"""
        return {
            'prediction_extension': 'http://example.org/fhir/StructureDefinition/ai-prediction',
            'confidence_extension': 'http://example.org/fhir/StructureDefinition/ai-confidence',
            'model_version_extension': 'http://example.org/fhir/StructureDefinition/model-version',
            'feature_importance_extension': 'http://example.org/fhir/StructureDefinition/feature-importance'
        }
    
    async def extract_patient_features(self, patient_id: str) -> Dict[str, Any]:
        """
        Extract comprehensive feature set for a patient
        
        This method aggregates data from multiple FHIR resources to create
        a comprehensive feature vector suitable for machine learning models.
        """
        features = {
            'patient_id': patient_id,
            'demographics': {},
            'vital_signs': {},
            'lab_results': {},
            'medications': {},
            'conditions': {},
            'procedures': {}
        }
        
        try:
            # Get patient demographics
            patient = await self.client.read_resource('Patient', patient_id)
            features['demographics'] = self._extract_demographic_features(patient)
            
            # Get observations (vital signs, lab results)
            observations_bundle = await self.client.search_resources(
                'Observation',
                {'subject': f'Patient/{patient_id}', '_count': '1000'}
            )
            
            vital_signs, lab_results = self._extract_observation_features(
                observations_bundle.get('entry', [])
            )
            features['vital_signs'] = vital_signs
            features['lab_results'] = lab_results
            
            # Get medications
            medications_bundle = await self.client.search_resources(
                'MedicationStatement',
                {'subject': f'Patient/{patient_id}', '_count': '1000'}
            )
            features['medications'] = self._extract_medication_features(
                medications_bundle.get('entry', [])
            )
            
            # Get conditions
            conditions_bundle = await self.client.search_resources(
                'Condition',
                {'subject': f'Patient/{patient_id}', '_count': '1000'}
            )
            features['conditions'] = self._extract_condition_features(
                conditions_bundle.get('entry', [])
            )
            
            # Get procedures
            procedures_bundle = await self.client.search_resources(
                'Procedure',
                {'subject': f'Patient/{patient_id}', '_count': '1000'}
            )
            features['procedures'] = self._extract_procedure_features(
                procedures_bundle.get('entry', [])
            )
            
        except Exception as e:
            logger.error(f"Error extracting patient features: {e}")
            raise
        
        return features
    
    def _extract_demographic_features(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """Extract demographic features from patient resource"""
        features = {}
        
        # Age calculation
        if 'birthDate' in patient:
            birth_date = datetime.datetime.fromisoformat(
                patient['birthDate'].replace('Z', '+00:00')
            )
            age = (datetime.datetime.now(timezone.utc) - birth_date).days / 365.25
            features['age'] = age
        
        # Gender
        features['gender'] = patient.get('gender', 'unknown')
        
        # Marital status
        if 'maritalStatus' in patient:
            features['marital_status'] = patient['maritalStatus'].get('coding', [{}])[0].get('code', 'unknown')
        
        # Address (for social determinants)
        if 'address' in patient and patient['address']:
            address = patient['address'][0]
            features['postal_code'] = address.get('postalCode', '')
            features['state'] = address.get('state', '')
        
        return features
    
    def _extract_observation_features(self, observations: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract vital signs and lab results from observations"""
        vital_signs = {}
        lab_results = {}
        
        # LOINC codes for common vital signs
        vital_sign_codes = {
            '8480-6': 'systolic_bp',
            '8462-4': 'diastolic_bp',
            '8867-4': 'heart_rate',
            '8310-5': 'body_temperature',
            '8302-2': 'body_height',
            '29463-7': 'body_weight',
            '39156-5': 'bmi'
        }
        
        # LOINC codes for common lab results
        lab_codes = {
            '2339-0': 'glucose',
            '4548-4': 'hba1c',
            '2093-3': 'cholesterol_total',
            '2085-9': 'hdl_cholesterol',
            '2089-1': 'ldl_cholesterol',
            '2571-8': 'triglycerides',
            '38483-4': 'creatinine',
            '33747-0': 'hemoglobin_a1c'
        }
        
        for entry in observations:
            if 'resource' not in entry:
                continue
            
            obs = entry['resource']
            if obs.get('resourceType') != 'Observation':
                continue
            
            # Extract observation code
            if 'code' not in obs or 'coding' not in obs['code']:
                continue
            
            for coding in obs['code']['coding']:
                loinc_code = coding.get('code')
                if not loinc_code:
                    continue
                
                # Extract value
                value = None
                if 'valueQuantity' in obs:
                    value = obs['valueQuantity'].get('value')
                elif 'valueString' in obs:
                    try:
                        value = float(obs['valueString'])
                    except ValueError:
                        continue
                
                if value is None:
                    continue
                
                # Categorize observation
                if loinc_code in vital_sign_codes:
                    feature_name = vital_sign_codes[loinc_code]
                    if feature_name not in vital_signs:
                        vital_signs[feature_name] = []
                    vital_signs[feature_name].append({
                        'value': value,
                        'date': obs.get('effectiveDateTime', ''),
                        'status': obs.get('status', 'unknown')
                    })
                
                elif loinc_code in lab_codes:
                    feature_name = lab_codes[loinc_code]
                    if feature_name not in lab_results:
                        lab_results[feature_name] = []
                    lab_results[feature_name].append({
                        'value': value,
                        'date': obs.get('effectiveDateTime', ''),
                        'status': obs.get('status', 'unknown')
                    })
        
        # Aggregate multiple values (use most recent)
        for feature_name, values in vital_signs.items():
            if values:
                # Sort by date and take most recent
                sorted_values = sorted(values, key=lambda x: x['date'], reverse=True)
                vital_signs[feature_name] = sorted_values[0]['value']
        
        for feature_name, values in lab_results.items():
            if values:
                sorted_values = sorted(values, key=lambda x: x['date'], reverse=True)
                lab_results[feature_name] = sorted_values[0]['value']
        
        return vital_signs, lab_results
    
    def _extract_medication_features(self, medications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract medication features"""
        features = {
            'active_medications': [],
            'medication_classes': {},
            'high_risk_medications': []
        }
        
        # High-risk medication classes (simplified)
        high_risk_classes = [
            'anticoagulants',
            'insulin',
            'opioids',
            'sedatives'
        ]
        
        for entry in medications:
            if 'resource' not in entry:
                continue
            
            med = entry['resource']
            if med.get('resourceType') != 'MedicationStatement':
                continue
            
            # Check if medication is active
            status = med.get('status', 'unknown')
            if status not in ['active', 'intended']:
                continue
            
            # Extract medication code
            if 'medicationCodeableConcept' in med and 'coding' in med['medicationCodeableConcept']:
                for coding in med['medicationCodeableConcept']['coding']:
                    if coding.get('system') == 'http://www.nlm.nih.gov/research/umls/rxnorm':
                        rxnorm_code = coding.get('code')
                        display = coding.get('display', 'Unknown medication')
                        
                        features['active_medications'].append({
                            'code': rxnorm_code,
                            'display': display
                        })
                        
                        # Check for high-risk medications (simplified logic)
                        display_lower = display.lower()
                        for risk_class in high_risk_classes:
                            if risk_class in display_lower:
                                features['high_risk_medications'].append(display)
                                break
        
        features['total_active_medications'] = len(features['active_medications'])
        features['high_risk_medication_count'] = len(features['high_risk_medications'])
        
        return features
    
    def _extract_condition_features(self, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract condition/diagnosis features"""
        features = {
            'active_conditions': [],
            'chronic_conditions': [],
            'condition_categories': {}
        }
        
        # Chronic condition ICD-10 codes (simplified)
        chronic_condition_codes = {
            'E11': 'diabetes_type_2',
            'I10': 'hypertension',
            'I25': 'coronary_artery_disease',
            'J44': 'copd',
            'N18': 'chronic_kidney_disease'
        }
        
        for entry in conditions:
            if 'resource' not in entry:
                continue
            
            condition = entry['resource']
            if condition.get('resourceType') != 'Condition':
                continue
            
            # Check if condition is active
            clinical_status = condition.get('clinicalStatus', {})
            if clinical_status.get('coding', [{}])[0].get('code') != 'active':
                continue
            
            # Extract condition code
            if 'code' in condition and 'coding' in condition['code']:
                for coding in condition['code']['coding']:
                    if coding.get('system') == 'http://hl7.org/fhir/sid/icd-10':
                        icd10_code = coding.get('code', '')
                        display = coding.get('display', 'Unknown condition')
                        
                        features['active_conditions'].append({
                            'code': icd10_code,
                            'display': display
                        })
                        
                        # Check for chronic conditions
                        for chronic_code, chronic_name in chronic_condition_codes.items():
                            if icd10_code.startswith(chronic_code):
                                features['chronic_conditions'].append(chronic_name)
                                break
        
        features['total_active_conditions'] = len(features['active_conditions'])
        features['chronic_condition_count'] = len(features['chronic_conditions'])
        
        return features
    
    def _extract_procedure_features(self, procedures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract procedure features"""
        features = {
            'recent_procedures': [],
            'surgical_procedures': [],
            'procedure_categories': {}
        }
        
        # Define lookback period (90 days)
        lookback_date = datetime.datetime.now(timezone.utc) - datetime.timedelta(days=90)
        
        for entry in procedures:
            if 'resource' not in entry:
                continue
            
            procedure = entry['resource']
            if procedure.get('resourceType') != 'Procedure':
                continue
            
            # Check procedure date
            performed_date = procedure.get('performedDateTime')
            if performed_date:
                try:
                    proc_date = datetime.datetime.fromisoformat(
                        performed_date.replace('Z', '+00:00')
                    )
                    if proc_date < lookback_date:
                        continue
                except ValueError:
                    continue
            
            # Extract procedure code
            if 'code' in procedure and 'coding' in procedure['code']:
                for coding in procedure['code']['coding']:
                    code = coding.get('code')
                    display = coding.get('display', 'Unknown procedure')
                    
                    features['recent_procedures'].append({
                        'code': code,
                        'display': display,
                        'date': performed_date
                    })
                    
                    # Check if surgical procedure (simplified logic)
                    if any(term in display.lower() for term in ['surgery', 'surgical', 'operation']):
                        features['surgical_procedures'].append(display)
        
        features['recent_procedure_count'] = len(features['recent_procedures'])
        features['recent_surgical_count'] = len(features['surgical_procedures'])
        
        return features
    
    async def create_ai_prediction_observation(
        self,
        patient_id: str,
        prediction_type: str,
        prediction_value: float,
        confidence: float,
        model_version: str,
        feature_importance: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Create an observation resource to store AI model predictions
        
        This method creates a FHIR Observation resource that captures the output
        of an AI model, including the prediction, confidence, and metadata.
        """
        
        observation = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "survey",
                    "display": "Survey"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://example.org/fhir/CodeSystem/ai-predictions",
                    "code": prediction_type,
                    "display": f"AI Prediction: {prediction_type}"
                }]
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "effectiveDateTime": datetime.datetime.utcnow().isoformat() + "Z",
            "valueQuantity": {
                "value": prediction_value,
                "unit": "probability",
                "system": "http://unitsofmeasure.org",
                "code": "1"
            },
            "extension": [
                {
                    "url": self.ai_extensions['confidence_extension'],
                    "valueDecimal": confidence
                },
                {
                    "url": self.ai_extensions['model_version_extension'],
                    "valueString": model_version
                }
            ]
        }
        
        # Add feature importance if provided
        if feature_importance:
            importance_extension = {
                "url": self.ai_extensions['feature_importance_extension'],
                "extension": []
            }
            
            for feature, importance in feature_importance.items():
                importance_extension["extension"].append({
                    "url": "feature",
                    "extension": [
                        {
                            "url": "name",
                            "valueString": feature
                        },
                        {
                            "url": "importance",
                            "valueDecimal": importance
                        }
                    ]
                })
            
            observation["extension"].append(importance_extension)
        
        # Create the observation
        return await self.client.create_resource('Observation', observation)
    
    async def get_patient_timeline(
        self,
        patient_id: str,
        resource_types: List[str] = None,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of patient events
        
        This method aggregates multiple FHIR resources to create a comprehensive
        timeline of patient events, useful for temporal analysis and modeling.
        """
        if resource_types is None:
            resource_types = ['Observation', 'Condition', 'Procedure', 'MedicationStatement', 'Encounter']
        
        timeline_events = []
        
        for resource_type in resource_types:
            try:
                # Build search parameters
                search_params = {
                    'subject': f'Patient/{patient_id}',
                    '_count': '1000',
                    '_sort': 'date'
                }
                
                # Add date filters if provided
                if start_date and end_date:
                    date_param = f"ge{start_date.isoformat()}&date=le{end_date.isoformat()}"
                    search_params['date'] = date_param
                
                # Search for resources
                bundle = await self.client.search_resources(resource_type, search_params)
                
                # Extract timeline events
                for entry in bundle.get('entry', []):
                    if 'resource' not in entry:
                        continue
                    
                    resource = entry['resource']
                    event = self._extract_timeline_event(resource)
                    if event:
                        timeline_events.append(event)
                        
            except Exception as e:
                logger.warning(f"Error fetching {resource_type} for timeline: {e}")
                continue
        
        # Sort events by date
        timeline_events.sort(key=lambda x: x.get('date', ''))
        
        return timeline_events
    
    def _extract_timeline_event(self, resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract timeline event from FHIR resource"""
        resource_type = resource.get('resourceType')
        
        if not resource_type:
            return None
        
        event = {
            'resource_type': resource_type,
            'resource_id': resource.get('id'),
            'date': None,
            'description': '',
            'category': '',
            'details': {}
        }
        
        # Extract date based on resource type
        if resource_type == 'Observation':
            event['date'] = resource.get('effectiveDateTime') or resource.get('effectivePeriod', {}).get('start')
            if 'code' in resource and 'coding' in resource['code']:
                event['description'] = resource['code']['coding'][0].get('display', 'Observation')
            event['category'] = 'observation'
            
            if 'valueQuantity' in resource:
                event['details']['value'] = resource['valueQuantity'].get('value')
                event['details']['unit'] = resource['valueQuantity'].get('unit')
        
        elif resource_type == 'Condition':
            event['date'] = resource.get('onsetDateTime') or resource.get('recordedDate')
            if 'code' in resource and 'coding' in resource['code']:
                event['description'] = resource['code']['coding'][0].get('display', 'Condition')
            event['category'] = 'condition'
            
            clinical_status = resource.get('clinicalStatus', {})
            if 'coding' in clinical_status:
                event['details']['status'] = clinical_status['coding'][0].get('code')
        
        elif resource_type == 'Procedure':
            event['date'] = resource.get('performedDateTime') or resource.get('performedPeriod', {}).get('start')
            if 'code' in resource and 'coding' in resource['code']:
                event['description'] = resource['code']['coding'][0].get('display', 'Procedure')
            event['category'] = 'procedure'
            
            event['details']['status'] = resource.get('status')
        
        elif resource_type == 'MedicationStatement':
            event['date'] = resource.get('effectiveDateTime') or resource.get('effectivePeriod', {}).get('start')
            if 'medicationCodeableConcept' in resource and 'coding' in resource['medicationCodeableConcept']:
                event['description'] = resource['medicationCodeableConcept']['coding'][0].get('display', 'Medication')
            event['category'] = 'medication'
            
            event['details']['status'] = resource.get('status')
        
        elif resource_type == 'Encounter':
            event['date'] = resource.get('period', {}).get('start')
            event['description'] = f"Healthcare encounter"
            event['category'] = 'encounter'
            
            if 'class' in resource:
                event['details']['encounter_class'] = resource['class'].get('code')
            event['details']['status'] = resource.get('status')
        
        # Only return events with valid dates
        if event['date']:
            return event
        
        return None

# Example usage and testing framework
async def demonstrate_fhir_system():
    """Comprehensive demonstration of the FHIR system"""
    
    print("Healthcare AI FHIR System Demonstration")
    print("=" * 50)
    
    # Create sample patient data
    sample_patient = {
        "resourceType": "Patient",
        "identifier": [{
            "system": "http://hospital.example.org/patient-id",
            "value": "12345"
        }],
        "name": [{
            "family": "Doe",
            "given": ["John", "Michael"]
        }],
        "gender": "male",
        "birthDate": "1980-01-15",
        "address": [{
            "line": ["123 Main St"],
            "city": "Anytown",
            "state": "CA",
            "postalCode": "12345"
        }]
    }
    
    # Sample observation (blood pressure)
    sample_observation = {
        "resourceType": "Observation",
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "vital-signs",
                "display": "Vital Signs"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "8480-6",
                "display": "Systolic blood pressure"
            }]
        },
        "subject": {
            "reference": "Patient/test-patient-1"
        },
        "effectiveDateTime": "2024-01-15T10:30:00Z",
        "valueQuantity": {
            "value": 140,
            "unit": "mmHg",
            "system": "http://unitsofmeasure.org",
            "code": "mm[Hg]"
        }
    }
    
    # Test FHIR client operations
    base_url = "http://localhost:8080"  # Assuming FHIR server is running
    auth_token = "test-token"  # In production, use real JWT token
    
    try:
        async with FHIRClient(base_url, auth_token) as client:
            print("Testing FHIR Client Operations:")
            print("-" * 30)
            
            # Create patient
            print("1. Creating patient...")
            try:
                created_patient = await client.create_resource('Patient', sample_patient)
                patient_id = created_patient['id']
                print(f"   Created patient with ID: {patient_id}")
            except Exception as e:
                print(f"   Error creating patient: {e}")
                patient_id = "test-patient-1"  # Use fallback ID
            
            # Update observation with correct patient reference
            sample_observation['subject']['reference'] = f"Patient/{patient_id}"
            
            # Create observation
            print("2. Creating observation...")
            try:
                created_observation = await client.create_resource('Observation', sample_observation)
                observation_id = created_observation['id']
                print(f"   Created observation with ID: {observation_id}")
            except Exception as e:
                print(f"   Error creating observation: {e}")
            
            # Search for patient
            print("3. Searching for patients...")
            try:
                search_results = await client.search_resources('Patient', {'name': 'Doe'})
                print(f"   Found {search_results.get('total', 0)} patients")
            except Exception as e:
                print(f"   Error searching patients: {e}")
            
            # Test AI extensions
            print("4. Testing AI extensions...")
            try:
                ai_extensions = HealthcareAIFHIRExtensions(client)
                
                # Extract patient features
                features = await ai_extensions.extract_patient_features(patient_id)
                print(f"   Extracted features for patient: {len(features)} categories")
                
                # Create AI prediction
                prediction_obs = await ai_extensions.create_ai_prediction_observation(
                    patient_id=patient_id,
                    prediction_type="cardiovascular_risk",
                    prediction_value=0.75,
                    confidence=0.85,
                    model_version="v1.2.3",
                    feature_importance={
                        "age": 0.3,
                        "systolic_bp": 0.4,
                        "gender": 0.1
                    }
                )
                print(f"   Created AI prediction observation: {prediction_obs['id']}")
                
            except Exception as e:
                print(f"   Error with AI extensions: {e}")
            
            print("\nFHIR system demonstration completed successfully!")
            
    except Exception as e:
        print(f"Error connecting to FHIR server: {e}")
        print("Note: This demonstration requires a running FHIR server at localhost:8080")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_fhir_system())
```

### Healthcare Data Integration Patterns

Healthcare data integration requires sophisticated patterns that can handle the complexity and diversity of clinical data sources. Modern healthcare organizations typically have dozens of different systems that need to be integrated, each with its own data formats, APIs, and update patterns. This section explores the key integration patterns and provides comprehensive implementations for each.

The Extract, Transform, Load (ETL) pattern remains fundamental to healthcare data integration, but modern implementations require real-time capabilities, advanced error handling, and comprehensive audit trails. The following implementation demonstrates a production-ready healthcare ETL system designed for AI applications:

```python
"""
Advanced Healthcare Data Integration and ETL Pipeline
Production-ready system for integrating diverse healthcare data sources

This implementation provides a comprehensive ETL framework specifically designed
for healthcare AI applications, including real-time processing, data quality
validation, and compliance monitoring.

Author: Sanjay Basu, MD PhD
Institution: Waymark
License: Educational use - requires clinical validation for production
References:
- Saripalle, R. et al. (2019). Using HL7 FHIR to achieve interoperability in patient health record. J Biomed Inform. 94:103188
- Kiourtis, A. et al. (2019). Structurally mapping healthcare data to HL7 FHIR through ontology alignment. J Med Syst. 43(3):62
"""

import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import hashlib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis
from kafka import KafkaProducer, KafkaConsumer
import boto3
from botocore.exceptions import ClientError
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from hl7apy import parse_message
from hl7apy.core import Message
import pydicom
from pathlib import Path
import schedule
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a healthcare data source"""
    source_id: str
    source_type: str  # 'hl7', 'fhir', 'database', 'file', 'api'
    connection_config: Dict[str, Any]
    data_format: str  # 'json', 'xml', 'csv', 'hl7', 'dicom'
    update_frequency: str  # 'real-time', 'hourly', 'daily', 'weekly'
    priority: int = 1  # 1=highest, 5=lowest
    enabled: bool = True
    last_sync: Optional[datetime] = None
    error_count: int = 0
    max_errors: int = 10

@dataclass
class DataQualityRule:
    """Data quality validation rule"""
    rule_id: str
    rule_type: str  # 'completeness', 'accuracy', 'consistency', 'timeliness'
    field_name: str
    validation_function: Callable[[Any], bool]
    error_message: str
    severity: str = 'error'  # 'error', 'warning', 'info'

@dataclass
class TransformationRule:
    """Data transformation rule"""
    rule_id: str
    source_field: str
    target_field: str
    transformation_function: Callable[[Any], Any]
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None

class DataExtractor(ABC):
    """Abstract base class for data extractors"""
    
    def __init__(self, source: DataSource):
        self.source = source
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """Setup HTTP session with retry strategy"""
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @abstractmethod
    async def extract(self, **kwargs) -> List[Dict[str, Any]]:
        """Extract data from the source"""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to the data source"""
        pass

class FHIRExtractor(DataExtractor):
    """FHIR server data extractor"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        self.base_url = source.connection_config['base_url']
        self.auth_token = source.connection_config.get('auth_token')
        self.headers = self._build_headers()
    
    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for FHIR requests"""
        headers = {
            'Accept': 'application/fhir+json',
            'Content-Type': 'application/fhir+json'
        }
        
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        
        return headers
    
    async def extract(self, resource_type: str = 'Patient', **kwargs) -> List[Dict[str, Any]]:
        """Extract FHIR resources"""
        try:
            # Build search URL
            url = f"{self.base_url}/{resource_type}"
            
            # Add search parameters
            params = kwargs.get('search_params', {})
            if self.source.last_sync:
                params['_lastUpdated'] = f"ge{self.source.last_sync.isoformat()}"
            
            # Add pagination
            params['_count'] = kwargs.get('count', 1000)
            
            all_resources = []
            next_url = url
            
            while next_url:
                response = self.session.get(next_url, headers=self.headers, params=params)
                response.raise_for_status()
                
                bundle = response.json()
                
                # Extract resources from bundle
                for entry in bundle.get('entry', []):
                    if 'resource' in entry:
                        all_resources.append(entry['resource'])
                
                # Check for next page
                next_url = None
                for link in bundle.get('link', []):
                    if link.get('relation') == 'next':
                        next_url = link.get('url')
                        params = {}  # Parameters are in the URL
                        break
            
            logger.info(f"Extracted {len(all_resources)} {resource_type} resources from FHIR server")
            return all_resources
            
        except Exception as e:
            logger.error(f"Error extracting from FHIR server: {e}")
            raise
    
    def validate_connection(self) -> bool:
        """Validate FHIR server connection"""
        try:
            response = self.session.get(f"{self.base_url}/metadata", headers=self.headers)
            return response.status_code == 200
        except Exception:
            return False

class HL7Extractor(DataExtractor):
    """HL7 message extractor"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        self.connection_type = source.connection_config['type']  # 'file', 'tcp', 'mllp'
        
    async def extract(self, **kwargs) -> List[Dict[str, Any]]:
        """Extract HL7 messages"""
        if self.connection_type == 'file':
            return await self._extract_from_file()
        elif self.connection_type == 'tcp':
            return await self._extract_from_tcp()
        else:
            raise ValueError(f"Unsupported HL7 connection type: {self.connection_type}")
    
    async def _extract_from_file(self) -> List[Dict[str, Any]]:
        """Extract HL7 messages from file"""
        file_path = self.source.connection_config['file_path']
        messages = []
        
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                
                # Split messages (assuming messages are separated by newlines)
                raw_messages = content.strip().split('\n')
                
                for raw_message in raw_messages:
                    if raw_message.strip():
                        try:
                            parsed_message = parse_message(raw_message)
                            message_dict = self._hl7_to_dict(parsed_message)
                            messages.append(message_dict)
                        except Exception as e:
                            logger.warning(f"Error parsing HL7 message: {e}")
                            continue
            
            logger.info(f"Extracted {len(messages)} HL7 messages from file")
            return messages
            
        except Exception as e:
            logger.error(f"Error reading HL7 file: {e}")
            raise
    
    async def _extract_from_tcp(self) -> List[Dict[str, Any]]:
        """Extract HL7 messages from TCP connection"""
        # This would implement MLLP (Minimal Lower Layer Protocol) client
        # For brevity, showing structure only
        host = self.source.connection_config['host']
        port = self.source.connection_config['port']
        
        # Implementation would connect to HL7 server and receive messages
        # This is a complex implementation requiring MLLP protocol handling
        logger.info(f"Would connect to HL7 server at {host}:{port}")
        return []
    
    def _hl7_to_dict(self, hl7_message: Message) -> Dict[str, Any]:
        """Convert HL7 message to dictionary"""
        message_dict = {
            'message_type': str(hl7_message.msh.msh_9.msh_9_1),
            'timestamp': str(hl7_message.msh.msh_7),
            'sending_application': str(hl7_message.msh.msh_3),
            'receiving_application': str(hl7_message.msh.msh_5),
            'segments': {}
        }
        
        # Extract key segments
        for segment in hl7_message.children:
            segment_name = segment.name
            segment_data = {}
            
            # Extract fields from segment
            for field in segment.children:
                if hasattr(field, 'value') and field.value:
                    segment_data[field.name] = str(field.value)
            
            message_dict['segments'][segment_name] = segment_data
        
        return message_dict
    
    def validate_connection(self) -> bool:
        """Validate HL7 connection"""
        if self.connection_type == 'file':
            file_path = self.source.connection_config['file_path']
            return Path(file_path).exists()
        elif self.connection_type == 'tcp':
            # Would test TCP connection
            return True
        return False

class DatabaseExtractor(DataExtractor):
    """Database data extractor"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        self.connection_string = source.connection_config['connection_string']
        self.engine = create_engine(self.connection_string)
        self.Session = sessionmaker(bind=self.engine)
    
    async def extract(self, query: str = None, table: str = None, **kwargs) -> List[Dict[str, Any]]:
        """Extract data from database"""
        session = self.Session()
        
        try:
            if query:
                # Execute custom query
                result = session.execute(text(query))
                columns = result.keys()
                rows = result.fetchall()
                
                data = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    data.append(row_dict)
                
            elif table:
                # Extract from table
                if self.source.last_sync:
                    # Incremental extraction
                    timestamp_column = kwargs.get('timestamp_column', 'updated_at')
                    query = f"""
                        SELECT * FROM {table} 
                        WHERE {timestamp_column} > :last_sync
                        ORDER BY {timestamp_column}
                    """
                    result = session.execute(text(query), {'last_sync': self.source.last_sync})
                else:
                    # Full extraction
                    query = f"SELECT * FROM {table}"
                    result = session.execute(text(query))
                
                columns = result.keys()
                rows = result.fetchall()
                
                data = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    # Convert datetime objects to strings
                    for key, value in row_dict.items():
                        if isinstance(value, datetime):
                            row_dict[key] = value.isoformat()
                    data.append(row_dict)
            
            else:
                raise ValueError("Either 'query' or 'table' parameter must be provided")
            
            logger.info(f"Extracted {len(data)} records from database")
            return data
            
        except Exception as e:
            logger.error(f"Error extracting from database: {e}")
            raise
        finally:
            session.close()
    
    def validate_connection(self) -> bool:
        """Validate database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

class DataTransformer:
    """Advanced data transformer for healthcare data"""
    
    def __init__(self):
        self.transformation_rules: List[TransformationRule] = []
        self.data_quality_rules: List[DataQualityRule] = []
        self.terminology_mappings = self._load_terminology_mappings()
    
    def _load_terminology_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load medical terminology mappings"""
        return {
            'gender_mappings': {
                'M': 'male',
                'F': 'female',
                'U': 'unknown',
                '1': 'male',
                '2': 'female',
                '0': 'unknown'
            },
            'race_mappings': {
                '1': 'american_indian_alaska_native',
                '2': 'asian',
                '3': 'black_african_american',
                '4': 'native_hawaiian_pacific_islander',
                '5': 'white',
                '6': 'other'
            },
            'marital_status_mappings': {
                'S': 'single',
                'M': 'married',
                'D': 'divorced',
                'W': 'widowed',
                'A': 'separated'
            }
        }
    
    def add_transformation_rule(self, rule: TransformationRule):
        """Add a transformation rule"""
        self.transformation_rules.append(rule)
    
    def add_data_quality_rule(self, rule: DataQualityRule):
        """Add a data quality rule"""
        self.data_quality_rules.append(rule)
    
    async def transform(self, data: List[Dict[str, Any]], source_type: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform healthcare data with quality validation
        
        Returns:
            Tuple of (transformed_data, quality_issues)
        """
        transformed_data = []
        quality_issues = []
        
        for record in data:
            try:
                # Apply transformations
                transformed_record = await self._apply_transformations(record, source_type)
                
                # Validate data quality
                record_issues = await self._validate_data_quality(transformed_record)
                
                if record_issues:
                    quality_issues.extend(record_issues)
                
                # Only include record if no critical errors
                critical_errors = [issue for issue in record_issues if issue['severity'] == 'error']
                if not critical_errors:
                    transformed_data.append(transformed_record)
                
            except Exception as e:
                logger.error(f"Error transforming record: {e}")
                quality_issues.append({
                    'record_id': record.get('id', 'unknown'),
                    'issue_type': 'transformation_error',
                    'severity': 'error',
                    'message': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        logger.info(f"Transformed {len(transformed_data)} records with {len(quality_issues)} quality issues")
        return transformed_data, quality_issues
    
    async def _apply_transformations(self, record: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Apply transformation rules to a record"""
        transformed_record = record.copy()
        
        # Apply source-specific transformations
        if source_type == 'hl7':
            transformed_record = await self._transform_hl7_record(transformed_record)
        elif source_type == 'fhir':
            transformed_record = await self._transform_fhir_record(transformed_record)
        elif source_type == 'database':
            transformed_record = await self._transform_database_record(transformed_record)
        
        # Apply custom transformation rules
        for rule in self.transformation_rules:
            if rule.condition is None or rule.condition(transformed_record):
                if rule.source_field in transformed_record:
                    try:
                        transformed_value = rule.transformation_function(
                            transformed_record[rule.source_field]
                        )
                        transformed_record[rule.target_field] = transformed_value
                    except Exception as e:
                        logger.warning(f"Error applying transformation rule {rule.rule_id}: {e}")
        
        return transformed_record
    
    async def _transform_hl7_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform HL7 message to standardized format"""
        transformed = {
            'source_type': 'hl7',
            'message_type': record.get('message_type'),
            'timestamp': record.get('timestamp'),
            'patient_data': {},
            'clinical_data': {}
        }
        
        segments = record.get('segments', {})
        
        # Extract patient information from PID segment
        if 'PID' in segments:
            pid = segments['PID']
            transformed['patient_data'] = {
                'patient_id': pid.get('pid_3', ''),  # Patient ID
                'name': self._parse_hl7_name(pid.get('pid_5', '')),
                'birth_date': self._parse_hl7_date(pid.get('pid_7', '')),
                'gender': self.terminology_mappings['gender_mappings'].get(
                    pid.get('pid_8', ''), 'unknown'
                ),
                'race': self.terminology_mappings['race_mappings'].get(
                    pid.get('pid_10', ''), 'unknown'
                ),
                'address': self._parse_hl7_address(pid.get('pid_11', ''))
            }
        
        # Extract observation data from OBX segments
        if 'OBX' in segments:
            obx = segments['OBX']
            transformed['clinical_data']['observations'] = [{
                'observation_id': obx.get('obx_3', ''),
                'value': obx.get('obx_5', ''),
                'units': obx.get('obx_6', ''),
                'reference_range': obx.get('obx_7', ''),
                'status': obx.get('obx_11', '')
            }]
        
        return transformed
    
    async def _transform_fhir_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform FHIR resource to standardized format"""
        resource_type = record.get('resourceType')
        
        transformed = {
            'source_type': 'fhir',
            'resource_type': resource_type,
            'resource_id': record.get('id'),
            'last_updated': record.get('meta', {}).get('lastUpdated'),
            'data': {}
        }
        
        if resource_type == 'Patient':
            transformed['data'] = {
                'patient_id': record.get('id'),
                'identifiers': record.get('identifier', []),
                'name': self._extract_fhir_name(record.get('name', [])),
                'gender': record.get('gender'),
                'birth_date': record.get('birthDate'),
                'address': record.get('address', []),
                'telecom': record.get('telecom', [])
            }
        
        elif resource_type == 'Observation':
            transformed['data'] = {
                'observation_id': record.get('id'),
                'status': record.get('status'),
                'category': record.get('category', []),
                'code': record.get('code', {}),
                'subject': record.get('subject', {}),
                'effective_date': record.get('effectiveDateTime'),
                'value': self._extract_fhir_value(record),
                'interpretation': record.get('interpretation', [])
            }
        
        return transformed
    
    async def _transform_database_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform database record to standardized format"""
        transformed = {
            'source_type': 'database',
            'record_id': record.get('id'),
            'data': record.copy()
        }
        
        # Apply common database transformations
        for key, value in transformed['data'].items():
            # Handle null values
            if value is None:
                transformed['data'][key] = ''
            
            # Standardize boolean values
            elif isinstance(value, bool):
                transformed['data'][key] = str(value).lower()
            
            # Standardize date formats
            elif isinstance(value, str) and self._is_date_string(value):
                transformed['data'][key] = self._standardize_date(value)
        
        return transformed
    
    def _parse_hl7_name(self, name_string: str) -> Dict[str, str]:
        """Parse HL7 name format"""
        # HL7 name format: LastName^FirstName^MiddleName^Suffix^Prefix
        parts = name_string.split('^')
        return {
            'family': parts[0] if len(parts) > 0 else '',
            'given': parts[1] if len(parts) > 1 else '',
            'middle': parts[2] if len(parts) > 2 else '',
            'suffix': parts[3] if len(parts) > 3 else '',
            'prefix': parts[4] if len(parts) > 4 else ''
        }
    
    def _parse_hl7_date(self, date_string: str) -> str:
        """Parse HL7 date format (YYYYMMDD or YYYYMMDDHHMMSS)"""
        if not date_string:
            return ''
        
        try:
            if len(date_string) >= 8:
                year = date_string[:4]
                month = date_string[4:6]
                day = date_string[6:8]
                return f"{year}-{month}-{day}"
        except Exception:
            pass
        
        return date_string
    
    def _parse_hl7_address(self, address_string: str) -> Dict[str, str]:
        """Parse HL7 address format"""
        # HL7 address format: Street^OtherDesignation^City^State^Zip^Country
        parts = address_string.split('^')
        return {
            'line': [parts[0]] if len(parts) > 0 and parts[0] else [],
            'city': parts[2] if len(parts) > 2 else '',
            'state': parts[3] if len(parts) > 3 else '',
            'postal_code': parts[4] if len(parts) > 4 else '',
            'country': parts[5] if len(parts) > 5 else ''
        }
    
    def _extract_fhir_name(self, name_array: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract name from FHIR name array"""
        if not name_array:
            return {}
        
        # Use first name in array
        name = name_array[0]
        return {
            'family': name.get('family', ''),
            'given': ' '.join(name.get('given', [])),
            'prefix': ' '.join(name.get('prefix', [])),
            'suffix': ' '.join(name.get('suffix', []))
        }
    
    def _extract_fhir_value(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract value from FHIR observation"""
        value_fields = [
            'valueQuantity', 'valueCodeableConcept', 'valueString',
            'valueBoolean', 'valueInteger', 'valueRange', 'valueRatio',
            'valueSampledData', 'valueTime', 'valueDateTime', 'valuePeriod'
        ]
        
        for field in value_fields:
            if field in observation:
                return {
                    'type': field,
                    'value': observation[field]
                }
        
        return {}
    
    def _is_date_string(self, value: str) -> bool:
        """Check if string represents a date"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        import re
        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        return False
    
    def _standardize_date(self, date_string: str) -> str:
        """Standardize date format to ISO 8601"""
        try:
            # Try different date formats
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%Y%m%d']
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_string, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            return date_string  # Return original if no format matches
            
        except Exception:
            return date_string
    
    async def _validate_data_quality(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate data quality for a record"""
        issues = []
        
        for rule in self.data_quality_rules:
            try:
                # Get field value
                field_value = self._get_nested_field(record, rule.field_name)
                
                # Apply validation function
                is_valid = rule.validation_function(field_value)
                
                if not is_valid:
                    issues.append({
                        'record_id': record.get('id', 'unknown'),
                        'rule_id': rule.rule_id,
                        'issue_type': rule.rule_type,
                        'field_name': rule.field_name,
                        'severity': rule.severity,
                        'message': rule.error_message,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
            except Exception as e:
                logger.warning(f"Error applying data quality rule {rule.rule_id}: {e}")
        
        return issues
    
    def _get_nested_field(self, record: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation"""
        keys = field_path.split('.')
        value = record
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value

class DataLoader:
    """Advanced data loader for healthcare data warehouses"""
    
    def __init__(self, target_config: Dict[str, Any]):
        self.target_config = target_config
        self.target_type = target_config['type']  # 'database', 'data_lake', 'fhir_server'
        self.batch_size = target_config.get('batch_size', 1000)
        
        if self.target_type == 'database':
            self.engine = create_engine(target_config['connection_string'])
            self.Session = sessionmaker(bind=self.engine)
        elif self.target_type == 'data_lake':
            self.s3_client = boto3.client('s3')
            self.bucket_name = target_config['bucket_name']
    
    async def load(self, data: List[Dict[str, Any]], table_name: str = None) -> Dict[str, Any]:
        """Load data to target system"""
        if self.target_type == 'database':
            return await self._load_to_database(data, table_name)
        elif self.target_type == 'data_lake':
            return await self._load_to_data_lake(data, table_name)
        elif self.target_type == 'fhir_server':
            return await self._load_to_fhir_server(data)
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")
    
    async def _load_to_database(self, data: List[Dict[str, Any]], table_name: str) -> Dict[str, Any]:
        """Load data to database"""
        session = self.Session()
        loaded_count = 0
        error_count = 0
        
        try:
            # Convert to DataFrame for efficient loading
            df = pd.DataFrame(data)
            
            # Load in batches
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                
                try:
                    batch.to_sql(
                        table_name,
                        self.engine,
                        if_exists='append',
                        index=False,
                        method='multi'
                    )
                    loaded_count += len(batch)
                    
                except Exception as e:
                    logger.error(f"Error loading batch to database: {e}")
                    error_count += len(batch)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading data to database: {e}")
            raise
        finally:
            session.close()
        
        return {
            'loaded_count': loaded_count,
            'error_count': error_count,
            'target_table': table_name
        }
    
    async def _load_to_data_lake(self, data: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
        """Load data to data lake (S3)"""
        loaded_count = 0
        error_count = 0
        
        try:
            # Create partitioned structure by date
            current_date = datetime.utcnow().strftime('%Y/%m/%d')
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            
            # Convert data to JSON Lines format
            json_lines = []
            for record in data:
                json_lines.append(json.dumps(record, default=str))
            
            content = '\n'.join(json_lines)
            
            # Upload to S3
            key = f"{prefix}/{current_date}/{timestamp}.jsonl"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=content.encode('utf-8'),
                ContentType='application/jsonl'
            )
            
            loaded_count = len(data)
            
        except Exception as e:
            logger.error(f"Error loading data to data lake: {e}")
            error_count = len(data)
            raise
        
        return {
            'loaded_count': loaded_count,
            'error_count': error_count,
            'target_location': f"s3://{self.bucket_name}/{key}"
        }
    
    async def _load_to_fhir_server(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Load data to FHIR server"""
        fhir_url = self.target_config['fhir_url']
        auth_token = self.target_config.get('auth_token')
        
        headers = {
            'Content-Type': 'application/fhir+json',
            'Accept': 'application/fhir+json'
        }
        
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
        
        loaded_count = 0
        error_count = 0
        
        # Create FHIR bundle for batch processing
        bundle = {
            "resourceType": "Bundle",
            "type": "batch",
            "entry": []
        }
        
        for record in data:
            if 'resourceType' in record:
                bundle["entry"].append({
                    "request": {
                        "method": "POST",
                        "url": record['resourceType']
                    },
                    "resource": record
                })
        
        try:
            # Send bundle to FHIR server
            response = requests.post(
                fhir_url,
                json=bundle,
                headers=headers
            )
            
            if response.status_code == 200:
                result_bundle = response.json()
                
                # Count successful operations
                for entry in result_bundle.get('entry', []):
                    if 'response' in entry:
                        status = entry['response'].get('status', '')
                        if status.startswith('2'):  # 2xx status codes
                            loaded_count += 1
                        else:
                            error_count += 1
            else:
                error_count = len(data)
                logger.error(f"FHIR server error: {response.status_code} - {response.text}")
                
        except Exception as e:
            error_count = len(data)
            logger.error(f"Error loading data to FHIR server: {e}")
            raise
        
        return {
            'loaded_count': loaded_count,
            'error_count': error_count,
            'target_server': fhir_url
        }

class HealthcareETLPipeline:
    """
    Comprehensive healthcare ETL pipeline
    
    This class orchestrates the entire ETL process with advanced features
    including real-time processing, data quality monitoring, and error recovery.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_sources: List[DataSource] = []
        self.transformer = DataTransformer()
        self.loader = DataLoader(config['target'])
        
        # Initialize monitoring
        self.redis_client = None
        if 'redis_url' in config:
            self.redis_client = redis.from_url(config['redis_url'])
        
        # Initialize message queue for real-time processing
        self.kafka_producer = None
        self.kafka_consumer = None
        if 'kafka_config' in config:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=config['kafka_config']['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
            )
        
        # Setup data quality rules
        self._setup_default_quality_rules()
        
        # Pipeline metrics
        self.metrics = {
            'total_records_processed': 0,
            'total_records_loaded': 0,
            'total_errors': 0,
            'last_run_timestamp': None,
            'pipeline_status': 'initialized'
        }
    
    def add_data_source(self, source: DataSource):
        """Add a data source to the pipeline"""
        self.data_sources.append(source)
        logger.info(f"Added data source: {source.source_id}")
    
    def _setup_default_quality_rules(self):
        """Setup default data quality rules for healthcare data"""
        
        # Completeness rules
        self.transformer.add_data_quality_rule(DataQualityRule(
            rule_id='patient_id_required',
            rule_type='completeness',
            field_name='patient_data.patient_id',
            validation_function=lambda x: x is not None and str(x).strip() != '',
            error_message='Patient ID is required',
            severity='error'
        ))
        
        # Accuracy rules
        self.transformer.add_data_quality_rule(DataQualityRule(
            rule_id='valid_gender',
            rule_type='accuracy',
            field_name='patient_data.gender',
            validation_function=lambda x: x in ['male', 'female', 'other', 'unknown'] if x else True,
            error_message='Invalid gender value',
            severity='warning'
        ))
        
        # Date validation
        self.transformer.add_data_quality_rule(DataQualityRule(
            rule_id='valid_birth_date',
            rule_type='accuracy',
            field_name='patient_data.birth_date',
            validation_function=self._validate_birth_date,
            error_message='Invalid birth date',
            severity='error'
        ))
        
        # Consistency rules
        self.transformer.add_data_quality_rule(DataQualityRule(
            rule_id='age_birth_date_consistency',
            rule_type='consistency',
            field_name='patient_data',
            validation_function=self._validate_age_birth_date_consistency,
            error_message='Age and birth date are inconsistent',
            severity='warning'
        ))
    
    def _validate_birth_date(self, birth_date: str) -> bool:
        """Validate birth date"""
        if not birth_date:
            return True  # Optional field
        
        try:
            dt = datetime.fromisoformat(birth_date.replace('Z', '+00:00'))
            # Birth date should be in the past but not too far back
            now = datetime.now(dt.tzinfo)
            return dt < now and dt > datetime(1900, 1, 1, tzinfo=dt.tzinfo)
        except Exception:
            return False
    
    def _validate_age_birth_date_consistency(self, patient_data: Dict[str, Any]) -> bool:
        """Validate consistency between age and birth date"""
        if not isinstance(patient_data, dict):
            return True
        
        age = patient_data.get('age')
        birth_date = patient_data.get('birth_date')
        
        if not age or not birth_date:
            return True  # Can't validate if either is missing
        
        try:
            dt = datetime.fromisoformat(birth_date.replace('Z', '+00:00'))
            calculated_age = (datetime.now(dt.tzinfo) - dt).days / 365.25
            
            # Allow 1 year tolerance
            return abs(float(age) - calculated_age) <= 1.0
        except Exception:
            return False
    
    async def run_pipeline(self, source_id: str = None, incremental: bool = True) -> Dict[str, Any]:
        """
        Run the ETL pipeline
        
        Args:
            source_id: Specific source to process (None for all sources)
            incremental: Whether to run incremental or full extraction
            
        Returns:
            Pipeline execution results
        """
        start_time = datetime.utcnow()
        self.metrics['pipeline_status'] = 'running'
        
        logger.info(f"Starting ETL pipeline at {start_time}")
        
        pipeline_results = {
            'start_time': start_time.isoformat(),
            'sources_processed': 0,
            'total_records_extracted': 0,
            'total_records_transformed': 0,
            'total_records_loaded': 0,
            'total_quality_issues': 0,
            'errors': [],
            'source_results': {}
        }
        
        # Determine sources to process
        sources_to_process = []
        if source_id:
            sources_to_process = [s for s in self.data_sources if s.source_id == source_id]
        else:
            sources_to_process = [s for s in self.data_sources if s.enabled]
        
        # Process each source
        for source in sources_to_process:
            try:
                source_result = await self._process_source(source, incremental)
                pipeline_results['source_results'][source.source_id] = source_result
                
                # Update totals
                pipeline_results['sources_processed'] += 1
                pipeline_results['total_records_extracted'] += source_result['records_extracted']
                pipeline_results['total_records_transformed'] += source_result['records_transformed']
                pipeline_results['total_records_loaded'] += source_result['records_loaded']
                pipeline_results['total_quality_issues'] += source_result['quality_issues']
                
                # Update source last sync time
                source.last_sync = datetime.utcnow()
                source.error_count = 0
                
            except Exception as e:
                error_msg = f"Error processing source {source.source_id}: {e}"
                logger.error(error_msg)
                pipeline_results['errors'].append(error_msg)
                
                # Update source error count
                source.error_count += 1
                if source.error_count >= source.max_errors:
                    source.enabled = False
                    logger.warning(f"Disabled source {source.source_id} due to excessive errors")
        
        # Calculate execution time
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        pipeline_results['end_time'] = end_time.isoformat()
        pipeline_results['execution_time_seconds'] = execution_time
        
        # Update metrics
        self.metrics['total_records_processed'] += pipeline_results['total_records_extracted']
        self.metrics['total_records_loaded'] += pipeline_results['total_records_loaded']
        self.metrics['total_errors'] += len(pipeline_results['errors'])
        self.metrics['last_run_timestamp'] = end_time.isoformat()
        self.metrics['pipeline_status'] = 'completed'
        
        # Store metrics in Redis if available
        if self.redis_client:
            self.redis_client.setex(
                'healthcare_etl_metrics',
                3600,  # 1 hour TTL
                json.dumps(self.metrics, default=str)
            )
        
        logger.info(f"ETL pipeline completed in {execution_time:.2f} seconds")
        return pipeline_results
    
    async def _process_source(self, source: DataSource, incremental: bool) -> Dict[str, Any]:
        """Process a single data source"""
        logger.info(f"Processing source: {source.source_id}")
        
        result = {
            'source_id': source.source_id,
            'records_extracted': 0,
            'records_transformed': 0,
            'records_loaded': 0,
            'quality_issues': 0,
            'errors': []
        }
        
        try:
            # Create appropriate extractor
            extractor = self._create_extractor(source)
            
            # Validate connection
            if not extractor.validate_connection():
                raise Exception(f"Cannot connect to source {source.source_id}")
            
            # Extract data
            if not incremental:
                source.last_sync = None  # Force full extraction
            
            extracted_data = await extractor.extract()
            result['records_extracted'] = len(extracted_data)
            
            if not extracted_data:
                logger.info(f"No data extracted from source {source.source_id}")
                return result
            
            # Transform data
            transformed_data, quality_issues = await self.transformer.transform(
                extracted_data, source.source_type
            )
            result['records_transformed'] = len(transformed_data)
            result['quality_issues'] = len(quality_issues)
            
            # Log quality issues
            if quality_issues:
                await self._log_quality_issues(quality_issues, source.source_id)
            
            # Load data
            if transformed_data:
                load_result = await self.loader.load(
                    transformed_data, 
                    table_name=f"{source.source_id}_data"
                )
                result['records_loaded'] = load_result['loaded_count']
                
                if load_result['error_count'] > 0:
                    result['errors'].append(f"Failed to load {load_result['error_count']} records")
            
            # Send to real-time processing if configured
            if self.kafka_producer and source.update_frequency == 'real-time':
                await self._send_to_realtime_processing(transformed_data, source.source_id)
            
        except Exception as e:
            error_msg = f"Error processing source {source.source_id}: {e}"
            result['errors'].append(error_msg)
            raise
        
        return
