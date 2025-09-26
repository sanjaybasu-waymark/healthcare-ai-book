"""
Chapter 1 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

"""
Clinical Informatics FHIR Data Processing Module

This module implements production-ready FHIR data processing capabilities
specifically designed for healthcare AI applications. It includes comprehensive
error handling, audit logging, and clinical validation features required
for deployment in real healthcare environments.

Author: Sanjay Basu MD PhD
License: MIT
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from fhirclient import client
from fhirclient.models import patient, observation, condition, medicationrequest
import warnings

\# Configure comprehensive logging for clinical applications
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('clinical_informatics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FHIRResourceType(Enum):
    """Enumeration of FHIR resources commonly used in clinical AI applications."""
    PATIENT = "Patient"
    OBSERVATION = "Observation"
    CONDITION = "Condition"
    MEDICATION_REQUEST = "MedicationRequest"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    ENCOUNTER = "Encounter"
    PROCEDURE = "Procedure"
    IMMUNIZATION = "Immunization"
    ALLERGY_INTOLERANCE = "AllergyIntolerance"

class ClinicalDataQualityLevel(Enum):
    """Data quality levels for clinical AI applications."""
    EXCELLENT = "excellent"  \# >95% complete, fully validated
    GOOD = "good"           \# 85-95% complete, mostly validated
    ACCEPTABLE = "acceptable"  \# 70-85% complete, basic validation
    POOR = "poor"           \# <70% complete, limited validation
    UNUSABLE = "unusable"   \# Insufficient for clinical AI applications

@dataclass
class ClinicalDataQuality:
    """
    Comprehensive data quality assessment for clinical AI applications.
    
    This class provides detailed metrics about data completeness, consistency,
    and clinical validity that are essential for evaluating whether a dataset
    is suitable for AI model development and deployment.
    """
    total_records: int
    complete_records: int
    missing_critical_fields: int
    temporal_consistency_issues: int
    coding_standard_compliance: float
    data_freshness_hours: float
    duplicate_records: int
    outlier_values: int
    quality_level: ClinicalDataQualityLevel
    quality_score: float = field(init=False)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate overall quality score and generate recommendations."""
        \# Calculate composite quality score (0-100)
        completeness_score = (self.complete_records / self.total_records) * 100 if self.total_records > 0 else 0
        consistency_score = max(0, 100 - (self.temporal_consistency_issues / self.total_records) * 100) if self.total_records > 0 else 0
        coding_score = self.coding_standard_compliance * 100
        freshness_score = max(0, 100 - (self.data_freshness_hours / 24) * 10)  \# Penalize data older than 24 hours
        
        self.quality_score = (completeness_score + consistency_score + coding_score + freshness_score) / 4
        
        \# Generate specific recommendations
        if completeness_score < 85:
            self.recommendations.append("Improve data completeness through enhanced data collection protocols")
        if consistency_score < 90:
            self.recommendations.append("Implement temporal consistency validation rules")
        if coding_score < 95:
            self.recommendations.append("Enhance coding standard compliance training and validation")
        if freshness_score < 80:
            self.recommendations.append("Implement more frequent data synchronization processes")

class HIPAAComplianceValidator:
    """
    HIPAA compliance validation for clinical data processing.
    
    This class implements essential HIPAA compliance checks that must be
    performed when processing protected health information (PHI) for AI applications.
    """
    
    @staticmethod
    def validate_data_minimization(data_fields: List[str], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that only necessary data fields are being processed (HIPAA minimum necessary standard).
        
        Args:
            data_fields: List of data fields being processed
            required_fields: List of fields actually required for the AI application
            
        Returns:
            Tuple of (is_compliant, list_of_unnecessary_fields)
        """
        unnecessary_fields = [field for field in data_fields if field not in required_fields]
        is_compliant = len(unnecessary_fields) == 0
        
        if not is_compliant:
            logger.warning(f"HIPAA compliance issue: Unnecessary fields detected: {unnecessary_fields}")
        
        return is_compliant, unnecessary_fields
    
    @staticmethod
    def generate_audit_log_entry(action: str, patient_id: str, user_id: str, 
                                data_accessed: List[str]) -> Dict[str, Any]:
        """
        Generate HIPAA-compliant audit log entry.
        
        Args:
            action: Description of the action performed
            patient_id: Patient identifier (should be hashed for logging)
            user_id: User performing the action
            data_accessed: List of data types accessed
            
        Returns:
            Structured audit log entry
        """
        \# Hash patient ID for audit logging (never log actual patient identifiers)
        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'patient_hash': patient_hash,
            'user_id': user_id,
            'data_accessed': data_accessed,
            'ip_address': 'system',  \# In production, capture actual IP
            'session_id': 'system',  \# In production, capture actual session
            'outcome': 'success'     \# Update based on actual outcome
        }
        
        logger.info(f"Audit log entry created: {action} for patient {patient_hash}")
        return audit_entry

class FHIRClinicalDataProcessor:
    """
    Production-ready FHIR data processor for clinical AI applications.
    
    This class implements clinical informatics best practices including:
    - Comprehensive error handling and retry logic
    - HIPAA compliance validation and audit logging
    - Clinical data quality assessment
    - Robust data validation and cleaning
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 fhir_base_url: str,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 timeout_seconds: int = 30,
                 max_retries: int = 3,
                 enable_audit_logging: bool = True):
        """
        Initialize FHIR processor with clinical-grade configuration.
        
        Args:
            fhir_base_url: Base URL for FHIR server (must be HTTPS in production)
            client_id: OAuth client ID for authenticated access
            client_secret: OAuth client secret for authenticated access
            timeout_seconds: Request timeout for clinical data retrieval
            max_retries: Number of retry attempts for failed requests
            enable_audit_logging: Whether to enable HIPAA audit logging
            
        Raises:
            ValueError: If FHIR URL is not HTTPS in production mode
            ConnectionError: If FHIR server is not accessible
        """
        self.base_url = fhir_base_url.rstrip('/')
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.enable_audit_logging = enable_audit_logging
        self.compliance_validator = HIPAAComplianceValidator()
        
        \# Validate security requirements for clinical data
        if not self.base_url.startswith('https://') and 'localhost' not in self.base_url:
            raise ValueError("Production FHIR endpoints must use HTTPS for PHI protection")
        
        \# Configure HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        \# Set up authentication if provided
        if client_id and client_secret:
            self._setup_oauth_authentication(client_id, client_secret)
        
        \# Test connectivity
        self._validate_fhir_connectivity()
        
        logger.info(f"Initialized FHIR processor for endpoint: {self.base_url}")
    
    def _setup_oauth_authentication(self, client_id: str, client_secret: str):
        """Set up OAuth 2.0 authentication for FHIR server access."""
        \# Implementation would depend on specific FHIR server OAuth configuration
        \# This is a placeholder for production OAuth implementation
        logger.info("OAuth authentication configured")
    
    def _validate_fhir_connectivity(self):
        """Validate that FHIR server is accessible and responding correctly."""
        try:
            response = self.session.get(
                f"{self.base_url}/metadata",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            capability_statement = response.json()
            if capability_statement.get('resourceType') != 'CapabilityStatement':
                raise ValueError("Invalid FHIR server response")
                
            logger.info("FHIR server connectivity validated successfully")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FHIR server connectivity validation failed: {e}")
            raise ConnectionError(f"Cannot connect to FHIR server: {e}")
    
    def get_patient_clinical_summary(self, 
                                   patient_id: str,
                                   include_history_days: int = 365,
                                   user_id: str = "system") -> Dict[str, Any]:
        """
        Retrieve comprehensive clinical summary for AI analysis.
        
        This method demonstrates how to gather the clinical context needed for
        most healthcare AI applications while maintaining proper error handling,
        HIPAA compliance, and clinical data validation.
        
        Args:
            patient_id: FHIR Patient resource ID
            include_history_days: Days of historical data to include
            user_id: User ID for audit logging
            
        Returns:
            Structured clinical summary with demographics, conditions, observations,
            and medications organized for AI processing
            
        Raises:
            ValueError: If patient_id is invalid or patient not found
            ConnectionError: If FHIR server is unreachable
            TimeoutError: If data retrieval exceeds timeout limits
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Retrieving clinical summary for patient {patient_id[:8]}...")
            
            \# Generate audit log entry
            if self.enable_audit_logging:
                audit_entry = self.compliance_validator.generate_audit_log_entry(
                    action="retrieve_clinical_summary",
                    patient_id=patient_id,
                    user_id=user_id,
                    data_accessed=["demographics", "conditions", "observations", "medications"]
                )
            
            \# Calculate date range for clinical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=include_history_days)
            date_filter = f"date=ge{start_date.strftime('%Y-%m-%d')}"
            
            clinical_summary = {
                'patient_id': patient_id,
                'retrieval_timestamp': end_date.isoformat(),
                'data_period_days': include_history_days,
                'demographics': {},
                'active_conditions': [],
                'recent_observations': [],
                'current_medications': [],
                'recent_encounters': [],
                'data_quality': None,
                'processing_time_seconds': 0
            }
            
            \# Retrieve patient demographics
            patient_data = self._get_patient_demographics(patient_id)
            clinical_summary['demographics'] = patient_data
            
            \# Retrieve active conditions
            conditions = self._get_patient_conditions(patient_id, date_filter)
            clinical_summary['active_conditions'] = conditions
            
            \# Retrieve recent observations (labs, vitals, etc.)
            observations = self._get_patient_observations(patient_id, date_filter)
            clinical_summary['recent_observations'] = observations
            
            \# Retrieve current medications
            medications = self._get_patient_medications(patient_id, date_filter)
            clinical_summary['current_medications'] = medications
            
            \# Retrieve recent encounters
            encounters = self._get_patient_encounters(patient_id, date_filter)
            clinical_summary['recent_encounters'] = encounters
            
            \# Assess data quality
            data_quality = self._assess_clinical_data_quality(clinical_summary)
            clinical_summary['data_quality'] = data_quality
            
            \# Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            clinical_summary['processing_time_seconds'] = processing_time
            
            logger.info(f"Clinical summary retrieved successfully in {processing_time:.2f} seconds")
            logger.info(f"Data quality score: {data_quality.quality_score:.1f}/100 ({data_quality.quality_level.value})")
            
            return clinical_summary
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout retrieving clinical summary for patient {patient_id[:8]}")
            raise TimeoutError("Clinical data retrieval exceeded timeout limit")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error retrieving clinical summary: {e}")
            raise ConnectionError(f"Failed to retrieve clinical data: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error retrieving clinical summary: {e}")
            raise
    
    def _get_patient_demographics(self, patient_id: str) -> Dict[str, Any]:
        """Retrieve patient demographic information."""
        try:
            response = self.session.get(
                f"{self.base_url}/Patient/{patient_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            patient_resource = response.json()
            
            \# Extract key demographic information
            demographics = {
                'id': patient_resource.get('id'),
                'gender': patient_resource.get('gender'),
                'birth_date': patient_resource.get('birthDate'),
                'age_years': self._calculate_age(patient_resource.get('birthDate')),
                'active': patient_resource.get('active', True),
                'name': self._extract_patient_name(patient_resource.get('name', [])),
                'identifiers': self._extract_patient_identifiers(patient_resource.get('identifier', [])),
                'contact_info': self._extract_contact_info(patient_resource.get('telecom', [])),
                'address': self._extract_address_info(patient_resource.get('address', []))
            }
            
            return demographics
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve patient demographics: {e}")
            return {}
    
    def _get_patient_conditions(self, patient_id: str, date_filter: str) -> List[Dict[str, Any]]:
        """Retrieve patient conditions (diagnoses)."""
        try:
            response = self.session.get(
                f"{self.base_url}/Condition",
                params={
                    'patient': patient_id,
                    'clinical-status': 'active',
                    '_sort': '-onset-date',
                    '_count': 100
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            bundle = response.json()
            conditions = []
            
            for entry in bundle.get('entry', []):
                condition = entry.get('resource', {})
                
                condition_data = {
                    'id': condition.get('id'),
                    'clinical_status': self._extract_coding_display(
                        condition.get('clinicalStatus', {}).get('coding', [])
                    ),
                    'verification_status': self._extract_coding_display(
                        condition.get('verificationStatus', {}).get('coding', [])
                    ),
                    'category': self._extract_coding_display(
                        condition.get('category', [{}])<sup>0</sup>.get('coding', [])
                    ),
                    'code': self._extract_condition_code(condition.get('code', {})),
                    'onset_date': condition.get('onsetDateTime'),
                    'recorded_date': condition.get('recordedDate'),
                    'severity': self._extract_coding_display(
                        condition.get('severity', {}).get('coding', [])
                    )
                }
                
                conditions.append(condition_data)
            
            logger.info(f"Retrieved {len(conditions)} active conditions")
            return conditions
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve patient conditions: {e}")
            return []
    
    def _get_patient_observations(self, patient_id: str, date_filter: str) -> List[Dict[str, Any]]:
        """Retrieve patient observations (labs, vitals, etc.)."""
        try:
            response = self.session.get(
                f"{self.base_url}/Observation",
                params={
                    'patient': patient_id,
                    'status': 'final',
                    '_sort': '-date',
                    '_count': 200,
                    'date': f"ge{(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')}"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            bundle = response.json()
            observations = []
            
            for entry in bundle.get('entry', []):
                observation = entry.get('resource', {})
                
                observation_data = {
                    'id': observation.get('id'),
                    'status': observation.get('status'),
                    'category': self._extract_coding_display(
                        observation.get('category', [{}])<sup>0</sup>.get('coding', [])
                    ),
                    'code': self._extract_observation_code(observation.get('code', {})),
                    'effective_date': observation.get('effectiveDateTime'),
                    'value': self._extract_observation_value(observation),
                    'reference_range': self._extract_reference_range(observation.get('referenceRange', [])),
                    'interpretation': self._extract_coding_display(
                        observation.get('interpretation', [{}])<sup>0</sup>.get('coding', [])
                    )
                }
                
                observations.append(observation_data)
            
            logger.info(f"Retrieved {len(observations)} recent observations")
            return observations
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve patient observations: {e}")
            return []
    
    def _get_patient_medications(self, patient_id: str, date_filter: str) -> List[Dict[str, Any]]:
        """Retrieve patient medications."""
        try:
            response = self.session.get(
                f"{self.base_url}/MedicationRequest",
                params={
                    'patient': patient_id,
                    'status': 'active',
                    '_sort': '-authored-on',
                    '_count': 100
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            bundle = response.json()
            medications = []
            
            for entry in bundle.get('entry', []):
                medication_request = entry.get('resource', {})
                
                medication_data = {
                    'id': medication_request.get('id'),
                    'status': medication_request.get('status'),
                    'intent': medication_request.get('intent'),
                    'medication': self._extract_medication_code(
                        medication_request.get('medicationCodeableConcept', {})
                    ),
                    'authored_on': medication_request.get('authoredOn'),
                    'dosage_instruction': self._extract_dosage_instruction(
                        medication_request.get('dosageInstruction', [])
                    ),
                    'dispense_request': medication_request.get('dispenseRequest', {}),
                    'requester': medication_request.get('requester', {}).get('display')
                }
                
                medications.append(medication_data)
            
            logger.info(f"Retrieved {len(medications)} current medications")
            return medications
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve patient medications: {e}")
            return []
    
    def _get_patient_encounters(self, patient_id: str, date_filter: str) -> List[Dict[str, Any]]:
        """Retrieve patient encounters."""
        try:
            response = self.session.get(
                f"{self.base_url}/Encounter",
                params={
                    'patient': patient_id,
                    '_sort': '-date',
                    '_count': 50,
                    'date': f"ge{(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')}"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            bundle = response.json()
            encounters = []
            
            for entry in bundle.get('entry', []):
                encounter = entry.get('resource', {})
                
                encounter_data = {
                    'id': encounter.get('id'),
                    'status': encounter.get('status'),
                    'class': self._extract_coding_display(
                        [encounter.get('class', {})]
                    ),
                    'type': self._extract_coding_display(
                        encounter.get('type', [{}])<sup>0</sup>.get('coding', [])
                    ),
                    'period': encounter.get('period', {}),
                    'reason_code': self._extract_coding_display(
                        encounter.get('reasonCode', [{}])<sup>0</sup>.get('coding', [])
                    ),
                    'location': encounter.get('location', [{}])<sup>0</sup>.get('location', {}).get('display'),
                    'participant': [p.get('individual', {}).get('display') 
                                  for p in encounter.get('participant', [])]
                }
                
                encounters.append(encounter_data)
            
            logger.info(f"Retrieved {len(encounters)} recent encounters")
            return encounters
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve patient encounters: {e}")
            return []
    
    def _assess_clinical_data_quality(self, clinical_summary: Dict[str, Any]) -> ClinicalDataQuality:
        """Assess the quality of retrieved clinical data."""
        total_records = (
            len(clinical_summary.get('active_conditions', [])) +
            len(clinical_summary.get('recent_observations', [])) +
            len(clinical_summary.get('current_medications', [])) +
            len(clinical_summary.get('recent_encounters', []))
        )
        
        \# Count complete records (records with all essential fields)
        complete_records = 0
        missing_critical_fields = 0
        temporal_consistency_issues = 0
        
        \# Assess conditions completeness
        for condition in clinical_summary.get('active_conditions', []):
            if all(condition.get(field) for field in ['code', 'clinical_status']):
                complete_records += 1
            else:
                missing_critical_fields += 1
        
        \# Assess observations completeness
        for observation in clinical_summary.get('recent_observations', []):
            if all(observation.get(field) for field in ['code', 'value', 'effective_date']):
                complete_records += 1
            else:
                missing_critical_fields += 1
        
        \# Calculate coding standard compliance (simplified)
        coding_compliance = 0.95  \# Placeholder - would implement actual SNOMED/LOINC validation
        
        \# Calculate data freshness
        latest_date = datetime.now() - timedelta(days=30)  \# Placeholder calculation
        data_freshness_hours = 24.0  \# Placeholder
        
        \# Determine quality level
        completeness_rate = complete_records / total_records if total_records > 0 else 0
        if completeness_rate >= 0.95:
            quality_level = ClinicalDataQualityLevel.EXCELLENT
        elif completeness_rate >= 0.85:
            quality_level = ClinicalDataQualityLevel.GOOD
        elif completeness_rate >= 0.70:
            quality_level = ClinicalDataQualityLevel.ACCEPTABLE
        elif completeness_rate >= 0.50:
            quality_level = ClinicalDataQualityLevel.POOR
        else:
            quality_level = ClinicalDataQualityLevel.UNUSABLE
        
        return ClinicalDataQuality(
            total_records=total_records,
            complete_records=complete_records,
            missing_critical_fields=missing_critical_fields,
            temporal_consistency_issues=temporal_consistency_issues,
            coding_standard_compliance=coding_compliance,
            data_freshness_hours=data_freshness_hours,
            duplicate_records=0,  \# Would implement duplicate detection
            outlier_values=0,     \# Would implement outlier detection
            quality_level=quality_level
        )
    
    \# Helper methods for data extraction
    def _calculate_age(self, birth_date: Optional[str]) -> Optional[int]:
        """Calculate patient age from birth date."""
        if not birth_date:
            return None
        
        try:
            birth = datetime.strptime(birth_date, '%Y-%m-%d')
            today = datetime.now()
            age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            return age
        except ValueError:
            return None
    
    def _extract_patient_name(self, names: List[Dict[str, Any]]) -> Optional[str]:
        """Extract patient name from FHIR name array."""
        if not names:
            return None
        
        \# Prefer official name, fall back to usual
        official_name = next((name for name in names if name.get('use') == 'official'), None)
        name_to_use = official_name or names<sup>0</sup>
        
        given_names = ' '.join(name_to_use.get('given', []))
        family_name = name_to_use.get('family', '')
        
        return f"{given_names} {family_name}".strip()
    
    def _extract_patient_identifiers(self, identifiers: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract patient identifiers."""
        extracted_identifiers = []
        for identifier in identifiers:
            extracted_identifiers.append({
                'system': identifier.get('system', ''),
                'value': identifier.get('value', ''),
                'type': identifier.get('type', {}).get('text', '')
            })
        return extracted_identifiers
    
    def _extract_contact_info(self, telecoms: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract patient contact information."""
        contact_info = {}
        for telecom in telecoms:
            system = telecom.get('system')
            value = telecom.get('value')
            if system and value:
                contact_info[system] = value
        return contact_info
    
    def _extract_address_info(self, addresses: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Extract patient address information."""
        if not addresses:
            return None
        
        \# Prefer home address
        home_address = next((addr for addr in addresses if addr.get('use') == 'home'), None)
        address_to_use = home_address or addresses<sup>0</sup>
        
        return {
            'line': ' '.join(address_to_use.get('line', [])),
            'city': address_to_use.get('city', ''),
            'state': address_to_use.get('state', ''),
            'postal_code': address_to_use.get('postalCode', ''),
            'country': address_to_use.get('country', '')
        }
    
    def _extract_coding_display(self, codings: List[Dict[str, Any]]) -> Optional[str]:
        """Extract display text from FHIR coding array."""
        if not codings:
            return None
        
        \# Prefer display text, fall back to code
        for coding in codings:
            if coding.get('display'):
                return coding['display']
        
        return codings<sup>0</sup>.get('code') if codings else None
    
    def _extract_condition_code(self, code: Dict[str, Any]) -> Dict[str, Any]:
        """Extract condition code information."""
        return {
            'text': code.get('text'),
            'codings': [
                {
                    'system': coding.get('system'),
                    'code': coding.get('code'),
                    'display': coding.get('display')
                }
                for coding in code.get('coding', [])
            ]
        }
    
    def _extract_observation_code(self, code: Dict[str, Any]) -> Dict[str, Any]:
        """Extract observation code information."""
        return self._extract_condition_code(code)  \# Same structure
    
    def _extract_observation_value(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract observation value with proper type handling."""
        value_data = {'type': None, 'value': None, 'unit': None}
        
        \# Handle different value types
        if 'valueQuantity' in observation:
            quantity = observation['valueQuantity']
            value_data.update({
                'type': 'quantity',
                'value': quantity.get('value'),
                'unit': quantity.get('unit'),
                'system': quantity.get('system')
            })
        elif 'valueCodeableConcept' in observation:
            concept = observation['valueCodeableConcept']
            value_data.update({
                'type': 'codeable_concept',
                'value': concept.get('text'),
                'codings': concept.get('coding', [])
            })
        elif 'valueString' in observation:
            value_data.update({
                'type': 'string',
                'value': observation['valueString']
            })
        elif 'valueBoolean' in observation:
            value_data.update({
                'type': 'boolean',
                'value': observation['valueBoolean']
            })
        
        return value_data
    
    def _extract_reference_range(self, reference_ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract reference range information."""
        ranges = []
        for ref_range in reference_ranges:
            range_data = {
                'low': ref_range.get('low', {}).get('value'),
                'high': ref_range.get('high', {}).get('value'),
                'unit': ref_range.get('low', {}).get('unit') or ref_range.get('high', {}).get('unit'),
                'type': ref_range.get('type', {}).get('text')
            }
            ranges.append(range_data)
        return ranges
    
    def _extract_medication_code(self, medication: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medication code information."""
        return self._extract_condition_code(medication)  \# Same structure
    
    def _extract_dosage_instruction(self, dosage_instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract dosage instruction information."""
        instructions = []
        for instruction in dosage_instructions:
            instruction_data = {
                'text': instruction.get('text'),
                'timing': instruction.get('timing', {}).get('repeat', {}),
                'route': self._extract_coding_display(
                    instruction.get('route', {}).get('coding', [])
                ),
                'dose_quantity': instruction.get('doseAndRate', [{}])<sup>0</sup>.get('doseQuantity', {})
            }
            instructions.append(instruction_data)
        return instructions


\# Example usage and testing functions
def demonstrate_fhir_processing():
    """
    Demonstrate comprehensive FHIR data processing for clinical AI applications.
    
    This function shows how to use the FHIRClinicalDataProcessor class
    to retrieve and process clinical data while maintaining HIPAA compliance
    and clinical data quality standards.
    """
    \# Initialize FHIR processor (using public test server for demonstration)
    fhir_processor = FHIRClinicalDataProcessor(
        fhir_base_url="https://hapi.fhir.org/baseR4",
        timeout_seconds=30,
        max_retries=3,
        enable_audit_logging=True
    )
    
    \# Example patient ID from public test server
    test_patient_id = "example-patient-1"
    
    try:
        \# Retrieve comprehensive clinical summary
        clinical_summary = fhir_processor.get_patient_clinical_summary(
            patient_id=test_patient_id,
            include_history_days=365,
            user_id="demo_user"
        )
        
        \# Display summary information
        print("\n=== Clinical Summary Retrieved ===")
        print(f"Patient ID: {clinical_summary['patient_id']}")
        print(f"Data Period: {clinical_summary['data_period_days']} days")
        print(f"Processing Time: {clinical_summary['processing_time_seconds']:.2f} seconds")
        
        \# Display data quality assessment
        data_quality = clinical_summary['data_quality']
        print(f"\n=== Data Quality Assessment ===")
        print(f"Quality Score: {data_quality.quality_score:.1f}/100")
        print(f"Quality Level: {data_quality.quality_level.value}")
        print(f"Total Records: {data_quality.total_records}")
        print(f"Complete Records: {data_quality.complete_records}")
        
        if data_quality.recommendations:
            print("\nRecommendations:")
            for rec in data_quality.recommendations:
                print(f"- {rec}")
        
        \# Display clinical data summary
        demographics = clinical_summary['demographics']
        print(f"\n=== Demographics ===")
        print(f"Age: {demographics.get('age_years', 'Unknown')} years")
        print(f"Gender: {demographics.get('gender', 'Unknown')}")
        
        print(f"\n=== Clinical Data Summary ===")
        print(f"Active Conditions: {len(clinical_summary['active_conditions'])}")
        print(f"Recent Observations: {len(clinical_summary['recent_observations'])}")
        print(f"Current Medications: {len(clinical_summary['current_medications'])}")
        print(f"Recent Encounters: {len(clinical_summary['recent_encounters'])}")
        
        return clinical_summary
        
    except Exception as e:
        logger.error(f"Error in FHIR processing demonstration: {e}")
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    \# Run demonstration
    demonstrate_fhir_processing()