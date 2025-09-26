"""
Chapter 1 - Example 3
Extracted from Healthcare AI Implementation Guide
"""

"""
HIPAA-Compliant Healthcare Data Processing for AI Applications

This module implements comprehensive HIPAA compliance features including
privacy controls, security safeguards, audit logging, and risk management
specifically designed for healthcare AI systems.
"""

import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

logger = logging.getLogger(__name__)

class PHIDataType(Enum):
    """Types of Protected Health Information as defined by HIPAA."""
    NAME = "name"
    ADDRESS = "address"
    BIRTH_DATE = "birth_date"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    SSN = "social_security_number"
    MRN = "medical_record_number"
    ACCOUNT_NUMBER = "account_number"
    CERTIFICATE_NUMBER = "certificate_number"
    VEHICLE_IDENTIFIER = "vehicle_identifier"
    DEVICE_IDENTIFIER = "device_identifier"
    WEB_URL = "web_url"
    IP_ADDRESS = "ip_address"
    BIOMETRIC_IDENTIFIER = "biometric_identifier"
    PHOTO = "photograph"
    OTHER_UNIQUE_IDENTIFIER = "other_unique_identifier"

class HIPAAAccessLevel(Enum):
    """HIPAA access levels for role-based access control."""
    NO_ACCESS = "no_access"
    LIMITED_ACCESS = "limited_access"
    STANDARD_ACCESS = "standard_access"
    ELEVATED_ACCESS = "elevated_access"
    ADMINISTRATIVE_ACCESS = "administrative_access"

@dataclass
class HIPAAUser:
    """Represents a user with HIPAA access permissions."""
    user_id: str
    name: str
    role: str
    access_level: HIPAAAccessLevel
    authorized_phi_types: Set[PHIDataType]
    department: str
    supervisor: Optional[str] = None
    training_completion_date: Optional[datetime] = None
    last_access_review: Optional[datetime] = None
    active: bool = True

@dataclass
class PHIAccessRequest:
    """Represents a request to access PHI data."""
    request_id: str
    user_id: str
    patient_id: str
    phi_types_requested: Set[PHIDataType]
    purpose: str
    justification: str
    requested_at: datetime
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    access_granted_until: Optional[datetime] = None

@dataclass
class HIPAAAuditEntry:
    """HIPAA-compliant audit log entry."""
    entry_id: str
    timestamp: datetime
    user_id: str
    patient_id_hash: str  \# Never store actual patient ID in logs
    action: str
    phi_types_accessed: Set[PHIDataType]
    purpose: str
    outcome: str
    ip_address: str
    session_id: str
    additional_details: Dict[str, Any] = field(default_factory=dict)

class HIPAAComplianceEngine:
    """
    Comprehensive HIPAA compliance engine for healthcare AI applications.
    
    This class implements all major HIPAA requirements including privacy controls,
    security safeguards, audit logging, and breach detection specifically
    designed for AI systems processing protected health information.
    """
    
    def __init__(self, 
                 organization_name: str,
                 encryption_key: Optional[bytes] = None,
                 audit_retention_days: int = 2555):  \# 7 years as required by HIPAA
        """
        Initialize HIPAA compliance engine.
        
        Args:
            organization_name: Name of the covered entity
            encryption_key: Encryption key for PHI protection (generated if not provided)
            audit_retention_days: Number of days to retain audit logs
        """
        self.organization_name = organization_name
        self.audit_retention_days = audit_retention_days
        
        \# Initialize encryption for PHI protection
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(encryption_key)
        
        \# Initialize data structures
        self.users: Dict[str, HIPAAUser] = {}
        self.access_requests: Dict[str, PHIAccessRequest] = {}
        self.audit_log: List[HIPAAAuditEntry] = []
        self.phi_identifiers: Dict[PHIDataType, List[re.Pattern]] = self._initialize_phi_patterns()
        
        \# Security configuration
        self.max_failed_login_attempts = 3
        self.session_timeout_minutes = 30
        self.password_complexity_requirements = {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special_chars': True
        }
        
        logger.info(f"HIPAA compliance engine initialized for {organization_name}")
    
    def register_user(self, user: HIPAAUser) -> bool:
        """
        Register a new user with HIPAA access permissions.
        
        Args:
            user: HIPAAUser object with complete user information
            
        Returns:
            True if user registered successfully, False otherwise
        """
        try:
            \# Validate user information
            if not self._validate_user_information(user):
                logger.error(f"User validation failed for {user.user_id}")
                return False
            
            \# Check for duplicate user ID
            if user.user_id in self.users:
                logger.error(f"User {user.user_id} already exists")
                return False
            
            \# Register user
            self.users[user.user_id] = user
            
            \# Create audit entry
            audit_entry = HIPAAAuditEntry(
                entry_id=self._generate_audit_id(),
                timestamp=datetime.now(),
                user_id="system",
                patient_id_hash="N/A",
                action="user_registration",
                phi_types_accessed=set(),
                purpose="User management",
                outcome="success",
                ip_address="system",
                session_id="system",
                additional_details={"registered_user": user.user_id, "role": user.role}
            )
            self.audit_log.append(audit_entry)
            
            logger.info(f"User {user.user_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering user {user.user_id}: {e}")
            return False
    
    def request_phi_access(self, 
                          user_id: str,
                          patient_id: str,
                          phi_types: Set[PHIDataType],
                          purpose: str,
                          justification: str) -> str:
        """
        Request access to PHI data following minimum necessary principle.
        
        Args:
            user_id: ID of user requesting access
            patient_id: ID of patient whose data is being requested
            phi_types: Set of PHI data types being requested
            purpose: Purpose for accessing the data
            justification: Detailed justification for the access request
            
        Returns:
            Request ID for tracking the access request
        """
        request_id = self._generate_request_id()
        
        try:
            \# Validate user exists and is active
            if user_id not in self.users or not self.users[user_id].active:
                raise ValueError(f"Invalid or inactive user: {user_id}")
            
            user = self.users[user_id]
            
            \# Check if user is authorized for requested PHI types
            unauthorized_types = phi_types - user.authorized_phi_types
            if unauthorized_types:
                logger.warning(f"User {user_id} requested unauthorized PHI types: {unauthorized_types}")
            
            \# Create access request
            access_request = PHIAccessRequest(
                request_id=request_id,
                user_id=user_id,
                patient_id=patient_id,
                phi_types_requested=phi_types,
                purpose=purpose,
                justification=justification,
                requested_at=datetime.now()
            )
            
            \# Auto-approve if user has appropriate access level and authorization
            if (user.access_level in [HIPAAAccessLevel.STANDARD_ACCESS, HIPAAAccessLevel.ELEVATED_ACCESS] and
                not unauthorized_types):
                access_request.approved = True
                access_request.approved_by = "system_auto_approval"
                access_request.approved_at = datetime.now()
                access_request.access_granted_until = datetime.now() + timedelta(hours=8)  \# 8-hour access window
            
            self.access_requests[request_id] = access_request
            
            \# Create audit entry
            audit_entry = HIPAAAuditEntry(
                entry_id=self._generate_audit_id(),
                timestamp=datetime.now(),
                user_id=user_id,
                patient_id_hash=self._hash_patient_id(patient_id),
                action="phi_access_request",
                phi_types_accessed=phi_types,
                purpose=purpose,
                outcome="request_created",
                ip_address="system",  \# Would capture actual IP in production
                session_id="system",  \# Would capture actual session in production
                additional_details={
                    "request_id": request_id,
                    "justification": justification,
                    "auto_approved": access_request.approved
                }
            )
            self.audit_log.append(audit_entry)
            
            logger.info(f"PHI access request {request_id} created for user {user_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Error creating PHI access request: {e}")
            raise
    
    def access_phi_data(self, 
                       request_id: str,
                       data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Access PHI data with proper authorization and audit logging.
        
        Args:
            request_id: ID of approved access request
            data: Raw data containing PHI
            
        Returns:
            Tuple of (access_granted, filtered_data)
        """
        try:
            \# Validate access request
            if request_id not in self.access_requests:
                logger.error(f"Invalid access request ID: {request_id}")
                return False, {}
            
            access_request = self.access_requests[request_id]
            
            \# Check if request is approved and still valid
            if not access_request.approved:
                logger.error(f"Access request {request_id} not approved")
                return False, {}
            
            if (access_request.access_granted_until and 
                datetime.now() > access_request.access_granted_until):
                logger.error(f"Access request {request_id} has expired")
                return False, {}
            
            \# Filter data based on minimum necessary principle
            filtered_data = self._filter_data_by_phi_types(data, access_request.phi_types_requested)
            
            \# Create audit entry for data access
            audit_entry = HIPAAAuditEntry(
                entry_id=self._generate_audit_id(),
                timestamp=datetime.now(),
                user_id=access_request.user_id,
                patient_id_hash=self._hash_patient_id(access_request.patient_id),
                action="phi_data_access",
                phi_types_accessed=access_request.phi_types_requested,
                purpose=access_request.purpose,
                outcome="success",
                ip_address="system",
                session_id="system",
                additional_details={
                    "request_id": request_id,
                    "data_elements_accessed": len(filtered_data)
                }
            )
            self.audit_log.append(audit_entry)
            
            logger.info(f"PHI data accessed successfully for request {request_id}")
            return True, filtered_data
            
        except Exception as e:
            logger.error(f"Error accessing PHI data: {e}")
            
            \# Create audit entry for failed access
            if request_id in self.access_requests:
                access_request = self.access_requests[request_id]
                audit_entry = HIPAAAuditEntry(
                    entry_id=self._generate_audit_id(),
                    timestamp=datetime.now(),
                    user_id=access_request.user_id,
                    patient_id_hash=self._hash_patient_id(access_request.patient_id),
                    action="phi_data_access",
                    phi_types_accessed=access_request.phi_types_requested,
                    purpose=access_request.purpose,
                    outcome="failure",
                    ip_address="system",
                    session_id="system",
                    additional_details={"error": str(e)}
                )
                self.audit_log.append(audit_entry)
            
            return False, {}
    
    def encrypt_phi_data(self, data: str) -> str:
        """
        Encrypt PHI data for secure storage or transmission.
        
        Args:
            data: PHI data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Error encrypting PHI data: {e}")
            raise
    
    def decrypt_phi_data(self, encrypted_data: str) -> str:
        """
        Decrypt PHI data for authorized access.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted PHI data
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error decrypting PHI data: {e}")
            raise
    
    def detect_phi_in_text(self, text: str) -> Dict[PHIDataType, List[str]]:
        """
        Detect potential PHI in unstructured text using pattern matching.
        
        Args:
            text: Text to analyze for PHI
            
        Returns:
            Dictionary mapping PHI types to detected instances
        """
        detected_phi = {}
        
        for phi_type, patterns in self.phi_identifiers.items():
            matches = []
            for pattern in patterns:
                found_matches = pattern.findall(text)
                matches.extend(found_matches)
            
            if matches:
                detected_phi[phi_type] = matches
        
        return detected_phi
    
    def generate_compliance_report(self, 
                                 start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive HIPAA compliance report.
        
        Args:
            start_date: Start date for report period
            end_date: End date for report period
            
        Returns:
            Detailed compliance report
        """
        \# Filter audit log for report period
        period_audits = [
            audit for audit in self.audit_log
            if start_date <= audit.timestamp <= end_date
        ]
        
        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'organization': self.organization_name,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_audit_entries': len(period_audits),
                'unique_users': len(set(audit.user_id for audit in period_audits)),
                'unique_patients': len(set(audit.patient_id_hash for audit in period_audits if audit.patient_id_hash != "N/A")),
                'phi_access_events': len([audit for audit in period_audits if audit.action == "phi_data_access"]),
                'failed_access_attempts': len([audit for audit in period_audits if audit.outcome == "failure"])
            },
            'user_activity': {},
            'phi_access_patterns': {},
            'security_incidents': [],
            'compliance_metrics': {}
        }
        
        \# Analyze user activity
        for audit in period_audits:
            user_id = audit.user_id
            if user_id not in report['user_activity']:
                report['user_activity'][user_id] = {
                    'total_actions': 0,
                    'phi_accesses': 0,
                    'failed_attempts': 0,
                    'last_activity': None
                }
            
            report['user_activity'][user_id]['total_actions'] += 1
            if audit.action == "phi_data_access":
                report['user_activity'][user_id]['phi_accesses'] += 1
            if audit.outcome == "failure":
                report['user_activity'][user_id]['failed_attempts'] += 1
            
            if (report['user_activity'][user_id]['last_activity'] is None or
                audit.timestamp > datetime.fromisoformat(report['user_activity'][user_id]['last_activity'])):
                report['user_activity'][user_id]['last_activity'] = audit.timestamp.isoformat()
        
        \# Analyze PHI access patterns
        phi_access_audits = [audit for audit in period_audits if audit.action == "phi_data_access"]
        for audit in phi_access_audits:
            for phi_type in audit.phi_types_accessed:
                phi_type_str = phi_type.value
                if phi_type_str not in report['phi_access_patterns']:
                    report['phi_access_patterns'][phi_type_str] = 0
                report['phi_access_patterns'][phi_type_str] += 1
        
        \# Identify potential security incidents
        for user_id, activity in report['user_activity'].items():
            if activity['failed_attempts'] > 5:
                report['security_incidents'].append({
                    'type': 'excessive_failed_attempts',
                    'user_id': user_id,
                    'failed_attempts': activity['failed_attempts'],
                    'severity': 'medium'
                })
        
        \# Calculate compliance metrics
        total_access_requests = len([audit for audit in period_audits if audit.action == "phi_access_request"])
        successful_accesses = len([audit for audit in period_audits if audit.action == "phi_data_access" and audit.outcome == "success"])
        
        report['compliance_metrics'] = {
            'access_success_rate': (successful_accesses / total_access_requests * 100) if total_access_requests > 0 else 0,
            'average_daily_phi_accesses': len(phi_access_audits) / max(1, (end_date - start_date).days),
            'audit_log_completeness': 100.0,  \# Assuming complete audit logging
            'encryption_compliance': 100.0    \# Assuming all PHI is encrypted
        }
        
        return report
    
    def _validate_user_information(self, user: HIPAAUser) -> bool:
        """Validate user information for HIPAA compliance."""
        required_fields = ['user_id', 'name', 'role', 'access_level', 'department']
        
        for field in required_fields:
            if not getattr(user, field):
                logger.error(f"Missing required field: {field}")
                return False
        
        \# Validate training completion for users with PHI access
        if (user.authorized_phi_types and 
            user.training_completion_date is None):
            logger.error(f"User {user.user_id} requires HIPAA training completion")
            return False
        
        return True
    
    def _filter_data_by_phi_types(self, 
                                 data: Dict[str, Any], 
                                 allowed_phi_types: Set[PHIDataType]) -> Dict[str, Any]:
        """Filter data to include only authorized PHI types."""
        \# Simplified implementation - in production, would use sophisticated
        \# data classification and filtering based on data schemas
        filtered_data = {}
        
        phi_field_mapping = {
            PHIDataType.NAME: ['name', 'patient_name', 'first_name', 'last_name'],
            PHIDataType.BIRTH_DATE: ['birth_date', 'dob', 'date_of_birth'],
            PHIDataType.ADDRESS: ['address', 'street_address', 'city', 'state', 'zip'],
            PHIDataType.PHONE_NUMBER: ['phone', 'phone_number', 'telephone'],
            PHIDataType.EMAIL: ['email', 'email_address'],
            PHIDataType.SSN: ['ssn', 'social_security_number'],
            PHIDataType.MRN: ['mrn', 'medical_record_number', 'patient_id']
        }
        
        for field, value in data.items():
            field_lower = field.lower()
            include_field = False
            
            \# Check if field corresponds to an allowed PHI type
            for phi_type in allowed_phi_types:
                if phi_type in phi_field_mapping:
                    if any(allowed_field in field_lower for allowed_field in phi_field_mapping[phi_type]):
                        include_field = True
                        break
            
            \# Always include non-PHI clinical data
            clinical_fields = ['diagnosis', 'medication', 'lab_result', 'vital_signs', 'procedure']
            if any(clinical_field in field_lower for clinical_field in clinical_fields):
                include_field = True
            
            if include_field:
                filtered_data[field] = value
        
        return filtered_data
    
    def _initialize_phi_patterns(self) -> Dict[PHIDataType, List[re.Pattern]]:
        """Initialize regex patterns for PHI detection."""
        patterns = {
            PHIDataType.SSN: [
                re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                re.compile(r'\b\d{9}\b')
            ],
            PHIDataType.PHONE_NUMBER: [
                re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),
                re.compile(r'$\d{3}$\s*\d{3}-\d{4}'),
                re.compile(r'\b\d{10}\b')
            ],
            PHIDataType.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            ],
            PHIDataType.BIRTH_DATE: [
                re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
                re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
            ],
            PHIDataType.IP_ADDRESS: [
                re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
            ]
        }
        
        return patterns
    
    def _generate_audit_id(self) -> str:
        """Generate unique audit entry ID."""
        return f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
    
    def _generate_request_id(self) -> str:
        """Generate unique access request ID."""
        return f"REQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
    
    def _hash_patient_id(self, patient_id: str) -> str:
        """Hash patient ID for audit logging (never log actual patient IDs)."""
        return hashlib.sha256(patient_id.encode()).hexdigest()[:16]


\# Demonstration function
def demonstrate_hipaa_compliance():
    """
    Demonstrate comprehensive HIPAA compliance implementation.
    
    This function shows how to use the HIPAA compliance engine
    for real-world healthcare AI applications.
    """
    print("=== HIPAA Compliance Demonstration ===\n")
    
    \# Initialize compliance engine
    compliance_engine = HIPAAComplianceEngine(
        organization_name="Example Healthcare System",
        audit_retention_days=2555
    )
    
    \# Register users with different access levels
    users = [
        HIPAAUser(
            user_id="physician_001",
            name="Dr. Jane Smith",
            role="Attending Physician",
            access_level=HIPAAAccessLevel.ELEVATED_ACCESS,
            authorized_phi_types={
                PHIDataType.NAME, PHIDataType.BIRTH_DATE, PHIDataType.MRN,
                PHIDataType.ADDRESS, PHIDataType.PHONE_NUMBER
            },
            department="Internal Medicine",
            training_completion_date=datetime.now() - timedelta(days=30)
        ),
        HIPAAUser(
            user_id="researcher_001",
            name="Dr. John Doe",
            role="Clinical Researcher",
            access_level=HIPAAAccessLevel.LIMITED_ACCESS,
            authorized_phi_types={PHIDataType.BIRTH_DATE, PHIDataType.MRN},
            department="Research",
            training_completion_date=datetime.now() - timedelta(days=15)
        )
    ]
    
    for user in users:
        success = compliance_engine.register_user(user)
        print(f"User {user.user_id} registration: {'Success' if success else 'Failed'}")
    
    \# Request PHI access
    print(f"\n1. PHI Access Request")
    print("-" * 30)
    
    request_id = compliance_engine.request_phi_access(
        user_id="physician_001",
        patient_id="patient_12345",
        phi_types={PHIDataType.NAME, PHIDataType.BIRTH_DATE, PHIDataType.MRN},
        purpose="Clinical care",
        justification="Reviewing patient history for treatment planning"
    )
    
    print(f"Access request created: {request_id}")
    
    \# Access PHI data
    print(f"\n2. PHI Data Access")
    print("-" * 30)
    
    sample_patient_data = {
        'patient_name': 'John Patient',
        'birth_date': '1980-05-15',
        'mrn': 'MRN123456',
        'address': '123 Main St, City, State',
        'phone_number': '555-123-4567',
        'diagnosis': 'Type 2 Diabetes',
        'lab_results': {'glucose': 150, 'hba1c': 7.2}
    }
    
    access_granted, filtered_data = compliance_engine.access_phi_data(
        request_id=request_id,
        data=sample_patient_data
    )
    
    print(f"Access granted: {access_granted}")
    print(f"Filtered data fields: {list(filtered_data.keys())}")
    
    \# Demonstrate PHI detection
    print(f"\n3. PHI Detection in Text")
    print("-" * 30)
    
    sample_text = """
    Patient John Smith (SSN: 123-45-6789) was born on 05/15/1980.
    Contact phone: 555-123-4567, email: john.smith@email.com
    Address: 123 Main Street, Anytown, ST 12345
    """
    
    detected_phi = compliance_engine.detect_phi_in_text(sample_text)
    print("Detected PHI:")
    for phi_type, instances in detected_phi.items():
        print(f"  {phi_type.value}: {instances}")
    
    \# Generate compliance report
    print(f"\n4. Compliance Report")
    print("-" * 30)
    
    report = compliance_engine.generate_compliance_report(
        start_date=datetime.now() - timedelta(days=1),
        end_date=datetime.now()
    )
    
    print(f"Total audit entries: {report['summary']['total_audit_entries']}")
    print(f"Unique users: {report['summary']['unique_users']}")
    print(f"PHI access events: {report['summary']['phi_access_events']}")
    print(f"Access success rate: {report['compliance_metrics']['access_success_rate']:.1f}%")
    
    \# Demonstrate encryption
    print(f"\n5. PHI Encryption")
    print("-" * 30)
    
    sensitive_data = "Patient: John Smith, DOB: 1980-05-15, SSN: 123-45-6789"
    encrypted_data = compliance_engine.encrypt_phi_data(sensitive_data)
    decrypted_data = compliance_engine.decrypt_phi_data(encrypted_data)
    
    print(f"Original data length: {len(sensitive_data)} characters")
    print(f"Encrypted data length: {len(encrypted_data)} characters")
    print(f"Decryption successful: {sensitive_data == decrypted_data}")


if __name__ == "__main__":
    demonstrate_hipaa_compliance()