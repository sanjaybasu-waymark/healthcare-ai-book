#!/usr/bin/env python3
"""
Healthcare AI Implementation Guide - Chapter 3 Examples
Healthcare Data Engineering

This module provides comprehensive examples for healthcare data engineering
including FHIR processing, real-time pipelines, and data quality frameworks.

Author: Healthcare AI Implementation Guide
License: MIT
"""

import pandas as pd
import numpy as np
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FHIRResource:
    """
    Represents a FHIR resource with validation and processing capabilities.
    """
    resource_type: str
    id: str
    data: Dict[str, Any]
    
    def validate(self) -> bool:
        """Validate FHIR resource structure."""
        required_fields = ['resourceType', 'id']
        return all(field in self.data for field in required_fields)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.data, indent=2)


class FHIRProcessor:
    """
    Comprehensive FHIR data processor for healthcare interoperability.
    
    This class handles FHIR resource processing, validation, and transformation
    for healthcare AI applications.
    """
    
    def __init__(self):
        """Initialize FHIR processor."""
        self.processed_resources = []
        self.validation_errors = []
        
        # FHIR resource schemas (simplified)
        self.resource_schemas = {
            'Patient': {
                'required': ['resourceType', 'id'],
                'optional': ['identifier', 'name', 'gender', 'birthDate', 'address']
            },
            'Observation': {
                'required': ['resourceType', 'status', 'code', 'subject'],
                'optional': ['valueQuantity', 'valueString', 'effectiveDateTime']
            },
            'Condition': {
                'required': ['resourceType', 'subject', 'code'],
                'optional': ['clinicalStatus', 'verificationStatus', 'onsetDateTime']
            }
        }
    
    def validate_fhir_resource(self, resource_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate FHIR resource against schema.
        
        Parameters:
        -----------
        resource_data : dict
            FHIR resource data
            
        Returns:
        --------
        tuple : (is_valid, error_messages)
        """
        errors = []
        
        # Check if resourceType exists
        if 'resourceType' not in resource_data:
            errors.append("Missing required field: resourceType")
            return False, errors
        
        resource_type = resource_data['resourceType']
        
        # Check if we have a schema for this resource type
        if resource_type not in self.resource_schemas:
            errors.append(f"Unknown resource type: {resource_type}")
            return False, errors
        
        schema = self.resource_schemas[resource_type]
        
        # Check required fields
        for field in schema['required']:
            if field not in resource_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate specific field formats
        if resource_type == 'Patient':
            if 'birthDate' in resource_data:
                try:
                    datetime.strptime(resource_data['birthDate'], '%Y-%m-%d')
                except ValueError:
                    errors.append("Invalid birthDate format. Expected YYYY-MM-DD")
        
        if resource_type == 'Observation':
            if 'status' in resource_data:
                valid_statuses = ['registered', 'preliminary', 'final', 'amended', 'cancelled']
                if resource_data['status'] not in valid_statuses:
                    errors.append(f"Invalid observation status: {resource_data['status']}")
        
        return len(errors) == 0, errors
    
    def process_fhir_bundle(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a FHIR bundle containing multiple resources.
        
        Parameters:
        -----------
        bundle_data : dict
            FHIR Bundle resource
            
        Returns:
        --------
        dict : Processing results
        """
        results = {
            'processed_count': 0,
            'error_count': 0,
            'resources_by_type': {},
            'validation_errors': []
        }
        
        if 'entry' not in bundle_data:
            results['validation_errors'].append("Bundle missing 'entry' field")
            return results
        
        for entry in bundle_data['entry']:
            if 'resource' not in entry:
                results['error_count'] += 1
                results['validation_errors'].append("Bundle entry missing 'resource' field")
                continue
            
            resource = entry['resource']
            is_valid, errors = self.validate_fhir_resource(resource)
            
            if is_valid:
                resource_type = resource['resourceType']
                if resource_type not in results['resources_by_type']:
                    results['resources_by_type'][resource_type] = []
                
                results['resources_by_type'][resource_type].append(resource)
                results['processed_count'] += 1
            else:
                results['error_count'] += 1
                results['validation_errors'].extend(errors)
        
        return results
    
    def transform_to_tabular(self, resources: List[Dict[str, Any]], resource_type: str) -> pd.DataFrame:
        """
        Transform FHIR resources to tabular format for analysis.
        
        Parameters:
        -----------
        resources : list
            List of FHIR resources
        resource_type : str
            Type of FHIR resource
            
        Returns:
        --------
        DataFrame : Tabular representation of resources
        """
        if resource_type == 'Patient':
            return self._transform_patients(resources)
        elif resource_type == 'Observation':
            return self._transform_observations(resources)
        elif resource_type == 'Condition':
            return self._transform_conditions(resources)
        else:
            raise ValueError(f"Unsupported resource type: {resource_type}")
    
    def _transform_patients(self, patients: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform Patient resources to DataFrame."""
        patient_data = []
        
        for patient in patients:
            row = {
                'patient_id': patient.get('id', ''),
                'gender': patient.get('gender', ''),
                'birth_date': patient.get('birthDate', ''),
            }
            
            # Extract name
            if 'name' in patient and len(patient['name']) > 0:
                name = patient['name'][0]
                if 'family' in name:
                    row['family_name'] = name['family']
                if 'given' in name and len(name['given']) > 0:
                    row['given_name'] = name['given'][0]
            
            # Extract identifiers
            if 'identifier' in patient:
                for identifier in patient['identifier']:
                    if identifier.get('system') == 'http://hospital.example.com/patient-ids':
                        row['mrn'] = identifier.get('value', '')
            
            patient_data.append(row)
        
        return pd.DataFrame(patient_data)
    
    def _transform_observations(self, observations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform Observation resources to DataFrame."""
        obs_data = []
        
        for obs in observations:
            row = {
                'observation_id': obs.get('id', ''),
                'patient_id': '',
                'status': obs.get('status', ''),
                'effective_date': obs.get('effectiveDateTime', ''),
                'code': '',
                'display': '',
                'value': '',
                'unit': ''
            }
            
            # Extract patient reference
            if 'subject' in obs and 'reference' in obs['subject']:
                row['patient_id'] = obs['subject']['reference'].replace('Patient/', '')
            
            # Extract code information
            if 'code' in obs and 'coding' in obs['code']:
                coding = obs['code']['coding'][0]
                row['code'] = coding.get('code', '')
                row['display'] = coding.get('display', '')
            
            # Extract value
            if 'valueQuantity' in obs:
                value_qty = obs['valueQuantity']
                row['value'] = value_qty.get('value', '')
                row['unit'] = value_qty.get('unit', '')
            elif 'valueString' in obs:
                row['value'] = obs['valueString']
            
            obs_data.append(row)
        
        return pd.DataFrame(obs_data)
    
    def _transform_conditions(self, conditions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform Condition resources to DataFrame."""
        condition_data = []
        
        for condition in conditions:
            row = {
                'condition_id': condition.get('id', ''),
                'patient_id': '',
                'clinical_status': '',
                'verification_status': '',
                'code': '',
                'display': '',
                'onset_date': condition.get('onsetDateTime', '')
            }
            
            # Extract patient reference
            if 'subject' in condition and 'reference' in condition['subject']:
                row['patient_id'] = condition['subject']['reference'].replace('Patient/', '')
            
            # Extract clinical status
            if 'clinicalStatus' in condition and 'coding' in condition['clinicalStatus']:
                row['clinical_status'] = condition['clinicalStatus']['coding'][0].get('code', '')
            
            # Extract verification status
            if 'verificationStatus' in condition and 'coding' in condition['verificationStatus']:
                row['verification_status'] = condition['verificationStatus']['coding'][0].get('code', '')
            
            # Extract condition code
            if 'code' in condition and 'coding' in condition['code']:
                coding = condition['code']['coding'][0]
                row['code'] = coding.get('code', '')
                row['display'] = coding.get('display', '')
            
            condition_data.append(row)
        
        return pd.DataFrame(condition_data)


class RealTimeHealthcareDataPipeline:
    """
    Real-time healthcare data processing pipeline.
    
    This class implements a scalable pipeline for processing streaming
    healthcare data with quality checks and alerting.
    """
    
    def __init__(self, db_path: str = "healthcare_pipeline.db"):
        """Initialize the real-time pipeline."""
        self.db_path = db_path
        self.setup_database()
        self.alert_thresholds = {
            'heart_rate': {'min': 50, 'max': 120},
            'blood_pressure_systolic': {'min': 90, 'max': 180},
            'temperature': {'min': 96.0, 'max': 102.0},
            'oxygen_saturation': {'min': 90, 'max': 100}
        }
        self.processed_count = 0
        self.alert_count = 0
    
    def setup_database(self):
        """Setup SQLite database for pipeline storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_vitals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                vital_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                alert_triggered BOOLEAN DEFAULT FALSE,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def process_vital_signs_stream(self, vital_signs_data: List[Dict[str, Any]]):
        """
        Process streaming vital signs data.
        
        Parameters:
        -----------
        vital_signs_data : list
            List of vital signs measurements
        """
        tasks = []
        
        for vital_data in vital_signs_data:
            task = asyncio.create_task(self.process_single_vital(vital_data))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log processing results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        logger.info(f"Processed {successful} vital signs, {failed} failed")
        self.processed_count += successful
    
    async def process_single_vital(self, vital_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single vital sign measurement.
        
        Parameters:
        -----------
        vital_data : dict
            Single vital sign measurement
            
        Returns:
        --------
        dict : Processing result
        """
        try:
            # Validate required fields
            required_fields = ['patient_id', 'vital_type', 'value', 'timestamp']
            if not all(field in vital_data for field in required_fields):
                raise ValueError(f"Missing required fields: {required_fields}")
            
            patient_id = vital_data['patient_id']
            vital_type = vital_data['vital_type']
            value = float(vital_data['value'])
            timestamp = vital_data['timestamp']
            unit = vital_data.get('unit', '')
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO patient_vitals (patient_id, timestamp, vital_type, value, unit)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, timestamp, vital_type, value, unit))
            
            vital_id = cursor.lastrowid
            
            # Check for alerts
            alert_triggered = await self.check_vital_alerts(patient_id, vital_type, value)
            
            if alert_triggered:
                cursor.execute('''
                    UPDATE patient_vitals SET alert_triggered = TRUE WHERE id = ?
                ''', (vital_id,))
                self.alert_count += 1
            
            conn.commit()
            conn.close()
            
            return {
                'status': 'success',
                'vital_id': vital_id,
                'alert_triggered': alert_triggered
            }
            
        except Exception as e:
            logger.error(f"Error processing vital sign: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def check_vital_alerts(self, patient_id: str, vital_type: str, value: float) -> bool:
        """
        Check if vital sign triggers an alert.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        vital_type : str
            Type of vital sign
        value : float
            Vital sign value
            
        Returns:
        --------
        bool : True if alert triggered
        """
        if vital_type not in self.alert_thresholds:
            return False
        
        thresholds = self.alert_thresholds[vital_type]
        
        if value < thresholds['min'] or value > thresholds['max']:
            # Determine severity
            if vital_type == 'heart_rate':
                if value < 40 or value > 150:
                    severity = 'CRITICAL'
                else:
                    severity = 'WARNING'
            elif vital_type == 'blood_pressure_systolic':
                if value < 70 or value > 200:
                    severity = 'CRITICAL'
                else:
                    severity = 'WARNING'
            else:
                severity = 'WARNING'
            
            # Create alert
            await self.create_alert(
                patient_id=patient_id,
                alert_type=f"{vital_type}_abnormal",
                message=f"{vital_type.replace('_', ' ').title()} value {value} is outside normal range",
                severity=severity
            )
            
            return True
        
        return False
    
    async def create_alert(self, patient_id: str, alert_type: str, message: str, severity: str):
        """Create a new alert in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (patient_id, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, alert_type, message, severity))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"ALERT [{severity}] for patient {patient_id}: {message}")
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        conn = sqlite3.connect(self.db_path)
        
        # Get vital signs statistics
        vitals_df = pd.read_sql_query('''
            SELECT vital_type, COUNT(*) as count, AVG(value) as avg_value
            FROM patient_vitals
            GROUP BY vital_type
        ''', conn)
        
        # Get alert statistics
        alerts_df = pd.read_sql_query('''
            SELECT severity, COUNT(*) as count
            FROM alerts
            GROUP BY severity
        ''', conn)
        
        conn.close()
        
        return {
            'total_processed': self.processed_count,
            'total_alerts': self.alert_count,
            'vitals_by_type': vitals_df.to_dict('records'),
            'alerts_by_severity': alerts_df.to_dict('records')
        }


class HealthcareDataQualityFramework:
    """
    Comprehensive data quality framework for healthcare data.
    
    This class implements data quality assessment, monitoring, and
    improvement strategies specific to healthcare applications.
    """
    
    def __init__(self):
        """Initialize data quality framework."""
        self.quality_rules = self._define_quality_rules()
        self.quality_reports = []
    
    def _define_quality_rules(self) -> Dict[str, Dict]:
        """Define healthcare-specific data quality rules."""
        return {
            'completeness': {
                'critical_fields': ['patient_id', 'timestamp', 'value'],
                'threshold': 0.95  # 95% completeness required
            },
            'validity': {
                'ranges': {
                    'age': (0, 120),
                    'heart_rate': (20, 300),
                    'blood_pressure_systolic': (50, 300),
                    'blood_pressure_diastolic': (20, 200),
                    'temperature': (90, 110),
                    'weight': (1, 500),  # kg
                    'height': (30, 250)  # cm
                },
                'formats': {
                    'patient_id': r'^[A-Z0-9]{6,12}$',
                    'timestamp': r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                }
            },
            'consistency': {
                'relationships': [
                    ('systolic_bp', 'diastolic_bp', 'systolic >= diastolic'),
                    ('height', 'weight', 'bmi_reasonable')
                ]
            },
            'timeliness': {
                'max_delay_hours': 24,  # Data should be no more than 24 hours old
                'expected_frequency': {
                    'vital_signs': 'hourly',
                    'lab_results': 'daily'
                }
            }
        }
    
    def assess_data_quality(self, data: pd.DataFrame, data_type: str = 'general') -> Dict[str, Any]:
        """
        Comprehensive data quality assessment.
        
        Parameters:
        -----------
        data : DataFrame
            Healthcare dataset to assess
        data_type : str
            Type of healthcare data (vital_signs, lab_results, etc.)
            
        Returns:
        --------
        dict : Comprehensive quality assessment report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'total_records': len(data),
            'quality_scores': {},
            'issues': [],
            'recommendations': []
        }
        
        # Completeness assessment
        completeness_score, completeness_issues = self._assess_completeness(data)
        report['quality_scores']['completeness'] = completeness_score
        report['issues'].extend(completeness_issues)
        
        # Validity assessment
        validity_score, validity_issues = self._assess_validity(data)
        report['quality_scores']['validity'] = validity_score
        report['issues'].extend(validity_issues)
        
        # Consistency assessment
        consistency_score, consistency_issues = self._assess_consistency(data)
        report['quality_scores']['consistency'] = consistency_score
        report['issues'].extend(consistency_issues)
        
        # Timeliness assessment (if timestamp column exists)
        if 'timestamp' in data.columns:
            timeliness_score, timeliness_issues = self._assess_timeliness(data)
            report['quality_scores']['timeliness'] = timeliness_score
            report['issues'].extend(timeliness_issues)
        
        # Overall quality score
        scores = list(report['quality_scores'].values())
        report['overall_quality_score'] = np.mean(scores) if scores else 0
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        self.quality_reports.append(report)
        return report
    
    def _assess_completeness(self, data: pd.DataFrame) -> tuple[float, List[str]]:
        """Assess data completeness."""
        issues = []
        
        # Check critical fields
        critical_fields = self.quality_rules['completeness']['critical_fields']
        missing_critical = []
        
        for field in critical_fields:
            if field in data.columns:
                missing_pct = data[field].isnull().mean()
                if missing_pct > 0:
                    issues.append(f"Critical field '{field}' has {missing_pct:.1%} missing values")
                    if missing_pct > 0.05:  # More than 5% missing
                        missing_critical.append(field)
        
        # Overall completeness score
        total_missing = data.isnull().sum().sum()
        total_values = data.size
        completeness_score = 1 - (total_missing / total_values) if total_values > 0 else 0
        
        if completeness_score < self.quality_rules['completeness']['threshold']:
            issues.append(f"Overall completeness {completeness_score:.1%} below threshold")
        
        return completeness_score, issues
    
    def _assess_validity(self, data: pd.DataFrame) -> tuple[float, List[str]]:
        """Assess data validity."""
        issues = []
        validity_scores = []
        
        # Check value ranges
        ranges = self.quality_rules['validity']['ranges']
        for field, (min_val, max_val) in ranges.items():
            if field in data.columns:
                out_of_range = ((data[field] < min_val) | (data[field] > max_val)).sum()
                total_non_null = data[field].notna().sum()
                
                if total_non_null > 0:
                    validity_rate = 1 - (out_of_range / total_non_null)
                    validity_scores.append(validity_rate)
                    
                    if out_of_range > 0:
                        issues.append(f"Field '{field}' has {out_of_range} values outside range [{min_val}, {max_val}]")
        
        # Check format patterns
        formats = self.quality_rules['validity']['formats']
        for field, pattern in formats.items():
            if field in data.columns:
                import re
                valid_format = data[field].astype(str).str.match(pattern, na=False).sum()
                total_non_null = data[field].notna().sum()
                
                if total_non_null > 0:
                    format_validity = valid_format / total_non_null
                    validity_scores.append(format_validity)
                    
                    if format_validity < 0.95:
                        issues.append(f"Field '{field}' has {(1-format_validity):.1%} invalid format values")
        
        overall_validity = np.mean(validity_scores) if validity_scores else 1.0
        return overall_validity, issues
    
    def _assess_consistency(self, data: pd.DataFrame) -> tuple[float, List[str]]:
        """Assess data consistency."""
        issues = []
        consistency_scores = []
        
        # Check blood pressure consistency
        if 'systolic_bp' in data.columns and 'diastolic_bp' in data.columns:
            bp_data = data[['systolic_bp', 'diastolic_bp']].dropna()
            if len(bp_data) > 0:
                inconsistent_bp = (bp_data['systolic_bp'] <= bp_data['diastolic_bp']).sum()
                consistency_rate = 1 - (inconsistent_bp / len(bp_data))
                consistency_scores.append(consistency_rate)
                
                if inconsistent_bp > 0:
                    issues.append(f"{inconsistent_bp} records have systolic BP <= diastolic BP")
        
        # Check BMI reasonableness
        if all(col in data.columns for col in ['height', 'weight']):
            hw_data = data[['height', 'weight']].dropna()
            if len(hw_data) > 0:
                # Calculate BMI (assuming height in cm, weight in kg)
                bmi = hw_data['weight'] / ((hw_data['height'] / 100) ** 2)
                unreasonable_bmi = ((bmi < 10) | (bmi > 80)).sum()
                bmi_consistency = 1 - (unreasonable_bmi / len(hw_data))
                consistency_scores.append(bmi_consistency)
                
                if unreasonable_bmi > 0:
                    issues.append(f"{unreasonable_bmi} records have unreasonable BMI values")
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        return overall_consistency, issues
    
    def _assess_timeliness(self, data: pd.DataFrame) -> tuple[float, List[str]]:
        """Assess data timeliness."""
        issues = []
        
        # Convert timestamp column to datetime
        try:
            timestamps = pd.to_datetime(data['timestamp'])
            now = datetime.now()
            
            # Check for future timestamps
            future_timestamps = (timestamps > now).sum()
            if future_timestamps > 0:
                issues.append(f"{future_timestamps} records have future timestamps")
            
            # Check for very old data
            max_delay = timedelta(hours=self.quality_rules['timeliness']['max_delay_hours'])
            old_data = (now - timestamps > max_delay).sum()
            
            timeliness_score = 1 - (old_data / len(data)) if len(data) > 0 else 1.0
            
            if old_data > 0:
                issues.append(f"{old_data} records are older than {max_delay}")
            
        except Exception as e:
            issues.append(f"Error processing timestamps: {e}")
            timeliness_score = 0.0
        
        return timeliness_score, issues
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Completeness recommendations
        if report['quality_scores'].get('completeness', 1.0) < 0.9:
            recommendations.append("Implement data validation at point of entry to improve completeness")
            recommendations.append("Review data collection processes for critical fields")
        
        # Validity recommendations
        if report['quality_scores'].get('validity', 1.0) < 0.9:
            recommendations.append("Add range validation checks for numeric fields")
            recommendations.append("Implement format validation for identifier fields")
        
        # Consistency recommendations
        if report['quality_scores'].get('consistency', 1.0) < 0.9:
            recommendations.append("Add cross-field validation rules")
            recommendations.append("Implement automated consistency checks in data pipeline")
        
        # Timeliness recommendations
        if report['quality_scores'].get('timeliness', 1.0) < 0.9:
            recommendations.append("Review data ingestion delays")
            recommendations.append("Implement real-time data quality monitoring")
        
        return recommendations
    
    def generate_quality_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for quality monitoring dashboard."""
        if not self.quality_reports:
            return {'message': 'No quality reports available'}
        
        latest_report = self.quality_reports[-1]
        
        # Historical trend data
        historical_scores = []
        for report in self.quality_reports[-10:]:  # Last 10 reports
            historical_scores.append({
                'timestamp': report['timestamp'],
                'overall_score': report['overall_quality_score'],
                'completeness': report['quality_scores'].get('completeness', 0),
                'validity': report['quality_scores'].get('validity', 0),
                'consistency': report['quality_scores'].get('consistency', 0),
                'timeliness': report['quality_scores'].get('timeliness', 0)
            })
        
        return {
            'current_quality': latest_report,
            'historical_trends': historical_scores,
            'total_reports': len(self.quality_reports)
        }


def generate_sample_healthcare_data() -> Dict[str, pd.DataFrame]:
    """Generate sample healthcare data for demonstration."""
    np.random.seed(42)
    
    # Generate patient data
    n_patients = 100
    patients = []
    
    for i in range(n_patients):
        patient = {
            'patient_id': f'P{i:06d}',
            'age': np.random.randint(18, 90),
            'gender': np.random.choice(['M', 'F']),
            'height': np.random.normal(170, 15),  # cm
            'weight': np.random.normal(75, 20),   # kg
        }
        patients.append(patient)
    
    patients_df = pd.DataFrame(patients)
    
    # Generate vital signs data
    vital_signs = []
    
    for patient_id in patients_df['patient_id']:
        # Generate 24 hours of hourly vital signs
        for hour in range(24):
            timestamp = datetime.now() - timedelta(hours=23-hour)
            
            # Add some realistic variation and occasional outliers
            base_hr = 70 + np.random.normal(0, 10)
            if np.random.random() < 0.05:  # 5% chance of outlier
                base_hr += np.random.choice([-30, 40])
            
            vital_signs.append({
                'patient_id': patient_id,
                'timestamp': timestamp.isoformat(),
                'vital_type': 'heart_rate',
                'value': max(30, min(200, base_hr)),
                'unit': 'bpm'
            })
            
            # Blood pressure
            systolic = 120 + np.random.normal(0, 15)
            diastolic = 80 + np.random.normal(0, 10)
            
            # Ensure systolic > diastolic (mostly)
            if np.random.random() > 0.02:  # 98% of the time
                if systolic <= diastolic:
                    systolic = diastolic + 20
            
            vital_signs.extend([
                {
                    'patient_id': patient_id,
                    'timestamp': timestamp.isoformat(),
                    'vital_type': 'blood_pressure_systolic',
                    'value': max(70, min(250, systolic)),
                    'unit': 'mmHg'
                },
                {
                    'patient_id': patient_id,
                    'timestamp': timestamp.isoformat(),
                    'vital_type': 'blood_pressure_diastolic',
                    'value': max(40, min(150, diastolic)),
                    'unit': 'mmHg'
                }
            ])
    
    vitals_df = pd.DataFrame(vital_signs)
    
    return {
        'patients': patients_df,
        'vital_signs': vitals_df
    }


async def main():
    """
    Main function demonstrating healthcare data engineering workflows.
    """
    print("Healthcare AI Implementation Guide - Chapter 3 Examples")
    print("Healthcare Data Engineering")
    print("=" * 70)
    
    # 1. FHIR Processing Example
    print("\n1. FHIR Resource Processing...")
    
    # Create sample FHIR bundle
    sample_bundle = {
        "resourceType": "Bundle",
        "id": "example-bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "patient-001",
                    "identifier": [
                        {
                            "system": "http://hospital.example.com/patient-ids",
                            "value": "MRN123456"
                        }
                    ],
                    "name": [
                        {
                            "family": "Smith",
                            "given": ["John"]
                        }
                    ],
                    "gender": "male",
                    "birthDate": "1980-01-15"
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-001",
                    "status": "final",
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "8480-6",
                                "display": "Systolic blood pressure"
                            }
                        ]
                    },
                    "subject": {
                        "reference": "Patient/patient-001"
                    },
                    "effectiveDateTime": "2024-01-15T10:30:00Z",
                    "valueQuantity": {
                        "value": 120,
                        "unit": "mmHg"
                    }
                }
            }
        ]
    }
    
    # Process FHIR bundle
    fhir_processor = FHIRProcessor()
    bundle_results = fhir_processor.process_fhir_bundle(sample_bundle)
    
    print(f"FHIR Bundle Processing Results:")
    print(f"  Processed: {bundle_results['processed_count']} resources")
    print(f"  Errors: {bundle_results['error_count']}")
    print(f"  Resource types: {list(bundle_results['resources_by_type'].keys())}")
    
    # Transform to tabular format
    if 'Patient' in bundle_results['resources_by_type']:
        patients_df = fhir_processor.transform_to_tabular(
            bundle_results['resources_by_type']['Patient'], 'Patient'
        )
        print(f"  Transformed {len(patients_df)} patients to tabular format")
    
    # 2. Real-time Data Pipeline Example
    print("\n2. Real-time Healthcare Data Pipeline...")
    
    # Generate sample streaming data
    sample_data = generate_sample_healthcare_data()
    
    # Convert vital signs to streaming format
    streaming_vitals = sample_data['vital_signs'].to_dict('records')[:50]  # First 50 records
    
    # Process with real-time pipeline
    pipeline = RealTimeHealthcareDataPipeline()
    await pipeline.process_vital_signs_stream(streaming_vitals)
    
    # Get pipeline statistics
    stats = pipeline.get_pipeline_statistics()
    print(f"Pipeline Statistics:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  Vitals by type: {len(stats['vitals_by_type'])} types")
    
    # 3. Data Quality Assessment
    print("\n3. Data Quality Assessment...")
    
    # Assess quality of sample data
    quality_framework = HealthcareDataQualityFramework()
    
    # Add some data quality issues for demonstration
    vitals_with_issues = sample_data['vital_signs'].copy()
    
    # Introduce missing values
    vitals_with_issues.loc[0:5, 'value'] = np.nan
    
    # Introduce invalid values
    vitals_with_issues.loc[10:12, 'value'] = -999
    
    # Assess quality
    quality_report = quality_framework.assess_data_quality(vitals_with_issues, 'vital_signs')
    
    print(f"Data Quality Assessment:")
    print(f"  Overall quality score: {quality_report['overall_quality_score']:.1%}")
    print(f"  Completeness: {quality_report['quality_scores']['completeness']:.1%}")
    print(f"  Validity: {quality_report['quality_scores']['validity']:.1%}")
    print(f"  Consistency: {quality_report['quality_scores']['consistency']:.1%}")
    print(f"  Issues found: {len(quality_report['issues'])}")
    
    if quality_report['issues']:
        print("  Top issues:")
        for issue in quality_report['issues'][:3]:
            print(f"    • {issue}")
    
    print(f"  Recommendations: {len(quality_report['recommendations'])}")
    if quality_report['recommendations']:
        print("  Top recommendations:")
        for rec in quality_report['recommendations'][:2]:
            print(f"    • {rec}")
    
    # 4. Save Results
    print("\n4. Saving Results...")
    
    # Save sample data
    sample_data['patients'].to_csv('sample_patients.csv', index=False)
    sample_data['vital_signs'].to_csv('sample_vital_signs.csv', index=False)
    
    # Save quality report
    with open('quality_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    print("✓ Sample data saved as 'sample_patients.csv' and 'sample_vital_signs.csv'")
    print("✓ Quality report saved as 'quality_report.json'")
    print("✓ Pipeline database saved as 'healthcare_pipeline.db'")
    
    print(f"\nChapter 3 examples completed successfully!")
    print("Ready to proceed to Chapter 4: Structured Machine Learning")


if __name__ == "__main__":
    asyncio.run(main())
