"""
Chapter 3 - Example 2
Extracted from Healthcare AI Implementation Guide
"""

"""
Comprehensive Healthcare Data Architecture Implementation

This implementation provides a scalable, secure healthcare data architecture
designed to support clinical operations, population health management, and
AI system development with full HIPAA compliance and clinical workflow integration.

Author: Sanjay Basu MD PhD
License: MIT
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import uuid
from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import redis
from kafka import KafkaProducer, KafkaConsumer
from elasticsearch import Elasticsearch
import boto3
from cryptography.fernet import Fernet
import hashlib
import jwt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncpg
import aioredis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

\# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

\# Metrics for monitoring
REQUEST_COUNT = Counter('healthcare_requests_total', 'Total healthcare data requests', ['operation', 'resource_type'])
REQUEST_DURATION = Histogram('healthcare_request_duration_seconds', 'Healthcare request duration')
ACTIVE_CONNECTIONS = Gauge('healthcare_active_connections', 'Active database connections')
DATA_QUALITY_SCORE = Gauge('healthcare_data_quality_score', 'Data quality score', ['resource_type'])

class DataTier(Enum):
    """Data storage tiers for healthcare data lifecycle management"""
    HOT = "hot"          \# Frequently accessed, high-performance storage
    WARM = "warm"        \# Occasionally accessed, balanced performance/cost
    COLD = "cold"        \# Rarely accessed, cost-optimized storage
    ARCHIVE = "archive"  \# Long-term retention, compliance storage

class SecurityLevel(Enum):
    """Security levels for healthcare data classification"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PHI = "phi"          \# Protected Health Information
    SENSITIVE_PHI = "sensitive_phi"  \# Highly sensitive PHI (mental health, etc.)

@dataclass
class DataGovernancePolicy:
    """Data governance policy configuration"""
    retention_period_days: int
    encryption_required: bool
    access_logging_required: bool
    data_classification: SecurityLevel
    allowed_regions: List[str]
    backup_frequency_hours: int
    audit_frequency_days: int
    data_quality_threshold: float = 0.95

class HealthcareDataArchitecture:
    """
    Comprehensive healthcare data architecture with multi-tier storage,
    real-time processing, and advanced security features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize healthcare data architecture"""
        self.config = config
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        \# Initialize storage layers
        self._init_storage_layers()
        
        \# Initialize processing engines
        self._init_processing_engines()
        
        \# Initialize security and monitoring
        self._init_security_monitoring()
        
        \# Data governance policies
        self.governance_policies = self._load_governance_policies()
        
        logger.info("Healthcare data architecture initialized")
    
    def _init_storage_layers(self):
        """Initialize multi-tier storage architecture"""
        \# Hot tier: High-performance transactional database
        self.hot_db = create_engine(
            self.config['hot_db_url'],
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            echo=False
        )
        
        \# Warm tier: Analytics database
        self.warm_db = create_engine(
            self.config['warm_db_url'],
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            echo=False
        )
        
        \# Cold tier: Object storage for large datasets
        self.cold_storage = boto3.client(
            's3',
            aws_access_key_id=self.config.get('aws_access_key'),
            aws_secret_access_key=self.config.get('aws_secret_key'),
            region_name=self.config.get('aws_region', 'us-east-1')
        )
        
        \# Search and analytics
        self.elasticsearch = Elasticsearch(
            [self.config['elasticsearch_url']],
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        
        \# Caching layer
        self.redis_client = redis.from_url(
            self.config['redis_url'],
            decode_responses=True,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
        
        logger.info("Storage layers initialized")
    
    def _init_processing_engines(self):
        """Initialize data processing engines"""
        \# Real-time streaming
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=self.config['kafka_brokers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            batch_size=16384,
            linger_ms=10
        )
        
        \# Batch processing executors
        self.thread_executor = ThreadPoolExecutor(max_workers=10)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        
        logger.info("Processing engines initialized")
    
    def _init_security_monitoring(self):
        """Initialize security and monitoring systems"""
        \# Start Prometheus metrics server
        start_http_server(8000)
        
        \# Initialize audit logging
        self.audit_logger = logging.getLogger('healthcare.audit')
        audit_handler = logging.FileHandler('healthcare_audit.log')
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)
        
        logger.info("Security and monitoring initialized")
    
    def _load_governance_policies(self) -> Dict[str, DataGovernancePolicy]:
        """Load data governance policies"""
        return {
            'patient_data': DataGovernancePolicy(
                retention_period_days=2555,  \# 7 years
                encryption_required=True,
                access_logging_required=True,
                data_classification=SecurityLevel.PHI,
                allowed_regions=['us-east-1', 'us-west-2'],
                backup_frequency_hours=6,
                audit_frequency_days=30
            ),
            'clinical_notes': DataGovernancePolicy(
                retention_period_days=3650,  \# 10 years
                encryption_required=True,
                access_logging_required=True,
                data_classification=SecurityLevel.SENSITIVE_PHI,
                allowed_regions=['us-east-1'],
                backup_frequency_hours=4,
                audit_frequency_days=7
            ),
            'population_metrics': DataGovernancePolicy(
                retention_period_days=1825,  \# 5 years
                encryption_required=True,
                access_logging_required=True,
                data_classification=SecurityLevel.CONFIDENTIAL,
                allowed_regions=['us-east-1', 'us-west-2'],
                backup_frequency_hours=12,
                audit_frequency_days=90
            )
        }
    
    async def store_healthcare_data(self, 
                                  data: Dict[str, Any],
                                  data_type: str,
                                  priority: str = 'normal',
                                  user_id: str = None) -> str:
        """
        Store healthcare data with appropriate tier and security
        
        Args:
            data: Healthcare data to store
            data_type: Type of data (patient_data, clinical_notes, etc.)
            priority: Storage priority (urgent, normal, batch)
            user_id: User performing the operation
            
        Returns:
            str: Data identifier
        """
        start_time = datetime.utcnow()
        data_id = str(uuid.uuid4())
        
        try:
            \# Apply data governance policy
            policy = self.governance_policies.get(data_type)
            if not policy:
                raise ValueError(f"No governance policy found for data type: {data_type}")
            
            \# Encrypt sensitive data
            if policy.encryption_required:
                data = self._encrypt_sensitive_fields(data, policy.data_classification)
            
            \# Determine storage tier based on priority and data type
            storage_tier = self._determine_storage_tier(data_type, priority)
            
            \# Store data in appropriate tier
            if storage_tier == DataTier.HOT:
                await self._store_hot_tier(data_id, data, data_type)
            elif storage_tier == DataTier.WARM:
                await self._store_warm_tier(data_id, data, data_type)
            elif storage_tier == DataTier.COLD:
                await self._store_cold_tier(data_id, data, data_type)
            
            \# Index for search if needed
            if data_type in ['clinical_notes', 'patient_data']:
                await self._index_for_search(data_id, data, data_type)
            
            \# Cache frequently accessed data
            if priority == 'urgent' or data_type == 'patient_data':
                await self._cache_data(data_id, data, ttl=3600)
            
            \# Audit logging
            if policy.access_logging_required:
                self._log_data_access('STORE', data_id, data_type, user_id)
            
            \# Update metrics
            REQUEST_COUNT.labels(operation='store', resource_type=data_type).inc()
            
            \# Publish to real-time stream if needed
            if priority == 'urgent':
                await self._publish_to_stream(data_id, data, data_type)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            REQUEST_DURATION.observe(duration)
            
            logger.info(f"Stored healthcare data: {data_id} ({data_type})")
            return data_id
            
        except Exception as e:
            logger.error(f"Error storing healthcare data: {e}")
            raise
    
    async def retrieve_healthcare_data(self,
                                     data_id: str,
                                     data_type: str,
                                     user_id: str = None,
                                     include_audit: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve healthcare data with security and audit logging
        
        Args:
            data_id: Data identifier
            data_type: Type of data
            user_id: User requesting the data
            include_audit: Whether to include audit trail
            
        Returns:
            Dict containing healthcare data or None if not found
        """
        start_time = datetime.utcnow()
        
        try:
            \# Check governance policy
            policy = self.governance_policies.get(data_type)
            if not policy:
                raise ValueError(f"No governance policy found for data type: {data_type}")
            
            \# Check cache first
            cached_data = await self._get_cached_data(data_id)
            if cached_data:
                data = cached_data
                logger.info(f"Retrieved from cache: {data_id}")
            else:
                \# Determine storage tier and retrieve
                data = await self._retrieve_from_storage_tiers(data_id, data_type)
                
                if data:
                    \# Cache for future access
                    await self._cache_data(data_id, data, ttl=1800)
            
            if data:
                \# Decrypt sensitive data
                if policy.encryption_required:
                    data = self._decrypt_sensitive_fields(data, policy.data_classification)
                
                \# Audit logging
                if policy.access_logging_required:
                    self._log_data_access('RETRIEVE', data_id, data_type, user_id)
                
                \# Include audit trail if requested
                if include_audit:
                    data['_audit_trail'] = await self._get_audit_trail(data_id)
                
                \# Update metrics
                REQUEST_COUNT.labels(operation='retrieve', resource_type=data_type).inc()
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                REQUEST_DURATION.observe(duration)
                
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving healthcare data: {e}")
            raise
    
    async def search_healthcare_data(self,
                                   query: Dict[str, Any],
                                   data_types: List[str],
                                   user_id: str = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search healthcare data across multiple types and storage tiers
        
        Args:
            query: Search query parameters
            data_types: Types of data to search
            user_id: User performing the search
            limit: Maximum number of results
            
        Returns:
            List of matching healthcare data records
        """
        start_time = datetime.utcnow()
        
        try:
            \# Build Elasticsearch query
            es_query = self._build_elasticsearch_query(query, data_types)
            
            \# Execute search
            response = self.elasticsearch.search(
                index='healthcare-data',
                body=es_query,
                size=limit
            )
            
            results = []
            for hit in response['hits']['hits']:
                data_id = hit['_id']
                data_type = hit['_source']['data_type']
                
                \# Retrieve full data
                full_data = await self.retrieve_healthcare_data(
                    data_id, data_type, user_id
                )
                
                if full_data:
                    full_data['_score'] = hit['_score']
                    results.append(full_data)
            
            \# Audit logging
            self._log_data_access('SEARCH', f"query:{json.dumps(query)}", 
                                'multiple', user_id)
            
            \# Update metrics
            REQUEST_COUNT.labels(operation='search', resource_type='multiple').inc()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            REQUEST_DURATION.observe(duration)
            
            logger.info(f"Search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching healthcare data: {e}")
            raise
    
    async def analyze_population_health(self,
                                      analysis_config: Dict[str, Any],
                                      user_id: str = None) -> Dict[str, Any]:
        """
        Perform population health analysis across healthcare data
        
        Args:
            analysis_config: Analysis configuration
            user_id: User requesting the analysis
            
        Returns:
            Dict containing analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            analysis_type = analysis_config['analysis_type']
            population_filters = analysis_config.get('population_filters', {})
            time_period = analysis_config.get('time_period', {})
            
            \# Execute analysis based on type
            if analysis_type == 'health_equity':
                results = await self._analyze_health_equity(
                    population_filters, time_period
                )
            elif analysis_type == 'outcome_trends':
                results = await self._analyze_outcome_trends(
                    population_filters, time_period
                )
            elif analysis_type == 'risk_stratification':
                results = await self._analyze_risk_stratification(
                    population_filters, time_period
                )
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            \# Store analysis results
            analysis_id = await self.store_healthcare_data(
                data=results,
                data_type='population_analysis',
                priority='normal',
                user_id=user_id
            )
            
            results['analysis_id'] = analysis_id
            results['generated_at'] = datetime.utcnow().isoformat()
            results['generated_by'] = user_id
            
            \# Audit logging
            self._log_data_access('ANALYZE', analysis_id, 'population_analysis', user_id)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            REQUEST_DURATION.observe(duration)
            
            logger.info(f"Population health analysis completed: {analysis_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error in population health analysis: {e}")
            raise
    
    async def _store_hot_tier(self, data_id: str, data: Dict[str, Any], data_type: str):
        """Store data in hot tier (high-performance database)"""
        with self.hot_db.connect() as conn:
            conn.execute(text("""
                INSERT INTO healthcare_data_hot (id, data_type, data, created_at)
                VALUES (:id, :data_type, :data, :created_at)
            """), {
                'id': data_id,
                'data_type': data_type,
                'data': json.dumps(data),
                'created_at': datetime.utcnow()
            })
            conn.commit()
    
    async def _store_warm_tier(self, data_id: str, data: Dict[str, Any], data_type: str):
        """Store data in warm tier (analytics database)"""
        with self.warm_db.connect() as conn:
            conn.execute(text("""
                INSERT INTO healthcare_data_warm (id, data_type, data, created_at)
                VALUES (:id, :data_type, :data, :created_at)
            """), {
                'id': data_id,
                'data_type': data_type,
                'data': json.dumps(data),
                'created_at': datetime.utcnow()
            })
            conn.commit()
    
    async def _store_cold_tier(self, data_id: str, data: Dict[str, Any], data_type: str):
        """Store data in cold tier (object storage)"""
        bucket_name = self.config['cold_storage_bucket']
        key = f"{data_type}/{datetime.utcnow().year}/{datetime.utcnow().month}/{data_id}.json"
        
        self.cold_storage.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json.dumps(data),
            ServerSideEncryption='AES256',
            Metadata={
                'data_type': data_type,
                'created_at': datetime.utcnow().isoformat()
            }
        )
    
    def _determine_storage_tier(self, data_type: str, priority: str) -> DataTier:
        """Determine appropriate storage tier for data"""
        if priority == 'urgent' or data_type in ['patient_data', 'clinical_alerts']:
            return DataTier.HOT
        elif data_type in ['clinical_notes', 'lab_results']:
            return DataTier.WARM
        else:
            return DataTier.COLD
    
    def _encrypt_sensitive_fields(self, data: Dict[str, Any], 
                                security_level: SecurityLevel) -> Dict[str, Any]:
        """Encrypt sensitive fields based on security level"""
        if security_level in [SecurityLevel.PHI, SecurityLevel.SENSITIVE_PHI]:
            sensitive_fields = ['ssn', 'mrn', 'name', 'address', 'phone', 'email']
            
            for field in sensitive_fields:
                if field in data:
                    if isinstance(data[field], str):
                        data[field] = self.cipher_suite.encrypt(
                            data[field].encode()
                        ).decode()
        
        return data
    
    def _decrypt_sensitive_fields(self, data: Dict[str, Any], 
                                security_level: SecurityLevel) -> Dict[str, Any]:
        """Decrypt sensitive fields based on security level"""
        if security_level in [SecurityLevel.PHI, SecurityLevel.SENSITIVE_PHI]:
            sensitive_fields = ['ssn', 'mrn', 'name', 'address', 'phone', 'email']
            
            for field in sensitive_fields:
                if field in data and isinstance(data[field], str):
                    try:
                        data[field] = self.cipher_suite.decrypt(
                            data[field].encode()
                        ).decode()
                    except Exception:
                        \# Field may not be encrypted
                        pass
        
        return data
    
    def _log_data_access(self, operation: str, data_id: str, 
                        data_type: str, user_id: str):
        """Log data access for audit purposes"""
        self.audit_logger.info(json.dumps({
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'data_id': data_id,
            'data_type': data_type,
            'user_id': user_id,
            'ip_address': 'unknown',  \# Would be populated from request context
            'session_id': 'unknown'   \# Would be populated from request context
        }))
    
    async def _cache_data(self, data_id: str, data: Dict[str, Any], ttl: int):
        """Cache data in Redis"""
        cache_key = f"healthcare:{data_id}"
        self.redis_client.setex(
            cache_key,
            ttl,
            json.dumps(data)
        )
    
    async def _get_cached_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache"""
        cache_key = f"healthcare:{data_id}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        return None
    
    async def _publish_to_stream(self, data_id: str, data: Dict[str, Any], data_type: str):
        """Publish data to real-time stream"""
        message = {
            'data_id': data_id,
            'data_type': data_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.kafka_producer.send(
            topic='healthcare-data-stream',
            key=data_id,
            value=message
        )


\# Demonstration function
def demonstrate_healthcare_architecture():
    """Demonstrate healthcare data architecture capabilities"""
    print("=== Healthcare Data Architecture Demonstration ===\n")
    
    \# Configuration for demonstration
    config = {
        'hot_db_url': 'sqlite:///healthcare_hot.db',
        'warm_db_url': 'sqlite:///healthcare_warm.db',
        'redis_url': 'redis://localhost:6379/0',
        'elasticsearch_url': 'http://localhost:9200',
        'kafka_brokers': ['localhost:9092'],
        'cold_storage_bucket': 'healthcare-cold-storage',
        'aws_region': 'us-east-1'
    }
    
    print("Healthcare data architecture would be initialized with:")
    print(f"  Hot tier: High-performance transactional database")
    print(f"  Warm tier: Analytics database for complex queries")
    print(f"  Cold tier: Object storage for long-term retention")
    print(f"  Search: Elasticsearch for full-text search")
    print(f"  Cache: Redis for high-speed data access")
    print(f"  Streaming: Kafka for real-time data processing")
    
    print(f"\nData governance policies:")
    print(f"  Patient data: 7-year retention, PHI encryption")
    print(f"  Clinical notes: 10-year retention, sensitive PHI encryption")
    print(f"  Population metrics: 5-year retention, confidential encryption")
    
    print(f"\nSecurity features:")
    print(f"  Field-level encryption for sensitive data")
    print(f"  Comprehensive audit logging")
    print(f"  Role-based access control")
    print(f"  Data classification and handling")
    
    print(f"\nMonitoring and metrics:")
    print(f"  Prometheus metrics for performance monitoring")
    print(f"  Request counting and duration tracking")
    print(f"  Data quality score monitoring")
    print(f"  Active connection monitoring")


if __name__ == "__main__":
    demonstrate_healthcare_architecture()