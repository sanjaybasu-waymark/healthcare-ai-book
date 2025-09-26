---
layout: default
title: "Chapter 13: Real World Deployment Strategies"
nav_order: 13
parent: Chapters
---

# Chapter 13: Real-World Deployment Strategies - Production Implementation of Healthcare AI Systems

*By Sanjay Basu MD PhD*

## Learning Objectives

By the end of this chapter, physician data scientists will be able to:

- Design comprehensive deployment architectures for healthcare AI systems across different clinical environments, including on-premises, cloud, hybrid, and edge deployment models that address the unique requirements of healthcare settings while maintaining scalability, security, and regulatory compliance
- Implement robust integration strategies with existing healthcare IT infrastructure and clinical workflows, including EHR integration, PACS connectivity, laboratory system interfaces, and real-time data streaming that enhance rather than disrupt existing care processes
- Develop scalable deployment pipelines with automated testing, continuous integration/continuous deployment (CI/CD), monitoring, and rollback capabilities that ensure reliable and safe deployment of AI systems in production healthcare environments
- Establish comprehensive governance frameworks for AI system deployment, maintenance, and continuous improvement that address change management, version control, performance monitoring, and stakeholder communication throughout the system lifecycle
- Navigate complex regulatory and compliance requirements for AI system deployment in healthcare settings, including HIPAA compliance, FDA regulations, state healthcare laws, and international standards that ensure legal and ethical operation
- Implement advanced monitoring and alerting systems for production AI systems that provide real-time visibility into system performance, clinical outcomes, user adoption, and potential issues while supporting proactive maintenance and optimization
- Apply DevOps and MLOps best practices specifically adapted for healthcare environments, including infrastructure as code, automated testing, model versioning, and deployment automation that accelerate development while maintaining safety and compliance

## 13.1 Introduction to Healthcare AI Deployment

The deployment of artificial intelligence systems in healthcare represents one of the most complex and critical challenges in modern medical technology implementation. Unlike traditional software deployments that primarily focus on functional requirements and user experience, healthcare AI systems must integrate seamlessly with existing clinical workflows while maintaining the highest standards of patient safety, data security, regulatory compliance, and clinical effectiveness. The stakes are inherently higher in healthcare, where system failures can directly impact patient outcomes, and the regulatory environment is more stringent than in most other industries.

Real-world deployment of healthcare AI systems requires careful orchestration of multiple interconnected factors that span technical, clinical, regulatory, and organizational domains. **Technical infrastructure** must support AI workloads reliably and securely while providing the scalability needed to handle varying clinical demands and the flexibility to adapt to evolving requirements. **Clinical workflow integration** must enhance rather than disrupt existing care processes, requiring deep understanding of clinical practices, user needs, and organizational culture. **Regulatory compliance** must address multiple overlapping requirements including HIPAA privacy and security rules, FDA medical device regulations, state healthcare laws, and institutional policies that govern the use of AI in clinical practice.

**User training and adoption** strategies must ensure that clinical staff can effectively utilize AI capabilities while maintaining their clinical judgment and decision-making autonomy. **Continuous monitoring and improvement** processes must maintain system performance over time while adapting to changing clinical needs, evolving best practices, and emerging regulatory requirements. **Change management** throughout the deployment process must address the human factors that determine the ultimate success or failure of AI system implementation.

### 13.1.1 Deployment Complexity in Healthcare Environments

Healthcare AI deployment complexity stems from the unique characteristics and constraints of healthcare environments that distinguish them from other technology deployment contexts. **Mission-critical nature** of healthcare applications means that system failures, performance degradation, or incorrect outputs can directly impact patient safety and clinical outcomes, requiring deployment strategies that prioritize reliability, fault tolerance, and graceful degradation over rapid feature deployment or cost optimization.

**Regulatory oversight** creates a complex compliance landscape that includes federal regulations such as HIPAA privacy and security rules, FDA medical device regulations, and CMS reimbursement requirements, as well as state-specific healthcare laws, institutional policies, and professional licensing requirements. Each of these regulatory frameworks imposes specific technical and operational requirements that must be addressed throughout the deployment process.

**Legacy system integration** presents significant technical challenges as healthcare organizations typically operate complex ecosystems of electronic health records (EHRs), picture archiving and communication systems (PACS), laboratory information systems (LIS), radiology information systems (RIS), pharmacy systems, billing systems, and numerous specialized clinical applications. These systems often use different data formats, communication protocols, security models, and integration standards, requiring sophisticated middleware and integration platforms to enable seamless data flow and interoperability.

**Clinical workflow diversity** across different healthcare settings means that AI systems must be flexible enough to accommodate varying practices, protocols, and organizational structures while maintaining consistent performance and clinical effectiveness. **Emergency departments** operate under time pressure with high patient turnover and variable acuity, requiring AI systems that can provide rapid, accurate results without disrupting critical care processes. **Intensive care units** require continuous monitoring capabilities with real-time alerting and decision support that integrates with existing monitoring systems and clinical protocols.

**Outpatient clinics** have different workflow patterns with scheduled appointments, routine screenings, and chronic disease management that require AI systems optimized for efficiency and patient engagement. **Surgical suites** require AI systems that can operate in sterile environments with minimal disruption to surgical workflows while providing real-time guidance and decision support. **Diagnostic imaging departments** require AI systems that integrate with PACS and radiology workflows while maintaining image quality and diagnostic accuracy.

**Stakeholder complexity** involves multiple groups with different priorities, concerns, and decision-making authority that must be aligned throughout the deployment process. **Clinicians** are primarily focused on patient care quality, workflow efficiency, and clinical outcomes, requiring AI systems that demonstrably improve their ability to provide effective care. **IT administrators** are concerned with system security, reliability, integration complexity, and resource utilization, requiring robust technical architecture and comprehensive monitoring capabilities.

**Compliance officers** must ensure adherence to all applicable regulations and institutional policies, requiring comprehensive documentation, audit trails, and risk management processes. **Executives** are focused on strategic objectives, cost management, and organizational performance, requiring clear value propositions and measurable return on investment. **Patients** are increasingly concerned about privacy, transparency, and the role of AI in their care, requiring clear communication and consent processes.

### 13.1.2 Deployment Models for Healthcare AI Systems

Healthcare AI systems can be deployed using various models, each with distinct advantages, challenges, and appropriate use cases that must be carefully evaluated based on organizational requirements, technical constraints, and regulatory considerations. **On-premises deployment** provides maximum control over data, infrastructure, and security but requires significant internal IT resources, expertise, and capital investment. This model is often preferred by large healthcare systems with robust IT capabilities, strict data governance requirements, and the resources to maintain complex AI infrastructure.

On-premises deployment offers several advantages including **complete data control** that ensures sensitive patient information never leaves the organization's infrastructure, **customization flexibility** that allows for deep integration with existing systems and workflows, **performance optimization** that can be tailored to specific use cases and workloads, and **regulatory compliance** that can be more easily demonstrated and audited. However, on-premises deployment also presents challenges including **high capital costs** for hardware and infrastructure, **ongoing maintenance** requirements for complex AI systems, **scalability limitations** that may constrain growth and adaptation, and **expertise requirements** for specialized AI infrastructure management.

**Cloud-based deployment** offers scalability, reduced infrastructure costs, and access to advanced AI services and platforms but raises important concerns about data security, regulatory compliance, and vendor dependency. **Public cloud** platforms such as Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform (GCP) provide comprehensive AI and machine learning services, global infrastructure, and elastic scaling capabilities that can significantly reduce deployment complexity and time-to-market.

Cloud deployment advantages include **reduced capital costs** through pay-as-you-use pricing models, **rapid scalability** that can handle varying workloads and growth, **managed services** that reduce operational complexity, **global availability** that supports multi-site deployments, and **innovation access** to cutting-edge AI services and capabilities. However, cloud deployment challenges include **data sovereignty** concerns about patient data leaving organizational control, **compliance complexity** in demonstrating regulatory adherence across cloud environments, **vendor lock-in** risks that may limit future flexibility, and **connectivity dependencies** that require reliable internet access for critical operations.

**Hybrid cloud models** attempt to balance the advantages of both on-premises and cloud deployment by keeping sensitive data and critical workloads on-premises while leveraging cloud resources for computation, development, testing, and non-critical operations. This approach allows organizations to maintain control over sensitive data while benefiting from cloud scalability and services.

**Edge deployment** brings AI capabilities closer to the point of care, reducing latency, improving reliability for time-critical applications, and enabling operation in environments with limited connectivity. **Medical devices** with embedded AI, such as portable ultrasound systems with automated image analysis, **bedside monitoring systems** with real-time patient assessment capabilities, and **mobile diagnostic tools** with immediate result interpretation represent different approaches to edge deployment.

Edge deployment offers advantages including **reduced latency** for time-critical applications, **improved reliability** through local processing capabilities, **bandwidth efficiency** by processing data locally rather than transmitting to central systems, and **privacy protection** by keeping sensitive data on local devices. However, edge deployment challenges include **limited computational resources** that may constrain AI model complexity, **device management** complexity across distributed deployments, **security concerns** for devices that may be physically accessible, and **update management** for distributed AI systems.

**Software as a Service (SaaS)** models provide AI capabilities through web-based interfaces, reducing deployment complexity and infrastructure requirements but potentially limiting customization and integration capabilities. SaaS AI solutions are typically developed by specialized vendors who maintain the underlying infrastructure, AI models, and application software while providing access through standardized interfaces.

**Platform as a Service (PaaS)** models offer more flexibility than SaaS while still providing managed infrastructure and development tools. PaaS platforms allow organizations to develop and deploy custom AI applications while leveraging managed services for infrastructure, data storage, and AI frameworks.

### 13.1.3 Deployment Lifecycle Management

Successful healthcare AI deployment requires comprehensive lifecycle management that addresses all phases from initial planning through ongoing maintenance and eventual retirement or replacement. This lifecycle approach ensures that AI systems continue to provide value throughout their operational life while adapting to changing requirements and maintaining compliance with evolving regulations.

**Planning and assessment** phases involve comprehensive stakeholder analysis to identify all parties affected by the AI system deployment, requirements gathering to understand functional and non-functional needs, risk assessment to identify potential technical and clinical risks, and resource planning to ensure adequate budget, personnel, and infrastructure for successful deployment. This phase also includes **feasibility studies** that evaluate technical and organizational readiness, **cost-benefit analysis** that quantifies expected value and return on investment, and **regulatory assessment** that identifies all applicable compliance requirements.

**Development and testing** phases include system development using appropriate software engineering practices, comprehensive integration testing to ensure compatibility with existing systems, user acceptance testing with clinical staff to validate usability and workflow integration, and regulatory validation to demonstrate compliance with applicable requirements. This phase also includes **performance testing** to validate system scalability and reliability, **security testing** to identify and address potential vulnerabilities, and **clinical validation** to demonstrate safety and effectiveness in realistic use scenarios.

**Pilot deployment** allows for controlled testing in limited clinical environments before full-scale rollout, providing opportunities to identify and address issues in a controlled setting while building confidence among clinical staff and stakeholders. Pilot deployments should include **limited scope** implementation with carefully selected use cases and user groups, **intensive monitoring** to identify performance issues and user feedback, **iterative improvement** based on pilot results and user input, and **success criteria** that determine readiness for broader deployment.

**Production deployment** involves full system implementation with comprehensive monitoring, support, and maintenance capabilities that ensure reliable operation in the production healthcare environment. This phase includes **phased rollout** strategies that gradually expand system usage while monitoring for issues, **comprehensive training** for all users and support staff, **24/7 monitoring** and support capabilities to address issues promptly, and **change management** processes to facilitate user adoption and workflow integration.

**Continuous improvement** processes ensure that AI systems evolve to meet changing clinical needs, incorporate new capabilities, and maintain optimal performance over time. This includes **performance monitoring** to track system effectiveness and identify optimization opportunities, **user feedback** collection and analysis to understand evolving needs, **model updates** to improve accuracy and incorporate new data, and **feature enhancement** to add new capabilities and improve user experience.

**Change management** throughout the deployment lifecycle helps ensure successful adoption by clinical staff and integration with existing workflows while minimizing disruption to patient care. Effective change management includes **communication strategies** that keep stakeholders informed about deployment progress and benefits, **training programs** that ensure users have the knowledge and skills needed to effectively use AI systems, **feedback mechanisms** that allow users to report issues and suggest improvements, and **support systems** that provide ongoing assistance and troubleshooting.

## 13.2 Infrastructure Architecture for Healthcare AI

### 13.2.1 Scalable Computing Infrastructure Design

Healthcare AI systems require robust computing infrastructure that can handle varying workloads while maintaining consistent performance, availability, and security. The infrastructure must be designed to support both the computational demands of AI workloads and the reliability requirements of healthcare applications, where system downtime or performance degradation can directly impact patient care.

**Horizontal scaling** capabilities allow systems to handle increased demand by adding additional computing resources, providing the flexibility to accommodate growth in user base, data volume, or computational complexity. This approach involves distributing workloads across multiple servers or containers, enabling the system to scale out rather than up. **Load balancing** mechanisms ensure that incoming requests are distributed evenly across available resources, preventing any single component from becoming a bottleneck.

**Container orchestration** using platforms like Kubernetes provides sophisticated capabilities for managing containerized AI applications at scale. Kubernetes offers **automatic scaling** based on resource utilization or custom metrics, **rolling updates** that enable zero-downtime deployments, **service discovery** that allows components to find and communicate with each other, and **resource management** that ensures efficient utilization of computing resources.

**Vertical scaling** provides more powerful individual resources for computationally intensive tasks such as deep learning model training or inference on large datasets. This approach involves increasing the CPU, memory, or storage capacity of individual servers to handle more demanding workloads. While vertical scaling has limits, it can be more cost-effective for certain types of AI workloads that cannot be easily parallelized.

**GPU acceleration** is often essential for deep learning workloads, requiring careful consideration of GPU resource allocation, memory management, and workload scheduling. **NVIDIA Tesla** and **AMD Instinct** GPUs provide the computational power needed for training and inference of complex neural networks. **GPU clusters** enable distributed training of large models across multiple GPUs and servers, while **GPU sharing** technologies allow multiple workloads to efficiently utilize GPU resources.

**Multi-GPU** and **distributed training** capabilities enable handling of large-scale AI models and datasets that cannot fit on a single GPU or server. **Data parallelism** distributes training data across multiple GPUs while keeping the model architecture consistent, while **model parallelism** distributes different parts of large models across multiple GPUs. **Gradient synchronization** mechanisms ensure that distributed training converges to optimal solutions.

**High availability** design ensures that AI systems remain operational even during hardware failures, software issues, or maintenance activities. **Redundancy** at multiple levels including servers, network connections, and storage systems prevents single points of failure. **Failover mechanisms** automatically redirect traffic and workloads to healthy components when failures are detected. **Disaster recovery** capabilities enable rapid restoration of services in the event of major outages or catastrophic failures.

**Auto-scaling policies** automatically adjust resource allocation based on demand, ensuring optimal performance while controlling costs. **CPU-based scaling** adjusts resources based on processor utilization, while **memory-based scaling** responds to memory pressure. **Custom metric scaling** can use application-specific metrics such as queue length or response time to trigger scaling actions. **Predictive scaling** uses historical patterns and machine learning to anticipate demand and pre-scale resources.

### 13.2.2 Data Infrastructure and Management Systems

Healthcare AI systems require sophisticated data infrastructure that can handle large volumes of diverse data types while maintaining security, privacy, and regulatory compliance. The data infrastructure must support both real-time processing for immediate clinical decision support and batch processing for model training and analytics.

**Data lakes** provide flexible storage for structured and unstructured healthcare data, accommodating the variety of data types found in healthcare environments including clinical notes, medical images, laboratory results, sensor data, and administrative records. **Apache Hadoop** and **Apache Spark** provide distributed storage and processing capabilities for large-scale data analytics. **Delta Lake** and **Apache Iceberg** add ACID transaction support and schema evolution capabilities to data lakes.

**Data warehouses** offer optimized storage and query performance for analytical workloads, providing structured access to healthcare data for reporting, analytics, and AI model training. **Snowflake**, **Amazon Redshift**, and **Google BigQuery** provide cloud-based data warehouse solutions with automatic scaling and optimization. **Apache Druid** and **ClickHouse** offer real-time analytics capabilities for time-series data and high-frequency queries.

**Real-time data streaming** capabilities enable AI systems to process data as it is generated, supporting time-critical applications like patient monitoring, clinical decision support, and real-time alerting. **Apache Kafka** provides distributed streaming platform capabilities with high throughput and fault tolerance. **Apache Storm** and **Apache Flink** offer stream processing frameworks for complex event processing and real-time analytics.

**Data preprocessing pipelines** must handle the complexity of healthcare data including **data cleaning** to remove errors and inconsistencies, **normalization** to standardize formats and units, **feature extraction** to derive meaningful variables for AI models, and **quality validation** to ensure data meets standards for clinical use. **Apache Airflow** provides workflow orchestration capabilities for complex data processing tasks, while **Kubeflow Pipelines** offers machine learning-specific workflow management.

**Data versioning and lineage** tracking ensures reproducibility and enables audit trails for regulatory compliance. **DVC (Data Version Control)** provides Git-like versioning for datasets and machine learning models. **MLflow** offers experiment tracking and model management capabilities. **Apache Atlas** and **DataHub** provide data governance and lineage tracking for complex data ecosystems.

**Data quality monitoring** continuously assesses data integrity, completeness, and accuracy to ensure that AI systems receive high-quality input data. **Great Expectations** provides data validation and testing frameworks, while **Apache Griffin** offers data quality monitoring for big data environments. **Anomaly detection** systems can identify unusual patterns in data that may indicate quality issues or security threats.

### 13.2.3 Security and Compliance Architecture

Healthcare AI deployment requires comprehensive security architecture that addresses multiple layers of protection while maintaining compliance with healthcare regulations and industry standards. The security architecture must protect against both external threats and internal risks while enabling legitimate access to data and systems for clinical and administrative purposes.

**Network security** forms the foundation of healthcare AI security, including **firewalls** that control traffic between network segments, **intrusion detection systems** that monitor for suspicious activity, and **network segmentation** that limits the impact of security breaches by isolating critical systems. **Virtual private networks (VPNs)** provide secure remote access for authorized users, while **network access control (NAC)** systems ensure that only authorized devices can connect to the network.

**Zero-trust architecture** assumes that no network traffic should be trusted by default and requires verification for all access requests, regardless of the source location or user credentials. This approach includes **identity verification** for all users and devices, **least privilege access** that grants only the minimum permissions necessary for specific tasks, **continuous monitoring** of all network activity, and **micro-segmentation** that creates secure zones around critical resources.

**Application security** involves secure coding practices, input validation, authentication, and authorization mechanisms that protect AI applications from common vulnerabilities. **Web application firewalls (WAF)** protect against common web-based attacks, while **API gateways** provide security and access control for application programming interfaces. **Container security** scanning identifies vulnerabilities in containerized applications and their dependencies.

**Data encryption** must protect data both at rest and in transit, using industry-standard encryption algorithms and key management practices. **AES-256 encryption** provides strong protection for stored data, while **TLS 1.3** secures data transmission over networks. **End-to-end encryption** ensures that data remains protected throughout its lifecycle, from collection through processing to storage and eventual deletion.

**Key management** systems securely generate, distribute, and rotate encryption keys while maintaining access controls and audit trails. **Hardware security modules (HSMs)** provide tamper-resistant storage for cryptographic keys, while **key management services** in cloud platforms offer managed key lifecycle management. **Certificate management** ensures that digital certificates used for authentication and encryption remain valid and properly configured.

**Identity and access management (IAM)** systems control who can access what resources under what circumstances, providing **single sign-on (SSO)** capabilities that simplify user authentication while maintaining security. **Multi-factor authentication (MFA)** adds additional security layers beyond passwords, while **role-based access control (RBAC)** ensures that users have appropriate permissions based on their job functions.

**Audit logging** and **monitoring** capabilities provide comprehensive tracking of system access, data usage, and administrative activities. **Security Information and Event Management (SIEM)** systems collect and analyze security logs from multiple sources to detect potential threats and security incidents. **User and Entity Behavior Analytics (UEBA)** systems use machine learning to identify unusual patterns that may indicate security threats or policy violations.

## 13.3 Comprehensive Healthcare AI Deployment Framework

### 13.3.1 Production-Ready Deployment Infrastructure

```python
"""
Comprehensive Healthcare AI Deployment Framework

This implementation provides a complete deployment infrastructure for healthcare AI systems,
including container orchestration, monitoring, security, and compliance capabilities
specifically designed for healthcare environments.

Author: Sanjay Basu MD PhD
License: MIT
"""

import asyncio
import logging
import json
import uuid
import hashlib
import time
import os
import yaml
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Infrastructure and orchestration
import docker
import kubernetes
from kubernetes import client, config
import redis
import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Cloud services
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs

# Web framework and API
import requests
from flask import Flask, request, jsonify, g
from celery import Celery
import gunicorn

# Monitoring and metrics
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import grafana_api

# Security and encryption
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
import bcrypt
from functools import wraps

# Scheduling and automation
import schedule
from crontab import CronTab

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/healthcare-ai-deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"

class DeploymentStrategy(Enum):
    """Deployment strategy types."""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"

class ScalingPolicy(Enum):
    """Auto-scaling policy types."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"

class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComplianceFramework(Enum):
    """Healthcare compliance frameworks."""
    HIPAA = "hipaa"
    GDPR = "gdpr"
    FDA_510K = "fda_510k"
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"

@dataclass
class DeploymentConfig:
    """Comprehensive deployment configuration."""
    deployment_id: str
    service_name: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    replicas: int
    resource_limits: Dict[str, str]
    health_check_config: Dict[str, Any]
    security_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    scaling_config: Dict[str, Any]
    backup_config: Dict[str, Any]
    compliance_requirements: List[ComplianceFramework]
    network_policies: Dict[str, Any]
    data_retention_policy: Dict[str, Any]
    disaster_recovery_config: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'deployment_id': self.deployment_id,
            'service_name': self.service_name,
            'environment': self.environment.value,
            'strategy': self.strategy.value,
            'replicas': self.replicas,
            'resource_limits': self.resource_limits,
            'health_check_config': self.health_check_config,
            'security_config': self.security_config,
            'monitoring_config': self.monitoring_config,
            'scaling_config': self.scaling_config,
            'backup_config': self.backup_config,
            'compliance_requirements': [req.value for req in self.compliance_requirements],
            'network_policies': self.network_policies,
            'data_retention_policy': self.data_retention_policy,
            'disaster_recovery_config': self.disaster_recovery_config,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class ServiceMetrics:
    """Comprehensive service performance metrics."""
    service_name: str
    deployment_id: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    request_rate: float
    error_rate: float
    response_time: float
    availability: float
    active_connections: int
    queue_depth: int
    model_inference_time: float
    model_accuracy: Optional[float]
    compliance_score: float
    security_events: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DeploymentEvent:
    """Deployment event tracking."""
    event_id: str
    deployment_id: str
    event_type: str
    event_data: Dict[str, Any]
    status: str
    user_id: Optional[str]
    source_ip: Optional[str]
    compliance_impact: bool
    security_relevant: bool
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SecurityAlert:
    """Security alert information."""
    alert_id: str
    deployment_id: str
    alert_type: str
    severity: str
    description: str
    affected_components: List[str]
    remediation_steps: List[str]
    compliance_implications: List[str]
    auto_remediated: bool
    timestamp: datetime = field(default_factory=datetime.now)

class SecurityManager:
    """Comprehensive security management for healthcare AI deployments."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize security manager."""
        self.config = config
        self.encryption_key = self._generate_encryption_key()
        self.audit_logger = self._setup_audit_logging()
        self.access_control = self._setup_access_control()
        
        logger.info("Initialized security manager for healthcare AI deployment")
    
    def _generate_encryption_key(self) -> Fernet:
        """Generate encryption key for data protection."""
        password = self.config.get('encryption_password', 'default_password').encode()
        salt = self.config.get('encryption_salt', b'salt_1234567890123456')
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password)
        return Fernet(key)
    
    def _setup_audit_logging(self) -> logging.Logger:
        """Setup comprehensive audit logging."""
        audit_logger = logging.getLogger('healthcare_ai_audit')
        audit_logger.setLevel(logging.INFO)
        
        # Create audit log handler
        audit_handler = logging.FileHandler('/var/log/healthcare-ai-audit.log')
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
        
        return audit_logger
    
    def _setup_access_control(self) -> Dict[str, Any]:
        """Setup role-based access control."""
        return {
            'roles': {
                'admin': ['read', 'write', 'deploy', 'configure'],
                'clinician': ['read', 'inference'],
                'researcher': ['read', 'analyze'],
                'auditor': ['read', 'audit']
            },
            'resources': {
                'models': ['read', 'write', 'deploy'],
                'data': ['read', 'write'],
                'configs': ['read', 'write', 'configure'],
                'logs': ['read', 'audit']
            }
        }
    
    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt sensitive data."""
        if isinstance(data, str):
            data = data.encode()
        
        encrypted_data = self.encryption_key.encrypt(data)
        
        self.audit_logger.info(f"Data encrypted - size: {len(data)} bytes")
        
        return encrypted_data
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data."""
        try:
            decrypted_data = self.encryption_key.decrypt(encrypted_data)
            self.audit_logger.info(f"Data decrypted - size: {len(decrypted_data)} bytes")
            return decrypted_data
        except Exception as e:
            self.audit_logger.error(f"Decryption failed: {str(e)}")
            raise
    
    def generate_access_token(self, user_id: str, role: str, expiration_hours: int = 24) -> str:
        """Generate JWT access token."""
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=expiration_hours),
            'iat': datetime.utcnow(),
            'iss': 'healthcare-ai-system'
        }
        
        token = jwt.encode(payload, self.config.get('jwt_secret', 'default_secret'), algorithm='HS256')
        
        self.audit_logger.info(f"Access token generated for user: {user_id}, role: {role}")
        
        return token
    
    def validate_access_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT access token."""
        try:
            payload = jwt.decode(token, self.config.get('jwt_secret', 'default_secret'), algorithms=['HS256'])
            
            self.audit_logger.info(f"Access token validated for user: {payload.get('user_id')}")
            
            return payload
        except jwt.ExpiredSignatureError:
            self.audit_logger.warning("Access token expired")
            raise
        except jwt.InvalidTokenError:
            self.audit_logger.error("Invalid access token")
            raise
    
    def check_permissions(self, user_role: str, resource: str, action: str) -> bool:
        """Check user permissions for resource and action."""
        role_permissions = self.access_control['roles'].get(user_role, [])
        resource_actions = self.access_control['resources'].get(resource, [])
        
        has_permission = action in role_permissions and action in resource_actions
        
        self.audit_logger.info(
            f"Permission check - role: {user_role}, resource: {resource}, "
            f"action: {action}, granted: {has_permission}"
        )
        
        return has_permission
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = "INFO"):
        """Log security-relevant events."""
        event_data = {
            'event_type': event_type,
            'details': details,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.audit_logger.info(f"SECURITY_EVENT: {json.dumps(event_data)}")

class MonitoringManager:
    """Comprehensive monitoring and alerting for healthcare AI deployments."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize monitoring manager."""
        self.config = config
        self.metrics_registry = CollectorRegistry()
        self.metrics = self._setup_metrics()
        self.alert_thresholds = self._setup_alert_thresholds()
        self.notification_channels = self._setup_notification_channels()
        
        logger.info("Initialized monitoring manager for healthcare AI deployment")
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup Prometheus metrics."""
        metrics = {
            'request_count': Counter(
                'healthcare_ai_requests_total',
                'Total number of requests',
                ['service', 'endpoint', 'status'],
                registry=self.metrics_registry
            ),
            'request_duration': Histogram(
                'healthcare_ai_request_duration_seconds',
                'Request duration in seconds',
                ['service', 'endpoint'],
                registry=self.metrics_registry
            ),
            'model_inference_time': Histogram(
                'healthcare_ai_model_inference_seconds',
                'Model inference time in seconds',
                ['model_name', 'model_version'],
                registry=self.metrics_registry
            ),
            'model_accuracy': Gauge(
                'healthcare_ai_model_accuracy',
                'Model accuracy score',
                ['model_name', 'model_version'],
                registry=self.metrics_registry
            ),
            'system_health': Gauge(
                'healthcare_ai_system_health',
                'System health status (1=healthy, 0=unhealthy)',
                ['service', 'component'],
                registry=self.metrics_registry
            ),
            'compliance_score': Gauge(
                'healthcare_ai_compliance_score',
                'Compliance score (0-1)',
                ['framework', 'service'],
                registry=self.metrics_registry
            ),
            'security_events': Counter(
                'healthcare_ai_security_events_total',
                'Total number of security events',
                ['event_type', 'severity'],
                registry=self.metrics_registry
            )
        }
        
        return metrics
    
    def _setup_alert_thresholds(self) -> Dict[str, Any]:
        """Setup alerting thresholds."""
        return {
            'response_time': {
                'warning': 2.0,  # seconds
                'critical': 5.0
            },
            'error_rate': {
                'warning': 0.05,  # 5%
                'critical': 0.10   # 10%
            },
            'cpu_usage': {
                'warning': 0.80,  # 80%
                'critical': 0.95   # 95%
            },
            'memory_usage': {
                'warning': 0.85,  # 85%
                'critical': 0.95   # 95%
            },
            'model_accuracy': {
                'warning': 0.85,  # Below 85%
                'critical': 0.80   # Below 80%
            },
            'compliance_score': {
                'warning': 0.90,  # Below 90%
                'critical': 0.80   # Below 80%
            }
        }
    
    def _setup_notification_channels(self) -> Dict[str, Any]:
        """Setup notification channels for alerts."""
        return {
            'email': {
                'enabled': True,
                'recipients': self.config.get('alert_emails', []),
                'smtp_server': self.config.get('smtp_server', 'localhost')
            },
            'slack': {
                'enabled': self.config.get('slack_enabled', False),
                'webhook_url': self.config.get('slack_webhook_url'),
                'channel': self.config.get('slack_channel', '#healthcare-ai-alerts')
            },
            'pagerduty': {
                'enabled': self.config.get('pagerduty_enabled', False),
                'integration_key': self.config.get('pagerduty_integration_key')
            }
        }
    
    def record_request(self, service: str, endpoint: str, status: str, duration: float):
        """Record request metrics."""
        self.metrics['request_count'].labels(
            service=service,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.metrics['request_duration'].labels(
            service=service,
            endpoint=endpoint
        ).observe(duration)
    
    def record_model_inference(self, model_name: str, model_version: str, inference_time: float):
        """Record model inference metrics."""
        self.metrics['model_inference_time'].labels(
            model_name=model_name,
            model_version=model_version
        ).observe(inference_time)
    
    def update_model_accuracy(self, model_name: str, model_version: str, accuracy: float):
        """Update model accuracy metrics."""
        self.metrics['model_accuracy'].labels(
            model_name=model_name,
            model_version=model_version
        ).set(accuracy)
        
        # Check for accuracy alerts
        if accuracy < self.alert_thresholds['model_accuracy']['critical']:
            self._send_alert(
                'Model Accuracy Critical',
                f'Model {model_name} v{model_version} accuracy dropped to {accuracy:.3f}',
                'critical'
            )
        elif accuracy < self.alert_thresholds['model_accuracy']['warning']:
            self._send_alert(
                'Model Accuracy Warning',
                f'Model {model_name} v{model_version} accuracy dropped to {accuracy:.3f}',
                'warning'
            )
    
    def update_system_health(self, service: str, component: str, is_healthy: bool):
        """Update system health metrics."""
        self.metrics['system_health'].labels(
            service=service,
            component=component
        ).set(1 if is_healthy else 0)
        
        if not is_healthy:
            self._send_alert(
                'System Health Alert',
                f'Component {component} in service {service} is unhealthy',
                'critical'
            )
    
    def update_compliance_score(self, framework: str, service: str, score: float):
        """Update compliance score metrics."""
        self.metrics['compliance_score'].labels(
            framework=framework,
            service=service
        ).set(score)
        
        # Check for compliance alerts
        if score < self.alert_thresholds['compliance_score']['critical']:
            self._send_alert(
                'Compliance Critical',
                f'Service {service} compliance score for {framework} dropped to {score:.3f}',
                'critical'
            )
        elif score < self.alert_thresholds['compliance_score']['warning']:
            self._send_alert(
                'Compliance Warning',
                f'Service {service} compliance score for {framework} dropped to {score:.3f}',
                'warning'
            )
    
    def record_security_event(self, event_type: str, severity: str):
        """Record security event metrics."""
        self.metrics['security_events'].labels(
            event_type=event_type,
            severity=severity
        ).inc()
        
        if severity in ['critical', 'high']:
            self._send_alert(
                'Security Event',
                f'Security event of type {event_type} with severity {severity}',
                severity
            )
    
    def _send_alert(self, title: str, message: str, severity: str):
        """Send alert through configured notification channels."""
        alert_data = {
            'title': title,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'healthcare-ai-deployment'
        }
        
        # Send email alerts
        if self.notification_channels['email']['enabled']:
            self._send_email_alert(alert_data)
        
        # Send Slack alerts
        if self.notification_channels['slack']['enabled']:
            self._send_slack_alert(alert_data)
        
        # Send PagerDuty alerts for critical issues
        if (self.notification_channels['pagerduty']['enabled'] and 
            severity in ['critical', 'high']):
            self._send_pagerduty_alert(alert_data)
        
        logger.warning(f"Alert sent: {title} - {message}")
    
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send email alert."""
        # Implementation would use SMTP to send email
        logger.info(f"Email alert sent: {alert_data['title']}")
    
    def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send Slack alert."""
        # Implementation would use Slack webhook
        logger.info(f"Slack alert sent: {alert_data['title']}")
    
    def _send_pagerduty_alert(self, alert_data: Dict[str, Any]):
        """Send PagerDuty alert."""
        # Implementation would use PagerDuty API
        logger.info(f"PagerDuty alert sent: {alert_data['title']}")

class DeploymentOrchestrator:
    """Comprehensive deployment orchestration for healthcare AI systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize deployment orchestrator."""
        self.config = config
        self.security_manager = SecurityManager(config.get('security', {}))
        self.monitoring_manager = MonitoringManager(config.get('monitoring', {}))
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_networking_v1 = client.NetworkingV1Api()
        
        # Initialize Docker client
        self.docker_client = docker.from_env()
        
        # Deployment tracking
        self.deployments = {}
        self.deployment_history = []
        
        logger.info("Initialized deployment orchestrator for healthcare AI")
    
    def create_deployment(self, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """Create a new healthcare AI deployment."""
        try:
            # Validate deployment configuration
            self._validate_deployment_config(deployment_config)
            
            # Create Kubernetes deployment
            k8s_deployment = self._create_kubernetes_deployment(deployment_config)
            
            # Create service
            k8s_service = self._create_kubernetes_service(deployment_config)
            
            # Create network policies
            network_policies = self._create_network_policies(deployment_config)
            
            # Setup monitoring
            self._setup_deployment_monitoring(deployment_config)
            
            # Setup security policies
            self._setup_security_policies(deployment_config)
            
            # Record deployment
            deployment_info = {
                'deployment_id': deployment_config.deployment_id,
                'kubernetes_deployment': k8s_deployment.metadata.name,
                'kubernetes_service': k8s_service.metadata.name,
                'network_policies': [np.metadata.name for np in network_policies],
                'status': 'deployed',
                'created_at': datetime.utcnow(),
                'config': deployment_config.to_dict()
            }
            
            self.deployments[deployment_config.deployment_id] = deployment_info
            self.deployment_history.append(deployment_info)
            
            # Log deployment event
            self.security_manager.log_security_event(
                'deployment_created',
                {
                    'deployment_id': deployment_config.deployment_id,
                    'service_name': deployment_config.service_name,
                    'environment': deployment_config.environment.value
                }
            )
            
            logger.info(f"Successfully created deployment: {deployment_config.deployment_id}")
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Failed to create deployment: {str(e)}")
            raise
    
    def _validate_deployment_config(self, config: DeploymentConfig):
        """Validate deployment configuration."""
        # Check required fields
        required_fields = ['deployment_id', 'service_name', 'environment']
        for field in required_fields:
            if not getattr(config, field):
                raise ValueError(f"Missing required field: {field}")
        
        # Validate resource limits
        if not config.resource_limits:
            raise ValueError("Resource limits must be specified")
        
        # Validate compliance requirements
        if not config.compliance_requirements:
            logger.warning("No compliance requirements specified")
        
        # Validate security configuration
        if not config.security_config:
            raise ValueError("Security configuration must be specified")
        
        logger.info(f"Deployment configuration validated: {config.deployment_id}")
    
    def _create_kubernetes_deployment(self, config: DeploymentConfig) -> client.V1Deployment:
        """Create Kubernetes deployment."""
        
        # Define container
        container = client.V1Container(
            name=config.service_name,
            image=config.security_config.get('container_image'),
            ports=[client.V1ContainerPort(container_port=8080)],
            resources=client.V1ResourceRequirements(
                limits=config.resource_limits,
                requests={
                    'cpu': str(float(config.resource_limits.get('cpu', '1')) * 0.5),
                    'memory': str(int(config.resource_limits.get('memory', '1Gi')[:-2]) // 2) + 'Mi'
                }
            ),
            env=[
                client.V1EnvVar(name='ENVIRONMENT', value=config.environment.value),
                client.V1EnvVar(name='DEPLOYMENT_ID', value=config.deployment_id)
            ],
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=config.health_check_config.get('liveness_path', '/health'),
                    port=8080
                ),
                initial_delay_seconds=30,
                period_seconds=10
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=config.health_check_config.get('readiness_path', '/ready'),
                    port=8080
                ),
                initial_delay_seconds=5,
                period_seconds=5
            )
        )
        
        # Define pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    'app': config.service_name,
                    'deployment-id': config.deployment_id,
                    'environment': config.environment.value
                }
            ),
            spec=client.V1PodSpec(
                containers=[container],
                security_context=client.V1PodSecurityContext(
                    run_as_non_root=True,
                    run_as_user=1000,
                    fs_group=2000
                )
            )
        )
        
        # Define deployment spec
        deployment_spec = client.V1DeploymentSpec(
            replicas=config.replicas,
            selector=client.V1LabelSelector(
                match_labels={
                    'app': config.service_name,
                    'deployment-id': config.deployment_id
                }
            ),
            template=pod_template,
            strategy=self._get_deployment_strategy(config.strategy)
        )
        
        # Create deployment
        deployment = client.V1Deployment(
            api_version='apps/v1',
            kind='Deployment',
            metadata=client.V1ObjectMeta(
                name=f"{config.service_name}-{config.deployment_id[:8]}",
                namespace=config.security_config.get('namespace', 'default'),
                labels={
                    'app': config.service_name,
                    'deployment-id': config.deployment_id,
                    'environment': config.environment.value
                }
            ),
            spec=deployment_spec
        )
        
        # Apply deployment
        created_deployment = self.k8s_apps_v1.create_namespaced_deployment(
            namespace=config.security_config.get('namespace', 'default'),
            body=deployment
        )
        
        logger.info(f"Created Kubernetes deployment: {created_deployment.metadata.name}")
        
        return created_deployment
    
    def _create_kubernetes_service(self, config: DeploymentConfig) -> client.V1Service:
        """Create Kubernetes service."""
        
        service_spec = client.V1ServiceSpec(
            selector={
                'app': config.service_name,
                'deployment-id': config.deployment_id
            },
            ports=[
                client.V1ServicePort(
                    port=80,
                    target_port=8080,
                    protocol='TCP'
                )
            ],
            type='ClusterIP'
        )
        
        service = client.V1Service(
            api_version='v1',
            kind='Service',
            metadata=client.V1ObjectMeta(
                name=f"{config.service_name}-service-{config.deployment_id[:8]}",
                namespace=config.security_config.get('namespace', 'default'),
                labels={
                    'app': config.service_name,
                    'deployment-id': config.deployment_id
                }
            ),
            spec=service_spec
        )
        
        created_service = self.k8s_core_v1.create_namespaced_service(
            namespace=config.security_config.get('namespace', 'default'),
            body=service
        )
        
        logger.info(f"Created Kubernetes service: {created_service.metadata.name}")
        
        return created_service
    
    def _create_network_policies(self, config: DeploymentConfig) -> List[client.V1NetworkPolicy]:
        """Create network security policies."""
        
        policies = []
        
        # Default deny all policy
        deny_all_policy = client.V1NetworkPolicy(
            api_version='networking.k8s.io/v1',
            kind='NetworkPolicy',
            metadata=client.V1ObjectMeta(
                name=f"{config.service_name}-deny-all-{config.deployment_id[:8]}",
                namespace=config.security_config.get('namespace', 'default')
            ),
            spec=client.V1NetworkPolicySpec(
                pod_selector=client.V1LabelSelector(
                    match_labels={
                        'app': config.service_name,
                        'deployment-id': config.deployment_id
                    }
                ),
                policy_types=['Ingress', 'Egress']
            )
        )
        
        # Allow specific ingress
        allow_ingress_policy = client.V1NetworkPolicy(
            api_version='networking.k8s.io/v1',
            kind='NetworkPolicy',
            metadata=client.V1ObjectMeta(
                name=f"{config.service_name}-allow-ingress-{config.deployment_id[:8]}",
                namespace=config.security_config.get('namespace', 'default')
            ),
            spec=client.V1NetworkPolicySpec(
                pod_selector=client.V1LabelSelector(
                    match_labels={
                        'app': config.service_name,
                        'deployment-id': config.deployment_id
                    }
                ),
                ingress=[
                    client.V1NetworkPolicyIngressRule(
                        from_=[
                            client.V1NetworkPolicyPeer(
                                namespace_selector=client.V1LabelSelector(
                                    match_labels={'name': 'healthcare-ai'}
                                )
                            )
                        ],
                        ports=[
                            client.V1NetworkPolicyPort(port=8080, protocol='TCP')
                        ]
                    )
                ],
                policy_types=['Ingress']
            )
        )
        
        # Create policies
        for policy in [deny_all_policy, allow_ingress_policy]:
            created_policy = self.k8s_networking_v1.create_namespaced_network_policy(
                namespace=config.security_config.get('namespace', 'default'),
                body=policy
            )
            policies.append(created_policy)
            logger.info(f"Created network policy: {created_policy.metadata.name}")
        
        return policies
    
    def _get_deployment_strategy(self, strategy: DeploymentStrategy) -> client.V1DeploymentStrategy:
        """Get Kubernetes deployment strategy."""
        
        if strategy == DeploymentStrategy.ROLLING:
            return client.V1DeploymentStrategy(
                type='RollingUpdate',
                rolling_update=client.V1RollingUpdateDeployment(
                    max_surge='25%',
                    max_unavailable='25%'
                )
            )
        elif strategy == DeploymentStrategy.RECREATE:
            return client.V1DeploymentStrategy(type='Recreate')
        else:
            # Default to rolling update
            return client.V1DeploymentStrategy(
                type='RollingUpdate',
                rolling_update=client.V1RollingUpdateDeployment(
                    max_surge='25%',
                    max_unavailable='25%'
                )
            )
    
    def _setup_deployment_monitoring(self, config: DeploymentConfig):
        """Setup monitoring for deployment."""
        
        # Update system health
        self.monitoring_manager.update_system_health(
            service=config.service_name,
            component='deployment',
            is_healthy=True
        )
        
        # Initialize compliance scores
        for framework in config.compliance_requirements:
            self.monitoring_manager.update_compliance_score(
                framework=framework.value,
                service=config.service_name,
                score=1.0  # Start with perfect compliance
            )
        
        logger.info(f"Setup monitoring for deployment: {config.deployment_id}")
    
    def _setup_security_policies(self, config: DeploymentConfig):
        """Setup security policies for deployment."""
        
        # Log security policy setup
        self.security_manager.log_security_event(
            'security_policies_applied',
            {
                'deployment_id': config.deployment_id,
                'compliance_requirements': [req.value for req in config.compliance_requirements],
                'security_config': config.security_config
            }
        )
        
        logger.info(f"Setup security policies for deployment: {config.deployment_id}")
    
    def update_deployment(self, deployment_id: str, new_config: DeploymentConfig) -> Dict[str, Any]:
        """Update existing deployment."""
        
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        try:
            # Get current deployment
            current_deployment = self.deployments[deployment_id]
            
            # Update Kubernetes deployment
            self._update_kubernetes_deployment(deployment_id, new_config)
            
            # Update monitoring
            self._setup_deployment_monitoring(new_config)
            
            # Update security policies
            self._setup_security_policies(new_config)
            
            # Update deployment record
            current_deployment['config'] = new_config.to_dict()
            current_deployment['updated_at'] = datetime.utcnow()
            
            # Log update event
            self.security_manager.log_security_event(
                'deployment_updated',
                {
                    'deployment_id': deployment_id,
                    'service_name': new_config.service_name
                }
            )
            
            logger.info(f"Successfully updated deployment: {deployment_id}")
            
            return current_deployment
            
        except Exception as e:
            logger.error(f"Failed to update deployment: {str(e)}")
            raise
    
    def _update_kubernetes_deployment(self, deployment_id: str, config: DeploymentConfig):
        """Update Kubernetes deployment."""
        
        deployment_name = f"{config.service_name}-{deployment_id[:8]}"
        namespace = config.security_config.get('namespace', 'default')
        
        # Get current deployment
        current_deployment = self.k8s_apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )
        
        # Update deployment spec
        current_deployment.spec.replicas = config.replicas
        current_deployment.spec.template.spec.containers[0].resources = client.V1ResourceRequirements(
            limits=config.resource_limits,
            requests={
                'cpu': str(float(config.resource_limits.get('cpu', '1')) * 0.5),
                'memory': str(int(config.resource_limits.get('memory', '1Gi')[:-2]) // 2) + 'Mi'
            }
        )
        
        # Apply update
        self.k8s_apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=current_deployment
        )
        
        logger.info(f"Updated Kubernetes deployment: {deployment_name}")
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete deployment."""
        
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        try:
            deployment_info = self.deployments[deployment_id]
            namespace = deployment_info['config']['security_config'].get('namespace', 'default')
            
            # Delete Kubernetes resources
            self.k8s_apps_v1.delete_namespaced_deployment(
                name=deployment_info['kubernetes_deployment'],
                namespace=namespace
            )
            
            self.k8s_core_v1.delete_namespaced_service(
                name=deployment_info['kubernetes_service'],
                namespace=namespace
            )
            
            # Delete network policies
            for policy_name in deployment_info['network_policies']:
                self.k8s_networking_v1.delete_namespaced_network_policy(
                    name=policy_name,
                    namespace=namespace
                )
            
            # Remove from tracking
            del self.deployments[deployment_id]
            
            # Log deletion event
            self.security_manager.log_security_event(
                'deployment_deleted',
                {'deployment_id': deployment_id}
            )
            
            logger.info(f"Successfully deleted deployment: {deployment_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete deployment: {str(e)}")
            raise
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status and health."""
        
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment_info = self.deployments[deployment_id]
        namespace = deployment_info['config']['security_config'].get('namespace', 'default')
        
        # Get Kubernetes deployment status
        k8s_deployment = self.k8s_apps_v1.read_namespaced_deployment(
            name=deployment_info['kubernetes_deployment'],
            namespace=namespace
        )
        
        # Get pod status
        pods = self.k8s_core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"deployment-id={deployment_id}"
        )
        
        pod_statuses = []
        for pod in pods.items:
            pod_statuses.append({
                'name': pod.metadata.name,
                'phase': pod.status.phase,
                'ready': all(condition.status == 'True' 
                           for condition in pod.status.conditions or []
                           if condition.type == 'Ready')
            })
        
        status = {
            'deployment_id': deployment_id,
            'kubernetes_status': {
                'replicas': k8s_deployment.status.replicas,
                'ready_replicas': k8s_deployment.status.ready_replicas,
                'available_replicas': k8s_deployment.status.available_replicas,
                'updated_replicas': k8s_deployment.status.updated_replicas
            },
            'pod_statuses': pod_statuses,
            'overall_health': self._calculate_deployment_health(k8s_deployment, pod_statuses),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return status
    
    def _calculate_deployment_health(self, deployment, pod_statuses) -> HealthStatus:
        """Calculate overall deployment health."""
        
        if not deployment.status.ready_replicas:
            return HealthStatus.UNHEALTHY
        
        if deployment.status.ready_replicas < deployment.spec.replicas:
            return HealthStatus.DEGRADED
        
        # Check pod health
        healthy_pods = sum(1 for pod in pod_statuses 
                          if pod['phase'] == 'Running' and pod['ready'])
        
        if healthy_pods == len(pod_statuses):
            return HealthStatus.HEALTHY
        elif healthy_pods > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments."""
        
        deployments_list = []
        for deployment_id, deployment_info in self.deployments.items():
            status = self.get_deployment_status(deployment_id)
            deployments_list.append({
                'deployment_id': deployment_id,
                'service_name': deployment_info['config']['service_name'],
                'environment': deployment_info['config']['environment'],
                'status': deployment_info['status'],
                'health': status['overall_health'].value,
                'created_at': deployment_info['created_at'].isoformat()
            })
        
        return deployments_list

class HealthcareAIDeploymentFramework:
    """
    Comprehensive healthcare AI deployment framework.
    
    This class provides end-to-end deployment capabilities for healthcare AI systems,
    including security, monitoring, compliance, and orchestration.
    """
    
    def __init__(self, config_path: str):
        """Initialize healthcare AI deployment framework."""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.orchestrator = DeploymentOrchestrator(self.config)
        self.security_manager = self.orchestrator.security_manager
        self.monitoring_manager = self.orchestrator.monitoring_manager
        
        # Setup API server
        self.app = Flask(__name__)
        self._setup_api_routes()
        
        logger.info("Initialized healthcare AI deployment framework")
    
    def _setup_api_routes(self):
        """Setup API routes for deployment management."""
        
        @self.app.before_request
        def authenticate_request():
            """Authenticate API requests."""
            if request.endpoint in ['health', 'metrics']:
                return  # Skip authentication for health and metrics endpoints
            
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Missing or invalid authorization header'}), 401
            
            token = auth_header.split(' ')[1]
            try:
                payload = self.security_manager.validate_access_token(token)
                g.user_id = payload['user_id']
                g.user_role = payload['role']
            except Exception as e:
                return jsonify({'error': 'Invalid access token'}), 401
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Prometheus metrics endpoint."""
            return prometheus_client.generate_latest(self.monitoring_manager.metrics_registry)
        
        @self.app.route('/deployments', methods=['POST'])
        def create_deployment():
            """Create new deployment."""
            try:
                # Check permissions
                if not self.security_manager.check_permissions(g.user_role, 'deployments', 'deploy'):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                data = request.get_json()
                
                # Create deployment config
                deployment_config = DeploymentConfig(
                    deployment_id=str(uuid.uuid4()),
                    service_name=data['service_name'],
                    environment=DeploymentEnvironment(data['environment']),
                    strategy=DeploymentStrategy(data.get('strategy', 'rolling')),
                    replicas=data.get('replicas', 1),
                    resource_limits=data['resource_limits'],
                    health_check_config=data.get('health_check_config', {}),
                    security_config=data['security_config'],
                    monitoring_config=data.get('monitoring_config', {}),
                    scaling_config=data.get('scaling_config', {}),
                    backup_config=data.get('backup_config', {}),
                    compliance_requirements=[ComplianceFramework(req) for req in data.get('compliance_requirements', [])],
                    network_policies=data.get('network_policies', {}),
                    data_retention_policy=data.get('data_retention_policy', {}),
                    disaster_recovery_config=data.get('disaster_recovery_config', {})
                )
                
                # Create deployment
                deployment_info = self.orchestrator.create_deployment(deployment_config)
                
                return jsonify(deployment_info), 201
                
            except Exception as e:
                logger.error(f"Failed to create deployment: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/deployments', methods=['GET'])
        def list_deployments():
            """List all deployments."""
            try:
                # Check permissions
                if not self.security_manager.check_permissions(g.user_role, 'deployments', 'read'):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                deployments = self.orchestrator.list_deployments()
                return jsonify(deployments)
                
            except Exception as e:
                logger.error(f"Failed to list deployments: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/deployments/<deployment_id>', methods=['GET'])
        def get_deployment(deployment_id):
            """Get deployment details."""
            try:
                # Check permissions
                if not self.security_manager.check_permissions(g.user_role, 'deployments', 'read'):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                status = self.orchestrator.get_deployment_status(deployment_id)
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"Failed to get deployment: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/deployments/<deployment_id>', methods=['DELETE'])
        def delete_deployment(deployment_id):
            """Delete deployment."""
            try:
                # Check permissions
                if not self.security_manager.check_permissions(g.user_role, 'deployments', 'deploy'):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                success = self.orchestrator.delete_deployment(deployment_id)
                return jsonify({'success': success})
                
            except Exception as e:
                logger.error(f"Failed to delete deployment: {str(e)}")
                return jsonify({'error': str(e)}), 500
    
    def start_api_server(self, host: str = '0.0.0.0', port: int = 8080):
        """Start API server."""
        logger.info(f"Starting healthcare AI deployment API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)
    
    def deploy_ai_model(
        self,
        model_name: str,
        model_version: str,
        container_image: str,
        environment: DeploymentEnvironment,
        compliance_requirements: List[ComplianceFramework],
        resource_requirements: Dict[str, str]
    ) -> str:
        """Deploy AI model with healthcare-specific configurations."""
        
        deployment_config = DeploymentConfig(
            deployment_id=str(uuid.uuid4()),
            service_name=model_name,
            environment=environment,
            strategy=DeploymentStrategy.ROLLING,
            replicas=2,  # Default to 2 for high availability
            resource_limits=resource_requirements,
            health_check_config={
                'liveness_path': '/health',
                'readiness_path': '/ready'
            },
            security_config={
                'container_image': container_image,
                'namespace': 'healthcare-ai',
                'security_context': {
                    'run_as_non_root': True,
                    'read_only_root_filesystem': True
                }
            },
            monitoring_config={
                'metrics_enabled': True,
                'logging_level': 'INFO'
            },
            scaling_config={
                'min_replicas': 2,
                'max_replicas': 10,
                'target_cpu_utilization': 70
            },
            backup_config={
                'enabled': True,
                'schedule': '0 2 * * *'  # Daily at 2 AM
            },
            compliance_requirements=compliance_requirements,
            network_policies={
                'ingress_allowed': ['healthcare-ai-gateway'],
                'egress_allowed': ['healthcare-data-services']
            },
            data_retention_policy={
                'logs_retention_days': 90,
                'metrics_retention_days': 365
            },
            disaster_recovery_config={
                'backup_region': 'us-west-2',
                'rto_minutes': 60,  # Recovery Time Objective
                'rpo_minutes': 15   # Recovery Point Objective
            }
        )
        
        deployment_info = self.orchestrator.create_deployment(deployment_config)
        
        logger.info(f"Deployed AI model {model_name} v{model_version} with deployment ID: {deployment_config.deployment_id}")
        
        return deployment_config.deployment_id

## Bibliography and References

### Infrastructure and Deployment Architecture

1. **Burns, B., & Beda, J.** (2019). *Kubernetes: Up and Running: Dive into the Future of Infrastructure*. O'Reilly Media. [Kubernetes orchestration]

2. **Fowler, M.** (2013). Microservices. *Martin Fowler's Blog*. Retrieved from https://martinfowler.com/articles/microservices.html [Microservices architecture]

3. **Newman, S.** (2021). *Building Microservices: Designing Fine-Grained Systems*. O'Reilly Media. [Microservices design patterns]

4. **Richardson, C.** (2018). *Microservices Patterns: With Examples in Java*. Manning Publications. [Microservices patterns]

### DevOps and MLOps for Healthcare

5. **Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., ... & Young, M.** (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems*, 28, 2503-2511. [ML technical debt]

6. **Amershi, S., Begel, A., Bird, C., DeLine, R., Gall, H., Kamar, E., ... & Zimmermann, T.** (2019). Software engineering for machine learning: A case study. *Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice*, 291-300. [ML software engineering]

7. **Paleyes, A., Urma, R. G., & Lawrence, N. D.** (2022). Challenges in deploying machine learning: a survey of case studies. *ACM Computing Surveys*, 55(6), 1-29. [ML deployment challenges]

8. **Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., ... & Heidmann, L.** (2020). *Introducing MLOps*. O'Reilly Media. [MLOps practices]

### Healthcare IT Infrastructure and Integration

9. **Benson, T., & Grieve, G.** (2021). *Principles of Health Interoperability: SNOMED CT, HL7 and FHIR*. Springer. [Healthcare interoperability]

10. **Mandl, K. D., & Kohane, I. S.** (2012). Escaping the EHR trap—the future of health IT. *New England Journal of Medicine*, 366(24), 2240-2242. [EHR integration challenges]

11. **Sittig, D. F., & Singh, H.** (2010). A new sociotechnical model for studying health information technology in complex adaptive healthcare systems. *Quality and Safety in Health Care*, 19(Suppl 3), i68-i74. [Healthcare IT systems]

12. **Kruse, C. S., Stein, A., Thomas, H., & Kaur, H.** (2018). The use of Electronic Health Records to support population health: a systematic review of the literature. *Journal of Medical Systems*, 42(11), 214. [EHR population health]

### Security and Compliance in Healthcare

13. **Luna, R., Rhine, E., Myhra, M., Sullivan, R., & Kruse, C. S.** (2016). Cyber threats to health information systems: a systematic review. *Technology and Health Care*, 24(1), 1-9. [Healthcare cybersecurity threats]

14. **Argaw, S. T., Troncoso-Pastoriza, J. R., Lacey, D., Florin, M. V., Calcavecchia, F., Anderson, D., ... & Floreano, D.** (2020). Cybersecurity of hospitals: discussing the challenges and working towards mitigating the risks. *BMC Medical Informatics and Decision Making*, 20(1), 146. [Hospital cybersecurity]

15. **Coventry, L., & Branley, D.** (2018). Cybersecurity in healthcare: A narrative review of trends, threats and ways forward. *Maturitas*, 113, 48-52. [Healthcare cybersecurity review]

16. **Kruse, C. S., Frederick, B., Jacobson, T., & Monticone, D. K.** (2017). Cybersecurity in healthcare: A systematic review of modern threats and trends. *Technology and Health Care*, 25(1), 1-10. [Healthcare cybersecurity trends]

### Monitoring and Observability

17. **Beyer, B., Jones, C., Petoff, J., & Murphy, N. R.** (2016). *Site Reliability Engineering: How Google Runs Production Systems*. O'Reilly Media. [Site reliability engineering]

18. **Majors, C., Fong-Jones, L., & Miranda, G.** (2022). *Observability Engineering: Achieving Production Excellence*. O'Reilly Media. [Observability practices]

19. **Godard, S.** (2020). *Systems Performance: Enterprise and the Cloud*. Addison-Wesley Professional. [Systems performance monitoring]

20. **Ligus, J.** (2019). *Effective Monitoring and Alerting: For Web Operations*. O'Reilly Media. [Monitoring and alerting]

This chapter provides a comprehensive framework for real-world deployment of healthcare AI systems, addressing the unique challenges and requirements of healthcare environments. The implementations provide practical tools for infrastructure management, security, monitoring, and compliance that enable safe and effective deployment of AI systems in clinical practice. The next chapter will explore population health AI systems, building upon these deployment concepts to address large-scale health analytics and intervention strategies.
