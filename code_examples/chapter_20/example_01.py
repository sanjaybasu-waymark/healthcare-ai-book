"""
Chapter 20 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

"""
Comprehensive Edge Computing System for Healthcare

This implementation provides a complete framework for edge computing
in healthcare environments with model optimization, real-time processing,
and secure communication capabilities.

Author: Sanjay Basu MD PhD
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.quantization as quant
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import asyncio
import concurrent.futures
from collections import defaultdict, deque
import pickle
import threading
import queue
import time
import copy
import psutil
import GPUtil
from pathlib import Path
import cv2
import pyaudio
import wave
import struct
from scipy import signal
from scipy.stats import zscore
import websockets
import ssl
from cryptography.fernet import Fernet
import hashlib
import hmac
import base64
import requests
import socket
import platform
import subprocess

warnings.filterwarnings('ignore')

\# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/edge-computing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EdgeDeviceType(Enum):
    """Types of edge devices."""
    WEARABLE = "wearable"
    BEDSIDE_MONITOR = "bedside_monitor"
    PORTABLE_DEVICE = "portable_device"
    GATEWAY = "gateway"
    WORKSTATION = "workstation"

class CompressionMethod(Enum):
    """Model compression methods."""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    COMBINED = "combined"

class ProcessingMode(Enum):
    """Edge processing modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    EVENT_DRIVEN = "event_driven"

@dataclass
class EdgeDeviceSpec:
    """Edge device specifications."""
    device_type: EdgeDeviceType
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    power_budget_watts: float = 10.0
    network_bandwidth_mbps: float = 100.0
    operating_system: str = "linux"
    
    \# Clinical specifications
    clinical_environment: str = "hospital"
    regulatory_class: str = "class_ii"
    real_time_requirements: bool = True
    max_latency_ms: float = 100.0
    availability_requirement: float = 0.999
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'device_type': self.device_type.value,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'storage_gb': self.storage_gb,
            'gpu_available': self.gpu_available,
            'gpu_memory_gb': self.gpu_memory_gb,
            'power_budget_watts': self.power_budget_watts,
            'network_bandwidth_mbps': self.network_bandwidth_mbps,
            'operating_system': self.operating_system,
            'clinical_environment': self.clinical_environment,
            'regulatory_class': self.regulatory_class,
            'real_time_requirements': self.real_time_requirements,
            'max_latency_ms': self.max_latency_ms,
            'availability_requirement': self.availability_requirement
        }

@dataclass
class EdgeConfig:
    """Configuration for edge computing system."""
    device_spec: EdgeDeviceSpec
    compression_method: CompressionMethod = CompressionMethod.QUANTIZATION
    processing_mode: ProcessingMode = ProcessingMode.REAL_TIME
    
    \# Model optimization
    target_model_size_mb: float = 10.0
    target_inference_time_ms: float = 50.0
    min_accuracy_threshold: float = 0.95
    quantization_bits: int = 8
    pruning_ratio: float = 0.5
    
    \# Real-time processing
    buffer_size: int = 1000
    sampling_rate_hz: float = 100.0
    window_size_seconds: float = 10.0
    overlap_ratio: float = 0.5
    
    \# Communication
    cloud_endpoint: str = "https://api.healthcare-cloud.com"
    secure_communication: bool = True
    compression_enabled: bool = True
    encryption_key: Optional[str] = None
    
    \# Monitoring and alerts
    enable_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 80.0,
        'memory_usage': 85.0,
        'temperature': 70.0,
        'inference_latency': 100.0
    })
    
    \# Clinical integration
    hl7_fhir_enabled: bool = True
    clinical_decision_support: bool = True
    audit_logging: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'device_spec': self.device_spec.to_dict(),
            'compression_method': self.compression_method.value,
            'processing_mode': self.processing_mode.value,
            'target_model_size_mb': self.target_model_size_mb,
            'target_inference_time_ms': self.target_inference_time_ms,
            'min_accuracy_threshold': self.min_accuracy_threshold,
            'quantization_bits': self.quantization_bits,
            'pruning_ratio': self.pruning_ratio,
            'buffer_size': self.buffer_size,
            'sampling_rate_hz': self.sampling_rate_hz,
            'window_size_seconds': self.window_size_seconds,
            'overlap_ratio': self.overlap_ratio,
            'cloud_endpoint': self.cloud_endpoint,
            'secure_communication': self.secure_communication,
            'compression_enabled': self.compression_enabled,
            'enable_monitoring': self.enable_monitoring,
            'alert_thresholds': self.alert_thresholds,
            'hl7_fhir_enabled': self.hl7_fhir_enabled,
            'clinical_decision_support': self.clinical_decision_support,
            'audit_logging': self.audit_logging
        }

class ModelOptimizer:
    """Model optimization for edge deployment."""
    
    def __init__(self, config: EdgeConfig):
        """Initialize model optimizer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.device_spec.gpu_available else "cpu")
        
        logger.info(f"Initialized model optimizer for {config.device_spec.device_type.value}")
    
    def quantize_model(self, model: nn.Module, calibration_loader: DataLoader) -> nn.Module:
        """Quantize model for edge deployment."""
        logger.info("Starting model quantization...")
        
        \# Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        \# Fuse modules if possible
        try:
            model = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
        except:
            logger.warning("Module fusion not applicable for this model")
        
        \# Prepare for quantization
        model_prepared = torch.quantization.prepare(model)
        
        \# Calibration
        logger.info("Calibrating quantized model...")
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                if batch_idx >= 100:  \# Limit calibration samples
                    break
                data = data.to(self.device)
                model_prepared(data)
        
        \# Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        \# Measure model size
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = original_size / quantized_size
        
        logger.info(f"Quantization completed: {compression_ratio:.2f}x compression")
        logger.info(f"Model size: {original_size:.2f} MB -> {quantized_size:.2f} MB")
        
        return quantized_model
    
    def prune_model(self, model: nn.Module, pruning_ratio: float = 0.5) -> nn.Module:
        """Prune model for edge deployment."""
        import torch.nn.utils.prune as prune
        
        logger.info(f"Starting model pruning with ratio {pruning_ratio}...")
        
        \# Identify modules to prune
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                modules_to_prune.append((module, 'weight'))
        
        \# Apply global magnitude pruning
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio
        )
        
        \# Remove pruning reparameterization
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        \# Calculate sparsity
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        sparsity = zero_params / total_params
        
        logger.info(f"Pruning completed: {sparsity:.2f} sparsity achieved")
        
        return model
    
    def distill_model(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        temperature: float = 4.0,
        alpha: float = 0.7
    ) -> nn.Module:
        """Distill knowledge from teacher to student model."""
        logger.info("Starting knowledge distillation...")
        
        teacher_model.eval()
        student_model.train()
        
        optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                \# Student predictions
                student_output = student_model(data)
                
                \# Teacher predictions
                with torch.no_grad():
                    teacher_output = teacher_model(data)
                
                \# Compute losses
                ce_loss = criterion_ce(student_output, target)
                
                kd_loss = criterion_kd(
                    F.log_softmax(student_output / temperature, dim=1),
                    F.softmax(teacher_output / temperature, dim=1)
                ) * (temperature ** 2)
                
                total_loss_batch = alpha * kd_loss + (1 - alpha) * ce_loss
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            \# Validation
            val_acc = self._evaluate_model(student_model, val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        logger.info(f"Distillation completed: Best validation accuracy: {best_val_acc:.4f}")
        
        return student_model
    
    def optimize_for_edge(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        calibration_loader: Optional[DataLoader] = None
    ) -> nn.Module:
        """Comprehensive model optimization for edge deployment."""
        logger.info("Starting comprehensive edge optimization...")
        
        if calibration_loader is None:
            calibration_loader = val_loader
        
        optimized_model = model
        
        \# Apply compression based on configuration
        if self.config.compression_method == CompressionMethod.QUANTIZATION:
            optimized_model = self.quantize_model(optimized_model, calibration_loader)
        
        elif self.config.compression_method == CompressionMethod.PRUNING:
            optimized_model = self.prune_model(optimized_model, self.config.pruning_ratio)
        
        elif self.config.compression_method == CompressionMethod.COMBINED:
            \# First prune, then quantize
            optimized_model = self.prune_model(optimized_model, self.config.pruning_ratio)
            optimized_model = self.quantize_model(optimized_model, calibration_loader)
        
        \# Validate optimized model
        original_acc = self._evaluate_model(model, val_loader)
        optimized_acc = self._evaluate_model(optimized_model, val_loader)
        
        logger.info(f"Optimization results:")
        logger.info(f"  Original accuracy: {original_acc:.4f}")
        logger.info(f"  Optimized accuracy: {optimized_acc:.4f}")
        logger.info(f"  Accuracy retention: {optimized_acc/original_acc:.4f}")
        
        \# Check if optimization meets requirements
        if optimized_acc < self.config.min_accuracy_threshold:
            logger.warning(f"Optimized model accuracy {optimized_acc:.4f} below threshold {self.config.min_accuracy_threshold}")
        
        return optimized_model
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb
    
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        return accuracy

class RealTimeProcessor:
    """Real-time data processing for edge devices."""
    
    def __init__(self, config: EdgeConfig):
        """Initialize real-time processor."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.device_spec.gpu_available else "cpu")
        
        \# Data buffers
        self.data_buffer = deque(maxlen=config.buffer_size)
        self.result_buffer = deque(maxlen=1000)
        
        \# Processing state
        self.is_processing = False
        self.processing_thread = None
        self.data_queue = queue.Queue(maxsize=config.buffer_size)
        
        \# Performance monitoring
        self.processing_times = deque(maxlen=1000)
        self.throughput_counter = 0
        self.last_throughput_time = time.time()
        
        \# Model
        self.model = None
        
        logger.info("Initialized real-time processor")
    
    def load_model(self, model: nn.Module):
        """Load optimized model for inference."""
        self.model = model.to(self.device)
        self.model.eval()
        
        logger.info("Loaded model for real-time inference")
    
    def start_processing(self):
        """Start real-time processing."""
        if self.is_processing:
            logger.warning("Processing already started")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        logger.info("Started real-time processing")
    
    def stop_processing(self):
        """Stop real-time processing."""
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join()
        
        logger.info("Stopped real-time processing")
    
    def add_data(self, data: np.ndarray, timestamp: Optional[float] = None):
        """Add data for processing."""
        if timestamp is None:
            timestamp = time.time()
        
        try:
            self.data_queue.put((data, timestamp), block=False)
        except queue.Full:
            logger.warning("Data queue full, dropping sample")
    
    def get_latest_results(self, num_results: int = 10) -> List[Dict[str, Any]]:
        """Get latest processing results."""
        results = []
        
        for _ in range(min(num_results, len(self.result_buffer))):
            if self.result_buffer:
                results.append(self.result_buffer.pop())
        
        return results[::-1]  \# Return in chronological order
    
    def _processing_loop(self):
        """Main processing loop."""
        logger.info("Starting processing loop")
        
        while self.is_processing:
            try:
                \# Get data from queue
                data, timestamp = self.data_queue.get(timeout=1.0)
                
                \# Process data
                start_time = time.time()
                result = self._process_sample(data, timestamp)
                processing_time = (time.time() - start_time) * 1000  \# ms
                
                \# Store result
                self.result_buffer.append(result)
                
                \# Update performance metrics
                self.processing_times.append(processing_time)
                self.throughput_counter += 1
                
                \# Log performance periodically
                current_time = time.time()
                if current_time - self.last_throughput_time >= 10.0:
                    throughput = self.throughput_counter / (current_time - self.last_throughput_time)
                    avg_latency = np.mean(self.processing_times) if self.processing_times else 0
                    
                    logger.info(f"Performance: {throughput:.1f} samples/sec, {avg_latency:.1f} ms avg latency")
                    
                    self.throughput_counter = 0
                    self.last_throughput_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
                continue
    
    def _process_sample(self, data: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Process a single data sample."""
        if self.model is None:
            return {
                'timestamp': timestamp,
                'error': 'No model loaded',
                'processing_time_ms': 0
            }
        
        try:
            \# Convert to tensor
            if len(data.shape) == 1:
                data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
            
            \# Inference
            with torch.no_grad():
                output = self.model(data_tensor)
                
                if output.dim() > 1:
                    probabilities = F.softmax(output, dim=1).cpu().numpy()<sup>0</sup>
                    prediction = output.argmax(dim=1).cpu().numpy()<sup>0</sup>
                else:
                    probabilities = output.cpu().numpy()
                    prediction = (output > 0.5).cpu().numpy()<sup>0</sup>
            
            return {
                'timestamp': timestamp,
                'prediction': int(prediction),
                'probabilities': probabilities.tolist(),
                'confidence': float(np.max(probabilities)),
                'processing_time_ms': 0  \# Will be filled by caller
            }
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'timestamp': timestamp,
                'error': str(e),
                'processing_time_ms': 0
            }

class EdgeCommunication:
    """Secure communication for edge devices."""
    
    def __init__(self, config: EdgeConfig):
        """Initialize edge communication."""
        self.config = config
        
        \# Encryption
        if config.secure_communication:
            if config.encryption_key:
                self.cipher = Fernet(config.encryption_key.encode())
            else:
                key = Fernet.generate_key()
                self.cipher = Fernet(key)
                logger.info(f"Generated encryption key: {key.decode()}")
        else:
            self.cipher = None
        
        \# Communication state
        self.connected = False
        self.last_heartbeat = time.time()
        
        logger.info("Initialized edge communication")
    
    def encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt data for transmission."""
        if self.cipher is None:
            return json.dumps(data).encode()
        
        serialized_data = json.dumps(data).encode()
        encrypted_data = self.cipher.encrypt(serialized_data)
        
        return encrypted_data
    
    def decrypt_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt received data."""
        if self.cipher is None:
            return json.loads(encrypted_data.decode())
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        data = json.loads(decrypted_data.decode())
        
        return data
    
    def send_to_cloud(self, data: Dict[str, Any]) -> bool:
        """Send data to cloud endpoint."""
        try:
            \# Encrypt data
            encrypted_data = self.encrypt_data(data)
            
            \# Compress if enabled
            if self.config.compression_enabled:
                import gzip
                encrypted_data = gzip.compress(encrypted_data)
            
            \# Send to cloud
            headers = {
                'Content-Type': 'application/octet-stream',
                'X-Device-ID': self.config.device_spec.device_type.value,
                'X-Timestamp': str(time.time())
            }
            
            response = requests.post(
                self.config.cloud_endpoint,
                data=encrypted_data,
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 200:
                self.connected = True
                self.last_heartbeat = time.time()
                return True
            else:
                logger.warning(f"Cloud communication failed: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Cloud communication error: {e}")
            self.connected = False
            return False
    
    def send_alert(self, alert_type: str, message: str, severity: str = "medium"):
        """Send alert to cloud."""
        alert_data = {
            'type': 'alert',
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': time.time(),
            'device_id': self.config.device_spec.device_type.value
        }
        
        success = self.send_to_cloud(alert_data)
        
        if success:
            logger.info(f"Alert sent: {alert_type} - {message}")
        else:
            logger.error(f"Failed to send alert: {alert_type} - {message}")
        
        return success

class SystemMonitor:
    """System monitoring for edge devices."""
    
    def __init__(self, config: EdgeConfig):
        """Initialize system monitor."""
        self.config = config
        
        \# Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        \# Metrics history
        self.cpu_usage_history = deque(maxlen=1000)
        self.memory_usage_history = deque(maxlen=1000)
        self.temperature_history = deque(maxlen=1000)
        self.inference_latency_history = deque(maxlen=1000)
        
        \# Communication
        self.communication = EdgeCommunication(config)
        
        logger.info("Initialized system monitor")
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()
        
        logger.info("Started system monitoring")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Stopped system monitoring")
    
    def add_inference_latency(self, latency_ms: float):
        """Add inference latency measurement."""
        self.inference_latency_history.append(latency_ms)
        
        \# Check threshold
        if latency_ms > self.config.alert_thresholds['inference_latency']:
            self.communication.send_alert(
                'high_latency',
                f'Inference latency {latency_ms:.1f}ms exceeds threshold',
                'high'
            )
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")
        
        while self.monitoring_active:
            try:
                \# Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent
                
                \# Temperature (if available)
                temperature = self._get_temperature()
                
                \# Store metrics
                self.cpu_usage_history.append(cpu_usage)
                self.memory_usage_history.append(memory_usage)
                if temperature is not None:
                    self.temperature_history.append(temperature)
                
                \# Check thresholds
                self._check_thresholds(cpu_usage, memory_usage, temperature)
                
                \# Send metrics to cloud periodically
                if len(self.cpu_usage_history) % 60 == 0:  \# Every minute
                    self._send_metrics_to_cloud()
                
                time.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5.0)
    
    def _get_temperature(self) -> Optional[float]:
        """Get system temperature."""
        try:
            if platform.system() == "Linux":
                \# Try to read from thermal zone
                thermal_files = [
                    "/sys/class/thermal/thermal_zone0/temp",
                    "/sys/class/thermal/thermal_zone1/temp"
                ]
                
                for thermal_file in thermal_files:
                    try:
                        with open(thermal_file, 'r') as f:
                            temp_millicelsius = int(f.read().strip())
                            return temp_millicelsius / 1000.0
                    except:
                        continue
            
            return None
        
        except Exception:
            return None
    
    def _check_thresholds(self, cpu_usage: float, memory_usage: float, temperature: Optional[float]):
        """Check if metrics exceed thresholds."""
        if cpu_usage > self.config.alert_thresholds['cpu_usage']:
            self.communication.send_alert(
                'high_cpu',
                f'CPU usage {cpu_usage:.1f}% exceeds threshold',
                'medium'
            )
        
        if memory_usage > self.config.alert_thresholds['memory_usage']:
            self.communication.send_alert(
                'high_memory',
                f'Memory usage {memory_usage:.1f}% exceeds threshold',
                'medium'
            )
        
        if temperature and temperature > self.config.alert_thresholds['temperature']:
            self.communication.send_alert(
                'high_temperature',
                f'Temperature {temperature:.1f}°C exceeds threshold',
                'high'
            )
    
    def _send_metrics_to_cloud(self):
        """Send metrics to cloud."""
        metrics_data = {
            'type': 'metrics',
            'timestamp': time.time(),
            'device_id': self.config.device_spec.device_type.value,
            'metrics': {
                'cpu_usage': list(self.cpu_usage_history)[-10:],  \# Last 10 samples
                'memory_usage': list(self.memory_usage_history)[-10:],
                'temperature': list(self.temperature_history)[-10:] if self.temperature_history else [],
                'inference_latency': list(self.inference_latency_history)[-10:]
            }
        }
        
        self.communication.send_to_cloud(metrics_data)

class EdgeComputingSystem:
    """Complete edge computing system."""
    
    def __init__(self, config: EdgeConfig):
        """Initialize edge computing system."""
        self.config = config
        
        \# Components
        self.optimizer = ModelOptimizer(config)
        self.processor = RealTimeProcessor(config)
        self.monitor = SystemMonitor(config)
        
        \# System state
        self.is_running = False
        self.model_loaded = False
        
        logger.info(f"Initialized edge computing system for {config.device_spec.device_type.value}")
    
    def load_and_optimize_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> nn.Module:
        """Load and optimize model for edge deployment."""
        logger.info("Loading and optimizing model for edge deployment...")
        
        \# Optimize model
        optimized_model = self.optimizer.optimize_for_edge(
            model, train_loader, val_loader
        )
        
        \# Load into processor
        self.processor.load_model(optimized_model)
        self.model_loaded = True
        
        logger.info("Model loaded and optimized successfully")
        
        return optimized_model
    
    def start_system(self):
        """Start the edge computing system."""
        if not self.model_loaded:
            logger.error("Cannot start system: No model loaded")
            return False
        
        if self.is_running:
            logger.warning("System already running")
            return True
        
        try:
            \# Start components
            self.processor.start_processing()
            self.monitor.start_monitoring()
            
            self.is_running = True
            logger.info("Edge computing system started successfully")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            return False
    
    def stop_system(self):
        """Stop the edge computing system."""
        if not self.is_running:
            logger.warning("System not running")
            return
        
        try:
            \# Stop components
            self.processor.stop_processing()
            self.monitor.stop_monitoring()
            
            self.is_running = False
            logger.info("Edge computing system stopped")
        
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def process_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Process data through the edge system."""
        if not self.is_running:
            return {'error': 'System not running'}
        
        \# Add data to processor
        self.processor.add_data(data)
        
        \# Get latest result
        results = self.processor.get_latest_results(1)
        
        if results:
            result = results<sup>0</sup>
            
            \# Update monitoring
            if 'processing_time_ms' in result:
                self.monitor.add_inference_latency(result['processing_time_ms'])
            
            return result
        else:
            return {'error': 'No results available'}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            'is_running': self.is_running,
            'model_loaded': self.model_loaded,
            'device_type': self.config.device_spec.device_type.value,
            'processing_mode': self.config.processing_mode.value,
            'timestamp': time.time()
        }
        
        if self.is_running:
            \# Add performance metrics
            if self.processor.processing_times:
                status['avg_latency_ms'] = np.mean(self.processor.processing_times)
                status['max_latency_ms'] = np.max(self.processor.processing_times)
            
            if self.monitor.cpu_usage_history:
                status['cpu_usage'] = self.monitor.cpu_usage_history[-1]
            
            if self.monitor.memory_usage_history:
                status['memory_usage'] = self.monitor.memory_usage_history[-1]
        
        return status

\# Example usage and demonstration
def create_sample_edge_model():
    """Create a sample model for edge deployment."""
    class SimpleEdgeModel(nn.Module):
        def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_classes: int = 2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    return SimpleEdgeModel()

def create_sample_data():
    """Create sample data for demonstration."""
    from torch.utils.data import TensorDataset
    
    \# Generate synthetic physiological data
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = 1000
    num_features = 10
    
    \# Normal vs abnormal patterns
    normal_data = np.random.randn(num_samples // 2, num_features)
    abnormal_data = np.random.randn(num_samples // 2, num_features) + 2.0
    
    X = np.vstack([normal_data, abnormal_data])
    y = np.hstack([np.zeros(num_samples // 2), np.ones(num_samples // 2)])
    
    \# Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    \# Create datasets
    dataset = TensorDataset(X_tensor, y_tensor)
    
    \# Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

def demonstrate_edge_computing():
    """Demonstrate edge computing system."""
    print("Edge Computing System Demonstration")
    print("=" * 50)
    
    \# Create device specification
    device_spec = EdgeDeviceSpec(
        device_type=EdgeDeviceType.BEDSIDE_MONITOR,
        cpu_cores=4,
        memory_gb=8.0,
        storage_gb=64.0,
        gpu_available=False,
        power_budget_watts=15.0,
        clinical_environment="icu",
        real_time_requirements=True,
        max_latency_ms=50.0
    )
    
    \# Create configuration
    config = EdgeConfig(
        device_spec=device_spec,
        compression_method=CompressionMethod.QUANTIZATION,
        processing_mode=ProcessingMode.REAL_TIME,
        target_model_size_mb=5.0,
        target_inference_time_ms=25.0,
        quantization_bits=8,
        enable_monitoring=True
    )
    
    print(f"Device Configuration:")
    print(f"  Type: {device_spec.device_type.value}")
    print(f"  CPU cores: {device_spec.cpu_cores}")
    print(f"  Memory: {device_spec.memory_gb} GB")
    print(f"  Max latency: {device_spec.max_latency_ms} ms")
    print(f"  Compression: {config.compression_method.value}")
    
    \# Create sample data
    train_dataset, val_dataset = create_sample_data()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    \# Create model
    model = create_sample_edge_model()
    
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    \# Initialize edge system
    edge_system = EdgeComputingSystem(config)
    
    print(f"\nEdge computing system initialized")
    print("Note: This is a demonstration with synthetic data")
    print("In practice, you would:")
    print("1. Deploy on actual edge hardware")
    print("2. Integrate with medical devices")
    print("3. Implement clinical protocols")
    print("4. Ensure regulatory compliance")
    print("5. Monitor system performance")

if __name__ == "__main__":
    demonstrate_edge_computing()