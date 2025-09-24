# Chapter 20: Edge Computing in Healthcare

## Learning Objectives

By the end of this chapter, readers will be able to:

1. **Design and implement edge computing architectures** for real-time healthcare applications
2. **Deploy AI models on resource-constrained devices** including wearables and medical IoT devices
3. **Optimize model performance** for edge deployment through quantization, pruning, and knowledge distillation
4. **Implement secure edge-to-cloud communication** protocols for healthcare data
5. **Handle real-time data processing** for critical healthcare monitoring applications
6. **Ensure regulatory compliance** for edge-based medical devices and systems

## Introduction

Edge computing represents a paradigm shift in healthcare AI, bringing computational capabilities closer to the point of care and data generation. This approach addresses critical challenges in healthcare delivery including latency requirements for real-time monitoring, bandwidth limitations in remote areas, privacy concerns with sensitive patient data, and the need for continuous operation even when connectivity is limited.

Traditional cloud-based healthcare AI systems face several limitations that edge computing can address. First, the latency introduced by round-trip communication to distant cloud servers can be unacceptable for time-critical applications such as cardiac monitoring, fall detection, or emergency response systems. Second, the continuous transmission of high-resolution medical data to the cloud can overwhelm network bandwidth, particularly in resource-constrained environments. Third, sending sensitive patient data to external cloud services raises privacy and security concerns, especially in light of increasing regulatory requirements.

Edge computing in healthcare encompasses a broad spectrum of applications and deployment scenarios. At the device level, wearable sensors and implantable devices can perform local processing to detect anomalies and trigger alerts. At the facility level, edge servers in hospitals and clinics can provide real-time analysis of medical imaging, support surgical navigation systems, and enable immediate clinical decision support. At the regional level, edge infrastructure can support telemedicine applications, emergency response coordination, and population health monitoring.

The benefits of edge computing in healthcare are substantial. Reduced latency enables real-time monitoring and immediate response to critical events. Local processing preserves patient privacy by keeping sensitive data within institutional boundaries. Improved reliability ensures continuous operation even during network outages. Cost efficiency is achieved by reducing bandwidth requirements and cloud computing costs. Enhanced scalability allows systems to handle increasing data volumes without proportional increases in infrastructure costs.

However, edge computing also introduces unique challenges. Resource constraints on edge devices limit the complexity of AI models that can be deployed. Power consumption becomes critical for battery-operated devices. Thermal management is essential for maintaining performance in compact form factors. Security vulnerabilities may be introduced by distributed processing. Model management becomes complex when dealing with numerous edge devices requiring updates and maintenance.

This chapter provides a comprehensive guide to implementing edge computing solutions for healthcare applications. We cover the theoretical foundations of edge AI, practical implementation strategies for model optimization and deployment, real-time data processing architectures, and regulatory compliance considerations. The approaches presented here represent the current state-of-the-art in healthcare edge computing and have been validated through extensive research and clinical deployments.

## Theoretical Foundations

### Edge Computing Architecture

Edge computing architectures in healthcare typically follow a hierarchical structure with multiple tiers of processing capabilities. The mathematical framework for edge computing can be modeled as a distributed system with heterogeneous computational resources:

$$\mathcal{E} = \{E_1, E_2, \ldots, E_n\}$$

where each edge node $E_i$ is characterized by its computational capacity $C_i$, memory capacity $M_i$, power consumption $P_i$, and communication bandwidth $B_i$.

The optimization problem for task allocation across edge nodes can be formulated as:

$$\min \sum_{i=1}^n \sum_{j=1}^m x_{ij} \cdot (T_{ij} + \alpha \cdot E_{ij})$$

subject to:
- $\sum_{i=1}^n x_{ij} = 1$ (each task assigned to exactly one node)
- $\sum_{j=1}^m x_{ij} \cdot R_j \leq C_i$ (capacity constraints)
- $x_{ij} \in \{0, 1\}$ (binary assignment variables)

where $T_{ij}$ is the execution time for task $j$ on node $i$, $E_{ij}$ is the energy consumption, $\alpha$ is a weighting factor, and $R_j$ is the resource requirement for task $j$.

### Model Compression Techniques

Edge deployment requires significant model compression to fit within resource constraints. The primary techniques include:

**Quantization**: Reduces the precision of model weights and activations. Post-training quantization can be formulated as:

$$\hat{w} = \text{round}\left(\frac{w - z}{s}\right)$$

where $w$ is the original weight, $s$ is the scale factor, $z$ is the zero point, and $\hat{w}$ is the quantized weight.

**Pruning**: Removes redundant connections or neurons. Structured pruning removes entire channels or layers:

$$\mathcal{L}_{pruned} = \mathcal{L}_{original} + \lambda \sum_{i} \|W_i\|_1$$

where $\lambda$ controls the sparsity level and $W_i$ represents weight groups.

**Knowledge Distillation**: Transfers knowledge from a large teacher model to a smaller student model:

$$\mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y, \sigma(z_s)) + (1-\alpha) \mathcal{L}_{KL}(\sigma(z_t/T), \sigma(z_s/T))$$

where $z_s$ and $z_t$ are student and teacher logits, $T$ is temperature, and $\alpha$ balances the losses.

### Real-Time Processing Constraints

Real-time healthcare applications must satisfy strict timing constraints. The system can be modeled as a real-time task set $\tau = \{\tau_1, \tau_2, \ldots, \tau_n\}$ where each task $\tau_i$ has:

- **Period** $T_i$: Time between successive task instances
- **Deadline** $D_i$: Maximum allowable response time
- **Worst-case execution time** $C_i$: Maximum processing time

The schedulability condition for rate-monotonic scheduling is:

$$\sum_{i=1}^n \frac{C_i}{T_i} \leq n(2^{1/n} - 1)$$

For healthcare applications, deadlines are often much stricter than periods, requiring earliest deadline first (EDF) scheduling with the condition:

$$\sum_{i=1}^n \frac{C_i}{T_i} \leq 1$$

### Energy-Efficient Computing

Power consumption is critical for battery-operated healthcare devices. The power model for edge devices can be expressed as:

$$P_{total} = P_{static} + P_{dynamic}$$

where:
$$P_{dynamic} = \alpha \cdot C_{eff} \cdot V_{dd}^2 \cdot f$$

$\alpha$ is the activity factor, $C_{eff}$ is the effective capacitance, $V_{dd}$ is the supply voltage, and $f$ is the operating frequency.

Dynamic voltage and frequency scaling (DVFS) can optimize energy consumption:

$$E = \int_0^T P(t) dt = \int_0^T (\alpha \cdot C_{eff} \cdot V_{dd}(t)^2 \cdot f(t) + P_{static}) dt$$

## Implementation Framework

### Comprehensive Edge Computing System

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import pickle
import threading
import time
import queue
import asyncio
import websockets
import psutil
import GPUtil
from collections import deque
import cv2
import pyaudio
import wave
import struct
import socket
import ssl
from cryptography.fernet import Fernet
import hashlib
import hmac
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeDevice:
    """
    Base class for edge computing devices in healthcare.
    
    Provides common functionality for resource monitoring, model deployment,
    and communication with cloud services.
    """
    
    def __init__(self,
                 device_id: str,
                 device_type: str,
                 computational_capacity: float,
                 memory_capacity: float,
                 power_budget: float,
                 communication_bandwidth: float):
        """
        Initialize edge device.
        
        Args:
            device_id: Unique device identifier
            device_type: Type of device ('wearable', 'gateway', 'server')
            computational_capacity: FLOPS capacity
            memory_capacity: Memory in GB
            power_budget: Power budget in watts
            communication_bandwidth: Bandwidth in Mbps
        """
        self.device_id = device_id
        self.device_type = device_type
        self.computational_capacity = computational_capacity
        self.memory_capacity = memory_capacity
        self.power_budget = power_budget
        self.communication_bandwidth = communication_bandwidth
        
        # Resource monitoring
        self.cpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.power_usage_history = deque(maxlen=100)
        self.temperature_history = deque(maxlen=100)
        
        # Model management
        self.deployed_models = {}
        self.model_performance_history = {}
        
        # Communication
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Real-time processing
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        
        logger.info(f"Initialized edge device {device_id} of type {device_type}")
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor device resources."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Power usage (simulated for demonstration)
            power_usage = np.random.normal(self.power_budget * 0.7, self.power_budget * 0.1)
            power_usage = max(0, min(power_usage, self.power_budget))
            
            # Temperature (simulated)
            temperature = np.random.normal(45, 5)  # Celsius
            
            # Update history
            self.cpu_usage_history.append(cpu_usage)
            self.memory_usage_history.append(memory_usage)
            self.power_usage_history.append(power_usage)
            self.temperature_history.append(temperature)
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'power_usage': power_usage,
                'temperature': temperature,
                'available_memory': memory.available / (1024**3)  # GB
            }
        
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            return {}
    
    def deploy_model(self, model_name: str, model: nn.Module, optimization_config: Dict[str, Any] = None):
        """Deploy AI model to edge device."""
        try:
            # Optimize model for edge deployment
            if optimization_config:
                model = self._optimize_model(model, optimization_config)
            
            # Store model
            self.deployed_models[model_name] = {
                'model': model,
                'deployment_time': datetime.now(),
                'inference_count': 0,
                'total_inference_time': 0.0
            }
            
            logger.info(f"Deployed model {model_name} to device {self.device_id}")
            
        except Exception as e:
            logger.error(f"Error deploying model {model_name}: {e}")
    
    def _optimize_model(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Optimize model for edge deployment."""
        optimized_model = model
        
        # Quantization
        if config.get('quantize', False):
            optimized_model = torch.quantization.quantize_dynamic(
                optimized_model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
        
        # Pruning (simplified implementation)
        if config.get('prune', False):
            prune_ratio = config.get('prune_ratio', 0.2)
            optimized_model = self._prune_model(optimized_model, prune_ratio)
            logger.info(f"Applied pruning with ratio {prune_ratio}")
        
        return optimized_model
    
    def _prune_model(self, model: nn.Module, prune_ratio: float) -> nn.Module:
        """Apply magnitude-based pruning to model."""
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=prune_ratio)
                prune.remove(module, 'weight')
        
        return model
    
    def inference(self, model_name: str, input_data: torch.Tensor) -> Dict[str, Any]:
        """Perform inference on edge device."""
        if model_name not in self.deployed_models:
            raise ValueError(f"Model {model_name} not deployed on device {self.device_id}")
        
        model_info = self.deployed_models[model_name]
        model = model_info['model']
        
        start_time = time.time()
        
        try:
            model.eval()
            with torch.no_grad():
                output = model(input_data)
            
            inference_time = time.time() - start_time
            
            # Update performance metrics
            model_info['inference_count'] += 1
            model_info['total_inference_time'] += inference_time
            
            return {
                'output': output,
                'inference_time': inference_time,
                'device_id': self.device_id,
                'model_name': model_name,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return {'error': str(e)}
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data for secure transmission."""
        return self.cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt received data."""
        return self.cipher.decrypt(encrypted_data)
    
    def start_processing(self):
        """Start real-time processing thread."""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        logger.info(f"Started processing on device {self.device_id}")
    
    def stop_processing(self):
        """Stop real-time processing."""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        logger.info(f"Stopped processing on device {self.device_id}")
    
    def _processing_loop(self):
        """Main processing loop for real-time data."""
        while self.is_running:
            try:
                # Check for new data
                if not self.processing_queue.empty():
                    task = self.processing_queue.get(timeout=1)
                    
                    # Process task
                    result = self._process_task(task)
                    
                    # Store result
                    self.result_queue.put(result)
                
                # Monitor resources
                resources = self.monitor_resources()
                
                # Adaptive processing based on resources
                if resources.get('cpu_usage', 0) > 90:
                    time.sleep(0.1)  # Throttle processing
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual task."""
        task_type = task.get('type', 'unknown')
        
        if task_type == 'inference':
            return self.inference(task['model_name'], task['input_data'])
        elif task_type == 'monitoring':
            return self.monitor_resources()
        else:
            return {'error': f'Unknown task type: {task_type}'}

class WearableDevice(EdgeDevice):
    """
    Wearable healthcare device with sensor data processing capabilities.
    
    Specialized for continuous monitoring applications with strict power
    and size constraints.
    """
    
    def __init__(self,
                 device_id: str,
                 sensor_types: List[str],
                 sampling_rate: float = 100.0,
                 battery_capacity: float = 1000.0):  # mAh
        """
        Initialize wearable device.
        
        Args:
            device_id: Unique device identifier
            sensor_types: List of sensor types ('ecg', 'ppg', 'accelerometer', etc.)
            sampling_rate: Sensor sampling rate in Hz
            battery_capacity: Battery capacity in mAh
        """
        super().__init__(
            device_id=device_id,
            device_type='wearable',
            computational_capacity=1e9,  # 1 GFLOPS
            memory_capacity=0.5,  # 512 MB
            power_budget=0.1,  # 100 mW
            communication_bandwidth=1.0  # 1 Mbps
        )
        
        self.sensor_types = sensor_types
        self.sampling_rate = sampling_rate
        self.battery_capacity = battery_capacity
        self.current_battery = battery_capacity
        
        # Sensor data buffers
        self.sensor_buffers = {sensor: deque(maxlen=int(sampling_rate * 10)) 
                              for sensor in sensor_types}
        
        # Anomaly detection models
        self.anomaly_thresholds = {}
        
        logger.info(f"Initialized wearable device {device_id} with sensors: {sensor_types}")
    
    def simulate_sensor_data(self, sensor_type: str, duration: float = 1.0) -> np.ndarray:
        """Simulate sensor data for testing."""
        num_samples = int(self.sampling_rate * duration)
        
        if sensor_type == 'ecg':
            # Simulate ECG signal
            t = np.linspace(0, duration, num_samples)
            heart_rate = 70  # BPM
            signal = np.sin(2 * np.pi * heart_rate / 60 * t) + 0.1 * np.random.randn(num_samples)
            
        elif sensor_type == 'ppg':
            # Simulate PPG signal
            t = np.linspace(0, duration, num_samples)
            heart_rate = 70  # BPM
            signal = 0.5 * np.sin(2 * np.pi * heart_rate / 60 * t) + 0.05 * np.random.randn(num_samples)
            
        elif sensor_type == 'accelerometer':
            # Simulate accelerometer data (3-axis)
            signal = np.random.randn(num_samples, 3) * 0.1
            signal[:, 2] += 9.81  # Add gravity
            
        elif sensor_type == 'temperature':
            # Simulate body temperature
            base_temp = 37.0  # Celsius
            signal = base_temp + 0.5 * np.random.randn(num_samples)
            
        else:
            # Generic sensor
            signal = np.random.randn(num_samples)
        
        return signal
    
    def collect_sensor_data(self):
        """Collect data from all sensors."""
        sensor_data = {}
        
        for sensor_type in self.sensor_types:
            data = self.simulate_sensor_data(sensor_type, duration=0.1)  # 100ms of data
            
            # Add to buffer
            if len(data.shape) == 1:
                self.sensor_buffers[sensor_type].extend(data)
            else:
                for sample in data:
                    self.sensor_buffers[sensor_type].append(sample)
            
            sensor_data[sensor_type] = data
        
        # Update battery (simplified model)
        power_consumption = 0.01  # 10mW for 100ms
        self.current_battery -= power_consumption
        
        return sensor_data
    
    def detect_anomalies(self, sensor_data: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Detect anomalies in sensor data."""
        anomalies = {}
        
        for sensor_type, data in sensor_data.items():
            if sensor_type == 'ecg':
                # Simple heart rate anomaly detection
                if len(data) > 0:
                    heart_rate = self._estimate_heart_rate(data)
                    anomalies[sensor_type] = heart_rate < 50 or heart_rate > 120
                else:
                    anomalies[sensor_type] = False
                    
            elif sensor_type == 'accelerometer':
                # Fall detection based on acceleration magnitude
                if len(data) > 0:
                    magnitude = np.linalg.norm(data, axis=1) if len(data.shape) > 1 else np.abs(data)
                    anomalies[sensor_type] = np.any(magnitude > 20)  # High acceleration threshold
                else:
                    anomalies[sensor_type] = False
                    
            elif sensor_type == 'temperature':
                # Temperature anomaly detection
                if len(data) > 0:
                    avg_temp = np.mean(data)
                    anomalies[sensor_type] = avg_temp < 35.0 or avg_temp > 39.0
                else:
                    anomalies[sensor_type] = False
                    
            else:
                anomalies[sensor_type] = False
        
        return anomalies
    
    def _estimate_heart_rate(self, ecg_data: np.ndarray) -> float:
        """Estimate heart rate from ECG data."""
        if len(ecg_data) < self.sampling_rate:
            return 70.0  # Default heart rate
        
        # Simple peak detection
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(ecg_data, height=0.5, distance=int(self.sampling_rate * 0.4))
        
        if len(peaks) > 1:
            # Calculate heart rate from peak intervals
            intervals = np.diff(peaks) / self.sampling_rate  # seconds
            heart_rate = 60.0 / np.mean(intervals)  # BPM
            return heart_rate
        else:
            return 70.0  # Default if no peaks detected
    
    def get_battery_status(self) -> Dict[str, float]:
        """Get current battery status."""
        battery_percentage = (self.current_battery / self.battery_capacity) * 100
        
        return {
            'current_battery': self.current_battery,
            'battery_percentage': battery_percentage,
            'estimated_runtime': self.current_battery / 10.0  # hours at 10mA
        }

class EdgeGateway(EdgeDevice):
    """
    Edge gateway device for aggregating and processing data from multiple sources.
    
    Provides intermediate processing capabilities between edge devices and cloud.
    """
    
    def __init__(self,
                 device_id: str,
                 max_connected_devices: int = 50,
                 processing_capacity: float = 100e9):  # 100 GFLOPS
        """
        Initialize edge gateway.
        
        Args:
            device_id: Unique device identifier
            max_connected_devices: Maximum number of connected devices
            processing_capacity: Processing capacity in FLOPS
        """
        super().__init__(
            device_id=device_id,
            device_type='gateway',
            computational_capacity=processing_capacity,
            memory_capacity=8.0,  # 8 GB
            power_budget=50.0,  # 50 W
            communication_bandwidth=100.0  # 100 Mbps
        )
        
        self.max_connected_devices = max_connected_devices
        self.connected_devices = {}
        self.data_aggregation_buffer = deque(maxlen=1000)
        
        # Load balancing
        self.task_queue = queue.PriorityQueue()
        self.worker_threads = []
        
        logger.info(f"Initialized edge gateway {device_id}")
    
    def register_device(self, device: EdgeDevice):
        """Register a device with the gateway."""
        if len(self.connected_devices) >= self.max_connected_devices:
            raise ValueError("Maximum number of connected devices reached")
        
        self.connected_devices[device.device_id] = device
        logger.info(f"Registered device {device.device_id} with gateway {self.device_id}")
    
    def aggregate_data(self, time_window: float = 1.0) -> Dict[str, Any]:
        """Aggregate data from connected devices."""
        aggregated_data = {
            'timestamp': datetime.now(),
            'device_count': len(self.connected_devices),
            'data_summary': {}
        }
        
        for device_id, device in self.connected_devices.items():
            if isinstance(device, WearableDevice):
                # Collect recent sensor data
                device_data = {}
                for sensor_type, buffer in device.sensor_buffers.items():
                    if buffer:
                        recent_data = list(buffer)[-int(device.sampling_rate * time_window):]
                        if recent_data:
                            device_data[sensor_type] = {
                                'mean': float(np.mean(recent_data)),
                                'std': float(np.std(recent_data)),
                                'min': float(np.min(recent_data)),
                                'max': float(np.max(recent_data)),
                                'samples': len(recent_data)
                            }
                
                aggregated_data['data_summary'][device_id] = device_data
        
        self.data_aggregation_buffer.append(aggregated_data)
        return aggregated_data
    
    def detect_population_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies across the population of connected devices."""
        if len(self.data_aggregation_buffer) < 10:
            return {'status': 'insufficient_data'}
        
        # Analyze recent aggregated data
        recent_data = list(self.data_aggregation_buffer)[-10:]
        
        anomalies = {
            'timestamp': datetime.now(),
            'device_anomalies': {},
            'population_trends': {}
        }
        
        # Check for device-level anomalies
        for device_id in self.connected_devices.keys():
            device_anomalies = []
            
            for data_point in recent_data:
                if device_id in data_point['data_summary']:
                    device_data = data_point['data_summary'][device_id]
                    
                    # Check for extreme values
                    for sensor_type, stats in device_data.items():
                        if sensor_type == 'ecg':
                            # Heart rate anomaly
                            if stats['mean'] < 50 or stats['mean'] > 120:
                                device_anomalies.append(f"Abnormal heart rate: {stats['mean']:.1f}")
                        
                        elif sensor_type == 'temperature':
                            # Temperature anomaly
                            if stats['mean'] < 35.0 or stats['mean'] > 39.0:
                                device_anomalies.append(f"Abnormal temperature: {stats['mean']:.1f}°C")
            
            if device_anomalies:
                anomalies['device_anomalies'][device_id] = device_anomalies
        
        return anomalies
    
    def optimize_task_allocation(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Optimize task allocation across connected devices."""
        device_allocations = {device_id: [] for device_id in self.connected_devices.keys()}
        
        # Sort tasks by priority and computational requirements
        sorted_tasks = sorted(tasks, key=lambda x: (x.get('priority', 0), x.get('compute_requirement', 0)))
        
        for task in sorted_tasks:
            # Find best device for this task
            best_device = None
            best_score = float('inf')
            
            for device_id, device in self.connected_devices.items():
                # Calculate allocation score based on current load and capabilities
                current_load = len(device_allocations[device_id])
                capability_match = device.computational_capacity / task.get('compute_requirement', 1)
                
                score = current_load / capability_match
                
                if score < best_score:
                    best_score = score
                    best_device = device_id
            
            if best_device:
                device_allocations[best_device].append(task)
        
        return device_allocations

class RealTimeHealthMonitor:
    """
    Real-time health monitoring system using edge computing.
    
    Coordinates multiple edge devices for continuous patient monitoring
    with immediate anomaly detection and alert generation.
    """
    
    def __init__(self,
                 monitor_id: str,
                 alert_thresholds: Dict[str, Dict[str, float]] = None):
        """
        Initialize real-time health monitor.
        
        Args:
            monitor_id: Unique monitor identifier
            alert_thresholds: Thresholds for different alert types
        """
        self.monitor_id = monitor_id
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        
        # Device management
        self.registered_devices = {}
        self.device_status = {}
        
        # Alert system
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Data streaming
        self.data_streams = {}
        self.stream_processors = {}
        
        # Real-time processing
        self.is_monitoring = False
        self.monitoring_thread = None
        
        logger.info(f"Initialized real-time health monitor {monitor_id}")
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default alert thresholds for various health parameters."""
        return {
            'heart_rate': {
                'low_critical': 40,
                'low_warning': 50,
                'high_warning': 120,
                'high_critical': 150
            },
            'temperature': {
                'low_critical': 35.0,
                'low_warning': 36.0,
                'high_warning': 38.5,
                'high_critical': 40.0
            },
            'blood_pressure_systolic': {
                'low_critical': 70,
                'low_warning': 90,
                'high_warning': 140,
                'high_critical': 180
            },
            'blood_pressure_diastolic': {
                'low_critical': 40,
                'low_warning': 60,
                'high_warning': 90,
                'high_critical': 120
            },
            'oxygen_saturation': {
                'low_critical': 85,
                'low_warning': 90,
                'high_warning': 100,
                'high_critical': 100
            }
        }
    
    def register_device(self, device: EdgeDevice, patient_id: str = None):
        """Register device for monitoring."""
        self.registered_devices[device.device_id] = {
            'device': device,
            'patient_id': patient_id,
            'registration_time': datetime.now(),
            'last_data_time': None
        }
        
        self.device_status[device.device_id] = 'active'
        
        logger.info(f"Registered device {device.device_id} for patient {patient_id}")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()
        
        logger.info("Started real-time health monitoring")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Stopped real-time health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check all registered devices
                for device_id, device_info in self.registered_devices.items():
                    device = device_info['device']
                    patient_id = device_info['patient_id']
                    
                    # Collect data from device
                    if isinstance(device, WearableDevice):
                        sensor_data = device.collect_sensor_data()
                        
                        # Process data and check for anomalies
                        anomalies = device.detect_anomalies(sensor_data)
                        
                        # Generate alerts if necessary
                        self._process_anomalies(device_id, patient_id, anomalies, sensor_data)
                        
                        # Update device status
                        device_info['last_data_time'] = datetime.now()
                        self.device_status[device_id] = 'active'
                    
                    # Check device health
                    resources = device.monitor_resources()
                    self._check_device_health(device_id, resources)
                
                # Check for device timeouts
                self._check_device_timeouts()
                
                time.sleep(0.1)  # 100ms monitoring cycle
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _process_anomalies(self,
                          device_id: str,
                          patient_id: str,
                          anomalies: Dict[str, bool],
                          sensor_data: Dict[str, np.ndarray]):
        """Process detected anomalies and generate alerts."""
        for sensor_type, is_anomaly in anomalies.items():
            if is_anomaly:
                # Extract relevant values
                if sensor_type == 'ecg' and len(sensor_data[sensor_type]) > 0:
                    device = self.registered_devices[device_id]['device']
                    heart_rate = device._estimate_heart_rate(sensor_data[sensor_type])
                    self._generate_alert(device_id, patient_id, 'heart_rate', heart_rate)
                
                elif sensor_type == 'temperature' and len(sensor_data[sensor_type]) > 0:
                    temperature = np.mean(sensor_data[sensor_type])
                    self._generate_alert(device_id, patient_id, 'temperature', temperature)
                
                elif sensor_type == 'accelerometer':
                    self._generate_alert(device_id, patient_id, 'fall_detected', 1.0)
    
    def _generate_alert(self,
                       device_id: str,
                       patient_id: str,
                       alert_type: str,
                       value: float):
        """Generate health alert."""
        alert_id = f"{device_id}_{alert_type}_{int(time.time())}"
        
        # Determine alert severity
        severity = self._determine_alert_severity(alert_type, value)
        
        alert = {
            'alert_id': alert_id,
            'device_id': device_id,
            'patient_id': patient_id,
            'alert_type': alert_type,
            'value': value,
            'severity': severity,
            'timestamp': datetime.now(),
            'acknowledged': False
        }
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(f"HEALTH ALERT: {severity} {alert_type} for patient {patient_id} "
                      f"(device {device_id}): {value}")
        
        # Trigger immediate response for critical alerts
        if severity == 'critical':
            self._trigger_emergency_response(alert)
    
    def _determine_alert_severity(self, alert_type: str, value: float) -> str:
        """Determine alert severity based on thresholds."""
        if alert_type in self.alert_thresholds:
            thresholds = self.alert_thresholds[alert_type]
            
            if value <= thresholds.get('low_critical', float('-inf')) or \
               value >= thresholds.get('high_critical', float('inf')):
                return 'critical'
            elif value <= thresholds.get('low_warning', float('-inf')) or \
                 value >= thresholds.get('high_warning', float('inf')):
                return 'warning'
            else:
                return 'info'
        
        return 'info'
    
    def _trigger_emergency_response(self, alert: Dict[str, Any]):
        """Trigger emergency response for critical alerts."""
        logger.critical(f"EMERGENCY RESPONSE TRIGGERED: {alert}")
        
        # In a real system, this would:
        # 1. Notify emergency contacts
        # 2. Alert medical staff
        # 3. Potentially contact emergency services
        # 4. Activate emergency protocols
        
        # For demonstration, we'll just log the emergency
        emergency_log = {
            'timestamp': datetime.now(),
            'alert_id': alert['alert_id'],
            'patient_id': alert['patient_id'],
            'emergency_type': alert['alert_type'],
            'severity': alert['severity'],
            'response_actions': [
                'Medical staff notified',
                'Emergency contacts alerted',
                'Patient location tracked'
            ]
        }
        
        logger.critical(f"Emergency response log: {emergency_log}")
    
    def _check_device_health(self, device_id: str, resources: Dict[str, float]):
        """Check device health and performance."""
        # Check for resource issues
        if resources.get('cpu_usage', 0) > 95:
            logger.warning(f"High CPU usage on device {device_id}: {resources['cpu_usage']:.1f}%")
        
        if resources.get('memory_usage', 0) > 90:
            logger.warning(f"High memory usage on device {device_id}: {resources['memory_usage']:.1f}%")
        
        if resources.get('temperature', 0) > 70:
            logger.warning(f"High temperature on device {device_id}: {resources['temperature']:.1f}°C")
        
        # Check battery for wearable devices
        device = self.registered_devices[device_id]['device']
        if isinstance(device, WearableDevice):
            battery_status = device.get_battery_status()
            if battery_status['battery_percentage'] < 20:
                logger.warning(f"Low battery on device {device_id}: {battery_status['battery_percentage']:.1f}%")
    
    def _check_device_timeouts(self):
        """Check for devices that haven't reported data recently."""
        current_time = datetime.now()
        timeout_threshold = timedelta(seconds=30)  # 30 second timeout
        
        for device_id, device_info in self.registered_devices.items():
            last_data_time = device_info['last_data_time']
            
            if last_data_time and (current_time - last_data_time) > timeout_threshold:
                if self.device_status[device_id] != 'timeout':
                    logger.error(f"Device {device_id} timeout - no data received")
                    self.device_status[device_id] = 'timeout'
                    
                    # Generate device timeout alert
                    self._generate_alert(
                        device_id,
                        device_info['patient_id'],
                        'device_timeout',
                        (current_time - last_data_time).total_seconds()
                    )
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        active_device_count = sum(1 for status in self.device_status.values() if status == 'active')
        timeout_device_count = sum(1 for status in self.device_status.values() if status == 'timeout')
        
        critical_alerts = [alert for alert in self.active_alerts.values() if alert['severity'] == 'critical']
        warning_alerts = [alert for alert in self.active_alerts.values() if alert['severity'] == 'warning']
        
        return {
            'monitor_id': self.monitor_id,
            'is_monitoring': self.is_monitoring,
            'total_devices': len(self.registered_devices),
            'active_devices': active_device_count,
            'timeout_devices': timeout_device_count,
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len(critical_alerts),
            'warning_alerts': len(warning_alerts),
            'total_alerts_today': len([alert for alert in self.alert_history 
                                     if alert['timestamp'].date() == datetime.now().date()])
        }

class EdgeModelOptimizer:
    """
    Model optimization toolkit for edge deployment.
    
    Provides comprehensive optimization techniques including quantization,
    pruning, knowledge distillation, and architecture search.
    """
    
    def __init__(self):
        """Initialize edge model optimizer."""
        self.optimization_history = []
        logger.info("Initialized edge model optimizer")
    
    def quantize_model(self,
                      model: nn.Module,
                      quantization_type: str = 'dynamic',
                      calibration_data: DataLoader = None) -> nn.Module:
        """
        Quantize model for edge deployment.
        
        Args:
            model: PyTorch model to quantize
            quantization_type: Type of quantization ('dynamic', 'static', 'qat')
            calibration_data: Calibration data for static quantization
            
        Returns:
            quantized_model: Quantized model
        """
        if quantization_type == 'dynamic':
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
        elif quantization_type == 'static':
            # Static quantization
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate with representative data
            with torch.no_grad():
                for data, _ in calibration_data:
                    model(data)
            
            quantized_model = torch.quantization.convert(model, inplace=False)
            
        elif quantization_type == 'qat':
            # Quantization-aware training
            model.train()
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(model, inplace=True)
            
            # Note: In practice, you would train the model here
            # For demonstration, we'll just convert directly
            model.eval()
            quantized_model = torch.quantization.convert(model, inplace=False)
            
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        # Record optimization
        self.optimization_history.append({
            'type': 'quantization',
            'method': quantization_type,
            'timestamp': datetime.now(),
            'original_size': self._get_model_size(model),
            'optimized_size': self._get_model_size(quantized_model)
        })
        
        logger.info(f"Applied {quantization_type} quantization")
        return quantized_model
    
    def prune_model(self,
                   model: nn.Module,
                   pruning_ratio: float = 0.2,
                   pruning_type: str = 'magnitude') -> nn.Module:
        """
        Prune model to reduce size and computation.
        
        Args:
            model: PyTorch model to prune
            pruning_ratio: Fraction of weights to prune
            pruning_type: Type of pruning ('magnitude', 'structured')
            
        Returns:
            pruned_model: Pruned model
        """
        import torch.nn.utils.prune as prune
        
        pruned_model = copy.deepcopy(model)
        
        if pruning_type == 'magnitude':
            # Magnitude-based unstructured pruning
            for name, module in pruned_model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                    prune.remove(module, 'weight')
                    
        elif pruning_type == 'structured':
            # Structured pruning (remove entire channels/filters)
            for name, module in pruned_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
                    prune.remove(module, 'weight')
                elif isinstance(module, nn.Linear):
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
                    prune.remove(module, 'weight')
        
        # Record optimization
        self.optimization_history.append({
            'type': 'pruning',
            'method': pruning_type,
            'pruning_ratio': pruning_ratio,
            'timestamp': datetime.now(),
            'original_size': self._get_model_size(model),
            'optimized_size': self._get_model_size(pruned_model)
        })
        
        logger.info(f"Applied {pruning_type} pruning with ratio {pruning_ratio}")
        return pruned_model
    
    def knowledge_distillation(self,
                              teacher_model: nn.Module,
                              student_model: nn.Module,
                              train_loader: DataLoader,
                              num_epochs: int = 10,
                              temperature: float = 4.0,
                              alpha: float = 0.7) -> nn.Module:
        """
        Apply knowledge distillation to create smaller student model.
        
        Args:
            teacher_model: Large teacher model
            student_model: Smaller student model
            train_loader: Training data loader
            num_epochs: Number of training epochs
            temperature: Temperature for softmax
            alpha: Weight for distillation loss
            
        Returns:
            distilled_model: Trained student model
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        teacher_model = teacher_model.to(device)
        student_model = student_model.to(device)
        
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                
                # Student predictions
                student_outputs = student_model(data)
                
                # Compute losses
                ce_loss = criterion_ce(student_outputs, target)
                
                # Distillation loss
                teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)
                student_soft = F.log_softmax(student_outputs / temperature, dim=1)
                kl_loss = criterion_kl(student_soft, teacher_soft) * (temperature ** 2)
                
                # Combined loss
                loss = alpha * kl_loss + (1 - alpha) * ce_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Distillation epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Record optimization
        self.optimization_history.append({
            'type': 'knowledge_distillation',
            'temperature': temperature,
            'alpha': alpha,
            'epochs': num_epochs,
            'timestamp': datetime.now(),
            'teacher_size': self._get_model_size(teacher_model),
            'student_size': self._get_model_size(student_model)
        })
        
        logger.info("Completed knowledge distillation")
        return student_model
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def benchmark_model(self,
                       model: nn.Module,
                       input_shape: Tuple[int, ...],
                       num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance on edge device.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            benchmark_results: Performance metrics
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark inference time
        inference_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        # Calculate statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        
        # Calculate throughput
        throughput = 1.0 / avg_inference_time
        
        # Model size
        model_size = self._get_model_size(model)
        
        benchmark_results = {
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time,
            'min_inference_time': min_inference_time,
            'max_inference_time': max_inference_time,
            'throughput': throughput,
            'model_size_bytes': model_size,
            'model_size_mb': model_size / (1024 * 1024),
            'device': str(device)
        }
        
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results

# Example usage and demonstration
def main():
    """Demonstrate the edge computing system."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create edge computing demonstration
    logger.info("Creating edge computing demonstration...")
    
    # Initialize real-time health monitor
    monitor = RealTimeHealthMonitor(
        monitor_id="hospital_edge_monitor_001",
        alert_thresholds=None  # Use defaults
    )
    
    # Create wearable devices for multiple patients
    wearable_devices = []
    patient_ids = ['patient_001', 'patient_002', 'patient_003']
    
    for i, patient_id in enumerate(patient_ids):
        device = WearableDevice(
            device_id=f"wearable_{i+1:03d}",
            sensor_types=['ecg', 'ppg', 'accelerometer', 'temperature'],
            sampling_rate=100.0,
            battery_capacity=1000.0
        )
        
        # Register device with monitor
        monitor.register_device(device, patient_id)
        wearable_devices.append(device)
    
    # Create edge gateway
    gateway = EdgeGateway(
        device_id="gateway_001",
        max_connected_devices=50,
        processing_capacity=100e9
    )
    
    # Register wearable devices with gateway
    for device in wearable_devices:
        gateway.register_device(device)
    
    # Create and optimize models for edge deployment
    logger.info("Creating and optimizing models for edge deployment...")
    
    # Create a simple health monitoring model
    class HealthMonitoringModel(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=64, num_classes=2):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Create original model
    original_model = HealthMonitoringModel(input_dim=4, hidden_dim=128, num_classes=2)
    
    # Initialize model optimizer
    optimizer = EdgeModelOptimizer()
    
    # Apply various optimizations
    logger.info("Applying model optimizations...")
    
    # Quantization
    quantized_model = optimizer.quantize_model(
        copy.deepcopy(original_model),
        quantization_type='dynamic'
    )
    
    # Pruning
    pruned_model = optimizer.prune_model(
        copy.deepcopy(original_model),
        pruning_ratio=0.3,
        pruning_type='magnitude'
    )
    
    # Deploy models to edge devices
    for device in wearable_devices:
        device.deploy_model('health_monitor', copy.deepcopy(quantized_model))
    
    gateway.deploy_model('health_aggregator', copy.deepcopy(pruned_model))
    
    # Benchmark models
    logger.info("Benchmarking model performance...")
    
    input_shape = (1, 4)  # Batch size 1, 4 features
    
    original_benchmark = optimizer.benchmark_model(original_model, input_shape)
    quantized_benchmark = optimizer.benchmark_model(quantized_model, input_shape)
    pruned_benchmark = optimizer.benchmark_model(pruned_model, input_shape)
    
    # Start real-time monitoring
    logger.info("Starting real-time monitoring simulation...")
    
    monitor.start_monitoring()
    
    # Simulate real-time data collection and processing
    simulation_duration = 30  # seconds
    start_time = time.time()
    
    data_collection_history = []
    
    while time.time() - start_time < simulation_duration:
        # Collect data from all devices
        for device in wearable_devices:
            # Simulate sensor data collection
            sensor_data = device.collect_sensor_data()
            
            # Process data with deployed model
            if 'health_monitor' in device.deployed_models:
                # Create feature vector from sensor data
                features = []
                for sensor_type in ['ecg', 'ppg', 'temperature']:
                    if sensor_type in sensor_data and len(sensor_data[sensor_type]) > 0:
                        features.append(np.mean(sensor_data[sensor_type]))
                    else:
                        features.append(0.0)
                
                # Add accelerometer magnitude
                if 'accelerometer' in sensor_data and len(sensor_data['accelerometer']) > 0:
                    acc_data = sensor_data['accelerometer']
                    if len(acc_data.shape) > 1:
                        magnitude = np.mean(np.linalg.norm(acc_data, axis=1))
                    else:
                        magnitude = np.mean(np.abs(acc_data))
                    features.append(magnitude)
                else:
                    features.append(9.81)  # Default gravity
                
                # Perform inference
                if len(features) == 4:
                    input_tensor = torch.tensor([features], dtype=torch.float32)
                    result = device.inference('health_monitor', input_tensor)
                    
                    data_collection_history.append({
                        'timestamp': datetime.now(),
                        'device_id': device.device_id,
                        'features': features,
                        'inference_time': result.get('inference_time', 0),
                        'battery_status': device.get_battery_status()
                    })
        
        # Gateway aggregation
        aggregated_data = gateway.aggregate_data(time_window=1.0)
        population_anomalies = gateway.detect_population_anomalies()
        
        time.sleep(0.5)  # 500ms cycle
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Generate results and analysis
    logger.info("Generating results and analysis...")
    
    # Get monitoring status
    monitoring_status = monitor.get_monitoring_status()
    
    # Save results
    results_dir = Path("edge_computing_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save benchmark results
    benchmark_results = {
        'original_model': original_benchmark,
        'quantized_model': quantized_benchmark,
        'pruned_model': pruned_benchmark,
        'optimization_history': optimizer.optimization_history
    }
    
    with open(results_dir / 'benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    # Save monitoring results
    monitoring_results = {
        'monitoring_status': monitoring_status,
        'data_collection_history': data_collection_history[-100:],  # Last 100 entries
        'alert_history': [dict(alert) for alert in list(monitor.alert_history)[-50:]]  # Last 50 alerts
    }
    
    with open(results_dir / 'monitoring_results.json', 'w') as f:
        json.dump(monitoring_results, f, indent=2, default=str)
    
    # Create visualizations
    plt.figure(figsize=(20, 12))
    
    # Model performance comparison
    plt.subplot(3, 4, 1)
    models = ['Original', 'Quantized', 'Pruned']
    inference_times = [
        original_benchmark['avg_inference_time'] * 1000,
        quantized_benchmark['avg_inference_time'] * 1000,
        pruned_benchmark['avg_inference_time'] * 1000
    ]
    
    plt.bar(models, inference_times)
    plt.ylabel('Inference Time (ms)')
    plt.title('Model Inference Time Comparison')
    plt.grid(True, alpha=0.3)
    
    # Model size comparison
    plt.subplot(3, 4, 2)
    model_sizes = [
        original_benchmark['model_size_mb'],
        quantized_benchmark['model_size_mb'],
        pruned_benchmark['model_size_mb']
    ]
    
    plt.bar(models, model_sizes)
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.grid(True, alpha=0.3)
    
    # Throughput comparison
    plt.subplot(3, 4, 3)
    throughputs = [
        original_benchmark['throughput'],
        quantized_benchmark['throughput'],
        pruned_benchmark['throughput']
    ]
    
    plt.bar(models, throughputs)
    plt.ylabel('Throughput (inferences/sec)')
    plt.title('Model Throughput Comparison')
    plt.grid(True, alpha=0.3)
    
    # Device resource usage over time
    plt.subplot(3, 4, 4)
    if wearable_devices[0].cpu_usage_history:
        plt.plot(list(wearable_devices[0].cpu_usage_history), label='CPU Usage')
        plt.plot(list(wearable_devices[0].memory_usage_history), label='Memory Usage')
        plt.ylabel('Usage (%)')
        plt.title('Device Resource Usage')
        plt.legend()
        plt.grid(True)
    
    # Battery usage over time
    plt.subplot(3, 4, 5)
    battery_history = []
    timestamps = []
    
    for entry in data_collection_history:
        if entry['device_id'] == wearable_devices[0].device_id:
            battery_history.append(entry['battery_status']['battery_percentage'])
            timestamps.append(entry['timestamp'])
    
    if battery_history:
        plt.plot(battery_history)
        plt.ylabel('Battery (%)')
        plt.title('Battery Usage Over Time')
        plt.grid(True)
    
    # Inference time distribution
    plt.subplot(3, 4, 6)
    inference_times_hist = [entry['inference_time'] * 1000 for entry in data_collection_history]
    
    if inference_times_hist:
        plt.hist(inference_times_hist, bins=20, alpha=0.7)
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Time Distribution')
        plt.grid(True, alpha=0.3)
    
    # Alert distribution by type
    plt.subplot(3, 4, 7)
    alert_types = {}
    for alert in monitor.alert_history:
        alert_type = alert['alert_type']
        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
    
    if alert_types:
        plt.bar(alert_types.keys(), alert_types.values())
        plt.xlabel('Alert Type')
        plt.ylabel('Count')
        plt.title('Alert Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Device status overview
    plt.subplot(3, 4, 8)
    device_statuses = list(monitor.device_status.values())
    status_counts = {}
    for status in device_statuses:
        status_counts[status] = status_counts.get(status, 0) + 1
    
    plt.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
    plt.title('Device Status Distribution')
    
    # Power consumption over time
    plt.subplot(3, 4, 9)
    if wearable_devices[0].power_usage_history:
        plt.plot(list(wearable_devices[0].power_usage_history))
        plt.ylabel('Power (W)')
        plt.title('Power Consumption')
        plt.grid(True)
    
    # Temperature monitoring
    plt.subplot(3, 4, 10)
    if wearable_devices[0].temperature_history:
        plt.plot(list(wearable_devices[0].temperature_history))
        plt.ylabel('Temperature (°C)')
        plt.title('Device Temperature')
        plt.grid(True)
    
    # Model optimization impact
    plt.subplot(3, 4, 11)
    optimization_types = ['Original', 'Quantized', 'Pruned']
    size_reduction = [
        0,  # Original
        (1 - quantized_benchmark['model_size_mb'] / original_benchmark['model_size_mb']) * 100,
        (1 - pruned_benchmark['model_size_mb'] / original_benchmark['model_size_mb']) * 100
    ]
    
    plt.bar(optimization_types, size_reduction)
    plt.ylabel('Size Reduction (%)')
    plt.title('Model Size Reduction')
    plt.grid(True, alpha=0.3)
    
    # Latency improvement
    plt.subplot(3, 4, 12)
    latency_improvement = [
        0,  # Original
        (1 - quantized_benchmark['avg_inference_time'] / original_benchmark['avg_inference_time']) * 100,
        (1 - pruned_benchmark['avg_inference_time'] / original_benchmark['avg_inference_time']) * 100
    ]
    
    plt.bar(optimization_types, latency_improvement)
    plt.ylabel('Latency Improvement (%)')
    plt.title('Inference Latency Improvement')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'edge_computing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed performance analysis
    plt.figure(figsize=(15, 10))
    
    # Real-time processing timeline
    plt.subplot(2, 3, 1)
    processing_times = [entry['timestamp'] for entry in data_collection_history]
    inference_times = [entry['inference_time'] * 1000 for entry in data_collection_history]
    
    if processing_times and inference_times:
        # Convert timestamps to relative seconds
        start_timestamp = processing_times[0]
        relative_times = [(t - start_timestamp).total_seconds() for t in processing_times]
        
        plt.scatter(relative_times, inference_times, alpha=0.6)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Inference Time (ms)')
        plt.title('Real-time Processing Performance')
        plt.grid(True)
    
    # Battery drain analysis
    plt.subplot(2, 3, 2)
    device_battery_data = {}
    
    for entry in data_collection_history:
        device_id = entry['device_id']
        if device_id not in device_battery_data:
            device_battery_data[device_id] = []
        device_battery_data[device_id].append(entry['battery_status']['battery_percentage'])
    
    for device_id, battery_data in device_battery_data.items():
        plt.plot(battery_data, label=device_id)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Battery (%)')
    plt.title('Battery Drain Comparison')
    plt.legend()
    plt.grid(True)
    
    # Resource utilization heatmap
    plt.subplot(2, 3, 3)
    resource_data = []
    
    for device in wearable_devices:
        device_resources = [
            np.mean(list(device.cpu_usage_history)) if device.cpu_usage_history else 0,
            np.mean(list(device.memory_usage_history)) if device.memory_usage_history else 0,
            np.mean(list(device.power_usage_history)) if device.power_usage_history else 0,
            np.mean(list(device.temperature_history)) if device.temperature_history else 0
        ]
        resource_data.append(device_resources)
    
    if resource_data:
        resource_data = np.array(resource_data)
        im = plt.imshow(resource_data, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im)
        plt.xlabel('Resource Type')
        plt.ylabel('Device')
        plt.title('Resource Utilization Heatmap')
        plt.xticks(range(4), ['CPU', 'Memory', 'Power', 'Temperature'])
        plt.yticks(range(len(wearable_devices)), [d.device_id for d in wearable_devices])
    
    # Alert severity timeline
    plt.subplot(2, 3, 4)
    alert_times = []
    alert_severities = []
    severity_map = {'info': 1, 'warning': 2, 'critical': 3}
    
    for alert in monitor.alert_history:
        alert_times.append(alert['timestamp'])
        alert_severities.append(severity_map.get(alert['severity'], 1))
    
    if alert_times:
        start_time = alert_times[0]
        relative_alert_times = [(t - start_time).total_seconds() for t in alert_times]
        
        colors = ['blue' if s == 1 else 'orange' if s == 2 else 'red' for s in alert_severities]
        plt.scatter(relative_alert_times, alert_severities, c=colors, alpha=0.7)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Alert Severity')
        plt.title('Alert Timeline')
        plt.yticks([1, 2, 3], ['Info', 'Warning', 'Critical'])
        plt.grid(True)
    
    # Communication efficiency
    plt.subplot(2, 3, 5)
    data_sizes = []
    transmission_times = []
    
    # Simulate communication data
    for entry in data_collection_history:
        # Estimate data size based on features
        data_size = len(entry['features']) * 4  # 4 bytes per float
        data_sizes.append(data_size)
        
        # Simulate transmission time based on bandwidth
        transmission_time = data_size / (1024 * 1024)  # Assume 1 MB/s
        transmission_times.append(transmission_time * 1000)  # Convert to ms
    
    if data_sizes and transmission_times:
        plt.scatter(data_sizes, transmission_times, alpha=0.6)
        plt.xlabel('Data Size (bytes)')
        plt.ylabel('Transmission Time (ms)')
        plt.title('Communication Efficiency')
        plt.grid(True)
    
    # Overall system performance
    plt.subplot(2, 3, 6)
    performance_metrics = [
        monitoring_status['active_devices'] / monitoring_status['total_devices'] * 100,
        (1 - monitoring_status['critical_alerts'] / max(monitoring_status['active_alerts'], 1)) * 100,
        np.mean([entry['inference_time'] for entry in data_collection_history]) * 1000 if data_collection_history else 0
    ]
    
    metric_names = ['Device\nUptime (%)', 'System\nHealth (%)', 'Avg Latency\n(ms)']
    
    bars = plt.bar(metric_names, performance_metrics)
    bars[0].set_color('green')
    bars[1].set_color('blue')
    bars[2].set_color('orange')
    
    plt.title('Overall System Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'edge_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Edge computing demonstration completed!")
    logger.info(f"Results saved to {results_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("EDGE COMPUTING DEMONSTRATION SUMMARY")
    print("="*60)
    print(f"Monitoring Status: {monitoring_status}")
    print(f"\nModel Performance Comparison:")
    print(f"Original Model: {original_benchmark['avg_inference_time']*1000:.2f}ms, {original_benchmark['model_size_mb']:.2f}MB")
    print(f"Quantized Model: {quantized_benchmark['avg_inference_time']*1000:.2f}ms, {quantized_benchmark['model_size_mb']:.2f}MB")
    print(f"Pruned Model: {pruned_benchmark['avg_inference_time']*1000:.2f}ms, {pruned_benchmark['model_size_mb']:.2f}MB")
    print(f"\nData Collection: {len(data_collection_history)} samples processed")
    print(f"Alerts Generated: {len(monitor.alert_history)}")
    print(f"Critical Alerts: {monitoring_status['critical_alerts']}")
    
    return monitor, wearable_devices, gateway, optimizer

if __name__ == "__main__":
    main()
```

## Advanced Edge Computing Techniques

### Adaptive Model Selection

Edge devices can dynamically select between multiple models based on current resource availability and performance requirements:

$$M^*(t) = \arg\min_{M_i \in \mathcal{M}} \alpha \cdot L_i(t) + \beta \cdot C_i(t) + \gamma \cdot E_i(t)$$

where $L_i(t)$ is the latency, $C_i(t)$ is the computational cost, $E_i(t)$ is the energy consumption, and $\alpha$, $\beta$, $\gamma$ are weighting factors.

### Hierarchical Edge Computing

Multi-tier edge architectures distribute computation across device, edge, and cloud layers:

**Device Layer**: Immediate processing for time-critical tasks
**Edge Layer**: Intermediate processing for complex analysis
**Cloud Layer**: Comprehensive processing for population-level insights

The task allocation problem can be formulated as:

$$\min \sum_{i=1}^N \sum_{j=1}^3 x_{ij} \cdot (T_{ij} + \lambda E_{ij})$$

subject to latency and resource constraints.

### Federated Edge Learning

Combining federated learning with edge computing enables collaborative model training while maintaining data locality:

$$\theta_{t+1} = \sum_{k=1}^K \frac{n_k}{n} \theta_t^{(k)}$$

where edge devices perform local training and share only model updates.

## Clinical Applications and Case Studies

### Case Study 1: Real-Time Cardiac Monitoring

Edge-based cardiac monitoring systems provide immediate detection of arrhythmias and cardiac events:

1. **Wearable ECG Devices**: Continuous monitoring with local anomaly detection
2. **Edge Processing**: Real-time signal analysis and pattern recognition
3. **Immediate Alerts**: Sub-second response times for critical events
4. **Clinical Integration**: Seamless integration with hospital monitoring systems

### Case Study 2: Surgical Navigation Systems

Edge computing enables real-time surgical guidance with minimal latency:

1. **Intraoperative Imaging**: Real-time processing of surgical imaging data
2. **Navigation Assistance**: Immediate feedback for surgical instruments
3. **Safety Monitoring**: Continuous assessment of surgical progress
4. **Outcome Prediction**: Real-time analysis of surgical success indicators

### Case Study 3: Emergency Response Systems

Edge-based emergency response systems provide immediate triage and coordination:

1. **Ambulance Systems**: Real-time patient assessment during transport
2. **Emergency Departments**: Immediate triage and resource allocation
3. **Disaster Response**: Coordinated response in resource-constrained environments
4. **Remote Medicine**: Healthcare delivery in areas with limited connectivity

## Security and Privacy Considerations

### Edge Security Architecture

Edge devices require comprehensive security frameworks:

1. **Device Authentication**: Cryptographic device identity verification
2. **Data Encryption**: End-to-end encryption for all data transmission
3. **Secure Boot**: Verified boot process to prevent tampering
4. **Runtime Protection**: Continuous monitoring for security threats

### Privacy-Preserving Edge Computing

Edge computing can enhance privacy through local processing:

1. **Data Minimization**: Only necessary data leaves the device
2. **Differential Privacy**: Noise addition for privacy protection
3. **Homomorphic Encryption**: Computation on encrypted data
4. **Secure Multiparty Computation**: Collaborative computation without data sharing

### Regulatory Compliance

Edge medical devices must comply with healthcare regulations:

1. **FDA 510(k)**: Medical device clearance for edge AI systems
2. **HIPAA Compliance**: Protection of patient health information
3. **GDPR Compliance**: European data protection requirements
4. **ISO 27001**: Information security management standards

## Future Directions and Emerging Technologies

### Neuromorphic Computing

Brain-inspired computing architectures offer ultra-low power edge processing:

1. **Spiking Neural Networks**: Event-driven processing for sensor data
2. **Memristive Devices**: Non-volatile memory for edge AI
3. **Neuromorphic Chips**: Specialized hardware for neural computation
4. **Adaptive Learning**: Real-time learning and adaptation

### 5G and Beyond

Next-generation wireless technologies enable new edge computing paradigms:

1. **Ultra-Low Latency**: Sub-millisecond communication delays
2. **Massive IoT**: Support for millions of connected devices
3. **Network Slicing**: Dedicated network resources for healthcare
4. **Mobile Edge Computing**: Computation at cellular base stations

### Quantum Edge Computing

Quantum technologies may revolutionize edge computing capabilities:

1. **Quantum Sensors**: Ultra-sensitive medical measurements
2. **Quantum Communication**: Unconditionally secure data transmission
3. **Quantum Machine Learning**: Exponential speedups for certain algorithms
4. **Hybrid Systems**: Classical-quantum edge computing architectures

## Summary

Edge computing represents a transformative approach to healthcare AI that brings computation closer to the point of care, enabling real-time processing, enhanced privacy, and improved reliability. This chapter has provided comprehensive coverage of edge computing architectures, model optimization techniques, real-time processing frameworks, and clinical applications.

Key takeaways include:

1. **Architectural Design**: Hierarchical edge computing architectures optimize resource utilization and performance
2. **Model Optimization**: Quantization, pruning, and knowledge distillation enable deployment on resource-constrained devices
3. **Real-Time Processing**: Careful system design ensures meeting strict timing requirements for healthcare applications
4. **Security and Privacy**: Edge computing can enhance privacy while requiring robust security frameworks
5. **Clinical Applications**: Diverse use cases from wearable monitoring to surgical navigation demonstrate the versatility of edge computing

The field continues to evolve rapidly, with emerging technologies such as neuromorphic computing, 5G networks, and quantum technologies promising to further enhance edge computing capabilities. However, the fundamental benefits of edge computing—reduced latency, enhanced privacy, and improved reliability—remain central to its value proposition in healthcare applications.

## References

1. Shi, W., et al. (2016). Edge computing: Vision and challenges. *IEEE Internet of Things Journal*, 3(5), 637-646. DOI: 10.1109/JIOT.2016.2579198

2. Yu, W., et al. (2017). A survey on the edge computing for the Internet of Things. *IEEE Access*, 6, 6900-6919. DOI: 10.1109/ACCESS.2017.2778504

3. Li, H., et al. (2018). Learning IoT in edge: Deep learning for the Internet of Things with edge computing. *IEEE Network*, 32(1), 96-101. DOI: 10.1109/MNET.2018.1700202

4. Chen, J., & Ran, X. (2019). Deep learning with edge computing: A review. *Proceedings of the IEEE*, 107(8), 1655-1674. DOI: 10.1109/JPROC.2019.2921977

5. Zhou, Z., et al. (2019). Edge intelligence: Paving the last mile of artificial intelligence with edge computing. *Proceedings of the IEEE*, 107(8), 1738-1762. DOI: 10.1109/JPROC.2019.2918951

6. Deng, S., et al. (2020). Edge intelligence: The confluence of edge computing and artificial intelligence. *IEEE Internet of Things Journal*, 7(8), 7457-7469. DOI: 10.1109/JIOT.2020.2984887

7. Wang, X., et al. (2020). Convergence of edge computing and deep learning: A comprehensive survey. *IEEE Communications Surveys & Tutorials*, 22(2), 869-904. DOI: 10.1109/COMST.2020.2970550

8. Zhang, J., et al. (2021). AI for edge computing: A survey. *ACM Computing Surveys*, 54(7), 1-37. DOI: 10.1145/3459992

9. Xu, D., et al. (2021). Edge computing for healthcare applications: Opportunities and challenges. *IEEE Network*, 35(6), 98-104. DOI: 10.1109/MNET.011.2100012

10. Liu, Y., et al. (2022). Privacy-preserving edge computing for healthcare IoT: A survey. *IEEE Internet of Things Journal*, 9(21), 20991-21007. DOI: 10.1109/JIOT.2022.3181899
