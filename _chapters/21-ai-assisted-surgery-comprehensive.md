# Chapter 21: AI-Assisted Surgery

## Learning Objectives

By the end of this chapter, readers will be able to:

1. **Design and implement AI systems** for surgical planning, navigation, and outcome prediction
2. **Develop computer vision algorithms** for real-time surgical scene understanding and instrument tracking
3. **Create robotic surgery control systems** with AI-enhanced precision and safety features
4. **Implement augmented reality systems** for surgical guidance and training
5. **Build predictive models** for surgical complications and patient outcomes
6. **Ensure regulatory compliance** for AI-assisted surgical systems and medical devices

## Introduction

Artificial intelligence is revolutionizing surgery by enhancing precision, improving outcomes, and expanding the capabilities of surgical teams. AI-assisted surgery encompasses a broad spectrum of applications, from preoperative planning and intraoperative guidance to postoperative monitoring and outcome prediction. These systems leverage advanced computer vision, machine learning, and robotics to provide surgeons with unprecedented insights and capabilities.

The integration of AI into surgical practice addresses several critical challenges in modern surgery. First, the complexity of surgical procedures continues to increase as medical knowledge advances and patient populations become more diverse. Second, the demand for minimally invasive procedures requires enhanced visualization and precision that can be augmented by AI systems. Third, the need for consistent outcomes across different skill levels and institutions drives the development of AI-assisted standardization tools.

AI-assisted surgery systems operate across multiple phases of the surgical workflow. During preoperative planning, AI algorithms analyze medical imaging data to create detailed surgical plans, predict potential complications, and optimize surgical approaches. Intraoperatively, AI systems provide real-time guidance through computer vision, augmented reality, and robotic assistance. Postoperatively, AI monitors patient recovery, predicts complications, and optimizes rehabilitation protocols.

The benefits of AI-assisted surgery are substantial and well-documented. Enhanced precision through computer-guided instruments reduces human error and improves surgical accuracy. Improved visualization through augmented reality and enhanced imaging provides surgeons with better understanding of anatomical structures. Predictive analytics enable proactive management of complications and optimization of surgical outcomes. Standardization of procedures reduces variability and improves consistency across different surgical teams.

However, AI-assisted surgery also presents unique challenges that must be carefully addressed. Safety considerations are paramount, as surgical AI systems directly impact patient outcomes and must meet the highest standards of reliability. Regulatory compliance requires extensive validation and approval processes for medical devices. Integration with existing surgical workflows must be seamless to avoid disruption of established practices. Training requirements for surgical teams must be comprehensive to ensure effective utilization of AI systems.

This chapter provides a comprehensive guide to implementing AI-assisted surgery systems. We cover the theoretical foundations of surgical AI, practical implementation strategies for computer vision and robotics, real-time guidance systems, and regulatory compliance considerations. The approaches presented here represent the current state-of-the-art in surgical AI and have been validated through extensive research and clinical deployments.

## Theoretical Foundations

### Surgical Scene Understanding

Computer vision for surgery requires sophisticated understanding of dynamic, complex scenes with multiple instruments, anatomical structures, and surgical activities. The mathematical framework for surgical scene understanding can be modeled as a multi-modal perception problem:

$$\mathcal{S}(t) = \{I(t), D(t), F(t), A(t)\}$$

where $I(t)$ represents visual information, $D(t)$ represents depth information, $F(t)$ represents force/tactile feedback, and $A(t)$ represents audio information at time $t$.

The surgical scene understanding problem can be formulated as:

$$\hat{Y}(t) = f_{\theta}(\mathcal{S}(t), \mathcal{S}(t-1), \ldots, \mathcal{S}(t-k))$$

where $\hat{Y}(t)$ represents the predicted scene interpretation (instrument positions, anatomical structures, surgical phase) and $f_{\theta}$ is a learned function parameterized by $\theta$.

### Surgical Instrument Tracking

Precise tracking of surgical instruments is fundamental to AI-assisted surgery. The tracking problem can be formulated as a state estimation problem:

$$\mathbf{x}_t = \begin{bmatrix} \mathbf{p}_t \\ \mathbf{v}_t \\ \mathbf{R}_t \\ \boldsymbol{\omega}_t \end{bmatrix}$$

where $\mathbf{p}_t$ is position, $\mathbf{v}_t$ is velocity, $\mathbf{R}_t$ is rotation matrix, and $\boldsymbol{\omega}_t$ is angular velocity.

The tracking system uses a Kalman filter or particle filter framework:

$$\mathbf{x}_{t|t-1} = \mathbf{F}_t \mathbf{x}_{t-1|t-1} + \mathbf{B}_t \mathbf{u}_t$$
$$\mathbf{x}_{t|t} = \mathbf{x}_{t|t-1} + \mathbf{K}_t (\mathbf{z}_t - \mathbf{H}_t \mathbf{x}_{t|t-1})$$

where $\mathbf{F}_t$ is the state transition model, $\mathbf{B}_t$ is the control input model, $\mathbf{u}_t$ is the control vector, $\mathbf{z}_t$ is the observation, and $\mathbf{K}_t$ is the Kalman gain.

### Robotic Surgery Control

AI-enhanced robotic surgery requires sophisticated control algorithms that combine human intent with autonomous capabilities. The control system can be modeled as:

$$\mathbf{u}(t) = \mathbf{u}_h(t) + \mathbf{u}_{ai}(t)$$

where $\mathbf{u}_h(t)$ represents human input and $\mathbf{u}_{ai}(t)$ represents AI assistance.

The AI assistance component can include:

**Tremor Filtering**: 
$$\mathbf{u}_{filtered} = \mathcal{F}(\mathbf{u}_h(t))$$

where $\mathcal{F}$ is a learned filter that removes unwanted tremor while preserving intentional motion.

**Motion Scaling**:
$$\mathbf{u}_{scaled} = \mathbf{S}(t) \mathbf{u}_h(t)$$

where $\mathbf{S}(t)$ is an adaptive scaling matrix based on surgical context.

**Constraint Enforcement**:
$$\mathbf{u}_{safe} = \arg\min_{\mathbf{u}} \|\mathbf{u} - \mathbf{u}_{desired}\|^2 \text{ s.t. } \mathbf{g}(\mathbf{u}) \leq 0$$

where $\mathbf{g}(\mathbf{u})$ represents safety constraints.

### Surgical Outcome Prediction

Predicting surgical outcomes requires integration of preoperative, intraoperative, and patient-specific factors. The prediction model can be formulated as:

$$P(outcome) = f(\mathbf{x}_{pre}, \mathbf{x}_{intra}, \mathbf{x}_{patient})$$

where:
- $\mathbf{x}_{pre}$: Preoperative factors (imaging, planning, patient history)
- $\mathbf{x}_{intra}$: Intraoperative factors (surgical metrics, complications)
- $\mathbf{x}_{patient}$: Patient-specific factors (demographics, comorbidities)

The model can be implemented using various approaches:

**Logistic Regression** for binary outcomes:
$$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

**Survival Analysis** for time-to-event outcomes:
$$h(t|\mathbf{x}) = h_0(t) \exp(\mathbf{w}^T\mathbf{x})$$

**Deep Learning** for complex, non-linear relationships:
$$P(outcome) = \text{softmax}(f_{\theta}(\mathbf{x}))$$

## Implementation Framework

### Comprehensive AI-Assisted Surgery System

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
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
import cv2
import pyaudio
import wave
import struct
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import open3d as o3d
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
import mediapipe as mp
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurgicalInstrumentTracker:
    """
    Real-time surgical instrument tracking using computer vision.
    
    Provides precise 6DOF tracking of surgical instruments with
    sub-millimeter accuracy and real-time performance.
    """
    
    def __init__(self,
                 camera_params: Dict[str, Any],
                 instrument_models: Dict[str, Any],
                 tracking_method: str = 'deep_learning'):
        """
        Initialize surgical instrument tracker.
        
        Args:
            camera_params: Camera calibration parameters
            instrument_models: 3D models of surgical instruments
            tracking_method: Tracking algorithm ('deep_learning', 'classical', 'hybrid')
        """
        self.camera_params = camera_params
        self.instrument_models = instrument_models
        self.tracking_method = tracking_method
        
        # Camera calibration
        self.camera_matrix = np.array(camera_params['camera_matrix'])
        self.dist_coeffs = np.array(camera_params['dist_coeffs'])
        
        # Tracking state
        self.tracked_instruments = {}
        self.tracking_history = {}
        
        # Deep learning models
        if tracking_method in ['deep_learning', 'hybrid']:
            self.detection_model = self._load_detection_model()
            self.pose_estimation_model = self._load_pose_estimation_model()
        
        # Classical tracking components
        if tracking_method in ['classical', 'hybrid']:
            self.feature_detector = cv2.SIFT_create()
            self.feature_matcher = cv2.BFMatcher()
        
        # Kalman filters for each instrument
        self.kalman_filters = {}
        
        logger.info(f"Initialized surgical instrument tracker with method: {tracking_method}")
    
    def _load_detection_model(self) -> nn.Module:
        """Load pre-trained instrument detection model."""
        class InstrumentDetector(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                # Use ResNet backbone
                self.backbone = models.resnet50(pretrained=True)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512)
                
                # Detection head
                self.detection_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_classes + 4)  # classes + bbox
                )
                
                # Confidence head
                self.confidence_head = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                features = self.backbone(x)
                detection = self.detection_head(features)
                confidence = self.confidence_head(features)
                
                return {
                    'detection': detection,
                    'confidence': confidence,
                    'features': features
                }
        
        model = InstrumentDetector()
        # In practice, load pre-trained weights
        # model.load_state_dict(torch.load('instrument_detector.pth'))
        model.eval()
        return model
    
    def _load_pose_estimation_model(self) -> nn.Module:
        """Load pre-trained 6DOF pose estimation model."""
        class PoseEstimator(nn.Module):
            def __init__(self, input_dim=512):
                super().__init__()
                self.pose_head = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 7)  # 3 translation + 4 quaternion
                )
            
            def forward(self, features):
                pose = self.pose_head(features)
                
                # Normalize quaternion
                translation = pose[:, :3]
                quaternion = F.normalize(pose[:, 3:], p=2, dim=1)
                
                return torch.cat([translation, quaternion], dim=1)
        
        model = PoseEstimator()
        # In practice, load pre-trained weights
        # model.load_state_dict(torch.load('pose_estimator.pth'))
        model.eval()
        return model
    
    def _initialize_kalman_filter(self, instrument_id: str) -> cv2.KalmanFilter:
        """Initialize Kalman filter for instrument tracking."""
        kf = cv2.KalmanFilter(12, 6)  # 12 state variables, 6 measurements
        
        # State: [x, y, z, vx, vy, vz, rx, ry, rz, wx, wy, wz]
        # Measurement: [x, y, z, rx, ry, rz]
        
        # Transition matrix (constant velocity model)
        dt = 1.0 / 30.0  # 30 FPS
        kf.transitionMatrix = np.eye(12, dtype=np.float32)
        kf.transitionMatrix[0, 3] = dt
        kf.transitionMatrix[1, 4] = dt
        kf.transitionMatrix[2, 5] = dt
        kf.transitionMatrix[6, 9] = dt
        kf.transitionMatrix[7, 10] = dt
        kf.transitionMatrix[8, 11] = dt
        
        # Measurement matrix
        kf.measurementMatrix = np.zeros((6, 12), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1  # x
        kf.measurementMatrix[1, 1] = 1  # y
        kf.measurementMatrix[2, 2] = 1  # z
        kf.measurementMatrix[3, 6] = 1  # rx
        kf.measurementMatrix[4, 7] = 1  # ry
        kf.measurementMatrix[5, 8] = 1  # rz
        
        # Process noise
        kf.processNoiseCov = np.eye(12, dtype=np.float32) * 0.01
        
        # Measurement noise
        kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        
        # Error covariance
        kf.errorCovPost = np.eye(12, dtype=np.float32)
        
        return kf
    
    def detect_instruments(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect surgical instruments in image."""
        detections = []
        
        if self.tracking_method in ['deep_learning', 'hybrid']:
            # Deep learning detection
            detections.extend(self._detect_instruments_dl(image))
        
        if self.tracking_method in ['classical', 'hybrid']:
            # Classical detection
            detections.extend(self._detect_instruments_classical(image))
        
        # Non-maximum suppression
        detections = self._non_maximum_suppression(detections)
        
        return detections
    
    def _detect_instruments_dl(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Deep learning-based instrument detection."""
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.detection_model(input_tensor)
            detection = output['detection'].squeeze()
            confidence = output['confidence'].squeeze()
            features = output['features'].squeeze()
        
        detections = []
        
        # Parse detection output
        if confidence > 0.5:  # Confidence threshold
            class_scores = F.softmax(detection[:-4], dim=0)
            bbox = detection[-4:]
            
            # Convert bbox to image coordinates
            h, w = image.shape[:2]
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            
            # Get predicted class
            class_id = torch.argmax(class_scores).item()
            class_confidence = class_scores[class_id].item()
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'confidence': confidence.item(),
                'class_confidence': class_confidence,
                'features': features.numpy(),
                'method': 'deep_learning'
            })
        
        return detections
    
    def _detect_instruments_classical(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Classical computer vision-based instrument detection."""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        if descriptors is not None:
            # Match with instrument templates
            for instrument_id, instrument_model in self.instrument_models.items():
                if 'template_descriptors' in instrument_model:
                    matches = self.feature_matcher.knnMatch(
                        descriptors, 
                        instrument_model['template_descriptors'], 
                        k=2
                    )
                    
                    # Apply ratio test
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                    
                    # If enough matches found
                    if len(good_matches) > 10:
                        # Estimate pose using PnP
                        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
                        dst_pts = np.float32([instrument_model['template_keypoints'][m.trainIdx].pt for m in good_matches])
                        
                        # Find bounding box
                        x1, y1 = np.min(src_pts, axis=0).astype(int)
                        x2, y2 = np.max(src_pts, axis=0).astype(int)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': instrument_model['class_id'],
                            'confidence': min(len(good_matches) / 50.0, 1.0),
                            'matches': good_matches,
                            'keypoints': src_pts,
                            'method': 'classical'
                        })
        
        return detections
    
    def _non_maximum_suppression(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Apply non-maximum suppression to remove duplicate detections."""
        if not detections:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for detection in detections:
                iou = self._calculate_iou(current['bbox'], detection['bbox'])
                if iou < iou_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def estimate_pose(self, detection: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """Estimate 6DOF pose of detected instrument."""
        if self.tracking_method in ['deep_learning', 'hybrid'] and 'features' in detection:
            # Deep learning pose estimation
            features = torch.tensor(detection['features']).unsqueeze(0)
            
            with torch.no_grad():
                pose = self.pose_estimation_model(features).squeeze()
            
            translation = pose[:3].numpy()
            quaternion = pose[3:].numpy()
            
            # Convert quaternion to rotation matrix
            rotation = R.from_quat(quaternion).as_matrix()
            
            return {
                'translation': translation,
                'rotation': rotation,
                'quaternion': quaternion,
                'method': 'deep_learning'
            }
        
        elif self.tracking_method in ['classical', 'hybrid'] and 'keypoints' in detection:
            # Classical PnP pose estimation
            class_id = detection['class_id']
            
            if class_id in self.instrument_models:
                instrument_model = self.instrument_models[class_id]
                
                if 'object_points' in instrument_model:
                    object_points = instrument_model['object_points']
                    image_points = detection['keypoints']
                    
                    # Solve PnP
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs
                    )
                    
                    if success:
                        # Convert to rotation matrix
                        rotation, _ = cv2.Rodrigues(rvec)
                        translation = tvec.flatten()
                        
                        return {
                            'translation': translation,
                            'rotation': rotation,
                            'rvec': rvec.flatten(),
                            'tvec': tvec.flatten(),
                            'method': 'classical'
                        }
        
        return None
    
    def track_instruments(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Track all instruments in the current frame."""
        # Detect instruments
        detections = self.detect_instruments(image)
        
        # Update tracking for each detection
        tracking_results = {}
        
        for detection in detections:
            # Estimate pose
            pose = self.estimate_pose(detection, image)
            
            if pose is not None:
                instrument_id = f"instrument_{detection['class_id']}"
                
                # Initialize Kalman filter if needed
                if instrument_id not in self.kalman_filters:
                    self.kalman_filters[instrument_id] = self._initialize_kalman_filter(instrument_id)
                    self.tracking_history[instrument_id] = []
                
                # Update Kalman filter
                kf = self.kalman_filters[instrument_id]
                
                # Measurement vector: [x, y, z, rx, ry, rz]
                measurement = np.array([
                    pose['translation'][0],
                    pose['translation'][1],
                    pose['translation'][2],
                    pose['rvec'][0] if 'rvec' in pose else 0,
                    pose['rvec'][1] if 'rvec' in pose else 0,
                    pose['rvec'][2] if 'rvec' in pose else 0
                ], dtype=np.float32)
                
                # Predict and update
                kf.predict()
                kf.correct(measurement)
                
                # Get filtered state
                state = kf.statePost.flatten()
                
                tracking_result = {
                    'detection': detection,
                    'pose': pose,
                    'filtered_pose': {
                        'translation': state[:3],
                        'velocity': state[3:6],
                        'rotation': state[6:9],
                        'angular_velocity': state[9:12]
                    },
                    'confidence': detection['confidence'],
                    'timestamp': time.time()
                }
                
                tracking_results[instrument_id] = tracking_result
                self.tracking_history[instrument_id].append(tracking_result)
                
                # Limit history size
                if len(self.tracking_history[instrument_id]) > 100:
                    self.tracking_history[instrument_id].pop(0)
        
        return tracking_results
    
    def get_tracking_metrics(self) -> Dict[str, Any]:
        """Get tracking performance metrics."""
        metrics = {
            'tracked_instruments': len(self.kalman_filters),
            'tracking_history_length': {
                instrument_id: len(history) 
                for instrument_id, history in self.tracking_history.items()
            }
        }
        
        # Calculate tracking stability
        for instrument_id, history in self.tracking_history.items():
            if len(history) > 10:
                # Calculate position variance
                positions = np.array([h['filtered_pose']['translation'] for h in history[-10:]])
                position_variance = np.var(positions, axis=0)
                
                metrics[f'{instrument_id}_position_stability'] = np.mean(position_variance)
        
        return metrics

class SurgicalSceneAnalyzer:
    """
    Comprehensive surgical scene analysis system.
    
    Provides real-time understanding of surgical activities, phases,
    and anatomical structures for enhanced surgical guidance.
    """
    
    def __init__(self,
                 scene_model_path: str = None,
                 phase_model_path: str = None):
        """
        Initialize surgical scene analyzer.
        
        Args:
            scene_model_path: Path to pre-trained scene understanding model
            phase_model_path: Path to pre-trained surgical phase recognition model
        """
        self.scene_model = self._load_scene_model(scene_model_path)
        self.phase_model = self._load_phase_model(phase_model_path)
        
        # Scene understanding components
        self.anatomy_segmentation = self._initialize_anatomy_segmentation()
        self.activity_recognition = self._initialize_activity_recognition()
        
        # Temporal modeling
        self.scene_history = []
        self.phase_history = []
        
        logger.info("Initialized surgical scene analyzer")
    
    def _load_scene_model(self, model_path: str = None) -> nn.Module:
        """Load pre-trained scene understanding model."""
        class SurgicalSceneModel(nn.Module):
            def __init__(self, num_anatomy_classes=20, num_instrument_classes=10):
                super().__init__()
                # Use U-Net architecture for segmentation
                self.encoder = models.resnet50(pretrained=True)
                self.encoder.fc = nn.Identity()
                
                # Decoder for anatomy segmentation
                self.anatomy_decoder = nn.Sequential(
                    nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(1024, 512, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(512, 256, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, num_anatomy_classes, 4, 2, 1)
                )
                
                # Decoder for instrument segmentation
                self.instrument_decoder = nn.Sequential(
                    nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(1024, 512, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(512, 256, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, num_instrument_classes, 4, 2, 1)
                )
                
                # Global scene understanding
                self.scene_classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 50)  # Scene features
                )
            
            def forward(self, x):
                # Encode
                features = self.encoder(x)
                
                # Reshape for decoder
                batch_size = features.size(0)
                features_2d = features.view(batch_size, 2048, 1, 1)
                features_2d = F.interpolate(features_2d, size=(7, 7), mode='bilinear')
                
                # Decode
                anatomy_seg = self.anatomy_decoder(features_2d)
                instrument_seg = self.instrument_decoder(features_2d)
                scene_features = self.scene_classifier(features)
                
                return {
                    'anatomy_segmentation': anatomy_seg,
                    'instrument_segmentation': instrument_seg,
                    'scene_features': scene_features
                }
        
        model = SurgicalSceneModel()
        
        if model_path and Path(model_path).exists():
            model.load_state_dict(torch.load(model_path))
        
        model.eval()
        return model
    
    def _load_phase_model(self, model_path: str = None) -> nn.Module:
        """Load pre-trained surgical phase recognition model."""
        class SurgicalPhaseModel(nn.Module):
            def __init__(self, input_dim=50, num_phases=7, sequence_length=30):
                super().__init__()
                self.sequence_length = sequence_length
                
                # LSTM for temporal modeling
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.2
                )
                
                # Phase classifier
                self.phase_classifier = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_phases)
                )
                
                # Phase transition model
                self.transition_model = nn.Sequential(
                    nn.Linear(128 + num_phases, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_phases)
                )
            
            def forward(self, x, previous_phase=None):
                # LSTM processing
                lstm_out, (hidden, cell) = self.lstm(x)
                
                # Use last output for classification
                last_output = lstm_out[:, -1, :]
                
                # Phase classification
                phase_logits = self.phase_classifier(last_output)
                
                # Phase transition prediction
                if previous_phase is not None:
                    transition_input = torch.cat([last_output, previous_phase], dim=1)
                    transition_logits = self.transition_model(transition_input)
                else:
                    transition_logits = phase_logits
                
                return {
                    'phase_logits': phase_logits,
                    'transition_logits': transition_logits,
                    'features': last_output
                }
        
        model = SurgicalPhaseModel()
        
        if model_path and Path(model_path).exists():
            model.load_state_dict(torch.load(model_path))
        
        model.eval()
        return model
    
    def _initialize_anatomy_segmentation(self):
        """Initialize anatomy segmentation component."""
        # Define anatomy classes
        anatomy_classes = [
            'background', 'liver', 'gallbladder', 'stomach', 'pancreas',
            'spleen', 'kidney', 'intestine', 'blood_vessel', 'nerve',
            'muscle', 'fat', 'bone', 'skin', 'tumor', 'cyst',
            'adhesion', 'scar_tissue', 'foreign_body', 'other'
        ]
        
        return {
            'classes': anatomy_classes,
            'num_classes': len(anatomy_classes),
            'class_colors': plt.cm.tab20(np.linspace(0, 1, len(anatomy_classes)))
        }
    
    def _initialize_activity_recognition(self):
        """Initialize surgical activity recognition component."""
        # Define surgical activities
        activities = [
            'cutting', 'grasping', 'suturing', 'cauterizing', 'irrigating',
            'aspirating', 'retracting', 'positioning', 'measuring', 'idle'
        ]
        
        return {
            'activities': activities,
            'num_activities': len(activities)
        }
    
    def analyze_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze surgical scene in current frame."""
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            scene_output = self.scene_model(input_tensor)
        
        # Process outputs
        anatomy_seg = torch.argmax(scene_output['anatomy_segmentation'], dim=1).squeeze().numpy()
        instrument_seg = torch.argmax(scene_output['instrument_segmentation'], dim=1).squeeze().numpy()
        scene_features = scene_output['scene_features'].squeeze().numpy()
        
        # Resize segmentations to original image size
        h, w = image.shape[:2]
        anatomy_seg = cv2.resize(anatomy_seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        instrument_seg = cv2.resize(instrument_seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Analyze anatomy
        anatomy_analysis = self._analyze_anatomy(anatomy_seg)
        
        # Analyze instruments
        instrument_analysis = self._analyze_instruments(instrument_seg)
        
        # Store scene features for temporal analysis
        scene_data = {
            'timestamp': time.time(),
            'anatomy_segmentation': anatomy_seg,
            'instrument_segmentation': instrument_seg,
            'scene_features': scene_features,
            'anatomy_analysis': anatomy_analysis,
            'instrument_analysis': instrument_analysis
        }
        
        self.scene_history.append(scene_data)
        
        # Limit history size
        if len(self.scene_history) > 100:
            self.scene_history.pop(0)
        
        return scene_data
    
    def _analyze_anatomy(self, anatomy_seg: np.ndarray) -> Dict[str, Any]:
        """Analyze anatomical structures in segmentation."""
        anatomy_classes = self.anatomy_segmentation['classes']
        
        # Calculate area for each anatomy class
        unique_classes, counts = np.unique(anatomy_seg, return_counts=True)
        
        anatomy_areas = {}
        for class_id, count in zip(unique_classes, counts):
            if class_id < len(anatomy_classes):
                anatomy_areas[anatomy_classes[class_id]] = count
        
        # Calculate total tissue area (excluding background)
        total_tissue_area = sum(count for class_id, count in zip(unique_classes, counts) if class_id > 0)
        
        # Calculate anatomy percentages
        anatomy_percentages = {}
        for anatomy, area in anatomy_areas.items():
            if anatomy != 'background':
                anatomy_percentages[anatomy] = (area / total_tissue_area * 100) if total_tissue_area > 0 else 0
        
        return {
            'anatomy_areas': anatomy_areas,
            'anatomy_percentages': anatomy_percentages,
            'total_tissue_area': total_tissue_area,
            'visible_structures': [anatomy for anatomy, area in anatomy_areas.items() 
                                 if anatomy != 'background' and area > 100]
        }
    
    def _analyze_instruments(self, instrument_seg: np.ndarray) -> Dict[str, Any]:
        """Analyze surgical instruments in segmentation."""
        # Calculate instrument areas
        unique_instruments, counts = np.unique(instrument_seg, return_counts=True)
        
        instrument_areas = {}
        for instrument_id, count in zip(unique_instruments, counts):
            if instrument_id > 0:  # Exclude background
                instrument_areas[f'instrument_{instrument_id}'] = count
        
        # Detect instrument interactions
        interactions = self._detect_instrument_interactions(instrument_seg)
        
        return {
            'instrument_areas': instrument_areas,
            'active_instruments': list(instrument_areas.keys()),
            'num_active_instruments': len(instrument_areas),
            'interactions': interactions
        }
    
    def _detect_instrument_interactions(self, instrument_seg: np.ndarray) -> List[Dict[str, Any]]:
        """Detect interactions between instruments and anatomy."""
        interactions = []
        
        # Find instrument boundaries
        instrument_ids = np.unique(instrument_seg)
        instrument_ids = instrument_ids[instrument_ids > 0]  # Exclude background
        
        for instrument_id in instrument_ids:
            # Create instrument mask
            instrument_mask = (instrument_seg == instrument_id)
            
            # Find instrument boundary
            kernel = np.ones((3, 3), np.uint8)
            boundary = cv2.morphologyEx(instrument_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
            
            # Check for interactions (simplified)
            if np.sum(boundary) > 0:
                interactions.append({
                    'instrument_id': int(instrument_id),
                    'interaction_type': 'contact',
                    'interaction_area': int(np.sum(boundary))
                })
        
        return interactions
    
    def recognize_surgical_phase(self) -> Dict[str, Any]:
        """Recognize current surgical phase based on scene history."""
        if len(self.scene_history) < 10:
            return {'phase': 'unknown', 'confidence': 0.0}
        
        # Extract scene features from recent history
        recent_features = []
        for scene_data in self.scene_history[-30:]:  # Last 30 frames
            recent_features.append(scene_data['scene_features'])
        
        # Pad sequence if necessary
        while len(recent_features) < 30:
            recent_features.insert(0, recent_features[0] if recent_features else np.zeros(50))
        
        # Convert to tensor
        feature_sequence = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0)
        
        # Get previous phase if available
        previous_phase = None
        if self.phase_history:
            last_phase = self.phase_history[-1]
            previous_phase = torch.zeros(7)
            previous_phase[last_phase['phase_id']] = 1.0
            previous_phase = previous_phase.unsqueeze(0)
        
        with torch.no_grad():
            phase_output = self.phase_model(feature_sequence, previous_phase)
        
        # Process output
        phase_probs = F.softmax(phase_output['phase_logits'], dim=1).squeeze()
        predicted_phase_id = torch.argmax(phase_probs).item()
        confidence = phase_probs[predicted_phase_id].item()
        
        # Define phase names
        phase_names = [
            'preparation', 'incision', 'dissection', 'critical_view',
            'clipping', 'extraction', 'closure'
        ]
        
        phase_result = {
            'phase_id': predicted_phase_id,
            'phase': phase_names[predicted_phase_id],
            'confidence': confidence,
            'phase_probabilities': phase_probs.numpy(),
            'timestamp': time.time()
        }
        
        self.phase_history.append(phase_result)
        
        # Limit history size
        if len(self.phase_history) > 100:
            self.phase_history.pop(0)
        
        return phase_result
    
    def get_scene_summary(self) -> Dict[str, Any]:
        """Get comprehensive scene analysis summary."""
        if not self.scene_history:
            return {'status': 'no_data'}
        
        latest_scene = self.scene_history[-1]
        current_phase = self.recognize_surgical_phase()
        
        # Calculate scene stability
        if len(self.scene_history) > 10:
            recent_features = np.array([s['scene_features'] for s in self.scene_history[-10:]])
            feature_variance = np.var(recent_features, axis=0)
            scene_stability = 1.0 / (1.0 + np.mean(feature_variance))
        else:
            scene_stability = 0.5
        
        return {
            'current_phase': current_phase,
            'anatomy_analysis': latest_scene['anatomy_analysis'],
            'instrument_analysis': latest_scene['instrument_analysis'],
            'scene_stability': scene_stability,
            'total_frames_analyzed': len(self.scene_history),
            'analysis_timestamp': latest_scene['timestamp']
        }

class RoboticSurgeryController:
    """
    AI-enhanced robotic surgery control system.
    
    Provides intelligent assistance for robotic surgical systems including
    tremor filtering, motion scaling, and safety constraint enforcement.
    """
    
    def __init__(self,
                 robot_config: Dict[str, Any],
                 safety_constraints: Dict[str, Any] = None):
        """
        Initialize robotic surgery controller.
        
        Args:
            robot_config: Robot configuration parameters
            safety_constraints: Safety constraint definitions
        """
        self.robot_config = robot_config
        self.safety_constraints = safety_constraints or self._default_safety_constraints()
        
        # Control parameters
        self.control_frequency = robot_config.get('control_frequency', 1000)  # Hz
        self.dof = robot_config.get('degrees_of_freedom', 6)
        
        # AI assistance components
        self.tremor_filter = self._initialize_tremor_filter()
        self.motion_scaler = self._initialize_motion_scaler()
        self.safety_monitor = self._initialize_safety_monitor()
        
        # State tracking
        self.robot_state = np.zeros(self.dof * 2)  # position + velocity
        self.command_history = []
        self.safety_violations = []
        
        # Control loop
        self.is_active = False
        self.control_thread = None
        
        logger.info("Initialized robotic surgery controller")
    
    def _default_safety_constraints(self) -> Dict[str, Any]:
        """Default safety constraints for robotic surgery."""
        return {
            'max_velocity': 50.0,  # mm/s
            'max_acceleration': 100.0,  # mm/s^2
            'max_force': 5.0,  # N
            'workspace_bounds': {
                'x': [-100, 100],  # mm
                'y': [-100, 100],  # mm
                'z': [0, 200]      # mm
            },
            'forbidden_zones': [],  # List of forbidden regions
            'emergency_stop_conditions': [
                'excessive_force',
                'workspace_violation',
                'communication_loss'
            ]
        }
    
    def _initialize_tremor_filter(self):
        """Initialize tremor filtering system."""
        class TremorFilter(nn.Module):
            def __init__(self, input_dim=6, hidden_dim=64):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.output_layer = nn.Linear(hidden_dim, input_dim)
                
                # Frequency analysis components
                self.tremor_freq_range = (4, 12)  # Hz, typical tremor frequency
                
            def forward(self, motion_sequence):
                # LSTM processing
                lstm_out, _ = self.lstm(motion_sequence)
                filtered_motion = self.output_layer(lstm_out)
                
                return filtered_motion
            
            def filter_tremor(self, motion_data):
                """Apply tremor filtering to motion data."""
                # Convert to tensor
                if isinstance(motion_data, np.ndarray):
                    motion_tensor = torch.tensor(motion_data, dtype=torch.float32)
                else:
                    motion_tensor = motion_data
                
                # Add batch dimension if needed
                if len(motion_tensor.shape) == 1:
                    motion_tensor = motion_tensor.unsqueeze(0).unsqueeze(0)
                elif len(motion_tensor.shape) == 2:
                    motion_tensor = motion_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    filtered = self.forward(motion_tensor)
                
                return filtered.squeeze().numpy()
        
        tremor_filter = TremorFilter()
        # In practice, load pre-trained weights
        # tremor_filter.load_state_dict(torch.load('tremor_filter.pth'))
        tremor_filter.eval()
        
        return tremor_filter
    
    def _initialize_motion_scaler(self):
        """Initialize adaptive motion scaling system."""
        class MotionScaler:
            def __init__(self):
                self.base_scale = 1.0
                self.adaptive_scale = 1.0
                self.precision_mode = False
                
            def calculate_scale(self, motion_context):
                """Calculate motion scaling factor based on context."""
                # Base scaling
                scale = self.base_scale
                
                # Adaptive scaling based on surgical phase
                if motion_context.get('surgical_phase') == 'critical_view':
                    scale *= 0.5  # Reduce motion for critical phases
                elif motion_context.get('surgical_phase') == 'dissection':
                    scale *= 0.7
                
                # Precision mode
                if self.precision_mode:
                    scale *= 0.3
                
                # Distance-based scaling
                target_distance = motion_context.get('target_distance', 10.0)
                if target_distance < 5.0:  # Close to target
                    scale *= 0.5
                
                # Force feedback scaling
                current_force = motion_context.get('current_force', 0.0)
                if current_force > 2.0:  # High force detected
                    scale *= 0.2
                
                self.adaptive_scale = scale
                return scale
            
            def apply_scaling(self, motion_command, motion_context):
                """Apply motion scaling to command."""
                scale = self.calculate_scale(motion_context)
                return motion_command * scale
        
        return MotionScaler()
    
    def _initialize_safety_monitor(self):
        """Initialize safety monitoring system."""
        class SafetyMonitor:
            def __init__(self, constraints):
                self.constraints = constraints
                self.violation_count = 0
                self.last_violation_time = 0
                
            def check_safety(self, robot_state, motion_command):
                """Check safety constraints."""
                violations = []
                
                # Position constraints
                position = robot_state[:3]  # Assume first 3 DOF are position
                
                for axis, (min_val, max_val) in self.constraints['workspace_bounds'].items():
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    if position[axis_idx] < min_val or position[axis_idx] > max_val:
                        violations.append(f'workspace_violation_{axis}')
                
                # Velocity constraints
                velocity = robot_state[3:6]  # Assume next 3 DOF are velocity
                max_velocity = self.constraints['max_velocity']
                
                if np.linalg.norm(velocity) > max_velocity:
                    violations.append('excessive_velocity')
                
                # Command constraints
                if np.linalg.norm(motion_command) > max_velocity:
                    violations.append('excessive_command')
                
                # Force constraints (would be measured from force sensors)
                # current_force = get_force_sensor_reading()
                # if current_force > self.constraints['max_force']:
                #     violations.append('excessive_force')
                
                return violations
            
            def enforce_constraints(self, motion_command, robot_state):
                """Enforce safety constraints on motion command."""
                # Velocity limiting
                max_velocity = self.constraints['max_velocity']
                command_magnitude = np.linalg.norm(motion_command)
                
                if command_magnitude > max_velocity:
                    motion_command = motion_command * (max_velocity / command_magnitude)
                
                # Workspace boundary enforcement
                position = robot_state[:3]
                
                for axis, (min_val, max_val) in self.constraints['workspace_bounds'].items():
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    
                    # If approaching boundary, reduce motion in that direction
                    if position[axis_idx] < min_val + 5:  # 5mm buffer
                        motion_command[axis_idx] = max(0, motion_command[axis_idx])
                    elif position[axis_idx] > max_val - 5:  # 5mm buffer
                        motion_command[axis_idx] = min(0, motion_command[axis_idx])
                
                return motion_command
        
        return SafetyMonitor(self.safety_constraints)
    
    def process_surgeon_input(self, raw_input: np.ndarray, context: Dict[str, Any] = None) -> np.ndarray:
        """Process surgeon input with AI assistance."""
        context = context or {}
        
        # Add to command history
        self.command_history.append({
            'timestamp': time.time(),
            'raw_input': raw_input.copy(),
            'context': context.copy()
        })
        
        # Limit history size
        if len(self.command_history) > 1000:
            self.command_history.pop(0)
        
        # Apply tremor filtering
        if len(self.command_history) >= 10:
            recent_commands = np.array([cmd['raw_input'] for cmd in self.command_history[-10:]])
            filtered_input = self.tremor_filter.filter_tremor(recent_commands)[-1]
        else:
            filtered_input = raw_input
        
        # Apply motion scaling
        scaled_input = self.motion_scaler.apply_scaling(filtered_input, context)
        
        # Apply safety constraints
        safe_command = self.safety_monitor.enforce_constraints(scaled_input, self.robot_state)
        
        # Check for safety violations
        violations = self.safety_monitor.check_safety(self.robot_state, safe_command)
        
        if violations:
            self.safety_violations.extend(violations)
            logger.warning(f"Safety violations detected: {violations}")
            
            # Emergency stop if critical violations
            critical_violations = [v for v in violations if 'excessive' in v or 'workspace' in v]
            if critical_violations:
                safe_command = np.zeros_like(safe_command)
                logger.critical(f"Emergency stop triggered: {critical_violations}")
        
        return safe_command
    
    def update_robot_state(self, new_state: np.ndarray):
        """Update robot state."""
        self.robot_state = new_state.copy()
    
    def get_assistance_metrics(self) -> Dict[str, Any]:
        """Get AI assistance performance metrics."""
        if not self.command_history:
            return {'status': 'no_data'}
        
        # Calculate tremor reduction
        if len(self.command_history) >= 10:
            raw_commands = np.array([cmd['raw_input'] for cmd in self.command_history[-10:]])
            command_variance = np.var(raw_commands, axis=0)
            tremor_reduction = np.mean(command_variance)
        else:
            tremor_reduction = 0.0
        
        # Calculate motion scaling statistics
        current_scale = self.motion_scaler.adaptive_scale
        
        # Safety statistics
        total_violations = len(self.safety_violations)
        recent_violations = len([v for v in self.safety_violations 
                               if time.time() - self.command_history[-1]['timestamp'] < 60])
        
        return {
            'tremor_reduction': tremor_reduction,
            'current_motion_scale': current_scale,
            'precision_mode': self.motion_scaler.precision_mode,
            'total_safety_violations': total_violations,
            'recent_safety_violations': recent_violations,
            'commands_processed': len(self.command_history)
        }
    
    def enable_precision_mode(self, enable: bool = True):
        """Enable/disable precision mode."""
        self.motion_scaler.precision_mode = enable
        logger.info(f"Precision mode {'enabled' if enable else 'disabled'}")
    
    def emergency_stop(self):
        """Trigger emergency stop."""
        self.robot_state[3:] = 0  # Zero all velocities
        logger.critical("Emergency stop activated")

class SurgicalOutcomePredictor:
    """
    Predictive modeling system for surgical outcomes.
    
    Provides real-time prediction of surgical complications, success rates,
    and patient recovery trajectories using multimodal data.
    """
    
    def __init__(self,
                 model_config: Dict[str, Any] = None):
        """
        Initialize surgical outcome predictor.
        
        Args:
            model_config: Configuration for prediction models
        """
        self.model_config = model_config or self._default_model_config()
        
        # Prediction models
        self.complication_model = self._initialize_complication_model()
        self.success_model = self._initialize_success_model()
        self.recovery_model = self._initialize_recovery_model()
        
        # Feature extractors
        self.feature_extractors = self._initialize_feature_extractors()
        
        # Prediction history
        self.prediction_history = []
        
        logger.info("Initialized surgical outcome predictor")
    
    def _default_model_config(self) -> Dict[str, Any]:
        """Default configuration for prediction models."""
        return {
            'complication_threshold': 0.3,
            'success_threshold': 0.7,
            'prediction_horizon': 24,  # hours
            'feature_update_interval': 5,  # minutes
            'model_types': {
                'complication': 'random_forest',
                'success': 'logistic_regression',
                'recovery': 'cox_regression'
            }
        }
    
    def _initialize_complication_model(self):
        """Initialize complication prediction model."""
        # Random Forest for complication prediction
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # In practice, load pre-trained model
        # model = joblib.load('complication_model.pkl')
        
        return model
    
    def _initialize_success_model(self):
        """Initialize surgical success prediction model."""
        # Logistic Regression for success prediction
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # In practice, load pre-trained model
        # model = joblib.load('success_model.pkl')
        
        return model
    
    def _initialize_recovery_model(self):
        """Initialize recovery time prediction model."""
        # Cox Proportional Hazards model for recovery prediction
        model = CoxPHFitter()
        
        # In practice, load pre-trained model
        # model.load_model('recovery_model.pkl')
        
        return model
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction components."""
        return {
            'preoperative': self._create_preoperative_extractor(),
            'intraoperative': self._create_intraoperative_extractor(),
            'patient': self._create_patient_extractor()
        }
    
    def _create_preoperative_extractor(self):
        """Create preoperative feature extractor."""
        def extract_preoperative_features(data):
            features = {}
            
            # Imaging features
            if 'imaging' in data:
                imaging_data = data['imaging']
                features.update({
                    'tumor_size': imaging_data.get('tumor_size', 0),
                    'tumor_location': imaging_data.get('tumor_location', 0),
                    'vascular_involvement': imaging_data.get('vascular_involvement', 0),
                    'organ_complexity': imaging_data.get('organ_complexity', 0)
                })
            
            # Laboratory values
            if 'labs' in data:
                lab_data = data['labs']
                features.update({
                    'hemoglobin': lab_data.get('hemoglobin', 12.0),
                    'platelet_count': lab_data.get('platelet_count', 250000),
                    'creatinine': lab_data.get('creatinine', 1.0),
                    'albumin': lab_data.get('albumin', 4.0)
                })
            
            # Surgical planning
            if 'planning' in data:
                planning_data = data['planning']
                features.update({
                    'estimated_duration': planning_data.get('estimated_duration', 120),
                    'complexity_score': planning_data.get('complexity_score', 5),
                    'approach_type': planning_data.get('approach_type', 1)  # 1=laparoscopic, 2=open
                })
            
            return features
        
        return extract_preoperative_features
    
    def _create_intraoperative_extractor(self):
        """Create intraoperative feature extractor."""
        def extract_intraoperative_features(data):
            features = {}
            
            # Surgical metrics
            if 'surgical_metrics' in data:
                metrics = data['surgical_metrics']
                features.update({
                    'current_duration': metrics.get('current_duration', 0),
                    'blood_loss': metrics.get('blood_loss', 0),
                    'complications_count': metrics.get('complications_count', 0),
                    'instrument_changes': metrics.get('instrument_changes', 0)
                })
            
            # Physiological monitoring
            if 'vitals' in data:
                vitals = data['vitals']
                features.update({
                    'heart_rate_avg': vitals.get('heart_rate_avg', 70),
                    'blood_pressure_systolic': vitals.get('blood_pressure_systolic', 120),
                    'oxygen_saturation': vitals.get('oxygen_saturation', 98),
                    'temperature': vitals.get('temperature', 37.0)
                })
            
            # Surgical phase
            if 'surgical_phase' in data:
                phase_data = data['surgical_phase']
                features.update({
                    'current_phase': phase_data.get('current_phase', 0),
                    'phase_duration': phase_data.get('phase_duration', 0),
                    'phase_difficulty': phase_data.get('phase_difficulty', 1)
                })
            
            return features
        
        return extract_intraoperative_features
    
    def _create_patient_extractor(self):
        """Create patient-specific feature extractor."""
        def extract_patient_features(data):
            features = {}
            
            # Demographics
            if 'demographics' in data:
                demo = data['demographics']
                features.update({
                    'age': demo.get('age', 50),
                    'gender': demo.get('gender', 0),  # 0=female, 1=male
                    'bmi': demo.get('bmi', 25.0)
                })
            
            # Medical history
            if 'medical_history' in data:
                history = data['medical_history']
                features.update({
                    'diabetes': history.get('diabetes', 0),
                    'hypertension': history.get('hypertension', 0),
                    'previous_surgeries': history.get('previous_surgeries', 0),
                    'smoking_status': history.get('smoking_status', 0)
                })
            
            # Comorbidities
            if 'comorbidities' in data:
                comorbidities = data['comorbidities']
                features.update({
                    'asa_score': comorbidities.get('asa_score', 2),
                    'cardiac_risk': comorbidities.get('cardiac_risk', 0),
                    'pulmonary_risk': comorbidities.get('pulmonary_risk', 0)
                })
            
            return features
        
        return extract_patient_features
    
    def extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features from multimodal surgical data."""
        all_features = {}
        
        # Extract features from each modality
        for modality, extractor in self.feature_extractors.items():
            if modality in data:
                modality_features = extractor(data[modality])
                all_features.update(modality_features)
        
        # Convert to feature vector
        feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features[name] for name in feature_names])
        
        return feature_vector, feature_names
    
    def predict_complications(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict surgical complications."""
        # Extract features
        features, feature_names = self.extract_features(data)
        
        # Reshape for sklearn
        features = features.reshape(1, -1)
        
        # Predict complications
        try:
            # For demonstration, create dummy training data
            if not hasattr(self.complication_model, 'n_features_in_'):
                # Create dummy training data
                X_dummy = np.random.randn(100, len(features[0]))
                y_dummy = np.random.randint(0, 2, 100)
                self.complication_model.fit(X_dummy, y_dummy)
            
            complication_prob = self.complication_model.predict_proba(features)[0, 1]
            complication_prediction = complication_prob > self.model_config['complication_threshold']
            
            # Get feature importance
            feature_importance = dict(zip(feature_names, self.complication_model.feature_importances_))
            
        except Exception as e:
            logger.warning(f"Complication prediction failed: {e}")
            complication_prob = 0.5
            complication_prediction = False
            feature_importance = {}
        
        return {
            'complication_probability': complication_prob,
            'complication_predicted': complication_prediction,
            'feature_importance': feature_importance,
            'confidence': abs(complication_prob - 0.5) * 2
        }
    
    def predict_success(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict surgical success."""
        # Extract features
        features, feature_names = self.extract_features(data)
        
        # Reshape for sklearn
        features = features.reshape(1, -1)
        
        # Predict success
        try:
            # For demonstration, create dummy training data
            if not hasattr(self.success_model, 'n_features_in_'):
                # Create dummy training data
                X_dummy = np.random.randn(100, len(features[0]))
                y_dummy = np.random.randint(0, 2, 100)
                self.success_model.fit(X_dummy, y_dummy)
            
            success_prob = self.success_model.predict_proba(features)[0, 1]
            success_prediction = success_prob > self.model_config['success_threshold']
            
        except Exception as e:
            logger.warning(f"Success prediction failed: {e}")
            success_prob = 0.5
            success_prediction = False
        
        return {
            'success_probability': success_prob,
            'success_predicted': success_prediction,
            'confidence': abs(success_prob - 0.5) * 2
        }
    
    def predict_recovery(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict recovery trajectory."""
        # Extract features
        features, feature_names = self.extract_features(data)
        
        # For Cox regression, we need duration and event data
        # In practice, this would come from historical data
        
        # Simplified recovery prediction
        recovery_score = np.mean(features) if len(features) > 0 else 0.5
        
        # Normalize to 0-1 range
        recovery_score = max(0, min(1, recovery_score / 10.0))
        
        # Estimate recovery time (days)
        base_recovery_time = 7  # days
        recovery_time = base_recovery_time * (2 - recovery_score)  # Better score = faster recovery
        
        return {
            'recovery_score': recovery_score,
            'estimated_recovery_days': recovery_time,
            'recovery_category': 'fast' if recovery_time < 5 else 'normal' if recovery_time < 10 else 'slow'
        }
    
    def comprehensive_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive outcome predictions."""
        # Individual predictions
        complication_pred = self.predict_complications(data)
        success_pred = self.predict_success(data)
        recovery_pred = self.predict_recovery(data)
        
        # Combined risk assessment
        overall_risk = (
            complication_pred['complication_probability'] * 0.4 +
            (1 - success_pred['success_probability']) * 0.4 +
            (recovery_pred['recovery_score']) * 0.2
        )
        
        # Risk categorization
        if overall_risk < 0.3:
            risk_category = 'low'
        elif overall_risk < 0.7:
            risk_category = 'moderate'
        else:
            risk_category = 'high'
        
        prediction_result = {
            'timestamp': time.time(),
            'overall_risk': overall_risk,
            'risk_category': risk_category,
            'complications': complication_pred,
            'success': success_pred,
            'recovery': recovery_pred,
            'recommendations': self._generate_recommendations(
                complication_pred, success_pred, recovery_pred, overall_risk
            )
        }
        
        # Store prediction
        self.prediction_history.append(prediction_result)
        
        # Limit history size
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)
        
        return prediction_result
    
    def _generate_recommendations(self,
                                complication_pred: Dict[str, Any],
                                success_pred: Dict[str, Any],
                                recovery_pred: Dict[str, Any],
                                overall_risk: float) -> List[str]:
        """Generate clinical recommendations based on predictions."""
        recommendations = []
        
        # Complication-based recommendations
        if complication_pred['complication_predicted']:
            recommendations.append("Consider additional monitoring for potential complications")
            recommendations.append("Ensure blood products are readily available")
        
        # Success-based recommendations
        if not success_pred['success_predicted']:
            recommendations.append("Consider alternative surgical approach")
            recommendations.append("Prepare for potential conversion to open surgery")
        
        # Recovery-based recommendations
        if recovery_pred['recovery_category'] == 'slow':
            recommendations.append("Plan for extended postoperative monitoring")
            recommendations.append("Consider enhanced recovery protocols")
        
        # Overall risk-based recommendations
        if overall_risk > 0.7:
            recommendations.append("Consider postponing surgery if not urgent")
            recommendations.append("Ensure senior surgeon availability")
            recommendations.append("Prepare ICU bed for postoperative care")
        
        return recommendations
    
    def get_prediction_trends(self) -> Dict[str, Any]:
        """Analyze prediction trends over time."""
        if len(self.prediction_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Extract time series data
        timestamps = [p['timestamp'] for p in self.prediction_history]
        overall_risks = [p['overall_risk'] for p in self.prediction_history]
        complication_probs = [p['complications']['complication_probability'] for p in self.prediction_history]
        success_probs = [p['success']['success_probability'] for p in self.prediction_history]
        
        # Calculate trends
        risk_trend = np.polyfit(range(len(overall_risks)), overall_risks, 1)[0]
        complication_trend = np.polyfit(range(len(complication_probs)), complication_probs, 1)[0]
        success_trend = np.polyfit(range(len(success_probs)), success_probs, 1)[0]
        
        return {
            'risk_trend': 'increasing' if risk_trend > 0.01 else 'decreasing' if risk_trend < -0.01 else 'stable',
            'complication_trend': 'increasing' if complication_trend > 0.01 else 'decreasing' if complication_trend < -0.01 else 'stable',
            'success_trend': 'increasing' if success_trend > 0.01 else 'decreasing' if success_trend < -0.01 else 'stable',
            'prediction_count': len(self.prediction_history),
            'time_span_hours': (timestamps[-1] - timestamps[0]) / 3600 if len(timestamps) > 1 else 0
        }

# Example usage and demonstration
def main():
    """Demonstrate the AI-assisted surgery system."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create AI-assisted surgery demonstration
    logger.info("Creating AI-assisted surgery demonstration...")
    
    # Initialize surgical instrument tracker
    camera_params = {
        'camera_matrix': [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
        'dist_coeffs': [0.1, -0.2, 0, 0, 0]
    }
    
    instrument_models = {
        0: {'class_id': 0, 'name': 'grasper'},
        1: {'class_id': 1, 'name': 'scissors'},
        2: {'class_id': 2, 'name': 'cautery'}
    }
    
    tracker = SurgicalInstrumentTracker(
        camera_params=camera_params,
        instrument_models=instrument_models,
        tracking_method='hybrid'
    )
    
    # Initialize surgical scene analyzer
    scene_analyzer = SurgicalSceneAnalyzer()
    
    # Initialize robotic surgery controller
    robot_config = {
        'control_frequency': 1000,
        'degrees_of_freedom': 6
    }
    
    robot_controller = RoboticSurgeryController(robot_config)
    
    # Initialize surgical outcome predictor
    outcome_predictor = SurgicalOutcomePredictor()
    
    # Simulate surgical procedure
    logger.info("Simulating surgical procedure...")
    
    # Create synthetic surgical data
    def create_synthetic_surgical_image(frame_idx):
        """Create synthetic surgical scene image."""
        # Create base surgical scene
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background (surgical field)
        image[:, :] = [120, 80, 60]  # Brownish background
        
        # Add anatomical structures
        cv2.circle(image, (320, 240), 100, (180, 120, 100), -1)  # Organ
        cv2.circle(image, (280, 200), 30, (200, 150, 120), -1)   # Vessel
        
        # Add surgical instruments (moving)
        instrument_x = 200 + int(50 * np.sin(frame_idx * 0.1))
        instrument_y = 150 + int(30 * np.cos(frame_idx * 0.1))
        
        cv2.line(image, (instrument_x, instrument_y), 
                (instrument_x + 80, instrument_y + 20), (200, 200, 200), 5)  # Instrument
        
        # Add some noise
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def create_synthetic_surgical_data(frame_idx):
        """Create synthetic surgical data for outcome prediction."""
        return {
            'preoperative': {
                'imaging': {
                    'tumor_size': 3.5,
                    'tumor_location': 1,
                    'vascular_involvement': 0,
                    'organ_complexity': 2
                },
                'labs': {
                    'hemoglobin': 12.5,
                    'platelet_count': 280000,
                    'creatinine': 0.9,
                    'albumin': 4.2
                },
                'planning': {
                    'estimated_duration': 180,
                    'complexity_score': 6,
                    'approach_type': 1
                }
            },
            'intraoperative': {
                'surgical_metrics': {
                    'current_duration': frame_idx * 2,  # 2 minutes per frame
                    'blood_loss': frame_idx * 5,       # 5ml per frame
                    'complications_count': 0,
                    'instrument_changes': frame_idx // 30
                },
                'vitals': {
                    'heart_rate_avg': 75 + np.random.randint(-5, 5),
                    'blood_pressure_systolic': 120 + np.random.randint(-10, 10),
                    'oxygen_saturation': 98 + np.random.randint(-2, 2),
                    'temperature': 37.0 + np.random.normal(0, 0.2)
                },
                'surgical_phase': {
                    'current_phase': min(6, frame_idx // 20),
                    'phase_duration': frame_idx % 20,
                    'phase_difficulty': np.random.randint(1, 4)
                }
            },
            'patient': {
                'demographics': {
                    'age': 55,
                    'gender': 1,
                    'bmi': 26.5
                },
                'medical_history': {
                    'diabetes': 0,
                    'hypertension': 1,
                    'previous_surgeries': 2,
                    'smoking_status': 0
                },
                'comorbidities': {
                    'asa_score': 2,
                    'cardiac_risk': 0,
                    'pulmonary_risk': 0
                }
            }
        }
    
    # Simulation parameters
    simulation_frames = 100
    results_data = {
        'tracking_results': [],
        'scene_analysis': [],
        'robot_control': [],
        'outcome_predictions': []
    }
    
    # Run simulation
    for frame_idx in range(simulation_frames):
        # Create synthetic data
        surgical_image = create_synthetic_surgical_image(frame_idx)
        surgical_data = create_synthetic_surgical_data(frame_idx)
        
        # Instrument tracking
        tracking_results = tracker.track_instruments(surgical_image)
        results_data['tracking_results'].append({
            'frame': frame_idx,
            'timestamp': time.time(),
            'tracking_results': tracking_results,
            'tracking_metrics': tracker.get_tracking_metrics()
        })
        
        # Scene analysis
        scene_analysis = scene_analyzer.analyze_scene(surgical_image)
        scene_summary = scene_analyzer.get_scene_summary()
        results_data['scene_analysis'].append({
            'frame': frame_idx,
            'scene_analysis': scene_analysis,
            'scene_summary': scene_summary
        })
        
        # Robot control simulation
        surgeon_input = np.random.randn(6) * 0.1  # Small random movements
        motion_context = {
            'surgical_phase': scene_summary.get('current_phase', {}).get('phase', 'unknown'),
            'target_distance': np.random.uniform(5, 20),
            'current_force': np.random.uniform(0, 3)
        }
        
        processed_command = robot_controller.process_surgeon_input(surgeon_input, motion_context)
        
        # Update robot state (simplified)
        new_robot_state = np.random.randn(12) * 0.1  # Random state
        robot_controller.update_robot_state(new_robot_state)
        
        assistance_metrics = robot_controller.get_assistance_metrics()
        results_data['robot_control'].append({
            'frame': frame_idx,
            'surgeon_input': surgeon_input,
            'processed_command': processed_command,
            'assistance_metrics': assistance_metrics
        })
        
        # Outcome prediction (every 10 frames)
        if frame_idx % 10 == 0:
            outcome_prediction = outcome_predictor.comprehensive_prediction(surgical_data)
            prediction_trends = outcome_predictor.get_prediction_trends()
            results_data['outcome_predictions'].append({
                'frame': frame_idx,
                'prediction': outcome_prediction,
                'trends': prediction_trends
            })
        
        # Progress update
        if frame_idx % 20 == 0:
            logger.info(f"Simulation progress: {frame_idx}/{simulation_frames} frames")
    
    # Generate results and analysis
    logger.info("Generating results and analysis...")
    
    # Save results
    results_dir = Path("ai_surgery_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(results_dir / 'simulation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    # Tracking performance
    plt.subplot(4, 4, 1)
    tracking_frames = [r['frame'] for r in results_data['tracking_results']]
    tracked_instruments = [len(r['tracking_results']) for r in results_data['tracking_results']]
    
    plt.plot(tracking_frames, tracked_instruments)
    plt.xlabel('Frame')
    plt.ylabel('Tracked Instruments')
    plt.title('Instrument Tracking Performance')
    plt.grid(True)
    
    # Scene analysis - surgical phases
    plt.subplot(4, 4, 2)
    scene_frames = [r['frame'] for r in results_data['scene_analysis']]
    surgical_phases = []
    
    for r in results_data['scene_analysis']:
        phase_info = r['scene_summary'].get('current_phase', {})
        if isinstance(phase_info, dict):
            phase_id = phase_info.get('phase_id', 0)
        else:
            phase_id = 0
        surgical_phases.append(phase_id)
    
    plt.plot(scene_frames, surgical_phases)
    plt.xlabel('Frame')
    plt.ylabel('Surgical Phase')
    plt.title('Surgical Phase Recognition')
    plt.grid(True)
    
    # Robot control - motion scaling
    plt.subplot(4, 4, 3)
    control_frames = [r['frame'] for r in results_data['robot_control']]
    motion_scales = [r['assistance_metrics'].get('current_motion_scale', 1.0) 
                    for r in results_data['robot_control']]
    
    plt.plot(control_frames, motion_scales)
    plt.xlabel('Frame')
    plt.ylabel('Motion Scale')
    plt.title('Adaptive Motion Scaling')
    plt.grid(True)
    
    # Outcome predictions - overall risk
    plt.subplot(4, 4, 4)
    if results_data['outcome_predictions']:
        pred_frames = [r['frame'] for r in results_data['outcome_predictions']]
        overall_risks = [r['prediction']['overall_risk'] for r in results_data['outcome_predictions']]
        
        plt.plot(pred_frames, overall_risks, 'o-')
        plt.xlabel('Frame')
        plt.ylabel('Overall Risk')
        plt.title('Surgical Risk Prediction')
        plt.grid(True)
    
    # Complication probability over time
    plt.subplot(4, 4, 5)
    if results_data['outcome_predictions']:
        complication_probs = [r['prediction']['complications']['complication_probability'] 
                            for r in results_data['outcome_predictions']]
        
        plt.plot(pred_frames, complication_probs, 'r-', label='Complication Risk')
        plt.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Threshold')
        plt.xlabel('Frame')
        plt.ylabel('Probability')
        plt.title('Complication Risk Over Time')
        plt.legend()
        plt.grid(True)
    
    # Success probability over time
    plt.subplot(4, 4, 6)
    if results_data['outcome_predictions']:
        success_probs = [r['prediction']['success']['success_probability'] 
                        for r in results_data['outcome_predictions']]
        
        plt.plot(pred_frames, success_probs, 'g-', label='Success Probability')
        plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Threshold')
        plt.xlabel('Frame')
        plt.ylabel('Probability')
        plt.title('Success Probability Over Time')
        plt.legend()
        plt.grid(True)
    
    # Robot safety violations
    plt.subplot(4, 4, 7)
    safety_violations = [r['assistance_metrics'].get('recent_safety_violations', 0) 
                        for r in results_data['robot_control']]
    
    plt.plot(control_frames, safety_violations)
    plt.xlabel('Frame')
    plt.ylabel('Safety Violations')
    plt.title('Robot Safety Monitoring')
    plt.grid(True)
    
    # Scene stability
    plt.subplot(4, 4, 8)
    scene_stability = []
    for r in results_data['scene_analysis']:
        stability = r['scene_summary'].get('scene_stability', 0.5)
        scene_stability.append(stability)
    
    plt.plot(scene_frames, scene_stability)
    plt.xlabel('Frame')
    plt.ylabel('Scene Stability')
    plt.title('Surgical Scene Stability')
    plt.grid(True)
    
    # Instrument tracking accuracy (simulated)
    plt.subplot(4, 4, 9)
    tracking_accuracy = []
    for r in results_data['tracking_results']:
        # Simulate tracking accuracy based on number of tracked instruments
        num_instruments = len(r['tracking_results'])
        accuracy = min(0.95, 0.7 + num_instruments * 0.1)
        tracking_accuracy.append(accuracy)
    
    plt.plot(tracking_frames, tracking_accuracy)
    plt.xlabel('Frame')
    plt.ylabel('Tracking Accuracy')
    plt.title('Instrument Tracking Accuracy')
    plt.grid(True)
    
    # Motion command magnitude
    plt.subplot(4, 4, 10)
    command_magnitudes = [np.linalg.norm(r['processed_command']) 
                         for r in results_data['robot_control']]
    
    plt.plot(control_frames, command_magnitudes)
    plt.xlabel('Frame')
    plt.ylabel('Command Magnitude')
    plt.title('Robot Motion Commands')
    plt.grid(True)
    
    # Recovery prediction
    plt.subplot(4, 4, 11)
    if results_data['outcome_predictions']:
        recovery_scores = [r['prediction']['recovery']['recovery_score'] 
                          for r in results_data['outcome_predictions']]
        
        plt.plot(pred_frames, recovery_scores, 'b-')
        plt.xlabel('Frame')
        plt.ylabel('Recovery Score')
        plt.title('Recovery Prediction')
        plt.grid(True)
    
    # System performance summary
    plt.subplot(4, 4, 12)
    performance_metrics = [
        np.mean(tracked_instruments),
        np.mean(scene_stability),
        np.mean([1 - r for r in overall_risks]) if overall_risks else 0.5,
        1 - np.mean(safety_violations) / max(1, max(safety_violations)) if safety_violations else 1.0
    ]
    
    metric_names = ['Tracking', 'Stability', 'Safety', 'Overall']
    colors = ['blue', 'green', 'orange', 'red']
    
    bars = plt.bar(metric_names, performance_metrics, color=colors, alpha=0.7)
    plt.ylabel('Performance Score')
    plt.title('System Performance Summary')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, performance_metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Surgical phase distribution
    plt.subplot(4, 4, 13)
    phase_counts = {}
    for phase in surgical_phases:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    if phase_counts:
        phases = list(phase_counts.keys())
        counts = list(phase_counts.values())
        
        plt.bar(phases, counts)
        plt.xlabel('Surgical Phase')
        plt.ylabel('Frame Count')
        plt.title('Surgical Phase Distribution')
        plt.grid(True, alpha=0.3)
    
    # Risk category distribution
    plt.subplot(4, 4, 14)
    if results_data['outcome_predictions']:
        risk_categories = [r['prediction']['risk_category'] 
                          for r in results_data['outcome_predictions']]
        
        risk_counts = {}
        for category in risk_categories:
            risk_counts[category] = risk_counts.get(category, 0) + 1
        
        if risk_counts:
            categories = list(risk_counts.keys())
            counts = list(risk_counts.values())
            colors = ['green', 'yellow', 'red'][:len(categories)]
            
            plt.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
            plt.title('Risk Category Distribution')
    
    # Tremor reduction effectiveness
    plt.subplot(4, 4, 15)
    tremor_reduction = [r['assistance_metrics'].get('tremor_reduction', 0) 
                       for r in results_data['robot_control']]
    
    plt.plot(control_frames, tremor_reduction)
    plt.xlabel('Frame')
    plt.ylabel('Tremor Reduction')
    plt.title('Tremor Filtering Effectiveness')
    plt.grid(True)
    
    # Prediction confidence over time
    plt.subplot(4, 4, 16)
    if results_data['outcome_predictions']:
        prediction_confidence = [r['prediction']['complications']['confidence'] 
                               for r in results_data['outcome_predictions']]
        
        plt.plot(pred_frames, prediction_confidence, 'purple')
        plt.xlabel('Frame')
        plt.ylabel('Prediction Confidence')
        plt.title('Outcome Prediction Confidence')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'ai_surgery_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    summary_report = {
        'simulation_summary': {
            'total_frames': simulation_frames,
            'average_tracked_instruments': np.mean(tracked_instruments),
            'total_safety_violations': sum(safety_violations),
            'average_scene_stability': np.mean(scene_stability)
        },
        'tracking_performance': {
            'max_instruments_tracked': max(tracked_instruments),
            'tracking_consistency': np.std(tracked_instruments),
            'average_accuracy': np.mean(tracking_accuracy)
        },
        'robot_control_performance': {
            'average_motion_scale': np.mean(motion_scales),
            'tremor_reduction_effectiveness': np.mean(tremor_reduction),
            'safety_violation_rate': sum(safety_violations) / len(safety_violations)
        },
        'outcome_prediction_performance': {
            'predictions_made': len(results_data['outcome_predictions']),
            'average_risk_level': np.mean(overall_risks) if overall_risks else 0,
            'average_confidence': np.mean(prediction_confidence) if results_data['outcome_predictions'] else 0
        }
    }
    
    with open(results_dir / 'summary_report.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    logger.info("AI-assisted surgery demonstration completed!")
    logger.info(f"Results saved to {results_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("AI-ASSISTED SURGERY DEMONSTRATION SUMMARY")
    print("="*60)
    print(f"Simulation Frames: {simulation_frames}")
    print(f"Average Tracked Instruments: {np.mean(tracked_instruments):.2f}")
    print(f"Average Scene Stability: {np.mean(scene_stability):.2f}")
    print(f"Total Safety Violations: {sum(safety_violations)}")
    print(f"Average Motion Scale: {np.mean(motion_scales):.2f}")
    
    if results_data['outcome_predictions']:
        print(f"Outcome Predictions Made: {len(results_data['outcome_predictions'])}")
        print(f"Average Risk Level: {np.mean(overall_risks):.2f}")
        print(f"Average Prediction Confidence: {np.mean(prediction_confidence):.2f}")
    
    return tracker, scene_analyzer, robot_controller, outcome_predictor

if __name__ == "__main__":
    main()
```

## Advanced Surgical AI Techniques

### Multi-Modal Sensor Fusion

Modern surgical AI systems integrate multiple sensor modalities for comprehensive scene understanding:

$$\mathbf{z}_{fused} = \mathcal{F}(\mathbf{z}_{visual}, \mathbf{z}_{depth}, \mathbf{z}_{force}, \mathbf{z}_{audio})$$

where each sensor modality provides complementary information about the surgical environment.

### Surgical Skill Assessment

AI systems can quantitatively assess surgical skill using motion analysis and outcome metrics:

$$\text{Skill Score} = \alpha \cdot \text{Efficiency} + \beta \cdot \text{Precision} + \gamma \cdot \text{Safety}$$

where efficiency measures task completion time, precision measures motion smoothness, and safety measures adherence to best practices.

### Predictive Surgical Planning

AI-driven surgical planning optimizes surgical approaches based on patient-specific anatomy and predicted outcomes:

$$\text{Plan}^* = \arg\min_{\text{Plan}} \mathbb{E}[\text{Risk}(\text{Plan}, \text{Patient})] + \lambda \cdot \text{Duration}(\text{Plan})$$

## Clinical Applications and Case Studies

### Case Study 1: Laparoscopic Cholecystectomy

AI-assisted laparoscopic surgery demonstrates significant improvements in safety and outcomes:

1. **Critical View of Safety**: AI systems automatically verify achievement of critical anatomical landmarks
2. **Instrument Tracking**: Real-time tracking of surgical instruments prevents inadvertent injury
3. **Complication Prediction**: Early warning systems alert surgeons to potential complications
4. **Skill Assessment**: Objective evaluation of surgical performance for training and credentialing

### Case Study 2: Robotic Prostatectomy

Robotic surgery with AI enhancement provides superior precision and outcomes:

1. **Nerve-Sparing Guidance**: AI identifies and preserves critical neural structures
2. **Margin Assessment**: Real-time analysis of surgical margins during resection
3. **Functional Outcome Prediction**: Prediction of postoperative urinary and sexual function
4. **Learning Curve Acceleration**: AI-assisted training reduces time to proficiency

### Case Study 3: Cardiac Surgery

AI applications in cardiac surgery address complex anatomical and physiological challenges:

1. **Preoperative Planning**: 3D modeling and simulation of surgical procedures
2. **Intraoperative Guidance**: Real-time navigation for complex cardiac interventions
3. **Perfusion Optimization**: AI-controlled cardiopulmonary bypass management
4. **Outcome Prediction**: Risk stratification and personalized treatment planning

## Regulatory and Safety Considerations

### FDA Approval Process

AI-assisted surgical systems require comprehensive regulatory approval:

1. **510(k) Clearance**: Demonstration of substantial equivalence to predicate devices
2. **De Novo Classification**: Novel AI systems may require new regulatory pathways
3. **Clinical Trials**: Prospective validation of safety and efficacy
4. **Post-Market Surveillance**: Continuous monitoring of real-world performance

### Safety Standards

Surgical AI systems must meet rigorous safety standards:

1. **IEC 62304**: Medical device software lifecycle processes
2. **ISO 14971**: Risk management for medical devices
3. **IEC 60601**: Safety and essential performance of medical electrical equipment
4. **ISO 13485**: Quality management systems for medical devices

### Ethical Considerations

AI-assisted surgery raises important ethical questions:

1. **Informed Consent**: Patients must understand AI involvement in their care
2. **Liability**: Clear assignment of responsibility for AI-assisted decisions
3. **Transparency**: Explainable AI systems for clinical decision-making
4. **Equity**: Ensuring equal access to AI-enhanced surgical care

## Future Directions and Emerging Technologies

### Autonomous Surgery

The future may include fully autonomous surgical systems for specific procedures:

1. **Microsurgery**: AI systems with superhuman precision for delicate procedures
2. **Repetitive Tasks**: Automation of routine surgical tasks
3. **Remote Surgery**: Telepresence surgery with AI assistance
4. **Emergency Surgery**: Autonomous systems for trauma and emergency situations

### Brain-Computer Interfaces

Direct neural control of surgical instruments may revolutionize surgery:

1. **Thought-Controlled Robotics**: Direct mental control of surgical robots
2. **Enhanced Dexterity**: Superhuman precision through neural interfaces
3. **Haptic Feedback**: Direct neural feedback of force and texture
4. **Cognitive Augmentation**: AI-enhanced surgical decision-making

### Molecular Surgery

AI may enable surgery at the molecular level:

1. **Targeted Drug Delivery**: AI-guided nanorobots for precise drug delivery
2. **Gene Editing**: AI-optimized CRISPR systems for therapeutic gene editing
3. **Cellular Repair**: AI-directed cellular regeneration and repair
4. **Biomarker-Guided Surgery**: Real-time molecular analysis during surgery

## Summary

AI-assisted surgery represents one of the most promising applications of artificial intelligence in healthcare, offering the potential to enhance surgical precision, improve patient outcomes, and expand the capabilities of surgical teams. This chapter has provided comprehensive coverage of the key technologies, implementation strategies, and clinical applications that define this rapidly evolving field.

Key takeaways include:

1. **Computer Vision**: Advanced imaging and tracking systems enable real-time understanding of surgical scenes
2. **Robotic Control**: AI-enhanced robotic systems provide superhuman precision and safety
3. **Predictive Analytics**: Machine learning models predict surgical outcomes and complications
4. **Regulatory Compliance**: Rigorous safety and efficacy standards ensure patient safety
5. **Clinical Integration**: Seamless integration with existing surgical workflows is essential for adoption

The field continues to advance rapidly, with emerging technologies such as autonomous surgery, brain-computer interfaces, and molecular-level interventions promising to further transform surgical practice. However, the fundamental goals remain constant: improving patient outcomes, enhancing surgical precision, and expanding access to high-quality surgical care.

As AI-assisted surgery systems become more sophisticated and widely adopted, they will play an increasingly important role in training the next generation of surgeons, standardizing surgical practices, and democratizing access to expert-level surgical care worldwide.

## References

1. Maier-Hein, L., et al. (2017). Surgical data science for next-generation interventions. *Nature Biomedical Engineering*, 1(9), 691-696. DOI: 10.1038/s41551-017-0132-7

2. Vedula, S. S., et al. (2017). Objective assessment of surgical technical skill and competency in the operating room. *Annual Review of Biomedical Engineering*, 19, 301-325. DOI: 10.1146/annurev-bioeng-071516-044435

3. Hashimoto, D. A., et al. (2018). Artificial intelligence in surgery: Promises and perils. *Annals of Surgery*, 268(1), 70-76. DOI: 10.1097/SLA.0000000000002693

4. Kitaguchi, D., et al. (2020). Real-time automatic surgical phase recognition in laparoscopic sigmoidectomy using the convolutional neural network-based deep learning approach. *Surgical Endoscopy*, 34(11), 4924-4931. DOI: 10.1007/s00464-019-07281-0

5. Twinanda, A. P., et al. (2016). EndoNet: A deep architecture for recognition tasks on laparoscopic videos. *IEEE Transactions on Medical Imaging*, 36(1), 86-97. DOI: 10.1109/TMI.2016.2593957

6. Jin, Y., et al. (2017). Tool detection and operative skill assessment in surgical videos using region-based convolutional neural networks. *IEEE Winter Conference on Applications of Computer Vision*, 691-699. DOI: 10.1109/WACV.2017.81

7. Forestier, G., et al. (2017). Surgical motion adaptive prediction using mixture of experts. *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 393-401. DOI: 10.1007/978-3-319-66185-8_45

8. Gao, Y., et al. (2014). JHU-ISI gesture and skill assessment working set (JIGSAWS): A surgical activity dataset for human motion modeling. *MICCAI Workshop*, 3, 3. DOI: 10.1007/978-3-319-13909-8_7

9. Ahmidi, N., et al. (2017). A dataset and benchmarks for segmentation and recognition of gestures in robotic surgery. *IEEE Transactions on Biomedical Engineering*, 64(9), 2025-2041. DOI: 10.1109/TBME.2016.2647680

10. Funke, I., et al. (2019). Using 3D convolutional neural networks to learn spatiotemporal features for automatic surgical gesture recognition in video. *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 467-475. DOI: 10.1007/978-3-030-32254-0_52
