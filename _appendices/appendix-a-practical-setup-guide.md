# Appendix A: Practical Setup Guide for Healthcare AI Development
## Complete Environment Configuration and Best Practices

### Overview

This comprehensive setup guide provides step-by-step instructions for establishing a complete healthcare AI development environment. Whether you're setting up a local development machine, cloud infrastructure, or hybrid deployment, this guide covers all essential components with security, compliance, and best practices considerations.

### Table of Contents

1. [Local Development Environment Setup](#local-development-environment-setup)
2. [Cloud Platform Configuration](#cloud-platform-configuration)
3. [Healthcare-Specific Security and Compliance](#healthcare-specific-security-and-compliance)
4. [Development Tools and IDEs](#development-tools-and-ides)
5. [Package Management and Dependencies](#package-management-and-dependencies)
6. [Data Management and Storage](#data-management-and-storage)
7. [Model Development and Training Infrastructure](#model-development-and-training-infrastructure)
8. [Deployment and Production Setup](#deployment-and-production-setup)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)
10. [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## Local Development Environment Setup

### System Requirements

**Minimum Requirements:**
- CPU: 8-core processor (Intel i7 or AMD Ryzen 7 equivalent)
- RAM: 32GB (64GB recommended for large models)
- Storage: 1TB NVMe SSD (2TB recommended)
- GPU: NVIDIA RTX 3070 or better (RTX 4090 or A100 for production)
- OS: Ubuntu 22.04 LTS, macOS 12+, or Windows 11 with WSL2

**Recommended Specifications:**
- CPU: 16-core processor (Intel i9 or AMD Ryzen 9)
- RAM: 64GB or higher
- Storage: 2TB NVMe SSD + additional storage for datasets
- GPU: NVIDIA RTX 4090, A100, or H100
- Network: Gigabit ethernet for data transfer

### Operating System Setup

#### Ubuntu 22.04 LTS (Recommended)

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential curl wget git vim htop tree unzip

# Install Python development dependencies
sudo apt install -y python3-dev python3-pip python3-venv python3-wheel

# Install system libraries for scientific computing
sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev gfortran
sudo apt install -y libhdf5-dev libnetcdf-dev

# Install multimedia and image processing libraries
sudo apt install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0
sudo apt install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# Install database clients
sudo apt install -y postgresql-client mysql-client sqlite3

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

#### macOS Setup

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install essential tools
brew install git wget curl htop tree unzip

# Install Python
brew install python@3.11

# Install development tools
brew install cmake pkg-config

# Install database clients
brew install postgresql mysql-client sqlite

# Install Docker Desktop
brew install --cask docker

# Install additional tools
brew install node npm
```

#### Windows with WSL2

```powershell
# Enable WSL2 (run as Administrator)
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart computer, then set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Install Windows Terminal
winget install Microsoft.WindowsTerminal

# Install Docker Desktop for Windows
winget install Docker.DockerDesktop
```

### NVIDIA GPU Setup

```bash
# Check GPU availability
nvidia-smi

# Install NVIDIA drivers (if not already installed)
sudo apt install -y nvidia-driver-525

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install cuDNN
# Download cuDNN from NVIDIA Developer website (requires registration)
# Extract and copy files to CUDA installation directory
```

### Python Environment Management

#### Using Conda (Recommended for Data Science)

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
conda init bash
source ~/.bashrc

# Create healthcare AI environment
conda create -n healthcare-ai python=3.11 -y
conda activate healthcare-ai

# Install essential packages
conda install -y numpy pandas scikit-learn matplotlib seaborn jupyter
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y tensorflow-gpu -c conda-forge

# Install additional scientific packages
conda install -y scipy statsmodels networkx
conda install -y plotly bokeh altair
conda install -y dask distributed
```

#### Using pyenv and virtualenv

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.11
pyenv install 3.11.5
pyenv global 3.11.5

# Create virtual environment
python -m venv healthcare-ai-env
source healthcare-ai-env/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools
```

### Essential Python Packages

Create a comprehensive `requirements.txt` file:

```txt
# Core scientific computing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
statsmodels>=0.14.0

# Deep learning frameworks
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
tensorflow>=2.13.0
transformers>=4.30.0
datasets>=2.12.0

# Medical AI specific
monai>=1.2.0
nibabel>=5.1.0
pydicom>=2.4.0
SimpleITK>=2.2.0
medpy>=0.4.0

# Natural language processing
spacy>=3.6.0
nltk>=3.8.0
gensim>=4.3.0
sentence-transformers>=2.2.0

# Computer vision
opencv-python>=4.8.0
pillow>=10.0.0
albumentations>=1.3.0
timm>=0.9.0

# Data visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
bokeh>=3.2.0
altair>=5.0.0

# Model interpretability
shap>=0.42.0
lime>=0.2.0
captum>=0.6.0

# Experiment tracking
mlflow>=2.5.0
wandb>=0.15.0
tensorboard>=2.13.0

# Data processing
dask>=2023.6.0
polars>=0.18.0
pyarrow>=12.0.0
h5py>=3.9.0

# Database connectivity
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
pymongo>=4.4.0
redis>=4.6.0

# API development
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
httpx>=0.24.0

# Cloud services
boto3>=1.28.0
google-cloud-storage>=2.10.0
azure-storage-blob>=12.17.0

# Utilities
tqdm>=4.65.0
joblib>=1.3.0
python-dotenv>=1.0.0
pyyaml>=6.0.0
click>=8.1.0

# Development tools
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.3.0

# Jupyter ecosystem
jupyter>=1.0.0
jupyterlab>=4.0.0
ipywidgets>=8.0.0
```

Install packages:

```bash
pip install -r requirements.txt

# Install additional packages for specific use cases
pip install lifelines  # Survival analysis
pip install causal-learn  # Causal inference
pip install fairlearn  # Fairness in ML
pip install interpret  # Model interpretability
pip install optuna  # Hyperparameter optimization
```

---

## Cloud Platform Configuration

### Amazon Web Services (AWS)

#### Initial Setup

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Access Key, region, and output format

# Install additional AWS tools
pip install boto3 sagemaker awswrangler
```

#### Healthcare-Specific AWS Services

```python
# AWS HealthLake setup example
import boto3

# Initialize HealthLake client
healthlake = boto3.client('healthlake', region_name='us-east-1')

# Create FHIR data store
response = healthlake.create_fhir_datastore(
    DatastoreName='healthcare-ai-datastore',
    DatastoreTypeVersion='R4',
    PreloadDataConfig={
        'PreloadDataType': 'SYNTHEA'
    }
)

# AWS Comprehend Medical setup
comprehend_medical = boto3.client('comprehendmedical', region_name='us-east-1')

# Example: Extract medical entities
text = "Patient presents with chest pain and shortness of breath."
response = comprehend_medical.detect_entities_v2(Text=text)
```

#### SageMaker Setup

```python
import sagemaker
from sagemaker import get_execution_role

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Create training job configuration
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='src',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001
    }
)
```

### Google Cloud Platform (GCP)

#### Initial Setup

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Install additional tools
pip install google-cloud-storage google-cloud-bigquery google-cloud-aiplatform
```

#### Healthcare API Setup

```python
from google.cloud import healthcare_v1

# Initialize Healthcare API client
client = healthcare_v1.HealthcareServiceClient()

# Create dataset
project_id = "your-project-id"
location = "us-central1"
dataset_id = "healthcare-ai-dataset"

parent = f"projects/{project_id}/locations/{location}"
dataset = healthcare_v1.Dataset()

response = client.create_dataset(
    parent=parent,
    dataset_id=dataset_id,
    dataset=dataset
)

# Create FHIR store
fhir_store = healthcare_v1.FhirStore()
fhir_store.version = healthcare_v1.FhirStore.Version.R4

fhir_store_id = "healthcare-ai-fhir-store"
response = client.create_fhir_store(
    parent=f"{parent}/datasets/{dataset_id}",
    fhir_store_id=fhir_store_id,
    fhir_store=fhir_store
)
```

#### Vertex AI Setup

```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(
    project="your-project-id",
    location="us-central1",
    staging_bucket="gs://your-bucket"
)

# Create custom training job
job = aiplatform.CustomTrainingJob(
    display_name="healthcare-ai-training",
    script_path="train.py",
    container_uri="gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest",
    requirements=["torch", "transformers", "datasets"],
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
```

### Microsoft Azure

#### Initial Setup

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Install Azure ML SDK
pip install azure-ai-ml azure-identity azure-storage-blob
```

#### Azure Health Data Services

```python
from azure.identity import DefaultAzureCredential
from azure.mgmt.healthcareapis import HealthcareApisManagementClient

# Initialize client
credential = DefaultAzureCredential()
client = HealthcareApisManagementClient(credential, subscription_id)

# Create FHIR service
from azure.mgmt.healthcareapis.models import FhirService

fhir_service = FhirService(
    location="East US",
    kind="fhir-R4",
    identity={"type": "SystemAssigned"},
    properties={
        "authenticationConfiguration": {
            "authority": "https://login.microsoftonline.com/tenant-id",
            "audience": "https://healthcare-ai-fhir.azurehealthcareapis.com",
            "smartProxyEnabled": False
        }
    }
)
```

#### Azure Machine Learning

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext

# Initialize ML client
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Create custom environment
env = Environment(
    name="healthcare-ai-env",
    description="Healthcare AI development environment",
    build=BuildContext(path="./docker-context"),
    conda_file="environment.yml",
)

ml_client.environments.create_or_update(env)
```

---

## Healthcare-Specific Security and Compliance

### HIPAA Compliance Setup

#### Encryption Configuration

```python
import cryptography
from cryptography.fernet import Fernet
import hashlib
import os

class HIPAACompliantEncryption:
    """HIPAA-compliant encryption utilities."""
    
    def __init__(self):
        self.key = self._generate_key()
        self.cipher = Fernet(self.key)
    
    def _generate_key(self):
        """Generate encryption key."""
        return Fernet.generate_key()
    
    def encrypt_phi(self, data):
        """Encrypt PHI data."""
        if isinstance(data, str):
            data = data.encode()
        return self.cipher.encrypt(data)
    
    def decrypt_phi(self, encrypted_data):
        """Decrypt PHI data."""
        return self.cipher.decrypt(encrypted_data).decode()
    
    def hash_identifier(self, identifier):
        """Create irreversible hash of identifier."""
        return hashlib.sha256(identifier.encode()).hexdigest()

# Example usage
encryptor = HIPAACompliantEncryption()
encrypted_mrn = encryptor.encrypt_phi("MRN123456")
hashed_id = encryptor.hash_identifier("patient_id_789")
```

#### Access Control and Audit Logging

```python
import logging
from datetime import datetime
from functools import wraps

# Configure audit logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/healthcare-ai/audit.log'),
        logging.StreamHandler()
    ]
)

audit_logger = logging.getLogger('healthcare_ai_audit')

def audit_access(resource_type):
    """Decorator for auditing data access."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get('user_id', 'unknown')
            audit_logger.info(
                f"ACCESS: User {user_id} accessed {resource_type} "
                f"via {func.__name__} at {datetime.now()}"
            )
            try:
                result = func(*args, **kwargs)
                audit_logger.info(
                    f"SUCCESS: User {user_id} successfully accessed {resource_type}"
                )
                return result
            except Exception as e:
                audit_logger.error(
                    f"FAILURE: User {user_id} failed to access {resource_type}: {str(e)}"
                )
                raise
        return wrapper
    return decorator

# Example usage
@audit_access("patient_data")
def get_patient_data(patient_id, user_id):
    # Implementation here
    pass
```

### Data De-identification

```python
import re
from datetime import datetime, timedelta
import random

class DataDeidentifier:
    """De-identify healthcare data for AI training."""
    
    def __init__(self):
        self.date_shift_days = random.randint(-365, 365)
        self.replacement_map = {}
    
    def deidentify_text(self, text):
        """Remove PHI from text data."""
        
        # Remove common PHI patterns
        patterns = {
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b': '[PHONE]',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',  # Email
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b': '[DATE]',  # Dates
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b': '[NAME]',  # Names (simple pattern)
        }
        
        deidentified_text = text
        for pattern, replacement in patterns.items():
            deidentified_text = re.sub(pattern, replacement, deidentified_text)
        
        return deidentified_text
    
    def shift_dates(self, date_obj):
        """Shift dates by consistent random offset."""
        if isinstance(date_obj, str):
            date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
        
        shifted_date = date_obj + timedelta(days=self.date_shift_days)
        return shifted_date
    
    def generate_synthetic_id(self, original_id):
        """Generate consistent synthetic ID."""
        if original_id not in self.replacement_map:
            self.replacement_map[original_id] = f"SYNTH_{len(self.replacement_map):06d}"
        return self.replacement_map[original_id]

# Example usage
deidentifier = DataDeidentifier()
clean_text = deidentifier.deidentify_text("Patient John Doe, SSN 123-45-6789, called on 01/15/2023")
synthetic_id = deidentifier.generate_synthetic_id("PATIENT_12345")
```

---

## Development Tools and IDEs

### Visual Studio Code Setup

#### Essential Extensions

```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.black-formatter",
        "ms-python.flake8",
        "ms-python.mypy-type-checker",
        "ms-toolsai.jupyter",
        "ms-toolsai.vscode-jupyter-cell-tags",
        "ms-toolsai.vscode-jupyter-slideshow",
        "ms-vscode.vscode-json",
        "redhat.vscode-yaml",
        "ms-vscode-remote.remote-ssh",
        "ms-vscode-remote.remote-containers",
        "ms-azuretools.vscode-docker",
        "github.copilot",
        "github.copilot-chat",
        "ms-vscode.vscode-github-issue-notebooks"
    ]
}
```

#### VS Code Settings

```json
{
    "python.defaultInterpreterPath": "./healthcare-ai-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.linting.mypyEnabled": true,
    "jupyter.askForKernelRestart": false,
    "jupyter.interactiveWindow.textEditor.executeSelection": true,
    "files.associations": {
        "*.yml": "yaml",
        "*.yaml": "yaml"
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### JupyterLab Configuration

```bash
# Install JupyterLab with extensions
pip install jupyterlab
pip install jupyterlab-git jupyterlab-lsp
pip install jupyter-ai  # AI assistant for Jupyter

# Install additional kernels
pip install ipykernel
python -m ipykernel install --user --name healthcare-ai --display-name "Healthcare AI"

# Configure JupyterLab
jupyter lab --generate-config

# Start JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### Custom JupyterLab Configuration

```python
# ~/.jupyter/jupyter_lab_config.py
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = 'your-secure-token'
c.ServerApp.password = 'your-hashed-password'

# Enable extensions
c.LabApp.collaborative = True
c.LabServerApp.extra_labextensions_path = ['/opt/conda/share/jupyter/labextensions']
```

### PyCharm Professional Setup

#### Project Configuration

```python
# .idea/misc.xml
<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectRootManager" version="2" project-jdk-name="Python 3.11 (healthcare-ai)" project-jdk-type="Python SDK" />
</project>

# .idea/healthcare-ai.iml
<?xml version="1.0" encoding="UTF-8"?>
<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager">
    <content url="file://$MODULE_DIR$">
      <excludeFolder url="file://$MODULE_DIR$/venv" />
      <excludeFolder url="file://$MODULE_DIR$/.pytest_cache" />
      <excludeFolder url="file://$MODULE_DIR$/data" />
      <excludeFolder url="file://$MODULE_DIR$/models" />
    </content>
    <orderEntry type="inheritedJdk" />
    <orderEntry type="sourceFolder" forTests="false" />
  </component>
</module>
```

---

## Package Management and Dependencies

### Conda Environment Management

```yaml
# environment.yml
name: healthcare-ai
channels:
  - conda-forge
  - pytorch
  - nvidia
  - huggingface
dependencies:
  - python=3.11
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scikit-learn>=1.3.0
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - torchaudio>=2.0.0
  - pytorch-cuda=12.1
  - tensorflow>=2.13.0
  - transformers>=4.30.0
  - datasets>=2.12.0
  - jupyter>=1.0.0
  - jupyterlab>=4.0.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - plotly>=5.15.0
  - pip
  - pip:
    - monai>=1.2.0
    - nibabel>=5.1.0
    - pydicom>=2.4.0
    - SimpleITK>=2.2.0
    - medpy>=0.4.0
    - shap>=0.42.0
    - lime>=0.2.0
    - mlflow>=2.5.0
    - wandb>=0.15.0
    - fastapi>=0.100.0
    - uvicorn>=0.22.0
```

Create environment:

```bash
conda env create -f environment.yml
conda activate healthcare-ai
```

### Docker Configuration

#### Base Dockerfile

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    unzip \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libnetcdf-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up non-root user for security
RUN useradd -m -u 1000 healthcare-ai && \
    chown -R healthcare-ai:healthcare-ai /app
USER healthcare-ai

# Expose ports
EXPOSE 8888 8000

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

#### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  healthcare-ai:
    build: .
    ports:
      - "8888:8888"
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./notebooks:/app/notebooks
      - ./src:/app/src
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - JUPYTER_TOKEN=your-secure-token
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - healthcare-ai-network

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: healthcare_ai
      POSTGRES_USER: healthcare_ai
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - healthcare-ai-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - healthcare-ai-network

  mlflow:
    image: python:3.11-slim
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri postgresql://healthcare_ai:secure_password@postgres:5432/healthcare_ai
               --default-artifact-root /mlflow/artifacts"
    ports:
      - "5000:5000"
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      - postgres
    networks:
      - healthcare-ai-network

volumes:
  postgres_data:
  redis_data:
  mlflow_artifacts:

networks:
  healthcare-ai-network:
    driver: bridge
```

---

## Data Management and Storage

### Database Setup

#### PostgreSQL Configuration

```sql
-- Create healthcare AI database
CREATE DATABASE healthcare_ai;
CREATE USER healthcare_ai WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE healthcare_ai TO healthcare_ai;

-- Connect to healthcare_ai database
\c healthcare_ai;

-- Create tables for patient data
CREATE TABLE patients (
    patient_id SERIAL PRIMARY KEY,
    mrn VARCHAR(50) UNIQUE NOT NULL,
    date_of_birth DATE,
    gender VARCHAR(10),
    race VARCHAR(50),
    ethnicity VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for clinical encounters
CREATE TABLE encounters (
    encounter_id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    encounter_date DATE NOT NULL,
    encounter_type VARCHAR(50),
    department VARCHAR(100),
    provider_id VARCHAR(50),
    diagnosis_codes TEXT[],
    procedure_codes TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for model predictions
CREATE TABLE model_predictions (
    prediction_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    patient_id INTEGER REFERENCES patients(patient_id),
    encounter_id INTEGER REFERENCES encounters(encounter_id),
    prediction_value FLOAT,
    prediction_probability FLOAT,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    features JSONB,
    metadata JSONB
);

-- Create indexes for performance
CREATE INDEX idx_patients_mrn ON patients(mrn);
CREATE INDEX idx_encounters_patient_id ON encounters(patient_id);
CREATE INDEX idx_encounters_date ON encounters(encounter_date);
CREATE INDEX idx_predictions_model ON model_predictions(model_name, model_version);
CREATE INDEX idx_predictions_patient ON model_predictions(patient_id);
```

#### Database Connection Setup

```python
# database.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import psycopg2
from contextlib import contextmanager

class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self):
        self.db_url = os.getenv(
            'DATABASE_URL',
            'postgresql://healthcare_ai:secure_password@localhost:5432/healthcare_ai'
        )
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_query(self, query, params=None):
        """Execute SQL query and return results."""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchall()
    
    def read_dataframe(self, query, params=None):
        """Read SQL query results into pandas DataFrame."""
        return pd.read_sql(query, self.engine, params=params)
    
    def write_dataframe(self, df, table_name, if_exists='append'):
        """Write pandas DataFrame to database table."""
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)

# Example usage
db = DatabaseManager()

# Read patient data
patients_df = db.read_dataframe("""
    SELECT p.patient_id, p.mrn, p.date_of_birth, p.gender,
           COUNT(e.encounter_id) as encounter_count
    FROM patients p
    LEFT JOIN encounters e ON p.patient_id = e.patient_id
    GROUP BY p.patient_id, p.mrn, p.date_of_birth, p.gender
""")

# Write predictions
predictions_df = pd.DataFrame({
    'model_name': ['risk_model_v1'] * 100,
    'model_version': ['1.0'] * 100,
    'patient_id': range(1, 101),
    'prediction_value': np.random.rand(100),
    'prediction_probability': np.random.rand(100)
})

db.write_dataframe(predictions_df, 'model_predictions')
```

### File Storage and Management

#### Local Storage Organization

```bash
# Create organized directory structure
mkdir -p healthcare-ai-project/{data,models,notebooks,src,config,logs,tests,docs}
mkdir -p healthcare-ai-project/data/{raw,processed,external,interim}
mkdir -p healthcare-ai-project/models/{trained,checkpoints,exports}
mkdir -p healthcare-ai-project/src/{data,features,models,visualization}

# Set up data versioning with DVC
cd healthcare-ai-project
pip install dvc[s3]  # or [gcs], [azure] for cloud storage
dvc init
dvc remote add -d myremote s3://your-bucket/dvc-cache
```

#### Cloud Storage Configuration

```python
# cloud_storage.py
import boto3
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient
import os

class CloudStorageManager:
    """Manage cloud storage operations across providers."""
    
    def __init__(self, provider='aws'):
        self.provider = provider
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize cloud storage client."""
        if self.provider == 'aws':
            self.client = boto3.client('s3')
        elif self.provider == 'gcp':
            self.client = gcs.Client()
        elif self.provider == 'azure':
            self.client = BlobServiceClient.from_connection_string(
                os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            )
    
    def upload_file(self, local_path, remote_path, bucket_name):
        """Upload file to cloud storage."""
        if self.provider == 'aws':
            self.client.upload_file(local_path, bucket_name, remote_path)
        elif self.provider == 'gcp':
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
        elif self.provider == 'azure':
            blob_client = self.client.get_blob_client(
                container=bucket_name, blob=remote_path
            )
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
    
    def download_file(self, remote_path, local_path, bucket_name):
        """Download file from cloud storage."""
        if self.provider == 'aws':
            self.client.download_file(bucket_name, remote_path, local_path)
        elif self.provider == 'gcp':
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(remote_path)
            blob.download_to_filename(local_path)
        elif self.provider == 'azure':
            blob_client = self.client.get_blob_client(
                container=bucket_name, blob=remote_path
            )
            with open(local_path, 'wb') as download_file:
                download_file.write(blob_client.download_blob().readall())

# Example usage
storage = CloudStorageManager(provider='aws')
storage.upload_file('model.pkl', 'models/v1/model.pkl', 'healthcare-ai-bucket')
```

---

## Model Development and Training Infrastructure

### MLflow Setup

```python
# mlflow_setup.py
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("healthcare-ai-experiments")

class MLflowManager:
    """Manage MLflow experiments and model tracking."""
    
    def __init__(self, experiment_name="healthcare-ai-experiments"):
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        self._create_experiment()
    
    def _create_experiment(self):
        """Create MLflow experiment if it doesn't exist."""
        try:
            self.experiment_id = self.client.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            self.experiment_id = self.client.get_experiment_by_name(
                self.experiment_name
            ).experiment_id
    
    def start_run(self, run_name=None):
        """Start MLflow run."""
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
    
    def log_model_metrics(self, model, X_test, y_test, model_type='sklearn'):
        """Log model and metrics."""
        if model_type == 'sklearn':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred = model.predict(X_test)
            
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))
            
            mlflow.sklearn.log_model(model, "model")
        
        elif model_type == 'pytorch':
            mlflow.pytorch.log_model(model, "model")

# Example usage
mlflow_manager = MLflowManager()

with mlflow_manager.start_run(run_name="diabetes_prediction_v1"):
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # Log model and metrics
    mlflow_manager.log_model_metrics(model, X_test, y_test, model_type='sklearn')
```

### Weights & Biases Setup

```python
# wandb_setup.py
import wandb
import os

# Initialize Weights & Biases
wandb.login(key=os.getenv('WANDB_API_KEY'))

class WandBManager:
    """Manage Weights & Biases experiments."""
    
    def __init__(self, project_name="healthcare-ai"):
        self.project_name = project_name
    
    def init_run(self, config, run_name=None, tags=None):
        """Initialize W&B run."""
        return wandb.init(
            project=self.project_name,
            config=config,
            name=run_name,
            tags=tags or []
        )
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to W&B."""
        wandb.log(metrics, step=step)
    
    def log_model(self, model_path, name="model"):
        """Log model artifact to W&B."""
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    def finish_run(self):
        """Finish W&B run."""
        wandb.finish()

# Example usage
wandb_manager = WandBManager()

config = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
    "model_type": "transformer"
}

run = wandb_manager.init_run(
    config=config,
    run_name="healthcare_transformer_v1",
    tags=["transformer", "healthcare", "classification"]
)

# During training
for epoch in range(config["epochs"]):
    # Training code here
    train_loss = 0.5 - epoch * 0.05  # Example
    val_loss = 0.6 - epoch * 0.04    # Example
    
    wandb_manager.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch
    })

wandb_manager.finish_run()
```

### Training Infrastructure

```python
# training_infrastructure.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import os
import json

class HealthcareAITrainer:
    """Healthcare AI model training infrastructure."""
    
    def __init__(self, model, device='cuda', mixed_precision=True):
        self.model = model
        self.device = device
        self.mixed_precision = mixed_precision
        
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.model.to(device)
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch to device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
                    loss = criterion(outputs.logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
            
            # Calculate metrics
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_samples:.4f}'
            })
        
        return total_loss / len(dataloader), correct_predictions / total_samples
    
    def validate(self, dataloader, criterion):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, labels)
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
        
        return total_loss / len(dataloader), correct_predictions / total_samples
    
    def save_checkpoint(self, epoch, optimizer, loss, save_path):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']

# Example training script
def train_healthcare_model():
    """Example training function."""
    
    # Initialize model (example with BERT for clinical text)
    from transformers import AutoModel, AutoConfig
    
    config = AutoConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 2  # Binary classification
    
    model = AutoModel.from_pretrained('bert-base-uncased', config=config)
    
    # Add classification head
    model.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    # Initialize trainer
    trainer = HealthcareAITrainer(model, device='cuda', mixed_precision=True)
    
    # Setup optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop (pseudo-code - replace with actual data loaders)
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_dataloader, optimizer, criterion)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_dataloader, criterion)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                epoch, optimizer, val_loss,
                f'models/best_model_epoch_{epoch}.pt'
            )
        
        # Save regular checkpoint
        trainer.save_checkpoint(
            epoch, optimizer, val_loss,
            f'models/checkpoint_epoch_{epoch}.pt'
        )

if __name__ == "__main__":
    train_healthcare_model()
```

---

## Deployment and Production Setup

### FastAPI Application Setup

```python
# app.py
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import joblib
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare AI API",
    description="Production API for healthcare AI models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Models storage
models = {}

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    patient_id: str
    features: dict
    model_name: str = "default"

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    patient_id: str
    prediction: float
    probability: float
    model_name: str
    model_version: str
    timestamp: datetime

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API token."""
    token = credentials.credentials
    # Implement your token verification logic here
    if token != os.getenv("API_TOKEN", "your-secure-token"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    try:
        # Load your trained models here
        models["risk_model"] = joblib.load("models/risk_model.pkl")
        models["readmission_model"] = joblib.load("models/readmission_model.pkl")
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """Make prediction using specified model."""
    try:
        # Get model
        if request.model_name not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = models[request.model_name]
        
        # Prepare features
        feature_array = np.array([list(request.features.values())])
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0].max()
        
        # Log prediction
        logger.info(f"Prediction made for patient {request.patient_id}")
        
        return PredictionResponse(
            patient_id=request.patient_id,
            prediction=float(prediction),
            probability=float(probability),
            model_name=request.model_name,
            model_version="1.0",
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models."""
    return {"models": list(models.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: healthcare-ai-api
  labels:
    app: healthcare-ai-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: healthcare-ai-api
  template:
    metadata:
      labels:
        app: healthcare-ai-api
    spec:
      containers:
      - name: healthcare-ai-api
        image: healthcare-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_TOKEN
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: token
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: healthcare-ai-service
spec:
  selector:
    app: healthcare-ai-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
data:
  token: <base64-encoded-token>

---
apiVersion: v1
kind: Secret
metadata:
  name: db-secrets
type: Opaque
data:
  url: <base64-encoded-database-url>
```

### Monitoring Setup

```python
# monitoring.py
import psutil
import GPUtil
import time
import logging
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import threading

# Prometheus metrics
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
gpu_usage = Gauge('gpu_usage_percent', 'GPU usage percentage', ['gpu_id'])
gpu_memory = Gauge('gpu_memory_usage_percent', 'GPU memory usage percentage', ['gpu_id'])
prediction_counter = Counter('predictions_total', 'Total predictions made', ['model_name'])
prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency', ['model_name'])

class SystemMonitor:
    """Monitor system resources and model performance."""
    
    def __init__(self, update_interval=10):
        self.update_interval = update_interval
        self.running = False
        
    def start_monitoring(self):
        """Start monitoring in background thread."""
        self.running = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start Prometheus metrics server
        start_http_server(8001)
        logging.info("Monitoring started on port 8001")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                cpu_usage.set(cpu_percent)
                memory_usage.set(memory_percent)
                
                # GPU monitoring
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        gpu_usage.labels(gpu_id=str(i)).set(gpu.load * 100)
                        gpu_memory.labels(gpu_id=str(i)).set(gpu.memoryUtil * 100)
                except:
                    pass  # GPU monitoring might fail if no GPUs available
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Monitoring error: {str(e)}")
                time.sleep(self.update_interval)

# Initialize monitoring
monitor = SystemMonitor()
monitor.start_monitoring()

# Decorator for tracking predictions
def track_prediction(model_name):
    """Decorator to track prediction metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                prediction_counter.labels(model_name=model_name).inc()
                return result
            finally:
                duration = time.time() - start_time
                prediction_latency.labels(model_name=model_name).observe(duration)
        
        return wrapper
    return decorator

# Example usage in FastAPI
@track_prediction("risk_model")
def make_risk_prediction(features):
    # Your prediction logic here
    pass
```

---

## Troubleshooting Common Issues

### CUDA and GPU Issues

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Fix common CUDA issues
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues

```python
# memory_optimization.py
import torch
import gc
import psutil
import os

def optimize_memory():
    """Optimize memory usage for large models."""
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Set memory fraction for PyTorch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    
    # Enable memory mapping for large datasets
    torch.multiprocessing.set_sharing_strategy('file_system')

def monitor_memory():
    """Monitor memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"RSS Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS Memory: {memory_info.vms / 1024 / 1024:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

# Use gradient checkpointing for large models
def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing to save memory."""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model
```

### Package Conflicts

```bash
# Create clean environment
conda create -n healthcare-ai-clean python=3.11
conda activate healthcare-ai-clean

# Install packages in specific order
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tensorflow -c conda-forge
pip install transformers datasets

# Check for conflicts
pip check

# Fix specific conflicts
pip install --upgrade --force-reinstall package-name
```

### Database Connection Issues

```python
# database_troubleshooting.py
import psycopg2
import sqlalchemy
from sqlalchemy import create_engine, text
import time

def test_database_connection(db_url, max_retries=5):
    """Test database connection with retries."""
    
    for attempt in range(max_retries):
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                print(f"Database connection successful on attempt {attempt + 1}")
                return True
        
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("All connection attempts failed")
                return False

# Test connection
db_url = "postgresql://healthcare_ai:secure_password@localhost:5432/healthcare_ai"
test_database_connection(db_url)
```

### Performance Optimization

```python
# performance_optimization.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class OptimizedDataLoader:
    """Optimized data loader for healthcare AI."""
    
    def __init__(self, dataset, batch_size=32, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_dataloader(self):
        """Get optimized DataLoader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=2,  # Prefetch batches
        )

def optimize_model_for_inference(model):
    """Optimize model for inference."""
    
    # Set to evaluation mode
    model.eval()
    
    # Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Enable inference mode
    torch.inference_mode()
    
    return model

def benchmark_model(model, sample_input, num_iterations=100):
    """Benchmark model performance."""
    
    model = optimize_model_for_inference(model)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(sample_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(sample_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"Average inference time: {avg_time * 1000:.2f} ms")
    
    return avg_time
```

---

## Summary

This comprehensive setup guide provides everything needed to establish a production-ready healthcare AI development environment. Key components include:

1. **Complete Environment Setup**: Local, cloud, and hybrid configurations
2. **Security and Compliance**: HIPAA-compliant encryption and audit logging
3. **Development Tools**: IDEs, notebooks, and debugging tools
4. **Data Management**: Database setup, cloud storage, and data versioning
5. **Training Infrastructure**: MLflow, Weights & Biases, and distributed training
6. **Production Deployment**: FastAPI, Kubernetes, and monitoring
7. **Troubleshooting**: Common issues and performance optimization

Following this guide ensures a robust, scalable, and compliant healthcare AI development environment that supports the full machine learning lifecycle from research to production deployment.

### Additional Resources

- [Healthcare AI Security Best Practices](https://www.hhs.gov/hipaa/for-professionals/security/guidance/cybersecurity/index.html)
- [MONAI Documentation](https://docs.monai.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)

This setup guide serves as the foundation for all healthcare AI development activities covered in the main chapters of this book.
