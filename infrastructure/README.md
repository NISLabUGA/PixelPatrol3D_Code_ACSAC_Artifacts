# Infrastructure Requirements and Public Cloud Compatibility

This document outlines the infrastructure requirements for the PP3D artifact and provides guidance for running the evaluation on public cloud infrastructure.

## Public Infrastructure Compatibility

The PP3D artifact is designed to be **fully compatible with public cloud infrastructure** and can be successfully executed on various public platforms without requiring special hardware or proprietary systems.

### Recommended Public Infrastructure Platforms

#### 1. **Google Colab** (Recommended for Quick Evaluation)

- **Compatibility**: ✅ Fully supported
- **GPU Access**: Available with Colab Pro for faster evaluation
- **Storage**: Sufficient for dataset (Google Drive integration available)
- **Setup**: Standard Python environment with pip package installation
- **Limitations**: Session timeouts may require splitting long evaluations

#### 2. **SPHERE**

- **Compatibility**: ✅ Fully supported
- **Advantages**: Persistent storage, longer execution times
- **GPU Access**: Available on GPU-enabled nodes
- **Setup**: Standard Linux environment with Python 3.8+

#### 3. **Chameleon Cloud**

- **Compatibility**: ✅ Fully supported
- **Advantages**: Bare metal access, flexible resource allocation
- **GPU Access**: Available on GPU-enabled instances
- **Setup**: Ubuntu/CentOS with Python environment

#### 4. **CloudLab**

- **Compatibility**: ✅ Fully supported
- **Advantages**: Research-focused infrastructure, high-performance nodes
- **GPU Access**: Available on select node types
- **Setup**: Standard Linux distribution with Python

#### 5. **FABRIC**

- **Compatibility**: ✅ Fully supported
- **Advantages**: High-performance research infrastructure
- **GPU Access**: Available on GPU-enabled slices
- **Setup**: Federated research environment

## Resource Requirements for Public Infrastructure

### Minimum Requirements

- **CPU**: 4 cores (any modern x86_64 processor)
- **RAM**: 8GB (16GB recommended for faster processing)
- **Storage**: 500GB free space
- **Network**: Stable internet connection for dataset download
- **OS**: Linux (Ubuntu 18.04+, CentOS 7+) or macOS

### Recommended Configuration

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: SSD with 500GB+ free space
- **Network**: High-bandwidth connection for dataset download

## Setup Instructions for Public Infrastructure

### 1. Environment Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y  # Ubuntu/Debian
# or
sudo yum update -y  # CentOS/RHEL

# Install Python 3.8+ if not available
sudo apt install python3 python3-pip python3-venv  # Ubuntu/Debian
# or
sudo yum install python3 python3-pip  # CentOS/RHEL

# Install git if not available
sudo apt install git  # Ubuntu/Debian
# or
sudo yum install git  # CentOS/RHEL
```

### 2. Artifact Setup

```bash
# Clone or download the artifact
git clone <artifact-repository-url>
cd PixelPatrol3D_Code_ACSAC_Artifacts

# Run installation script
bash install.sh

# Download dataset
python3 download_data.py
```

### 3. GPU Setup (Optional but Recommended)

```bash
# For NVIDIA GPU support
# Install CUDA toolkit (version 11.0+)
# This varies by platform - follow platform-specific CUDA installation guides

# Verify GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Platform-Specific Considerations

### Google Colab

- **Dataset Storage**: Use Google Drive mount for persistent dataset storage
- **Session Management**: Save intermediate results to prevent loss during timeouts
- **GPU Access**: Enable GPU runtime in notebook settings
- **Memory Management**: Monitor RAM usage, restart runtime if needed

### SPHERE/Chameleon/CloudLab/FABRIC

- **Persistent Storage**: Ensure adequate persistent storage allocation
- **Network Access**: Verify outbound internet access for dataset download
- **GPU Allocation**: Request GPU-enabled nodes when available
- **Environment**: Use virtual environments to avoid system conflicts

## Execution Strategy for Public Infrastructure

### Option 1: Sequential Execution (Recommended for Limited Resources)

```bash
# Run claims one at a time to manage resource usage
cd claims/claim1 && bash run.sh
cd ../claim2 && bash run.sh
cd ../claim3 && bash run.sh
cd ../claim4 && bash run.sh
```

### Option 2: Parallel Execution (For High-Resource Environments)

```bash
# Run multiple claims in parallel if resources allow
# Monitor system resources to avoid overloading
```

### Option 3: Subset Evaluation (For Quick Validation)

```bash
# Start with smaller claims for quick validation
cd claims/claim1 && bash run.sh  # ~30 minutes
cd ../claim4 && bash run.sh      # ~1 hour
```

## Expected Performance on Public Infrastructure

### CPU-Only Execution

- **Claim 1 (RQ1 & RQ4)**: ~45-60 minutes
- **Claim 2 (RQ2)**: ~6-8 hours
- **Claim 3 (RQ3)**: ~8-10 hours
- **Claim 4 (RQ5)**: ~1.5-2 hours

### GPU-Accelerated Execution

- **Claim 1 (RQ1 & RQ4)**: ~30 minutes
- **Claim 2 (RQ2)**: ~4 hours
- **Claim 3 (RQ3)**: ~6 hours
- **Claim 4 (RQ5)**: ~1 hour

## Troubleshooting Public Infrastructure Issues

### Common Issues and Solutions

1. **Dataset Download Timeouts**

   - Use `python3 download_data.py --no-extract` to download only
   - Extract files separately: `python3 download_data.py --verify-only`

2. **Memory Limitations**

   - Reduce batch sizes in evaluation scripts
   - Run claims sequentially instead of parallel
   - Use swap space if available

3. **Storage Limitations**

   - Use `python3 download_data.py --cleanup` to remove zip files after extraction
   - Download only required datasets for specific research questions

4. **Network Connectivity**

   - Verify outbound HTTPS access to pp3d-data.sdkhomelab.com
   - Use alternative download methods if direct access is blocked

5. **GPU Access Issues**

   - Verify CUDA installation and compatibility
   - Fall back to CPU-only execution if GPU unavailable
   - Check platform-specific GPU allocation procedures

## Cost Estimation for Public Cloud Platforms

### Google Colab

- **Free Tier**: Sufficient for basic evaluation (with session management)
- **Colab Pro**: ~$10/month, recommended for full evaluation with GPU

### Commercial Cloud Platforms (AWS/GCP/Azure)

- **CPU Instance**: ~$50-100 for complete evaluation
- **GPU Instance**: ~$100-200 for complete evaluation
- **Storage**: ~$20-30 for dataset storage

### Research Platforms (SPHERE/Chameleon/CloudLab/FABRIC)

- **Cost**: Free for academic research use
- **Allocation**: Requires research project allocation/proposal

## Conclusion

The PP3D artifact is fully compatible with public infrastructure and does not require any special hardware, proprietary systems, or restricted access. The evaluation can be successfully completed on any of the recommended public platforms with standard computational resources.

For questions about specific platform setup or troubleshooting, please refer to the main README.md or contact the artifact authors.
