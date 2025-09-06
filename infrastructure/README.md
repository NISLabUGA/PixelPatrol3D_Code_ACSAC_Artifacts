# Infrastructure Requirements and Public Cloud Compatibility

This document outlines the infrastructure requirements for the PP3D artifact and provides guidance for running the evaluation on public cloud infrastructure.

## Public Infrastructure Compatibility

The PP3D artifact is designed to be **fully compatible with public cloud infrastructure** and can be successfully executed on various public platforms without requiring special hardware or proprietary systems.

### Recommended Public Infrastructure Platforms

#### 1. **Google Colab** (Recommended for Quick Evaluation)

- **Compatibility**: ✅ Supported with limitations
- **GPU Access**: Available with Colab Pro for faster evaluation
- **Storage**: **Limited** - 500GB dataset may require Google Drive Pro or external storage
- **Setup**: Standard Python environment with pip package installation
- **Limitations**:
  - Session timeouts (12-24 hours max) may interrupt long evaluations
  - Storage constraints may require careful dataset management
  - Network download speeds may vary

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
- **Storage**: 500GB free space (**Note**: This is substantial - verify platform storage limits)
- **Network**: Stable internet connection for dataset download (218GB download required)
- **OS**: Linux (Ubuntu 22.04 recommended, Ubuntu 18.04+, CentOS 7+) or macOS

### Recommended Configuration

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: SSD with 500GB+ free space
- **Network**: High-bandwidth connection for dataset download

### Tested Environment

The artifact has been extensively tested on **Ubuntu 22.04** and the `install.sh` script works reliably in this environment. We specifically recommend:

- **Ubuntu 22.04 LTS** for optimal compatibility
- Docker containerization for isolated and reproducible execution
- VM or bare metal deployment with Ubuntu 22.04

## Setup Instructions for Public Infrastructure

### 1. Artifact Setup

```bash
# Clone or download the artifact
git clone <artifact-repository-url>
cd PixelPatrol3D_Code_ACSAC_Artifacts

# Run installation script
bash install.sh

# Download dataset
python3 download_data.py
```

### 2. Docker Setup (Recommended for Reproducible Environment)

For the most reproducible and isolated execution environment, we recommend using Docker with Ubuntu 22.04:

```bash
# Run Docker container with GPU support (if available)
docker run -d -i -t \
  -v /path/to/PixelPatrol3D_Code_ACSAC_Artifacts:/mnt/pp3d \
  --ipc=host \
  --gpus '"device=0"' \
  ubuntu:22.04 /bin/bash

# For CPU-only execution (no GPU required)
docker run -d -i -t \
  -v /path/to/PixelPatrol3D_Code_ACSAC_Artifacts:/mnt/pp3d \
  --ipc=host \
  ubuntu:22.04 /bin/bash

# Enter the container
docker exec -it <container_id> /bin/bash

# Inside the container, navigate to the artifact
cd /mnt/pp3d

# Run the installation script (tested and verified on Ubuntu 22.04)
bash install.sh

# Download dataset
python3 download_data.py
```

**Note**: Replace `/path/to/PixelPatrol3D_Code_ACSAC_Artifacts` with the actual path to your artifact directory on the host system.

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

- **Dataset Storage**: **Challenge** - 500GB dataset exceeds free Google Drive (15GB). Consider:
  - Google Drive Pro subscription (2TB for $10/month)
  - Download dataset in chunks and process sequentially
  - Use external cloud storage integration
- **Session Management**: **Critical** - Save intermediate results frequently due to timeouts
- **GPU Access**: Enable GPU runtime in notebook settings (limited hours on free tier)
- **Memory Management**: Monitor RAM usage, restart runtime if needed
- **Recommendation**: Best suited for individual claims rather than full evaluation

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

## Expected Performance on Public Infrastructure

### CPU-Only Execution

- **Claim 1 (RQ1 & RQ4)**: ~15-20 minutes
- **Claim 2 (RQ2)**: ~45-60 minutes
- **Claim 3 (RQ3)**: ~45-60 minutes
- **Claim 4 (RQ5)**: ~45-60 minutes

### GPU-Accelerated Execution

- **Claim 1 (RQ1 & RQ4)**: <10 minutes
- **Claim 2 (RQ2)**: <30 minutes
- **Claim 3 (RQ3)**: <30 minutes
- **Claim 4 (RQ5)**: <30 minutes

### Training Verification Claims (Optional)

- **Claim 5 (RQ1 & RQ4 Training)**: ~15-30 minutes (1-2 epochs verification)
- **Claim 6 (RQ5 Adversarial Training)**: ~45-90 minutes (1-2 epochs verification)
- **Claim 7 (RQ2 Resolution Training)**: ~15-30 minutes (1-2 epochs verification)
- **Claim 8 (RQ3 Campaign Training)**: ~15-30 minutes (1-2 epochs verification)

## Troubleshooting Public Infrastructure Issues

### Common Issues and Solutions

1. **Dataset Download Timeouts**

   - Use `python3 download_data.py --no-extract` to download only
   - Extract files separately: `python3 download_data.py --verify-only`
   - **For constrained environments**: Contact authors for alternative download methods

2. **Memory Limitations**

   - Reduce batch sizes in evaluation scripts
   - Run claims sequentially instead of parallel
   - Use swap space if available
   - **Severe limitations**: Focus on Claim 1 only for core validation

3. **Storage Limitations**

   - Use `python3 download_data.py --cleanup` to remove zip files after extraction
   - Download only required datasets for specific research questions
   - **Critical limitation**: 500GB requirement may exceed some platform limits

4. **Network Connectivity**

   - Verify outbound HTTPS access to pp3d-data.sdkhomelab.com
   - Use alternative download methods if direct access is blocked
   - **Restricted networks**: May require pre-downloaded dataset transfer

5. **GPU Access Issues**

   - Verify CUDA installation and compatibility
   - Fall back to CPU-only execution if GPU unavailable
   - Check platform-specific GPU allocation procedures
   - **No GPU access**: Expect 2-4x longer execution times

6. **Platform-Specific Limitations**

   - **Google Colab**: Storage and session timeout constraints
   - **Research Platforms**: May require allocation approval and scheduling
   - **Commercial Cloud**: Costs may accumulate for large dataset storage

## Important Limitations and Considerations

### Storage Requirements

- **500GB dataset** is substantial and may exceed free tier limits on some platforms
- **Verify storage availability** before beginning download
- Consider **cost implications** for cloud storage on commercial platforms

### Network Requirements

- **218GB download** requires stable, high-bandwidth connection
- Some institutional or restricted networks may block large downloads
- **Plan for several hours** of download time depending on connection speed

### Platform Suitability

- **Google Colab**: Best for individual claims, challenging for full evaluation
- **Research Platforms**: Excellent for full evaluation but may require allocation approval
- **Commercial Cloud**: Fully capable but consider storage and compute costs

## Conclusion

The PP3D artifact is **compatible with public infrastructure** but has **significant resource requirements** (500GB storage, 218GB download). While the evaluation can be completed on the recommended platforms, reviewers should:

1. **Verify storage capacity** before starting
2. **Plan for substantial download time** (218GB dataset)
3. **Consider platform limitations** (especially Google Colab storage constraints)
4. **Start with individual claims** to validate setup before full evaluation

For questions about specific platform setup, alternative approaches for constrained environments, or troubleshooting, please refer to the main README.md or contact the artifact authors.
