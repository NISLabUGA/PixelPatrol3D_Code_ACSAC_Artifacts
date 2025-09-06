#!/bin/bash

# PixelPatrol3D Artifact Installation Script
# This script sets up the environment for reproducing the paper results

echo "Installing PixelPatrol3D Artifact Dependencies..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "Python version check passed: $python_version"

# Check if CUDA is available (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: CUDA not detected. Training will use CPU (much slower)."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify key dependencies
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python3 -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"

# Check if CUDA is available in PyTorch
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python3 -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
fi

echo ""
echo "Installation complete!"
echo ""
echo "To run the reproducibility claims:"
echo "1. Navigate to a claim directory: cd claims/claim1"
echo "2. Run the evaluation script: bash run.sh"
echo "3. Check results in the expected/ directory"
echo ""
echo "Available claims:"
echo "- claim1: RQ1 & RQ4 (New BMA instances and fresh attacks)"
echo "- claim2: RQ2 (New screen resolutions)"
echo "- claim3: RQ3 (Never-before-seen campaigns)"
echo "- claim4: RQ5 (Adversarial robustness)"
