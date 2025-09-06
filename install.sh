#!/bin/bash
set -e  # exit immediately if any command fails

# PixelPatrol3D Artifact Installation Script
# This script sets up the environment for reproducing the paper results

echo "Installing PixelPatrol3D Artifact Dependencies..."

# -------------------------------
# Python check / install
# -------------------------------
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    echo "Error: Python 3.8 or higher is required."
    if command -v python3 &>/dev/null; then
        echo "Found: $(python3 --version 2>&1)"
    else
        echo "Python is not installed."
    fi

    read -p "Would you like to install Python now? (y/n) " choice
    case "$choice" in 
      y|Y )
        echo "Installing Python (latest available from distro)..."
        if command -v apt-get &>/dev/null; then
            PKG="apt-get"
        elif command -v yum &>/dev/null; then
            PKG="yum"
        fi

        if [ -n "$PKG" ]; then
            if [ "$(id -u)" -eq 0 ]; then
                $PKG update -y || true
                if [ "$PKG" = "apt-get" ]; then
                    $PKG install -y python3 python3-pip python3-venv python3-dev
                else
                    $PKG install -y python3 python3-pip
                fi
            else
                sudo $PKG update -y || true
                if [ "$PKG" = "apt-get" ]; then
                    sudo $PKG install -y python3 python3-pip python3-venv python3-dev
                else
                    sudo $PKG install -y python3 python3-pip
                fi
            fi
        else
            echo "Unsupported package manager. Please install Python manually."
            exit 1
        fi
        ;;
      * )
        echo "Please install Python >= 3.8 manually and rerun this script."
        exit 1
        ;;
    esac
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version check passed: $python_version"

# -------------------------------
# CUDA check
# -------------------------------
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_CUDA=1
else
    echo "Warning: CUDA not detected. Training will use CPU (much slower)."
    HAS_CUDA=0
fi

# -------------------------------
# Install PyTorch first
# -------------------------------
echo "Installing PyTorch..."
if [ $HAS_CUDA -eq 1 ]; then
    python3 -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --extra-index-url https://download.pytorch.org/whl/cu121
else
    python3 -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
fi

# -------------------------------
# Install other dependencies
# -------------------------------
echo "Installing other Python dependencies..."
python3 -m pip install -r requirements.txt

# -------------------------------
# Verify key dependencies
# -------------------------------
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python3 -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
fi

# -------------------------------
# Finish
# -------------------------------
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
