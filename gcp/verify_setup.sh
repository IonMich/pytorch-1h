#!/bin/bash

# Verification script to check your DDP setup on GCP
# Run this AFTER connecting to your instance

echo "=========================================="
echo "DDP Setup Verification"
echo "=========================================="

echo "1. Checking NVIDIA driver installation..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ NVIDIA drivers installed"
else
    echo "x NVIDIA drivers not found. Wait 5-10 minutes and try again."
    exit 1
fi

echo ""
echo "2. Checking Python and PyTorch..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    
    if torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print('✓ PyTorch can access GPUs')
    else:
        print('x No GPUs detected by PyTorch')
        
except ImportError:
    print('x PyTorch not installed')
"

echo ""
echo "3. Checking torchrun availability..."
if command -v torchrun &> /dev/null; then
    echo "✓ torchrun is available"
    torchrun --help | head -5
else
    echo "x torchrun not found"
fi

echo ""
echo "4. Checking for DDP training script..."
if [ -f "ddp_training.py" ]; then
    echo "✓ ddp_training.py found"
    echo "File size: $(wc -l < ddp_training.py) lines"
else
    echo "x ddp_training.py not found in current directory"
    echo "Upload it using: gcloud compute scp ddp_training.py pytorch-ddp-test:~/ --zone=us-west1-b"
fi

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="

if [ -f "ddp_training.py" ]; then
    echo ""
    echo "Ready to run DDP training!"
    echo "Execute: torchrun --nproc_per_node=2 ddp_training.py"
else
    echo ""
    echo "Upload your training script first, then run this verification again."
fi
