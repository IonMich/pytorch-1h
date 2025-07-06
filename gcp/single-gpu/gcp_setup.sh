#!/bin/bash

# GCP DDP PyTorch Setup Script
# Project: ddp-starter
# This script sets up a multi-GPU instance for testing DDP PyTorch code

set -e  # Exit on any error

# Configuration
PROJECT_ID="ddp-starter"
INSTANCE_NAME="pytorch-ddp-test"
ZONE="us-east1-c"
MACHINE_TYPE="n1-standard-8"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=2
BOOT_DISK_SIZE="100GB"

echo "=========================================="
echo "GCP DDP PyTorch Setup"
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "GPUs: $GPU_COUNT x $GPU_TYPE"
echo "=========================================="

# Set the project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Verify project is set
CURRENT_PROJECT=$(gcloud config get-value project)
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo "Error: Project not set correctly. Current: $CURRENT_PROJECT, Expected: $PROJECT_ID"
    exit 1
fi

echo "✓ Project set to: $CURRENT_PROJECT"

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
    echo "Instance $INSTANCE_NAME already exists in zone $ZONE"
    echo "Delete it first or choose a different name."
    exit 1
fi

echo "Creating instance $INSTANCE_NAME..."

# Create the instance
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-standard \
    --metadata="install-nvidia-driver=True" \
    --tags=pytorch-ddp

echo "✓ Instance created successfully!"

# Wait a moment for instance to be ready
echo "Waiting for instance to be ready..."
sleep 10

# Show instance information
echo ""
echo "Instance Information:"
gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="table(
    name,
    status,
    machineType.scope(machineTypes):label=MACHINE_TYPE,
    zone.scope(zones):label=ZONE
)"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Wait 5-10 minutes for NVIDIA drivers to install"
echo "2. Connect to instance:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "3. Verify GPUs are available:"
echo "   nvidia-smi"
echo ""
echo "4. Upload your DDP script:"
echo "   gcloud compute scp ddp_training.py $INSTANCE_NAME:~/ --zone=$ZONE"
echo ""
echo "5. Run DDP training:"
echo "   torchrun --nproc_per_node=$GPU_COUNT ddp_training.py"
echo ""
echo "To delete instance when done:"
echo "   gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Estimated cost: ~$2.00/hour with 2x T4 GPUs"
echo "=========================================="
