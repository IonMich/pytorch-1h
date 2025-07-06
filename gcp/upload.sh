#!/bin/bash

# Upload DDP training script to GCP instance
# Project: ddp-starter

PROJECT_ID="ddp-starter"
INSTANCE_NAME="pytorch-ddp-test"
ZONE="us-east1-c"
LOCAL_FILE="ddp_training.py"

echo "Uploading $LOCAL_FILE to $INSTANCE_NAME..."

# Set project
gcloud config set project $PROJECT_ID

# Check if local file exists
if [ ! -f "$LOCAL_FILE" ]; then
    echo "Error: $LOCAL_FILE not found in current directory"
    echo "Make sure you're in the pytorch-1h directory"
    exit 1
fi

# Upload the file
gcloud compute scp $LOCAL_FILE $INSTANCE_NAME:~/ --zone=$ZONE

echo "✓ File uploaded successfully!"
echo ""
echo "Now connect to the instance and run:"
echo "  ./connect.sh"
echo "  torchrun --nproc_per_node=2 ddp_training.py"
