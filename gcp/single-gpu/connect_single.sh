#!/bin/bash

# Connect to single GPU instance for testing
# Project: ddp-starter

PROJECT_ID="ddp-starter"
INSTANCE_NAME="pytorch-ddp-test-single"
ZONE="us-east1-c"

echo "Connecting to single GPU instance $INSTANCE_NAME..."

# Set project
gcloud config set project $PROJECT_ID

# SSH into the instance
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE
