#!/bin/bash

# Quick connect script to your DDP instance
# Project: ddp-starter

PROJECT_ID="ddp-starter"
INSTANCE_NAME="pytorch-ddp-test"
ZONE="us-east1-c"

echo "Connecting to $INSTANCE_NAME in project $PROJECT_ID..."

# Set project
gcloud config set project $PROJECT_ID

# SSH into the instance
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE
