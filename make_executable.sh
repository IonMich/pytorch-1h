#!/bin/bash

# Make all GCP scripts executable
# Project: ddp-starter

echo "Making all scripts executable..."

chmod +x gcp_setup.sh
chmod +x connect.sh
chmod +x upload.sh
chmod +x cleanup.sh
chmod +x verify_setup.sh

echo "✓ All scripts are now executable!"
echo ""
echo "You can now run:"
echo "  ./gcp_setup.sh    - Create your GCP instance" 
echo "  ./upload.sh       - Upload DDP script"
echo "  ./connect.sh      - Connect to instance"
echo "  ./cleanup.sh      - Clean up when done"
