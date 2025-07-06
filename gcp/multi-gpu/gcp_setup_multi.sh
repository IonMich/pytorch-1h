#!/bin/bash

# GCP Multi-GPU DDP Setup Script
# This script tries multiple GPU types and zones to find available resources

PROJECT_ID="ddp-starter"
INSTANCE_NAME="pytorch-ddp-test-multi"

echo "🚀 Attempting to create multi-GPU instance for DDP testing..."
echo "This script will try different GPU types and zones until it finds available resources."
echo ""

# Set project
gcloud config set project $PROJECT_ID

# Define GPU configurations to try (in order of preference)
declare -a GPU_CONFIGS=(
    "nvidia-tesla-t4:2:n1-standard-8:T4 GPUs (most cost-effective)"
    "nvidia-tesla-v100:2:n1-standard-8:V100 GPUs (good performance)"
    "nvidia-tesla-a100:2:a2-highgpu-2g:A100 GPUs (best performance)"
    "nvidia-l4:1:g2-standard-8:L4 GPU (single GPU only)"
)

# Define zones to try
declare -a ZONES=(
    "us-central1-a"
    "us-central1-b" 
    "us-central1-c"
    "us-central1-f"
    "us-east1-b"
    "us-east1-c"
    "us-east1-d"
    "us-west1-a"
    "us-west1-b"
    "us-west1-c"
    "europe-west1-b"
    "europe-west1-c"
    "europe-west1-d"
    "asia-east1-a"
    "asia-east1-b"
    "asia-east1-c"
)

# Function to try creating instance
try_create_instance() {
    local gpu_type=$1
    local gpu_count=$2
    local machine_type=$3
    local zone=$4
    local description=$5
    
    echo "🔄 Trying: $description in zone $zone..."
    
    # Clean up any existing instance with the same name
    gcloud compute instances delete $INSTANCE_NAME --zone=$zone --quiet 2>/dev/null || true
    
    # Try to create the instance
    if gcloud compute instances create $INSTANCE_NAME \
        --zone=$zone \
        --machine-type=$machine_type \
        --accelerator=type=$gpu_type,count=$gpu_count \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=200GB \
        --boot-disk-type=pd-standard \
        --maintenance-policy=TERMINATE \
        --restart-on-failure \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --tags=pytorch-ddp \
        --quiet 2>/dev/null; then
        
        echo "✅ SUCCESS! Created instance with $description in $zone"
        echo "Instance name: $INSTANCE_NAME"
        echo "Zone: $zone"
        echo "GPU: $gpu_type x$gpu_count"
        echo "Machine type: $machine_type"
        echo ""
        echo "Waiting for instance to start..."
        
        # Wait for instance to be ready
        sleep 30
        
        echo "Instance details:"
        gcloud compute instances describe $INSTANCE_NAME --zone=$zone \
            --format="table(name,status,machineType.basename(),guestAccelerators[0].acceleratorType.basename(),guestAccelerators[0].acceleratorCount)"
        
        echo ""
        echo "🎉 Multi-GPU instance is ready!"
        echo ""
        echo "Next steps:"
        echo "1. Upload your DDP script:"
        echo "   ./upload_multi.sh"
        echo ""
        echo "2. Connect to the instance:"
        echo "   ./connect_multi.sh"
        echo ""
        echo "3. Run multi-GPU DDP training:"
        echo "   torchrun --nproc_per_node=$gpu_count ddp_training.py"
        
        # Create zone-specific scripts
        create_helper_scripts $zone $gpu_count
        
        return 0
    else
        echo "❌ Failed: Resource not available"
        return 1
    fi
}

# Function to create helper scripts for the successful configuration
create_helper_scripts() {
    local zone=$1
    local gpu_count=$2
    
    # Create upload script
    cat > upload_multi.sh << EOF
#!/bin/bash

# Upload DDP training script to multi-GPU instance
PROJECT_ID="$PROJECT_ID"
INSTANCE_NAME="$INSTANCE_NAME"
ZONE="$zone"
LOCAL_FILE="ddp_training.py"

echo "Uploading \$LOCAL_FILE to multi-GPU instance \$INSTANCE_NAME in \$ZONE..."

gcloud config set project \$PROJECT_ID

if [ ! -f "\$LOCAL_FILE" ]; then
    echo "Error: \$LOCAL_FILE not found in current directory"
    exit 1
fi

gcloud compute scp \$LOCAL_FILE \$INSTANCE_NAME:~/ --zone=\$ZONE

echo "✓ File uploaded successfully!"
echo ""
echo "Now connect to the instance:"
echo "  ./connect_multi.sh"
echo ""
echo "On the instance, run multi-GPU DDP:"
echo "  torchrun --nproc_per_node=$gpu_count ddp_training.py"
EOF

    # Create connect script
    cat > connect_multi.sh << EOF
#!/bin/bash

# Connect to multi-GPU instance
PROJECT_ID="$PROJECT_ID"
INSTANCE_NAME="$INSTANCE_NAME"
ZONE="$zone"

echo "Connecting to multi-GPU instance \$INSTANCE_NAME in \$ZONE..."

gcloud config set project \$PROJECT_ID
gcloud compute ssh \$INSTANCE_NAME --zone=\$ZONE
EOF

    # Create cleanup script
    cat > cleanup_multi.sh << EOF
#!/bin/bash

# Cleanup multi-GPU instance
PROJECT_ID="$PROJECT_ID"
INSTANCE_NAME="$INSTANCE_NAME"
ZONE="$zone"

echo "Cleaning up multi-GPU instance \$INSTANCE_NAME in \$ZONE..."

gcloud config set project \$PROJECT_ID
gcloud compute instances delete \$INSTANCE_NAME --zone=\$ZONE

echo "✓ Multi-GPU instance deleted"
EOF

    chmod +x upload_multi.sh connect_multi.sh cleanup_multi.sh
    
    echo ""
    echo "📝 Created helper scripts:"
    echo "  - upload_multi.sh  : Upload DDP script to the instance"
    echo "  - connect_multi.sh : Connect to the instance"
    echo "  - cleanup_multi.sh : Delete the instance"
}

# Main execution
echo "Trying different GPU configurations and zones..."
echo ""

success=false

for config in "${GPU_CONFIGS[@]}"; do
    if [ "$success" = true ]; then
        break
    fi
    
    IFS=':' read -r gpu_type gpu_count machine_type description <<< "$config"
    
    echo "🎯 Trying GPU configuration: $description"
    
    for zone in "${ZONES[@]}"; do
        if try_create_instance "$gpu_type" "$gpu_count" "$machine_type" "$zone" "$description"; then
            success=true
            break
        fi
        sleep 2  # Brief pause between attempts
    done
    
    if [ "$success" = false ]; then
        echo "❌ No resources available for $description in any zone"
        echo ""
    fi
done

if [ "$success" = false ]; then
    echo ""
    echo "😞 No multi-GPU resources available in any zone at this time."
    echo ""
    echo "Alternative approaches:"
    echo ""
    echo "1. 📋 Request quota increase:"
    echo "   Visit: https://console.cloud.google.com/iam-admin/quotas"
    echo "   Search for 'GPUs' and request increase for specific regions"
    echo ""
    echo "2. ⏰ Try again later:"
    echo "   GPU availability changes throughout the day"
    echo "   Run this script again in a few hours"
    echo ""
    echo "3. 🌍 Try other regions:"
    echo "   Edit this script to include more international zones"
    echo ""
    echo "4. 💰 Consider Spot instances:"
    echo "   Add --preemptible flag for cheaper, temporary instances"
    echo ""
    echo "5. 🔄 Single GPU simulation:"
    echo "   Your current single GPU setup can simulate multi-GPU DDP"
    echo "   The DDP code structure is the same, just with 1 process"
    echo ""
    echo "6. ☁️ Alternative platforms:"
    echo "   Consider AWS, Azure, or specialized ML platforms like:"
    echo "   - Paperspace Gradient"
    echo "   - Lambda Labs"
    echo "   - RunPod"
    echo "   - Vast.ai"
    
    exit 1
fi
