# Multi-GPU DDP Setup

This setup attempts to create a multi-GPU instance on GCP for true distributed PyTorch DDP training across multiple GPUs.

## ❌ Status: Blocked by Resource Availability

Currently blocked due to GCP resource exhaustion for multi-GPU instances in all attempted regions and GPU types.

## 🎯 What This Would Demonstrate

With multiple GPUs, you would experience:
- ✅ True parallel training across GPUs
- ✅ Real distributed communication
- ✅ Significant training speedup
- ✅ Multi-process coordination
- ✅ Production-ready DDP patterns

## 🚀 How to Retry

### Option 1: Automated Retry Script

```bash
./gcp_setup_multi.sh
```

This script automatically tries:
- Multiple GPU types (T4, V100, A100, L4)
- Multiple zones across US and international regions
- Different machine configurations
- Creates helper scripts when successful

### Option 2: Manual Retry

Try specific configurations:

```bash
# Try T4 GPUs in different zones
gcloud compute instances create pytorch-ddp-test-multi \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=2 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-standard \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=pytorch-ddp
```

## 🔄 Retry Strategies

### Timing
- **Peak hours**: Avoid 9 AM - 5 PM PST (high demand)
- **Off-peak**: Try evenings, weekends, early mornings
- **International**: Try European/Asian regions

### GPU Types (in order of availability)
1. **T4**: Most cost-effective, high demand
2. **V100**: Good performance, moderate availability  
3. **A100**: Best performance, premium pricing
4. **L4**: Newest, limited to single GPU

### Regions to Try
```bash
# US regions
us-central1-a, us-central1-b, us-central1-c, us-central1-f
us-east1-b, us-east1-c, us-east1-d
us-west1-a, us-west1-b, us-west1-c

# International (may have better availability)
europe-west1-b, europe-west1-c, europe-west1-d
asia-east1-a, asia-east1-b, asia-east1-c
```

## 📋 Quota Requirements

Check your quotas:
```bash
gcloud compute project-info describe --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -i gpu
```

Request increases at:
https://console.cloud.google.com/iam-admin/quotas

## 💰 Alternative Approaches

### 1. Preemptible Instances (Cheaper)
Add `--preemptible` flag to reduce costs by ~70%

### 2. Different Cloud Providers
- **AWS**: EC2 P3/P4 instances
- **Azure**: NC/ND series VMs
- **Paperspace**: Gradient platform
- **Lambda Labs**: Dedicated GPU cloud
- **RunPod**: Community GPU cloud

### 3. Local Multi-GPU
If you have local hardware:
- 2+ NVIDIA GPUs
- CUDA drivers installed
- Same DDP code works locally

## 🎯 When Successful

Once you get a multi-GPU instance:

1. **Upload your script**:
   ```bash
   ./upload_multi.sh
   ```

2. **Connect**:
   ```bash
   ./connect_multi.sh
   ```

3. **Run true multi-GPU DDP**:
   ```bash
   # On the instance
   torchrun --nproc_per_node=2 ddp_training.py
   ```

4. **Clean up**:
   ```bash
   ./cleanup_multi.sh
   ```

## 📂 Files in this folder

- `gcp_setup_multi.sh` - Automated multi-GPU instance creation script
- Generated files (when successful):
  - `upload_multi.sh` - Upload script for successful instance
  - `connect_multi.sh` - Connect script for successful instance  
  - `cleanup_multi.sh` - Cleanup script for successful instance

## 🆘 Troubleshooting

### If the automated script finds resources:
✅ Use the generated helper scripts

### If no resources are found:
1. Try again at different times
2. Request quota increases
3. Try the single-GPU setup first
4. Consider alternative platforms

### Cost monitoring:
```bash
# Always check for running instances
gcloud compute instances list
```

## ➡️ Recommended Next Steps

1. **Start with single GPU**: Master DDP concepts first
2. **Set up monitoring**: Get alerts for resource availability
3. **Request quotas**: Increase limits for your preferred regions
4. **Try alternatives**: Explore other cloud providers
5. **Local setup**: Consider local multi-GPU if available
