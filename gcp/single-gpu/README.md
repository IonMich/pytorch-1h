# Single GPU DDP Setup

This setup creates a single T4 GPU instance on GCP for learning PyTorch DDP concepts. While it uses only one GPU, it demonstrates the complete DDP structure and workflow.

## ✅ Status: Working and Tested

This setup has been successfully tested and verified to work.

## 🚀 Quick Start

1. **Create the instance**:
   ```bash
   ./gcp_setup.sh
   ```

2. **Upload your DDP script**:
   ```bash
   ./upload_single.sh
   ```

3. **Connect to the instance**:
   ```bash
   ./connect_single.sh
   ```

4. **Run DDP training** (on the instance):
   ```bash
   # Verify setup first
   ./verify_setup.sh
   
   # Run single-GPU DDP
   torchrun --nproc_per_node=1 ddp_training.py
   ```

## 📋 What This Demonstrates

Even with a single GPU, you'll learn:
- ✅ DDP initialization and setup
- ✅ Distributed sampler usage
- ✅ Model synchronization concepts
- ✅ Proper cleanup and finalization
- ✅ Complete training workflow

## 💰 Cost

- **Instance**: ~$0.35/hour for T4 GPU
- **Storage**: ~$0.04/month for 100GB
- **Total**: Very affordable for learning

## 🔄 How to Retry

1. **If instance creation fails**:
   ```bash
   # Try again - T4 availability is usually good
   ./gcp_setup.sh
   ```

2. **If connection fails**:
   ```bash
   # Wait 1-2 minutes after creation, then:
   ./connect_single.sh
   ```

3. **To clean up**:
   ```bash
   cd .. && ./cleanup.sh
   ```

## 📂 Files in this folder

- `gcp_setup.sh` - Creates single T4 GPU instance
- `upload_single.sh` - Uploads DDP training script
- `connect_single.sh` - Connects to the instance

## 🎯 Learning Outcomes

This setup teaches you:
1. Complete DDP code structure
2. PyTorch distributed initialization
3. Proper data loading with DistributedSampler
4. Model wrapping with DDP
5. Cleanup and resource management

## ➡️ Next Steps

Once comfortable with single GPU:
1. Try the multi-GPU setup when resources are available
2. Experiment with different model architectures
3. Test with your own datasets
4. Explore other distributed training strategies
