# GCP PyTorch DDP Setup

This folder contains all the scripts and documentation for setting up PyTorch Distributed Data Parallel (DDP) training on Google Cloud Platform.

## 📁 Folder Structure

```
gcp/
├── single-gpu/          # Single GPU setup (working, cost-effective)
├── multi-gpu/           # Multi-GPU setup (requires quota/availability)
├── cleanup.sh           # General cleanup script
├── verify_setup.sh      # Verify GPU and PyTorch setup
├── connect.sh           # Legacy connect script
├── upload.sh            # Legacy upload script
└── README.md            # This file
```

## 🎯 Current Status

- ✅ **Single GPU Setup**: Working and tested
- ❌ **Multi GPU Setup**: Blocked by resource availability on GCP

## 🚀 Quick Start

### For Single GPU (Recommended)
```bash
cd single-gpu
./gcp_setup.sh
```

### For Multi GPU (When resources are available)
```bash
cd multi-gpu
./gcp_setup_multi.sh
```

## 💰 Cost Management

### Check for Running Instances
```bash
gcloud compute instances list
```

### Stop All Instances
```bash
./cleanup.sh
```

### Current Billing Status
- ✅ No instances currently running
- ✅ No ongoing charges
- ✅ Only standard storage costs apply (minimal)

## 📋 Next Steps

1. **Immediate**: Use single GPU setup to learn DDP concepts
2. **Later**: Try multi-GPU when GCP resources become available
3. **Alternative**: Consider other cloud providers or local multi-GPU setup

## 🆘 Support

If you encounter issues:
1. Check the specific README in single-gpu/ or multi-gpu/ folders
2. Ensure no instances are running with `gcloud compute instances list`
3. Use `./cleanup.sh` to clean up any resources

## 🔄 Retry Strategy

Resource availability changes frequently. Try:
- Different times of day
- Different regions
- Different GPU types
- Requesting quota increases
