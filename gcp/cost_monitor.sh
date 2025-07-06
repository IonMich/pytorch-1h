#!/bin/bash

# Cost monitoring script for GCP PyTorch DDP project
# Project: ddp-starter

PROJECT_ID="ddp-starter"

echo "💰 GCP Cost Monitoring for PyTorch DDP Project"
echo "Project: $PROJECT_ID"
echo ""

# Set project
gcloud config set project $PROJECT_ID

echo "📋 Current Running Instances:"
INSTANCES=$(gcloud compute instances list --format="table(name,zone,status,machineType.basename(),guestAccelerators[0].acceleratorType.basename(),guestAccelerators[0].acceleratorCount,creationTimestamp)" 2>/dev/null)

if [ -z "$INSTANCES" ] || [ "$INSTANCES" = "Listed 0 items." ]; then
    echo "✅ No instances running - No compute charges!"
else
    echo "$INSTANCES"
    echo ""
    echo "⚠️  WARNING: You have running instances that are being charged!"
    echo "   Use './cleanup.sh' to stop all instances"
fi

echo ""
echo "🗄️  Storage Resources:"
echo "Persistent Disks:"
DISKS=$(gcloud compute disks list --format="table(name,zone,sizeGb,type,status)" 2>/dev/null)
if [ -z "$DISKS" ] || echo "$DISKS" | grep -q "Listed 0 items"; then
    echo "✅ No persistent disks - No storage charges!"
else
    echo "$DISKS"
fi

echo ""
echo "Custom Images:"
IMAGES=$(gcloud compute images list --no-standard-images --format="table(name,diskSizeGb,status)" 2>/dev/null)
if [ -z "$IMAGES" ] || echo "$IMAGES" | grep -q "Listed 0 items"; then
    echo "✅ No custom images - No image storage charges!"
else
    echo "$IMAGES"
fi

echo ""
echo "Snapshots:"
SNAPSHOTS=$(gcloud compute snapshots list --format="table(name,diskSizeGb,status)" 2>/dev/null)
if [ -z "$SNAPSHOTS" ] || echo "$SNAPSHOTS" | grep -q "Listed 0 items"; then
    echo "✅ No snapshots - No snapshot charges!"
else
    echo "$SNAPSHOTS"
fi

echo ""
echo "🔍 Resource Summary:"

# Count instances and storage
RUNNING_COUNT=$(gcloud compute instances list --format="value(name)" | wc -l | tr -d ' ')
DISK_COUNT=$(gcloud compute disks list --format="value(name)" | wc -l | tr -d ' ')
IMAGE_COUNT=$(gcloud compute images list --no-standard-images --format="value(name)" | wc -l | tr -d ' ')
SNAPSHOT_COUNT=$(gcloud compute snapshots list --format="value(name)" | wc -l | tr -d ' ')

echo "   Total instances: $RUNNING_COUNT"
echo "   Persistent disks: $DISK_COUNT"
echo "   Custom images: $IMAGE_COUNT"
echo "   Snapshots: $SNAPSHOT_COUNT"

echo ""
echo "💸 Estimated Current Costs:"
if [ "$RUNNING_COUNT" -eq 0 ]; then
    echo "   ✅ Compute: \$0.00/hour (no instances)"
else
    echo "   ⚠️  Compute: ~\$0.35-2.50/hour (depending on GPU type)"
fi

TOTAL_STORAGE=$((DISK_COUNT + IMAGE_COUNT + SNAPSHOT_COUNT))
if [ "$TOTAL_STORAGE" -eq 0 ]; then
    echo "   ✅ Storage: \$0.00/month (no storage resources)"
    echo "   🎉 TOTAL CURRENT COST: \$0.00"
else
    echo "   ⚠️  Storage: ~\$0.04-0.20/month (depending on storage size)"
fi

echo ""
echo "🎯 Cost Optimization Tips:"
echo "   1. Stop instances when not in use (preserves disks)"
echo "   2. Delete instances completely when done (removes everything)"
echo "   3. Use preemptible instances for experimentation (-70% cost)"
echo "   4. Clean up unused disks"

echo ""
echo "🚨 To stop all charges immediately:"
echo "   ./cleanup.sh"

echo ""
echo "📊 For detailed billing:"
echo "   Visit: https://console.cloud.google.com/billing"
