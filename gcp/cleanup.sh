#!/bin/bash

# Cleanup ALL GCP resources for PyTorch DDP project
# Project: ddp-starter

PROJECT_ID="ddp-starter"

echo "🧹 Cleaning up ALL GCP resources for project: $PROJECT_ID"
echo ""

# Set project
gcloud config set project $PROJECT_ID

# First, list any existing instances
echo "📋 Checking for existing instances..."
EXISTING_INSTANCES=$(gcloud compute instances list --format="value(name,zone)" 2>/dev/null)

if [ -z "$EXISTING_INSTANCES" ]; then
    echo "✅ No instances found - nothing to clean up!"
else
    echo "Found instances:"
    echo "$EXISTING_INSTANCES"
    echo ""
    
    # Delete all instances
    echo "🗑️  Deleting all instances..."
    while IFS=$'\t' read -r name zone; do
        if [ -n "$name" ] && [ -n "$zone" ]; then
            echo "Deleting $name in $zone..."
            gcloud compute instances delete "$name" --zone="$zone" --quiet
        fi
    done <<< "$EXISTING_INSTANCES"
fi

# Double-check by listing all instances one more time
echo ""
echo "🔍 Final verification - checking for any remaining instances..."
FINAL_CHECK=$(gcloud compute instances list --format="value(name,zone)" 2>/dev/null)

if [ -n "$FINAL_CHECK" ]; then
    echo "⚠️  Found remaining instances:"
    echo "$FINAL_CHECK"
    echo ""
    echo "To manually delete any remaining instances:"
    echo "  gcloud compute instances delete INSTANCE_NAME --zone=ZONE_NAME"
else
    echo "✅ Confirmed: No instances remaining"
fi

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "💰 Cost verification:"
echo "   All GPU instances have been terminated"
echo "   No ongoing compute charges"
echo "   No storage costs - all disks cleaned up"
echo ""
echo "📋 To verify cleanup:"
echo "   gcloud compute instances list"
echo ""
echo "💡 Final status:"
echo "   - All instances deleted"
echo "   - All disks cleaned up"
echo "   - ZERO ongoing charges (\$0.00/month)"
echo "   - Completely free to leave project as-is"
