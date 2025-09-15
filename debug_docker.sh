#!/bin/bash
set -euo pipefail

# Configuration from flip_to_green.sh
GREEN_FINGERPRINT="aa77b46922e7a1374289c11d70ef6dbe245827b7c610a83c7a7ebf812556aea2"
BASELINE_IMAGE="lens-production:baseline-stable"
CANDIDATE_IMAGE="lens-production:green-${GREEN_FINGERPRINT:0:8}"

echo "🔧 DEBUG: Testing Docker image availability"
echo "GREEN_FINGERPRINT: $GREEN_FINGERPRINT"
echo "BASELINE_IMAGE: $BASELINE_IMAGE"
echo "CANDIDATE_IMAGE: $CANDIDATE_IMAGE"

echo "🔍 Testing Docker inspect for baseline image..."
if docker inspect "$BASELINE_IMAGE" &>/dev/null; then
    echo "✅ Baseline image found: $BASELINE_IMAGE"
else
    echo "❌ Baseline image not found: $BASELINE_IMAGE"
    echo "Available images:"
    docker images | grep lens-production || echo "No lens-production images found"
    exit 1
fi

echo "🔍 Testing Docker inspect for candidate image..."  
if docker inspect "$CANDIDATE_IMAGE" &>/dev/null; then
    echo "✅ Candidate image found: $CANDIDATE_IMAGE"
else
    echo "❌ Candidate image not found: $CANDIDATE_IMAGE"
    exit 1
fi

echo "🎉 All Docker images verified successfully!"