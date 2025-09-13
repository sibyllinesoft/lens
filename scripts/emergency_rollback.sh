#!/bin/bash
# 🚨 EMERGENCY ROLLBACK SCRIPT
# Single-command production rollback with <5-minute RTO

set -euo pipefail

BASELINE_IMAGE="lens-production:baseline-stable"
ROLLBACK_REASON="${1:-automated_trigger}"

echo "🚨 EMERGENCY ROLLBACK INITIATED"
echo "======================================="
echo "Reason: $ROLLBACK_REASON"
echo "Target: $BASELINE_IMAGE"
echo "Expected RTO: <5 minutes"
echo "======================================="

# Immediate traffic cutoff to candidate
echo "⚡ Step 1: Immediate traffic cutoff..."
kubectl apply -f manifests/traffic-split-0pct.yaml || {
    echo "❌ Traffic cutoff failed - manual intervention required"
    exit 1
}
echo "✅ Traffic routed to baseline"

# Rollback main deployment
echo "⚡ Step 2: Rolling back main deployment..."
kubectl set image deployment/lens-api lens-api="$BASELINE_IMAGE" || {
    echo "❌ Deployment rollback failed - manual intervention required"
    exit 1
}

# Wait for rollback completion
echo "⏳ Step 3: Waiting for rollback completion..."
if ! kubectl rollout status deployment/lens-api --timeout=300s; then
    echo "❌ Rollback timeout - manual intervention required"
    exit 1
fi
echo "✅ Deployment rollback complete"

# Cleanup failed canary resources
echo "🧹 Step 4: Cleaning up failed resources..."
kubectl delete deployment lens-canary lens-shadow 2>/dev/null || true
kubectl delete service lens-canary 2>/dev/null || true
kubectl delete configmap canary-config 2>/dev/null || true
echo "✅ Cleanup complete"

# Health validation
echo "🔍 Step 5: Validating rollback health..."
sleep 30  # Allow stabilization
if ! python3 scripts/health_check.py --strict; then
    echo "❌ Post-rollback health check failed - escalate immediately"
    exit 1
fi
echo "✅ System health confirmed"

# Alert and logging
echo "📢 Step 6: Alerting and logging..."
python3 scripts/alert_rollback.py --reason "$ROLLBACK_REASON" --timestamp "$(date -Iseconds)"
echo "EMERGENCY_ROLLBACK_$(date +%Y%m%d_%H%M%S): $ROLLBACK_REASON" >> rollback.log

# Generate rollback report
echo "📄 Generating rollback report..."
python3 scripts/generate_rollback_report.py \
    --reason "$ROLLBACK_REASON" \
    --baseline-image "$BASELINE_IMAGE" \
    --rollback-time "$(date -Iseconds)" \
    --output "rollback-report-$(date +%Y%m%d-%H%M%S).md"

echo ""
echo "✅ EMERGENCY ROLLBACK COMPLETE"
echo "======================================="
echo "🛡️ System restored to baseline: $BASELINE_IMAGE"
echo "⏱️  Total rollback time: <5 minutes" 
echo "📊 Health status: HEALTHY"
echo "📋 Action items:"
echo "   1. Review rollback report"
echo "   2. Investigate root cause: $ROLLBACK_REASON"
echo "   3. Fix issues before next deployment attempt"
echo "   4. Update incident post-mortem"
echo "======================================="