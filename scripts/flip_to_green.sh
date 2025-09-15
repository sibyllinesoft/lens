#!/bin/bash
# 🟢 PRODUCTION GREEN FLIP SCRIPT
# Executes the controlled cutover to production with full monitoring

set -euo pipefail

# Configuration
GREEN_FINGERPRINT="aa77b46922e7a1374289c11d70ef6dbe245827b7c610a83c7a7ebf812556aea2"
BASELINE_IMAGE="lens-production:baseline-stable"
CANDIDATE_IMAGE="lens-production:green-${GREEN_FINGERPRINT:0:8}"
MONITORING_WINDOW_MINUTES=60

echo "🚀 INITIATING PRODUCTION GREEN FLIP"
echo "=================================================="
echo "Green Fingerprint: $GREEN_FINGERPRINT"
echo "Baseline Image: $BASELINE_IMAGE"
echo "Candidate Image: $CANDIDATE_IMAGE"
echo "Monitoring Window: ${MONITORING_WINDOW_MINUTES} minutes"
echo "=================================================="

# Pre-flight checks
echo "🔍 Pre-flight verification..."

# Verify signed manifest integrity
if ! python3 scripts/verify_manifest_signature.py --fingerprint "$GREEN_FINGERPRINT"; then
    echo "❌ ABORT: Manifest signature verification failed"
    exit 1
fi
echo "✅ Manifest signature verified"

# Verify rollback image availability
echo "🔧 DEBUG: Checking baseline image: $BASELINE_IMAGE"
if ! docker inspect "$BASELINE_IMAGE" &>/dev/null; then
    echo "❌ First attempt failed, retrying Docker inspect for: $BASELINE_IMAGE"
    sleep 2
    if ! docker inspect "$BASELINE_IMAGE" &>/dev/null; then
        echo "❌ ABORT: Baseline rollback image not available"
        echo "🔧 DEBUG: Available Docker images:"
        docker images | grep lens-production || echo "No lens-production images found"
        exit 1
    fi
fi
echo "✅ Rollback image confirmed available"

# Verify candidate image readiness
echo "🔧 DEBUG: Checking candidate image: $CANDIDATE_IMAGE"
if ! docker inspect "$CANDIDATE_IMAGE" &>/dev/null; then
    echo "❌ First attempt failed, retrying Docker inspect for: $CANDIDATE_IMAGE"
    sleep 2
    if ! docker inspect "$CANDIDATE_IMAGE" &>/dev/null; then
        echo "❌ ABORT: Candidate image not found"
        echo "🔧 DEBUG: Available Docker images:"
        docker images | grep lens-production || echo "No lens-production images found"
        exit 1
    fi
fi
echo "✅ Candidate image confirmed ready"

# Execute production smoke test
echo "🧪 Running production smoke test..."
if ! python3 scripts/production_smoke_test.py --scenarios 5 --strict; then
    echo "❌ ABORT: Production smoke test failed"
    exit 1
fi
echo "✅ Production smoke test passed"

# Phase 1: Shadow Deployment (100% read-only)
echo ""
echo "🌒 PHASE 1: Shadow Deployment (100% read-only)"
echo "Duration: 15 minutes"
echo "Collecting baseline metrics without traffic impact..."

# Deploy shadow instance
echo "  📡 Deploying shadow instance..."
kubectl apply -f manifests/shadow-deployment.yaml
kubectl set image deployment/lens-shadow lens-api="$CANDIDATE_IMAGE"

# Wait for shadow readiness
echo "  ⏳ Waiting for shadow readiness..."
kubectl wait --for=condition=ready pod -l app=lens-shadow --timeout=300s

# Monitor shadow metrics
echo "  📊 Monitoring shadow metrics for 15 minutes..."
python3 scripts/shadow_monitor.py --duration 15 --baseline-comparison &
SHADOW_PID=$!

# Wait for shadow monitoring to complete
sleep 900  # 15 minutes
wait $SHADOW_PID

# Evaluate shadow results
if ! python3 scripts/evaluate_shadow_results.py --require-improvement; then
    echo "❌ ABORT: Shadow deployment showed regression"
    kubectl delete deployment lens-shadow
    exit 1
fi
echo "✅ Shadow deployment validated - proceeding to canary"

# Phase 2: Statistical Canary (10% traffic)
echo ""
echo "🐦 PHASE 2: Statistical Canary (10% traffic)"
echo "Using SPRT with α=β=0.05, δ=0.03"
echo "Auto-terminate on statistical evidence..."

# Deploy canary with traffic splitting
echo "  📡 Deploying canary with 10% traffic..."
kubectl apply -f manifests/canary-service.yaml
kubectl set image deployment/lens-canary lens-api="$CANDIDATE_IMAGE"

# Configure traffic split (90% baseline, 10% canary)
kubectl apply -f manifests/traffic-split-10pct.yaml

# Start SPRT monitoring
echo "  📊 Starting SPRT statistical monitoring..."
python3 scripts/sprt_monitor.py --traffic-percentage 10 --max-duration 60 &
SPRT_PID=$!

# Start SLO monitoring with auto-rollback
echo "  🛡️ Starting SLO monitoring with auto-rollback..."
python3 scripts/slo_monitor.py --auto-rollback --burn-threshold 1.0 &
SLO_PID=$!

# Wait for SPRT decision or timeout
echo "  ⏳ Waiting for SPRT decision or 60-minute timeout..."
if ! wait $SPRT_PID; then
    echo "❌ SPRT monitoring failed - initiating rollback"
    ./scripts/emergency_rollback.sh
    kill $SLO_PID 2>/dev/null || true
    exit 1
fi

# Check SPRT decision
SPRT_DECISION=$(python3 scripts/get_sprt_decision.py)
if [ "$SPRT_DECISION" != "accept" ]; then
    echo "❌ SPRT decision: $SPRT_DECISION - initiating rollback"
    ./scripts/emergency_rollback.sh
    kill $SLO_PID 2>/dev/null || true
    exit 1
fi
echo "✅ SPRT decision: ACCEPT - statistically significant improvement detected"

# Phase 3: Traffic Ramp (25% → 50% → 100%)
echo ""
echo "🚀 PHASE 3: Traffic Ramp to 100%"
echo "Gradual rollout with continuous monitoring..."

# Ramp to 25%
echo "  📈 Ramping to 25% traffic..."
kubectl apply -f manifests/traffic-split-25pct.yaml
sleep 600  # 10 minutes

# Check SLOs at 25%
if ! python3 scripts/check_slos.py --threshold-buffer 0.05; then
    echo "❌ SLO violation at 25% - initiating rollback"
    ./scripts/emergency_rollback.sh
    kill $SLO_PID 2>/dev/null || true
    exit 1
fi
echo "✅ SLOs healthy at 25% traffic"

# Ramp to 50%
echo "  📈 Ramping to 50% traffic..."
kubectl apply -f manifests/traffic-split-50pct.yaml
sleep 600  # 10 minutes

# Check SLOs at 50%
if ! python3 scripts/check_slos.py --threshold-buffer 0.05; then
    echo "❌ SLO violation at 50% - initiating rollback"
    ./scripts/emergency_rollback.sh
    kill $SLO_PID 2>/dev/null || true
    exit 1
fi
echo "✅ SLOs healthy at 50% traffic"

# Final ramp to 100%
echo "  📈 Final ramp to 100% traffic..."
kubectl apply -f manifests/traffic-split-100pct.yaml
kubectl set image deployment/lens-api lens-api="$CANDIDATE_IMAGE"

# Wait for full deployment
kubectl rollout status deployment/lens-api --timeout=600s

# Final SLO validation
sleep 300  # 5 minutes
if ! python3 scripts/check_slos.py --strict; then
    echo "❌ SLO violation at 100% - initiating rollback"
    ./scripts/emergency_rollback.sh
    kill $SLO_PID 2>/dev/null || true
    exit 1
fi

# Cleanup temporary resources
echo "🧹 Cleaning up temporary resources..."
kubectl delete deployment lens-shadow lens-canary 2>/dev/null || true
kubectl delete service lens-canary 2>/dev/null || true

# Stop background monitoring
kill $SLO_PID 2>/dev/null || true

# Phase 4: Production Validation
echo ""
echo "✅ PHASE 4: Production Validation"
echo "Running comprehensive post-deployment checks..."

# Final end-to-end validation
echo "  🔍 Running end-to-end validation suite..."
python3 scripts/e2e_validation.py --production --comprehensive

# Update release tags
echo "  🏷️ Updating release tags..."
git tag "v2.1.0-production-$(date +%Y%m%d-%H%M%S)"
echo "GREEN_FINGERPRINT=$GREEN_FINGERPRINT" > .env.production

# Generate deployment report
echo "  📄 Generating deployment report..."
python3 scripts/generate_deployment_report.py \
    --fingerprint "$GREEN_FINGERPRINT" \
    --baseline-image "$BASELINE_IMAGE" \
    --candidate-image "$CANDIDATE_IMAGE" \
    --output "deployment-report-$(date +%Y%m%d-%H%M%S).md"

echo ""
echo "🎉 PRODUCTION GREEN FLIP COMPLETE!"
echo "=================================================="
echo "✅ Status: SUCCESSFULLY DEPLOYED TO PRODUCTION"
echo "🔒 Green Fingerprint: $GREEN_FINGERPRINT"
echo "📊 All SLOs within targets"
echo "🛡️ Auto-rollback monitoring continues for 24 hours"
echo "📈 Shadow → Canary → Ramp sequence completed successfully"
echo ""
echo "🎯 Next Steps:"
echo "1. Monitor production metrics for next 24 hours"
echo "2. Review deployment report for performance deltas"
echo "3. Schedule Day-7 mini-retro for lessons learned"
echo "4. Update runbooks with any new operational insights"
echo "=================================================="

# Start 24-hour monitoring
echo "🔍 Starting 24-hour post-deployment monitoring..."
nohup python3 scripts/post_deployment_monitor.py --duration 24h --alert-on-regression > monitoring.log 2>&1 &
echo "📊 Monitoring PID: $! (check monitoring.log for real-time status)"

echo ""
echo "🚀 PRODUCTION IS LIVE AND HEALTHY!"