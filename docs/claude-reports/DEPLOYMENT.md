# Deployment Guide - Production Rollout System

This guide covers the complete production deployment system implementing the TODO.md deployment plan. The system provides automated canary rollouts, online calibration, drift monitoring, and emergency controls.

## 🚀 Quick Start

### Complete Deployment Pipeline

Run the entire TODO.md deployment sequence with a single command:

```bash
# Start complete deployment pipeline
npm run deploy:start

# Or with options
npm run deploy:start -- --skip-bench --accelerated
```

This executes all phases:
1. **Tag + Freeze**: Version artifacts with config fingerprints
2. **Final Bench**: AnchorSmoke + LadderFull validation on pinned dataset  
3. **Canary Rollout**: 3-block deployment (A→B→C) with promotion gates
4. **Online Calibration**: Daily reliability curve updates
5. **Production Monitoring**: CUSUM drift alarms  
6. **Sentinel Activation**: Kill switches and emergency controls

### Monitor Deployment

```bash
# Check overall status
npm run deploy:status

# Monitor specific systems  
npm run deploy:canary     # Canary rollout status
npm run deploy:monitor    # CUSUM alarms and health  
npm run deploy:sentinel   # Sentinel probes and kill switches
```

### Emergency Controls

```bash
# Emergency abort entire pipeline
npm run deploy:abort "Critical issue found"

# Manual canary rollback
npm run deploy -- canary rollback deployment_id "Performance degradation"

# Activate kill switch
npm run deploy -- sentinel killswitch activate zero_results_emergency
```

## 📋 System Architecture

### Deployment Pipeline Overview

```
User Request → CLI → Orchestrator → Phase Execution
    │
    ├─ Tag + Freeze ────────────────── Version Manager
    ├─ Final Bench ─────────────────── Benchmark System + Pinned Dataset
    ├─ Canary A→B→C ────────────────── Rollout System + Traffic Control
    ├─ Online Calibration ──────────── Reliability Curve Updates  
    ├─ Production Monitoring ───────── CUSUM Alarms + Drift Detection
    └─ Sentinel Activation ─────────── Hourly Probes + Kill Switches
```

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Version Manager** | Config fingerprinting | Policy versioning, artifact freezing, rollback |
| **Final Bench System** | Pre-deployment validation | AnchorSmoke + LadderFull on pinned dataset |
| **Canary Rollout** | Graduated deployment | Block A→B→C with automatic promotion/rollback |
| **Online Calibration** | Runtime optimization | Daily τ updates, 5±2 results/query target |
| **Production Monitoring** | Drift detection | CUSUM alarms, coverage tracking |
| **Sentinel System** | Emergency controls | Hourly probes, kill switches |

## 🎯 Phase-by-Phase Guide

### Phase 1: Tag + Freeze

Creates versioned deployment artifacts with complete configuration fingerprints.

```bash
# Manual version creation (usually automatic)
npm run deploy -- version create --baseline-override baseline.json
npm run deploy -- version list
npm run deploy -- version show 1.2.3
```

**Artifacts Created:**
- `config_fingerprint_1.2.3.json` - Complete configuration snapshot
- Baseline metrics with performance targets  
- LTR model hash and feature schema
- Git commit SHA and build metadata

### Phase 2: Final Bench Validation

Validates deployment against pinned dataset with comprehensive gates.

```bash
# Run final benchmark
npm run deploy:bench

# Detailed validation
npm run deploy -- bench run --version 1.2.3
```

**Validation Gates:**
- ΔnDCG@10 ≥ 0 (no regression)
- Recall@50 Δ ≥ 0 (maintained or improved)  
- P95 latency ≤ +10% vs baseline
- P99/P95 ratio ≤ 2.0
- Span coverage = 100%

**Outputs:**
- `final_bench_results.json` - Complete metrics
- `sign_off_summary.md` - Stakeholder approval document
- `metrics.csv` - Analysis-ready data

### Phase 3: Canary Rollout (Blocks A→B→C)

Graduated rollout with automatic promotion based on success gates.

```bash
# Check canary status
npm run deploy:canary

# Manual rollback if needed  
npm run deploy -- canary rollback deployment_123 "High error rate detected"
```

**Block A: Early-Exit Optimization**
- Traffic: 5% → 25% → 100%
- Feature: Margin=0.12, min_probes=96
- Gates: 24h validation at each stage

**Block B: Dynamic TopN Calibration**  
- Traffic: 5% → 25% → 100%
- Feature: τ optimization, reliability curves
- Gates: Results/query target maintenance

**Block C: Gentle Deduplication**
- Traffic: 5% → 25% → 100%  
- Feature: Simhash k=5, hamming_max=2, keep=3
- Gates: Recall preservation validation

**Automatic Rollback Triggers:**
- Hard negative leakage >1.0% absolute
- Results/query drift >±1 from target envelope
- P99 latency >2×P95 sustained
- CUSUM alarms active >24 hours

### Phase 4: Online Calibration

Continuous reliability curve updates with feedback loop prevention.

```bash
# Check calibration status
npm run deploy -- monitor calibration
```

**Daily Process:**
1. Collect canary clicks/impressions  
2. Recompute reliability diagram
3. Re-solve τ = argmin |E[1{p≥τ}]−5| (5±2 target)
4. Apply update after 2-day holdout period

**Safety Mechanisms:**
- Feature drift monitoring (>3σ triggers LTR fallback)
- Isotonic regression as final calibration layer
- Manual override capabilities for emergencies

### Phase 5: Production Monitoring

CUSUM-based drift detection with smart alerting.

```bash
# Monitor system health
npm run deploy:monitor

# Reset false alarm
npm run deploy -- monitor reset anchor_p_at_1
```

**Monitored Metrics:**
- **Anchor P@1**: CUSUM detection for precision drift
- **Recall@50**: Availability regression detection  
- **Ladder Positives**: Candidate quality monitoring
- **LSIF/Tree-sitter Coverage**: Parse success rates

**Alert Conditions:**
- Sustained deviation >24 hours triggers escalation
- Page on critical thresholds (P99 >5s, error rate >1%)
- Email on drift detection, webhook on emergencies

### Phase 6: Sentinel Activation

Automated health validation with kill switch integration.

```bash
# Check sentinel status
npm run deploy:sentinel

# Manual probe execution
npm run deploy -- sentinel probe class_probe

# Emergency kill switch
npm run deploy -- sentinel killswitch activate manual_emergency --reason "Critical bug"
```

**Sentinel Probes (Hourly):**
- `class` query must return results (zero-result detection)
- `def` query must return results (basic functionality)  
- General search health validation

**Kill Switch Actions:**
- **Zero Results Emergency**: Route to basic search fallback
- **System Failure**: Automatic rollback to previous version
- **Manual Emergency**: Disable advanced features, static responses

**Recovery Conditions:**
- Automatic recovery when probes pass for 30+ minutes
- Manual override available for emergency situations

## 📊 Monitoring and Dashboards

### Real-time Status

```bash
# Complete system overview
npm run deploy:status --detailed

# JSON output for dashboards  
npm run deploy:status --json | jq .
```

### Key Metrics Dashboard

The system tracks critical metrics for immediate visibility:

**Core Search Metrics:**
- P@1: ≥75% target (currently tracking at baseline)
- nDCG@10: +5-8pts improvement from optimization 
- Recall@50: Baseline maintenance (no regression)
- Results/query: 5±2 target range

**System Health:**
- Span coverage: 100% requirement
- CUSUM alarm status: Quiet period validation
- Sentinel success rate: >95% expected
- Kill switch status: Emergency readiness

**Performance Monitoring:**  
- P95 latency: ≤110% baseline (with 10% buffer)
- P99/P95 ratio: ≤2.0 (tail latency control)
- Error rate: <1% threshold
- QPS handling: Load capacity validation

### Alert Integration

The system integrates with external alerting:

```bash
# Configure webhook alerts (example)
curl -X POST http://localhost:3000/alerts/webhook \
  -d '{"url": "https://alerts.company.com/lens-deploy"}' \
  -H "Content-Type: application/json"
```

**Alert Severity Levels:**
- **INFO**: Routine calibration updates, successful promotions
- **WARN**: Gate delays, minor drift detected  
- **HIGH**: CUSUM alarms, canary rollbacks
- **CRITICAL**: Kill switch activations, system failures

## 🔧 Configuration Management

### Deployment Configuration

The system supports extensive configuration customization:

```javascript
// deployment-config.json
{
  "target_version": "1.2.3",
  "skip_final_bench": false,
  "required_gate_success": true,
  
  "canary_config": {
    "accelerated_rollout": false,
    "stage_duration_hours": 24,
    "manual_promotion": false
  },
  
  "monitoring_config": {
    "enable_cusum_alarms": true,
    "enable_drift_detection": true,
    "alert_webhooks": ["https://alerts.example.com"]
  },
  
  "sentinel_config": {
    "probe_frequency_minutes": 60,
    "kill_switch_enabled": true
  }
}
```

### Environment-Specific Settings

```bash
# Development (accelerated timelines)
npm run deploy:start -- --accelerated --skip-bench

# Staging (full validation) 
npm run deploy:start -- --force  # Override gate failures

# Production (maximum safety)
npm run deploy:start  # All safety checks enabled
```

## 🚨 Emergency Procedures

### Deployment Failures

When deployment fails at any phase:

1. **Immediate Actions:**
   ```bash
   # Get failure details
   npm run deploy:status --detailed
   
   # Review failure logs
   cat deployment-artifacts/orchestrator/failure_report_*.json
   ```

2. **Recovery Options:**
   ```bash
   # Emergency abort (triggers cleanup)
   npm run deploy:abort "Detailed failure reason"
   
   # Manual rollback if canary was started
   npm run deploy -- canary rollback deployment_id "Failure reason"
   ```

3. **Post-Incident:**
   - Review failure report in `deployment-artifacts/orchestrator/`
   - Check system health: `npm run deploy:status`
   - Validate cleanup: `npm run deploy:monitor`

### Production Incidents

For live production issues:

1. **Kill Switch Activation:**
   ```bash
   # Immediate traffic cutoff
   npm run deploy -- sentinel killswitch activate manual_emergency \
     --reason "Critical performance degradation"
   ```

2. **Selective Feature Disable:**
   ```bash
   # Disable specific features
   npm run deploy -- sentinel killswitch activate search_system_failure
   ```

3. **System Rollback:**
   ```bash
   # Complete system rollback
   npm run deploy:abort "Production incident - rolling back"
   ```

### Drift Detection Response

When CUSUM alarms trigger:

1. **Investigate Cause:**
   ```bash
   # Check specific metrics
   npm run deploy -- monitor status
   
   # Review recent changes
   npm run deploy -- version list
   ```

2. **Mitigation Options:**
   ```bash  
   # Reset false alarms
   npm run deploy -- monitor reset metric_name
   
   # Activate LTR fallback
   # (Automatic when feature drift >3σ)
   ```

3. **Long-term Resolution:**
   - Recalibrate baselines if environment changed
   - Update feature schemas if code evolution detected
   - Retrain models if systematic drift confirmed

## 📚 Troubleshooting Guide

### Common Issues

**Deployment Stuck in Phase:**
```bash
# Check phase status
npm run deploy:status --detailed

# Look for specific errors
cat deployment-artifacts/orchestrator/current_pipeline.json | jq .phase_history
```

**Canary Not Promoting:**
```bash
# Check promotion gates
npm run deploy:canary

# Review metrics that are failing
npm run deploy:monitor
```

**Sentinel Probes Failing:**
```bash
# Execute probes manually
npm run deploy -- sentinel probe class_probe
npm run deploy -- sentinel probe def_probe

# Check search system health
npm run deploy -- monitor status
```

**CUSUM False Alarms:**
```bash
# Identify noisy metrics
npm run deploy -- monitor status

# Reset specific detectors
npm run deploy -- monitor reset metric_name

# Review baseline accuracy
npm run deploy -- version show current
```

### Log Locations

All system components maintain detailed logs:

- **Orchestrator**: `./deployment-artifacts/orchestrator/`
- **Benchmarks**: `./deployment-artifacts/benchmarks/`  
- **Canary**: `./deployment-artifacts/canary/`
- **Monitoring**: `./deployment-artifacts/monitoring/`
- **Sentinel**: `./deployment-artifacts/sentinel/`
- **Calibration**: `./deployment-artifacts/calibration/`

### Debug Commands

```bash
# Enable verbose logging
DEBUG=* npm run deploy:start

# Dry run mode (planning only)  
npm run deploy:start -- --dry-run

# Skip problematic phases for testing
npm run deploy:start -- --skip-bench --skip-canary

# Force override safety checks
npm run deploy:start -- --force
```

## 🔄 Advanced Usage

### Custom Deployment Flows

For specialized deployment needs:

```bash
# Benchmark-only validation
npm run deploy:bench

# Canary-only rollout (existing version)
npm run deploy -- canary start --version 1.2.3

# Monitoring activation (post-deployment)
npm run deploy -- monitor start
```

### API Integration

The deployment system exposes REST APIs:

```bash
# Start deployment via API
curl -X POST http://localhost:3000/deploy/start \
  -d '{"version": "1.2.3", "config": {...}}' \
  -H "Content-Type: application/json"

# Monitor progress  
curl -s http://localhost:3000/deploy/status | jq .

# Emergency controls
curl -X POST http://localhost:3000/deploy/abort \
  -d '{"reason": "Emergency stop"}' \
  -H "Content-Type: application/json"
```

### Scheduled Deployments

For automated deployment workflows:

```bash
# Cron job example (daily deployment check)
0 2 * * * cd /path/to/lens && npm run deploy:status --json > /tmp/deploy-status.json

# Weekly benchmark validation
0 0 * * 0 cd /path/to/lens && npm run deploy:bench >> /var/log/lens-bench.log 2>&1
```

---

## 📖 Additional Resources

- **TODO.md**: Original deployment requirements specification
- **Architecture Documentation**: `./docs/architecture/`
- **API Reference**: `./docs/api/deployment-endpoints.md`
- **Runbook**: `./docs/operations/deployment-runbook.md`
- **Troubleshooting**: `./docs/troubleshooting/deployment-issues.md`

## 🎉 Success Criteria

A deployment is considered successful when:

✅ All 9 pipeline phases complete without errors  
✅ Canary rollout reaches 100% traffic (all 3 blocks)  
✅ Online calibration maintains 5±2 results/query target  
✅ CUSUM alarms remain quiet for 24+ hours  
✅ Sentinel probes pass with >95% success rate  
✅ Kill switches are inactive (ready but not triggered)  
✅ Performance metrics within baseline tolerances  
✅ No production incidents or emergency rollbacks

**Expected Timeline:**
- **Immediate**: Tag+Freeze, Final Bench (< 1 hour)
- **Short-term**: Block A canary completion (24-48 hours)
- **Medium-term**: Full canary rollout (3-7 days)  
- **Long-term**: Stable production operation (ongoing)

The system is designed for continuous operation with minimal manual intervention while maintaining the highest levels of safety and observability.