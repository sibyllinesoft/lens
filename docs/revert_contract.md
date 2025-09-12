# 🚨 Revert Contract - T₀ Baseline Protection

**Baseline Version**: T0-2025-09-12T04:47:39Z  
**Error Budget Period**: 28 days from baseline establishment  
**Auto-Revert Policy**: Two consecutive violation windows trigger immediate revert

---

## 🎯 Revert Thresholds

### Quality Guardrails (Global)
- **nDCG@10**: Must remain ≥ 0.340 (baseline - 0.5pp)
- **SLA-Recall@50**: Must remain ≥ 0.669 (baseline - 0.3pp)
- **Violation Trigger**: Any single measurement below threshold

### Latency Guardrails (SLA Protection)  
- **p95 Latency**: Must remain ≤ 119ms (baseline + 1ms)
- **p99 Latency**: Must remain ≤ 144ms (baseline + 2ms)
- **Violation Trigger**: Any measurement above threshold for >5 minutes

### Calibration Guardrails (CALIB_V22)
- **AECE Score**: Must remain ≤ 0.024 (baseline + 0.01)
- **Cross-language Parity**: |ŷ_rust - ŷ_ts|∞ ≤ 1e-6
- **Violation Trigger**: AECE degradation or parity break

### Safety Guardrails (Operational)
- **File Credit**: Must remain ≤ 5.0% (hard safety limit)
- **Consecutive Window Rule**: 2 violation windows = immediate revert
- **Emergency Override**: Manual revert capability <15 minutes

---

## 📊 Mathematical Revert Logic

Let $B = [nDCG, R@50, p95, p99, AECE, cost]$ be baseline vector with CI half-widths $w$.

**Quality Constraint**:
$$nDCG_{new} - nDCG_B \geq \max(0, -w_{nDCG}) = -0.5\text{pp}$$

**Latency Constraint**: 
$$p95_{new} - p95_B \leq \min(1.0\text{ms}, w_{p95}) = 1.0\text{ms}$$

**Calibration Constraint**:
$$\Delta AECE \leq 0.01$$

**Safety Constraint**:
$$\text{file-credit} \leq 5.0\%$$

**Revert Condition**:
$$\text{violation\_count} \geq 2 \text{ consecutive windows} \Rightarrow \text{AUTO\_REVERT}$$

---

## 🔧 Revert Execution Runbook

### Phase 1: Detection (Automated - 0-2 minutes)
1. **Monitoring Alert**: Gate violation detected in continuous monitoring
2. **Validation Window**: Confirm violation persists for 2 consecutive 5-minute windows
3. **Auto-Revert Trigger**: System automatically initiates revert sequence

### Phase 2: Revert Execution (Automated - 2-10 minutes)
1. **Traffic Ramp Down**: Reduce hero traffic 100% → 50% → 0% (2 minutes each step)
2. **Baseline Restore**: Route all traffic to T₀ baseline configurations  
3. **Health Verification**: Confirm all metrics return within baseline bounds
4. **Alert Escalation**: Page on-call team with revert completion status

### Phase 3: Analysis (Manual - 10-60 minutes)
1. **Root Cause Investigation**: Determine cause of metric degradation
2. **Impact Assessment**: Measure user impact and business metrics
3. **Fix Strategy**: Plan remediation approach for next deployment attempt
4. **Documentation**: Update incident log and lessons learned

### Emergency Manual Revert (Human Override)
```bash
# Emergency revert commands (SRE use only)
./scripts/emergency_revert.sh --baseline T0-2025-09-12T04:47:39Z --reason "manual_override"
./scripts/verify_baseline_restore.sh --check-all-gates
```

---

## 🎛️ Delta Gate System

### CI Whisker Validation
- **Requirement**: All future changes must clear CI whiskers from weekly micro-suites
- **Shadow Traffic Rule**: Changes failing CI whiskers run shadow-only until validated
- **Promotion Gate**: Only changes with positive CI-cleared improvements advance to canary

### Monitoring Windows
- **Short-term**: 5-minute rolling windows for latency and errors
- **Medium-term**: 1-hour rolling windows for quality metrics  
- **Long-term**: 24-hour rolling windows for calibration drift
- **Weekly**: Comprehensive micro-suite validation with bootstrapped CI

### Traffic Mix Monitoring
- **NL vs Lexical Ratio**: Monitor query classification drift
- **Calibration Trigger**: >5% mix shift requires CALIB_V22 spot-check
- **Adaptation Window**: 7-day sliding window for mix analysis

---

## 🚨 Panic Procedures

### Kill-Switch Activation
```bash
# Emergency kill-switch (bypasses all gates)
./scripts/panic_revert.sh --immediate --reason "production_incident"
```

### Escalation Chain
1. **L1**: Automated revert (0-15 minutes)
2. **L2**: SRE on-call engagement (15-30 minutes)  
3. **L3**: Engineering team lead escalation (30-60 minutes)
4. **L4**: Director/VP escalation (>60 minutes for business impact)

### Communication Templates
- **Internal Alert**: "Hero config auto-reverted due to [METRIC] violation. Baseline restored. ETA for investigation: X minutes."
- **Customer Communication**: "Brief search performance issue detected and automatically resolved. No action required."

---

## 📋 Validation Checklist

### Pre-Deployment (Delta Gate)
- [ ] Change clears CI whiskers on weekly micro-suite
- [ ] Shadow traffic validates metric improvements  
- [ ] No calibration drift detected in 7-day window
- [ ] Traffic mix within ±5% of baseline assumptions
- [ ] All safety thresholds respected in simulation

### Post-Deployment (Monitoring)
- [ ] All baseline metrics within error budget bounds
- [ ] No consecutive violation windows detected
- [ ] Latency SLA maintained with safety margin
- [ ] Calibration parity preserved (|ŷ_rust - ŷ_ts|∞ ≤ 1e-6)
- [ ] File credit below 5% safety threshold

### Quarterly Review (Strategic)
- [ ] Error budget consumption analysis
- [ ] Revert frequency and root cause trends
- [ ] Baseline update consideration (new T₁?)
- [ ] Pool refresh cadence optimization

---

**Contract Owner**: Site Reliability Engineering  
**Technical Owner**: Search Engineering Team  
**Business Owner**: Product Management  

**Status**: ✅ ACTIVE - T₀ Baseline Protection Engaged  
**Next Review**: 2025-10-10 (28-day error budget cycle)

---

*This contract establishes mathematical guardrails to protect the T₀ baseline while enabling safe innovation. All changes must respect these bounds or face automatic revert to preserve system reliability.*