# Lens v1.0 Operational Runbook - Phase D Production

## 🚀 Overview

This runbook provides operational procedures for lens v1.0 production deployment, monitoring, and incident response during Phase D rollout and steady-state operations.

## 📊 Phase D Rollout Status

### Current Release: v1.0.0-rc.1 → v1.0.0

**Canary Deployment Stages:**
- 🟡 **Stage 1**: 5% traffic (RC validation)  
- 🟡 **Stage 2**: 25% traffic (Extended validation)
- 🟢 **Stage 3**: 100% traffic (Full production)

**Quality Gates Status:**
- ✅ SemVer compliance
- ✅ Compatibility check passed
- ✅ UPGRADE.md documentation
- ✅ Security artifacts (SBOM/SAST clean)
- ✅ Performance SLAs met
- ✅ Three-night validation completed

---

## 🎯 Service Level Objectives (SLOs)

### Performance SLAs
```yaml
Stage-A (Lexical):
  - p95_latency: ≤5ms
  - p99_latency: ≤10ms (≤2× p95)
  - throughput: ≥1000 RPS

Stage-B (Symbol/AST):  
  - p95_latency: ≤300ms
  - LSIF_coverage: ≥95%
  - cache_hit_rate: ≥80%

Stage-C (Rerank):
  - p95_latency: ≤300ms  
  - semantic_gating_rate: ≥70%
  - confidence_cutoff_rate: ≥12%

End-to-End:
  - p95_latency: ≤+10% vs baseline
  - error_rate: ≤5%
  - uptime: ≥99.9%
```

### Quality SLAs
```yaml
Search Quality:
  - span_coverage: ≥98%
  - recall_at_50: ≥baseline (0.85)
  - nDCG@10_improvement: ≥+2% or unchanged with perf win
  - consistency_violations: 0

System Quality:
  - test_coverage: ≥90%
  - security_vulnerabilities: 0 critical
  - documentation_coverage: ≥85%
```

---

## 📈 Monitoring & Dashboards

### Primary Dashboards

1. **Phase D Production Dashboard**: `/monitoring/dashboard`
   - Real-time performance metrics per stage
   - Quality gates status
   - Canary deployment progress
   - Alert status and history

2. **Canary Deployment Dashboard**: `/canary/status`
   - Traffic percentage distribution
   - Kill switch status
   - Progressive rollout stages
   - Error rates and rollback events

3. **Three-Night Validation Dashboard**: `/validation/status`
   - Nightly validation results
   - Consecutive pass count
   - Sign-off eligibility status
   - Quality trends over time

### Key Metrics to Monitor

**Performance Metrics:**
```bash
# Stage-A latency monitoring
GET /monitoring/dashboard | jq '.dashboard_state.metrics.performance.stageA.p95_latency_ms'

# Tail latency compliance check
GET /monitoring/dashboard | jq '.dashboard_state.sla_compliance.tail_latency_compliant'

# Overall health status
GET /monitoring/dashboard | jq '.dashboard_state.health.status'
```

**Quality Metrics:**
```bash
# Span coverage check
GET /validation/quality-gates | jq '.quality_gates_report.gates[] | select(.gate == "span_coverage")'

# Three-night validation status
GET /validation/status | jq '.validation_status.promotion_ready'
```

---

## 🚨 Alert Thresholds & Response

### Critical Alerts (Immediate Response Required)

| Alert | Threshold | Response Time | Escalation |
|-------|-----------|---------------|------------|
| Stage-A p95 > 5ms | >5ms | <5 min | On-call engineer |
| Tail latency violation | p99 > 2×p95 | <5 min | On-call engineer |
| Span coverage drop | <98% | <10 min | Platform team |
| Error rate spike | >5% | <5 min | On-call + Product |
| Kill switch triggered | Any activation | <2 min | All teams |

### Warning Alerts (Response within SLA)

| Alert | Threshold | Response Time | Owner |
|-------|-----------|---------------|-------|
| LSIF coverage regression | 5% drop | <30 min | Platform team |
| Cache hit rate low | <80% | <1 hour | Platform team |
| Quality gate failure | Any failure | <15 min | Platform team |
| Canary error rate | >3% | <10 min | On-call engineer |

---

## 🛠 Incident Response Procedures

### Kill Switch Activation

**When to Activate:**
- Critical performance regression (Stage-A p95 >10ms)
- Quality degradation (span coverage <95%)
- High error rates (>10%)
- Security incidents
- System instability

**Activation Steps:**
```bash
# 1. Immediate kill switch activation
curl -X POST http://localhost:3000/canary/killswitch \
  -H "Content-Type: application/json" \
  -d '{"reason": "Performance regression - Stage-A p95 exceeds 10ms"}'

# 2. Verify traffic rollback
curl http://localhost:3000/canary/status | jq '.canary_deployment.trafficPercentage'
# Expected: 0

# 3. Monitor system recovery
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.health.status'
```

**Post-Activation:**
1. Create incident report in GitHub Issues
2. Notify all stakeholders via Slack/Email
3. Begin root cause analysis
4. Plan remediation strategy
5. Document lessons learned

### Progressive Rollout Management

**Normal Progression (5% → 25% → 100%):**
```bash
# Check current status
curl http://localhost:3000/canary/status

# Progress to next stage (only if metrics are healthy)
curl -X POST http://localhost:3000/canary/progress

# Verify progression success
curl http://localhost:3000/canary/status | jq '.canary_deployment.nextStage'
```

**Rollback Procedure:**
```bash
# Immediate rollback to previous stage
curl -X POST http://localhost:3000/canary/killswitch \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual rollback - quality concerns"}'

# Verify rollback
curl http://localhost:3000/canary/status | jq '.canary_deployment'
```

### Quality Gate Failures

**Response to Quality Gate Failure:**
```bash
# 1. Run immediate quality assessment  
curl -X POST http://localhost:3000/validation/quality-gates

# 2. Check specific failed gates
curl http://localhost:3000/validation/quality-gates | jq '.quality_gates_report.blocking_issues'

# 3. Generate recommendations
curl http://localhost:3000/validation/quality-gates | jq '.quality_gates_report.recommendations'
```

**Common Quality Issues & Solutions:**

| Issue | Symptoms | Immediate Action | Long-term Fix |
|-------|----------|------------------|---------------|
| Span coverage drop | <98% coverage | Investigate indexing completeness | Review parser updates |
| Latency regression | p95 >5ms Stage-A | Enable native scanner flag | Optimize lexical processing |
| Consistency violations | Inconsistent results | Review recent model changes | Revalidate training data |
| LSIF coverage drop | Symbol resolution issues | Check language server health | Update language definitions |

---

## 🔧 Troubleshooting Guide

### Performance Issues

**Stage-A Latency (>5ms p95)**
```bash
# Check current latency metrics
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.metrics.performance.stageA'

# Enable native SIMD scanner (emergency optimization)
# This requires feature flag adjustment - contact platform team

# Check early termination rates
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.metrics.performance.stageA.early_termination_rate'
```

**Stage-B Symbol Processing Issues**
```bash
# Check LSIF coverage
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.metrics.performance.stageB.lsif_coverage_percent'

# Verify cache hit rates
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.metrics.performance.stageB.lru_cache_hit_rate'

# Pattern compilation time check
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.metrics.performance.stageB.pattern_compile_time_ms'
```

**Stage-C Reranking Problems**
```bash
# Check semantic gating rates
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.metrics.performance.stageC.semantic_gating_rate'

# Confidence cutoff validation
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.metrics.performance.stageC.confidence_cutoff_rate'
```

### System Health Checks

**Overall Health Status:**
```bash
# Comprehensive health check
curl http://localhost:3000/health | jq '.'

# Dashboard health summary
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.health'

# Active alerts summary
curl http://localhost:3000/monitoring/dashboard | jq '.dashboard_state.recent_alerts'
```

**Compatibility Verification:**
```bash
# API compatibility check
curl "http://localhost:3000/compat/check?api_version=v1&index_version=v1"

# Bundle compatibility validation
curl "http://localhost:3000/compat/bundles?allow_compat=false"
```

---

## 👥 On-Call Team Structure

### Primary On-Call Rotation

**Platform Team** (Primary)
- **Role**: System reliability, performance issues, feature flag management
- **Contact**: @platform-team
- **Escalation**: 15 minutes for critical alerts

**Security Team** (Secondary)  
- **Role**: Security incidents, vulnerability management, compliance
- **Contact**: @security-team
- **Escalation**: 30 minutes for security alerts

**Product Team** (Tertiary)
- **Role**: Quality regressions, user experience issues, rollback decisions
- **Contact**: @product-team
- **Escalation**: 1 hour for quality alerts

### Escalation Matrix

```
Critical Alert → Primary On-Call (5 min) → Secondary (15 min) → Management (30 min) → All Hands (1 hour)
```

### Communication Channels

- **Slack**: `#lens-production-alerts` (immediate alerts)
- **PagerDuty**: Critical alert escalation
- **Email**: Weekly summary reports
- **GitHub Issues**: Incident tracking and post-mortems

---

## 📚 Key Commands Reference

### Production Deployment
```bash
# Cut new RC
lens cut-rc --version 1.0.0-rc.1

# Run compatibility drill  
lens compat-drill --version 1.0.0-rc.1

# Execute nightly validation
lens nightly-validation --duration 120

# Check sign-off status
lens check-signoff --version 1.0.0-rc.1

# Promote to production (requires sign-off)
lens promote --version 1.0.0-rc.1
```

### Monitoring & Status
```bash
# Real-time dashboard
curl http://localhost:3000/monitoring/dashboard | jq '.'

# Canary deployment status
curl http://localhost:3000/canary/status | jq '.'

# Quality gates validation
curl -X POST http://localhost:3000/validation/quality-gates | jq '.'

# Three-night validation status
curl http://localhost:3000/validation/status | jq '.'

# Sign-off report generation
curl http://localhost:3000/validation/signoff-report | jq '.'
```

### Emergency Procedures
```bash
# Kill switch activation
curl -X POST http://localhost:3000/canary/killswitch \
  -d '{"reason": "Emergency: Critical performance regression"}'

# Canary rollout progression
curl -X POST http://localhost:3000/canary/progress

# Force nightly validation (testing)
curl -X POST http://localhost:3000/validation/nightly \
  -d '{"force_night": 1, "duration_minutes": 60}'
```

---

## 📋 Pre-Production Checklist

### Before RC Cut
- [ ] All quality gates passing
- [ ] Security scans completed (SBOM/SAST)
- [ ] Documentation updated (UPGRADE.md)
- [ ] Feature flags configured (kill switches enabled)
- [ ] Monitoring dashboards operational
- [ ] On-call rotation confirmed

### Before Canary Rollout
- [ ] RC compatibility validated
- [ ] Performance baselines established
- [ ] Alert thresholds configured
- [ ] Kill switch procedures tested
- [ ] Stakeholder notifications sent

### Before Production Promotion
- [ ] Three consecutive nights validation passed
- [ ] All stakeholder sign-offs obtained
- [ ] Emergency procedures validated
- [ ] Rollback plans confirmed
- [ ] Documentation finalized

---

## 🎯 Success Metrics

### Deployment Success Criteria
- **Quality Gates**: 100% pass rate on all critical gates
- **Performance**: All SLAs met during rollout
- **Reliability**: 99.9%+ uptime during transition  
- **Quality**: No regression in search quality metrics
- **Security**: Zero critical vulnerabilities
- **Monitoring**: <5% false positive alert rate

### Post-Deployment Validation
- **7-day health check**: Monitor all metrics for stability
- **Performance trend analysis**: Validate sustained improvements  
- **Quality metrics validation**: Confirm nDCG@10 improvements
- **User feedback collection**: Product team monitors satisfaction
- **Incident analysis**: Review any issues and document learnings

---

## 📞 Emergency Contacts

**Critical Issues (24/7)**
- On-call Engineer: @platform-on-call
- Security Team: @security-on-call  
- Management: @engineering-leadership

**Business Hours Support**
- Platform Team: @platform-team
- Product Team: @product-team
- DevOps Team: @devops-team

**Vendor Support**
- Cloud Provider: [Support Case Portal]
- Monitoring Service: [Vendor Support]
- Security Scanner: [Vendor Support]

---

*Last Updated: Phase D Implementation*  
*Next Review: Post-v1.0 GA Release*