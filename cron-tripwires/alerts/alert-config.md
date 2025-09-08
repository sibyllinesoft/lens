# Lens v2.2 Alert Configuration

## Alert Channels

### Slack Integration
- **Webhook URL:** Set via `SLACK_WEBHOOK_URL` environment variable
- **Channel:** #lens-alerts (recommended)
- **Severity:** P0, P1, P2 alerts
- **Format:** Rich blocks with action buttons

### Email Notifications  
- **Recipients:** Set via `ALERT_EMAIL_RECIPIENTS` environment variable
- **Format:** `eng-alerts@lens.dev,ops-team@lens.dev`
- **Severity:** P0, P1 alerts only
- **Format:** Plain text with action links

### PagerDuty Integration
- **Integration Key:** Set via `PAGERDUTY_INTEGRATION_KEY` environment variable
- **Service:** lens-v22-tripwires
- **Severity:** P0 alerts only (immediate escalation)
- **Deduplication:** Based on tripwire + fingerprint + date

### Discord (Optional)
- **Webhook URL:** Set via `DISCORD_WEBHOOK_URL` environment variable
- **Channel:** #system-alerts  
- **Severity:** All alerts
- **Format:** Embedded messages with color coding

## Alert Severity Levels

### P0 - Critical (Immediate Response)
**Triggers:**
- Flatline sentinels fail (variance < 1e-4 or range < 0.02)
- Pool health degradation (system contribution < 30%)
- Calibration failure (ECE > 0.02 or tail ratio > 2.0)
- Auto-revert system failure

**Response:**
- Immediate Slack notification
- Email to on-call engineer
- PagerDuty escalation
- Auto-revert triggered (if applicable)

**Escalation:** 15 minutes if not acknowledged

### P1 - High (Same Day Response)  
**Triggers:**
- Power discipline violation (N < 800 queries/suite)
- Credit audit failure (span-only usage < 95%)
- Adapter sanity issues (Jaccard similarity > 0.8)

**Response:**
- Slack notification
- Email to engineering team
- No auto-revert (investigation required)

**Escalation:** 1 hour if not acknowledged

### P2 - Medium (Next Business Day)
**Triggers:**
- Minor configuration drift
- Non-critical monitoring issues
- Baseline validation warnings

**Response:**
- Slack notification only
- No email or PagerDuty

**Escalation:** 4 hours if not acknowledged

## Auto-Revert Policy

### Immediate Auto-Revert (P0)
- Flatline behavior detected
- Pool health critical failure  
- Calibration metrics exceed thresholds
- System instability indicators

### Investigation Required (P1)
- Power discipline violations
- Credit audit failures
- Adapter behavior anomalies
- Quality metric boundary cases

### Manual Review (P2)
- Configuration drift
- Baseline inconsistencies
- Performance trend concerns

## Alert Fatigue Prevention

### Deduplication
- Same tripwire + fingerprint + date = single alert
- Repeat failures within 4 hours are suppressed
- Weekly digest for recurring P2 issues

### Noise Reduction
- Baseline validation before alerting
- Confidence interval checks for metric drift
- Statistical significance testing for trend alerts

### Smart Grouping
- Related tripwire failures grouped into single notification
- Cascade failure detection (prevent alert storms)
- Root cause correlation for complex scenarios

## Testing & Validation

### Alert Testing
```bash
# Test all alert channels
node alert-manager.js test

# Send specific alert from file
node alert-manager.js send test-alert.json
```

### Monitoring Health
- Weekly alert channel health checks
- End-to-end delivery verification
- Response time monitoring for critical alerts

### Runbook Integration
- Each alert type has associated runbook link
- Standard operating procedures for common failures
- Escalation procedures for unresolved alerts

Generated: 2025-09-08T16:20:56.592Z  
Ready for: Production deployment and team training
