# Weekly Cron Tripwires - Setup Guide

## Overview

The Weekly Cron Tripwires system provides continuous validation of Lens v2.2 baseline performance with automatic drift detection and revert capabilities.

## Installation

### 1. Install Cron Job

```bash
# Make installation script executable
chmod +x ./cron-tripwires/scripts/install-cron.sh

# Install the weekly cron job
./cron-tripwires/scripts/install-cron.sh

# Verify installation
crontab -l | grep weekly-validation
```

### 2. Setup Baseline Configuration

```bash
# Capture current configuration as baseline
node ./cron-tripwires/scripts/baseline-manager.js capture v22_1f3db391_1757345166574

# Verify baseline was captured
node ./cron-tripwires/scripts/baseline-manager.js list
```

### 3. Configure Alert Channels

```bash
# Set environment variables for alerting
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
export ALERT_EMAIL_RECIPIENTS="ops-team@lens.dev,eng-alerts@lens.dev"
export PAGERDUTY_INTEGRATION_KEY="your-pagerduty-integration-key"

# Test alert configuration
node ./cron-tripwires/alerts/alert-manager.js test
```

### 4. Validate System Health

```bash
# Run manual tripwire validation
node ./cron-tripwires/scripts/validate-weekly-tripwires.js --baseline v22_1f3db391_1757345166574

# Test auto-revert system (safe - uses test mode)
node ./cron-tripwires/scripts/auto-revert.js "Manual test of revert system"
```

## Configuration

### Tripwire Thresholds

The system monitors these standing tripwires:

| Tripwire | Threshold | Severity | Auto-Revert |
|----------|-----------|----------|-------------|
| Flatline Variance | Var(nDCG@10) > 1e-4 | P0 | Yes |
| Flatline Range | Range ≥ 0.02 | P0 | Yes |
| Pool Contribution | ≥30% per system | P0 | Yes |
| Credit Mode | Span-only ≥95% | P1 | No |
| Adapter Sanity | Jaccard < 0.8 median | P1 | No |
| Power Discipline | N ≥ 800/suite | P1 | No |
| CI Width | ≤ 0.03 | P1 | No |
| Max Slice ECE | ≤ 0.02 | P0 | Yes |
| Tail Ratio | p99/p95 ≤ 2.0 | P0 | Yes |

### Cron Schedule

- **Default:** Sundays at 02:00 local time
- **Frequency:** Weekly
- **Duration:** ~15-30 minutes per run
- **Logs:** Stored in `./cron-tripwires/logs/`

## Operations

### Manual Execution

```bash
# Run weekly validation immediately
./cron-tripwires/scripts/weekly-validation.sh

# Check specific baseline
node ./cron-tripwires/scripts/validate-weekly-tripwires.js --baseline v22_custom_fingerprint
```

### Log Management

```bash
# View recent validation logs
ls -la ./cron-tripwires/logs/cron-validation-*.log

# View P0 alerts
ls -la ./cron-tripwires/alerts/p0-alert-*.json

# Cleanup old logs (automatic, but can run manually)
find ./cron-tripwires/logs/ -name "*.log" -mtime +30 -delete
```

### Baseline Management

```bash
# List available baselines
node ./cron-tripwires/scripts/baseline-manager.js list

# Validate baseline integrity
node ./cron-tripwires/scripts/baseline-manager.js validate v22_1f3db391_1757345166574

# Capture new baseline after intentional changes
node ./cron-tripwires/scripts/baseline-manager.js capture v22_new_fingerprint
```

## Monitoring & Alerts

### Alert Channels

1. **Slack:** Real-time notifications with action buttons
2. **Email:** Summary reports for P0/P1 alerts
3. **PagerDuty:** Immediate escalation for P0 issues
4. **GitHub Issues:** Automatic issue creation for failures

### Alert Response

#### P0 Alerts (Critical)
- **Response Time:** Immediate (< 5 minutes)
- **Auto Actions:** Automatic revert triggered
- **Escalation:** PagerDuty → On-call engineer
- **Follow-up:** Root cause analysis within 24 hours

#### P1 Alerts (High)  
- **Response Time:** Same day (< 4 hours)
- **Auto Actions:** Investigation required, no revert
- **Escalation:** Email → Engineering team
- **Follow-up:** Fix within 2 business days

### Runbook Links

- **Flatline Failure:** [Internal Runbook Link]
- **Pool Health Issues:** [Internal Runbook Link]
- **Auto-Revert Failed:** [Internal Runbook Link]
- **Calibration Drift:** [Internal Runbook Link]

## GitHub Actions Integration

### Workflow Setup

```bash
# Copy GitHub Actions workflows to repository
cp ./cron-tripwires/github-actions/*.yml ./.github/workflows/

# Commit and push to enable
git add .github/workflows/
git commit -m "Add weekly tripwire validation workflow"
git push
```

### Environment Secrets

Configure these secrets in GitHub repository settings:

- `SLACK_WEBHOOK_URL`: Slack webhook for notifications
- `PAGERDUTY_INTEGRATION_KEY`: PagerDuty integration key
- `ALERT_EMAIL_RECIPIENTS`: Email addresses for alerts

## Troubleshooting

### Common Issues

#### Cron Job Not Running
```bash
# Check cron service status
systemctl status cron

# Verify cron entry exists
crontab -l | grep weekly-validation

# Check cron logs
grep CRON /var/log/syslog | tail -20
```

#### Baseline Not Found
```bash
# Recapture baseline
node ./cron-tripwires/scripts/baseline-manager.js capture v22_1f3db391_1757345166574

# Verify baseline files
ls -la ./cron-tripwires/baselines/
```

#### Auto-Revert Failed
```bash
# Check auto-revert logs
ls -la ./cron-tripwires/logs/auto-revert-*.json

# Manual revert to baseline
node ./cron-tripwires/scripts/auto-revert.js "Manual revert after auto-revert failure"
```

#### Alert Delivery Issues
```bash
# Test alert channels
node ./cron-tripwires/alerts/alert-manager.js test

# Check environment variables
env | grep -E "(SLACK|PAGER|DISCORD|EMAIL)"
```

### Health Checks

```bash
# System health check script
#!/bin/bash
echo "🔍 Weekly Cron Tripwires Health Check"
echo "======================================"

echo "📅 Cron Job Status:"
crontab -l | grep weekly-validation || echo "❌ Cron job not found"

echo ""
echo "📚 Baseline Status:"
node ./cron-tripwires/scripts/baseline-manager.js list | head -5

echo ""
echo "📊 Recent Validations:"
ls -la ./cron-tripwires/logs/cron-validation-*.log | tail -3

echo ""
echo "🚨 Recent Alerts:"
ls -la ./cron-tripwires/alerts/p0-alert-*.json | tail -3 || echo "No recent P0 alerts"

echo ""
echo "✅ Health check complete"
```

## Maintenance

### Weekly Tasks
- Review validation logs for trends
- Verify alert channel health
- Update baseline after intentional changes

### Monthly Tasks  
- Audit tripwire threshold effectiveness
- Review auto-revert success rates
- Update documentation and runbooks

### Quarterly Tasks
- Performance tune tripwire sensitivity
- Evaluate new tripwire opportunities
- Team training on alert response procedures

Generated: 2025-09-08T16:20:56.592Z  
Version: 1.0  
Ready for: Production deployment
