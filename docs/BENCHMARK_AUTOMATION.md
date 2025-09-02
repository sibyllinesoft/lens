# Lens Benchmark Automation System

Comprehensive automated benchmarking system for continuous quality monitoring of the Lens code search system.

## Overview

The Lens benchmark automation system provides:

- **Nightly Benchmark Execution**: Automated full benchmark suite execution
- **Performance Regression Detection**: Statistical analysis with configurable thresholds
- **Comprehensive Reporting**: Multi-format reports with trend analysis
- **Automated Alerting**: Slack/email notifications for failures and regressions
- **Historical Tracking**: Performance trend monitoring over time
- **Quality Gates**: Automated promotion/rollback decisions

## Architecture

### Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  GitHub Actions │───▶│  Nightly Script │───▶│ Report Generator│
│   Scheduler     │    │   Orchestrator  │    │   & Analysis    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   NATS Server   │    │  Lens Search    │    │  Historical     │
│   (Telemetry)   │    │    Engine       │    │     Data        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Files

- `.github/workflows/nightly-benchmark.yml` - GitHub Actions workflow
- `scripts/nightly-benchmark.js` - Main automation orchestrator
- `scripts/generate-report.js` - Comprehensive report generation
- `scripts/analyze-regressions.js` - Performance regression analysis
- `scripts/generate-badge.js` - Performance badge generation

## Features

### 1. Automated Nightly Execution

**Schedule**: Every night at 2 AM UTC to avoid peak usage

**Process**:
1. Health check of search server and NATS
2. Corpus-golden consistency validation
3. Full benchmark suite execution (1,916 golden items)
4. Performance regression analysis
5. Report generation and archival
6. Notification dispatch

**Timeout Protection**: 4-hour total timeout with component-level timeouts

### 2. Performance Monitoring

**Tracked Metrics**:
- **Recall@10**: Primary search quality metric (target: 70%)
- **Recall@50**: Extended search coverage (target: 85%)
- **NDCG@10**: Ranking quality metric (target: 65%)
- **P95 Latency**: 95th percentile response time (target: <200ms)
- **Error Rate**: Query failure rate (target: <5%)

**Regression Thresholds**:
- **Warning**: 5-15% degradation depending on metric
- **Critical**: 10-25% degradation depending on metric

### 3. Multi-Stage Pipeline Testing

**Stage Coverage**:
- **Stage-A**: Lexical matching (baseline)
- **Stage-B**: Structural search with symbols
- **Stage-C**: Hybrid semantic + structural

**Test Matrix**:
- All 3 stages tested independently
- Cold and warm cache scenarios
- Multiple random seeds for statistical significance

### 4. Comprehensive Reporting

**Output Formats**:
- **JSON**: Machine-readable data for API integration
- **Markdown**: Human-readable GitHub-friendly reports
- **HTML**: Rich dashboard with charts (placeholder for interactive elements)

**Report Sections**:
- Executive summary with overall health status
- Key metrics vs. targets comparison table
- Trend analysis with historical comparison
- Performance regression alerts
- Actionable recommendations
- Corpus-golden consistency validation results

### 5. Alerting & Notifications

**Slack Integration**:
- **#lens-alerts**: Critical failures and system issues
- **#lens-performance**: Performance regressions and health updates

**Notification Triggers**:
- Benchmark execution failures
- Performance regressions (warning or critical)
- Corpus-golden inconsistencies
- Optional success notifications

**Alert Content**:
- Run ID and timestamp
- Affected metrics with percentage changes
- Links to detailed reports and workflow logs

## Usage

### Manual Execution

```bash
# Quick smoke test (for PR validation)
npm run benchmark:smoke

# Full benchmark suite (nightly equivalent)
npm run benchmark:full

# Health check only
npm run benchmark:health

# Corpus consistency validation
npm run benchmark:validate
```

### Manual Workflow Trigger

1. Go to GitHub Actions tab
2. Select "Nightly Benchmark Suite" workflow
3. Click "Run workflow"
4. Choose options:
   - **Suite Type**: `full` or `smoke`
   - **Notify on Success**: Enable for success notifications

### Local Development

```bash
# Start required services
docker-compose up -d  # NATS and other dependencies
npm run start         # Lens search server

# Run benchmark with custom parameters
node scripts/nightly-benchmark.js run-suite \
  --suite-type smoke \
  --output-dir custom-output \
  --verbose

# Generate reports from existing results
node scripts/generate-report.js \
  --input-dir benchmark-results/nightly-20240901-123456 \
  --output-formats json,markdown,html \
  --compare-with-history

# Analyze regressions
node scripts/analyze-regressions.js \
  --current-run benchmark-results/nightly-20240901-123456 \
  --history-dir benchmark-results/history \
  --output-format github-actions
```

## Configuration

### Environment Variables

```bash
# GitHub Actions Environment
NATS_URL=nats://localhost:4222
BENCHMARK_TIMEOUT_MINUTES=180
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# Local Development
LENS_SERVER_URL=http://localhost:4000
BENCHMARK_OUTPUT_DIR=./benchmark-results
```

### Performance Thresholds

Configurable in `scripts/analyze-regressions.js`:

```javascript
REGRESSION_THRESHOLDS: {
  recall_at_10: { warning: 0.05, critical: 0.10 },
  e2e_p95_latency: { warning: 0.15, critical: 0.25 },
  error_rate: { warning: 0.02, critical: 0.05 }
}
```

### Notification Settings

Configured via GitHub Secrets:
- `SLACK_WEBHOOK_URL`: Slack incoming webhook URL
- Additional integrations can be added to the workflow

## Data Storage

### Directory Structure

```
benchmark-results/
├── nightly-20240901-020000/          # Current run
│   ├── summary.json                   # Quick metrics overview
│   ├── benchmark-results.json         # Full benchmark data
│   ├── execution-metadata.json        # Runtime information
│   ├── consistency-validation.json    # Corpus health check
│   ├── regression-analysis.json       # Regression analysis
│   └── benchmark-report-*.{json,md,html}  # Generated reports
├── history/                           # Historical runs (last 30)
│   ├── nightly-20240831-020000/
│   └── nightly-20240830-020000/
└── artifacts/                         # CI artifacts
    └── benchmark-results-nightly-*/   # GitHub Actions artifacts
```

### Data Retention

- **GitHub Artifacts**: 30 days
- **Local History**: Last 30 runs (automatic cleanup)
- **Long-term Storage**: Configure external archival as needed

## Performance Targets

### Quality Metrics (Minimum Acceptable)

| Metric | Target | Excellent | Good | Needs Improvement |
|--------|--------|-----------|------|-----------------|
| Recall@10 | 70% | ≥80% | ≥70% | <60% |
| Recall@50 | 85% | ≥90% | ≥80% | <70% |
| NDCG@10 | 65% | ≥75% | ≥65% | <55% |
| P95 Latency | 200ms | ≤150ms | ≤200ms | >300ms |
| Error Rate | <5% | <1% | <3% | >5% |

### Promotion Gate Criteria

**Required for Production Deployment**:
- No critical performance regressions
- All quality metrics meet minimum targets
- Error rate below 5%
- Corpus-golden consistency validation passes

## Troubleshooting

### Common Issues

**1. Benchmark Execution Timeout**
```bash
# Check search server logs
docker-compose logs lens-api

# Verify NATS connectivity
node scripts/nightly-benchmark.js health-check
```

**2. Corpus-Golden Inconsistencies**
```bash
# Run validation manually
npm run benchmark:validate

# Check inconsistency report
cat benchmark-results/latest/inconsistency.ndjson
```

**3. Regression False Positives**
```bash
# Analyze with verbose output
node scripts/analyze-regressions.js \
  --current-run benchmark-results/latest \
  --verbose \
  --output-format summary
```

**4. GitHub Actions Failures**
- Check workflow logs in GitHub Actions tab
- Verify secrets are properly configured
- Ensure Docker services are healthy

### Debugging Commands

```bash
# Enable verbose logging for all scripts
export DEBUG=lens:benchmark:*

# Test individual components
node scripts/nightly-benchmark.js health-check --verbose
node scripts/generate-report.js --input-dir test-data --verbose

# Manual NATS connection test
node -e "const nats = require('nats'); nats.connect().then(nc => { console.log('NATS OK'); nc.close(); });"
```

## Monitoring Dashboard

### Status Badge

Add to README.md:
```markdown
![Benchmark Status](./benchmark-badge.svg)
```

### Key Metrics Overview

Monitor these indicators daily:
- **Overall Status**: Green/Yellow/Red health indicator
- **Trend Direction**: Improving/stable/declining over 7 days
- **Regression Count**: Number of active performance regressions
- **Last Success**: Timestamp of last successful full benchmark

### Alerting Escalation

1. **Green**: Automated monitoring, no action needed
2. **Yellow**: Review trends, investigate if pattern emerges
3. **Red**: Immediate investigation required, consider deployment freeze

## Future Enhancements

### Planned Improvements

- **Interactive Dashboards**: Grafana integration for real-time metrics
- **A/B Testing Integration**: Automated feature flag performance testing
- **Multi-Environment Support**: Staging vs. production benchmark comparison
- **Performance Profiling**: Automated bottleneck identification and suggestions
- **Custom Query Sets**: Support for domain-specific benchmark suites

### Integration Opportunities

- **Datadog/New Relic**: APM integration for deeper performance insights
- **PagerDuty**: Critical alert escalation
- **Jira**: Automatic ticket creation for regressions
- **Jenkins/TeamCity**: Alternative CI/CD platform support

## Contributing

### Adding New Metrics

1. Update `src/benchmark/metrics-calculator.ts` to compute the metric
2. Add threshold configuration to `scripts/analyze-regressions.js`
3. Update report templates in `scripts/generate-report.js`
4. Add badge support in `scripts/generate-badge.js`

### Extending Notifications

1. Add new notification provider to GitHub Actions workflow
2. Update environment variable documentation
3. Configure message formatting in workflow YAML

### Performance Tuning

Benchmark execution time optimization:
- Parallel query execution where possible
- Optimize golden dataset size vs. statistical significance
- Implement adaptive timeout based on historical performance
- Cache optimization for repeated benchmark runs

---

**For support or questions about the benchmark automation system, please:**
- Check the troubleshooting section above
- Review GitHub Actions workflow logs
- Contact the Lens development team
- Create an issue in the repository with benchmark logs
