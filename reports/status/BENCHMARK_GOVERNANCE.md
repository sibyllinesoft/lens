# Benchmark Governance & Anti-Fraud Protocol

## Pre-Registration Requirements

All benchmark experiments MUST be pre-registered with a one-page document containing:

### Required Fields
- **Metrics**: Specific performance metrics to be measured (QPS, latency, memory, etc.)
- **Dataset**: Exact dataset to be used with SHA256 hash and provenance
- **SUT Commit**: Exact git commit SHA of system under test
- **Acceptance Thresholds**: Specific criteria for success/failure
- **Methodology**: Step-by-step procedure for benchmark execution
- **Environment**: Target hardware configuration and requirements

### Example Pre-Registration

```yaml
experiment_id: lens-rust-v1-baseline
title: "Rust Implementation vs TypeScript Baseline Performance"
date: 2025-09-06
investigator: [Name]

metrics:
  primary:
    - qps_at_p95_latency_200ms
    - p50_latency_ms  
    - p95_latency_ms
    - memory_rss_mb
  
  acceptance_thresholds:
    qps_improvement: ">20%"
    p95_latency: "<200ms" 
    memory_usage: "<500MB"

dataset:
  name: "clean-baseline-v0"
  uri: "file://./clean-baseline/minimal-dataset-v0.json"
  sha256: "3f7e0179c726ce79edb12a734cd241ffe7f520bc03da2785739ca93c96a3d22b"
  
sut_commit: "887bdac42ffa3495cef4fb099a66c813c4bc764a"

methodology: |
  1. Build Rust service from sut_commit
  2. Load dataset and verify SHA256
  3. Perform service handshake with nonce/response
  4. Execute 1000 queries with 10 concurrent clients
  5. Measure latency distribution and resource usage
  6. Generate attestation report with full provenance

environment:
  cpu: "AMD Ryzen 7 5800X"
  cores: 16
  memory: "32GB"
  governor: "performance"
```

## PR Compliance Requirements

All pull requests adding performance figures MUST:

1. **Link Exact Report JSON**: Include direct link to CI-generated report JSON
2. **No Screenshots/PDFs**: Raw data only, no processed visualizations without source
3. **Full Attestation Chain**: Binary hash, dataset hash, environment capture
4. **Handshake Verification**: Successful nonce/response from SUT

### Non-Compliant Examples
❌ "Performance improved by 20%" (no evidence)  
❌ Screenshot of graph without raw data  
❌ PDF report without source JSON  
❌ Benchmark results without handshake field  

### Compliant Examples  
✅ Links to `benchmark-results/experiment-123.json` with full attestation  
✅ Raw CSV data with dataset SHA256 and environment capture  
✅ Git commit with validated report artifacts  

## Audit Trail Requirements

Every performance claim MUST have:

- **Reproducible Command**: Exact command line to reproduce results
- **Environment Fingerprint**: CPU, kernel, memory, network config  
- **Dataset Provenance**: Immutable dataset with cryptographic verification
- **Binary Attestation**: Hash chain from source to executable
- **Network Logs**: Evidence of real service communication (no mocks)

## Violation Consequences

Violations of this protocol result in:

1. **Immediate Quarantine**: Results marked invalid and excluded from citations
2. **PR Rejection**: Non-compliant PRs automatically closed
3. **Retraction Requirements**: Published results must be corrected/withdrawn
4. **Enhanced Review**: Future submissions subject to additional verification

## Review Process

1. **Automated Validation**: CI checks schema compliance and tripwires
2. **Peer Review**: At least one independent verification of methodology  
3. **Attestation Verification**: Manual check of provenance chain
4. **Publication Approval**: Final sign-off before external use

This protocol ensures research integrity and prevents synthetic data contamination.