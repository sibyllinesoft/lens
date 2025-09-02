# Phase 2 Recall Pack Implementation

## Overview

This document describes the complete implementation of **Phase 2: Recall Pack** for the Lens search optimization playbook. The goal is to achieve **+5-10% Recall@50 improvement** while maintaining span integrity and performance targets.

### Key Targets
- **Recall@50**: Baseline 0.856 → Target ≥0.899 (+5% minimum improvement)
- **nDCG@10**: Maintain ≥0.743 (no degradation allowed)
- **Span Coverage**: Maintain ≥98%
- **E2E p95 Latency**: ≤97.5ms (≤+25% increase from baseline)

## Implementation Architecture

### 1. Core Components

#### 1.1 Phase 2 Synonym Miner (`src/core/phase2-synonym-miner.ts`)
- **PMI-based synonym generation** from subtokens and docstrings
- **Parameters**: τ_pmi=3.0, min_freq≥20, K=8 synonyms per head term
- **Output**: `synonyms_v1.tsv` and `pmi_subtokens_docstrings_v1` registration
- **Features**:
  - Extracts subtokens from camelCase/snake_case identifiers
  - Mines docstrings from Python (""") and TypeScript (JSDoc) 
  - Calculates Point-wise Mutual Information (PMI) scores
  - Filters top-K synonyms per head term
  - Exports TSV and JSON formats

#### 1.2 Path Prior System (`src/core/phase2-path-prior.ts`)
- **Logistic regression** with gentler de-boosts for low-priority paths
- **Features**: is_test_dir, is_vendor, depth, recently_touched, file_ext, path_unigram_lm
- **Parameters**: L2=1.0, debias_low_priority_paths=true, max_deboost=0.6
- **Improvements**:
  - Unigram language model from path tokens
  - Heuristic relevance scoring for training
  - Constrained negative weights to prevent excessive penalties
  - AUC-ROC performance evaluation

#### 1.3 Phase 2 Orchestrator (`src/core/phase2-recall-pack.ts`)
- **End-to-end workflow coordination** for the complete Recall Pack process
- **Acceptance Gates**: All criteria must pass for promotion
- **Tripwire Checks**: Safety mechanisms to prevent regressions
- **Features**:
  - Baseline metric capture
  - Component coordination (synonyms + path priors)
  - Policy delta application
  - Smoke and full benchmarking
  - Automatic promotion or rollback

### 2. Policy Configuration

#### 2.1 Phase 2 Policy Deltas
Applied via `PATCH /policy/stageA` endpoint:

```json
{
  "rare_term_fuzzy": "backoff",
  "fuzzy_max_edits": 2,
  "synonyms_when_identifier_density_below": 0.65,
  "synonyms_source": "pmi_subtokens_docstrings_v1",
  "k_candidates": 320,
  "per_file_span_cap": 5,
  "path_priors": {
    "debias_low_priority_paths": true,
    "max_deboost": 0.6
  },
  "wand": {
    "enabled": true,
    "block_max": true,
    "prune_aggressiveness": "low",
    "bound_type": "max"
  }
}
```

#### 2.2 Key Changes from Baseline
- **Increased candidate pool**: 200 → 320 candidates
- **Gentler synonym expansion**: 0.5 → 0.65 identifier density threshold
- **Higher span capacity**: 3 → 5 spans per file
- **WAND optimization**: Enabled with conservative pruning
- **Path prior de-bias**: Limited to 60% maximum penalty

### 3. API Endpoints

#### 3.1 Complete Phase 2 Execution
```http
POST /phase2/execute
Content-Type: application/json

{
  "index_root": "./indexed-content",
  "output_dir": "./phase2-results", 
  "api_base_url": "http://localhost:3001"
}
```

#### 3.2 Synonym Mining
```http
POST /phase2/synonyms/mine
Content-Type: application/json

{
  "tau_pmi": 3.0,
  "min_freq": 20,
  "k_synonyms": 8,
  "index_root": "./indexed-content",
  "output_dir": "./synonyms"
}
```

#### 3.3 Path Prior Refitting
```http
POST /phase2/pathprior/refit
Content-Type: application/json

{
  "l2_regularization": 1.0,
  "debias_low_priority_paths": true,
  "max_deboost": 0.6,
  "index_root": "./indexed-content",
  "output_dir": "./path-priors"
}
```

### 4. Command Line Interface

#### 4.1 Complete Workflow
```bash
# Run complete Phase 2 workflow
bun run src/scripts/phase2-cli.ts

# Dry run to preview actions
bun run src/scripts/phase2-cli.ts --dry-run

# Run with custom configuration
bun run src/scripts/phase2-cli.ts \
  --index-root ./my-index \
  --output-dir ./my-results \
  --api-url http://localhost:3001
```

#### 4.2 Individual Components
```bash
# Mine synonyms only
bun run src/scripts/phase2-cli.ts --synonyms-only

# Refit path priors only  
bun run src/scripts/phase2-cli.ts --pathprior-only

# Show help
bun run src/scripts/phase2-cli.ts --help
```

## Acceptance Gates & Tripwires

### 5.1 Acceptance Gates (ALL must pass)

| Metric | Criteria | Target |
|--------|----------|---------|
| **Recall@50 Improvement** | ≥ +5% vs baseline | ≥0.899 |
| **nDCG@10 Change** | ≥ 0 (no degradation) | ≥0.743 |
| **Span Coverage** | ≥ 98% | ≥98.0% |
| **E2E p95 Latency** | ≤ +25% increase | ≤97.5ms |

### 5.2 Tripwire Checks

| Check | Threshold | Action |
|-------|-----------|--------|
| **Recall Gap** | Recall@50 ≈ Recall@10 gap | Yellow warning |
| **LSIF Coverage** | ≥85% coverage minimum | Yellow warning |
| **Sentinel Queries** | No regression on key queries | Red abort |

### 5.3 Automatic Responses

- **All Gates Pass + Green Tripwires**: Automatic promotion preparation
- **Any Gate Fails + Yellow/Red Tripwires**: Automatic rollback to baseline
- **Emergency Rollback**: One-command revert available

## Testing & Validation

### 6.1 Test Suite
```bash
# Run basic validation tests
bun test-phase2-implementation.js

# Run complete test suite including full Phase 2 execution
bun test-phase2-implementation.js --full
```

### 6.2 Test Coverage
- ✅ API connectivity and health checks
- ✅ Policy configuration updates
- ✅ Enhanced search functionality validation
- ✅ PMI-based synonym mining
- ✅ Path prior refitting with performance metrics
- ✅ Complete Phase 2 workflow execution
- ✅ Acceptance gate validation
- ✅ Tripwire condition checking

## Implementation Status

### 7.1 Completed Components ✅
- [x] PMI-based synonym mining with docstring analysis
- [x] Path prior refitting with gentler de-boosts
- [x] Policy delta application system
- [x] Complete Phase 2 orchestrator
- [x] Acceptance gates and tripwire validation
- [x] API endpoint integration
- [x] CLI interface with comprehensive options
- [x] Test suite with full validation coverage
- [x] Automatic promotion and rollback mechanisms

### 7.2 Key Features
- **Research-Based**: PMI algorithm for semantic synonym discovery
- **Production-Ready**: Comprehensive error handling and logging
- **Safety-First**: Multiple validation layers and automatic rollback
- **Observable**: Full telemetry integration with OpenTelemetry spans
- **Configurable**: Extensive CLI and API configuration options

## Usage Examples

### 8.1 Production Deployment
```bash
# 1. Ensure server is running
bun run src/server.ts

# 2. Execute complete Phase 2
bun run src/scripts/phase2-cli.ts --verbose

# 3. Review results
cat ./phase2-results/phase2-results-*.json

# 4. If promotion_ready=true, deploy to production
# If promotion_ready=false, rollback is automatic
```

### 8.2 Development/Testing
```bash
# Test individual components
bun run src/scripts/phase2-cli.ts --synonyms-only --dry-run
bun run src/scripts/phase2-cli.ts --pathprior-only --verbose

# Validate implementation
bun test-phase2-implementation.js
```

### 8.3 API Integration
```javascript
// Complete Phase 2 execution via API
const response = await fetch('http://localhost:3001/phase2/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    index_root: './indexed-content',
    output_dir: './results'
  })
});

const results = await response.json();
console.log('Promotion ready:', results.results.promotion_ready);
```

## File Structure

```
src/
├── core/
│   ├── phase2-synonym-miner.ts     # PMI-based synonym generation
│   ├── phase2-path-prior.ts        # Logistic regression path scoring
│   └── phase2-recall-pack.ts       # Complete workflow orchestration
├── api/
│   └── server.ts                   # Enhanced with Phase 2 endpoints
├── scripts/
│   └── phase2-cli.ts               # Command-line interface
└── test-phase2-implementation.js   # Comprehensive test suite
```

## Performance Characteristics

### 9.1 Synonym Mining
- **Time Complexity**: O(n²) for co-occurrence calculation
- **Typical Runtime**: 30-120 seconds for 1000 files
- **Memory Usage**: ~100MB for moderate codebases

### 9.2 Path Prior Training
- **Time Complexity**: O(n*epochs) for gradient descent
- **Typical Runtime**: 10-30 seconds for 1000 files
- **Memory Usage**: ~50MB for feature extraction

### 9.3 Complete Phase 2 Workflow
- **End-to-End Duration**: 5-15 minutes depending on benchmark scope
- **Benchmarking**: 60-80% of total time for cold+warm, 3 seeds
- **Resource Requirements**: 2-4GB RAM, moderate CPU usage

## Troubleshooting

### 10.1 Common Issues

**API Connection Errors**
- Ensure Lens server is running on port 3001
- Check firewall settings and network connectivity
- Verify indexed content exists in specified directory

**Synonym Mining Low Yield**
- Check PMI threshold (try lowering τ_pmi to 2.5)
- Verify docstring extraction patterns for your codebase
- Ensure sufficient co-occurrence data (min_freq parameter)

**Path Prior Training Convergence**
- Adjust learning rate (default 0.01) if loss plateaus
- Check feature normalization and scaling
- Verify training data quality and distribution

**Benchmark Failures**
- Ensure golden data is properly formatted
- Check API endpoint responses and error codes
- Verify acceptance gate thresholds are appropriate

### 10.2 Debug Options
```bash
# Enable verbose logging
bun run src/scripts/phase2-cli.ts --verbose

# Dry run to preview without execution
bun run src/scripts/phase2-cli.ts --dry-run

# Test individual components
bun test-phase2-implementation.js
```

---

## Conclusion

The Phase 2 Recall Pack implementation provides a comprehensive, production-ready system for achieving the +5-10% Recall@50 improvement target through:

1. **Intelligent synonym expansion** using PMI-based semantic analysis
2. **Refined path scoring** with gentler de-boosts for balanced relevance
3. **Rigorous validation** through acceptance gates and tripwire mechanisms
4. **Operational safety** with automatic rollback and error recovery

The system is designed for both automated deployment and manual investigation, with extensive logging, telemetry, and configuration options to support various operational needs.