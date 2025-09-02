# Lens Search Engine - Benchmark Report

**Generated**: 2025-09-01T01:41:55.012Z
**System**: Lens Search Engine
**Server**: http://localhost:3001

## Executive Summary

**Overall Assessment**: ACCEPTABLE

### ✅ Achievements
- Service health check passed
- Search latency within acceptable range
- Built-in benchmark suite operational

### ⚠️ Issues Identified
- Low success rate: 0.0%

## Service Performance Analysis

### API Response Performance
- **Success Rate**: 0.0%
- **Average Latency**: 0.00ms
- **P95 Latency**: 0.00ms
- **Total Queries**: 5
- **Successful**: 0
- **Failed**: 5

### Result Quality Distribution
- **No distribution data available**

## Built-in Benchmark Results

✅ **Smoke Test**: PASSED
- **Duration**: 1318ms
- **Trace ID**: 8bccc4fb-c517-492a-b04d-cf3987bb7d0d
- **Promotion Gate**: unknown
- **Generated Reports**:
  - pdf_path: `/media/nathan/Seagate Hub/Projects/lens/benchmark-results/lens-benchmark-report-2025-09-01T01-41-56-870Z.pdf`
  - markdown_path: `/media/nathan/Seagate Hub/Projects/lens/benchmark-results/lens-benchmark-report-2025-09-01T01-41-56-870Z.md`
  - json_path: `/media/nathan/Seagate Hub/Projects/lens/benchmark-results/lens-benchmark-report-2025-09-01T01-41-56-870Z.json`

## Detailed Query Analysis

| Query | Mode | Status | Latency | Results | Notes |
|-------|------|--------|---------|---------|-------|
| function | lex | ❌ | 2.87ms | 0 | Error: [object Object] |
| class | lex | ❌ | 5.53ms | 0 | Error: [object Object] |
| user | struct | ❌ | 7.08ms | 0 | Error: [object Object] |
| UserService | struct | ❌ | 5.35ms | 0 | Error: [object Object] |
| async function | hybrid | ❌ | 5.55ms | 0 | Error: [object Object] |

## Recommendations & Next Steps

1. ⚠️ HIGH: Investigate search API failures - many queries are not returning results
2. 📋 PROCEED WITH CAUTION: Address identified issues while continuing development
3. 🔄 NEXT: Set up automated benchmark runs in CI/CD pipeline
4. 🔄 NEXT: Establish SLA targets and monitoring based on current performance baseline