# Lens Search Engine Benchmark Report

## Summary
- **Total Queries**: 3
- **Time Range**: 2025-09-01T02:43:52.362Z to 2025-09-01T02:43:52.363Z
- **Queries per Second**: 3000.00

## Latency Metrics

### Total Response Time
- **Average**: 15.00ms
- **p50**: 13.00ms
- **p95**: 22.00ms
- **p99**: 22.80ms

### Stage A (Lexical + Fuzzy)
- **Average**: 4.00ms
- **p95**: 4.90ms
- **SLA Compliance**: 100.0% under 8ms

### Stage B (Symbol + AST)
- **Average**: 7.00ms
- **p95**: 7.90ms
- **SLA Compliance**: 100.0% under 10ms

### Stage C (Semantic Rerank)
- **Average**: 12.00ms
- **p95**: 12.00ms

## Result Quality

### Result Counts
- **Average Results per Query**: 15.0
- **Max Results**: 22

### Quality Scores
- **F1 Score**: 0.760 (avg)
- **Precision**: 0.817 (avg)
- **Recall**: 0.717 (avg)

## Performance Analysis
- **Error Rate**: 0.00%
- **Total SLA Compliance**: 66.7% under 20ms

## Recommendations
- Overall response time needs improvement for SLA compliance