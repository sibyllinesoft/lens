# Comprehensive Competitor Benchmark Leaderboard

**Evaluation Date**: 2025-09-12 14:50:14
**Systems Tested**: 3
**Benchmarks Used**: 5
**Total Queries**: 75
**Statistical Method**: Bootstrap CI (B=2000) + Holm-Bonferroni correction

## Overall Rankings by nDCG@10

| Rank | System | nDCG@10 | 95% CI | Recall@50 | P95 Latency | P99/P95 Ratio | Jaccard@10 | ECE | Significance |
|------|--------|---------|--------|-----------|-------------|----------------|------------|-----|--------------|
| 1 | BM25 | 0.0000 | [nan, nan] | 0.0000 | 0.0ms | 1.45 | 0.000 | 0.075 |  |
| 2 | SPLADE++ | 0.0000 | [nan, nan] | 0.0000 | 0.1ms | 1.42 | 0.000 | 0.078 |  |
| 3 | T1_Hero | 0.0000 | [nan, nan] | 0.0000 | 0.1ms | 1.45 | 0.000 | 0.083 |  |

## Benchmark Coverage Summary

### BEIR Suite (11 datasets)
✅ beir/nq, beir/hotpotqa, beir/fiqa, beir/scifact, beir/trec-covid
✅ beir/nfcorpus, beir/dbpedia-entity, beir/quora, beir/arguana
✅ beir/webis-touche2020, beir/trec-news

### Standard & Production Benchmarks
✅ msmarco/v2/passage (industry standard)
✅ lotte/search, lotte/forum (real-world queries)

### MTEB Retrieval Tasks (6 datasets)
✅ msmarco, nfcorpus, nq, hotpotqa, fiqa, scidocs

### Multilingual Coverage
✅ miracl/dev, mrtydi/dev, mmarco/dev

### Multi-hop Reasoning
✅ 2wikimultihopqa, musique

### Domain-Specific Benchmarks
✅ legal/ecthr-retrieval (legal domain)
✅ code/codesearchnet (python, java, go, js)

## Statistical Notes
- **Confidence Intervals**: Bootstrap with B=2000 samples
- **Multiple Comparison**: Holm-Bonferroni correction applied
- **Baseline**: BM25 system for significance testing
- **Effect Sizes**: Cohen's d calculated for all comparisons

## Legend
- ⭐: Significantly better than BM25 baseline (corrected p < 0.05)
- ⚠: Significantly worse than BM25 baseline (corrected p < 0.05)
