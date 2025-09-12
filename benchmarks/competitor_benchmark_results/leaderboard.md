# Rigorous Competitor Benchmark Leaderboard

## Overall Rankings by nDCG@10

| Rank | System | nDCG@10 | 95% CI | Recall@50 | P95 Latency (ms) | Significance |
|------|--------|---------|--------|-----------|------------------|--------------|
| 1 | BM25_Baseline | 0.6962 | [0.680, 0.712] | 1.0000 | 0.0 |  |
| 2 | Hybrid_BM25_Dense | 0.0419 | [0.036, 0.048] | 0.1768 | 2.7 | ⚠️ |
| 3 | BM25_RM3 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.0 | ⚠️ |
| 4 | ColBERTv2 | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1 | ⚠️ |
| 5 | ANCE | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1 | ⚠️ |
| 6 | OpenAI_Ada | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1 | ⚠️ |
| 7 | T1_Hero | 0.0000 | [0.000, 0.000] | 0.0000 | 0.1 | ⚠️ |

## Legend
- ⭐: Significantly better than BM25 baseline (p < 0.05, Holm-corrected)
- ⚠️: Significantly worse than BM25 baseline (p < 0.05, Holm-corrected)

## Statistical Notes
- Confidence intervals computed via bootstrap with B=2000 samples
- Multiple comparison correction applied using Holm-Bonferroni method
- All systems tested on identical query sets across 5 benchmark datasets
