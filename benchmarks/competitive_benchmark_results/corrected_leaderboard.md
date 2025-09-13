# 🏆 CORRECTED Competitive Benchmark Leaderboard

**🔧 BUG FIXES APPLIED:**
- ✅ Fixed system ID canonicalization (t1_hero key mismatch)
- ✅ Corrected pairwise win rate algorithm
- ✅ Removed constant fill values (0.1 → proper calculation)
- ✅ Added NaN masking instead of fillna

**Systems Ranked**: 11
**Valid Results**: Available systems only

## 📊 Rankings

| Rank | System | ΔnDCG@10 | Win Rate | p95 Latency | Benchmarks |
|------|--------|----------|----------|-------------|------------|
| **#1** | **t1_hero** | +0.173 | 100.0% | 223ms | 5 |
| **#2** | **openai_text_embedding_3_large** | +0.151 | 88.0% | 298ms | 5 |
| **#3** | **colbertv2** | +0.138 | 74.0% | 177ms | 5 |
| **#4** | **hybrid_bm25_dense** | +0.136 | 72.0% | 188ms | 5 |
| **#5** | **e5-large-v2** | +0.128 | 62.0% | 171ms | 5 |
| **#6** | **tasb** | +0.117 | 46.0% | 155ms | 5 |
| **#7** | **spladepp** | +0.106 | 48.0% | 140ms | 5 |
| **#8** | **unicoil** | +0.069 | 26.0% | 126ms | 5 |
| **#9** | **contriever** | +0.066 | 24.0% | 139ms | 5 |
| **#10** | **bm25** | +0.005 | 4.0% | 107ms | 5 |
| **#11** | **bm25+rm3** | +0.004 | 6.0% | 118ms | 5 |

## 🔧 Bug Fix Summary

**Original Issues:**
1. System key mismatch: `t1_hero` data existed but lookups failed
2. Heatmap algorithm: Incorrect pairwise comparison logic
3. Constant fill: NaN values imputed with 0.1 instead of proper masking

**Fixes Applied:**
1. Canonical system ID mapping with validation
2. Correct pairwise win rate: `(wins + 0.5*ties) / total_comparisons`
3. Variance checks to detect future constant fill bugs
4. NaN masking in visualizations (no more fillna)

## 📋 Technical Details

- **Algorithm**: Pairwise comparison with tie handling (ties = 0.5)
- **Missing Data**: NaN-masked, not imputed with constants
- **Validation**: Row variance checks, anti-symmetry validation
- **Provenance**: All calculations traceable to competitor_matrix.csv

---
*Corrected analysis generated with bug fixes applied*