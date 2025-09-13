
# SANITY PYRAMID SCORECARD
Generated: 2025-09-12 23:58:31

## 🎯 HARD GATES STATUS: ❌ FAIL (6/10)

**Pre-Generation Pass Rate:** 0.0% / 85% target
**Queries Validated:** 2
**Latency P95:** 250ms

## 📊 OPERATION PERFORMANCE

| Operation | Pass Rate | Avg ESS | Queries |
|-----------|-----------|---------|---------|
| extract   |     0.0% |    0.60 |       1 |
| explain   |     0.0% |    0.54 |       1 |

## 🚫 TOP FAILURE REASONS
- ESS 0.60 < threshold 0.75: 1 queries (50.0%)
- ESS 0.54 < threshold 0.6: 1 queries (50.0%)

## 🔬 ABLATION SENSITIVITY
- Context Shuffle: 12.0% F1 drop (target: ≥10%)
- Drop Top-1: 8.0% F1 drop
- ESS-Answer Correlation: 0.65

## 🎚️ HARD GATES DETAIL
- ❌ Overall Pass Rate ≥85%
- ✅ Locate Pass Rate ≥90%
- ❌ Extract Pass Rate ≥85%
- ❌ Explain Pass Rate ≥70%
- ✅ Answerable@k ≥70%
- ❌ SpanRecall ≥50%
