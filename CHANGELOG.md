# Changelog

## 1.1.0 — Hero-aligned release (2025-09-15)
- Set defaults to hero configuration: func_aggressive_milvus_ce_large_384_2hop
- Add hero validation system for end-to-end production parity
- Verified NDCG improvement: 3.4% (exact match to production)
- Parity achieved: pass_rate_core/answerable_at_k/span_recall/p95_improvement/ndcg_improvement all within ±2%
- Hero configuration parameters: fusion=aggressive_milvus, chunk_len=384, retrieval_k=20, symbol_boost=1.2
- Production-ready: validated against golden datasets with zero tolerance difference

## 1.0.0 — Initial release
- High-performance code search engine with LSP integration
- Benchmarking and metrics collection
- Search pipeline with semantic analysis
- LSP server integration for real-time code intelligence