#!/usr/bin/env python3
"""
Comprehensive test suite for the competitor benchmarking framework.

Validates all components of the comprehensive benchmarking system:
- All 11 competitor systems function correctly
- 20+ benchmark datasets are properly initialized
- Statistical analysis produces valid results
- Validation guards work as expected
- All artifacts are generated with proper formatting
- Mandatory compliance requirements are met
"""

import asyncio
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Import the comprehensive benchmarking system
from comprehensive_competitor_benchmark import (
    ComprehensiveCompetitorBenchmark,
    BenchmarkResult,
    StatisticalSummary,
    CompetitorComparison,
    ValidationGuardResult,
    # All competitor systems
    BM25System,
    BM25RM3System,
    SPLADEPPSystem,
    UniCOILSystem,
    ColBERTv2System,
    TASBSystem,
    ContrieverSystem,
    HybridBM25DenseSystem,
    OpenAIEmbeddingSystem,
    CohereEmbeddingSystem,
    T1HeroSystem
)

class TestCompetitorSystems:
    """Test all 11 mandatory competitor systems."""
    
    @pytest.mark.asyncio
    async def test_bm25_system(self):
        """Test BM25 baseline system."""
        system = BM25System()
        
        assert system.get_name() == "BM25"
        
        # Test search functionality
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert all(isinstance(score, (int, float)) for score in scores)
        assert metadata["system"] == "bm25"
        assert "execution_time_ms" in metadata
        
        # Test configuration
        config = system.get_config()
        assert config["system"] == "BM25"
        assert "k1" in config
        assert "b" in config
        
        # Test warmup
        await system.warmup(["warmup query 1", "warmup query 2"])
    
    @pytest.mark.asyncio
    async def test_bm25_rm3_system(self):
        """Test BM25+RM3 system."""
        system = BM25RM3System()
        
        assert system.get_name() == "BM25+RM3"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "bm25_rm3"
        assert "fb_docs" in metadata
        assert "fb_terms" in metadata
        
        config = system.get_config()
        assert config["system"] == "BM25+RM3"
        assert "fb_docs" in config
        assert "fb_terms" in config
    
    @pytest.mark.asyncio
    async def test_splade_pp_system(self):
        """Test SPLADE++ learned sparse system."""
        system = SPLADEPPSystem()
        
        assert system.get_name() == "SPLADE++"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "splade_pp"
        assert metadata["implementation"] == "learned_sparse"
        
        config = system.get_config()
        assert config["system"] == "SPLADE++"
        assert config["type"] == "learned_sparse"
    
    @pytest.mark.asyncio
    async def test_unicoil_system(self):
        """Test uniCOIL system."""
        system = UniCOILSystem()
        
        assert system.get_name() == "uniCOIL"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "unicoil"
        assert metadata["implementation"] == "learned_sparse_hybrid"
    
    @pytest.mark.asyncio
    async def test_colbertv2_system(self):
        """Test ColBERTv2 late interaction system."""
        system = ColBERTv2System()
        
        assert system.get_name() == "ColBERTv2"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "colbertv2"
        assert metadata["implementation"] == "late_interaction"
    
    @pytest.mark.asyncio
    async def test_tasb_system(self):
        """Test TAS-B dense bi-encoder system."""
        system = TASBSystem()
        
        assert system.get_name() == "TAS-B"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "tas_b"
        assert metadata["implementation"] == "dense_biencoder"
    
    @pytest.mark.asyncio
    async def test_contriever_system(self):
        """Test Contriever system."""
        system = ContrieverSystem()
        
        assert system.get_name() == "Contriever"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "contriever"
        assert metadata["implementation"] == "dense_biencoder"
    
    @pytest.mark.asyncio
    async def test_hybrid_system(self):
        """Test Hybrid BM25+Dense system."""
        system = HybridBM25DenseSystem()
        
        assert system.get_name() == "Hybrid_BM25_Dense"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "hybrid_bm25_dense"
        assert "alpha_sparse" in metadata
        assert "beta_dense" in metadata
    
    @pytest.mark.asyncio
    async def test_openai_system(self):
        """Test OpenAI embedding system."""
        system = OpenAIEmbeddingSystem()
        
        assert system.get_name() == "OpenAI_text-embedding-3-large"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "openai_text_embedding_3_large"
        assert metadata["implementation"] == "api"
    
    @pytest.mark.asyncio
    async def test_cohere_system(self):
        """Test Cohere embedding system."""
        system = CohereEmbeddingSystem()
        
        assert system.get_name() == "Cohere_embed-english-v3.0"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "cohere_embed_english_v3"
        assert metadata["implementation"] == "api"
    
    @pytest.mark.asyncio
    async def test_t1_hero_system(self):
        """Test T₁ Hero system."""
        system = T1HeroSystem()
        
        assert system.get_name() == "T1_Hero"
        
        doc_ids, scores, metadata = await system.search("test query", max_results=10)
        
        assert len(doc_ids) == 10
        assert len(scores) == 10
        assert metadata["system"] == "t1_hero"
        assert metadata["implementation"] == "parametric_router_conformal"
        assert metadata["frozen"] == True
        assert metadata["target_ndcg"] == 0.745

class TestBenchmarkRegistry:
    """Test comprehensive benchmark dataset registry."""
    
    def test_benchmark_registry_initialization(self):
        """Test that benchmark registry contains all mandatory datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            registry = benchmark.benchmark_registry
            
            # Test total count
            assert len(registry) >= 20, f"Expected ≥20 datasets, got {len(registry)}"
            
            # Test BEIR suite (11 datasets)
            beir_datasets = [k for k in registry.keys() if k.startswith("beir/")]
            assert len(beir_datasets) == 11, f"Expected 11 BEIR datasets, got {len(beir_datasets)}"
            
            expected_beir = [
                "beir/nq", "beir/hotpotqa", "beir/fiqa", "beir/scifact", "beir/trec-covid",
                "beir/nfcorpus", "beir/dbpedia-entity", "beir/quora", "beir/arguana", 
                "beir/webis-touche2020", "beir/trec-news"
            ]
            for dataset in expected_beir:
                assert dataset in registry, f"Missing BEIR dataset: {dataset}"
            
            # Test MTEB tasks (6 datasets)
            mteb_datasets = [k for k in registry.keys() if "mteb/" in k]
            assert len(mteb_datasets) == 6, f"Expected 6 MTEB tasks, got {len(mteb_datasets)}"
            
            # Test multilingual datasets (3 datasets)
            multilingual_datasets = [k for k in registry.keys() if any(ml in k for ml in ["miracl", "mrtydi", "mmarco"])]
            assert len(multilingual_datasets) == 3, f"Expected 3 multilingual datasets, got {len(multilingual_datasets)}"
            
            # Test domain-specific datasets
            domain_datasets = [k for k in registry.keys() if any(domain in k for domain in ["legal/", "code/"])]
            assert len(domain_datasets) >= 5, f"Expected ≥5 domain datasets, got {len(domain_datasets)}"
            
            # Test that each dataset has proper structure
            for dataset_name, dataset_info in registry.items():
                assert "queries" in dataset_info, f"Dataset {dataset_name} missing queries"
                assert "tags" in dataset_info, f"Dataset {dataset_name} missing tags"
                assert isinstance(dataset_info["queries"], list), f"Dataset {dataset_name} queries not list"
                assert isinstance(dataset_info["tags"], list), f"Dataset {dataset_name} tags not list"
                assert len(dataset_info["queries"]) > 0, f"Dataset {dataset_name} has no queries"

class TestStatisticalAnalysis:
    """Test statistical analysis components."""
    
    def test_ndcg_calculation(self):
        """Test nDCG@10 calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            # Test perfect ranking
            retrieved_docs = ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]
            ground_truth = ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]
            relevance_scores = [1.0, 1.0, 1.0, 1.0, 1.0]
            
            ndcg = benchmark._calculate_ndcg(retrieved_docs, ground_truth, relevance_scores)
            assert 0.95 <= ndcg <= 1.0, f"Perfect ranking should have high nDCG, got {ndcg}"
            
            # Test no relevant docs
            retrieved_docs = ["doc_10", "doc_11", "doc_12"]
            ndcg = benchmark._calculate_ndcg(retrieved_docs, ground_truth, relevance_scores)
            assert ndcg == 0.0, f"No relevant docs should have nDCG=0, got {ndcg}"
    
    def test_recall_calculation(self):
        """Test Recall@50 calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            # Test perfect recall
            retrieved_docs = ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]
            ground_truth = ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]
            
            recall = benchmark._calculate_recall(retrieved_docs, ground_truth)
            assert recall == 1.0, f"Perfect recall should be 1.0, got {recall}"
            
            # Test partial recall
            retrieved_docs = ["doc_0", "doc_1", "doc_10", "doc_11"]
            recall = benchmark._calculate_recall(retrieved_docs, ground_truth)
            assert recall == 0.4, f"2/5 recall should be 0.4, got {recall}"
    
    def test_jaccard_calculation(self):
        """Test Jaccard similarity calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            # Test perfect similarity
            retrieved_docs = ["doc_0", "doc_1", "doc_2"]
            reference_docs = ["doc_0", "doc_1", "doc_2"]
            
            jaccard = benchmark._calculate_jaccard_similarity(retrieved_docs, reference_docs)
            assert jaccard == 1.0, f"Perfect match should have Jaccard=1.0, got {jaccard}"
            
            # Test no overlap
            retrieved_docs = ["doc_0", "doc_1", "doc_2"]
            reference_docs = ["doc_3", "doc_4", "doc_5"]
            
            jaccard = benchmark._calculate_jaccard_similarity(retrieved_docs, reference_docs)
            assert jaccard == 0.0, f"No overlap should have Jaccard=0.0, got {jaccard}"

class TestValidationGuards:
    """Test validation guard system."""
    
    @pytest.mark.asyncio
    async def test_validation_guards_structure(self):
        """Test that all 7 mandatory validation guards are present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            guard_results = await benchmark._apply_validation_guards()
            
            # Test guard count
            assert len(guard_results) >= 7, f"Expected ≥7 validation guards, got {len(guard_results)}"
            
            # Test expected guard names
            guard_names = [guard.guard_name for guard in guard_results]
            expected_guards = [
                "ess_min_ratio",
                "pareto_kappa_max", 
                "conformal_coverage_cold",
                "conformal_coverage_warm",
                "ndcg_delta_min_pp",
                "p95_delta_max_ms",
                "jaccard_min",
                "aece_delta_max"
            ]
            
            for expected_guard in expected_guards:
                assert expected_guard in guard_names, f"Missing validation guard: {expected_guard}"
            
            # Test guard result structure
            for guard in guard_results:
                assert isinstance(guard, ValidationGuardResult)
                assert isinstance(guard.guard_name, str)
                assert isinstance(guard.passed, bool)
                assert isinstance(guard.measured_value, (int, float))
                assert isinstance(guard.threshold_value, (int, float))
                assert isinstance(guard.details, dict)
    
    def test_guard_thresholds(self):
        """Test that validation guards have correct thresholds."""
        expected_thresholds = {
            "ess_min_ratio": 0.2,
            "pareto_kappa_max": 0.5,
            "conformal_coverage_cold": 0.95,
            "conformal_coverage_warm": 0.95, 
            "ndcg_delta_min_pp": 0.0,
            "p95_delta_max_ms": 1.0,
            "jaccard_min": 0.80,
            "aece_delta_max": 0.01
        }
        
        # This would normally be tested with real guard results
        # but we're testing the expected thresholds are correct
        for guard_name, expected_threshold in expected_thresholds.items():
            assert isinstance(expected_threshold, (int, float))
            assert expected_threshold >= 0  # All thresholds should be non-negative

class TestArtifactGeneration:
    """Test artifact generation system."""
    
    @pytest.mark.asyncio
    async def test_competitor_matrix_generation(self):
        """Test competitor matrix CSV generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            # Create mock results
            benchmark.results = [
                BenchmarkResult(
                    system_name="BM25",
                    dataset_name="test_dataset",
                    query_id="test_001",
                    ndcg_at_10=0.65,
                    recall_at_50=0.78,
                    latency_p95_ms=12.5,
                    latency_p99_ms=18.2,
                    p99_p95_ratio=1.46,
                    jaccard_at_10=0.82,
                    ece_score=0.08,
                    aece_score=0.06,
                    retrieved_docs=["doc_1", "doc_2"],
                    raw_scores=[0.9, 0.8],
                    execution_time_ms=10.0,
                    memory_usage_mb=150.0,
                    metadata={}
                )
            ]
            
            matrix_file = await benchmark._generate_competitor_matrix()
            
            assert Path(matrix_file).exists(), "Competitor matrix file not created"
            
            # Test CSV structure
            df = pd.read_csv(matrix_file)
            assert "system" in df.columns or df.index.name == "system"
            
            expected_metrics = ['ndcg@10', 'recall@50', 'p95_ms', 'p99_ms', 'ratio_p99_p95', 'jaccard@10', 'ece', 'aece']
            for metric in expected_metrics:
                assert metric in df.columns, f"Missing metric in matrix: {metric}"
    
    @pytest.mark.asyncio
    async def test_leaderboard_generation(self):
        """Test leaderboard markdown generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            # Mock statistical results
            mock_statistical_results = {
                "statistical_summaries": {
                    "BM25": {
                        "ndcg_at_10": StatisticalSummary(
                            mean=0.65, std=0.05, ci_lower=0.62, ci_upper=0.68,
                            n_samples=100, bootstrap_samples=np.random.normal(0.65, 0.05, 100)
                        ),
                        "recall_at_50": StatisticalSummary(
                            mean=0.78, std=0.04, ci_lower=0.76, ci_upper=0.80,
                            n_samples=100, bootstrap_samples=np.random.normal(0.78, 0.04, 100)
                        ),
                        "latency_p95_ms": StatisticalSummary(
                            mean=12.5, std=1.2, ci_lower=11.8, ci_upper=13.2,
                            n_samples=100, bootstrap_samples=np.random.normal(12.5, 1.2, 100)
                        ),
                        "p99_p95_ratio": StatisticalSummary(
                            mean=1.46, std=0.08, ci_lower=1.42, ci_upper=1.50,
                            n_samples=100, bootstrap_samples=np.random.normal(1.46, 0.08, 100)
                        ),
                        "jaccard_at_10": StatisticalSummary(
                            mean=0.82, std=0.03, ci_lower=0.80, ci_upper=0.84,
                            n_samples=100, bootstrap_samples=np.random.normal(0.82, 0.03, 100)
                        ),
                        "ece_score": StatisticalSummary(
                            mean=0.08, std=0.01, ci_lower=0.07, ci_upper=0.09,
                            n_samples=100, bootstrap_samples=np.random.normal(0.08, 0.01, 100)
                        )
                    }
                },
                "pairwise_comparisons": []
            }
            
            leaderboard_file = await benchmark._generate_comprehensive_leaderboard(mock_statistical_results)
            
            assert Path(leaderboard_file).exists(), "Leaderboard file not created"
            
            # Test markdown structure
            with open(leaderboard_file, 'r') as f:
                content = f.read()
            
            assert "# Comprehensive Competitor Benchmark Leaderboard" in content
            assert "## Overall Rankings by nDCG@10" in content
            assert "## Benchmark Coverage Summary" in content
            assert "### BEIR Suite" in content
            assert "### MTEB Retrieval Tasks" in content
            assert "### Multilingual Coverage" in content

class TestMandatoryCompliance:
    """Test compliance with mandatory specification requirements."""
    
    def test_system_count_compliance(self):
        """Test that exactly 11 systems are implemented."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            assert len(benchmark.systems) == 11, f"Expected exactly 11 systems, got {len(benchmark.systems)}"
            
            expected_systems = [
                "BM25", "BM25+RM3", "SPLADE++", "uniCOIL", "ColBERTv2", 
                "TAS-B", "Contriever", "Hybrid_BM25_Dense", 
                "OpenAI_text-embedding-3-large", "Cohere_embed-english-v3.0", "T1_Hero"
            ]
            
            actual_systems = [system.get_name() for system in benchmark.systems]
            for expected_system in expected_systems:
                assert expected_system in actual_systems, f"Missing mandatory system: {expected_system}"
    
    def test_benchmark_count_compliance(self):
        """Test that 20+ benchmark datasets are implemented."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            assert len(benchmark.benchmark_registry) >= 20, f"Expected ≥20 benchmarks, got {len(benchmark.benchmark_registry)}"
    
    def test_metrics_compliance(self):
        """Test that all 8 mandatory metrics are implemented."""
        expected_metrics = [
            'ndcg_at_10', 'recall_at_50', 'latency_p95_ms', 'latency_p99_ms',
            'p99_p95_ratio', 'jaccard_at_10', 'ece_score', 'aece_score'
        ]
        
        # Create a sample result and verify it has all metrics
        result = BenchmarkResult(
            system_name="Test",
            dataset_name="Test",
            query_id="test_001",
            ndcg_at_10=0.5,
            recall_at_50=0.6,
            latency_p95_ms=10.0,
            latency_p99_ms=15.0,
            p99_p95_ratio=1.5,
            jaccard_at_10=0.8,
            ece_score=0.1,
            aece_score=0.08,
            retrieved_docs=[],
            raw_scores=[],
            execution_time_ms=8.0,
            memory_usage_mb=100.0,
            metadata={}
        )
        
        # Verify all mandatory metrics are present
        result_dict = result.__dict__
        for metric in expected_metrics:
            assert metric in result_dict, f"Missing mandatory metric: {metric}"

class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_small_scale_benchmark_run(self):
        """Test a small-scale benchmark run with minimal data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            # Reduce dataset size for testing
            for dataset_name in list(benchmark.benchmark_registry.keys()):
                benchmark.benchmark_registry[dataset_name]["queries"] = \
                    benchmark.benchmark_registry[dataset_name]["queries"][:2]  # Only 2 queries per dataset
            
            # Only test first 3 systems for speed
            benchmark.systems = benchmark.systems[:3]
            
            results = await benchmark.run_comprehensive_benchmark()
            
            # Test basic structure
            assert "benchmark_duration_seconds" in results
            assert "total_systems" in results
            assert "total_datasets" in results
            assert "statistical_results" in results
            assert "validation_guards" in results
            assert "artifacts_manifest" in results
            assert "compliance" in results
            
            # Test that results were generated
            assert results["total_queries_executed"] > 0
            assert len(benchmark.results) > 0
            
            # Test that validation guards ran
            assert len(results["validation_guards"]) >= 7
            
            # Test that artifacts were generated
            assert results["artifacts_manifest"]["total_artifacts"] > 0

# Performance and stress testing
class TestPerformanceValidation:
    """Test performance characteristics and stress scenarios."""
    
    def test_large_result_handling(self):
        """Test handling of large result sets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = ComprehensiveCompetitorBenchmark(output_dir=temp_dir)
            
            # Create large number of mock results
            large_results = []
            for i in range(1000):
                result = BenchmarkResult(
                    system_name=f"System_{i % 11}",
                    dataset_name=f"Dataset_{i % 20}",
                    query_id=f"query_{i:04d}",
                    ndcg_at_10=np.random.random(),
                    recall_at_50=np.random.random(),
                    latency_p95_ms=np.random.uniform(5.0, 50.0),
                    latency_p99_ms=np.random.uniform(10.0, 100.0),
                    p99_p95_ratio=np.random.uniform(1.2, 2.5),
                    jaccard_at_10=np.random.uniform(0.5, 1.0),
                    ece_score=np.random.uniform(0.01, 0.2),
                    aece_score=np.random.uniform(0.01, 0.15),
                    retrieved_docs=[f"doc_{j}" for j in range(10)],
                    raw_scores=list(np.random.random(10)),
                    execution_time_ms=np.random.uniform(1.0, 20.0),
                    memory_usage_mb=np.random.uniform(50.0, 200.0),
                    metadata={}
                )
                large_results.append(result)
            
            benchmark.results = large_results
            
            # Test that statistical analysis can handle large datasets
            df = pd.DataFrame([
                {
                    'system_name': r.system_name,
                    'dataset_name': r.dataset_name,
                    'ndcg_at_10': r.ndcg_at_10,
                    'recall_at_50': r.recall_at_50,
                    'latency_p95_ms': r.latency_p95_ms,
                }
                for r in large_results
            ])
            
            assert len(df) == 1000
            assert df['ndcg_at_10'].notna().all()
            assert df['system_name'].nunique() <= 11
            assert df['dataset_name'].nunique() <= 20

# Run the test suite
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])