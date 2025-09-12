#!/usr/bin/env python3
"""
Test Suite for Rigorous Competitor Benchmarking Framework

Comprehensive test suite that validates:
1. All 7 competitor systems work correctly
2. Statistical analysis produces valid results
3. Validation guards function properly
4. All mandatory artifacts are generated
5. Integration with Rust infrastructure works
6. Mathematical correctness of metrics

Run with: python test_rigorous_benchmark.py
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# Import the benchmarking framework
from rigorous_competitor_benchmark import (
    BM25BaselineSystem, BM25RM3System, ColBERTv2System, ANCESystem,
    HybridBM25DenseSystem, OpenAIAdaSystem, T1HeroSystem,
    RigorousCompetitorBenchmark, BenchmarkResult, ValidationGuardResult
)

from rust_integration import (
    LensT1HeroSystem, LensServerConfig, RustDatasetLoader,
    IntegratedRigorousBenchmark
)

class TestCompetitorSystems(unittest.TestCase):
    """Test all 7 competitor systems for basic functionality."""
    
    def setUp(self):
        self.test_query = "search test query"
        self.max_results = 20
    
    async def _test_system_interface(self, system):
        """Test that a system implements the required interface correctly."""
        
        # Test basic interface
        self.assertIsInstance(system.get_name(), str)
        self.assertTrue(len(system.get_name()) > 0)
        
        # Test configuration
        config = system.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("system", config)
        
        # Test warmup
        await system.warmup(["warmup query 1", "warmup query 2"])
        
        # Test search
        doc_ids, scores, metadata = await system.search(self.test_query, self.max_results)
        
        self.assertIsInstance(doc_ids, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(len(doc_ids), len(scores))
        self.assertLessEqual(len(doc_ids), self.max_results)
        
        # Validate scores are reasonable
        for score in scores:
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0)
        
        # Validate metadata
        self.assertIn("execution_time_ms", metadata)
        self.assertIsInstance(metadata["execution_time_ms"], (int, float))
        self.assertGreater(metadata["execution_time_ms"], 0)
    
    def test_bm25_baseline_system(self):
        """Test BM25 baseline system."""
        system = BM25BaselineSystem("./test_index")
        asyncio.run(self._test_system_interface(system))
    
    def test_bm25_rm3_system(self):
        """Test BM25 + RM3 system."""
        system = BM25RM3System("./test_index")
        asyncio.run(self._test_system_interface(system))
    
    def test_colbertv2_system(self):
        """Test ColBERTv2 system."""
        system = ColBERTv2System()
        asyncio.run(self._test_system_interface(system))
    
    def test_ance_system(self):
        """Test ANCE system."""
        system = ANCESystem()
        asyncio.run(self._test_system_interface(system))
    
    def test_hybrid_system(self):
        """Test Hybrid BM25+Dense system."""
        system = HybridBM25DenseSystem()
        asyncio.run(self._test_system_interface(system))
    
    def test_openai_ada_system(self):
        """Test OpenAI Ada system."""
        system = OpenAIAdaSystem()
        asyncio.run(self._test_system_interface(system))
    
    def test_t1_hero_system(self):
        """Test T1 Hero system."""
        system = T1HeroSystem()
        asyncio.run(self._test_system_interface(system))

class TestStatisticalAnalysis(unittest.TestCase):
    """Test statistical analysis components."""
    
    def test_ndcg_calculation(self):
        """Test nDCG calculation correctness."""
        benchmark = RigorousCompetitorBenchmark()
        
        # Perfect ranking
        retrieved = ["doc_1", "doc_2", "doc_3"]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        relevance = [1.0, 1.0, 1.0]
        
        ndcg = benchmark._calculate_ndcg(retrieved, ground_truth, relevance)
        self.assertAlmostEqual(ndcg, 1.0, places=3)
        
        # No relevant documents
        retrieved = ["doc_x", "doc_y", "doc_z"]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        relevance = [1.0, 1.0, 1.0]
        
        ndcg = benchmark._calculate_ndcg(retrieved, ground_truth, relevance)
        self.assertEqual(ndcg, 0.0)
        
        # Partial relevance
        retrieved = ["doc_1", "doc_x", "doc_2"]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        relevance = [1.0, 0.8, 0.6]
        
        ndcg = benchmark._calculate_ndcg(retrieved, ground_truth, relevance)
        self.assertGreater(ndcg, 0.0)
        self.assertLess(ndcg, 1.0)
    
    def test_recall_calculation(self):
        """Test recall calculation correctness."""
        benchmark = RigorousCompetitorBenchmark()
        
        # Perfect recall
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        
        recall = benchmark._calculate_recall(retrieved, ground_truth)
        self.assertAlmostEqual(recall, 1.0, places=3)
        
        # Zero recall
        retrieved = ["doc_x", "doc_y", "doc_z"]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        
        recall = benchmark._calculate_recall(retrieved, ground_truth)
        self.assertEqual(recall, 0.0)
        
        # Partial recall
        retrieved = ["doc_1", "doc_x", "doc_y"]
        ground_truth = ["doc_1", "doc_2", "doc_3"]
        
        recall = benchmark._calculate_recall(retrieved, ground_truth)
        self.assertAlmostEqual(recall, 1.0/3.0, places=3)
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        benchmark = RigorousCompetitorBenchmark()
        
        # Identical sets
        set_a = ["doc_1", "doc_2", "doc_3"]
        set_b = ["doc_1", "doc_2", "doc_3"]
        
        jaccard = benchmark._calculate_jaccard_similarity(set_a, set_b)
        self.assertAlmostEqual(jaccard, 1.0, places=3)
        
        # Disjoint sets
        set_a = ["doc_1", "doc_2", "doc_3"]
        set_b = ["doc_4", "doc_5", "doc_6"]
        
        jaccard = benchmark._calculate_jaccard_similarity(set_a, set_b)
        self.assertEqual(jaccard, 0.0)
        
        # Partial overlap
        set_a = ["doc_1", "doc_2", "doc_3"]
        set_b = ["doc_2", "doc_3", "doc_4"]
        
        jaccard = benchmark._calculate_jaccard_similarity(set_a, set_b)
        expected = 2.0 / 4.0  # 2 intersection, 4 union
        self.assertAlmostEqual(jaccard, expected, places=3)

class TestValidationGuards(unittest.TestCase):
    """Test validation guard implementations."""
    
    async def test_validation_guards_structure(self):
        """Test that validation guards return proper structure."""
        benchmark = RigorousCompetitorBenchmark()
        
        # Run validation guards
        guard_results = await benchmark._apply_validation_guards()
        
        # Verify structure
        self.assertIsInstance(guard_results, list)
        self.assertGreater(len(guard_results), 0)
        
        for guard in guard_results:
            self.assertIsInstance(guard, ValidationGuardResult)
            self.assertIsInstance(guard.guard_name, str)
            self.assertIsInstance(guard.passed, bool)
            self.assertIsInstance(guard.measured_value, (int, float))
            self.assertIsInstance(guard.threshold_value, (int, float))
            self.assertIsInstance(guard.details, dict)
    
    async def test_guard_names_and_thresholds(self):
        """Test that expected guards exist with correct thresholds."""
        benchmark = RigorousCompetitorBenchmark()
        guard_results = await benchmark._apply_validation_guards()
        
        guard_names = [guard.guard_name for guard in guard_results]
        
        # Required guards
        expected_guards = [
            "Counterfactual_ESS",
            "Conformal_Coverage_Cold", 
            "Conformal_Coverage_Warm",
            "Performance_nDCG_Improvement",
            "Performance_Latency_SLA",
            "Performance_Jaccard_Stability",
            "Performance_AECE_Calibration"
        ]
        
        for expected_guard in expected_guards:
            self.assertIn(expected_guard, guard_names, 
                         f"Required guard {expected_guard} not found")
        
        # Check specific thresholds
        for guard in guard_results:
            if guard.guard_name == "Counterfactual_ESS":
                self.assertEqual(guard.threshold_value, 0.2)
            elif guard.guard_name == "Performance_Latency_SLA":
                self.assertEqual(guard.threshold_value, 1.0)
            elif guard.guard_name == "Performance_Jaccard_Stability":
                self.assertEqual(guard.threshold_value, 0.80)

class TestArtifactGeneration(unittest.TestCase):
    """Test artifact generation and output."""
    
    async def test_csv_matrix_generation(self):
        """Test CSV matrix generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = RigorousCompetitorBenchmark(temp_dir)
            
            # Run a minimal benchmark to generate results
            await benchmark._warmup_all_systems()
            
            # Generate some mock results
            for system in benchmark.systems[:2]:  # Test with 2 systems
                for dataset_name in ["TestDataset"]:
                    result = BenchmarkResult(
                        system_name=system.get_name(),
                        dataset_name=dataset_name,
                        query_id="test_001",
                        ndcg_at_10=np.random.uniform(0.5, 0.9),
                        recall_at_50=np.random.uniform(0.6, 1.0),
                        latency_p95_ms=np.random.uniform(50, 200),
                        latency_p99_ms=np.random.uniform(80, 300),
                        jaccard_at_10=np.random.uniform(0.7, 0.9),
                        ece_score=np.random.uniform(0.05, 0.15),
                        aece_score=np.random.uniform(0.03, 0.12),
                        retrieved_docs=["doc_1", "doc_2", "doc_3"],
                        raw_scores=[0.9, 0.8, 0.7],
                        execution_time_ms=np.random.uniform(20, 100),
                        memory_usage_mb=np.random.uniform(100, 500)
                    )
                    benchmark.results.append(result)
            
            # Generate statistical analysis
            statistical_results = await benchmark._perform_statistical_analysis()
            
            # Generate CSV matrices
            await benchmark._generate_csv_matrices(statistical_results)
            
            # Check that files were created
            output_path = Path(temp_dir)
            self.assertTrue((output_path / "competitor_matrix.csv").exists())
            self.assertTrue((output_path / "ci_intervals.csv").exists())
            
            # Validate CSV content
            competitor_df = pd.read_csv(output_path / "competitor_matrix.csv")
            self.assertGreater(len(competitor_df), 0)
            self.assertIn("ndcg_at_10", competitor_df.columns)
            self.assertIn("recall_at_50", competitor_df.columns)
            
            ci_df = pd.read_csv(output_path / "ci_intervals.csv")
            self.assertGreater(len(ci_df), 0)
            self.assertIn("mean", ci_df.columns)
            self.assertIn("ci_lower", ci_df.columns)
            self.assertIn("ci_upper", ci_df.columns)
    
    async def test_plot_generation(self):
        """Test plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = RigorousCompetitorBenchmark(temp_dir)
            
            # Generate mock results
            for system in benchmark.systems[:3]:
                result = BenchmarkResult(
                    system_name=system.get_name(),
                    dataset_name="TestDataset",
                    query_id="test_001",
                    ndcg_at_10=np.random.uniform(0.5, 0.9),
                    recall_at_50=np.random.uniform(0.6, 1.0),
                    latency_p95_ms=np.random.uniform(50, 200),
                    latency_p99_ms=np.random.uniform(80, 300),
                    jaccard_at_10=np.random.uniform(0.7, 0.9),
                    ece_score=np.random.uniform(0.05, 0.15),
                    aece_score=np.random.uniform(0.03, 0.12),
                    retrieved_docs=["doc_1", "doc_2"],
                    raw_scores=[0.9, 0.8],
                    execution_time_ms=np.random.uniform(20, 100),
                    memory_usage_mb=np.random.uniform(100, 500)
                )
                benchmark.results.append(result)
            
            # Generate statistical analysis and plots
            statistical_results = await benchmark._perform_statistical_analysis()
            await benchmark._generate_plots(statistical_results)
            
            # Check that plot files were created
            plots_dir = Path(temp_dir) / "plots"
            self.assertTrue(plots_dir.exists())
            self.assertTrue((plots_dir / "scatter_ndcg_vs_p95.png").exists())
            self.assertTrue((plots_dir / "per_benchmark_bars.png").exists())
    
    async def test_leaderboard_generation(self):
        """Test leaderboard generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = RigorousCompetitorBenchmark(temp_dir)
            
            # Generate mock results for all systems
            for system in benchmark.systems:
                result = BenchmarkResult(
                    system_name=system.get_name(),
                    dataset_name="TestDataset",
                    query_id="test_001",
                    ndcg_at_10=np.random.uniform(0.5, 0.9),
                    recall_at_50=np.random.uniform(0.6, 1.0),
                    latency_p95_ms=np.random.uniform(50, 200),
                    latency_p99_ms=np.random.uniform(80, 300),
                    jaccard_at_10=np.random.uniform(0.7, 0.9),
                    ece_score=np.random.uniform(0.05, 0.15),
                    aece_score=np.random.uniform(0.03, 0.12),
                    retrieved_docs=["doc_1", "doc_2"],
                    raw_scores=[0.9, 0.8],
                    execution_time_ms=np.random.uniform(20, 100),
                    memory_usage_mb=np.random.uniform(100, 500)
                )
                benchmark.results.append(result)
            
            # Generate statistical analysis and leaderboard
            statistical_results = await benchmark._perform_statistical_analysis()
            await benchmark._generate_leaderboard(statistical_results)
            
            # Check that leaderboard was created
            leaderboard_path = Path(temp_dir) / "leaderboard.md"
            self.assertTrue(leaderboard_path.exists())
            
            # Validate leaderboard content
            with open(leaderboard_path, 'r') as f:
                content = f.read()
            
            self.assertIn("# Rigorous Competitor Benchmark Leaderboard", content)
            self.assertIn("| Rank | System |", content)
            self.assertIn("nDCG@10", content)
            self.assertIn("Statistical Notes", content)

class TestRustIntegration(unittest.TestCase):
    """Test integration with Rust infrastructure."""
    
    def test_lens_server_config(self):
        """Test Lens server configuration."""
        config = LensServerConfig(
            http_endpoint="http://localhost:3001",
            enable_lsp=True,
            enable_semantic=False
        )
        
        self.assertEqual(config.http_endpoint, "http://localhost:3001")
        self.assertTrue(config.enable_lsp)
        self.assertFalse(config.enable_semantic)
    
    def test_rust_dataset_loader_initialization(self):
        """Test RustDatasetLoader initialization."""
        loader = RustDatasetLoader("./test_datasets")
        
        self.assertEqual(str(loader.dataset_path), "./test_datasets")
        self.assertEqual(len(loader.available_datasets), 0)
    
    async def test_mock_dataset_discovery(self):
        """Test dataset discovery with mock files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock pinned dataset
            mock_dataset = {
                "version": "test_version",
                "created_at": "2024-01-01T00:00:00Z",
                "corpus_consistency_validated": True,
                "queries": [
                    {
                        "query": "test query 1",
                        "expected_files": ["file1.py", "file2.py"],
                        "relevance_scores": [1.0, 0.8]
                    }
                ]
            }
            
            dataset_path = Path(temp_dir)
            with open(dataset_path / "golden-pinned-test_version.json", 'w') as f:
                json.dump(mock_dataset, f)
            
            # Test discovery
            loader = RustDatasetLoader(temp_dir)
            discovered = await loader.discover_available_datasets()
            
            self.assertEqual(len(discovered), 1)
            self.assertIn("test_version", discovered)
            
            dataset_info = discovered["test_version"]
            self.assertEqual(dataset_info.total_queries, 1)
            self.assertTrue(dataset_info.consistency_validated)
            
            # Test loading
            loaded_queries = await loader.load_dataset("test_version")
            self.assertEqual(len(loaded_queries), 1)
            
            query = loaded_queries[0]
            self.assertEqual(query["query"], "test query 1")
            self.assertEqual(query["ground_truth"], ["file1.py", "file2.py"])

class TestEndToEndBenchmark(unittest.TestCase):
    """End-to-end integration tests."""
    
    async def test_minimal_benchmark_execution(self):
        """Test minimal benchmark execution with mock data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create benchmark with limited systems and datasets
            benchmark = RigorousCompetitorBenchmark(temp_dir)
            
            # Use only 2 systems for speed
            benchmark.systems = benchmark.systems[:2]
            
            # Use minimal dataset
            benchmark.datasets = {
                "TestSet": benchmark._create_mock_dataset("TestSet", 5)
            }
            
            # Execute benchmark
            results = await benchmark.run_comprehensive_benchmark()
            
            # Validate results structure
            self.assertIsInstance(results, dict)
            self.assertIn("benchmark_duration_seconds", results)
            self.assertIn("total_systems", results)
            self.assertIn("total_datasets", results)
            self.assertIn("statistical_results", results)
            self.assertIn("validation_guards", results)
            
            # Check that artifacts were created
            output_path = Path(temp_dir)
            self.assertTrue((output_path / "competitor_matrix.csv").exists())
            self.assertTrue((output_path / "leaderboard.md").exists())
            self.assertTrue((output_path / "plots").exists())
    
    async def test_system_count_validation(self):
        """Test that exactly 7 systems are configured."""
        benchmark = RigorousCompetitorBenchmark()
        
        self.assertEqual(len(benchmark.systems), 7)
        
        system_names = [system.get_name() for system in benchmark.systems]
        expected_systems = [
            "BM25_Baseline",
            "BM25_RM3", 
            "ColBERTv2",
            "ANCE",
            "Hybrid_BM25_Dense",
            "OpenAI_Ada",
            "T1_Hero"
        ]
        
        for expected in expected_systems:
            self.assertIn(expected, system_names)

class TestBenchmarkRunner:
    """Test runner that executes all tests."""
    
    def run_all_tests(self):
        """Run all test suites."""
        test_suites = [
            TestCompetitorSystems,
            TestStatisticalAnalysis,
            TestValidationGuards,
            TestArtifactGeneration,
            TestRustIntegration,
            TestEndToEndBenchmark
        ]
        
        total_tests = 0
        passed_tests = 0
        failed_tests = []
        
        for test_class in test_suites:
            print(f"\n{'='*60}")
            print(f"Running {test_class.__name__}")
            print('='*60)
            
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            for test in suite:
                total_tests += 1
                test_name = f"{test_class.__name__}.{test._testMethodName}"
                
                try:
                    # Handle async tests
                    if asyncio.iscoroutinefunction(getattr(test, test._testMethodName)):
                        asyncio.run(test.debug())
                    else:
                        test.debug()
                    
                    print(f"✅ {test_name}")
                    passed_tests += 1
                    
                except Exception as e:
                    print(f"❌ {test_name}: {e}")
                    failed_tests.append((test_name, str(e)))
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print('='*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Success Rate: {100 * passed_tests / total_tests:.1f}%")
        
        if failed_tests:
            print("\nFailed Tests:")
            for test_name, error in failed_tests:
                print(f"  ❌ {test_name}: {error}")
        
        return len(failed_tests) == 0

async def main():
    """Main test runner."""
    print("🧪 RIGOROUS COMPETITOR BENCHMARK TEST SUITE")
    print("="*80)
    print("Testing all components of the rigorous benchmarking framework...")
    print("This validates:")
    print("• All 7 competitor systems function correctly")
    print("• Statistical analysis produces valid results")  
    print("• Validation guards work properly")
    print("• All mandatory artifacts are generated")
    print("• Mathematical correctness of metrics")
    print("• Integration with Rust infrastructure")
    
    runner = TestBenchmarkRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Benchmark framework is ready for production use!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Please review and fix issues before deployment")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))