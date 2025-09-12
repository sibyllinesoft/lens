#!/usr/bin/env python3
"""
Quick Validation Script for Rigorous Competitor Benchmarking Framework

Performs rapid validation to ensure the framework is correctly installed
and all components function properly before running the full benchmark.

Usage: python validate_framework.py
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_imports() -> Dict[str, Any]:
    """Check that all required imports are available."""
    checks = {}
    
    # Core framework imports
    try:
        from rigorous_competitor_benchmark import RigorousCompetitorBenchmark
        checks['rigorous_framework'] = {"status": "✅", "details": "Main framework imported"}
    except Exception as e:
        checks['rigorous_framework'] = {"status": "❌", "details": f"Import failed: {e}"}
    
    try:
        from rust_integration import IntegratedRigorousBenchmark, LensServerConfig
        checks['rust_integration'] = {"status": "✅", "details": "Rust integration imported"}
    except Exception as e:
        checks['rust_integration'] = {"status": "❌", "details": f"Import failed: {e}"}
    
    try:
        from test_rigorous_benchmark import TestBenchmarkRunner
        checks['test_framework'] = {"status": "✅", "details": "Test framework imported"}
    except Exception as e:
        checks['test_framework'] = {"status": "❌", "details": f"Import failed: {e}"}
    
    # Required dependencies
    required_deps = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'aiohttp', 'statsmodels'
    ]
    
    for dep in required_deps:
        try:
            __import__(dep)
            checks[f'dep_{dep}'] = {"status": "✅", "details": f"Dependency available"}
        except ImportError:
            checks[f'dep_{dep}'] = {"status": "❌", "details": f"Dependency missing"}
    
    return checks

async def check_competitor_systems() -> Dict[str, Any]:
    """Check that all 7 competitor systems initialize correctly."""
    checks = {}
    
    try:
        from rigorous_competitor_benchmark import (
            BM25BaselineSystem, BM25RM3System, ColBERTv2System, ANCESystem,
            HybridBM25DenseSystem, OpenAIAdaSystem, T1HeroSystem
        )
        
        systems = [
            BM25BaselineSystem("./test"),
            BM25RM3System("./test"),
            ColBERTv2System(),
            ANCESystem(),
            HybridBM25DenseSystem(),
            OpenAIAdaSystem(),
            T1HeroSystem()
        ]
        
        for system in systems:
            system_name = system.get_name()
            try:
                # Test basic interface
                config = system.get_config()
                assert isinstance(config, dict)
                assert 'system' in config
                
                # Test search interface (with mock data)
                doc_ids, scores, metadata = await system.search("test query", 5)
                assert isinstance(doc_ids, list)
                assert isinstance(scores, list)
                assert isinstance(metadata, dict)
                assert len(doc_ids) == len(scores)
                
                checks[f'system_{system_name}'] = {
                    "status": "✅", 
                    "details": f"System interface working"
                }
                
            except Exception as e:
                checks[f'system_{system_name}'] = {
                    "status": "❌", 
                    "details": f"System failed: {e}"
                }
    
    except Exception as e:
        checks['systems_import'] = {"status": "❌", "details": f"Failed to import systems: {e}"}
    
    return checks

async def check_statistical_functions() -> Dict[str, Any]:
    """Check that statistical analysis functions work correctly."""
    checks = {}
    
    try:
        from rigorous_competitor_benchmark import RigorousCompetitorBenchmark
        import numpy as np
        
        benchmark = RigorousCompetitorBenchmark()
        
        # Test nDCG calculation
        try:
            retrieved = ["doc_1", "doc_2", "doc_3"]
            ground_truth = ["doc_1", "doc_2", "doc_3"]
            relevance = [1.0, 1.0, 1.0]
            
            ndcg = benchmark._calculate_ndcg(retrieved, ground_truth, relevance)
            assert abs(ndcg - 1.0) < 0.001, f"Perfect nDCG should be 1.0, got {ndcg}"
            
            checks['ndcg_calculation'] = {"status": "✅", "details": "nDCG calculation correct"}
            
        except Exception as e:
            checks['ndcg_calculation'] = {"status": "❌", "details": f"nDCG failed: {e}"}
        
        # Test recall calculation
        try:
            retrieved = ["doc_1", "doc_2", "doc_x"]
            ground_truth = ["doc_1", "doc_2", "doc_3"]
            
            recall = benchmark._calculate_recall(retrieved, ground_truth)
            expected = 2.0 / 3.0
            assert abs(recall - expected) < 0.001, f"Expected recall {expected}, got {recall}"
            
            checks['recall_calculation'] = {"status": "✅", "details": "Recall calculation correct"}
            
        except Exception as e:
            checks['recall_calculation'] = {"status": "❌", "details": f"Recall failed: {e}"}
        
        # Test Jaccard similarity
        try:
            set_a = ["doc_1", "doc_2", "doc_3"]
            set_b = ["doc_1", "doc_2", "doc_3"]
            
            jaccard = benchmark._calculate_jaccard_similarity(set_a, set_b)
            assert abs(jaccard - 1.0) < 0.001, f"Identical sets should have Jaccard=1.0, got {jaccard}"
            
            checks['jaccard_calculation'] = {"status": "✅", "details": "Jaccard calculation correct"}
            
        except Exception as e:
            checks['jaccard_calculation'] = {"status": "❌", "details": f"Jaccard failed: {e}"}
    
    except Exception as e:
        checks['statistical_import'] = {"status": "❌", "details": f"Failed to import statistical functions: {e}"}
    
    return checks

async def check_validation_guards() -> Dict[str, Any]:
    """Check that validation guards function correctly."""
    checks = {}
    
    try:
        from rigorous_competitor_benchmark import RigorousCompetitorBenchmark
        
        benchmark = RigorousCompetitorBenchmark()
        
        # Test validation guard structure
        guard_results = await benchmark._apply_validation_guards()
        
        if not isinstance(guard_results, list) or len(guard_results) == 0:
            checks['guards_structure'] = {"status": "❌", "details": "No validation guards returned"}
        else:
            # Check required guards exist
            guard_names = [guard.guard_name for guard in guard_results]
            required_guards = [
                "Counterfactual_ESS",
                "Conformal_Coverage_Cold",
                "Conformal_Coverage_Warm", 
                "Performance_nDCG_Improvement",
                "Performance_Latency_SLA",
                "Performance_Jaccard_Stability",
                "Performance_AECE_Calibration"
            ]
            
            missing_guards = []
            for required_guard in required_guards:
                if required_guard not in guard_names:
                    missing_guards.append(required_guard)
            
            if missing_guards:
                checks['guards_completeness'] = {
                    "status": "❌", 
                    "details": f"Missing guards: {missing_guards}"
                }
            else:
                checks['guards_completeness'] = {
                    "status": "✅", 
                    "details": f"All {len(required_guards)} required guards present"
                }
            
            # Check guard structure
            for guard in guard_results:
                if not all(hasattr(guard, attr) for attr in ['guard_name', 'passed', 'measured_value', 'threshold_value']):
                    checks['guards_structure'] = {"status": "❌", "details": "Invalid guard structure"}
                    break
            else:
                checks['guards_structure'] = {"status": "✅", "details": "Validation guard structure correct"}
    
    except Exception as e:
        checks['guards_error'] = {"status": "❌", "details": f"Validation guards failed: {e}"}
    
    return checks

async def check_artifact_generation() -> Dict[str, Any]:
    """Check that artifacts can be generated correctly."""
    checks = {}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            from rigorous_competitor_benchmark import RigorousCompetitorBenchmark, BenchmarkResult
            import numpy as np
            
            benchmark = RigorousCompetitorBenchmark(temp_dir)
            
            # Generate minimal mock results
            benchmark.results = [
                BenchmarkResult(
                    system_name="TestSystem1",
                    dataset_name="TestDataset", 
                    query_id="test_001",
                    ndcg_at_10=0.75,
                    recall_at_50=0.85,
                    latency_p95_ms=120.0,
                    latency_p99_ms=180.0,
                    jaccard_at_10=0.8,
                    ece_score=0.1,
                    aece_score=0.08,
                    retrieved_docs=["doc1", "doc2"],
                    raw_scores=[0.9, 0.8],
                    execution_time_ms=95.0,
                    memory_usage_mb=256.0
                ),
                BenchmarkResult(
                    system_name="TestSystem2",
                    dataset_name="TestDataset",
                    query_id="test_001", 
                    ndcg_at_10=0.68,
                    recall_at_50=0.78,
                    latency_p95_ms=95.0,
                    latency_p99_ms=135.0,
                    jaccard_at_10=0.75,
                    ece_score=0.12,
                    aece_score=0.09,
                    retrieved_docs=["doc1", "doc3"],
                    raw_scores=[0.85, 0.7],
                    execution_time_ms=80.0,
                    memory_usage_mb=198.0
                )
            ]
            
            # Test statistical analysis
            try:
                statistical_results = await benchmark._perform_statistical_analysis()
                assert 'statistical_summaries' in statistical_results
                assert 'pairwise_comparisons' in statistical_results
                checks['statistical_analysis'] = {"status": "✅", "details": "Statistical analysis working"}
            except Exception as e:
                checks['statistical_analysis'] = {"status": "❌", "details": f"Statistical analysis failed: {e}"}
            
            # Test CSV generation
            try:
                await benchmark._generate_csv_matrices(statistical_results)
                
                output_path = Path(temp_dir)
                csv_files = ['competitor_matrix.csv', 'ci_intervals.csv']
                missing_files = [f for f in csv_files if not (output_path / f).exists()]
                
                if missing_files:
                    checks['csv_generation'] = {"status": "❌", "details": f"Missing files: {missing_files}"}
                else:
                    checks['csv_generation'] = {"status": "✅", "details": "CSV matrices generated"}
                    
            except Exception as e:
                checks['csv_generation'] = {"status": "❌", "details": f"CSV generation failed: {e}"}
            
            # Test plot generation
            try:
                await benchmark._generate_plots(statistical_results)
                
                plots_dir = output_path / "plots"
                plot_files = ['scatter_ndcg_vs_p95.png', 'per_benchmark_bars.png']
                missing_plots = [f for f in plot_files if not (plots_dir / f).exists()]
                
                if missing_plots:
                    checks['plot_generation'] = {"status": "⚠️", "details": f"Some plots missing: {missing_plots}"}
                else:
                    checks['plot_generation'] = {"status": "✅", "details": "Plots generated"}
                    
            except Exception as e:
                checks['plot_generation'] = {"status": "❌", "details": f"Plot generation failed: {e}"}
            
            # Test leaderboard generation
            try:
                await benchmark._generate_leaderboard(statistical_results)
                
                if (output_path / "leaderboard.md").exists():
                    checks['leaderboard_generation'] = {"status": "✅", "details": "Leaderboard generated"}
                else:
                    checks['leaderboard_generation'] = {"status": "❌", "details": "Leaderboard file missing"}
                    
            except Exception as e:
                checks['leaderboard_generation'] = {"status": "❌", "details": f"Leaderboard generation failed: {e}"}
    
    except Exception as e:
        checks['artifact_setup'] = {"status": "❌", "details": f"Artifact generation setup failed: {e}"}
    
    return checks

async def run_comprehensive_validation():
    """Run all validation checks."""
    print("🔍 RIGOROUS COMPETITOR BENCHMARK FRAMEWORK VALIDATION")
    print("="*70)
    print("Checking framework installation and functionality...\n")
    
    all_checks = {}
    
    # Import checks
    print("📦 Checking imports and dependencies...")
    import_checks = check_imports()
    all_checks.update(import_checks)
    
    # System checks
    print("🤖 Checking competitor systems...")
    system_checks = await check_competitor_systems()
    all_checks.update(system_checks)
    
    # Statistical function checks
    print("📊 Checking statistical functions...")
    stats_checks = await check_statistical_functions()
    all_checks.update(stats_checks)
    
    # Validation guard checks
    print("🛡️ Checking validation guards...")
    guard_checks = await check_validation_guards()
    all_checks.update(guard_checks)
    
    # Artifact generation checks
    print("📈 Checking artifact generation...")
    artifact_checks = await check_artifact_generation()
    all_checks.update(artifact_checks)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for check in all_checks.values() if check['status'] == '✅')
    warned = sum(1 for check in all_checks.values() if check['status'] == '⚠️')
    failed = sum(1 for check in all_checks.values() if check['status'] == '❌')
    total = len(all_checks)
    
    print(f"Total Checks: {total}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️ Warnings: {warned}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {100 * passed / total:.1f}%")
    
    # Detailed results
    if failed > 0 or warned > 0:
        print("\n📋 Detailed Results:")
        for check_name, result in all_checks.items():
            if result['status'] != '✅':
                print(f"  {result['status']} {check_name}: {result['details']}")
    
    # Overall assessment
    print(f"\n🎯 Framework Status:")
    if failed == 0:
        if warned == 0:
            print("🌟 FULLY READY - All checks passed, framework ready for production")
            return 0
        else:
            print("⚠️  MOSTLY READY - Minor issues detected, framework functional")
            return 1
    else:
        print("❌ NOT READY - Critical issues detected, please fix before use")
        print("\n🔧 Recommended Actions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python version compatibility (3.8+)")
        print("3. Verify all framework files are present")
        return 2

def main():
    """Main validation entry point."""
    try:
        exit_code = asyncio.run(run_comprehensive_validation())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()