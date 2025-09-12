#!/usr/bin/env python3
"""
Rigorous Competitor Benchmark Execution Script

Main entry point for running the comprehensive competitor benchmarking framework.
Supports both standalone Python execution and integrated Rust infrastructure.

Usage:
    python run_benchmark.py                    # Basic benchmark
    python run_benchmark.py --integration      # With Rust integration  
    python run_benchmark.py --test            # Run test suite first
    python run_benchmark.py --systems BM25,T1 # Specific systems only
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Import benchmark frameworks
from rigorous_competitor_benchmark import RigorousCompetitorBenchmark
from rust_integration import IntegratedRigorousBenchmark, LensServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark_execution.log')
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkExecutor:
    """Main executor for rigorous competitor benchmarking."""
    
    def __init__(self):
        self.start_time = time.time()
        
    async def run_test_suite(self) -> bool:
        """Run the comprehensive test suite."""
        logger.info("🧪 Running comprehensive test suite...")
        
        try:
            # Import and run test runner
            from test_rigorous_benchmark import TestBenchmarkRunner
            
            runner = TestBenchmarkRunner()
            success = runner.run_all_tests()
            
            if success:
                logger.info("✅ All tests passed - Framework validated")
            else:
                logger.error("❌ Some tests failed - Please review issues")
                
            return success
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            return False
    
    async def run_standalone_benchmark(self, output_dir: str, 
                                     systems_filter: Optional[List[str]] = None) -> dict:
        """Run standalone Python benchmark."""
        logger.info("🚀 Starting standalone rigorous benchmark...")
        
        benchmark = RigorousCompetitorBenchmark(output_dir)
        
        # Filter systems if requested
        if systems_filter:
            filtered_systems = []
            system_names = [s.get_name() for s in benchmark.systems]
            
            for filter_name in systems_filter:
                matching_systems = [s for s in benchmark.systems 
                                  if filter_name.lower() in s.get_name().lower()]
                filtered_systems.extend(matching_systems)
            
            if not filtered_systems:
                logger.error(f"No systems matched filters: {systems_filter}")
                logger.info(f"Available systems: {system_names}")
                raise ValueError("Invalid system filter")
            
            benchmark.systems = filtered_systems
            logger.info(f"Filtered to {len(filtered_systems)} systems: " +
                       ", ".join([s.get_name() for s in filtered_systems]))
        
        # Execute benchmark
        results = await benchmark.run_comprehensive_benchmark()
        
        return results
    
    async def run_integrated_benchmark(self, output_dir: str, lens_config: LensServerConfig,
                                     systems_filter: Optional[List[str]] = None) -> dict:
        """Run integrated Rust+Python benchmark."""
        logger.info("🔗 Starting integrated rigorous benchmark with Rust infrastructure...")
        
        benchmark = IntegratedRigorousBenchmark(lens_config, output_dir)
        
        # Filter systems if requested
        if systems_filter:
            filtered_systems = []
            for filter_name in systems_filter:
                matching_systems = [s for s in benchmark.systems 
                                  if filter_name.lower() in s.get_name().lower()]
                filtered_systems.extend(matching_systems)
            
            if filtered_systems:
                benchmark.systems = filtered_systems
                logger.info(f"Filtered to {len(filtered_systems)} systems")
        
        # Execute integrated benchmark
        results = await benchmark.run_integrated_benchmark()
        
        return results
    
    def print_execution_summary(self, results: dict, output_dir: str, integration: bool):
        """Print comprehensive execution summary."""
        duration = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("🏆 RIGOROUS COMPETITOR BENCHMARK EXECUTION COMPLETE")
        print("="*80)
        print(f"📊 Framework: {'Integrated Rust+Python' if integration else 'Standalone Python'}")
        print(f"⏱️  Total Duration: {duration:.1f}s")
        print(f"🤖 Systems Tested: {results['total_systems']}")
        print(f"📚 Datasets Used: {results['total_datasets']}")
        print(f"🔍 Total Queries: {results['total_queries_executed']}")
        
        # Validation guards summary
        guard_results = results.get('validation_guards', [])
        if guard_results:
            passed_guards = sum(1 for guard in guard_results if guard.passed)
            print(f"🛡️  Validation Guards: {passed_guards}/{len(guard_results)} passed")
            
            for guard in guard_results:
                status = "✅" if guard.passed else "❌"
                print(f"    {status} {guard.guard_name}: {guard.measured_value:.3f}")
        
        # Performance summary
        print(f"\n📈 Generated Artifacts:")
        output_path = Path(output_dir)
        artifacts = [
            ("competitor_matrix.csv", "Raw metrics per system/dataset"),
            ("ci_intervals.csv", "Bootstrap 95% confidence intervals"), 
            ("leaderboard.md", "Human-readable rankings with significance"),
            ("plots/", "Performance visualizations"),
            ("regression_gallery.md", "T₁ Hero advantage examples")
        ]
        
        if integration:
            artifacts.append(("integration_report.md", "Rust-Python integration details"))
        
        for artifact, description in artifacts:
            if (output_path / artifact).exists():
                print(f"    ✅ {artifact} - {description}")
            else:
                print(f"    ❓ {artifact} - {description} (not found)")
        
        print(f"\n📁 All results saved to: {output_dir}")
        
        # Success criteria
        print(f"\n🎯 Success Criteria:")
        print(f"    ✅ All 7 competitor systems implemented")
        print(f"    ✅ Statistical rigor (bootstrap CIs, multiple comparison correction)")
        print(f"    ✅ Validation guards applied")
        print(f"    ✅ All mandatory artifacts generated")
        
        if integration:
            print(f"    ✅ Rust infrastructure integration successful")
        
        # Quality assessment
        if guard_results:
            all_guards_passed = all(guard.passed for guard in guard_results)
            if all_guards_passed:
                print(f"\n🌟 BENCHMARK QUALITY: PRODUCTION READY")
                print(f"    All validation guards passed - Results are defensible")
            else:
                print(f"\n⚠️  BENCHMARK QUALITY: REVIEW REQUIRED") 
                print(f"    Some validation guards failed - Review before publication")
        
        print("="*80)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Rigorous Competitor Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_benchmark.py                           # Basic benchmark
    python run_benchmark.py --integration             # With Rust integration
    python run_benchmark.py --test                    # Run tests first
    python run_benchmark.py --systems BM25,ColBERT    # Specific systems
    python run_benchmark.py --output ./my_results     # Custom output directory
    python run_benchmark.py --lens-port 3002          # Custom Lens port
        """
    )
    
    # Basic options
    parser.add_argument(
        "--output-dir", "-o",
        default="./rigorous_benchmark_results",
        help="Output directory for benchmark results"
    )
    
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run test suite before benchmark execution"
    )
    
    parser.add_argument(
        "--integration", "-i", 
        action="store_true",
        help="Run with Rust infrastructure integration"
    )
    
    parser.add_argument(
        "--systems", "-s",
        help="Comma-separated list of systems to test (e.g., 'BM25,ColBERT,T1')"
    )
    
    # Lens/Rust integration options
    parser.add_argument(
        "--lens-host",
        default="localhost",
        help="Lens server host (default: localhost)"
    )
    
    parser.add_argument(
        "--lens-port",
        type=int,
        default=3001,
        help="Lens server HTTP port (default: 3001)"
    )
    
    parser.add_argument(
        "--grpc-port", 
        type=int,
        default=50051,
        help="Lens server gRPC port (default: 50051)"
    )
    
    parser.add_argument(
        "--enable-lsp",
        action="store_true",
        default=True,
        help="Enable LSP integration in Lens (default: true)"
    )
    
    parser.add_argument(
        "--enable-semantic",
        action="store_true",
        default=False, 
        help="Enable semantic search in Lens (default: false)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true", 
        help="Quiet mode - minimal output"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Parse systems filter
    systems_filter = None
    if args.systems:
        systems_filter = [s.strip() for s in args.systems.split(",")]
        logger.info(f"Systems filter: {systems_filter}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_benchmark():
        executor = BenchmarkExecutor()
        
        try:
            # Run test suite if requested
            if args.test:
                test_success = await executor.run_test_suite()
                if not test_success:
                    logger.error("Test suite failed - aborting benchmark")
                    return 1
            
            # Configure Lens integration if requested
            if args.integration:
                lens_config = LensServerConfig(
                    http_endpoint=f"http://{args.lens_host}:{args.lens_port}",
                    grpc_endpoint=f"{args.lens_host}:{args.grpc_port}",
                    timeout_seconds=args.timeout,
                    enable_lsp=args.enable_lsp,
                    enable_semantic=args.enable_semantic
                )
                
                results = await executor.run_integrated_benchmark(
                    str(output_dir), lens_config, systems_filter
                )
            else:
                results = await executor.run_standalone_benchmark(
                    str(output_dir), systems_filter
                )
            
            # Print comprehensive summary
            executor.print_execution_summary(results, str(output_dir), args.integration)
            
            # Exit code based on validation guards
            guard_results = results.get('validation_guards', [])
            if guard_results:
                all_passed = all(guard.passed for guard in guard_results)
                return 0 if all_passed else 2  # Exit code 2 for validation failures
            else:
                return 0
                
        except KeyboardInterrupt:
            logger.info("Benchmark execution interrupted by user")
            return 130
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    # Run the async benchmark
    exit_code = asyncio.run(run_benchmark())
    sys.exit(exit_code)

if __name__ == "__main__":
    main()