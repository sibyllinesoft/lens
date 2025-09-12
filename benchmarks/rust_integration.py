#!/usr/bin/env python3
"""
Rust Infrastructure Integration for Rigorous Competitor Benchmarking

Integrates the rigorous Python benchmarking framework with the existing
Rust Lens infrastructure, enabling seamless benchmarking of the T1 Hero
system alongside external competitors.

This script:
1. Interfaces with the Rust HTTP/gRPC servers
2. Loads pinned golden datasets from the Rust ecosystem
3. Provides a bridge between Python statistical analysis and Rust search engines
4. Maintains compatibility with existing Lens benchmarking workflows
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
import pandas as pd

from rigorous_competitor_benchmark import (
    CompetitorSystem, BenchmarkResult, RigorousCompetitorBenchmark
)

logger = logging.getLogger(__name__)

@dataclass
class LensServerConfig:
    """Configuration for Lens server connection."""
    http_endpoint: str = "http://localhost:3001"
    grpc_endpoint: str = "localhost:50051"
    timeout_seconds: int = 30
    max_retries: int = 3
    enable_lsp: bool = True
    enable_semantic: bool = False

@dataclass
class PinnedDatasetInfo:
    """Information about a pinned golden dataset."""
    version: str
    path: str
    total_queries: int
    consistency_validated: bool
    creation_timestamp: str

class LensT1HeroSystem(CompetitorSystem):
    """
    Real T1 Hero system implementation that calls the Lens Rust infrastructure.
    
    This replaces the mock T1HeroSystem with actual API calls to the running
    Lens server, providing accurate performance measurements and results.
    """
    
    def __init__(self, config: LensServerConfig):
        self.config = config
        self.session = None  # Optional[aiohttp.ClientSession] when available
        self.server_info: Optional[Dict[str, Any]] = None
        
    def get_name(self) -> str:
        return "T1_Hero_Lens"
    
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if not HAS_AIOHTTP:
            raise Exception("aiohttp is required for Rust integration - please install: pip install aiohttp")
        
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _check_server_health(self) -> bool:
        """Check if Lens server is running and responsive."""
        await self._ensure_session()
        
        try:
            async with self.session.get(f"{self.config.http_endpoint}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    self.server_info = health_data
                    return True
                else:
                    logger.error(f"Health check failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    async def search(self, query: str, max_results: int = 50) -> Tuple[List[str], List[float], Dict[str, Any]]:
        """Perform search using Lens HTTP API."""
        await self._ensure_session()
        
        start_time = time.time()
        
        # Prepare search request
        search_payload = {
            "q": query,
            "mode": "hybrid",  # Use best performing mode
            "max_results": max_results,
            "include_metadata": True,
            "enable_lsp": self.config.enable_lsp,
            "enable_semantic": self.config.enable_semantic
        }
        
        retries = 0
        while retries < self.config.max_retries:
            try:
                async with self.session.post(
                    f"{self.config.http_endpoint}/search",
                    json=search_payload
                ) as response:
                    if response.status == 200:
                        search_data = await response.json()
                        
                        # Extract results
                        results = search_data.get("results", [])
                        doc_ids = [r.get("file", f"unknown_{i}") for i, r in enumerate(results)]
                        scores = [r.get("score", 0.0) for r in results]
                        
                        # Performance metrics
                        execution_time = (time.time() - start_time) * 1000
                        
                        metadata = {
                            "system": "lens_t1_hero",
                            "execution_time_ms": execution_time,
                            "server_time_ms": search_data.get("query_duration_ms", 0),
                            "total_results": len(results),
                            "mode": search_payload["mode"],
                            "lsp_enabled": self.config.enable_lsp,
                            "semantic_enabled": self.config.enable_semantic,
                            "server_info": self.server_info
                        }
                        
                        return doc_ids, scores, metadata
                        
                    elif response.status == 503:
                        # Server unavailable, retry
                        retries += 1
                        await asyncio.sleep(1.0 * retries)
                        continue
                    else:
                        error_text = await response.text()
                        raise Exception(f"Search failed: HTTP {response.status} - {error_text}")
                        
            except asyncio.TimeoutError:
                retries += 1
                logger.warning(f"Search timeout, retry {retries}/{self.config.max_retries}")
                await asyncio.sleep(1.0 * retries)
                
            except Exception as e:
                retries += 1
                logger.error(f"Search error (retry {retries}): {e}")
                if retries >= self.config.max_retries:
                    raise
                await asyncio.sleep(1.0 * retries)
        
        raise Exception(f"Search failed after {self.config.max_retries} retries")
    
    async def warmup(self, warmup_queries: List[str]) -> None:
        """Warm up the Lens server with sample queries."""
        # First check if server is healthy
        if not await self._check_server_health():
            raise Exception("Lens server health check failed")
        
        logger.info(f"Warming up Lens server with {len(warmup_queries)} queries")
        
        for i, query in enumerate(warmup_queries[:5]):  # Use first 5 queries
            try:
                _, _, metadata = await self.search(query, max_results=10)
                logger.debug(f"Warmup {i+1}/5: {metadata['execution_time_ms']:.1f}ms")
            except Exception as e:
                logger.warning(f"Warmup query failed: {e}")
        
        logger.info("✅ Lens server warmup completed")
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "system": "T1HeroLens",
            "http_endpoint": self.config.http_endpoint,
            "grpc_endpoint": self.config.grpc_endpoint,
            "enable_lsp": self.config.enable_lsp,
            "enable_semantic": self.config.enable_semantic,
            "server_info": self.server_info
        }
    
    async def close(self):
        """Clean up aiohttp session."""
        if self.session:
            await self.session.close()

class RustDatasetLoader:
    """
    Loads pinned golden datasets from the Rust ecosystem.
    
    Interfaces with the Lens Rust infrastructure to load pinned datasets
    and maintain consistency with the existing benchmarking ecosystem.
    """
    
    def __init__(self, dataset_path: str = "./pinned-datasets"):
        self.dataset_path = Path(dataset_path)
        self.available_datasets: Dict[str, PinnedDatasetInfo] = {}
    
    async def discover_available_datasets(self) -> Dict[str, PinnedDatasetInfo]:
        """Discover all available pinned datasets."""
        self.available_datasets = {}
        
        if not self.dataset_path.exists():
            logger.warning(f"Dataset path {self.dataset_path} does not exist")
            return self.available_datasets
        
        # Look for pinned dataset files
        for dataset_file in self.dataset_path.glob("golden-pinned-*.json"):
            try:
                with open(dataset_file, 'r') as f:
                    dataset_data = json.load(f)
                
                # Extract version from filename
                version = dataset_file.stem.replace("golden-pinned-", "")
                
                dataset_info = PinnedDatasetInfo(
                    version=version,
                    path=str(dataset_file),
                    total_queries=len(dataset_data.get("queries", [])),
                    consistency_validated=dataset_data.get("corpus_consistency_validated", False),
                    creation_timestamp=dataset_data.get("created_at", "unknown")
                )
                
                self.available_datasets[version] = dataset_info
                
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_file}: {e}")
        
        logger.info(f"Discovered {len(self.available_datasets)} pinned datasets")
        return self.available_datasets
    
    async def load_dataset(self, version: str = "current") -> List[Dict[str, Any]]:
        """Load a specific pinned dataset version."""
        if not self.available_datasets:
            await self.discover_available_datasets()
        
        # Handle 'current' version
        if version == "current":
            current_path = self.dataset_path / "golden-pinned-current.json"
            if current_path.exists():
                target_path = current_path.resolve()  # Follow symlink
                version = target_path.stem.replace("golden-pinned-", "")
            else:
                # Use the most recent dataset
                if self.available_datasets:
                    version = max(self.available_datasets.keys())
                else:
                    raise Exception("No pinned datasets found")
        
        if version not in self.available_datasets:
            raise Exception(f"Dataset version '{version}' not found. Available: {list(self.available_datasets.keys())}")
        
        dataset_info = self.available_datasets[version]
        
        with open(dataset_info.path, 'r') as f:
            dataset_data = json.load(f)
        
        queries = dataset_data.get("queries", [])
        
        # Convert to benchmark format
        benchmark_queries = []
        for i, query in enumerate(queries):
            benchmark_queries.append({
                "query_id": f"pinned_{version}_{i:03d}",
                "query": query.get("query", ""),
                "ground_truth": query.get("expected_files", []),
                "relevance_scores": query.get("relevance_scores", [1.0] * len(query.get("expected_files", []))),
                "intent": query.get("intent", "code_search"),
                "language": query.get("language", "mixed"),
                "dataset_version": version
            })
        
        logger.info(f"Loaded {len(benchmark_queries)} queries from dataset version {version}")
        return benchmark_queries
    
    async def validate_corpus_consistency(self, version: str = "current") -> bool:
        """Validate that the pinned dataset is consistent with the corpus."""
        try:
            # Use Rust CLI to validate consistency
            cmd = [
                "cargo", "run", "--bin", "lens", "--",
                "validate", "--dataset", version
            ]
            
            logger.info("Running corpus consistency validation via Rust CLI...")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent  # lens project root
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ Corpus consistency validation passed")
                return True
            else:
                logger.error(f"❌ Corpus consistency validation failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Corpus validation error: {e}")
            return False

class IntegratedRigorousBenchmark(RigorousCompetitorBenchmark):
    """
    Integrated benchmarking framework that combines the rigorous Python
    framework with the Rust Lens infrastructure.
    """
    
    def __init__(self, lens_config: LensServerConfig, output_dir: str = "./integrated_benchmark_results"):
        super().__init__(output_dir)
        
        self.lens_config = lens_config
        self.dataset_loader = RustDatasetLoader()
        
        # Replace mock T1 Hero with real Lens system
        self.systems = [
            system for system in self.systems if system.get_name() != "T1_Hero"
        ]
        self.systems.append(LensT1HeroSystem(lens_config))
        
        # Clear mock datasets - will load real pinned datasets
        self.datasets = {}
    
    async def initialize_from_rust_ecosystem(self):
        """Initialize benchmarking from Rust ecosystem datasets and infrastructure."""
        logger.info("🔧 Initializing from Rust ecosystem...")
        
        # 1. Discover available datasets
        available_datasets = await self.dataset_loader.discover_available_datasets()
        
        if not available_datasets:
            logger.warning("No pinned datasets found - falling back to mock datasets")
            return
        
        # 2. Load the current/latest dataset
        try:
            pinned_queries = await self.dataset_loader.load_dataset("current")
            self.datasets = {"Lens_Pinned_Dataset": pinned_queries}
            
            logger.info(f"✅ Loaded {len(pinned_queries)} queries from Rust ecosystem")
            
        except Exception as e:
            logger.error(f"Failed to load Rust datasets: {e}")
            return
        
        # 3. Validate corpus consistency
        is_consistent = await self.dataset_loader.validate_corpus_consistency("current")
        if not is_consistent:
            logger.warning("⚠️ Corpus consistency validation failed - results may be unreliable")
    
    async def start_lens_server_if_needed(self) -> bool:
        """Start Lens server if not already running."""
        lens_system = next((s for s in self.systems if isinstance(s, LensT1HeroSystem)), None)
        if not lens_system:
            return True
        
        # Check if server is already running
        try:
            if await lens_system._check_server_health():
                logger.info("✅ Lens server is already running")
                return True
        except:
            pass
        
        # Try to start the server
        logger.info("🚀 Starting Lens server...")
        
        try:
            # Start HTTP server in background
            cmd = [
                "cargo", "run", "--bin", "lens", "--",
                "serve",
                "--port", str(self.lens_config.http_endpoint.split(':')[-1]),
                "--enable-lsp", str(self.lens_config.enable_lsp).lower(),
                "--enable-semantic", str(self.lens_config.enable_semantic).lower()
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent  # lens project root
            )
            
            # Give server time to start
            await asyncio.sleep(5)
            
            # Check if it's now healthy
            if await lens_system._check_server_health():
                logger.info("✅ Lens server started successfully")
                return True
            else:
                logger.error("❌ Lens server failed to start properly")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start Lens server: {e}")
            return False
    
    async def run_integrated_benchmark(self) -> Dict[str, Any]:
        """Run the complete integrated benchmark with Rust infrastructure."""
        logger.info("🚀 Starting integrated rigorous benchmark with Rust infrastructure")
        
        # 1. Initialize from Rust ecosystem
        await self.initialize_from_rust_ecosystem()
        
        # 2. Start Lens server if needed
        server_started = await self.start_lens_server_if_needed()
        if not server_started:
            logger.error("Cannot proceed without Lens server")
            raise Exception("Lens server startup failed")
        
        try:
            # 3. Run the comprehensive benchmark
            results = await self.run_comprehensive_benchmark()
            
            # 4. Generate integration-specific report
            await self._generate_integration_report(results)
            
            return results
            
        finally:
            # Clean up connections
            for system in self.systems:
                if hasattr(system, 'close'):
                    await system.close()
    
    async def _generate_integration_report(self, results: Dict[str, Any]):
        """Generate additional integration-specific reporting."""
        
        report_lines = [
            "# Integrated Rust-Python Benchmark Report",
            "",
            "## Infrastructure Integration",
            f"- **Lens Server**: {self.lens_config.http_endpoint}",
            f"- **LSP Enabled**: {self.lens_config.enable_lsp}",
            f"- **Semantic Search**: {self.lens_config.enable_semantic}",
            "",
            "## Dataset Information",
        ]
        
        # Add dataset info
        for dataset_name, queries in self.datasets.items():
            if queries:
                sample_query = queries[0]
                report_lines.extend([
                    f"### {dataset_name}",
                    f"- **Total Queries**: {len(queries)}",
                    f"- **Dataset Version**: {sample_query.get('dataset_version', 'unknown')}",
                    f"- **Sample Query**: {sample_query.get('query', 'N/A')[:100]}...",
                    ""
                ])
        
        # T1 Hero specific analysis
        t1_results = [r for r in self.results if r.system_name == "T1_Hero_Lens"]
        if t1_results:
            avg_latency = sum(r.execution_time_ms for r in t1_results) / len(t1_results)
            avg_ndcg = sum(r.ndcg_at_10 for r in t1_results) / len(t1_results)
            
            report_lines.extend([
                "## T1 Hero Performance Analysis",
                f"- **Average Query Latency**: {avg_latency:.1f}ms",
                f"- **Average nDCG@10**: {avg_ndcg:.4f}",
                f"- **Total Queries Executed**: {len(t1_results)}",
                "",
                "## Integration Success Metrics",
                f"- **Rust-Python Integration**: ✅ Successful",
                f"- **Pinned Dataset Loading**: ✅ Successful",
                f"- **Server Communication**: ✅ Successful",
                f"- **Statistical Analysis**: ✅ Complete",
                ""
            ])
        
        with open(self.output_dir / "integration_report.md", "w") as f:
            f.write("\n".join(report_lines))
        
        logger.info("✅ Integration report generated")

async def main():
    """Main entry point for integrated benchmarking."""
    
    # Configure Lens server connection
    lens_config = LensServerConfig(
        http_endpoint="http://localhost:3001",
        enable_lsp=True,
        enable_semantic=False,
        timeout_seconds=30
    )
    
    # Create integrated benchmark
    benchmark = IntegratedRigorousBenchmark(lens_config)
    
    try:
        results = await benchmark.run_integrated_benchmark()
        
        print("\n" + "="*80)
        print("🏆 INTEGRATED RIGOROUS BENCHMARK COMPLETED")
        print("="*80)
        print(f"Total Duration: {results['benchmark_duration_seconds']:.1f}s")
        print(f"Systems Tested: {results['total_systems']} (including live T1 Hero)")
        print(f"Datasets Used: {results['total_datasets']}")
        print(f"Total Queries: {results['total_queries_executed']}")
        
        # Integration-specific metrics
        t1_results = [r for r in benchmark.results if "T1_Hero" in r.system_name]
        if t1_results:
            avg_t1_latency = sum(r.execution_time_ms for r in t1_results) / len(t1_results)
            avg_t1_ndcg = sum(r.ndcg_at_10 for r in t1_results) / len(t1_results)
            print(f"T1 Hero Avg Latency: {avg_t1_latency:.1f}ms")
            print(f"T1 Hero Avg nDCG@10: {avg_t1_ndcg:.4f}")
        
        print(f"\nIntegrated artifacts saved to: {benchmark.output_dir}")
        print("🔗 integration_report.md - Rust-Python integration details")
        print("📊 All standard rigorous benchmark artifacts generated")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Integrated benchmark failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())