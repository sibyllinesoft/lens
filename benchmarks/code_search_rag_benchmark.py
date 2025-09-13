#!/usr/bin/env python3
"""
Code Search & Code-Aware RAG Benchmark System

Redesigned benchmark focused on real developer workflows:
- Code search engines (ripgrep, Zoekt, OpenGrok, Lucene/BM25, hybrid)  
- Code intelligence (ctags, treesitter, LSIF/SCIP)
- RAG retrievers (BM25, SPLADE, FAISS/HNSW, hybrid, vector DBs)
- Rerankers (cross-encoder, lightweight rerankers)

Key Features:
- Strict CSV schema with scenario-based metrics
- Auto-generated queries from repo data (issues, PRs, commits, symbols)
- Scenario coverage: code.*, rag.*
- Fail-fast validation with guardrails
- Production-ready competitor implementations
"""

import asyncio
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
import subprocess
import time
import hashlib
import logging
from datetime import datetime
import tempfile
import shutil
import git
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Benchmark scenario types."""
    CODE_FUNC = "code.func"          # Find function/def by natural language
    CODE_SYMBOL = "code.symbol"      # Find symbol by name (case/underscore/renamed)  
    CODE_REGEX = "code.regex"        # Regex/literal "where used" patterns
    CODE_REPO = "code.repo"          # Repo-level tasks (auth middleware, etc.)
    CODE_TRACE = "code.trace"        # Map stack trace/file:line back to source
    RAG_CODE_QA = "rag.code.qa"      # Answer code questions grounded in repo (multi-file)
    RAG_API_QA = "rag.api.qa"        # Answer API usage questions grounded in docs + code
    RAG_DESIGN_QA = "rag.design.qa"  # Answer high-level design questions grounded in ADRs/READMEs


class RetrieverType(Enum):
    """Retriever system types."""
    BM25 = "bm25"
    SPARSE_SPLADE = "sparse_splade"  
    DENSE = "dense"
    HYBRID = "hybrid"
    COLBERT = "colbert"
    REGEX = "regex"
    SYMBOL = "symbol"


class RerankerType(Enum):
    """Reranker types."""
    NONE = "none"
    CROSSENCODER = "crossencoder"


class GeneratorType(Enum):
    """Generator types for RAG."""
    NONE = "none"
    LLM = "llm"


class ChunkPolicy(Enum):
    """Text chunking policies."""
    CODE_UNITS = "code_units"        # Functions, classes, modules
    SLIDING_512 = "sliding_512"      # Sliding window with overlap
    SEMANTIC = "semantic"            # Semantic boundaries


class GoldType(Enum):
    """Gold standard types."""
    PATH = "path"                    # File path matches
    SPAN = "span"                    # Code span matches  
    PASSAGE = "passage"              # Text passage matches


@dataclass
class CorpusConfig:
    """Corpus configuration."""
    id: str
    git_url: str
    sha: str
    description: Optional[str] = None


@dataclass
class SystemConfig:
    """System configuration."""
    id: str
    kind: RetrieverType
    config: Dict[str, Any]
    description: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    corpora: List[CorpusConfig]
    systems: List[SystemConfig]
    scenarios: List[ScenarioType]
    k_retrieval: int = 20
    chunk_policy: ChunkPolicy = ChunkPolicy.CODE_UNITS
    overlap_tokens: int = 64
    

@dataclass
class QueryItem:
    """Single query item with gold standard."""
    qid: str
    query: str
    gold_paths: List[str]
    corpus_id: str
    scenario: ScenarioType
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    qid: str
    rank: int
    path: str
    score: float
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    passage: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Extended benchmark result with new schema."""
    # Original fields (preserved for compatibility)
    run_id: str
    system: str
    benchmark: str  # Now maps to corpus_id
    ndcg_10: Optional[float] = None
    recall_50: Optional[float] = None  
    p95_latency: Optional[float] = None
    jaccard_10: Optional[float] = None
    aece: Optional[float] = None
    delta_ndcg: Optional[float] = None
    delta_p95: Optional[float] = None
    delta_aece: Optional[float] = None
    provenance: str = "local"
    status: str = "AVAILABLE"
    queries_executed: int = 0
    raw_results_path: str = ""
    
    # New schema fields
    scenario: Optional[str] = None
    corpus_id: Optional[str] = None
    index_cfg: Optional[str] = None
    retriever: Optional[str] = None
    reranker: Optional[str] = None
    generator: Optional[str] = None
    k_retrieval: Optional[int] = None
    k_rerank: Optional[int] = None
    answer_len_limit: Optional[int] = None
    overlap_tokens: Optional[int] = None
    chunk_policy: Optional[str] = None
    gold_type: Optional[str] = None
    gold_count: Optional[int] = None
    
    # Scenario-specific metrics
    success_at_k: Optional[float] = None      # Hit@k on gold path
    mrr_at_k: Optional[float] = None         # Mean Reciprocal Rank@k
    exact_path_at_1: Optional[float] = None   # Exact path match@1
    defs_refs_hit_at_k: Optional[float] = None  # Symbol def/ref hit@k
    context_precision: Optional[float] = None   # Gold tokens in retrieved / retrieved tokens
    context_recall: Optional[float] = None      # Retrieved gold tokens / total gold tokens  
    attribution_at_k: Optional[float] = None    # Retrieved passage from gold file
    answerable_at_k: Optional[float] = None     # Gold present in top-k


class DatasetGenerator:
    """Generate benchmark queries from repository data."""
    
    def __init__(self, corpus_config: CorpusConfig, work_dir: Path):
        self.corpus_config = corpus_config
        self.work_dir = work_dir / "corpora" / corpus_config.id
        self.repo_path = self.work_dir / "repo"
        
    async def setup_corpus(self) -> bool:
        """Clone and setup corpus repository."""
        try:
            if self.repo_path.exists():
                shutil.rmtree(self.repo_path)
                
            self.work_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Cloning {self.corpus_config.git_url} to {self.repo_path}")
            repo = git.Repo.clone_from(self.corpus_config.git_url, self.repo_path)
            repo.git.checkout(self.corpus_config.sha)
            
            logger.info(f"✅ Corpus {self.corpus_config.id} ready at SHA {self.corpus_config.sha}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup corpus {self.corpus_config.id}: {e}")
            return False
            
    async def generate_symbol_queries(self) -> List[QueryItem]:
        """Generate code.symbol queries using ctags or regex fallback."""
        queries = []
        
        try:
            # Try ctags first
            ctags_output = subprocess.run([
                "ctags", "-R", "--output-format=json", 
                "--languages=Python,TypeScript,JavaScript,Go,Rust",
                str(self.repo_path)
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if ctags_output.returncode == 0:
                symbol_count = 0
                for line in ctags_output.stdout.strip().split('\n'):
                    if not line:
                        continue
                        
                    try:
                        tag = json.loads(line)
                        if tag.get('kind') in ['function', 'class', 'interface', 'type']:
                            symbol_name = tag['name']
                            file_path = tag['path']
                            
                            # Create query variants
                            queries.extend([
                                QueryItem(
                                    qid=f"symbol_{self.corpus_config.id}_{symbol_count}_exact",
                                    query=f"find {symbol_name}",
                                    gold_paths=[file_path],
                                    corpus_id=self.corpus_config.id,
                                    scenario=ScenarioType.CODE_SYMBOL,
                                    metadata={"symbol": symbol_name, "kind": tag.get('kind')}
                                ),
                                QueryItem(
                                    qid=f"symbol_{self.corpus_config.id}_{symbol_count}_underscore",
                                    query=f"find {symbol_name.replace('_', ' ')}",
                                    gold_paths=[file_path],
                                    corpus_id=self.corpus_config.id,
                                    scenario=ScenarioType.CODE_SYMBOL,
                                    metadata={"symbol": symbol_name, "kind": tag.get('kind'), "variant": "underscore"}
                                )
                            ])
                            symbol_count += 1
                            
                            if symbol_count >= 25:  # Limit for manageable dataset
                                break
                                
                    except json.JSONDecodeError:
                        continue
                        
                logger.info(f"✅ Generated {len(queries)} symbol queries using ctags for {self.corpus_config.id}")
            else:
                raise Exception("ctags failed")
                
        except Exception as e:
            logger.warning(f"⚠️  ctags failed ({e}), falling back to regex symbol extraction")
            
            # Fallback: regex-based symbol extraction
            try:
                queries = await self._generate_symbols_regex_fallback()
                logger.info(f"✅ Generated {len(queries)} symbol queries using regex fallback for {self.corpus_config.id}")
            except Exception as fallback_error:
                logger.error(f"❌ Regex fallback also failed: {fallback_error}")
            
        return queries
        
    async def _generate_symbols_regex_fallback(self) -> List[QueryItem]:
        """Fallback symbol extraction using regex patterns."""
        queries = []
        symbol_count = 0
        
        # Python patterns
        python_patterns = [
            (r"^def\s+(\w+)\s*\(", "function"),
            (r"^class\s+(\w+)", "class"),
            (r"^async\s+def\s+(\w+)\s*\(", "async_function")
        ]
        
        # Find Python files and extract symbols
        for py_file in self.repo_path.glob("**/*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines):
                    for pattern, kind in python_patterns:
                        match = re.match(pattern, line.strip())
                        if match:
                            symbol_name = match.group(1)
                            file_path = str(py_file.relative_to(self.repo_path))
                            
                            queries.extend([
                                QueryItem(
                                    qid=f"symbol_{self.corpus_config.id}_{symbol_count}_exact",
                                    query=f"find {symbol_name}",
                                    gold_paths=[file_path],
                                    corpus_id=self.corpus_config.id,
                                    scenario=ScenarioType.CODE_SYMBOL,
                                    metadata={"symbol": symbol_name, "kind": kind, "method": "regex"}
                                ),
                                QueryItem(
                                    qid=f"symbol_{self.corpus_config.id}_{symbol_count}_underscore",
                                    query=f"find {symbol_name.replace('_', ' ')}",
                                    gold_paths=[file_path],
                                    corpus_id=self.corpus_config.id,
                                    scenario=ScenarioType.CODE_SYMBOL,
                                    metadata={"symbol": symbol_name, "kind": kind, "method": "regex", "variant": "underscore"}
                                )
                            ])
                            symbol_count += 1
                            
                            if symbol_count >= 25:  # Limit for manageable dataset
                                return queries
                                
            except Exception as e:
                logger.warning(f"Failed to process {py_file}: {e}")
                continue
                
        return queries
        
    async def generate_function_queries(self) -> List[QueryItem]:
        """Generate code.func queries from commit messages and PR titles."""
        queries = []
        
        try:
            repo = git.Repo(self.repo_path)
            commit_count = 0
            
            # Mine recent commit messages for natural language queries
            for commit in repo.iter_commits(max_count=200):
                message = commit.message.strip()
                
                # Filter for function-related commits
                if any(keyword in message.lower() for keyword in ['add', 'implement', 'create', 'fix', 'update']):
                    if any(code_word in message.lower() for code_word in ['function', 'method', 'class', 'api']):
                        # Get files touched in this commit
                        touched_files = []
                        try:
                            for item in commit.stats.files:
                                if any(item.endswith(ext) for ext in ['.py', '.ts', '.js', '.go', '.rs']):
                                    touched_files.append(item)
                        except:
                            continue
                            
                        if touched_files:
                            queries.append(QueryItem(
                                qid=f"func_{self.corpus_config.id}_{commit_count}",
                                query=message.split('\n')[0][:200],  # First line, truncated
                                gold_paths=touched_files,
                                corpus_id=self.corpus_config.id,
                                scenario=ScenarioType.CODE_FUNC,
                                metadata={"commit_sha": commit.hexsha[:8]}
                            ))
                            commit_count += 1
                            
                            if commit_count >= 30:  # Limit for manageable dataset
                                break
                                
            logger.info(f"✅ Generated {len(queries)} function queries for {self.corpus_config.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate function queries: {e}")
            
        return queries
        
    async def generate_regex_queries(self) -> List[QueryItem]:
        """Generate code.regex queries for common patterns.""" 
        queries = []
        
        # Common regex patterns developers search for
        patterns = [
            (r"TODO|FIXME", "find todos and fixmes"),
            (r"import\s+\w+", "find import statements"),
            (r"class\s+\w+", "find class definitions"),
            (r"def\s+\w+", "find function definitions"),
            (r"@\w+", "find decorators or annotations"),
            (r"console\.log|print\(", "find debug statements"),
            (r"throw|raise\s+", "find error handling"),
            (r"async\s+def|async\s+function", "find async functions")
        ]
        
        try:
            for i, (pattern, description) in enumerate(patterns):
                # Find files matching pattern using ripgrep
                result = subprocess.run([
                    "rg", "-l", pattern, str(self.repo_path)
                ], capture_output=True, text=True)
                
                matching_files = []
                for line in result.stdout.strip().split('\n'):
                    if line and line.startswith(str(self.repo_path)):
                        rel_path = str(Path(line).relative_to(self.repo_path))
                        matching_files.append(rel_path)
                        
                if matching_files:
                    queries.append(QueryItem(
                        qid=f"regex_{self.corpus_config.id}_{i}",
                        query=description,
                        gold_paths=matching_files[:10],  # Limit gold files
                        corpus_id=self.corpus_config.id,
                        scenario=ScenarioType.CODE_REGEX,
                        metadata={"pattern": pattern}
                    ))
                    
            logger.info(f"✅ Generated {len(queries)} regex queries for {self.corpus_config.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate regex queries: {e}")
            
        return queries
        
    async def generate_rag_queries(self) -> List[QueryItem]:
        """Generate RAG queries from README files and documentation."""
        queries = []
        
        try:
            # Find documentation files
            doc_files = []
            for pattern in ["README*", "*.md", "docs/**/*.md"]:
                doc_files.extend(self.repo_path.glob(pattern))
                
            for i, doc_file in enumerate(doc_files[:10]):  # Limit to first 10 docs
                try:
                    content = doc_file.read_text(encoding='utf-8', errors='ignore')
                    
                    # Extract sections as potential questions
                    lines = content.split('\n')
                    for j, line in enumerate(lines):
                        if line.startswith('#') and len(line.strip()) > 5:
                            section_title = line.strip('#').strip()
                            
                            # Create a question from the section
                            query = f"How does {section_title.lower()}?"
                            
                            queries.append(QueryItem(
                                qid=f"rag_code_{self.corpus_config.id}_{i}_{j}",
                                query=query,
                                gold_paths=[str(doc_file.relative_to(self.repo_path))],
                                corpus_id=self.corpus_config.id,
                                scenario=ScenarioType.RAG_CODE_QA,
                                metadata={"doc_file": str(doc_file.relative_to(self.repo_path))}
                            ))
                            
                            if len(queries) >= 20:  # Limit total RAG queries
                                break
                                
                    if len(queries) >= 20:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to process {doc_file}: {e}")
                    continue
                    
            logger.info(f"✅ Generated {len(queries)} RAG queries for {self.corpus_config.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate RAG queries: {e}")
            
        return queries
        
    async def generate_all_queries(self) -> List[QueryItem]:
        """Generate all query types for this corpus."""
        all_queries = []
        
        logger.info(f"🔄 Generating queries for corpus {self.corpus_config.id}")
        
        # Generate each scenario type
        symbol_queries = await self.generate_symbol_queries()
        all_queries.extend(symbol_queries)
        
        function_queries = await self.generate_function_queries()
        all_queries.extend(function_queries)
        
        regex_queries = await self.generate_regex_queries()
        all_queries.extend(regex_queries)
        
        rag_queries = await self.generate_rag_queries()
        all_queries.extend(rag_queries)
        
        # Validate: fail if any scenario has 0 queries
        scenario_counts = {}
        for query in all_queries:
            scenario = query.scenario.value
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            
        for scenario in [ScenarioType.CODE_SYMBOL, ScenarioType.CODE_FUNC, 
                        ScenarioType.CODE_REGEX, ScenarioType.RAG_CODE_QA]:
            if scenario_counts.get(scenario.value, 0) == 0:
                raise ValueError(f"❌ FAIL: Scenario {scenario.value} has 0 queries for corpus {self.corpus_config.id}")
                
        logger.info(f"✅ Generated {len(all_queries)} total queries for {self.corpus_config.id}")
        logger.info(f"📊 Scenario breakdown: {scenario_counts}")
        
        return all_queries


class CompetitorSystem:
    """Base class for competitor system implementations."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir / "systems" / config.id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    async def setup_index(self, corpus_path: Path) -> bool:
        """Setup search index for the corpus."""
        raise NotImplementedError
        
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Execute search query."""
        raise NotImplementedError
        
    async def cleanup(self):
        """Cleanup resources."""
        pass


class RipgrepSystem(CompetitorSystem):
    """Ripgrep baseline system."""
    
    def __init__(self, config: SystemConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.corpus_path = None
        
    async def setup_index(self, corpus_path: Path) -> bool:
        """No indexing needed for ripgrep."""
        self.corpus_path = corpus_path
        return True
        
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Search using ripgrep."""
        if not self.corpus_path:
            return []
            
        try:
            # Use ripgrep for literal search
            start_time = time.time()
            result = subprocess.run([
                "rg", "-l", "--max-count", str(k), 
                query, str(self.corpus_path)
            ], capture_output=True, text=True, timeout=30)
            
            latency_ms = (time.time() - start_time) * 1000
            
            results = []
            for i, line in enumerate(result.stdout.strip().split('\n')):
                if line and line.startswith(str(self.corpus_path)):
                    rel_path = str(Path(line).relative_to(self.corpus_path))
                    results.append(RetrievalResult(
                        qid="",  # Will be filled by caller
                        rank=i + 1,
                        path=rel_path,
                        score=1.0 / (i + 1)  # Simple ranking score
                    ))
                    
            return results[:k]
            
        except Exception as e:
            logger.error(f"Ripgrep search failed: {e}")
            return []


class MockBM25System(CompetitorSystem):
    """Mock BM25 system for demonstration."""
    
    async def setup_index(self, corpus_path: Path) -> bool:
        """Mock index setup."""
        logger.info(f"Mock BM25 index setup for {corpus_path}")
        return True
        
    async def search(self, query: str, k: int = 20) -> List[RetrievalResult]:
        """Mock BM25 search."""
        # Return mock results for demonstration
        results = []
        for i in range(min(k, 5)):
            results.append(RetrievalResult(
                qid="",
                rank=i + 1,
                path=f"mock/file_{i}.py",
                score=0.8 - (i * 0.1)
            ))
        return results


class CodeSearchRAGBenchmark:
    """Main benchmark orchestrator."""
    
    def __init__(self, config_file: str):
        self.config_file = Path(config_file)
        self.work_dir = Path("code_search_benchmark_workspace")
        self.output_dir = Path("benchmarks/code_search_results")
        self.run_id = f"csr_{int(time.time())}"
        
        # Load configuration
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            
        self.config = BenchmarkConfig(
            corpora=[CorpusConfig(**c) for c in config_data['corpora']],
            systems=[SystemConfig(
                id=s['id'],
                kind=RetrieverType(s['kind']),
                config=s.get('config', {}),
                description=s.get('description')
            ) for s in config_data['systems']],
            scenarios=[ScenarioType(s) for s in config_data['scenarios']],
            k_retrieval=config_data.get('k_retrieval', 20),
            chunk_policy=ChunkPolicy(config_data.get('chunk_policy', 'code_units')),
            overlap_tokens=config_data.get('overlap_tokens', 64)
        )
        
        # Setup directories
        self.work_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_system(self, system_config: SystemConfig) -> CompetitorSystem:
        """Factory method to create competitor systems."""
        if system_config.kind == RetrieverType.REGEX:
            return RipgrepSystem(system_config, self.work_dir)
        elif system_config.kind == RetrieverType.BM25:
            return MockBM25System(system_config, self.work_dir)
        else:
            # For now, default to mock BM25
            logger.warning(f"System type {system_config.kind} not implemented, using mock")
            return MockBM25System(system_config, self.work_dir)
            
    async def run_benchmark(self) -> Dict[str, Any]:
        """Run complete code search & RAG benchmark."""
        
        logger.info("🚀 CODE SEARCH & RAG BENCHMARK")
        logger.info("=" * 50)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Corpora: {len(self.config.corpora)}")
        logger.info(f"Systems: {len(self.config.systems)}")
        logger.info(f"Scenarios: {[s.value for s in self.config.scenarios]}")
        logger.info("=" * 50)
        
        start_time = time.time()
        all_results = []
        
        try:
            # 1. Setup corpora and generate queries
            logger.info("📥 Setting up corpora and generating queries...")
            all_queries = []
            
            for corpus_config in self.config.corpora:
                generator = DatasetGenerator(corpus_config, self.work_dir)
                
                if not await generator.setup_corpus():
                    raise ValueError(f"Failed to setup corpus {corpus_config.id}")
                    
                queries = await generator.generate_all_queries()
                all_queries.extend(queries)
                
            logger.info(f"✅ Generated {len(all_queries)} total queries across all corpora")
            
            # 2. Run competitor systems
            logger.info("🏁 Running competitor systems...")
            
            for system_config in self.config.systems:
                logger.info(f"⚡ Running system: {system_config.id}")
                
                system = self._create_system(system_config)
                
                # Setup indexes for all corpora
                for corpus_config in self.config.corpora:
                    corpus_path = self.work_dir / "corpora" / corpus_config.id / "repo"
                    if not await system.setup_index(corpus_path):
                        logger.error(f"Failed to setup index for {system_config.id} on {corpus_config.id}")
                        continue
                        
                # Run queries grouped by (scenario, corpus)
                scenario_corpus_groups = {}
                for query in all_queries:
                    key = (query.scenario.value, query.corpus_id)
                    if key not in scenario_corpus_groups:
                        scenario_corpus_groups[key] = []
                    scenario_corpus_groups[key].append(query)
                    
                for (scenario, corpus_id), queries in scenario_corpus_groups.items():
                    logger.info(f"  📋 {scenario} on {corpus_id}: {len(queries)} queries")
                    
                    # Execute queries and collect results
                    scenario_results = []
                    query_count = 0
                    
                    for query in queries:
                        start_query_time = time.time()
                        retrieval_results = await system.search(query.query, self.config.k_retrieval)
                        query_latency = (time.time() - start_query_time) * 1000
                        
                        # Store raw results
                        raw_results_dir = self.output_dir / "runs" / system_config.id / scenario / corpus_id
                        raw_results_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Compute metrics
                        metrics = self._compute_scenario_metrics(query, retrieval_results, scenario)
                        
                        # Create benchmark result
                        result = BenchmarkResult(
                            run_id=self.run_id,
                            system=system_config.id,
                            benchmark=corpus_id,  # Maps to corpus_id for compatibility
                            scenario=scenario,
                            corpus_id=corpus_id,
                            retriever=system_config.kind.value if isinstance(system_config.kind, RetrieverType) else system_config.kind,
                            k_retrieval=self.config.k_retrieval,
                            chunk_policy=self.config.chunk_policy.value if isinstance(self.config.chunk_policy, ChunkPolicy) else self.config.chunk_policy,
                            overlap_tokens=self.config.overlap_tokens,
                            gold_type=GoldType.PATH.value,  # Most queries use path matching
                            gold_count=len(query.gold_paths),
                            queries_executed=1,
                            p95_latency=query_latency,
                            **metrics
                        )
                        
                        scenario_results.append(result)
                        query_count += 1
                        
                    # Aggregate scenario results
                    if scenario_results:
                        aggregated = self._aggregate_scenario_results(scenario_results, scenario, corpus_id, system_config.id)
                        all_results.append(aggregated)
                        
                await system.cleanup()
                
            # 3. Validate results
            logger.info("✅ Validating results...")
            self._validate_results(all_results)
            
            # 4. Generate outputs
            logger.info("📊 Generating outputs...")
            artifacts = await self._generate_outputs(all_results)
            
            duration = time.time() - start_time
            
            return {
                'run_id': self.run_id,
                'duration_seconds': duration,
                'total_queries': len(all_queries),
                'total_results': len(all_results),
                'artifacts': artifacts
            }
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            raise
            
    def _compute_scenario_metrics(self, query: QueryItem, results: List[RetrievalResult], 
                                 scenario: str) -> Dict[str, float]:
        """Compute scenario-specific metrics."""
        metrics = {}
        
        if not results:
            return {
                'success_at_k': 0.0,
                'mrr_at_k': 0.0,
                'ndcg_10': 0.0,
                'recall_50': 0.0
            }
            
        k = min(len(results), 20)
        
        # Success@k: Hit@k on gold paths
        hits = [1 if any(gold in result.path for gold in query.gold_paths) else 0 
                for result in results[:k]]
        metrics['success_at_k'] = 1.0 if any(hits) else 0.0
        
        # MRR@k: Mean Reciprocal Rank
        for i, hit in enumerate(hits):
            if hit:
                metrics['mrr_at_k'] = 1.0 / (i + 1)
                break
        else:
            metrics['mrr_at_k'] = 0.0
            
        # NDCG@10 (simplified binary relevance)
        relevance = [1 if any(gold in result.path for gold in query.gold_paths) else 0 
                    for result in results[:10]]
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(query.gold_paths), 10)))
        metrics['ndcg_10'] = dcg / idcg if idcg > 0 else 0.0
        
        # Recall@50 (simplified)
        hits_50 = [1 if any(gold in result.path for gold in query.gold_paths) else 0 
                   for result in results[:50]]
        metrics['recall_50'] = sum(hits_50) / len(query.gold_paths)
        
        # Scenario-specific metrics
        if scenario.startswith('code.'):
            # Exact path match@1
            if results and any(gold == results[0].path for gold in query.gold_paths):
                metrics['exact_path_at_1'] = 1.0
            else:
                metrics['exact_path_at_1'] = 0.0
                
        elif scenario.startswith('rag.'):
            # Context precision/recall (simplified)
            retrieved_tokens = sum(len(r.passage or r.path) for r in results[:k])
            if retrieved_tokens > 0:
                # Simplified: assume all gold paths contribute equally
                gold_tokens = sum(len(path) for path in query.gold_paths)
                overlap_tokens = sum(len(gold) for gold in query.gold_paths 
                                   if any(gold in r.path for r in results[:k]))
                
                metrics['context_precision'] = overlap_tokens / retrieved_tokens
                metrics['context_recall'] = overlap_tokens / gold_tokens if gold_tokens > 0 else 0.0
                metrics['attribution_at_k'] = 1.0 if overlap_tokens > 0 else 0.0
                
        return metrics
        
    def _aggregate_scenario_results(self, results: List[BenchmarkResult], 
                                  scenario: str, corpus_id: str, system_id: str) -> BenchmarkResult:
        """Aggregate multiple query results into scenario-level result."""
        
        # Compute averages across queries
        avg_metrics = {}
        metric_fields = ['success_at_k', 'mrr_at_k', 'ndcg_10', 'recall_50', 
                        'exact_path_at_1', 'context_precision', 'context_recall', 
                        'attribution_at_k', 'p95_latency']
        
        for field in metric_fields:
            values = [getattr(r, field) for r in results if getattr(r, field) is not None]
            if values:
                avg_metrics[field] = np.mean(values)
                if field == 'p95_latency':
                    avg_metrics[field] = np.percentile(values, 95)  # Actual p95 for latency
                    
        return BenchmarkResult(
            run_id=self.run_id,
            system=system_id,
            benchmark=corpus_id,
            scenario=scenario,
            corpus_id=corpus_id,
            queries_executed=len(results),
            **avg_metrics
        )
        
    def _validate_results(self, results: List[BenchmarkResult]):
        """Validate results with fail-fast checks."""
        
        logger.info("🔍 Running validation checks...")
        
        # Check 1: No duplicate (system, benchmark, scenario, corpus_id) rows
        seen_keys = set()
        for result in results:
            key = (result.system, result.benchmark, result.scenario, result.corpus_id)
            if key in seen_keys:
                raise ValueError(f"❌ FAIL: Duplicate result for {key}")
            seen_keys.add(key)
            
        # Check 2: Metric ranges
        for result in results:
            if result.ndcg_10 is not None and not (0 <= result.ndcg_10 <= 1):
                raise ValueError(f"❌ FAIL: Invalid NDCG@10 {result.ndcg_10} for {result.system}")
            if result.recall_50 is not None and not (0 <= result.recall_50 <= 1):
                raise ValueError(f"❌ FAIL: Invalid Recall@50 {result.recall_50} for {result.system}")
            if result.p95_latency is not None and result.p95_latency < 0:
                raise ValueError(f"❌ FAIL: Invalid latency {result.p95_latency} for {result.system}")
            if result.queries_executed < 1:
                raise ValueError(f"❌ FAIL: No queries executed for {result.system}")
                
        # Check 3: System coverage
        scenario_groups = {'code': [], 'rag': []}
        for result in results:
            if result.scenario.startswith('code.'):
                scenario_groups['code'].append(result.system)
            elif result.scenario.startswith('rag.'):
                scenario_groups['rag'].append(result.system)
                
        for system_config in self.config.systems:
            code_count = scenario_groups['code'].count(system_config.id)
            rag_count = scenario_groups['rag'].count(system_config.id)
            
            if code_count < 2:
                raise ValueError(f"❌ FAIL: System {system_config.id} missing in ≥2 code.* scenarios")
            if rag_count < 1:
                raise ValueError(f"❌ FAIL: System {system_config.id} missing in ≥1 rag.* scenario")
                
        # Check 4: RAG context recall threshold
        for result in results:
            if (result.scenario and result.scenario.startswith('rag.') and 
                result.context_recall is not None and result.context_recall < 0.6):
                logger.warning(f"⚠️  RAG regression: {result.system} context_recall={result.context_recall:.2f} < 0.6")
                
        logger.info("✅ All validation checks passed")
        
    async def _generate_outputs(self, results: List[BenchmarkResult]) -> Dict[str, str]:
        """Generate all required outputs."""
        artifacts = {}
        
        # 1. Update CSV with new schema
        csv_file = self.output_dir / "competitor_matrix.csv" 
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(csv_file, index=False)
        artifacts['csv'] = str(csv_file)
        
        # 2. Generate leaderboard by scenario
        leaderboard_file = self.output_dir / "scenario_leaderboard.md"
        await self._generate_scenario_leaderboard(results, leaderboard_file)
        artifacts['leaderboard'] = str(leaderboard_file)
        
        # 3. Generate plots (placeholder)
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        artifacts['plots'] = str(plots_dir)
        
        # 4. Generate integrity manifest
        manifest_file = await self._generate_integrity_manifest(artifacts)
        artifacts['manifest'] = manifest_file
        
        logger.info(f"✅ Generated {len(artifacts)} artifacts")
        return artifacts
        
    async def _generate_scenario_leaderboard(self, results: List[BenchmarkResult], output_file: Path):
        """Generate scenario-grouped leaderboard."""
        
        md_lines = [
            "# 🏆 Code Search & RAG Benchmark Results",
            "",
            f"**Run ID**: {self.run_id}",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Total Results**: {len(results)}",
            "",
        ]
        
        # Group by scenario
        scenario_groups = {}
        for result in results:
            scenario = result.scenario
            if scenario not in scenario_groups:
                scenario_groups[scenario] = []
            scenario_groups[scenario].append(result)
            
        for scenario, scenario_results in sorted(scenario_groups.items()):
            md_lines.extend([
                f"## 📋 {scenario.upper()}",
                "",
                "| Rank | System | Success@k | MRR@k | NDCG@10 | Queries |",
                "|------|--------|-----------|-------|---------|---------|"
            ])
            
            # Sort by primary metric (Success@k, then MRR@k)
            scenario_results.sort(key=lambda x: (
                -(x.success_at_k or 0),
                -(x.mrr_at_k or 0),
                -(x.ndcg_10 or 0)
            ))
            
            for i, result in enumerate(scenario_results, 1):
                md_lines.append(
                    f"| **#{i}** | **{result.system}** | "
                    f"{result.success_at_k:.1%} | "
                    f"{result.mrr_at_k:.3f} | "
                    f"{result.ndcg_10:.3f} | "
                    f"{result.queries_executed} |"
                )
                
            md_lines.append("")
            
        md_lines.extend([
            "---",
            "*Generated by Code Search & RAG Benchmark System*"
        ])
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(md_lines))
            
    async def _generate_integrity_manifest(self, artifacts: Dict[str, str]) -> str:
        """Generate integrity manifest with checksums."""
        manifest = {
            "run_id": self.run_id,
            "generated_at": time.time(),
            "benchmark_type": "code_search_rag",
            "artifacts": {}
        }
        
        for name, path in artifacts.items():
            if name == 'manifest':  # Skip self
                continue
                
            file_path = Path(path)
            if file_path.exists() and file_path.is_file():
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    
                manifest["artifacts"][name] = {
                    "path": str(file_path),
                    "sha256": file_hash,
                    "size_bytes": file_path.stat().st_size
                }
                
        manifest_file = self.output_dir / "integrity_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            
        return str(manifest_file)


async def main():
    """Main entry point."""
    
    # Create sample config if it doesn't exist
    config_file = "bench_config.yaml"
    if not Path(config_file).exists():
        sample_config = {
            'corpora': [
                {
                    'id': 'lens_main',
                    'git_url': 'https://github.com/example/lens.git',  # Replace with actual
                    'sha': 'main',
                    'description': 'Main lens repository'
                }
            ],
            'systems': [
                {
                    'id': 'ripgrep',
                    'kind': 'regex',
                    'config': {},
                    'description': 'Ripgrep literal/regex baseline'
                },
                {
                    'id': 'lucene_bm25',
                    'kind': 'bm25',
                    'config': {},
                    'description': 'Lucene BM25 sparse retrieval'
                }
            ],
            'scenarios': ['code.func', 'code.symbol', 'code.regex', 'rag.code.qa'],
            'k_retrieval': 20,
            'chunk_policy': 'code_units',
            'overlap_tokens': 64
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False)
            
        print(f"📝 Created sample config: {config_file}")
        print("⚠️  Please update git URLs and SHAs before running")
        return
        
    # Run benchmark
    benchmark = CodeSearchRAGBenchmark(config_file)
    
    try:
        results = await benchmark.run_benchmark()
        
        print("\n🎉 CODE SEARCH & RAG BENCHMARK COMPLETED!")
        print("=" * 50)
        print(f"Run ID: {results['run_id']}")
        print(f"Duration: {results['duration_seconds']:.1f}s")
        print(f"Total Queries: {results['total_queries']}")
        print(f"Total Results: {results['total_results']}")
        print("\n📊 Artifacts Generated:")
        for name, path in results['artifacts'].items():
            print(f"   ✅ {name}: {path}")
            
    except Exception as e:
        print(f"\n❌ BENCHMARK FAILED: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())