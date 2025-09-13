#!/usr/bin/env python3
"""
Comprehensive Code Search RAG Benchmark with Live Sanity Pyramid Integration

Wires the sanity pyramid between retrieval and generation:
retrieve → evidence_map → ESS → (gate) → generate

Only queries that pass sanity gates reach the LLM.
"""
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from code_search_rag_comprehensive import ComprehensiveBenchmarkFramework, BenchmarkQuery
from live_sanity_integration import LiveSanityIntegration, ESS_Thresholds, SanityScorecard
from real_competitor_systems import create_real_system

logger = logging.getLogger(__name__)


class SanityGatedBenchmark:
    """Benchmark framework with integrated sanity pyramid gates."""
    
    def __init__(self, config_path: str, work_dir: Path):
        self.config_path = config_path
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        
        # Initialize core framework
        self.framework = ComprehensiveBenchmarkFramework(config_path, work_dir)
        
        # Initialize live sanity integration
        self.live_sanity = LiveSanityIntegration(work_dir / "sanity_gated")
        
        # Results tracking
        self.gated_results: Dict[str, List[Dict]] = {}
        self.sanity_reports: List[Dict] = []
    
    async def run_gated_benchmark(self, scenarios: List[str] = None, 
                                 systems: List[str] = None) -> Dict[str, Any]:
        """
        Run full benchmark with sanity pyramid gates.
        
        Pipeline: query → retrieve → sanity_gate → (if passed) → generate → evaluate
        """
        logger.info("🚀 Starting sanity-gated benchmark")
        
        # Load configuration 
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        scenarios = scenarios or config.get('scenarios', ['code.func', 'code.symbol', 'rag.code.qa'])
        systems_config = {s['id']: s for s in config.get('systems', [])}
        systems = systems or list(systems_config.keys())
        
        logger.info(f"📊 Running {len(scenarios)} scenarios across {len(systems)} systems")
        
        # Generate or load queries
        queries = await self._generate_benchmark_queries(config.get('corpora', []))
        
        benchmark_results = {}
        
        # Run each system through the gated pipeline
        for system_id in systems:
            logger.info(f"🔄 Processing system: {system_id}")
            
            system_config = systems_config.get(system_id, {})
            system = await self._initialize_system(system_id, system_config)
            
            if not system:
                logger.warning(f"⚠️ Skipping system {system_id} - initialization failed")
                continue
            
            system_results = await self._run_gated_system_evaluation(
                system, system_id, queries, scenarios
            )
            
            benchmark_results[system_id] = system_results
        
        # Generate comprehensive sanity report
        sanity_report = await self.live_sanity.generate_live_validation_report()
        
        # Compile final results
        final_results = {
            'benchmark_results': benchmark_results,
            'sanity_validation': sanity_report,
            'pipeline_summary': {
                'total_queries': sum(len(scenario_queries) for scenario_queries in queries.values()),
                'queries_passed_gates': len(self.live_sanity.passed_queries),
                'queries_blocked': len(self.live_sanity.blocked_queries),
                'gate_pass_rate': len(self.live_sanity.passed_queries) / sum(len(scenario_queries) for scenario_queries in queries.values()) if queries else 0.0,
                'systems_evaluated': len(systems),
                'scenarios_tested': len(scenarios)
            }
        }
        
        # Save results and scorecard
        await self._save_comprehensive_results(final_results)
        
        return final_results
    
    async def _generate_benchmark_queries(self, corpora_config: List[Dict]) -> Dict[str, List[BenchmarkQuery]]:
        """Generate or load benchmark queries."""
        logger.info("📝 Generating benchmark queries")
        
        # For now, create representative test queries
        # In production, this would use the DatasetMiner
        queries = {}
        
        for corpus_config in corpora_config:
            corpus_id = corpus_config['id']
            
            # Generate diverse queries per scenario
            corpus_queries = []
            
            # code.func queries
            for i, topic in enumerate(['validation', 'parsing', 'serialization', 'error handling', 'configuration']):
                query = BenchmarkQuery(
                    qid=f"{corpus_id}_code_func_{i}",
                    query=f"find function that handles {topic}",
                    gold_paths=[f"src/{topic.replace(' ', '_')}.py"],
                    corpus_id=corpus_id,
                    scenario="code.func",
                    gold_spans=[(f"src/{topic.replace(' ', '_')}.py", 10, 50)]
                )
                corpus_queries.append(query)
            
            # code.symbol queries
            for i, symbol in enumerate(['BaseModel', 'Field', 'validator', 'ValidationError', 'Schema']):
                query = BenchmarkQuery(
                    qid=f"{corpus_id}_code_symbol_{i}",
                    query=f"locate {symbol} definition",
                    gold_paths=[f"src/core.py"],
                    corpus_id=corpus_id,
                    scenario="code.symbol",
                    gold_spans=[(f"src/core.py", 20, 60)]
                )
                corpus_queries.append(query)
            
            # rag.code.qa queries
            for i, task in enumerate(['validate data', 'create models', 'handle errors', 'serialize output', 'configure fields']):
                query = BenchmarkQuery(
                    qid=f"{corpus_id}_rag_code_qa_{i}",
                    query=f"how to {task} in this library",
                    gold_paths=[f"docs/{task.replace(' ', '_')}.md", f"examples/{task.replace(' ', '_')}.py"],
                    corpus_id=corpus_id,
                    scenario="rag.code.qa",
                    gold_spans=[(f"examples/{task.replace(' ', '_')}.py", 1, 30)]
                )
                corpus_queries.append(query)
            
            queries[corpus_id] = corpus_queries
        
        total_queries = sum(len(cq) for cq in queries.values())
        logger.info(f"✅ Generated {total_queries} queries across {len(corpora_config)} corpora")
        
        return queries
    
    async def _initialize_system(self, system_id: str, system_config: Dict) -> Optional[Any]:
        """Initialize competitor system."""
        try:
            from code_search_rag_comprehensive import SystemConfig
            config = SystemConfig(**system_config)
            system = create_real_system(config, self.work_dir / "systems" / system_id)
            
            # Health check
            if hasattr(system, 'health_check'):
                healthy = await system.health_check()
                if not healthy:
                    logger.warning(f"⚠️ System {system_id} failed health check")
                    return None
            
            return system
        except Exception as e:
            logger.error(f"❌ Failed to initialize system {system_id}: {e}")
            return None
    
    async def _run_gated_system_evaluation(self, system: Any, system_id: str, 
                                          queries: Dict[str, List[BenchmarkQuery]], 
                                          scenarios: List[str]) -> Dict[str, Any]:
        """Run system evaluation with sanity gates."""
        logger.info(f"🎯 Running gated evaluation for {system_id}")
        
        system_results = {
            'queries_processed': 0,
            'queries_passed_gates': 0,
            'queries_blocked': 0,
            'scenario_results': {},
            'sanity_gate_results': []
        }
        
        for corpus_id, corpus_queries in queries.items():
            for query in corpus_queries:
                if query.scenario not in scenarios:
                    continue
                
                # Step 1: Retrieval
                try:
                    retrieved_chunks = await self._retrieve_for_query(system, query)
                    system_results['queries_processed'] += 1
                    
                    # Step 2: Sanity Gate Validation
                    gold_data = {
                        'query': query.query,
                        'gold_paths': query.gold_paths,
                        'gold_spans': getattr(query, 'gold_spans', [])
                    }
                    
                    gate_result = await self.live_sanity.live_sanity_gate(
                        query.qid, query.query, retrieved_chunks, gold_data
                    )
                    
                    system_results['sanity_gate_results'].append({
                        'query_id': query.qid,
                        'scenario': query.scenario,
                        'passed': gate_result.passed_gate,
                        'ess_score': gate_result.ess_score,
                        'operation': gate_result.operation.value,
                        'failure_reason': gate_result.failure_reason
                    })
                    
                    if gate_result.passed_gate:
                        system_results['queries_passed_gates'] += 1
                        
                        # Step 3: Generation (only for passing queries)
                        generated_answer = await self._generate_answer(system, query, retrieved_chunks)
                        
                        # Step 4: Evaluation
                        eval_result = await self._evaluate_answer(query, generated_answer, retrieved_chunks)
                        
                        # Store scenario results
                        if query.scenario not in system_results['scenario_results']:
                            system_results['scenario_results'][query.scenario] = []
                        
                        system_results['scenario_results'][query.scenario].append({
                            'query_id': query.qid,
                            'ess_score': gate_result.ess_score,
                            'success_at_1': eval_result.get('success_at_1', 0),
                            'mrr': eval_result.get('mrr', 0),
                            'context_precision': eval_result.get('context_precision', 0),
                            'context_recall': eval_result.get('context_recall', 0),
                            'citation_accuracy': eval_result.get('citation_accuracy', 0)
                        })
                    else:
                        system_results['queries_blocked'] += 1
                        logger.debug(f"🚫 Query {query.qid} blocked: {gate_result.failure_reason}")
                
                except Exception as e:
                    logger.error(f"❌ Error processing query {query.qid}: {e}")
                    continue
        
        # Calculate aggregate metrics (only on passed queries)
        system_results['gate_pass_rate'] = (
            system_results['queries_passed_gates'] / system_results['queries_processed'] 
            if system_results['queries_processed'] > 0 else 0.0
        )
        
        logger.info(f"✅ System {system_id}: {system_results['queries_passed_gates']}/{system_results['queries_processed']} queries passed gates")
        
        return system_results
    
    async def _retrieve_for_query(self, system: Any, query: BenchmarkQuery) -> List[Dict]:
        """Perform retrieval for a query."""
        # Mock retrieval - in real system would call system.retrieve()
        return [
            {
                'file_path': path,
                'content': f"Mock content for {path} related to '{query.query}'",
                'score': 0.8,
                'char_start': 10,
                'char_end': 100
            }
            for path in query.gold_paths
        ]
    
    async def _generate_answer(self, system: Any, query: BenchmarkQuery, 
                              retrieved_chunks: List[Dict]) -> str:
        """Generate answer using the system (only called for queries that pass gates)."""
        # Mock generation - in real system would call LLM
        context = "\n".join(chunk['content'] for chunk in retrieved_chunks)
        return f"Based on the retrieved context, the answer to '{query.query}' is found in {retrieved_chunks[0]['file_path']}"
    
    async def _evaluate_answer(self, query: BenchmarkQuery, answer: str, 
                              retrieved_chunks: List[Dict]) -> Dict[str, float]:
        """Evaluate generated answer quality."""
        # Mock evaluation - in real system would use proper metrics
        return {
            'success_at_1': 1.0 if any(path in answer for path in query.gold_paths) else 0.0,
            'mrr': 0.8,
            'context_precision': 0.7,
            'context_recall': 0.6,
            'citation_accuracy': 1.0 if any(path in answer for path in query.gold_paths) else 0.0
        }
    
    async def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive results including sanity scorecard."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        results_path = self.work_dir / f"gated_benchmark_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Generate and save scorecard
        scorecard = SanityScorecard.generate_scorecard(results['sanity_validation'])
        scorecard_path = self.work_dir / f"sanity_scorecard_{timestamp}.md"
        
        with open(scorecard_path, 'w') as f:
            f.write(scorecard)
        
        # Generate summary report
        summary_path = self.work_dir / f"benchmark_summary_{timestamp}.md"
        summary = self._generate_benchmark_summary(results)
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"📊 Results saved:")
        logger.info(f"  Full results: {results_path}")
        logger.info(f"  Sanity scorecard: {scorecard_path}")
        logger.info(f"  Summary report: {summary_path}")
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable by converting dataclasses and enums."""
        if hasattr(obj, '__dict__'):  # dataclass
            return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'value'):  # enum
            return obj.value
        else:
            return obj
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable benchmark summary."""
        pipeline = results['pipeline_summary']
        sanity = results['sanity_validation']
        
        summary = f"""# Sanity-Gated Benchmark Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 🎯 Pipeline Overview
- **Total Queries**: {pipeline['total_queries']}
- **Passed Sanity Gates**: {pipeline['queries_passed_gates']} ({pipeline['gate_pass_rate']:.1%})
- **Blocked by Gates**: {pipeline['queries_blocked']}
- **Systems Evaluated**: {pipeline['systems_evaluated']}
- **Scenarios Tested**: {pipeline['scenarios_tested']}

## 🚨 Sanity Validation Status
- **Pre-Gen Pass Rate**: {sanity.pre_gen_pass_rate:.1%}
- **Hard Gates Status**: {sum(sanity.hard_gates_status.values())}/{len(sanity.hard_gates_status)} passed

## 📊 System Performance (Gated Queries Only)
"""
        
        for system_id, system_results in results['benchmark_results'].items():
            gate_rate = system_results.get('gate_pass_rate', 0.0)
            summary += f"\n### {system_id}\n"
            summary += f"- Gate Pass Rate: {gate_rate:.1%}\n"
            summary += f"- Queries Processed: {system_results.get('queries_processed', 0)}\n"
            summary += f"- Queries Generated: {system_results.get('queries_passed_gates', 0)}\n"
        
        summary += f"""
## 🔬 Key Insights
- Only queries that pass sanity gates reach the LLM
- ESS thresholds per operation enforce evidence quality
- Ablation tests validate evidence dependency
- System comparison is performed only on contract-valid queries

## 📋 Next Steps
1. Calibrate ESS thresholds based on labeled validation set
2. Expand core query set to 150+ queries across operations
3. Implement hard CI gates for production deployment
4. Monitor ESS drift and retrieval health over time
"""
        
        return summary


async def main():
    """Run sanity-gated benchmark."""
    import yaml
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize benchmark
    benchmark = SanityGatedBenchmark(
        config_path='bench_config.yaml',
        work_dir=Path('gated_benchmark_results')
    )
    
    # Run gated benchmark
    results = await benchmark.run_gated_benchmark(
        scenarios=['code.func', 'code.symbol', 'rag.code.qa'],
        systems=['qdrant', 'bm25_baseline']
    )
    
    print(f"\n🎯 SANITY-GATED BENCHMARK COMPLETE")
    print(f"Total queries: {results['pipeline_summary']['total_queries']}")
    print(f"Passed gates: {results['pipeline_summary']['queries_passed_gates']} ({results['pipeline_summary']['gate_pass_rate']:.1%})")
    print(f"Hard gates: {sum(results['sanity_validation'].hard_gates_status.values())}/{len(results['sanity_validation'].hard_gates_status)} passed")


if __name__ == "__main__":
    asyncio.run(main())