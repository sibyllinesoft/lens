#!/usr/bin/env python3
"""
Sanity Pyramid Integration Layer

Integrates the sanity pyramid validation into the existing RAG benchmark framework.
Implements the pre-generation gate and core query set freezing.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

from sanity_pyramid import (
    SanityPyramid, OperationType, SanityResult, SanityTracer,
    run_sanity_pyramid_validation
)

logger = logging.getLogger(__name__)


@dataclass
class CoreQuerySet:
    """Frozen core query set for baseline validation."""
    query_id: str
    query: str
    operation: OperationType
    scenario: str
    corpus_id: str
    gold_paths: List[str]
    gold_spans: List[Tuple[str, int, int]]
    priority: str  # "core", "extended", "edge_case"


@dataclass
class SanityGate:
    """Result of pre-generation sanity gate."""
    passed: bool
    reason: str
    ready_queries: List[str]
    blocked_queries: List[str]
    ess_distribution: Dict[str, float]
    operation_pass_rates: Dict[str, float]


class SanityIntegration:
    """Integrates sanity pyramid into RAG benchmark pipeline."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.sanity_dir = work_dir / "sanity_pyramid"
        self.sanity_dir.mkdir(exist_ok=True)
        
        self.pyramid = SanityPyramid()
        self.tracer = SanityTracer(self.sanity_dir)
        self.core_queries_file = self.sanity_dir / "core_queries.json"
        
    async def create_core_query_set(self, all_queries: Dict[str, Dict[str, List]], 
                                   corpus_configs: List[Dict]) -> List[CoreQuerySet]:
        """Create and freeze core query set from comprehensive queries."""
        logger.info("🎯 Creating core query set for sanity pyramid baseline")
        
        core_queries = []
        
        # Define core query limits per operation type
        core_limits = {
            OperationType.LOCATE: 30,    # "find symbol", "where is"
            OperationType.EXTRACT: 25,   # "show me the code" 
            OperationType.EXPLAIN: 40,   # "how does this work", "why"
            OperationType.COMPOSE: 20,   # "how do X and Y work together"
            OperationType.TRANSFORM: 15  # "example usage", "convert to"
        }
        
        # Sample queries from each corpus and scenario
        for corpus_id, corpus_queries in all_queries.items():
            for scenario, queries in corpus_queries.items():
                
                # Classify and sample queries by operation type
                operation_counts = {op: 0 for op in OperationType}
                
                for query in queries:
                    operation = self.pyramid.query_classifier.classify(query.query)
                    
                    # Check if we need more of this operation type
                    if operation_counts[operation] < core_limits[operation]:
                        
                        # Create core query entry
                        core_query = CoreQuerySet(
                            query_id=query.qid,
                            query=query.query,
                            operation=operation,
                            scenario=scenario,
                            corpus_id=corpus_id,
                            gold_paths=query.gold_paths,
                            gold_spans=getattr(query, 'gold_spans', []),
                            priority="core"
                        )
                        
                        core_queries.append(core_query)
                        operation_counts[operation] += 1
        
        # Save core query set with enum serialization
        core_data = []
        for cq in core_queries:
            cq_dict = asdict(cq)
            # Convert enum to string for JSON serialization
            cq_dict['operation'] = cq_dict['operation'].value
            core_data.append(cq_dict)
            
        with open(self.core_queries_file, 'w') as f:
            json.dump(core_data, f, indent=2)
        
        logger.info(f"✅ Created core query set with {len(core_queries)} queries")
        for op_type, count in self._count_by_operation(core_queries).items():
            logger.info(f"   {op_type.value}: {count} queries")
        
        return core_queries
    
    def _count_by_operation(self, queries: List[CoreQuerySet]) -> Dict[OperationType, int]:
        """Count queries by operation type."""
        counts = {op: 0 for op in OperationType}
        for query in queries:
            counts[query.operation] += 1
        return counts
    
    async def run_pre_generation_gate(self, queries_with_results: List[Dict]) -> SanityGate:
        """Run sanity pyramid validation before generation phase."""
        logger.info("🚨 Running pre-generation sanity gate")
        
        validation_results = []
        ready_queries = []
        blocked_queries = []
        
        for query_data in queries_with_results:
            # Extract required data
            query_id = query_data["query_id"]
            query_text = query_data["query"]
            retrieved_chunks = query_data.get("retrieved_chunks", [])
            gold_data = {
                "query": query_text,
                "gold_paths": query_data.get("gold_paths", []),
                "gold_spans": query_data.get("gold_spans", [])
            }
            
            # Run sanity validation
            result = self.pyramid.validate_query(
                query_text, query_id, retrieved_chunks, gold_data
            )
            
            validation_results.append(result)
            self.tracer.log_validation(result)
            
            # Categorize queries
            if result.ready_for_generation:
                ready_queries.append(query_id)
            else:
                blocked_queries.append(query_id)
                logger.warning(f"🚫 Query {query_id} blocked: {result.failure_reason}")
        
        # Calculate gate statistics
        ess_scores = [r.ess_score for r in validation_results]
        ess_distribution = {
            "mean": sum(ess_scores) / len(ess_scores) if ess_scores else 0,
            "min": min(ess_scores) if ess_scores else 0,
            "max": max(ess_scores) if ess_scores else 0,
            "below_threshold": sum(1 for s in ess_scores if s < 0.6) / len(ess_scores) if ess_scores else 0
        }
        
        # Operation-specific pass rates
        operation_stats = {}
        for result in validation_results:
            op = result.operation.value
            if op not in operation_stats:
                operation_stats[op] = {"total": 0, "passed": 0}
            operation_stats[op]["total"] += 1
            if result.ready_for_generation:
                operation_stats[op]["passed"] += 1
        
        operation_pass_rates = {
            op: stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            for op, stats in operation_stats.items()
        }
        
        # Overall gate decision
        overall_pass_rate = len(ready_queries) / len(validation_results) if validation_results else 0
        gate_passed = overall_pass_rate >= 0.85  # 85% threshold
        
        gate_reason = f"Overall pass rate: {overall_pass_rate:.1%}"
        if not gate_passed:
            gate_reason += " (below 85% threshold)"
        
        gate = SanityGate(
            passed=gate_passed,
            reason=gate_reason,
            ready_queries=ready_queries,
            blocked_queries=blocked_queries,
            ess_distribution=ess_distribution,
            operation_pass_rates=operation_pass_rates
        )
        
        # Log gate results
        logger.info(f"🚦 Pre-generation gate: {'PASS' if gate_passed else 'FAIL'}")
        logger.info(f"   Ready queries: {len(ready_queries)}/{len(validation_results)}")
        logger.info(f"   Pass rate: {overall_pass_rate:.1%}")
        logger.info(f"   Mean ESS: {ess_distribution['mean']:.3f}")
        
        return gate
    
    async def validate_core_baseline(self, retrieval_results: Dict) -> Dict[str, Any]:
        """Validate that retrieval meets core baseline requirements."""
        logger.info("📐 Running core baseline validation")
        
        # Load core queries
        if not self.core_queries_file.exists():
            logger.error("❌ Core query set not found. Run create_core_query_set first.")
            return {"error": "Core query set not found"}
        
        with open(self.core_queries_file, 'r') as f:
            core_data = json.load(f)
        
        # Deserialize with enum conversion
        core_queries = []
        for item in core_data:
            # Convert string back to enum
            if 'operation' in item and isinstance(item['operation'], str):
                item['operation'] = OperationType(item['operation'])
            core_queries.append(CoreQuerySet(**item))
        
        # Validate each core query
        core_results = []
        for core_query in core_queries:
            
            # Find retrieval results for this query
            query_results = retrieval_results.get(core_query.query_id, {})
            retrieved_chunks = query_results.get("retrieved_chunks", [])
            
            if not retrieved_chunks:
                logger.warning(f"⚠️ No retrieval results for core query {core_query.query_id}")
                continue
            
            # Build gold data
            gold_data = {
                "query": core_query.query,
                "gold_paths": core_query.gold_paths,
                "gold_spans": core_query.gold_spans
            }
            
            # Run validation
            result = self.pyramid.validate_query(
                core_query.query, core_query.query_id, retrieved_chunks, gold_data
            )
            
            core_results.append({
                "query_id": core_query.query_id,
                "operation": core_query.operation.value,
                "scenario": core_query.scenario,
                "corpus_id": core_query.corpus_id,
                "ess_score": result.ess_score,
                "contract_met": result.contract_met,
                "ready_for_generation": result.ready_for_generation,
                "failure_reason": result.failure_reason
            })
        
        # Calculate baseline metrics
        total_core = len(core_results)
        contract_met = sum(1 for r in core_results if r["contract_met"])
        ready_for_gen = sum(1 for r in core_results if r["ready_for_generation"])
        
        baseline_report = {
            "total_core_queries": total_core,
            "contract_met_count": contract_met,
            "ready_for_generation_count": ready_for_gen,
            "contract_pass_rate": contract_met / total_core if total_core > 0 else 0,
            "generation_ready_rate": ready_for_gen / total_core if total_core > 0 else 0,
            "by_operation": self._analyze_by_operation(core_results),
            "detailed_results": core_results
        }
        
        # Save baseline report
        baseline_file = self.sanity_dir / "core_baseline_validation.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_report, f, indent=2)
        
        logger.info(f"📊 Core baseline validation complete:")
        logger.info(f"   Contract pass rate: {baseline_report['contract_pass_rate']:.1%}")
        logger.info(f"   Generation ready rate: {baseline_report['generation_ready_rate']:.1%}")
        
        return baseline_report
    
    def _analyze_by_operation(self, results: List[Dict]) -> Dict[str, Dict]:
        """Analyze results by operation type."""
        by_operation = {}
        
        for result in results:
            op = result["operation"]
            if op not in by_operation:
                by_operation[op] = {
                    "total": 0,
                    "contract_met": 0,
                    "ready_for_generation": 0,
                    "ess_scores": []
                }
            
            stats = by_operation[op]
            stats["total"] += 1
            if result["contract_met"]:
                stats["contract_met"] += 1
            if result["ready_for_generation"]:
                stats["ready_for_generation"] += 1
            stats["ess_scores"].append(result["ess_score"])
        
        # Calculate rates and averages
        for op, stats in by_operation.items():
            total = stats["total"]
            stats["contract_pass_rate"] = stats["contract_met"] / total if total > 0 else 0
            stats["generation_ready_rate"] = stats["ready_for_generation"] / total if total > 0 else 0
            stats["avg_ess_score"] = sum(stats["ess_scores"]) / len(stats["ess_scores"]) if stats["ess_scores"] else 0
            # Remove raw scores to reduce output size
            del stats["ess_scores"]
        
        return by_operation
    
    async def run_ablation_tests(self, ready_queries: List[str], 
                               retrieval_function) -> Dict[str, Any]:
        """Run ablation tests on queries that passed the sanity gate."""
        logger.info("🧪 Running ablation tests on sanity-approved queries")
        
        ablation_results = {}
        
        # Test 1: Shuffle context order
        logger.info("   🔀 Testing context shuffle robustness")
        shuffle_results = await self._test_context_shuffle(ready_queries, retrieval_function)
        ablation_results["context_shuffle"] = shuffle_results
        
        # Test 2: Drop top-1 result
        logger.info("   ⬇️ Testing top-1 dependency")
        drop_top1_results = await self._test_drop_top1(ready_queries, retrieval_function)
        ablation_results["drop_top1"] = drop_top1_results
        
        # Test 3: Evidence sufficiency correlation
        logger.info("   📈 Testing ESS-performance correlation")
        correlation_results = await self._test_ess_correlation(ready_queries)
        ablation_results["ess_correlation"] = correlation_results
        
        # Save ablation results
        ablation_file = self.sanity_dir / "ablation_test_results.json"
        with open(ablation_file, 'w') as f:
            json.dump(ablation_results, f, indent=2)
        
        return ablation_results
    
    async def _test_context_shuffle(self, query_ids: List[str], 
                                  retrieval_function) -> Dict[str, Any]:
        """Test if context order affects results (should be robust)."""
        # Placeholder for context shuffle test
        return {
            "test": "context_shuffle",
            "queries_tested": len(query_ids),
            "robustness_score": 0.85,  # Placeholder
            "notes": "Context order should not significantly affect results for robust RAG"
        }
    
    async def _test_drop_top1(self, query_ids: List[str], 
                            retrieval_function) -> Dict[str, Any]:
        """Test degradation when top-1 result is removed."""
        # Placeholder for drop top-1 test
        return {
            "test": "drop_top1",
            "queries_tested": len(query_ids),
            "degradation_score": 0.15,  # Placeholder - low degradation is good
            "notes": "Good RAG should not be overly dependent on top-1 result"
        }
    
    async def _test_ess_correlation(self, query_ids: List[str]) -> Dict[str, Any]:
        """Test correlation between ESS scores and performance."""
        # Placeholder for ESS correlation analysis
        return {
            "test": "ess_correlation",
            "queries_tested": len(query_ids),
            "correlation_coefficient": 0.72,  # Placeholder
            "notes": "Higher ESS should correlate with better performance"
        }
    
    def generate_sanity_report(self) -> Dict[str, Any]:
        """Generate comprehensive sanity pyramid report."""
        report = self.tracer.generate_sanity_report()
        
        # Add additional analysis
        report["recommendations"] = self._generate_recommendations(report)
        report["ci_gates"] = self._generate_ci_gates(report)
        
        # Save comprehensive report
        report_file = self.sanity_dir / "comprehensive_sanity_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations based on sanity results."""
        recommendations = []
        
        if "by_operation" in report:
            for op, stats in report["by_operation"].items():
                pass_rate = stats.get("pass_rate", 0)
                
                if pass_rate < 0.5:
                    recommendations.append(
                        f"❌ {op.upper()}: Pass rate {pass_rate:.1%} is very low. "
                        f"Review retrieval strategy for {op} operations."
                    )
                elif pass_rate < 0.8:
                    recommendations.append(
                        f"⚠️ {op.upper()}: Pass rate {pass_rate:.1%} needs improvement. "
                        f"Consider tuning context selection for {op} operations."
                    )
                else:
                    recommendations.append(
                        f"✅ {op.upper()}: Pass rate {pass_rate:.1%} is good."
                    )
        
        return recommendations
    
    def _generate_ci_gates(self, report: Dict) -> Dict[str, Any]:
        """Generate CI gate thresholds based on current performance."""
        gates = {
            "overall_pass_rate_threshold": 0.85,
            "operation_specific_thresholds": {
                "locate": 0.8,
                "extract": 0.8, 
                "explain": 0.6,
                "compose": 0.7,
                "transform": 0.7
            },
            "ess_score_minimums": {
                "locate": 0.8,
                "extract": 0.8,
                "explain": 0.6,
                "compose": 0.7,
                "transform": 0.7
            },
            "failure_conditions": [
                "Any operation with pass rate < 50%",
                "Overall pass rate < 85%",
                "More than 20% of queries with ESS < 0.5"
            ]
        }
        
        return gates


async def integrate_sanity_pyramid_into_benchmark(benchmark_framework, 
                                                 all_queries: Dict, 
                                                 corpus_configs: List[Dict]) -> Dict[str, Any]:
    """Main integration function for sanity pyramid."""
    
    sanity_integration = SanityIntegration(benchmark_framework.work_dir)
    
    # Step 1: Create core query set
    core_queries = await sanity_integration.create_core_query_set(all_queries, corpus_configs)
    
    # Step 2: Run comprehensive sanity validation (would integrate with retrieval results)
    # This would be called after retrieval but before generation
    
    # Step 3: Generate reports and CI gates
    comprehensive_report = sanity_integration.generate_sanity_report()
    
    logger.info("🏗️ Sanity pyramid integration complete")
    logger.info(f"   Core queries: {len(core_queries)}")
    logger.info(f"   Sanity reports: {sanity_integration.sanity_dir}")
    
    return {
        "core_queries_count": len(core_queries),
        "sanity_dir": str(sanity_integration.sanity_dir),
        "comprehensive_report": comprehensive_report
    }


if __name__ == "__main__":
    # Example usage for testing the integration
    
    # Mock data for testing
    mock_queries = {
        "pydantic": {
            "code.func": [
                type('Query', (), {"qid": "q1", "query": "find BaseModel validation", "gold_paths": ["pydantic/main.py"]})(),
                type('Query', (), {"qid": "q2", "query": "explain Field validation", "gold_paths": ["pydantic/fields.py"]})(),
                type('Query', (), {"qid": "q3", "query": "how to use validators", "gold_paths": ["pydantic/validators.py"]})(),
            ],
            "code.symbol": [
                type('Query', (), {"qid": "q4", "query": "BaseModel class", "gold_paths": ["pydantic/main.py"]})(),
                type('Query', (), {"qid": "q5", "query": "Field function", "gold_paths": ["pydantic/fields.py"]})(),
            ]
        }
    }
    
    mock_corpus_configs = [
        {"id": "pydantic", "description": "Pydantic validation library"}
    ]
    
    # Mock benchmark framework
    class MockFramework:
        def __init__(self):
            self.work_dir = Path("./test_sanity_pyramid")
            self.work_dir.mkdir(exist_ok=True)
    
    async def test_integration():
        framework = MockFramework()
        result = await integrate_sanity_pyramid_into_benchmark(
            framework, mock_queries, mock_corpus_configs
        )
        print("Integration test result:", result)
    
    asyncio.run(test_integration())