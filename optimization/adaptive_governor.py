#!/usr/bin/env python3
"""
Adaptive Cost/Latency Governor
Per-query optimization using composite objective scoring
score = ΔNDCG − λ·max(0, P95/budget − 1)
"""
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

class QueryType(Enum):
    IDENTIFIER = "identifier"
    STRUCTURAL = "structural"  
    SEMANTIC = "semantic"
    EXACT_MATCH = "exact_match"

@dataclass
class QueryContext:
    query_id: str
    query_type: QueryType
    tenant_id: str
    ess_score: float  # Existing semantic score
    historical_answerable_score: float
    query_complexity: float  # 0.0-1.0
    timestamp: str

@dataclass
class AdaptiveDecision:
    deep_pool_k: int
    enable_reranker: bool
    bypass_cold_paths: bool
    expected_latency_ms: float
    expected_cost: float
    confidence: float

class AdaptiveGovernor:
    def __init__(self, lambda_param: float = 2.0, p95_budget_ms: float = 185.0):
        """
        Initialize adaptive governor with tuned parameters
        
        Args:
            lambda_param: Trade-off weight between quality and latency (higher = prioritize latency)
            p95_budget_ms: Target P95 latency budget (175-185ms range)
        """
        self.lambda_param = lambda_param
        self.p95_budget_ms = p95_budget_ms
        
        # Configuration thresholds
        self.ess_high_threshold = 0.8  # ESS > 0.8 = high confidence
        self.answerable_high_threshold = 0.8  # Historical answerable > 0.8
        
        # Adaptive parameters by scenario
        self.scenario_configs = {
            QueryType.IDENTIFIER: {
                "base_k": 300,
                "high_confidence_k": 200,
                "low_confidence_k": 400,
                "reranker_threshold": 0.6
            },
            QueryType.STRUCTURAL: {
                "base_k": 400, 
                "high_confidence_k": 300,
                "low_confidence_k": 500,
                "reranker_threshold": 0.7
            },
            QueryType.SEMANTIC: {
                "base_k": 500,
                "high_confidence_k": 400, 
                "low_confidence_k": 600,
                "reranker_threshold": 0.8
            },
            QueryType.EXACT_MATCH: {
                "base_k": 150,
                "high_confidence_k": 100,
                "low_confidence_k": 200,
                "reranker_threshold": 0.5
            }
        }
        
        # Performance models (latency vs k, reranker overhead)
        self.latency_model = self._init_latency_model()
        self.cost_model = self._init_cost_model()
        
        # Historical performance tracking
        self.performance_history: List[Dict] = []
        self.current_p95_ms = 175.0  # Current system P95
    
    def _init_latency_model(self) -> Dict:
        """Initialize latency performance models"""
        return {
            "retrieval_base_ms": 45,  # Base retrieval time
            "ms_per_100k": 12,        # Additional time per 100 docs in deep pool
            "reranker_overhead_ms": 42,  # Full reranker overhead
            "marshaling_base_ms": 28,    # Base marshaling time
            "cache_hit_speedup": 0.6     # 60% speedup on cache hit
        }
    
    def _init_cost_model(self) -> Dict:
        """Initialize cost performance models"""
        return {
            "cost_per_100k_docs": 0.0008,  # Cost per 100k docs retrieved
            "reranker_cost": 0.0015,       # Cost of running reranker
            "base_cost": 0.0005            # Base query processing cost
        }
    
    def predict_latency(self, k: int, enable_reranker: bool, bypass_cold: bool = False) -> float:
        """Predict query latency based on configuration"""
        model = self.latency_model
        
        # Retrieval time scales with k
        retrieval_time = model["retrieval_base_ms"] + (k / 100) * model["ms_per_100k"]
        
        # Reranker overhead
        reranker_time = model["reranker_overhead_ms"] if enable_reranker else 0
        
        # Marshaling time
        marshaling_time = model["marshaling_base_ms"]
        
        # Cold path bypass saves ~30% retrieval time
        if bypass_cold:
            retrieval_time *= 0.7
        
        total_latency = retrieval_time + reranker_time + marshaling_time
        
        # Add some realistic variance
        variance = np.random.normal(0, total_latency * 0.05)  # 5% variance
        return max(50, total_latency + variance)
    
    def predict_cost(self, k: int, enable_reranker: bool) -> float:
        """Predict query cost based on configuration"""
        model = self.cost_model
        
        retrieval_cost = (k / 100) * model["cost_per_100k_docs"]
        reranker_cost = model["reranker_cost"] if enable_reranker else 0
        base_cost = model["base_cost"]
        
        return base_cost + retrieval_cost + reranker_cost
    
    def estimate_ndcg_delta(self, base_k: int, candidate_k: int, enable_reranker: bool, 
                          query_context: QueryContext) -> float:
        """Estimate NDCG change from configuration adjustment"""
        
        # NDCG modeling based on query characteristics
        if candidate_k < base_k:
            # Reducing k - risk of recall loss
            k_ratio = candidate_k / base_k
            
            # High ESS + high historical answerable = low risk of loss
            if query_context.ess_score > 0.8 and query_context.historical_answerable_score > 0.8:
                ndcg_delta = -0.02 * (1 - k_ratio)  # Minimal loss for high-confidence queries
            else:
                ndcg_delta = -0.05 * (1 - k_ratio)  # Moderate loss for uncertain queries
        else:
            # Increasing k - diminishing returns
            k_ratio = candidate_k / base_k
            ndcg_delta = 0.03 * np.log(k_ratio)  # Logarithmic improvement
        
        # Reranker boost
        if enable_reranker:
            ndcg_delta += 0.04  # 4% boost from reranking
        
        return ndcg_delta
    
    def composite_objective(self, ndcg_delta: float, predicted_latency: float) -> float:
        """
        Compute composite objective score
        score = ΔNDCG − λ·max(0, P95/budget − 1)
        """
        latency_penalty = max(0, predicted_latency / self.p95_budget_ms - 1)
        score = ndcg_delta - self.lambda_param * latency_penalty
        return score
    
    def make_adaptive_decision(self, query_context: QueryContext) -> AdaptiveDecision:
        """Make adaptive configuration decision for a query"""
        
        query_config = self.scenario_configs[query_context.query_type]
        base_k = query_config["base_k"]
        
        # Determine confidence level
        high_confidence = (
            query_context.ess_score > self.ess_high_threshold and
            query_context.historical_answerable_score > self.answerable_high_threshold
        )
        
        low_confidence = (
            query_context.ess_score < 0.5 or
            query_context.historical_answerable_score < 0.6
        )
        
        # Generate candidate configurations
        candidates = []
        
        if high_confidence:
            # High confidence - can reduce k for speed
            candidate_k = query_config["high_confidence_k"]
            enable_reranker = query_context.ess_score < 0.9  # Skip reranker for very high ESS
        elif low_confidence:
            # Low confidence - increase k for quality
            candidate_k = query_config["low_confidence_k"] 
            enable_reranker = True  # Always rerank for uncertain queries
        else:
            # Medium confidence - baseline
            candidate_k = base_k
            enable_reranker = query_context.ess_score < query_config["reranker_threshold"]
        
        # Predict performance
        predicted_latency = self.predict_latency(candidate_k, enable_reranker)
        predicted_cost = self.predict_cost(candidate_k, enable_reranker)
        
        # Estimate quality impact
        ndcg_delta = self.estimate_ndcg_delta(base_k, candidate_k, enable_reranker, query_context)
        
        # Compute composite score
        composite_score = self.composite_objective(ndcg_delta, predicted_latency)
        
        # Cold path bypass for emergency latency reduction
        bypass_cold = predicted_latency > self.p95_budget_ms * 1.1  # 10% over budget
        if bypass_cold:
            predicted_latency = self.predict_latency(candidate_k, enable_reranker, bypass_cold=True)
            predicted_cost *= 0.9  # Small cost reduction from bypassing cold paths
        
        # Confidence based on query characteristics
        confidence = min(1.0, (query_context.ess_score + query_context.historical_answerable_score) / 2)
        
        decision = AdaptiveDecision(
            deep_pool_k=candidate_k,
            enable_reranker=enable_reranker,
            bypass_cold_paths=bypass_cold,
            expected_latency_ms=predicted_latency,
            expected_cost=predicted_cost,
            confidence=confidence
        )
        
        return decision
    
    def update_performance_tracking(self, query_context: QueryContext, decision: AdaptiveDecision, 
                                  actual_latency: float, actual_cost: float, actual_ndcg: float):
        """Update performance history for model improvement"""
        performance_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query_context": asdict(query_context),
            "decision": asdict(decision),
            "actual_latency_ms": actual_latency,
            "actual_cost": actual_cost,
            "actual_ndcg": actual_ndcg,
            "latency_error": abs(actual_latency - decision.expected_latency_ms),
            "cost_error": abs(actual_cost - decision.expected_cost),
            "composite_score": self.composite_objective(actual_ndcg, actual_latency)
        }
        
        self.performance_history.append(performance_record)
        
        # Keep rolling window
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def tune_lambda_parameter(self) -> float:
        """Auto-tune lambda parameter based on recent performance"""
        if len(self.performance_history) < 50:
            return self.lambda_param
        
        recent_history = self.performance_history[-100:]
        
        # Calculate current P95 latency
        recent_latencies = [r["actual_latency_ms"] for r in recent_history]
        current_p95 = np.percentile(recent_latencies, 95)
        
        # Adjust lambda to hit target P95
        if current_p95 > self.p95_budget_ms * 1.05:  # 5% over budget
            self.lambda_param *= 1.1  # Increase latency penalty
        elif current_p95 < self.p95_budget_ms * 0.95:  # 5% under budget
            self.lambda_param *= 0.95  # Decrease latency penalty (allow more quality focus)
        
        return self.lambda_param
    
    def get_system_status(self) -> Dict:
        """Get current governor status and performance"""
        if not self.performance_history:
            return {"status": "INITIALIZING", "history_size": 0}
        
        recent = self.performance_history[-50:] if len(self.performance_history) >= 50 else self.performance_history
        
        latencies = [r["actual_latency_ms"] for r in recent]
        costs = [r["actual_cost"] for r in recent]
        scores = [r["composite_score"] for r in recent]
        
        return {
            "status": "ACTIVE",
            "lambda_parameter": self.lambda_param,
            "p95_budget_ms": self.p95_budget_ms,
            "current_p95_ms": np.percentile(latencies, 95) if latencies else 0,
            "avg_cost_per_query": np.mean(costs) if costs else 0,
            "avg_composite_score": np.mean(scores) if scores else 0,
            "decisions_made": len(self.performance_history),
            "tuning_status": "AUTO" if len(self.performance_history) > 50 else "LEARNING"
        }

def generate_demo_queries() -> List[QueryContext]:
    """Generate demo query contexts for testing"""
    queries = []
    
    # High-confidence identifier query
    queries.append(QueryContext(
        query_id="q1_high_conf",
        query_type=QueryType.IDENTIFIER,
        tenant_id="tenant_a", 
        ess_score=0.92,
        historical_answerable_score=0.89,
        query_complexity=0.3,
        timestamp=datetime.utcnow().isoformat() + "Z"
    ))
    
    # Low-confidence semantic query
    queries.append(QueryContext(
        query_id="q2_low_conf",
        query_type=QueryType.SEMANTIC,
        tenant_id="tenant_b",
        ess_score=0.45,
        historical_answerable_score=0.52,
        query_complexity=0.8,
        timestamp=datetime.utcnow().isoformat() + "Z"
    ))
    
    # Medium-confidence structural query
    queries.append(QueryContext(
        query_id="q3_medium_conf", 
        query_type=QueryType.STRUCTURAL,
        tenant_id="tenant_c",
        ess_score=0.67,
        historical_answerable_score=0.71,
        query_complexity=0.6,
        timestamp=datetime.utcnow().isoformat() + "Z"
    ))
    
    # Exact match (should be fast)
    queries.append(QueryContext(
        query_id="q4_exact",
        query_type=QueryType.EXACT_MATCH,
        tenant_id="tenant_a",
        ess_score=0.98,
        historical_answerable_score=0.95,
        query_complexity=0.1,
        timestamp=datetime.utcnow().isoformat() + "Z"
    ))
    
    return queries

def simulate_query_execution(governor: AdaptiveGovernor, query: QueryContext) -> Tuple[float, float, float]:
    """Simulate query execution and return actual latency, cost, NDCG"""
    decision = governor.make_adaptive_decision(query)
    
    # Add some realistic variance to predictions
    actual_latency = decision.expected_latency_ms + np.random.normal(0, 10)  # ±10ms variance
    actual_cost = decision.expected_cost + np.random.normal(0, 0.0002)      # Small cost variance
    
    # NDCG simulation based on configuration
    base_ndcg = 0.75  # Baseline NDCG
    if decision.deep_pool_k > 300:
        base_ndcg += 0.03  # Boost from higher k
    if decision.enable_reranker:
        base_ndcg += 0.04  # Boost from reranking
    if decision.bypass_cold_paths:
        base_ndcg -= 0.02  # Small penalty from bypassing cold paths
    
    actual_ndcg = max(0, min(1, base_ndcg + np.random.normal(0, 0.02)))
    
    return actual_latency, actual_cost, actual_ndcg

def main():
    """Demo adaptive governor system"""
    print("⚡ ADAPTIVE COST/LATENCY GOVERNOR")
    print("=" * 50)
    
    # Initialize governor with tuned parameters
    governor = AdaptiveGovernor(lambda_param=2.0, p95_budget_ms=185.0)
    
    print(f"🎯 Target P95: {governor.p95_budget_ms}ms | Lambda: {governor.lambda_param}")
    
    # Generate demo queries
    demo_queries = generate_demo_queries() * 25  # 100 total queries
    
    print(f"\n🔄 Processing {len(demo_queries)} queries with adaptive decisions...")
    
    results = []
    for query in demo_queries:
        # Make adaptive decision
        decision = governor.make_adaptive_decision(query)
        
        # Simulate execution 
        actual_latency, actual_cost, actual_ndcg = simulate_query_execution(governor, query)
        
        # Update performance tracking
        governor.update_performance_tracking(query, decision, actual_latency, actual_cost, actual_ndcg)
        
        results.append({
            "query_id": query.query_id,
            "query_type": query.query_type.value,
            "ess_score": query.ess_score,
            "decision_k": decision.deep_pool_k,
            "enable_reranker": decision.enable_reranker,
            "bypass_cold": decision.bypass_cold_paths,
            "predicted_latency": decision.expected_latency_ms,
            "actual_latency": actual_latency,
            "predicted_cost": decision.expected_cost,
            "actual_cost": actual_cost,
            "confidence": decision.confidence
        })
    
    # Auto-tune lambda
    new_lambda = governor.tune_lambda_parameter()
    if new_lambda != 2.0:
        print(f"🔧 Lambda auto-tuned: {2.0:.2f} → {new_lambda:.2f}")
    
    # System status
    status = governor.get_system_status()
    print(f"\n📊 GOVERNOR STATUS")
    print(f"   Status: {status['status']} | Decisions: {status['decisions_made']}")
    print(f"   Current P95: {status['current_p95_ms']:.1f}ms | Budget: {status['p95_budget_ms']}ms")
    print(f"   Avg Cost/Query: ${status['avg_cost_per_query']:.4f}")
    print(f"   Avg Composite Score: {status['avg_composite_score']:.3f}")
    
    # Optimization summary
    latencies = [r["actual_latency"] for r in results]
    costs = [r["actual_cost"] for r in results]
    
    print(f"\n💰 OPTIMIZATION RESULTS")
    print(f"   Latency P95: {np.percentile(latencies, 95):.1f}ms (target: {governor.p95_budget_ms}ms)")
    print(f"   Avg Cost: ${np.mean(costs):.4f} per query")
    print(f"   Adaptive k Range: {min(r['decision_k'] for r in results)} - {max(r['decision_k'] for r in results)}")
    print(f"   Reranker Usage: {sum(1 for r in results if r['enable_reranker']) / len(results) * 100:.1f}%")
    print(f"   Cold Path Bypass: {sum(1 for r in results if r['bypass_cold']) / len(results) * 100:.1f}%")
    
    # Save detailed results
    with open('adaptive-governor-results.json', 'w') as f:
        json.dump({
            "system_status": status,
            "optimization_summary": {
                "p95_latency_ms": np.percentile(latencies, 95),
                "avg_cost_per_query": np.mean(costs),
                "lambda_parameter": governor.lambda_param,
                "total_queries": len(results)
            },
            "query_results": results[-10:]  # Last 10 for brevity
        }, f, indent=2)
    
    print(f"\n✅ Adaptive governor results saved: adaptive-governor-results.json")

if __name__ == "__main__":
    main()