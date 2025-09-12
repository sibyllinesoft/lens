#!/usr/bin/env python3
"""
Counterfactual Logging System with IPS/DR Estimators
====================================================

Implements per-decision logging with propensities and outcomes for unbiased off-policy evaluation.
Supports both Inverse Propensity Scoring (IPS) and Doubly Robust (DR) estimation methods.

Key Features:
- Per-decision propensity π(a|x) logging for router and ANN arms
- Context feature storage with query embeddings and metadata
- Latency counter tracking with cache awareness
- IPS/DR estimation for unbiased metric evaluation
- Integration with Thompson sampling exploration policies

Author: Lens Search Team
Date: 2025-09-12
"""

import logging
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
from datetime import datetime, timedelta


@dataclass
class DecisionContext:
    """Context features for a search decision point"""
    query_id: str
    query_text: str
    query_length: int
    query_entropy: float
    nl_confidence: float
    query_embedding: List[float]  # 768-dim semantic embedding
    user_session: str
    timestamp_ms: int
    is_hard_nl: bool  # entropy>2.5 AND nl_conf>0.8 AND length>6


@dataclass
class RouterDecision:
    """Router arm selection with propensity"""
    context: DecisionContext
    selected_arm: int  # arm_id from contextual bandit
    tau: float
    spend_cap_ms: int
    min_conf_gain: float
    propensity: float  # π(arm|context) from Thompson sampling
    exploration_policy: str  # "thompson_sampling"


@dataclass
class ANNDecision:
    """ANN configuration selection with propensity"""
    context: DecisionContext
    selected_config: int  # config_id from Pareto optimizer
    ef_search: int
    refine_topk: int
    cache_residency: float
    propensity: float  # π(config|context) from optimization policy
    exploration_policy: str  # "pareto_thompson" or "epsilon_greedy"


@dataclass
class DecisionOutcome:
    """Realized outcomes from a search decision"""
    query_id: str
    decision_type: str  # "router" or "ann"
    selected_id: int
    
    # Quality metrics
    ndcg_at_10: float
    sla_recall_at_50: float
    
    # Latency metrics (microseconds for precision)
    p95_latency_us: int
    p99_latency_us: int
    cache_hit_rate: float
    
    # Business metrics
    click_through_rate: float
    user_satisfaction: float
    
    # Technical counters
    lexical_matches: int
    semantic_matches: int
    total_candidates: int
    
    # Timestamp
    completion_timestamp_ms: int


class CounterfactualLogger:
    """
    Logs counterfactual data for unbiased off-policy evaluation
    
    Maintains thread-safe logging of decisions and outcomes with efficient
    storage and retrieval for IPS/DR estimation.
    """
    
    def __init__(self, log_dir: str = "./counterfactual_logs"):
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe storage
        self._decisions_lock = threading.RLock()
        self._outcomes_lock = threading.RLock()
        
        # In-memory buffers for fast access
        self.router_decisions: Dict[str, RouterDecision] = {}
        self.ann_decisions: Dict[str, ANNDecision] = {}
        self.outcomes: Dict[str, DecisionOutcome] = {}
        
        # Batch writing controls
        self.batch_size = 1000
        self.last_flush_time = time.time()
        self.flush_interval_sec = 30
        
        # Metrics tracking
        self.decisions_logged = 0
        self.outcomes_logged = 0
        
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Create log directory if it doesn't exist"""
        import os
        os.makedirs(self.log_dir, exist_ok=True)
    
    def log_router_decision(self, decision: RouterDecision):
        """Log a router arm selection with propensity"""
        with self._decisions_lock:
            self.router_decisions[decision.context.query_id] = decision
            self.decisions_logged += 1
            
        self.logger.debug(f"Router decision logged: query={decision.context.query_id}, "
                         f"arm={decision.selected_arm}, propensity={decision.propensity:.4f}")
        
        self._maybe_flush_batch()
    
    def log_ann_decision(self, decision: ANNDecision):
        """Log an ANN configuration selection with propensity"""
        with self._decisions_lock:
            self.ann_decisions[decision.context.query_id] = decision
            self.decisions_logged += 1
            
        self.logger.debug(f"ANN decision logged: query={decision.context.query_id}, "
                         f"config={decision.selected_config}, propensity={decision.propensity:.4f}")
        
        self._maybe_flush_batch()
    
    def log_outcome(self, outcome: DecisionOutcome):
        """Log realized outcome for a decision"""
        with self._outcomes_lock:
            self.outcomes[outcome.query_id] = outcome
            self.outcomes_logged += 1
            
        self.logger.debug(f"Outcome logged: query={outcome.query_id}, "
                         f"ndcg={outcome.ndcg_at_10:.4f}, p95_us={outcome.p95_latency_us}")
        
        self._maybe_flush_batch()
    
    def _maybe_flush_batch(self):
        """Flush to disk if batch size or time threshold reached"""
        now = time.time()
        should_flush = (
            (self.decisions_logged + self.outcomes_logged) % self.batch_size == 0 or
            (now - self.last_flush_time) > self.flush_interval_sec
        )
        
        if should_flush:
            self.flush_to_disk()
    
    def flush_to_disk(self):
        """Write buffered data to NDJSON files"""
        timestamp = int(time.time())
        
        # Write router decisions
        router_file = f"{self.log_dir}/router_decisions_{timestamp}.ndjson"
        with open(router_file, 'a') as f:
            with self._decisions_lock:
                for decision in self.router_decisions.values():
                    json.dump(asdict(decision), f)
                    f.write('\n')
        
        # Write ANN decisions  
        ann_file = f"{self.log_dir}/ann_decisions_{timestamp}.ndjson"
        with open(ann_file, 'a') as f:
            with self._decisions_lock:
                for decision in self.ann_decisions.values():
                    json.dump(asdict(decision), f)
                    f.write('\n')
        
        # Write outcomes
        outcomes_file = f"{self.log_dir}/outcomes_{timestamp}.ndjson"
        with open(outcomes_file, 'a') as f:
            with self._outcomes_lock:
                for outcome in self.outcomes.values():
                    json.dump(asdict(outcome), f)
                    f.write('\n')
        
        self.last_flush_time = time.time()
        self.logger.info(f"Flushed {self.decisions_logged} decisions, {self.outcomes_logged} outcomes to disk")


class IPSDREstimator:
    """
    Inverse Propensity Scoring and Doubly Robust estimation for unbiased metrics
    
    Implements both IPS and DR estimators for off-policy evaluation of counterfactual
    policies without deploying them to production traffic.
    """
    
    def __init__(self, logger: CounterfactualLogger):
        self.logger_instance = logger
        self.estimator_logger = logging.getLogger(__name__ + ".estimator")
    
    def estimate_policy_value_ips(self, 
                                  target_policy_probs: Dict[str, float],
                                  metric_name: str = "ndcg_at_10") -> Tuple[float, float]:
        """
        Estimate policy value using Inverse Propensity Scoring
        
        V^π = E[w(x,a) * r(x,a)] where w(x,a) = π(a|x) / μ(a|x)
        
        Args:
            target_policy_probs: π(a|x) for each query_id
            metric_name: Which outcome metric to estimate
            
        Returns:
            (estimated_value, confidence_interval)
        """
        weights = []
        rewards = []
        
        # Collect IPS weights and rewards
        for query_id, outcome in self.logger_instance.outcomes.items():
            if query_id not in target_policy_probs:
                continue
                
            # Get logging policy propensity μ(a|x)
            logging_propensity = self._get_logging_propensity(query_id, outcome.decision_type)
            if logging_propensity == 0:
                continue
                
            # Compute importance weight w(x,a) = π(a|x) / μ(a|x)
            target_propensity = target_policy_probs[query_id]
            importance_weight = target_propensity / logging_propensity
            
            # Get reward r(x,a)
            reward = getattr(outcome, metric_name)
            
            weights.append(importance_weight)
            rewards.append(reward)
        
        if not weights:
            return 0.0, 0.0
        
        # Compute IPS estimate
        weights = np.array(weights)
        rewards = np.array(rewards)
        
        ips_estimate = np.mean(weights * rewards)
        
        # Confidence interval (bootstrap or asymptotic)
        n = len(weights)
        se = np.sqrt(np.var(weights * rewards) / n)
        ci_half_width = 1.96 * se  # 95% CI
        
        self.estimator_logger.info(f"IPS estimate for {metric_name}: "
                                  f"{ips_estimate:.4f} ± {ci_half_width:.4f} (n={n})")
        
        return ips_estimate, ci_half_width
    
    def estimate_policy_value_dr(self,
                                 target_policy_probs: Dict[str, float],
                                 reward_model: Any,  # Pre-trained reward model q̂(x,a)
                                 metric_name: str = "ndcg_at_10") -> Tuple[float, float]:
        """
        Estimate policy value using Doubly Robust estimation
        
        V^π_DR = E[q̂(x,π(x)) + w(x,a) * (r(x,a) - q̂(x,a))]
        
        Args:
            target_policy_probs: π(a|x) for each query_id
            reward_model: Model that predicts q̂(x,a) 
            metric_name: Which outcome metric to estimate
            
        Returns:
            (estimated_value, confidence_interval)
        """
        dr_terms = []
        
        for query_id, outcome in self.logger_instance.outcomes.items():
            if query_id not in target_policy_probs:
                continue
            
            # Get context features
            context = self._get_decision_context(query_id)
            if context is None:
                continue
            
            # Direct method term: q̂(x,π(x))
            target_action = self._get_target_action(query_id, target_policy_probs[query_id])
            predicted_reward = reward_model.predict(context, target_action)
            
            # IPS correction term: w(x,a) * (r(x,a) - q̂(x,a))
            logging_propensity = self._get_logging_propensity(query_id, outcome.decision_type)
            if logging_propensity > 0:
                target_propensity = target_policy_probs[query_id]
                importance_weight = target_propensity / logging_propensity
                
                observed_reward = getattr(outcome, metric_name)
                predicted_observed_reward = reward_model.predict(context, outcome.selected_id)
                
                ips_correction = importance_weight * (observed_reward - predicted_observed_reward)
            else:
                ips_correction = 0.0
            
            dr_estimate = predicted_reward + ips_correction
            dr_terms.append(dr_estimate)
        
        if not dr_terms:
            return 0.0, 0.0
        
        # Compute DR estimate
        dr_terms = np.array(dr_terms)
        dr_estimate = np.mean(dr_terms)
        
        # Confidence interval
        n = len(dr_terms)
        se = np.sqrt(np.var(dr_terms) / n)
        ci_half_width = 1.96 * se
        
        self.estimator_logger.info(f"DR estimate for {metric_name}: "
                                  f"{dr_estimate:.4f} ± {ci_half_width:.4f} (n={n})")
        
        return dr_estimate, ci_half_width
    
    def compare_policies(self,
                        policy_a_probs: Dict[str, float],
                        policy_b_probs: Dict[str, float],
                        metric_name: str = "ndcg_at_10",
                        use_dr: bool = True,
                        reward_model: Any = None) -> Dict[str, Any]:
        """
        Compare two policies using counterfactual evaluation
        
        Returns statistical test results for A vs B comparison
        """
        if use_dr and reward_model is not None:
            value_a, ci_a = self.estimate_policy_value_dr(policy_a_probs, reward_model, metric_name)
            value_b, ci_b = self.estimate_policy_value_dr(policy_b_probs, reward_model, metric_name)
            method = "DR"
        else:
            value_a, ci_a = self.estimate_policy_value_ips(policy_a_probs, metric_name)
            value_b, ci_b = self.estimate_policy_value_ips(policy_b_probs, metric_name)
            method = "IPS"
        
        # Statistical significance test
        difference = value_a - value_b
        pooled_se = np.sqrt(ci_a**2 + ci_b**2) / 1.96  # Back-calculate SE from CI
        t_stat = difference / pooled_se if pooled_se > 0 else 0
        
        # Two-tailed p-value approximation
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(t_stat)))
        
        is_significant = p_value < 0.05
        better_policy = "A" if difference > 0 else "B"
        
        return {
            "method": method,
            "metric": metric_name,
            "policy_a_value": value_a,
            "policy_a_ci": ci_a,
            "policy_b_value": value_b, 
            "policy_b_ci": ci_b,
            "difference": difference,
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "better_policy": better_policy,
            "confidence_level": 0.95
        }
    
    def _get_logging_propensity(self, query_id: str, decision_type: str) -> float:
        """Get the logging policy propensity μ(a|x) for a decision"""
        if decision_type == "router":
            decision = self.logger_instance.router_decisions.get(query_id)
            return decision.propensity if decision else 0.0
        elif decision_type == "ann":
            decision = self.logger_instance.ann_decisions.get(query_id)
            return decision.propensity if decision else 0.0
        else:
            return 0.0
    
    def _get_decision_context(self, query_id: str) -> Optional[DecisionContext]:
        """Get context features for a query"""
        router_decision = self.logger_instance.router_decisions.get(query_id)
        if router_decision:
            return router_decision.context
            
        ann_decision = self.logger_instance.ann_decisions.get(query_id)
        if ann_decision:
            return ann_decision.context
            
        return None
    
    def _get_target_action(self, query_id: str, target_prob: float) -> int:
        """Determine target action from target policy probability"""
        # This is a simplified version - in practice you'd need the full target policy
        # to determine which action has the given probability
        context = self._get_decision_context(query_id)
        if context is None:
            return 0
            
        # For demo purposes, assume target action is deterministic given high probability
        if target_prob > 0.5:
            return 1  # Take action
        else:
            return 0  # Don't take action


def test_counterfactual_logging():
    """Test the counterfactual logging system"""
    logger = CounterfactualLogger("./test_counterfactual_logs")
    
    # Create test decision context
    context = DecisionContext(
        query_id="test_query_001",
        query_text="implement async database connection pooling",
        query_length=5,
        query_entropy=3.2,
        nl_confidence=0.85,
        query_embedding=[0.1] * 768,  # Mock embedding
        user_session="session_123",
        timestamp_ms=int(time.time() * 1000),
        is_hard_nl=True
    )
    
    # Log router decision
    router_decision = RouterDecision(
        context=context,
        selected_arm=42,
        tau=0.55,
        spend_cap_ms=4,
        min_conf_gain=0.12,
        propensity=0.15,  # Thompson sampling probability
        exploration_policy="thompson_sampling"
    )
    logger.log_router_decision(router_decision)
    
    # Log ANN decision  
    ann_decision = ANNDecision(
        context=context,
        selected_config=7,
        ef_search=32,
        refine_topk=100,
        cache_residency=0.85,
        propensity=0.22,  # Pareto-Thompson probability
        exploration_policy="pareto_thompson"
    )
    logger.log_ann_decision(ann_decision)
    
    # Log outcome
    outcome = DecisionOutcome(
        query_id="test_query_001",
        decision_type="router",
        selected_id=42,
        ndcg_at_10=0.367,
        sla_recall_at_50=0.689,
        p95_latency_us=121000,  # 121ms
        p99_latency_us=145000,  # 145ms
        cache_hit_rate=0.78,
        click_through_rate=0.23,
        user_satisfaction=4.2,
        lexical_matches=15,
        semantic_matches=28,
        total_candidates=156,
        completion_timestamp_ms=int(time.time() * 1000)
    )
    logger.log_outcome(outcome)
    
    # Force flush to disk
    logger.flush_to_disk()
    
    # Test IPS estimation
    estimator = IPSDREstimator(logger)
    target_probs = {"test_query_001": 0.25}  # Target policy probability
    
    ips_value, ips_ci = estimator.estimate_policy_value_ips(target_probs, "ndcg_at_10")
    print(f"IPS Estimate: {ips_value:.4f} ± {ips_ci:.4f}")
    
    print("✅ Counterfactual logging test completed successfully")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_counterfactual_logging()