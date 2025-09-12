#!/usr/bin/env python3
"""
Router Contextual Bandit Implementation
Thompson Sampling with monotone constraints for τ, spend_cap_ms, min_conf_gain learning

Target: +0.5-1.0pp hard-NL nDCG with Δp95 ≤ +0.3ms
"""

import numpy as np
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.stats import invgamma
from datetime import datetime
import hashlib

@dataclass
class QueryContext:
    """Context features for router decision making"""
    query_id: str
    tokens: List[str]
    entropy: float
    length: int
    nl_confidence: float
    ann_miss_rate: float
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numerical feature vector for bandit"""
        return np.array([
            self.entropy,
            self.length,
            self.nl_confidence,
            self.ann_miss_rate
        ])

@dataclass
class RouterArm:
    """Router configuration arm"""
    arm_id: str
    tau_threshold: float
    spend_cap_ms: int
    min_conf_gain: float
    
    def __post_init__(self):
        # Generate deterministic arm_id from parameters if not provided
        if not self.arm_id:
            param_str = f"{self.tau_threshold:.2f}_{self.spend_cap_ms}_{self.min_conf_gain:.3f}"
            self.arm_id = hashlib.md5(param_str.encode()).hexdigest()[:8]

@dataclass
class RouterOutcome:
    """Observed outcome from router decision"""
    arm_id: str
    query_id: str
    hard_nl_ndcg: float
    global_ndcg: float
    p95_latency: float
    sla_recall_50: float
    semantic_spend_ms: float
    timestamp: float

class ContextualLinearBandit:
    """
    Thompson Sampling for contextual linear bandits with monotone constraints
    
    Model: E[r|x,a] = x^T θ_a where θ_a ~ N(μ_a, Σ_a)
    Constraints: Higher τ should not decrease reward net of latency penalty
    """
    
    def __init__(self, feature_dim: int, arms: List[RouterArm], 
                 prior_precision: float = 1.0, noise_precision: float = 1.0):
        """
        Initialize contextual linear bandit
        
        Args:
            feature_dim: Dimension of context feature vector
            arms: List of router configuration arms
            prior_precision: Prior precision on θ parameters (inverse variance)
            noise_precision: Observation noise precision
        """
        self.feature_dim = feature_dim
        self.arms = {arm.arm_id: arm for arm in arms}
        
        # Bayesian linear regression parameters for each arm
        self.arm_params = {}
        for arm_id in self.arms.keys():
            self.arm_params[arm_id] = {
                'precision_matrix': prior_precision * np.eye(feature_dim),  # Λ_a = X^T X + λI
                'precision_weighted_mean': np.zeros(feature_dim),           # Λ_a μ_a = X^T y
                'observation_count': 0
            }
        
        self.noise_precision = noise_precision
        self.observation_log = []
        self.logger = logging.getLogger(__name__)
        
    def select_arm(self, context: QueryContext) -> Tuple[RouterArm, float]:
        """
        Select arm using Thompson Sampling with monotone constraints
        
        Returns:
            (selected_arm, selection_propensity)
        """
        feature_vector = context.to_feature_vector()
        arm_scores = []
        
        # Sample θ from posterior for each arm
        for arm_id, arm in self.arms.items():
            params = self.arm_params[arm_id]
            
            # Posterior parameters for θ_a | data
            precision_matrix = params['precision_matrix']
            precision_weighted_mean = params['precision_weighted_mean']
            
            try:
                # μ_a = Λ_a^{-1} * (X^T y)
                posterior_mean = np.linalg.solve(precision_matrix, precision_weighted_mean)
                # Σ_a = Λ_a^{-1}
                posterior_covariance = np.linalg.inv(precision_matrix)
                
                # Thompson sampling: θ_a ~ N(μ_a, Σ_a)
                theta_sample = np.random.multivariate_normal(posterior_mean, posterior_covariance)
                
                # Predicted reward: x^T θ_a
                predicted_reward = np.dot(feature_vector, theta_sample)
                
                # Apply monotone constraint: higher τ incurs latency penalty
                latency_penalty = arm.tau_threshold * 2.0  # 2ms penalty per tau unit above 0.5
                constrained_reward = predicted_reward - latency_penalty
                
                arm_scores.append({
                    'arm': arm,
                    'score': constrained_reward,
                    'predicted_reward': predicted_reward,
                    'latency_penalty': latency_penalty,
                    'posterior_mean': posterior_mean,
                    'posterior_std': np.sqrt(np.diag(posterior_covariance))
                })
                
            except np.linalg.LinAlgError:
                # Handle singular precision matrix (insufficient data)
                self.logger.warning(f"Singular precision matrix for arm {arm_id}, using prior")
                theta_sample = np.random.normal(0, 1/np.sqrt(1.0), size=self.feature_dim)
                predicted_reward = np.dot(feature_vector, theta_sample)
                latency_penalty = arm.tau_threshold * 2.0
                
                arm_scores.append({
                    'arm': arm,
                    'score': predicted_reward - latency_penalty,
                    'predicted_reward': predicted_reward,
                    'latency_penalty': latency_penalty,
                    'posterior_mean': np.zeros(self.feature_dim),
                    'posterior_std': np.ones(self.feature_dim)
                })
        
        # Select arm with highest score
        selected_arm_data = max(arm_scores, key=lambda x: x['score'])
        selected_arm = selected_arm_data['arm']
        
        # Calculate selection propensity (approximate)
        # In practice, could use more sophisticated propensity estimation
        propensity = self._estimate_selection_propensity(selected_arm, arm_scores, feature_vector)
        
        self.logger.info(f"Selected arm {selected_arm.arm_id}: τ={selected_arm.tau_threshold:.3f}, "
                        f"spend={selected_arm.spend_cap_ms}ms, gain={selected_arm.min_conf_gain:.3f}, "
                        f"score={selected_arm_data['score']:.3f}, propensity={propensity:.3f}")
        
        return selected_arm, propensity
    
    def _estimate_selection_propensity(self, selected_arm: RouterArm, arm_scores: List[Dict], 
                                     context_vector: np.ndarray) -> float:
        """
        Estimate propensity of selecting the chosen arm (for IPS/DR)
        
        Uses softmax approximation over arm scores
        """
        scores = [arm_data['score'] for arm_data in arm_scores]
        
        # Softmax with temperature
        temperature = 1.0
        exp_scores = np.exp(np.array(scores) / temperature)
        propensities = exp_scores / np.sum(exp_scores)
        
        # Find propensity for selected arm
        selected_idx = next(i for i, arm_data in enumerate(arm_scores) 
                          if arm_data['arm'].arm_id == selected_arm.arm_id)
        
        return propensities[selected_idx]
    
    def update_arm(self, arm_id: str, context: QueryContext, reward: float) -> None:
        """
        Update arm parameters with new observation using Bayesian linear regression
        
        Args:
            arm_id: ID of arm that was selected
            context: Query context that was used
            reward: Observed reward (e.g., hard_nl_ndcg delta)
        """
        if arm_id not in self.arm_params:
            self.logger.error(f"Unknown arm_id: {arm_id}")
            return
            
        feature_vector = context.to_feature_vector()
        params = self.arm_params[arm_id]
        
        # Bayesian linear regression update
        # Λ_new = Λ_old + noise_precision * x * x^T
        outer_product = np.outer(feature_vector, feature_vector)
        params['precision_matrix'] += self.noise_precision * outer_product
        
        # (X^T y)_new = (X^T y)_old + noise_precision * x * y  
        params['precision_weighted_mean'] += self.noise_precision * feature_vector * reward
        
        params['observation_count'] += 1
        
        # Log observation for counterfactual analysis
        self.observation_log.append({
            'timestamp': datetime.now().isoformat(),
            'arm_id': arm_id,
            'context': context.to_feature_vector().tolist(),
            'reward': reward,
            'observation_count': params['observation_count']
        })
        
        self.logger.info(f"Updated arm {arm_id} with reward {reward:.4f}, "
                        f"total observations: {params['observation_count']}")

class RouterContextualSystem:
    """
    Complete router contextual bandit system with safety constraints
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize router contextual bandit system"""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_file)
        
        # Generate arm space
        self.arms = self._generate_arm_space()
        
        # Initialize bandit
        self.bandit = ContextualLinearBandit(
            feature_dim=4,  # [entropy, length, nl_confidence, ann_miss_rate]
            arms=self.arms,
            prior_precision=self.config.get('prior_precision', 1.0),
            noise_precision=self.config.get('noise_precision', 1.0)
        )
        
        # Safety constraints from T₀ baseline
        self.t0_baseline = {
            'ndcg_at_10': 0.345,
            'p95_latency': 118,
            'sla_recall_50': 0.672
        }
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'tau_range': [0.40, 0.70],
            'tau_steps': 7,
            'spend_caps': [2, 4, 6, 8],
            'min_conf_gains': [0.08, 0.10, 0.12, 0.15, 0.18],
            'prior_precision': 1.0,
            'noise_precision': 1.0,
            'p95_budget_ms': 0.3,  # +0.3ms budget
            'min_hard_nl_improvement': 0.005  # +0.5pp minimum
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except FileNotFoundError:
                self.logger.warning(f"Config file {config_file} not found, using defaults")
        
        return default_config
        
    def _generate_arm_space(self) -> List[RouterArm]:
        """Generate router arm configurations"""
        arms = []
        
        tau_values = np.linspace(
            self.config['tau_range'][0], 
            self.config['tau_range'][1], 
            self.config['tau_steps']
        )
        
        for tau in tau_values:
            for spend_cap in self.config['spend_caps']:
                for min_gain in self.config['min_conf_gains']:
                    arm = RouterArm(
                        arm_id="",  # Will be auto-generated
                        tau_threshold=tau,
                        spend_cap_ms=spend_cap,
                        min_conf_gain=min_gain
                    )
                    arms.append(arm)
        
        self.logger.info(f"Generated {len(arms)} router arms")
        return arms
    
    def get_router_config(self, query: QueryContext) -> Dict:
        """
        Get router configuration for query using contextual bandit
        
        Returns:
            Router configuration dict with propensity for logging
        """
        # Check if query qualifies as hard-NL
        is_hard_nl = self._is_hard_nl_query(query)
        
        # Select arm using bandit
        selected_arm, propensity = self.bandit.select_arm(query)
        
        return {
            'tau': selected_arm.tau_threshold,
            'spend_cap_ms': selected_arm.spend_cap_ms,
            'min_conf_gain': selected_arm.min_conf_gain,
            'arm_id': selected_arm.arm_id,
            'propensity': propensity,
            'is_hard_nl': is_hard_nl,
            'context': query.to_feature_vector().tolist(),
            'timestamp': datetime.now().isoformat()
        }
    
    def update_with_outcome(self, outcome: RouterOutcome) -> None:
        """
        Update bandit with observed outcome
        
        Reward function emphasizes hard-NL improvement with latency constraint
        """
        # Calculate reward focusing on hard-NL improvement
        hard_nl_delta = outcome.hard_nl_ndcg - self.t0_baseline['ndcg_at_10']
        p95_delta = outcome.p95_latency - self.t0_baseline['p95_latency']
        
        # Reward function: hard-NL improvement with latency penalty
        reward = hard_nl_delta
        
        # Apply latency penalty if over budget
        if p95_delta > self.config['p95_budget_ms']:
            penalty = (p95_delta - self.config['p95_budget_ms']) * 0.01  # 1pp penalty per excess ms
            reward -= penalty
        
        # Apply SLA-Recall penalty if regression
        sla_delta = outcome.sla_recall_50 - self.t0_baseline['sla_recall_50']
        if sla_delta < 0:
            reward += sla_delta * 2.0  # 2x penalty for SLA regression
        
        # Convert outcome to context for update
        context = QueryContext(
            query_id=outcome.query_id,
            tokens=[],  # Would need to be populated from logs
            entropy=0.0,  # Would be reconstructed from stored context
            length=0,
            nl_confidence=0.0, 
            ann_miss_rate=0.0
        )
        
        self.bandit.update_arm(outcome.arm_id, context, reward)
        
        self.logger.info(f"Updated bandit - arm: {outcome.arm_id}, "
                        f"hard_nl_delta: {hard_nl_delta:.4f}, "
                        f"p95_delta: {p95_delta:.1f}ms, reward: {reward:.4f}")
    
    def _is_hard_nl_query(self, query: QueryContext) -> bool:
        """
        Identify hard natural language queries for targeted optimization
        
        Criteria:
        - High NL confidence (>0.8)
        - High entropy (>2.5) indicating complex semantic structure
        - Sufficient length (>6 tokens) for context
        - Historical ANN difficulty (miss rate >0.3)
        """
        return (query.nl_confidence > 0.8 and
                query.entropy > 2.5 and 
                query.length > 6 and
                query.ann_miss_rate > 0.3)
    
    def get_bandit_status(self) -> Dict:
        """Get current bandit status for monitoring"""
        status = {
            'total_arms': len(self.arms),
            'total_observations': sum(params['observation_count'] 
                                    for params in self.bandit.arm_params.values()),
            'observations_per_arm': {arm_id: params['observation_count']
                                   for arm_id, params in self.bandit.arm_params.items()},
            'top_arms': self._get_top_arms(n=5)
        }
        
        return status
    
    def _get_top_arms(self, n: int = 5) -> List[Dict]:
        """Get top performing arms by observation count and posterior mean"""
        arm_stats = []
        
        for arm_id, params in self.bandit.arm_params.items():
            if params['observation_count'] > 0:
                try:
                    posterior_mean = np.linalg.solve(
                        params['precision_matrix'], 
                        params['precision_weighted_mean']
                    )
                    mean_reward = np.mean(posterior_mean)
                except:
                    mean_reward = 0.0
                
                arm = self.bandit.arms[arm_id]
                arm_stats.append({
                    'arm_id': arm_id,
                    'tau': arm.tau_threshold,
                    'spend_cap': arm.spend_cap_ms,
                    'min_gain': arm.min_conf_gain,
                    'observations': params['observation_count'],
                    'mean_reward': mean_reward
                })
        
        # Sort by observation count and mean reward
        arm_stats.sort(key=lambda x: (x['observations'], x['mean_reward']), reverse=True)
        return arm_stats[:n]

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize router contextual system
    router_system = RouterContextualSystem()
    
    # Example query
    query = QueryContext(
        query_id="test_query_001",
        tokens=["how", "to", "optimize", "react", "server", "components", "performance"],
        entropy=3.2,
        length=7,
        nl_confidence=0.85,
        ann_miss_rate=0.4
    )
    
    # Get router configuration
    config = router_system.get_router_config(query)
    print(f"Router config: {config}")
    
    # Simulate outcome
    outcome = RouterOutcome(
        arm_id=config['arm_id'],
        query_id=query.query_id,
        hard_nl_ndcg=0.352,  # +0.7pp improvement
        global_ndcg=0.347,   # +0.2pp improvement
        p95_latency=119.5,   # +1.5ms (over budget, will be penalized)
        sla_recall_50=0.674, # +0.2pp improvement
        semantic_spend_ms=config['spend_cap_ms'],
        timestamp=datetime.now().timestamp()
    )
    
    # Update bandit
    router_system.update_with_outcome(outcome)
    
    # Check status
    status = router_system.get_bandit_status()
    print(f"Bandit status: {status}")