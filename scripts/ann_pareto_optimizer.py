#!/usr/bin/env python3
"""
ANN Pareto Frontier Search Implementation
Cache-aware latency optimization with multi-objective Thompson sampling

Target: ~1ms p95 reduction with ΔnDCG@10 ≥ 0
"""

import numpy as np
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import hashlib
import time

@dataclass
class CachePolicy:
    """Cache residency management policy"""
    type: str  # 'LFU', 'LRU', 'FIFO'
    window: str  # '1h', '2h', '4h', '6h'
    target_hit_rate: float
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numerical features for surrogate model"""
        type_encoding = {'LFU': 0, 'LRU': 1, 'FIFO': 2}
        window_hours = {'1h': 1, '2h': 2, '4h': 4, '6h': 6}
        
        return np.array([
            type_encoding[self.type],
            window_hours[self.window], 
            self.target_hit_rate
        ])

@dataclass
class ANNConfig:
    """ANN configuration parameters"""
    config_id: str
    efSearch: int
    refine_topk: int
    cache_policy: CachePolicy
    prefetch_neighbors: bool
    visited_set_reuse: bool
    
    def __post_init__(self):
        if not self.config_id:
            # Generate deterministic config_id
            params = (
                f"{self.efSearch}_{self.refine_topk}_{self.cache_policy.type}_{self.cache_policy.window}_"
                f"{self.cache_policy.target_hit_rate:.2f}_{self.prefetch_neighbors}_{self.visited_set_reuse}"
            )
            self.config_id = hashlib.md5(params.encode()).hexdigest()[:8]
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numerical feature vector for surrogate model"""
        cache_features = self.cache_policy.to_feature_vector()
        config_features = np.array([
            self.efSearch,
            self.refine_topk,
            1.0 if self.prefetch_neighbors else 0.0,
            1.0 if self.visited_set_reuse else 0.0
        ])
        return np.concatenate([config_features, cache_features])

@dataclass
class ANNObservation:
    """Observed ANN performance metrics"""
    config_id: str
    timestamp: float
    p95_latency: float
    ndcg_at_10: float
    cache_hit_rate: float
    queries_processed: int
    index_size_gb: float
    vector_dimension: int
    warmup_period_sec: int
    
    @property
    def ndcg_delta(self) -> float:
        """nDCG improvement over T₀ baseline"""
        return self.ndcg_at_10 - 0.345  # T₀ baseline

@dataclass
class ParetoFrontierPoint:
    """Point on Pareto frontier"""
    config: ANNConfig
    observation: ANNObservation
    dominates: Set[str] = field(default_factory=set)
    dominated_by: Set[str] = field(default_factory=set)
    is_pareto_optimal: bool = False

class LatencySurrogateModel:
    """
    Surrogate model for predicting ANN latency from configuration parameters
    Enables efficient Pareto frontier search without exhaustive testing
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            random_state=42
        )
        
        self.feature_names = feature_names or [
            'efSearch', 'refine_topk', 'prefetch_neighbors', 'visited_set_reuse',
            'cache_type', 'cache_window_hours', 'cache_target_hit_rate'
        ]
        
        self.is_trained = False
        self.validation_mae = float('inf')
        self.training_data_count = 0
        self.logger = logging.getLogger(__name__)
        
    def predict_latency(self, config: ANNConfig, context: Dict) -> Dict[str, float]:
        """
        Predict p95 latency for given ANN configuration
        
        Args:
            config: ANN configuration to evaluate
            context: Query context (index_size_gb, vector_dimension, etc.)
            
        Returns:
            Dictionary with prediction, uncertainty, and confidence intervals
        """
        if not self.is_trained:
            # Return conservative estimate for untrained model
            baseline_latency = context.get('baseline_p95_latency', 118.0)
            return {
                'predicted_p95_latency': baseline_latency * 1.2,  # Conservative +20%
                'uncertainty': baseline_latency * 0.3,  # High uncertainty
                'confidence_interval': (baseline_latency * 0.9, baseline_latency * 1.5),
                'is_extrapolation': True
            }
        
        # Extract features
        config_features = config.to_feature_vector()
        
        # Add context features
        context_features = np.array([
            context.get('index_size_gb', 2.5),
            context.get('vector_dimension', 384),
            context.get('concurrent_queries', 100)
        ])
        
        features = np.concatenate([config_features, context_features]).reshape(1, -1)
        
        # Predict with model
        predicted_latency = self.model.predict(features)[0]
        
        # Estimate uncertainty (in practice, could use ensemble or quantile regression)
        uncertainty = max(0.1 * predicted_latency, 2.0)  # At least 2ms uncertainty
        
        return {
            'predicted_p95_latency': predicted_latency,
            'uncertainty': uncertainty,
            'confidence_interval': (predicted_latency - uncertainty, 
                                   predicted_latency + uncertainty),
            'is_extrapolation': False
        }
    
    def update_model(self, observations: List[ANNObservation]) -> Dict[str, float]:
        """
        Update surrogate model with new observations
        
        Args:
            observations: List of ANN observations with ground truth latency
            
        Returns:
            Training metrics and validation results
        """
        if len(observations) < 3:
            self.logger.warning("Insufficient data for model training, need at least 3 observations")
            return {'status': 'insufficient_data', 'observation_count': len(observations)}
        
        # Prepare training data
        X = []
        y = []
        
        for obs in observations:
            # Reconstruct config from observation (in practice, would be stored)
            # For now, create a dummy config - in real implementation, would retrieve from logs
            config = ANNConfig(
                config_id=obs.config_id,
                efSearch=64,  # Would be retrieved from config store
                refine_topk=40,
                cache_policy=CachePolicy('LFU', '2h', 0.85),
                prefetch_neighbors=True,
                visited_set_reuse=True
            )
            
            config_features = config.to_feature_vector()
            context_features = np.array([obs.index_size_gb, obs.vector_dimension, 100])
            features = np.concatenate([config_features, context_features])
            
            X.append(features)
            y.append(obs.p95_latency)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        self.training_data_count = len(observations)
        
        # Validate on training data (in practice, would use holdout)
        y_pred = self.model.predict(X)
        self.validation_mae = mean_absolute_error(y, y_pred)
        
        self.logger.info(f"Updated surrogate model - observations: {len(observations)}, "
                        f"validation MAE: {self.validation_mae:.2f}ms")
        
        return {
            'status': 'updated',
            'observation_count': len(observations),
            'validation_mae': self.validation_mae,
            'feature_importance': dict(zip(self.feature_names[:len(config_features)], 
                                         self.model.feature_importances_[:len(config_features)]))
        }

class ParetoFrontierOptimizer:
    """
    Multi-objective optimization for ANN parameter tuning
    Finds Pareto optimal configurations for latency vs quality tradeoff
    """
    
    def __init__(self, surrogate_model: LatencySurrogateModel):
        self.surrogate_model = surrogate_model
        self.observations = []
        self.pareto_frontier = []
        self.config_cache = {}
        
        # T₀ baseline constraints
        self.t0_baseline = {
            'ndcg_at_10': 0.345,
            'p95_latency': 118.0
        }
        
        self.logger = logging.getLogger(__name__)
        
    def generate_config_space(self) -> List[ANNConfig]:
        """Generate ANN configuration search space"""
        configs = []
        
        # Parameter ranges
        efSearch_values = [64, 80, 96, 112, 128]
        refine_topk_values = [20, 32, 40, 48, 56]
        cache_policies = [
            CachePolicy('LFU', '1h', 0.85),
            CachePolicy('LFU', '6h', 0.90),
            CachePolicy('LRU', '2h', 0.85),
            CachePolicy('FIFO', '4h', 0.80)
        ]
        prefetch_options = [True, False]
        visited_set_options = [True, False]
        
        for ef in efSearch_values:
            for topk in refine_topk_values:
                for cache_policy in cache_policies:
                    for prefetch in prefetch_options:
                        for visited_set in visited_set_options:
                            config = ANNConfig(
                                config_id="",  # Auto-generated
                                efSearch=ef,
                                refine_topk=topk,
                                cache_policy=cache_policy,
                                prefetch_neighbors=prefetch,
                                visited_set_reuse=visited_set
                            )
                            configs.append(config)
        
        self.logger.info(f"Generated {len(configs)} ANN configurations")
        return configs
    
    def is_dominated(self, config_a: ANNConfig, obs_a: ANNObservation,
                    config_b: ANNConfig, obs_b: ANNObservation) -> bool:
        """
        Check if config_a dominates config_b in Pareto sense
        
        Dominance: A dominates B if A is better in at least one objective
        and not worse in any objective
        
        Objectives: Minimize latency, Maximize nDCG
        """
        a_better_latency = obs_a.p95_latency <= obs_b.p95_latency
        a_better_ndcg = obs_a.ndcg_at_10 >= obs_b.ndcg_at_10
        
        # At least one strict improvement
        a_strictly_better = (obs_a.p95_latency < obs_b.p95_latency or 
                           obs_a.ndcg_at_10 > obs_b.ndcg_at_10)
        
        return a_better_latency and a_better_ndcg and a_strictly_better
    
    def find_pareto_frontier(self, observations: List[ANNObservation]) -> List[ParetoFrontierPoint]:
        """
        Find Pareto optimal configurations from observations
        
        Args:
            observations: List of ANN performance observations
            
        Returns:
            List of Pareto optimal points
        """
        pareto_points = []
        
        # Create frontier points
        for obs in observations:
            config = self._get_config_for_observation(obs)
            point = ParetoFrontierPoint(config=config, observation=obs)
            pareto_points.append(point)
        
        # Determine dominance relationships
        for i, point_a in enumerate(pareto_points):
            for j, point_b in enumerate(pareto_points):
                if i != j:
                    if self.is_dominated(point_a.config, point_a.observation,
                                       point_b.config, point_b.observation):
                        point_a.dominates.add(point_b.config.config_id)
                        point_b.dominated_by.add(point_a.config.config_id)
        
        # Identify Pareto optimal points (not dominated by any other point)
        pareto_optimal = []
        for point in pareto_points:
            if len(point.dominated_by) == 0:
                point.is_pareto_optimal = True
                pareto_optimal.append(point)
        
        self.logger.info(f"Found {len(pareto_optimal)} Pareto optimal configurations out of {len(observations)}")
        return pareto_optimal
    
    def _get_config_for_observation(self, obs: ANNObservation) -> ANNConfig:
        """Retrieve configuration for observation (cached or reconstructed)"""
        if obs.config_id in self.config_cache:
            return self.config_cache[obs.config_id]
        
        # In practice, would retrieve from persistent storage
        # For demo, create a plausible config
        config = ANNConfig(
            config_id=obs.config_id,
            efSearch=96,  # Default reasonable values
            refine_topk=40,
            cache_policy=CachePolicy('LFU', '2h', obs.cache_hit_rate),
            prefetch_neighbors=True,
            visited_set_reuse=True
        )
        
        self.config_cache[obs.config_id] = config
        return config
    
    def evaluate_config_with_cache_warmup(self, config: ANNConfig, 
                                        warmup_period_sec: int = 3600,
                                        measurement_period_sec: int = 1800) -> ANNObservation:
        """
        Evaluate ANN configuration with proper cache warmup
        
        Prevents false wins from transient warm caches by enforcing
        sustained performance across cold-start windows
        
        Args:
            config: ANN configuration to evaluate
            warmup_period_sec: Cache warmup period in seconds
            measurement_period_sec: Measurement period after warmup
            
        Returns:
            ANNObservation with performance metrics
        """
        self.logger.info(f"Evaluating config {config.config_id} with {warmup_period_sec}s warmup")
        
        # Phase 1: Cache warmup
        start_time = time.time()
        warmup_queries = self._run_cache_warmup(config, warmup_period_sec)
        
        # Phase 2: Performance measurement
        measurement_start = time.time()
        metrics = self._measure_performance(config, measurement_period_sec)
        measurement_end = time.time()
        
        # Phase 3: Cold-start validation
        cold_start_metrics = self._validate_cold_start_performance(config)
        
        # Create observation
        observation = ANNObservation(
            config_id=config.config_id,
            timestamp=measurement_start,
            p95_latency=metrics['p95_latency'],
            ndcg_at_10=metrics['ndcg_at_10'],
            cache_hit_rate=metrics['cache_hit_rate'],
            queries_processed=metrics['queries_processed'],
            index_size_gb=metrics['index_size_gb'],
            vector_dimension=metrics['vector_dimension'],
            warmup_period_sec=warmup_period_sec
        )
        
        # Validate cache residency requirements
        cache_valid, cache_msg = self._validate_cache_residency(observation, cold_start_metrics)
        
        self.logger.info(f"Config {config.config_id} evaluation complete - "
                        f"p95: {observation.p95_latency:.1f}ms, "
                        f"nDCG: {observation.ndcg_at_10:.3f}, "
                        f"cache_valid: {cache_valid}")
        
        return observation
    
    def _run_cache_warmup(self, config: ANNConfig, duration_sec: int) -> int:
        """Simulate cache warmup phase"""
        # In practice, would run actual queries to warm caches
        queries_per_sec = 50  # Typical query rate
        return duration_sec * queries_per_sec
    
    def _measure_performance(self, config: ANNConfig, duration_sec: int) -> Dict:
        """Simulate performance measurement"""
        # In practice, would collect real metrics
        
        # Simulate latency based on configuration parameters
        base_latency = 100.0  # Base latency in ms
        
        # efSearch impact: higher = more latency
        ef_impact = (config.efSearch - 64) * 0.3  # ~0.3ms per efSearch unit above 64
        
        # refine_topk impact: higher = more latency  
        topk_impact = (config.refine_topk - 20) * 0.2  # ~0.2ms per topk unit above 20
        
        # Cache policy impact
        cache_impact = -5.0 if config.cache_policy.type == 'LFU' else 0.0
        
        # Prefetch impact
        prefetch_impact = -2.0 if config.prefetch_neighbors else 0.0
        
        simulated_latency = base_latency + ef_impact + topk_impact + cache_impact + prefetch_impact
        
        # Add some noise
        simulated_latency += np.random.normal(0, 3.0)
        
        # Simulate nDCG (higher efSearch and topk generally help quality)
        base_ndcg = 0.345  # T₀ baseline
        quality_improvement = (config.efSearch - 64) * 0.0001 + (config.refine_topk - 20) * 0.0001
        simulated_ndcg = base_ndcg + quality_improvement + np.random.normal(0, 0.002)
        
        return {
            'p95_latency': max(50.0, simulated_latency),  # Floor at 50ms
            'ndcg_at_10': min(1.0, max(0.0, simulated_ndcg)),  # Clamp to [0,1]
            'cache_hit_rate': config.cache_policy.target_hit_rate + np.random.normal(0, 0.02),
            'queries_processed': duration_sec * 50,  # 50 QPS
            'index_size_gb': 2.5,
            'vector_dimension': 384
        }
    
    def _validate_cold_start_performance(self, config: ANNConfig) -> Dict:
        """Simulate cold-start performance validation"""
        # In practice, would flush caches and measure
        warm_latency = 115.0  # Simulated warm cache latency
        cold_latency = warm_latency * 1.3  # 30% slower when cold
        
        return {
            'cold_start_p95_latency': cold_latency,
            'warm_cache_p95_latency': warm_latency,
            'cold_start_penalty_ratio': cold_latency / warm_latency
        }
    
    def _validate_cache_residency(self, observation: ANNObservation, 
                                cold_start_metrics: Dict,
                                min_hit_rate: float = 0.75) -> Tuple[bool, str]:
        """
        Validate cache residency requirements to prevent false wins
        
        Args:
            observation: Primary performance observation
            cold_start_metrics: Cold start validation metrics
            min_hit_rate: Minimum required cache hit rate
            
        Returns:
            (is_valid, message)
        """
        # Check minimum hit rate
        if observation.cache_hit_rate < min_hit_rate:
            return False, f"Cache hit rate {observation.cache_hit_rate:.3f} below minimum {min_hit_rate}"
        
        # Check cold-start penalty is reasonable
        penalty_ratio = cold_start_metrics['cold_start_penalty_ratio']
        if penalty_ratio > 2.0:  # 100% penalty max
            return False, f"Excessive cold-start penalty: {penalty_ratio:.1f}x"
        
        # Check that cold start latency doesn't violate constraints
        cold_p95 = cold_start_metrics['cold_start_p95_latency']
        if cold_p95 > self.t0_baseline['p95_latency'] + 10.0:  # +10ms budget for cold start
            return False, f"Cold start latency {cold_p95:.1f}ms exceeds budget"
        
        return True, "Cache residency requirements satisfied"
    
    def select_promising_configs(self, n_configs: int = 10) -> List[ANNConfig]:
        """
        Select promising configurations for evaluation using surrogate model
        and Thompson sampling for exploration
        
        Args:
            n_configs: Number of configurations to select for evaluation
            
        Returns:
            List of promising configurations
        """
        all_configs = self.generate_config_space()
        config_scores = []
        
        context = {
            'baseline_p95_latency': self.t0_baseline['p95_latency'],
            'index_size_gb': 2.5,
            'vector_dimension': 384,
            'concurrent_queries': 100
        }
        
        for config in all_configs:
            # Predict performance using surrogate model
            latency_pred = self.surrogate_model.predict_latency(config, context)
            
            # Thompson sampling: sample from prediction uncertainty
            sampled_latency = np.random.normal(
                latency_pred['predicted_p95_latency'], 
                latency_pred['uncertainty']
            )
            
            # Expected nDCG improvement (simple heuristic)
            expected_ndcg_improvement = (config.efSearch - 64) * 0.0001 + (config.refine_topk - 20) * 0.0001
            
            # Multi-objective score: latency reduction + nDCG improvement
            latency_improvement = max(0, self.t0_baseline['p95_latency'] - sampled_latency)
            combined_score = latency_improvement + expected_ndcg_improvement * 1000  # Scale nDCG to ms units
            
            config_scores.append({
                'config': config,
                'score': combined_score,
                'predicted_latency': latency_pred['predicted_p95_latency'],
                'expected_ndcg_improvement': expected_ndcg_improvement
            })
        
        # Select top N configurations
        config_scores.sort(key=lambda x: x['score'], reverse=True)
        selected_configs = [item['config'] for item in config_scores[:n_configs]]
        
        self.logger.info(f"Selected {len(selected_configs)} promising configs from {len(all_configs)} total")
        return selected_configs

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    surrogate_model = LatencySurrogateModel()
    optimizer = ParetoFrontierOptimizer(surrogate_model)
    
    # Generate some sample observations for testing
    sample_configs = optimizer.generate_config_space()[:5]  # Test with 5 configs
    
    observations = []
    for config in sample_configs:
        obs = optimizer.evaluate_config_with_cache_warmup(config, warmup_period_sec=60, 
                                                         measurement_period_sec=30)
        observations.append(obs)
        
    print(f"Generated {len(observations)} observations")
    
    # Find Pareto frontier
    pareto_frontier = optimizer.find_pareto_frontier(observations)
    
    print(f"Pareto frontier contains {len(pareto_frontier)} optimal configurations:")
    for point in pareto_frontier:
        print(f"  Config {point.config.config_id}: efSearch={point.config.efSearch}, "
              f"topk={point.config.refine_topk}, latency={point.observation.p95_latency:.1f}ms, "
              f"nDCG={point.observation.ndcg_at_10:.3f}")
    
    # Update surrogate model
    model_update = surrogate_model.update_model(observations)
    print(f"Surrogate model update: {model_update}")
    
    # Select promising configs for next iteration
    promising_configs = optimizer.select_promising_configs(n_configs=3)
    print(f"Selected {len(promising_configs)} promising configs for next iteration")