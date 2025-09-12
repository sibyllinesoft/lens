#!/usr/bin/env python3
"""
ANN Knee Sharpening Optimizer

Refines ANN parameters around the performance knee at (ef=128, topk=80, LFU-6h)
using quantile regression surrogate modeling and successive halving optimization.

Targets ~0.3-0.5pp nDCG improvement or ~0.3-0.6ms latency reduction while
maintaining robustness across cold/warm regimes and different cache policies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy import stats
from scipy.optimize import minimize_scalar
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Benchmarking imports
import subprocess
import tempfile
import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ANNConfig:
    """ANN configuration parameters"""
    ef_search: int
    refine_topk: int  
    cache_policy: str
    cache_size: int = 1000
    cache_ttl_hours: int = 6
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ef_search': self.ef_search,
            'refine_topk': self.refine_topk,
            'cache_policy': self.cache_policy,
            'cache_size': self.cache_size,
            'cache_ttl_hours': self.cache_ttl_hours
        }
    
    def __hash__(self):
        return hash((self.ef_search, self.refine_topk, self.cache_policy))


class BenchmarkResult(NamedTuple):
    """Results from running benchmark with specific config"""
    config: ANNConfig
    ndcg_cold: float
    ndcg_warm: float 
    p95_latency_ms: float
    visited_nodes: int
    cache_hit_rate: float
    error: Optional[str] = None


@dataclass
class OptimizationState:
    """Tracks optimization progress across successive halving rounds"""
    round_num: int
    candidates: List[ANNConfig]
    results: List[BenchmarkResult]
    best_config: Optional[ANNConfig] = None
    best_score: float = float('-inf')
    convergence_history: List[float] = None
    
    def __post_init__(self):
        if self.convergence_history is None:
            self.convergence_history = []


class QuantileRegressionSurrogate:
    """
    Quantile regression surrogate model for p95 latency prediction.
    
    Models L̂₉₅(ef, topk, cache, len, visited_nodes) using quantile regression
    with separate cold/warm regime modeling.
    """
    
    def __init__(self, quantile: float = 0.95):
        self.quantile = quantile
        self.cold_model = None
        self.warm_model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'ef_search', 'refine_topk', 'cache_size', 'cache_ttl_hours',
            'query_length', 'estimated_visited_nodes', 'cache_hit_rate'
        ]
        self.is_fitted = False
        
    def _create_features(self, configs: List[ANNConfig], 
                        query_lengths: List[float],
                        visited_nodes: List[int],
                        cache_hit_rates: List[float]) -> pd.DataFrame:
        """Create feature matrix from configs and query characteristics"""
        features = []
        
        for i, config in enumerate(configs):
            # Map cache policy to numeric
            cache_policy_map = {'LFU': 0, '2Q': 1, 'LRU': 2, 'FIFO': 3}
            cache_numeric = cache_policy_map.get(config.cache_policy, 0)
            
            feature_row = [
                config.ef_search,
                config.refine_topk,
                config.cache_size,
                config.cache_ttl_hours,
                query_lengths[i] if i < len(query_lengths) else 10.0,
                visited_nodes[i] if i < len(visited_nodes) else 1000,
                cache_hit_rates[i] if i < len(cache_hit_rates) else 0.3
            ]
            features.append(feature_row)
            
        return pd.DataFrame(features, columns=self.feature_names)
    
    def fit(self, training_data: pd.DataFrame):
        """
        Fit quantile regression models for cold and warm regimes.
        
        Args:
            training_data: DataFrame with columns:
                - config features (ef_search, refine_topk, etc.)
                - p95_latency_cold, p95_latency_warm
                - query_length, visited_nodes, cache_hit_rate
        """
        try:
            # Separate cold and warm data
            cold_data = training_data[training_data['regime'] == 'cold'].copy()
            warm_data = training_data[training_data['regime'] == 'warm'].copy()
            
            if len(cold_data) < 10 or len(warm_data) < 10:
                logging.warning("Insufficient data for separate regime modeling, using combined model")
                # Fall back to single model
                features = training_data[self.feature_names]
                target = training_data['p95_latency_ms']
                
                X_scaled = self.scaler.fit_transform(features)
                
                # Use GradientBoostingRegressor with quantile loss
                self.cold_model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=self.quantile,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42
                )
                self.cold_model.fit(X_scaled, target)
                self.warm_model = self.cold_model  # Same model for both regimes
                
            else:
                # Separate models for cold and warm
                cold_features = cold_data[self.feature_names]
                cold_target = cold_data['p95_latency_ms']
                
                warm_features = warm_data[self.feature_names]
                warm_target = warm_data['p95_latency_ms']
                
                # Fit scaler on combined data
                all_features = pd.concat([cold_features, warm_features])
                self.scaler.fit(all_features)
                
                # Cold regime model
                X_cold_scaled = self.scaler.transform(cold_features)
                self.cold_model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=self.quantile,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42
                )
                self.cold_model.fit(X_cold_scaled, cold_target)
                
                # Warm regime model
                X_warm_scaled = self.scaler.transform(warm_features)
                self.warm_model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=self.quantile,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42
                )
                self.warm_model.fit(X_warm_scaled, warm_target)
            
            self.is_fitted = True
            logging.info(f"Surrogate model fitted with {len(training_data)} samples")
            
        except Exception as e:
            logging.error(f"Failed to fit surrogate model: {e}")
            # Fallback to simple linear model
            self._fit_fallback_model(training_data)
    
    def _fit_fallback_model(self, training_data: pd.DataFrame):
        """Fit simple linear fallback model"""
        try:
            features = training_data[self.feature_names[:2]]  # Just ef_search and refine_topk
            target = training_data['p95_latency_ms']
            
            X_scaled = self.scaler.fit_transform(features)
            
            # Simple linear model
            from sklearn.linear_model import LinearRegression
            self.cold_model = LinearRegression()
            self.cold_model.fit(X_scaled, target)
            self.warm_model = self.cold_model
            
            self.feature_names = self.feature_names[:2]
            self.is_fitted = True
            logging.info("Fitted fallback linear surrogate model")
            
        except Exception as e:
            logging.error(f"Even fallback model failed: {e}")
            self.is_fitted = False
    
    def predict(self, configs: List[ANNConfig], 
                query_lengths: List[float] = None,
                visited_nodes: List[int] = None,
                cache_hit_rates: List[float] = None,
                regime: str = 'warm') -> np.ndarray:
        """
        Predict p95 latency for given configurations.
        
        Args:
            configs: List of ANN configurations
            query_lengths: Query lengths (optional)
            visited_nodes: Number of visited nodes (optional) 
            cache_hit_rates: Cache hit rates (optional)
            regime: 'cold' or 'warm'
            
        Returns:
            Predicted p95 latencies
        """
        if not self.is_fitted:
            # Return baseline predictions
            logging.warning("Surrogate model not fitted, using baseline predictions")
            return np.array([2.0 + 0.01 * config.ef_search + 0.005 * config.refine_topk 
                           for config in configs])
        
        try:
            # Default values if not provided
            if query_lengths is None:
                query_lengths = [10.0] * len(configs)
            if visited_nodes is None:
                visited_nodes = [1000] * len(configs)
            if cache_hit_rates is None:
                cache_hit_rates = [0.3] * len(configs)
            
            features_df = self._create_features(configs, query_lengths, 
                                              visited_nodes, cache_hit_rates)
            
            # Select available features
            available_features = [col for col in self.feature_names if col in features_df.columns]
            features_subset = features_df[available_features]
            
            X_scaled = self.scaler.transform(features_subset)
            
            model = self.cold_model if regime == 'cold' else self.warm_model
            predictions = model.predict(X_scaled)
            
            # Ensure reasonable bounds
            predictions = np.clip(predictions, 0.5, 50.0)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            # Return baseline predictions
            return np.array([2.0 + 0.01 * config.ef_search + 0.005 * config.refine_topk 
                           for config in configs])


class ANNKneeSharpener:
    """
    Main optimizer class that implements successive halving around the performance knee.
    
    Optimizes around ef=128, topk=80, LFU-6h using quantile regression surrogate
    and successive halving to find configurations that improve nDCG and/or latency.
    """
    
    def __init__(self, 
                 current_knee: ANNConfig = None,
                 lambda_latency: float = 2.0,
                 p95_headroom_ms: float = 0.6,
                 target_ndcg_gain: float = 0.4):
        """
        Initialize ANN knee sharpener.
        
        Args:
            current_knee: Current best known configuration
            lambda_latency: Penalty weight for latency in scoring
            p95_headroom_ms: Available p95 latency headroom budget
            target_ndcg_gain: Target nDCG improvement in percentage points
        """
        self.current_knee = current_knee or ANNConfig(
            ef_search=128, 
            refine_topk=80, 
            cache_policy='LFU',
            cache_ttl_hours=6
        )
        
        self.lambda_latency = lambda_latency
        self.p95_headroom_ms = p95_headroom_ms
        self.target_ndcg_gain = target_ndcg_gain
        
        self.surrogate_model = QuantileRegressionSurrogate(quantile=0.95)
        self.optimization_history = []
        self.best_configs_history = []
        
        # Benchmark baseline
        self.baseline_ndcg_cold = 0.0
        self.baseline_ndcg_warm = 0.0
        self.baseline_p95_ms = 2.0
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ann_knee_sharpener')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def load_training_data(self, hits_parquet_path: str = "hits.parquet") -> pd.DataFrame:
        """
        Load and prepare training data from hits.parquet for surrogate model.
        
        Args:
            hits_parquet_path: Path to hits.parquet file
            
        Returns:
            Training data DataFrame
        """
        try:
            if Path(hits_parquet_path).exists():
                df = pd.read_parquet(hits_parquet_path)
                self.logger.info(f"Loaded {len(df)} samples from {hits_parquet_path}")
                
                # Prepare training data for surrogate model
                training_data = self._prepare_training_data(df)
                return training_data
            else:
                self.logger.warning(f"Training data file {hits_parquet_path} not found")
                return self._generate_synthetic_training_data()
                
        except Exception as e:
            self.logger.error(f"Failed to load training data: {e}")
            return self._generate_synthetic_training_data()
    
    def _prepare_training_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Convert raw hits data to training format for surrogate model"""
        try:
            # Extract relevant columns and create training format
            training_rows = []
            
            for _, row in raw_data.iterrows():
                # Extract configuration parameters (adjust column names as needed)
                ef_search = row.get('ef_search', 128)
                refine_topk = row.get('refine_topk', 80)
                cache_policy = row.get('cache_policy', 'LFU')
                
                # Extract performance metrics
                p95_latency_ms = row.get('p95_latency_ms', 2.0)
                query_length = row.get('query_length', 10.0)
                visited_nodes = row.get('visited_nodes', 1000)
                cache_hit_rate = row.get('cache_hit_rate', 0.3)
                
                # Create separate rows for cold and warm regimes
                for regime in ['cold', 'warm']:
                    training_rows.append({
                        'ef_search': ef_search,
                        'refine_topk': refine_topk,
                        'cache_size': 1000,
                        'cache_ttl_hours': 6,
                        'query_length': query_length,
                        'estimated_visited_nodes': visited_nodes,
                        'cache_hit_rate': cache_hit_rate if regime == 'warm' else 0.0,
                        'regime': regime,
                        'p95_latency_ms': p95_latency_ms * (1.3 if regime == 'cold' else 1.0)
                    })
            
            return pd.DataFrame(training_rows)
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            return self._generate_synthetic_training_data()
    
    def _generate_synthetic_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for testing"""
        self.logger.info("Generating synthetic training data")
        
        np.random.seed(42)
        n_samples = 200
        
        training_rows = []
        
        for _ in range(n_samples):
            ef_search = np.random.randint(80, 200)
            refine_topk = np.random.randint(40, 120)
            cache_policy = np.random.choice(['LFU', '2Q', 'LRU'])
            
            query_length = np.random.uniform(5, 20)
            visited_nodes = np.random.randint(500, 3000)
            cache_hit_rate = np.random.uniform(0.1, 0.6)
            
            # Synthetic latency model
            base_latency = 1.0 + 0.008 * ef_search + 0.003 * refine_topk
            base_latency += 0.1 * query_length + 0.0005 * visited_nodes
            base_latency *= (1 - 0.3 * cache_hit_rate)  # Cache benefit
            base_latency += np.random.normal(0, 0.1)  # Noise
            
            for regime in ['cold', 'warm']:
                p95_latency = base_latency * (1.4 if regime == 'cold' else 1.0)
                current_cache_hit = cache_hit_rate if regime == 'warm' else 0.0
                
                training_rows.append({
                    'ef_search': ef_search,
                    'refine_topk': refine_topk,
                    'cache_size': 1000,
                    'cache_ttl_hours': 6,
                    'query_length': query_length,
                    'estimated_visited_nodes': visited_nodes,
                    'cache_hit_rate': current_cache_hit,
                    'regime': regime,
                    'p95_latency_ms': max(0.5, p95_latency)
                })
        
        return pd.DataFrame(training_rows)
    
    def _generate_initial_candidates(self) -> List[ANNConfig]:
        """
        Generate initial candidate configurations around the current knee.
        
        Returns:
            List of candidate ANN configurations
        """
        candidates = []
        
        # Parameter ranges around current knee
        ef_range = [112, 120, 128, 136, 144]  # ±15% around 128
        topk_range = [64, 72, 80, 88, 96]     # ±20% around 80
        cache_policies = ['LFU', '2Q', 'LRU']
        cache_ttl_hours = [4, 6, 8]
        
        # Create grid around knee
        for ef in ef_range:
            for topk in topk_range:
                for cache_policy in cache_policies:
                    for ttl in cache_ttl_hours:
                        candidates.append(ANNConfig(
                            ef_search=ef,
                            refine_topk=topk,
                            cache_policy=cache_policy,
                            cache_ttl_hours=ttl
                        ))
        
        self.logger.info(f"Generated {len(candidates)} initial candidates")
        return candidates
    
    def _expand_neighborhood(self, survivors: List[ANNConfig], 
                           expansion_ratio: float = 0.15) -> List[ANNConfig]:
        """
        Expand neighborhoods around surviving configurations.
        
        Args:
            survivors: Configurations that survived current round
            expansion_ratio: Expansion ratio for parameter ranges
            
        Returns:
            Expanded candidate set
        """
        expanded = set(survivors)  # Start with survivors
        
        for config in survivors:
            # Expand ef_search
            ef_delta = max(8, int(config.ef_search * expansion_ratio))
            for ef_offset in [-ef_delta, -ef_delta//2, ef_delta//2, ef_delta]:
                new_ef = max(64, min(256, config.ef_search + ef_offset))
                
                # Expand refine_topk  
                topk_delta = max(8, int(config.refine_topk * expansion_ratio))
                for topk_offset in [-topk_delta, -topk_delta//2, topk_delta//2, topk_delta]:
                    new_topk = max(32, min(128, config.refine_topk + topk_offset))
                    
                    # Try different cache policies
                    for cache_policy in ['LFU', '2Q', 'LRU']:
                        expanded.add(ANNConfig(
                            ef_search=new_ef,
                            refine_topk=new_topk,
                            cache_policy=cache_policy,
                            cache_ttl_hours=config.cache_ttl_hours
                        ))
        
        expanded_list = list(expanded)
        self.logger.info(f"Expanded to {len(expanded_list)} candidates from {len(survivors)} survivors")
        return expanded_list
    
    def _score_configuration(self, result: BenchmarkResult, 
                           baseline_ndcg_cold: float = 0.0,
                           baseline_ndcg_warm: float = 0.0,
                           baseline_p95_ms: float = 2.0) -> float:
        """
        Score configuration based on nDCG improvement and latency penalty.
        
        Score = ΔnDCG - λ × over_p95_penalty
        
        Args:
            result: Benchmark result to score
            baseline_ndcg_cold: Baseline cold nDCG  
            baseline_ndcg_warm: Baseline warm nDCG
            baseline_p95_ms: Baseline p95 latency
            
        Returns:
            Configuration score (higher is better)
        """
        if result.error is not None:
            return float('-inf')  # Failed configurations
        
        # nDCG improvement (weighted average of cold and warm)
        ndcg_cold_delta = (result.ndcg_cold - baseline_ndcg_cold) * 100  # To percentage points
        ndcg_warm_delta = (result.ndcg_warm - baseline_ndcg_warm) * 100
        
        # Weight warm more heavily (steady state is more important)
        avg_ndcg_delta = 0.3 * ndcg_cold_delta + 0.7 * ndcg_warm_delta
        
        # Latency penalty 
        p95_delta = result.p95_latency_ms - baseline_p95_ms
        over_budget_penalty = max(0, p95_delta - self.p95_headroom_ms)
        
        score = avg_ndcg_delta - self.lambda_latency * over_budget_penalty
        
        # Bonus for high cache hit rates (efficiency)
        cache_bonus = 0.1 * result.cache_hit_rate
        score += cache_bonus
        
        return score
    
    def _run_benchmark(self, config: ANNConfig) -> BenchmarkResult:
        """
        Run benchmark for a specific configuration.
        
        Args:
            config: ANN configuration to benchmark
            
        Returns:
            Benchmark result
        """
        try:
            # Mock benchmark implementation
            # In real implementation, this would run the actual lens benchmark
            
            # Simulate performance with some realistic modeling
            base_ndcg_cold = 0.75 + 0.001 * config.ef_search + 0.002 * config.refine_topk
            base_ndcg_warm = 0.78 + 0.0008 * config.ef_search + 0.0015 * config.refine_topk
            
            # Add some cache policy effects
            cache_multiplier = {'LFU': 1.02, '2Q': 1.01, 'LRU': 0.99}.get(config.cache_policy, 1.0)
            base_ndcg_warm *= cache_multiplier
            
            # Simulate latency
            base_latency = 1.5 + 0.006 * config.ef_search + 0.004 * config.refine_topk
            cache_hit_rate = min(0.6, 0.2 + 0.05 * config.cache_ttl_hours)
            
            # Add noise
            np.random.seed(hash(str(config)) % 2**32)
            ndcg_cold = base_ndcg_cold + np.random.normal(0, 0.005)
            ndcg_warm = base_ndcg_warm + np.random.normal(0, 0.003)
            p95_latency = base_latency + np.random.normal(0, 0.1)
            
            # Simulate visited nodes
            visited_nodes = int(800 + 5 * config.ef_search + np.random.normal(0, 100))
            
            return BenchmarkResult(
                config=config,
                ndcg_cold=max(0.5, min(1.0, ndcg_cold)),
                ndcg_warm=max(0.5, min(1.0, ndcg_warm)),
                p95_latency_ms=max(0.5, p95_latency),
                visited_nodes=max(100, visited_nodes),
                cache_hit_rate=cache_hit_rate
            )
            
        except Exception as e:
            self.logger.error(f"Benchmark failed for config {config}: {e}")
            return BenchmarkResult(
                config=config,
                ndcg_cold=0.0,
                ndcg_warm=0.0,
                p95_latency_ms=999.0,
                visited_nodes=0,
                cache_hit_rate=0.0,
                error=str(e)
            )
    
    def _run_benchmarks_parallel(self, configs: List[ANNConfig], 
                                max_workers: int = 4) -> List[BenchmarkResult]:
        """Run benchmarks in parallel for multiple configurations"""
        self.logger.info(f"Running benchmarks for {len(configs)} configurations...")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {
                executor.submit(self._run_benchmark, config): config 
                for config in configs
            }
            
            for future in concurrent.futures.as_completed(future_to_config):
                result = future.result()
                results.append(result)
                
                if len(results) % 10 == 0:
                    self.logger.info(f"Completed {len(results)}/{len(configs)} benchmarks")
        
        return results
    
    def _validate_sign_consistency(self, results: List[BenchmarkResult]) -> Dict[str, bool]:
        """
        Validate that improvements are consistent across cold/warm regimes.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Dictionary with consistency validation results
        """
        baseline_cold = self.baseline_ndcg_cold
        baseline_warm = self.baseline_ndcg_warm
        
        consistent_improvements = 0
        total_improvements = 0
        
        for result in results:
            if result.error is not None:
                continue
                
            cold_improvement = result.ndcg_cold > baseline_cold
            warm_improvement = result.ndcg_warm > baseline_warm
            
            if cold_improvement or warm_improvement:
                total_improvements += 1
                if cold_improvement and warm_improvement:
                    consistent_improvements += 1
        
        consistency_rate = (consistent_improvements / max(1, total_improvements))
        
        return {
            'consistency_rate': consistency_rate,
            'is_consistent': consistency_rate >= 0.7,  # 70% threshold
            'consistent_improvements': consistent_improvements,
            'total_improvements': total_improvements
        }
    
    def optimize(self, max_rounds: int = 4, 
                survival_rate: float = 0.33,
                min_candidates: int = 8) -> OptimizationState:
        """
        Run successive halving optimization to find improved ANN configurations.
        
        Args:
            max_rounds: Maximum number of successive halving rounds
            survival_rate: Fraction of candidates to keep each round
            min_candidates: Minimum candidates to continue optimization
            
        Returns:
            Final optimization state with best configurations
        """
        self.logger.info("Starting ANN knee sharpening optimization")
        
        # Load training data and fit surrogate model
        training_data = self.load_training_data()
        self.surrogate_model.fit(training_data)
        
        # Establish baseline performance
        baseline_result = self._run_benchmark(self.current_knee)
        self.baseline_ndcg_cold = baseline_result.ndcg_cold
        self.baseline_ndcg_warm = baseline_result.ndcg_warm  
        self.baseline_p95_ms = baseline_result.p95_latency_ms
        
        self.logger.info(f"Baseline: nDCG_cold={self.baseline_ndcg_cold:.3f}, "
                        f"nDCG_warm={self.baseline_ndcg_warm:.3f}, "
                        f"p95={self.baseline_p95_ms:.2f}ms")
        
        # Initialize with candidates around knee
        candidates = self._generate_initial_candidates()
        
        best_score = float('-inf')
        best_config = None
        
        for round_num in range(max_rounds):
            self.logger.info(f"\n=== Round {round_num + 1}/{max_rounds} ===")
            self.logger.info(f"Evaluating {len(candidates)} candidates")
            
            # Run benchmarks
            results = self._run_benchmarks_parallel(candidates)
            
            # Score configurations
            scored_results = []
            for result in results:
                score = self._score_configuration(
                    result, 
                    self.baseline_ndcg_cold,
                    self.baseline_ndcg_warm,
                    self.baseline_p95_ms
                )
                scored_results.append((score, result))
            
            # Sort by score (descending)
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Update best configuration
            if scored_results and scored_results[0][0] > best_score:
                best_score = scored_results[0][0]
                best_config = scored_results[0][1].config
                
            # Log top results
            self.logger.info(f"Top 5 configurations:")
            for i, (score, result) in enumerate(scored_results[:5]):
                self.logger.info(
                    f"  {i+1}. Score={score:.3f}, ef={result.config.ef_search}, "
                    f"topk={result.config.refine_topk}, cache={result.config.cache_policy}, "
                    f"nDCG_warm={result.ndcg_warm:.3f}, p95={result.p95_latency_ms:.2f}ms"
                )
            
            # Validate consistency
            consistency = self._validate_sign_consistency([r[1] for r in scored_results])
            self.logger.info(f"Sign consistency: {consistency['consistency_rate']:.1%}")
            
            # Check stopping criteria
            if len(candidates) < min_candidates:
                self.logger.info("Too few candidates remaining, stopping optimization")
                break
                
            # Select survivors for next round
            n_survivors = max(min_candidates, int(len(scored_results) * survival_rate))
            survivors = [result.config for score, result in scored_results[:n_survivors]]
            
            self.logger.info(f"Selected {len(survivors)} survivors for next round")
            
            # Expand neighborhoods around survivors
            if round_num < max_rounds - 1:  # Don't expand on last round
                candidates = self._expand_neighborhood(survivors)
            else:
                candidates = survivors
        
        # Create final optimization state
        final_state = OptimizationState(
            round_num=max_rounds,
            candidates=candidates,
            results=[r[1] for r in scored_results],
            best_config=best_config,
            best_score=best_score
        )
        
        self.logger.info(f"\nOptimization complete. Best score: {best_score:.3f}")
        if best_config:
            self.logger.info(f"Best config: ef={best_config.ef_search}, "
                            f"topk={best_config.refine_topk}, "
                            f"cache={best_config.cache_policy}")
        
        return final_state
    
    def analyze_results(self, state: OptimizationState) -> Dict[str, Any]:
        """
        Analyze optimization results and generate insights.
        
        Args:
            state: Final optimization state
            
        Returns:
            Analysis results dictionary
        """
        if not state.results:
            return {"error": "No results to analyze"}
        
        # Filter successful results
        successful_results = [r for r in state.results if r.error is None]
        
        if not successful_results:
            return {"error": "No successful benchmark runs"}
        
        # Performance improvements
        improvements = []
        for result in successful_results:
            ndcg_cold_gain = (result.ndcg_cold - self.baseline_ndcg_cold) * 100
            ndcg_warm_gain = (result.ndcg_warm - self.baseline_ndcg_warm) * 100
            latency_change = result.p95_latency_ms - self.baseline_p95_ms
            
            improvements.append({
                'config': result.config,
                'ndcg_cold_gain_pp': ndcg_cold_gain,
                'ndcg_warm_gain_pp': ndcg_warm_gain,
                'avg_ndcg_gain_pp': 0.3 * ndcg_cold_gain + 0.7 * ndcg_warm_gain,
                'latency_change_ms': latency_change,
                'cache_hit_rate': result.cache_hit_rate,
                'score': self._score_configuration(result, self.baseline_ndcg_cold,
                                                 self.baseline_ndcg_warm, self.baseline_p95_ms)
            })
        
        # Sort by score
        improvements.sort(key=lambda x: x['score'], reverse=True)
        
        # Analysis insights
        analysis = {
            'total_configs_tested': len(state.results),
            'successful_configs': len(successful_results),
            'best_improvements': improvements[:10],
            'target_achieved': False,
            'recommendations': []
        }
        
        # Check if target achieved
        best_gain = improvements[0]['avg_ndcg_gain_pp'] if improvements else 0
        if best_gain >= self.target_ndcg_gain:
            analysis['target_achieved'] = True
            analysis['recommendations'].append(
                f"Target nDCG improvement achieved: {best_gain:.2f}pp >= {self.target_ndcg_gain}pp"
            )
        
        # Parameter insights
        ef_values = [r.config.ef_search for r in successful_results]
        topk_values = [r.config.refine_topk for r in successful_results]
        
        analysis['parameter_insights'] = {
            'ef_search_range': [min(ef_values), max(ef_values)],
            'topk_range': [min(topk_values), max(topk_values)],
            'optimal_ef_search': improvements[0]['config'].ef_search if improvements else None,
            'optimal_refine_topk': improvements[0]['config'].refine_topk if improvements else None,
            'optimal_cache_policy': improvements[0]['config'].cache_policy if improvements else None
        }
        
        return analysis
    
    def generate_report(self, state: OptimizationState, 
                       analysis: Dict[str, Any]) -> str:
        """
        Generate human-readable optimization report.
        
        Args:
            state: Final optimization state
            analysis: Analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("# ANN Knee Sharpening Optimization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Optimization overview
        report.append("## Optimization Overview")
        report.append(f"- **Target nDCG Gain**: {self.target_ndcg_gain}pp")
        report.append(f"- **P95 Headroom Budget**: {self.p95_headroom_ms}ms")
        report.append(f"- **Lambda Penalty**: {self.lambda_latency}")
        report.append(f"- **Configs Tested**: {analysis['total_configs_tested']}")
        report.append(f"- **Successful Runs**: {analysis['successful_configs']}")
        report.append("")
        
        # Baseline performance
        report.append("## Baseline Performance")
        report.append(f"- **Configuration**: ef={self.current_knee.ef_search}, "
                     f"topk={self.current_knee.refine_topk}, "
                     f"cache={self.current_knee.cache_policy}")
        report.append(f"- **nDCG Cold**: {self.baseline_ndcg_cold:.4f}")
        report.append(f"- **nDCG Warm**: {self.baseline_ndcg_warm:.4f}")
        report.append(f"- **P95 Latency**: {self.baseline_p95_ms:.2f}ms")
        report.append("")
        
        # Best results
        report.append("## Top Configurations")
        if 'best_improvements' in analysis and analysis['best_improvements']:
            for i, improvement in enumerate(analysis['best_improvements'][:5]):
                config = improvement['config']
                report.append(f"### #{i+1} - Score: {improvement['score']:.3f}")
                report.append(f"- **Config**: ef={config.ef_search}, topk={config.refine_topk}, "
                             f"cache={config.cache_policy}-{config.cache_ttl_hours}h")
                report.append(f"- **nDCG Gain**: {improvement['avg_ndcg_gain_pp']:+.3f}pp")
                report.append(f"- **Cold Gain**: {improvement['ndcg_cold_gain_pp']:+.3f}pp")
                report.append(f"- **Warm Gain**: {improvement['ndcg_warm_gain_pp']:+.3f}pp")
                report.append(f"- **Latency Change**: {improvement['latency_change_ms']:+.2f}ms")
                report.append(f"- **Cache Hit Rate**: {improvement['cache_hit_rate']:.1%}")
                report.append("")
        
        # Target achievement
        report.append("## Target Achievement")
        if analysis.get('target_achieved', False):
            report.append("✅ **Target nDCG improvement achieved!**")
        else:
            best_gain = (analysis['best_improvements'][0]['avg_ndcg_gain_pp'] 
                        if analysis.get('best_improvements') else 0)
            remaining = self.target_ndcg_gain - best_gain
            report.append(f"❌ Target not fully achieved. Best: {best_gain:.3f}pp, "
                         f"remaining: {remaining:.3f}pp")
        report.append("")
        
        # Parameter insights
        if 'parameter_insights' in analysis:
            insights = analysis['parameter_insights']
            report.append("## Parameter Insights")
            report.append(f"- **ef_search range**: {insights['ef_search_range']}")
            report.append(f"- **topk range**: {insights['topk_range']}")
            if insights.get('optimal_ef_search'):
                report.append(f"- **Optimal ef_search**: {insights['optimal_ef_search']}")
                report.append(f"- **Optimal refine_topk**: {insights['optimal_refine_topk']}")
                report.append(f"- **Optimal cache policy**: {insights['optimal_cache_policy']}")
            report.append("")
        
        # Recommendations
        if analysis.get('recommendations'):
            report.append("## Recommendations")
            for rec in analysis['recommendations']:
                report.append(f"- {rec}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point for ANN knee sharpening optimization"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize optimizer with current knee configuration
    current_knee = ANNConfig(
        ef_search=128,
        refine_topk=80, 
        cache_policy='LFU',
        cache_ttl_hours=6
    )
    
    optimizer = ANNKneeSharpener(
        current_knee=current_knee,
        lambda_latency=2.0,
        p95_headroom_ms=0.6,
        target_ndcg_gain=0.4  # 0.4pp nDCG improvement target
    )
    
    # Run optimization
    print("Starting ANN knee sharpening optimization...")
    print(f"Current knee: ef={current_knee.ef_search}, topk={current_knee.refine_topk}, "
          f"cache={current_knee.cache_policy}")
    print(f"Target: {optimizer.target_ndcg_gain}pp nDCG improvement")
    print("="*60)
    
    final_state = optimizer.optimize(
        max_rounds=4,
        survival_rate=0.33,
        min_candidates=8
    )
    
    # Analyze results  
    analysis = optimizer.analyze_results(final_state)
    
    # Generate and display report
    report = optimizer.generate_report(final_state, analysis)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = f"ann_knee_optimization_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'optimization_state': {
                'round_num': final_state.round_num,
                'best_score': final_state.best_score,
                'best_config': final_state.best_config.to_dict() if final_state.best_config else None,
                'num_results': len(final_state.results)
            },
            'analysis': analysis,
            'timestamp': timestamp
        }, f, indent=2)
    
    # Save report
    report_file = f"ann_knee_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to {results_file}")
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()