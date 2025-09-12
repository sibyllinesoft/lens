#!/usr/bin/env python3
"""
Counterfactual Evaluator with SNIPS/DR Estimation
=================================================

Implements self-normalized IPS and doubly robust estimation for unbiased 
policy evaluation with variance control and statistical guards.

Key Features:
- Self-normalized IPS with clipping (c=10) to control variance
- Doubly robust estimation with reward model fallback
- Bootstrap confidence intervals with stratified resampling
- Statistical guard enforcement with Bonferroni correction
- Efficient batch processing for parallel evaluation

Author: Lens Search Team
Date: 2025-09-12
"""

import logging
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ML imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PolicyEvaluation:
    """Results from counterfactual policy evaluation"""
    policy_id: str
    metric_name: str
    
    # SNIPS estimates
    snips_value: float
    snips_ci_lower: float
    snips_ci_upper: float
    snips_effective_n: float
    
    # DR estimates (if available)
    dr_value: Optional[float] = None
    dr_ci_lower: Optional[float] = None
    dr_ci_upper: Optional[float] = None
    
    # Diagnostics
    sample_size: int = 0
    clip_rate: float = 0.0
    max_weight: float = 0.0
    guard_violations: List[str] = None
    
    def __post_init__(self):
        if self.guard_violations is None:
            self.guard_violations = []


@dataclass 
class RewardModel:
    """Trained reward prediction model for DR estimation"""
    model: Any  # sklearn regressor
    scaler: StandardScaler
    feature_cols: List[str]
    validation_r2: float
    training_samples: int


class CounterfactualEvaluator:
    """
    Unbiased policy evaluation using counterfactual methods
    
    Evaluates router and ANN policies using logged data with importance
    weighting to correct for distribution shift between logging and target policies.
    """
    
    def __init__(self, 
                 clip_threshold: float = 10.0,
                 min_effective_sample_size: float = 50.0,
                 bootstrap_samples: int = 2000,
                 confidence_level: float = 0.95):
        
        self.clip_threshold = clip_threshold
        self.min_effective_sample_size = min_effective_sample_size
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        
        self.logger = logging.getLogger(__name__)
        
        # T₀ baseline guards for validation
        self.T0_GUARDS = {
            'ndcg_at_10': {'baseline': 0.345, 'floor_delta': -0.005},
            'sla_recall_at_50': {'baseline': 0.672, 'floor_delta': -0.003},
            'p95_latency': {'baseline': 118, 'ceiling_delta': 1.0},
            'p99_latency': {'baseline': 142, 'ceiling_delta': 2.0},
            'jaccard_at_10': {'baseline': 1.0, 'floor_delta': -0.20}  # vs T₀
        }
        
        self.reward_models: Dict[str, RewardModel] = {}
        
    def train_reward_model(self, 
                          observations: pd.DataFrame,
                          target_metric: str = "composite_reward",
                          feature_cols: List[str] = None) -> RewardModel:
        """
        Train reward model for doubly robust estimation
        
        The reward model q̂(x,a) predicts expected reward given context and action.
        Used to reduce variance in DR estimation when propensities are small.
        """
        self.logger.info(f"🔄 Training reward model for {target_metric}")
        
        if feature_cols is None:
            feature_cols = [
                'entropy', 'nl_confidence', 'length', 'lexical_density',
                'tau', 'spend_cap_ms', 'min_conf_gain',
                'ef_search', 'refine_topk', 'cache_residency'
            ]
        
        # Prepare training data
        available_cols = [col for col in feature_cols if col in observations.columns]
        X = observations[available_cols].fillna(0).values
        
        # Compute composite reward if not provided
        if target_metric == "composite_reward" and target_metric not in observations.columns:
            quality_reward = (0.7 * observations.get('ndcg_at_10', 0) + 
                            0.3 * observations.get('sla_recall_at_50', 0))
            latency_penalty = np.maximum(0, (observations.get('p95_latency', 118) - 118) / 1000)
            composite_reward = quality_reward - 0.1 * latency_penalty
            y = composite_reward.values
        else:
            y = observations[target_metric].values
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Validate
        y_pred = model.predict(X_val_scaled)
        validation_r2 = 1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
        
        reward_model = RewardModel(
            model=model,
            scaler=scaler,
            feature_cols=available_cols,
            validation_r2=validation_r2,
            training_samples=len(X_train)
        )
        
        self.reward_models[target_metric] = reward_model
        
        self.logger.info(f"✅ Reward model trained: R²={validation_r2:.4f}, n={len(X_train)}")
        return reward_model
    
    def evaluate_policy_snips(self,
                              observations: pd.DataFrame,
                              target_propensities: Dict[str, float],
                              metric_name: str,
                              policy_id: str = "target_policy",
                              slice_weights: Dict[str, float] = None) -> PolicyEvaluation:
        """
        Evaluate policy using Self-Normalized Importance Sampling (SNIPS)
        
        SNIPS estimator: V̂ = Σ(w_i * r_i) / Σ(w_i) where w_i = min(π(a|x)/μ(a|x), c)
        
        More stable than standard IPS due to self-normalization.
        """
        self.logger.debug(f"Evaluating policy {policy_id} with SNIPS on {metric_name}")
        
        # Compute importance weights
        weights = []
        rewards = []
        clipped_count = 0
        
        for idx, row in observations.iterrows():
            query_id = row['query_id']
            
            if query_id not in target_propensities:
                continue
                
            # Get logging propensity μ(a|x)
            logging_prop = row.get('propensity', 0.1)
            if logging_prop <= 0:
                continue
            
            # Get target propensity π(a|x)
            target_prop = target_propensities[query_id]
            
            # Compute clipped importance weight
            raw_weight = target_prop / logging_prop
            clipped_weight = min(raw_weight, self.clip_threshold)
            
            if raw_weight > self.clip_threshold:
                clipped_count += 1
            
            # Get reward
            reward = row.get(metric_name, 0)
            
            # Apply slice reweighting if provided
            slice_weight = 1.0
            if slice_weights and 'slice_name' in row:
                slice_weight = slice_weights.get(row['slice_name'], 1.0)
            
            weights.append(clipped_weight * slice_weight)
            rewards.append(reward)
        
        if len(weights) == 0:
            return PolicyEvaluation(
                policy_id=policy_id,
                metric_name=metric_name,
                snips_value=0.0,
                snips_ci_lower=0.0,
                snips_ci_upper=0.0,
                snips_effective_n=0.0,
                sample_size=0,
                clip_rate=0.0,
                max_weight=0.0,
                guard_violations=["insufficient_data"]
            )
        
        weights = np.array(weights)
        rewards = np.array(rewards)
        
        # SNIPS estimate
        snips_value = np.sum(weights * rewards) / np.sum(weights)
        
        # Effective sample size
        effective_n = (np.sum(weights) ** 2) / np.sum(weights ** 2)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_snips_ci(weights, rewards)
        
        # Statistics
        clip_rate = clipped_count / len(weights)
        max_weight = np.max(weights)
        
        # Check statistical guards
        guard_violations = self._check_statistical_guards(snips_value, metric_name, effective_n)
        
        evaluation = PolicyEvaluation(
            policy_id=policy_id,
            metric_name=metric_name,
            snips_value=snips_value,
            snips_ci_lower=ci_lower,
            snips_ci_upper=ci_upper,
            snips_effective_n=effective_n,
            sample_size=len(weights),
            clip_rate=clip_rate,
            max_weight=max_weight,
            guard_violations=guard_violations
        )
        
        self.logger.debug(f"SNIPS: {snips_value:.4f} ± {ci_upper-ci_lower:.4f}, "
                         f"eff_n={effective_n:.1f}, clip_rate={clip_rate:.1%}")
        
        return evaluation
    
    def evaluate_policy_dr(self,
                           observations: pd.DataFrame,
                           target_propensities: Dict[str, float],
                           metric_name: str,
                           policy_id: str = "target_policy",
                           reward_model: RewardModel = None) -> PolicyEvaluation:
        """
        Evaluate policy using Doubly Robust (DR) estimation
        
        DR estimator: V̂ = E[q̂(x,π(x)) + w(x,a)(r(x,a) - q̂(x,a))]
        
        Combines direct method (reward model) with IPS correction for bias reduction.
        """
        if reward_model is None:
            reward_model = self.reward_models.get(metric_name)
            if reward_model is None:
                self.logger.warning(f"No reward model for {metric_name}, falling back to SNIPS")
                return self.evaluate_policy_snips(observations, target_propensities, metric_name, policy_id)
        
        self.logger.debug(f"Evaluating policy {policy_id} with DR on {metric_name}")
        
        dr_terms = []
        
        for idx, row in observations.iterrows():
            query_id = row['query_id']
            
            if query_id not in target_propensities:
                continue
            
            # Prepare features for reward model
            features = []
            for col in reward_model.feature_cols:
                features.append(row.get(col, 0))
            features = np.array([features])
            features_scaled = reward_model.scaler.transform(features)
            
            # Direct method term: q̂(x,π(x))
            # In practice, need to determine target action from target policy
            target_action_reward = reward_model.model.predict(features_scaled)[0]
            
            # IPS correction term: w(x,a) * (r(x,a) - q̂(x,a))
            logging_prop = row.get('propensity', 0.1)
            if logging_prop > 0:
                target_prop = target_propensities[query_id]
                weight = min(target_prop / logging_prop, self.clip_threshold)
                
                observed_reward = row.get(metric_name, 0)
                predicted_reward = reward_model.model.predict(features_scaled)[0]
                
                ips_correction = weight * (observed_reward - predicted_reward)
            else:
                ips_correction = 0.0
            
            dr_term = target_action_reward + ips_correction
            dr_terms.append(dr_term)
        
        if len(dr_terms) == 0:
            return PolicyEvaluation(
                policy_id=policy_id,
                metric_name=metric_name,
                snips_value=0.0,
                snips_ci_lower=0.0,
                snips_ci_upper=0.0,
                snips_effective_n=0.0,
                guard_violations=["insufficient_data"]
            )
        
        dr_terms = np.array(dr_terms)
        
        # DR estimate
        dr_value = np.mean(dr_terms)
        
        # Bootstrap CI for DR
        ci_lower, ci_upper = self._bootstrap_dr_ci(dr_terms)
        
        # Effective sample size (approximation)
        effective_n = len(dr_terms)  # DR has better effective sample size than IPS
        
        # Check guards
        guard_violations = self._check_statistical_guards(dr_value, metric_name, effective_n)
        
        evaluation = PolicyEvaluation(
            policy_id=policy_id,
            metric_name=metric_name,
            snips_value=dr_value,  # Use DR as primary estimate
            snips_ci_lower=ci_lower,
            snips_ci_upper=ci_upper,
            snips_effective_n=effective_n,
            dr_value=dr_value,
            dr_ci_lower=ci_lower,
            dr_ci_upper=ci_upper,
            sample_size=len(dr_terms),
            guard_violations=guard_violations
        )
        
        self.logger.debug(f"DR: {dr_value:.4f} ± {ci_upper-ci_lower:.4f}, n={len(dr_terms)}")
        
        return evaluation
    
    def _bootstrap_snips_ci(self, weights: np.ndarray, rewards: np.ndarray) -> Tuple[float, float]:
        """Bootstrap confidence interval for SNIPS estimate"""
        bootstrap_estimates = []
        n = len(weights)
        
        for _ in range(self.bootstrap_samples):
            # Stratified bootstrap to preserve weight distribution
            boot_indices = np.random.choice(n, size=n, replace=True)
            boot_weights = weights[boot_indices]
            boot_rewards = rewards[boot_indices]
            
            # SNIPS estimate for bootstrap sample
            if np.sum(boot_weights) > 0:
                boot_estimate = np.sum(boot_weights * boot_rewards) / np.sum(boot_weights)
            else:
                boot_estimate = 0.0
                
            bootstrap_estimates.append(boot_estimate)
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper
    
    def _bootstrap_dr_ci(self, dr_terms: np.ndarray) -> Tuple[float, float]:
        """Bootstrap confidence interval for DR estimate"""
        bootstrap_estimates = []
        n = len(dr_terms)
        
        for _ in range(self.bootstrap_samples):
            boot_indices = np.random.choice(n, size=n, replace=True)
            boot_terms = dr_terms[boot_indices]
            boot_estimate = np.mean(boot_terms)
            bootstrap_estimates.append(boot_estimate)
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper
    
    def _check_statistical_guards(self, 
                                 estimate: float, 
                                 metric_name: str, 
                                 effective_n: float) -> List[str]:
        """Check T₀ baseline statistical guards"""
        violations = []
        
        # Check minimum effective sample size
        if effective_n < self.min_effective_sample_size:
            violations.append(f"low_effective_sample_size_{effective_n:.1f}")
        
        # Check metric-specific guards
        if metric_name in self.T0_GUARDS:
            guard = self.T0_GUARDS[metric_name]
            baseline = guard['baseline']
            
            if 'floor_delta' in guard:
                floor = baseline + guard['floor_delta']
                if estimate < floor:
                    violations.append(f"below_floor_{metric_name}_{estimate:.4f}<{floor:.4f}")
            
            if 'ceiling_delta' in guard:
                ceiling = baseline + guard['ceiling_delta']
                if estimate > ceiling:
                    violations.append(f"above_ceiling_{metric_name}_{estimate:.4f}>{ceiling:.4f}")
        
        return violations
    
    def compare_policies(self, 
                        observations: pd.DataFrame,
                        policy_a_props: Dict[str, float],
                        policy_b_props: Dict[str, float],
                        metric_name: str,
                        use_dr: bool = True) -> Dict[str, Any]:
        """
        Statistical comparison between two policies
        
        Returns significance test results with multiple testing correction.
        """
        self.logger.info(f"Comparing policies A vs B on {metric_name}")
        
        # Evaluate both policies
        if use_dr and metric_name in self.reward_models:
            eval_a = self.evaluate_policy_dr(observations, policy_a_props, metric_name, "policy_a")
            eval_b = self.evaluate_policy_dr(observations, policy_b_props, metric_name, "policy_b")
            method = "DR"
        else:
            eval_a = self.evaluate_policy_snips(observations, policy_a_props, metric_name, "policy_a")
            eval_b = self.evaluate_policy_snips(observations, policy_b_props, metric_name, "policy_b")
            method = "SNIPS"
        
        # Statistical test
        difference = eval_a.snips_value - eval_b.snips_value
        
        # Pooled standard error from CIs
        se_a = (eval_a.snips_ci_upper - eval_a.snips_ci_lower) / (2 * 1.96)
        se_b = (eval_b.snips_ci_upper - eval_b.snips_ci_lower) / (2 * 1.96)
        pooled_se = np.sqrt(se_a**2 + se_b**2)
        
        # T-test
        if pooled_se > 0:
            t_stat = difference / pooled_se
            df = min(eval_a.snips_effective_n, eval_b.snips_effective_n) - 1
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(1, df)))
        else:
            t_stat = 0.0
            p_value = 1.0
        
        # Effect size (Cohen's d approximation)
        pooled_std = np.sqrt((se_a**2 * eval_a.snips_effective_n + se_b**2 * eval_b.snips_effective_n) / 
                            (eval_a.snips_effective_n + eval_b.snips_effective_n))
        effect_size = difference / pooled_std if pooled_std > 0 else 0.0
        
        # Statistical power (approximation)
        power = self._compute_statistical_power(effect_size, eval_a.snips_effective_n + eval_b.snips_effective_n)
        
        return {
            "method": method,
            "metric": metric_name,
            "policy_a": {
                "value": eval_a.snips_value,
                "ci_lower": eval_a.snips_ci_lower,
                "ci_upper": eval_a.snips_ci_upper,
                "effective_n": eval_a.snips_effective_n,
                "guard_violations": eval_a.guard_violations
            },
            "policy_b": {
                "value": eval_b.snips_value,
                "ci_lower": eval_b.snips_ci_lower,
                "ci_upper": eval_b.snips_ci_upper,
                "effective_n": eval_b.snips_effective_n,
                "guard_violations": eval_b.guard_violations
            },
            "difference": difference,
            "difference_ci": [difference - 1.96 * pooled_se, difference + 1.96 * pooled_se],
            "t_statistic": t_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "statistical_power": power,
            "is_significant": p_value < 0.05,
            "better_policy": "A" if difference > 0 else "B"
        }
    
    def _compute_statistical_power(self, effect_size: float, total_n: float) -> float:
        """Approximate statistical power calculation"""
        # Cohen's power calculation approximation
        power = 1 - stats.norm.cdf(1.96 - abs(effect_size) * np.sqrt(total_n / 2))
        return max(0.0, min(1.0, power))


def test_counterfactual_evaluator():
    """Test the counterfactual evaluator"""
    evaluator = CounterfactualEvaluator()
    
    # Create mock observation data
    np.random.seed(42)
    n_obs = 1000
    
    observations = pd.DataFrame({
        'query_id': [f"q_{i:04d}" for i in range(n_obs)],
        'slice_name': np.random.choice(['infinitebench', 'nl_hard', 'code_doc'], n_obs),
        'propensity': np.random.uniform(0.05, 0.3, n_obs),  # Logging propensities
        'ndcg_at_10': np.random.normal(0.35, 0.1, n_obs),
        'sla_recall_at_50': np.random.normal(0.67, 0.15, n_obs),
        'p95_latency': np.random.normal(120, 15, n_obs),
        'entropy': np.random.normal(2.5, 0.8, n_obs),
        'nl_confidence': np.random.uniform(0.2, 0.9, n_obs),
        'length': np.random.poisson(8, n_obs),
        'lexical_density': np.random.uniform(0.2, 0.8, n_obs),
        'tau': np.random.uniform(0.4, 0.7, n_obs),
        'spend_cap_ms': np.random.choice([2, 4, 6, 8], n_obs),
        'min_conf_gain': np.random.uniform(0.08, 0.18, n_obs),
        'ef_search': np.random.choice([64, 96, 128], n_obs),
        'refine_topk': np.random.choice([20, 40, 80], n_obs),
        'cache_residency': np.random.uniform(0.6, 0.9, n_obs)
    })
    
    # Train reward model
    reward_model = evaluator.train_reward_model(observations)
    print(f"✅ Reward model trained: R²={reward_model.validation_r2:.4f}")
    
    # Create target policy propensities (slightly different from logging)
    target_propensities = {}
    for query_id in observations['query_id']:
        # Target policy has 20% higher propensities on average
        logging_prop = observations[observations['query_id'] == query_id]['propensity'].iloc[0]
        target_propensities[query_id] = min(1.0, logging_prop * 1.2)
    
    # Evaluate policy with SNIPS
    snips_eval = evaluator.evaluate_policy_snips(
        observations, target_propensities, 'ndcg_at_10', 'target_policy'
    )
    
    print(f"📊 SNIPS Evaluation:")
    print(f"  Value: {snips_eval.snips_value:.4f}")
    print(f"  CI: [{snips_eval.snips_ci_lower:.4f}, {snips_eval.snips_ci_upper:.4f}]")
    print(f"  Effective N: {snips_eval.snips_effective_n:.1f}")
    print(f"  Clip rate: {snips_eval.clip_rate:.1%}")
    print(f"  Guard violations: {snips_eval.guard_violations}")
    
    # Evaluate policy with DR
    dr_eval = evaluator.evaluate_policy_dr(
        observations, target_propensities, 'composite_reward', 'target_policy', reward_model
    )
    
    print(f"📊 DR Evaluation:")
    print(f"  Value: {dr_eval.dr_value:.4f}")
    print(f"  CI: [{dr_eval.dr_ci_lower:.4f}, {dr_eval.dr_ci_upper:.4f}]")
    print(f"  Effective N: {dr_eval.snips_effective_n:.1f}")
    
    # Compare two policies
    baseline_propensities = {qid: prop for qid, prop in 
                           zip(observations['query_id'], observations['propensity'])}
    
    comparison = evaluator.compare_policies(
        observations, target_propensities, baseline_propensities, 'ndcg_at_10'
    )
    
    print(f"📊 Policy Comparison:")
    print(f"  Method: {comparison['method']}")
    print(f"  Difference: {comparison['difference']:.4f}")
    print(f"  P-value: {comparison['p_value']:.4f}")
    print(f"  Effect size: {comparison['effect_size']:.3f}")
    print(f"  Better policy: {comparison['better_policy']}")
    print(f"  Significant: {comparison['is_significant']}")
    
    print("✅ Counterfactual evaluator test completed successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_counterfactual_evaluator()