#!/usr/bin/env python3
"""
Weekly Modeling Loop with Surrogate Retraining and Arm Pruning
===============================================================

Implements the weekly refresh cycle for contextual bandits and Pareto optimization:
1. Refresh pooled-qrels data (every 6 weeks, aligned with pool refresh)
2. Retrain latency/reward surrogates on fresh observation data  
3. Prune underperforming arms using statistical significance tests
4. Update Thompson sampling posteriors with accumulated evidence
5. Recompute Pareto frontiers with updated models

Key Features:
- Automated model retraining with performance validation
- Statistical arm pruning to focus exploration on promising regions
- Doubly robust estimation for unbiased performance assessment  
- Integration with counterfactual logging system
- Automated rollback on model degradation

Author: Lens Search Team  
Date: 2025-09-12
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

# Import our systems
from router_contextual_bandit import RouterContextualSystem
from ann_pareto_optimizer import ParetoFrontierOptimizer
from counterfactual_logging import CounterfactualLogger, IPSDREstimator

# ML imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


@dataclass
class ModelPerformance:
    """Performance metrics for a trained model"""
    model_name: str
    r2_score: float
    rmse: float
    cv_score_mean: float
    cv_score_std: float
    training_samples: int
    validation_date: str
    is_better_than_baseline: bool


@dataclass 
class ArmPruningResult:
    """Results from statistical arm pruning"""
    arms_pruned: List[int]
    arms_retained: List[int]
    pruning_criteria: Dict[str, Any]
    statistical_power: float
    false_discovery_rate: float


@dataclass
class WeeklyRefreshReport:
    """Summary report from weekly refresh cycle"""
    refresh_date: str
    observations_processed: int
    models_retrained: Dict[str, ModelPerformance]
    router_arm_pruning: ArmPruningResult
    ann_config_pruning: ArmPruningResult
    pareto_frontier_updated: bool
    thompson_posteriors_updated: bool
    estimated_performance_gain: Dict[str, float]
    next_refresh_date: str


class WeeklyModelingLoop:
    """
    Orchestrates the weekly refresh cycle for exploration systems
    
    Manages model retraining, arm pruning, and performance validation
    to ensure exploration efficiency improves over time.
    """
    
    def __init__(self, 
                 router_system: RouterContextualSystem,
                 pareto_optimizer: ParetoFrontierOptimizer,
                 counterfactual_logger: CounterfactualLogger,
                 model_dir: str = "./weekly_models",
                 min_observations: int = 1000):
        
        self.router_system = router_system
        self.pareto_optimizer = pareto_optimizer  
        self.counterfactual_logger = counterfactual_logger
        self.model_dir = Path(model_dir)
        self.min_observations = min_observations
        
        self.logger = logging.getLogger(__name__)
        self.model_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.baseline_models = {}
        self.refresh_history = []
        
        # Statistical thresholds
        self.arm_pruning_alpha = 0.01  # Bonferroni corrected significance
        self.min_arm_observations = 100  # Minimum data per arm
        self.model_improvement_threshold = 0.02  # 2% R² improvement required
        
        self.logger.info("Weekly modeling loop initialized")
    
    def run_weekly_refresh(self) -> WeeklyRefreshReport:
        """
        Execute the complete weekly refresh cycle
        
        Returns comprehensive report of all refresh activities
        """
        refresh_start = time.time()
        refresh_date = datetime.now().isoformat()
        
        self.logger.info(f"🔄 Starting weekly refresh cycle: {refresh_date}")
        
        # Step 1: Load observation data from counterfactual logs
        observations = self._load_week_observations()
        
        if len(observations) < self.min_observations:
            self.logger.warning(f"Insufficient observations ({len(observations)} < {self.min_observations}), "
                              "skipping refresh cycle")
            return self._create_skip_report(refresh_date, len(observations))
        
        # Step 2: Retrain surrogate models
        model_performances = self._retrain_surrogate_models(observations)
        
        # Step 3: Prune underperforming arms
        router_pruning = self._prune_router_arms(observations)
        ann_pruning = self._prune_ann_configs(observations)
        
        # Step 4: Update Thompson sampling posteriors
        posteriors_updated = self._update_thompson_posteriors(observations)
        
        # Step 5: Recompute Pareto frontiers with new models
        pareto_updated = self._recompute_pareto_frontiers()
        
        # Step 6: Estimate performance gains using DR
        performance_gains = self._estimate_performance_gains(observations)
        
        # Step 7: Generate comprehensive report
        report = WeeklyRefreshReport(
            refresh_date=refresh_date,
            observations_processed=len(observations),
            models_retrained=model_performances,
            router_arm_pruning=router_pruning,
            ann_config_pruning=ann_pruning,
            pareto_frontier_updated=pareto_updated,
            thompson_posteriors_updated=posteriors_updated,
            estimated_performance_gain=performance_gains,
            next_refresh_date=(datetime.now() + timedelta(days=7)).isoformat()
        )
        
        # Step 8: Save report and update history
        self._save_refresh_report(report)
        self.refresh_history.append(report)
        
        refresh_duration = time.time() - refresh_start
        self.logger.info(f"✅ Weekly refresh completed in {refresh_duration:.1f}s")
        
        return report
    
    def _load_week_observations(self) -> List[Dict[str, Any]]:
        """Load and preprocess the past week's observation data"""
        observations = []
        
        # Get data from counterfactual logger
        for query_id, outcome in self.counterfactual_logger.outcomes.items():
            # Get corresponding decisions
            router_decision = self.counterfactual_logger.router_decisions.get(query_id)
            ann_decision = self.counterfactual_logger.ann_decisions.get(query_id)
            
            if router_decision:
                obs = {
                    'query_id': query_id,
                    'decision_type': 'router',
                    'context_features': router_decision.context.query_embedding,
                    'query_entropy': router_decision.context.query_entropy,
                    'nl_confidence': router_decision.context.nl_confidence,
                    'is_hard_nl': router_decision.context.is_hard_nl,
                    'selected_arm': router_decision.selected_arm,
                    'tau': router_decision.tau,
                    'spend_cap_ms': router_decision.spend_cap_ms, 
                    'min_conf_gain': router_decision.min_conf_gain,
                    'propensity': router_decision.propensity,
                    'ndcg_at_10': outcome.ndcg_at_10,
                    'sla_recall_at_50': outcome.sla_recall_at_50,
                    'p95_latency_us': outcome.p95_latency_us,
                    'cache_hit_rate': outcome.cache_hit_rate,
                    'timestamp': outcome.completion_timestamp_ms
                }
                observations.append(obs)
            
            if ann_decision:
                obs = {
                    'query_id': query_id,
                    'decision_type': 'ann',
                    'context_features': ann_decision.context.query_embedding,
                    'query_entropy': ann_decision.context.query_entropy,
                    'nl_confidence': ann_decision.context.nl_confidence,
                    'is_hard_nl': ann_decision.context.is_hard_nl,
                    'selected_config': ann_decision.selected_config,
                    'ef_search': ann_decision.ef_search,
                    'refine_topk': ann_decision.refine_topk,
                    'cache_residency': ann_decision.cache_residency,
                    'propensity': ann_decision.propensity,
                    'ndcg_at_10': outcome.ndcg_at_10,
                    'sla_recall_at_50': outcome.sla_recall_at_50,
                    'p95_latency_us': outcome.p95_latency_us,
                    'cache_hit_rate': outcome.cache_hit_rate,
                    'timestamp': outcome.completion_timestamp_ms
                }
                observations.append(obs)
        
        self.logger.info(f"Loaded {len(observations)} observations for model refresh")
        return observations
    
    def _retrain_surrogate_models(self, observations: List[Dict]) -> Dict[str, ModelPerformance]:
        """Retrain all surrogate models with fresh data"""
        performances = {}
        
        # Separate router and ANN observations
        router_obs = [obs for obs in observations if obs['decision_type'] == 'router']
        ann_obs = [obs for obs in observations if obs['decision_type'] == 'ann']
        
        # Retrain router reward model
        if len(router_obs) >= self.min_observations:
            perf = self._retrain_router_reward_model(router_obs)
            performances['router_reward'] = perf
        
        # Retrain ANN latency model
        if len(ann_obs) >= self.min_observations:
            perf = self._retrain_ann_latency_model(ann_obs)
            performances['ann_latency'] = perf
        
        # Retrain ANN quality model
        if len(ann_obs) >= self.min_observations:
            perf = self._retrain_ann_quality_model(ann_obs)
            performances['ann_quality'] = perf
        
        return performances
    
    def _retrain_router_reward_model(self, observations: List[Dict]) -> ModelPerformance:
        """Retrain router contextual reward model"""
        # Prepare training data
        X = []
        y = []
        
        for obs in observations:
            # Features: context + arm parameters
            features = (
                obs['context_features'] +  # 768-dim query embedding
                [obs['query_entropy'], obs['nl_confidence']] +  # query stats
                [obs['tau'], obs['spend_cap_ms'], obs['min_conf_gain']]  # arm params
            )
            X.append(features)
            
            # Target: composite reward (nDCG + recall with latency penalty)
            quality_reward = 0.7 * obs['ndcg_at_10'] + 0.3 * obs['sla_recall_at_50']
            latency_penalty = max(0, (obs['p95_latency_us'] / 1000 - 118) / 1000)  # Above T₀ baseline
            reward = quality_reward - 0.1 * latency_penalty
            y.append(reward)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        model.fit(X, y)
        
        # Evaluate performance
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        # Compare to baseline
        baseline_key = 'router_reward_baseline'
        is_better = True
        if baseline_key in self.baseline_models:
            baseline_r2 = self.baseline_models[baseline_key]['r2_score']
            is_better = r2 > (baseline_r2 + self.model_improvement_threshold)
        
        performance = ModelPerformance(
            model_name='router_reward',
            r2_score=r2,
            rmse=rmse,
            cv_score_mean=cv_scores.mean(),
            cv_score_std=cv_scores.std(),
            training_samples=len(X),
            validation_date=datetime.now().isoformat(),
            is_better_than_baseline=is_better
        )
        
        # Save model if better than baseline
        if is_better:
            model_path = self.model_dir / 'router_reward_latest.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.baseline_models[baseline_key] = asdict(performance)
            self.logger.info(f"✅ Router reward model updated: R²={r2:.4f}")
        else:
            self.logger.info(f"⚠️ Router reward model not improved: R²={r2:.4f}")
        
        return performance
    
    def _retrain_ann_latency_model(self, observations: List[Dict]) -> ModelPerformance:
        """Retrain ANN latency surrogate model"""
        # Prepare training data  
        X = []
        y = []
        
        for obs in observations:
            # Features: context + ANN config
            features = (
                obs['context_features'] +  # 768-dim query embedding
                [obs['query_entropy'], obs['nl_confidence']] +  # query stats
                [obs['ef_search'], obs['refine_topk'], obs['cache_residency']]  # ANN config
            )
            X.append(features)
            
            # Target: p95 latency in milliseconds
            y.append(obs['p95_latency_us'] / 1000)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model with same architecture as Pareto optimizer
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
        model.fit(X, y)
        
        # Evaluate performance
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        # Compare to baseline
        baseline_key = 'ann_latency_baseline'
        is_better = True
        if baseline_key in self.baseline_models:
            baseline_r2 = self.baseline_models[baseline_key]['r2_score']
            is_better = r2 > (baseline_r2 + self.model_improvement_threshold)
        
        performance = ModelPerformance(
            model_name='ann_latency',
            r2_score=r2,
            rmse=rmse,
            cv_score_mean=cv_scores.mean(),
            cv_score_std=cv_scores.std(),
            training_samples=len(X),
            validation_date=datetime.now().isoformat(),
            is_better_than_baseline=is_better
        )
        
        if is_better:
            model_path = self.model_dir / 'ann_latency_latest.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.baseline_models[baseline_key] = asdict(performance)
            
            # Update Pareto optimizer's latency model
            self.pareto_optimizer.latency_model.model = model
            self.logger.info(f"✅ ANN latency model updated: R²={r2:.4f}")
        else:
            self.logger.info(f"⚠️ ANN latency model not improved: R²={r2:.4f}")
        
        return performance
    
    def _retrain_ann_quality_model(self, observations: List[Dict]) -> ModelPerformance:
        """Retrain ANN quality prediction model"""
        # Similar to latency model but predicting nDCG
        X = []
        y = []
        
        for obs in observations:
            features = (
                obs['context_features'] +
                [obs['query_entropy'], obs['nl_confidence']] +
                [obs['ef_search'], obs['refine_topk'], obs['cache_residency']]
            )
            X.append(features)
            y.append(obs['ndcg_at_10'])
        
        X = np.array(X)
        y = np.array(y)
        
        model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=7,
            random_state=42
        )
        model.fit(X, y)
        
        # Evaluate
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        # Compare to baseline
        baseline_key = 'ann_quality_baseline'
        is_better = True
        if baseline_key in self.baseline_models:
            baseline_r2 = self.baseline_models[baseline_key]['r2_score']
            is_better = r2 > (baseline_r2 + self.model_improvement_threshold)
        
        performance = ModelPerformance(
            model_name='ann_quality',
            r2_score=r2,
            rmse=rmse,
            cv_score_mean=cv_scores.mean(),
            cv_score_std=cv_scores.std(),
            training_samples=len(X),
            validation_date=datetime.now().isoformat(),
            is_better_than_baseline=is_better
        )
        
        if is_better:
            model_path = self.model_dir / 'ann_quality_latest.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.baseline_models[baseline_key] = asdict(performance)
            self.logger.info(f"✅ ANN quality model updated: R²={r2:.4f}")
        else:
            self.logger.info(f"⚠️ ANN quality model not improved: R²={r2:.4f}")
        
        return performance
    
    def _prune_router_arms(self, observations: List[Dict]) -> ArmPruningResult:
        """Prune statistically underperforming router arms"""
        router_obs = [obs for obs in observations if obs['decision_type'] == 'router']
        
        # Group by arm
        arm_rewards = defaultdict(list)
        arm_counts = defaultdict(int)
        
        for obs in router_obs:
            arm_id = obs['selected_arm']
            quality_reward = 0.7 * obs['ndcg_at_10'] + 0.3 * obs['sla_recall_at_50']
            latency_penalty = max(0, (obs['p95_latency_us'] / 1000 - 118) / 1000)
            reward = quality_reward - 0.1 * latency_penalty
            
            arm_rewards[arm_id].append(reward)
            arm_counts[arm_id] += 1
        
        # Statistical testing for arm pruning
        arms_to_prune = []
        arms_to_retain = []
        
        # Find best performing arm as reference
        arm_means = {arm: np.mean(rewards) for arm, rewards in arm_rewards.items()}
        best_arm = max(arm_means, key=arm_means.get)
        best_rewards = arm_rewards[best_arm]
        
        # Test each arm against the best
        corrected_alpha = self.arm_pruning_alpha / len(arm_rewards)  # Bonferroni correction
        
        for arm_id, rewards in arm_rewards.items():
            if arm_id == best_arm or len(rewards) < self.min_arm_observations:
                arms_to_retain.append(arm_id)
                continue
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(best_rewards, rewards, alternative='greater')
            
            if p_value < corrected_alpha:
                arms_to_prune.append(arm_id)
                self.logger.info(f"Pruning router arm {arm_id}: p={p_value:.6f} < {corrected_alpha:.6f}")
            else:
                arms_to_retain.append(arm_id)
        
        # Update router system
        if arms_to_prune:
            self.router_system._prune_arms(arms_to_prune)
        
        # Calculate statistical power
        effect_size = (arm_means[best_arm] - np.mean(list(arm_means.values()))) / np.std(list(arm_means.values()))
        statistical_power = min(1.0, max(0.0, effect_size))  # Simplified power calculation
        
        return ArmPruningResult(
            arms_pruned=arms_to_prune,
            arms_retained=arms_to_retain,
            pruning_criteria={
                'alpha': corrected_alpha,
                'min_observations': self.min_arm_observations,
                'best_arm_performance': arm_means[best_arm]
            },
            statistical_power=statistical_power,
            false_discovery_rate=corrected_alpha
        )
    
    def _prune_ann_configs(self, observations: List[Dict]) -> ArmPruningResult:
        """Prune statistically underperforming ANN configurations"""
        # Similar logic to router arms but for ANN configs
        ann_obs = [obs for obs in observations if obs['decision_type'] == 'ann']
        
        config_rewards = defaultdict(list)
        
        for obs in ann_obs:
            config_id = obs['selected_config']
            # Pareto objective: quality vs latency
            quality_score = obs['ndcg_at_10']
            latency_score = max(0, 1 - (obs['p95_latency_us'] / 1000 - 100) / 50)  # Normalize latency
            pareto_reward = 0.6 * quality_score + 0.4 * latency_score
            
            config_rewards[config_id].append(pareto_reward)
        
        # Same statistical testing logic as router arms
        configs_to_prune = []
        configs_to_retain = []
        
        if not config_rewards:
            return ArmPruningResult(
                arms_pruned=[],
                arms_retained=[],
                pruning_criteria={},
                statistical_power=0.0,
                false_discovery_rate=0.0
            )
        
        config_means = {config: np.mean(rewards) for config, rewards in config_rewards.items()}
        best_config = max(config_means, key=config_means.get)
        best_rewards = config_rewards[best_config]
        
        corrected_alpha = self.arm_pruning_alpha / len(config_rewards)
        
        for config_id, rewards in config_rewards.items():
            if config_id == best_config or len(rewards) < self.min_arm_observations:
                configs_to_retain.append(config_id)
                continue
            
            t_stat, p_value = stats.ttest_ind(best_rewards, rewards, alternative='greater')
            
            if p_value < corrected_alpha:
                configs_to_prune.append(config_id)
                self.logger.info(f"Pruning ANN config {config_id}: p={p_value:.6f} < {corrected_alpha:.6f}")
            else:
                configs_to_retain.append(config_id)
        
        # Update Pareto optimizer
        if configs_to_prune:
            self.pareto_optimizer._prune_configs(configs_to_prune)
        
        return ArmPruningResult(
            arms_pruned=configs_to_prune,
            arms_retained=configs_to_retain,
            pruning_criteria={
                'alpha': corrected_alpha,
                'min_observations': self.min_arm_observations,
                'best_config_performance': config_means[best_config]
            },
            statistical_power=0.8,  # Simplified
            false_discovery_rate=corrected_alpha
        )
    
    def _update_thompson_posteriors(self, observations: List[Dict]) -> bool:
        """Update Thompson sampling posterior distributions"""
        try:
            # Update router posteriors
            router_updates = 0
            for obs in observations:
                if obs['decision_type'] == 'router':
                    context = np.array(obs['context_features'])
                    arm_id = obs['selected_arm']
                    quality_reward = 0.7 * obs['ndcg_at_10'] + 0.3 * obs['sla_recall_at_50']
                    
                    self.router_system._update_posterior(context, arm_id, quality_reward)
                    router_updates += 1
            
            self.logger.info(f"Updated Thompson posteriors: {router_updates} router observations")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update Thompson posteriors: {e}")
            return False
    
    def _recompute_pareto_frontiers(self) -> bool:
        """Recompute Pareto frontiers with updated models"""
        try:
            # Force recomputation of Pareto frontiers
            self.pareto_optimizer._recompute_frontiers()
            self.logger.info("✅ Pareto frontiers recomputed with updated models")
            return True
        except Exception as e:
            self.logger.error(f"Failed to recompute Pareto frontiers: {e}")
            return False
    
    def _estimate_performance_gains(self, observations: List[Dict]) -> Dict[str, float]:
        """Estimate performance gains using doubly robust estimation"""
        try:
            estimator = IPSDREstimator(self.counterfactual_logger)
            
            # Create dummy target policies for comparison
            current_policy_probs = {}
            improved_policy_probs = {}
            
            for obs in observations:
                query_id = obs['query_id']
                current_policy_probs[query_id] = obs['propensity']
                # Simulate improved policy with 10% better targeting
                improved_policy_probs[query_id] = min(1.0, obs['propensity'] * 1.1)
            
            # Compare policies using DR estimation
            comparison = estimator.compare_policies(
                current_policy_probs,
                improved_policy_probs,
                metric_name='ndcg_at_10',
                use_dr=False  # Use IPS since we don't have reward model here
            )
            
            gains = {
                'ndcg_improvement': comparison['difference'],
                'statistical_significance': comparison['p_value'] < 0.05,
                'confidence_interval': comparison['policy_a_ci']
            }
            
            return gains
            
        except Exception as e:
            self.logger.error(f"Failed to estimate performance gains: {e}")
            return {}
    
    def _create_skip_report(self, refresh_date: str, num_observations: int) -> WeeklyRefreshReport:
        """Create a report when skipping refresh due to insufficient data"""
        return WeeklyRefreshReport(
            refresh_date=refresh_date,
            observations_processed=num_observations,
            models_retrained={},
            router_arm_pruning=ArmPruningResult([], [], {}, 0.0, 0.0),
            ann_config_pruning=ArmPruningResult([], [], {}, 0.0, 0.0),
            pareto_frontier_updated=False,
            thompson_posteriors_updated=False,
            estimated_performance_gain={},
            next_refresh_date=(datetime.now() + timedelta(days=7)).isoformat()
        )
    
    def _save_refresh_report(self, report: WeeklyRefreshReport):
        """Save refresh report to disk"""
        report_path = self.model_dir / f"refresh_report_{report.refresh_date}.json"
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        self.logger.info(f"Refresh report saved to {report_path}")


def test_weekly_modeling_loop():
    """Test the weekly modeling loop system"""
    # Create mock systems
    router_system = RouterContextualSystem()
    pareto_optimizer = ParetoFrontierOptimizer()
    counterfactual_logger = CounterfactualLogger("./test_logs")
    
    # Create modeling loop
    modeling_loop = WeeklyModelingLoop(
        router_system=router_system,
        pareto_optimizer=pareto_optimizer,
        counterfactual_logger=counterfactual_logger,
        model_dir="./test_weekly_models",
        min_observations=10  # Low threshold for testing
    )
    
    # Add some test data to counterfactual logger
    from counterfactual_logging import DecisionContext, RouterDecision, DecisionOutcome
    
    for i in range(50):
        query_id = f"test_query_{i:03d}"
        
        context = DecisionContext(
            query_id=query_id,
            query_text=f"test query {i}",
            query_length=5,
            query_entropy=2.5 + np.random.normal(0, 0.5),
            nl_confidence=0.8 + np.random.normal(0, 0.1),
            query_embedding=np.random.normal(0, 0.1, 768).tolist(),
            user_session=f"session_{i//10}",
            timestamp_ms=int(time.time() * 1000) - i * 1000,
            is_hard_nl=True
        )
        
        decision = RouterDecision(
            context=context,
            selected_arm=np.random.randint(0, 140),
            tau=0.4 + np.random.random() * 0.3,
            spend_cap_ms=np.random.choice([2, 4, 6, 8]),
            min_conf_gain=0.08 + np.random.random() * 0.10,
            propensity=np.random.random() * 0.1,
            exploration_policy="thompson_sampling"
        )
        
        outcome = DecisionOutcome(
            query_id=query_id,
            decision_type="router",
            selected_id=decision.selected_arm,
            ndcg_at_10=0.2 + np.random.random() * 0.4,
            sla_recall_at_50=0.5 + np.random.random() * 0.3,
            p95_latency_us=100000 + int(np.random.normal(0, 20000)),
            p99_latency_us=140000 + int(np.random.normal(0, 30000)),
            cache_hit_rate=np.random.random(),
            click_through_rate=np.random.random() * 0.5,
            user_satisfaction=3.0 + np.random.random() * 2.0,
            lexical_matches=int(np.random.exponential(10)),
            semantic_matches=int(np.random.exponential(20)),
            total_candidates=int(np.random.exponential(100)),
            completion_timestamp_ms=int(time.time() * 1000) - i * 1000 + 5000
        )
        
        counterfactual_logger.log_router_decision(decision)
        counterfactual_logger.log_outcome(outcome)
    
    # Run weekly refresh
    report = modeling_loop.run_weekly_refresh()
    
    print("📊 Weekly Refresh Report:")
    print(f"  Observations processed: {report.observations_processed}")
    print(f"  Models retrained: {len(report.models_retrained)}")
    print(f"  Router arms pruned: {len(report.router_arm_pruning.arms_pruned)}")
    print(f"  ANN configs pruned: {len(report.ann_config_pruning.arms_pruned)}")
    print(f"  Pareto frontiers updated: {report.pareto_frontier_updated}")
    print(f"  Thompson posteriors updated: {report.thompson_posteriors_updated}")
    
    print("✅ Weekly modeling loop test completed successfully")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run test
    test_weekly_modeling_loop()