"""
Parametric Router Policy System

Replaces discrete 140-arm grid with continuous functions τ(x), spend_cap(x), min_gain(x)
based on context features using Bayesian linear regression and Thompson Sampling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContextFeatures:
    """Context features for parametric routing"""
    entropy: float
    query_length: int
    nl_confidence: float
    prior_miss_rate: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([self.entropy, self.query_length, self.nl_confidence, self.prior_miss_rate])

@dataclass
class PolicyOutput:
    """Output of parametric policy"""
    tau: float  # Time allocation parameter
    spend_cap_ms: float  # Maximum spend in milliseconds
    min_conf_gain: float  # Minimum confidence gain threshold
    
    def validate_constraints(self) -> bool:
        """Validate policy output constraints"""
        return (
            0.1 <= self.tau <= 10.0 and
            10.0 <= self.spend_cap_ms <= 1000.0 and
            0.0 <= self.min_conf_gain <= 1.0
        )

@dataclass
class ObservationRecord:
    """Single observation for policy learning"""
    context: ContextFeatures
    action: PolicyOutput
    reward: float
    p95_latency: float
    p99_latency: float
    jaccard_10: float
    ndcg_delta: float
    r50_delta: float
    aece_delta: float
    file_credit_pct: float
    propensity_score: float  # For importance weighting

class DoublyRobustEstimator:
    """Doubly-Robust reward estimator with importance weighting"""
    
    def __init__(self, max_weight_clip: float = 10.0):
        self.max_weight_clip = max_weight_clip
        self.bias_model = BayesianRidge()  # For bias correction
        
    def estimate_value(self, observations: List[ObservationRecord]) -> float:
        """
        Estimate value using doubly-robust estimator
        V̂ = Σ(w_i * r_i) / Σ(w_i) + bias correction
        """
        if not observations:
            return 0.0
            
        weights = []
        rewards = []
        bias_features = []
        
        for obs in observations:
            # Clip importance weights
            weight = min(1.0 / max(obs.propensity_score, 1e-6), self.max_weight_clip)
            weights.append(weight)
            rewards.append(obs.reward)
            bias_features.append(obs.context.to_array())
            
        weights = np.array(weights)
        rewards = np.array(rewards)
        bias_features = np.array(bias_features)
        
        # IPS estimate
        ips_estimate = np.sum(weights * rewards) / np.sum(weights) if np.sum(weights) > 0 else 0.0
        
        # Bias correction (if we have enough data)
        bias_correction = 0.0
        if len(observations) > 10:
            try:
                self.bias_model.fit(bias_features, rewards)
                predicted_rewards = self.bias_model.predict(bias_features)
                bias_correction = np.mean(predicted_rewards - rewards)
            except Exception as e:
                logger.warning(f"Bias correction failed: {e}")
                
        return ips_estimate + bias_correction
    
    def compute_composite_reward(self, obs: ObservationRecord) -> float:
        """
        Compute composite reward: 0.7×ΔnDCG + 0.3×ΔR@50 - 0.1×[p95-(p95_T₀+0.3)]₊
        """
        # Performance gains
        perf_reward = 0.7 * obs.ndcg_delta + 0.3 * obs.r50_delta
        
        # Latency penalty (assuming baseline p95 + 0.3s tolerance)
        baseline_p95_ms = 200.0  # Configurable baseline
        tolerance_ms = 300.0
        latency_penalty = max(0, obs.p95_latency - (baseline_p95_ms + tolerance_ms)) / 1000.0
        
        return perf_reward - 0.1 * latency_penalty

class MonotoneConstraintValidator:
    """Validates monotone constraints for policy outputs"""
    
    def __init__(self):
        self.p95_predictor = BayesianRidge()  # Predict p95 given context + action
        self.trained = False
        
    def train_predictor(self, observations: List[ObservationRecord]):
        """Train p95 predictor from observations"""
        if len(observations) < 10:
            return
            
        features = []
        targets = []
        
        for obs in observations:
            # Combine context and action features
            context_features = obs.context.to_array()
            action_features = np.array([obs.action.tau, obs.action.spend_cap_ms, obs.action.min_conf_gain])
            combined_features = np.concatenate([context_features, action_features])
            
            features.append(combined_features)
            targets.append(obs.p95_latency)
            
        features = np.array(features)
        targets = np.array(targets)
        
        self.p95_predictor.fit(features, targets)
        self.trained = True
        
    def validate_monotone_constraint(self, context: ContextFeatures, 
                                   action1: PolicyOutput, action2: PolicyOutput) -> bool:
        """
        Validate: ↑τ or ↓spend can't worsen predicted p95
        """
        if not self.trained:
            return True  # Skip validation if not trained
            
        try:
            # Predict p95 for both actions
            context_arr = context.to_array()
            
            features1 = np.concatenate([context_arr, 
                                      [action1.tau, action1.spend_cap_ms, action1.min_conf_gain]])
            features2 = np.concatenate([context_arr, 
                                      [action2.tau, action2.spend_cap_ms, action2.min_conf_gain]])
            
            p95_1 = self.p95_predictor.predict([features1])[0]
            p95_2 = self.p95_predictor.predict([features2])[0]
            
            # Check monotone constraints
            if action2.tau > action1.tau or action2.spend_cap_ms < action1.spend_cap_ms:
                return p95_2 <= p95_1 * 1.05  # Allow 5% tolerance
                
            return True
            
        except Exception as e:
            logger.warning(f"Monotone validation failed: {e}")
            return True

class ParametricPolicyModel(ABC):
    """Abstract base class for parametric policy models"""
    
    @abstractmethod
    def fit(self, contexts: List[ContextFeatures], actions: List[PolicyOutput], 
            rewards: List[float]) -> None:
        """Fit the model to observations"""
        pass
    
    @abstractmethod
    def sample_action(self, context: ContextFeatures) -> PolicyOutput:
        """Sample action using Thompson Sampling"""
        pass
    
    @abstractmethod
    def predict_action(self, context: ContextFeatures) -> PolicyOutput:
        """Predict best action (no sampling)"""
        pass

class BayesianLinearPolicy(ParametricPolicyModel):
    """Bayesian linear regression for parametric policy"""
    
    def __init__(self):
        self.tau_model = BayesianRidge()
        self.spend_model = BayesianRidge()
        self.gain_model = BayesianRidge()
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, contexts: List[ContextFeatures], actions: List[PolicyOutput], 
            rewards: List[float]) -> None:
        """Fit Bayesian linear models"""
        if len(contexts) < 5:
            return
            
        # Prepare features
        X = np.array([ctx.to_array() for ctx in contexts])
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit separate models for each action component
        tau_targets = [action.tau for action in actions]
        spend_targets = [action.spend_cap_ms for action in actions]
        gain_targets = [action.min_conf_gain for action in actions]
        
        # Weight by rewards (higher reward = more influence)
        sample_weights = np.array(rewards) - np.min(rewards) + 1e-3
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
        
        self.tau_model.fit(X_scaled, tau_targets, sample_weight=sample_weights)
        self.spend_model.fit(X_scaled, spend_targets, sample_weight=sample_weights)
        self.gain_model.fit(X_scaled, gain_targets, sample_weight=sample_weights)
        
        self.fitted = True
        
    def sample_action(self, context: ContextFeatures) -> PolicyOutput:
        """Sample action using Thompson Sampling on parameters"""
        if not self.fitted:
            # Default policy for cold start
            return PolicyOutput(
                tau=2.0 + np.random.normal(0, 0.5),
                spend_cap_ms=200.0 + np.random.normal(0, 50),
                min_conf_gain=0.1 + np.random.beta(2, 8)
            )
            
        X = self.scaler.transform([context.to_array()])
        
        # Sample from posterior predictive distribution
        tau_sample = self.tau_model.predict(X, return_std=True)
        tau = np.random.normal(tau_sample[0][0], tau_sample[1][0])
        
        spend_sample = self.spend_model.predict(X, return_std=True)
        spend = np.random.normal(spend_sample[0][0], spend_sample[1][0])
        
        gain_sample = self.gain_model.predict(X, return_std=True)
        gain = np.random.normal(gain_sample[0][0], gain_sample[1][0])
        
        # Apply constraints
        action = PolicyOutput(
            tau=np.clip(tau, 0.1, 10.0),
            spend_cap_ms=np.clip(spend, 10.0, 1000.0),
            min_conf_gain=np.clip(gain, 0.0, 1.0)
        )
        
        return action
    
    def predict_action(self, context: ContextFeatures) -> PolicyOutput:
        """Predict best action (no sampling)"""
        if not self.fitted:
            return PolicyOutput(tau=2.0, spend_cap_ms=200.0, min_conf_gain=0.1)
            
        X = self.scaler.transform([context.to_array()])
        
        tau = self.tau_model.predict(X)[0]
        spend = self.spend_model.predict(X)[0]
        gain = self.gain_model.predict(X)[0]
        
        return PolicyOutput(
            tau=np.clip(tau, 0.1, 10.0),
            spend_cap_ms=np.clip(spend, 10.0, 1000.0),
            min_conf_gain=np.clip(gain, 0.0, 1.0)
        )

class MonotoneGBMPolicy(ParametricPolicyModel):
    """Monotone Gradient Boosting for parametric policy"""
    
    def __init__(self):
        # Monotone constraints: [entropy↑, length?, nl_conf↑, miss_rate↓]
        self.tau_model = HistGradientBoostingRegressor(
            monotonic_cst=[1, 0, 1, -1],  # tau should increase with entropy, nl_conf
            random_state=42
        )
        self.spend_model = HistGradientBoostingRegressor(
            monotonic_cst=[1, 1, 1, 1],  # spend should increase with complexity
            random_state=42
        )
        self.gain_model = HistGradientBoostingRegressor(
            monotonic_cst=[1, 0, -1, 1],  # gain threshold higher for low confidence
            random_state=42
        )
        self.fitted = False
        
    def fit(self, contexts: List[ContextFeatures], actions: List[PolicyOutput], 
            rewards: List[float]) -> None:
        """Fit gradient boosting models"""
        if len(contexts) < 20:  # Need more data for GBM
            return
            
        X = np.array([ctx.to_array() for ctx in contexts])
        
        tau_targets = [action.tau for action in actions]
        spend_targets = [action.spend_cap_ms for action in actions]
        gain_targets = [action.min_conf_gain for action in actions]
        
        # Weight by rewards
        sample_weights = np.array(rewards) - np.min(rewards) + 1e-3
        
        self.tau_model.fit(X, tau_targets, sample_weight=sample_weights)
        self.spend_model.fit(X, spend_targets, sample_weight=sample_weights)
        self.gain_model.fit(X, gain_targets, sample_weight=sample_weights)
        
        self.fitted = True
        
    def sample_action(self, context: ContextFeatures) -> PolicyOutput:
        """Sample with noise for exploration"""
        base_action = self.predict_action(context)
        
        # Add exploration noise
        tau_noise = np.random.normal(0, 0.2)
        spend_noise = np.random.normal(0, 20)
        gain_noise = np.random.normal(0, 0.05)
        
        return PolicyOutput(
            tau=np.clip(base_action.tau + tau_noise, 0.1, 10.0),
            spend_cap_ms=np.clip(base_action.spend_cap_ms + spend_noise, 10.0, 1000.0),
            min_conf_gain=np.clip(base_action.min_conf_gain + gain_noise, 0.0, 1.0)
        )
    
    def predict_action(self, context: ContextFeatures) -> PolicyOutput:
        """Predict best action"""
        if not self.fitted:
            return PolicyOutput(tau=2.0, spend_cap_ms=200.0, min_conf_gain=0.1)
            
        X = [context.to_array()]
        
        tau = self.tau_model.predict(X)[0]
        spend = self.spend_model.predict(X)[0]
        gain = self.gain_model.predict(X)[0]
        
        return PolicyOutput(
            tau=np.clip(tau, 0.1, 10.0),
            spend_cap_ms=np.clip(spend, 10.0, 1000.0),
            min_conf_gain=np.clip(gain, 0.0, 1.0)
        )

class GuardSystem:
    """Guard systems for quality and safety constraints"""
    
    def __init__(self):
        self.jaccard_threshold = 0.80
        self.p99_p95_ratio_threshold = 2.0
        self.aece_threshold = 0.01
        self.file_credit_threshold = 0.05
        
    def validate_observation(self, obs: ObservationRecord) -> Dict[str, bool]:
        """Validate all guard constraints"""
        guards = {
            'jaccard_10': obs.jaccard_10 >= self.jaccard_threshold,
            'p99_p95_ratio': (obs.p99_latency / max(obs.p95_latency, 1)) <= self.p99_p95_ratio_threshold,
            'aece_delta': obs.aece_delta <= self.aece_threshold,
            'file_credit': obs.file_credit_pct <= self.file_credit_threshold
        }
        
        return guards
    
    def is_safe_observation(self, obs: ObservationRecord) -> bool:
        """Check if observation passes all guards"""
        guards = self.validate_observation(obs)
        return all(guards.values())

class ParametricRouterPolicy:
    """Main parametric router policy system"""
    
    def __init__(self, model_type: str = "bayesian"):
        self.model_type = model_type
        
        if model_type == "bayesian":
            self.policy_model = BayesianLinearPolicy()
        elif model_type == "gbm":
            self.policy_model = MonotoneGBMPolicy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.dr_estimator = DoublyRobustEstimator()
        self.constraint_validator = MonotoneConstraintValidator()
        self.guard_system = GuardSystem()
        
        self.observations: List[ObservationRecord] = []
        self.cold_query_count = 0  # Track cold start
        
    def add_observation(self, obs: ObservationRecord) -> None:
        """Add new observation and update models"""
        # Compute composite reward
        obs.reward = self.dr_estimator.compute_composite_reward(obs)
        
        # Store observation
        self.observations.append(obs)
        
        # Retrain models periodically
        if len(self.observations) % 10 == 0:
            self._retrain_models()
            
    def _retrain_models(self):
        """Retrain all models with current observations"""
        # Filter safe observations for training
        safe_observations = [obs for obs in self.observations 
                           if self.guard_system.is_safe_observation(obs)]
        
        if len(safe_observations) < 5:
            logger.warning("Not enough safe observations for retraining")
            return
            
        contexts = [obs.context for obs in safe_observations]
        actions = [obs.action for obs in safe_observations]
        rewards = [obs.reward for obs in safe_observations]
        
        # Retrain policy model
        self.policy_model.fit(contexts, actions, rewards)
        
        # Retrain constraint validator
        self.constraint_validator.train_predictor(safe_observations)
        
        logger.info(f"Retrained models with {len(safe_observations)} safe observations")
        
    def get_action(self, context: ContextFeatures, explore: bool = True) -> PolicyOutput:
        """Get routing action for given context"""
        self.cold_query_count += 1
        
        # Use conservative policy for first few queries
        if self.cold_query_count <= 3:
            action = PolicyOutput(
                tau=1.0 + 0.5 * context.entropy,  # Conservative tau
                spend_cap_ms=150.0 + 50 * context.prior_miss_rate,
                min_conf_gain=0.05 + 0.1 * (1 - context.nl_confidence)
            )
        else:
            if explore:
                action = self.policy_model.sample_action(context)
            else:
                action = self.policy_model.predict_action(context)
        
        # Validate constraints
        if not action.validate_constraints():
            logger.warning("Action violates constraints, using conservative fallback")
            action = PolicyOutput(tau=1.0, spend_cap_ms=200.0, min_conf_gain=0.1)
            
        return action
    
    def evaluate_policy(self) -> Dict[str, float]:
        """Evaluate current policy performance"""
        if not self.observations:
            return {}
            
        # Compute doubly-robust value estimate
        policy_value = self.dr_estimator.estimate_value(self.observations)
        
        # Guard compliance rates
        guard_results = [self.guard_system.validate_observation(obs) 
                        for obs in self.observations]
        
        guard_rates = {}
        if guard_results:
            for guard_name in ['jaccard_10', 'p99_p95_ratio', 'aece_delta', 'file_credit']:
                guard_rates[f'{guard_name}_rate'] = np.mean([gr[guard_name] for gr in guard_results])
        
        # Performance metrics
        recent_obs = self.observations[-50:] if len(self.observations) >= 50 else self.observations
        
        metrics = {
            'policy_value': policy_value,
            'num_observations': len(self.observations),
            'safe_observation_rate': np.mean([self.guard_system.is_safe_observation(obs) 
                                            for obs in recent_obs]),
            'avg_p95_latency': np.mean([obs.p95_latency for obs in recent_obs]),
            'avg_ndcg_delta': np.mean([obs.ndcg_delta for obs in recent_obs]),
            'avg_r50_delta': np.mean([obs.r50_delta for obs in recent_obs]),
            **guard_rates
        }
        
        return metrics
    
    def export_policy_state(self) -> Dict[str, Any]:
        """Export policy state for analysis"""
        return {
            'model_type': self.model_type,
            'num_observations': len(self.observations),
            'cold_query_count': self.cold_query_count,
            'policy_fitted': self.policy_model.fitted if hasattr(self.policy_model, 'fitted') else False,
            'evaluation_metrics': self.evaluate_policy()
        }

# Integration with flight simulator
class ParametricFlightSimulator:
    """Flight simulator integration for parametric policy"""
    
    def __init__(self, policy: ParametricRouterPolicy):
        self.policy = policy
        self.query_count = 0
        
    def route_query(self, query_text: str, ground_truth: Optional[Any] = None) -> Dict[str, Any]:
        """Route query using parametric policy"""
        self.query_count += 1
        
        # Extract context features (placeholder implementation)
        context = self._extract_context(query_text)
        
        # Get routing action
        action = self.policy.get_action(context, explore=True)
        
        # Simulate query execution (placeholder)
        results = self._simulate_execution(query_text, action, ground_truth)
        
        # Create observation record
        obs = ObservationRecord(
            context=context,
            action=action,
            reward=0.0,  # Will be computed by policy
            p95_latency=results['p95_latency'],
            p99_latency=results['p99_latency'],
            jaccard_10=results['jaccard_10'],
            ndcg_delta=results['ndcg_delta'],
            r50_delta=results['r50_delta'],
            aece_delta=results['aece_delta'],
            file_credit_pct=results['file_credit_pct'],
            propensity_score=0.1  # Placeholder
        )
        
        # Add observation to policy
        self.policy.add_observation(obs)
        
        return {
            'query_id': self.query_count,
            'action': action,
            'results': results,
            'context': context
        }
    
    def _extract_context(self, query_text: str) -> ContextFeatures:
        """Extract context features from query (placeholder)"""
        import math
        
        # Placeholder feature extraction
        entropy = len(set(query_text.lower())) / len(query_text) if query_text else 0.0
        query_length = len(query_text.split())
        nl_confidence = 0.8 if any(word in query_text.lower() 
                                 for word in ['find', 'show', 'list', 'get']) else 0.5
        prior_miss_rate = 0.1  # Would be looked up from history
        
        return ContextFeatures(
            entropy=entropy,
            query_length=query_length,
            nl_confidence=nl_confidence,
            prior_miss_rate=prior_miss_rate
        )
    
    def _simulate_execution(self, query_text: str, action: PolicyOutput, 
                          ground_truth: Optional[Any]) -> Dict[str, float]:
        """Simulate query execution (placeholder)"""
        # Placeholder simulation
        base_latency = 100 + action.spend_cap_ms * 0.8
        noise = np.random.normal(0, 20)
        
        return {
            'p95_latency': base_latency + noise,
            'p99_latency': (base_latency + noise) * 1.3,
            'jaccard_10': 0.85 + np.random.normal(0, 0.05),
            'ndcg_delta': 0.02 + np.random.normal(0, 0.01),
            'r50_delta': 0.03 + np.random.normal(0, 0.01),
            'aece_delta': 0.005 + np.random.normal(0, 0.002),
            'file_credit_pct': 0.02 + np.random.normal(0, 0.01)
        }

def main():
    """Example usage of parametric router policy"""
    # Initialize policy
    policy = ParametricRouterPolicy(model_type="bayesian")
    
    # Initialize flight simulator
    simulator = ParametricFlightSimulator(policy)
    
    # Run some queries
    test_queries = [
        "find function definitions",
        "show class implementations", 
        "list all imports",
        "get error handling code",
        "search for database connections"
    ]
    
    print("Running parametric router policy simulation...")
    
    for i, query in enumerate(test_queries * 4):  # Run multiple times
        result = simulator.route_query(query)
        
        if (i + 1) % 5 == 0:
            metrics = policy.evaluate_policy()
            print(f"\nAfter {i + 1} queries:")
            print(f"  Policy Value: {metrics.get('policy_value', 0):.4f}")
            print(f"  Safe Rate: {metrics.get('safe_observation_rate', 0):.2f}")
            print(f"  Avg P95: {metrics.get('avg_p95_latency', 0):.1f}ms")
            print(f"  Avg nDCG Δ: {metrics.get('avg_ndcg_delta', 0):.4f}")
    
    # Final evaluation
    final_metrics = policy.evaluate_policy()
    print(f"\n=== Final Policy Evaluation ===")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Export policy state
    state = policy.export_policy_state()
    print(f"\nPolicy State: {state}")

if __name__ == "__main__":
    main()