# 🎯 Protected Innovation Loop - 6-Week Exploration Epoch Specification

**Epoch Period**: September 12 - October 23, 2025 (6 weeks)  
**Baseline Contract**: T₀-2025-09-12T04:47:39Z (frozen CI widths)  
**Error Budget Strategy**: Controlled burn with protected recovery  

---

## 🔒 Frozen Baseline Contracts

### Mathematical CI Contracts (Non-negotiable)
```python
# Frozen T₀ baseline parameters - DO NOT MODIFY during exploration
T0_CONTRACTS = {
    'ndcg_at_10': {'value': 0.345, 'ci_half_width': 0.008, 'frozen': True},
    'p95_latency': {'value': 118, 'ci_half_width': 3.2, 'frozen': True},
    'aece_score': {'value': 0.014, 'ci_half_width': 0.003, 'frozen': True},
    'sla_recall_50': {'value': 0.672, 'ci_half_width': 0.012, 'frozen': True},
    'file_credit_pct': {'value': 2.8, 'ci_half_width': 0.4, 'frozen': True}
}

def validate_ci_contract_compliance(current_metrics):
    """Ensure exploration doesn't violate frozen baseline CI bounds"""
    violations = []
    
    for metric, contract in T0_CONTRACTS.items():
        current_value = current_metrics[metric]
        baseline_value = contract['value']
        ci_bound = contract['ci_half_width']
        
        # Lower bound check (quality metrics must not degrade beyond CI)
        if metric in ['ndcg_at_10', 'sla_recall_50']:
            min_allowed = baseline_value - ci_bound
            if current_value < min_allowed:
                violations.append(f"{metric}: {current_value} < {min_allowed} (CI violation)")
        
        # Upper bound check (latency/cost metrics must not increase beyond CI + error budget)
        elif metric in ['p95_latency']:
            max_allowed = baseline_value + ci_bound + 1.0  # +1ms error budget
            if current_value > max_allowed:
                violations.append(f"{metric}: {current_value} > {max_allowed} (budget violation)")
                
    return violations
```

---

## 🎲 Exploration Track Specifications

### Track 1: Router Contextual Bandit

#### Arm Space Definition
```python
ROUTER_ARM_SPACE = {
    'tau_threshold': np.linspace(0.4, 0.7, 7),           # [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    'spend_cap_ms': [2, 4, 6, 8],                         # Discrete spend budgets
    'min_conf_gain': [0.08, 0.10, 0.12, 0.15, 0.18],    # Confidence gain thresholds
    'context_features': [
        'query_entropy',      # Calculated from token distribution
        'query_length',       # Number of tokens
        'nl_confidence',      # NL vs lexical classifier score
        'ann_miss_rate'       # Historical ANN failure rate for similar queries
    ]
}

# Total arm combinations: 7 × 4 × 5 = 140 arms
# Thompson sampling reduces to ~8-12 active arms
```

#### Contextual Policy Learning
```python
def contextual_router_policy(query_features, bandit_state):
    """
    Learn optimal router parameters as function of query context
    
    Target: +0.5-1.0pp on hard-NL nDCG with Δp95 ≤ +0.3ms
    """
    # Extract cheap context features
    entropy = calculate_query_entropy(query_features.tokens)
    length = len(query_features.tokens)
    nl_conf = query_features.nl_classifier_score
    ann_miss = query_features.historical_ann_miss_rate
    
    context_vector = np.array([entropy, length, nl_conf, ann_miss])
    
    # Thompson sampling with contextual linear bandits
    arm_rewards = []
    for arm in bandit_state.active_arms:
        # Posterior sampling for reward prediction
        theta_sample = bandit_state.sample_posterior(arm.arm_id)
        predicted_reward = np.dot(context_vector, theta_sample)
        
        # Apply monotone constraints (higher tau = higher latency)
        latency_penalty = arm.tau_threshold * 2.0  # ms penalty per tau unit
        constrained_reward = predicted_reward - latency_penalty
        
        arm_rewards.append({
            'arm': arm,
            'expected_reward': constrained_reward,
            'tau': arm.tau_threshold,
            'spend_cap': arm.spend_cap_ms,
            'min_gain': arm.min_conf_gain
        })
    
    # Select arm with highest expected reward under constraints
    selected_arm = max(arm_rewards, key=lambda x: x['expected_reward'])
    
    return {
        'tau': selected_arm['tau'],
        'spend_cap_ms': selected_arm['spend_cap'],
        'min_conf_gain': selected_arm['min_gain'],
        'context': context_vector,
        'propensity': bandit_state.get_selection_probability(selected_arm['arm'])
    }
```

#### Hard-NL Target Definition
```python
def identify_hard_nl_queries(query_batch):
    """
    Identify high-value target queries for router optimization
    
    Hard-NL criteria:
    - NL confidence > 0.8
    - Query entropy > 2.5 (complex semantic structure)
    - Length > 6 tokens (substantial context)
    - Historical ANN success rate < 0.7 (challenging for current system)
    """
    hard_nl_candidates = []
    
    for query in query_batch:
        if (query.nl_confidence > 0.8 and 
            query.entropy > 2.5 and 
            len(query.tokens) > 6 and
            query.ann_historical_success < 0.7):
            
            hard_nl_candidates.append(query)
    
    return hard_nl_candidates
```

### Track 2: ANN Pareto Frontier Search

#### Multi-dimensional Parameter Space
```python
ANN_ARM_SPACE = {
    'efSearch': [64, 80, 96, 112, 128],                    # HNSW exploration factor
    'refine_topk': [20, 32, 40, 48, 56],                   # PQ refinement candidates
    'cache_residency_policy': [                            # Cache management strategy
        {'type': 'LFU', 'window': '1h', 'target_hit_rate': 0.85},
        {'type': 'LFU', 'window': '6h', 'target_hit_rate': 0.90},
        {'type': 'LRU', 'window': '2h', 'target_hit_rate': 0.85},
        {'type': 'FIFO', 'window': '4h', 'target_hit_rate': 0.80}
    ],
    'prefetch_neighbors': [True, False],                    # I/O optimization
    'visited_set_reuse': [True, False]                      # Memory optimization
}

# Total combinations: 5 × 5 × 4 × 2 × 2 = 200 arms
# Surrogate model + Pareto dominance reduces to ~6-10 arms
```

#### Latency Surrogate Model
```python
class LatencySurrogateModel:
    """
    Predict ANN latency from configuration parameters
    
    Goal: Enable efficient Pareto frontier search without exhaustive testing
    """
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.feature_names = [
            'efSearch', 'refine_topk', 'cache_hit_rate', 
            'index_size_gb', 'query_vector_dim', 'prefetch_enabled'
        ]
        
    def predict_latency(self, config, query_context):
        """Predict p95 latency for given ANN configuration"""
        features = np.array([
            config.efSearch,
            config.refine_topk, 
            config.cache_policy.expected_hit_rate,
            query_context.index_size_gb,
            query_context.vector_dimension,
            1.0 if config.prefetch_neighbors else 0.0
        ]).reshape(1, -1)
        
        predicted_latency = self.model.predict(features)[0]
        
        # Add uncertainty estimate
        # (In practice, use ensemble or Bayesian approach)
        uncertainty = 0.1 * predicted_latency  # 10% uncertainty
        
        return {
            'predicted_p95_latency': predicted_latency,
            'uncertainty': uncertainty,
            'confidence_interval': (predicted_latency - uncertainty, 
                                   predicted_latency + uncertainty)
        }
        
    def update_model(self, new_observations):
        """Retrain surrogate with fresh latency observations"""
        X = np.array([obs.features for obs in new_observations])
        y = np.array([obs.observed_latency for obs in new_observations])
        
        # Incremental update
        self.model.fit(X, y)
        
        # Validate model accuracy
        validation_mae = mean_absolute_error(y, self.model.predict(X))
        return {'validation_mae': validation_mae, 'sample_count': len(new_observations)}
```

#### Pareto Dominance with Cache Awareness
```python
def pareto_frontier_search(arms, surrogate_model, cache_warmup_period=3600):
    """
    Find non-dominated ANN configurations with cache-aware validation
    
    Target: ~1ms p95 reduction with ΔnDCG@10 ≥ 0
    """
    evaluated_arms = []
    
    for arm in arms:
        # Predict performance using surrogate
        latency_pred = surrogate_model.predict_latency(arm.config, arm.context)
        
        # Skip dominated arms early (surrogate-based pruning)
        if is_dominated_by_surrogate(arm, evaluated_arms, latency_pred):
            continue
            
        # Evaluate promising arms with cache warmup
        observed_metrics = evaluate_arm_with_cache_warmup(
            arm=arm,
            warmup_period_sec=cache_warmup_period,
            measurement_period_sec=1800  # 30 minutes
        )
        
        evaluated_arms.append({
            'arm': arm,
            'predicted_latency': latency_pred['predicted_p95_latency'],
            'observed_latency': observed_metrics['p95_latency'],
            'ndcg_delta': observed_metrics['ndcg_at_10'] - T0_CONTRACTS['ndcg_at_10']['value'],
            'cache_hit_rate': observed_metrics['cache_hit_rate'],
            'is_pareto_optimal': None  # Will be computed after all evaluations
        })
    
    # Identify Pareto frontier
    pareto_optimal_arms = find_pareto_frontier(
        evaluated_arms, 
        objectives=['observed_latency', 'ndcg_delta'],
        maximize=[False, True]  # Minimize latency, maximize nDCG
    )
    
    return pareto_optimal_arms

def enforce_cache_residency_floor(arm_config, min_hit_rate=0.75):
    """
    Prevent false wins from transient warm caches
    
    Require sustained performance across cold-start windows
    """
    if arm_config.expected_cache_hit_rate < min_hit_rate:
        return False, f"Cache hit rate {arm_config.expected_cache_hit_rate} below floor {min_hit_rate}"
    
    # Test performance during cache cold-start
    cold_start_metrics = evaluate_during_cache_flush(arm_config)
    
    if cold_start_metrics.p95_latency > arm_config.warm_cache_p95 * 1.5:
        return False, "Excessive cold-start latency degradation"
        
    return True, "Cache residency requirements satisfied"
```

---

## 📊 Counterfactual Logging & Off-Policy Evaluation

### IPS/DR Implementation
```python
class CounterfactualLogger:
    """
    Emit propensities and outcomes for unbiased off-policy evaluation
    
    Enables IPS (Inverse Propensity Scoring) and DR (Doubly Robust) estimators
    """
    def __init__(self, logging_backend):
        self.backend = logging_backend
        self.propensity_model = None
        
    def log_decision(self, context, action, propensity, outcome):
        """Log decision data for offline analysis"""
        log_entry = {
            'timestamp': time.time(),
            'context_features': context.to_dict(),
            'action_taken': action.to_dict(), 
            'propensity_score': propensity,
            'observed_outcome': outcome.to_dict(),
            'exploration_track': action.track  # 'router' or 'ann'
        }
        
        self.backend.write(log_entry)
        
    def compute_ips_estimate(self, target_policy, logged_data):
        """
        Inverse Propensity Scoring for off-policy evaluation
        
        E[R(π)] ≈ (1/n) Σ (π(a|x) / π₀(a|x)) * r
        """
        ips_estimates = []
        
        for entry in logged_data:
            context = entry['context_features']
            action = entry['action_taken']
            logged_propensity = entry['propensity_score']
            reward = entry['observed_outcome']['reward']
            
            # Target policy propensity
            target_propensity = target_policy.get_action_probability(context, action)
            
            # IPS weight
            importance_weight = target_propensity / logged_propensity
            
            # Clamp weights to reduce variance
            importance_weight = np.clip(importance_weight, 0.1, 10.0)
            
            ips_estimates.append(importance_weight * reward)
            
        return {
            'ips_estimate': np.mean(ips_estimates),
            'ips_variance': np.var(ips_estimates),
            'effective_sample_size': len(ips_estimates)
        }
        
    def compute_doubly_robust_estimate(self, target_policy, reward_model, logged_data):
        """
        Doubly Robust estimator combining IPS + direct method
        
        More robust than IPS alone when either propensity or reward model is correct
        """
        dr_estimates = []
        
        for entry in logged_data:
            context = entry['context_features']
            action = entry['action_taken'] 
            logged_propensity = entry['propensity_score']
            observed_reward = entry['observed_outcome']['reward']
            
            # Target policy propensity
            target_propensity = target_policy.get_action_probability(context, action)
            
            # Predicted reward under target policy
            predicted_reward = reward_model.predict(context, target_policy.get_action(context))
            
            # DR estimator formula
            importance_weight = target_propensity / logged_propensity
            ips_term = importance_weight * observed_reward
            dm_term = predicted_reward
            correction_term = importance_weight * (observed_reward - reward_model.predict(context, action))
            
            dr_estimate = dm_term + correction_term
            dr_estimates.append(dr_estimate)
            
        return {
            'dr_estimate': np.mean(dr_estimates),
            'dr_variance': np.var(dr_estimates),
            'bias_estimate': np.abs(np.mean(dr_estimates) - np.mean([e['observed_outcome']['reward'] for e in logged_data]))
        }
```

---

## 🚦 Delta-Gate Promotion Logic

### CI Lower Bound Validation
```python
def validate_promotion_criteria(arm_metrics, baseline_contracts):
    """
    Promotion requires lower CI bound clearing zero on key slices
    
    LCB(ΔnDCG@10) ≥ 0 on global + hard-NL slices for promotion to canary
    """
    promotion_checks = []
    
    # Global nDCG improvement requirement
    global_ndcg_delta = arm_metrics.global_ndcg - baseline_contracts['ndcg_at_10']['value']
    global_ndcg_lcb = global_ndcg_delta - arm_metrics.global_ndcg_ci_half_width
    
    if global_ndcg_lcb >= 0:
        promotion_checks.append({'slice': 'global', 'metric': 'nDCG@10', 'status': 'PASS', 
                               'lcb': global_ndcg_lcb, 'delta': global_ndcg_delta})
    else:
        promotion_checks.append({'slice': 'global', 'metric': 'nDCG@10', 'status': 'FAIL',
                               'lcb': global_ndcg_lcb, 'required': 0.0})
    
    # Hard-NL slice nDCG improvement
    hard_nl_ndcg_delta = arm_metrics.hard_nl_ndcg - baseline_contracts['ndcg_at_10']['value']
    hard_nl_ndcg_lcb = hard_nl_ndcg_delta - arm_metrics.hard_nl_ndcg_ci_half_width
    
    if hard_nl_ndcg_lcb >= 0:
        promotion_checks.append({'slice': 'hard_nl', 'metric': 'nDCG@10', 'status': 'PASS',
                               'lcb': hard_nl_ndcg_lcb, 'delta': hard_nl_ndcg_delta})
    else:
        promotion_checks.append({'slice': 'hard_nl', 'metric': 'nDCG@10', 'status': 'FAIL',
                               'lcb': hard_nl_ndcg_lcb, 'required': 0.0})
    
    # Latency constraint check
    p95_delta = arm_metrics.p95_latency - baseline_contracts['p95_latency']['value']
    if p95_delta <= 0.3:  # +0.3ms budget
        promotion_checks.append({'metric': 'p95_latency', 'status': 'PASS',
                               'delta': p95_delta, 'budget': 0.3})
    else:
        promotion_checks.append({'metric': 'p95_latency', 'status': 'FAIL',
                               'delta': p95_delta, 'budget_exceeded': p95_delta - 0.3})
    
    # SLA-Recall maintenance
    sla_recall_delta = arm_metrics.sla_recall_50 - baseline_contracts['sla_recall_50']['value']
    if sla_recall_delta >= 0:
        promotion_checks.append({'metric': 'SLA_recall_50', 'status': 'PASS', 'delta': sla_recall_delta})
    else:
        promotion_checks.append({'metric': 'SLA_recall_50', 'status': 'FAIL', 'delta': sla_recall_delta})
    
    all_passed = all(check['status'] == 'PASS' for check in promotion_checks)
    
    return {
        'promotion_eligible': all_passed,
        'checks': promotion_checks,
        'summary': f"{'PASS' if all_passed else 'FAIL'}: {sum(1 for c in promotion_checks if c['status'] == 'PASS')}/{len(promotion_checks)} criteria met"
    }
```

### Adapter Collapse & Exactifier Monitoring
```python
def monitor_adaptation_health(current_metrics, t0_baseline):
    """
    Guard against adapter collapse and exactifier cost bugs
    
    Alert conditions:
    - Jaccard@10 vs T₀ mean < 0.80 (adapter collapse)
    - Clamp rate trending up (exactifier cost bug)
    """
    health_alerts = []
    
    # Adapter collapse detection
    jaccard_similarity = calculate_jaccard_at_10(current_metrics.results, t0_baseline.results)
    if jaccard_similarity.mean < 0.80:
        health_alerts.append({
            'type': 'adapter_collapse',
            'severity': 'HIGH',
            'jaccard_mean': jaccard_similarity.mean,
            'threshold': 0.80,
            'recommendation': 'Investigate ranking drift, consider arm retirement'
        })
    
    # Exactifier cost monitoring
    current_clamp_rate = current_metrics.clamp_rate_percent
    baseline_clamp_rate = t0_baseline.clamp_rate_percent
    clamp_increase = current_clamp_rate - baseline_clamp_rate
    
    if clamp_increase > 1.0:  # >1% increase in clamping
        health_alerts.append({
            'type': 'exactifier_cost_increase', 
            'severity': 'MEDIUM',
            'current_clamp_rate': current_clamp_rate,
            'baseline_clamp_rate': baseline_clamp_rate,
            'increase': clamp_increase,
            'recommendation': 'Monitor entropy query distribution, check for cost regression'
        })
    
    # Merged bin rate monitoring (related to exactifier behavior)
    current_merged_bin_rate = current_metrics.merged_bin_rate_percent
    baseline_merged_bin_rate = t0_baseline.merged_bin_rate_percent
    
    if current_merged_bin_rate > baseline_merged_bin_rate * 1.5:
        health_alerts.append({
            'type': 'merged_bin_rate_spike',
            'severity': 'LOW',
            'current_rate': current_merged_bin_rate,
            'baseline_rate': baseline_merged_bin_rate,
            'recommendation': 'Investigate query complexity distribution changes'
        })
    
    return {
        'health_status': 'HEALTHY' if len(health_alerts) == 0 else 'DEGRADED',
        'alerts': health_alerts,
        'jaccard_similarity': jaccard_similarity.mean,
        'clamp_rate_delta': clamp_increase
    }
```

---

**Exploration Epoch Owner**: Search Engineering Team  
**Router Track Lead**: Senior Engineer (Contextual Bandits)  
**ANN Track Lead**: Senior Engineer (Pareto Optimization)  
**Safety Monitoring**: Site Reliability Engineering  

**Status**: ✅ READY - Protected Innovation Loop Initialized  
**Next Checkpoint**: September 26, 2025 (2-week progress review)

---

*This specification establishes the protected framework for aggressive optimization exploration while maintaining mathematical guarantees of T₀ baseline safety through frozen contracts and rigorous promotion criteria.*