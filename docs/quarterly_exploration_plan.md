# 🎯 Quarterly Exploration Plan - Next Optimization Deltas

**Planning Period**: Q4 2025 (Oct-Dec)  
**Baseline**: T0-2025-09-12T04:47:39Z  
**Objective**: Chase +1-2 pp improvements through router spend shaping + ANN recall@latency frontier

---

## 🧠 Router Spend Shaping - Adaptive Tau Policy

### Current State Analysis
- **Fixed Tau**: 0.62 across all query types
- **Spend Cap**: 6ms uniformly applied
- **Opportunity**: Per-slice adaptive spending based on query characteristics

### Exploration Hypothesis
**Hard NL Queries** (high entropy, complex semantic needs) should get **higher tau thresholds** and **adaptive spend budgets** to maximize semantic upshift benefits while maintaining latency SLA.

### Mathematical Framework
```python
def adaptive_tau_policy(query_features):
    """
    Adaptive tau with per-slice priors using Thompson sampling
    
    Goal: +0.5 pp on "hard NL" at Δp95 ≤ +0.3ms
    """
    base_tau = 0.62
    
    # Query complexity features
    entropy_score = calculate_entropy(query_features.tokens)
    nl_confidence = query_features.nl_classifier_score  
    query_length = len(query_features.tokens)
    
    # Per-slice adaptive adjustments
    if nl_confidence > 0.8 and entropy_score > 2.5:
        # Hard NL queries - increase semantic budget
        tau_adjustment = +0.08  # 0.62 → 0.70
        spend_cap_ms = 8        # 6ms → 8ms
        min_conf_gain = 0.10    # Lower barrier for upshift
        
    elif nl_confidence > 0.6 and query_length > 8:
        # Medium complexity NL - modest increase  
        tau_adjustment = +0.04  # 0.62 → 0.66
        spend_cap_ms = 7        # 6ms → 7ms
        min_conf_gain = 0.12
        
    elif nl_confidence < 0.3:
        # Lexical queries - reduce semantic spending
        tau_adjustment = -0.05  # 0.62 → 0.57  
        spend_cap_ms = 4        # 6ms → 4ms (save compute)
        min_conf_gain = 0.18    # Higher barrier
        
    else:
        # Default/mixed queries - baseline behavior
        tau_adjustment = 0.0
        spend_cap_ms = 6
        min_conf_gain = 0.15
    
    return {
        'tau': base_tau + tau_adjustment,
        'spend_cap_ms': spend_cap_ms,
        'min_conf_gain': min_conf_gain
    }
```

### Thompson Sampling Exploration
- **Exploration Window**: 6 weeks (aligns with pool refresh)
- **Arms**: 8 different (tau, spend_cap, min_conf_gain) combinations
- **Prior**: Beta(1,1) for each arm's success probability
- **Success Metric**: Δ(nDCG@10) on hard NL queries subject to latency constraint
- **Sample Allocation**: Thompson sampling with minimum 5% traffic per arm

### Validation Protocol
```yaml
week_1_2:
  traffic_split: "shadow_only"
  sample_size_per_arm: "N≥1000 hard NL queries"
  metrics: ["ndcg@10", "p95_latency", "semantic_spend_rate"]
  
week_3_4:
  traffic_split: "micro_canary_1%"
  gate_validation: "ci_whisker_cleared"
  safety_bounds: "T0_error_budgets"
  
week_5_6:
  traffic_split: "progressive_canary"
  winning_arm: "highest_posterior_mean"
  fallback: "revert_to_T0_on_violation"
```

---

## 🔍 ANN Recall@Latency Pareto Frontier

### Current State Analysis
- **efSearch**: 32 (fixed)
- **refine_topk**: 48 (fixed)  
- **Cache Residency**: ~85% hit rate
- **Opportunity**: Multi-dimensional optimization for -1ms p95 at ΔnDCG@10 ≥ 0

### Pareto Frontier Search Space
```python
search_space = {
    'efSearch': [24, 28, 32, 36, 40],           # Current: 32
    'refine_topk': [32, 40, 48, 56, 64],       # Current: 48
    'cache_residency_target': [0.80, 0.85, 0.90, 0.95],  # Current: ~0.85
    'prefetch_neighbors': [True, False],        # Current: True
    'visited_set_reuse': [True, False],         # Current: True
}

# Total combinations: 5 × 5 × 4 × 2 × 2 = 200 configs
# Thompson sampling reduces to ~12-16 promising arms
```

### Multi-Objective Optimization
**Primary Objective**: Minimize p95 latency  
**Constraint**: ΔnDCG@10 ≥ 0 (no quality regression)  
**Secondary**: Maximize throughput (queries/sec/core)

```python
def pareto_dominance_check(config_a, config_b):
    """
    Config A dominates B if:
    - A.latency ≤ B.latency AND A.quality ≥ B.quality
    - At least one inequality is strict
    """
    latency_better = config_a.p95_latency <= config_b.p95_latency
    quality_better = config_a.ndcg_at_10 >= config_b.ndcg_at_10
    
    strict_improvement = (
        config_a.p95_latency < config_b.p95_latency or 
        config_a.ndcg_at_10 > config_b.ndcg_at_10
    )
    
    return latency_better and quality_better and strict_improvement

def thompson_sampling_pareto(arms, observations):
    """
    Thompson sampling adapted for Pareto frontier exploration
    """
    # Model each arm's (latency, quality) as bivariate normal
    posteriors = []
    for arm in arms:
        obs = observations[arm.config_id]
        if len(obs) > 0:
            # Bayesian posterior updates
            latency_posterior = update_gaussian_posterior(obs.latencies)
            quality_posterior = update_gaussian_posterior(obs.qualities) 
        else:
            # Uninformative priors
            latency_posterior = GaussianPrior(mu=120, sigma=10)
            quality_posterior = GaussianPrior(mu=0.345, sigma=0.01)
        
        posteriors.append({
            'arm': arm,
            'latency': latency_posterior,
            'quality': quality_posterior
        })
    
    # Sample from posteriors and select non-dominated arms
    samples = []
    for p in posteriors:
        latency_sample = p['latency'].sample()
        quality_sample = p['quality'].sample()
        samples.append({
            'arm': p['arm'],
            'latency': latency_sample,
            'quality': quality_sample
        })
    
    # Find Pareto frontier in sampled space
    pareto_arms = find_pareto_frontier(samples)
    
    # Allocation proportional to frontier proximity
    return allocate_traffic_to_pareto_arms(pareto_arms)
```

### Exploration Timeline
```yaml
month_1_october:
  focus: "efSearch + refine_topk grid search"
  arms: 25  # 5×5 combinations
  traffic: "shadow_only"
  goal: "identify top 6 arms for month 2"
  
month_2_november:  
  focus: "cache_residency + prefetch optimization"
  arms: 6   # Top arms from month 1 × cache variations
  traffic: "micro_canary_2%"
  goal: "pareto frontier identification"
  
month_3_december:
  focus: "production validation of pareto winners"
  arms: 2-3 # Non-dominated configurations only
  traffic: "full_canary_progressive"
  goal: "T1_baseline_candidate selection"
```

---

## 🔤 Lexical Enhancement - Query Length Adaptive Boosting

### Current State Analysis  
- **Phrase Boost**: 1.25 (uniform across query lengths)
- **Window Tokens**: 16 (fixed)
- **Opportunity**: Query length-sensitive phrase boosting for NL queries

### Adaptive Boosting Strategy
```python
def adaptive_phrase_boost(query_tokens, nl_confidence):
    """
    Query length and NL confidence adaptive phrase boosting
    
    Goal: Ensure ΔSLA-Recall@50 ≥ 0 on NL queries
    """
    base_boost = 1.25
    token_count = len(query_tokens)
    
    if nl_confidence > 0.7:  # High confidence NL
        if token_count <= 4:
            # Short NL: reduce phrase emphasis (favor semantic)
            phrase_boost = 1.10
            window_tokens = 12
        elif token_count >= 10:
            # Long NL: increase phrase emphasis (structure matters)
            phrase_boost = 1.40  
            window_tokens = 24
        else:
            # Medium NL: baseline behavior
            phrase_boost = 1.25
            window_tokens = 16
            
    else:  # Lexical or mixed queries
        if token_count <= 3:
            # Short lexical: strong phrase boost (exact matching)
            phrase_boost = 1.50
            window_tokens = 8
        else:
            # Longer lexical: moderate boost
            phrase_boost = 1.30
            window_tokens = 20
    
    return {
        'phrase_boost': phrase_boost,
        'window_tokens': window_tokens,
        'ordered_boost': min(1.20, phrase_boost * 0.8)  # Cap ordered boost
    }
```

### A/B Testing Framework
- **Control**: Fixed phrase_boost=1.25, window_tokens=16
- **Treatment**: Adaptive boosting based on query length + NL confidence
- **Slices**: Short NL (≤4 tokens), Long NL (≥10 tokens), Lexical queries
- **Primary Metric**: SLA-Recall@50 (must maintain Δ ≥ 0)
- **Secondary**: nDCG@10, file credit efficiency

---

## 📅 Integrated Exploration Schedule

### October 2025 - Foundation Building
```yaml
week_1:
  router: "baseline_tau_analysis + hard_nl_identification"
  ann: "efSearch_grid_search_shadow"
  lexical: "query_length_distribution_analysis"
  
week_2:
  router: "thompson_sampling_setup + prior_initialization"
  ann: "refine_topk_exploration_shadow"  
  lexical: "adaptive_boost_logic_validation"
  
week_3:
  router: "micro_canary_1%_tau_adaptive"
  ann: "efSearch_refine_combined_optimization"
  lexical: "shadow_traffic_adaptive_boost"
  
week_4:
  router: "performance_analysis + arm_elimination"
  ann: "pareto_frontier_identification"
  lexical: "CI_whisker_validation"
```

### November 2025 - Production Validation
```yaml
week_1:
  router: "winning_tau_policy_canary_5%"
  ann: "cache_residency_optimization"
  lexical: "micro_canary_adaptive_boost"
  
week_2:
  router: "progressive_canary_25%"
  ann: "prefetch_strategy_validation"  
  lexical: "A/B_test_full_deployment"
  
week_3:
  router: "full_deployment_candidate"
  ann: "pareto_winner_canary"
  lexical: "performance_monitoring"
  
week_4:
  router: "performance_measurement + ROI"
  ann: "production_stability_validation"
  lexical: "SLA_recall_verification"
```

### December 2025 - Integration & Next Baseline
```yaml
week_1:
  integration: "three_system_combined_testing"
  measurement: "aggregate_improvement_validation"
  safety: "comprehensive_gate_validation"
  
week_2:  
  integration: "interaction_effect_analysis"
  measurement: "CI_bounds_establishment"
  safety: "failure_mode_testing"
  
week_3:
  integration: "production_ready_validation"  
  measurement: "baseline_update_preparation"
  safety: "T1_baseline_candidate_finalization"
  
week_4:
  integration: "year_end_performance_review"
  measurement: "2026_roadmap_planning"
  safety: "holiday_freeze_preparation"
```

---

## 🎯 Success Criteria & KPIs

### Primary Objectives (Quarter End)
- **Router**: +0.5 pp nDCG@10 on hard NL queries, Δp95 ≤ +0.3ms
- **ANN**: -1ms p95 latency at ΔnDCG@10 ≥ 0
- **Lexical**: ΔSLA-Recall@50 ≥ 0 on NL queries with improved efficiency

### Combined System Goals
- **Aggregate nDCG@10**: 0.345 → 0.355 (+1.0 pp minimum)
- **Latency Profile**: p95 ≤ 118ms (maintain or improve)
- **Cost Efficiency**: ≤$0.0023/request (maintain or improve)
- **SLA Compliance**: 150ms p95 with >95% queries

### Innovation Metrics
- **Exploration Velocity**: 3 major optimization areas in parallel
- **Thompson Sampling Efficiency**: >80% of traffic allocated to top 3 arms by month 2
- **Pareto Frontier Discovery**: ≥2 non-dominated ANN configurations identified
- **Risk Management**: Zero T₀ baseline violations during exploration

---

## 🔄 Pool Refresh Integration

### 6-Week Cadence Alignment
- **Router Exploration**: Aligns with 6-week Thompson sampling cycles
- **ANN Optimization**: Monthly cycles with 6-week validation periods  
- **Lexical Testing**: 6-week A/B test minimum for statistical power
- **Pool Refresh**: Every 6 weeks maintains CI stability during exploration

### Edge Intent Backfill Strategy
- **New Query Types**: Emerging programming languages, frameworks, APIs
- **Edge Cases**: Error handling patterns, performance optimization queries  
- **User Behavior Evolution**: Natural language programming questions
- **Quality Gates**: All new intents validated against CALIB_V22 compatibility

---

**Quarterly Plan Owner**: Search Engineering Team  
**Optimization Leads**: Router (Sr. Eng A), ANN (Sr. Eng B), Lexical (Sr. Eng C)  
**Statistical Validation**: Data Science Team  

**Status**: ✅ APPROVED - Q4 2025 Exploration Plan Active  
**Next Review**: Monthly progress check (October 31, 2025)

---

*This quarterly plan enables systematic exploration of the next performance frontier while maintaining T₀ baseline safety through Thompson sampling and Pareto optimization.*