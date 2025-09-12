# 📊 Pooled-Qrels Refresh Cadence & Data Hygiene Pipeline

**Refresh Frequency**: 6-week cycles  
**Current Pool Version**: v2.2  
**Next Refresh**: October 23, 2025 (6 weeks from T₀ baseline)  
**Alignment**: Synchronized with quarterly exploration cycles

---

## 🔄 6-Week Refresh Cycle Rationale

### Statistical Stability Requirements
- **CI Width Maintenance**: 6-week cycle keeps confidence intervals stable during exploration
- **Bootstrap Sample Size**: N≥800 queries per slice maintained across refreshes  
- **Temporal Correlation**: Minimizes measurement drift while capturing real user behavior evolution
- **Exploration Alignment**: Perfect sync with Thompson sampling cycles for router/ANN optimization

### Business Cycle Integration  
- **Development Sprints**: Aligns with 3×2-week development cycles
- **Quarterly Planning**: 2 refreshes per quarter enable mid-quarter course corrections
- **Holiday Planning**: Avoids major refreshes during freeze periods (Dec 15-Jan 5)
- **Team Capacity**: Sustainable workload for data engineering team

---

## 🎯 Pooled-Qrels Composition Management

### Current Pool Statistics (v2.2)
```json
{
  "total_queries": 2400,
  "slices": {
    "global": {"queries": 800, "coverage": "all query types"},
    "nl_queries": {"queries": 400, "coverage": "natural language, semantic"},
    "lexical_queries": {"queries": 400, "coverage": "exact terms, API names"},  
    "long_queries": {"queries": 400, "coverage": ">8 tokens"},
    "short_queries": {"queries": 400, "coverage": "≤4 tokens"}
  },
  "intent_coverage": {
    "debugging": 15,
    "implementation": 25,
    "conceptual": 20,
    "api_reference": 18,
    "troubleshooting": 12,
    "performance": 10
  }
}
```

### Edge Intent Backfill Strategy
```python
def identify_edge_intents(user_query_logs, current_pool):
    """
    Systematic identification of emerging query patterns for pool refresh
    """
    # Analyze 6-week query volume by intent classification
    query_trends = analyze_query_classification_drift(
        start_date=datetime.now() - timedelta(weeks=6),
        end_date=datetime.now(),
        min_query_volume=100
    )
    
    # Identify underrepresented intents in current pool
    coverage_gaps = []
    for intent, volume in query_trends.items():
        current_representation = current_pool.count_intent(intent)
        expected_representation = volume * 0.0003  # 0.03% sampling rate
        
        if current_representation < expected_representation * 0.5:
            coverage_gaps.append({
                'intent': intent,
                'current': current_representation,
                'needed': int(expected_representation - current_representation),
                'priority': volume  # Higher volume = higher priority
            })
    
    return sorted(coverage_gaps, key=lambda x: x['priority'], reverse=True)

# Example output for October 2025 refresh
edge_intents_backfill = [
    {'intent': 'rust_async_patterns', 'needed': 25, 'priority': 3200},
    {'intent': 'react_server_components', 'needed': 18, 'priority': 2100},
    {'intent': 'python_type_narrowing', 'needed': 15, 'priority': 1800},
    {'intent': 'go_generics_advanced', 'needed': 12, 'priority': 1200}
]
```

### Stale Query Retirement Process
```python
def identify_stale_queries(pool_queries, user_behavior_data):
    """
    Identify queries that no longer reflect current user behavior patterns
    """
    stale_candidates = []
    
    for query in pool_queries:
        # Check if query pattern still appears in user logs
        recent_similar = count_similar_queries_recent(query.text, days=42)  # 6 weeks
        historical_similar = query.historical_frequency
        
        # Technology relevance check
        tech_mentions = extract_technologies(query.text)
        deprecated_tech = check_deprecation_status(tech_mentions)
        
        # User engagement signals
        avg_click_through = calculate_ctr_recent(query.text, days=42)
        
        staleness_score = calculate_staleness(
            frequency_ratio=recent_similar / max(historical_similar, 1),
            deprecated_tech_count=len(deprecated_tech),
            ctr_drop=max(0, query.baseline_ctr - avg_click_through)
        )
        
        if staleness_score > 0.7:  # High staleness threshold
            stale_candidates.append({
                'query': query,
                'staleness_score': staleness_score,
                'reasons': {
                    'low_frequency': recent_similar < historical_similar * 0.3,
                    'deprecated_tech': len(deprecated_tech) > 0,
                    'poor_engagement': avg_click_through < query.baseline_ctr * 0.5
                }
            })
    
    return sorted(stale_candidates, key=lambda x: x['staleness_score'], reverse=True)
```

---

## 📋 Refresh Execution Checklist

### Pre-Refresh Analysis (Week -1)
- [ ] **User Query Analysis**: 6-week log analysis for intent drift
- [ ] **Technology Trend Review**: New frameworks, deprecated APIs, emerging patterns  
- [ ] **Current Pool Performance**: Identify underperforming queries (low CTR, poor relevance)
- [ ] **Edge Case Discovery**: Rare but important query patterns missing from pool
- [ ] **Stakeholder Review**: Engineering team input on new domain areas

### Pool Composition Updates (Week 0)
- [ ] **Stale Query Removal**: Remove queries with staleness_score > 0.7
- [ ] **Edge Intent Addition**: Add identified edge cases maintaining slice balance
- [ ] **Quality Validation**: All new queries validated by domain experts  
- [ ] **CALIB_V22 Compatibility**: Ensure new queries work with existing calibration
- [ ] **A/B Test Preparation**: Split-test new pool vs current for validation

### Validation & Deployment (Week +1)
- [ ] **Bootstrap Validation**: Generate CI whiskers with new pool composition
- [ ] **Performance Baseline**: Measure metric stability with refreshed pool
- [ ] **Regression Testing**: No significant metric shifts due to pool changes alone
- [ ] **Documentation Update**: Pool v2.3 composition and rationale documented
- [ ] **Team Communication**: Share refresh results and impact analysis

---

## 🎛️ Automated Pool Management System

### Continuous Monitoring Dashboard
```yaml
metrics_tracked:
  pool_coverage:
    - "intent_distribution_drift"
    - "technology_relevance_score" 
    - "query_freshness_index"
    - "user_behavior_alignment"
    
  statistical_stability:
    - "bootstrap_ci_width_trend"
    - "inter_refresh_correlation"
    - "metric_variance_stability"
    - "sample_size_adequacy"
    
  performance_impact:
    - "pool_change_metric_delta"
    - "refresh_induced_variance"
    - "baseline_drift_attribution"
    - "exploration_ci_stability"
```

### Automated Alerts & Triggers
```python
class PoolMonitoringSystem:
    def __init__(self):
        self.alert_thresholds = {
            'intent_drift': 0.15,           # 15% change in intent distribution
            'ci_width_increase': 0.20,      # 20% CI width expansion
            'staleness_accumulation': 0.30,  # 30% of pool queries stale
            'coverage_gap': 0.05            # 5% of user queries uncovered
        }
    
    def check_early_refresh_trigger(self, current_metrics):
        """
        Determine if early refresh needed before 6-week cycle
        """
        triggers = []
        
        if current_metrics.intent_drift > self.alert_thresholds['intent_drift']:
            triggers.append('significant_intent_distribution_shift')
            
        if current_metrics.ci_width_ratio > 1 + self.alert_thresholds['ci_width_increase']:
            triggers.append('bootstrap_ci_degradation')
            
        if current_metrics.stale_query_percentage > self.alert_thresholds['staleness_accumulation']:
            triggers.append('excessive_staleness_detected')
            
        return {
            'early_refresh_recommended': len(triggers) > 0,
            'triggers': triggers,
            'urgency': 'high' if len(triggers) >= 2 else 'medium'
        }
```

---

## 🔍 Data Hygiene Protocols

### Query Quality Standards
```yaml
inclusion_criteria:
  - "represents_real_user_intent: true"
  - "has_clear_correct_answer: true" 
  - "technology_currently_relevant: true"
  - "sufficient_corpus_coverage: >5_relevant_results"
  - "expert_validation_passed: true"
  
exclusion_criteria:
  - "deprecated_technology_primary: true"
  - "spam_or_nonsense: true"
  - "personally_identifiable_info: true"
  - "copyright_violation: true"
  - "single_result_dependency: true"  # Queries with only 1 good answer
```

### Expert Validation Process
```python
class QueryValidationWorkflow:
    def __init__(self):
        self.validation_stages = [
            'automated_quality_check',
            'domain_expert_review', 
            'corpus_coverage_validation',
            'calib_v22_compatibility_test',
            'peer_review_confirmation'
        ]
    
    def validate_new_query_batch(self, queries):
        """
        Multi-stage validation for new pool queries
        """
        results = {'passed': [], 'failed': [], 'needs_revision': []}
        
        for query in queries:
            validation_score = 0
            feedback = []
            
            # Stage 1: Automated checks
            if self.passes_automated_quality_check(query):
                validation_score += 20
            else:
                feedback.append('automated_quality_issues')
            
            # Stage 2: Domain expert review (async)
            expert_rating = self.get_expert_rating(query)  # 0-40 points
            validation_score += expert_rating
            if expert_rating < 25:
                feedback.append('expert_concerns')
            
            # Stage 3: Corpus coverage
            coverage_score = self.check_corpus_coverage(query)  # 0-20 points
            validation_score += coverage_score
            if coverage_score < 10:
                feedback.append('insufficient_corpus_coverage')
            
            # Stage 4: CALIB_V22 compatibility
            if self.check_calibration_compatibility(query):
                validation_score += 20
            else:
                feedback.append('calibration_incompatible')
            
            # Classification based on total score (0-100)
            if validation_score >= 80:
                results['passed'].append(query)
            elif validation_score >= 60:
                results['needs_revision'].append({'query': query, 'feedback': feedback})
            else:
                results['failed'].append({'query': query, 'feedback': feedback})
        
        return results
```

---

## 📅 Refresh Schedule & Timeline

### 2025-2026 Refresh Calendar
```yaml
refresh_v2_3:
  date: "2025-10-23"
  focus: "rust_async + react_server_components edge intents"
  queries_added: 50
  queries_removed: 25
  
refresh_v2_4: 
  date: "2025-12-04"
  focus: "year_end_technology_trends + holiday_freeze_prep"
  queries_added: 30
  queries_removed: 15
  
refresh_v2_5:
  date: "2026-01-15"
  focus: "post_holiday_catchup + q1_tech_trends"
  queries_added: 60
  queries_removed: 35
  
refresh_v2_6:
  date: "2026-02-26"  
  focus: "spring_framework_updates + ai_tooling_queries"
  queries_added: 45
  queries_removed: 20
```

### Holiday & Freeze Considerations
- **December 15 - January 5**: No pool refreshes during holiday freeze
- **Conference Seasons**: Align refreshes with major tech conferences for trend capture
- **Framework Release Cycles**: Monitor major framework releases for backfill opportunities
- **Team PTO Planning**: Ensure sufficient expert validator availability

---

## 🎯 Success Metrics & KPIs

### Pool Quality Metrics
- **Intent Coverage**: ≥95% of user query intents represented
- **Technology Relevance**: ≤5% queries using deprecated technologies  
- **Expert Validation Rate**: ≥90% of new queries pass validation
- **Staleness Rate**: ≤10% of pool queries flagged as stale between refreshes

### Statistical Stability Metrics  
- **CI Width Stability**: ≤10% variance in bootstrap CI widths between refreshes
- **Metric Correlation**: ≥0.95 correlation in key metrics before/after refresh
- **Exploration Impact**: Pool refreshes don't interfere with ongoing optimization exploration
- **Baseline Preservation**: T₀ baseline metrics maintained within error budgets

### Operational Efficiency Metrics
- **Refresh Execution Time**: ≤5 business days per refresh cycle
- **Expert Validator Utilization**: ≤8 hours per validator per refresh
- **Automated Processing Rate**: ≥85% of pool management automated
- **Cost per Query**: ≤$0.50 total cost per new query added

---

## 🔧 Implementation Scripts

### Pool Refresh Execution
```bash
#!/bin/bash
# Automated pool refresh pipeline

POOL_VERSION="$1"  # e.g., "v2.3"
REFRESH_DATE="$2"  # e.g., "2025-10-23"

echo "🔄 Starting pooled-qrels refresh to $POOL_VERSION"

# Step 1: Generate edge intent analysis
python3 scripts/analyze_edge_intents.py \
  --lookback-weeks 6 \
  --min-query-volume 100 \
  --output "edge_intents_$POOL_VERSION.json"

# Step 2: Identify stale queries
python3 scripts/identify_stale_queries.py \
  --current-pool "pooled_qrels_v2.2.json" \
  --staleness-threshold 0.7 \
  --output "stale_queries_$POOL_VERSION.json"

# Step 3: Generate new pool composition
python3 scripts/generate_pool_refresh.py \
  --current-pool "pooled_qrels_v2.2.json" \
  --edge-intents "edge_intents_$POOL_VERSION.json" \
  --stale-queries "stale_queries_$POOL_VERSION.json" \
  --target-size 2400 \
  --output "pooled_qrels_$POOL_VERSION.json"

# Step 4: Validate new pool
python3 scripts/validate_pool_composition.py \
  --pool "pooled_qrels_$POOL_VERSION.json" \
  --expert-validation \
  --calib-v22-check \
  --output "validation_report_$POOL_VERSION.json"

# Step 5: Bootstrap CI generation
python3 scripts/generate_bootstrap_cis.py \
  --pool "pooled_qrels_$POOL_VERSION.json" \
  --bootstrap-iterations 2000 \
  --confidence-level 0.95 \
  --output "ci_whiskers_$POOL_VERSION.json"

echo "✅ Pool refresh to $POOL_VERSION completed"
echo "📊 Validation report: validation_report_$POOL_VERSION.json"
echo "📈 CI whiskers: ci_whiskers_$POOL_VERSION.json"
```

---

**Pool Management Owner**: Data Engineering Team  
**Validation Coordinators**: Search Engineering + Domain Experts  
**Statistical Analysis**: Data Science Team  

**Status**: ✅ ACTIVE - 6-Week Refresh Cadence Established  
**Next Refresh**: October 23, 2025 (Pool v2.3)

---

*This pooled-qrels refresh system maintains statistical stability while capturing evolving user behavior and technology trends, enabling continuous optimization without measurement drift.*