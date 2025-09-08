#!/usr/bin/env python3
"""
Statistical Analysis Pipeline - Bootstrap + Permutation Testing
Provides rigorous statistical analysis for Benchmark Protocol v2.0
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import bootstrap, permutation_test
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceInterval:
    lower: float
    upper: float
    confidence_level: float
    method: str

@dataclass 
class EffectSize:
    statistic: float
    magnitude: str  # "negligible", "small", "medium", "large"
    interpretation: str

@dataclass
class StatisticalTest:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str
    confidence_interval: Optional[ConfidenceInterval] = None
    effect_size: Optional[EffectSize] = None

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis with bootstrap and permutation testing
    Implements authentic scientific methodology for benchmark analysis
    """
    
    def __init__(self, confidence_level: float = 0.95, bootstrap_samples: int = 10000, random_state: int = 42):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        logger.info(f"Statistical analyzer initialized: confidence={confidence_level}, bootstrap_samples={bootstrap_samples}")
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of benchmark results
        """
        logger.info("Starting comprehensive statistical analysis...")
        
        analysis = {
            "metadata": {
                "confidence_level": self.confidence_level,
                "bootstrap_samples": self.bootstrap_samples,
                "total_observations": len(df),
                "systems": df['system'].unique().tolist(),
                "scenarios": df['scenario'].unique().tolist()
            }
        }
        
        # 1. Descriptive Statistics
        analysis["descriptive"] = self._compute_descriptive_statistics(df)
        
        # 2. Performance Comparison
        analysis["performance_comparison"] = self._analyze_performance_comparison(df)
        
        # 3. SLA Analysis
        analysis["sla_analysis"] = self._analyze_sla_compliance(df)
        
        # 4. Quality Metrics Analysis
        analysis["quality_analysis"] = self._analyze_quality_metrics(df)
        
        # 5. System Rankings
        analysis["rankings"] = self._compute_system_rankings(df)
        
        # 6. Gap Analysis
        analysis["gap_analysis"] = self._compute_gap_analysis(df)
        
        logger.info("Statistical analysis complete")
        return analysis
    
    def _compute_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute descriptive statistics for all metrics"""
        logger.info("Computing descriptive statistics...")
        
        metrics = ['lat_ms', 'ndcg@10', 'recall@50', 'success@10', 'memory_gb', 'qps150x']
        
        descriptive = {}
        
        for metric in metrics:
            if metric in df.columns:
                descriptive[metric] = {
                    "mean": float(df[metric].mean()),
                    "std": float(df[metric].std()),
                    "median": float(df[metric].median()),
                    "q25": float(df[metric].quantile(0.25)),
                    "q75": float(df[metric].quantile(0.75)),
                    "min": float(df[metric].min()),
                    "max": float(df[metric].max()),
                    "skewness": float(stats.skew(df[metric].dropna())),
                    "kurtosis": float(stats.kurtosis(df[metric].dropna()))
                }
                
                # Add confidence interval for mean
                ci = self._bootstrap_confidence_interval(df[metric].values, np.mean)
                descriptive[metric]["mean_ci"] = {
                    "lower": ci.lower,
                    "upper": ci.upper,
                    "confidence_level": ci.confidence_level
                }
        
        return descriptive
    
    def _analyze_performance_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance across systems using rigorous statistical tests"""
        logger.info("Analyzing performance comparisons...")
        
        performance_analysis = {}
        
        # Group by system
        systems = df['system'].unique()
        
        # Pairwise comparisons for latency
        latency_comparisons = {}
        
        for i, system1 in enumerate(systems):
            for system2 in systems[i+1:]:
                group1 = df[df['system'] == system1]['lat_ms'].values
                group2 = df[df['system'] == system2]['lat_ms'].values
                
                comparison_key = f"{system1}_vs_{system2}"
                
                # Permutation test for difference in means
                permutation_result = self._permutation_test_difference(group1, group2, 'latency')
                
                # Bootstrap confidence interval for difference
                diff_ci = self._bootstrap_difference_ci(group1, group2, np.mean)
                
                # Effect size (Cohen's d)
                effect_size = self._compute_cohens_d(group1, group2)
                
                latency_comparisons[comparison_key] = {
                    "system1": system1,
                    "system2": system2,
                    "system1_mean": float(np.mean(group1)),
                    "system2_mean": float(np.mean(group2)),
                    "difference": float(np.mean(group1) - np.mean(group2)),
                    "permutation_test": permutation_result,
                    "difference_ci": diff_ci,
                    "effect_size": effect_size
                }
        
        performance_analysis["latency_comparisons"] = latency_comparisons
        
        # Quality metric comparisons (NDCG@10)
        if 'ndcg@10' in df.columns:
            ndcg_comparisons = {}
            
            for i, system1 in enumerate(systems):
                for system2 in systems[i+1:]:
                    group1 = df[df['system'] == system1]['ndcg@10'].values
                    group2 = df[df['system'] == system2]['ndcg@10'].values
                    
                    comparison_key = f"{system1}_vs_{system2}"
                    
                    # Permutation test
                    permutation_result = self._permutation_test_difference(group1, group2, 'ndcg')
                    
                    # Bootstrap CI for difference
                    diff_ci = self._bootstrap_difference_ci(group1, group2, np.mean)
                    
                    # Effect size
                    effect_size = self._compute_cohens_d(group1, group2)
                    
                    ndcg_comparisons[comparison_key] = {
                        "system1": system1,
                        "system2": system2,
                        "system1_mean": float(np.mean(group1)),
                        "system2_mean": float(np.mean(group2)),
                        "difference": float(np.mean(group1) - np.mean(group2)),
                        "permutation_test": permutation_result,
                        "difference_ci": diff_ci,
                        "effect_size": effect_size
                    }
            
            performance_analysis["ndcg_comparisons"] = ndcg_comparisons
        
        return performance_analysis
    
    def _analyze_sla_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze SLA compliance across systems and scenarios"""
        logger.info("Analyzing SLA compliance...")
        
        # Calculate SLA violation rates by system
        sla_violations = df[df['lat_ms'] > df['sla_ms']]
        
        sla_analysis = {
            "overall_violation_rate": len(sla_violations) / len(df),
            "by_system": {},
            "by_scenario": {}
        }
        
        # By system
        for system in df['system'].unique():
            system_data = df[df['system'] == system]
            violations = len(system_data[system_data['lat_ms'] > system_data['sla_ms']])
            total = len(system_data)
            
            violation_rate = violations / total if total > 0 else 0
            
            # Bootstrap confidence interval for violation rate
            violation_indicators = (system_data['lat_ms'] > system_data['sla_ms']).astype(int).values
            violation_rate_ci = self._bootstrap_confidence_interval(violation_indicators, np.mean)
            
            sla_analysis["by_system"][system] = {
                "violation_rate": violation_rate,
                "violations": violations,
                "total": total,
                "violation_rate_ci": {
                    "lower": violation_rate_ci.lower,
                    "upper": violation_rate_ci.upper,
                    "confidence_level": violation_rate_ci.confidence_level
                }
            }
        
        # By scenario
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            violations = len(scenario_data[scenario_data['lat_ms'] > scenario_data['sla_ms']])
            total = len(scenario_data)
            
            violation_rate = violations / total if total > 0 else 0
            
            sla_analysis["by_scenario"][scenario] = {
                "violation_rate": violation_rate,
                "violations": violations,
                "total": total
            }
        
        return sla_analysis
    
    def _analyze_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality metrics with statistical rigor"""
        logger.info("Analyzing quality metrics...")
        
        quality_metrics = ['ndcg@10', 'recall@50', 'success@10']
        
        quality_analysis = {
            "by_system": {},
            "correlation_analysis": {}
        }
        
        # Quality by system
        for system in df['system'].unique():
            system_data = df[df['system'] == system]
            system_analysis = {}
            
            for metric in quality_metrics:
                if metric in system_data.columns:
                    values = system_data[metric].values
                    
                    # Bootstrap statistics
                    mean_ci = self._bootstrap_confidence_interval(values, np.mean)
                    median_ci = self._bootstrap_confidence_interval(values, np.median)
                    
                    system_analysis[metric] = {
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "std": float(np.std(values)),
                        "mean_ci": {
                            "lower": mean_ci.lower,
                            "upper": mean_ci.upper,
                            "confidence_level": mean_ci.confidence_level
                        },
                        "median_ci": {
                            "lower": median_ci.lower,
                            "upper": median_ci.upper,
                            "confidence_level": median_ci.confidence_level
                        }
                    }
            
            quality_analysis["by_system"][system] = system_analysis
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        quality_analysis["correlation_analysis"] = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "key_correlations": {
                "latency_vs_quality": correlation_matrix.loc['lat_ms', 'ndcg@10'] if 'lat_ms' in correlation_matrix.index and 'ndcg@10' in correlation_matrix.columns else None,
                "memory_vs_latency": correlation_matrix.loc['memory_gb', 'lat_ms'] if 'memory_gb' in correlation_matrix.index and 'lat_ms' in correlation_matrix.columns else None
            }
        }
        
        return quality_analysis
    
    def _compute_system_rankings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute system rankings across different metrics"""
        logger.info("Computing system rankings...")
        
        rankings = {}
        
        # Group by system and compute means
        system_means = df.groupby('system').agg({
            'lat_ms': 'mean',
            'ndcg@10': 'mean',
            'recall@50': 'mean',  
            'success@10': 'mean',
            'memory_gb': 'mean',
            'qps150x': 'mean'
        }).round(4)
        
        # Rank systems (lower is better for latency and memory, higher is better for quality metrics)
        rankings['latency'] = system_means['lat_ms'].rank(ascending=True).to_dict()
        rankings['ndcg'] = system_means['ndcg@10'].rank(ascending=False).to_dict()
        rankings['recall'] = system_means['recall@50'].rank(ascending=False).to_dict()
        rankings['success'] = system_means['success@10'].rank(ascending=False).to_dict()
        rankings['memory'] = system_means['memory_gb'].rank(ascending=True).to_dict()
        rankings['qps'] = system_means['qps150x'].rank(ascending=False).to_dict()
        
        # Compute overall ranking (weighted combination)
        weights = {
            'latency': 0.3,   # Lower latency is better
            'ndcg': 0.25,     # Higher quality is better
            'recall': 0.25,   # Higher recall is better
            'memory': 0.2     # Lower memory is better
        }
        
        overall_scores = {}
        for system in system_means.index:
            score = 0
            score += weights['latency'] * (1 / rankings['latency'][system])  # Invert for lower-is-better
            score += weights['ndcg'] * (1 / rankings['ndcg'][system])
            score += weights['recall'] * (1 / rankings['recall'][system])
            score += weights['memory'] * (1 / rankings['memory'][system])  # Invert for lower-is-better
            overall_scores[system] = score
        
        rankings['overall'] = {system: rank for rank, system in enumerate(
            sorted(overall_scores.keys(), key=lambda x: overall_scores[x], reverse=True), 1
        )}
        
        return rankings
    
    def _compute_gap_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute gap analysis: Lens vs best competitor"""
        logger.info("Computing gap analysis...")
        
        gap_analysis = {}
        
        # Find Lens performance
        lens_data = df[df['system'] == 'lens']
        if lens_data.empty:
            logger.warning("No Lens data found for gap analysis")
            return {"error": "No Lens data available"}
        
        # For each scenario, find best non-Lens system
        scenarios = df['scenario'].unique()
        
        for scenario in scenarios:
            scenario_data = df[df['scenario'] == scenario]
            lens_scenario = lens_data[lens_data['scenario'] == scenario]
            
            if lens_scenario.empty:
                continue
                
            # Find best competitor (highest NDCG@10)
            non_lens_data = scenario_data[scenario_data['system'] != 'lens']
            if non_lens_data.empty:
                continue
                
            best_competitor_ndcg = non_lens_data.groupby('system')['ndcg@10'].mean().idxmax()
            best_competitor_data = non_lens_data[non_lens_data['system'] == best_competitor_ndcg]
            
            # Calculate gaps
            lens_ndcg = lens_scenario['ndcg@10'].mean()
            competitor_ndcg = best_competitor_data['ndcg@10'].mean()
            ndcg_gap = lens_ndcg - competitor_ndcg
            
            lens_latency = lens_scenario['lat_ms'].mean()
            competitor_latency = best_competitor_data['lat_ms'].mean()
            latency_gap = lens_latency - competitor_latency
            
            # Statistical test for significance
            ndcg_test = self._permutation_test_difference(
                lens_scenario['ndcg@10'].values,
                best_competitor_data['ndcg@10'].values,
                'ndcg'
            )
            
            gap_analysis[scenario] = {
                "best_competitor": best_competitor_ndcg,
                "lens_ndcg": lens_ndcg,
                "competitor_ndcg": competitor_ndcg,
                "ndcg_gap": ndcg_gap,
                "lens_latency": lens_latency,
                "competitor_latency": competitor_latency, 
                "latency_gap": latency_gap,
                "ndcg_gap_significant": ndcg_test["significant"],
                "ndcg_gap_p_value": ndcg_test["p_value"]
            }
        
        return gap_analysis
    
    # Statistical utility methods
    
    def _bootstrap_confidence_interval(self, data: np.ndarray, statistic_func) -> ConfidenceInterval:
        """Compute bootstrap confidence interval"""
        if len(data) == 0:
            return ConfidenceInterval(0, 0, self.confidence_level, "bootstrap")
            
        try:
            res = bootstrap((data,), statistic_func, n_resamples=self.bootstrap_samples, 
                          confidence_level=self.confidence_level, random_state=self.rng)
            return ConfidenceInterval(
                lower=float(res.confidence_interval.low),
                upper=float(res.confidence_interval.high),
                confidence_level=self.confidence_level,
                method="bootstrap"
            )
        except Exception as e:
            logger.warning(f"Bootstrap CI failed: {e}")
            return ConfidenceInterval(0, 0, self.confidence_level, "bootstrap")
    
    def _bootstrap_difference_ci(self, group1: np.ndarray, group2: np.ndarray, statistic_func) -> ConfidenceInterval:
        """Compute bootstrap CI for difference between two groups"""
        if len(group1) == 0 or len(group2) == 0:
            return ConfidenceInterval(0, 0, self.confidence_level, "bootstrap_difference")
        
        def difference_statistic(x, y):
            return statistic_func(x) - statistic_func(y)
        
        try:
            res = bootstrap((group1, group2), difference_statistic, n_resamples=self.bootstrap_samples,
                          confidence_level=self.confidence_level, random_state=self.rng)
            return ConfidenceInterval(
                lower=float(res.confidence_interval.low),
                upper=float(res.confidence_interval.high),
                confidence_level=self.confidence_level,
                method="bootstrap_difference"
            )
        except Exception as e:
            logger.warning(f"Bootstrap difference CI failed: {e}")
            return ConfidenceInterval(0, 0, self.confidence_level, "bootstrap_difference")
    
    def _permutation_test_difference(self, group1: np.ndarray, group2: np.ndarray, metric_type: str) -> StatisticalTest:
        """Perform permutation test for difference in means"""
        if len(group1) == 0 or len(group2) == 0:
            return StatisticalTest("permutation", 0, 1, False, "Insufficient data")
        
        def statistic(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        
        try:
            # Determine alternative hypothesis based on metric type
            alternative = 'two-sided'  # Default
            if metric_type == 'latency':
                alternative = 'less'  # We expect Lens to have lower latency
            elif metric_type == 'ndcg':
                alternative = 'greater'  # We expect Lens to have higher quality
                
            res = permutation_test((group1, group2), statistic, 
                                 n_resamples=self.bootstrap_samples,
                                 alternative=alternative,
                                 random_state=self.rng)
            
            significant = res.pvalue < self.alpha
            
            interpretation = f"Difference is {'significant' if significant else 'not significant'} at α={self.alpha}"
            
            return StatisticalTest(
                test_name="permutation_test",
                statistic=float(res.statistic),
                p_value=float(res.pvalue),
                significant=significant,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.warning(f"Permutation test failed: {e}")
            return StatisticalTest("permutation", 0, 1, False, f"Test failed: {e}")
    
    def _compute_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> EffectSize:
        """Compute Cohen's d effect size"""
        if len(group1) == 0 or len(group2) == 0:
            return EffectSize(0, "negligible", "Insufficient data")
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Interpret magnitude
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = "negligible"
            interpretation = "Very small practical difference"
        elif abs_d < 0.5:
            magnitude = "small"
            interpretation = "Small practical difference"
        elif abs_d < 0.8:
            magnitude = "medium"
            interpretation = "Medium practical difference"
        else:
            magnitude = "large"
            interpretation = "Large practical difference"
        
        return EffectSize(
            statistic=float(cohens_d),
            magnitude=magnitude,
            interpretation=interpretation
        )