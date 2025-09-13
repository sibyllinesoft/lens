#!/usr/bin/env python3
"""
Sequential Probability Ratio Test (SPRT) for Statistical Canary Deployment
Implements rigorous statistical testing for production deployments with early stopping.

Mathematical Foundation:
- H₀: p = p₀ (baseline performance)  
- H₁: p = p₁ = p₀ + δ (improved performance)
- Accept if Λₙ ≥ log((1-β)/α)
- Reject if Λₙ ≤ log(β/(1-α))
- Continue if between thresholds

Where Λₙ = Σᵢ log(p₁^xᵢ(1-p₁)^(1-xᵢ) / p₀^xᵢ(1-p₀)^(1-xᵢ))
"""

import math
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SPRTResult:
    """Result of SPRT test evaluation"""
    decision: str  # "accept", "reject", "continue"
    terminate: bool
    log_likelihood_ratio: float
    sample_count: int
    accept_threshold: float
    reject_threshold: float
    p_value_estimate: Optional[float] = None

class SPRTCanaryTester:
    """Statistical canary testing using Sequential Probability Ratio Test"""
    
    def __init__(self, 
                 baseline_p0: float = 0.90,
                 min_detectable_delta: float = 0.03,
                 alpha: float = 0.05,
                 beta: float = 0.05,
                 max_samples: int = 10000):
        """
        Initialize SPRT tester
        
        Args:
            baseline_p0: Baseline success rate (null hypothesis)
            min_detectable_delta: Minimum detectable improvement
            alpha: Type I error rate (false positive)
            beta: Type II error rate (false negative) 
            max_samples: Maximum samples before forced decision
        """
        self.baseline_p0 = baseline_p0
        self.alternative_p1 = baseline_p0 + min_detectable_delta
        self.alpha = alpha
        self.beta = beta
        self.max_samples = max_samples
        
        # Calculate SPRT thresholds
        self.accept_threshold = math.log((1 - beta) / alpha)
        self.reject_threshold = math.log(beta / (1 - alpha))
        
        # Test state
        self.reset()
        
        logger.info(f"SPRT Canary Tester initialized:")
        logger.info(f"  H₀: p = {baseline_p0} (baseline)")
        logger.info(f"  H₁: p = {self.alternative_p1} (alternative)")
        logger.info(f"  α = {alpha}, β = {beta}")
        logger.info(f"  Accept threshold: {self.accept_threshold:.3f}")
        logger.info(f"  Reject threshold: {self.reject_threshold:.3f}")
    
    def reset(self):
        """Reset test state for new experiment"""
        self.log_likelihood_ratio = 0.0
        self.sample_count = 0
        self.observations = []
        self.test_history = []
        
    def add_observation(self, success: bool) -> SPRTResult:
        """
        Add a single observation and evaluate SPRT decision
        
        Args:
            success: Whether the query/test was successful
            
        Returns:
            SPRTResult with current decision and statistics
        """
        x_i = 1 if success else 0
        self.observations.append(x_i)
        self.sample_count += 1
        
        # Update log likelihood ratio using SPRT formula
        if x_i == 1:
            # Success: log(p₁/p₀)
            delta_lambda = math.log(self.alternative_p1 / self.baseline_p0)
        else:
            # Failure: log((1-p₁)/(1-p₀))
            delta_lambda = math.log((1 - self.alternative_p1) / (1 - self.baseline_p0))
            
        self.log_likelihood_ratio += delta_lambda
        
        # Evaluate decision
        result = self._evaluate_decision()
        
        # Record in history
        self.test_history.append({
            "sample": self.sample_count,
            "observation": x_i,
            "lambda": self.log_likelihood_ratio,
            "decision": result.decision
        })
        
        if result.terminate or self.sample_count >= self.max_samples:
            logger.info(f"SPRT Decision after {self.sample_count} samples: {result.decision}")
            if self.sample_count >= self.max_samples and not result.terminate:
                logger.warning("Reached maximum samples - forcing decision based on current evidence")
        
        return result
    
    def add_batch_observations(self, successes: List[bool]) -> SPRTResult:
        """
        Add multiple observations and return final decision
        
        Args:
            successes: List of success/failure outcomes
            
        Returns:
            SPRTResult after processing all observations
        """
        result = None
        for success in successes:
            result = self.add_observation(success)
            if result.terminate:
                break
                
        return result
    
    def _evaluate_decision(self) -> SPRTResult:
        """Evaluate SPRT decision based on current log likelihood ratio"""
        
        # Check termination conditions
        if self.log_likelihood_ratio >= self.accept_threshold:
            decision = "accept"
            terminate = True
        elif self.log_likelihood_ratio <= self.reject_threshold:
            decision = "reject" 
            terminate = True
        elif self.sample_count >= self.max_samples:
            # Force decision based on current evidence
            if self.log_likelihood_ratio > 0:
                decision = "accept"
            else:
                decision = "reject"
            terminate = True
        else:
            decision = "continue"
            terminate = False
        
        # Estimate p-value (approximate)
        p_value_estimate = self._estimate_p_value() if self.sample_count > 10 else None
        
        return SPRTResult(
            decision=decision,
            terminate=terminate,
            log_likelihood_ratio=self.log_likelihood_ratio,
            sample_count=self.sample_count,
            accept_threshold=self.accept_threshold,
            reject_threshold=self.reject_threshold,
            p_value_estimate=p_value_estimate
        )
    
    def _estimate_p_value(self) -> float:
        """Estimate p-value based on current sample statistics"""
        if self.sample_count == 0:
            return 1.0
            
        # Simple empirical p-value estimation
        observed_success_rate = sum(self.observations) / len(self.observations)
        
        # Use normal approximation for large samples
        if self.sample_count >= 30:
            # Z-test approximation
            expected_variance = self.baseline_p0 * (1 - self.baseline_p0)
            z_score = (observed_success_rate - self.baseline_p0) / math.sqrt(expected_variance / self.sample_count)
            
            # Two-tailed p-value approximation
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            return max(0.001, min(0.999, p_value))  # Clamp to reasonable range
        else:
            # For small samples, return conservative estimate
            return 0.5
    
    def get_power_analysis(self, effect_size: float = None) -> Dict:
        """
        Calculate statistical power and required sample size
        
        Args:
            effect_size: Effect size to test (defaults to configured delta)
            
        Returns:
            Dictionary with power analysis results
        """
        if effect_size is None:
            effect_size = self.alternative_p1 - self.baseline_p0
            
        # Expected sample sizes under H₀ and H₁ (theoretical)
        # These are approximations based on SPRT theory
        
        # Under H₀ (null hypothesis)
        e_n0_accept = self.alpha * self.accept_threshold + (1 - self.alpha) * self.reject_threshold
        e_n0_reject = (1 - self.alpha) * self.reject_threshold + self.alpha * self.accept_threshold
        
        # Under H₁ (alternative hypothesis)  
        e_n1_accept = (1 - self.beta) * self.accept_threshold + self.beta * self.reject_threshold
        e_n1_reject = self.beta * self.reject_threshold + (1 - self.beta) * self.accept_threshold
        
        # Expected sample size (weighted average)
        expected_n_h0 = self.alpha * abs(e_n0_accept) + (1 - self.alpha) * abs(e_n0_reject)
        expected_n_h1 = (1 - self.beta) * abs(e_n1_accept) + self.beta * abs(e_n1_reject)
        
        # Approximate using log-odds difference
        log_odds_diff = math.log(self.alternative_p1 / (1 - self.alternative_p1)) - math.log(self.baseline_p0 / (1 - self.baseline_p0))
        
        if abs(log_odds_diff) > 0.001:
            expected_n_h0 /= abs(log_odds_diff)
            expected_n_h1 /= abs(log_odds_diff)
        
        return {
            "effect_size": effect_size,
            "statistical_power": 1 - self.beta,
            "significance_level": self.alpha,
            "expected_samples_h0": max(1, int(expected_n_h0)),
            "expected_samples_h1": max(1, int(expected_n_h1)),
            "baseline_p0": self.baseline_p0,
            "alternative_p1": self.alternative_p1,
            "accept_threshold": self.accept_threshold,
            "reject_threshold": self.reject_threshold
        }
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        if self.sample_count == 0:
            return "No observations recorded yet."
            
        report = []
        report.append("# 📊 SPRT Canary Test Report")
        report.append("=" * 50)
        report.append("")
        
        # Test parameters
        report.append("## Test Configuration")
        report.append(f"- **Baseline (H₀):** p₀ = {self.baseline_p0}")
        report.append(f"- **Alternative (H₁):** p₁ = {self.alternative_p1}")
        report.append(f"- **Min Detectable Delta:** {self.alternative_p1 - self.baseline_p0:.3f}")
        report.append(f"- **Type I Error (α):** {self.alpha}")
        report.append(f"- **Type II Error (β):** {self.beta}")
        report.append(f"- **Statistical Power:** {1 - self.beta}")
        report.append("")
        
        # Current results
        current_result = self._evaluate_decision()
        observed_rate = sum(self.observations) / len(self.observations)
        
        report.append("## Test Results")
        report.append(f"- **Sample Count:** {self.sample_count}")
        report.append(f"- **Observed Success Rate:** {observed_rate:.3f}")
        report.append(f"- **Log Likelihood Ratio:** {self.log_likelihood_ratio:.3f}")
        report.append(f"- **Current Decision:** {current_result.decision.upper()}")
        
        if current_result.p_value_estimate:
            report.append(f"- **Estimated p-value:** {current_result.p_value_estimate:.4f}")
        
        # Decision boundaries
        report.append("")
        report.append("## Decision Boundaries")
        report.append(f"- **Accept Threshold:** {self.accept_threshold:.3f}")
        report.append(f"- **Reject Threshold:** {self.reject_threshold:.3f}")
        report.append(f"- **Current Position:** {self.log_likelihood_ratio:.3f}")
        
        # Interpretation
        report.append("")
        report.append("## Interpretation")
        if current_result.decision == "accept":
            report.append("✅ **ACCEPT**: Strong statistical evidence of improvement")
            report.append("   → Candidate performs significantly better than baseline")
        elif current_result.decision == "reject":
            report.append("❌ **REJECT**: Statistical evidence against improvement")  
            report.append("   → Candidate does not show significant improvement")
        else:
            report.append("⏳ **CONTINUE**: Insufficient evidence for decision")
            report.append("   → More samples needed to reach statistical conclusion")
        
        # Power analysis
        power_analysis = self.get_power_analysis()
        report.append("")
        report.append("## Power Analysis")
        report.append(f"- **Expected Samples (H₀):** {power_analysis['expected_samples_h0']}")
        report.append(f"- **Expected Samples (H₁):** {power_analysis['expected_samples_h1']}")
        
        return "\n".join(report)
    
    def export_test_data(self) -> Dict:
        """Export complete test data for analysis"""
        current_result = self._evaluate_decision()
        
        return {
            "test_config": {
                "baseline_p0": self.baseline_p0,
                "alternative_p1": self.alternative_p1,
                "alpha": self.alpha,
                "beta": self.beta,
                "min_detectable_delta": self.alternative_p1 - self.baseline_p0
            },
            "thresholds": {
                "accept": self.accept_threshold,
                "reject": self.reject_threshold
            },
            "results": {
                "sample_count": self.sample_count,
                "log_likelihood_ratio": self.log_likelihood_ratio,
                "decision": current_result.decision,
                "terminate": current_result.terminate,
                "observed_success_rate": sum(self.observations) / len(self.observations) if self.observations else 0.0,
                "p_value_estimate": current_result.p_value_estimate
            },
            "observations": self.observations,
            "test_history": self.test_history,
            "power_analysis": self.get_power_analysis(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

# Example usage and testing
if __name__ == "__main__":
    # Example: Test with simulated data
    import argparse
    
    parser = argparse.ArgumentParser(description="SPRT Canary Tester")
    parser.add_argument("--baseline", type=float, default=0.90, help="Baseline success rate")
    parser.add_argument("--delta", type=float, default=0.03, help="Minimum detectable improvement")
    parser.add_argument("--alpha", type=float, default=0.05, help="Type I error rate")
    parser.add_argument("--beta", type=float, default=0.05, help="Type II error rate")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SPRTCanaryTester(
        baseline_p0=args.baseline,
        min_detectable_delta=args.delta,
        alpha=args.alpha,
        beta=args.beta
    )
    
    if args.simulate:
        # Simulate canary test with improved performance
        print("🔬 Simulating SPRT canary test...")
        
        # Simulate data with actual improvement (p = 0.93 vs baseline 0.90)
        true_success_rate = args.baseline + args.delta + 0.01  # Slightly better than minimum
        
        for i in range(1000):  # Maximum 1000 samples
            # Generate observation based on true success rate
            success = np.random.random() < true_success_rate
            result = tester.add_observation(success)
            
            if result.terminate:
                break
        
        # Print results
        print(f"\n{tester.generate_test_report()}")
        
        # Export data
        test_data = tester.export_test_data()
        with open("sprt_test_results.json", "w") as f:
            json.dump(test_data, f, indent=2)
        
        print(f"\n📁 Test data exported to: sprt_test_results.json")
    
    else:
        print("Use --simulate to run a simulation, or import this module to use in production")
        print(f"Configured for: H₀: p={args.baseline}, H₁: p={args.baseline + args.delta}")