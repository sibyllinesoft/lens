//! # Critical Calibration Gate Enforcer
//!
//! Implements the strict ECE ≤ 0.02 requirement with proper gate enforcement
//! as specified in TODO.md. This is the BLOCKER fix for the ECE gate failure.
//!
//! **CRITICAL**: ECE = 0.021 > 0.02 threshold was showing as false PASS - FIXED
//!
//! Gates enforced:
//! - max-slice ECE ≤ 0.02 (not average)
//! - ΔnDCG@10 ≥ 0 (NL)
//! - SLA-Recall@50 ≥ 0
//! - p99 ≤ 150ms
//! - Slope ∈ [0.9, 1.1] for isotonic calibration

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{error, info, warn};

/// Strict gate enforcement system
pub struct CalibrationGateEnforcer {
    config: GateConfig,
    violation_history: Vec<GateViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateConfig {
    /// CRITICAL: Maximum ECE per slice (not average)
    pub max_slice_ece: f32,
    /// Minimum nDCG@10 delta for NL queries (pp)
    pub min_ndcg10_delta_nl_pp: f32,
    /// Minimum SLA-Recall@50 delta
    pub min_sla_recall_delta: f32,
    /// Maximum p99 latency (ms)
    pub max_p99_ms: f32,
    /// Minimum slope for isotonic calibration
    pub min_slope: f32,
    /// Maximum slope for isotonic calibration
    pub max_slope: f32,
    /// Tolerance for floating point comparisons
    pub tolerance: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            max_slice_ece: 0.020,    // CRITICAL: 0.02 threshold from TODO.md
            min_ndcg10_delta_nl_pp: 4.0,  // +4pp minimum improvement
            min_sla_recall_delta: 0.0,    // No regression allowed
            max_p99_ms: 150.0,            // 150ms SLA requirement
            min_slope: 0.9,               // Slope constraints [0.9, 1.1]
            max_slope: 1.1,
            tolerance: 1e-6,              // Floating point tolerance
        }
    }
}

/// Gate validation result with detailed failure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateValidationResult {
    pub calibration_gate: GateResult,
    pub sla_gate: GateResult,
    pub quality_gate: GateResult,
    pub overall_pass: bool,
    pub violations: Vec<GateViolation>,
    pub validation_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub passed: bool,
    pub metric_name: String,
    pub actual_value: f32,
    pub threshold: f32,
    pub violation_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateViolation {
    pub gate_name: String,
    pub metric_name: String,
    pub actual_value: f32,
    pub threshold: f32,
    pub severity: ViolationSeverity,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub error_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Critical,  // Blocks deployment
    Warning,   // Logged but not blocking
}

/// Metrics for gate validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// ECE per {intent×language} slice - CRITICAL: max must be ≤ 0.02
    pub ece_by_slice: HashMap<String, f32>,
    /// nDCG@10 improvement for NL queries (percentage points)
    pub ndcg10_delta_nl_pp: f32,
    /// SLA-Recall@50 delta (must be ≥ 0)
    pub sla_recall_delta: f32,
    /// P99 latency in milliseconds
    pub p99_latency_ms: f32,
    /// Isotonic calibration slopes by slice
    pub slopes_by_slice: HashMap<String, f32>,
    /// Additional quality metrics
    pub quality_metrics: QualityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub core_at_10: f32,
    pub diversity_at_10: f32,
    pub coverage_percentage: f32,
    pub total_queries_processed: usize,
}

impl CalibrationGateEnforcer {
    /// Create new gate enforcer with strict configuration
    pub fn new(config: GateConfig) -> Self {
        info!("Initializing strict calibration gate enforcer");
        info!("CRITICAL ECE threshold: {:.3} (max per slice)", config.max_slice_ece);
        info!("SLA requirements: p99 ≤ {}ms, nDCG@10 ≥ +{:.1}pp", 
              config.max_p99_ms, config.min_ndcg10_delta_nl_pp);
        
        Self {
            config,
            violation_history: Vec::new(),
        }
    }

    /// CRITICAL: Enforce all gates with proper failure reporting
    pub fn validate_gates(&mut self, metrics: &ValidationMetrics) -> Result<GateValidationResult> {
        info!("Enforcing critical gates on validation metrics");
        
        let mut violations = Vec::new();
        
        // GATE 1: CALIBRATION - max-slice ECE ≤ 0.02 (CRITICAL FIX)
        let calibration_gate = self.validate_calibration_gate(metrics, &mut violations)?;
        
        // GATE 2: SLA - p99 ≤ 150ms, SLA-Recall@50 ≥ 0
        let sla_gate = self.validate_sla_gate(metrics, &mut violations)?;
        
        // GATE 3: QUALITY - ΔnDCG@10 ≥ +4pp
        let quality_gate = self.validate_quality_gate(metrics, &mut violations)?;
        
        let overall_pass = calibration_gate.passed && sla_gate.passed && quality_gate.passed;
        
        // Store violations in history
        self.violation_history.extend(violations.clone());
        
        // Log critical failures
        if !overall_pass {
            error!("GATE VALIDATION FAILED - DEPLOYMENT BLOCKED");
            for violation in &violations {
                if matches!(violation.severity, ViolationSeverity::Critical) {
                    error!("CRITICAL VIOLATION: {}", violation.error_message);
                }
            }
        } else {
            info!("All gates PASSED - validation successful");
        }
        
        Ok(GateValidationResult {
            calibration_gate,
            sla_gate,
            quality_gate,
            overall_pass,
            violations,
            validation_timestamp: chrono::Utc::now(),
        })
    }

    /// CRITICAL: Validate calibration gate - max ECE per slice ≤ 0.02
    fn validate_calibration_gate(
        &self, 
        metrics: &ValidationMetrics,
        violations: &mut Vec<GateViolation>
    ) -> Result<GateResult> {
        // Find maximum ECE across all slices - CRITICAL: not average!
        let max_slice_ece = metrics.ece_by_slice.values()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(1.0); // Fail-safe: assume worst case if no data
        
        let passed = max_slice_ece <= self.config.max_slice_ece + self.config.tolerance;
        
        let result = if passed {
            info!("CALIBRATION GATE: PASS - max ECE {:.4} ≤ {:.3}", 
                  max_slice_ece, self.config.max_slice_ece);
            GateResult {
                passed: true,
                metric_name: "max_slice_ECE".to_string(),
                actual_value: max_slice_ece,
                threshold: self.config.max_slice_ece,
                violation_message: None,
            }
        } else {
            let error_msg = format!("FAIL:ECE {:.4} > {:.3}", max_slice_ece, self.config.max_slice_ece);
            error!("CALIBRATION GATE: {}", error_msg);
            
            // Find which slice(s) violated the threshold
            let violating_slices: Vec<String> = metrics.ece_by_slice.iter()
                .filter(|(_, &ece)| ece > self.config.max_slice_ece + self.config.tolerance)
                .map(|(slice, ece)| format!("{}:{:.4}", slice, ece))
                .collect();
            
            let violation = GateViolation {
                gate_name: "CALIBRATION".to_string(),
                metric_name: "max_slice_ECE".to_string(),
                actual_value: max_slice_ece,
                threshold: self.config.max_slice_ece,
                severity: ViolationSeverity::Critical,
                timestamp: chrono::Utc::now(),
                error_message: format!("{} (violating slices: {})", error_msg, violating_slices.join(", ")),
            };
            violations.push(violation);
            
            GateResult {
                passed: false,
                metric_name: "max_slice_ECE".to_string(),
                actual_value: max_slice_ece,
                threshold: self.config.max_slice_ece,
                violation_message: Some(error_msg),
            }
        };
        
        // Validate slope constraints for isotonic calibration
        self.validate_slope_constraints(metrics, violations)?;
        
        Ok(result)
    }

    /// Validate SLA gate - p99 ≤ 150ms, SLA-Recall@50 ≥ 0
    fn validate_sla_gate(
        &self,
        metrics: &ValidationMetrics,
        violations: &mut Vec<GateViolation>
    ) -> Result<GateResult> {
        let p99_passed = metrics.p99_latency_ms <= self.config.max_p99_ms + self.config.tolerance;
        let recall_passed = metrics.sla_recall_delta >= self.config.min_sla_recall_delta - self.config.tolerance;
        
        let overall_passed = p99_passed && recall_passed;
        
        if !p99_passed {
            let error_msg = format!("FAIL:P99 {:.1}ms > {:.0}ms", 
                                   metrics.p99_latency_ms, self.config.max_p99_ms);
            error!("SLA GATE: {}", error_msg);
            
            violations.push(GateViolation {
                gate_name: "SLA".to_string(),
                metric_name: "p99_latency_ms".to_string(),
                actual_value: metrics.p99_latency_ms,
                threshold: self.config.max_p99_ms,
                severity: ViolationSeverity::Critical,
                timestamp: chrono::Utc::now(),
                error_message: error_msg,
            });
        }
        
        if !recall_passed {
            let error_msg = format!("FAIL:SLA-Recall {:.3} < {:.1}", 
                                   metrics.sla_recall_delta, self.config.min_sla_recall_delta);
            error!("SLA GATE: {}", error_msg);
            
            violations.push(GateViolation {
                gate_name: "SLA".to_string(),
                metric_name: "sla_recall_delta".to_string(),
                actual_value: metrics.sla_recall_delta,
                threshold: self.config.min_sla_recall_delta,
                severity: ViolationSeverity::Critical,
                timestamp: chrono::Utc::now(),
                error_message: error_msg,
            });
        }
        
        if overall_passed {
            info!("SLA GATE: PASS - p99 {:.1}ms ≤ {}ms, SLA-Recall Δ {:.3} ≥ 0", 
                  metrics.p99_latency_ms, self.config.max_p99_ms, metrics.sla_recall_delta);
        }
        
        Ok(GateResult {
            passed: overall_passed,
            metric_name: "SLA_composite".to_string(),
            actual_value: metrics.p99_latency_ms, // Primary SLA metric
            threshold: self.config.max_p99_ms,
            violation_message: if overall_passed { None } else { Some("SLA requirements not met".to_string()) },
        })
    }

    /// Validate quality gate - ΔnDCG@10 ≥ +4pp
    fn validate_quality_gate(
        &self,
        metrics: &ValidationMetrics,
        violations: &mut Vec<GateViolation>
    ) -> Result<GateResult> {
        let passed = metrics.ndcg10_delta_nl_pp >= self.config.min_ndcg10_delta_nl_pp - self.config.tolerance;
        
        let result = if passed {
            info!("QUALITY GATE: PASS - nDCG@10 Δ {:.1}pp ≥ +{:.0}pp", 
                  metrics.ndcg10_delta_nl_pp, self.config.min_ndcg10_delta_nl_pp);
            GateResult {
                passed: true,
                metric_name: "ndcg10_delta_nl_pp".to_string(),
                actual_value: metrics.ndcg10_delta_nl_pp,
                threshold: self.config.min_ndcg10_delta_nl_pp,
                violation_message: None,
            }
        } else {
            let error_msg = format!("FAIL:ΔnDCG {:.1}pp < +{:.0}pp", 
                                   metrics.ndcg10_delta_nl_pp, self.config.min_ndcg10_delta_nl_pp);
            error!("QUALITY GATE: {}", error_msg);
            
            violations.push(GateViolation {
                gate_name: "QUALITY".to_string(),
                metric_name: "ndcg10_delta_nl_pp".to_string(),
                actual_value: metrics.ndcg10_delta_nl_pp,
                threshold: self.config.min_ndcg10_delta_nl_pp,
                severity: ViolationSeverity::Critical,
                timestamp: chrono::Utc::now(),
                error_message: error_msg,
            });
            
            GateResult {
                passed: false,
                metric_name: "ndcg10_delta_nl_pp".to_string(),
                actual_value: metrics.ndcg10_delta_nl_pp,
                threshold: self.config.min_ndcg10_delta_nl_pp,
                violation_message: Some(error_msg),
            }
        };
        
        Ok(result)
    }

    /// Validate isotonic calibration slope constraints
    fn validate_slope_constraints(
        &self,
        metrics: &ValidationMetrics,
        violations: &mut Vec<GateViolation>
    ) -> Result<()> {
        for (slice, &slope) in &metrics.slopes_by_slice {
            if slope < self.config.min_slope - self.config.tolerance || 
               slope > self.config.max_slope + self.config.tolerance {
                
                let error_msg = format!("Slope {:.3} outside [{:.1}, {:.1}] for slice {}", 
                                       slope, self.config.min_slope, self.config.max_slope, slice);
                warn!("CALIBRATION CONSTRAINT: {}", error_msg);
                
                violations.push(GateViolation {
                    gate_name: "CALIBRATION".to_string(),
                    metric_name: format!("slope_{}", slice),
                    actual_value: slope,
                    threshold: if slope < self.config.min_slope { self.config.min_slope } else { self.config.max_slope },
                    severity: ViolationSeverity::Warning, // Not critical but logged
                    timestamp: chrono::Utc::now(),
                    error_message: error_msg,
                });
            }
        }
        
        Ok(())
    }

    /// Generate enforcement block for automated testing
    pub fn generate_enforcement_block(&self, metrics: &ValidationMetrics) -> String {
        let max_slice_ece = metrics.ece_by_slice.values()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(1.0);

        format!(r#"
// CRITICAL GATE ENFORCEMENT - AUTO-GENERATED
assert!({:.6} <= 0.020, "FAIL:ECE {:.4}");
assert!({:.1} <= 150.0, "FAIL:P99 {:.1}ms");
assert!({:.3} >= 0.0, "FAIL:SLA-Recall {:.3}");
assert!({:.1} >= 4.0, "FAIL:ΔnDCG {:.1}pp");
"#, 
            max_slice_ece, max_slice_ece,
            metrics.p99_latency_ms, metrics.p99_latency_ms,
            metrics.sla_recall_delta, metrics.sla_recall_delta,
            metrics.ndcg10_delta_nl_pp, metrics.ndcg10_delta_nl_pp
        )
    }

    /// Get violation history for analysis
    pub fn get_violation_history(&self) -> &[GateViolation] {
        &self.violation_history
    }

    /// Reset violation history
    pub fn clear_violation_history(&mut self) {
        self.violation_history.clear();
    }

    /// Check if system is in valid state for deployment
    pub fn is_deployment_ready(&self, metrics: &ValidationMetrics) -> Result<bool> {
        let validation_result = self.clone().validate_gates(metrics)?;
        Ok(validation_result.overall_pass)
    }
}

impl Clone for CalibrationGateEnforcer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            violation_history: self.violation_history.clone(),
        }
    }
}

/// Create gate enforcer with strict TODO.md requirements
pub fn create_strict_enforcer() -> CalibrationGateEnforcer {
    CalibrationGateEnforcer::new(GateConfig::default())
}

/// Utility function to create sample metrics for testing
pub fn create_sample_metrics() -> ValidationMetrics {
    let mut ece_by_slice = HashMap::new();
    ece_by_slice.insert("NL×python".to_string(), 0.018);
    ece_by_slice.insert("NL×typescript".to_string(), 0.019);
    ece_by_slice.insert("identifier×rust".to_string(), 0.015);
    
    let mut slopes_by_slice = HashMap::new();
    slopes_by_slice.insert("NL×python".to_string(), 1.05);
    slopes_by_slice.insert("NL×typescript".to_string(), 0.95);
    slopes_by_slice.insert("identifier×rust".to_string(), 1.02);
    
    ValidationMetrics {
        ece_by_slice,
        ndcg10_delta_nl_pp: 4.6, // Above +4pp requirement
        sla_recall_delta: 0.02,  // No regression
        p99_latency_ms: 145.0,   // Under 150ms SLA
        slopes_by_slice,
        quality_metrics: QualityMetrics {
            core_at_10: 0.85,
            diversity_at_10: 0.72,
            coverage_percentage: 96.5,
            total_queries_processed: 1000,
        },
    }
}

/// Create failing metrics to test gate enforcement
pub fn create_failing_metrics() -> ValidationMetrics {
    let mut ece_by_slice = HashMap::new();
    ece_by_slice.insert("NL×python".to_string(), 0.021); // FAILS: 0.021 > 0.02
    ece_by_slice.insert("NL×typescript".to_string(), 0.019);
    ece_by_slice.insert("identifier×rust".to_string(), 0.015);
    
    let mut slopes_by_slice = HashMap::new();
    slopes_by_slice.insert("NL×python".to_string(), 1.05);
    slopes_by_slice.insert("NL×typescript".to_string(), 0.95);
    slopes_by_slice.insert("identifier×rust".to_string(), 1.02);
    
    ValidationMetrics {
        ece_by_slice,
        ndcg10_delta_nl_pp: 4.6,
        sla_recall_delta: 0.02,
        p99_latency_ms: 145.0,
        slopes_by_slice,
        quality_metrics: QualityMetrics {
            core_at_10: 0.85,
            diversity_at_10: 0.72,
            coverage_percentage: 96.5,
            total_queries_processed: 1000,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passing_gates() {
        let mut enforcer = create_strict_enforcer();
        let metrics = create_sample_metrics();
        
        let result = enforcer.validate_gates(&metrics).unwrap();
        assert!(result.overall_pass);
        assert!(result.calibration_gate.passed);
        assert!(result.sla_gate.passed);
        assert!(result.quality_gate.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_critical_ece_failure() {
        let mut enforcer = create_strict_enforcer();
        let metrics = create_failing_metrics(); // ECE = 0.021 > 0.02
        
        let result = enforcer.validate_gates(&metrics).unwrap();
        assert!(!result.overall_pass);
        assert!(!result.calibration_gate.passed);
        
        let violations: Vec<_> = result.violations.iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::Critical))
            .collect();
        assert!(!violations.is_empty());
        
        let ece_violation = violations.iter()
            .find(|v| v.metric_name == "max_slice_ECE")
            .unwrap();
        assert!(ece_violation.actual_value > 0.02);
    }

    #[test]
    fn test_enforcement_block_generation() {
        let enforcer = create_strict_enforcer();
        let metrics = create_sample_metrics();
        
        let block = enforcer.generate_enforcement_block(&metrics);
        assert!(block.contains("FAIL:ECE"));
        assert!(block.contains("FAIL:P99"));
        assert!(block.contains("FAIL:SLA-Recall"));
        assert!(block.contains("FAIL:ΔnDCG"));
    }

    #[test]
    fn test_deployment_readiness_check() {
        let enforcer = create_strict_enforcer();
        
        let passing_metrics = create_sample_metrics();
        assert!(enforcer.is_deployment_ready(&passing_metrics).unwrap());
        
        let failing_metrics = create_failing_metrics();
        assert!(!enforcer.is_deployment_ready(&failing_metrics).unwrap());
    }
}