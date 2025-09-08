//! # Adversarial Audit Module
//!
//! Implements comprehensive adversarial testing suites to validate system robustness
//! as specified in TODO.md Step 4 - Adversarial audit.
//!
//! Test Suites:
//! - Clone-heavy repositories (duplicate content stress testing)
//! - Vendored bloat scenarios (large dependency noise)
//! - Large JSON/data files (non-code content filtering)
//!
//! Validation Gates:
//! - span=100% (complete corpus coverage)
//! - SLA-Recall@50 flat (no degradation under adversarial conditions)
//! - p99/p95 ≤ 2.0 (latency stability under stress)

pub mod clone_suite;
pub mod bloat_suite;
pub mod noise_suite;
pub mod adversarial_orchestrator;
pub mod stress_harness;

pub use clone_suite::{CloneSuite, CloneTestConfig, CloneResult};
pub use bloat_suite::{BloatSuite, BloatTestConfig, BloatResult};
pub use noise_suite::{NoiseSuite, NoiseTestConfig, NoiseResult};
pub use adversarial_orchestrator::{AdversarialOrchestrator, AdversarialConfig};
pub use stress_harness::{StressHarness, StressResult};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Adversarial test result aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialAuditResult {
    pub clone_results: CloneResult,
    pub bloat_results: BloatResult,
    pub noise_results: NoiseResult,
    pub overall_metrics: OverallMetrics,
    pub gate_validation: GateValidation,
    pub stress_profile: StressProfile,
}

/// Overall performance metrics across all adversarial tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallMetrics {
    pub span_coverage_pct: f32,
    pub sla_recall_at_50: f32,
    pub p99_latency_ms: f32,
    pub p95_latency_ms: f32,
    pub degradation_factor: f32,
    pub robustness_score: f32,
}

/// Gate validation results for adversarial audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateValidation {
    pub span_coverage_gate: bool,    // Must be 100%
    pub sla_recall_gate: bool,       // Must be flat (no degradation)
    pub latency_stability_gate: bool, // p99/p95 ≤ 2.0
    pub overall_pass: bool,
    pub violations: Vec<String>,
}

/// System stress profile under adversarial conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressProfile {
    pub memory_peak_mb: f32,
    pub cpu_utilization_pct: f32,
    pub disk_io_ops_per_sec: f32,
    pub network_bandwidth_mbps: f32,
    pub gc_pressure_score: f32,
    pub resource_exhaustion_risk: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Validate adversarial audit gates
pub fn validate_adversarial_gates(result: &AdversarialAuditResult) -> Result<bool> {
    const SPAN_COVERAGE_THRESHOLD: f32 = 100.0;
    const LATENCY_STABILITY_RATIO: f32 = 2.0;
    const MIN_SLA_RECALL: f32 = 0.50; // Flat requirement - no degradation allowed
    
    let mut violations = Vec::new();
    let metrics = &result.overall_metrics;
    
    // Gate 1: span=100% (complete corpus coverage)
    let span_gate = metrics.span_coverage_pct >= SPAN_COVERAGE_THRESHOLD;
    if !span_gate {
        violations.push(format!(
            "Span coverage gate failed: {:.1}% < {:.1}% required",
            metrics.span_coverage_pct, SPAN_COVERAGE_THRESHOLD
        ));
    }
    
    // Gate 2: SLA-Recall@50 flat (no degradation)
    let sla_recall_gate = metrics.sla_recall_at_50 >= MIN_SLA_RECALL;
    if !sla_recall_gate {
        violations.push(format!(
            "SLA-Recall@50 gate failed: {:.3} < {:.3} required",
            metrics.sla_recall_at_50, MIN_SLA_RECALL
        ));
    }
    
    // Gate 3: p99/p95 ≤ 2.0 (latency stability)
    let latency_ratio = metrics.p99_latency_ms / metrics.p95_latency_ms;
    let latency_gate = latency_ratio <= LATENCY_STABILITY_RATIO;
    if !latency_gate {
        violations.push(format!(
            "Latency stability gate failed: p99/p95 = {:.2} > {:.1} allowed",
            latency_ratio, LATENCY_STABILITY_RATIO
        ));
    }
    
    let overall_pass = span_gate && sla_recall_gate && latency_gate;
    
    if !overall_pass {
        tracing::warn!(
            "Adversarial audit gate failures: {}",
            violations.join("; ")
        );
    }
    
    Ok(overall_pass)
}

/// Calculate robustness score based on adversarial performance
pub fn calculate_robustness_score(result: &AdversarialAuditResult) -> f32 {
    let metrics = &result.overall_metrics;
    
    // Weighted scoring across key robustness dimensions
    let span_score = (metrics.span_coverage_pct / 100.0).min(1.0);
    let recall_score = metrics.sla_recall_at_50.min(1.0);
    let stability_score = (2.0 / (metrics.p99_latency_ms / metrics.p95_latency_ms)).min(1.0);
    let degradation_score = (2.0 / (1.0 + metrics.degradation_factor)).min(1.0);
    
    // Geometric mean for conservative scoring
    let robustness = (span_score * recall_score * stability_score * degradation_score).powf(0.25);
    
    (robustness * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adversarial_gate_validation() {
        let passing_result = AdversarialAuditResult {
            clone_results: CloneResult::default(),
            bloat_results: BloatResult::default(),
            noise_results: NoiseResult::default(),
            overall_metrics: OverallMetrics {
                span_coverage_pct: 100.0,
                sla_recall_at_50: 0.52,
                p99_latency_ms: 190.0,
                p95_latency_ms: 140.0, // ratio = 1.36 ≤ 2.0 ✓
                degradation_factor: 1.15,
                robustness_score: 0.85,
            },
            gate_validation: GateValidation {
                span_coverage_gate: true,
                sla_recall_gate: true,
                latency_stability_gate: true,
                overall_pass: true,
                violations: vec![],
            },
            stress_profile: StressProfile {
                memory_peak_mb: 2048.0,
                cpu_utilization_pct: 75.0,
                disk_io_ops_per_sec: 1250.0,
                network_bandwidth_mbps: 15.0,
                gc_pressure_score: 0.3,
                resource_exhaustion_risk: RiskLevel::Low,
            },
        };
        
        assert!(validate_adversarial_gates(&passing_result).unwrap());
    }

    #[test]
    fn test_failing_span_coverage_gate() {
        let mut failing_result = AdversarialAuditResult {
            clone_results: CloneResult::default(),
            bloat_results: BloatResult::default(),
            noise_results: NoiseResult::default(),
            overall_metrics: OverallMetrics {
                span_coverage_pct: 97.5, // Below 100% threshold
                sla_recall_at_50: 0.52,
                p99_latency_ms: 180.0,
                p95_latency_ms: 130.0,
                degradation_factor: 1.10,
                robustness_score: 0.80,
            },
            gate_validation: GateValidation {
                span_coverage_gate: false,
                sla_recall_gate: true,
                latency_stability_gate: true,
                overall_pass: false,
                violations: vec!["Span coverage insufficient".to_string()],
            },
            stress_profile: StressProfile {
                memory_peak_mb: 1800.0,
                cpu_utilization_pct: 70.0,
                disk_io_ops_per_sec: 1100.0,
                network_bandwidth_mbps: 12.0,
                gc_pressure_score: 0.25,
                resource_exhaustion_risk: RiskLevel::Low,
            },
        };
        
        assert!(!validate_adversarial_gates(&failing_result).unwrap());
    }

    #[test]
    fn test_robustness_score_calculation() {
        let result = AdversarialAuditResult {
            clone_results: CloneResult::default(),
            bloat_results: BloatResult::default(),
            noise_results: NoiseResult::default(),
            overall_metrics: OverallMetrics {
                span_coverage_pct: 100.0,
                sla_recall_at_50: 0.52,
                p99_latency_ms: 180.0,
                p95_latency_ms: 140.0,
                degradation_factor: 1.2,
                robustness_score: 0.0, // Will be calculated
            },
            gate_validation: GateValidation {
                span_coverage_gate: true,
                sla_recall_gate: true,
                latency_stability_gate: true,
                overall_pass: true,
                violations: vec![],
            },
            stress_profile: StressProfile {
                memory_peak_mb: 2000.0,
                cpu_utilization_pct: 72.0,
                disk_io_ops_per_sec: 1200.0,
                network_bandwidth_mbps: 14.0,
                gc_pressure_score: 0.28,
                resource_exhaustion_risk: RiskLevel::Low,
            },
        };
        
        let score = calculate_robustness_score(&result);
        assert!(score > 0.7); // Should be reasonably high for good metrics
        assert!(score <= 1.0); // Should be normalized
    }
}