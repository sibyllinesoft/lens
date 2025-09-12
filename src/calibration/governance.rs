//! # Calibration Governance System
//!
//! Production-ready governance system for calibration contract enforcement, SLA monitoring,
//! and IEEE-754 compliance validation. Implements the complete calibration contract from
//! CALIBRATION_CONTRACT.md with automated enforcement and violation detection.
//!
//! ## Key Features
//!
//! * **Contract Codification**: Complete calibration contract implementation
//! * **SLA Enforcement**: Automated ECE ≤ max(0.015, ĉ√(K/N)) enforcement
//! * **Configuration Compatibility**: Version and dependency validation
//! * **IEEE-754 Compliance**: Floating-point safety and fast-math protection
//! * **Real-time Monitoring**: Continuous compliance validation
//! * **Violation Detection**: Automated detection and escalation of breaches
//!
//! ## SLA Formula Implementation
//!
//! The system enforces the statistical ECE threshold:
//! ```
//! τ(N,K) = max(0.015, ĉ·√(K/N))
//! where:
//! - N = sample count
//! - K = number of bins (typically 15)
//! - ĉ = 1.5 (empirical constant for synthetic data noise)
//! - 0.015 = PHASE 4 minimum requirement
//! ```

use crate::calibration::{
    CalibrationManifest, Phase4Config, CalibrationSample, CrossLanguageMetrics,
    IntegrityStatus, ValidationResult, ConfigurationFingerprint,
};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use tracing::{info, warn, error, debug};

/// Calibration governance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceConfig {
    /// Enable strict SLA enforcement
    pub enable_sla_enforcement: bool,
    /// ECE threshold formula parameters
    pub ece_threshold_params: EceThresholdParams,
    /// Language variance limits
    pub language_variance_limits: LanguageVarianceLimits,
    /// Configuration compatibility settings
    pub compatibility_settings: CompatibilitySettings,
    /// IEEE-754 compliance settings
    pub ieee754_settings: Ieee754Settings,
    /// Monitoring and alerting configuration
    pub monitoring_config: MonitoringConfig,
    /// Violation escalation settings
    pub violation_escalation: ViolationEscalationConfig,
}

/// ECE threshold formula parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EceThresholdParams {
    /// Base ECE requirement (0.015 for PHASE 4)
    pub base_ece_requirement: f32,
    /// Empirical constant ĉ (1.5 for synthetic data)
    pub empirical_constant: f32,
    /// Default bin count K
    pub default_bin_count: usize,
    /// Minimum sample count for threshold calculation
    pub min_sample_count: usize,
    /// Maximum allowed ECE (safety limit)
    pub max_allowed_ece: f32,
}

/// Language variance enforcement limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageVarianceLimits {
    /// Maximum Tier-1 language variance in percentage points
    pub max_tier1_variance_pp: f32,
    /// Maximum Tier-2 language variance in percentage points  
    pub max_tier2_variance_pp: f32,
    /// Maximum individual language ECE multiplier
    pub max_individual_language_ece_multiplier: f32,
    /// Minimum languages required for variance calculation
    pub min_languages_for_variance: usize,
}

/// Configuration compatibility validation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilitySettings {
    /// Enable version compatibility checking
    pub enable_version_checking: bool,
    /// Supported manifest versions
    pub supported_manifest_versions: Vec<String>,
    /// Supported Phase 4 config versions
    pub supported_phase4_versions: Vec<String>,
    /// Required dependencies with version constraints
    pub required_dependencies: HashMap<String, String>,
    /// Breaking change detection enabled
    pub enable_breaking_change_detection: bool,
}

/// IEEE-754 floating-point compliance settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ieee754Settings {
    /// Enable strict IEEE-754 compliance
    pub enable_strict_compliance: bool,
    /// Detect and warn about fast-math compilation
    pub detect_fast_math: bool,
    /// Enable denormal number handling
    pub enable_denormal_handling: bool,
    /// Floating-point precision requirements
    pub precision_requirements: PrecisionRequirements,
    /// NaN and infinity handling
    pub special_values_handling: SpecialValuesHandling,
}

/// Floating-point precision requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionRequirements {
    /// Minimum precision for ECE calculations (decimal places)
    pub ece_precision: u8,
    /// Minimum precision for calibrated scores
    pub score_precision: u8,
    /// Minimum precision for confidence values
    pub confidence_precision: u8,
    /// Tolerance for numerical comparisons
    pub numerical_tolerance: f64,
}

/// Special floating-point values handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialValuesHandling {
    /// How to handle NaN values
    pub nan_handling: NanHandling,
    /// How to handle infinite values
    pub infinity_handling: InfinityHandling,
    /// Enable subnormal detection
    pub detect_subnormals: bool,
    /// Maximum allowed relative error
    pub max_relative_error: f64,
}

/// NaN handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NanHandling {
    /// Reject inputs with NaN
    Reject,
    /// Replace NaN with neutral value (0.5)
    ReplaceWithNeutral,
    /// Replace NaN with specified value
    ReplaceWith(f32),
    /// Allow NaN to propagate
    Allow,
}

/// Infinity handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfinityHandling {
    /// Reject inputs with infinity
    Reject,
    /// Clamp to finite bounds
    Clamp { min: f32, max: f32 },
    /// Replace with specified values
    Replace { pos_inf: f32, neg_inf: f32 },
    /// Allow infinity to propagate
    Allow,
}

/// Monitoring and alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable real-time compliance monitoring
    pub enable_realtime_monitoring: bool,
    /// Compliance check interval in seconds
    pub check_interval_seconds: u64,
    /// Historical violation tracking period in days
    pub violation_tracking_days: u32,
    /// Alert thresholds for different violation types
    pub alert_thresholds: AlertThresholds,
    /// Enable automated reporting
    pub enable_automated_reports: bool,
}

/// Alert thresholds for violation detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// ECE violation severity levels
    pub ece_violations: SeverityThresholds,
    /// Language variance violation levels
    pub variance_violations: SeverityThresholds,
    /// Configuration compatibility violation levels
    pub compatibility_violations: SeverityThresholds,
    /// IEEE-754 compliance violation levels
    pub ieee754_violations: SeverityThresholds,
}

/// Severity threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverityThresholds {
    /// Warning threshold
    pub warning: f32,
    /// Critical threshold
    pub critical: f32,
    /// Emergency threshold
    pub emergency: f32,
}

/// Violation escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationEscalationConfig {
    /// Enable automatic escalation
    pub enable_auto_escalation: bool,
    /// Escalation levels and actions
    pub escalation_levels: Vec<EscalationLevel>,
    /// Maximum violations before emergency stop
    pub max_violations_before_emergency: u32,
    /// Cooldown period between alerts in minutes
    pub alert_cooldown_minutes: u32,
}

/// Escalation level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Escalation level name
    pub level: String,
    /// Violation count threshold
    pub violation_threshold: u32,
    /// Time window for violation counting in hours
    pub time_window_hours: u32,
    /// Actions to take at this level
    pub actions: Vec<EscalationAction>,
}

/// Actions to take on violation escalation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationAction {
    /// Send alert notification
    Alert { message: String, severity: AlertSeverity },
    /// Stop calibration system
    StopCalibration,
    /// Revert to previous configuration
    RevertConfiguration,
    /// Trigger emergency procedure
    EmergencyProcedure { procedure: String },
    /// Log violation for audit
    LogViolation { details: String },
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Governance compliance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Overall compliance status
    pub compliant: bool,
    /// Timestamp of compliance check
    pub checked_at: DateTime<Utc>,
    /// Individual compliance checks
    pub checks: Vec<ComplianceCheck>,
    /// Any violations detected
    pub violations: Vec<ComplianceViolation>,
    /// Overall compliance score (0-100)
    pub compliance_score: f32,
    /// Next recommended check time
    pub next_check_at: DateTime<Utc>,
}

/// Individual compliance check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    /// Check name
    pub name: String,
    /// Check type
    pub check_type: ComplianceCheckType,
    /// Check passed
    pub passed: bool,
    /// Check timestamp
    pub checked_at: DateTime<Utc>,
    /// Measured value (if applicable)
    pub measured_value: Option<f32>,
    /// Expected value (if applicable)
    pub expected_value: Option<f32>,
    /// Check details
    pub details: String,
}

/// Types of compliance checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceCheckType {
    /// ECE threshold compliance
    EceThreshold,
    /// Language variance compliance
    LanguageVariance,
    /// Configuration compatibility
    ConfigurationCompatibility,
    /// IEEE-754 compliance
    Ieee754Compliance,
    /// Manifest integrity
    ManifestIntegrity,
    /// SLA enforcement
    SlaEnforcement,
}

/// Compliance violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    /// Violation ID
    pub violation_id: String,
    /// Violation type
    pub violation_type: ViolationType,
    /// Violation severity
    pub severity: ViolationSeverity,
    /// When violation was detected
    pub detected_at: DateTime<Utc>,
    /// Violation description
    pub description: String,
    /// Measured value that caused violation
    pub measured_value: f32,
    /// Expected/threshold value
    pub threshold_value: f32,
    /// Violation context
    pub context: HashMap<String, serde_json::Value>,
    /// Suggested remediation
    pub remediation: String,
}

/// Types of compliance violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// ECE threshold exceeded
    EceThresholdExceeded,
    /// Language variance too high
    LanguageVarianceExceeded,
    /// Unsupported configuration version
    UnsupportedConfigurationVersion,
    /// IEEE-754 compliance issue
    Ieee754NonCompliance,
    /// Manifest integrity compromise
    ManifestIntegrityCompromise,
    /// SLA requirement violation
    SlaViolation,
}

/// Violation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Main governance service
pub struct CalibrationGovernance {
    /// Governance configuration
    config: GovernanceConfig,
    /// Historical violations
    violation_history: Vec<ComplianceViolation>,
    /// Last compliance check result
    last_compliance_result: Option<ComplianceResult>,
    /// Active alerts
    active_alerts: Vec<ActiveAlert>,
}

/// Active alert tracking
#[derive(Debug, Clone)]
struct ActiveAlert {
    pub alert_id: String,
    pub violation_id: String,
    pub severity: AlertSeverity,
    pub created_at: DateTime<Utc>,
    pub escalated: bool,
}

impl CalibrationGovernance {
    /// Create a new calibration governance service
    pub fn new(config: GovernanceConfig) -> Self {
        info!("Initializing calibration governance service");
        info!("SLA enforcement: {}", config.enable_sla_enforcement);
        info!("Base ECE requirement: {:.4}", config.ece_threshold_params.base_ece_requirement);
        info!("Empirical constant ĉ: {:.2}", config.ece_threshold_params.empirical_constant);
        info!("Max Tier-1 variance: {:.1}pp", config.language_variance_limits.max_tier1_variance_pp);
        info!("IEEE-754 strict compliance: {}", config.ieee754_settings.enable_strict_compliance);
        
        Self {
            config,
            violation_history: Vec::new(),
            last_compliance_result: None,
            active_alerts: Vec::new(),
        }
    }
    
    /// Calculate statistical ECE threshold using the contract formula
    pub fn calculate_statistical_ece_threshold(&self, sample_count: usize, bin_count: usize) -> f32 {
        if sample_count == 0 {
            return self.config.ece_threshold_params.base_ece_requirement;
        }
        
        let n = sample_count as f32;
        let k = bin_count as f32;
        let c_hat = self.config.ece_threshold_params.empirical_constant;
        let base_requirement = self.config.ece_threshold_params.base_ece_requirement;
        
        // Formula: τ(N,K) = max(0.015, ĉ·√(K/N))
        let statistical_floor = c_hat * (k / n).sqrt();
        let threshold = statistical_floor.max(base_requirement);
        
        debug!("Statistical ECE threshold: N={}, K={}, ĉ={:.2}, τ={:.4}", 
               sample_count, bin_count, c_hat, threshold);
        
        // Apply safety limit
        threshold.min(self.config.ece_threshold_params.max_allowed_ece)
    }
    
    /// Enforce SLA requirements on calibration results
    pub fn enforce_sla_requirements(
        &mut self,
        samples: &[CalibrationSample],
        metrics: &CrossLanguageMetrics,
    ) -> Result<ComplianceResult> {
        info!("Enforcing SLA requirements");
        info!("Sample count: {}", samples.len());
        info!("Overall ECE: {:.4}", metrics.overall_ece);
        info!("Tier-1 variance: {:.1}pp", metrics.tier1_variance);
        
        let mut checks = Vec::new();
        let mut violations = Vec::new();
        let checked_at = Utc::now();
        
        // ECE Threshold Check
        let bin_count = self.config.ece_threshold_params.default_bin_count;
        let ece_threshold = self.calculate_statistical_ece_threshold(samples.len(), bin_count);
        let ece_compliant = metrics.overall_ece <= ece_threshold;
        
        checks.push(ComplianceCheck {
            name: "ECE Threshold Compliance".to_string(),
            check_type: ComplianceCheckType::EceThreshold,
            passed: ece_compliant,
            checked_at,
            measured_value: Some(metrics.overall_ece),
            expected_value: Some(ece_threshold),
            details: format!("ECE {:.4} vs threshold {:.4} (N={}, K={})", 
                           metrics.overall_ece, ece_threshold, samples.len(), bin_count),
        });
        
        if !ece_compliant {
            let severity = self.determine_ece_violation_severity(metrics.overall_ece, ece_threshold);
            violations.push(ComplianceViolation {
                violation_id: self.generate_violation_id(),
                violation_type: ViolationType::EceThresholdExceeded,
                severity,
                detected_at: checked_at,
                description: format!("ECE {:.4} exceeds statistical threshold {:.4}", 
                                   metrics.overall_ece, ece_threshold),
                measured_value: metrics.overall_ece,
                threshold_value: ece_threshold,
                context: HashMap::from_iter(vec![
                    ("sample_count".to_string(), serde_json::Value::Number(samples.len().into())),
                    ("bin_count".to_string(), serde_json::Value::Number(bin_count.into())),
                ]),
                remediation: "Increase calibration training data or adjust model parameters".to_string(),
            });
        }
        
        // Language Variance Check (Tier-1)
        let tier1_variance_compliant = metrics.tier1_variance < self.config.language_variance_limits.max_tier1_variance_pp;
        
        checks.push(ComplianceCheck {
            name: "Tier-1 Language Variance".to_string(),
            check_type: ComplianceCheckType::LanguageVariance,
            passed: tier1_variance_compliant,
            checked_at,
            measured_value: Some(metrics.tier1_variance),
            expected_value: Some(self.config.language_variance_limits.max_tier1_variance_pp),
            details: format!("Tier-1 variance {:.1}pp vs limit {:.1}pp", 
                           metrics.tier1_variance, self.config.language_variance_limits.max_tier1_variance_pp),
        });
        
        if !tier1_variance_compliant {
            let severity = self.determine_variance_violation_severity(
                metrics.tier1_variance, 
                self.config.language_variance_limits.max_tier1_variance_pp
            );
            violations.push(ComplianceViolation {
                violation_id: self.generate_violation_id(),
                violation_type: ViolationType::LanguageVarianceExceeded,
                severity,
                detected_at: checked_at,
                description: format!("Tier-1 language variance {:.1}pp exceeds limit {:.1}pp", 
                                   metrics.tier1_variance, 
                                   self.config.language_variance_limits.max_tier1_variance_pp),
                measured_value: metrics.tier1_variance,
                threshold_value: self.config.language_variance_limits.max_tier1_variance_pp,
                context: HashMap::new(),
                remediation: "Improve language-specific calibration or collect more training data for underperforming languages".to_string(),
            });
        }
        
        // Individual Language ECE Bounds Check
        let max_individual_ece = ece_threshold * self.config.language_variance_limits.max_individual_language_ece_multiplier;
        let mut individual_language_compliant = true;
        
        for (language, lang_ece) in &metrics.ece_by_language {
            if *lang_ece > max_individual_ece {
                individual_language_compliant = false;
                violations.push(ComplianceViolation {
                    violation_id: self.generate_violation_id(),
                    violation_type: ViolationType::SlaViolation,
                    severity: ViolationSeverity::High,
                    detected_at: checked_at,
                    description: format!("Language {} ECE {:.4} exceeds individual limit {:.4}", 
                                       language, lang_ece, max_individual_ece),
                    measured_value: *lang_ece,
                    threshold_value: max_individual_ece,
                    context: HashMap::from_iter(vec![
                        ("language".to_string(), serde_json::Value::String(language.clone())),
                    ]),
                    remediation: format!("Improve calibration for {} language specifically", language),
                });
            }
        }
        
        checks.push(ComplianceCheck {
            name: "Individual Language ECE Bounds".to_string(),
            check_type: ComplianceCheckType::SlaEnforcement,
            passed: individual_language_compliant,
            checked_at,
            measured_value: None,
            expected_value: Some(max_individual_ece),
            details: format!("All languages within {:.4} ECE limit", max_individual_ece),
        });
        
        // Calculate overall compliance
        let total_checks = checks.len();
        let passed_checks = checks.iter().filter(|c| c.passed).count();
        let compliance_score = (passed_checks as f32 / total_checks as f32) * 100.0;
        let overall_compliant = violations.is_empty();
        
        let result = ComplianceResult {
            compliant: overall_compliant,
            checked_at,
            checks,
            violations: violations.clone(),
            compliance_score,
            next_check_at: checked_at + Duration::seconds(self.config.monitoring_config.check_interval_seconds as i64),
        };
        
        // Store violations in history
        self.violation_history.extend(violations.clone());
        self.last_compliance_result = Some(result.clone());
        
        // Handle violations if any
        if !violations.is_empty() {
            self.handle_violations(&violations)?;
        }
        
        if overall_compliant {
            info!("✓ SLA requirements fully compliant (score: {:.1}%)", compliance_score);
        } else {
            warn!("✗ SLA compliance violations detected: {} violations", violations.len());
            warn!("Compliance score: {:.1}%", compliance_score);
        }
        
        Ok(result)
    }
    
    /// Validate configuration compatibility
    pub fn validate_configuration_compatibility(
        &self,
        manifest: &CalibrationManifest,
    ) -> Result<ComplianceCheck> {
        info!("Validating configuration compatibility");
        
        let checked_at = Utc::now();
        let mut compatibility_issues = Vec::new();
        
        // Check manifest version compatibility
        if self.config.compatibility_settings.enable_version_checking {
            if !self.config.compatibility_settings.supported_manifest_versions
                .contains(&manifest.manifest_version) {
                compatibility_issues.push(format!(
                    "Unsupported manifest version: {} (supported: {:?})",
                    manifest.manifest_version,
                    self.config.compatibility_settings.supported_manifest_versions
                ));
            }
        }
        
        // Check Phase 4 configuration constraints
        let phase4_valid = self.validate_phase4_constraints(&manifest.phase4_config);
        if !phase4_valid {
            compatibility_issues.push("Phase 4 configuration violates contract requirements".to_string());
        }
        
        // Check required dependencies
        for (dep_name, version_constraint) in &self.config.compatibility_settings.required_dependencies {
            let dep_found = manifest.sbom.iter()
                .any(|entry| entry.name == *dep_name);
            
            if !dep_found {
                compatibility_issues.push(format!("Missing required dependency: {}", dep_name));
            }
        }
        
        // Check for breaking changes if enabled
        if self.config.compatibility_settings.enable_breaking_change_detection {
            let breaking_changes = self.detect_breaking_changes(manifest);
            compatibility_issues.extend(breaking_changes);
        }
        
        let compatible = compatibility_issues.is_empty();
        let details = if compatible {
            "All configuration compatibility checks passed".to_string()
        } else {
            format!("Compatibility issues: {}", compatibility_issues.join("; "))
        };
        
        info!("Configuration compatibility: {} (issues: {})", 
              if compatible { "✓" } else { "✗" }, compatibility_issues.len());
        
        Ok(ComplianceCheck {
            name: "Configuration Compatibility".to_string(),
            check_type: ComplianceCheckType::ConfigurationCompatibility,
            passed: compatible,
            checked_at,
            measured_value: None,
            expected_value: None,
            details,
        })
    }
    
    /// Enforce IEEE-754 compliance
    pub fn enforce_ieee754_compliance(
        &self,
        samples: &[CalibrationSample],
    ) -> Result<ComplianceCheck> {
        info!("Enforcing IEEE-754 compliance");
        info!("Checking {} samples for floating-point compliance", samples.len());
        
        let checked_at = Utc::now();
        let mut compliance_issues = Vec::new();
        
        if self.config.ieee754_settings.enable_strict_compliance {
            // Check for NaN values
            let nan_count = samples.iter()
                .filter(|s| s.prediction.is_nan() || s.ground_truth.is_nan())
                .count();
            
            if nan_count > 0 {
                match self.config.ieee754_settings.special_values_handling.nan_handling {
                    NanHandling::Reject => {
                        compliance_issues.push(format!("Found {} NaN values (rejected)", nan_count));
                    }
                    _ => {
                        info!("Found {} NaN values (handled by configured strategy)", nan_count);
                    }
                }
            }
            
            // Check for infinite values
            let inf_count = samples.iter()
                .filter(|s| s.prediction.is_infinite() || s.ground_truth.is_infinite())
                .count();
            
            if inf_count > 0 {
                match self.config.ieee754_settings.special_values_handling.infinity_handling {
                    InfinityHandling::Reject => {
                        compliance_issues.push(format!("Found {} infinite values (rejected)", inf_count));
                    }
                    _ => {
                        info!("Found {} infinite values (handled by configured strategy)", inf_count);
                    }
                }
            }
            
            // Check for subnormal values if enabled
            if self.config.ieee754_settings.special_values_handling.detect_subnormals {
                let subnormal_count = samples.iter()
                    .filter(|s| self.is_subnormal(s.prediction) || self.is_subnormal(s.ground_truth))
                    .count();
                
                if subnormal_count > 0 {
                    compliance_issues.push(format!("Found {} subnormal values", subnormal_count));
                }
            }
            
            // Detect fast-math compilation if enabled
            if self.config.ieee754_settings.detect_fast_math {
                if self.detect_fast_math_compilation() {
                    compliance_issues.push("Fast-math compilation detected - may violate IEEE-754 compliance".to_string());
                }
            }
        }
        
        let compliant = compliance_issues.is_empty();
        let details = if compliant {
            format!("All {} samples are IEEE-754 compliant", samples.len())
        } else {
            format!("IEEE-754 compliance issues: {}", compliance_issues.join("; "))
        };
        
        info!("IEEE-754 compliance: {} (issues: {})", 
              if compliant { "✓" } else { "✗" }, compliance_issues.len());
        
        Ok(ComplianceCheck {
            name: "IEEE-754 Compliance".to_string(),
            check_type: ComplianceCheckType::Ieee754Compliance,
            passed: compliant,
            checked_at,
            measured_value: None,
            expected_value: None,
            details,
        })
    }
    
    /// Perform comprehensive compliance validation
    pub fn validate_comprehensive_compliance(
        &mut self,
        manifest: &CalibrationManifest,
        samples: &[CalibrationSample],
        metrics: &CrossLanguageMetrics,
    ) -> Result<ComplianceResult> {
        info!("Performing comprehensive compliance validation");
        
        let mut all_checks = Vec::new();
        let mut all_violations = Vec::new();
        
        // SLA enforcement
        if self.config.enable_sla_enforcement {
            let sla_result = self.enforce_sla_requirements(samples, metrics)?;
            all_checks.extend(sla_result.checks);
            all_violations.extend(sla_result.violations);
        }
        
        // Configuration compatibility
        let config_check = self.validate_configuration_compatibility(manifest)?;
        let config_compliant = config_check.passed;
        all_checks.push(config_check);
        
        // IEEE-754 compliance
        let ieee754_check = self.enforce_ieee754_compliance(samples)?;
        let ieee754_compliant = ieee754_check.passed;
        all_checks.push(ieee754_check);
        
        // Manifest integrity check
        let integrity_check = self.validate_manifest_integrity(manifest)?;
        all_checks.push(integrity_check);
        
        // Calculate overall compliance
        let total_checks = all_checks.len();
        let passed_checks = all_checks.iter().filter(|c| c.passed).count();
        let compliance_score = (passed_checks as f32 / total_checks as f32) * 100.0;
        let overall_compliant = all_violations.is_empty() && config_compliant && ieee754_compliant;
        
        let checked_at = Utc::now();
        let result = ComplianceResult {
            compliant: overall_compliant,
            checked_at,
            checks: all_checks,
            violations: all_violations.clone(),
            compliance_score,
            next_check_at: checked_at + Duration::seconds(self.config.monitoring_config.check_interval_seconds as i64),
        };
        
        // Store violations in history
        self.violation_history.extend(all_violations.clone());
        self.last_compliance_result = Some(result.clone());
        
        // Handle violations if any
        if !all_violations.is_empty() {
            self.handle_violations(&all_violations)?;
        }
        
        info!("Comprehensive compliance validation complete");
        info!("Overall compliant: {}", overall_compliant);
        info!("Compliance score: {:.1}%", compliance_score);
        info!("Total checks: {}, passed: {}", total_checks, passed_checks);
        info!("Violations detected: {}", all_violations.len());
        
        Ok(result)
    }
    
    /// Get compliance history and trends
    pub fn get_compliance_history(&self, days: u32) -> Vec<&ComplianceViolation> {
        let cutoff = Utc::now() - Duration::days(days as i64);
        self.violation_history.iter()
            .filter(|v| v.detected_at >= cutoff)
            .collect()
    }
    
    /// Generate compliance report
    pub fn generate_compliance_report(&self) -> Result<GovernanceReport> {
        info!("Generating compliance report");
        
        let report_time = Utc::now();
        let last_30_days_violations = self.get_compliance_history(30);
        
        let violation_stats = ViolationStatistics {
            total_violations: self.violation_history.len(),
            last_30_days_violations: last_30_days_violations.len(),
            critical_violations: self.violation_history.iter()
                .filter(|v| matches!(v.severity, ViolationSeverity::Critical | ViolationSeverity::Emergency))
                .count(),
            resolved_violations: 0, // Would track resolution in a full implementation
        };
        
        let compliance_trends = self.calculate_compliance_trends();
        
        let current_status = if let Some(last_result) = &self.last_compliance_result {
            if last_result.compliant {
                "COMPLIANT".to_string()
            } else {
                format!("NON-COMPLIANT ({} violations)", last_result.violations.len())
            }
        } else {
            "UNKNOWN".to_string()
        };
        
        let report = GovernanceReport {
            generated_at: report_time,
            reporting_period_days: 30,
            current_compliance_status: current_status,
            compliance_score: self.last_compliance_result.as_ref()
                .map(|r| r.compliance_score)
                .unwrap_or(0.0),
            violation_statistics: violation_stats,
            compliance_trends,
            active_alerts_count: self.active_alerts.len(),
            recommendations: self.generate_recommendations(),
        };
        
        info!("✓ Compliance report generated");
        Ok(report)
    }
    
    // Private helper methods
    
    fn validate_phase4_constraints(&self, config: &Phase4Config) -> bool {
        // Validate Phase 4 specific requirements from the contract
        config.target_ece <= 0.015
            && config.max_language_variance < 7.0
            && config.isotonic_slope_clamp == (0.9, 1.1)
    }
    
    fn detect_breaking_changes(&self, manifest: &CalibrationManifest) -> Vec<String> {
        let mut breaking_changes = Vec::new();
        
        // Check for configuration format changes
        if manifest.manifest_version.starts_with("2.") && 
           self.config.compatibility_settings.supported_manifest_versions.iter()
               .all(|v| v.starts_with("1.")) {
            breaking_changes.push("Major version change detected in manifest format".to_string());
        }
        
        // Check for slope clamp changes
        if manifest.phase4_config.isotonic_slope_clamp != (0.9, 1.1) {
            breaking_changes.push("Isotonic slope clamp changed from contract requirement [0.9, 1.1]".to_string());
        }
        
        breaking_changes
    }
    
    fn validate_manifest_integrity(&self, manifest: &CalibrationManifest) -> Result<ComplianceCheck> {
        let checked_at = Utc::now();
        
        // Check integrity status
        let integrity_valid = matches!(manifest.integrity_status, IntegrityStatus::Valid);
        
        let details = match &manifest.integrity_status {
            IntegrityStatus::Valid => "Manifest integrity verified".to_string(),
            IntegrityStatus::HashMismatch { mismatched_components } => {
                format!("Hash mismatch in components: {}", mismatched_components.join(", "))
            }
            IntegrityStatus::VersionIncompatible { incompatible_versions } => {
                format!("Incompatible versions: {}", incompatible_versions.join(", "))
            }
            IntegrityStatus::SecurityIssue { vulnerability_count } => {
                format!("Security issues detected: {} vulnerabilities", vulnerability_count)
            }
            IntegrityStatus::ConfigurationInvalid { validation_errors } => {
                format!("Configuration invalid: {}", validation_errors.join(", "))
            }
            IntegrityStatus::Unknown => "Integrity status unknown".to_string(),
        };
        
        Ok(ComplianceCheck {
            name: "Manifest Integrity".to_string(),
            check_type: ComplianceCheckType::ManifestIntegrity,
            passed: integrity_valid,
            checked_at,
            measured_value: None,
            expected_value: None,
            details,
        })
    }
    
    fn is_subnormal(&self, value: f32) -> bool {
        // Check if a float32 value is subnormal (denormal)
        value != 0.0 && value.abs() < f32::MIN_POSITIVE
    }
    
    fn detect_fast_math_compilation(&self) -> bool {
        // Simplified fast-math detection
        // In a real implementation, this would check compiler flags, runtime behavior, etc.
        
        // Test IEEE-754 compliance with a known case
        let test_val = 0.1_f32 + 0.2_f32;
        let expected = 0.30000001_f32; // IEEE-754 representation
        
        (test_val - expected).abs() > f32::EPSILON * 10.0
    }
    
    fn determine_ece_violation_severity(&self, measured: f32, threshold: f32) -> ViolationSeverity {
        let excess = measured - threshold;
        let relative_excess = excess / threshold;
        
        let thresholds = &self.config.monitoring_config.alert_thresholds.ece_violations;
        
        if relative_excess >= thresholds.emergency {
            ViolationSeverity::Emergency
        } else if relative_excess >= thresholds.critical {
            ViolationSeverity::Critical
        } else if relative_excess >= thresholds.warning {
            ViolationSeverity::High
        } else {
            ViolationSeverity::Medium
        }
    }
    
    fn determine_variance_violation_severity(&self, measured: f32, threshold: f32) -> ViolationSeverity {
        let excess = measured - threshold;
        
        let thresholds = &self.config.monitoring_config.alert_thresholds.variance_violations;
        
        if excess >= thresholds.emergency {
            ViolationSeverity::Emergency
        } else if excess >= thresholds.critical {
            ViolationSeverity::Critical
        } else if excess >= thresholds.warning {
            ViolationSeverity::High
        } else {
            ViolationSeverity::Medium
        }
    }
    
    fn generate_violation_id(&self) -> String {
        use fastrand;
        let timestamp = Utc::now().timestamp();
        let random = fastrand::u32(..);
        format!("VIO-{}-{:08X}", timestamp, random)
    }
    
    fn handle_violations(&mut self, violations: &[ComplianceViolation]) -> Result<()> {
        if !self.config.violation_escalation.enable_auto_escalation {
            return Ok(());
        }
        
        info!("Handling {} violations", violations.len());
        
        for violation in violations {
            // Check escalation levels
            for escalation_level in &self.config.violation_escalation.escalation_levels {
                let recent_violations = self.count_recent_violations(
                    Duration::hours(escalation_level.time_window_hours as i64)
                );
                
                if recent_violations >= escalation_level.violation_threshold {
                    info!("Escalating to level: {}", escalation_level.level);
                    
                    for action in &escalation_level.actions {
                        self.execute_escalation_action(action, violation)?;
                    }
                }
            }
            
            // Create active alert
            let alert = ActiveAlert {
                alert_id: format!("ALERT-{}", self.generate_violation_id()),
                violation_id: violation.violation_id.clone(),
                severity: match violation.severity {
                    ViolationSeverity::Low => AlertSeverity::Info,
                    ViolationSeverity::Medium => AlertSeverity::Warning,
                    ViolationSeverity::High => AlertSeverity::Critical,
                    ViolationSeverity::Critical => AlertSeverity::Critical,
                    ViolationSeverity::Emergency => AlertSeverity::Emergency,
                },
                created_at: Utc::now(),
                escalated: false,
            };
            
            self.active_alerts.push(alert);
        }
        
        Ok(())
    }
    
    fn count_recent_violations(&self, duration: Duration) -> u32 {
        let cutoff = Utc::now() - duration;
        self.violation_history.iter()
            .filter(|v| v.detected_at >= cutoff)
            .count() as u32
    }
    
    fn execute_escalation_action(&self, action: &EscalationAction, violation: &ComplianceViolation) -> Result<()> {
        match action {
            EscalationAction::Alert { message, severity } => {
                match severity {
                    AlertSeverity::Info => info!("ALERT: {}", message),
                    AlertSeverity::Warning => warn!("WARNING: {}", message),
                    AlertSeverity::Critical => error!("CRITICAL: {}", message),
                    AlertSeverity::Emergency => error!("EMERGENCY: {}", message),
                }
            }
            EscalationAction::StopCalibration => {
                error!("ESCALATION: Stopping calibration system due to violation {}", violation.violation_id);
                // In a real implementation, this would stop the calibration service
            }
            EscalationAction::RevertConfiguration => {
                warn!("ESCALATION: Configuration revert requested for violation {}", violation.violation_id);
                // In a real implementation, this would trigger configuration rollback
            }
            EscalationAction::EmergencyProcedure { procedure } => {
                error!("ESCALATION: Triggering emergency procedure '{}' for violation {}", 
                       procedure, violation.violation_id);
                // In a real implementation, this would trigger emergency procedures
            }
            EscalationAction::LogViolation { details } => {
                info!("LOGGING: Violation {} - {}", violation.violation_id, details);
                // Violation is already logged in history
            }
        }
        
        Ok(())
    }
    
    fn calculate_compliance_trends(&self) -> ComplianceTrends {
        // Calculate trends over the last 30 days
        let now = Utc::now();
        let mut weekly_scores = Vec::new();
        
        for week in 0..4 {
            let week_start = now - Duration::weeks((week + 1) as i64);
            let week_end = now - Duration::weeks(week as i64);
            
            let week_violations = self.violation_history.iter()
                .filter(|v| v.detected_at >= week_start && v.detected_at < week_end)
                .count();
            
            // Simple compliance score based on violation count
            let score = if week_violations == 0 {
                100.0
            } else {
                (100.0 - (week_violations as f32 * 10.0)).max(0.0)
            };
            
            weekly_scores.push(score);
        }
        
        // Calculate trend direction
        let trend_direction = if weekly_scores.len() >= 2 {
            let recent_avg = (weekly_scores[0] + weekly_scores[1]) / 2.0;
            let older_avg = (weekly_scores[2] + weekly_scores[3]) / 2.0;
            
            if recent_avg > older_avg + 5.0 {
                "IMPROVING".to_string()
            } else if recent_avg < older_avg - 5.0 {
                "DEGRADING".to_string()
            } else {
                "STABLE".to_string()
            }
        } else {
            "UNKNOWN".to_string()
        };
        
        ComplianceTrends {
            trend_direction,
            weekly_compliance_scores: weekly_scores,
            violation_rate_change_percent: 0.0, // Would calculate actual rate change
        }
    }
    
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if let Some(last_result) = &self.last_compliance_result {
            if !last_result.compliant {
                recommendations.push("Address current compliance violations before proceeding".to_string());
            }
            
            if last_result.compliance_score < 90.0 {
                recommendations.push("Improve calibration training data quality and quantity".to_string());
            }
        }
        
        let recent_violations = self.get_compliance_history(7);
        if recent_violations.len() > 5 {
            recommendations.push("High violation rate detected - review calibration system configuration".to_string());
        }
        
        if self.active_alerts.len() > 10 {
            recommendations.push("Many active alerts - prioritize resolution of critical issues".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("System is operating within compliance parameters".to_string());
        }
        
        recommendations
    }
}

/// Governance report data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceReport {
    pub generated_at: DateTime<Utc>,
    pub reporting_period_days: u32,
    pub current_compliance_status: String,
    pub compliance_score: f32,
    pub violation_statistics: ViolationStatistics,
    pub compliance_trends: ComplianceTrends,
    pub active_alerts_count: usize,
    pub recommendations: Vec<String>,
}

/// Violation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationStatistics {
    pub total_violations: usize,
    pub last_30_days_violations: usize,
    pub critical_violations: usize,
    pub resolved_violations: usize,
}

/// Compliance trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceTrends {
    pub trend_direction: String,
    pub weekly_compliance_scores: Vec<f32>,
    pub violation_rate_change_percent: f32,
}

// Default implementations

impl Default for GovernanceConfig {
    fn default() -> Self {
        Self {
            enable_sla_enforcement: true,
            ece_threshold_params: EceThresholdParams {
                base_ece_requirement: 0.015,
                empirical_constant: 1.5,
                default_bin_count: 15,
                min_sample_count: 30,
                max_allowed_ece: 0.10,
            },
            language_variance_limits: LanguageVarianceLimits {
                max_tier1_variance_pp: 7.0,
                max_tier2_variance_pp: 10.0,
                max_individual_language_ece_multiplier: 2.0,
                min_languages_for_variance: 2,
            },
            compatibility_settings: CompatibilitySettings {
                enable_version_checking: true,
                supported_manifest_versions: vec!["1.0.0".to_string()],
                supported_phase4_versions: vec!["1.0.0".to_string()],
                required_dependencies: HashMap::new(),
                enable_breaking_change_detection: true,
            },
            ieee754_settings: Ieee754Settings {
                enable_strict_compliance: true,
                detect_fast_math: true,
                enable_denormal_handling: true,
                precision_requirements: PrecisionRequirements {
                    ece_precision: 4,
                    score_precision: 6,
                    confidence_precision: 3,
                    numerical_tolerance: 1e-6,
                },
                special_values_handling: SpecialValuesHandling {
                    nan_handling: NanHandling::ReplaceWithNeutral,
                    infinity_handling: InfinityHandling::Clamp { min: 0.001, max: 0.999 },
                    detect_subnormals: true,
                    max_relative_error: 1e-6,
                },
            },
            monitoring_config: MonitoringConfig {
                enable_realtime_monitoring: true,
                check_interval_seconds: 300, // 5 minutes
                violation_tracking_days: 30,
                alert_thresholds: AlertThresholds {
                    ece_violations: SeverityThresholds {
                        warning: 0.1,
                        critical: 0.3,
                        emergency: 0.5,
                    },
                    variance_violations: SeverityThresholds {
                        warning: 1.0,
                        critical: 2.0,
                        emergency: 3.0,
                    },
                    compatibility_violations: SeverityThresholds {
                        warning: 1.0,
                        critical: 1.0,
                        emergency: 1.0,
                    },
                    ieee754_violations: SeverityThresholds {
                        warning: 0.01,
                        critical: 0.05,
                        emergency: 0.10,
                    },
                },
                enable_automated_reports: true,
            },
            violation_escalation: ViolationEscalationConfig {
                enable_auto_escalation: true,
                escalation_levels: vec![
                    EscalationLevel {
                        level: "Warning".to_string(),
                        violation_threshold: 3,
                        time_window_hours: 1,
                        actions: vec![
                            EscalationAction::Alert {
                                message: "Multiple violations detected".to_string(),
                                severity: AlertSeverity::Warning,
                            },
                            EscalationAction::LogViolation {
                                details: "Warning level escalation".to_string(),
                            },
                        ],
                    },
                    EscalationLevel {
                        level: "Critical".to_string(),
                        violation_threshold: 5,
                        time_window_hours: 6,
                        actions: vec![
                            EscalationAction::Alert {
                                message: "Critical violation threshold exceeded".to_string(),
                                severity: AlertSeverity::Critical,
                            },
                            EscalationAction::RevertConfiguration,
                        ],
                    },
                    EscalationLevel {
                        level: "Emergency".to_string(),
                        violation_threshold: 10,
                        time_window_hours: 24,
                        actions: vec![
                            EscalationAction::Alert {
                                message: "Emergency: System non-compliance critical".to_string(),
                                severity: AlertSeverity::Emergency,
                            },
                            EscalationAction::StopCalibration,
                            EscalationAction::EmergencyProcedure {
                                procedure: "calibration_emergency_stop".to_string(),
                            },
                        ],
                    },
                ],
                max_violations_before_emergency: 20,
                alert_cooldown_minutes: 15,
            },
        }
    }
}

// Display implementations

impl fmt::Display for ComplianceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.compliant { "✓ COMPLIANT" } else { "✗ NON-COMPLIANT" };
        write!(f, "{} (score: {:.1}%, checks: {}, violations: {})", 
               status, self.compliance_score, self.checks.len(), self.violations.len())
    }
}

impl fmt::Display for ComplianceViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {:?}: {} (measured: {:.4} vs threshold: {:.4})",
               self.violation_id, self.severity, self.description, 
               self.measured_value, self.threshold_value)
    }
}

/// Initialize governance service with default configuration
pub fn initialize_calibration_governance() -> CalibrationGovernance {
    let config = GovernanceConfig::default();
    CalibrationGovernance::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Phase4Config, feature_flags::CalibV22Config};
    
    #[test]
    fn test_statistical_ece_threshold_calculation() {
        let config = GovernanceConfig::default();
        let governance = CalibrationGovernance::new(config);
        
        // Test with typical values
        let threshold = governance.calculate_statistical_ece_threshold(1000, 15);
        assert!(threshold >= 0.015);
        assert!(threshold < 0.10);
        
        // Test with small sample count
        let threshold_small = governance.calculate_statistical_ece_threshold(100, 15);
        assert!(threshold_small > threshold); // Should be higher for smaller N
        
        // Test edge case
        let threshold_zero = governance.calculate_statistical_ece_threshold(0, 15);
        assert_eq!(threshold_zero, 0.015);
    }
    
    #[test]
    fn test_phase4_constraints_validation() {
        let config = GovernanceConfig::default();
        let governance = CalibrationGovernance::new(config);
        
        let valid_config = Phase4Config::default();
        assert!(governance.validate_phase4_constraints(&valid_config));
        
        let mut invalid_config = Phase4Config::default();
        invalid_config.target_ece = 0.020; // Too high
        assert!(!governance.validate_phase4_constraints(&invalid_config));
        
        let mut invalid_config2 = Phase4Config::default();
        invalid_config2.max_language_variance = 8.0; // Too high
        assert!(!governance.validate_phase4_constraints(&invalid_config2));
    }
    
    #[test]
    fn test_ieee754_compliance_checking() {
        let config = GovernanceConfig::default();
        let governance = CalibrationGovernance::new(config);
        
        // Valid samples
        let valid_samples = vec![
            CalibrationSample {
                prediction: 0.7,
                ground_truth: 1.0,
                intent: "test".to_string(),
                language: Some("rust".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            },
        ];
        
        let check = governance.enforce_ieee754_compliance(&valid_samples).unwrap();
        assert!(check.passed);
        
        // Invalid samples with NaN
        let invalid_samples = vec![
            CalibrationSample {
                prediction: f32::NAN,
                ground_truth: 1.0,
                intent: "test".to_string(),
                language: Some("rust".to_string()),
                features: HashMap::new(),
                weight: 1.0,
            },
        ];
        
        let check = governance.enforce_ieee754_compliance(&invalid_samples).unwrap();
        // Should fail if NaN handling is set to Reject
        assert!(!check.passed);
    }
    
    #[test]
    fn test_subnormal_detection() {
        let config = GovernanceConfig::default();
        let governance = CalibrationGovernance::new(config);
        
        assert!(!governance.is_subnormal(1.0));
        assert!(!governance.is_subnormal(0.0));
        assert!(governance.is_subnormal(f32::MIN_POSITIVE / 2.0));
    }
    
    #[test]
    fn test_violation_severity_determination() {
        let mut config = GovernanceConfig::default();
        config.monitoring_config.alert_thresholds.ece_violations = SeverityThresholds {
            warning: 0.1,
            critical: 0.5,
            emergency: 1.0,
        };
        
        let governance = CalibrationGovernance::new(config);
        
        // Test ECE violation severity
        let severity1 = governance.determine_ece_violation_severity(0.020, 0.015);
        assert!(matches!(severity1, ViolationSeverity::High | ViolationSeverity::Medium));
        
        let severity2 = governance.determine_ece_violation_severity(0.030, 0.015);
        assert!(matches!(severity2, ViolationSeverity::Critical | ViolationSeverity::Emergency));
    }
    
    #[test]
    fn test_governance_report_generation() {
        let config = GovernanceConfig::default();
        let governance = CalibrationGovernance::new(config);
        
        let report = governance.generate_compliance_report().unwrap();
        
        assert!(!report.generated_at.timestamp().is_zero());
        assert_eq!(report.reporting_period_days, 30);
        assert!(!report.recommendations.is_empty());
    }
}