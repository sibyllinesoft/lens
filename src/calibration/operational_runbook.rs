//! Operational Runbook for Calibration System
//! 
//! Single-page runbook providing automated decision trees, data collection,
//! and communication templates for calibration incidents and maintenance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, warn, info, debug};

use crate::calibration::drift_slos::{SloViolation, AlertSeverity, CalibrationMetrics};

/// Calibration system symptoms for automated diagnosis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationSymptom {
    /// AECE drift exceeding thresholds
    AeceDrift { current: f64, baseline: f64, threshold: f64 },
    /// DECE drift exceeding thresholds  
    DeceDrift { current: f64, baseline: f64, threshold: f64 },
    /// Alpha parameter drift
    AlphaDrift { current: f64, baseline: f64, threshold: f64 },
    /// High clamp rate indicating distribution shift
    HighClampRate { rate: f64, threshold: f64 },
    /// Excessive merged bins indicating calibration instability
    ExcessiveMergedBins { rate: f64, warn_threshold: f64, fail_threshold: f64 },
    /// Score range violations (scores ∉ [0,1])
    ScoreRangeViolations { count: u64 },
    /// Mask mismatch detection
    MaskMismatch { count: u64 },
    /// General performance degradation
    PerformanceDegradation { metric: String, current: f64, baseline: f64 },
}

/// Automated remediation action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemediationAction {
    /// Raise confidence threshold ĉ for specific class
    RaiseConfidenceThreshold { class_id: usize, from: f64, to: f64, reason: String },
    /// Revert to previous calibration model
    RevertToPreviousModel { reason: String, rollback_timestamp: u64 },
    /// Trigger recalibration with updated parameters
    TriggerRecalibration { parameters: HashMap<String, f64> },
    /// Alert human operators
    EscalateToHuman { severity: AlertSeverity, context: String },
    /// Monitor and wait for stabilization
    MonitorAndWait { duration_hours: u8, check_interval_minutes: u8 },
    /// Temporarily disable affected features
    DisableFeatures { features: Vec<String>, duration_hours: u8 },
}

/// Decision tree node for automated incident response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    pub condition: String,
    pub action: RemediationAction,
    pub verification_steps: Vec<String>,
    pub rollback_plan: Option<String>,
    pub escalation_criteria: Vec<String>,
}

/// Automated data collection package for incidents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentDataPackage {
    pub timestamp: u64,
    pub symptoms: Vec<CalibrationSymptom>,
    pub bin_table: HashMap<String, BinData>,
    pub alpha_values: HashMap<String, f64>,
    pub tau_values: HashMap<String, f64>,
    pub aece_tau_values: HashMap<String, f64>,
    pub recent_metrics_history: Vec<CalibrationMetrics>,
    pub system_context: HashMap<String, String>,
}

/// Calibration bin data for diagnostic purposes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinData {
    pub bin_id: usize,
    pub confidence_range: (f64, f64),
    pub sample_count: u64,
    pub accuracy: f64,
    pub merged_with: Option<usize>,
    pub clamp_count: u64,
}

/// Communication template for different stakeholders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationTemplate {
    pub audience: String,
    pub severity: AlertSeverity,
    pub subject_template: String,
    pub body_template: String,
    pub required_context: Vec<String>,
}

/// Single-page operational runbook for calibration incidents
#[derive(Debug)]
pub struct OperationalRunbook {
    decision_tree: HashMap<String, DecisionNode>,
    communication_templates: HashMap<String, CommunicationTemplate>,
    data_collectors: HashMap<String, Box<dyn Fn() -> Result<IncidentDataPackage, String>>>,
}

impl OperationalRunbook {
    /// Create new runbook with default decision trees and templates
    pub fn new() -> Self {
        let mut runbook = Self {
            decision_tree: HashMap::new(),
            communication_templates: HashMap::new(),
            data_collectors: HashMap::new(),
        };
        
        runbook.initialize_decision_tree();
        runbook.initialize_communication_templates();
        
        runbook
    }

    /// Initialize automated decision tree for common calibration issues
    fn initialize_decision_tree(&mut self) {
        // AECE drift decision tree
        self.decision_tree.insert("aece_drift".to_string(), DecisionNode {
            condition: "AECE drift > threshold AND drift increasing".to_string(),
            action: RemediationAction::RaiseConfidenceThreshold {
                class_id: 0, // Will be determined dynamically
                from: 0.5,
                to: 0.6,
                reason: "AECE drift mitigation".to_string(),
            },
            verification_steps: vec![
                "Re-measure AECE after 1 hour".to_string(),
                "Verify AECE trend is decreasing".to_string(),
                "Check for side effects on other metrics".to_string(),
            ],
            rollback_plan: Some("Revert confidence threshold if AECE doesn't improve in 2 hours".to_string()),
            escalation_criteria: vec![
                "AECE continues increasing after remediation".to_string(),
                "Other calibration metrics degrade significantly".to_string(),
            ],
        });

        // DECE drift decision tree
        self.decision_tree.insert("dece_drift".to_string(), DecisionNode {
            condition: "DECE drift > threshold AND affecting multiple classes".to_string(),
            action: RemediationAction::TriggerRecalibration {
                parameters: [
                    ("learning_rate".to_string(), 0.01),
                    ("regularization".to_string(), 0.1),
                ].iter().cloned().collect(),
            },
            verification_steps: vec![
                "Monitor DECE during recalibration".to_string(),
                "Verify distribution alignment improves".to_string(),
                "Check that bin merging stabilizes".to_string(),
            ],
            rollback_plan: Some("Revert to previous model if recalibration fails".to_string()),
            escalation_criteria: vec![
                "Recalibration doesn't complete within 30 minutes".to_string(),
                "DECE gets worse during recalibration".to_string(),
            ],
        });

        // High clamp rate decision tree  
        self.decision_tree.insert("high_clamp_rate".to_string(), DecisionNode {
            condition: "Clamp rate > 10% AND trending upward".to_string(),
            action: RemediationAction::MonitorAndWait {
                duration_hours: 2,
                check_interval_minutes: 15,
            },
            verification_steps: vec![
                "Track clamp rate trend every 15 minutes".to_string(),
                "Identify which classes have highest clamp rates".to_string(),
                "Check if distribution shift is temporary".to_string(),
            ],
            rollback_plan: Some("Trigger recalibration if clamp rate doesn't stabilize".to_string()),
            escalation_criteria: vec![
                "Clamp rate exceeds 20%".to_string(),
                "Clamp rate doesn't decrease within 2 hours".to_string(),
            ],
        });

        // Score range violation decision tree
        self.decision_tree.insert("score_range_violations".to_string(), DecisionNode {
            condition: "Any scores ∉ [0,1] detected".to_string(),
            action: RemediationAction::RevertToPreviousModel {
                reason: "Score range integrity violation - immediate rollback required".to_string(),
                rollback_timestamp: 0, // Will be set dynamically
            },
            verification_steps: vec![
                "Verify all scores ∈ [0,1] after rollback".to_string(),
                "Check that calibration metrics return to baseline".to_string(),
                "Validate system stability for 1 hour".to_string(),
            ],
            rollback_plan: Some("If rollback fails, disable scoring system and escalate".to_string()),
            escalation_criteria: vec![
                "Rollback doesn't resolve score range issues".to_string(),
                "Previous model also shows score violations".to_string(),
            ],
        });

        // Excessive merged bins decision tree
        self.decision_tree.insert("excessive_merged_bins".to_string(), DecisionNode {
            condition: "Merged bin rate > 20% (failure threshold)".to_string(),
            action: RemediationAction::EscalateToHuman {
                severity: AlertSeverity::Critical,
                context: "Calibration stability compromised - manual intervention required".to_string(),
            },
            verification_steps: vec![
                "Document current bin configuration".to_string(),
                "Identify root cause of bin instability".to_string(),
                "Assess impact on calibration accuracy".to_string(),
            ],
            rollback_plan: Some("Prepared for potential full system recalibration".to_string()),
            escalation_criteria: vec![
                "Immediate escalation - no automated remediation".to_string(),
            ],
        });
    }

    /// Initialize communication templates for different audiences
    fn initialize_communication_templates(&mut self) {
        // Technical team alert
        self.communication_templates.insert("technical_team".to_string(), CommunicationTemplate {
            audience: "technical_team".to_string(),
            severity: AlertSeverity::High,
            subject_template: "[CALIBRATION] {severity} - {symptom} detected".to_string(),
            body_template: 
                "CALIBRATION INCIDENT ALERT\n\
                 ========================\n\
                 \n\
                 Symptom: {symptom}\n\
                 Severity: {severity}\n\
                 Time: {timestamp}\n\
                 \n\
                 Current Metrics:\n\
                 - AECE: {aece:.6}\n\
                 - DECE: {dece:.6}\n\
                 - Alpha: {alpha:.6}\n\
                 - Clamp Rate: {clamp_rate:.2}%\n\
                 - Merged Bins: {merged_bin_rate:.2}%\n\
                 \n\
                 Automated Action: {action}\n\
                 \n\
                 Verification Steps:\n\
                 {verification_steps}\n\
                 \n\
                 Escalation Criteria:\n\
                 {escalation_criteria}\n\
                 \n\
                 Data Package: {data_package_location}".to_string(),
            required_context: vec![
                "symptom".to_string(), "severity".to_string(), "timestamp".to_string(),
                "aece".to_string(), "dece".to_string(), "alpha".to_string(),
                "clamp_rate".to_string(), "merged_bin_rate".to_string(),
                "action".to_string(), "verification_steps".to_string(),
                "escalation_criteria".to_string(), "data_package_location".to_string(),
            ],
        });

        // Management summary
        self.communication_templates.insert("management".to_string(), CommunicationTemplate {
            audience: "management".to_string(),
            severity: AlertSeverity::Medium,
            subject_template: "Calibration System Status - {severity} Alert".to_string(),
            body_template:
                "CALIBRATION STATUS UPDATE\n\
                 ========================\n\
                 \n\
                 Status: {status}\n\
                 Impact: {impact}\n\
                 Action Taken: {action_summary}\n\
                 Expected Resolution: {eta}\n\
                 \n\
                 System Health:\n\
                 - Calibration Accuracy: {health_summary}\n\
                 - User Impact: {user_impact}\n\
                 - Service Availability: {availability}\n\
                 \n\
                 Next Update: {next_update}".to_string(),
            required_context: vec![
                "status".to_string(), "impact".to_string(), "action_summary".to_string(),
                "eta".to_string(), "health_summary".to_string(), "user_impact".to_string(),
                "availability".to_string(), "next_update".to_string(),
            ],
        });

        // Customer communication  
        self.communication_templates.insert("customers".to_string(), CommunicationTemplate {
            audience: "customers".to_string(),
            severity: AlertSeverity::Low,
            subject_template: "Service Update - Calibration Optimization".to_string(),
            body_template:
                "SERVICE UPDATE\n\
                 =============\n\
                 \n\
                 We're currently optimizing our calibration system to improve accuracy.\n\
                 \n\
                 Impact: {customer_impact}\n\
                 Duration: {estimated_duration}\n\
                 \n\
                 What we're doing:\n\
                 {customer_friendly_description}\n\
                 \n\
                 We'll update you when optimization is complete.\n\
                 \n\
                 Thank you for your patience.".to_string(),
            required_context: vec![
                "customer_impact".to_string(), "estimated_duration".to_string(),
                "customer_friendly_description".to_string(),
            ],
        });
    }

    /// Execute automated incident response for given symptoms
    pub fn execute_incident_response(&self, symptoms: Vec<CalibrationSymptom>) -> Vec<RemediationAction> {
        let mut actions = Vec::new();
        
        for symptom in symptoms {
            let symptom_key = match &symptom {
                CalibrationSymptom::AeceDrift { .. } => "aece_drift",
                CalibrationSymptom::DeceDrift { .. } => "dece_drift", 
                CalibrationSymptom::AlphaDrift { .. } => "alpha_drift",
                CalibrationSymptom::HighClampRate { .. } => "high_clamp_rate",
                CalibrationSymptom::ExcessiveMergedBins { .. } => "excessive_merged_bins",
                CalibrationSymptom::ScoreRangeViolations { .. } => "score_range_violations",
                CalibrationSymptom::MaskMismatch { .. } => "mask_mismatch",
                CalibrationSymptom::PerformanceDegradation { .. } => "performance_degradation",
            };

            if let Some(decision_node) = self.decision_tree.get(symptom_key) {
                info!("Executing automated response for {}: {}", symptom_key, decision_node.condition);
                actions.push(decision_node.action.clone());
                
                // Log verification plan
                info!("Verification steps for {}: {:?}", symptom_key, decision_node.verification_steps);
                
                // Log escalation criteria
                info!("Escalation criteria for {}: {:?}", symptom_key, decision_node.escalation_criteria);
            } else {
                warn!("No decision tree found for symptom: {}", symptom_key);
                actions.push(RemediationAction::EscalateToHuman {
                    severity: AlertSeverity::Medium,
                    context: format!("Unknown symptom type: {}", symptom_key),
                });
            }
        }

        actions
    }

    /// Collect comprehensive diagnostic data for incident analysis
    pub fn collect_incident_data(&self, symptoms: Vec<CalibrationSymptom>) -> IncidentDataPackage {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        // In a real implementation, these would fetch actual system data
        let bin_table = self.collect_bin_table();
        let alpha_values = self.collect_alpha_values(); 
        let tau_values = self.collect_tau_values();
        let aece_tau_values = self.collect_aece_tau_values();
        let recent_metrics = self.collect_recent_metrics_history();
        let system_context = self.collect_system_context();

        IncidentDataPackage {
            timestamp,
            symptoms,
            bin_table,
            alpha_values,
            tau_values, 
            aece_tau_values,
            recent_metrics_history: recent_metrics,
            system_context,
        }
    }

    /// Generate communication for specific audience and incident
    pub fn generate_communication(
        &self, 
        audience: &str,
        incident_data: &IncidentDataPackage,
        actions: &[RemediationAction]
    ) -> Result<String, String> {
        let template = self.communication_templates.get(audience)
            .ok_or_else(|| format!("No communication template for audience: {}", audience))?;

        let mut context = HashMap::new();
        
        // Build context based on incident data
        self.build_communication_context(&mut context, incident_data, actions);
        
        // Replace template variables
        let mut message = template.body_template.clone();
        for (key, value) in context {
            message = message.replace(&format!("{{{}}}", key), &value);
        }

        Ok(message)
    }

    /// Build context variables for communication templates
    fn build_communication_context(
        &self,
        context: &mut HashMap<String, String>,
        incident_data: &IncidentDataPackage,
        actions: &[RemediationAction]
    ) {
        // Timestamp
        context.insert("timestamp".to_string(), format!("{}", incident_data.timestamp));
        
        // Symptom summary
        let symptom_summary = incident_data.symptoms.iter()
            .map(|s| self.format_symptom(s))
            .collect::<Vec<_>>()
            .join(", ");
        context.insert("symptom".to_string(), symptom_summary);
        
        // Severity assessment
        let severity = self.assess_overall_severity(&incident_data.symptoms);
        context.insert("severity".to_string(), format!("{:?}", severity));
        
        // Action summary
        let action_summary = actions.iter()
            .map(|a| self.format_action(a))
            .collect::<Vec<_>>()
            .join("; ");
        context.insert("action".to_string(), action_summary);
        context.insert("action_summary".to_string(), action_summary);
        
        // Latest metrics if available
        if let Some(latest) = incident_data.recent_metrics_history.first() {
            context.insert("aece".to_string(), format!("{:.6}", latest.aece));
            context.insert("dece".to_string(), format!("{:.6}", latest.dece));
            context.insert("alpha".to_string(), format!("{:.6}", latest.alpha));
            context.insert("clamp_rate".to_string(), format!("{:.2}", latest.clamp_rate * 100.0));
            context.insert("merged_bin_rate".to_string(), format!("{:.2}", latest.merged_bin_rate * 100.0));
        }
        
        // System context
        for (key, value) in &incident_data.system_context {
            context.insert(key.clone(), value.clone());
        }
        
        // Default values for optional template variables
        context.insert("status".to_string(), "Under Investigation".to_string());
        context.insert("impact".to_string(), "Monitoring for user impact".to_string());
        context.insert("eta".to_string(), "1-2 hours".to_string());
        context.insert("health_summary".to_string(), "Degraded but operational".to_string());
        context.insert("user_impact".to_string(), "Minimal expected impact".to_string());
        context.insert("availability".to_string(), "System operational".to_string());
        context.insert("next_update".to_string(), "In 1 hour or when resolved".to_string());
        context.insert("verification_steps".to_string(), "Automated verification in progress".to_string());
        context.insert("escalation_criteria".to_string(), "Defined per incident type".to_string());
        context.insert("data_package_location".to_string(), "/var/log/calibration/incidents".to_string());
        context.insert("customer_impact".to_string(), "No expected impact on service quality".to_string());
        context.insert("estimated_duration".to_string(), "15-30 minutes".to_string());
        context.insert("customer_friendly_description".to_string(), "Improving accuracy of our recommendation system".to_string());
    }

    /// Format symptom for human readable output
    fn format_symptom(&self, symptom: &CalibrationSymptom) -> String {
        match symptom {
            CalibrationSymptom::AeceDrift { current, baseline, threshold } =>
                format!("AECE drift: {:.4} (was {:.4}, threshold {:.4})", current, baseline, threshold),
            CalibrationSymptom::DeceDrift { current, baseline, threshold } =>
                format!("DECE drift: {:.4} (was {:.4}, threshold {:.4})", current, baseline, threshold),
            CalibrationSymptom::AlphaDrift { current, baseline, threshold } =>
                format!("Alpha drift: {:.4} (was {:.4}, threshold {:.4})", current, baseline, threshold),
            CalibrationSymptom::HighClampRate { rate, threshold } =>
                format!("High clamp rate: {:.2}% (threshold {:.2}%)", rate * 100.0, threshold * 100.0),
            CalibrationSymptom::ExcessiveMergedBins { rate, warn_threshold, fail_threshold: _ } =>
                format!("Excessive merged bins: {:.2}% (warn threshold {:.2}%)", rate * 100.0, warn_threshold * 100.0),
            CalibrationSymptom::ScoreRangeViolations { count } =>
                format!("Score range violations: {} instances", count),
            CalibrationSymptom::MaskMismatch { count } =>
                format!("Mask mismatches: {} instances", count),
            CalibrationSymptom::PerformanceDegradation { metric, current, baseline } =>
                format!("Performance degradation in {}: {:.4} (was {:.4})", metric, current, baseline),
        }
    }

    /// Format remediation action for human readable output
    fn format_action(&self, action: &RemediationAction) -> String {
        match action {
            RemediationAction::RaiseConfidenceThreshold { class_id, from, to, reason } =>
                format!("Raise confidence threshold for class {} from {:.2} to {:.2} ({})", class_id, from, to, reason),
            RemediationAction::RevertToPreviousModel { reason, .. } =>
                format!("Revert to previous model ({})", reason),
            RemediationAction::TriggerRecalibration { parameters } =>
                format!("Trigger recalibration with {} parameters", parameters.len()),
            RemediationAction::EscalateToHuman { severity, context } =>
                format!("Escalate to human ({:?}): {}", severity, context),
            RemediationAction::MonitorAndWait { duration_hours, check_interval_minutes } =>
                format!("Monitor for {} hours (check every {} minutes)", duration_hours, check_interval_minutes),
            RemediationAction::DisableFeatures { features, duration_hours } =>
                format!("Temporarily disable {} features for {} hours", features.len(), duration_hours),
        }
    }

    /// Assess overall severity from multiple symptoms
    fn assess_overall_severity(&self, symptoms: &[CalibrationSymptom]) -> AlertSeverity {
        let mut max_severity = AlertSeverity::Low;
        
        for symptom in symptoms {
            let severity = match symptom {
                CalibrationSymptom::ScoreRangeViolations { .. } => AlertSeverity::Critical,
                CalibrationSymptom::ExcessiveMergedBins { rate, fail_threshold, .. } if rate > fail_threshold => AlertSeverity::Critical,
                CalibrationSymptom::AeceDrift { .. } | CalibrationSymptom::DeceDrift { .. } => AlertSeverity::High,
                CalibrationSymptom::HighClampRate { rate, threshold } if rate > threshold * 2.0 => AlertSeverity::High,
                CalibrationSymptom::MaskMismatch { .. } => AlertSeverity::High,
                _ => AlertSeverity::Medium,
            };
            
            if (severity as u8) > (max_severity as u8) {
                max_severity = severity;
            }
        }
        
        max_severity
    }

    // Mock data collection methods (would be replaced with real implementations)
    fn collect_bin_table(&self) -> HashMap<String, BinData> {
        // Mock implementation - would collect actual bin data
        HashMap::new()
    }

    fn collect_alpha_values(&self) -> HashMap<String, f64> {
        // Mock implementation - would collect actual alpha values
        HashMap::new()
    }

    fn collect_tau_values(&self) -> HashMap<String, f64> {
        // Mock implementation - would collect actual tau values  
        HashMap::new()
    }

    fn collect_aece_tau_values(&self) -> HashMap<String, f64> {
        // Mock implementation - would collect actual AECE-τ values
        HashMap::new()
    }

    fn collect_recent_metrics_history(&self) -> Vec<CalibrationMetrics> {
        // Mock implementation - would collect actual metrics history
        Vec::new()
    }

    fn collect_system_context(&self) -> HashMap<String, String> {
        // Mock implementation - would collect actual system context
        let mut context = HashMap::new();
        context.insert("version".to_string(), "1.0.0".to_string());
        context.insert("environment".to_string(), "production".to_string());
        context
    }

    /// Print single-page runbook summary
    pub fn print_runbook_summary(&self) {
        println!("CALIBRATION OPERATIONAL RUNBOOK");
        println!("===============================");
        println!();
        println!("🚨 SYMPTOMS → DATA → DECISION → VERIFICATION → COMMS");
        println!();
        
        println!("📊 AUTOMATED DATA COLLECTION:");
        println!("  • Bin table with confidence ranges and sample counts");
        println!("  • Alpha (α), Tau (τ), and AECE-τ values by class");
        println!("  • Recent metrics history (AECE, DECE, clamp rates)");
        println!("  • System context and configuration state");
        println!();
        
        println!("🌳 DECISION TREE:");
        for (symptom, node) in &self.decision_tree {
            println!("  {} → {}", symptom.to_uppercase(), node.condition);
            println!("    Action: {}", self.format_action(&node.action));
            println!("    Verify: {} steps", node.verification_steps.len());
            println!();
        }
        
        println!("📞 COMMUNICATION TEMPLATES:");
        for (audience, template) in &self.communication_templates {
            println!("  {} ({:?}): {}", audience, template.severity, template.subject_template);
        }
        println!();
        
        println!("⚡ ESCALATION TRIGGERS:");
        println!("  • Score range violations → Immediate rollback");
        println!("  • >20% merged bins → Human escalation");
        println!("  • Failed automated remediation → Human escalation");
        println!("  • Continued degradation → Management notification");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incident_response_aece_drift() {
        let runbook = OperationalRunbook::new();
        let symptoms = vec![CalibrationSymptom::AeceDrift {
            current: 0.035,
            baseline: 0.02,
            threshold: 0.01,
        }];
        
        let actions = runbook.execute_incident_response(symptoms);
        assert!(!actions.is_empty());
        assert!(matches!(actions[0], RemediationAction::RaiseConfidenceThreshold { .. }));
    }

    #[test]  
    fn test_incident_response_score_violations() {
        let runbook = OperationalRunbook::new();
        let symptoms = vec![CalibrationSymptom::ScoreRangeViolations { count: 5 }];
        
        let actions = runbook.execute_incident_response(symptoms);
        assert!(!actions.is_empty());
        assert!(matches!(actions[0], RemediationAction::RevertToPreviousModel { .. }));
    }

    #[test]
    fn test_communication_generation() {
        let runbook = OperationalRunbook::new();
        let incident_data = IncidentDataPackage {
            timestamp: 1234567890,
            symptoms: vec![CalibrationSymptom::HighClampRate { rate: 0.15, threshold: 0.10 }],
            bin_table: HashMap::new(),
            alpha_values: HashMap::new(),
            tau_values: HashMap::new(),
            aece_tau_values: HashMap::new(),
            recent_metrics_history: Vec::new(),
            system_context: HashMap::new(),
        };
        
        let actions = vec![RemediationAction::MonitorAndWait { 
            duration_hours: 2, 
            check_interval_minutes: 15 
        }];
        
        let message = runbook.generate_communication("technical_team", &incident_data, &actions).unwrap();
        assert!(message.contains("CALIBRATION INCIDENT ALERT"));
        assert!(message.contains("High clamp rate"));
    }

    #[test]
    fn test_severity_assessment() {
        let runbook = OperationalRunbook::new();
        
        // Critical symptoms
        let critical_symptoms = vec![CalibrationSymptom::ScoreRangeViolations { count: 1 }];
        assert_eq!(runbook.assess_overall_severity(&critical_symptoms), AlertSeverity::Critical);
        
        // High severity symptoms
        let high_symptoms = vec![CalibrationSymptom::AeceDrift { 
            current: 0.05, baseline: 0.02, threshold: 0.01 
        }];
        assert_eq!(runbook.assess_overall_severity(&high_symptoms), AlertSeverity::High);
        
        // Medium severity symptoms
        let medium_symptoms = vec![CalibrationSymptom::HighClampRate { 
            rate: 0.12, threshold: 0.10 
        }];
        assert_eq!(runbook.assess_overall_severity(&medium_symptoms), AlertSeverity::Medium);
    }

    #[test]
    fn test_data_collection() {
        let runbook = OperationalRunbook::new();
        let symptoms = vec![CalibrationSymptom::DeceDrift { 
            current: 0.025, baseline: 0.015, threshold: 0.01 
        }];
        
        let data_package = runbook.collect_incident_data(symptoms.clone());
        assert_eq!(data_package.symptoms.len(), 1);
        assert!(data_package.timestamp > 0);
    }
}