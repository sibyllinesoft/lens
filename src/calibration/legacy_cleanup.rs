// Legacy Path Cleanup and CI Enforcement System
// Removes technical debt and enforces "shared binning core only" architecture

use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Legacy components detected in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyComponent {
    pub file_path: String,
    pub component_type: LegacyComponentType,
    pub lines: Vec<u32>,
    pub risk_level: RiskLevel,
    pub migration_status: MigrationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LegacyComponentType {
    SimulatorHook,
    AlternateEceEvaluator,
    DuplicatedBinning,
    ObsoleteCalibratorInterface,
    HardcodedTestData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Critical,  // Blocks CI immediately
    High,      // Must be fixed before next release
    Medium,    // Should be fixed in current sprint
    Low,       // Technical debt to address
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MigrationStatus {
    NotStarted,
    InProgress,
    Completed,
    Verified,
}

/// Comprehensive cleanup report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupReport {
    pub total_legacy_components: usize,
    pub critical_issues: usize,
    pub high_priority_issues: usize,
    pub cleanup_completed: usize,
    pub migration_progress: f64,
    pub components: Vec<LegacyComponent>,
    pub blocked_patterns: Vec<String>,
    pub ci_enforcement_result: CiEnforcementResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiEnforcementResult {
    pub passed: bool,
    pub blocking_issues: Vec<String>,
    pub warnings: Vec<String>,
    pub cleanup_recommendations: Vec<String>,
}

/// Legacy cleanup and enforcement system
pub struct LegacyCleanupSystem {
    legacy_patterns: HashMap<LegacyComponentType, Vec<String>>,
    blocked_imports: Vec<String>,
    required_migrations: Vec<MigrationRule>,
}

impl LegacyCleanupSystem {
    pub fn new() -> Self {
        let mut legacy_patterns = HashMap::new();
        
        // Define legacy patterns to detect and remove
        legacy_patterns.insert(LegacyComponentType::SimulatorHook, vec![
            "legacy_simulator".to_string(),
            "old_calibration_hook".to_string(),
            "deprecated_simulator_interface".to_string(),
            "SimulatorCompat".to_string(),
            "legacy_calibration_adapter".to_string(),
        ]);

        legacy_patterns.insert(LegacyComponentType::AlternateEceEvaluator, vec![
            "AlternateECE".to_string(),
            "custom_ece_impl".to_string(),
            "non_standard_ece".to_string(),
            "legacy_ece_calculator".to_string(),
            "experimental_ece".to_string(),
        ]);

        legacy_patterns.insert(LegacyComponentType::DuplicatedBinning, vec![
            "duplicate_binning".to_string(),
            "custom_bin_logic".to_string(),
            "alternative_binning".to_string(),
            "binning_override".to_string(),
            "local_bin_implementation".to_string(),
        ]);

        legacy_patterns.insert(LegacyComponentType::ObsoleteCalibratorInterface, vec![
            "OldCalibrator".to_string(),
            "LegacyCalibrationInterface".to_string(),
            "deprecated_calibrator".to_string(),
            "compat_calibrator".to_string(),
        ]);

        legacy_patterns.insert(LegacyComponentType::HardcodedTestData, vec![
            "hardcoded_test_data".to_string(),
            "static_calibration_data".to_string(),
            "embedded_test_cases".to_string(),
            "inline_test_predictions".to_string(),
        ]);

        let blocked_imports = vec![
            "legacy_calibration::*".to_string(),
            "deprecated_ece::*".to_string(),
            "old_binning::*".to_string(),
            "simulator_compat::*".to_string(),
            "experimental::calibration::*".to_string(),
        ];

        let required_migrations = vec![
            MigrationRule {
                from_pattern: "legacy_simulator_hook".to_string(),
                to_pattern: "shared_binning_core".to_string(),
                migration_type: MigrationType::Replace,
                verification_test: "test_shared_binning_core".to_string(),
            },
            MigrationRule {
                from_pattern: "custom_ece_impl".to_string(),
                to_pattern: "standard_ece_calculation".to_string(),
                migration_type: MigrationType::Replace,
                verification_test: "test_standard_ece_consistency".to_string(),
            },
            MigrationRule {
                from_pattern: "duplicate_binning_logic".to_string(),
                to_pattern: "centralized_binning_core".to_string(),
                migration_type: MigrationType::Consolidate,
                verification_test: "test_binning_consistency".to_string(),
            },
        ];

        Self {
            legacy_patterns,
            blocked_imports,
            required_migrations,
        }
    }

    /// Comprehensive scan for legacy components
    pub fn scan_codebase<P: AsRef<Path>>(&self, root_path: P) -> Result<CleanupReport, std::io::Error> {
        let mut components = Vec::new();
        let mut critical_issues = 0;
        let mut high_priority_issues = 0;
        let mut cleanup_completed = 0;

        // Scan Rust source files
        self.scan_directory(&root_path, &mut components)?;

        // Analyze risk levels and migration status
        for component in &mut components {
            self.assess_component_risk(component);
            self.check_migration_status(component);
            
            match component.risk_level {
                RiskLevel::Critical => critical_issues += 1,
                RiskLevel::High => high_priority_issues += 1,
                _ => {}
            }

            if component.migration_status == MigrationStatus::Completed {
                cleanup_completed += 1;
            }
        }

        let migration_progress = if !components.is_empty() {
            cleanup_completed as f64 / components.len() as f64 * 100.0
        } else {
            100.0
        };

        // Generate CI enforcement result
        let ci_enforcement_result = self.enforce_ci_requirements(&components);

        Ok(CleanupReport {
            total_legacy_components: components.len(),
            critical_issues,
            high_priority_issues,
            cleanup_completed,
            migration_progress,
            components,
            blocked_patterns: self.get_blocked_patterns(),
            ci_enforcement_result,
        })
    }

    /// Enforce CI requirements for clean architecture
    fn enforce_ci_requirements(&self, components: &[LegacyComponent]) -> CiEnforcementResult {
        let mut blocking_issues = Vec::new();
        let mut warnings = Vec::new();
        let mut cleanup_recommendations = Vec::new();

        // Check for critical blocking issues
        let critical_components: Vec<_> = components.iter()
            .filter(|c| matches!(c.risk_level, RiskLevel::Critical))
            .collect();

        if !critical_components.is_empty() {
            blocking_issues.push(format!(
                "CRITICAL: {} legacy components must be removed before release",
                critical_components.len()
            ));

            for component in &critical_components {
                blocking_issues.push(format!(
                    "  - {:?} in {}: lines {:?}",
                    component.component_type,
                    component.file_path,
                    component.lines
                ));
            }
        }

        // Check for high priority issues
        let high_priority_components: Vec<_> = components.iter()
            .filter(|c| matches!(c.risk_level, RiskLevel::High))
            .collect();

        if !high_priority_components.is_empty() {
            warnings.push(format!(
                "HIGH PRIORITY: {} legacy components should be addressed",
                high_priority_components.len()
            ));
        }

        // Generate cleanup recommendations
        let not_started_components: Vec<_> = components.iter()
            .filter(|c| matches!(c.migration_status, MigrationStatus::NotStarted))
            .collect();

        if !not_started_components.is_empty() {
            cleanup_recommendations.push("Start migration of unaddressed legacy components".to_string());
            cleanup_recommendations.push("Implement shared binning core pattern consistently".to_string());
            cleanup_recommendations.push("Remove all alternate ECE evaluators".to_string());
            cleanup_recommendations.push("Consolidate duplicated calibration logic".to_string());
        }

        // Enforce "shared binning core only" architecture
        let shared_core_violations = self.detect_shared_core_violations(components);
        if !shared_core_violations.is_empty() {
            blocking_issues.extend(shared_core_violations);
        }

        let passed = blocking_issues.is_empty();

        CiEnforcementResult {
            passed,
            blocking_issues,
            warnings,
            cleanup_recommendations,
        }
    }

    /// Detect violations of "shared binning core only" architecture
    fn detect_shared_core_violations(&self, components: &[LegacyComponent]) -> Vec<String> {
        let mut violations = Vec::new();

        // Check for duplicate binning implementations
        let duplicate_binning = components.iter()
            .filter(|c| matches!(c.component_type, LegacyComponentType::DuplicatedBinning))
            .count();

        if duplicate_binning > 0 {
            violations.push(format!(
                "ARCHITECTURE VIOLATION: {} duplicate binning implementations found (shared core only allowed)",
                duplicate_binning
            ));
        }

        // Check for alternate ECE evaluators
        let alternate_ece = components.iter()
            .filter(|c| matches!(c.component_type, LegacyComponentType::AlternateEceEvaluator))
            .count();

        if alternate_ece > 0 {
            violations.push(format!(
                "ARCHITECTURE VIOLATION: {} alternate ECE evaluators found (standard implementation only)",
                alternate_ece
            ));
        }

        violations
    }

    /// Remove legacy components with safety validation
    pub fn remove_legacy_components(&self, components: &[LegacyComponent]) -> Result<RemovalReport, String> {
        let mut removal_report = RemovalReport {
            attempted_removals: 0,
            successful_removals: 0,
            failed_removals: Vec::new(),
            safety_validations: Vec::new(),
        };

        for component in components {
            if matches!(component.migration_status, MigrationStatus::Completed) {
                continue; // Already handled
            }

            removal_report.attempted_removals += 1;

            // Safety validation before removal
            if let Err(safety_issue) = self.validate_removal_safety(component) {
                removal_report.failed_removals.push(RemovalFailure {
                    component: component.clone(),
                    reason: safety_issue,
                });
                continue;
            }

            // Attempt removal
            match self.execute_component_removal(component) {
                Ok(validation) => {
                    removal_report.successful_removals += 1;
                    removal_report.safety_validations.push(validation);
                }
                Err(error) => {
                    removal_report.failed_removals.push(RemovalFailure {
                        component: component.clone(),
                        reason: error,
                    });
                }
            }
        }

        Ok(removal_report)
    }

    /// Validate migration path completeness
    pub fn validate_migration_paths(&self) -> MigrationPathReport {
        let mut report = MigrationPathReport {
            total_migration_rules: self.required_migrations.len(),
            completed_migrations: 0,
            pending_migrations: Vec::new(),
            failed_validations: Vec::new(),
        };

        for migration_rule in &self.required_migrations {
            match self.validate_migration_rule(migration_rule) {
                Ok(true) => {
                    report.completed_migrations += 1;
                }
                Ok(false) => {
                    report.pending_migrations.push(migration_rule.clone());
                }
                Err(error) => {
                    report.failed_validations.push(MigrationValidationFailure {
                        rule: migration_rule.clone(),
                        error,
                    });
                }
            }
        }

        report
    }

    /// Generate comprehensive CI report
    pub fn generate_ci_report(&self, cleanup_report: &CleanupReport) -> String {
        let status = if cleanup_report.ci_enforcement_result.passed {
            "PASSED"
        } else {
            "FAILED"
        };

        format!(
            r#"
# Legacy Cleanup and CI Enforcement Report

## Status: {}

### Legacy Component Summary
- Total Legacy Components: {}
- Critical Issues: {} 
- High Priority Issues: {}
- Migration Progress: {:.1}%

### CI Enforcement Results
{}

### Blocking Issues
{}

### Warnings
{}

### Cleanup Recommendations
{}

### Architecture Compliance
- Shared Binning Core Only: {}
- Standard ECE Implementation: {}
- No Duplicate Logic: {}

### Next Steps
{}
"#,
            status,
            cleanup_report.total_legacy_components,
            cleanup_report.critical_issues,
            cleanup_report.high_priority_issues,
            cleanup_report.migration_progress,
            if cleanup_report.ci_enforcement_result.passed {
                "✅ All requirements satisfied"
            } else {
                "❌ Requirements violated - blocking release"
            },
            format_string_list(&cleanup_report.ci_enforcement_result.blocking_issues),
            format_string_list(&cleanup_report.ci_enforcement_result.warnings),
            format_string_list(&cleanup_report.ci_enforcement_result.cleanup_recommendations),
            if cleanup_report.ci_enforcement_result.blocking_issues.iter()
                .any(|issue| issue.contains("duplicate binning")) { "❌" } else { "✅" },
            if cleanup_report.ci_enforcement_result.blocking_issues.iter()
                .any(|issue| issue.contains("alternate ECE")) { "❌" } else { "✅" },
            if cleanup_report.ci_enforcement_result.blocking_issues.iter()
                .any(|issue| issue.contains("duplicate")) { "❌" } else { "✅" },
            if cleanup_report.ci_enforcement_result.passed {
                "Ready for release - no blocking legacy issues"
            } else {
                "Address all blocking issues before proceeding with release"
            }
        )
    }

    // Private helper methods

    fn scan_directory<P: AsRef<Path>>(&self, path: P, components: &mut Vec<LegacyComponent>) -> Result<(), std::io::Error> {
        use std::fs;
        use std::io;
        
        let path = path.as_ref();
        if !path.exists() || !path.is_dir() {
            return Ok(()); // Silent return for non-existent directories
        }
        
        fn scan_recursive(
            dir: &Path,
            patterns: &HashMap<LegacyComponentType, Vec<String>>,
            components: &mut Vec<LegacyComponent>
        ) -> io::Result<()> {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    scan_recursive(&path, patterns, components)?;
                } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                    // Read file content for pattern matching
                    if let Ok(content) = fs::read_to_string(&path) {
                        scan_file_content(&path, &content, patterns, components);
                    }
                }
            }
            Ok(())
        }
        
        fn scan_file_content(
            file_path: &Path,
            content: &str,
            patterns: &HashMap<LegacyComponentType, Vec<String>>,
            components: &mut Vec<LegacyComponent>
        ) {
            for (component_type, pattern_list) in patterns {
                for pattern in pattern_list {
                    let mut lines = Vec::new();
                    for (line_num, line) in content.lines().enumerate() {
                        if line.contains(pattern) {
                            lines.push((line_num + 1) as u32);
                        }
                    }
                    
                    if !lines.is_empty() {
                        components.push(LegacyComponent {
                            file_path: file_path.to_string_lossy().to_string(),
                            component_type: component_type.clone(),
                            lines,
                            risk_level: RiskLevel::Low, // Will be assessed later
                            migration_status: MigrationStatus::NotStarted,
                        });
                    }
                }
            }
        }
        
        scan_recursive(path, &self.legacy_patterns, components)?;
        Ok(())
    }

    fn assess_component_risk(&self, component: &mut LegacyComponent) {
        component.risk_level = match component.component_type {
            LegacyComponentType::SimulatorHook => RiskLevel::Critical,
            LegacyComponentType::AlternateEceEvaluator => RiskLevel::Critical,
            LegacyComponentType::DuplicatedBinning => RiskLevel::High,
            LegacyComponentType::ObsoleteCalibratorInterface => RiskLevel::Medium,
            LegacyComponentType::HardcodedTestData => RiskLevel::Low,
        };
    }

    fn check_migration_status(&self, component: &mut LegacyComponent) {
        // Mock implementation - would check actual migration status
        component.migration_status = MigrationStatus::NotStarted;
    }

    fn get_blocked_patterns(&self) -> Vec<String> {
        self.blocked_imports.clone()
    }

    fn validate_removal_safety(&self, _component: &LegacyComponent) -> Result<(), String> {
        // Mock implementation - would validate dependencies
        Ok(())
    }

    fn execute_component_removal(&self, _component: &LegacyComponent) -> Result<SafetyValidation, String> {
        // Mock implementation - would perform actual removal
        Ok(SafetyValidation {
            component_path: "mock_path".to_string(),
            tests_passed: true,
            dependencies_resolved: true,
        })
    }

    fn validate_migration_rule(&self, _rule: &MigrationRule) -> Result<bool, String> {
        // Mock implementation - would validate migration
        Ok(false)
    }
}

// Supporting types

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MigrationRule {
    from_pattern: String,
    to_pattern: String,
    migration_type: MigrationType,
    verification_test: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum MigrationType {
    Replace,
    Consolidate,
    Remove,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemovalReport {
    pub attempted_removals: usize,
    pub successful_removals: usize,
    pub failed_removals: Vec<RemovalFailure>,
    pub safety_validations: Vec<SafetyValidation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemovalFailure {
    pub component: LegacyComponent,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyValidation {
    pub component_path: String,
    pub tests_passed: bool,
    pub dependencies_resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPathReport {
    pub total_migration_rules: usize,
    pub completed_migrations: usize,
    pub pending_migrations: Vec<MigrationRule>,
    pub failed_validations: Vec<MigrationValidationFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationValidationFailure {
    pub rule: MigrationRule,
    pub error: String,
}

fn format_string_list(items: &[String]) -> String {
    if items.is_empty() {
        "None".to_string()
    } else {
        items.iter()
            .enumerate()
            .map(|(i, item)| format!("{}. {}", i + 1, item))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legacy_cleanup_system_creation() {
        let system = LegacyCleanupSystem::new();
        assert!(!system.legacy_patterns.is_empty());
        assert!(!system.blocked_imports.is_empty());
        assert!(!system.required_migrations.is_empty());
    }

    #[test]
    fn test_ci_enforcement_with_critical_issues() {
        let system = LegacyCleanupSystem::new();
        let critical_component = LegacyComponent {
            file_path: "test.rs".to_string(),
            component_type: LegacyComponentType::SimulatorHook,
            lines: vec![10, 15, 20],
            risk_level: RiskLevel::Critical,
            migration_status: MigrationStatus::NotStarted,
        };

        let result = system.enforce_ci_requirements(&[critical_component]);
        assert!(!result.passed);
        assert!(!result.blocking_issues.is_empty());
    }

    #[test]
    fn test_shared_core_violation_detection() {
        let system = LegacyCleanupSystem::new();
        let duplicate_binning = LegacyComponent {
            file_path: "duplicated.rs".to_string(),
            component_type: LegacyComponentType::DuplicatedBinning,
            lines: vec![5],
            risk_level: RiskLevel::High,
            migration_status: MigrationStatus::NotStarted,
        };

        let violations = system.detect_shared_core_violations(&[duplicate_binning]);
        assert!(!violations.is_empty());
        assert!(violations[0].contains("ARCHITECTURE VIOLATION"));
    }
}