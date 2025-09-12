use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use regex::Regex;
use tracing::{error, info, warn};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

/// Legacy Retirement Enforcer for CALIB_V22
/// Ensures complete removal of legacy calibration systems and enforces
/// "shared binning core only" architecture with CI validation
#[derive(Debug, Clone)]
pub struct LegacyRetirementEnforcer {
    project_root: PathBuf,
    config: RetirementConfig,
    legacy_patterns: LegacyPatternRegistry,
    violation_tracker: ViolationTracker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetirementConfig {
    /// Files and directories that MUST NOT exist
    pub forbidden_paths: Vec<String>,
    
    /// Code patterns that indicate legacy usage
    pub forbidden_patterns: Vec<ForbiddenPattern>,
    
    /// Required architecture patterns
    pub required_patterns: Vec<RequiredPattern>,
    
    /// CI enforcement settings
    pub ci_enforcement: CiEnforcementConfig,
    
    /// Exception handling
    pub temporary_exceptions: Vec<TemporaryException>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForbiddenPattern {
    pub name: String,
    pub pattern: String,
    pub severity: ViolationSeverity,
    pub message: String,
    pub suggested_replacement: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredPattern {
    pub name: String,
    pub description: String,
    pub validation_regex: String,
    pub required_in_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiEnforcementConfig {
    pub fail_on_legacy_links: bool,
    pub fail_on_forbidden_imports: bool,
    pub fail_on_missing_shared_core: bool,
    pub generate_violation_report: bool,
    pub auto_fix_violations: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporaryException {
    pub path: String,
    pub reason: String,
    pub expires_at: String, // ISO date
    pub approved_by: String,
}

#[derive(Debug, Clone)]
struct LegacyPatternRegistry {
    /// Legacy calibration system patterns to detect and remove
    legacy_imports: Vec<String>,
    legacy_function_calls: Vec<String>,
    legacy_class_names: Vec<String>,
    legacy_config_keys: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct ViolationTracker {
    violations: Vec<ArchitectureViolation>,
    files_scanned: usize,
    legacy_references_found: usize,
    auto_fixes_applied: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureViolation {
    pub violation_type: ViolationType,
    pub file_path: String,
    pub line_number: Option<usize>,
    pub content: String,
    pub severity: ViolationSeverity,
    pub message: String,
    pub suggested_fix: Option<String>,
    pub auto_fixable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViolationType {
    ForbiddenImport,
    LegacyFunctionCall,
    LegacyClassUsage,
    MissingSharedCore,
    ForbiddenPath,
    InvalidArchitecture,
    DeprecatedConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViolationSeverity {
    Error,   // CI must fail
    Warning, // CI warning but continues
    Info,    // Informational only
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RetirementReport {
    pub timestamp: String,
    pub total_files_scanned: usize,
    pub violations_found: usize,
    pub violations_by_type: HashMap<ViolationType, usize>,
    pub violations_by_severity: HashMap<ViolationSeverity, usize>,
    pub auto_fixes_applied: usize,
    pub ci_should_fail: bool,
    pub violations: Vec<ArchitectureViolation>,
    pub retirement_status: RetirementStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RetirementStatus {
    Complete,      // All legacy systems removed
    InProgress,    // Some violations remain
    Critical,      // Critical violations block deployment
    Blocked,       // Systematic issues prevent progress
}

impl Default for RetirementConfig {
    fn default() -> Self {
        Self {
            forbidden_paths: vec![
                "src/calibration/legacy/".to_string(),
                "src/calibration/old_platt.rs".to_string(),
                "src/calibration/deprecated/".to_string(),
                "src/calibration/v1/".to_string(),
                "src/calibration/simulator/".to_string(), // Legacy simulators
            ],
            forbidden_patterns: vec![
                ForbiddenPattern {
                    name: "legacy_platt_import".to_string(),
                    pattern: r"use.*legacy.*platt".to_string(),
                    severity: ViolationSeverity::Error,
                    message: "Legacy Platt calibrator imports are forbidden".to_string(),
                    suggested_replacement: Some("use crate::calibration::platt::PlattCalibrator".to_string()),
                },
                ForbiddenPattern {
                    name: "simulator_linkage".to_string(),
                    pattern: r"Simulator::|::Simulator".to_string(),
                    severity: ViolationSeverity::Error,
                    message: "Legacy simulator linkage detected - must use shared binning core only".to_string(),
                    suggested_replacement: Some("Use shared calibration core through BinningCore interface".to_string()),
                },
                ForbiddenPattern {
                    name: "deprecated_config".to_string(),
                    pattern: r"use_legacy_calibration.*=.*true".to_string(),
                    severity: ViolationSeverity::Error,
                    message: "Legacy calibration configuration must be removed".to_string(),
                    suggested_replacement: Some("Remove legacy configuration entirely".to_string()),
                },
            ],
            required_patterns: vec![
                RequiredPattern {
                    name: "shared_binning_core".to_string(),
                    description: "All calibration must use shared binning core".to_string(),
                    validation_regex: r"BinningCore::|SharedBinning::".to_string(),
                    required_in_files: vec!["src/calibration/**/*.rs".to_string()],
                },
            ],
            ci_enforcement: CiEnforcementConfig {
                fail_on_legacy_links: true,
                fail_on_forbidden_imports: true,
                fail_on_missing_shared_core: true,
                generate_violation_report: true,
                auto_fix_violations: false, // Disabled for safety
            },
            temporary_exceptions: Vec::new(),
        }
    }
}

impl Default for LegacyPatternRegistry {
    fn default() -> Self {
        Self {
            legacy_imports: vec![
                "use.*legacy.*calibration".to_string(),
                "use.*deprecated.*platt".to_string(),
                "use.*simulator.*calibration".to_string(),
                "use.*old.*isotonic".to_string(),
            ],
            legacy_function_calls: vec![
                "legacy_calibrate\\(".to_string(),
                "deprecated_platt_fit\\(".to_string(),
                "simulator_predict\\(".to_string(),
                "old_isotonic_map\\(".to_string(),
            ],
            legacy_class_names: vec![
                "LegacyPlattCalibrator".to_string(),
                "DeprecatedIsotonic".to_string(),
                "SimulatorCalibration".to_string(),
                "OldBinningCore".to_string(),
            ],
            legacy_config_keys: vec![
                "use_legacy_calibration".to_string(),
                "enable_simulator_mode".to_string(),
                "deprecated_platt_settings".to_string(),
                "old_isotonic_config".to_string(),
            ],
        }
    }
}

impl LegacyRetirementEnforcer {
    /// Create new legacy retirement enforcer
    pub fn new(project_root: PathBuf) -> Result<Self, RetirementError> {
        if !project_root.exists() {
            return Err(RetirementError::InvalidProjectRoot(
                format!("Project root does not exist: {}", project_root.display())
            ));
        }

        Ok(Self {
            project_root,
            config: RetirementConfig::default(),
            legacy_patterns: LegacyPatternRegistry::default(),
            violation_tracker: ViolationTracker::default(),
        })
    }

    /// Execute comprehensive legacy retirement validation
    pub async fn enforce_legacy_retirement(&mut self) -> Result<RetirementReport, RetirementError> {
        info!("🔍 Starting comprehensive legacy retirement enforcement");
        
        // Step 1: Check for forbidden paths
        self.check_forbidden_paths().await?;
        
        // Step 2: Scan code for legacy patterns
        self.scan_code_for_legacy_patterns().await?;
        
        // Step 3: Validate shared binning core usage
        self.validate_shared_binning_core_architecture().await?;
        
        // Step 4: Check CI/CD configuration
        self.validate_ci_configuration().await?;
        
        // Step 5: Apply auto-fixes if enabled
        if self.config.ci_enforcement.auto_fix_violations {
            self.apply_auto_fixes().await?;
        }
        
        // Generate comprehensive report
        let report = self.generate_retirement_report().await?;
        
        info!("✅ Legacy retirement enforcement completed: {} violations found", 
              report.violations_found);
        
        Ok(report)
    }

    /// Check for existence of forbidden paths and files
    async fn check_forbidden_paths(&mut self) -> Result<(), RetirementError> {
        info!("📁 Checking for forbidden paths and files");
        
        for forbidden_path in &self.config.forbidden_paths.clone() {
            let full_path = self.project_root.join(forbidden_path);
            
            if full_path.exists() {
                let violation = ArchitectureViolation {
                    violation_type: ViolationType::ForbiddenPath,
                    file_path: forbidden_path.clone(),
                    line_number: None,
                    content: format!("Forbidden path exists: {}", forbidden_path),
                    severity: ViolationSeverity::Error,
                    message: format!("Legacy path '{}' must be completely removed", forbidden_path),
                    suggested_fix: Some(format!("rm -rf {}", forbidden_path)),
                    auto_fixable: true,
                };
                
                self.violation_tracker.violations.push(violation);
                warn!("❌ Found forbidden path: {}", forbidden_path);
            }
        }
        
        Ok(())
    }

    /// Scan source code for legacy patterns
    async fn scan_code_for_legacy_patterns(&mut self) -> Result<(), RetirementError> {
        info!("🔍 Scanning source code for legacy patterns");
        
        let source_extensions = vec!["rs", "ts", "js", "toml", "yaml", "yml", "json"];
        
        for entry in WalkDir::new(&self.project_root)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            
            // Skip non-source files
            let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if !source_extensions.contains(&extension) {
                continue;
            }
            
            // Skip excluded directories
            if self.should_skip_path(path) {
                continue;
            }
            
            self.scan_file_for_violations(path).await?;
            self.violation_tracker.files_scanned += 1;
        }
        
        info!("📊 Scanned {} files for legacy patterns", self.violation_tracker.files_scanned);
        Ok(())
    }

    /// Scan individual file for legacy pattern violations
    async fn scan_file_for_violations(&mut self, file_path: &Path) -> Result<(), RetirementError> {
        let content = fs::read_to_string(file_path)
            .map_err(|e| RetirementError::FileReadError(format!("Failed to read {}: {}", file_path.display(), e)))?;
        
        let lines: Vec<&str> = content.lines().collect();
        let file_path_str = file_path.strip_prefix(&self.project_root)
            .unwrap_or(file_path)
            .to_string_lossy()
            .to_string();
        
        // Check forbidden patterns
        for pattern_config in &self.config.forbidden_patterns.clone() {
            let regex = Regex::new(&pattern_config.pattern)
                .map_err(|e| RetirementError::RegexError(format!("Invalid pattern '{}': {}", pattern_config.pattern, e)))?;
            
            for (line_num, line) in lines.iter().enumerate() {
                if regex.is_match(line) {
                    let violation = ArchitectureViolation {
                        violation_type: self.pattern_to_violation_type(&pattern_config.name),
                        file_path: file_path_str.clone(),
                        line_number: Some(line_num + 1),
                        content: line.to_string(),
                        severity: pattern_config.severity.clone(),
                        message: pattern_config.message.clone(),
                        suggested_fix: pattern_config.suggested_replacement.clone(),
                        auto_fixable: pattern_config.suggested_replacement.is_some(),
                    };
                    
                    self.violation_tracker.violations.push(violation);
                    self.violation_tracker.legacy_references_found += 1;
                }
            }
        }
        
        // Check legacy patterns from registry
        self.check_legacy_imports(&lines, &file_path_str).await?;
        self.check_legacy_function_calls(&lines, &file_path_str).await?;
        self.check_legacy_class_usage(&lines, &file_path_str).await?;
        
        Ok(())
    }

    /// Check for legacy import statements
    async fn check_legacy_imports(&mut self, lines: &[&str], file_path: &str) -> Result<(), RetirementError> {
        for (line_num, line) in lines.iter().enumerate() {
            for legacy_import in &self.legacy_patterns.legacy_imports {
                let regex = Regex::new(legacy_import)
                    .map_err(|e| RetirementError::RegexError(e.to_string()))?;
                
                if regex.is_match(line) {
                    let violation = ArchitectureViolation {
                        violation_type: ViolationType::ForbiddenImport,
                        file_path: file_path.to_string(),
                        line_number: Some(line_num + 1),
                        content: line.to_string(),
                        severity: ViolationSeverity::Error,
                        message: "Legacy calibration import detected".to_string(),
                        suggested_fix: Some("Replace with shared binning core import".to_string()),
                        auto_fixable: false,
                    };
                    
                    self.violation_tracker.violations.push(violation);
                }
            }
        }
        Ok(())
    }

    /// Check for legacy function calls
    async fn check_legacy_function_calls(&mut self, lines: &[&str], file_path: &str) -> Result<(), RetirementError> {
        for (line_num, line) in lines.iter().enumerate() {
            for legacy_func in &self.legacy_patterns.legacy_function_calls {
                let regex = Regex::new(legacy_func)
                    .map_err(|e| RetirementError::RegexError(e.to_string()))?;
                
                if regex.is_match(line) {
                    let violation = ArchitectureViolation {
                        violation_type: ViolationType::LegacyFunctionCall,
                        file_path: file_path.to_string(),
                        line_number: Some(line_num + 1),
                        content: line.to_string(),
                        severity: ViolationSeverity::Error,
                        message: "Legacy function call detected".to_string(),
                        suggested_fix: Some("Replace with shared binning core equivalent".to_string()),
                        auto_fixable: false,
                    };
                    
                    self.violation_tracker.violations.push(violation);
                }
            }
        }
        Ok(())
    }

    /// Check for legacy class usage
    async fn check_legacy_class_usage(&mut self, lines: &[&str], file_path: &str) -> Result<(), RetirementError> {
        for (line_num, line) in lines.iter().enumerate() {
            for legacy_class in &self.legacy_patterns.legacy_class_names {
                if line.contains(legacy_class) {
                    let violation = ArchitectureViolation {
                        violation_type: ViolationType::LegacyClassUsage,
                        file_path: file_path.to_string(),
                        line_number: Some(line_num + 1),
                        content: line.to_string(),
                        severity: ViolationSeverity::Error,
                        message: "Legacy class usage detected".to_string(),
                        suggested_fix: Some("Replace with shared binning core class".to_string()),
                        auto_fixable: false,
                    };
                    
                    self.violation_tracker.violations.push(violation);
                }
            }
        }
        Ok(())
    }

    /// Validate shared binning core architecture compliance
    async fn validate_shared_binning_core_architecture(&mut self) -> Result<(), RetirementError> {
        info!("🏗️  Validating shared binning core architecture");
        
        // Check that all calibration files use the shared core
        let calibration_files = self.find_calibration_files().await?;
        
        for file_path in &calibration_files {
            let content = fs::read_to_string(&self.project_root.join(file_path))
                .map_err(|e| RetirementError::FileReadError(e.to_string()))?;
            
            let has_shared_core = self.config.required_patterns.iter()
                .any(|pattern| {
                    Regex::new(&pattern.validation_regex)
                        .map(|regex| regex.is_match(&content))
                        .unwrap_or(false)
                });
            
            if !has_shared_core {
                let violation = ArchitectureViolation {
                    violation_type: ViolationType::MissingSharedCore,
                    file_path: file_path.clone(),
                    line_number: None,
                    content: "File does not use shared binning core".to_string(),
                    severity: ViolationSeverity::Error,
                    message: "Calibration file must use shared binning core architecture".to_string(),
                    suggested_fix: Some("Add shared binning core import and usage".to_string()),
                    auto_fixable: false,
                };
                
                self.violation_tracker.violations.push(violation);
            }
        }
        
        Ok(())
    }

    /// Validate CI/CD configuration for legacy retirement enforcement
    async fn validate_ci_configuration(&mut self) -> Result<(), RetirementError> {
        info!("⚙️  Validating CI/CD configuration");
        
        let ci_files = vec![
            ".github/workflows/ci.yml",
            ".github/workflows/test.yml",
            "Cargo.toml",
            "package.json",
        ];
        
        for ci_file in &ci_files {
            let ci_path = self.project_root.join(ci_file);
            if ci_path.exists() {
                self.validate_ci_file(&ci_path).await?;
            }
        }
        
        Ok(())
    }

    /// Validate specific CI configuration file
    async fn validate_ci_file(&mut self, file_path: &Path) -> Result<(), RetirementError> {
        let content = fs::read_to_string(file_path)
            .map_err(|e| RetirementError::FileReadError(e.to_string()))?;
        
        // Check for legacy configuration keys
        for legacy_key in &self.legacy_patterns.legacy_config_keys {
            if content.contains(legacy_key) {
                let violation = ArchitectureViolation {
                    violation_type: ViolationType::DeprecatedConfig,
                    file_path: file_path.to_string_lossy().to_string(),
                    line_number: None,
                    content: format!("Deprecated config key: {}", legacy_key),
                    severity: ViolationSeverity::Error,
                    message: "Legacy configuration key must be removed from CI".to_string(),
                    suggested_fix: Some(format!("Remove '{}' from configuration", legacy_key)),
                    auto_fixable: true,
                };
                
                self.violation_tracker.violations.push(violation);
            }
        }
        
        Ok(())
    }

    /// Apply automatic fixes for violations that can be safely auto-fixed
    async fn apply_auto_fixes(&mut self) -> Result<(), RetirementError> {
        if !self.config.ci_enforcement.auto_fix_violations {
            return Ok(());
        }
        
        info!("🔧 Applying automatic fixes for violations");
        
        for violation in &self.violation_tracker.violations.clone() {
            if violation.auto_fixable {
                match self.apply_single_auto_fix(violation).await {
                    Ok(()) => {
                        self.violation_tracker.auto_fixes_applied += 1;
                        info!("✅ Auto-fixed violation in {}", violation.file_path);
                    }
                    Err(e) => {
                        warn!("⚠️  Failed to auto-fix violation in {}: {}", violation.file_path, e);
                    }
                }
            }
        }
        
        info!("🔧 Applied {} automatic fixes", self.violation_tracker.auto_fixes_applied);
        Ok(())
    }

    /// Apply a single automatic fix
    async fn apply_single_auto_fix(&self, violation: &ArchitectureViolation) -> Result<(), RetirementError> {
        match violation.violation_type {
            ViolationType::ForbiddenPath => {
                let path = self.project_root.join(&violation.file_path);
                if path.exists() {
                    if path.is_dir() {
                        fs::remove_dir_all(&path)
                            .map_err(|e| RetirementError::AutoFixError(format!("Failed to remove directory {}: {}", path.display(), e)))?;
                    } else {
                        fs::remove_file(&path)
                            .map_err(|e| RetirementError::AutoFixError(format!("Failed to remove file {}: {}", path.display(), e)))?;
                    }
                }
            }
            _ => {
                // Other auto-fixes would be implemented here
                return Err(RetirementError::AutoFixError("Auto-fix not implemented for this violation type".to_string()));
            }
        }
        Ok(())
    }

    /// Generate comprehensive retirement report
    async fn generate_retirement_report(&self) -> Result<RetirementReport, RetirementError> {
        let mut violations_by_type = HashMap::new();
        let mut violations_by_severity = HashMap::new();
        
        for violation in &self.violation_tracker.violations {
            *violations_by_type.entry(violation.violation_type.clone()).or_insert(0) += 1;
            *violations_by_severity.entry(violation.severity.clone()).or_insert(0) += 1;
        }
        
        let error_count = violations_by_severity.get(&ViolationSeverity::Error).copied().unwrap_or(0);
        let ci_should_fail = error_count > 0 && self.config.ci_enforcement.fail_on_legacy_links;
        
        let retirement_status = self.determine_retirement_status(error_count);
        
        Ok(RetirementReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            total_files_scanned: self.violation_tracker.files_scanned,
            violations_found: self.violation_tracker.violations.len(),
            violations_by_type,
            violations_by_severity,
            auto_fixes_applied: self.violation_tracker.auto_fixes_applied,
            ci_should_fail,
            violations: self.violation_tracker.violations.clone(),
            retirement_status,
        })
    }

    /// Determine overall retirement status based on violations
    fn determine_retirement_status(&self, error_count: usize) -> RetirementStatus {
        if error_count == 0 && self.violation_tracker.violations.is_empty() {
            RetirementStatus::Complete
        } else if error_count > 10 {
            RetirementStatus::Critical
        } else if error_count > 0 {
            RetirementStatus::InProgress
        } else {
            RetirementStatus::InProgress
        }
    }

    /// Check if CI should fail based on current violations
    pub fn should_fail_ci(&self) -> bool {
        let error_violations = self.violation_tracker.violations.iter()
            .any(|v| v.severity == ViolationSeverity::Error);
        
        error_violations && (
            self.config.ci_enforcement.fail_on_legacy_links ||
            self.config.ci_enforcement.fail_on_forbidden_imports ||
            self.config.ci_enforcement.fail_on_missing_shared_core
        )
    }

    /// Generate CI failure message
    pub fn generate_ci_failure_message(&self) -> String {
        let error_count = self.violation_tracker.violations.iter()
            .filter(|v| v.severity == ViolationSeverity::Error)
            .count();
        
        format!(
            "🚨 Legacy Retirement Enforcement FAILED\n\
             Found {} critical violations that must be fixed before deployment:\n\
             - {} legacy imports/references detected\n\
             - Use 'cargo run --bin legacy-retirement-checker' for detailed report\n\
             - All legacy calibration systems must be completely removed\n\
             - Only shared binning core architecture is permitted",
            error_count,
            self.violation_tracker.legacy_references_found
        )
    }

    // Helper methods

    fn should_skip_path(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        path_str.contains("target/") ||
        path_str.contains(".git/") ||
        path_str.contains("node_modules/") ||
        path_str.contains("coverage/")
    }

    async fn find_calibration_files(&self) -> Result<Vec<String>, RetirementError> {
        let mut calibration_files = Vec::new();
        
        for entry in WalkDir::new(self.project_root.join("src/calibration"))
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Ok(relative_path) = path.strip_prefix(&self.project_root) {
                    calibration_files.push(relative_path.to_string_lossy().to_string());
                }
            }
        }
        
        Ok(calibration_files)
    }

    fn pattern_to_violation_type(&self, pattern_name: &str) -> ViolationType {
        match pattern_name {
            name if name.contains("import") => ViolationType::ForbiddenImport,
            name if name.contains("simulator") => ViolationType::LegacyFunctionCall,
            name if name.contains("config") => ViolationType::DeprecatedConfig,
            _ => ViolationType::InvalidArchitecture,
        }
    }

    /// Export report as JSON for CI integration
    pub fn export_report_json(&self, report: &RetirementReport) -> Result<String, RetirementError> {
        serde_json::to_string_pretty(report)
            .map_err(|e| RetirementError::SerializationError(e.to_string()))
    }

    /// Export report as GitHub-compatible format
    pub fn export_github_annotations(&self, report: &RetirementReport) -> Vec<String> {
        let mut annotations = Vec::new();
        
        for violation in &report.violations {
            let level = match violation.severity {
                ViolationSeverity::Error => "error",
                ViolationSeverity::Warning => "warning",
                ViolationSeverity::Info => "notice",
            };
            
            let annotation = if let Some(line_num) = violation.line_number {
                format!(
                    "::{} file={},line={}::{}", 
                    level, violation.file_path, line_num, violation.message
                )
            } else {
                format!(
                    "::{} file={}::{}", 
                    level, violation.file_path, violation.message
                )
            };
            
            annotations.push(annotation);
        }
        
        annotations
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RetirementError {
    #[error("Invalid project root: {0}")]
    InvalidProjectRoot(String),
    
    #[error("File read error: {0}")]
    FileReadError(String),
    
    #[error("Regex error: {0}")]
    RegexError(String),
    
    #[error("Auto-fix error: {0}")]
    AutoFixError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("CI enforcement failed: {0}")]
    CiEnforcementError(String),
}

/// CLI binary for running legacy retirement checks
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::init();
    
    let args: Vec<String> = std::env::args().collect();
    let project_root = args.get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    
    let mut enforcer = LegacyRetirementEnforcer::new(project_root)?;
    let report = enforcer.enforce_legacy_retirement().await?;
    
    // Print summary
    println!("🔍 Legacy Retirement Enforcement Report");
    println!("=======================================");
    println!("Files scanned: {}", report.total_files_scanned);
    println!("Violations found: {}", report.violations_found);
    println!("Auto-fixes applied: {}", report.auto_fixes_applied);
    println!("Retirement status: {:?}", report.retirement_status);
    
    if report.ci_should_fail {
        eprintln!("\n{}", enforcer.generate_ci_failure_message());
        std::process::exit(1);
    } else {
        println!("✅ Legacy retirement validation passed");
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_enforcer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let enforcer = LegacyRetirementEnforcer::new(temp_dir.path().to_path_buf());
        assert!(enforcer.is_ok());
    }

    #[tokio::test]
    async fn test_forbidden_path_detection() {
        let temp_dir = TempDir::new().unwrap();
        let legacy_dir = temp_dir.path().join("src/calibration/legacy");
        fs::create_dir_all(&legacy_dir).unwrap();
        fs::write(legacy_dir.join("old_file.rs"), "// legacy code").unwrap();
        
        let mut enforcer = LegacyRetirementEnforcer::new(temp_dir.path().to_path_buf()).unwrap();
        enforcer.check_forbidden_paths().await.unwrap();
        
        assert!(!enforcer.violation_tracker.violations.is_empty());
        assert!(enforcer.violation_tracker.violations.iter()
            .any(|v| v.violation_type == ViolationType::ForbiddenPath));
    }

    #[tokio::test]
    async fn test_legacy_pattern_detection() {
        let temp_dir = TempDir::new().unwrap();
        let src_dir = temp_dir.path().join("src");
        fs::create_dir_all(&src_dir).unwrap();
        
        let test_file = src_dir.join("test.rs");
        fs::write(&test_file, "use legacy::platt::LegacyPlattCalibrator;\n").unwrap();
        
        let mut enforcer = LegacyRetirementEnforcer::new(temp_dir.path().to_path_buf()).unwrap();
        enforcer.scan_code_for_legacy_patterns().await.unwrap();
        
        assert!(enforcer.violation_tracker.legacy_references_found > 0);
    }

    #[tokio::test]
    async fn test_ci_failure_detection() {
        let temp_dir = TempDir::new().unwrap();
        let mut enforcer = LegacyRetirementEnforcer::new(temp_dir.path().to_path_buf()).unwrap();
        
        // Add a critical violation
        enforcer.violation_tracker.violations.push(ArchitectureViolation {
            violation_type: ViolationType::ForbiddenImport,
            file_path: "test.rs".to_string(),
            line_number: Some(1),
            content: "use legacy::calibration".to_string(),
            severity: ViolationSeverity::Error,
            message: "Legacy import".to_string(),
            suggested_fix: None,
            auto_fixable: false,
        });
        
        assert!(enforcer.should_fail_ci());
    }

    #[tokio::test]
    async fn test_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let mut enforcer = LegacyRetirementEnforcer::new(temp_dir.path().to_path_buf()).unwrap();
        
        let report = enforcer.generate_retirement_report().await.unwrap();
        assert_eq!(report.violations_found, 0);
        assert_eq!(report.retirement_status, RetirementStatus::Complete);
    }
}