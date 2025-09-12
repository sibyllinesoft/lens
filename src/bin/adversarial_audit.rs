//! # Adversarial Audit Binary
//!
//! Command-line interface for executing comprehensive adversarial testing
//! as specified in TODO.md Step 4 - Adversarial audit.
//!
//! Usage:
//! ```bash
//! cargo run --bin adversarial_audit [OPTIONS]
//! ```
//!
//! This binary orchestrates all adversarial test suites and validates
//! the critical gate requirements:
//! - span=100% (complete corpus coverage)
//! - SLA-Recall@50 flat (no degradation under adversarial conditions)
//! - p99/p95 ≤ 2.0 (latency stability under stress)

use anyhow::{Context, Result};
use clap::Parser;
use lens_core::adversarial::{
    AdversarialOrchestrator, AdversarialConfig, validate_adversarial_gates,
    calculate_robustness_score,
};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, Level};
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "adversarial_audit")]
#[command(about = "Execute comprehensive adversarial testing for lens search system")]
#[command(version = "1.0.0")]
struct Args {
    /// Path to the base corpus directory
    #[arg(short = 'c', long, default_value = "./indexed-content")]
    corpus_path: PathBuf,

    /// Output directory for test results and artifacts
    #[arg(short = 'o', long, default_value = "./adversarial-results")]
    output_path: PathBuf,

    /// Enable clone-heavy repository testing
    #[arg(long, default_value = "true")]
    enable_clone_testing: bool,

    /// Enable vendored bloat scenario testing
    #[arg(long, default_value = "true")]
    enable_bloat_testing: bool,

    /// Enable large JSON/data file noise testing
    #[arg(long, default_value = "true")]
    enable_noise_testing: bool,

    /// Enable system stress testing
    #[arg(long, default_value = "true")]
    enable_stress_testing: bool,

    /// Execute test suites in parallel (higher resource usage)
    #[arg(long, default_value = "false")]
    parallel_execution: bool,

    /// Timeout for entire audit in minutes
    #[arg(short = 't', long, default_value = "30")]
    timeout_minutes: u64,

    /// Memory limit in GB
    #[arg(short = 'm', long, default_value = "12")]
    memory_limit_gb: u64,

    /// Clean up test artifacts after completion
    #[arg(long, default_value = "true")]
    cleanup_artifacts: bool,

    /// Generate detailed JSON report
    #[arg(long, default_value = "true")]
    json_report: bool,

    /// Generate human-readable markdown report
    #[arg(long, default_value = "true")]
    markdown_report: bool,

    /// Verbose logging output
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Fail fast on first gate violation
    #[arg(long, default_value = "false")]
    fail_fast: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { Level::DEBUG } else { Level::INFO };
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    info!("🎭 Lens Adversarial Audit System v1.0.0");
    info!("📊 Corpus: {}", args.corpus_path.display());
    info!("📂 Output: {}", args.output_path.display());

    // Validate input parameters
    validate_arguments(&args)?;

    // Create adversarial configuration
    let config = create_adversarial_config(&args);
    
    // Execute comprehensive adversarial audit
    let execution_start = Instant::now();
    let audit_result = execute_adversarial_audit(config).await?;
    let total_execution_time = execution_start.elapsed();

    // Validate gate requirements
    let gates_passed = validate_adversarial_gates(&audit_result)?;
    let robustness_score = calculate_robustness_score(&audit_result);

    // Generate reports
    if args.json_report {
        generate_json_report(&audit_result, &args.output_path).await?;
    }

    if args.markdown_report {
        generate_markdown_report(&audit_result, &args.output_path, total_execution_time).await?;
    }

    // Log final summary
    log_final_summary(&audit_result, gates_passed, robustness_score, total_execution_time);

    // Exit with appropriate code
    if gates_passed {
        info!("✅ All adversarial gates PASSED - System ready for production");
        std::process::exit(0);
    } else {
        error!("❌ Some adversarial gates FAILED - System requires improvements");
        if args.fail_fast {
            std::process::exit(1);
        } else {
            std::process::exit(2); // Different exit code for gate failures
        }
    }
}

fn validate_arguments(args: &Args) -> Result<()> {
    // Validate corpus path exists
    if !args.corpus_path.exists() {
        return Err(anyhow::anyhow!(
            "Corpus path does not exist: {}",
            args.corpus_path.display()
        ));
    }

    // Validate corpus contains files
    let corpus_files = std::fs::read_dir(&args.corpus_path)
        .context("Failed to read corpus directory")?
        .count();
    
    if corpus_files == 0 {
        return Err(anyhow::anyhow!(
            "Corpus directory is empty: {}",
            args.corpus_path.display()
        ));
    }

    info!("📁 Corpus validation: {} items found", corpus_files);

    // Validate resource limits are reasonable
    if args.memory_limit_gb < 4 {
        return Err(anyhow::anyhow!(
            "Memory limit too low: {}GB (minimum 4GB required)",
            args.memory_limit_gb
        ));
    }

    if args.timeout_minutes < 10 {
        return Err(anyhow::anyhow!(
            "Timeout too low: {}min (minimum 10min required)",
            args.timeout_minutes
        ));
    }

    // Validate at least one test suite is enabled
    if !args.enable_clone_testing 
        && !args.enable_bloat_testing 
        && !args.enable_noise_testing 
        && !args.enable_stress_testing {
        return Err(anyhow::anyhow!(
            "At least one test suite must be enabled"
        ));
    }

    Ok(())
}

fn create_adversarial_config(args: &Args) -> AdversarialConfig {
    AdversarialConfig {
        base_corpus_path: args.corpus_path.clone(),
        output_base_path: args.output_path.clone(),
        enable_clone_testing: args.enable_clone_testing,
        enable_bloat_testing: args.enable_bloat_testing,
        enable_noise_testing: args.enable_noise_testing,
        enable_stress_testing: args.enable_stress_testing,
        parallel_execution: args.parallel_execution,
        timeout_minutes: args.timeout_minutes,
        memory_limit_gb: args.memory_limit_gb,
        cleanup_artifacts: args.cleanup_artifacts,
    }
}

async fn execute_adversarial_audit(config: AdversarialConfig) -> Result<lens_core::adversarial::AdversarialAuditResult> {
    info!("🚀 Starting comprehensive adversarial audit");
    
    let mut orchestrator = AdversarialOrchestrator::new(config);
    
    let result = orchestrator.execute_full_audit().await
        .context("Adversarial audit execution failed")?;
    
    info!("✅ Adversarial audit execution completed");
    Ok(result)
}

async fn generate_json_report(
    result: &lens_core::adversarial::AdversarialAuditResult, 
    output_path: &PathBuf
) -> Result<()> {
    let report_path = output_path.join("adversarial_audit_report.json");
    
    let json_content = serde_json::to_string_pretty(result)
        .context("Failed to serialize audit results to JSON")?;
    
    tokio::fs::write(&report_path, json_content).await
        .context("Failed to write JSON report")?;
    
    info!("📄 JSON report generated: {}", report_path.display());
    Ok(())
}

async fn generate_markdown_report(
    result: &lens_core::adversarial::AdversarialAuditResult,
    output_path: &PathBuf,
    execution_time: std::time::Duration,
) -> Result<()> {
    let report_path = output_path.join("adversarial_audit_report.md");
    
    let markdown_content = format_markdown_report(result, execution_time)?;
    
    tokio::fs::write(&report_path, markdown_content).await
        .context("Failed to write markdown report")?;
    
    info!("📝 Markdown report generated: {}", report_path.display());
    Ok(())
}

fn format_markdown_report(
    result: &lens_core::adversarial::AdversarialAuditResult,
    execution_time: std::time::Duration,
) -> Result<String> {
    let mut report = String::new();
    
    report.push_str("# Lens Adversarial Audit Report\n\n");
    report.push_str(&format!("**Generated:** {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    report.push_str(&format!("**Execution Time:** {:.1}s\n\n", execution_time.as_secs_f64()));
    
    // Overall metrics
    let metrics = &result.overall_metrics;
    report.push_str("## Overall Metrics\n\n");
    report.push_str(&format!("- **Span Coverage:** {:.1}%\n", metrics.span_coverage_pct));
    report.push_str(&format!("- **SLA-Recall@50:** {:.3}\n", metrics.sla_recall_at_50));
    report.push_str(&format!("- **p99 Latency:** {:.1}ms\n", metrics.p99_latency_ms));
    report.push_str(&format!("- **p95 Latency:** {:.1}ms\n", metrics.p95_latency_ms));
    report.push_str(&format!("- **Degradation Factor:** {:.2}x\n", metrics.degradation_factor));
    report.push_str(&format!("- **Robustness Score:** {:.2}\n\n", metrics.robustness_score));
    
    // Gate validation results
    let gates = &result.gate_validation;
    report.push_str("## Gate Validation Results\n\n");
    
    let span_status = if gates.span_coverage_gate { "✅ PASS" } else { "❌ FAIL" };
    let recall_status = if gates.sla_recall_gate { "✅ PASS" } else { "❌ FAIL" };
    let latency_status = if gates.latency_stability_gate { "✅ PASS" } else { "❌ FAIL" };
    
    report.push_str(&format!("- **Span Coverage (100%):** {}\n", span_status));
    report.push_str(&format!("- **SLA-Recall@50 Flat:** {}\n", recall_status));
    report.push_str(&format!("- **Latency Stability (p99/p95 ≤ 2.0):** {}\n", latency_status));
    
    let overall_status = if gates.overall_pass { "✅ PASS" } else { "❌ FAIL" };
    report.push_str(&format!("- **Overall Gate Status:** {}\n\n", overall_status));
    
    if !gates.violations.is_empty() {
        report.push_str("### Gate Violations\n\n");
        for violation in &gates.violations {
            report.push_str(&format!("- {}\n", violation));
        }
        report.push_str("\n");
    }
    
    // Clone testing results
    report.push_str("## Clone Testing Results\n\n");
    let clone = &result.clone_results;
    report.push_str(&format!("- **Max Duplication Handled:** {}x\n", clone.overall_metrics.max_duplication_handled));
    report.push_str(&format!("- **Deduplication Effectiveness:** {:.2}\n", clone.overall_metrics.deduplication_effectiveness));
    report.push_str(&format!("- **Search Consistency Score:** {:.2}\n", clone.overall_metrics.search_consistency_score));
    report.push_str(&format!("- **Resource Efficiency Score:** {:.2}\n\n", clone.overall_metrics.resource_efficiency_score));
    
    // Bloat testing results
    report.push_str("## Bloat Testing Results\n\n");
    let bloat = &result.bloat_results;
    report.push_str(&format!("- **Filtering Effectiveness:** {:.1}%\n", bloat.filtering_effectiveness.overall_precision * 100.0));
    report.push_str(&format!("- **Search Latency Multiplier:** {:.2}x\n", bloat.performance_impact.search_latency_multiplier));
    report.push_str(&format!("- **Result Quality Score:** {:.2}\n", bloat.performance_impact.result_quality_score));
    report.push_str(&format!("- **Resource Efficiency:** {:.2}\n\n", bloat.performance_impact.resource_efficiency));
    
    // Noise testing results  
    report.push_str("## Noise Testing Results\n\n");
    let noise = &result.noise_results;
    report.push_str(&format!("- **Average Parse Time:** {:.1}ms\n", noise.parsing_performance.average_parse_time_ms));
    report.push_str(&format!("- **Parsing Throughput:** {:.1} MB/s\n", noise.parsing_performance.parsing_throughput_mb_per_sec));
    report.push_str(&format!("- **Content Filtering Precision:** {:.1}%\n", noise.content_filtering.overall_precision * 100.0));
    report.push_str(&format!("- **System Stability Score:** {:.2}\n\n", noise.robustness_metrics.system_stability_under_load));
    
    // Stress profile
    report.push_str("## System Stress Profile\n\n");
    let stress = &result.stress_profile;
    report.push_str(&format!("- **Peak Memory Usage:** {:.0}MB\n", stress.memory_peak_mb));
    report.push_str(&format!("- **CPU Utilization:** {:.1}%\n", stress.cpu_utilization_pct));
    report.push_str(&format!("- **Disk I/O:** {:.0} ops/sec\n", stress.disk_io_ops_per_sec));
    report.push_str(&format!("- **Network Bandwidth:** {:.1} Mbps\n", stress.network_bandwidth_mbps));
    report.push_str(&format!("- **GC Pressure Score:** {:.2}\n", stress.gc_pressure_score));
    report.push_str(&format!("- **Resource Risk Level:** {:?}\n\n", stress.resource_exhaustion_risk));
    
    // Recommendations
    report.push_str("## Recommendations\n\n");
    
    if gates.overall_pass {
        report.push_str("✅ **System Status:** Ready for production deployment\n\n");
        report.push_str("The system has successfully passed all adversarial testing gates and demonstrates robust performance under stress conditions.\n\n");
    } else {
        report.push_str("⚠️ **System Status:** Requires improvements before production\n\n");
        
        if !gates.span_coverage_gate {
            report.push_str("- **Span Coverage:** Improve corpus coverage to ensure 100% file accessibility\n");
        }
        
        if !gates.sla_recall_gate {
            report.push_str("- **SLA Recall:** Optimize search quality to maintain flat recall under adversarial conditions\n");
        }
        
        if !gates.latency_stability_gate {
            report.push_str("- **Latency Stability:** Improve p99/p95 latency ratio for better consistency\n");
        }
        
        report.push_str("\n");
    }
    
    // Performance optimization suggestions
    if metrics.robustness_score < 0.80 {
        report.push_str("### Performance Optimization Suggestions\n\n");
        report.push_str("- Consider implementing more aggressive caching strategies\n");
        report.push_str("- Optimize memory management for large-scale operations\n");
        report.push_str("- Enhance error recovery mechanisms\n");
        report.push_str("- Implement adaptive resource management\n\n");
    }
    
    report.push_str("---\n\n");
    report.push_str("*This report was automatically generated by the Lens Adversarial Audit System*\n");
    
    Ok(report)
}

fn log_final_summary(
    result: &lens_core::adversarial::AdversarialAuditResult,
    gates_passed: bool,
    robustness_score: f32,
    execution_time: std::time::Duration,
) {
    let metrics = &result.overall_metrics;
    
    info!("🎭 === ADVERSARIAL AUDIT FINAL SUMMARY ===");
    info!("⏱️ Execution Time: {:.1}s", execution_time.as_secs_f64());
    info!("📊 Span Coverage: {:.1}%", metrics.span_coverage_pct);
    info!("🎯 SLA-Recall@50: {:.3}", metrics.sla_recall_at_50);
    info!("⚡ Latency p99/p95: {:.1}ms / {:.1}ms (ratio: {:.2})", 
          metrics.p99_latency_ms, metrics.p95_latency_ms, 
          metrics.p99_latency_ms / metrics.p95_latency_ms);
    info!("📉 Degradation Factor: {:.2}x", metrics.degradation_factor);
    info!("💪 Robustness Score: {:.2}", robustness_score);
    
    if gates_passed {
        info!("✅ RESULT: All adversarial gates PASSED");
        info!("🚀 System is ready for production deployment");
    } else {
        error!("❌ RESULT: Some adversarial gates FAILED");
        error!("⚠️ System requires improvements before production");
        
        for violation in &result.gate_validation.violations {
            error!("   • {}", violation);
        }
    }
    
    info!("🎭 === END ADVERSARIAL AUDIT SUMMARY ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_args_parsing() {
        let args = Args::try_parse_from(&[
            "adversarial_audit",
            "--corpus-path", "/tmp/corpus",
            "--output-path", "/tmp/results",
            "--timeout-minutes", "60",
            "--memory-limit-gb", "8",
        ]).unwrap();
        
        assert_eq!(args.corpus_path, PathBuf::from("/tmp/corpus"));
        assert_eq!(args.output_path, PathBuf::from("/tmp/results"));
        assert_eq!(args.timeout_minutes, 60);
        assert_eq!(args.memory_limit_gb, 8);
    }

    #[test]
    fn test_config_creation() {
        let args = Args {
            corpus_path: PathBuf::from("/tmp/corpus"),
            output_path: PathBuf::from("/tmp/results"),
            enable_clone_testing: true,
            enable_bloat_testing: false,
            enable_noise_testing: true,
            enable_stress_testing: true,
            parallel_execution: true,
            timeout_minutes: 45,
            memory_limit_gb: 16,
            cleanup_artifacts: false,
            json_report: true,
            markdown_report: true,
            verbose: false,
            fail_fast: false,
        };
        
        let config = create_adversarial_config(&args);
        
        assert_eq!(config.base_corpus_path, PathBuf::from("/tmp/corpus"));
        assert_eq!(config.output_base_path, PathBuf::from("/tmp/results"));
        assert!(config.enable_clone_testing);
        assert!(!config.enable_bloat_testing);
        assert!(config.enable_noise_testing);
        assert!(config.enable_stress_testing);
        assert!(config.parallel_execution);
        assert_eq!(config.timeout_minutes, 45);
        assert_eq!(config.memory_limit_gb, 16);
        assert!(!config.cleanup_artifacts);
    }
}