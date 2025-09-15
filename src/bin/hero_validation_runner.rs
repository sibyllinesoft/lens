use std::process;
use tokio;
use tracing_subscriber;

// Import the hero validation module
use lens_core::hero_validation::HeroValidationSuite;

#[tokio::main]
async fn main() {
    println!("🚀 Starting Hero Configuration Validation");
    println!("==========================================");
    
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "lens_core=info,hero_validation_runner=debug".into()),
        )
        .init();
    
    // Create validation suite (with 2% tolerance)
    let suite = match HeroValidationSuite::new(0.02) {
        Ok(suite) => {
            println!("✅ Hero validation suite initialized successfully");
            suite
        },
        Err(e) => {
            eprintln!("❌ Failed to initialize hero validation suite: {}", e);
            process::exit(1);
        }
    };
    
    // Validate hero configuration structure
    match suite.validate_config() {
        Ok(is_valid) => {
            if is_valid {
                println!("✅ Hero configuration validation passed");
            } else {
                println!("❌ Hero configuration validation failed");
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("❌ Configuration validation failed: {}", e);
            process::exit(1);
        }
    }
    
    println!("\n🎯 Running end-to-end validation against golden dataset...");
    
    // Generate comprehensive validation report
    let report = match suite.generate_validation_report().await {
        Ok(report) => {
            println!("✅ Hero validation report generated successfully");
            report
        },
        Err(e) => {
            eprintln!("❌ Hero validation failed: {}", e);
            process::exit(1);
        }
    };
    
    // Print executive summary
    println!("\n📊 VALIDATION RESULTS");
    println!("=====================");
    println!("Hero Config ID: {}", report.hero_config.config_id);
    println!("Hero Fingerprint: {}", report.hero_config.fingerprint);
    println!("Validation Passed: {}", if report.results.passed { "✅ YES" } else { "❌ NO" });
    println!("Tolerance: {:.1}%", report.results.tolerance * 100.0);
    println!("Validation Timestamp: {}", report.results.validation_timestamp);
    println!();
    
    // Print baseline vs actual metrics
    println!("🎯 BASELINE vs ACTUAL METRICS");
    println!("==============================");
    for (metric, actual) in &report.results.rust_metrics {
        if let (Some(&baseline), Some(&diff)) = (
            report.results.baseline_metrics.get(metric), 
            report.results.differences.get(metric)
        ) {
            let status = if diff <= report.results.tolerance { "✅ PASS" } else { "❌ FAIL" };
            println!("{}: {:.3} vs {:.3} {} (diff: {:.3})", 
                metric, actual, baseline, status, diff);
        }
    }
    println!();
    
    // Print hero configuration details
    println!("⚙️ HERO CONFIGURATION");
    println!("===================");
    println!("Fusion: {}", report.hero_config.params.fusion);
    println!("Chunk Policy: {}", report.hero_config.params.chunk_policy);
    println!("Chunk Length: {}", report.hero_config.params.chunk_len);
    println!("Retrieval K: {}", report.hero_config.params.retrieval_k);
    println!("RRF K0: {}", report.hero_config.params.rrf_k0);
    println!("Reranker: {}", report.hero_config.params.reranker);
    println!("Router: {}", report.hero_config.params.router);
    println!("Symbol Boost: {}", report.hero_config.params.symbol_boost);
    println!("Graph Expand Hops: {}", report.hero_config.params.graph_expand_hops);
    println!();
    
    // Print recommendations
    println!("💡 RECOMMENDATIONS");
    println!("==================");
    for recommendation in &report.recommendations {
        println!("  • {}", recommendation);
    }
    println!();
    
    // Final verdict
    if report.results.passed {
        println!("🎉 VALIDATION SUCCESSFUL!");
        println!("   The Rust implementation with hero defaults produces equivalent");
        println!("   results to the production hero canary configuration.");
        println!("   ✅ READY FOR PRODUCTION DEPLOYMENT");
        process::exit(0);
    } else {
        println!("⚠️  VALIDATION INCOMPLETE!");
        println!("   Some metrics do not meet production equivalence thresholds.");
        println!("   ❌ REVIEW REQUIRED BEFORE PRODUCTION DEPLOYMENT");
        process::exit(1);
    }
}