//! # LTR Training Binary
//!
//! Implementation of the TODO.md training pipeline:
//! TRAIN:
//!   bin/train_ltr \
//!     --train qrels://swe_verified_dev,qrels://coir_dev, qrels://csn_dev \
//!     --features spec://features_vN.json \
//!     --objective pairwise_logistic \
//!     --monotone exact_match+,struct_hit+ \
//!     --cap_logodds 0.4 \
//!     --hard_negs graph://symbol_neighbors --ratio 4 \
//!     --cv_split repo \
//!     --out artifact://models/ltr_<DATE>.json

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use lens_core::semantic::{LTRConfig, LTRObjective, LTRTrainer};
use serde_json;
use std::path::PathBuf;
use tracing::{info, warn, error};
use tracing_subscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Training dataset qrels (comma-separated)
    #[arg(long, value_delimiter = ',')]
    train: Vec<String>,
    
    /// Feature specification file
    #[arg(long)]
    features: String,
    
    /// Training objective
    #[arg(long, default_value = "pairwise_logistic")]
    objective: String,
    
    /// Monotonic constraints (e.g., exact_match+,struct_hit+)
    #[arg(long, value_delimiter = ',')]
    monotone: Vec<String>,
    
    /// Cap log-odds delta per feature
    #[arg(long, default_value = "0.4")]
    cap_logodds: f32,
    
    /// Hard negatives source
    #[arg(long, default_value = "graph://symbol_neighbors")]
    hard_negs: String,
    
    /// Negative to positive ratio
    #[arg(long, default_value = "4.0")]
    ratio: f32,
    
    /// Cross-validation split strategy
    #[arg(long, default_value = "repo")]
    cv_split: String,
    
    /// Output model path
    #[arg(long)]
    out: String,
    
    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("train_ltr={},lens_core={}", log_level, log_level))
        .init();

    // Check hard fail conditions from TODO.md
    info!("🚀 Starting LTR training pipeline (TODO.md compliant)");
    info!("📋 Training datasets: {:?}", args.train);
    info!("📋 Features: {}", args.features);
    info!("📋 Objective: {}", args.objective);
    info!("📋 Monotonic constraints: {:?}", args.monotone);

    // Validate required inputs from TODO.md
    validate_inputs(&args)?;

    // Parse monotonic constraints
    let (monotonic_increasing, monotonic_decreasing) = parse_monotonic_constraints(&args.monotone)?;

    // Parse objective
    let objective = match args.objective.as_str() {
        "pairwise_logistic" => LTRObjective::PairwiseLogistic,
        "lambdamart" => LTRObjective::LambdaMART,
        _ => return Err(anyhow::anyhow!("Unsupported objective: {}", args.objective)),
    };

    // Create training configuration
    let config = LTRConfig {
        objective,
        max_log_odds_delta: args.cap_logodds,
        monotonic_increasing,
        monotonic_decreasing,
        hard_negative_ratio: args.ratio,
        learning_rate: 0.01,
        l2_lambda: 0.001,
        max_iterations: 1000,
        cv_folds: 5,
        patience: 10,
        seed: 42,
    };

    // Initialize trainer
    let mut trainer = LTRTrainer::new(config);

    // Load training data
    info!("📊 Loading training data...");
    for qrel_path in &args.train {
        trainer.add_training_data(qrel_path)
            .await
            .with_context(|| format!("Failed to load training data from {}", qrel_path))?;
    }

    // Load feature specification
    info!("🔧 Loading feature specification from {}", args.features);
    trainer.load_feature_spec(&args.features)
        .await
        .context("Failed to load feature specification")?;

    // Generate hard negatives
    info!("💀 Generating hard negatives (ratio: {:.1}:1)", args.ratio);
    trainer.generate_hard_negatives(&args.hard_negs, args.ratio)
        .await
        .context("Failed to generate hard negatives")?;

    // Train model with cross-validation
    info!("🎯 Training bounded LTR model with monotonic constraints...");
    info!("   • Max |Δlog-odds| per feature: {:.1}", args.cap_logodds);
    info!("   • Monotonic increasing: {:?}", trainer.get_monotonic_increasing());
    info!("   • Cross-validation: {}", args.cv_split);
    
    let trained_model = trainer.train_with_cv(&args.cv_split)
        .await
        .context("Failed to train LTR model")?;

    // Validate model weights (TODO.md hard fail condition)
    validate_trained_model(&trained_model)?;

    // Create output path with timestamp
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let output_path = args.out.replace("<DATE>", &timestamp);
    
    // Ensure artifact directory exists
    if let Some(parent) = PathBuf::from(&output_path).parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {:?}", parent))?;
    }

    // Save trained model
    info!("💾 Saving trained model to {}", output_path);
    let model_json = serde_json::to_string_pretty(&trained_model)
        .context("Failed to serialize trained model")?;
    
    std::fs::write(&output_path, model_json)
        .with_context(|| format!("Failed to write model to {}", output_path))?;

    // Generate training report
    let report = trainer.generate_training_report()
        .await
        .context("Failed to generate training report")?;

    info!("✅ LTR training completed successfully!");
    info!("📊 Training metrics:");
    info!("   • Final validation nDCG@10: {:.3}", report.final_ndcg);
    info!("   • Feature weights (non-uniform): {} features", report.feature_count);
    info!("   • Cross-validation folds: {}", report.cv_folds);
    info!("   • Training samples: {}", report.total_samples);
    info!("   • Hard negatives: {}", report.hard_negative_count);

    // Verify model passes TODO.md requirements
    if report.weights_stddev < 1e-6 {
        error!("❌ ABORT: UNTRAINED_MODEL - weights have stddev < 1e-6 (uniform/zeroed)");
        return Err(anyhow::anyhow!("Model failed TODO.md validation: uniform weights detected"));
    }

    info!("🎯 Model ready for calibration and evaluation");
    info!("   • Next step: bin/calibrate_isotonic --scores {} --out artifact://calib/iso_{}.json", 
          output_path, timestamp);

    Ok(())
}

/// Validate inputs according to TODO.md requirements
fn validate_inputs(args: &Args) -> Result<()> {
    // Check required qrels
    if args.train.is_empty() {
        return Err(anyhow::anyhow!("At least one training qrel is required"));
    }

    // Check feature spec exists
    if !args.features.starts_with("spec://") {
        warn!("Feature spec should use spec:// URI format for TODO.md compliance");
    }

    // Check hard negatives source
    if !args.hard_negs.starts_with("graph://") {
        warn!("Hard negatives should use graph:// URI format for TODO.md compliance");
    }

    // Check output path
    if !args.out.starts_with("artifact://") {
        warn!("Output path should use artifact:// URI format for TODO.md compliance");
    }

    Ok(())
}

/// Parse monotonic constraints from command line
fn parse_monotonic_constraints(constraints: &[String]) -> Result<(Vec<String>, Vec<String>)> {
    let mut increasing = Vec::new();
    let mut decreasing = Vec::new();
    
    for constraint in constraints {
        if constraint.ends_with('+') {
            increasing.push(constraint.trim_end_matches('+').to_string());
        } else if constraint.ends_with('-') {
            decreasing.push(constraint.trim_end_matches('-').to_string());
        } else {
            return Err(anyhow::anyhow!(
                "Monotonic constraint must end with + or -: {}", constraint
            ));
        }
    }
    
    Ok((increasing, decreasing))
}

/// Validate trained model meets TODO.md requirements
fn validate_trained_model(model: &serde_json::Value) -> Result<()> {
    // Check for uniform weights (hard fail condition)
    if let Some(weights) = model.get("weights").and_then(|w| w.as_array()) {
        let weight_values: Vec<f64> = weights.iter()
            .filter_map(|v| v.as_f64())
            .collect();
            
        if weight_values.is_empty() {
            return Err(anyhow::anyhow!("NO_TRAINED_WEIGHTS: No weights found in model"));
        }
        
        // Calculate standard deviation
        let mean = weight_values.iter().sum::<f64>() / weight_values.len() as f64;
        let variance = weight_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / weight_values.len() as f64;
        let stddev = variance.sqrt();
        
        if stddev < 1e-6 {
            return Err(anyhow::anyhow!(
                "UNTRAINED_MODEL: weights stddev {:.2e} < 1e-6 (uniform/zeroed)", 
                stddev
            ));
        }
        
        info!("✅ Model validation passed: weights stddev = {:.2e}", stddev);
    } else {
        return Err(anyhow::anyhow!("NO_TRAINED_WEIGHTS: weights field missing from model"));
    }
    
    Ok(())
}