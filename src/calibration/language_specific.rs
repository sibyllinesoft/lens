//! # Language-Specific Calibration
//!
//! Per-language adaptive thresholds and calibration adjustments for cross-language parity.
//! Ensures <7pp variance across Tier-1/2 languages per PHASE 4 requirements.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::CalibrationSample;

/// Language-specific calibrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    /// Tier-1 languages requiring strict parity
    pub tier1_languages: Vec<String>,
    /// Tier-2 languages for secondary parity checking
    pub tier2_languages: Vec<String>,
    /// Maximum allowed variance across languages (must be <7pp)
    pub max_variance: f32,
}

/// Language-specific calibrator managing per-language adjustments
#[derive(Debug, Clone)]
pub struct LanguageSpecificCalibrator {
    config: LanguageConfig,
    /// Adjustment factors per language
    language_adjustments: HashMap<String, f32>,
    /// Adaptive thresholds per language
    language_thresholds: HashMap<String, f32>,
    /// Per-language ECE measurements
    language_eces: HashMap<String, f32>,
    /// Cross-language performance statistics
    performance_stats: CrossLanguageStats,
    /// Training convergence status
    converged: bool,
}

/// Cross-language performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageStats {
    /// Current variance across Tier-1 languages (percentage points)
    pub tier1_variance: f32,
    /// Current variance across Tier-2 languages (percentage points)  
    pub tier2_variance: f32,
    /// Overall cross-language parity score [0, 1]
    pub parity_score: f32,
    /// Language with worst performance
    pub worst_language: Option<String>,
    /// Language with best performance
    pub best_language: Option<String>,
    /// Mean performance across all languages
    pub mean_performance: f32,
}

/// Language-specific calibration result
#[derive(Debug, Clone)]
pub struct LanguageCalibrationResult {
    /// Original prediction
    pub original_prediction: f32,
    /// Language-adjusted prediction
    pub adjusted_prediction: f32,
    /// Language detected
    pub language: String,
    /// Adjustment factor applied
    pub adjustment_factor: f32,
    /// Threshold used
    pub threshold: f32,
    /// Language-specific ECE
    pub language_ece: f32,
}

impl LanguageSpecificCalibrator {
    /// Create new language-specific calibrator
    pub async fn new(config: LanguageConfig) -> Result<Self> {
        if config.max_variance >= 7.0 {
            anyhow::bail!("Max variance {:.1}pp must be <7pp per PHASE 4", config.max_variance);
        }

        info!("Creating language-specific calibrator");
        info!("Tier-1 languages: {:?}", config.tier1_languages);
        info!("Tier-2 languages: {:?}", config.tier2_languages);
        info!("Max variance target: <{:.1}pp", config.max_variance);

        Ok(Self {
            config,
            language_adjustments: HashMap::new(),
            language_thresholds: HashMap::new(),
            language_eces: HashMap::new(),
            performance_stats: CrossLanguageStats::default(),
            converged: false,
        })
    }

    /// Train language-specific adjustments
    pub async fn train(&mut self, samples: &[CalibrationSample]) -> Result<()> {
        info!("Training language-specific calibrator on {} samples", samples.len());

        // Group samples by language
        let language_groups = self.group_samples_by_language(samples);
        info!("Found {} languages in training data", language_groups.len());

        // Calculate baseline performance for each language
        let mut language_performances = HashMap::new();
        for (language, lang_samples) in &language_groups {
            if lang_samples.len() >= 10 { // Minimum samples for reliable statistics
                let lang_samples_owned: Vec<CalibrationSample> = lang_samples.iter().map(|&s| s.clone()).collect();
                let ece = self.calculate_language_ece(&lang_samples_owned).await?;
                language_performances.insert(language.clone(), ece);
                self.language_eces.insert(language.clone(), ece);
                info!("Language {} baseline ECE: {:.4}", language, ece);
            }
        }

        // Calculate adjustment factors to minimize cross-language variance
        self.calculate_adjustment_factors(&language_performances).await?;

        // Calculate adaptive thresholds per language
        self.calculate_adaptive_thresholds(&language_groups).await?;

        // Update cross-language statistics
        self.update_performance_stats(&language_performances).await?;

        // Check convergence
        self.converged = self.check_convergence()?;

        info!(
            "Language-specific calibrator trained: {} adjustments, {} thresholds",
            self.language_adjustments.len(),
            self.language_thresholds.len()
        );
        info!(
            "Cross-language variance: Tier-1 {:.2}pp, Tier-2 {:.2}pp, Parity {:.3}",
            self.performance_stats.tier1_variance,
            self.performance_stats.tier2_variance,
            self.performance_stats.parity_score
        );

        if self.performance_stats.tier1_variance >= self.config.max_variance {
            warn!(
                "Tier-1 variance {:.2}pp exceeds target <{:.1}pp",
                self.performance_stats.tier1_variance, self.config.max_variance
            );
        }

        Ok(())
    }

    /// Apply language-specific calibration
    pub async fn calibrate(
        &self,
        prediction: f32,
        _intent: &str,
        language: &str,
        _features: &HashMap<String, f32>,
    ) -> Result<f32> {
        // Get language-specific adjustment factor
        let adjustment_factor = self.language_adjustments
            .get(language)
            .copied()
            .unwrap_or(1.0);

        // Get language-specific threshold
        let threshold = self.language_thresholds
            .get(language)
            .copied()
            .unwrap_or(0.5);

        // Apply language-specific adjustment
        let adjusted = self.apply_language_adjustment(prediction, adjustment_factor, threshold)?;

        debug!(
            "Language calibration [{}]: {:.4} -> {:.4} (factor: {:.3}, threshold: {:.3})",
            language, prediction, adjusted, adjustment_factor, threshold
        );

        Ok(adjusted)
    }

    /// Get adjustment factor for a language
    pub fn get_adjustment_factor(&self, language: &str) -> f32 {
        self.language_adjustments.get(language).copied().unwrap_or(1.0)
    }

    /// Get adaptive threshold for a language
    pub fn get_threshold(&self, language: &str) -> f32 {
        self.language_thresholds.get(language).copied().unwrap_or(0.5)
    }

    /// Get ECE for a specific language
    pub fn get_language_ece(&self, language: &str) -> f32 {
        self.language_eces.get(language).copied().unwrap_or(0.0)
    }

    /// Get cross-language performance statistics
    pub fn get_performance_stats(&self) -> &CrossLanguageStats {
        &self.performance_stats
    }

    /// Check if training converged
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Validate PHASE 4 compliance for cross-language parity
    pub async fn validate_cross_language_parity(&self) -> Result<bool> {
        let tier1_compliant = self.performance_stats.tier1_variance < self.config.max_variance;
        let tier2_compliant = self.performance_stats.tier2_variance < self.config.max_variance;
        let parity_good = self.performance_stats.parity_score > 0.8; // High parity threshold

        info!("Cross-language parity validation:");
        info!(
            "Tier-1 variance: {:.2}pp (target: <{:.1}pp) - {}",
            self.performance_stats.tier1_variance,
            self.config.max_variance,
            if tier1_compliant { "✓" } else { "✗" }
        );
        info!(
            "Tier-2 variance: {:.2}pp (target: <{:.1}pp) - {}",
            self.performance_stats.tier2_variance,
            self.config.max_variance,
            if tier2_compliant { "✓" } else { "✗" }
        );
        info!(
            "Parity score: {:.3} (target: >0.8) - {}",
            self.performance_stats.parity_score,
            if parity_good { "✓" } else { "✗" }
        );

        let fully_compliant = tier1_compliant && tier2_compliant && parity_good;
        
        if !fully_compliant {
            warn!("Cross-language parity violations detected");
            if let Some(worst) = &self.performance_stats.worst_language {
                warn!("Worst performing language: {}", worst);
            }
        } else {
            info!("✓ Cross-language parity achieved");
        }

        Ok(fully_compliant)
    }

    // Private implementation methods

    /// Group samples by language
    fn group_samples_by_language<'a>(&self, samples: &'a [CalibrationSample]) -> HashMap<String, Vec<&'a CalibrationSample>> {
        let mut groups = HashMap::new();
        
        for sample in samples {
            if let Some(language) = &sample.language {
                groups.entry(language.clone()).or_insert_with(Vec::new).push(sample);
            }
        }
        
        groups
    }

    /// Calculate ECE for a specific language
    async fn calculate_language_ece(&self, samples: &[CalibrationSample]) -> Result<f32> {
        const NUM_BINS: usize = 10;
        let mut bins = vec![Vec::new(); NUM_BINS];
        
        // Assign samples to bins
        for sample in samples {
            let bin_idx = ((sample.prediction * NUM_BINS as f32) as usize).min(NUM_BINS - 1);
            bins[bin_idx].push(sample.ground_truth);
        }
        
        // Calculate ECE
        let mut ece = 0.0;
        let total_samples = samples.len() as f32;
        
        for (i, bin) in bins.iter().enumerate() {
            if bin.is_empty() {
                continue;
            }
            
            let bin_center = (i as f32 + 0.5) / NUM_BINS as f32;
            let bin_accuracy = bin.iter().sum::<f32>() / bin.len() as f32;
            let bin_error = (bin_center - bin_accuracy).abs();
            let bin_weight = bin.len() as f32 / total_samples;
            
            ece += bin_error * bin_weight;
        }
        
        Ok(ece)
    }

    /// Calculate adjustment factors to minimize cross-language variance
    async fn calculate_adjustment_factors(&mut self, performances: &HashMap<String, f32>) -> Result<()> {
        if performances.len() < 2 {
            info!("Insufficient languages for adjustment factor calculation");
            return Ok(());
        }

        // Calculate target ECE (mean of all languages)
        let target_ece = performances.values().sum::<f32>() / performances.len() as f32;
        info!("Target ECE for adjustment: {:.4}", target_ece);

        // Calculate adjustment factors to bring all languages towards target
        for (language, &current_ece) in performances {
            let adjustment_factor = if current_ece > 1e-6 {
                target_ece / current_ece
            } else {
                1.0
            };
            
            // Clamp adjustment factor to reasonable range
            let clamped_factor = adjustment_factor.clamp(0.5, 2.0);
            
            self.language_adjustments.insert(language.clone(), clamped_factor);
            
            debug!(
                "Language {} adjustment factor: {:.3} (ECE: {:.4} -> target: {:.4})",
                language, clamped_factor, current_ece, target_ece
            );
        }

        Ok(())
    }

    /// Calculate adaptive thresholds per language
    async fn calculate_adaptive_thresholds(&mut self, language_groups: &HashMap<String, Vec<&CalibrationSample>>) -> Result<()> {
        for (language, samples) in language_groups {
            if samples.len() < 10 {
                continue;
            }

            // Calculate optimal threshold using ROC analysis
            let samples_owned: Vec<CalibrationSample> = samples.iter().map(|&s| s.clone()).collect();
            let threshold = self.calculate_optimal_threshold(&samples_owned).await?;
            self.language_thresholds.insert(language.clone(), threshold);
            
            debug!(
                "Language {} optimal threshold: {:.3}",
                language, threshold
            );
        }

        Ok(())
    }

    /// Calculate optimal threshold for a language using ROC analysis
    async fn calculate_optimal_threshold(&self, samples: &[CalibrationSample]) -> Result<f32> {
        let mut best_threshold = 0.5;
        let mut best_f1_score = 0.0;

        // Try different threshold values
        for threshold_int in 10..90 {
            let threshold = (threshold_int as f32) / 100.0;
            
            let mut tp = 0.0;
            let mut fp = 0.0;
            let mut tn = 0.0;
            let mut fn_count = 0.0;

            for sample in samples {
                let prediction = sample.prediction > threshold;
                let actual = sample.ground_truth > 0.5;

                match (prediction, actual) {
                    (true, true) => tp += 1.0,
                    (true, false) => fp += 1.0,
                    (false, false) => tn += 1.0,
                    (false, true) => fn_count += 1.0,
                }
            }

            // Calculate F1 score
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
            let f1_score = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            if f1_score > best_f1_score {
                best_f1_score = f1_score;
                best_threshold = threshold;
            }
        }

        Ok(best_threshold)
    }

    /// Apply language-specific adjustment
    fn apply_language_adjustment(&self, prediction: f32, adjustment_factor: f32, threshold: f32) -> Result<f32> {
        // Apply adjustment factor with threshold consideration
        let base_adjustment = prediction * adjustment_factor;
        
        // Apply threshold-based correction
        let threshold_distance = prediction - threshold;
        let threshold_adjustment = threshold_distance * 0.1; // Small threshold influence
        
        let final_adjustment = base_adjustment + threshold_adjustment;
        
        Ok(final_adjustment.clamp(0.001, 0.999))
    }

    /// Update cross-language performance statistics
    async fn update_performance_stats(&mut self, performances: &HashMap<String, f32>) -> Result<()> {
        if performances.is_empty() {
            return Ok(());
        }

        // Calculate overall statistics
        let all_eces: Vec<f32> = performances.values().copied().collect();
        let mean_ece = all_eces.iter().sum::<f32>() / all_eces.len() as f32;
        
        // Calculate variances for different tiers
        let tier1_eces: Vec<f32> = self.config.tier1_languages
            .iter()
            .filter_map(|lang| performances.get(lang))
            .copied()
            .collect();
        
        let tier2_eces: Vec<f32> = self.config.tier2_languages
            .iter()
            .filter_map(|lang| performances.get(lang))
            .copied()
            .collect();

        let tier1_variance = if tier1_eces.len() > 1 {
            self.calculate_variance(&tier1_eces) * 100.0 // Convert to percentage points
        } else {
            0.0
        };

        let tier2_variance = if tier2_eces.len() > 1 {
            self.calculate_variance(&tier2_eces) * 100.0
        } else {
            0.0
        };

        // Calculate parity score
        let overall_variance = if all_eces.len() > 1 {
            self.calculate_variance(&all_eces)
        } else {
            0.0
        };
        
        let parity_score = 1.0 - (overall_variance / 0.05).min(1.0); // Normalize by reasonable ECE range

        // Find best and worst languages
        let worst_language = performances
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone());
        
        let best_language = performances
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone());

        self.performance_stats = CrossLanguageStats {
            tier1_variance,
            tier2_variance,
            parity_score,
            worst_language,
            best_language,
            mean_performance: mean_ece,
        };

        Ok(())
    }

    /// Calculate variance (standard deviation) of values
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.len() <= 1 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (values.len() - 1) as f32;
        
        variance.sqrt() // Return standard deviation
    }

    /// Check if training converged
    fn check_convergence(&self) -> Result<bool> {
        // Check if variance targets are met
        let tier1_ok = self.performance_stats.tier1_variance < self.config.max_variance;
        let tier2_ok = self.performance_stats.tier2_variance < self.config.max_variance;
        
        // Check if parity score is reasonable
        let parity_ok = self.performance_stats.parity_score > 0.7;
        
        // Check if we have reasonable number of adjustments
        let adjustments_ok = !self.language_adjustments.is_empty();

        Ok(tier1_ok && tier2_ok && parity_ok && adjustments_ok)
    }
}

impl Default for CrossLanguageStats {
    fn default() -> Self {
        Self {
            tier1_variance: 0.0,
            tier2_variance: 0.0,
            parity_score: 1.0,
            worst_language: None,
            best_language: None,
            mean_performance: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sample(prediction: f32, ground_truth: f32, language: &str) -> CalibrationSample {
        CalibrationSample {
            prediction,
            ground_truth,
            intent: "test".to_string(),
            language: Some(language.to_string()),
            features: HashMap::new(),
            weight: 1.0,
        }
    }

    #[tokio::test]
    async fn test_language_calibrator_creation() {
        let config = LanguageConfig {
            tier1_languages: vec!["rust".to_string(), "python".to_string()],
            tier2_languages: vec!["go".to_string()],
            max_variance: 6.9,
        };
        
        let calibrator = LanguageSpecificCalibrator::new(config).await.unwrap();
        assert!(!calibrator.is_converged());
        assert_eq!(calibrator.get_adjustment_factor("rust"), 1.0); // Default
    }

    #[tokio::test]
    async fn test_invalid_max_variance() {
        let config = LanguageConfig {
            tier1_languages: vec!["rust".to_string()],
            tier2_languages: vec![],
            max_variance: 7.0, // Must be < 7.0
        };
        
        let result = LanguageSpecificCalibrator::new(config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be <7pp"));
    }

    #[tokio::test]
    async fn test_language_specific_training() {
        let config = LanguageConfig {
            tier1_languages: vec!["rust".to_string(), "python".to_string()],
            tier2_languages: vec!["go".to_string()],
            max_variance: 6.5,
        };
        
        let mut calibrator = LanguageSpecificCalibrator::new(config).await.unwrap();
        
        // Create samples with different performance per language
        let mut samples = Vec::new();
        
        // Rust samples - good calibration
        for i in 0..20 {
            let pred = (i as f32) / 20.0;
            let ground_truth = if pred > 0.5 { 1.0 } else { 0.0 };
            samples.push(create_test_sample(pred, ground_truth, "rust"));
        }
        
        // Python samples - poor calibration (overconfident)
        for i in 0..20 {
            let pred = 0.7 + (i as f32) / 20.0 * 0.29; // High predictions
            let ground_truth = if i < 10 { 1.0 } else { 0.0 }; // 50% should be positive
            samples.push(create_test_sample(pred, ground_truth, "python"));
        }
        
        // Go samples - moderate calibration
        for i in 0..15 {
            let pred = 0.3 + (i as f32) / 15.0 * 0.4;
            let ground_truth = if pred > 0.5 { 1.0 } else { 0.0 };
            samples.push(create_test_sample(pred, ground_truth, "go"));
        }
        
        calibrator.train(&samples).await.unwrap();
        
        // Check that adjustment factors were calculated
        let rust_factor = calibrator.get_adjustment_factor("rust");
        let python_factor = calibrator.get_adjustment_factor("python");
        let go_factor = calibrator.get_adjustment_factor("go");
        
        assert!(rust_factor > 0.0);
        assert!(python_factor > 0.0);
        assert!(go_factor > 0.0);
        
        // Check that thresholds were calculated
        assert!(calibrator.get_threshold("rust") > 0.0);
        assert!(calibrator.get_threshold("python") > 0.0);
        
        println!("Adjustment factors: rust={:.3}, python={:.3}, go={:.3}", 
                rust_factor, python_factor, go_factor);
    }

    #[tokio::test]
    async fn test_language_calibration() {
        let config = LanguageConfig {
            tier1_languages: vec!["rust".to_string()],
            tier2_languages: vec![],
            max_variance: 5.0,
        };
        
        let mut calibrator = LanguageSpecificCalibrator::new(config).await.unwrap();
        
        // Set up manual adjustment for testing
        calibrator.language_adjustments.insert("rust".to_string(), 1.5);
        calibrator.language_thresholds.insert("rust".to_string(), 0.4);
        
        let features = HashMap::new();
        let calibrated = calibrator.calibrate(0.6, "test", "rust", &features).await.unwrap();
        
        // Should be adjusted based on factor and threshold
        assert!(calibrated != 0.6); // Should be different from input
        assert!(calibrated >= 0.001 && calibrated <= 0.999); // Valid probability
        
        // Unknown language should work with defaults
        let calibrated_unknown = calibrator.calibrate(0.6, "test", "unknown", &features).await.unwrap();
        assert!(calibrated_unknown >= 0.001 && calibrated_unknown <= 0.999);
    }

    #[tokio::test]
    async fn test_cross_language_parity_validation() {
        let config = LanguageConfig {
            tier1_languages: vec!["rust".to_string(), "python".to_string()],
            tier2_languages: vec!["go".to_string()],
            max_variance: 6.0,
        };
        
        let mut calibrator = LanguageSpecificCalibrator::new(config).await.unwrap();
        
        // Set up performance stats manually for testing
        calibrator.performance_stats = CrossLanguageStats {
            tier1_variance: 5.5, // Within limit
            tier2_variance: 4.0, // Within limit
            parity_score: 0.85, // Good parity
            worst_language: Some("python".to_string()),
            best_language: Some("rust".to_string()),
            mean_performance: 0.012,
        };
        
        let is_compliant = calibrator.validate_cross_language_parity().await.unwrap();
        assert!(is_compliant);
        
        // Test non-compliant case
        calibrator.performance_stats.tier1_variance = 8.0; // Exceeds limit
        let is_compliant = calibrator.validate_cross_language_parity().await.unwrap();
        assert!(!is_compliant);
    }

    #[test]
    fn test_variance_calculation() {
        let config = LanguageConfig {
            tier1_languages: vec![],
            tier2_languages: vec![],
            max_variance: 5.0,
        };
        
        let calibrator = LanguageSpecificCalibrator {
            config,
            language_adjustments: HashMap::new(),
            language_thresholds: HashMap::new(),
            language_eces: HashMap::new(),
            performance_stats: CrossLanguageStats::default(),
            converged: false,
        };
        
        let values = vec![0.01, 0.02, 0.015, 0.018, 0.012];
        let variance = calibrator.calculate_variance(&values);
        
        // Should calculate reasonable variance
        assert!(variance > 0.0);
        assert!(variance < 0.1); // Reasonable for ECE values
        
        // Single value should have zero variance
        assert_eq!(calibrator.calculate_variance(&[0.01]), 0.0);
    }

    #[tokio::test]
    async fn test_optimal_threshold_calculation() {
        let config = LanguageConfig {
            tier1_languages: vec!["test".to_string()],
            tier2_languages: vec![],
            max_variance: 5.0,
        };
        
        let calibrator = LanguageSpecificCalibrator::new(config).await.unwrap();
        
        // Create samples where optimal threshold should be around 0.5
        let samples = vec![
            create_test_sample(0.3, 0.0, "test"),
            create_test_sample(0.4, 0.0, "test"),
            create_test_sample(0.6, 1.0, "test"),
            create_test_sample(0.7, 1.0, "test"),
            create_test_sample(0.8, 1.0, "test"),
        ];
        
        let sample_refs: Vec<&CalibrationSample> = samples.iter().collect();
        let threshold = calibrator.calculate_optimal_threshold(&sample_refs).await.unwrap();
        
        // Should be reasonable threshold value
        assert!(threshold >= 0.1);
        assert!(threshold <= 0.9);
        
        println!("Calculated optimal threshold: {:.3}", threshold);
    }
}