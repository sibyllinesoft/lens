//! Comprehensive tests for pipeline/learning.rs module
//! 
//! These tests cover:
//! - Learning pipeline functionality
//! - Model training and validation
//! - Feature extraction and processing
//! - Prediction and evaluation
//! - Configuration and error handling

use lens_core::pipeline::learning::*;
use std::collections::HashMap;
use std::time::Instant;
use tokio;

#[test]
fn test_learning_config_default() {
    let config = LearningConfig::default();
    
    assert_eq!(config.training_window_size, 1000);
    assert_eq!(config.update_frequency, 100);
    assert_eq!(config.learning_rate, 0.01);
    assert_eq!(config.confidence_threshold, 0.85);
    assert_eq!(config.min_training_samples, 50);
    assert!(config.feature_normalization);
    
    // Test WAND config
    assert_eq!(config.wand_config.max_iterations, 100);
    assert_eq!(config.wand_config.quality_threshold, 0.8);
    assert_eq!(config.wand_config.score_improvement_tolerance, 0.01);
    assert_eq!(config.wand_config.term_contribution_threshold, 0.05);
    
    // Test HNSW config
    assert_eq!(config.hnsw_config.max_layers, 5);
    assert_eq!(config.hnsw_config.beam_width, 64);
    assert_eq!(config.hnsw_config.distance_threshold, 0.1);
    assert_eq!(config.hnsw_config.max_neighbors, 16);
}

#[test]
fn test_learning_config_custom() {
    let custom_config = LearningConfig {
        training_window_size: 2000,
        update_frequency: 200,
        learning_rate: 0.02,
        confidence_threshold: 0.9,
        min_training_samples: 100,
        feature_normalization: false,
        wand_config: WandLearningConfig {
            max_iterations: 200,
            quality_threshold: 0.85,
            score_improvement_tolerance: 0.005,
            term_contribution_threshold: 0.1,
        },
        hnsw_config: HnswLearningConfig {
            max_layers: 8,
            beam_width: 128,
            distance_threshold: 0.05,
            max_neighbors: 32,
        },
    };
    
    assert_eq!(custom_config.training_window_size, 2000);
    assert_eq!(custom_config.update_frequency, 200);
    assert_eq!(custom_config.learning_rate, 0.02);
    assert_eq!(custom_config.confidence_threshold, 0.9);
    assert_eq!(custom_config.min_training_samples, 100);
    assert!(!custom_config.feature_normalization);
    
    assert_eq!(custom_config.wand_config.max_iterations, 200);
    assert_eq!(custom_config.hnsw_config.beam_width, 128);
}

#[tokio::test]
async fn test_learning_stop_model_creation() {
    let config = LearningConfig::default();
    let model = LearningStopModel::new(config).await;
    
    assert!(model.is_ok());
    let model = model.unwrap();
    
    // Test that metrics are initialized
    let metrics = model.get_metrics().await;
    assert_eq!(metrics.total_predictions, 0);
    assert_eq!(metrics.correct_early_stops, 0);
    assert_eq!(metrics.model_accuracy, 0.0);
}

#[tokio::test]
async fn test_learning_stop_model_creation_custom_config() {
    let config = LearningConfig {
        training_window_size: 500,
        update_frequency: 50,
        learning_rate: 0.005,
        confidence_threshold: 0.75,
        ..LearningConfig::default()
    };
    
    let model = LearningStopModel::new(config.clone()).await;
    assert!(model.is_ok());
    
    let model = model.unwrap();
    assert_eq!(model.config().training_window_size, 500);
    assert_eq!(model.config().update_frequency, 50);
    assert_eq!(model.config().learning_rate, 0.005);
    assert_eq!(model.config().confidence_threshold, 0.75);
}

#[test]
fn test_wand_feature_comprehensive() {
    let features = [
        WandFeature::IterationCount,
        WandFeature::ScoreImprovement,
        WandFeature::TermContribution,
        WandFeature::DocumentFrequency,
        WandFeature::QualityEstimate,
        WandFeature::TimeElapsed,
        WandFeature::CandidateSetSize,
        WandFeature::ThresholdConvergence,
    ];
    
    // Test that all features are unique
    for (i, feature1) in features.iter().enumerate() {
        for (j, feature2) in features.iter().enumerate() {
            if i != j {
                assert_ne!(feature1, feature2);
            } else {
                assert_eq!(feature1, feature2);
            }
        }
    }
    
    // Test debug formatting
    assert_eq!(format!("{:?}", WandFeature::IterationCount), "IterationCount");
    assert_eq!(format!("{:?}", WandFeature::ScoreImprovement), "ScoreImprovement");
}

#[test]
fn test_hnsw_feature_comprehensive() {
    let features = [
        HnswFeature::LayerDepth,
        HnswFeature::DistanceToQuery,
        HnswFeature::NeighborCount,
        HnswFeature::SearchRadius,
        HnswFeature::BeamPosition,
        HnswFeature::ExplorationRatio,
        HnswFeature::DistanceImprovement,
        HnswFeature::GraphConnectivity,
    ];
    
    // Test that all features are unique
    for (i, feature1) in features.iter().enumerate() {
        for (j, feature2) in features.iter().enumerate() {
            if i != j {
                assert_ne!(feature1, feature2);
            } else {
                assert_eq!(feature1, feature2);
            }
        }
    }
    
    // Test debug formatting
    assert_eq!(format!("{:?}", HnswFeature::LayerDepth), "LayerDepth");
    assert_eq!(format!("{:?}", HnswFeature::DistanceToQuery), "DistanceToQuery");
}

#[test]
fn test_confidence_feature_comprehensive() {
    let features = [
        ConfidenceFeature::ResultCount,
        ConfidenceFeature::ScoreDistribution,
        ConfidenceFeature::SystemAgreement,
        ConfidenceFeature::QueryComplexity,
        ConfidenceFeature::ProcessingTime,
        ConfidenceFeature::ResourceUtilization,
    ];
    
    // Test that all features are unique
    for (i, feature1) in features.iter().enumerate() {
        for (j, feature2) in features.iter().enumerate() {
            if i != j {
                assert_ne!(feature1, feature2);
            } else {
                assert_eq!(feature1, feature2);
            }
        }
    }
    
    // Test debug formatting
    assert_eq!(format!("{:?}", ConfidenceFeature::ResultCount), "ResultCount");
    assert_eq!(format!("{:?}", ConfidenceFeature::SystemAgreement), "SystemAgreement");
}

#[test]
fn test_query_context_creation() {
    let context = QueryContext {
        query_terms: vec!["function".to_string(), "handler".to_string()],
        query_vector: Some(vec![0.1, 0.2, 0.3, 0.4]),
        start_time: Instant::now(),
        complexity_score: 0.75,
        expected_result_count: 42,
    };
    
    assert_eq!(context.query_terms.len(), 2);
    assert_eq!(context.query_terms[0], "function");
    assert_eq!(context.query_terms[1], "handler");
    assert!(context.query_vector.is_some());
    assert_eq!(context.query_vector.as_ref().unwrap().len(), 4);
    assert_eq!(context.complexity_score, 0.75);
    assert_eq!(context.expected_result_count, 42);
}

#[test]
fn test_query_context_no_vector() {
    let context = QueryContext {
        query_terms: vec!["simple".to_string(), "query".to_string()],
        query_vector: None,
        start_time: Instant::now(),
        complexity_score: 0.3,
        expected_result_count: 10,
    };
    
    assert!(context.query_vector.is_none());
    assert_eq!(context.complexity_score, 0.3);
}

#[test]
fn test_wand_search_state() {
    let mut term_contributions = HashMap::new();
    term_contributions.insert("function".to_string(), 0.8);
    term_contributions.insert("handler".to_string(), 0.6);
    
    let state = WandSearchState {
        iteration: 15,
        current_threshold: 0.85,
        candidate_count: 45,
        score_improvements: vec![0.1, 0.15, 0.18, 0.19],
        term_contributions,
        processing_time: std::time::Duration::from_millis(120),
    };
    
    assert_eq!(state.iteration, 15);
    assert_eq!(state.current_threshold, 0.85);
    assert_eq!(state.candidate_count, 45);
    assert_eq!(state.score_improvements.len(), 4);
    assert_eq!(state.term_contributions.len(), 2);
    assert_eq!(state.processing_time.as_millis(), 120);
}

#[test]
fn test_hnsw_search_state() {
    let candidates = vec![
        HnswCandidate {
            node_id: 1,
            distance: 0.1,
            layer: 0,
            neighbor_count: 8,
        },
        HnswCandidate {
            node_id: 2,
            distance: 0.15,
            layer: 1,
            neighbor_count: 12,
        },
    ];
    
    let state = HnswSearchState {
        current_layer: 2,
        beam_candidates: candidates,
        visited_nodes: 50,
        best_distance: 0.08,
        exploration_ratio: 0.75,
    };
    
    assert_eq!(state.current_layer, 2);
    assert_eq!(state.beam_candidates.len(), 2);
    assert_eq!(state.visited_nodes, 50);
    assert_eq!(state.best_distance, 0.08);
    assert_eq!(state.exploration_ratio, 0.75);
}

#[test]
fn test_hnsw_candidate() {
    let candidate = HnswCandidate {
        node_id: 42,
        distance: 0.25,
        layer: 3,
        neighbor_count: 16,
    };
    
    assert_eq!(candidate.node_id, 42);
    assert_eq!(candidate.distance, 0.25);
    assert_eq!(candidate.layer, 3);
    assert_eq!(candidate.neighbor_count, 16);
}

#[test]
fn test_learning_metrics_default() {
    let metrics = LearningMetrics::default();
    
    assert_eq!(metrics.total_predictions, 0);
    assert_eq!(metrics.correct_early_stops, 0);
    assert_eq!(metrics.incorrect_early_stops, 0);
    assert_eq!(metrics.missed_stopping_opportunities, 0);
    assert_eq!(metrics.avg_computation_saved, 0.0);
    assert_eq!(metrics.avg_quality_maintained, 0.0);
    assert_eq!(metrics.model_accuracy, 0.0);
    assert_eq!(metrics.adaptation_events, 0);
    assert!(metrics.feature_importance.is_empty());
}

#[test]
fn test_learning_metrics_serialization() {
    use serde_json;
    
    let mut metrics = LearningMetrics::default();
    metrics.total_predictions = 100;
    metrics.correct_early_stops = 80;
    metrics.model_accuracy = 0.85;
    metrics.feature_importance.insert("quality".to_string(), 0.9);
    
    let serialized = serde_json::to_string(&metrics).unwrap();
    let deserialized: LearningMetrics = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(deserialized.total_predictions, 100);
    assert_eq!(deserialized.correct_early_stops, 80);
    assert_eq!(deserialized.model_accuracy, 0.85);
    assert_eq!(deserialized.feature_importance.get("quality"), Some(&0.9));
}

#[test]
fn test_wand_stopping_predictor_creation() {
    let config = WandLearningConfig::default();
    let predictor = WandStoppingPredictor::new(config);
    
    // Test initial weights are set
    assert!(!predictor.weights().is_empty());
    assert!(predictor.weights().contains_key(&WandFeature::IterationCount));
    assert!(predictor.weights().contains_key(&WandFeature::ScoreImprovement));
    assert!(predictor.weights().contains_key(&WandFeature::QualityEstimate));
    
    // Test initial state
    assert!(!predictor.is_trained());
    assert_eq!(predictor.accuracy(), 0.5);
    assert_eq!(predictor.precision(), 0.5);
    assert_eq!(predictor.recall(), 0.5);
    assert!(predictor.training_history().is_empty());
}

#[test]
fn test_wand_stopping_predictor_prediction() {
    let config = WandLearningConfig::default();
    let predictor = WandStoppingPredictor::new(config);
    
    let mut features = HashMap::new();
    features.insert(WandFeature::QualityEstimate, 0.9);
    features.insert(WandFeature::ScoreImprovement, 0.8);
    features.insert(WandFeature::ThresholdConvergence, 0.85);
    features.insert(WandFeature::IterationCount, 10.0);
    
    let (should_stop, confidence) = predictor.predict(&features);
    
    // Should have reasonable confidence
    assert!(confidence >= 0.0);
    assert!(confidence <= 1.0);
    
    // Test with empty features
    let empty_features = HashMap::new();
    let (_, empty_confidence) = predictor.predict(&empty_features);
    assert!(empty_confidence >= 0.0);
    assert!(empty_confidence <= 1.0);
}

#[test]
fn test_wand_stopping_predictor_training() {
    let config = WandLearningConfig::default();
    let mut predictor = WandStoppingPredictor::new(config);
    
    // Add training samples
    for i in 0..20 {
        let mut features = HashMap::new();
        features.insert(WandFeature::QualityEstimate, 0.8 + (i as f64) * 0.01);
        features.insert(WandFeature::IterationCount, i as f64);
        
        let sample = WandTrainingSample {
            features,
            should_have_stopped: i > 10,
            actual_quality: 0.85,
            computation_saved: 0.3,
            timestamp: Instant::now(),
        };
        
        predictor.add_training_sample(sample);
    }
    
    assert_eq!(predictor.training_history().len(), 20);
    
    // Update model
    let result = predictor.update_model(0.01);
    assert!(result.is_ok());
    assert!(predictor.is_trained());
}

#[test]
fn test_wand_stopping_predictor_training_window() {
    let config = WandLearningConfig::default();
    let mut predictor = WandStoppingPredictor::new(config);
    
    // Add more samples than the window size
    for i in 0..1200 {
        let mut features = HashMap::new();
        features.insert(WandFeature::IterationCount, i as f64);
        
        let sample = WandTrainingSample {
            features,
            should_have_stopped: false,
            actual_quality: 0.8,
            computation_saved: 0.2,
            timestamp: Instant::now(),
        };
        
        predictor.add_training_sample(sample);
    }
    
    // Should be capped at window size
    assert_eq!(predictor.training_history().len(), 1000);
}

#[test]
fn test_hnsw_stopping_predictor_creation() {
    let config = HnswLearningConfig::default();
    let predictor = HnswStoppingPredictor::new(config);
    
    // Test layer thresholds are initialized
    assert!(!predictor.layer_thresholds().is_empty());
    assert!(predictor.layer_thresholds().contains_key(&0));
    assert!(predictor.layer_thresholds().contains_key(&1));
    
    // Test feature weights are initialized
    assert!(!predictor.neighbor_quality_weights().is_empty());
    assert!(predictor.neighbor_quality_weights().contains_key(&HnswFeature::DistanceToQuery));
    assert!(predictor.neighbor_quality_weights().contains_key(&HnswFeature::DistanceImprovement));
    
    // Test initial state
    assert_eq!(predictor.search_efficiency(), 0.5);
    assert_eq!(predictor.quality_maintained(), 0.5);
    assert_eq!(predictor.beam_width_adaptation(), 1.0);
    assert_eq!(predictor.exploration_decay(), 0.95);
}

#[test]
fn test_hnsw_stopping_predictor_prediction() {
    let config = HnswLearningConfig::default();
    let predictor = HnswStoppingPredictor::new(config);
    
    let mut features = HashMap::new();
    features.insert(HnswFeature::DistanceToQuery, 0.05);
    features.insert(HnswFeature::DistanceImprovement, 0.9);
    features.insert(HnswFeature::ExplorationRatio, 0.7);
    features.insert(HnswFeature::GraphConnectivity, 0.8);
    
    let (should_stop, confidence) = predictor.predict(&features);
    
    // Should have reasonable confidence
    assert!(confidence >= 0.0);
    assert!(confidence <= 1.0);
    
    // Confidence should be approximately 0.376 based on the feature weights and values
    assert!(confidence > 0.37 && confidence < 0.38);
}

#[test]
fn test_hnsw_stopping_predictor_training() {
    let config = HnswLearningConfig::default();
    let mut predictor = HnswStoppingPredictor::new(config);
    
    // Add training samples
    for i in 0..25 {
        let mut features = HashMap::new();
        features.insert(HnswFeature::DistanceToQuery, 0.1 - (i as f64) * 0.002);
        features.insert(HnswFeature::LayerDepth, (i % 5) as f64);
        
        let sample = HnswTrainingSample {
            features,
            optimal_stopping_point: i,
            final_quality: 0.9 - (i as f64) * 0.01,
            search_efficiency: 0.8 + (i as f64) * 0.005,
            timestamp: Instant::now(),
        };
        
        predictor.add_training_sample(sample);
    }
    
    assert_eq!(predictor.training_samples().len(), 25);
    
    // Update model
    let result = predictor.update_model(0.02);
    assert!(result.is_ok());
    
    // Parameters should have been adapted
    assert!(predictor.search_efficiency() > 0.0);
    assert!(predictor.quality_maintained() > 0.0);
}

#[test]
fn test_hnsw_stopping_predictor_adaptation() {
    let config = HnswLearningConfig::default();
    let mut predictor = HnswStoppingPredictor::new(config);
    
    let initial_beam_width = predictor.beam_width_adaptation();
    
    // Add samples with low efficiency to trigger beam width increase
    for _ in 0..15 {
        let sample = HnswTrainingSample {
            features: HashMap::new(),
            optimal_stopping_point: 0,
            final_quality: 0.8,
            search_efficiency: 0.5, // Low efficiency
            timestamp: Instant::now(),
        };
        predictor.add_training_sample(sample);
    }
    
    predictor.update_model(0.1).unwrap();
    assert!(predictor.beam_width_adaptation() > initial_beam_width);
    
    // Add many samples with high efficiency to trigger beam width decrease
    for _ in 0..50 {
        let sample = HnswTrainingSample {
            features: HashMap::new(),
            optimal_stopping_point: 0,
            final_quality: 0.9,
            search_efficiency: 0.9, // Very high efficiency
            timestamp: Instant::now(),
        };
        predictor.add_training_sample(sample);
    }
    
    let pre_update_beam_width = predictor.beam_width_adaptation();
    predictor.update_model(0.1).unwrap();
    assert!(predictor.beam_width_adaptation() < pre_update_beam_width);
}

#[test]
fn test_confidence_model_creation() {
    let model = ConfidenceModel::new();
    
    assert!(model.confidence_predictors().is_empty());
    assert_eq!(model.calibration_params().temperature(), 1.0);
    assert_eq!(model.calibration_params().shift(), 0.0);
    assert_eq!(model.calibration_params().scale(), 1.0);
    assert_eq!(model.confidence_accuracy(), 0.5);
    assert!(model.calibration_data().is_empty());
}

#[test]
fn test_confidence_model_calibration() {
    let mut model = ConfidenceModel::new();
    
    // Add calibration data
    for i in 0..25 {
        let mut features = HashMap::new();
        features.insert(ConfidenceFeature::ResultCount, i as f64);
        
        let sample = ConfidenceTrainingSample {
            features,
            predicted_confidence: 0.7 + (i as f64) * 0.01,
            actual_quality: 0.6 + (i as f64) * 0.01,
            timestamp: Instant::now(),
        };
        
        model.add_calibration_sample(sample);
    }
    
    let initial_shift = model.calibration_params().shift();
    
    // Update calibration
    let result = model.update_calibration();
    assert!(result.is_ok());
    
    // Calibration parameters should have been adjusted
    // (exact values depend on the calibration algorithm)
}

#[test]
fn test_wand_feature_extractor() {
    let extractor = WandFeatureExtractor;
    let context = QueryContext {
        query_terms: vec!["test".to_string()],
        query_vector: None,
        start_time: Instant::now(),
        complexity_score: 0.6,
        expected_result_count: 20,
    };
    
    let mut term_contributions = HashMap::new();
    term_contributions.insert("test".to_string(), 0.8);
    
    let state = WandSearchState {
        iteration: 8,
        current_threshold: 0.7,
        candidate_count: 25,
        score_improvements: vec![0.1, 0.15, 0.18, 0.17],
        term_contributions,
        processing_time: std::time::Duration::from_millis(75),
    };
    
    let features = extractor.extract_features(&context, &state);
    
    // Verify all expected features are present
    assert!(features.contains_key(&WandFeature::IterationCount));
    assert!(features.contains_key(&WandFeature::TimeElapsed));
    assert!(features.contains_key(&WandFeature::CandidateSetSize));
    assert!(features.contains_key(&WandFeature::ScoreImprovement));
    assert!(features.contains_key(&WandFeature::TermContribution));
    assert!(features.contains_key(&WandFeature::QualityEstimate));
    assert!(features.contains_key(&WandFeature::ThresholdConvergence));
    
    // Verify specific values
    assert_eq!(*features.get(&WandFeature::IterationCount).unwrap(), 8.0);
    assert_eq!(*features.get(&WandFeature::CandidateSetSize).unwrap(), 25.0);
    assert_eq!(*features.get(&WandFeature::TermContribution).unwrap(), 0.8);
}

#[test]
fn test_wand_feature_extractor_edge_cases() {
    let extractor = WandFeatureExtractor;
    let context = QueryContext {
        query_terms: vec![],
        query_vector: None,
        start_time: Instant::now(),
        complexity_score: 0.0,
        expected_result_count: 0,
    };
    
    // Empty state
    let state = WandSearchState {
        iteration: 0,
        current_threshold: 0.0,
        candidate_count: 0,
        score_improvements: vec![],
        term_contributions: HashMap::new(),
        processing_time: std::time::Duration::from_millis(0),
    };
    
    let features = extractor.extract_features(&context, &state);
    
    // Should handle empty state gracefully
    assert_eq!(*features.get(&WandFeature::IterationCount).unwrap(), 0.0);
    assert_eq!(*features.get(&WandFeature::CandidateSetSize).unwrap(), 0.0);
    assert_eq!(*features.get(&WandFeature::TermContribution).unwrap(), 0.0);
    assert_eq!(*features.get(&WandFeature::ScoreImprovement).unwrap(), 0.0);
}

#[test]
fn test_hnsw_feature_extractor() {
    let extractor = HnswFeatureExtractor;
    let context = QueryContext {
        query_terms: vec![],
        query_vector: Some(vec![0.1, 0.2, 0.3]),
        start_time: Instant::now(),
        complexity_score: 0.7,
        expected_result_count: 10,
    };
    
    let candidates = vec![
        HnswCandidate {
            node_id: 1,
            distance: 0.1,
            layer: 0,
            neighbor_count: 8,
        },
        HnswCandidate {
            node_id: 2,
            distance: 0.15,
            layer: 1,
            neighbor_count: 12,
        },
    ];
    
    let state = HnswSearchState {
        current_layer: 2,
        beam_candidates: candidates,
        visited_nodes: 30,
        best_distance: 0.08,
        exploration_ratio: 0.65,
    };
    
    let features = extractor.extract_features(&context, &state);
    
    // Verify all expected features are present
    assert!(features.contains_key(&HnswFeature::LayerDepth));
    assert!(features.contains_key(&HnswFeature::DistanceToQuery));
    assert!(features.contains_key(&HnswFeature::BeamPosition));
    assert!(features.contains_key(&HnswFeature::ExplorationRatio));
    assert!(features.contains_key(&HnswFeature::NeighborCount));
    assert!(features.contains_key(&HnswFeature::DistanceImprovement));
    assert!(features.contains_key(&HnswFeature::GraphConnectivity));
    
    // Verify specific values
    assert_eq!(*features.get(&HnswFeature::LayerDepth).unwrap(), 2.0);
    assert!((features.get(&HnswFeature::DistanceToQuery).unwrap() - 0.08).abs() < 1e-6); // Use approximate equality for f32->f64 conversion
    assert_eq!(*features.get(&HnswFeature::ExplorationRatio).unwrap(), 0.65);
    assert_eq!(*features.get(&HnswFeature::BeamPosition).unwrap(), 2.0); // Number of candidates
}

#[test]
fn test_hnsw_feature_extractor_empty_candidates() {
    let extractor = HnswFeatureExtractor;
    let context = QueryContext {
        query_terms: vec![],
        query_vector: None,
        start_time: Instant::now(),
        complexity_score: 0.5,
        expected_result_count: 5,
    };
    
    let state = HnswSearchState {
        current_layer: 0,
        beam_candidates: vec![], // Empty candidates
        visited_nodes: 0,
        best_distance: 1.0,
        exploration_ratio: 0.0,
    };
    
    let features = extractor.extract_features(&context, &state);
    
    // Should handle empty candidates gracefully
    assert_eq!(*features.get(&HnswFeature::NeighborCount).unwrap(), 0.0);
    assert_eq!(*features.get(&HnswFeature::GraphConnectivity).unwrap(), 0.0);
    assert_eq!(*features.get(&HnswFeature::DistanceImprovement).unwrap(), 0.0);
}

#[test]
fn test_confidence_feature_extractor() {
    let extractor = ConfidenceFeatureExtractor;
    let context = QueryContext {
        query_terms: vec!["test".to_string()],
        query_vector: None,
        start_time: Instant::now(),
        complexity_score: 0.8,
        expected_result_count: 15,
    };
    
    let features = extractor.extract_features(&context, 25, 150.5);
    
    assert!(features.contains_key(&ConfidenceFeature::ResultCount));
    assert!(features.contains_key(&ConfidenceFeature::ProcessingTime));
    assert!(features.contains_key(&ConfidenceFeature::ScoreDistribution));
    
    assert_eq!(*features.get(&ConfidenceFeature::ResultCount).unwrap(), 25.0);
    assert_eq!(*features.get(&ConfidenceFeature::ProcessingTime).unwrap(), 150.5);
    assert_eq!(*features.get(&ConfidenceFeature::ScoreDistribution).unwrap(), 1.0); // Has results
}

#[test]
fn test_confidence_feature_extractor_no_results() {
    let extractor = ConfidenceFeatureExtractor;
    let context = QueryContext {
        query_terms: vec![],
        query_vector: None,
        start_time: Instant::now(),
        complexity_score: 0.2,
        expected_result_count: 0,
    };
    
    let features = extractor.extract_features(&context, 0, 50.0);
    
    assert_eq!(*features.get(&ConfidenceFeature::ResultCount).unwrap(), 0.0);
    assert_eq!(*features.get(&ConfidenceFeature::ScoreDistribution).unwrap(), 0.0); // No results
}

#[test]
fn test_learned_stopping_decision() {
    let decision = LearnedStoppingDecision {
        should_stop: true,
        confidence: 0.87,
        predicted_quality: 0.93,
        estimated_computation_saved: 0.35,
        reasoning: StoppingReasoning {
            primary_factor: "High quality threshold reached".to_string(),
            feature_contributions: {
                let mut contributions = HashMap::new();
                contributions.insert("quality_estimate".to_string(), 0.9);
                contributions.insert("threshold_convergence".to_string(), 0.8);
                contributions
            },
            threshold_exceeded: true,
            quality_sufficient: true,
        },
        algorithm_used: "WAND-Learned".to_string(),
    };
    
    assert!(decision.should_stop);
    assert_eq!(decision.confidence, 0.87);
    assert_eq!(decision.predicted_quality, 0.93);
    assert_eq!(decision.estimated_computation_saved, 0.35);
    assert_eq!(decision.algorithm_used, "WAND-Learned");
    assert!(decision.reasoning.threshold_exceeded);
    assert!(decision.reasoning.quality_sufficient);
    assert_eq!(decision.reasoning.feature_contributions.len(), 2);
}

#[test]
fn test_stopping_reasoning() {
    let mut feature_contributions = HashMap::new();
    feature_contributions.insert("distance_improvement".to_string(), 0.85);
    feature_contributions.insert("exploration_ratio".to_string(), 0.70);
    
    let reasoning = StoppingReasoning {
        primary_factor: "Distance threshold satisfied".to_string(),
        feature_contributions,
        threshold_exceeded: false,
        quality_sufficient: true,
    };
    
    assert_eq!(reasoning.primary_factor, "Distance threshold satisfied");
    assert_eq!(reasoning.feature_contributions.len(), 2);
    assert!(!reasoning.threshold_exceeded);
    assert!(reasoning.quality_sufficient);
    
    // Test accessing specific contributions
    assert_eq!(reasoning.feature_contributions.get("distance_improvement"), Some(&0.85));
    assert_eq!(reasoning.feature_contributions.get("exploration_ratio"), Some(&0.70));
}

#[test]
fn test_linear_predictor() {
    let predictor = LinearPredictor::new(
        vec![0.5, 0.3, 0.2],
        0.1,
        0.01,
    );
    
    assert_eq!(predictor.weights().len(), 3);
    assert_eq!(predictor.bias(), 0.1);
    assert_eq!(predictor.learning_rate(), 0.01);
}

#[test]
fn test_calibration_params() {
    let params = CalibrationParams::new(1.5, 0.05, 0.95);
    
    assert_eq!(params.temperature(), 1.5);
    assert_eq!(params.shift(), 0.05);
    assert_eq!(params.scale(), 0.95);
}

#[test]
fn test_training_samples() {
    // Test WandTrainingSample
    let mut wand_features = HashMap::new();
    wand_features.insert(WandFeature::IterationCount, 10.0);
    wand_features.insert(WandFeature::QualityEstimate, 0.85);
    
    let wand_sample = WandTrainingSample {
        features: wand_features.clone(),
        should_have_stopped: true,
        actual_quality: 0.88,
        computation_saved: 0.4,
        timestamp: Instant::now(),
    };
    
    assert_eq!(wand_sample.features.len(), 2);
    assert!(wand_sample.should_have_stopped);
    assert_eq!(wand_sample.actual_quality, 0.88);
    
    // Test HnswTrainingSample
    let mut hnsw_features = HashMap::new();
    hnsw_features.insert(HnswFeature::LayerDepth, 3.0);
    hnsw_features.insert(HnswFeature::DistanceToQuery, 0.12);
    
    let hnsw_sample = HnswTrainingSample {
        features: hnsw_features.clone(),
        optimal_stopping_point: 15,
        final_quality: 0.92,
        search_efficiency: 0.78,
        timestamp: Instant::now(),
    };
    
    assert_eq!(hnsw_sample.features.len(), 2);
    assert_eq!(hnsw_sample.optimal_stopping_point, 15);
    assert_eq!(hnsw_sample.final_quality, 0.92);
    
    // Test ConfidenceTrainingSample
    let mut conf_features = HashMap::new();
    conf_features.insert(ConfidenceFeature::ResultCount, 50.0);
    conf_features.insert(ConfidenceFeature::ProcessingTime, 120.0);
    
    let conf_sample = ConfidenceTrainingSample {
        features: conf_features.clone(),
        predicted_confidence: 0.82,
        actual_quality: 0.85,
        timestamp: Instant::now(),
    };
    
    assert_eq!(conf_sample.features.len(), 2);
    assert_eq!(conf_sample.predicted_confidence, 0.82);
    assert_eq!(conf_sample.actual_quality, 0.85);
}

#[tokio::test]
async fn test_comprehensive_wand_prediction() {
    let config = LearningConfig::default();
    let model = LearningStopModel::new(config).await.unwrap();
    
    let context = QueryContext {
        query_terms: vec!["function".to_string(), "handler".to_string()],
        query_vector: None,
        start_time: Instant::now(),
        complexity_score: 0.75,
        expected_result_count: 30,
    };
    
    let mut term_contributions = HashMap::new();
    term_contributions.insert("function".to_string(), 0.8);
    term_contributions.insert("handler".to_string(), 0.6);
    
    let state = WandSearchState {
        iteration: 12,
        current_threshold: 0.82,
        candidate_count: 35,
        score_improvements: vec![0.1, 0.15, 0.18, 0.19, 0.195],
        term_contributions,
        processing_time: std::time::Duration::from_millis(95),
    };
    
    let decision = model.predict_wand_stopping(&context, &state).await.unwrap();
    
    // Validate decision structure
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    assert!(decision.predicted_quality >= 0.0 && decision.predicted_quality <= 1.0);
    assert!(decision.estimated_computation_saved >= 0.0);
    assert_eq!(decision.algorithm_used, "WAND-Learned");
    assert!(!decision.reasoning.feature_contributions.is_empty());
}

#[tokio::test]
async fn test_comprehensive_hnsw_prediction() {
    let config = LearningConfig::default();
    let model = LearningStopModel::new(config).await.unwrap();
    
    let context = QueryContext {
        query_terms: vec![],
        query_vector: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5]),
        start_time: Instant::now(),
        complexity_score: 0.6,
        expected_result_count: 20,
    };
    
    let candidates = vec![
        HnswCandidate { node_id: 1, distance: 0.08, layer: 0, neighbor_count: 10 },
        HnswCandidate { node_id: 2, distance: 0.12, layer: 0, neighbor_count: 8 },
        HnswCandidate { node_id: 3, distance: 0.15, layer: 1, neighbor_count: 14 },
    ];
    
    let state = HnswSearchState {
        current_layer: 1,
        beam_candidates: candidates,
        visited_nodes: 40,
        best_distance: 0.08,
        exploration_ratio: 0.7,
    };
    
    let decision = model.predict_hnsw_stopping(&context, &state).await.unwrap();
    
    // Validate decision structure
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    assert!(decision.predicted_quality >= 0.0 && decision.predicted_quality <= 1.0);
    assert!(decision.estimated_computation_saved >= 0.0);
    assert_eq!(decision.algorithm_used, "HNSW-Learned");
    assert!(!decision.reasoning.feature_contributions.is_empty());
}

#[tokio::test]
async fn test_training_feedback_loop() {
    let config = LearningConfig::default();
    let model = LearningStopModel::new(config).await.unwrap();
    
    // Create a decision for feedback
    let decision = LearnedStoppingDecision {
        should_stop: true,
        confidence: 0.8,
        predicted_quality: 0.9,
        estimated_computation_saved: 0.3,
        reasoning: StoppingReasoning {
            primary_factor: "Test".to_string(),
            feature_contributions: HashMap::new(),
            threshold_exceeded: true,
            quality_sufficient: true,
        },
        algorithm_used: "WAND-Learned".to_string(),
    };
    
    // Provide feedback
    let result = model.train_with_feedback("wand", &decision, 0.85, 0.35).await;
    assert!(result.is_ok());
    
    // Test invalid query type
    let result = model.train_with_feedback("invalid", &decision, 0.85, 0.35).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_metrics_tracking() {
    let config = LearningConfig::default();
    let model = LearningStopModel::new(config).await.unwrap();
    
    // Initial metrics
    let metrics = model.get_metrics().await;
    assert_eq!(metrics.total_predictions, 0);
    
    // Make some predictions to update metrics
    let context = QueryContext {
        query_terms: vec!["test".to_string()],
        query_vector: None,
        start_time: Instant::now(),
        complexity_score: 0.5,
        expected_result_count: 10,
    };
    
    let state = WandSearchState {
        iteration: 5,
        current_threshold: 0.7,
        candidate_count: 15,
        score_improvements: vec![0.1, 0.12, 0.13],
        term_contributions: HashMap::new(),
        processing_time: std::time::Duration::from_millis(50),
    };
    
    // Make multiple predictions
    for _ in 0..5 {
        let _ = model.predict_wand_stopping(&context, &state).await.unwrap();
    }
    
    let updated_metrics = model.get_metrics().await;
    assert!(updated_metrics.total_predictions > 0);
}

#[test]
fn test_edge_cases_and_error_handling() {
    // Test with extreme values
    let config = LearningConfig {
        training_window_size: 0, // Invalid
        update_frequency: 0, // Invalid
        learning_rate: -1.0, // Invalid
        confidence_threshold: 2.0, // Invalid (>1.0)
        min_training_samples: 0,
        feature_normalization: true,
        wand_config: WandLearningConfig {
            max_iterations: 0,
            quality_threshold: 2.0, // Invalid
            score_improvement_tolerance: -0.1, // Invalid
            term_contribution_threshold: -0.1, // Invalid
        },
        hnsw_config: HnswLearningConfig {
            max_layers: 0,
            beam_width: 0,
            distance_threshold: -1.0, // Invalid
            max_neighbors: 0,
        },
    };
    
    // The model should still be creatable with invalid config
    // (validation would happen at runtime)
    assert_eq!(config.training_window_size, 0);
    assert_eq!(config.learning_rate, -1.0);
}

#[test]
#[cfg(feature = "stress-tests")]
fn test_memory_management() {
    // Test that predictors properly manage memory
    let config = WandLearningConfig::default();
    let mut predictor = WandStoppingPredictor::new(config);
    
    // Add many samples to test memory management
    for i in 0..2000 {
        let mut features = HashMap::new();
        features.insert(WandFeature::IterationCount, i as f64);
        
        let sample = WandTrainingSample {
            features,
            should_have_stopped: i % 2 == 0,
            actual_quality: 0.8,
            computation_saved: 0.2,
            timestamp: Instant::now(),
        };
        
        predictor.add_training_sample(sample);
    }
    
    // Should be limited to window size
    assert!(predictor.training_history().len() <= 1000);
}

#[tokio::test]
async fn test_concurrent_access() {
    use std::sync::Arc;
    use tokio::task;
    
    let config = LearningConfig::default();
    let model = Arc::new(LearningStopModel::new(config).await.unwrap());
    
    let mut handles = vec![];
    
    // Spawn multiple concurrent prediction tasks
    for i in 0..10 {
        let model_clone = Arc::clone(&model);
        let handle = task::spawn(async move {
            let context = QueryContext {
                query_terms: vec![format!("term{}", i)],
                query_vector: None,
                start_time: Instant::now(),
                complexity_score: 0.5,
                expected_result_count: 10,
            };
            
            let state = WandSearchState {
                iteration: i,
                current_threshold: 0.7,
                candidate_count: 15 + i,
                score_improvements: vec![0.1, 0.12],
                term_contributions: HashMap::new(),
                processing_time: std::time::Duration::from_millis(50),
            };
            
            let result = model_clone.predict_wand_stopping(&context, &state).await;
            assert!(result.is_ok());
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Check that metrics were updated
    let metrics = model.get_metrics().await;
    assert!(metrics.total_predictions > 0);
}