//! Cross-Shard Early Stopping Algorithms
//!
//! Implements Threshold Algorithm (TA) and No Random Access (NRA) optimization
//! for distributed early termination across pipeline stages.
//! 
//! Target: >30% reduction in unnecessary computation per TODO.md

use anyhow::{anyhow, Result};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;
use std::cmp::{Ordering, Reverse};
use tokio::sync::{RwLock, watch};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Cross-shard stopping coordinator
pub struct CrossShardStopper {
    /// Current threshold for stopping decision
    threshold: Arc<RwLock<f64>>,
    
    /// Shard coordinators
    shard_coordinators: Vec<Arc<ShardCoordinator>>,
    
    /// Global stopping signal
    stop_signal: watch::Sender<bool>,
    
    /// Stopping statistics
    stats: Arc<RwLock<StoppingStats>>,
    
    /// Configuration
    config: StoppingConfig,
}

/// Configuration for stopping algorithms
#[derive(Debug, Clone)]
pub struct StoppingConfig {
    /// Minimum confidence threshold for early stopping
    pub confidence_threshold: f64,
    
    /// Quality threshold for result acceptance
    pub quality_threshold: f64,
    
    /// Maximum computation budget (percentage)
    pub max_computation_budget: f64,
    
    /// Number of shards for distributed processing
    pub num_shards: usize,
    
    /// TA algorithm parameters
    pub ta_config: TAConfig,
    
    /// NRA algorithm parameters  
    pub nra_config: NRAConfig,
    
    /// Learning parameters for adaptive stopping
    pub learning_rate: f64,
    pub adaptation_window: usize,
}

/// Threshold Algorithm (TA) configuration
#[derive(Debug, Clone)]
pub struct TAConfig {
    /// Initial threshold value
    pub initial_threshold: f64,
    
    /// Threshold decay rate
    pub decay_rate: f64,
    
    /// Minimum threshold value
    pub min_threshold: f64,
    
    /// Top-k results to consider
    pub top_k: usize,
}

/// No Random Access (NRA) configuration
#[derive(Debug, Clone)]
pub struct NRAConfig {
    /// Buffer size for sorted access
    pub buffer_size: usize,
    
    /// Minimum sorted access ratio
    pub min_sorted_ratio: f64,
    
    /// Random access penalty
    pub random_access_penalty: f64,
}

/// Shard coordinator for distributed stopping
pub struct ShardCoordinator {
    shard_id: usize,
    threshold_receiver: watch::Receiver<bool>,
    local_candidates: Arc<RwLock<CandidateSet>>,
    ta_processor: TAProcessor,
    nra_processor: NRAProcessor,
    metrics: Arc<RwLock<ShardMetrics>>,
}

/// Set of candidate results
#[derive(Debug, Clone)]
pub struct CandidateSet {
    candidates: BinaryHeap<ScoredCandidate>,
    threshold: f64,
    total_processed: usize,
    early_stopped: bool,
}

/// Scored candidate result
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCandidate {
    pub id: String,
    pub score: f64,
    pub source_shard: usize,
    pub confidence: f64,
    pub processing_cost: f64,
}

impl Eq for ScoredCandidate {}

impl PartialOrd for ScoredCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.score.partial_cmp(&self.score) // Reverse for max-heap
    }
}

impl Ord for ScoredCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Threshold Algorithm processor
pub struct TAProcessor {
    config: TAConfig,
    sorted_access_count: usize,
    random_access_count: usize,
    current_threshold: f64,
    top_k_heap: BinaryHeap<Reverse<ScoredCandidate>>, // Min-heap for top-k
}

/// No Random Access processor  
pub struct NRAProcessor {
    config: NRAConfig,
    sorted_buffers: HashMap<String, VecDeque<ScoredCandidate>>,
    seen_candidates: HashMap<String, PartialCandidate>,
    access_pattern_score: f64,
}

/// Partial candidate for NRA algorithm
#[derive(Debug, Clone)]
pub struct PartialCandidate {
    id: String,
    scores: HashMap<String, f64>,
    min_possible_score: f64,
    max_possible_score: f64,
    completeness: f64,
}

/// Stopping statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StoppingStats {
    pub total_queries: u64,
    pub early_stopped_queries: u64,
    pub avg_computation_saved: f64,
    pub avg_quality_maintained: f64,
    pub ta_stops: u64,
    pub nra_stops: u64,
    pub threshold_adaptations: u64,
    pub cross_shard_coordination_time_ms: f64,
}

/// Shard-level metrics
#[derive(Debug, Default, Clone)]
pub struct ShardMetrics {
    pub processed_candidates: usize,
    pub early_stops: usize,
    pub threshold_violations: usize,
    pub avg_processing_time_ms: f64,
}

impl Default for StoppingConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.85,
            quality_threshold: 0.8,
            max_computation_budget: 0.7, // 70% of time budget
            num_shards: 4,
            ta_config: TAConfig {
                initial_threshold: 0.5,
                decay_rate: 0.95,
                min_threshold: 0.1,
                top_k: 50,
            },
            nra_config: NRAConfig {
                buffer_size: 100,
                min_sorted_ratio: 0.3,
                random_access_penalty: 0.1,
            },
            learning_rate: 0.01,
            adaptation_window: 100,
        }
    }
}

impl CrossShardStopper {
    /// Create a new cross-shard stopper
    pub async fn new(config: StoppingConfig) -> Result<Self> {
        let threshold = Arc::new(RwLock::new(config.ta_config.initial_threshold));
        let (stop_sender, _) = watch::channel(false);
        let stats = Arc::new(RwLock::new(StoppingStats::default()));
        
        let mut shard_coordinators = Vec::new();
        
        // Create shard coordinators
        for shard_id in 0..config.num_shards {
            let coordinator = Arc::new(ShardCoordinator::new(
                shard_id,
                stop_sender.subscribe(),
                config.clone(),
            ).await?);
            shard_coordinators.push(coordinator);
        }
        
        info!("Initialized cross-shard stopper with {} shards", config.num_shards);
        
        Ok(Self {
            threshold,
            shard_coordinators,
            stop_signal: stop_sender,
            stats,
            config,
        })
    }
    
    /// Process query across all shards with early stopping
    pub async fn process_query(&self, query_id: &str, candidates: Vec<ScoredCandidate>) -> Result<StoppingDecision> {
        let start_time = std::time::Instant::now();
        
        // Distribute candidates across shards
        let distributed_candidates = self.distribute_candidates(candidates).await?;
        
        // Process in parallel across shards
        let mut handles = Vec::new();
        
        for (shard_idx, shard_candidates) in distributed_candidates.into_iter().enumerate() {
            let coordinator = self.shard_coordinators[shard_idx].clone();
            let query_id = query_id.to_string();
            
            let handle = tokio::spawn(async move {
                coordinator.process_shard_candidates(query_id, shard_candidates).await
            });
            handles.push(handle);
        }
        
        // Collect shard results
        let mut shard_results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(result)) => shard_results.push(result),
                Ok(Err(e)) => warn!("Shard processing failed: {:?}", e),
                Err(e) => warn!("Shard task failed: {:?}", e),
            }
        }
        
        // Make global stopping decision
        let decision = self.make_stopping_decision(shard_results).await?;
        
        // Update statistics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.update_stats(&decision, processing_time).await;
        
        // Adapt threshold based on results
        if decision.should_stop {
            self.adapt_threshold(decision.confidence, decision.quality_estimate).await;
        }
        
        debug!(
            "Query {} stopping decision: should_stop={}, confidence={:.3}, saved={:.1}%",
            query_id,
            decision.should_stop,
            decision.confidence,
            decision.computation_saved * 100.0
        );
        
        Ok(decision)
    }
    
    /// Distribute candidates across shards using hash-based partitioning
    async fn distribute_candidates(&self, candidates: Vec<ScoredCandidate>) -> Result<Vec<Vec<ScoredCandidate>>> {
        let mut distributed = vec![Vec::new(); self.config.num_shards];
        
        for candidate in candidates {
            // Use candidate ID for consistent hashing
            let shard_idx = self.hash_to_shard(&candidate.id);
            distributed[shard_idx].push(candidate);
        }
        
        Ok(distributed)
    }
    
    /// Hash candidate ID to shard index
    fn hash_to_shard(&self, id: &str) -> usize {
        // Simple hash function for consistent distribution
        let mut hash: u32 = 2166136261; // FNV-1a initial value
        
        for byte in id.bytes() {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(16777619); // FNV-1a prime
        }
        
        (hash as usize) % self.config.num_shards
    }
    
    /// Make global stopping decision from shard results
    async fn make_stopping_decision(&self, shard_results: Vec<ShardStoppingDecision>) -> Result<StoppingDecision> {
        if shard_results.is_empty() {
            return Ok(StoppingDecision {
                should_stop: true,
                reason: StoppingReason::NoResults,
                confidence: 0.0,
                quality_estimate: 0.0,
                computation_saved: 0.0,
                top_candidates: Vec::new(),
            });
        }
        
        // Aggregate shard decisions using voting
        let should_stop_votes = shard_results.iter()
            .filter(|r| r.should_stop)
            .count();
        
        let total_votes = shard_results.len();
        let stop_ratio = should_stop_votes as f64 / total_votes as f64;
        
        // Majority voting with confidence weighting
        let weighted_confidence: f64 = shard_results.iter()
            .map(|r| r.confidence * if r.should_stop { 1.0 } else { 0.5 })
            .sum::<f64>() / total_votes as f64;
        
        let avg_quality: f64 = shard_results.iter()
            .map(|r| r.quality_estimate)
            .sum::<f64>() / total_votes as f64;
        
        let avg_computation_saved: f64 = shard_results.iter()
            .map(|r| r.computation_saved)
            .sum::<f64>() / total_votes as f64;
        
        // Collect top candidates from all shards
        let mut all_candidates: Vec<_> = shard_results.iter()
            .flat_map(|r| r.top_candidates.iter())
            .cloned()
            .collect();
        
        all_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_candidates.truncate(self.config.ta_config.top_k);
        
        let should_stop = stop_ratio >= 0.5 && weighted_confidence >= self.config.confidence_threshold;
        
        let reason = if should_stop {
            if weighted_confidence >= 0.9 {
                StoppingReason::HighConfidence
            } else if avg_quality >= self.config.quality_threshold {
                StoppingReason::QualityThreshold
            } else {
                StoppingReason::MajorityVote
            }
        } else {
            StoppingReason::ContinueProcessing
        };
        
        Ok(StoppingDecision {
            should_stop,
            reason,
            confidence: weighted_confidence,
            quality_estimate: avg_quality,
            computation_saved: avg_computation_saved,
            top_candidates: all_candidates,
        })
    }
    
    /// Adapt threshold based on decision outcomes
    async fn adapt_threshold(&self, confidence: f64, quality: f64) {
        let mut threshold = self.threshold.write().await;
        let mut stats = self.stats.write().await;
        
        // Simple adaptive mechanism
        if confidence > 0.9 && quality > self.config.quality_threshold {
            // Very good results, can be more aggressive
            *threshold = (*threshold * (1.0 - self.config.learning_rate)).max(self.config.ta_config.min_threshold);
        } else if confidence < 0.7 || quality < self.config.quality_threshold * 0.8 {
            // Poor results, be more conservative
            *threshold = (*threshold * (1.0 + self.config.learning_rate)).min(1.0);
        }
        
        stats.threshold_adaptations += 1;
        
        debug!("Adapted threshold to {:.3} based on confidence={:.3}, quality={:.3}", 
               *threshold, confidence, quality);
    }
    
    /// Update stopping statistics
    async fn update_stats(&self, decision: &StoppingDecision, processing_time_ms: f64) {
        let mut stats = self.stats.write().await;
        
        stats.total_queries += 1;
        
        if decision.should_stop {
            stats.early_stopped_queries += 1;
            
            // Update rolling averages
            let total_stops = stats.early_stopped_queries as f64;
            stats.avg_computation_saved = (stats.avg_computation_saved * (total_stops - 1.0) + decision.computation_saved) / total_stops;
            stats.avg_quality_maintained = (stats.avg_quality_maintained * (total_stops - 1.0) + decision.quality_estimate) / total_stops;
        }
        
        // Update coordination time
        let total_queries = stats.total_queries as f64;
        stats.cross_shard_coordination_time_ms = (stats.cross_shard_coordination_time_ms * (total_queries - 1.0) + processing_time_ms) / total_queries;
    }
    
    /// Get current stopping statistics
    pub async fn get_stats(&self) -> StoppingStats {
        self.stats.read().await.clone()
    }
    
    /// Force stop all shard processing
    pub async fn force_stop(&self) -> Result<()> {
        self.stop_signal.send(true).map_err(|_| anyhow!("Failed to send stop signal"))?;
        info!("Forced stop signal sent to all shards");
        Ok(())
    }
}

/// Stopping decision result
#[derive(Debug, Clone)]
pub struct StoppingDecision {
    pub should_stop: bool,
    pub reason: StoppingReason,
    pub confidence: f64,
    pub quality_estimate: f64,
    pub computation_saved: f64,
    pub top_candidates: Vec<ScoredCandidate>,
}

/// Reasons for stopping decisions
#[derive(Debug, Clone, PartialEq)]
pub enum StoppingReason {
    HighConfidence,
    QualityThreshold,
    ComputationBudget,
    MajorityVote,
    NoResults,
    ContinueProcessing,
}

/// Shard-level stopping decision
#[derive(Debug, Clone)]
pub struct ShardStoppingDecision {
    pub shard_id: usize,
    pub should_stop: bool,
    pub confidence: f64,
    pub quality_estimate: f64,
    pub computation_saved: f64,
    pub top_candidates: Vec<ScoredCandidate>,
    pub algorithm_used: StoppingAlgorithm,
}

/// Stopping algorithms used
#[derive(Debug, Clone, PartialEq)]
pub enum StoppingAlgorithm {
    ThresholdAlgorithm,
    NoRandomAccess,
    Hybrid,
}

impl ShardCoordinator {
    /// Create new shard coordinator
    pub async fn new(
        shard_id: usize,
        stop_receiver: watch::Receiver<bool>,
        config: StoppingConfig,
    ) -> Result<Self> {
        let local_candidates = Arc::new(RwLock::new(CandidateSet::new(config.ta_config.initial_threshold)));
        let ta_processor = TAProcessor::new(config.ta_config.clone());
        let nra_processor = NRAProcessor::new(config.nra_config.clone());
        let metrics = Arc::new(RwLock::new(ShardMetrics::default()));
        
        Ok(Self {
            shard_id,
            threshold_receiver: stop_receiver,
            local_candidates,
            ta_processor,
            nra_processor,
            metrics,
        })
    }
    
    /// Process candidates for this shard
    pub async fn process_shard_candidates(&self, _query_id: String, candidates: Vec<ScoredCandidate>) -> Result<ShardStoppingDecision> {
        let start_time = std::time::Instant::now();
        
        // Check for global stop signal
        if *self.threshold_receiver.borrow() {
            return Ok(ShardStoppingDecision {
                shard_id: self.shard_id,
                should_stop: true,
                confidence: 1.0,
                quality_estimate: 0.0,
                computation_saved: 1.0,
                top_candidates: Vec::new(),
                algorithm_used: StoppingAlgorithm::ThresholdAlgorithm,
            });
        }
        
        // Determine which algorithm to use based on candidate characteristics
        let algorithm = self.select_algorithm(&candidates).await;
        
        let decision = match algorithm {
            StoppingAlgorithm::ThresholdAlgorithm => {
                self.process_with_ta(candidates).await?
            }
            StoppingAlgorithm::NoRandomAccess => {
                self.process_with_nra(candidates).await?
            }
            StoppingAlgorithm::Hybrid => {
                // Try TA first, fall back to NRA if needed
                let ta_result = self.process_with_ta(candidates.clone()).await;
                if ta_result.is_ok() {
                    ta_result?
                } else {
                    self.process_with_nra(candidates).await?
                }
            }
        };
        
        // Update shard metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.update_shard_metrics(processing_time, &decision).await;
        
        Ok(decision)
    }
    
    /// Select optimal algorithm based on candidate characteristics
    async fn select_algorithm(&self, candidates: &[ScoredCandidate]) -> StoppingAlgorithm {
        let avg_confidence: f64 = candidates.iter().map(|c| c.confidence).sum::<f64>() / candidates.len() as f64;
        let score_variance = self.calculate_score_variance(candidates);
        
        if avg_confidence > 0.8 && score_variance < 0.1 {
            // High confidence, low variance - TA works well
            StoppingAlgorithm::ThresholdAlgorithm
        } else if score_variance > 0.3 {
            // High variance - NRA handles uncertainty better
            StoppingAlgorithm::NoRandomAccess
        } else {
            // Mixed conditions - use hybrid approach
            StoppingAlgorithm::Hybrid
        }
    }
    
    /// Calculate score variance for algorithm selection
    fn calculate_score_variance(&self, candidates: &[ScoredCandidate]) -> f64 {
        if candidates.len() < 2 {
            return 0.0;
        }
        
        let mean: f64 = candidates.iter().map(|c| c.score).sum::<f64>() / candidates.len() as f64;
        let variance: f64 = candidates.iter()
            .map(|c| (c.score - mean).powi(2))
            .sum::<f64>() / candidates.len() as f64;
        
        variance.sqrt() / mean // Coefficient of variation
    }
    
    /// Process candidates using Threshold Algorithm
    async fn process_with_ta(&self, candidates: Vec<ScoredCandidate>) -> Result<ShardStoppingDecision> {
        // Implementation of TA algorithm logic
        let mut processed_count = 0;
        let mut top_candidates = Vec::new();
        
        // Sort candidates by score (descending)
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        let threshold = {
            let candidates_guard = self.local_candidates.read().await;
            candidates_guard.threshold
        };
        
        for candidate in sorted_candidates.iter() {
            processed_count += 1;
            
            if candidate.score >= threshold && candidate.confidence >= 0.7 {
                top_candidates.push(candidate.clone());
                
                // TA stopping condition: found enough high-quality candidates
                if top_candidates.len() >= 10 && candidate.confidence > 0.85 {
                    let computation_saved = 1.0 - (processed_count as f64 / sorted_candidates.len() as f64);
                    let avg_quality = top_candidates.iter().map(|c| c.confidence).sum::<f64>() / top_candidates.len() as f64;
                    
                    return Ok(ShardStoppingDecision {
                        shard_id: self.shard_id,
                        should_stop: true,
                        confidence: avg_quality,
                        quality_estimate: avg_quality,
                        computation_saved,
                        top_candidates,
                        algorithm_used: StoppingAlgorithm::ThresholdAlgorithm,
                    });
                }
            }
        }
        
        // Processed all candidates
        let avg_quality = if !top_candidates.is_empty() {
            top_candidates.iter().map(|c| c.confidence).sum::<f64>() / top_candidates.len() as f64
        } else {
            0.0
        };
        
        Ok(ShardStoppingDecision {
            shard_id: self.shard_id,
            should_stop: false,
            confidence: avg_quality,
            quality_estimate: avg_quality,
            computation_saved: 0.0,
            top_candidates,
            algorithm_used: StoppingAlgorithm::ThresholdAlgorithm,
        })
    }
    
    /// Process candidates using No Random Access algorithm
    async fn process_with_nra(&self, candidates: Vec<ScoredCandidate>) -> Result<ShardStoppingDecision> {
        // Implementation of NRA algorithm logic
        let mut top_candidates = Vec::new();
        let mut processed_count = 0;
        
        // NRA processes candidates in sorted order without random access
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        let mut candidate_bounds: HashMap<String, (f64, f64)> = HashMap::new();
        
        for candidate in sorted_candidates.iter() {
            processed_count += 1;
            
            // Update bounds for this candidate
            candidate_bounds.insert(
                candidate.id.clone(), 
                (candidate.score * 0.8, candidate.score * 1.2)
            );
            
            // NRA stopping condition: tight bounds and high confidence
            if candidate.confidence > 0.8 {
                let (min_bound, _max_bound) = candidate_bounds.get(&candidate.id).unwrap();
                
                if *min_bound > 0.5 { // Sufficiently high lower bound
                    top_candidates.push(candidate.clone());
                    
                    if top_candidates.len() >= 15 {
                        let computation_saved = 1.0 - (processed_count as f64 / sorted_candidates.len() as f64);
                        let avg_quality = top_candidates.iter().map(|c| c.confidence).sum::<f64>() / top_candidates.len() as f64;
                        
                        return Ok(ShardStoppingDecision {
                            shard_id: self.shard_id,
                            should_stop: true,
                            confidence: avg_quality,
                            quality_estimate: avg_quality,
                            computation_saved,
                            top_candidates,
                            algorithm_used: StoppingAlgorithm::NoRandomAccess,
                        });
                    }
                }
            }
        }
        
        let avg_quality = if !top_candidates.is_empty() {
            top_candidates.iter().map(|c| c.confidence).sum::<f64>() / top_candidates.len() as f64
        } else {
            0.0
        };
        
        Ok(ShardStoppingDecision {
            shard_id: self.shard_id,
            should_stop: false,
            confidence: avg_quality,
            quality_estimate: avg_quality,
            computation_saved: 0.0,
            top_candidates,
            algorithm_used: StoppingAlgorithm::NoRandomAccess,
        })
    }
    
    /// Update shard-level metrics
    async fn update_shard_metrics(&self, processing_time_ms: f64, decision: &ShardStoppingDecision) {
        let mut metrics = self.metrics.write().await;
        
        metrics.processed_candidates += decision.top_candidates.len();
        
        if decision.should_stop {
            metrics.early_stops += 1;
        }
        
        // Update rolling average processing time
        let total_processed = metrics.processed_candidates as f64;
        if total_processed > 0.0 {
            metrics.avg_processing_time_ms = (metrics.avg_processing_time_ms * (total_processed - 1.0) + processing_time_ms) / total_processed;
        }
    }
}

impl TAProcessor {
    pub fn new(config: TAConfig) -> Self {
        Self {
            config,
            sorted_access_count: 0,
            random_access_count: 0,
            current_threshold: config.initial_threshold,
            top_k_heap: BinaryHeap::new(),
        }
    }
}

impl NRAProcessor {
    pub fn new(config: NRAConfig) -> Self {
        Self {
            config,
            sorted_buffers: HashMap::new(),
            seen_candidates: HashMap::new(),
            access_pattern_score: 1.0,
        }
    }
}

impl CandidateSet {
    pub fn new(threshold: f64) -> Self {
        Self {
            candidates: BinaryHeap::new(),
            threshold,
            total_processed: 0,
            early_stopped: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candidate(id: &str, score: f64, confidence: f64) -> ScoredCandidate {
        ScoredCandidate {
            id: id.to_string(),
            score,
            source_shard: 0,
            confidence,
            processing_cost: 1.0,
        }
    }

    #[tokio::test]
    async fn test_cross_shard_stopper_creation() {
        let config = StoppingConfig::default();
        let stopper = CrossShardStopper::new(config).await;
        assert!(stopper.is_ok());
    }

    #[tokio::test]
    async fn test_candidate_distribution() {
        let config = StoppingConfig::default();
        let stopper = CrossShardStopper::new(config).await.unwrap();
        
        let candidates = vec![
            create_test_candidate("test1", 0.9, 0.8),
            create_test_candidate("test2", 0.8, 0.9),
            create_test_candidate("test3", 0.7, 0.7),
        ];
        
        let distributed = stopper.distribute_candidates(candidates).await.unwrap();
        
        // Should distribute across shards
        assert_eq!(distributed.len(), stopper.config.num_shards);
        
        // Total candidates should be preserved
        let total_distributed: usize = distributed.iter().map(|v| v.len()).sum();
        assert_eq!(total_distributed, 3);
    }

    #[tokio::test]
    async fn test_stopping_decision_high_confidence() {
        let config = StoppingConfig::default();
        let stopper = CrossShardStopper::new(config).await.unwrap();
        
        // High-confidence candidates that should trigger early stopping
        let candidates = vec![
            create_test_candidate("test1", 0.95, 0.95),
            create_test_candidate("test2", 0.90, 0.90),
            create_test_candidate("test3", 0.85, 0.88),
        ];
        
        let decision = stopper.process_query("test_query", candidates).await.unwrap();
        
        // Should decide to stop early with high confidence candidates
        assert_eq!(decision.should_stop, true);
        assert!(decision.confidence >= 0.8);
        assert!(!decision.top_candidates.is_empty());
    }

    #[tokio::test]
    async fn test_algorithm_selection() {
        let config = StoppingConfig::default();
        let stop_receiver = watch::channel(false).1;
        let coordinator = ShardCoordinator::new(0, stop_receiver, config).await.unwrap();
        
        // High confidence, low variance -> should prefer TA
        let high_conf_candidates = vec![
            create_test_candidate("test1", 0.9, 0.9),
            create_test_candidate("test2", 0.85, 0.85),
        ];
        
        let algorithm = coordinator.select_algorithm(&high_conf_candidates).await;
        assert_eq!(algorithm, StoppingAlgorithm::ThresholdAlgorithm);
        
        // High variance -> should prefer NRA
        let high_var_candidates = vec![
            create_test_candidate("test1", 0.9, 0.5),
            create_test_candidate("test2", 0.1, 0.8),
        ];
        
        let algorithm = coordinator.select_algorithm(&high_var_candidates).await;
        assert_eq!(algorithm, StoppingAlgorithm::NoRandomAccess);
    }

    #[tokio::test]
    async fn test_threshold_adaptation() {
        let config = StoppingConfig::default();
        let stopper = CrossShardStopper::new(config).await.unwrap();
        
        let initial_threshold = {
            let threshold = stopper.threshold.read().await;
            *threshold
        };
        
        // Adapt with high confidence and quality
        stopper.adapt_threshold(0.95, 0.9).await;
        
        let new_threshold = {
            let threshold = stopper.threshold.read().await;
            *threshold
        };
        
        // Threshold should decrease (more aggressive) with good results
        assert!(new_threshold <= initial_threshold);
    }
}