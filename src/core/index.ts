/**
 * Evergreen Systems Core Module Exports
 * 
 * This module exports all four evergreen optimization systems for easy integration:
 * 1. Program-Slice Recall (Stage-B++)
 * 2. Build/Test-Aware Priors
 * 3. Speculative Multi-Plan Planner  
 * 4. Cache Admission That Learns
 * 
 * Plus the integration coordinator and quality monitoring.
 */

// Program Slice Recall System
export {
  SymbolGraph,
  PathSensitiveSlicing,
  type SliceNode,
  type SliceEdge,
  type SliceResult,
  type SlicePath
} from './program-slice-recall.js';

// Build/Test-Aware Priors System
export {
  BuildTestPriors,
  BazelParser,
  GradleParser,
  CargoParser,
  type BuildTarget,
  type TestFailure,
  type ChangeEvent,
  type CodeOwner,
  type FilePrior
} from './build-test-priors.js';

// Speculative Multi-Plan Planner System
export {
  SpeculativeMultiPlanPlanner,
  QueryFeatureExtractor,
  PlanPredictor,
  type QueryPlan,
  type PlanPrediction,
  type PlanExecution,
  type CooperativeCancel
} from './speculative-multi-plan.js';

// Cache Admission That Learns System
export {
  CacheAdmissionLearner,
  CountMinSketch,
  SegmentedLRU,
  TinyLFUController,
  ReuseSignatureGenerator,
  type CacheEntry,
  type ReuseSignature,
  type TinyLFUConfig,
  type CacheStats
} from './cache-admission-learner.js';

// Quality Gates and Monitoring
export {
  EvergreenQualityMonitor,
  BaselineMetricsCollector,
  QualityGateEvaluator,
  type QualityGateResult,
  type SystemMetrics,
  type QualityGateConfig,
  type QualityMonitoringReport
} from './evergreen-quality-gates.js';

// Main Integration Coordinator
export {
  EvergreenSystemsIntegrator,
  type EvergreenSystemsConfig,
  type SystemsStatus,
  type SearchPipelineResult
} from './evergreen-systems-integration.js';

// Re-export core types that are needed
export type {
  SearchContext,
  Candidate,
  SymbolDefinition,
  SymbolReference
} from '../types/core.js';

export type {
  SearchHit
} from './span_resolver/types.js';