/**
 * Configuration for Sprint-1 tail-taming features
 * Implements Section 2 of TODO.md: hedged probes, cooperative cancel, TA/NRA, learning-to-stop
 */

export interface TailTamingConfig {
  // Feature flags
  TAIL_HEDGE: boolean;
  HEDGE_DELAY_MS: number;
  TA_STOP: boolean;          // Cross-shard TA/NRA
  LTS_STOP: boolean;         // Learning-to-stop

  // Hedged probe configuration
  hedge_delay_base: number;           // Base delay before sending hedge probe
  hedge_delay_p50_multiplier: number; // Multiplier for p50 latency
  max_hedge_delay: number;            // Maximum hedge delay cap
  cancel_on_first_success: boolean;   // Cancel other requests on first success

  // Cross-shard TA/NRA parameters
  global_min_score_threshold: number; // Global threshold for stopping
  shard_early_stop_threshold: number; // Individual shard stop threshold
  max_concurrent_shards: number;      // Maximum shards to query simultaneously

  // Learning-to-stop parameters
  lts_feature_weights: {
    top_k_gap: number;        // Weight for top-k score gap feature
    shard_residuals: number;  // Weight for shard residuals feature  
    elapsed_time: number;     // Weight for elapsed time feature
    ef_search: number;        // Weight for efSearch feature
  };
  lts_stop_probability_threshold: number; // Probability threshold for stopping
  lts_monotone_floor: number;            // Minimum probability floor

  // Performance gates (vs v2.2 baseline)
  gates: {
    p99_latency_improvement_min: number;  // -10% to -15%
    p99_latency_improvement_max: number;
    p99_p95_ratio_max: number;           // ≤ 2.0
    sla_recall_at_50_delta_min: number;  // ≥ 0.0 pp
    qps_at_150ms_improvement_min: number; // +10% to +15%
    qps_at_150ms_improvement_max: number;
    cost_increase_max: number;           // ≤ +5%
  };

  // Canary rollout configuration
  canary: {
    stages: number[];                    // [5, 25, 50, 100] % traffic
    stage_duration_minutes: number;     // Duration per stage
    gate_check_interval_minutes: number; // How often to check gates
    consecutive_failures_to_revert: number; // Failures before auto-revert
    repo_bucket_strategy: 'hash' | 'list'; // How to bucket repositories
  };
}

// Default configuration based on TODO.md specifications
export function getDefaultTailTamingConfig(): TailTamingConfig {
  return {
    // Feature flags - all disabled by default, enabled via environment
    TAIL_HEDGE: process.env.TAIL_HEDGE === 'true',
    HEDGE_DELAY_MS: parseInt(process.env.HEDGE_DELAY_MS || '6', 10),
    TA_STOP: process.env.TA_STOP === 'true', 
    LTS_STOP: process.env.LTS_STOP === 'true',

    // Hedged probe configuration
    hedge_delay_base: parseInt(process.env.HEDGE_DELAY_MS || '6', 10),
    hedge_delay_p50_multiplier: 0.1, // min(6ms, 0.1 * p50_shard)
    max_hedge_delay: 50, // Cap at 50ms
    cancel_on_first_success: true,

    // Cross-shard TA/NRA
    global_min_score_threshold: parseFloat(process.env.GLOBAL_MIN_SCORE || '0.5'),
    shard_early_stop_threshold: parseFloat(process.env.SHARD_STOP_THRESHOLD || '0.8'),
    max_concurrent_shards: parseInt(process.env.MAX_CONCURRENT_SHARDS || '4', 10),

    // Learning-to-stop weights
    lts_feature_weights: {
      top_k_gap: 0.3,
      shard_residuals: 0.25,
      elapsed_time: 0.25, 
      ef_search: 0.2
    },
    lts_stop_probability_threshold: 0.7,
    lts_monotone_floor: 0.1,

    // Performance gates from TODO.md
    gates: {
      p99_latency_improvement_min: -0.15,  // -15%
      p99_latency_improvement_max: -0.10,  // -10% 
      p99_p95_ratio_max: 2.0,
      sla_recall_at_50_delta_min: 0.0,     // >= 0.0 pp
      qps_at_150ms_improvement_min: 0.10,   // +10%
      qps_at_150ms_improvement_max: 0.15,   // +15%
      cost_increase_max: 0.05              // +5%
    },

    // Canary rollout from TODO.md
    canary: {
      stages: [5, 25, 50, 100],
      stage_duration_minutes: 60,         // 1 hour per stage
      gate_check_interval_minutes: 15,    // Check every 15 minutes
      consecutive_failures_to_revert: 2,  // 2 consecutive 15-min windows
      repo_bucket_strategy: 'hash'        // Hash-based bucketing
    }
  };
}

export function validateTailTamingConfig(config: TailTamingConfig): string[] {
  const errors: string[] = [];

  // Validate hedge delay
  if (config.HEDGE_DELAY_MS < 1 || config.HEDGE_DELAY_MS > 100) {
    errors.push('HEDGE_DELAY_MS must be between 1-100ms');
  }

  // Validate thresholds
  if (config.global_min_score_threshold < 0 || config.global_min_score_threshold > 1) {
    errors.push('global_min_score_threshold must be between 0.0 and 1.0');
  }

  if (config.shard_early_stop_threshold < 0 || config.shard_early_stop_threshold > 1) {
    errors.push('shard_early_stop_threshold must be between 0.0 and 1.0');
  }

  // Validate LTS weights sum to approximately 1.0
  const weightsSum = Object.values(config.lts_feature_weights).reduce((sum, w) => sum + w, 0);
  if (Math.abs(weightsSum - 1.0) > 0.1) {
    errors.push(`LTS feature weights should sum to ~1.0, got ${weightsSum}`);
  }

  // Validate gates
  if (config.gates.p99_latency_improvement_min >= config.gates.p99_latency_improvement_max) {
    errors.push('p99_latency_improvement_min must be less than max');
  }

  if (config.gates.qps_at_150ms_improvement_min >= config.gates.qps_at_150ms_improvement_max) {
    errors.push('qps_at_150ms_improvement_min must be less than max');
  }

  // Validate canary stages
  const stages = config.canary.stages;
  if (!stages.includes(100)) {
    errors.push('Canary stages must include 100% rollout');
  }

  for (let i = 1; i < stages.length; i++) {
    if (stages[i] <= stages[i-1]) {
      errors.push('Canary stages must be in ascending order');
    }
  }

  return errors;
}

// Feature flag helpers
export function isTailHedgingEnabled(config: TailTamingConfig): boolean {
  return config.TAIL_HEDGE;
}

export function isCrossShardTAEnabled(config: TailTamingConfig): boolean {
  return config.TA_STOP;
}

export function isLearningToStopEnabled(config: TailTamingConfig): boolean {
  return config.LTS_STOP;
}

// Calculate hedge delay based on p50 shard latency
export function calculateHedgeDelay(config: TailTamingConfig, p50ShardLatency: number): number {
  const calculated = Math.min(
    config.hedge_delay_base,
    config.hedge_delay_p50_multiplier * p50ShardLatency
  );
  
  return Math.min(calculated, config.max_hedge_delay);
}