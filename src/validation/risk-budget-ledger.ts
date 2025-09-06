/**
 * Risk Budget Ledger - Per-Query Risk Tracking System
 * 
 * Core component for operational validation that tracks per-query risk spending
 * and budget enforcement according to the "trust-but-verify" approach.
 * 
 * Tracks: {risk, spend: {768d?|efSearchΔ?|MMR?}, headroom_ms, entropy_bin, pos_in_cands, outcome}
 */

import { z } from 'zod';
import { LensTracer, tracer, meter } from '../telemetry/tracer.js';
import type { SearchContext, SearchHit } from '../types/core.js';

// Entropy bins for risk classification
export enum EntropyBin {
  LOW = 'low',      // < 2.0 bits
  MEDIUM = 'medium', // 2.0-4.0 bits  
  HIGH = 'high'     // > 4.0 bits
}

// Risk spend sources
export enum RiskSpendSource {
  EMBEDDING_768D = '768d',
  EFFICIENT_SEARCH = 'efSearchΔ',
  MMR = 'MMR'
}

// Query outcome classification  
export enum QueryOutcome {
  SUCCESS = 'success',
  TIMEOUT = 'timeout',
  ERROR = 'error',
  DEGRADED = 'degraded'
}

// Risk ledger entry schema
export const RiskLedgerEntrySchema = z.object({
  trace_id: z.string(),
  timestamp: z.date(),
  query: z.string(),
  repo_sha: z.string(),
  risk_score: z.number().min(0).max(1),
  spend: z.object({
    [RiskSpendSource.EMBEDDING_768D]: z.number().min(0).optional(),
    [RiskSpendSource.EFFICIENT_SEARCH]: z.number().min(0).optional(), 
    [RiskSpendSource.MMR]: z.number().min(0).optional(),
  }),
  headroom_ms: z.number(),
  entropy_bin: z.nativeEnum(EntropyBin),
  pos_in_candidates: z.number().int().min(0),
  outcome: z.nativeEnum(QueryOutcome),
  // Additional context for analysis
  latency_breakdown: z.object({
    stage_a: z.number(),
    stage_b: z.number(), 
    stage_c: z.number().optional(),
    total: z.number(),
  }),
  candidates_processed: z.number().int(),
  final_results: z.number().int(),
  budget_remaining: z.number().min(0).max(1),
});

export type RiskLedgerEntry = z.infer<typeof RiskLedgerEntrySchema>;

// Risk budget configuration
interface RiskBudgetConfig {
  daily_risk_budget: number;
  upshift_rate_target: number; // 5%±2pp
  max_headroom_spend_ratio: number; // Only spend when p95_headroom ≥ h
  entropy_thresholds: {
    low: number;
    medium: number; 
  };
  knapsack_constraints: {
    max_concurrent_high_risk: number;
    budget_refresh_interval_ms: number;
  };
}

// Default configuration based on TODO.md requirements
const DEFAULT_RISK_CONFIG: RiskBudgetConfig = {
  daily_risk_budget: 1.0,
  upshift_rate_target: 0.05, // 5%
  max_headroom_spend_ratio: 0.8, // 80% of headroom
  entropy_thresholds: {
    low: 2.0,
    medium: 4.0,
  },
  knapsack_constraints: {
    max_concurrent_high_risk: 10,
    budget_refresh_interval_ms: 3600000, // 1 hour
  },
};

// Metrics for monitoring
const riskSpendMetrics = {
  total_spend: meter.createUpDownCounter('lens_risk_budget_spent_total', {
    description: 'Total risk budget spent',
  }),
  spend_by_source: meter.createHistogram('lens_risk_spend_by_source', {
    description: 'Risk spend breakdown by source',
  }),
  budget_utilization: meter.createObservableGauge('lens_risk_budget_utilization', {
    description: 'Current risk budget utilization ratio',
  }),
  upshift_rate: meter.createObservableGauge('lens_upshift_rate', {
    description: 'Current query upshift rate',
  }),
  headroom_tracking: meter.createHistogram('lens_headroom_usage', {
    description: 'Headroom usage patterns',
  }),
};

/**
 * Risk Budget Ledger - Core risk tracking and budget enforcement
 */
export class RiskBudgetLedger {
  private config: RiskBudgetConfig;
  private ledger: Map<string, RiskLedgerEntry[]>; // Keyed by date (YYYY-MM-DD)
  private currentBudgetSpent: number = 0;
  private lastBudgetRefresh: Date;
  private recentQueries: RiskLedgerEntry[] = []; // Sliding window for rate calculations

  constructor(config: Partial<RiskBudgetConfig> = {}) {
    this.config = { ...DEFAULT_RISK_CONFIG, ...config };
    this.ledger = new Map();
    this.lastBudgetRefresh = new Date();
  }

  /**
   * Calculate query entropy from query characteristics
   */
  private calculateQueryEntropy(query: string, context: SearchContext): number {
    const span = LensTracer.createChildSpan('calculate_entropy', {
      'lens.query_length': query.length,
      'lens.mode': context.mode,
    });

    try {
      // Simple entropy calculation based on query characteristics
      const tokens = query.toLowerCase().split(/\s+/);
      const uniqueTokens = new Set(tokens);
      const tokenFreq = new Map<string, number>();

      tokens.forEach(token => {
        tokenFreq.set(token, (tokenFreq.get(token) || 0) + 1);
      });

      let entropy = 0;
      const totalTokens = tokens.length;

      tokenFreq.forEach(freq => {
        const prob = freq / totalTokens;
        entropy -= prob * Math.log2(prob);
      });

      // Adjust for query mode and complexity
      if (context.mode === 'hybrid') entropy *= 1.2;
      if (uniqueTokens.size / tokens.length > 0.8) entropy *= 1.1; // High uniqueness bonus
      if (query.includes('(') || query.includes('[')) entropy *= 1.15; // Structural complexity

      span.setAttributes({
        'lens.calculated_entropy': entropy,
        'lens.unique_tokens': uniqueTokens.size,
        'lens.total_tokens': totalTokens,
      });

      return entropy;
    } finally {
      span.end();
    }
  }

  /**
   * Classify entropy into bins
   */
  private classifyEntropy(entropy: number): EntropyBin {
    if (entropy < this.config.entropy_thresholds.low) return EntropyBin.LOW;
    if (entropy < this.config.entropy_thresholds.medium) return EntropyBin.MEDIUM;
    return EntropyBin.HIGH;
  }

  /**
   * Calculate risk score based on query characteristics
   */
  private calculateRiskScore(
    entropy: number,
    context: SearchContext,
    candidatePosition: number
  ): number {
    const span = LensTracer.createChildSpan('calculate_risk', {
      'lens.entropy': entropy,
      'lens.candidate_position': candidatePosition,
    });

    try {
      let riskScore = 0.0;

      // Base risk from entropy
      const entropyBin = this.classifyEntropy(entropy);
      switch (entropyBin) {
        case EntropyBin.LOW: riskScore += 0.1; break;
        case EntropyBin.MEDIUM: riskScore += 0.3; break;
        case EntropyBin.HIGH: riskScore += 0.5; break;
      }

      // Mode complexity risk
      if (context.mode === 'hybrid') riskScore += 0.2;
      if (context.mode === 'struct') riskScore += 0.15;

      // Position in candidates (earlier = riskier)
      const positionRisk = Math.max(0, (50 - candidatePosition) / 50 * 0.3);
      riskScore += positionRisk;

      // K value impact
      if (context.k > 100) riskScore += 0.1;

      // Fuzzy distance impact  
      riskScore += context.fuzzy_distance * 0.05;

      // Clamp to [0, 1]
      riskScore = Math.min(1.0, Math.max(0.0, riskScore));

      span.setAttributes({
        'lens.calculated_risk': riskScore,
        'lens.entropy_contribution': entropyBin,
        'lens.position_contribution': positionRisk,
      });

      return riskScore;
    } finally {
      span.end();
    }
  }

  /**
   * Check if budget allows spending for this query
   */
  canSpendBudget(riskScore: number, headroomMs: number): boolean {
    const span = LensTracer.createChildSpan('budget_check', {
      'lens.risk_score': riskScore,
      'lens.headroom_ms': headroomMs,
      'lens.current_budget_spent': this.currentBudgetSpent,
    });

    try {
      // Refresh budget if needed
      this.refreshBudgetIfNeeded();

      // Check overall budget availability
      if (this.currentBudgetSpent + riskScore > this.config.daily_risk_budget) {
        span.setAttributes({ 'lens.budget_rejection': 'insufficient_budget' });
        return false;
      }

      // Check headroom requirements
      const requiredHeadroom = riskScore * 100; // ms per risk unit
      if (headroomMs < requiredHeadroom) {
        span.setAttributes({ 'lens.budget_rejection': 'insufficient_headroom' });
        return false;
      }

      // Check upshift rate
      const currentUpshiftRate = this.calculateCurrentUpshiftRate();
      const targetRange = [0.03, 0.07]; // 3%-7% as per TODO.md
      if (currentUpshiftRate > targetRange[1]) {
        span.setAttributes({ 'lens.budget_rejection': 'upshift_rate_exceeded' });
        return false;
      }

      span.setAttributes({ 'lens.budget_approved': true });
      return true;
    } finally {
      span.end();
    }
  }

  /**
   * Record risk spend for a query
   */
  recordRiskSpend(
    context: SearchContext,
    spend: Partial<Record<RiskSpendSource, number>>,
    headroomMs: number,
    candidatePosition: number,
    outcome: QueryOutcome,
    latencyBreakdown: { stage_a: number; stage_b: number; stage_c?: number; total: number },
    candidatesProcessed: number,
    finalResults: number
  ): RiskLedgerEntry {
    const span = LensTracer.createChildSpan('record_risk_spend', {
      'lens.trace_id': context.trace_id,
      'lens.candidates_processed': candidatesProcessed,
      'lens.final_results': finalResults,
    });

    try {
      const entropy = this.calculateQueryEntropy(context.query, context);
      const riskScore = this.calculateRiskScore(entropy, context, candidatePosition);
      const entropyBin = this.classifyEntropy(entropy);

      // Calculate total spend
      const totalSpend = Object.values(spend).reduce((sum, val) => sum + (val || 0), 0);
      
      // Update budget
      this.currentBudgetSpent += riskScore;

      const entry: RiskLedgerEntry = {
        trace_id: context.trace_id,
        timestamp: new Date(),
        query: context.query,
        repo_sha: context.repo_sha || 'unknown',
        risk_score: riskScore,
        spend,
        headroom_ms: headroomMs,
        entropy_bin: entropyBin,
        pos_in_candidates: candidatePosition,
        outcome,
        latency_breakdown: {
          stage_a: latencyBreakdown.stage_a,
          stage_b: latencyBreakdown.stage_b,
          stage_c: latencyBreakdown.stage_c,
          total: latencyBreakdown.total,
        },
        candidates_processed: candidatesProcessed,
        final_results: finalResults,
        budget_remaining: Math.max(0, this.config.daily_risk_budget - this.currentBudgetSpent),
      };

      // Store in ledger
      const dateKey = entry.timestamp.toISOString().split('T')[0];
      if (!this.ledger.has(dateKey)) {
        this.ledger.set(dateKey, []);
      }
      this.ledger.get(dateKey)!.push(entry);

      // Add to recent queries for rate calculation
      this.recentQueries.push(entry);
      if (this.recentQueries.length > 1000) {
        this.recentQueries.shift(); // Keep sliding window
      }

      // Record metrics
      riskSpendMetrics.total_spend.add(riskScore, {
        entropy_bin: entropyBin,
        outcome,
      });

      Object.entries(spend).forEach(([source, amount]) => {
        if (amount && amount > 0) {
          riskSpendMetrics.spend_by_source.record(amount, { source });
        }
      });

      riskSpendMetrics.budget_utilization.record(
        this.currentBudgetSpent / this.config.daily_risk_budget,
        { date: dateKey }
      );

      const upshiftRate = this.calculateCurrentUpshiftRate();
      riskSpendMetrics.upshift_rate.record(upshiftRate, { date: dateKey });

      riskSpendMetrics.headroom_tracking.record(headroomMs, {
        entropy_bin: entropyBin,
        risk_level: riskScore > 0.5 ? 'high' : 'low',
      });

      span.setAttributes({
        'lens.entry_recorded': true,
        'lens.risk_score': riskScore,
        'lens.total_spend': totalSpend,
        'lens.budget_remaining': entry.budget_remaining,
      });

      return entry;
    } finally {
      span.end();
    }
  }

  /**
   * Get risk ledger entries for analysis
   */
  getLedgerEntries(date?: string): RiskLedgerEntry[] {
    if (date) {
      return this.ledger.get(date) || [];
    }

    // Return all entries if no date specified
    const allEntries: RiskLedgerEntry[] = [];
    for (const entries of this.ledger.values()) {
      allEntries.push(...entries);
    }
    return allEntries;
  }

  /**
   * Calculate current upshift rate
   */
  private calculateCurrentUpshiftRate(): number {
    const recentWindow = 100; // Last 100 queries
    const recent = this.recentQueries.slice(-recentWindow);
    
    if (recent.length === 0) return 0;

    const upshiftedQueries = recent.filter(entry => 
      entry.risk_score > 0.3 || // Medium/high risk
      Object.values(entry.spend).some(spend => (spend || 0) > 0) // Any spend
    );

    return upshiftedQueries.length / recent.length;
  }

  /**
   * Refresh budget if needed (hourly refresh)
   */
  private refreshBudgetIfNeeded(): void {
    const now = new Date();
    const timeSinceRefresh = now.getTime() - this.lastBudgetRefresh.getTime();

    if (timeSinceRefresh > this.config.knapsack_constraints.budget_refresh_interval_ms) {
      // Partial refresh - don't reset completely to enforce daily limits
      const refreshRatio = timeSinceRefresh / (24 * 60 * 60 * 1000); // Proportion of day
      const budgetRefresh = this.config.daily_risk_budget * refreshRatio * 0.1; // 10% refresh rate
      
      this.currentBudgetSpent = Math.max(0, this.currentBudgetSpent - budgetRefresh);
      this.lastBudgetRefresh = now;
    }
  }

  /**
   * Get current budget status
   */
  getBudgetStatus(): {
    spent: number;
    remaining: number;
    utilization: number;
    upshift_rate: number;
    last_refresh: Date;
  } {
    this.refreshBudgetIfNeeded();

    return {
      spent: this.currentBudgetSpent,
      remaining: Math.max(0, this.config.daily_risk_budget - this.currentBudgetSpent),
      utilization: this.currentBudgetSpent / this.config.daily_risk_budget,
      upshift_rate: this.calculateCurrentUpshiftRate(),
      last_refresh: this.lastBudgetRefresh,
    };
  }

  /**
   * Export ledger data for analysis
   */
  exportLedgerData(startDate?: string, endDate?: string): RiskLedgerEntry[] {
    const entries = this.getLedgerEntries();
    
    if (!startDate && !endDate) return entries;

    return entries.filter(entry => {
      const entryDate = entry.timestamp.toISOString().split('T')[0];
      if (startDate && entryDate < startDate) return false;
      if (endDate && entryDate > endDate) return false;
      return true;
    });
  }
}

// Global risk budget ledger instance
export const globalRiskLedger = new RiskBudgetLedger();