/**
 * Anti-Gaming Contract Validator
 * Enforces replay invariants and prevents gaming as specified in TODO.md
 */

import { createHash } from 'crypto';
import type { ConfigFingerprint, BenchmarkConfig } from '../types/benchmark.js';

export interface ContractViolation {
  rule: string;
  expected: any;
  actual: any;
  severity: 'error' | 'warning';
  message: string;
}

export interface ContractValidationResult {
  passed: boolean;
  violations: ContractViolation[];
  contractHash: string;
  enforcedInvariants: string[];
}

export interface ReplayContext {
  poolId: string;
  layoutConfig: any;
  dedupEnabled: boolean;
  causalMusts: boolean;
  kvBudget: number;
  candidatePool: any[];
  replayTimestamp: string;
}

export class AntiGamingContractValidator {
  private readonly strictMode: boolean;
  
  constructor(strictMode: boolean = true) {
    this.strictMode = strictMode;
  }

  /**
   * Validate contract compliance at replay time
   * Implements TODO.md anti-gaming invariants
   */
  validateContract(
    fingerprint: ConfigFingerprint,
    replayContext: ReplayContext,
    benchmarkConfig: BenchmarkConfig
  ): ContractValidationResult {
    const violations: ContractViolation[] = [];
    const enforcedInvariants: string[] = [];

    // 1. Fixed Layout Invariant
    if (!this.validateFixedLayout(fingerprint, replayContext)) {
      violations.push({
        rule: 'fixed_layout',
        expected: true,
        actual: false,
        severity: 'error',
        message: 'Layout configuration must be identical between runs to prevent result manipulation'
      });
    } else {
      enforcedInvariants.push('fixed_layout=true');
    }

    // 2. Pool ID Consistency
    if (!this.validatePoolConsistency(fingerprint, replayContext)) {
      violations.push({
        rule: 'pool_id_consistency',
        expected: fingerprint.pool_sha,
        actual: replayContext.poolId,
        severity: 'error',
        message: 'Candidate pool must match frozen configuration to ensure reproducibility'
      });
    } else {
      enforcedInvariants.push('pool_id==run.pool_id');
    }

    // 3. Deduplication Enforcement
    if (!this.validateDeduplication(fingerprint, replayContext)) {
      violations.push({
        rule: 'dedup_enabled',
        expected: true,
        actual: replayContext.dedupEnabled,
        severity: 'error',
        message: 'Deduplication must be enabled to prevent artificially inflated recall scores'
      });
    } else {
      enforcedInvariants.push('dedup=true');
    }

    // 4. Causal Must Requirements
    if (!this.validateCausalMusts(fingerprint, replayContext)) {
      violations.push({
        rule: 'causal_musts',
        expected: true,
        actual: replayContext.causalMusts,
        severity: 'error',
        message: 'Causal ordering requirements must be enforced to maintain result validity'
      });
    } else {
      enforcedInvariants.push('causal_musts=true');
    }

    // 5. KV Budget Cap Enforcement
    if (!this.validateKVBudget(fingerprint, replayContext, benchmarkConfig)) {
      violations.push({
        rule: 'kv_budget_cap',
        expected: `≤${fingerprint.kv_budget_cap}`,
        actual: replayContext.kvBudget,
        severity: 'error',
        message: `KV cache budget (${replayContext.kvBudget}) exceeds cap (${fingerprint.kv_budget_cap})`
      });
    } else {
      enforcedInvariants.push('KV_budget≤cap');
    }

    // 6. Candidate Pool Integrity
    const poolViolation = this.validateCandidatePoolIntegrity(fingerprint, replayContext);
    if (poolViolation) {
      violations.push(poolViolation);
    } else {
      enforcedInvariants.push('candidate_pool_integrity');
    }

    // 7. Temporal Ordering (prevent time-based gaming)
    const temporalViolation = this.validateTemporalOrdering(fingerprint, replayContext);
    if (temporalViolation) {
      violations.push(temporalViolation);
    } else {
      enforcedInvariants.push('temporal_ordering');
    }

    // Generate contract hash for this validation
    const contractHash = this.generateContractHash(fingerprint, replayContext, enforcedInvariants);

    const passed = this.strictMode ? violations.length === 0 : violations.filter(v => v.severity === 'error').length === 0;

    return {
      passed,
      violations,
      contractHash,
      enforcedInvariants
    };
  }

  private validateFixedLayout(fingerprint: ConfigFingerprint, context: ReplayContext): boolean {
    if (!fingerprint.fixed_layout) return false;
    
    // Layout configuration must be identical
    const expectedLayoutHash = createHash('sha256')
      .update(JSON.stringify(fingerprint.shard_layout))
      .digest('hex');
    
    const actualLayoutHash = createHash('sha256')
      .update(JSON.stringify(context.layoutConfig))
      .digest('hex');
    
    return expectedLayoutHash === actualLayoutHash;
  }

  private validatePoolConsistency(fingerprint: ConfigFingerprint, context: ReplayContext): boolean {
    return fingerprint.pool_sha === context.poolId;
  }

  private validateDeduplication(fingerprint: ConfigFingerprint, context: ReplayContext): boolean {
    return fingerprint.dedup_enabled && context.dedupEnabled;
  }

  private validateCausalMusts(fingerprint: ConfigFingerprint, context: ReplayContext): boolean {
    return fingerprint.causal_musts && context.causalMusts;
  }

  private validateKVBudget(
    fingerprint: ConfigFingerprint, 
    context: ReplayContext,
    config: BenchmarkConfig
  ): boolean {
    return context.kvBudget <= fingerprint.kv_budget_cap;
  }

  private validateCandidatePoolIntegrity(
    fingerprint: ConfigFingerprint, 
    context: ReplayContext
  ): ContractViolation | null {
    // Validate that candidate pool hasn't been tampered with
    const poolHash = createHash('sha256')
      .update(JSON.stringify(context.candidatePool.map(c => c.id || c.file).sort()))
      .digest('hex');
    
    if (poolHash !== fingerprint.pool_sha) {
      return {
        rule: 'candidate_pool_integrity',
        expected: fingerprint.pool_sha,
        actual: poolHash,
        severity: 'error',
        message: 'Candidate pool has been modified since freezing, invalidating reproducibility'
      };
    }
    
    return null;
  }

  private validateTemporalOrdering(
    fingerprint: ConfigFingerprint, 
    context: ReplayContext
  ): ContractViolation | null {
    const frozenTime = new Date(fingerprint.timestamp);
    const replayTime = new Date(context.replayTimestamp);
    
    // Replay must happen after freezing
    if (replayTime <= frozenTime) {
      return {
        rule: 'temporal_ordering',
        expected: `>${fingerprint.timestamp}`,
        actual: context.replayTimestamp,
        severity: 'warning',
        message: 'Replay timestamp suggests potential temporal gaming'
      };
    }
    
    return null;
  }

  private generateContractHash(
    fingerprint: ConfigFingerprint,
    context: ReplayContext,
    invariants: string[]
  ): string {
    return createHash('sha256')
      .update(JSON.stringify({
        config_hash: fingerprint.config_hash,
        pool_sha: fingerprint.pool_sha,
        invariants: invariants.sort(),
        validation_timestamp: new Date().toISOString()
      }))
      .digest('hex');
  }

  /**
   * Fast fail validation - immediately abort on critical violations
   */
  validateOrThrow(
    fingerprint: ConfigFingerprint,
    replayContext: ReplayContext,
    benchmarkConfig: BenchmarkConfig
  ): void {
    const result = this.validateContract(fingerprint, replayContext, benchmarkConfig);
    
    if (!result.passed) {
      const errorViolations = result.violations.filter(v => v.severity === 'error');
      if (errorViolations.length > 0) {
        const violationMessages = errorViolations.map(v => `${v.rule}: ${v.message}`).join('; ');
        throw new Error(`Contract validation failed: ${violationMessages}`);
      }
    }
  }

  /**
   * Generate report for audit trail
   */
  generateAuditReport(result: ContractValidationResult): string {
    const report = [
      '# Anti-Gaming Contract Validation Report',
      `**Timestamp**: ${new Date().toISOString()}`,
      `**Status**: ${result.passed ? '✅ PASSED' : '❌ FAILED'}`,
      `**Contract Hash**: ${result.contractHash}`,
      '',
      '## Enforced Invariants',
      ...result.enforcedInvariants.map(inv => `- ✅ ${inv}`)
    ];

    if (result.violations.length > 0) {
      report.push(
        '',
        '## Violations',
        ...result.violations.map(v => 
          `- ${v.severity === 'error' ? '❌' : '⚠️'} **${v.rule}**: ${v.message} (expected: ${v.expected}, actual: ${v.actual})`
        )
      );
    }

    return report.join('\n');
  }
}

/**
 * Factory function for creating validator with TODO.md compliance
 */
export function createContractValidator(strictMode: boolean = true): AntiGamingContractValidator {
  return new AntiGamingContractValidator(strictMode);
}