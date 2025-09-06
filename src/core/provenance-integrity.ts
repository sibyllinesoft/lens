/**
 * Provenance & Integrity Hardening System
 * 
 * 1. Segment Merkle trees over postings + SymbolGraph with config_fingerprint in root
 * 2. Span Normal Form (SNF) persisted with patience-diff line map for reproducible spans  
 * 3. Churn-indexed TTLs extended to RAPTOR/centrality priors and symbol sketches
 * 4. Integrity verification in /bench/health with zero span drift under HEAD↔SHA↔HEAD
 * 
 * Gates: Merkle verification 100% success, zero span drift, integrity check performance impact
 */

import { createHash } from 'crypto';
import type {
  SegmentMerkleTree,
  SpanNormalForm,
  PatienceDiffMap,
  DiffHunk,
  NormalizationRule,
  ChurnIndexedTTL,
  ChurnMetric,
  IntegrityVerification,
  AdvancedLeverMetrics
} from '../types/embedder-proof-levers.js';

export class ProvenanceIntegritySystem {
  private merkleTree: SegmentMerkleTree | null = null;
  private spanNormalForms: Map<string, SpanNormalForm> = new Map();
  private churnIndexedTTLs: Map<string, ChurnIndexedTTL> = new Map();
  private verificationResults: IntegrityVerification[] = [];
  private metrics: AdvancedLeverMetrics['provenance_integrity'];
  private configFingerprint: string = '';

  constructor() {
    this.initializeMetrics();
    this.setupPeriodicVerification();
  }

  /**
   * Build Merkle tree over segments with config fingerprint
   */
  public buildSegmentMerkleTree(
    segments: Array<{ id: string; data: Buffer }>,
    postingLists: Array<{ id: string; data: Buffer }>,
    symbolGraphData: Buffer,
    configFingerprint: string
  ): SegmentMerkleTree {
    const segmentHashes = new Map<string, string>();
    const postingListHashes = new Map<string, string>();

    // Hash individual segments
    for (const segment of segments) {
      const hash = this.computeHash(segment.data);
      segmentHashes.set(segment.id, hash);
    }

    // Hash posting lists
    for (const postingList of postingLists) {
      const hash = this.computeHash(postingList.data);
      postingListHashes.set(postingList.id, hash);
    }

    // Hash symbol graph
    const symbolGraphHash = this.computeHash(symbolGraphData);

    // Build Merkle root
    const allHashes = [
      ...Array.from(segmentHashes.values()),
      ...Array.from(postingListHashes.values()),
      symbolGraphHash,
      configFingerprint
    ].sort(); // Deterministic order

    const rootHash = this.computeMerkleRoot(allHashes);

    this.merkleTree = {
      root_hash: rootHash,
      config_fingerprint: configFingerprint,
      segment_hashes: segmentHashes,
      posting_list_hashes: postingListHashes,
      symbol_graph_hash: symbolGraphHash,
      created_at: new Date(),
      verification_depth: this.calculateVerificationDepth(segments.length + postingLists.length)
    };

    this.configFingerprint = configFingerprint;
    return this.merkleTree;
  }

  /**
   * Verify Merkle tree integrity - refuse to serve mixed trees
   */
  public verifyMerkleIntegrity(
    segments: Array<{ id: string; data: Buffer }>,
    postingLists: Array<{ id: string; data: Buffer }>,
    symbolGraphData: Buffer
  ): IntegrityVerification {
    const startTime = Date.now();
    
    if (!this.merkleTree) {
      return {
        verification_type: 'merkle',
        status: 'fail',
        details: 'No Merkle tree available for verification',
        checked_at: new Date(),
        performance_impact_ms: Date.now() - startTime,
        error_details: 'Missing Merkle tree'
      };
    }

    try {
      // Recompute hashes and compare
      const segmentHashes = new Map<string, string>();
      const postingListHashes = new Map<string, string>();

      for (const segment of segments) {
        const hash = this.computeHash(segment.data);
        const expectedHash = this.merkleTree.segment_hashes.get(segment.id);
        
        if (hash !== expectedHash) {
          return {
            verification_type: 'merkle',
            status: 'fail',
            details: `Segment ${segment.id} hash mismatch`,
            checked_at: new Date(),
            performance_impact_ms: Date.now() - startTime,
            error_details: `Expected: ${expectedHash}, Got: ${hash}`
          };
        }
        segmentHashes.set(segment.id, hash);
      }

      for (const postingList of postingLists) {
        const hash = this.computeHash(postingList.data);
        const expectedHash = this.merkleTree.posting_list_hashes.get(postingList.id);
        
        if (hash !== expectedHash) {
          return {
            verification_type: 'merkle',
            status: 'fail',
            details: `Posting list ${postingList.id} hash mismatch`,
            checked_at: new Date(),
            performance_impact_ms: Date.now() - startTime,
            error_details: `Expected: ${expectedHash}, Got: ${hash}`
          };
        }
        postingListHashes.set(postingList.id, hash);
      }

      // Verify symbol graph
      const symbolGraphHash = this.computeHash(symbolGraphData);
      if (symbolGraphHash !== this.merkleTree.symbol_graph_hash) {
        return {
          verification_type: 'merkle',
          status: 'fail',
          details: 'Symbol graph hash mismatch',
          checked_at: new Date(),
          performance_impact_ms: Date.now() - startTime,
          error_details: `Expected: ${this.merkleTree.symbol_graph_hash}, Got: ${symbolGraphHash}`
        };
      }

      // Verify root hash
      const allHashes = [
        ...Array.from(segmentHashes.values()),
        ...Array.from(postingListHashes.values()),
        symbolGraphHash,
        this.configFingerprint
      ].sort();

      const computedRoot = this.computeMerkleRoot(allHashes);
      if (computedRoot !== this.merkleTree.root_hash) {
        return {
          verification_type: 'merkle',
          status: 'fail',
          details: 'Root hash mismatch',
          checked_at: new Date(),
          performance_impact_ms: Date.now() - startTime,
          error_details: `Expected: ${this.merkleTree.root_hash}, Got: ${computedRoot}`
        };
      }

      const result: IntegrityVerification = {
        verification_type: 'merkle',
        status: 'pass',
        details: 'All hashes verified successfully',
        checked_at: new Date(),
        performance_impact_ms: Date.now() - startTime
      };

      this.verificationResults.push(result);
      this.updateMetrics();
      
      return result;

    } catch (error) {
      return {
        verification_type: 'merkle',
        status: 'fail',
        details: 'Verification failed with error',
        checked_at: new Date(),
        performance_impact_ms: Date.now() - startTime,
        error_details: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Create Span Normal Form with patience diff for time-travel
   */
  public createSpanNormalForm(
    filePath: string,
    lineStart: number,
    lineEnd: number,
    colStart: number,
    colEnd: number,
    content: string,
    gitSha: string,
    originalSha?: string
  ): SpanNormalForm {
    // Normalize content using standardization rules
    const normalizedContent = this.normalizeContent(content, filePath);
    const contentHash = this.computeHash(Buffer.from(normalizedContent, 'utf8'));

    // Create patience diff map if original SHA provided
    let patienceDiffMap: PatienceDiffMap | undefined;
    if (originalSha && originalSha !== gitSha) {
      patienceDiffMap = this.createPatienceDiffMap(originalSha, gitSha, filePath);
    } else {
      patienceDiffMap = {
        original_sha: gitSha,
        target_sha: gitSha,
        line_mappings: new Map([[lineStart, lineStart]]),
        diff_hunks: [],
        patience_algorithm_version: '1.0'
      };
    }

    const snf: SpanNormalForm = {
      file_path: filePath,
      line_start: lineStart,
      line_end: lineEnd,
      col_start: colStart,
      col_end: colEnd,
      content_hash: contentHash,
      patience_diff_map: patienceDiffMap,
      git_sha: gitSha,
      normalization_rules: this.getNormalizationRules(filePath)
    };

    const snfKey = `${filePath}:${gitSha}:${lineStart}:${lineEnd}`;
    this.spanNormalForms.set(snfKey, snf);

    return snf;
  }

  /**
   * Verify span drift under HEAD↔SHA↔HEAD round-trips
   */
  public verifySpanRoundTripFidelity(
    filePath: string,
    originalSpan: { lineStart: number; lineEnd: number; colStart: number; colEnd: number },
    headSha: string,
    targetSha: string
  ): IntegrityVerification {
    const startTime = Date.now();

    try {
      // Step 1: HEAD -> SHA
      const forwardSpan = this.mapSpanBetweenShas(
        filePath, originalSpan, headSha, targetSha
      );

      if (!forwardSpan) {
        return {
          verification_type: 'round_trip',
          status: 'fail',
          details: 'Forward span mapping failed',
          checked_at: new Date(),
          performance_impact_ms: Date.now() - startTime,
          error_details: `Could not map span from ${headSha} to ${targetSha}`
        };
      }

      // Step 2: SHA -> HEAD
      const roundTripSpan = this.mapSpanBetweenShas(
        filePath, forwardSpan, targetSha, headSha
      );

      if (!roundTripSpan) {
        return {
          verification_type: 'round_trip',
          status: 'fail',
          details: 'Backward span mapping failed',
          checked_at: new Date(),
          performance_impact_ms: Date.now() - startTime,
          error_details: `Could not map span from ${targetSha} to ${headSha}`
        };
      }

      // Step 3: Compare original vs round-trip
      const spanDrift = this.calculateSpanDrift(originalSpan, roundTripSpan);
      
      if (spanDrift > 0) {
        return {
          verification_type: 'round_trip',
          status: 'fail',
          details: `Span drift detected: ${spanDrift} lines`,
          checked_at: new Date(),
          performance_impact_ms: Date.now() - startTime,
          error_details: `Original: ${JSON.stringify(originalSpan)}, Round-trip: ${JSON.stringify(roundTripSpan)}`
        };
      }

      const result: IntegrityVerification = {
        verification_type: 'round_trip',
        status: 'pass',
        details: 'Zero span drift verified',
        checked_at: new Date(),
        performance_impact_ms: Date.now() - startTime
      };

      this.verificationResults.push(result);
      this.updateMetrics();
      
      return result;

    } catch (error) {
      return {
        verification_type: 'round_trip',
        status: 'fail',
        details: 'Round-trip verification failed',
        checked_at: new Date(),
        performance_impact_ms: Date.now() - startTime,
        error_details: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Setup churn-indexed TTLs for RAPTOR/centrality/symbol sketches
   */
  public setupChurnIndexedTTL(
    resourceType: 'raptor' | 'centrality' | 'symbol_sketch',
    baseTtlHours: number = 24,
    tauMin: number = 1,
    tauMax: number = 168 // 1 week
  ): ChurnIndexedTTL {
    const ttl: ChurnIndexedTTL = {
      resource_type: resourceType,
      base_ttl_hours: baseTtlHours,
      churn_lambda: 0.1, // Initial churn rate
      ttl_min_hours: tauMin,
      ttl_max_hours: tauMax,
      current_ttl: baseTtlHours,
      last_updated: new Date(),
      churn_history: []
    };

    this.churnIndexedTTLs.set(resourceType, ttl);
    return ttl;
  }

  /**
   * Update churn metrics and recompute TTL
   */
  public updateChurnMetrics(
    resourceType: 'raptor' | 'centrality' | 'symbol_sketch',
    filesChanged: number,
    linesAdded: number,
    linesDeleted: number,
    symbolsAffected: number
  ): void {
    const ttlConfig = this.churnIndexedTTLs.get(resourceType);
    if (!ttlConfig) return;

    // Calculate normalized churn rate
    const totalChanges = filesChanged + linesAdded + linesDeleted + symbolsAffected;
    const churnRate = totalChanges / Math.max(1, filesChanged); // Per-file intensity

    const churnMetric: ChurnMetric = {
      timestamp: new Date(),
      files_changed: filesChanged,
      lines_added: linesAdded,
      lines_deleted: linesDeleted,
      symbols_affected: symbolsAffected,
      churn_rate: churnRate
    };

    ttlConfig.churn_history.push(churnMetric);

    // Keep only recent history (last 30 days)
    const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    ttlConfig.churn_history = ttlConfig.churn_history.filter(m => m.timestamp >= cutoff);

    // Update lambda (exponential moving average)
    const alpha = 0.1;
    ttlConfig.churn_lambda = alpha * churnRate + (1 - alpha) * ttlConfig.churn_lambda;

    // Recompute TTL: clamp(τ_min, τ_max, c/λ_churn_slice)
    const c = ttlConfig.base_ttl_hours * ttlConfig.churn_lambda;
    const newTtl = Math.max(
      ttlConfig.ttl_min_hours,
      Math.min(ttlConfig.ttl_max_hours, c / Math.max(0.01, ttlConfig.churn_lambda))
    );

    ttlConfig.current_ttl = newTtl;
    ttlConfig.last_updated = new Date();
  }

  /**
   * Get current TTL for a resource type
   */
  public getCurrentTTL(resourceType: 'raptor' | 'centrality' | 'symbol_sketch'): number {
    const ttlConfig = this.churnIndexedTTLs.get(resourceType);
    return ttlConfig?.current_ttl || 24; // Default 24 hours
  }

  /**
   * Comprehensive health check for /bench/health endpoint
   */
  public async performHealthCheck(): Promise<{
    overall_status: 'healthy' | 'degraded' | 'unhealthy';
    checks: IntegrityVerification[];
    performance_summary: {
      avg_verification_time_ms: number;
      success_rate: number;
      last_24h_checks: number;
    };
  }> {
    const checks: IntegrityVerification[] = [];
    
    // Mock health checks - in practice would verify actual data
    const mockSegments = [{ id: 'segment1', data: Buffer.from('test') }];
    const mockPostings = [{ id: 'posting1', data: Buffer.from('test') }];
    const mockSymbolGraph = Buffer.from('symbol_graph');

    // Verify Merkle tree
    if (this.merkleTree) {
      const merkleCheck = this.verifyMerkleIntegrity(mockSegments, mockPostings, mockSymbolGraph);
      checks.push(merkleCheck);
    }

    // Verify span fidelity for sample spans
    const spanCheck = this.verifySpanRoundTripFidelity(
      'test/file.ts',
      { lineStart: 10, lineEnd: 20, colStart: 0, colEnd: 50 },
      'head_sha',
      'target_sha'
    );
    checks.push(spanCheck);

    // Calculate performance summary
    const recent24h = this.verificationResults.filter(
      r => r.checked_at.getTime() > Date.now() - 24 * 60 * 60 * 1000
    );

    const avgTime = recent24h.length > 0 
      ? recent24h.reduce((sum, r) => sum + r.performance_impact_ms, 0) / recent24h.length
      : 0;

    const successRate = recent24h.length > 0
      ? recent24h.filter(r => r.status === 'pass').length / recent24h.length
      : 1;

    const allPassed = checks.every(c => c.status === 'pass');
    const anyFailed = checks.some(c => c.status === 'fail');

    return {
      overall_status: allPassed ? 'healthy' : anyFailed ? 'unhealthy' : 'degraded',
      checks,
      performance_summary: {
        avg_verification_time_ms: avgTime,
        success_rate: successRate,
        last_24h_checks: recent24h.length
      }
    };
  }

  public getMetrics(): AdvancedLeverMetrics['provenance_integrity'] {
    return { ...this.metrics };
  }

  // Private helper methods

  private computeHash(data: Buffer): string {
    return createHash('sha256').update(data).digest('hex');
  }

  private computeMerkleRoot(hashes: string[]): string {
    if (hashes.length === 0) return '';
    if (hashes.length === 1) return hashes[0];

    const nextLevel: string[] = [];
    for (let i = 0; i < hashes.length; i += 2) {
      const left = hashes[i];
      const right = i + 1 < hashes.length ? hashes[i + 1] : left;
      const combined = this.computeHash(Buffer.from(left + right, 'hex'));
      nextLevel.push(combined);
    }

    return this.computeMerkleRoot(nextLevel);
  }

  private calculateVerificationDepth(nodeCount: number): number {
    return Math.ceil(Math.log2(nodeCount));
  }

  private normalizeContent(content: string, filePath: string): string {
    const rules = this.getNormalizationRules(filePath);
    let normalized = content;

    for (const rule of rules) {
      const regex = new RegExp(rule.pattern, 'g');
      normalized = normalized.replace(regex, rule.replacement);
    }

    return normalized;
  }

  private getNormalizationRules(filePath: string): NormalizationRule[] {
    const extension = filePath.split('.').pop()?.toLowerCase();
    const baseRules: NormalizationRule[] = [
      {
        rule_type: 'whitespace',
        pattern: '\\s+',
        replacement: ' ',
        preserve_semantics: true,
        language_specific: false
      }
    ];

    if (extension === 'ts' || extension === 'js') {
      baseRules.push({
        rule_type: 'comments',
        pattern: '//.*$',
        replacement: '',
        preserve_semantics: false,
        language_specific: true
      });
    }

    return baseRules;
  }

  private createPatienceDiffMap(originalSha: string, targetSha: string, filePath: string): PatienceDiffMap {
    // Simplified patience diff - in practice would use git diff with patience algorithm
    return {
      original_sha: originalSha,
      target_sha: targetSha,
      line_mappings: new Map([[1, 1], [10, 12], [20, 22]]), // Example mappings
      diff_hunks: [
        {
          original_start: 10,
          original_count: 5,
          target_start: 12,
          target_count: 5,
          operation: 'modify',
          confidence: 0.95
        }
      ],
      patience_algorithm_version: '1.0'
    };
  }

  private mapSpanBetweenShas(
    filePath: string,
    span: { lineStart: number; lineEnd: number; colStart: number; colEnd: number },
    fromSha: string,
    toSha: string
  ): typeof span | null {
    // Simplified span mapping - in practice would use actual diff analysis
    const diffMap = this.createPatienceDiffMap(fromSha, toSha, filePath);
    
    const mappedLineStart = diffMap.line_mappings.get(span.lineStart) || span.lineStart;
    const mappedLineEnd = diffMap.line_mappings.get(span.lineEnd) || span.lineEnd;

    return {
      lineStart: mappedLineStart,
      lineEnd: mappedLineEnd,
      colStart: span.colStart,
      colEnd: span.colEnd
    };
  }

  private calculateSpanDrift(
    original: { lineStart: number; lineEnd: number; colStart: number; colEnd: number },
    roundTrip: { lineStart: number; lineEnd: number; colStart: number; colEnd: number }
  ): number {
    return Math.abs(original.lineStart - roundTrip.lineStart) +
           Math.abs(original.lineEnd - roundTrip.lineEnd) +
           Math.abs(original.colStart - roundTrip.colStart) +
           Math.abs(original.colEnd - roundTrip.colEnd);
  }

  private initializeMetrics(): void {
    this.metrics = {
      merkle_verification_success_rate: 1.0,
      span_drift_incidents: 0,
      round_trip_fidelity: 1.0,
      integrity_check_latency_ms: 0,
      ttl_optimization_savings_pct: 0
    };
  }

  private setupPeriodicVerification(): void {
    // Run integrity checks every hour
    setInterval(() => {
      this.performHealthCheck().then(result => {
        console.log('Periodic integrity check:', {
          status: result.overall_status,
          checks: result.checks.length,
          avg_time: result.performance_summary.avg_verification_time_ms
        });
      });
    }, 60 * 60 * 1000); // 1 hour
  }

  private updateMetrics(): void {
    const recent = this.verificationResults.slice(-100); // Last 100 checks
    
    if (recent.length > 0) {
      const successCount = recent.filter(r => r.status === 'pass').length;
      this.metrics.merkle_verification_success_rate = successCount / recent.length;
      
      const avgLatency = recent.reduce((sum, r) => sum + r.performance_impact_ms, 0) / recent.length;
      this.metrics.integrity_check_latency_ms = avgLatency;

      this.metrics.span_drift_incidents = recent.filter(
        r => r.verification_type === 'round_trip' && r.status === 'fail'
      ).length;

      const roundTripSuccesses = recent.filter(
        r => r.verification_type === 'round_trip' && r.status === 'pass'
      ).length;
      const totalRoundTrips = recent.filter(r => r.verification_type === 'round_trip').length;
      
      this.metrics.round_trip_fidelity = totalRoundTrips > 0 ? roundTripSuccesses / totalRoundTrips : 1.0;
    }
  }
}

/**
 * Factory function to create provenance integrity system
 */
export function createProvenanceIntegrity(): ProvenanceIntegritySystem {
  return new ProvenanceIntegritySystem();
}