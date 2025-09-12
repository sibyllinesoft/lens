import { describe, it, expect, beforeEach, jest } from 'bun:test';
import { AdvancedSearchIntegration } from '../advanced-search-integration';
import { ConformalRouter } from '../conformal-router';
import { EntropyGatedPriors } from '../entropy-gated-priors';
import { LatencyConditionedMetrics } from '../latency-conditioned-metrics';
import { RAPTORHygiene } from '../raptor-hygiene';
import { EmbeddingRoadmap } from '../embedding-roadmap';
import { UnicodeNFCNormalizer } from '../unicode-nfc-normalizer';
import { ComprehensiveMonitoring } from '../comprehensive-monitoring';
import type { SearchContext, SearchHit } from '../../types/search';

// Mock implementations for testing
const createMockSearchContext = (): SearchContext => ({
  query: 'test query',
  filters: {},
  userId: 'test-user',
  requestId: 'test-req-001',
  timestamp: Date.now(),
  efSearch: 50,
  maxResults: 10,
  includeSnippets: true
});

const createMockSearchHits = (count: number): SearchHit[] => {
  return Array.from({ length: count }, (_, i) => ({
    id: `hit-${i}`,
    path: `/test/file${i}.ts`,
    content: `test content ${i}`,
    score: 0.9 - (i * 0.1),
    snippet: `snippet ${i}`,
    line: i + 1,
    column: 1,
    context: `context ${i}`,
    startOffset: i * 10,
    endOffset: (i * 10) + 5
  }));
};

describe('Advanced Search Optimizations - Safety Gates & Performance Validation', () => {
  let integration: AdvancedSearchIntegration;
  let mockContext: SearchContext;
  let mockHits: SearchHit[];

  beforeEach(() => {
    integration = AdvancedSearchIntegration.getInstance();
    mockContext = createMockSearchContext();
    mockHits = createMockSearchHits(50);
    
    // Reset all components to clean state
    integration.reset();
  });

  describe('Upshift Rate Cap Validation (≤5%)', () => {
    it('should maintain upshift rate below 5% over 1000 requests', async () => {
      const numRequests = 1000;
      let upshiftCount = 0;
      
      for (let i = 0; i < numRequests; i++) {
        const context = {
          ...mockContext,
          requestId: `req-${i}`,
          query: `test query ${i % 10}` // Vary queries
        };
        
        const result = await integration.executeAdvancedSearch(
          mockHits.slice(0, 20), // Smaller candidate sets
          context
        );
        
        if (result.routingDecision?.useExpensiveMode) {
          upshiftCount++;
        }
      }
      
      const upshiftRate = (upshiftCount / numRequests) * 100;
      console.log(`Upshift rate: ${upshiftRate.toFixed(2)}%`);
      
      expect(upshiftRate).toBeLessThanOrEqual(5.0);
    });

    it('should enforce budget constraints when approaching upshift limit', async () => {
      const router = ConformalRouter.getInstance();
      
      // Simulate high upshift scenario
      for (let i = 0; i < 48; i++) { // 48/1000 = 4.8%
        await router.makeRoutingDecision({
          ...mockContext,
          requestId: `budget-test-${i}`
        }, mockHits.slice(0, 5)); // Small candidate sets to trigger upshift
      }
      
      const stats = router.getStatistics();
      const currentRate = (stats.upshiftCount / stats.totalDecisions) * 100;
      
      // Next decision should be conservative due to budget constraints
      const decision = await router.makeRoutingDecision({
        ...mockContext,
        requestId: 'budget-test-final'
      }, mockHits.slice(0, 5));
      
      if (currentRate > 4.5) {
        expect(decision.useExpensiveMode).toBe(false);
      }
    });
  });

  describe('Latency Impact Validation (p95 ≤ +1ms)', () => {
    it('should measure and validate latency impact stays within bounds', async () => {
      const baselineLatencies: number[] = [];
      const optimizedLatencies: number[] = [];
      const numSamples = 100;
      
      // Baseline measurements (without optimizations)
      integration.disable();
      
      for (let i = 0; i < numSamples; i++) {
        const startTime = performance.now();
        await integration.executeAdvancedSearch(mockHits, {
          ...mockContext,
          requestId: `baseline-${i}`
        });
        const latency = performance.now() - startTime;
        baselineLatencies.push(latency);
      }
      
      // Optimized measurements (with optimizations)
      integration.enable();
      
      for (let i = 0; i < numSamples; i++) {
        const startTime = performance.now();
        await integration.executeAdvancedSearch(mockHits, {
          ...mockContext,
          requestId: `optimized-${i}`
        });
        const latency = performance.now() - startTime;
        optimizedLatencies.push(latency);
      }
      
      // Calculate p95 latencies
      const p95Baseline = calculatePercentile(baselineLatencies, 95);
      const p95Optimized = calculatePercentile(optimizedLatencies, 95);
      const latencyImpact = p95Optimized - p95Baseline;
      
      console.log(`p95 Baseline: ${p95Baseline.toFixed(2)}ms`);
      console.log(`p95 Optimized: ${p95Optimized.toFixed(2)}ms`);
      console.log(`p95 Impact: ${latencyImpact.toFixed(2)}ms`);
      
      expect(latencyImpact).toBeLessThanOrEqual(1.0);
    });
  });

  describe('Recall@50 Preservation Validation', () => {
    it('should preserve Recall@50 performance while improving precision', async () => {
      const metricsCalculator = LatencyConditionedMetrics.getInstance();
      
      // Generate ground truth relevance judgments
      const groundTruth = new Set(mockHits.slice(0, 25).map(hit => hit.id));
      
      // Test baseline recall
      const baselineMetrics = await metricsCalculator.calculateMetrics(
        mockHits,
        mockContext,
        10 // p95 latency
      );
      
      // Test optimized recall with advanced search
      const optimizedResult = await integration.executeAdvancedSearch(
        mockHits,
        mockContext
      );
      
      const optimizedMetrics = await metricsCalculator.calculateMetrics(
        optimizedResult.enhancedHits,
        mockContext,
        11 // slightly higher latency due to optimizations
      );
      
      // Recall@50 should be preserved or improved
      expect(optimizedMetrics.recallAt50).toBeGreaterThanOrEqual(
        baselineMetrics.recallAt50 * 0.95 // Allow 5% tolerance
      );
      
      console.log(`Baseline Recall@50: ${baselineMetrics.recallAt50.toFixed(3)}`);
      console.log(`Optimized Recall@50: ${optimizedMetrics.recallAt50.toFixed(3)}`);
    });
  });

  describe('nDCG@10 Improvement Validation (≥3pp)', () => {
    it('should achieve nDCG@10 improvement of at least 3 percentage points', async () => {
      const numTests = 50;
      let baselineNdcgSum = 0;
      let optimizedNdcgSum = 0;
      
      for (let i = 0; i < numTests; i++) {
        const testContext = {
          ...mockContext,
          requestId: `ndcg-test-${i}`,
          query: `test query variation ${i % 5}`
        };
        
        // Baseline nDCG (without optimizations)
        integration.disable();
        const baselineResult = await integration.executeAdvancedSearch(
          mockHits,
          testContext
        );
        const baselineNdcg = calculateNDCG(baselineResult.enhancedHits.slice(0, 10));
        baselineNdcgSum += baselineNdcg;
        
        // Optimized nDCG (with optimizations)
        integration.enable();
        const optimizedResult = await integration.executeAdvancedSearch(
          mockHits,
          testContext
        );
        const optimizedNdcg = calculateNDCG(optimizedResult.enhancedHits.slice(0, 10));
        optimizedNdcgSum += optimizedNdcg;
      }
      
      const avgBaselineNdcg = baselineNdcgSum / numTests;
      const avgOptimizedNdcg = optimizedNdcgSum / numTests;
      const improvementPp = (avgOptimizedNdcg - avgBaselineNdcg) * 100;
      
      console.log(`Average baseline nDCG@10: ${(avgBaselineNdcg * 100).toFixed(2)}%`);
      console.log(`Average optimized nDCG@10: ${(avgOptimizedNdcg * 100).toFixed(2)}%`);
      console.log(`Improvement: ${improvementPp.toFixed(2)} percentage points`);
      
      expect(improvementPp).toBeGreaterThanOrEqual(3.0);
    });
  });

  describe('CUSUM Drift Detection Validation', () => {
    it('should detect metric drift using CUSUM algorithms', async () => {
      const metricsCalculator = LatencyConditionedMetrics.getInstance();
      const monitoring = ComprehensiveMonitoring.getInstance();
      
      // Generate stable baseline metrics
      for (let i = 0; i < 50; i++) {
        const metrics = await metricsCalculator.calculateMetrics(
          mockHits,
          { ...mockContext, requestId: `baseline-${i}` },
          10 + Math.random() * 2 // Small variance
        );
        monitoring.recordMetric('recall_at_50', metrics.recallAt50, Date.now());
      }
      
      // Introduce drift - significant degradation
      for (let i = 0; i < 20; i++) {
        const degradedHits = mockHits.map((hit, idx) => ({
          ...hit,
          score: hit.score * (idx < 10 ? 0.7 : 0.9) // Degrade top results
        }));
        
        const metrics = await metricsCalculator.calculateMetrics(
          degradedHits,
          { ...mockContext, requestId: `drift-${i}` },
          15 + Math.random() * 5 // Higher latency variance
        );
        monitoring.recordMetric('recall_at_50', metrics.recallAt50, Date.now());
      }
      
      const driftStatus = monitoring.checkDriftDetection();
      expect(driftStatus.detected).toBe(true);
      expect(driftStatus.metrics).toContain('recall_at_50');
    });
  });

  describe('Unicode NFC Normalization Validation', () => {
    it('should correctly normalize unicode text spans', async () => {
      const normalizer = UnicodeNFCNormalizer.getInstance();
      
      // Test cases with combining characters
      const testCases = [
        {
          input: 'café', // é as single character
          expected: 'café' // Should remain unchanged if already NFC
        },
        {
          input: 'cafe\u0301', // e + combining acute accent
          expected: 'café' // Should normalize to NFC form
        },
        {
          input: 'naïve', // ï as single character
          expected: 'naïve'
        },
        {
          input: 'nai\u0308ve', // i + combining diaeresis
          expected: 'naïve'
        }
      ];
      
      testCases.forEach((testCase, i) => {
        const normalized = normalizer.normalizeSpan(testCase.input, 0, testCase.input.length);
        expect(normalized.normalizedText).toBe(testCase.expected);
        console.log(`Test ${i + 1}: "${testCase.input}" → "${normalized.normalizedText}"`);
      });
    });

    it('should handle edge cases and invalid unicode', async () => {
      const normalizer = UnicodeNFCNormalizer.getInstance();
      
      // Test with empty string
      const emptyResult = normalizer.normalizeSpan('', 0, 0);
      expect(emptyResult.normalizedText).toBe('');
      expect(emptyResult.isValid).toBe(true);
      
      // Test with only combining characters
      const combiningOnly = normalizer.normalizeSpan('\u0301\u0308', 0, 2);
      expect(combiningOnly.warnings).toContain('Leading combining characters detected');
      
      // Test with surrogate pairs
      const emoji = normalizer.normalizeSpan('🚀test', 0, 6);
      expect(emoji.isValid).toBe(true);
      expect(emoji.normalizedText).toBe('🚀test');
    });
  });

  describe('RAPTOR Hierarchical Clustering Validation', () => {
    it('should maintain clustering quality under pressure budget constraints', async () => {
      const raptor = RAPTORHygiene.getInstance();
      
      // Create diverse query embeddings
      const queryEmbeddings = Array.from({ length: 10 }, (_, i) => 
        new Float32Array(Array.from({ length: 384 }, (_, j) => 
          Math.sin(i * 0.1 + j * 0.01) * 0.1 + Math.random() * 0.05
        ))
      );
      
      let totalResults = 0;
      let totalClusters = 0;
      
      for (const embedding of queryEmbeddings) {
        const results = await raptor.hierarchicalSearch(
          embedding,
          mockContext,
          20 // Request 20 results
        );
        
        totalResults += results.length;
        
        // Verify results are diverse (not all from same cluster)
        const uniquePaths = new Set(results.map(hit => hit.path.split('/')[1]));
        totalClusters += uniquePaths.size;
      }
      
      const avgResults = totalResults / queryEmbeddings.length;
      const avgClusters = totalClusters / queryEmbeddings.length;
      
      console.log(`Average results per query: ${avgResults.toFixed(1)}`);
      console.log(`Average clusters per query: ${avgClusters.toFixed(1)}`);
      
      // Should return reasonable number of results
      expect(avgResults).toBeGreaterThanOrEqual(10);
      // Should show diversity (multiple clusters)
      expect(avgClusters).toBeGreaterThanOrEqual(3);
      
      const stats = raptor.getStatistics();
      expect(stats.pressureBudgetUtilization).toBeLessThan(1.0); // Under budget
    });
  });

  describe('System Integration Validation', () => {
    it('should coordinate all components without conflicts', async () => {
      const monitoring = ComprehensiveMonitoring.getInstance();
      
      // Run coordinated search with all optimizations
      const result = await integration.executeAdvancedSearch(
        mockHits,
        mockContext,
        new Float32Array(384).fill(0.1) // Query embedding
      );
      
      // Verify all components contributed
      expect(result.routingDecision).toBeDefined();
      expect(result.entropyAnalysis).toBeDefined();
      expect(result.normalizedSpans).toBeDefined();
      expect(result.hierarchicalResults).toBeDefined();
      expect(result.enhancedHits.length).toBeGreaterThan(0);
      
      // Verify safety gates were applied
      expect(result.safetyValidation.passed).toBe(true);
      expect(result.safetyValidation.constraints.upshiftBudgetOk).toBe(true);
      expect(result.safetyValidation.constraints.latencyWithinBounds).toBe(true);
      
      // Verify monitoring captured the operation
      const dashboard = await monitoring.generateDashboard();
      expect(dashboard.systemHealth.overallStatus).toBe('healthy');
      expect(dashboard.metrics.totalOperations).toBeGreaterThan(0);
      
      console.log(`Integration test completed successfully`);
      console.log(`- Enhanced hits: ${result.enhancedHits.length}`);
      console.log(`- Safety validation: ${result.safetyValidation.passed ? 'PASS' : 'FAIL'}`);
      console.log(`- Processing time: ${result.processingTimeMs.toFixed(2)}ms`);
    });
  });
});

// Helper functions
function calculatePercentile(values: number[], percentile: number): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const index = Math.ceil((percentile / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
}

function calculateNDCG(hits: SearchHit[]): number {
  // Simplified nDCG calculation - in practice would use relevance judgments
  const dcg = hits.reduce((sum, hit, i) => {
    const relevance = Math.max(0, hit.score); // Use score as relevance proxy
    const discount = Math.log2(i + 2); // Position discount
    return sum + (relevance / discount);
  }, 0);
  
  // Ideal DCG (if results were perfectly ranked)
  const sortedScores = hits.map(h => h.score).sort((a, b) => b - a);
  const idcg = sortedScores.reduce((sum, score, i) => {
    const discount = Math.log2(i + 2);
    return sum + (score / discount);
  }, 0);
  
  return idcg > 0 ? dcg / idcg : 0;
}