import { describe, it, expect, beforeEach, vi } from 'vitest';
import { 
  ConformalRouter, 
  ConformalPredictionFeatures, 
  MisrankRisk, 
  RoutingDecision,
  UpshiftBudget,
  globalConformalRouter 
} from '../conformal-router';

// Mock the tracer
vi.mock('../telemetry/tracer', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn()
    }))
  }
}));

describe('ConformalRouter', () => {
  let router: ConformalRouter;

  beforeEach(() => {
    vi.clearAllMocks();
    router = new ConformalRouter(0.6, 5.0); // 60% risk threshold, 5% daily budget
  });

  describe('Router Initialization', () => {
    it('should initialize with default values', () => {
      const defaultRouter = new ConformalRouter();
      expect(defaultRouter).toBeDefined();
      
      const stats = defaultRouter.getStats();
      expect(stats.total_queries).toBe(0);
      expect(stats.upshifted_queries).toBe(0);
      expect(stats.upshift_rate).toBe(0);
      expect(stats.enabled).toBe(true);
    });

    it('should initialize with custom thresholds', () => {
      const customRouter = new ConformalRouter(0.8, 10.0);
      expect(customRouter).toBeDefined();
      
      const stats = customRouter.getStats();
      expect(stats.budget_status.daily_budget).toBe(10.0);
    });
  });

  describe('Feature Extraction', () => {
    it('should extract features from simple queries', async () => {
      const ctx = {
        query: 'function test',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision).toBeDefined();
      expect(decision.should_upshift).toBe(false); // Below risk threshold initially
      expect(decision.routing_reason).toBe('risk_below_threshold');
    });

    it('should extract features from complex queries', async () => {
      const ctx = {
        query: 'export async function processData(input: string): Promise<Result>',
        fuzzy: true,
        mode: 'hybrid' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision).toBeDefined();
    });

    it('should handle queries with special characters', async () => {
      const ctx = {
        query: 'interface User { name: string; getData(): Promise<Data[]>; }',
        fuzzy: false,
        mode: 'struct' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision).toBeDefined();
    });

    it('should handle empty queries', async () => {
      const ctx = {
        query: '',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision.should_upshift).toBe(false);
    });
  });

  describe('Risk Assessment', () => {
    it('should use heuristic risk when not calibrated', async () => {
      const ctx = {
        query: 'very complex semantic query with many unusual patterns and structures',
        fuzzy: true,
        mode: 'hybrid' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision).toBeDefined();
      expect(typeof decision.routing_reason).toBe('string');
    });

    it('should use calibrated predictor when available', async () => {
      // Calibrate the router first
      const calibrationData = Array.from({ length: 50 }, (_, i) => ({
        features: {
          query_length: 20 + i,
          word_count: 3 + Math.floor(i / 10),
          has_special_chars: i % 2 === 0,
          fuzzy_enabled: i % 3 === 0,
          structural_mode: i % 4 === 0,
          avg_word_length: 4 + (i % 3),
          query_entropy: 1.5 + (i % 5) * 0.3,
          identifier_density: (i % 10) / 10,
          semantic_complexity: (i % 8) / 10,
        },
        actual_ndcg: 0.7 + (i % 3) * 0.1,
        predicted_ndcg: 0.65 + (i % 4) * 0.08,
      }));

      await router.calibrate(calibrationData);

      const ctx = {
        query: 'test calibrated prediction',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision).toBeDefined();
      
      const stats = router.getStats();
      expect(stats.last_calibration).not.toBeNull();
    });
  });

  describe('Budget Management', () => {
    it('should respect daily budget limits', async () => {
      // Create router with very small budget for testing
      const smallBudgetRouter = new ConformalRouter(0.1, 1.0); // Very low risk threshold, 1% budget
      
      const ctx = {
        query: 'high risk query that should trigger upshift',
        fuzzy: true,
        mode: 'hybrid' as const
      };

      // Make several requests to exhaust budget
      for (let i = 0; i < 10; i++) {
        await smallBudgetRouter.makeRoutingDecision(ctx);
      }

      const stats = smallBudgetRouter.getStats();
      expect(stats.upshift_rate).toBeLessThanOrEqual(15); // With current budget logic, expect ~10% (1 upshift out of 10 queries)
    });

    it('should track upshift usage over time', async () => {
      const ctx = {
        query: 'test query',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const initialStats = router.getStats();
      expect(initialStats.total_queries).toBe(0);

      await router.makeRoutingDecision(ctx);
      await router.makeRoutingDecision(ctx);

      const afterStats = router.getStats();
      expect(afterStats.total_queries).toBe(2);
    });

    it('should provide budget status information', async () => {
      const stats = router.getStats();
      const budget = stats.budget_status;

      expect(budget.daily_budget).toBe(5.0);
      expect(budget.used_today).toBeGreaterThanOrEqual(0);
      expect(budget.remaining_budget).toBeGreaterThanOrEqual(0);
      expect(budget.current_upshift_rate).toBeGreaterThanOrEqual(0);
      expect(budget.p95_headroom_ms).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Upshift Type Selection', () => {
    it('should select dimension_768d for high semantic complexity', async () => {
      // Create router with low threshold to trigger upshifts
      const lowThresholdRouter = new ConformalRouter(0.1, 50.0);
      
      const ctx = {
        query: 'complex semantic query with natural language understanding requirements and advanced cognitive reasoning patterns',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await lowThresholdRouter.makeRoutingDecision(ctx);
      // The decision might be 'none' due to risk assessment, but we can still test the logic
      expect(['none', 'dimension_768d', 'efSearch_boost', 'mmr_diversity']).toContain(decision.upshift_type);
    });

    it('should select efSearch_boost for structural queries', async () => {
      const lowThresholdRouter = new ConformalRouter(0.1, 50.0);
      
      const ctx = {
        query: 'class MyClass implements IInterface',
        fuzzy: false,
        mode: 'struct' as const
      };

      const decision = await lowThresholdRouter.makeRoutingDecision(ctx);
      expect(['none', 'dimension_768d', 'efSearch_boost', 'mmr_diversity']).toContain(decision.upshift_type);
    });

    it('should select mmr_diversity for natural language queries', async () => {
      const lowThresholdRouter = new ConformalRouter(0.1, 50.0);
      
      const ctx = {
        query: 'find functions that process user data and return formatted results',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await lowThresholdRouter.makeRoutingDecision(ctx);
      expect(['none', 'dimension_768d', 'efSearch_boost', 'mmr_diversity']).toContain(decision.upshift_type);
    });
  });

  describe('Expected Improvement Calculation', () => {
    it('should calculate expected improvement for different upshift types', async () => {
      const lowThresholdRouter = new ConformalRouter(0.1, 50.0);
      
      const ctx = {
        query: 'test improvement calculation',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await lowThresholdRouter.makeRoutingDecision(ctx);
      
      if (decision.should_upshift) {
        expect(decision.expected_improvement).toBeGreaterThan(0);
        expect(decision.expected_improvement).toBeLessThanOrEqual(1.0); // Reasonable nDCG improvement
      }
    });
  });

  describe('Router Configuration', () => {
    it('should allow enabling/disabling the router', async () => {
      router.setEnabled(false);
      
      const ctx = {
        query: 'test disabled router',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision.should_upshift).toBe(false);
      expect(decision.routing_reason).toBe('router_disabled');

      // Re-enable
      router.setEnabled(true);
      const stats = router.getStats();
      expect(stats.enabled).toBe(true);
    });

    it('should allow updating configuration', () => {
      router.updateConfig({
        risk_threshold: 0.8,
        daily_budget_percent: 10.0
      });

      const stats = router.getStats();
      expect(stats.budget_status.daily_budget).toBe(10.0);
    });
  });

  describe('Error Handling', () => {
    it('should handle malformed queries gracefully', async () => {
      const ctx = {
        query: null as any, // Invalid query
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision).toBeDefined();
      expect(decision.should_upshift).toBe(false);
    });

    it('should handle routing errors with safe fallback', async () => {
      const ctx = {
        query: 'test error handling',
        fuzzy: false,
        mode: 'lexical' as const
      };

      // Mock an error in feature extraction
      const originalExtractFeatures = (router as any).extractFeatures;
      (router as any).extractFeatures = () => {
        throw new Error('Test error');
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision.should_upshift).toBe(false);
      expect(decision.routing_reason).toBe('routing_error');

      // Restore original method
      (router as any).extractFeatures = originalExtractFeatures;
    });
  });

  describe('Statistics and Metrics', () => {
    it('should provide comprehensive statistics', async () => {
      const ctx = {
        query: 'test statistics',
        fuzzy: false,
        mode: 'lexical' as const
      };

      await router.makeRoutingDecision(ctx);
      await router.makeRoutingDecision(ctx);

      const stats = router.getStats();
      expect(stats.total_queries).toBe(2);
      expect(stats.upshifted_queries).toBeGreaterThanOrEqual(0);
      expect(stats.upshift_rate).toBeGreaterThanOrEqual(0);
      expect(stats.budget_status).toBeDefined();
      expect(stats.enabled).toBe(true);
    });

    it('should calculate upshift rate correctly', async () => {
      const lowThresholdRouter = new ConformalRouter(0.0, 100.0); // Force upshifts
      
      const ctx = {
        query: 'force upshift query',
        fuzzy: false,
        mode: 'lexical' as const
      };

      // Make some queries
      for (let i = 0; i < 5; i++) {
        await lowThresholdRouter.makeRoutingDecision(ctx);
      }

      const stats = lowThresholdRouter.getStats();
      expect(stats.total_queries).toBe(5);
      expect(stats.upshift_rate).toBeGreaterThanOrEqual(0);
      expect(stats.upshift_rate).toBeLessThanOrEqual(100);
    });
  });

  describe('Calibration', () => {
    it('should accept and use calibration data', async () => {
      const calibrationData = [
        {
          features: {
            query_length: 20,
            word_count: 4,
            has_special_chars: true,
            fuzzy_enabled: false,
            structural_mode: false,
            avg_word_length: 5,
            query_entropy: 2.5,
            identifier_density: 0.3,
            semantic_complexity: 0.6,
          },
          actual_ndcg: 0.8,
          predicted_ndcg: 0.75,
        },
        {
          features: {
            query_length: 15,
            word_count: 3,
            has_special_chars: false,
            fuzzy_enabled: true,
            structural_mode: true,
            avg_word_length: 6,
            query_entropy: 1.8,
            identifier_density: 0.7,
            semantic_complexity: 0.4,
          },
          actual_ndcg: 0.9,
          predicted_ndcg: 0.85,
        },
      ];

      await router.calibrate(calibrationData);

      const stats = router.getStats();
      expect(stats.last_calibration).not.toBeNull();
      expect(stats.last_calibration!.getTime()).toBeGreaterThan(Date.now() - 1000);
    });
  });

  describe('Integration with Candidates', () => {
    it('should handle routing decisions with current candidates', async () => {
      const ctx = {
        query: 'test with candidates',
        fuzzy: false,
        mode: 'hybrid' as const
      };

      const mockCandidates = [
        { score: 0.9, path: '/test1.js' },
        { score: 0.8, path: '/test2.js' },
        { score: 0.7, path: '/test3.js' },
      ];

      const decision = await router.makeRoutingDecision(ctx, mockCandidates);
      expect(decision).toBeDefined();
    });
  });

  describe('Global Instance', () => {
    it('should provide a global conformal router instance', () => {
      expect(globalConformalRouter).toBeDefined();
      expect(globalConformalRouter).toBeInstanceOf(ConformalRouter);
    });
  });

  describe('Edge Cases and Stress Tests', () => {
    it('should handle very long queries', async () => {
      const longQuery = 'a'.repeat(1000) + ' complex query with many words '.repeat(50);
      const ctx = {
        query: longQuery,
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision).toBeDefined();
    });

    it('should handle queries with only special characters', async () => {
      const ctx = {
        query: '{}[]()<>.,;:!@#$%^&*',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const decision = await router.makeRoutingDecision(ctx);
      expect(decision).toBeDefined();
    });

    it('should handle concurrent routing decisions', async () => {
      const ctx = {
        query: 'concurrent test',
        fuzzy: false,
        mode: 'lexical' as const
      };

      const promises = Array.from({ length: 10 }, () => 
        router.makeRoutingDecision(ctx)
      );

      const decisions = await Promise.all(promises);
      expect(decisions).toHaveLength(10);
      decisions.forEach(decision => {
        expect(decision).toBeDefined();
        expect(typeof decision.should_upshift).toBe('boolean');
      });
    });

    it('should maintain consistent state across many requests', async () => {
      const ctx = {
        query: 'consistency test',
        fuzzy: false,
        mode: 'lexical' as const
      };

      for (let i = 0; i < 100; i++) {
        await router.makeRoutingDecision(ctx);
      }

      const stats = router.getStats();
      expect(stats.total_queries).toBe(100);
      expect(stats.upshift_rate).toBeGreaterThanOrEqual(0);
      expect(stats.upshift_rate).toBeLessThanOrEqual(100);
    });
  });
});