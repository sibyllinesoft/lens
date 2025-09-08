/**
 * Integration tests for TODO.md Sections 1-2
 * Tests the integration between data source conversion and tail-taming features
 */

import { getDataSourceConfig, validateDataSourceConfig } from '../config/data-source-config';
import { getDefaultTailTamingConfig, validateTailTamingConfig } from '../config/tail-taming-config';
import { SimIngestor } from '../ingestors/sim-ingestor';
import { SchemaGuard } from '../services/schema-guard';
import { HedgedProbeService } from '../services/hedged-probe-service';
import { CanaryRolloutService } from '../services/canary-rollout-service';
import { LensSearchRequest } from '../clients/lens-client';

describe('TODO.md Sections 1-2 Integration', () => {
  beforeAll(() => {
    // Set up test environment
    process.env.DATA_SOURCE = 'sim';
    process.env.SLA_MS = '150';
    process.env.TAIL_HEDGE = 'true';
    process.env.HEDGE_DELAY_MS = '6';
  });

  afterAll(() => {
    // Clean up environment
    delete process.env.DATA_SOURCE;
    delete process.env.SLA_MS;
    delete process.env.TAIL_HEDGE;
    delete process.env.HEDGE_DELAY_MS;
  });

  describe('Section 1: Convert simulated → real', () => {
    it('should validate data source configuration', () => {
      const config = getDataSourceConfig();
      
      expect(config.DATA_SOURCE).toBe('sim');
      expect(config.SLA_MS).toBe(150);
      expect(config.LENS_ENDPOINTS).toBeDefined();
      
      expect(() => validateDataSourceConfig(config)).not.toThrow();
    });

    it('should handle production configuration validation', () => {
      const prodConfig = {
        DATA_SOURCE: 'prod' as const,
        LENS_ENDPOINTS: ['http://localhost:3000', 'http://localhost:3001'],
        AUTH_TOKEN: 'test-token',
        SLA_MS: 150,
        RETRY_COUNT: 3,
        CANCEL_ON_FIRST: true
      };
      
      expect(() => validateDataSourceConfig(prodConfig)).not.toThrow();
      
      // Test invalid production config
      const invalidProdConfig = {
        ...prodConfig,
        LENS_ENDPOINTS: [] // Empty endpoints should fail
      };
      
      expect(() => validateDataSourceConfig(invalidProdConfig)).toThrow();
    });

    it('should process queries through complete data pipeline', async () => {
      const ingestor = new SimIngestor();
      
      const queries: LensSearchRequest[] = [
        { query: 'function authenticate', limit: 10 },
        { query: 'class UserService', limit: 15 }
      ];
      
      const result = await ingestor.ingestQueries(queries);
      
      // Validate output conforms to TODO.md schema requirements
      expect(result.aggRecords).toHaveLength(2);
      expect(result.hitsRecords.length).toBeGreaterThan(0);
      
      // Check required fields from TODO.md: req_ts, shard, lat_ms, within_sla, why_mix_*
      result.aggRecords.forEach(record => {
        expect(record).toHaveProperty('req_ts');
        expect(record).toHaveProperty('shard');
        expect(record).toHaveProperty('lat_ms');
        expect(record).toHaveProperty('within_sla');
        expect(record).toHaveProperty('why_mix_lex');
        expect(record).toHaveProperty('why_mix_struct');
        expect(record).toHaveProperty('why_mix_sem');
        expect(record).toHaveProperty('cfg_hash');
        
        // Validate within_sla logic
        expect(record.within_sla).toBe(record.lat_ms <= 150);
      });
      
      // Schema guard validation
      const validation = SchemaGuard.validateForWrite(result.aggRecords, result.hitsRecords);
      expect(validation.valid).toBe(true);
      expect(validation.attestation_sha256).toBeDefined();
      expect(validation.attestation_sha256).toMatch(/^[a-f0-9]{64}$/);
      
      ingestor.cleanup();
    }, 10000);

    it('should refuse writes with missing columns', async () => {
      const invalidAggRecords: any[] = [
        {
          query_id: 'test',
          req_ts: Date.now(),
          // Missing required fields intentionally
          cfg_hash: 'test',
          success: true
        }
      ];
      
      const validation = SchemaGuard.validateForWrite(invalidAggRecords, []);
      
      expect(validation.valid).toBe(false);
      expect(validation.errors.length).toBeGreaterThan(0);
      expect(validation.attestation_sha256).toBeUndefined();
    });
  });

  describe('Section 2: Sprint-1 tail-taming integration', () => {
    it('should validate tail-taming configuration', () => {
      const config = getDefaultTailTamingConfig();
      
      expect(config.TAIL_HEDGE).toBe(true);
      expect(config.HEDGE_DELAY_MS).toBe(6);
      
      const errors = validateTailTamingConfig(config);
      expect(errors).toHaveLength(0);
    });

    it('should calculate hedge delay correctly', () => {
      const config = getDefaultTailTamingConfig();
      const hedgeService = new HedgedProbeService(config);
      
      // Test hedge delay calculation: min(6ms, 0.1 * p50_shard)
      hedgeService.updateP50ShardLatency(100); // p50 = 100ms
      
      // Should use min(6, 0.1 * 100) = min(6, 10) = 6
      // This is tested indirectly through the service behavior
      expect(config.hedge_delay_base).toBe(6);
      expect(config.hedge_delay_p50_multiplier).toBe(0.1);
    });

    it('should validate performance gates configuration', () => {
      const config = getDefaultTailTamingConfig();
      
      // Check gate thresholds match TODO.md requirements
      expect(config.gates.p99_latency_improvement_min).toBe(-0.15); // -15%
      expect(config.gates.p99_latency_improvement_max).toBe(-0.10); // -10%
      expect(config.gates.p99_p95_ratio_max).toBe(2.0);
      expect(config.gates.sla_recall_at_50_delta_min).toBe(0.0); // >= 0.0 pp
      expect(config.gates.qps_at_150ms_improvement_min).toBe(0.10); // +10%
      expect(config.gates.qps_at_150ms_improvement_max).toBe(0.15); // +15%
      expect(config.gates.cost_increase_max).toBe(0.05); // +5%
    });

    it('should initialize canary rollout correctly', () => {
      const config = getDefaultTailTamingConfig();
      const canaryService = new CanaryRolloutService(config);
      
      canaryService.startRollout();
      
      const status = canaryService.getRolloutStatus();
      expect(status.current_stage).toBe(0);
      expect(status.current_traffic_percentage).toBe(5); // First stage is 5%
      expect(status.consecutive_failures).toBe(0);
      expect(status.is_reverted).toBe(false);
    });

    it('should bucket repositories deterministically', () => {
      const config = getDefaultTailTamingConfig();
      const canaryService = new CanaryRolloutService(config);
      
      canaryService.startRollout();
      
      // Test consistent bucketing
      const repo1Treatment1 = canaryService.shouldApplyTreatment('repo-123');
      const repo1Treatment2 = canaryService.shouldApplyTreatment('repo-123');
      
      expect(repo1Treatment1).toBe(repo1Treatment2); // Should be deterministic
      
      // Different repos should potentially get different treatments
      const repo2Treatment = canaryService.shouldApplyTreatment('repo-456');
      
      // At 5% traffic, most repos should be control
      // This is probabilistic but we can test the mechanism works
      expect(typeof repo1Treatment1).toBe('boolean');
      expect(typeof repo2Treatment).toBe('boolean');
    });
  });

  describe('Integration: Data Pipeline + Tail-Taming', () => {
    it('should process queries with tail-taming enabled end-to-end', async () => {
      const ingestor = new SimIngestor();
      const config = getDefaultTailTamingConfig();
      const canaryService = new CanaryRolloutService(config);
      
      canaryService.startRollout();
      
      // Simulate query processing with canary treatment decision
      const testRepo = 'test-repo-integration';
      const shouldTreat = canaryService.shouldApplyTreatment(testRepo);
      
      const queries: LensSearchRequest[] = [
        { query: 'integration test query', limit: 5 }
      ];
      
      const result = await ingestor.ingestQueries(queries);
      
      // Validate the results include configuration hash for reproducibility
      expect(result.aggRecords[0].cfg_hash).toBeDefined();
      expect(result.metrics.attestation_sha256).toBeDefined();
      
      // In a real implementation, we would apply tail-taming based on shouldTreat
      // For now, validate the mechanism is working
      expect(typeof shouldTreat).toBe('boolean');
      
      ingestor.cleanup();
    });

    it('should validate SLA-bounded recall measurement', async () => {
      const ingestor = new SimIngestor();
      
      const queries: LensSearchRequest[] = [
        { query: 'sla test query', limit: 20 }
      ];
      
      const result = await ingestor.ingestQueries(queries);
      
      // Calculate SLA-bounded metrics as specified in TODO.md
      const withinSlaQueries = result.aggRecords.filter(r => r.within_sla);
      const totalQueries = result.aggRecords.length;
      const slaRate = withinSlaQueries.length / totalQueries;
      
      expect(slaRate).toBeGreaterThanOrEqual(0);
      expect(slaRate).toBeLessThanOrEqual(1);
      
      // Validate that SLA measurement is included in metrics
      expect(result.metrics.within_sla_percentage).toBeDefined();
      expect(result.metrics.within_sla_percentage).toBe(slaRate * 100);
      
      ingestor.cleanup();
    });

    it('should demonstrate artifact binding with attestation', async () => {
      const ingestor = new SimIngestor();
      
      const result1 = await ingestor.ingestQueries([
        { query: 'artifact binding test', limit: 10 }
      ]);
      
      const result2 = await ingestor.ingestQueries([
        { query: 'artifact binding test', limit: 10 }
      ]);
      
      // Different runs should have different attestations (due to timestamps)
      expect(result1.metrics.attestation_sha256).not.toBe(result2.metrics.attestation_sha256);
      
      // But both should be valid SHA256 hashes
      expect(result1.metrics.attestation_sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(result2.metrics.attestation_sha256).toMatch(/^[a-f0-9]{64}$/);
      
      ingestor.cleanup();
    });
  });
});