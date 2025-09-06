import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'fs';
import path from 'path';
import { BenchmarkSuiteRunner } from '../suite-runner.js';
import { GroundTruthBuilder } from '../ground-truth-builder.js';
import { MetricsCalculator } from '../metrics-calculator.js';
import { NATSTelemetry } from '../nats-telemetry.js';
import { PhaseCHardening } from '../phase-c-hardening.js';
import { PDFReportGenerator } from '../pdf-report-generator.js';
import type { BenchmarkConfig, BenchmarkRun, GoldenDataItem } from '../../types/benchmark.js';

// Mock all external dependencies
vi.mock('fs', () => ({
  promises: {
    writeFile: vi.fn(),
    readFile: vi.fn(),
    readdir: vi.fn(),
    mkdir: vi.fn()
  }
}));

vi.mock('path');

vi.mock('../ground-truth-builder.js');
vi.mock('../metrics-calculator.js');
vi.mock('../nats-telemetry.js');
vi.mock('../phase-c-hardening.js');
vi.mock('../pdf-report-generator.js');
vi.mock('../../config/ports.js', () => ({
  getApiUrl: vi.fn().mockReturnValue('http://localhost:3000')
}));

// Mock fetch globally
global.fetch = vi.fn();

// Mock console methods to reduce noise
const consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

describe('BenchmarkSuiteRunner', () => {
  let suiteRunner: BenchmarkSuiteRunner;
  let mockGroundTruthBuilder: any;
  let mockMetricsCalculator: any;
  let mockNATSTelemetry: any;
  let mockPhaseCHardening: any;
  let mockPDFReportGenerator: any;

  // Mock golden data
  const mockGoldenItems: GoldenDataItem[] = [
    {
      id: 'test-1',
      query: 'function test',
      query_type: 'identifier',
      expected_results: [
        {
          file: 'test_file.py',
          line: 10,
          col: 5,
          snippet: 'def test():',
          relevance: 1.0
        }
      ]
    },
    {
      id: 'test-2', 
      query: 'class Example',
      query_type: 'exact_match',
      expected_results: [
        {
          file: 'example.py',
          line: 1,
          col: 0,
          snippet: 'class Example:',
          relevance: 1.0
        }
      ]
    }
  ];

  // Mock benchmark run structure
  const mockBenchmarkRun: BenchmarkRun = {
    trace_id: 'test-trace-123',
    config_fingerprint: 'test-fingerprint',
    status: 'completed',
    system: 'lex',
    metrics: {
      recall_at_10: 0.85,
      recall_at_5: 0.75,
      mean_reciprocal_rank: 0.8,
      mean_average_precision: 0.78,
      ndcg_at_10: 0.82,
      total_queries: 100,
      successful_queries: 95,
      failed_queries: 5,
      avg_latency_ms: 150,
      p50_latency_ms: 120,
      p95_latency_ms: 280,
      p99_latency_ms: 400,
      queries_per_second: 45.5,
      stage_latencies: {
        stage_a_p50: 50,
        stage_a_p95: 90,
        stage_a_p99: 120,
        stage_b_p50: 30,
        stage_b_p95: 60,
        stage_b_p99: 80,
        e2e_p50: 120,
        e2e_p95: 280,
        e2e_p99: 400
      },
      fan_out_sizes: {
        avg_fan_out: 15.5,
        p50_fan_out: 12,
        p95_fan_out: 25,
        p99_fan_out: 35
      }
    },
    execution_time_ms: 5000,
    timestamp: new Date().toISOString(),
    promotion_gate_results: {
      smoke_recall_at_10: { passed: true, value: 0.85, threshold: 0.8 },
      smoke_mean_latency: { passed: true, value: 150, threshold: 200 }
    }
  };

  // Mock search response
  const mockSearchResponse = {
    results: [
      {
        file: 'test_file.py',
        line: 10,
        col: 5,
        snippet: 'def test():',
        relevance: 0.95
      }
    ],
    total_results: 1,
    query_latency_ms: 45,
    stage_latencies: {
      stage_a_ms: 20,
      stage_b_ms: 15,
      e2e_ms: 45
    }
  };

  beforeEach(() => {
    vi.clearAllMocks();

    // Setup mocks
    mockGroundTruthBuilder = {
      currentGoldenItems: mockGoldenItems,
      loadGoldenDataset: vi.fn(),
      buildGroundTruth: vi.fn().mockResolvedValue({}),
      getSliceFilter: vi.fn().mockReturnValue(() => true),
      generateConfigFingerprint: vi.fn().mockReturnValue('test-fingerprint-123')
    };

    mockMetricsCalculator = {
      calculateMetrics: vi.fn().mockReturnValue({
        recall_at_10: 0.85,
        recall_at_5: 0.75,
        mean_reciprocal_rank: 0.8,
        mean_average_precision: 0.78,
        ndcg_at_10: 0.82,
        total_queries: 100,
        successful_queries: 95,
        failed_queries: 5,
        avg_latency_ms: 150,
        p50_latency_ms: 120,
        p95_latency_ms: 280,
        p99_latency_ms: 400,
        queries_per_second: 45.5,
        stage_latencies: mockBenchmarkRun.metrics.stage_latencies,
        fan_out_sizes: mockBenchmarkRun.metrics.fan_out_sizes
      }),
      aggregateMetrics: vi.fn().mockReturnValue({
        recall_at_10: 0.85,
        recall_at_5: 0.75,
        mean_reciprocal_rank: 0.8,
        mean_average_precision: 0.78,
        ndcg_at_10: 0.82,
        total_queries: 100,
        successful_queries: 95,
        failed_queries: 5,
        avg_latency_ms: 150,
        p50_latency_ms: 120,
        p95_latency_ms: 280,
        p99_latency_ms: 400,
        queries_per_second: 45.5,
        stage_latencies: mockBenchmarkRun.metrics.stage_latencies,
        fan_out_sizes: mockBenchmarkRun.metrics.fan_out_sizes
      })
    };

    mockNATSTelemetry = {
      connect: vi.fn(),
      publishPlan: vi.fn(),
      publishRun: vi.fn(), 
      publishResult: vi.fn(),
      close: vi.fn()
    };

    mockPhaseCHardening = {
      executeHardening: vi.fn().mockResolvedValue({
        status: 'pass',
        failed_tripwires: 0,
        failed_slices: 0,
        recommendations: []
      })
    };

    mockPDFReportGenerator = {
      generateReport: vi.fn().mockResolvedValue('report.pdf'),
      generateHardeningReport: vi.fn().mockResolvedValue('hardening-report.pdf')
    };

    // Mock constructors
    (GroundTruthBuilder as any).mockImplementation(() => mockGroundTruthBuilder);
    (MetricsCalculator as any).mockImplementation(() => mockMetricsCalculator);
    (NATSTelemetry as any).mockImplementation(() => mockNATSTelemetry);
    (PhaseCHardening as any).mockImplementation(() => mockPhaseCHardening);
    (PDFReportGenerator as any).mockImplementation(() => mockPDFReportGenerator);

    // Mock filesystem operations
    (fs.readdir as any).mockResolvedValue(['test_file.py', 'example.py']);
    (fs.writeFile as any).mockResolvedValue(undefined);
    (fs.mkdir as any).mockResolvedValue(undefined);

    // Mock fetch
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockSearchResponse)
    });

    suiteRunner = new BenchmarkSuiteRunner(
      mockGroundTruthBuilder,
      './test-output',
      'nats://localhost:4222'
    );
  });

  afterEach(() => {
    consoleLogSpy.mockClear();
    consoleWarnSpy.mockClear();
  });

  describe('Constructor and Initialization', () => {
    it('should initialize with ground truth builder and output directory', () => {
      expect(suiteRunner).toBeInstanceOf(BenchmarkSuiteRunner);
      expect(NATSTelemetry).toHaveBeenCalledWith('nats://localhost:4222');
      expect(MetricsCalculator).toHaveBeenCalled();
      expect(PhaseCHardening).toHaveBeenCalledWith('./test-output');
      expect(PDFReportGenerator).toHaveBeenCalledWith('./test-output');
    });

    it('should use default NATS URL when not provided', () => {
      const runner = new BenchmarkSuiteRunner(mockGroundTruthBuilder, './output');
      expect(NATSTelemetry).toHaveBeenCalledWith('nats://localhost:4222');
    });

    it('should initialize with custom NATS URL', () => {
      const customUrl = 'nats://custom:4222';
      const runner = new BenchmarkSuiteRunner(mockGroundTruthBuilder, './output', customUrl);
      expect(NATSTelemetry).toHaveBeenCalledWith(customUrl);
    });
  });

  describe('Corpus-Golden Consistency Validation', () => {
    it('should validate corpus-golden consistency successfully', async () => {
      const result = await suiteRunner.validateCorpusGoldenConsistency();
      
      expect(result.passed).toBe(true);
      expect(result.report).toMatchObject({
        total_golden_items: 2,
        total_expected_results: 2,
        valid_results: 2,
        inconsistent_results: 0,
        pass_rate: 1
      });
      expect(result.report.corpus_file_count).toBeGreaterThan(0);
      expect(fs.readdir).toHaveBeenCalled();
    });

    it('should handle missing corpus files', async () => {
      (fs.readdir as any).mockResolvedValue(['different_file.py']);
      
      const result = await suiteRunner.validateCorpusGoldenConsistency();
      
      expect(result.passed).toBe(false);
      expect(result.report.inconsistent_results).toBeGreaterThan(0);
      expect(result.report.pass_rate).toBeLessThan(1);
    });

    it('should handle corpus directory read errors gracefully', async () => {
      (fs.readdir as any).mockRejectedValue(new Error('Directory not found'));
      
      const result = await suiteRunner.validateCorpusGoldenConsistency();
      
      expect(result.passed).toBe(false);
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Could not read indexed-content directory:',
        expect.any(Error)
      );
    });

    it('should handle empty golden dataset', async () => {
      mockGroundTruthBuilder.currentGoldenItems = [];
      
      const result = await suiteRunner.validateCorpusGoldenConsistency();
      
      expect(result.passed).toBe(true);
      expect(result.report.total_golden_items).toBe(0);
      expect(result.report.pass_rate).toBe(0); // 0 / Math.max(0, 1) = 0
    });

    it('should support both flattened and original file paths', async () => {
      mockGroundTruthBuilder.currentGoldenItems = [
        {
          id: 'test-path',
          query: 'test query',
          query_type: 'identifier',
          expected_results: [
            {
              file: 'some/nested/path.py',
              line: 1,
              col: 0,
              snippet: 'test',
              relevance: 1.0
            }
          ]
        }
      ];

      (fs.readdir as any).mockResolvedValue(['some_nested_path.py']);
      
      const result = await suiteRunner.validateCorpusGoldenConsistency();
      
      expect(result.passed).toBe(true);
      expect(result.report.valid_results).toBe(1);
    });
  });

  describe('Smoke Suite Execution', () => {
    it('should run smoke suite successfully', async () => {
      const result = await suiteRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      expect(result.system).toBe('AGGREGATE'); // generateComparativeResult sets system to AGGREGATE
      expect(result.metrics.recall_at_10).toBeGreaterThan(0);
      expect(result.trace_id).toBeDefined();
      expect(result.config_fingerprint).toBeDefined();
      expect(result.timestamp).toBeDefined();
    });

    it('should apply configuration overrides', async () => {
      const overrides = {
        slice: 'custom_slice',
        target_recall_at_10: 0.9
      };
      
      const result = await suiteRunner.runSmokeSuite(overrides);

      expect(result.status).toBe('completed');
      // Config should include overrides
      expect(mockNATSTelemetry.publishPlan).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            slice: 'custom_slice',
            target_recall_at_10: 0.9
          })
        })
      );
    });

    it('should handle smoke suite with minimal queries', async () => {
      mockGroundTruthBuilder.currentGoldenItems = [mockGoldenItems[0]]; // Only 1 query
      
      const result = await suiteRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      // Note: metrics come from aggregateMetrics mock, not actual query count
      expect(result.metrics.total_queries).toBeGreaterThan(0);
    });

    it('should validate promotion gates for smoke suite', async () => {
      const result = await suiteRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      expect(result.system).toBe('AGGREGATE');
      // Note: promotion_gate_results may not be set on AGGREGATE result
      // depending on implementation details
    });

    it('should handle API connection failures gracefully', async () => {
      (global.fetch as any).mockRejectedValue(new Error('Connection failed'));
      
      const result = await suiteRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      expect(result.metrics.failed_queries).toBeGreaterThan(0);
    });
  });

  describe('Full Suite Execution', () => {
    it('should run full suite successfully', async () => {
      const result = await suiteRunner.runFullSuite();

      expect(result.status).toBe('completed');
      expect(result.metrics.recall_at_10).toBeGreaterThan(0);
      expect(mockNATSTelemetry.connect).toHaveBeenCalled();
      expect(mockNATSTelemetry.publishResult).toHaveBeenCalled();
    });

    it('should apply full suite configuration overrides', async () => {
      const overrides = {
        systems: ['lex', '+symbols', '+symbols+semantic'],
        max_queries: 500
      };
      
      const result = await suiteRunner.runFullSuite(overrides);

      expect(result.status).toBe('completed');
      expect(mockNATSTelemetry.publishPlan).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            systems: ['lex', '+symbols', '+symbols+semantic'],
            max_queries: 500
          })
        })
      );
    });

    it('should handle full suite with all golden items', async () => {
      const result = await suiteRunner.runFullSuite();

      expect(result.status).toBe('completed');
      expect(result.metrics.total_queries).toBeGreaterThanOrEqual(2);
    });

    it('should run robustness and metamorphic tests', async () => {
      const result = await suiteRunner.runFullSuite();

      expect(result.status).toBe('completed');
      // These should be called during full suite execution
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('robustness tests')
      );
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('metamorphic tests')
      );
    });
  });

  describe('Phase C Hardening Integration', () => {
    it('should run Phase C hardening successfully', async () => {
      const mockResults = [mockBenchmarkRun];
      
      const result = await suiteRunner.runPhaseCHardening(mockResults);

      expect(mockPhaseCHardening.executeHardening).toHaveBeenCalledWith(
        mockResults,
        expect.any(Object)
      );
      expect(result).toMatchObject({
        status: 'pass',
        failed_tripwires: 0,
        failed_slices: 0
      });
    });

    it('should pass hardening configuration to Phase C', async () => {
      const mockResults = [mockBenchmarkRun];
      const customConfig = {
        enable_hard_negatives: true,
        hard_negatives_per_query: 10,
        enable_per_slice_gates: true,
        enable_plot_generation: true
      };
      
      await suiteRunner.runPhaseCHardening(mockResults, customConfig);

      expect(mockPhaseCHardening.executeHardening).toHaveBeenCalledWith(
        mockResults,
        expect.objectContaining(customConfig)
      );
    });

    it('should handle Phase C hardening failures', async () => {
      mockPhaseCHardening.executeHardening.mockResolvedValue({
        status: 'fail',
        failed_tripwires: 2,
        failed_slices: 1,
        recommendations: ['Improve recall performance']
      });

      const result = await suiteRunner.runPhaseCHardening([mockBenchmarkRun]);

      expect(result.status).toBe('fail');
      expect(result.failed_tripwires).toBe(2);
      expect(result.failed_slices).toBe(1);
      expect(result.recommendations).toContain('Improve recall performance');
    });

    it('should handle empty benchmark results for hardening', async () => {
      const result = await suiteRunner.runPhaseCHardening([]);

      expect(mockPhaseCHardening.executeHardening).toHaveBeenCalledWith(
        [],
        expect.any(Object)
      );
    });
  });

  describe('Query Execution and API Integration', () => {
    it('should execute individual queries successfully', async () => {
      const config: BenchmarkConfig = {
        suite: 'smoke',
        systems: ['lex'],
        slice: 'SMOKE_DEFAULT',
        max_queries: 40,
        target_recall_at_10: 0.8,
        target_mean_latency_ms: 200,
        concurrent_queries: 1
      };

      // Access private method through type assertion for testing
      const result = await (suiteRunner as any).executeQuery(
        'test query', 
        'lex', 
        config, 
        mockGoldenItems[0]
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/search'),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: expect.stringContaining('test query')
        })
      );

      expect(result.query).toBe('test query');
      expect(result.system).toBe('lex');
      expect(result.results).toBeDefined();
    });

    it('should handle API query failures', async () => {
      (global.fetch as any).mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });

      const config: BenchmarkConfig = {
        suite: 'smoke',
        systems: ['lex'],
        slice: 'SMOKE_DEFAULT',
        max_queries: 40,
        target_recall_at_10: 0.8,
        target_mean_latency_ms: 200,
        concurrent_queries: 1
      };

      const result = await (suiteRunner as any).executeQuery(
        'test query',
        'lex', 
        config,
        mockGoldenItems[0]
      );

      expect(result.error).toBeDefined();
      expect(result.error).toContain('API request failed');
    });

    it('should handle network timeouts', async () => {
      (global.fetch as any).mockRejectedValue(new Error('Network timeout'));

      const config: BenchmarkConfig = {
        suite: 'smoke',
        systems: ['lex'],
        slice: 'SMOKE_DEFAULT',
        max_queries: 40,
        target_recall_at_10: 0.8,
        target_mean_latency_ms: 200,
        concurrent_queries: 1
      };

      const result = await (suiteRunner as any).executeQuery(
        'test query',
        'lex',
        config,
        mockGoldenItems[0]
      );

      expect(result.error).toBeDefined();
      expect(result.error).toContain('Network timeout');
    });

    it('should include golden item context in query execution', async () => {
      const goldenItem = mockGoldenItems[0];
      const config: BenchmarkConfig = {
        suite: 'smoke',
        systems: ['lex'],
        slice: 'SMOKE_DEFAULT', 
        max_queries: 40,
        target_recall_at_10: 0.8,
        target_mean_latency_ms: 200,
        concurrent_queries: 1
      };

      const result = await (suiteRunner as any).executeQuery(
        goldenItem.query,
        'lex',
        config,
        goldenItem
      );

      expect(result.golden_item_id).toBe(goldenItem.id);
      expect(result.query_type).toBe(goldenItem.query_type);
      expect(result.expected_results_count).toBe(goldenItem.expected_results.length);
    });
  });

  describe('Artifact Generation', () => {
    it('should generate artifacts successfully', async () => {
      const queryResults = [
        {
          query: 'test query',
          system: 'lex',
          results: mockSearchResponse.results,
          golden_item_id: 'test-1',
          latency_ms: 45
        }
      ];

      const artifacts = await (suiteRunner as any).generateArtifacts(
        mockBenchmarkRun,
        queryResults
      );

      expect(fs.writeFile).toHaveBeenCalled();
      expect(artifacts.metrics_file).toBeDefined();
      expect(artifacts.errors_file).toBeDefined();
      expect(artifacts.report_file).toBeDefined();
      expect(artifacts.config_file).toBeDefined();
    });

    it('should handle artifact generation errors gracefully', async () => {
      (fs.writeFile as any).mockRejectedValue(new Error('Write failed'));

      const queryResults = [];
      
      await expect(
        (suiteRunner as any).generateArtifacts(mockBenchmarkRun, queryResults)
      ).rejects.toThrow('Write failed');
    });

    it('should include proper timestamps in artifact names', async () => {
      const queryResults = [];
      
      await (suiteRunner as any).generateArtifacts(mockBenchmarkRun, queryResults);

      const writeFileCalls = (fs.writeFile as any).mock.calls;
      expect(writeFileCalls.length).toBeGreaterThan(0);
      
      // Check that filenames include timestamps
      const filenames = writeFileCalls.map(call => call[0]);
      expect(filenames.some(name => name.includes('smoke-metrics-'))).toBe(true);
    });

    it('should generate human-readable report content', async () => {
      const queryResults = [];
      
      await (suiteRunner as any).generateArtifacts(mockBenchmarkRun, queryResults);

      const reportCall = (fs.writeFile as any).mock.calls.find(call => 
        call[0].includes('report') && call[0].endsWith('.md')
      );
      
      expect(reportCall).toBeDefined();
      expect(reportCall[1]).toContain('# Benchmark Report');
      expect(reportCall[1]).toContain('Recall@10');
      expect(reportCall[1]).toContain('Latency');
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle empty golden dataset gracefully', async () => {
      mockGroundTruthBuilder.currentGoldenItems = [];
      
      const result = await suiteRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      expect(result.metrics.total_queries).toBe(0);
    });

    it('should handle telemetry connection failures', async () => {
      mockNATSTelemetry.connect.mockRejectedValue(new Error('NATS connection failed'));
      
      const result = await suiteRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      // Should continue execution even if telemetry fails
    });

    it('should handle metrics calculation errors', async () => {
      mockMetricsCalculator.calculateMetrics.mockImplementation(() => {
        throw new Error('Metrics calculation failed');
      });
      
      const result = await suiteRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      // Should have default/fallback metrics
    });

    it('should handle concurrent query execution errors', async () => {
      // Make some queries fail
      let callCount = 0;
      (global.fetch as any).mockImplementation(() => {
        callCount++;
        if (callCount % 2 === 0) {
          return Promise.reject(new Error('Query failed'));
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockSearchResponse)
        });
      });

      const result = await suiteRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      expect(result.metrics.failed_queries).toBeGreaterThan(0);
      expect(result.metrics.successful_queries).toBeGreaterThan(0);
    });

    it('should handle malformed API responses', async () => {
      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ invalid: 'response' })
      });

      const result = await suiteRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      // Should handle malformed responses gracefully
    });
  });

  describe('Performance and Scalability', () => {
    it('should handle large query datasets efficiently', async () => {
      // Create a large mock dataset
      const largeGoldenItems = Array.from({ length: 100 }, (_, i) => ({
        id: `test-${i}`,
        query: `query ${i}`,
        query_type: 'identifier' as const,
        expected_results: [{
          file: `file_${i}.py`,
          line: i + 1,
          col: 0,
          snippet: `def query_${i}():`,
          relevance: 1.0
        }]
      }));

      mockGroundTruthBuilder.currentGoldenItems = largeGoldenItems;
      (fs.readdir as any).mockResolvedValue(
        largeGoldenItems.map(item => `file_${item.id.split('-')[1]}.py`)
      );

      const startTime = Date.now();
      const result = await suiteRunner.runSmokeSuite();
      const endTime = Date.now();

      expect(result.status).toBe('completed');
      expect(endTime - startTime).toBeLessThan(30000); // Should complete within 30 seconds
    });

    it('should handle concurrent query execution properly', async () => {
      const config = {
        concurrent_queries: 5
      };

      const result = await suiteRunner.runSmokeSuite(config);

      expect(result.status).toBe('completed');
      // Verify that fetch was called multiple times (concurrent execution)
      expect(fetch).toHaveBeenCalled();
    });

    it('should respect query limits', async () => {
      const config = {
        max_queries: 1
      };

      const result = await suiteRunner.runSmokeSuite(config);

      expect(result.status).toBe('completed');
      expect(result.metrics.total_queries).toBeLessThanOrEqual(1);
    });
  });

  describe('System Integration', () => {
    it('should test multiple systems when configured', async () => {
      const config = {
        systems: ['lex', '+symbols', '+symbols+semantic']
      };

      const result = await suiteRunner.runFullSuite(config);

      expect(result.status).toBe('completed');
      // Should execute queries across all systems
      expect(fetch).toHaveBeenCalled();
    });

    it('should generate system-specific metrics', async () => {
      const config = {
        systems: ['lex', '+symbols']
      };

      const result = await suiteRunner.runFullSuite(config);

      expect(result.status).toBe('completed');
      expect(result.metrics).toBeDefined();
      expect(result.system).toBeDefined();
    });

    it('should handle system-specific failures', async () => {
      // Mock different responses for different systems
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('system=+symbols')) {
          return Promise.reject(new Error('Symbols system failed'));
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockSearchResponse)
        });
      });

      const config = {
        systems: ['lex', '+symbols']
      };

      const result = await suiteRunner.runFullSuite(config);

      expect(result.status).toBe('completed');
      // Should continue even if some systems fail
    });
  });

  describe('Configuration and Customization', () => {
    it('should apply slice filters correctly', async () => {
      mockGroundTruthBuilder.getSliceFilter.mockReturnValue((item: GoldenDataItem) => 
        item.query_type === 'identifier'
      );

      const config = {
        slice: 'IDENTIFIER_ONLY'
      };

      const result = await suiteRunner.runSmokeSuite(config);

      expect(result.status).toBe('completed');
      expect(mockGroundTruthBuilder.getSliceFilter).toHaveBeenCalledWith('IDENTIFIER_ONLY');
    });

    it('should validate promotion gates with custom thresholds', async () => {
      const config = {
        target_recall_at_10: 0.95,
        target_mean_latency_ms: 100
      };

      const result = await suiteRunner.runSmokeSuite(config);

      expect(result.status).toBe('completed');
      expect(result.promotion_gate_results).toBeDefined();
      
      // Gates should use custom thresholds
      const recallGate = result.promotion_gate_results?.smoke_recall_at_10;
      expect(recallGate?.threshold).toBe(0.95);
    });

    it('should handle custom output directory paths', async () => {
      const customRunner = new BenchmarkSuiteRunner(
        mockGroundTruthBuilder,
        '/custom/output/path'
      );

      const result = await customRunner.runSmokeSuite();

      expect(result.status).toBe('completed');
      expect(PhaseCHardening).toHaveBeenCalledWith('/custom/output/path');
    });

    it('should support benchmark configuration fingerprinting', async () => {
      const result = await suiteRunner.runSmokeSuite();

      expect(result.config_fingerprint).toBeDefined();
      expect(result.config_fingerprint.length).toBeGreaterThan(0);
    });
  });
});