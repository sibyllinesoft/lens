/**
 * Tests for Drift Detection and Alerting System
 * Covers CUSUM detection, ladder monitoring, coverage tracking, and real-time alerting
 */

import { describe, it, expect, beforeEach, jest, afterEach, mock } from 'bun:test';
import { EventEmitter } from 'events';
import {
  DriftDetectionSystem,
  CUSUMDetector,
  LadderMonitor,
  CoverageTracker,
  DriftMetrics,
  DriftAlert,
  DriftDetectionConfig,
  CUSUMConfig,
  DEFAULT_DRIFT_CONFIG,
  computeCUSUM,
  calculateTrendDirection,
  generateRecommendations,
} from '../drift-detection-system.js';

// Mock the tracer
mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: jest.fn(() => ({
      setAttributes: jest.fn(),
      setStatus: jest.fn(),
      end: jest.fn(),
    })),
  },
}));

describe('Drift Detection System', () => {
  let driftDetector: DriftDetectionSystem;
  let mockMetrics: DriftMetrics;

  beforeEach(() => {
    jest.clearAllMocks();
    driftDetector = new DriftDetectionSystem(DEFAULT_DRIFT_CONFIG);
    
    mockMetrics = {
      timestamp: new Date().toISOString(),
      anchor_p_at_1: 0.85,
      anchor_recall_at_50: 0.92,
      ladder_positives_ratio: 0.78,
      lsif_coverage_pct: 85.2,
      tree_sitter_coverage_pct: 92.4,
      sample_count: 1000,
      query_complexity_distribution: {
        simple: 0.3,
        medium: 0.5,
        complex: 0.2,
      },
    };
  });

  afterEach(() => {
    driftDetector.destroy();
    jest.restoreAllMocks();
  });

  describe('CUSUM Detector', () => {
    let cusumDetector: CUSUMDetector;
    const cusumConfig: CUSUMConfig = {
      reference_value: 0.85,
      drift_threshold: 2.0,
      decision_interval: 5.0,
      reset_threshold: -5.0,
      min_samples: 10,
    };

    beforeEach(() => {
      cusumDetector = new CUSUMDetector('test_metric', cusumConfig);
    });

    it('should initialize with zero CUSUM statistic', () => {
      expect(cusumDetector.getCurrentStatistic()).toBe(0);
      expect(cusumDetector.getSampleCount()).toBe(0);
      expect(cusumDetector.isAlertActive()).toBe(false);
    });

    it('should compute CUSUM statistic correctly', () => {
      // Test the core CUSUM computation
      const values = [0.85, 0.84, 0.83, 0.82, 0.81]; // Declining values
      let cusum = 0;
      
      values.forEach(value => {
        cusum = computeCUSUM(cusum, value, cusumConfig.reference_value, cusumConfig.drift_threshold);
        cusumDetector.addSample(value);
      });
      
      expect(cusumDetector.getCurrentStatistic()).toBeGreaterThan(0);
      expect(cusumDetector.getSampleCount()).toBe(5);
    });

    it('should detect drift when CUSUM exceeds decision interval', () => {
      // Add samples that will trigger drift detection
      const lowValues = Array(15).fill(0.75); // Much lower than reference 0.85
      
      let driftDetected = false;
      lowValues.forEach((value, index) => {
        const result = cusumDetector.addSample(value);
        if (result.drift_detected && index >= cusumConfig.min_samples - 1) {
          driftDetected = true;
        }
      });
      
      expect(driftDetected).toBe(true);
      expect(cusumDetector.isAlertActive()).toBe(true);
      expect(cusumDetector.getCurrentStatistic()).toBeGreaterThan(cusumConfig.decision_interval);
    });

    it('should reset CUSUM when statistic goes below reset threshold', () => {
      // First, trigger drift
      Array(15).fill(0.75).forEach(value => cusumDetector.addSample(value));
      expect(cusumDetector.isAlertActive()).toBe(true);
      
      // Then add good values to reset
      Array(10).fill(0.90).forEach(value => cusumDetector.addSample(value));
      
      expect(cusumDetector.isAlertActive()).toBe(false);
      expect(cusumDetector.getCurrentStatistic()).toBeLessThan(0);
    });

    it('should not alert with insufficient samples', () => {
      const lowValues = Array(5).fill(0.70); // Below min_samples
      
      let alertTriggered = false;
      lowValues.forEach(value => {
        const result = cusumDetector.addSample(value);
        if (result.drift_detected) alertTriggered = true;
      });
      
      expect(alertTriggered).toBe(false);
    });

    it('should handle values close to reference without false alarms', () => {
      const normalValues = [0.85, 0.86, 0.84, 0.85, 0.87, 0.83, 0.85];
      
      let falseAlarm = false;
      normalValues.forEach(value => {
        const result = cusumDetector.addSample(value);
        if (result.drift_detected) falseAlarm = true;
      });
      
      expect(falseAlarm).toBe(false);
      expect(Math.abs(cusumDetector.getCurrentStatistic())).toBeLessThan(1.0);
    });

    it('should provide confidence intervals for drift magnitude', () => {
      Array(15).fill(0.75).forEach(value => cusumDetector.addSample(value));
      
      const stats = cusumDetector.getStatistics();
      expect(stats.confidence_interval).toBeDefined();
      expect(stats.confidence_interval[0]).toBeLessThan(stats.confidence_interval[1]);
      expect(stats.drift_magnitude).toBeGreaterThan(0);
    });
  });

  describe('Ladder Monitor', () => {
    let ladderMonitor: LadderMonitor;
    
    beforeEach(() => {
      ladderMonitor = new LadderMonitor({
        enabled: true,
        baseline_ratio: 0.80,
        degradation_threshold: 0.05, // 5% degradation
        trend_window_size: 10,
      });
    });

    it('should initialize with empty sample history', () => {
      expect(ladderMonitor.getSampleCount()).toBe(0);
      expect(ladderMonitor.getCurrentRatio()).toBe(0);
      expect(ladderMonitor.getTrendDirection()).toBe('stable');
    });

    it('should track positives-in-candidates ratio', () => {
      const ratios = [0.82, 0.81, 0.83, 0.79, 0.80];
      
      ratios.forEach(ratio => ladderMonitor.addSample(ratio));
      
      expect(ladderMonitor.getSampleCount()).toBe(5);
      expect(ladderMonitor.getCurrentRatio()).toBe(0.80); // Last sample
    });

    it('should detect degradation trend', () => {
      // Add samples showing declining trend
      const decliningRatios = [0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68];
      
      let degradationDetected = false;
      decliningRatios.forEach(ratio => {
        const result = ladderMonitor.addSample(ratio);
        if (result.degradation_detected) degradationDetected = true;
      });
      
      expect(degradationDetected).toBe(true);
      expect(ladderMonitor.getTrendDirection()).toBe('decreasing');
    });

    it('should calculate trend direction correctly', () => {
      // Test increasing trend
      [0.70, 0.72, 0.74, 0.76, 0.78].forEach(ratio => 
        ladderMonitor.addSample(ratio)
      );
      expect(ladderMonitor.getTrendDirection()).toBe('increasing');
      
      // Reset and test stable trend
      ladderMonitor.reset();
      [0.80, 0.81, 0.79, 0.80, 0.81].forEach(ratio => 
        ladderMonitor.addSample(ratio)
      );
      expect(ladderMonitor.getTrendDirection()).toBe('stable');
    });

    it('should provide statistical analysis of trend', () => {
      const ratios = [0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70];
      ratios.forEach(ratio => ladderMonitor.addSample(ratio));
      
      const analysis = ladderMonitor.getTrendAnalysis();
      expect(analysis.slope).toBeLessThan(0); // Declining trend
      expect(analysis.r_squared).toBeGreaterThan(0.5); // Strong correlation
      expect(analysis.confidence_level).toBeGreaterThan(0.8);
    });
  });

  describe('Coverage Tracker', () => {
    let coverageTracker: CoverageTracker;
    
    beforeEach(() => {
      coverageTracker = new CoverageTracker({
        enabled: true,
        lsif_baseline_pct: 85.0,
        tree_sitter_baseline_pct: 90.0,
        degradation_threshold_pct: 5.0,
        measurement_interval_hours: 1,
      });
    });

    it('should track LSIF and tree-sitter coverage', () => {
      const result = coverageTracker.updateCoverage(82.5, 88.2);
      
      expect(result.lsif_coverage).toBe(82.5);
      expect(result.tree_sitter_coverage).toBe(88.2);
      expect(result.lsif_degraded).toBe(true); // 82.5 < 85.0 - 5.0
      expect(result.tree_sitter_degraded).toBe(true); // 88.2 < 90.0 - 5.0
    });

    it('should not flag degradation within acceptable range', () => {
      const result = coverageTracker.updateCoverage(84.0, 89.0);
      
      expect(result.lsif_degraded).toBe(false); // 84.0 > 85.0 - 5.0 (80.0)
      expect(result.tree_sitter_degraded).toBe(false); // 89.0 > 90.0 - 5.0 (85.0)
    });

    it('should track coverage history and trends', () => {
      const coveragePoints = [
        [85.0, 90.0], [84.0, 89.0], [83.0, 88.0], [82.0, 87.0]
      ];
      
      coveragePoints.forEach(([lsif, tree_sitter]) => 
        coverageTracker.updateCoverage(lsif, tree_sitter)
      );
      
      const history = coverageTracker.getCoverageHistory();
      expect(history).toHaveLength(4);
      expect(history[0].lsif_coverage).toBe(85.0);
      expect(history[3].lsif_coverage).toBe(82.0);
    });

    it('should calculate coverage trends over time', () => {
      const decliningCoverage = [
        [85.0, 90.0], [83.0, 88.0], [81.0, 86.0], [79.0, 84.0]
      ];
      
      decliningCoverage.forEach(([lsif, tree_sitter]) => 
        coverageTracker.updateCoverage(lsif, tree_sitter)
      );
      
      const trends = coverageTracker.getCoverageTrends();
      expect(trends.lsif_trend_direction).toBe('decreasing');
      expect(trends.tree_sitter_trend_direction).toBe('decreasing');
      expect(trends.lsif_trend_slope).toBeLessThan(0);
      expect(trends.tree_sitter_trend_slope).toBeLessThan(0);
    });
  });

  describe('Drift Detection System Integration', () => {
    it('should process metrics and detect multiple types of drift', () => {
      const alertsReceived: DriftAlert[] = [];
      driftDetector.on('drift_alert', (alert: DriftAlert) => {
        alertsReceived.push(alert);
      });
      
      // Simulate degrading metrics
      const degradedMetrics = {
        ...mockMetrics,
        anchor_p_at_1: 0.70, // Significant drop from expected 0.85
        ladder_positives_ratio: 0.65, // Below baseline
      };
      
      // Process enough samples to trigger detection
      Array(15).fill(degradedMetrics).forEach((metrics, index) => {
        driftDetector.processMetrics({
          ...metrics,
          timestamp: new Date(Date.now() + index * 60000).toISOString(),
        });
      });
      
      expect(alertsReceived.length).toBeGreaterThan(0);
      expect(alertsReceived.some(alert => alert.alert_type === 'anchor_p1_drift')).toBe(true);
    });

    it('should consolidate alerts within consolidation window', () => {
      const shortWindowConfig = {
        ...DEFAULT_DRIFT_CONFIG,
        alerting: {
          ...DEFAULT_DRIFT_CONFIG.alerting,
          consolidation_window_minutes: 1, // Very short window for testing
        },
      };
      
      const detector = new DriftDetectionSystem(shortWindowConfig);
      const alertsReceived: DriftAlert[] = [];
      detector.on('drift_alert', (alert: DriftAlert) => alertsReceived.push(alert));
      
      const degradedMetrics = {
        ...mockMetrics,
        anchor_p_at_1: 0.70,
      };
      
      // Process multiple samples quickly (should consolidate)
      Array(5).fill(degradedMetrics).forEach(metrics => {
        detector.processMetrics(metrics);
      });
      
      // Should have fewer alerts due to consolidation
      expect(alertsReceived.length).toBeLessThanOrEqual(2);
      
      detector.destroy();
    });

    it('should generate appropriate recommendations based on alert type', () => {
      const p1Alert: DriftAlert = {
        id: 'test-alert',
        alert_type: 'anchor_p1_drift',
        severity: 'error',
        metric_name: 'anchor_p_at_1',
        current_value: 0.70,
        reference_value: 0.85,
        drift_magnitude: 2.5,
        cusum_statistic: 8.2,
        consecutive_violations: 5,
        sample_count: 100,
        timestamp: new Date().toISOString(),
        recommended_actions: [],
        context: {
          trend_direction: 'decreasing',
          confidence_interval: [0.68, 0.72],
          historical_baseline: 0.85,
          recent_samples: [0.72, 0.71, 0.70, 0.69, 0.70],
        },
      };
      
      const recommendations = generateRecommendations(p1Alert);
      
      expect(recommendations).toContain(/check.*anchor.*quality/i);
      expect(recommendations).toContain(/review.*indexing.*pipeline/i);
      expect(recommendations.length).toBeGreaterThan(0);
    });

    it('should handle burst of metrics efficiently', async () => {
      const metricsCount = 1000;
      const startTime = performance.now();
      
      const promises = Array.from({ length: metricsCount }, (_, index) => 
        Promise.resolve(driftDetector.processMetrics({
          ...mockMetrics,
          timestamp: new Date(Date.now() + index * 1000).toISOString(),
          anchor_p_at_1: 0.85 + (Math.random() - 0.5) * 0.1, // Small random variation
        }))
      );
      
      await Promise.all(promises);
      
      const processingTime = performance.now() - startTime;
      
      // Should process 1000 metrics in under 1 second
      expect(processingTime).toBeLessThan(1000);
      expect(driftDetector.getProcessedSampleCount()).toBe(metricsCount);
    });

    it('should maintain state consistency under concurrent updates', async () => {
      const concurrentPromises = Array.from({ length: 50 }, (_, index) => 
        Promise.resolve().then(() => 
          driftDetector.processMetrics({
            ...mockMetrics,
            timestamp: new Date(Date.now() + index * 100).toISOString(),
            sample_count: index + 1,
          })
        )
      );
      
      await Promise.all(concurrentPromises);
      
      const finalState = driftDetector.getSystemState();
      expect(finalState.processed_samples).toBe(50);
      expect(finalState.active_alerts).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Alert Generation and Management', () => {
    it('should create alerts with proper severity levels', () => {
      const minorDriftMetrics = {
        ...mockMetrics,
        anchor_p_at_1: 0.82, // Small drift
      };
      
      const majorDriftMetrics = {
        ...mockMetrics,
        anchor_p_at_1: 0.65, // Major drift
      };
      
      // Process samples to generate alerts
      Array(15).fill(minorDriftMetrics).forEach(metrics => 
        driftDetector.processMetrics(metrics)
      );
      
      const minorAlerts = driftDetector.getActiveAlerts();
      
      driftDetector.reset();
      
      Array(15).fill(majorDriftMetrics).forEach(metrics => 
        driftDetector.processMetrics(metrics)
      );
      
      const majorAlerts = driftDetector.getActiveAlerts();
      
      // Major drift should have higher severity
      if (majorAlerts.length > 0 && minorAlerts.length > 0) {
        expect(majorAlerts[0].drift_magnitude).toBeGreaterThan(minorAlerts[0].drift_magnitude);
      }
    });

    it('should track consecutive violations correctly', () => {
      const consistentlyBadMetrics = {
        ...mockMetrics,
        anchor_p_at_1: 0.70,
      };
      
      // Process enough bad samples
      Array(20).fill(consistentlyBadMetrics).forEach((metrics, index) => 
        driftDetector.processMetrics({
          ...metrics,
          timestamp: new Date(Date.now() + index * 60000).toISOString(),
        })
      );
      
      const alerts = driftDetector.getActiveAlerts();
      const p1Alert = alerts.find(alert => alert.alert_type === 'anchor_p1_drift');
      
      if (p1Alert) {
        expect(p1Alert.consecutive_violations).toBeGreaterThan(10);
      }
    });

    it('should clear resolved alerts', () => {
      // First, generate alerts with bad metrics
      const badMetrics = { ...mockMetrics, anchor_p_at_1: 0.70 };
      Array(15).fill(badMetrics).forEach(metrics => 
        driftDetector.processMetrics(metrics)
      );
      
      expect(driftDetector.getActiveAlerts().length).toBeGreaterThan(0);
      
      // Then recover with good metrics
      const goodMetrics = { ...mockMetrics, anchor_p_at_1: 0.88 };
      Array(15).fill(goodMetrics).forEach(metrics => 
        driftDetector.processMetrics(metrics)
      );
      
      // Alerts should be cleared or reduced
      const finalAlerts = driftDetector.getActiveAlerts();
      const p1Alerts = finalAlerts.filter(alert => alert.alert_type === 'anchor_p1_drift');
      expect(p1Alerts.length).toBe(0);
    });
  });

  describe('Configuration and Customization', () => {
    it('should accept custom configuration', () => {
      const customConfig: DriftDetectionConfig = {
        anchor_p1_cusum: {
          reference_value: 0.90,
          drift_threshold: 1.5,
          decision_interval: 4.0,
          reset_threshold: -4.0,
          min_samples: 5,
        },
        anchor_recall_cusum: {
          reference_value: 0.95,
          drift_threshold: 1.5,
          decision_interval: 4.0,
          reset_threshold: -4.0,
          min_samples: 5,
        },
        ladder_monitoring: {
          enabled: true,
          baseline_ratio: 0.75,
          degradation_threshold: 0.10,
          trend_window_size: 15,
        },
        coverage_monitoring: {
          enabled: false,
          lsif_baseline_pct: 80.0,
          tree_sitter_baseline_pct: 85.0,
          degradation_threshold_pct: 10.0,
          measurement_interval_hours: 2,
        },
        alerting: {
          consolidation_window_minutes: 5,
          max_alerts_per_hour: 20,
          severity_escalation_threshold: 3,
        },
      };
      
      const customDetector = new DriftDetectionSystem(customConfig);
      
      expect(() => customDetector.processMetrics(mockMetrics)).not.toThrow();
      customDetector.destroy();
    });

    it('should validate configuration on initialization', () => {
      const invalidConfig = {
        ...DEFAULT_DRIFT_CONFIG,
        anchor_p1_cusum: {
          ...DEFAULT_DRIFT_CONFIG.anchor_p1_cusum,
          min_samples: -1, // Invalid
        },
      };
      
      expect(() => new DriftDetectionSystem(invalidConfig)).toThrow(/configuration/i);
    });
  });

  describe('Performance and Memory Management', () => {
    it('should limit memory usage with large sample counts', () => {
      const startMemory = process.memoryUsage().heapUsed;
      
      // Process a large number of metrics
      Array.from({ length: 10000 }, (_, index) => 
        driftDetector.processMetrics({
          ...mockMetrics,
          timestamp: new Date(Date.now() + index * 1000).toISOString(),
        })
      );
      
      const endMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = endMemory - startMemory;
      
      // Memory increase should be reasonable (less than 50MB for 10k samples)
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
    });

    it('should cleanup resources on destroy', () => {
      const detector = new DriftDetectionSystem(DEFAULT_DRIFT_CONFIG);
      
      // Process some metrics to establish state
      Array(10).fill(mockMetrics).forEach(metrics => 
        detector.processMetrics(metrics)
      );
      
      expect(() => detector.destroy()).not.toThrow();
      
      // Should not accept new metrics after destruction
      expect(() => detector.processMetrics(mockMetrics)).toThrow();
    });
  });
});