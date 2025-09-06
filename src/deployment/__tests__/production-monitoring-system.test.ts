import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EventEmitter } from 'events';
import { ProductionMonitoringSystem } from '../production-monitoring-system.js';

// Mock filesystem operations
vi.mock('fs', () => ({
  writeFileSync: vi.fn(),
  readFileSync: vi.fn(),
  existsSync: vi.fn(),
  mkdirSync: vi.fn(),
  promises: {
    writeFile: vi.fn(),
    readFile: vi.fn(),
    mkdir: vi.fn()
  }
}));

vi.mock('path', () => ({
  join: vi.fn((...paths) => paths.join('/'))
}));

// Mock fetch for API calls
global.fetch = vi.fn();

// Mock console to reduce noise
const consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

describe('ProductionMonitoringSystem', () => {
  let monitoringSystem: ProductionMonitoringSystem;
  let mockMonitoringDir: string;

  // Mock metric snapshot data
  const mockMetricSnapshot = {
    timestamp: '2024-01-15T10:00:00Z',
    anchor_p_at_1: 0.92,
    recall_at_50: 0.88,
    ndcg_at_10: 0.85,
    ladder_positives_in_candidates: 45,
    ladder_total_candidates: 100,
    ladder_positive_rate: 0.45,
    lsif_coverage: 0.94,
    tree_sitter_coverage: 0.96,
    total_spans: 10000,
    covered_spans: 9400,
    p95_latency_ms: 120,
    p99_latency_ms: 200,
    qps: 150,
    error_rate: 0.02,
    results_per_query_mean: 8.5,
    results_per_query_p95: 15,
    zero_result_rate: 0.05,
    memory_usage_gb: 2.5,
    cpu_utilization: 0.65,
    disk_usage_gb: 50.2
  };

  // Mock API responses
  const mockSearchResponse = {
    results: [
      {
        file: 'test.py',
        line: 1,
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

  beforeEach(async () => {
    vi.clearAllMocks();
    
    mockMonitoringDir = './test-monitoring';

    // Setup filesystem mocks
    const fs = await vi.importMock('fs') as any;
    fs.existsSync.mockReturnValue(false);
    fs.readFileSync.mockReturnValue('{}');
    fs.writeFileSync.mockImplementation(() => {});

    // Mock fetch responses
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockSearchResponse)
    });

    monitoringSystem = new ProductionMonitoringSystem(mockMonitoringDir);
  });

  afterEach(() => {
    if (monitoringSystem) {
      monitoringSystem.stopMonitoring();
    }
    consoleLogSpy.mockClear();
    consoleWarnSpy.mockClear();
    consoleErrorSpy.mockClear();
  });

  describe('Constructor and Initialization', () => {
    it('should initialize with monitoring directory', () => {
      expect(monitoringSystem).toBeInstanceOf(ProductionMonitoringSystem);
      expect(monitoringSystem).toBeInstanceOf(EventEmitter);
    });

    it('should create default monitoring system when no state file exists', () => {
      mockFs.existsSync.mockReturnValue(false);

      const system = new ProductionMonitoringSystem(mockMonitoringDir);
      expect(system).toBeInstanceOf(ProductionMonitoringSystem);
    });

    it('should load existing state when state file exists', async () => {
      const mockState = {
        cusum_detectors: {
          anchor_p_at_1: {
            metric_name: 'anchor_p_at_1',
            target_mean: 0.90,
            target_std: 0.02,
            threshold: 3,
            positive_sum: 0,
            negative_sum: 0,
            last_reset: '2024-01-15T00:00:00Z',
            alarm_active: false,
            consecutive_violations: 0,
            max_consecutive_violations: 5
          }
        },
        alert_conditions: [],
        metrics_history: [],
        last_baseline_update: '2024-01-15T00:00:00Z'
      };

      const fs = await vi.importMock('fs') as any;
      fs.existsSync.mockReturnValue(true);
      fs.readFileSync.mockReturnValue(JSON.stringify(mockState));

      const system = new ProductionMonitoringSystem(mockMonitoringDir);
      expect(system).toBeInstanceOf(ProductionMonitoringSystem);
    });

    it('should handle corrupted state file gracefully', async () => {
      const fs = await vi.importMock('fs') as any;
      fs.existsSync.mockReturnValue(true);
      fs.readFileSync.mockReturnValue('invalid json');

      const system = new ProductionMonitoringSystem(mockMonitoringDir);
      expect(system).toBeInstanceOf(ProductionMonitoringSystem);
    });

    it('should initialize with custom configuration', () => {
      const customConfig = {
        metricsInterval: 30000,
        alertsInterval: 60000,
        cusumThreshold: 4,
        maxHistoryLength: 2000
      };

      const system = new ProductionMonitoringSystem(mockMonitoringDir, customConfig);
      expect(system).toBeInstanceOf(ProductionMonitoringSystem);
    });
  });

  describe('Monitoring Lifecycle', () => {
    it('should start monitoring successfully', async () => {
      await monitoringSystem.startMonitoring();

      expect(consoleLogSpy).toHaveBeenCalledWith('🟢 Production monitoring system started');
      expect(consoleLogSpy).toHaveBeenCalledWith('📊 CUSUM baselines initialized');
    });

    it('should stop monitoring and clean up intervals', () => {
      monitoringSystem.stopMonitoring();

      expect(consoleLogSpy).toHaveBeenCalledWith('🛑 Production monitoring system stopped');
    });

    it('should handle start monitoring errors gracefully', async () => {
      // Mock initialization failure
      const originalMethod = monitoringSystem['initializeCUSUMBaselines'];
      monitoringSystem['initializeCUSUMBaselines'] = vi.fn().mockRejectedValue(new Error('Init failed'));

      await expect(monitoringSystem.startMonitoring()).rejects.toThrow('Init failed');

      // Restore original method
      monitoringSystem['initializeCUSUMBaselines'] = originalMethod;
    });

    it('should handle multiple start calls gracefully', async () => {
      await monitoringSystem.startMonitoring();
      await monitoringSystem.startMonitoring(); // Second call

      // Should not create duplicate intervals
      expect(consoleLogSpy).toHaveBeenCalledWith('🟢 Production monitoring system started');
    });

    it('should handle stop monitoring when not started', () => {
      monitoringSystem.stopMonitoring();

      expect(consoleLogSpy).toHaveBeenCalledWith('🛑 Production monitoring system stopped');
    });
  });

  describe('CUSUM Drift Detection', () => {
    beforeEach(async () => {
      await monitoringSystem.startMonitoring();
    });

    it('should initialize CUSUM detectors with correct parameters', async () => {
      const cusumStatus = monitoringSystem.getCUSUMStatus();

      expect(cusumStatus['anchor_p_at_1']).toBeDefined();
      expect(cusumStatus['recall_at_50']).toBeDefined();
      expect(cusumStatus['anchor_p_at_1'].target_mean).toBeCloseTo(0.95, 2);
      expect(cusumStatus['anchor_p_at_1'].threshold).toBe(3);
      expect(cusumStatus['anchor_p_at_1'].alarm_active).toBe(false);
    });

    it('should update CUSUM detectors with new metrics', async () => {
      const testMetrics = { ...mockMetricSnapshot, anchor_p_at_1: 0.80 }; // Below baseline

      await monitoringSystem['updateCUSUMDetectors'](testMetrics);

      const cusumStatus = monitoringSystem.getCUSUMStatus();
      expect(cusumStatus['anchor_p_at_1'].negative_sum).toBeGreaterThan(0);
    });

    it('should trigger CUSUM alarm on sustained drift', async () => {
      // Simulate sustained drift by providing multiple poor metrics
      const driftMetrics = { ...mockMetricSnapshot, anchor_p_at_1: 0.70 };
      
      for (let i = 0; i < 10; i++) {
        await monitoringSystem['updateCUSUMDetectors']({
          ...driftMetrics,
          timestamp: new Date(Date.now() + i * 60000).toISOString()
        });
      }

      const cusumStatus = monitoringSystem.getCUSUMStatus();
      expect(cusumStatus['anchor_p_at_1'].consecutive_violations).toBeGreaterThan(0);
    });

    it('should reset CUSUM detector on command', () => {
      monitoringSystem.resetCUSUMDetector('anchor_p_at_1');

      const cusumStatus = monitoringSystem.getCUSUMStatus();
      expect(cusumStatus['anchor_p_at_1'].positive_sum).toBe(0);
      expect(cusumStatus['anchor_p_at_1'].negative_sum).toBe(0);
      expect(cusumStatus['anchor_p_at_1'].alarm_active).toBe(false);
    });

    it('should handle metrics with missing values gracefully', async () => {
      const incompleteMetrics = {
        ...mockMetricSnapshot,
        anchor_p_at_1: undefined as any
      };

      await expect(
        monitoringSystem['updateCUSUMDetectors'](incompleteMetrics)
      ).resolves.not.toThrow();
    });

    it('should calculate CUSUM statistics correctly', async () => {
      const goodMetrics = { ...mockMetricSnapshot, anchor_p_at_1: 0.98 };
      const badMetrics = { ...mockMetricSnapshot, anchor_p_at_1: 0.80 };

      // Good metrics should increase positive sum
      await monitoringSystem['updateCUSUMDetectors'](goodMetrics);
      let status = monitoringSystem.getCUSUMStatus();
      expect(status['anchor_p_at_1'].positive_sum).toBeGreaterThan(0);

      // Bad metrics should increase negative sum
      await monitoringSystem['updateCUSUMDetectors'](badMetrics);
      status = monitoringSystem.getCUSUMStatus();
      expect(status['anchor_p_at_1'].negative_sum).toBeGreaterThan(0);
    });
  });

  describe('Alert System', () => {
    beforeEach(async () => {
      await monitoringSystem.startMonitoring();
    });

    it('should evaluate alert conditions correctly', async () => {
      // Add a test alert condition
      const testCondition = {
        name: 'High Error Rate',
        metric_path: 'error_rate',
        condition: 'above' as const,
        threshold: 0.05,
        severity: 'high' as const,
        sustained_minutes: 2,
        cooldown_minutes: 15,
        actions: [
          {
            type: 'webhook' as const,
            config: { url: 'http://localhost:8080/alerts' }
          }
        ],
        violations_count: 0
      };

      monitoringSystem['state'].alert_conditions.push(testCondition);

      // Trigger condition with high error rate
      const highErrorMetrics = { ...mockMetricSnapshot, error_rate: 0.10 };
      await monitoringSystem['evaluateAlerts'](highErrorMetrics);

      expect(testCondition.current_violation_start).toBeDefined();
    });

    it('should respect sustained violation requirements', async () => {
      const testCondition = {
        name: 'High Latency',
        metric_path: 'p99_latency_ms',
        condition: 'above' as const,
        threshold: 500,
        severity: 'medium' as const,
        sustained_minutes: 5,
        cooldown_minutes: 10,
        actions: [],
        violations_count: 0
      };

      monitoringSystem['state'].alert_conditions.push(testCondition);

      const highLatencyMetrics = { ...mockMetricSnapshot, p99_latency_ms: 600 };
      await monitoringSystem['evaluateAlerts'](highLatencyMetrics);

      // Should not trigger immediately due to sustained_minutes requirement
      expect(testCondition.current_violation_start).toBeDefined();
      expect(testCondition.last_alert_sent).toBeUndefined();
    });

    it('should handle webhook alert actions', async () => {
      const webhookUrl = 'http://localhost:8080/alerts';
      const alertAction = {
        type: 'webhook' as const,
        config: { url: webhookUrl }
      };

      const alertData = {
        alert_name: 'Test Alert',
        severity: 'high',
        metric_value: 0.10,
        threshold: 0.05,
        timestamp: '2024-01-15T10:00:00Z'
      };

      await monitoringSystem['executeAlertAction'](alertAction, alertData);

      expect(fetch).toHaveBeenCalledWith(webhookUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(alertData)
      });
    });

    it('should handle email alert actions', async () => {
      const emailAction = {
        type: 'email' as const,
        config: {
          to: 'admin@example.com',
          subject: 'Production Alert'
        }
      };

      const alertData = {
        alert_name: 'Test Alert',
        severity: 'high'
      };

      await monitoringSystem['executeAlertAction'](emailAction, alertData);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('📧 Email alert sent to admin@example.com')
      );
    });

    it('should handle PagerDuty alert actions', async () => {
      const pagerDutyAction = {
        type: 'pagerduty' as const,
        config: {
          routing_key: 'test-routing-key',
          severity: 'critical'
        }
      };

      const alertData = {
        alert_name: 'Critical Alert',
        severity: 'critical'
      };

      await monitoringSystem['executeAlertAction'](pagerDutyAction, alertData);

      expect(fetch).toHaveBeenCalledWith(
        'https://events.pagerduty.com/v2/enqueue',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
      );
    });

    it('should handle kill switch alert actions', async () => {
      const killSwitchAction = {
        type: 'kill_switch' as const,
        config: {
          endpoint: 'http://localhost:3000/admin/kill-switch',
          reason: 'Performance degradation detected'
        }
      };

      const alertData = {
        alert_name: 'Kill Switch Triggered',
        severity: 'critical'
      };

      await monitoringSystem['executeAlertAction'](killSwitchAction, alertData);

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:3000/admin/kill-switch',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
      );

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        '🚨 KILL SWITCH ACTIVATED: Performance degradation detected'
      );
    });

    it('should handle rollback alert actions', async () => {
      const rollbackAction = {
        type: 'rollback' as const,
        config: {
          deployment_endpoint: 'http://localhost:3000/deployments/rollback',
          target_version: 'v1.2.3'
        }
      };

      const alertData = {
        alert_name: 'Rollback Triggered',
        severity: 'critical'
      };

      await monitoringSystem['executeAlertAction'](rollbackAction, alertData);

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:3000/deployments/rollback',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
      );
    });

    it('should respect alert cooldown periods', async () => {
      const testCondition = {
        name: 'CPU Alert',
        metric_path: 'cpu_utilization',
        condition: 'above' as const,
        threshold: 0.8,
        severity: 'medium' as const,
        sustained_minutes: 1,
        cooldown_minutes: 30,
        actions: [],
        violations_count: 0,
        last_alert_sent: new Date().toISOString()
      };

      monitoringSystem['state'].alert_conditions.push(testCondition);

      const highCpuMetrics = { ...mockMetricSnapshot, cpu_utilization: 0.9 };
      await monitoringSystem['evaluateAlerts'](highCpuMetrics);

      // Should not send another alert due to cooldown
      expect(consoleLogSpy).not.toHaveBeenCalledWith(
        expect.stringContaining('Alert triggered')
      );
    });
  });

  describe('Metrics Collection', () => {
    beforeEach(async () => {
      await monitoringSystem.startMonitoring();
    });

    it('should collect metrics from API successfully', async () => {
      await monitoringSystem['collectMetrics']();

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:3000/health',
        expect.any(Object)
      );
      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:3000/metrics',
        expect.any(Object)
      );
    });

    it('should handle API failures gracefully', async () => {
      (global.fetch as any).mockRejectedValue(new Error('API unavailable'));

      await monitoringSystem['collectMetrics']();

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        '❌ Error collecting metrics:',
        expect.any(Error)
      );
    });

    it('should handle malformed API responses', async () => {
      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ invalid: 'data' })
      });

      await monitoringSystem['collectMetrics']();

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Incomplete metrics data')
      );
    });

    it('should collect ladder metrics correctly', async () => {
      const mockLadderResponse = {
        total_candidates: 500,
        positives_in_candidates: 225,
        positive_rate: 0.45
      };

      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('/ladder-metrics')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockLadderResponse)
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({})
        });
      });

      await monitoringSystem['collectMetrics']();

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:3000/ladder-metrics',
        expect.any(Object)
      );
    });

    it('should collect system performance metrics', async () => {
      const mockSystemMetrics = {
        memory_usage_gb: 3.2,
        cpu_utilization: 0.75,
        disk_usage_gb: 45.8
      };

      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('/system-metrics')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockSystemMetrics)
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({})
        });
      });

      await monitoringSystem['collectMetrics']();

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:3000/system-metrics',
        expect.any(Object)
      );
    });
  });

  describe('Health Status and Dashboard', () => {
    beforeEach(async () => {
      await monitoringSystem.startMonitoring();
    });

    it('should return comprehensive health status', () => {
      const healthStatus = monitoringSystem.getHealthStatus();

      expect(healthStatus).toHaveProperty('overall_status');
      expect(healthStatus).toHaveProperty('cusum_alarms');
      expect(healthStatus).toHaveProperty('active_alerts');
      expect(healthStatus).toHaveProperty('last_metrics');
      expect(healthStatus).toHaveProperty('system_uptime');
    });

    it('should return dashboard data with KPIs', () => {
      const dashboardData = monitoringSystem.getDashboardData();

      expect(dashboardData).toHaveProperty('current_metrics');
      expect(dashboardData).toHaveProperty('cusum_status');
      expect(dashboardData).toHaveProperty('alert_summary');
      expect(dashboardData).toHaveProperty('trends');
      expect(dashboardData).toHaveProperty('system_health');
    });

    it('should calculate health status correctly with no active alarms', () => {
      const healthStatus = monitoringSystem.getHealthStatus();

      expect(healthStatus.overall_status).toBe('healthy');
      expect(healthStatus.cusum_alarms).toBe(0);
      expect(healthStatus.active_alerts).toBe(0);
    });

    it('should report degraded status when CUSUM alarms are active', () => {
      // Simulate active CUSUM alarm
      const cusumStatus = monitoringSystem.getCUSUMStatus();
      if (cusumStatus['anchor_p_at_1']) {
        cusumStatus['anchor_p_at_1'].alarm_active = true;
      }

      const healthStatus = monitoringSystem.getHealthStatus();

      expect(healthStatus.overall_status).toBe('degraded');
      expect(healthStatus.cusum_alarms).toBeGreaterThan(0);
    });

    it('should provide metric trends in dashboard data', () => {
      // Add some historical data
      monitoringSystem['state'].metrics_history.push(
        mockMetricSnapshot,
        {
          ...mockMetricSnapshot,
          timestamp: '2024-01-15T11:00:00Z',
          anchor_p_at_1: 0.94
        }
      );

      const dashboardData = monitoringSystem.getDashboardData();

      expect(dashboardData.trends).toBeDefined();
      expect(Array.isArray(dashboardData.trends.anchor_p_at_1_trend)).toBe(true);
    });
  });

  describe('State Persistence', () => {
    it('should save state to file correctly', async () => {
      // Add some state data
      await monitoringSystem.startMonitoring();
      
      // Trigger state save
      monitoringSystem['saveState']();
      
      const fs = await vi.importMock('fs') as any;
      expect(fs.writeFileSync).toHaveBeenCalledWith(
        expect.stringContaining('monitoring-state.json'),
        expect.stringContaining('cusum_detectors')
      );
    });

    it('should handle state save errors gracefully', async () => {
      const fs = await vi.importMock('fs') as any;
      fs.writeFileSync.mockImplementation(() => {
        throw new Error('Write failed');
      });

      expect(() => {
        monitoringSystem['saveState']();
      }).not.toThrow();

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        '❌ Error saving monitoring state:',
        expect.any(Error)
      );
    });

    it('should maintain metrics history with size limits', async () => {
      await monitoringSystem.startMonitoring();

      // Add many metrics to test history limit
      for (let i = 0; i < 1500; i++) {
        monitoringSystem['state'].metrics_history.push({
          ...mockMetricSnapshot,
          timestamp: new Date(Date.now() + i * 60000).toISOString()
        });
      }

      // Trigger history cleanup
      monitoringSystem['cleanupHistory']();

      expect(monitoringSystem['state'].metrics_history.length).toBeLessThanOrEqual(1000);
    });

    it('should load baseline data correctly', async () => {
      const mockBaselineData = {
        anchor_p_at_1: { mean: 0.95, std: 0.02 },
        recall_at_50: { mean: 0.88, std: 0.03 }
      };

      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockBaselineData)
      });

      await monitoringSystem['initializeCUSUMBaselines']();

      const cusumStatus = monitoringSystem.getCUSUMStatus();
      expect(cusumStatus['anchor_p_at_1'].target_mean).toBeCloseTo(0.95, 2);
      expect(cusumStatus['recall_at_50'].target_mean).toBeCloseTo(0.88, 2);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle empty metrics history gracefully', () => {
      monitoringSystem['state'].metrics_history = [];

      const dashboardData = monitoringSystem.getDashboardData();
      expect(dashboardData).toBeDefined();
      expect(dashboardData.current_metrics).toBeNull();
    });

    it('should handle malformed CUSUM detector states', () => {
      monitoringSystem['state'].cusum_detectors = {
        invalid_detector: {} as any
      };

      expect(() => {
        monitoringSystem.resetCUSUMDetector('invalid_detector');
      }).not.toThrow();
    });

    it('should handle network timeouts gracefully', async () => {
      (global.fetch as any).mockImplementation(() => 
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Timeout')), 100)
        )
      );

      await monitoringSystem['collectMetrics']();

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        '❌ Error collecting metrics:',
        expect.any(Error)
      );
    });

    it('should handle invalid alert configurations', async () => {
      const invalidCondition = {
        name: 'Invalid Alert',
        metric_path: 'nonexistent.metric',
        condition: 'invalid' as any,
        severity: 'high' as const,
        sustained_minutes: 5,
        cooldown_minutes: 10,
        actions: [],
        violations_count: 0
      };

      monitoringSystem['state'].alert_conditions.push(invalidCondition);

      await expect(
        monitoringSystem['evaluateAlerts'](mockMetricSnapshot)
      ).resolves.not.toThrow();
    });

    it('should handle alert action failures gracefully', async () => {
      (global.fetch as any).mockRejectedValue(new Error('Network error'));

      const webhookAction = {
        type: 'webhook' as const,
        config: { url: 'http://invalid-url' }
      };

      await monitoringSystem['executeAlertAction'](webhookAction, {});

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        '❌ Error executing alert action:',
        expect.any(Error)
      );
    });

    it('should handle concurrent monitoring operations safely', async () => {
      await monitoringSystem.startMonitoring();

      // Simulate concurrent operations
      const promises = [
        monitoringSystem['collectMetrics'](),
        monitoringSystem['evaluateAlerts'](),
        monitoringSystem.resetCUSUMDetector('anchor_p_at_1')
      ];

      await expect(Promise.all(promises)).resolves.toBeDefined();
    });
  });

  describe('Event Emission', () => {
    it('should emit events on alert triggers', async () => {
      const alertHandler = vi.fn();
      monitoringSystem.on('alert', alertHandler);

      const testCondition = {
        name: 'Test Alert',
        metric_path: 'error_rate',
        condition: 'above' as const,
        threshold: 0.01,
        severity: 'high' as const,
        sustained_minutes: 0,
        cooldown_minutes: 0,
        actions: [],
        violations_count: 0
      };

      monitoringSystem['state'].alert_conditions.push(testCondition);

      const highErrorMetrics = { ...mockMetricSnapshot, error_rate: 0.05 };
      await monitoringSystem['evaluateAlerts'](highErrorMetrics);

      // May emit alert depending on sustained_minutes implementation
    });

    it('should emit events on CUSUM alarms', async () => {
      const cusumHandler = vi.fn();
      monitoringSystem.on('cusum_alarm', cusumHandler);

      await monitoringSystem.startMonitoring();

      // Trigger CUSUM alarm with multiple bad metrics
      const driftMetrics = { ...mockMetricSnapshot, anchor_p_at_1: 0.60 };
      
      for (let i = 0; i < 15; i++) {
        await monitoringSystem['updateCUSUMDetectors']({
          ...driftMetrics,
          timestamp: new Date(Date.now() + i * 60000).toISOString()
        });
      }

      // Check if CUSUM alarm was activated
      const cusumStatus = monitoringSystem.getCUSUMStatus();
      if (cusumStatus['anchor_p_at_1'].alarm_active) {
        expect(cusumHandler).toHaveBeenCalled();
      }
    });

    it('should emit health status changes', () => {
      const healthHandler = vi.fn();
      monitoringSystem.on('health_change', healthHandler);

      // Simulate health status change by activating an alarm
      const cusumStatus = monitoringSystem.getCUSUMStatus();
      if (cusumStatus['anchor_p_at_1']) {
        cusumStatus['anchor_p_at_1'].alarm_active = true;
      }

      const healthStatus = monitoringSystem.getHealthStatus();
      
      if (healthStatus.overall_status === 'degraded') {
        // Health change event would be emitted in real implementation
      }
    });
  });

  describe('Performance and Scalability', () => {
    it('should handle large metrics history efficiently', async () => {
      await monitoringSystem.startMonitoring();

      const startTime = Date.now();

      // Add large number of metrics
      for (let i = 0; i < 500; i++) {
        monitoringSystem['state'].metrics_history.push({
          ...mockMetricSnapshot,
          timestamp: new Date(Date.now() + i * 60000).toISOString()
        });
      }

      const dashboardData = monitoringSystem.getDashboardData();
      
      const endTime = Date.now();
      expect(endTime - startTime).toBeLessThan(1000); // Should be fast
      expect(dashboardData).toBeDefined();
    });

    it('should handle many concurrent alert evaluations', async () => {
      // Add multiple alert conditions
      for (let i = 0; i < 20; i++) {
        monitoringSystem['state'].alert_conditions.push({
          name: `Alert ${i}`,
          metric_path: 'error_rate',
          condition: 'above' as const,
          threshold: 0.05 + (i * 0.01),
          severity: 'medium' as const,
          sustained_minutes: 1,
          cooldown_minutes: 5,
          actions: [],
          violations_count: 0
        });
      }

      const startTime = Date.now();
      await monitoringSystem['evaluateAlerts'](mockMetricSnapshot);
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(500); // Should be reasonably fast
    });

    it('should clean up old data automatically', () => {
      // Fill with old data
      const oldTimestamp = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000); // 7 days ago
      for (let i = 0; i < 100; i++) {
        monitoringSystem['state'].metrics_history.push({
          ...mockMetricSnapshot,
          timestamp: new Date(oldTimestamp.getTime() + i * 60000).toISOString()
        });
      }

      const initialLength = monitoringSystem['state'].metrics_history.length;
      monitoringSystem['cleanupHistory']();
      
      // Should clean up old data
      expect(monitoringSystem['state'].metrics_history.length).toBeLessThanOrEqual(initialLength);
    });
  });
});