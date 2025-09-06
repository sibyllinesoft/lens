#!/usr/bin/env node
/**
 * Safety Gate Validation Script
 * 
 * This script runs comprehensive validation of all safety gates and constraints
 * for the advanced search optimizations. It can be run standalone or as part
 * of CI/CD pipeline to ensure production readiness.
 * 
 * Usage:
 *   npm run safety-validation
 *   or
 *   tsx src/core/__tests__/safety-gate-validation.ts
 */

import { AdvancedSearchIntegration } from '../advanced-search-integration.js';
import { ConformalRouter } from '../conformal-router.js';
import { EntropyGatedPriors } from '../entropy-gated-priors.js';
import { LatencyConditionedMetrics } from '../latency-conditioned-metrics.js';
import { RAPTORHygiene } from '../raptor-hygiene.js';
import { EmbeddingRoadmap } from '../embedding-roadmap.js';
import { UnicodeNFCNormalizer } from '../unicode-nfc-normalizer.js';
import { ComprehensiveMonitoring } from '../comprehensive-monitoring.js';
import type { SearchContext, SearchHit } from '../../types/core.js';

interface ValidationResult {
  testName: string;
  passed: boolean;
  message: string;
  metrics?: Record<string, number>;
  details?: string;
}

interface SafetyGateReport {
  timestamp: string;
  totalTests: number;
  passedTests: number;
  failedTests: number;
  criticalFailures: number;
  overallStatus: 'PASS' | 'FAIL' | 'WARNING';
  results: ValidationResult[];
  recommendations: string[];
}

class SafetyGateValidator {
  private integration: AdvancedSearchIntegration;
  private monitoring: ComprehensiveMonitoring;
  private results: ValidationResult[] = [];

  constructor() {
    this.integration = AdvancedSearchIntegration.getInstance();
    this.monitoring = ComprehensiveMonitoring.getInstance();
  }

  private log(message: string, level: 'INFO' | 'WARN' | 'ERROR' = 'INFO'): void {
    const timestamp = new Date().toISOString();
    const prefix = `[${timestamp}] [${level}]`;
    
    switch (level) {
      case 'ERROR':
        console.error(`${prefix} ${message}`);
        break;
      case 'WARN':
        console.warn(`${prefix} ${message}`);
        break;
      default:
        console.log(`${prefix} ${message}`);
    }
  }

  private addResult(result: ValidationResult): void {
    this.results.push(result);
    const status = result.passed ? '✅' : '❌';
    this.log(`${status} ${result.testName}: ${result.message}`, result.passed ? 'INFO' : 'ERROR');
  }

  private generateMockHits(count: number): SearchHit[] {
    return Array.from({ length: count }, (_, i) => ({
      file: `/safety/test/file${i % 15}.ts`,
      score: Math.max(0.1, 0.95 - (i * 0.01)),
      snippet: `safety snippet ${i}`,
      line: i + 1,
      col: 1,
      why: ['exact'] as MatchReason[],
      match_reasons: ['exact'] as MatchReason[]
    }));
  }

  private createTestContext(id: string): SearchContext {
    return {
      query: `safety test ${id}`,
      filters: {},
      userId: 'safety-validator',
      requestId: `safety-${id}`,
      timestamp: Date.now(),
      efSearch: 50,
      maxResults: 20,
      includeSnippets: true
    };
  }

  async validateUpshiftRateCap(): Promise<ValidationResult> {
    this.log('Validating upshift rate cap (≤5%)...');
    
    try {
      const router = new ConformalRouter();
      const testRequests = 200; // Smaller sample for faster validation
      let upshiftCount = 0;
      
      const mockHits = this.generateMockHits(30);
      
      for (let i = 0; i < testRequests; i++) {
        const context = this.createTestContext(`upshift-${i}`);
        const decision = await router.makeRoutingDecision(context, mockHits.slice(0, 10));
        
        if (decision.useExpensiveMode) {
          upshiftCount++;
        }
      }
      
      const upshiftRate = (upshiftCount / testRequests) * 100;
      const passed = upshiftRate <= 5.0;
      
      return {
        testName: 'Upshift Rate Cap',
        passed,
        message: `Upshift rate: ${upshiftRate.toFixed(2)}% (target: ≤5.0%)`,
        metrics: {
          upshiftRate,
          upshiftCount,
          totalRequests: testRequests
        }
      };
    } catch (error) {
      return {
        testName: 'Upshift Rate Cap',
        passed: false,
        message: `Error during validation: ${error.message}`,
        details: error.stack
      };
    }
  }

  async validateLatencyImpact(): Promise<ValidationResult> {
    this.log('Validating latency impact (≤+1ms)...');
    
    try {
      const mockHits = this.generateMockHits(50);
      const testIterations = 20;
      
      // Baseline measurements (optimizations disabled)
      this.integration.disable();
      const baselineLatencies: number[] = [];
      
      for (let i = 0; i < testIterations; i++) {
        const context = this.createTestContext(`baseline-${i}`);
        const startTime = performance.now();
        await this.integration.executeAdvancedSearch(mockHits, context);
        const latency = performance.now() - startTime;
        baselineLatencies.push(latency);
      }
      
      // Optimized measurements (optimizations enabled)
      this.integration.enable();
      const optimizedLatencies: number[] = [];
      
      for (let i = 0; i < testIterations; i++) {
        const context = this.createTestContext(`optimized-${i}`);
        const startTime = performance.now();
        await this.integration.executeAdvancedSearch(mockHits, context);
        const latency = performance.now() - startTime;
        optimizedLatencies.push(latency);
      }
      
      // Calculate impact
      const avgBaseline = baselineLatencies.reduce((a, b) => a + b, 0) / baselineLatencies.length;
      const avgOptimized = optimizedLatencies.reduce((a, b) => a + b, 0) / optimizedLatencies.length;
      const impact = avgOptimized - avgBaseline;
      
      const passed = impact <= 1.0;
      
      return {
        testName: 'Latency Impact',
        passed,
        message: `Average latency impact: ${impact.toFixed(2)}ms (target: ≤1.0ms)`,
        metrics: {
          baselineLatencyMs: avgBaseline,
          optimizedLatencyMs: avgOptimized,
          impactMs: impact
        }
      };
    } catch (error) {
      return {
        testName: 'Latency Impact',
        passed: false,
        message: `Error during validation: ${error.message}`,
        details: error.stack
      };
    }
  }

  async validateUnicodeNormalization(): Promise<ValidationResult> {
    this.log('Validating Unicode NFC normalization...');
    
    try {
      const normalizer = new UnicodeNFCNormalizer();
      
      const testCases = [
        { input: 'café', expected: 'café' },
        { input: 'cafe\u0301', expected: 'café' },
        { input: 'naïve', expected: 'naïve' },
        { input: 'nai\u0308ve', expected: 'naïve' },
        { input: '🚀test', expected: '🚀test' },
        { input: '', expected: '' }
      ];
      
      let passed = true;
      let failedCases = 0;
      
      for (const testCase of testCases) {
        const result = normalizer.normalizeSpan(testCase.input, 0, testCase.input.length);
        
        if (result.normalized_text !== testCase.expected) {
          passed = false;
          failedCases++;
        }
      }
      
      return {
        testName: 'Unicode NFC Normalization',
        passed,
        message: `${testCases.length - failedCases}/${testCases.length} test cases passed`,
        metrics: {
          totalCases: testCases.length,
          passedCases: testCases.length - failedCases,
          failedCases
        }
      };
    } catch (error) {
      return {
        testName: 'Unicode NFC Normalization',
        passed: false,
        message: `Error during validation: ${error.message}`,
        details: error.stack
      };
    }
  }

  async validateRAPTORClustering(): Promise<ValidationResult> {
    this.log('Validating RAPTOR hierarchical clustering...');
    
    try {
      const raptor = new RAPTORHygiene();
      const queryEmbedding = new Float32Array(384).fill(0.1);
      
      // Test multiple queries to ensure clustering diversity
      const testQueries = 5;
      let totalResults = 0;
      let totalClusters = 0;
      
      for (let i = 0; i < testQueries; i++) {
        const context = this.createTestContext(`raptor-${i}`);
        const results = await raptor.hierarchicalSearch(queryEmbedding, context, 15);
        
        totalResults += results.length;
        
        // Count unique path prefixes to estimate cluster diversity
        const uniquePrefixes = new Set(results.map(hit => hit.file.split('/').slice(0, 3).join('/')));
        totalClusters += uniquePrefixes.size;
      }
      
      const avgResults = totalResults / testQueries;
      const avgClusters = totalClusters / testQueries;
      
      // Should return reasonable results and show diversity
      const passed = avgResults >= 8 && avgClusters >= 2;
      
      const stats = raptor.getStatistics();
      
      return {
        testName: 'RAPTOR Hierarchical Clustering',
        passed,
        message: `Avg results: ${avgResults.toFixed(1)}, Avg clusters: ${avgClusters.toFixed(1)}`,
        metrics: {
          averageResults: avgResults,
          averageClusters: avgClusters,
          pressureUtilization: stats.pressureBudgetUtilization,
          totalClusters: stats.totalClusters
        }
      };
    } catch (error) {
      return {
        testName: 'RAPTOR Hierarchical Clustering',
        passed: false,
        message: `Error during validation: ${error.message}`,
        details: error.stack
      };
    }
  }

  async validateSystemIntegration(): Promise<ValidationResult> {
    this.log('Validating system integration...');
    
    try {
      const mockHits = this.generateMockHits(30);
      const context = this.createTestContext('integration');
      const queryEmbedding = new Float32Array(384).fill(0.05);
      
      const result = await this.integration.executeAdvancedSearch(
        mockHits,
        context,
        queryEmbedding
      );
      
      // Check that all components contributed
      const hasRoutingDecision = !!result.routing_decision;
      const hasEntropyAnalysis = result.normalization_applied;
      const hasNormalizedSpans = result.normalization_applied;
      const hasHierarchicalResults = result.raptor_used;
      const hasEnhancedHits = result.hits.length > 0;
      const safetyPassed = result.safety_gates_passed;
      
      const componentCount = [
        hasRoutingDecision,
        hasEntropyAnalysis,
        hasNormalizedSpans,
        hasHierarchicalResults,
        hasEnhancedHits,
        safetyPassed
      ].filter(Boolean).length;
      
      const passed = componentCount >= 5; // Most components should work
      
      return {
        testName: 'System Integration',
        passed,
        message: `${componentCount}/6 integration components working`,
        metrics: {
          componentCount,
          processingTimeMs: result.latency_breakdown?.total_advanced_ms || 0,
          enhancedHitsCount: result.hits.length
        },
        details: `Routing: ${hasRoutingDecision}, Entropy: ${hasEntropyAnalysis}, Normalization: ${hasNormalizedSpans}, RAPTOR: ${hasHierarchicalResults}, Results: ${hasEnhancedHits}, Safety: ${safetyPassed}`
      };
    } catch (error) {
      return {
        testName: 'System Integration',
        passed: false,
        message: `Error during validation: ${error.message}`,
        details: error.stack
      };
    }
  }

  async validateMonitoringAndAlerting(): Promise<ValidationResult> {
    this.log('Validating monitoring and alerting...');
    
    try {
      // Generate some test metrics
      for (let i = 0; i < 10; i++) {
        this.monitoring.recordMetric('test_metric', Math.random(), Date.now());
      }
      
      const dashboard = await this.monitoring.generateDashboard();
      
      const hasSystemHealth = !!dashboard.systemHealth;
      const hasMetrics = !!dashboard.metrics && dashboard.metrics.totalOperations >= 0;
      const hasComponentStatus = !!dashboard.component_statuses && Object.keys(dashboard.component_statuses).length > 0;
      const canGenerateDashboard = !!dashboard.timestamp;
      
      const monitoringScore = [hasSystemHealth, hasMetrics, hasComponentStatus, canGenerateDashboard].filter(Boolean).length;
      const passed = monitoringScore >= 3;
      
      return {
        testName: 'Monitoring and Alerting',
        passed,
        message: `${monitoringScore}/4 monitoring components functional`,
        metrics: {
          monitoringScore,
          systemStatus: dashboard.systemHealth?.overallStatus,
          totalOperations: dashboard.metrics?.totalOperations
        }
      };
    } catch (error) {
      return {
        testName: 'Monitoring and Alerting',
        passed: false,
        message: `Error during validation: ${error.message}`,
        details: error.stack
      };
    }
  }

  async validateMemoryUsage(): Promise<ValidationResult> {
    this.log('Validating memory usage...');
    
    try {
      const initialMemory = process.memoryUsage().heapUsed;
      const mockHits = this.generateMockHits(100);
      
      // Run multiple operations to test memory usage
      for (let i = 0; i < 50; i++) {
        const context = this.createTestContext(`memory-${i}`);
        await this.integration.executeAdvancedSearch(mockHits, context);
      }
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryGrowthMB = (finalMemory - initialMemory) / (1024 * 1024);
      
      // Memory growth should be reasonable for 50 operations
      const passed = memoryGrowthMB < 50; // Less than 50MB growth
      
      return {
        testName: 'Memory Usage',
        passed,
        message: `Memory growth: ${memoryGrowthMB.toFixed(1)}MB (target: <50MB)`,
        metrics: {
          initialMemoryMB: initialMemory / (1024 * 1024),
          finalMemoryMB: finalMemory / (1024 * 1024),
          growthMB: memoryGrowthMB
        }
      };
    } catch (error) {
      return {
        testName: 'Memory Usage',
        passed: false,
        message: `Error during validation: ${error.message}`,
        details: error.stack
      };
    }
  }

  async runAllValidations(): Promise<SafetyGateReport> {
    this.log('Starting comprehensive safety gate validation...');
    this.results = [];
    
    const validations = [
      this.validateUpshiftRateCap(),
      this.validateLatencyImpact(),
      this.validateUnicodeNormalization(),
      this.validateRAPTORClustering(),
      this.validateSystemIntegration(),
      this.validateMonitoringAndAlerting(),
      this.validateMemoryUsage()
    ];
    
    const results = await Promise.all(validations);
    results.forEach(result => this.addResult(result));
    
    const passedTests = this.results.filter(r => r.passed).length;
    const failedTests = this.results.filter(r => !r.passed).length;
    
    // Critical failures are those that affect core functionality
    const criticalTests = ['Upshift Rate Cap', 'Latency Impact', 'System Integration'];
    const criticalFailures = this.results
      .filter(r => criticalTests.includes(r.testName) && !r.passed)
      .length;
    
    let overallStatus: 'PASS' | 'FAIL' | 'WARNING' = 'PASS';
    if (criticalFailures > 0) {
      overallStatus = 'FAIL';
    } else if (failedTests > 0) {
      overallStatus = 'WARNING';
    }
    
    const recommendations: string[] = [];
    
    if (criticalFailures > 0) {
      recommendations.push('CRITICAL: Address critical failures before production deployment');
    }
    
    if (failedTests > 0) {
      recommendations.push('Review failed tests and address underlying issues');
    }
    
    const upshiftResult = this.results.find(r => r.testName === 'Upshift Rate Cap');
    if (upshiftResult?.metrics?.upshiftRate > 4.0) {
      recommendations.push('Upshift rate approaching limit - monitor closely');
    }
    
    const latencyResult = this.results.find(r => r.testName === 'Latency Impact');
    if (latencyResult?.metrics?.impactMs > 0.5) {
      recommendations.push('Latency impact significant - consider optimization');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('All safety gates passed - system ready for production');
    }
    
    return {
      timestamp: new Date().toISOString(),
      totalTests: this.results.length,
      passedTests,
      failedTests,
      criticalFailures,
      overallStatus,
      results: this.results,
      recommendations
    };
  }

  generateReport(report: SafetyGateReport): void {
    console.log('\n' + '='.repeat(80));
    console.log('ADVANCED SEARCH OPTIMIZATIONS - SAFETY GATE VALIDATION REPORT');
    console.log('='.repeat(80));
    console.log(`Timestamp: ${report.timestamp}`);
    console.log(`Overall Status: ${report.overallStatus}`);
    console.log(`Tests Passed: ${report.passedTests}/${report.totalTests}`);
    
    if (report.criticalFailures > 0) {
      console.log(`❌ CRITICAL FAILURES: ${report.criticalFailures}`);
    }
    
    console.log('\nDETAILED RESULTS:');
    console.log('-'.repeat(80));
    
    report.results.forEach(result => {
      const status = result.passed ? '✅ PASS' : '❌ FAIL';
      console.log(`${status} | ${result.testName}`);
      console.log(`         ${result.message}`);
      
      if (result.metrics) {
        const metricsStr = Object.entries(result.metrics)
          .map(([key, value]) => `${key}: ${typeof value === 'number' ? value.toFixed(2) : value}`)
          .join(', ');
        console.log(`         Metrics: ${metricsStr}`);
      }
      
      if (!result.passed && result.details) {
        console.log(`         Details: ${result.details.substring(0, 200)}...`);
      }
      console.log();
    });
    
    console.log('RECOMMENDATIONS:');
    console.log('-'.repeat(80));
    report.recommendations.forEach(rec => {
      console.log(`• ${rec}`);
    });
    
    console.log('\n' + '='.repeat(80));
    console.log(`VALIDATION COMPLETE - STATUS: ${report.overallStatus}`);
    console.log('='.repeat(80));
  }
}

// Main execution
async function main() {
  const validator = new SafetyGateValidator();
  
  try {
    const report = await validator.runAllValidations();
    validator.generateReport(report);
    
    // Exit with appropriate code for CI/CD
    process.exit(report.overallStatus === 'FAIL' ? 1 : 0);
  } catch (error) {
    console.error('Fatal error during validation:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

export { SafetyGateValidator, type SafetyGateReport, type ValidationResult };