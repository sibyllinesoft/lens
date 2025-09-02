#!/usr/bin/env node

/**
 * Performance Regression Analysis Tool
 * 
 * Analyzes benchmark results to detect statistically significant performance regressions
 * using configurable thresholds and historical trend analysis.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

/**
 * Regression analysis configuration
 */
const REGRESSION_CONFIG = {
  // Statistical thresholds for regression detection
  THRESHOLDS: {
    recall_at_10: {
      warning: 0.05,   // 5% drop
      critical: 0.10   // 10% drop
    },
    recall_at_50: {
      warning: 0.05,   // 5% drop
      critical: 0.10   // 10% drop
    },
    ndcg_at_10: {
      warning: 0.05,   // 5% drop
      critical: 0.10   // 10% drop
    },
    e2e_p95_latency: {
      warning: 0.15,   // 15% increase
      critical: 0.25   // 25% increase
    },
    error_rate: {
      warning: 0.02,   // 2% increase (absolute)
      critical: 0.05   // 5% increase (absolute)
    }
  },
  
  // Analysis parameters
  MIN_HISTORICAL_SAMPLES: 3,
  OUTLIER_DETECTION_THRESHOLD: 2.0,  // Standard deviations
  TREND_ANALYSIS_WINDOW: 10,          // Last 10 runs
  CONFIDENCE_LEVEL: 0.95              // 95% confidence for statistical tests
};

/**
 * Performance regression analyzer
 */
class RegressionAnalyzer {
  constructor(options = {}) {
    this.currentRunDir = options.currentRun;
    this.historyDir = options.historyDir;
    this.threshold = options.threshold || 0.10;
    this.outputFormat = options.outputFormat || 'json';
    this.verbose = options.verbose || false;
    
    this.currentMetrics = null;
    this.historicalData = [];
    this.regressions = [];
    this.warnings = [];
  }
  
  /**
   * Load current run metrics
   */
  async loadCurrentMetrics() {
    console.log(`📂 Loading current run metrics from ${this.currentRunDir}...`);
    
    const summaryPath = path.join(this.currentRunDir, 'summary.json');
    
    try {
      const summary = await this.loadJsonFile(summaryPath);
      
      this.currentMetrics = {
        run_id: summary.run_id,
        timestamp: summary.timestamp,
        recall_at_10: summary.performance?.recall_at_10,
        recall_at_50: summary.performance?.recall_at_50,
        ndcg_at_10: summary.performance?.ndcg_at_10,
        e2e_p95_latency: summary.latency?.e2e_p95,
        error_rate: summary.quality?.error_rate
      };
      
      console.log(`✅ Current metrics loaded (Run: ${this.currentMetrics.run_id})`);
    } catch (error) {
      throw new Error(`Failed to load current metrics: ${error.message}`);
    }
  }
  
  /**
   * Load historical benchmark data for comparison
   */
  async loadHistoricalData() {
    console.log(`📈 Loading historical data from ${this.historyDir}...`);
    
    try {
      const entries = await fs.readdir(this.historyDir, { withFileTypes: true });
      const runDirs = entries
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name)
        .sort()
        .reverse() // Most recent first
        .slice(0, REGRESSION_CONFIG.TREND_ANALYSIS_WINDOW);
      
      for (const runDir of runDirs) {
        try {
          const summaryPath = path.join(this.historyDir, runDir, 'summary.json');
          const summary = await this.loadJsonFile(summaryPath);
          
          const metrics = {
            run_id: runDir,
            timestamp: summary.timestamp,
            recall_at_10: summary.performance?.recall_at_10,
            recall_at_50: summary.performance?.recall_at_50,
            ndcg_at_10: summary.performance?.ndcg_at_10,
            e2e_p95_latency: summary.latency?.e2e_p95,
            error_rate: summary.quality?.error_rate
          };
          
          // Only include runs with valid metrics
          if (this.hasValidMetrics(metrics)) {
            this.historicalData.push(metrics);
          }
        } catch (error) {
          if (this.verbose) {
            console.warn(`⚠️ Could not load historical run ${runDir}: ${error.message}`);
          }
        }
      }
      
      console.log(`✅ Loaded ${this.historicalData.length} historical runs`);
      
      if (this.historicalData.length < REGRESSION_CONFIG.MIN_HISTORICAL_SAMPLES) {
        console.warn(`⚠️ Insufficient historical data (${this.historicalData.length} < ${REGRESSION_CONFIG.MIN_HISTORICAL_SAMPLES})`);
      }
      
    } catch (error) {
      console.warn(`⚠️ Could not load historical data: ${error.message}`);
    }
  }
  
  /**
   * Check if metrics object has sufficient data for analysis
   */
  hasValidMetrics(metrics) {
    return metrics.recall_at_10 !== null && 
           metrics.recall_at_10 !== undefined &&
           metrics.ndcg_at_10 !== null &&
           metrics.ndcg_at_10 !== undefined;
  }
  
  /**
   * Perform comprehensive regression analysis
   */
  analyzeRegressions() {
    console.log('🗺 Analyzing performance regressions...');
    
    if (this.historicalData.length < REGRESSION_CONFIG.MIN_HISTORICAL_SAMPLES) {
      console.warn('⚠️ Skipping regression analysis - insufficient historical data');
      return {
        status: 'INSUFFICIENT_DATA',
        message: `Need at least ${REGRESSION_CONFIG.MIN_HISTORICAL_SAMPLES} historical samples`,
        historical_samples: this.historicalData.length
      };
    }
    
    const metrics = ['recall_at_10', 'recall_at_50', 'ndcg_at_10', 'e2e_p95_latency', 'error_rate'];
    
    for (const metric of metrics) {
      this.analyzeMetricRegression(metric);
    }
    
    // Determine overall status
    let overallStatus = 'HEALTHY';
    let statusMessage = 'No significant regressions detected';
    
    if (this.regressions.length > 0) {
      const criticalCount = this.regressions.filter(r => r.severity === 'critical').length;
      if (criticalCount > 0) {
        overallStatus = 'CRITICAL_REGRESSION';
        statusMessage = `${criticalCount} critical regression(s) detected`;
      } else {
        overallStatus = 'WARNING_REGRESSION';
        statusMessage = `${this.regressions.length} warning-level regression(s) detected`;
      }
    } else if (this.warnings.length > 0) {
      overallStatus = 'MINOR_CONCERNS';
      statusMessage = `${this.warnings.length} minor concern(s) detected`;
    }
    
    console.log(`✅ Regression analysis complete - Status: ${overallStatus}`);
    
    return {
      status: overallStatus,
      message: statusMessage,
      regressions: this.regressions,
      warnings: this.warnings,
      analysis_metadata: {
        current_run: this.currentMetrics.run_id,
        historical_samples: this.historicalData.length,
        analysis_timestamp: new Date().toISOString()
      }
    };
  }
  
  /**
   * Analyze regression for a specific metric
   */
  analyzeMetricRegression(metric) {
    const currentValue = this.currentMetrics[metric];
    if (currentValue === null || currentValue === undefined) {
      return; // Skip metrics without current data
    }
    
    const historicalValues = this.historicalData
      .map(run => run[metric])
      .filter(val => val !== null && val !== undefined);
    
    if (historicalValues.length === 0) {
      return; // Skip if no historical data for this metric
    }
    
    // Calculate statistical measures
    const stats = this.calculateStatistics(historicalValues);
    const isLatencyMetric = metric.includes('latency');
    const isErrorMetric = metric.includes('error');
    
    // Detect regression based on metric type
    let regressionDetected = false;
    let changeFromMean, changeFromMedian;
    
    if (isLatencyMetric || isErrorMetric) {
      // For latency and error rates, increases are bad
      changeFromMean = (currentValue - stats.mean) / stats.mean;
      changeFromMedian = (currentValue - stats.median) / stats.median;
      
      if (isErrorMetric) {
        // For error rates, use absolute thresholds
        const absoluteChange = currentValue - stats.mean;
        regressionDetected = absoluteChange > REGRESSION_CONFIG.THRESHOLDS[metric]?.warning || 0.02;
      } else {
        regressionDetected = changeFromMean > REGRESSION_CONFIG.THRESHOLDS[metric]?.warning || 0.15;
      }
    } else {
      // For quality metrics (recall, NDCG), decreases are bad
      changeFromMean = (stats.mean - currentValue) / stats.mean;
      changeFromMedian = (stats.median - currentValue) / stats.median;
      
      regressionDetected = changeFromMean > REGRESSION_CONFIG.THRESHOLDS[metric]?.warning || 0.05;
    }
    
    // Determine severity
    const criticalThreshold = REGRESSION_CONFIG.THRESHOLDS[metric]?.critical || 0.10;
    const warningThreshold = REGRESSION_CONFIG.THRESHOLDS[metric]?.warning || 0.05;
    
    let severity = 'none';
    if (regressionDetected) {
      if (Math.abs(changeFromMean) > criticalThreshold) {
        severity = 'critical';
      } else if (Math.abs(changeFromMean) > warningThreshold) {
        severity = 'warning';
      }
    }
    
    // Check for outliers
    const isOutlier = Math.abs(currentValue - stats.mean) > REGRESSION_CONFIG.OUTLIER_DETECTION_THRESHOLD * stats.stdDev;
    
    // Record findings
    const finding = {
      metric,
      current_value: currentValue,
      historical_stats: stats,
      change_from_mean: changeFromMean,
      change_from_median: changeFromMedian,
      is_outlier: isOutlier,
      severity,
      thresholds: REGRESSION_CONFIG.THRESHOLDS[metric]
    };
    
    if (severity === 'critical' || severity === 'warning') {
      this.regressions.push(finding);
    } else if (isOutlier) {
      this.warnings.push({
        ...finding,
        reason: 'Statistical outlier detected'
      });
    }
    
    if (this.verbose) {
      console.log(`   ${metric}: ${this.formatMetric(metric, currentValue)} (${changeFromMean > 0 ? '+' : ''}${(changeFromMean * 100).toFixed(1)}% from mean)`);
    }
  }
  
  /**
   * Calculate statistical measures for a dataset
   */
  calculateStatistics(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    
    const median = sorted.length % 2 === 0 ?
      (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2 :
      sorted[Math.floor(sorted.length / 2)];
    
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    return {
      mean,
      median,
      stdDev,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length,
      percentile_25: sorted[Math.floor(sorted.length * 0.25)],
      percentile_75: sorted[Math.floor(sorted.length * 0.75)]
    };
  }
  
  /**
   * Format output based on requested format
   */
  formatOutput(analysisResult) {
    switch (this.outputFormat) {
      case 'github-actions':
        return this.formatForGitHubActions(analysisResult);
      case 'slack':
        return this.formatForSlack(analysisResult);
      case 'json':
        return JSON.stringify(analysisResult, null, 2);
      case 'summary':
        return this.formatSummary(analysisResult);
      default:
        return JSON.stringify(analysisResult, null, 2);
    }
  }
  
  /**
   * Format output for GitHub Actions
   */
  formatForGitHubActions(result) {
    if (result.status === 'INSUFFICIENT_DATA') {
      return 'INSUFFICIENT_DATA';
    }
    
    if (result.regressions.length === 0) {
      return 'HEALTHY';
    }
    
    const criticalCount = result.regressions.filter(r => r.severity === 'critical').length;
    if (criticalCount > 0) {
      return `CRITICAL_REGRESSION:${criticalCount}_critical,${result.regressions.length - criticalCount}_warning`;
    } else {
      return `WARNING_REGRESSION:${result.regressions.length}_warning`;
    }
  }
  
  /**
   * Format output for Slack notifications
   */
  formatForSlack(result) {
    if (result.status === 'INSUFFICIENT_DATA') {
      return '📉 Insufficient data for regression analysis';
    }
    
    if (result.regressions.length === 0) {
      return '✅ No performance regressions detected';
    }
    
    let message = `⚠️ ${result.regressions.length} regression(s) detected:\n`;
    
    for (const regression of result.regressions.slice(0, 3)) { // Limit to top 3
      const icon = regression.severity === 'critical' ? '🚨' : '⚠️';
      const change = (regression.change_from_mean * 100).toFixed(1);
      message += `${icon} ${this.formatMetricName(regression.metric)}: ${change}% change\n`;
    }
    
    if (result.regressions.length > 3) {
      message += `... and ${result.regressions.length - 3} more`;
    }
    
    return message;
  }
  
  /**
   * Format summary output
   */
  formatSummary(result) {
    let summary = `Regression Analysis Summary\n`;
    summary += `Status: ${result.status}\n`;
    summary += `Message: ${result.message}\n`;
    
    if (result.regressions.length > 0) {
      summary += `\nRegressions Detected:\n`;
      for (const regression of result.regressions) {
        const change = (regression.change_from_mean * 100).toFixed(1);
        summary += `  - ${regression.metric}: ${change}% (${regression.severity})\n`;
      }
    }
    
    if (result.warnings.length > 0) {
      summary += `\nWarnings:\n`;
      for (const warning of result.warnings) {
        summary += `  - ${warning.metric}: ${warning.reason}\n`;
      }
    }
    
    return summary;
  }
  
  /**
   * Format metric names for display
   */
  formatMetricName(metric) {
    const names = {
      'recall_at_10': 'Recall@10',
      'recall_at_50': 'Recall@50',
      'ndcg_at_10': 'NDCG@10',
      'e2e_p95_latency': 'E2E P95 Latency',
      'error_rate': 'Error Rate'
    };
    return names[metric] || metric;
  }
  
  /**
   * Format metric values for display
   */
  formatMetric(metric, value) {
    if (value === null || value === undefined) return 'N/A';
    
    if (metric.includes('recall') || metric.includes('ndcg') || metric.includes('error')) {
      return `${(value * 100).toFixed(1)}%`;
    } else if (metric.includes('latency')) {
      return `${value.toFixed(0)}ms`;
    } else {
      return value.toFixed(3);
    }
  }
  
  /**
   * Utility function to safely load JSON files
   */
  async loadJsonFile(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      throw new Error(`Failed to load ${filePath}: ${error.message}`);
    }
  }
  
  /**
   * Run complete regression analysis
   */
  async runAnalysis() {
    await this.loadCurrentMetrics();
    await this.loadHistoricalData();
    
    const result = this.analyzeRegressions();
    
    // Save detailed analysis results
    const outputPath = path.join(this.currentRunDir, 'regression-analysis.json');
    await fs.writeFile(outputPath, JSON.stringify(result, null, 2));
    console.log(`✅ Detailed analysis saved to ${outputPath}`);
    
    return result;
  }
}

/**
 * Command-line interface
 */
async function main() {
  const args = process.argv.slice(2);
  
  function getArg(name, defaultValue = null) {
    const index = args.findIndex(arg => arg === `--${name}`);
    return index !== -1 && index + 1 < args.length ? args[index + 1] : defaultValue;
  }
  
  function hasFlag(name) {
    return args.includes(`--${name}`);
  }
  
  try {
    const analyzer = new RegressionAnalyzer({
      currentRun: getArg('current-run'),
      historyDir: getArg('history-dir'),
      threshold: parseFloat(getArg('threshold', '0.10')),
      outputFormat: getArg('output-format', 'json'),
      verbose: hasFlag('verbose')
    });
    
    if (!analyzer.currentRunDir) {
      console.error('❌ Missing required argument: --current-run');
      process.exit(1);
    }
    
    const result = await analyzer.runAnalysis();
    const formattedOutput = analyzer.formatOutput(result);
    
    // Output result (this is what GitHub Actions will capture)
    console.log(formattedOutput);
    
    // Exit with appropriate code for CI/CD integration
    if (result.status === 'CRITICAL_REGRESSION') {
      process.exit(2); // Critical regressions
    } else if (result.status === 'WARNING_REGRESSION') {
      process.exit(1); // Warning regressions
    } else {
      process.exit(0); // All good
    }
    
  } catch (error) {
    console.error(`❌ Regression analysis failed: ${error.message}`);
    if (hasFlag('verbose')) {
      console.error('Stack trace:', error.stack);
    }
    process.exit(1);
  }
}

// Execute main function if this script is run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  });
}

export { RegressionAnalyzer, REGRESSION_CONFIG };
