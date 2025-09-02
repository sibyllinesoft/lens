#!/usr/bin/env node

/**
 * Comprehensive Benchmark Report Generator
 * 
 * Generates detailed reports with trend analysis, performance comparisons,
 * and actionable insights for the nightly benchmark automation system.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

/**
 * Report configuration and templates
 */
const REPORT_CONFIG = {
  // Historical analysis window
  TREND_ANALYSIS_DAYS: 30,
  COMPARISON_RUNS: 7,  // Compare against last 7 runs
  
  // Performance thresholds
  PERFORMANCE_TARGETS: {
    recall_at_10: 0.70,     // 70% minimum
    recall_at_50: 0.85,     // 85% minimum
    ndcg_at_10: 0.65,       // 65% minimum
    e2e_p95_latency: 200,   // 200ms maximum
    error_rate: 0.05        // 5% maximum
  },
  
  // Regression detection
  REGRESSION_THRESHOLDS: {
    recall_at_10: -0.10,    // 10% drop
    latency_p95: 0.20,      // 20% increase
    error_rate: 0.05        // 5% increase
  }
};

/**
 * Comprehensive report generator
 */
class BenchmarkReportGenerator {
  constructor(options = {}) {
    this.inputDir = options.inputDir;
    this.runId = options.runId;
    this.outputFormats = options.outputFormats || ['json', 'markdown'];
    this.compareWithHistory = options.compareWithHistory || false;
    this.verbose = options.verbose || false;
    
    this.currentRun = null;
    this.historicalRuns = [];
    this.trendData = null;
  }
  
  /**
   * Load current run data from input directory
   */
  async loadCurrentRun() {
    console.log(`📂 Loading current run data from ${this.inputDir}...`);
    
    try {
      // Load main results
      const resultsPath = path.join(this.inputDir, 'benchmark-results.json');
      const summaryPath = path.join(this.inputDir, 'summary.json');
      const metadataPath = path.join(this.inputDir, 'execution-metadata.json');
      
      this.currentRun = {
        results: await this.loadJsonFile(resultsPath),
        summary: await this.loadJsonFile(summaryPath),
        metadata: await this.loadJsonFile(metadataPath)
      };
      
      // Load optional consistency report
      const consistencyPath = path.join(this.inputDir, 'consistency-validation.json');
      try {
        this.currentRun.consistency = await this.loadJsonFile(consistencyPath);
      } catch {
        console.warn('⚠️ Consistency validation data not found');
      }
      
      console.log(`✅ Current run data loaded (Run ID: ${this.currentRun.summary?.run_id || 'unknown'})`);
    } catch (error) {
      throw new Error(`Failed to load current run data: ${error.message}`);
    }
  }
  
  /**
   * Load historical run data for trend analysis
   */
  async loadHistoricalData() {
    if (!this.compareWithHistory) {
      console.log('🔍 Skipping historical data loading (not requested)');
      return;
    }
    
    console.log('📈 Loading historical data for trend analysis...');
    
    const historyDir = path.join(projectRoot, 'benchmark-results', 'history');
    
    try {
      const historyEntries = await fs.readdir(historyDir, { withFileTypes: true });
      const runDirs = historyEntries
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name)
        .sort()
        .reverse() // Most recent first
        .slice(0, REPORT_CONFIG.COMPARISON_RUNS);
      
      for (const runDir of runDirs) {
        try {
          const summaryPath = path.join(historyDir, runDir, 'summary.json');
          const summary = await this.loadJsonFile(summaryPath);
          
          this.historicalRuns.push({
            run_id: runDir,
            summary,
            timestamp: summary.timestamp
          });
        } catch (error) {
          console.warn(`⚠️ Could not load historical run ${runDir}: ${error.message}`);
        }
      }
      
      console.log(`✅ Loaded ${this.historicalRuns.length} historical runs for comparison`);
    } catch (error) {
      console.warn(`⚠️ Could not load historical data: ${error.message}`);
    }
  }
  
  /**
   * Analyze trends and detect regressions
   */
  analyzeTrends() {
    if (this.historicalRuns.length === 0) {
      console.log('📉 Skipping trend analysis (no historical data)');
      return null;
    }
    
    console.log('🗺 Analyzing performance trends...');
    
    const allRuns = [this.currentRun.summary, ...this.historicalRuns.map(r => r.summary)];
    const trends = {};
    const regressions = [];
    
    // Analyze each metric
    const metrics = ['recall_at_10', 'recall_at_50', 'ndcg_at_10', 'e2e_p95'];
    
    for (const metric of metrics) {
      const values = allRuns.map(run => this.getMetricValue(run, metric)).filter(v => v !== null);
      
      if (values.length >= 2) {
        const current = values[0];
        const previous = values[1];
        const average = values.reduce((sum, val) => sum + val, 0) / values.length;
        
        const changeFromPrevious = (current - previous) / previous;
        const changeFromAverage = (current - average) / average;
        
        trends[metric] = {
          current,
          previous,
          average,
          change_from_previous: changeFromPrevious,
          change_from_average: changeFromAverage,
          values: values.slice(0, 10), // Keep last 10 for visualization
          is_improving: current > previous
        };
        
        // Check for regressions
        const threshold = REPORT_CONFIG.REGRESSION_THRESHOLDS[metric] || -0.10;
        if (changeFromPrevious < threshold) {
          regressions.push({
            metric,
            change: changeFromPrevious,
            threshold,
            severity: Math.abs(changeFromPrevious) > Math.abs(threshold * 2) ? 'critical' : 'warning'
          });
        }
      }
    }
    
    this.trendData = {
      trends,
      regressions,
      historical_runs_analyzed: allRuns.length - 1
    };
    
    console.log(`✅ Trend analysis complete (${regressions.length} regressions detected)`);
    return this.trendData;
  }
  
  /**
   * Get metric value from a run summary with proper path navigation
   */
  getMetricValue(runSummary, metric) {
    switch (metric) {
      case 'recall_at_10':
        return runSummary.performance?.recall_at_10;
      case 'recall_at_50':
        return runSummary.performance?.recall_at_50;
      case 'ndcg_at_10':
        return runSummary.performance?.ndcg_at_10;
      case 'e2e_p95':
        return runSummary.latency?.e2e_p95;
      case 'error_rate':
        return runSummary.quality?.error_rate;
      default:
        return null;
    }
  }
  
  /**
   * Generate comprehensive JSON report
   */
  generateJsonReport() {
    console.log('📋 Generating JSON report...');
    
    const report = {
      meta: {
        report_version: '1.0',
        generated_at: new Date().toISOString(),
        run_id: this.currentRun.summary?.run_id || this.runId,
        generator: 'lens-benchmark-reporter'
      },
      
      current_run: {
        summary: this.currentRun.summary,
        metadata: this.currentRun.metadata,
        consistency_validation: this.currentRun.consistency
      },
      
      performance_analysis: this.analyzePerformance(),
      trend_analysis: this.trendData,
      recommendations: this.generateRecommendations(),
      
      // Raw historical data for external analysis
      historical_data: this.historicalRuns.map(run => ({
        run_id: run.run_id,
        timestamp: run.timestamp,
        key_metrics: {
          recall_at_10: this.getMetricValue(run.summary, 'recall_at_10'),
          ndcg_at_10: this.getMetricValue(run.summary, 'ndcg_at_10'),
          e2e_p95: this.getMetricValue(run.summary, 'e2e_p95'),
          error_rate: this.getMetricValue(run.summary, 'error_rate')
        }
      }))
    };
    
    console.log('✅ JSON report generated');
    return report;
  }
  
  /**
   * Analyze current performance against targets
   */
  analyzePerformance() {
    const summary = this.currentRun.summary;
    const analysis = {
      overall_status: 'unknown',
      target_compliance: {},
      issues: [],
      highlights: []
    };
    
    let targetsMetCount = 0;
    let totalTargets = 0;
    
    // Check each performance target
    for (const [metric, target] of Object.entries(REPORT_CONFIG.PERFORMANCE_TARGETS)) {
      const current = this.getMetricValue(summary, metric);
      
      if (current !== null) {
        totalTargets++;
        
        const isLatency = metric.includes('latency');
        const isError = metric.includes('error');
        const metTarget = isLatency || isError ? current <= target : current >= target;
        
        analysis.target_compliance[metric] = {
          current,
          target,
          met: metTarget,
          deviation: isLatency || isError ? current - target : target - current
        };
        
        if (metTarget) {
          targetsMetCount++;
          
          // Highlight exceptional performance
          const exceptionThreshold = isLatency || isError ? 0.8 : 1.2;
          const isExceptional = isLatency || isError ? 
            current <= target * exceptionThreshold :
            current >= target * exceptionThreshold;
            
          if (isExceptional) {
            analysis.highlights.push(`Exceptional ${metric}: ${this.formatMetric(metric, current)}`);
          }
        } else {
          analysis.issues.push(`${metric} below target: ${this.formatMetric(metric, current)} < ${this.formatMetric(metric, target)}`);
        }
      }
    }
    
    // Determine overall status
    const complianceRate = totalTargets > 0 ? targetsMetCount / totalTargets : 0;
    if (complianceRate >= 0.9) {
      analysis.overall_status = 'excellent';
    } else if (complianceRate >= 0.7) {
      analysis.overall_status = 'good';
    } else if (complianceRate >= 0.5) {
      analysis.overall_status = 'needs_improvement';
    } else {
      analysis.overall_status = 'poor';
    }
    
    analysis.compliance_rate = complianceRate;
    analysis.targets_met = targetsMetCount;
    analysis.total_targets = totalTargets;
    
    return analysis;
  }
  
  /**
   * Generate actionable recommendations
   */
  generateRecommendations() {
    const recommendations = [];
    const performanceAnalysis = this.analyzePerformance();
    
    // Performance-based recommendations
    if (performanceAnalysis.overall_status === 'poor') {
      recommendations.push({
        type: 'urgent',
        category: 'performance',
        title: 'Critical Performance Issues Detected',
        description: 'Multiple performance targets are not being met',
        actions: [
          'Review and optimize Stage-A lexical matching algorithms',
          'Analyze Stage-B structural search bottlenecks',
          'Consider index optimization and caching strategies',
          'Schedule performance debugging session'
        ]
      });
    }
    
    // Regression-based recommendations
    if (this.trendData?.regressions?.length > 0) {
      const criticalRegressions = this.trendData.regressions.filter(r => r.severity === 'critical');
      
      if (criticalRegressions.length > 0) {
        recommendations.push({
          type: 'critical',
          category: 'regression',
          title: 'Critical Performance Regressions',
          description: `${criticalRegressions.length} critical regression(s) detected`,
          actions: [
            'Investigate recent code changes for performance impact',
            'Consider rolling back recent changes if regression source is unclear',
            'Run targeted profiling on affected components',
            'Review resource utilization trends'
          ]
        });
      }
    }
    
    // Quality-based recommendations  
    const errorRate = this.currentRun.summary?.quality?.error_rate || 0;
    if (errorRate > 0.02) { // 2% threshold
      recommendations.push({
        type: 'warning',
        category: 'quality',
        title: 'Elevated Error Rate',
        description: `Error rate of ${(errorRate * 100).toFixed(1)}% detected`,
        actions: [
          'Review error logs for common failure patterns',
          'Check corpus-golden consistency validation results',
          'Verify search server health and resource availability',
          'Consider increasing timeout values if needed'
        ]
      });
    }
    
    // Positive recommendations
    if (performanceAnalysis.overall_status === 'excellent' && this.trendData?.regressions?.length === 0) {
      recommendations.push({
        type: 'success',
        category: 'maintenance',
        title: 'System Performing Excellently',
        description: 'All targets met with no regressions detected',
        actions: [
          'Continue current optimization strategies',
          'Consider documenting current configuration as baseline',
          'Explore opportunities for further performance improvements',
          'Review and update performance targets if consistently exceeded'
        ]
      });
    }
    
    return recommendations;
  }
  
  /**
   * Generate human-readable Markdown report
   */
  generateMarkdownReport() {
    console.log('📝 Generating Markdown report...');
    
    const summary = this.currentRun.summary;
    const metadata = this.currentRun.metadata;
    const performanceAnalysis = this.analyzePerformance();
    const recommendations = this.generateRecommendations();
    
    let markdown = `# Lens Benchmark Report\n\n`;
    
    // Header and metadata
    markdown += `**Run ID:** \`${summary.run_id}\`  \n`;
    markdown += `**Timestamp:** ${new Date(summary.timestamp).toLocaleString()}  \n`;
    markdown += `**Suite Type:** ${summary.suite_type}  \n`;
    markdown += `**Duration:** ${Math.round((metadata?.duration_ms || 0) / 1000)}s  \n`;
    markdown += `\n---\n\n`;
    
    // Executive Summary
    markdown += `## \ud83c\udfaf Executive Summary\n\n`;
    
    const statusEmoji = {
      'excellent': '✅',
      'good': '🟡',
      'needs_improvement': '🟠',
      'poor': '❌'
    };
    
    markdown += `**Overall Status:** ${statusEmoji[performanceAnalysis.overall_status] || '❓'} ${performanceAnalysis.overall_status}\n\n`;
    
    // Key metrics table
    markdown += `## \ud83d\udcca Key Metrics\n\n`;
    markdown += `| Metric | Current | Target | Status |\n`;
    markdown += `|--------|---------|--------|--------|\n`;
    
    for (const [metric, compliance] of Object.entries(performanceAnalysis.target_compliance)) {
      const status = compliance.met ? '✅ Met' : '❌ Missed';
      markdown += `| ${this.formatMetricName(metric)} | ${this.formatMetric(metric, compliance.current)} | ${this.formatMetric(metric, compliance.target)} | ${status} |\n`;
    }
    
    markdown += `\n`;
    
    // Quality summary
    markdown += `## \ud83d\udccb Quality Summary\n\n`;
    markdown += `- **Total Queries:** ${summary.quality.total_queries}\n`;
    markdown += `- **Completed:** ${summary.quality.completed_queries} (${(summary.quality.success_rate * 100).toFixed(1)}%)\n`;
    markdown += `- **Failed:** ${summary.quality.failed_queries} (${(summary.quality.error_rate * 100).toFixed(1)}%)\n`;
    markdown += `\n`;
    
    // Trend analysis if available
    if (this.trendData && this.trendData.trends) {
      markdown += `## \ud83d\uddfa Trend Analysis\n\n`;
      
      for (const [metric, trend] of Object.entries(this.trendData.trends)) {
        const changeIcon = trend.is_improving ? '📈' : '📉';
        const changePct = (trend.change_from_previous * 100).toFixed(1);
        
        markdown += `### ${this.formatMetricName(metric)} ${changeIcon}\n\n`;
        markdown += `- **Current:** ${this.formatMetric(metric, trend.current)}\n`;
        markdown += `- **Previous:** ${this.formatMetric(metric, trend.previous)}\n`;
        markdown += `- **Change:** ${changePct > 0 ? '+' : ''}${changePct}%\n`;
        markdown += `- **Trend:** ${trend.is_improving ? 'Improving' : 'Declining'}\n`;
        markdown += `\n`;
      }
    }
    
    // Regressions
    if (this.trendData?.regressions?.length > 0) {
      markdown += `## \u26a0\ufe0f Performance Regressions\n\n`;
      
      for (const regression of this.trendData.regressions) {
        const severityEmoji = regression.severity === 'critical' ? '🚨' : '⚠️';
        markdown += `${severityEmoji} **${this.formatMetricName(regression.metric)}** declined by ${(regression.change * 100).toFixed(1)}%\n`;
      }
      
      markdown += `\n`;
    }
    
    // Recommendations
    if (recommendations.length > 0) {
      markdown += `## \ud83d\udca1 Recommendations\n\n`;
      
      for (const rec of recommendations) {
        const typeEmoji = {
          'critical': '🚨',
          'urgent': '❌',
          'warning': '⚠️',
          'success': '✅'
        };
        
        markdown += `### ${typeEmoji[rec.type] || '📝'} ${rec.title}\n\n`;
        markdown += `${rec.description}\n\n`;
        
        if (rec.actions?.length > 0) {
          markdown += `**Actions:**\n`;
          for (const action of rec.actions) {
            markdown += `- ${action}\n`;
          }
        }
        
        markdown += `\n`;
      }
    }
    
    // Consistency validation if available
    if (this.currentRun.consistency) {
      const consistency = this.currentRun.consistency;
      markdown += `## \ud83d\udd0d Corpus-Golden Consistency\n\n`;
      markdown += `- **Status:** ${consistency.passed ? '✅ Passed' : '❌ Failed'}\n`;
      markdown += `- **Valid Results:** ${consistency.report.valid_results}/${consistency.report.total_expected_results}\n`;
      markdown += `- **Pass Rate:** ${(consistency.report.pass_rate * 100).toFixed(1)}%\n`;
      
      if (consistency.report.inconsistencies?.length > 0) {
        markdown += `- **Inconsistencies:** ${consistency.report.inconsistencies.length}\n`;
      }
      
      markdown += `\n`;
    }
    
    // Footer
    markdown += `---\n\n`;
    markdown += `*Report generated on ${new Date().toLocaleString()} by Lens Benchmark Automation*\n`;
    
    console.log('✅ Markdown report generated');
    return markdown;
  }
  
  /**
   * Generate simple HTML report with charts
   */
  generateHtmlReport(jsonReport, markdownReport) {
    console.log('🌐 Generating HTML report...');
    
    // Convert markdown to basic HTML structure
    let html = markdownReport
      .replace(/^# (.+)$/gm, '<h1>$1</h1>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      .replace(/^- (.+)$/gm, '<li>$1</li>')
      .replace(/(\n<li>.*?<\/li>)+/gs, '<ul>$&</ul>')
      .replace(/\n/g, '<br>');
    
    // Add CSS and structure
    const fullHtml = `
<!DOCTYPE html>
<html>
<head>
    <title>Lens Benchmark Report - ${jsonReport.meta.run_id}</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            line-height: 1.6;
        }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f5f5f5; font-weight: bold; }
        code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
        .status-excellent { color: #28a745; }
        .status-good { color: #17a2b8; }
        .status-warning { color: #ffc107; }
        .status-poor { color: #dc3545; }
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .chart-placeholder {
            background: #e9ecef;
            border: 2px dashed #adb5bd;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #6c757d;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    ${html}
    
    ${this.trendData ? `
    <div class="chart-placeholder">
        📊 Trend Chart Placeholder<br>
        <small>In a full implementation, this would contain interactive charts</small>
    </div>
    ` : ''}
    
    <script>
        // In a full implementation, this would include Chart.js or similar
        // for interactive trend visualizations
        console.log('Benchmark report data:', ${JSON.stringify(jsonReport, null, 2)});
    </script>
</body>
</html>`;
    
    console.log('✅ HTML report generated');
    return fullHtml;
  }
  
  /**
   * Format metric names for display
   */
  formatMetricName(metric) {
    const names = {
      'recall_at_10': 'Recall@10',
      'recall_at_50': 'Recall@50',
      'ndcg_at_10': 'NDCG@10',
      'e2e_p95': 'E2E P95 Latency',
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
   * Save report in the specified format
   */
  async saveReport(content, format, outputDir) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `benchmark-report-${this.runId}-${timestamp}.${format}`;
    const outputPath = path.join(outputDir || this.inputDir, filename);
    
    await fs.writeFile(outputPath, content, 'utf-8');
    console.log(`✅ ${format.toUpperCase()} report saved to ${outputPath}`);
    return outputPath;
  }
  
  /**
   * Generate all requested report formats
   */
  async generateAllReports() {
    console.log(`📊 Generating reports in formats: ${this.outputFormats.join(', ')}`);
    
    await this.loadCurrentRun();
    await this.loadHistoricalData();
    this.analyzeTrends();
    
    const generatedFiles = [];
    
    // Generate JSON report (always generated for other formats)
    const jsonReport = this.generateJsonReport();
    
    for (const format of this.outputFormats) {
      let content;
      
      switch (format) {
        case 'json':
          content = JSON.stringify(jsonReport, null, 2);
          break;
          
        case 'markdown':
          content = this.generateMarkdownReport();
          break;
          
        case 'html':
          const markdownReport = this.generateMarkdownReport();
          content = this.generateHtmlReport(jsonReport, markdownReport);
          break;
          
        default:
          console.warn(`⚠️ Unsupported format: ${format}`);
          continue;
      }
      
      const outputPath = await this.saveReport(content, format);
      generatedFiles.push(outputPath);
    }
    
    console.log(`✅ Report generation complete (${generatedFiles.length} files generated)`);
    return {
      files: generatedFiles,
      data: jsonReport
    };
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
  
  function getArrayArg(name, defaultValue = []) {
    const value = getArg(name);
    return value ? value.split(',') : defaultValue;
  }
  
  try {
    const generator = new BenchmarkReportGenerator({
      inputDir: getArg('input-dir'),
      runId: getArg('run-id'),
      outputFormats: getArrayArg('output-formats', ['json', 'markdown']),
      compareWithHistory: hasFlag('compare-with-history'),
      verbose: hasFlag('verbose')
    });
    
    const result = await generator.generateAllReports();
    
    console.log(`🎯 Report generation completed successfully`);
    console.log(`   Generated files: ${result.files.length}`);
    
    if (hasFlag('verbose')) {
      console.log('   Files:');
      for (const file of result.files) {
        console.log(`     - ${file}`);
      }
    }
    
  } catch (error) {
    console.error(`❌ Report generation failed: ${error.message}`);
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

export { BenchmarkReportGenerator, REPORT_CONFIG };
