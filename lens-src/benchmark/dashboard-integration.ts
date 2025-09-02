/**
 * Dashboard Integration for Lens Quality Gates
 * Provides real-time dashboard updates and alerting integration
 */

import { promises as fs } from 'fs';
import path from 'path';
import type { TestExecutionResult, TestExecutionConfig } from './test-orchestrator.js';
import type { BenchmarkRun } from '../types/benchmark.js';

export interface DashboardConfig {
  apiEndpoint?: string;
  alertWebhook?: string;
  slackWebhook?: string;
  emailConfig?: {
    smtp: string;
    from: string;
    to: string[];
  };
  retentionDays: number;
}

export interface QualityMetrics {
  timestamp: string;
  ndcg_at_10: number;
  recall_at_50: number;
  latency_p95: number;
  error_rate: number;
  quality_score: number;
  stability_score: number;
  performance_score: number;
  test_type: 'smoke' | 'nightly';
  passed: boolean;
}

export interface AlertCondition {
  metric: keyof QualityMetrics;
  threshold: number;
  comparison: 'greater_than' | 'less_than' | 'equals';
  severity: 'critical' | 'warning' | 'info';
  message_template: string;
}

export class DashboardIntegration {
  private readonly defaultAlertConditions: AlertCondition[] = [
    {
      metric: 'quality_score',
      threshold: 0.85,
      comparison: 'less_than',
      severity: 'critical',
      message_template: 'Quality score dropped to {value} (threshold: {threshold})'
    },
    {
      metric: 'stability_score',
      threshold: 0.95,
      comparison: 'less_than',
      severity: 'warning',
      message_template: 'Stability score dropped to {value} (threshold: {threshold})'
    },
    {
      metric: 'ndcg_at_10',
      threshold: 0.80,
      comparison: 'less_than',
      severity: 'critical',
      message_template: 'nDCG@10 dropped to {value} (threshold: {threshold})'
    },
    {
      metric: 'error_rate',
      threshold: 0.05,
      comparison: 'greater_than',
      severity: 'warning',
      message_template: 'Error rate increased to {value} (threshold: {threshold})'
    }
  ];

  constructor(
    private readonly outputDir: string,
    private readonly config: DashboardConfig = { retentionDays: 90 }
  ) {}

  /**
   * Update dashboard with latest test results
   */
  async updateDashboard(result: TestExecutionResult): Promise<void> {
    console.log('📊 Updating quality dashboard...');

    // Extract quality metrics
    const metrics = this.extractQualityMetrics(result);
    
    // Store metrics locally
    await this.storeMetrics(metrics);
    
    // Update dashboard data
    const dashboardData = await this.generateDashboardUpdate(metrics, result);
    
    // Send to external dashboard API if configured
    if (this.config.apiEndpoint) {
      await this.sendToDashboard(dashboardData);
    }
    
    // Generate local dashboard files
    await this.generateLocalDashboard(dashboardData);
    
    // Check alert conditions
    await this.checkAlerts(metrics);
    
    console.log('✅ Dashboard updated successfully');
  }

  /**
   * Generate PR comment data for GitHub integration
   */
  async generatePRComment(result: TestExecutionResult): Promise<{
    status: 'success' | 'failure' | 'warning';
    title: string;
    body: string;
  }> {
    const passed = result.passed;
    const status = passed ? 'success' : (result.test_type === 'smoke_pr' ? 'failure' : 'warning');
    
    const metrics = this.extractQualityMetrics(result);
    
    const title = `🔍 Lens Quality Gates - ${passed ? '✅ PASS' : '❌ FAIL'}`;
    
    const body = `## ${title}

**Test Type:** ${result.test_type === 'smoke_pr' ? 'Smoke Tests (PR Gate)' : 'Full Nightly Tests'}  
**Duration:** ${(result.duration_ms / 1000).toFixed(1)}s  
**Status:** ${passed ? '✅ PASS - Ready to merge' : '❌ FAIL - ' + (result.blocking_merge ? 'Merge blocked' : 'Issues detected')}

### 📈 Quality Metrics

| Metric | Value | Status |
|--------|-------|---------|
| Quality Score | ${(metrics.quality_score * 100).toFixed(1)}% | ${this.getStatusIcon(metrics.quality_score, 0.85)} |
| nDCG@10 | ${(metrics.ndcg_at_10 * 100).toFixed(1)}% | ${this.getStatusIcon(metrics.ndcg_at_10, 0.80)} |
| Recall@50 | ${(metrics.recall_at_50 * 100).toFixed(1)}% | ${this.getStatusIcon(metrics.recall_at_50, 0.80)} |
| Stability Score | ${(metrics.stability_score * 100).toFixed(1)}% | ${this.getStatusIcon(metrics.stability_score, 0.95)} |
| Latency P95 | ${metrics.latency_p95.toFixed(0)}ms | ${this.getStatusIcon(1 - metrics.latency_p95/1000, 0.2)} |
| Error Rate | ${(metrics.error_rate * 100).toFixed(2)}% | ${this.getStatusIcon(1 - metrics.error_rate, 0.98)} |

### 🔍 Gate Results

- **Preflight Checks:** ${result.preflight_passed ? '✅ PASS' : '❌ FAIL'}
- **Performance Gates:** ${result.performance_passed ? '✅ PASS' : '❌ FAIL'}

${result.benchmark_runs?.length > 0 ? `
### 📊 Test Execution Summary

- **Total Queries:** ${result.total_queries}
- **Benchmark Runs:** ${result.benchmark_runs.length}
- **Error Count:** ${result.error_count}
` : ''}

${!passed ? `
### ⚠️ Issues Detected

${this.getIssuesList(result)}
` : ''}

### 📋 Artifacts

- [📊 Test Summary](${path.basename(result.artifacts.summary_json)})
- [📈 Metrics Data](${path.basename(result.artifacts.metrics_parquet)})
- [🚨 Error Log](${path.basename(result.artifacts.errors_ndjson)})
- [📄 Full Report](${path.basename(result.artifacts.report_pdf)})

---
*Generated by Lens Quality Gates v${process.env.npm_package_version || '1.0.0'}* 🔍`;

    return { status, title, body };
  }

  /**
   * Generate dashboard status badge
   */
  async generateStatusBadge(latest_result?: TestExecutionResult): Promise<{
    label: string;
    message: string;
    color: string;
    svg: string;
  }> {
    let status = 'unknown';
    let color = 'lightgrey';
    let message = 'unknown';
    
    if (latest_result) {
      const metrics = this.extractQualityMetrics(latest_result);
      status = latest_result.passed ? 'passing' : 'failing';
      color = latest_result.passed ? 'brightgreen' : 'red';
      message = `${(metrics.quality_score * 100).toFixed(0)}% quality`;
    }
    
    // Generate simple SVG badge
    const svg = this.generateBadgeSVG('Lens Quality', message, color);
    
    return {
      label: 'Lens Quality',
      message,
      color,
      svg
    };
  }

  /**
   * Clean up old dashboard data
   */
  async cleanupOldData(): Promise<void> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - this.config.retentionDays);
    
    try {
      const files = await fs.readdir(this.outputDir);
      const metricsFiles = files.filter(f => f.includes('_metrics_') || f.includes('_dashboard_'));
      
      for (const file of metricsFiles) {
        const filePath = path.join(this.outputDir, file);
        const stats = await fs.stat(filePath);
        
        if (stats.mtime < cutoffDate) {
          await fs.unlink(filePath);
          console.log(`🗑️ Cleaned up old file: ${file}`);
        }
      }
    } catch (error) {
      console.warn('Warning: Could not clean up old dashboard data:', error);
    }
  }

  // Private helper methods

  private extractQualityMetrics(result: TestExecutionResult): QualityMetrics {
    return {
      timestamp: result.timestamp,
      ndcg_at_10: this.getAverageMetric(result.benchmark_runs, 'ndcg_at_10'),
      recall_at_50: this.getAverageMetric(result.benchmark_runs, 'recall_at_50'),
      latency_p95: this.getAverageLatency(result.benchmark_runs),
      error_rate: result.error_count / Math.max(result.total_queries, 1),
      quality_score: result.quality_score,
      stability_score: result.stability_score,
      performance_score: result.performance_score,
      test_type: result.test_type === 'smoke_pr' ? 'smoke' : 'nightly',
      passed: result.passed
    };
  }

  private getAverageMetric(runs: any[], metric: string): number {
    if (!runs || runs.length === 0) return 0;
    const sum = runs.reduce((acc, run) => acc + (run.metrics?.[metric] || 0), 0);
    return sum / runs.length;
  }

  private getAverageLatency(runs: any[]): number {
    if (!runs || runs.length === 0) return 0;
    const sum = runs.reduce((acc, run) => acc + (run.metrics?.stage_latencies?.e2e_p95 || 0), 0);
    return sum / runs.length;
  }

  private async storeMetrics(metrics: QualityMetrics): Promise<void> {
    const metricsFile = path.join(this.outputDir, 'quality_metrics_history.ndjson');
    const line = JSON.stringify(metrics) + '\n';
    await fs.appendFile(metricsFile, line);
  }

  private async generateDashboardUpdate(metrics: QualityMetrics, result: TestExecutionResult): Promise<any> {
    return {
      timestamp: metrics.timestamp,
      status: result.passed ? 'healthy' : 'degraded',
      test_type: metrics.test_type,
      
      // Current metrics
      current_metrics: {
        quality_score: metrics.quality_score,
        ndcg_at_10: metrics.ndcg_at_10,
        recall_at_50: metrics.recall_at_50,
        stability_score: metrics.stability_score,
        latency_p95: metrics.latency_p95,
        error_rate: metrics.error_rate
      },
      
      // Trend data (would load from history in production)
      trends: {
        quality_trend_24h: 0.02, // +2% (placeholder)
        stability_trend_24h: 0.001, // +0.1% (placeholder)
        error_rate_trend_24h: -0.005 // -0.5% (placeholder)
      },
      
      // Alert status
      alerts: await this.evaluateAlerts(metrics),
      
      // Test execution metadata
      execution_metadata: {
        duration_ms: result.duration_ms,
        total_queries: result.total_queries,
        benchmark_runs: result.benchmark_runs?.length || 0,
        artifacts_generated: Object.keys(result.artifacts).length
      }
    };
  }

  private async sendToDashboard(data: any): Promise<void> {
    if (!this.config.apiEndpoint) return;
    
    try {
      const response = await fetch(this.config.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      
      if (!response.ok) {
        throw new Error(`Dashboard API returned ${response.status}`);
      }
      
      console.log('📡 Dashboard API updated successfully');
    } catch (error) {
      console.warn('Warning: Could not update external dashboard:', error);
    }
  }

  private async generateLocalDashboard(data: any): Promise<void> {
    const dashboardPath = path.join(this.outputDir, 'dashboard_latest.json');
    await fs.writeFile(dashboardPath, JSON.stringify(data, null, 2));
    
    // Generate HTML dashboard (simple version)
    const htmlDashboard = this.generateHTMLDashboard(data);
    const htmlPath = path.join(this.outputDir, 'dashboard.html');
    await fs.writeFile(htmlPath, htmlDashboard);
  }

  private async checkAlerts(metrics: QualityMetrics): Promise<void> {
    const triggeredAlerts = [];
    
    for (const condition of this.defaultAlertConditions) {
      const value = metrics[condition.metric] as number;
      let triggered = false;
      
      switch (condition.comparison) {
        case 'greater_than':
          triggered = value > condition.threshold;
          break;
        case 'less_than':
          triggered = value < condition.threshold;
          break;
        case 'equals':
          triggered = Math.abs(value - condition.threshold) < 0.001;
          break;
      }
      
      if (triggered) {
        const alert = {
          ...condition,
          value,
          message: condition.message_template
            .replace('{value}', value.toString())
            .replace('{threshold}', condition.threshold.toString())
        };
        
        triggeredAlerts.push(alert);
      }
    }
    
    if (triggeredAlerts.length > 0) {
      await this.sendAlerts(triggeredAlerts);
    }
  }

  private async evaluateAlerts(metrics: QualityMetrics): Promise<any[]> {
    const alerts = [];
    
    if (metrics.quality_score < 0.85) {
      alerts.push({
        severity: 'critical',
        message: `Quality score dropped to ${(metrics.quality_score * 100).toFixed(1)}%`
      });
    }
    
    if (metrics.error_rate > 0.05) {
      alerts.push({
        severity: 'warning',
        message: `Error rate increased to ${(metrics.error_rate * 100).toFixed(2)}%`
      });
    }
    
    return alerts;
  }

  private async sendAlerts(alerts: any[]): Promise<void> {
    console.log(`🚨 ${alerts.length} alert(s) triggered:`);
    alerts.forEach(alert => {
      console.log(`  ${alert.severity.toUpperCase()}: ${alert.message}`);
    });
    
    // In production, this would send to Slack, email, etc.
    if (this.config.slackWebhook) {
      try {
        const payload = {
          text: `🚨 Lens Quality Alert`,
          attachments: alerts.map(alert => ({
            color: alert.severity === 'critical' ? 'danger' : 'warning',
            text: alert.message
          }))
        };
        
        await fetch(this.config.slackWebhook, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        
        console.log('📢 Slack alert sent');
      } catch (error) {
        console.warn('Warning: Could not send Slack alert:', error);
      }
    }
  }

  private getStatusIcon(value: number, threshold: number): string {
    return value >= threshold ? '✅' : '❌';
  }

  private getIssuesList(result: TestExecutionResult): string {
    const issues = [];
    
    if (!result.preflight_passed) {
      issues.push('- **Preflight checks failed**: Data consistency issues detected');
    }
    
    if (!result.performance_passed) {
      issues.push('- **Performance gates triggered**: Quality or latency thresholds exceeded');
    }
    
    if (result.error_count > 0) {
      issues.push(`- **Errors detected**: ${result.error_count} errors occurred during testing`);
    }
    
    return issues.length > 0 ? issues.join('\n') : 'No specific issues identified';
  }

  private generateBadgeSVG(label: string, message: string, color: string): string {
    const labelWidth = label.length * 6 + 10;
    const messageWidth = message.length * 6 + 10;
    const totalWidth = labelWidth + messageWidth;
    
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${totalWidth}" height="20">
      <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
      </linearGradient>
      <mask id="a">
        <rect width="${totalWidth}" height="20" rx="3" fill="#fff"/>
      </mask>
      <g mask="url(#a)">
        <path fill="#555" d="M0 0h${labelWidth}v20H0z"/>
        <path fill="${color}" d="M${labelWidth} 0h${messageWidth}v20H${labelWidth}z"/>
        <path fill="url(#b)" d="M0 0h${totalWidth}v20H0z"/>
      </g>
      <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="${labelWidth/2}" y="15" fill="#010101" fill-opacity=".3">${label}</text>
        <text x="${labelWidth/2}" y="14">${label}</text>
        <text x="${labelWidth + messageWidth/2}" y="15" fill="#010101" fill-opacity=".3">${message}</text>
        <text x="${labelWidth + messageWidth/2}" y="14">${message}</text>
      </g>
    </svg>`;
  }

  private generateHTMLDashboard(data: any): string {
    return `<!DOCTYPE html>
<html>
<head>
  <title>Lens Quality Dashboard</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .header { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
    .metric { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .metric-value { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
    .metric-label { color: #666; font-size: 14px; }
    .status-healthy { color: #28a745; }
    .status-degraded { color: #dc3545; }
    .alerts { background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 8px; margin-top: 20px; }
    .timestamp { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <div class="header">
    <h1>🔍 Lens Quality Dashboard</h1>
    <p class="timestamp">Last updated: ${new Date(data.timestamp).toLocaleString()}</p>
    <p class="status-${data.status}">Status: ${data.status.toUpperCase()}</p>
  </div>
  
  <div class="metrics">
    <div class="metric">
      <div class="metric-value">${(data.current_metrics.quality_score * 100).toFixed(1)}%</div>
      <div class="metric-label">Quality Score</div>
    </div>
    <div class="metric">
      <div class="metric-value">${(data.current_metrics.ndcg_at_10 * 100).toFixed(1)}%</div>
      <div class="metric-label">nDCG@10</div>
    </div>
    <div class="metric">
      <div class="metric-value">${(data.current_metrics.recall_at_50 * 100).toFixed(1)}%</div>
      <div class="metric-label">Recall@50</div>
    </div>
    <div class="metric">
      <div class="metric-value">${data.current_metrics.latency_p95.toFixed(0)}ms</div>
      <div class="metric-label">Latency P95</div>
    </div>
    <div class="metric">
      <div class="metric-value">${(data.current_metrics.error_rate * 100).toFixed(2)}%</div>
      <div class="metric-label">Error Rate</div>
    </div>
    <div class="metric">
      <div class="metric-value">${(data.current_metrics.stability_score * 100).toFixed(1)}%</div>
      <div class="metric-label">Stability Score</div>
    </div>
  </div>
  
  ${data.alerts.length > 0 ? `
  <div class="alerts">
    <h3>🚨 Active Alerts</h3>
    <ul>
      ${data.alerts.map((alert: any) => `<li><strong>${alert.severity.toUpperCase()}:</strong> ${alert.message}</li>`).join('')}
    </ul>
  </div>
  ` : ''}
  
  <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
    Generated by Lens Quality Gates • Test Type: ${data.test_type} • Duration: ${(data.execution_metadata.duration_ms / 1000).toFixed(1)}s
  </div>
</body>
</html>`;
  }
}