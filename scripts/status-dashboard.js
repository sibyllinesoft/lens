#!/usr/bin/env node

/**
 * Simple Status Dashboard Generator
 * 
 * Creates a simple HTML dashboard showing benchmark status and trends.
 * Can be served statically or integrated into existing monitoring systems.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

/**
 * Status dashboard generator
 */
class StatusDashboard {
  constructor(options = {}) {
    this.historyDir = options.historyDir || path.join(projectRoot, 'benchmark-results', 'history');
    this.outputFile = options.outputFile || path.join(projectRoot, 'benchmark-status.html');
    this.maxRuns = options.maxRuns || 30;
    
    this.runs = [];
    this.currentStatus = null;
  }
  
  async loadHistoricalRuns() {
    console.log(`📈 Loading historical runs from ${this.historyDir}...`);
    
    try {
      const entries = await fs.readdir(this.historyDir, { withFileTypes: true });
      const runDirs = entries
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name)
        .sort()
        .reverse()
        .slice(0, this.maxRuns);
      
      for (const runDir of runDirs) {
        try {
          const summaryPath = path.join(this.historyDir, runDir, 'summary.json');
          const summary = await this.loadJsonFile(summaryPath);
          
          // Try to load regression analysis if available
          let regressionData = null;
          try {
            const regressionPath = path.join(this.historyDir, runDir, 'regression-analysis.json');
            regressionData = await this.loadJsonFile(regressionPath);
          } catch {
            // Regression analysis not available for this run
          }
          
          this.runs.push({
            run_id: runDir,
            timestamp: new Date(summary.timestamp),
            summary,
            regressions: regressionData
          });
        } catch (error) {
          console.warn(`⚠️ Could not load run ${runDir}: ${error.message}`);
        }
      }
      
      console.log(`✅ Loaded ${this.runs.length} historical runs`);
    } catch (error) {
      console.warn(`⚠️ Could not load historical data: ${error.message}`);
    }
  }
  
  generateDashboard() {
    console.log('📊 Generating status dashboard...');
    
    const latestRun = this.runs[0];
    const overallStatus = this.determineOverallStatus(latestRun);
    
    const html = `<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Lens Benchmark Status</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="300"> <!-- Refresh every 5 minutes -->
    <style>
        ${this.getStyles()}
    </style>
</head>
<body>
    <div class="dashboard">
        <header>
            <h1>🔍 Lens Benchmark Status</h1>
            <div class="last-updated">Last updated: ${new Date().toLocaleString()}</div>
        </header>
        
        <main>
            ${this.generateOverviewSection(latestRun, overallStatus)}
            ${this.generateMetricsSection(latestRun)}
            ${this.generateTrendsSection()}
            ${this.generateRecentRunsSection()}
        </main>
        
        <footer>
            <p>Benchmark automation system - <a href="docs/BENCHMARK_AUTOMATION.md">Documentation</a></p>
        </footer>
    </div>
    
    <script>
        ${this.getJavaScript()}
    </script>
</body>
</html>`;
    
    return html;
  }
  
  determineOverallStatus(latestRun) {
    if (!latestRun) {
      return { level: 'unknown', message: 'No data available' };
    }
    
    const summary = latestRun.summary;
    const regressions = latestRun.regressions;
    
    // Check for critical regressions
    if (regressions?.regressions?.some(r => r.severity === 'critical')) {
      return { level: 'critical', message: 'Critical performance regressions detected' };
    }
    
    // Check quality metrics against targets
    const targets = {
      recall_at_10: 0.70,
      recall_at_50: 0.85,
      ndcg_at_10: 0.65,
      error_rate: 0.05
    };
    
    let missedTargets = 0;
    let totalTargets = 0;
    
    for (const [metric, target] of Object.entries(targets)) {
      const current = this.getMetricValue(summary, metric);
      if (current !== null) {
        totalTargets++;
        const metTarget = metric === 'error_rate' ? current <= target : current >= target;
        if (!metTarget) missedTargets++;
      }
    }
    
    const complianceRate = totalTargets > 0 ? (totalTargets - missedTargets) / totalTargets : 0;
    
    if (complianceRate >= 0.9) {
      return { level: 'healthy', message: 'All systems performing well' };
    } else if (complianceRate >= 0.7) {
      return { level: 'warning', message: 'Some performance targets missed' };
    } else {
      return { level: 'critical', message: 'Multiple performance targets missed' };
    }
  }
  
  generateOverviewSection(latestRun, status) {
    const statusClass = `status-${status.level}`;
    const statusIcon = {
      'healthy': '✅',
      'warning': '⚠️',
      'critical': '❌',
      'unknown': '❓'
    }[status.level] || '❓';
    
    return `
    <section class="overview">
        <div class="status-card ${statusClass}">
            <div class="status-icon">${statusIcon}</div>
            <div class="status-content">
                <h2>Overall Status</h2>
                <div class="status-level">${status.level.toUpperCase()}</div>
                <div class="status-message">${status.message}</div>
                ${latestRun ? `<div class="last-run">Last run: ${latestRun.timestamp.toLocaleString()}</div>` : ''}
            </div>
        </div>
        
        ${latestRun ? this.generateQuickStats(latestRun) : ''}
    </section>`;
  }
  
  generateQuickStats(run) {
    const summary = run.summary;
    const successRate = summary.quality?.success_rate || 0;
    const errorRate = summary.quality?.error_rate || 0;
    
    return `
    <div class="quick-stats">
        <div class="stat-card">
            <div class="stat-value">${(summary.performance?.recall_at_10 * 100 || 0).toFixed(1)}%</div>
            <div class="stat-label">Recall@10</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${(summary.performance?.ndcg_at_10 * 100 || 0).toFixed(1)}%</div>
            <div class="stat-label">NDCG@10</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${Math.round(summary.latency?.e2e_p95 || 0)}ms</div>
            <div class="stat-label">P95 Latency</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${(errorRate * 100).toFixed(1)}%</div>
            <div class="stat-label">Error Rate</div>
        </div>
    </div>`;
  }
  
  generateMetricsSection(latestRun) {
    if (!latestRun) {
      return '<section class="metrics"><h3>Metrics</h3><p>No data available</p></section>';
    }
    
    const summary = latestRun.summary;
    const metrics = [
      { key: 'recall_at_10', label: 'Recall@10', target: 0.70, format: 'percentage' },
      { key: 'recall_at_50', label: 'Recall@50', target: 0.85, format: 'percentage' },
      { key: 'ndcg_at_10', label: 'NDCG@10', target: 0.65, format: 'percentage' },
      { key: 'e2e_p95', label: 'E2E P95 Latency', target: 200, format: 'milliseconds', inverted: true },
      { key: 'error_rate', label: 'Error Rate', target: 0.05, format: 'percentage', inverted: true }
    ];
    
    const metricsHtml = metrics.map(metric => {
      const current = this.getMetricValue(summary, metric.key);
      const target = metric.target;
      
      if (current === null) {
        return `
        <div class="metric-row">
            <div class="metric-label">${metric.label}</div>
            <div class="metric-value">N/A</div>
            <div class="metric-target">Target: ${this.formatValue(target, metric.format)}</div>
            <div class="metric-status status-unknown">❓</div>
        </div>`;
      }
      
      const metTarget = metric.inverted ? current <= target : current >= target;
      const statusClass = metTarget ? 'status-healthy' : 'status-warning';
      const statusIcon = metTarget ? '✅' : '⚠️';
      
      return `
      <div class="metric-row">
          <div class="metric-label">${metric.label}</div>
          <div class="metric-value">${this.formatValue(current, metric.format)}</div>
          <div class="metric-target">Target: ${this.formatValue(target, metric.format)}</div>
          <div class="metric-status ${statusClass}">${statusIcon}</div>
      </div>`;
    }).join('');
    
    return `
    <section class="metrics">
        <h3>Performance Metrics</h3>
        <div class="metrics-grid">
            ${metricsHtml}
        </div>
    </section>`;
  }
  
  generateTrendsSection() {
    if (this.runs.length < 3) {
      return '<section class="trends"><h3>Trends</h3><p>Insufficient data for trend analysis</p></section>';
    }
    
    const recentRuns = this.runs.slice(0, 10).reverse(); // Last 10 runs, chronological
    const chartData = this.prepareChartData(recentRuns);
    
    return `
    <section class="trends">
        <h3>Performance Trends (Last 10 runs)</h3>
        <div class="chart-container">
            <canvas id="trendsChart" width="800" height="300"></canvas>
        </div>
        <script>
            window.chartData = ${JSON.stringify(chartData)};
        </script>
    </section>`;
  }
  
  prepareChartData(runs) {
    const labels = runs.map(run => run.timestamp.toLocaleDateString());
    
    return {
      labels,
      datasets: [
        {
          label: 'Recall@10 (%)',
          data: runs.map(run => (this.getMetricValue(run.summary, 'recall_at_10') || 0) * 100),
          borderColor: '#28a745',
          backgroundColor: 'rgba(40, 167, 69, 0.1)'
        },
        {
          label: 'NDCG@10 (%)',
          data: runs.map(run => (this.getMetricValue(run.summary, 'ndcg_at_10') || 0) * 100),
          borderColor: '#17a2b8',
          backgroundColor: 'rgba(23, 162, 184, 0.1)'
        },
        {
          label: 'P95 Latency (ms)',
          data: runs.map(run => this.getMetricValue(run.summary, 'e2e_p95') || 0),
          borderColor: '#ffc107',
          backgroundColor: 'rgba(255, 193, 7, 0.1)',
          yAxisID: 'latency'
        }
      ]
    };
  }
  
  generateRecentRunsSection() {
    const runsHtml = this.runs.slice(0, 10).map(run => {
      const status = run.regressions?.status || 'UNKNOWN';
      const statusClass = {
        'HEALTHY': 'status-healthy',
        'WARNING_REGRESSION': 'status-warning',
        'CRITICAL_REGRESSION': 'status-critical',
        'INSUFFICIENT_DATA': 'status-unknown'
      }[status] || 'status-unknown';
      
      const statusIcon = {
        'HEALTHY': '✅',
        'WARNING_REGRESSION': '⚠️',
        'CRITICAL_REGRESSION': '❌',
        'INSUFFICIENT_DATA': '❓'
      }[status] || '❓';
      
      return `
      <tr>
          <td>${run.run_id}</td>
          <td>${run.timestamp.toLocaleString()}</td>
          <td>${run.summary.suite_type || 'full'}</td>
          <td>${run.summary.quality?.completed_queries || 0}</td>
          <td>${(run.summary.performance?.recall_at_10 * 100 || 0).toFixed(1)}%</td>
          <td>${Math.round(run.summary.latency?.e2e_p95 || 0)}ms</td>
          <td class="${statusClass}">${statusIcon} ${status}</td>
      </tr>`;
    }).join('');
    
    return `
    <section class="recent-runs">
        <h3>Recent Benchmark Runs</h3>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Run ID</th>
                        <th>Timestamp</th>
                        <th>Type</th>
                        <th>Queries</th>
                        <th>Recall@10</th>
                        <th>P95 Latency</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${runsHtml}
                </tbody>
            </table>
        </div>
    </section>`;
  }
  
  getMetricValue(summary, metric) {
    switch (metric) {
      case 'recall_at_10':
        return summary.performance?.recall_at_10;
      case 'recall_at_50':
        return summary.performance?.recall_at_50;
      case 'ndcg_at_10':
        return summary.performance?.ndcg_at_10;
      case 'e2e_p95':
        return summary.latency?.e2e_p95;
      case 'error_rate':
        return summary.quality?.error_rate;
      default:
        return null;
    }
  }
  
  formatValue(value, format) {
    if (value === null || value === undefined) return 'N/A';
    
    switch (format) {
      case 'percentage':
        return `${(value * 100).toFixed(1)}%`;
      case 'milliseconds':
        return `${Math.round(value)}ms`;
      default:
        return value.toString();
    }
  }
  
  getStyles() {
    return `
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f8f9fa;
        }
        
        .dashboard {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }
        
        header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .last-updated {
            color: #6c757d;
            font-size: 0.9em;
        }
        
        section {
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        section h3 {
            color: #2c3e50;
            margin-bottom: 20px;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        
        .overview {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            align-items: center;
        }
        
        .status-card {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 30px;
            border-radius: 10px;
            border-left: 5px solid;
        }
        
        .status-healthy { border-color: #28a745; background: #d4edda; }
        .status-warning { border-color: #ffc107; background: #fff3cd; }
        .status-critical { border-color: #dc3545; background: #f8d7da; }
        .status-unknown { border-color: #6c757d; background: #e9ecef; }
        
        .status-icon {
            font-size: 3em;
        }
        
        .status-level {
            font-size: 1.5em;
            font-weight: bold;
            margin: 5px 0;
        }
        
        .status-message {
            color: #6c757d;
        }
        
        .last-run {
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 10px;
        }
        
        .quick-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
        }
        
        .stat-card {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .stat-label {
            color: #6c757d;
            margin-top: 5px;
        }
        
        .metrics-grid {
            display: grid;
            gap: 15px;
        }
        
        .metric-row {
            display: grid;
            grid-template-columns: 1fr auto auto auto;
            gap: 20px;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        
        .metric-label {
            font-weight: 500;
        }
        
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .metric-target {
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .metric-status {
            font-size: 1.2em;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            color: #6c757d;
        }
        
        .chart-container::before {
            content: '📊 Interactive charts would be displayed here in a full implementation';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-style: italic;
        }
        
        .table-container {
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        
        tbody tr:hover {
            background: #f8f9fa;
        }
        
        footer {
            text-align: center;
            padding: 30px;
            color: #6c757d;
            font-size: 0.9em;
        }
        
        footer a {
            color: #007bff;
            text-decoration: none;
        }
        
        footer a:hover {
            text-decoration: underline;
        }
        
        @media (max-width: 768px) {
            .overview {
                grid-template-columns: 1fr;
            }
            
            .status-card {
                flex-direction: column;
                text-align: center;
            }
            
            .metric-row {
                grid-template-columns: 1fr;
                text-align: center;
            }
            
            .quick-stats {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    `;
  }
  
  getJavaScript() {
    return `
        // Simple dashboard interactivity
        document.addEventListener('DOMContentLoaded', function() {
            // Auto-refresh functionality
            let refreshInterval;
            
            function startAutoRefresh() {
                refreshInterval = setInterval(() => {
                    window.location.reload();
                }, 5 * 60 * 1000); // 5 minutes
            }
            
            function stopAutoRefresh() {
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                    refreshInterval = null;
                }
            }
            
            // Start auto-refresh by default
            startAutoRefresh();
            
            // Stop auto-refresh when page is not visible
            document.addEventListener('visibilitychange', function() {
                if (document.hidden) {
                    stopAutoRefresh();
                } else {
                    startAutoRefresh();
                }
            });
            
            // Add click handlers for metric rows
            document.querySelectorAll('.metric-row').forEach(row => {
                row.style.cursor = 'pointer';
                row.addEventListener('click', function() {
                    // In a full implementation, this could show detailed metric history
                    const metricLabel = row.querySelector('.metric-label').textContent;
                    alert('Detailed ' + metricLabel + ' history would be shown here');
                });
            });
            
            console.log('Lens Benchmark Status Dashboard initialized');
            console.log('Auto-refresh every 5 minutes');
            console.log('Chart data:', window.chartData || 'No chart data available');
        });
    `;
  }
  
  async loadJsonFile(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      throw new Error(`Failed to load ${filePath}: ${error.message}`);
    }
  }
  
  async generateAndSave() {
    await this.loadHistoricalRuns();
    const html = this.generateDashboard();
    
    await fs.writeFile(this.outputFile, html);
    console.log(`✅ Status dashboard generated: ${this.outputFile}`);
    
    return this.outputFile;
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
    const dashboard = new StatusDashboard({
      historyDir: getArg('history-dir'),
      outputFile: getArg('output'),
      maxRuns: parseInt(getArg('max-runs', '30'))
    });
    
    const outputFile = await dashboard.generateAndSave();
    
    console.log(`🎯 Status dashboard generation completed`);
    console.log(`   Output: ${outputFile}`);
    console.log(`   Runs analyzed: ${dashboard.runs.length}`);
    
    // If running in a web server context, could start a simple server here
    if (hasFlag('serve')) {
      const port = parseInt(getArg('port', '8080'));
      console.log(`🌍 Starting server on http://localhost:${port}`);
      // Simple HTTP server implementation would go here
    }
    
  } catch (error) {
    console.error(`❌ Dashboard generation failed: ${error.message}`);
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

export { StatusDashboard };
