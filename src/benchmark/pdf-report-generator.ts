/**
 * PDF Report Generator for Phase C Hardening
 * Generates comprehensive PDF reports with embedded plots and analysis
 */

import { promises as fs } from 'fs';
import path from 'path';
import type { BenchmarkRun } from '../types/benchmark.js';
import type { HardeningReport, SliceMetrics, TripwireResult } from './phase-c-hardening.js';

export interface PDFReportConfig {
  title: string;
  subtitle?: string;
  author: string;
  template: 'comprehensive' | 'executive' | 'technical';
  include_plots: boolean;
  include_raw_data: boolean;
  output_format: 'pdf' | 'html' | 'markdown';
}

export interface PDFSection {
  title: string;
  content: string;
  subsections?: PDFSection[];
  plots?: string[];
  data_tables?: any[];
  metadata?: Record<string, any>;
}

export class PDFReportGenerator {
  
  constructor(private readonly outputDir: string) {}

  /**
   * Generate comprehensive PDF report from hardening results
   */
  async generateHardeningReport(
    hardeningReport: HardeningReport,
    benchmarkResults: BenchmarkRun[],
    config: PDFReportConfig
  ): Promise<string> {
    
    console.log(`📄 Generating ${config.template} PDF report...`);
    
    const sections = await this.buildReportSections(hardeningReport, benchmarkResults, config);
    const reportContent = this.assembleReport(sections, config);
    
    const outputPath = await this.writeReport(reportContent, config);
    
    console.log(`✅ PDF report generated: ${outputPath}`);
    return outputPath;
  }

  /**
   * Build all sections for the hardening report
   */
  private async buildReportSections(
    hardeningReport: HardeningReport,
    benchmarkResults: BenchmarkRun[],
    config: PDFReportConfig
  ): Promise<PDFSection[]> {
    
    const sections: PDFSection[] = [];

    // Executive Summary
    sections.push(await this.buildExecutiveSummary(hardeningReport));

    // Tripwire Analysis
    sections.push(await this.buildTripwireAnalysis(hardeningReport.tripwire_results));

    // Performance Analysis
    sections.push(await this.buildPerformanceAnalysis(benchmarkResults, hardeningReport));

    // Hard Negative Testing
    if (hardeningReport.hard_negatives.total_generated > 0) {
      sections.push(await this.buildHardNegativeAnalysis(hardeningReport.hard_negatives));
    }

    // Per-Slice Analysis
    if (hardeningReport.slice_results.length > 0) {
      sections.push(await this.buildSliceAnalysis(hardeningReport.slice_results));
    }

    // Visualization Gallery
    if (config.include_plots && hardeningReport.plots_generated) {
      sections.push(await this.buildVisualizationGallery(hardeningReport.plots_generated));
    }

    // Recommendations & Action Items
    sections.push(await this.buildRecommendations(hardeningReport.recommendations));

    // Appendix (Raw Data)
    if (config.include_raw_data) {
      sections.push(await this.buildRawDataAppendix(hardeningReport, benchmarkResults));
    }

    return sections;
  }

  /**
   * Build executive summary section
   */
  private async buildExecutiveSummary(hardeningReport: HardeningReport): Promise<PDFSection> {
    
    const statusEmoji = hardeningReport.hardening_status === 'pass' ? '✅' : '❌';
    const tripwireStatus = hardeningReport.tripwire_summary;
    const sliceStatus = hardeningReport.slice_gate_summary;

    const content = `
# Executive Summary

**Overall Hardening Status**: ${statusEmoji} ${hardeningReport.hardening_status.toUpperCase()}

## Key Findings

### Tripwire Status
- **${tripwireStatus.passed_tripwires}/${tripwireStatus.total_tripwires} Tripwires Passed** 
- Critical failures: ${tripwireStatus.failed_tripwires}
- Overall tripwire status: ${tripwireStatus.overall_status === 'pass' ? '✅ PASS' : '❌ FAIL'}

${sliceStatus ? `
### Per-Slice Performance Gates
- **${sliceStatus.passed_slices}/${sliceStatus.total_slices} Slices Passed**
- Failed slices: ${sliceStatus.failed_slices}
- Slice failure rate: ${((sliceStatus.failed_slices / sliceStatus.total_slices) * 100).toFixed(1)}%
` : ''}

### Hard Negative Testing Impact
- Total hard negatives generated: ${hardeningReport.hard_negatives.total_generated}
- Recall@10 degradation: ${hardeningReport.hard_negatives.impact_on_metrics?.degradation_percent?.toFixed(2) || 'N/A'}%
- Ranking robustness: ${(hardeningReport.hard_negatives.impact_on_metrics?.degradation_percent || 0) < 10 ? '✅ ROBUST' : '⚠️ SENSITIVE'}

## System Readiness Assessment

The lens system has been subjected to comprehensive hardening tests including:
- ⚡ **Tripwire validation**: Automated quality gates for span coverage, ranking diversity, and performance consistency
- 🎯 **Adversarial testing**: Hard negative injection to stress-test ranking robustness  
- 📊 **Slice validation**: Per-repository and per-language performance gates
- 📈 **Enhanced monitoring**: Comprehensive visualization suite for ongoing performance tracking

${hardeningReport.hardening_status === 'pass' 
  ? '✅ **RECOMMENDATION**: System is ready for production deployment with current hardening measures active.'
  : '❌ **RECOMMENDATION**: Address failing tripwires and slice issues before production deployment.'
}
`;

    return {
      title: 'Executive Summary',
      content: content.trim(),
      metadata: {
        overall_status: hardeningReport.hardening_status,
        timestamp: hardeningReport.timestamp,
        critical_failures: tripwireStatus.failed_tripwires + (sliceStatus?.failed_slices || 0)
      }
    };
  }

  /**
   * Build tripwire analysis section
   */
  private async buildTripwireAnalysis(tripwires: TripwireResult[]): Promise<PDFSection> {
    
    const failedTripwires = tripwires.filter(t => t.status === 'fail');
    const passedTripwires = tripwires.filter(t => t.status === 'pass');

    let content = `
# Tripwire Analysis

Tripwires are automated quality gates that enforce hard failure conditions to prevent system degradation.

## Tripwire Results Summary

| Tripwire | Status | Actual | Threshold | Description |
|----------|--------|--------|-----------|-------------|
`;

    for (const tripwire of tripwires) {
      const statusIcon = tripwire.status === 'pass' ? '✅' : '❌';
      const actualFormatted = this.formatTripwireValue(tripwire.name, tripwire.actual_value);
      const thresholdFormatted = this.formatTripwireValue(tripwire.name, tripwire.threshold);
      
      content += `| ${tripwire.name} | ${statusIcon} ${tripwire.status.toUpperCase()} | ${actualFormatted} | ${thresholdFormatted} | ${tripwire.description} |\n`;
    }

    if (failedTripwires.length > 0) {
      content += `\n## ❌ Failed Tripwires (${failedTripwires.length})\n\n`;
      
      for (const tripwire of failedTripwires) {
        content += `### ${tripwire.name}\n`;
        content += `- **Status**: FAILED\n`;
        content += `- **Actual Value**: ${this.formatTripwireValue(tripwire.name, tripwire.actual_value)}\n`;
        content += `- **Threshold**: ${this.formatTripwireValue(tripwire.name, tripwire.threshold)}\n`;
        content += `- **Impact**: ${this.describeTripwireImpact(tripwire.name)}\n`;
        content += `- **Remediation**: ${this.describeTripwireRemediation(tripwire.name)}\n\n`;
      }
    }

    if (passedTripwires.length > 0) {
      content += `\n## ✅ Passed Tripwires (${passedTripwires.length})\n\n`;
      content += `The following tripwires passed validation, indicating healthy system behavior:\n\n`;
      
      for (const tripwire of passedTripwires) {
        content += `- **${tripwire.name}**: ${this.formatTripwireValue(tripwire.name, tripwire.actual_value)} (threshold: ${this.formatTripwireValue(tripwire.name, tripwire.threshold)})\n`;
      }
    }

    return {
      title: 'Tripwire Analysis',
      content: content.trim(),
      data_tables: [tripwires],
      metadata: {
        total_tripwires: tripwires.length,
        failed_count: failedTripwires.length,
        failure_rate: (failedTripwires.length / tripwires.length) * 100
      }
    };
  }

  /**
   * Build performance analysis section
   */
  private async buildPerformanceAnalysis(
    benchmarkResults: BenchmarkRun[],
    hardeningReport: HardeningReport
  ): Promise<PDFSection> {
    
    const avgMetrics = this.calculateAverageMetrics(benchmarkResults);
    
    let content = `
# Performance Analysis

## Core Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Recall@10 | ${avgMetrics.recall_at_10.toFixed(3)} | ≥0.70 | ${avgMetrics.recall_at_10 >= 0.70 ? '✅' : '❌'} |
| Recall@50 | ${avgMetrics.recall_at_50.toFixed(3)} | ≥0.80 | ${avgMetrics.recall_at_50 >= 0.80 ? '✅' : '❌'} |
| nDCG@10 | ${avgMetrics.ndcg_at_10.toFixed(3)} | ≥0.60 | ${avgMetrics.ndcg_at_10 >= 0.60 ? '✅' : '❌'} |
| MRR | ${avgMetrics.mrr.toFixed(3)} | ≥0.50 | ${avgMetrics.mrr >= 0.50 ? '✅' : '❌'} |

## Latency Analysis

### Stage-by-Stage Performance

| Stage | P50 (ms) | P95 (ms) | Budget (ms) | Status |
|-------|----------|----------|-------------|--------|
| Stage A (Lexical) | ${avgMetrics.stage_latencies.stage_a_p50.toFixed(1)} | ${avgMetrics.stage_latencies.stage_a_p95.toFixed(1)} | 200 | ${avgMetrics.stage_latencies.stage_a_p95 <= 200 ? '✅' : '❌'} |
| Stage B (Symbolic) | ${avgMetrics.stage_latencies.stage_b_p50.toFixed(1)} | ${avgMetrics.stage_latencies.stage_b_p95.toFixed(1)} | 300 | ${avgMetrics.stage_latencies.stage_b_p95 <= 300 ? '✅' : '❌'} |
| Stage C (Semantic) | ${avgMetrics.stage_latencies.stage_c_p50?.toFixed(1) || 'N/A'} | ${avgMetrics.stage_latencies.stage_c_p95?.toFixed(1) || 'N/A'} | 300 | ${(avgMetrics.stage_latencies.stage_c_p95 || 0) <= 300 ? '✅' : '❌'} |
| **End-to-End** | **${avgMetrics.stage_latencies.e2e_p50.toFixed(1)}** | **${avgMetrics.stage_latencies.e2e_p95.toFixed(1)}** | **800** | **${avgMetrics.stage_latencies.e2e_p95 <= 800 ? '✅' : '❌'}** |

### Performance Insights

- **Stage A Efficiency**: ${this.assessStageEfficiency('stage_a', avgMetrics.stage_latencies.stage_a_p95)}
- **Stage B Efficiency**: ${this.assessStageEfficiency('stage_b', avgMetrics.stage_latencies.stage_b_p95)}
- **End-to-End Performance**: ${this.assessOverallPerformance(avgMetrics.stage_latencies.e2e_p95)}

## Fan-out Analysis

| Stage | Avg Candidates | Efficiency |
|-------|---------------|-------------|
| Stage A | ${avgMetrics.fan_out_sizes.stage_a} | ${this.assessFanOutEfficiency('stage_a', avgMetrics.fan_out_sizes.stage_a)} |
| Stage B | ${avgMetrics.fan_out_sizes.stage_b} | ${this.assessFanOutEfficiency('stage_b', avgMetrics.fan_out_sizes.stage_b)} |
| Stage C | ${avgMetrics.fan_out_sizes.stage_c || 'N/A'} | ${avgMetrics.fan_out_sizes.stage_c ? this.assessFanOutEfficiency('stage_c', avgMetrics.fan_out_sizes.stage_c) : 'N/A'} |
`;

    return {
      title: 'Performance Analysis',
      content: content.trim(),
      plots: ['latency_percentiles_by_stage'],
      data_tables: [benchmarkResults.map(r => r.metrics)],
      metadata: {
        avg_e2e_p95: avgMetrics.stage_latencies.e2e_p95,
        performance_score: this.calculatePerformanceScore(avgMetrics)
      }
    };
  }

  /**
   * Build hard negative analysis section
   */
  private async buildHardNegativeAnalysis(hardNegatives: HardeningReport['hard_negatives']): Promise<PDFSection> {
    
    let content = `
# Hard Negative Testing Analysis

Hard negative testing injects adversarial near-miss documents to stress-test ranking robustness.

## Hard Negative Generation Summary

- **Total Generated**: ${hardNegatives.total_generated} hard negatives
- **Average per Query**: ${(hardNegatives.total_generated / Object.keys(hardNegatives.per_query_stats).length).toFixed(1)}
- **Generation Strategies**: shared_class, shared_method, shared_variable, shared_imports

## Impact on Ranking Performance

| Metric | Baseline | With Hard Negatives | Degradation |
|--------|----------|-------------------|------------|
| Recall@10 | ${hardNegatives.impact_on_metrics.baseline_recall_at_10.toFixed(3)} | ${hardNegatives.impact_on_metrics.with_negatives_recall_at_10.toFixed(3)} | ${hardNegatives.impact_on_metrics.degradation_percent.toFixed(2)}% |

## Robustness Assessment

`;

    const degradation = hardNegatives.impact_on_metrics.degradation_percent;
    if (degradation < 5) {
      content += `✅ **EXCELLENT ROBUSTNESS**: ${degradation.toFixed(2)}% degradation indicates strong resistance to adversarial inputs.`;
    } else if (degradation < 10) {
      content += `🟡 **GOOD ROBUSTNESS**: ${degradation.toFixed(2)}% degradation is within acceptable bounds but could be improved.`;
    } else if (degradation < 20) {
      content += `⚠️ **MODERATE ROBUSTNESS**: ${degradation.toFixed(2)}% degradation indicates some sensitivity to hard negatives.`;
    } else {
      content += `❌ **POOR ROBUSTNESS**: ${degradation.toFixed(2)}% degradation indicates high sensitivity to adversarial inputs. Ranking improvements needed.`;
    }

    content += `\n\n## Per-Query Hard Negative Distribution\n\n`;
    
    const sortedStats = Object.entries(hardNegatives.per_query_stats)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10); // Top 10 queries

    content += `| Query ID | Hard Negatives Generated |\n|----------|------------------------|\n`;
    for (const [queryId, count] of sortedStats) {
      content += `| ${queryId} | ${count} |\n`;
    }

    return {
      title: 'Hard Negative Testing Analysis',
      content: content.trim(),
      data_tables: [hardNegatives],
      metadata: {
        degradation_percent: degradation,
        robustness_level: this.assessRobustnessLevel(degradation)
      }
    };
  }

  /**
   * Build slice analysis section
   */
  private async buildSliceAnalysis(sliceResults: SliceMetrics[]): Promise<PDFSection> {
    
    const failedSlices = sliceResults.filter(s => s.gate_status === 'fail');
    const passedSlices = sliceResults.filter(s => s.gate_status === 'pass');

    let content = `
# Per-Slice Performance Analysis

Analysis of performance across different repositories and programming languages.

## Slice Performance Summary

- **Total Slices**: ${sliceResults.length}
- **Passed Gates**: ${passedSlices.length} (${((passedSlices.length / sliceResults.length) * 100).toFixed(1)}%)
- **Failed Gates**: ${failedSlices.length} (${((failedSlices.length / sliceResults.length) * 100).toFixed(1)}%)

## Performance by Language

`;

    const languageStats = this.groupSlicesByLanguage(sliceResults);
    content += `| Language | Slices | Pass Rate | Avg Recall@10 | Avg nDCG@10 |\n`;
    content += `|----------|--------|-----------|---------------|-------------|\n`;

    for (const [language, slices] of languageStats.entries()) {
      const passRate = (slices.filter(s => s.gate_status === 'pass').length / slices.length) * 100;
      const avgRecall = slices.reduce((sum, s) => sum + s.metrics.recall_at_10, 0) / slices.length;
      const avgNdcg = slices.reduce((sum, s) => sum + s.metrics.ndcg_at_10, 0) / slices.length;
      
      content += `| ${language} | ${slices.length} | ${passRate.toFixed(1)}% | ${avgRecall.toFixed(3)} | ${avgNdcg.toFixed(3)} |\n`;
    }

    content += `\n## Performance by Repository\n\n`;

    const repoStats = this.groupSlicesByRepo(sliceResults);
    content += `| Repository | Slices | Pass Rate | Avg E2E P95 (ms) |\n`;
    content += `|------------|--------|-----------|------------------|\n`;

    for (const [repo, slices] of repoStats.entries()) {
      const passRate = (slices.filter(s => s.gate_status === 'pass').length / slices.length) * 100;
      const avgLatency = slices.reduce((sum, s) => sum + s.metrics.stage_latencies.e2e_p95, 0) / slices.length;
      
      content += `| ${repo} | ${slices.length} | ${passRate.toFixed(1)}% | ${avgLatency.toFixed(1)} |\n`;
    }

    if (failedSlices.length > 0) {
      content += `\n## ❌ Failed Slices (${failedSlices.length})\n\n`;
      
      for (const slice of failedSlices) {
        content += `### ${slice.slice_id}\n`;
        content += `- **Repository**: ${slice.repo}\n`;
        content += `- **Language**: ${slice.language}\n`;
        content += `- **Query Count**: ${slice.query_count}\n`;
        content += `- **Failing Criteria**:\n`;
        for (const criteria of slice.failing_criteria) {
          content += `  - ${criteria}\n`;
        }
        content += `\n`;
      }
    }

    return {
      title: 'Per-Slice Performance Analysis',
      content: content.trim(),
      data_tables: [sliceResults],
      metadata: {
        slice_failure_rate: (failedSlices.length / sliceResults.length) * 100,
        worst_performing_language: this.findWorstPerformingLanguage(languageStats),
        worst_performing_repo: this.findWorstPerformingRepo(repoStats)
      }
    };
  }

  /**
   * Build visualization gallery section
   */
  private async buildVisualizationGallery(plots: HardeningReport['plots_generated']): Promise<PDFSection> {
    
    let content = `
# Visualization Gallery

The following plots provide detailed insights into system performance and behavior.

## Available Visualizations

1. **Positives in Candidates**: Analysis of relevant documents within candidate sets
2. **Relevant Per Query Histogram**: Distribution of relevant results across queries
3. **Precision vs Score (Pre-Calibration)**: Precision analysis before score calibration
4. **Precision vs Score (Post-Calibration)**: Precision analysis after score calibration
5. **Latency Percentiles by Stage**: Performance breakdown across pipeline stages
6. **Early Termination Rate**: Analysis of pipeline termination patterns

> **Note**: Plot files are available in JSON format and can be rendered using visualization tools like D3.js, Matplotlib, or similar plotting libraries.

## Plot Files

`;

    for (const [plotName, plotPath] of Object.entries(plots)) {
      const formattedName = plotName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      content += `- **${formattedName}**: \`${plotPath}\`\n`;
    }

    return {
      title: 'Visualization Gallery',
      content: content.trim(),
      plots: Object.values(plots),
      metadata: {
        plot_count: Object.keys(plots).length,
        plot_types: Object.keys(plots)
      }
    };
  }

  /**
   * Build recommendations section
   */
  private async buildRecommendations(recommendations: string[]): Promise<PDFSection> {
    
    let content = `
# Recommendations & Action Items

Based on the hardening analysis, the following recommendations have been identified:

## Priority Actions

`;

    recommendations.forEach((rec, index) => {
      const priority = index < 2 ? '🔴 HIGH' : index < 4 ? '🟡 MEDIUM' : '🟢 LOW';
      content += `### ${index + 1}. ${priority}\n\n${rec}\n\n`;
    });

    if (recommendations.length === 0) {
      content += `✅ **No critical issues identified**\n\nThe system has passed all hardening checks and is ready for production deployment.`;
    }

    content += `\n## Implementation Timeline\n\n`;
    content += `- **Immediate (Next Sprint)**: Address HIGH priority items\n`;
    content += `- **Short-term (Next Release)**: Address MEDIUM priority items\n`;
    content += `- **Long-term (Future Releases)**: Address LOW priority items and optimizations\n`;

    return {
      title: 'Recommendations & Action Items',
      content: content.trim(),
      metadata: {
        total_recommendations: recommendations.length,
        high_priority_count: Math.min(2, recommendations.length),
        medium_priority_count: Math.max(0, Math.min(2, recommendations.length - 2)),
        low_priority_count: Math.max(0, recommendations.length - 4)
      }
    };
  }

  /**
   * Build raw data appendix
   */
  private async buildRawDataAppendix(
    hardeningReport: HardeningReport,
    benchmarkResults: BenchmarkRun[]
  ): Promise<PDFSection> {
    
    const content = `
# Appendix: Raw Data

## Configuration

\`\`\`json
${JSON.stringify(hardeningReport.config, null, 2)}
\`\`\`

## Benchmark Results Summary

- **Total Benchmark Runs**: ${benchmarkResults.length}
- **Systems Tested**: ${[...new Set(benchmarkResults.map(r => r.system))].join(', ')}
- **Report Generation Time**: ${hardeningReport.timestamp}

## Data Files

Raw data files are available in the output directory for further analysis:

- \`phase-c-hardening-report.json\`: Complete hardening report
- \`*_metrics.parquet.json\`: Per-run metrics data
- \`*_errors.ndjson\`: Error logs
- \`*_traces.ndjson\`: Performance traces
- \`hardening-plots/*.json\`: Visualization data

> **Note**: Use these files for custom analysis, debugging, or integration with external monitoring systems.
`;

    return {
      title: 'Appendix: Raw Data',
      content: content.trim(),
      data_tables: [benchmarkResults],
      metadata: {
        config_hash: this.generateConfigHash(hardeningReport.config),
        data_file_count: Object.keys(hardeningReport.plots_generated).length + 4
      }
    };
  }

  /**
   * Assemble all sections into final report
   */
  private assembleReport(sections: PDFSection[], config: PDFReportConfig): string {
    
    const header = `
---
title: "${config.title}"
subtitle: "${config.subtitle || 'Phase C Hardening Analysis'}"
author: "${config.author}"
date: "${new Date().toISOString().split('T')[0]}"
geometry: margin=1in
fontsize: 11pt
---

`;

    let content = header;

    for (const section of sections) {
      content += section.content + '\n\n---\n\n';
    }

    // Add metadata footer
    content += `
---

*Report generated on ${new Date().toISOString()} using lens Phase C Hardening Suite*
`;

    return content;
  }

  /**
   * Write report to file
   */
  private async writeReport(content: string, config: PDFReportConfig): Promise<string> {
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `lens-hardening-report-${timestamp}.${config.output_format}`;
    const outputPath = path.join(this.outputDir, filename);

    await fs.writeFile(outputPath, content, 'utf-8');
    
    return outputPath;
  }

  // Helper methods for formatting and analysis
  
  private formatTripwireValue(tripwireName: string, value: number): string {
    switch (tripwireName) {
      case 'span_coverage':
      case 'lsif_coverage_drop':
      case 'recall_convergence':
        return `${(value * 100).toFixed(2)}%`;
      case 'p99_p95_ratio':
        return `${value.toFixed(2)}×`;
      default:
        return value.toFixed(3);
    }
  }

  private describeTripwireImpact(tripwireName: string): string {
    const impacts = {
      'span_coverage': 'Reduced relevance quality - users may not find expected results',
      'recall_convergence': 'Poor ranking diversity - limited benefit from deeper search',
      'lsif_coverage_drop': 'Degraded symbol search quality - structural queries less effective',
      'p99_p95_ratio': 'Inconsistent tail latency - poor user experience for some queries'
    };
    return impacts[tripwireName as keyof typeof impacts] || 'Unknown impact';
  }

  private describeTripwireRemediation(tripwireName: string): string {
    const remediations = {
      'span_coverage': 'Review indexing pipeline and golden dataset alignment',
      'recall_convergence': 'Improve candidate generation and ranking diversity',
      'lsif_coverage_drop': 'Investigate symbol extraction and LSIF generation issues',
      'p99_p95_ratio': 'Identify and fix tail latency outliers in query processing'
    };
    return remediations[tripwireName as keyof typeof remediations] || 'Review system configuration';
  }

  private calculateAverageMetrics(benchmarkResults: BenchmarkRun[]): BenchmarkRun['metrics'] {
    if (benchmarkResults.length === 0) {
      throw new Error('Cannot calculate metrics for empty benchmark results');
    }

    const sum = benchmarkResults.reduce((acc, result) => ({
      recall_at_10: acc.recall_at_10 + result.metrics.recall_at_10,
      recall_at_50: acc.recall_at_50 + result.metrics.recall_at_50,
      ndcg_at_10: acc.ndcg_at_10 + result.metrics.ndcg_at_10,
      mrr: acc.mrr + result.metrics.mrr,
      first_relevant_tokens: acc.first_relevant_tokens + result.metrics.first_relevant_tokens,
      stage_latencies: {
        stage_a_p50: acc.stage_latencies.stage_a_p50 + result.metrics.stage_latencies.stage_a_p50,
        stage_a_p95: acc.stage_latencies.stage_a_p95 + result.metrics.stage_latencies.stage_a_p95,
        stage_b_p50: acc.stage_latencies.stage_b_p50 + result.metrics.stage_latencies.stage_b_p50,
        stage_b_p95: acc.stage_latencies.stage_b_p95 + result.metrics.stage_latencies.stage_b_p95,
        stage_c_p50: (acc.stage_latencies.stage_c_p50 || 0) + (result.metrics.stage_latencies.stage_c_p50 || 0),
        stage_c_p95: (acc.stage_latencies.stage_c_p95 || 0) + (result.metrics.stage_latencies.stage_c_p95 || 0),
        e2e_p50: acc.stage_latencies.e2e_p50 + result.metrics.stage_latencies.e2e_p50,
        e2e_p95: acc.stage_latencies.e2e_p95 + result.metrics.stage_latencies.e2e_p95
      },
      fan_out_sizes: {
        stage_a: acc.fan_out_sizes.stage_a + result.metrics.fan_out_sizes.stage_a,
        stage_b: acc.fan_out_sizes.stage_b + result.metrics.fan_out_sizes.stage_b,
        stage_c: (acc.fan_out_sizes.stage_c || 0) + (result.metrics.fan_out_sizes.stage_c || 0)
      },
      why_attributions: {} // Not averaged for simplicity
    }), {
      recall_at_10: 0, recall_at_50: 0, ndcg_at_10: 0, mrr: 0, first_relevant_tokens: 0,
      stage_latencies: { stage_a_p50: 0, stage_a_p95: 0, stage_b_p50: 0, stage_b_p95: 0, stage_c_p50: 0, stage_c_p95: 0, e2e_p50: 0, e2e_p95: 0 },
      fan_out_sizes: { stage_a: 0, stage_b: 0, stage_c: 0 },
      why_attributions: {}
    });

    const count = benchmarkResults.length;
    return {
      recall_at_10: sum.recall_at_10 / count,
      recall_at_50: sum.recall_at_50 / count,
      ndcg_at_10: sum.ndcg_at_10 / count,
      mrr: sum.mrr / count,
      first_relevant_tokens: sum.first_relevant_tokens / count,
      stage_latencies: {
        stage_a_p50: sum.stage_latencies.stage_a_p50 / count,
        stage_a_p95: sum.stage_latencies.stage_a_p95 / count,
        stage_b_p50: sum.stage_latencies.stage_b_p50 / count,
        stage_b_p95: sum.stage_latencies.stage_b_p95 / count,
        stage_c_p50: sum.stage_latencies.stage_c_p50 > 0 ? sum.stage_latencies.stage_c_p50 / count : undefined,
        stage_c_p95: sum.stage_latencies.stage_c_p95 > 0 ? sum.stage_latencies.stage_c_p95 / count : undefined,
        e2e_p50: sum.stage_latencies.e2e_p50 / count,
        e2e_p95: sum.stage_latencies.e2e_p95 / count
      },
      fan_out_sizes: {
        stage_a: Math.round(sum.fan_out_sizes.stage_a / count),
        stage_b: Math.round(sum.fan_out_sizes.stage_b / count),
        stage_c: sum.fan_out_sizes.stage_c > 0 ? Math.round(sum.fan_out_sizes.stage_c / count) : undefined
      },
      why_attributions: {}
    };
  }

  private assessStageEfficiency(stage: string, p95: number): string {
    const budgets = { stage_a: 200, stage_b: 300, stage_c: 300 };
    const budget = budgets[stage as keyof typeof budgets];
    const utilization = (p95 / budget) * 100;
    
    if (utilization < 50) return `Excellent (${utilization.toFixed(1)}% budget used)`;
    if (utilization < 75) return `Good (${utilization.toFixed(1)}% budget used)`;
    if (utilization < 95) return `Acceptable (${utilization.toFixed(1)}% budget used)`;
    return `Over budget (${utilization.toFixed(1)}% budget used)`;
  }

  private assessOverallPerformance(e2eP95: number): string {
    if (e2eP95 < 400) return 'Excellent - Well under budget';
    if (e2eP95 < 600) return 'Good - Within acceptable limits';
    if (e2eP95 < 800) return 'Acceptable - Near budget limits';
    return 'Poor - Over budget';
  }

  private assessFanOutEfficiency(stage: string, avgCandidates: number): string {
    const targets = { stage_a: 100, stage_b: 50, stage_c: 200 };
    const target = targets[stage as keyof typeof targets];
    const ratio = avgCandidates / target;
    
    if (ratio < 0.5) return 'Low';
    if (ratio < 1.2) return 'Optimal';
    if (ratio < 2.0) return 'High';
    return 'Excessive';
  }

  private calculatePerformanceScore(metrics: BenchmarkRun['metrics']): number {
    // Weighted composite score
    const weights = { recall: 0.3, ndcg: 0.3, latency: 0.4 };
    
    const recallScore = Math.min(1, metrics.recall_at_10 / 0.8);
    const ndcgScore = Math.min(1, metrics.ndcg_at_10 / 0.7);
    const latencyScore = Math.max(0, 1 - (metrics.stage_latencies.e2e_p95 / 800));
    
    return (weights.recall * recallScore + weights.ndcg * ndcgScore + weights.latency * latencyScore) * 100;
  }

  private assessRobustnessLevel(degradationPercent: number): string {
    if (degradationPercent < 5) return 'excellent';
    if (degradationPercent < 10) return 'good';
    if (degradationPercent < 20) return 'moderate';
    return 'poor';
  }

  private groupSlicesByLanguage(slices: SliceMetrics[]): Map<string, SliceMetrics[]> {
    const grouped = new Map<string, SliceMetrics[]>();
    for (const slice of slices) {
      const language = slice.language || 'unknown';
      if (!grouped.has(language)) grouped.set(language, []);
      grouped.get(language)!.push(slice);
    }
    return grouped;
  }

  private groupSlicesByRepo(slices: SliceMetrics[]): Map<string, SliceMetrics[]> {
    const grouped = new Map<string, SliceMetrics[]>();
    for (const slice of slices) {
      const repo = slice.repo || 'unknown';
      if (!grouped.has(repo)) grouped.set(repo, []);
      grouped.get(repo)!.push(slice);
    }
    return grouped;
  }

  private findWorstPerformingLanguage(languageStats: Map<string, SliceMetrics[]>): string {
    let worstLanguage = 'none';
    let worstPassRate = 100;
    
    for (const [language, slices] of languageStats.entries()) {
      const passRate = (slices.filter(s => s.gate_status === 'pass').length / slices.length) * 100;
      if (passRate < worstPassRate) {
        worstPassRate = passRate;
        worstLanguage = language;
      }
    }
    
    return worstLanguage;
  }

  private findWorstPerformingRepo(repoStats: Map<string, SliceMetrics[]>): string {
    let worstRepo = 'none';
    let worstPassRate = 100;
    
    for (const [repo, slices] of repoStats.entries()) {
      const passRate = (slices.filter(s => s.gate_status === 'pass').length / slices.length) * 100;
      if (passRate < worstPassRate) {
        worstPassRate = passRate;
        worstRepo = repo;
      }
    }
    
    return worstRepo;
  }

  private generateConfigHash(config: any): string {
    return Buffer.from(JSON.stringify(config)).toString('base64').slice(0, 12);
  }
}