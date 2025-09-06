/**
 * Gap vs Serena Calculation Fix
 * 
 * Addresses critical issue: "Fix the 'Gap vs Serena' column (compute Lens−Serena in percentage points)"
 * The calculation should show +3.5pp, not -7.1pp based on nDCG 0.815 vs 0.780
 */

export interface SystemMetrics {
  system: string;
  nDCG_10: number;
  P_at_1: number;
  Success_at_10: number;
  Recall_50: number;
  p95_latency: number;
  QPS_150ms: number;
  error_rate: number;
}

export interface GapCalculation {
  metric: string;
  lens_value: number;
  serena_value: number;
  absolute_gap: number;
  percentage_points: number;
  relative_improvement: number;
  direction: 'improvement' | 'degradation';
}

export interface GapAnalysisResult {
  overall_assessment: 'leading' | 'competitive' | 'underperforming';
  gaps: GapCalculation[];
  summary: {
    quality_wins: number;
    performance_wins: number;
    total_metrics: number;
    net_score: number;
  };
}

export class GapCalculator {
  private lensMetrics: SystemMetrics;
  private serenaMetrics: SystemMetrics;

  constructor(lensMetrics: SystemMetrics, serenaMetrics: SystemMetrics) {
    this.lensMetrics = lensMetrics;
    this.serenaMetrics = serenaMetrics;
  }

  /**
   * Calculate gaps between Lens and Serena with correct sign convention
   * Positive gap = Lens outperforms Serena
   * Negative gap = Serena outperforms Lens
   */
  calculateGaps(): GapAnalysisResult {
    const gaps: GapCalculation[] = [
      this.calculateMetricGap('nDCG@10', this.lensMetrics.nDCG_10, this.serenaMetrics.nDCG_10, 'higher_better'),
      this.calculateMetricGap('P@1', this.lensMetrics.P_at_1, this.serenaMetrics.P_at_1, 'higher_better'),
      this.calculateMetricGap('Success@10', this.lensMetrics.Success_at_10, this.serenaMetrics.Success_at_10, 'higher_better'),
      this.calculateMetricGap('Recall@50', this.lensMetrics.Recall_50, this.serenaMetrics.Recall_50, 'higher_better'),
      this.calculateMetricGap('p95_latency', this.lensMetrics.p95_latency, this.serenaMetrics.p95_latency, 'lower_better'),
      this.calculateMetricGap('QPS@150ms', this.lensMetrics.QPS_150ms, this.serenaMetrics.QPS_150ms, 'higher_better'),
      this.calculateMetricGap('error_rate', this.lensMetrics.error_rate, this.serenaMetrics.error_rate, 'lower_better')
    ];

    // Calculate summary statistics
    const qualityMetrics = ['nDCG@10', 'P@1', 'Success@10', 'Recall@50'];
    const performanceMetrics = ['p95_latency', 'QPS@150ms', 'error_rate'];

    const quality_wins = gaps.filter(g => 
      qualityMetrics.includes(g.metric) && g.direction === 'improvement'
    ).length;

    const performance_wins = gaps.filter(g => 
      performanceMetrics.includes(g.metric) && g.direction === 'improvement'
    ).length;

    // Net score: +1 for improvement, -1 for degradation
    const net_score = gaps.reduce((sum, gap) => 
      sum + (gap.direction === 'improvement' ? 1 : -1), 0
    );

    // Overall assessment
    let overall_assessment: 'leading' | 'competitive' | 'underperforming';
    if (net_score >= 3 && quality_wins >= 2) {
      overall_assessment = 'leading';
    } else if (net_score >= 0) {
      overall_assessment = 'competitive';
    } else {
      overall_assessment = 'underperforming';
    }

    return {
      overall_assessment,
      gaps,
      summary: {
        quality_wins,
        performance_wins,
        total_metrics: gaps.length,
        net_score
      }
    };
  }

  /**
   * Get the fixed Gap vs Serena for nDCG@10 specifically
   * This is the critical fix: should be +3.5pp, not -7.1pp
   */
  getFixedNDCGGap(): GapCalculation {
    return this.calculateMetricGap('nDCG@10', this.lensMetrics.nDCG_10, this.serenaMetrics.nDCG_10, 'higher_better');
  }

  private calculateMetricGap(
    metric: string, 
    lensValue: number, 
    serenaValue: number, 
    direction: 'higher_better' | 'lower_better'
  ): GapCalculation {
    // Correct sign convention: Lens - Serena
    const absolute_gap = lensValue - serenaValue;
    
    // Convert to percentage points (multiply by 100)
    const percentage_points = absolute_gap * 100;
    
    // Relative improvement percentage
    const relative_improvement = serenaValue === 0 ? 0 : (absolute_gap / serenaValue) * 100;
    
    // Determine if this is an improvement or degradation
    let is_improvement: boolean;
    if (direction === 'higher_better') {
      is_improvement = absolute_gap > 0; // Lens > Serena is good
    } else {
      is_improvement = absolute_gap < 0; // Lens < Serena is good (lower latency, error rate)
    }

    return {
      metric,
      lens_value: lensValue,
      serena_value: serenaValue,
      absolute_gap,
      percentage_points,
      relative_improvement,
      direction: is_improvement ? 'improvement' : 'degradation'
    };
  }

  /**
   * Generate formatted gap report
   */
  generateGapReport(): string {
    const analysis = this.calculateGaps();
    
    let report = '# Gap vs Serena Analysis Report\n\n';
    report += `Overall Assessment: **${analysis.overall_assessment.toUpperCase()}**\n\n`;
    
    report += '## Detailed Gap Analysis\n\n';
    report += '| Metric | Lens | Serena | Gap (pp) | Relative (%) | Direction |\n';
    report += '|--------|------|--------|----------|--------------|----------|\n';
    
    for (const gap of analysis.gaps) {
      const directionSymbol = gap.direction === 'improvement' ? '✅' : '❌';
      report += `| ${gap.metric} | ${gap.lens_value.toFixed(3)} | ${gap.serena_value.toFixed(3)} | `;
      report += `${gap.percentage_points >= 0 ? '+' : ''}${gap.percentage_points.toFixed(1)} | `;
      report += `${gap.relative_improvement >= 0 ? '+' : ''}${gap.relative_improvement.toFixed(1)}% | `;
      report += `${directionSymbol} ${gap.direction} |\n`;
    }
    
    report += '\n## Summary\n\n';
    report += `- Quality wins: ${analysis.summary.quality_wins}/4\n`;
    report += `- Performance wins: ${analysis.summary.performance_wins}/3\n`;
    report += `- Net score: ${analysis.summary.net_score}/${analysis.summary.total_metrics}\n\n`;
    
    // Highlight the critical nDCG fix
    const ndcgGap = analysis.gaps.find(g => g.metric === 'nDCG@10');
    if (ndcgGap) {
      report += '## Critical Fix: nDCG@10 Gap\n\n';
      report += `**CORRECTED CALCULATION:**\n`;
      report += `- Lens nDCG@10: ${ndcgGap.lens_value.toFixed(3)}\n`;
      report += `- Serena nDCG@10: ${ndcgGap.serena_value.toFixed(3)}\n`;
      report += `- Gap: ${ndcgGap.percentage_points >= 0 ? '+' : ''}${ndcgGap.percentage_points.toFixed(1)}pp\n`;
      report += `- Status: ${ndcgGap.direction === 'improvement' ? '✅ Lens leads' : '❌ Serena leads'}\n\n`;
      
      if (Math.abs(ndcgGap.percentage_points - 3.5) < 0.5) {
        report += '✅ **VERIFICATION PASSED**: Gap matches expected +3.5pp\n';
      } else {
        report += `⚠️ **VERIFICATION NEEDED**: Expected ~+3.5pp, got ${ndcgGap.percentage_points.toFixed(1)}pp\n`;
      }
    }
    
    return report;
  }
}

/**
 * Factory function to create GapCalculator from raw metrics
 */
export function createGapCalculator(
  lensMetrics: Partial<SystemMetrics>, 
  serenaMetrics: Partial<SystemMetrics>
): GapCalculator {
  // Fill in defaults for missing metrics
  const defaultMetrics: SystemMetrics = {
    system: '',
    nDCG_10: 0,
    P_at_1: 0,
    Success_at_10: 0,
    Recall_50: 0,
    p95_latency: 0,
    QPS_150ms: 0,
    error_rate: 0
  };

  const fullLensMetrics = { ...defaultMetrics, ...lensMetrics, system: 'lens' };
  const fullSerenaMetrics = { ...defaultMetrics, ...serenaMetrics, system: 'serena' };

  return new GapCalculator(fullLensMetrics, fullSerenaMetrics);
}

/**
 * Demo function with corrected values showing +3.5pp gap
 */
export function demoCorrectCalculation(): GapCalculator {
  const lensMetrics: SystemMetrics = {
    system: 'lens',
    nDCG_10: 0.815,  // As stated in TODO.md
    P_at_1: 0.452,
    Success_at_10: 0.527,
    Recall_50: 0.683,
    p95_latency: 142,
    QPS_150ms: 1.2,
    error_rate: 0.023
  };

  const serenaMetrics: SystemMetrics = {
    system: 'serena', 
    nDCG_10: 0.780,  // As stated in TODO.md
    P_at_1: 0.402,
    Success_at_10: 0.452,
    Recall_50: 0.654,
    p95_latency: 152,
    QPS_150ms: 1.0,
    error_rate: 0.031
  };

  return new GapCalculator(lensMetrics, serenaMetrics);
}

// CLI execution and verification
if (import.meta.main) {
  console.log('🔧 Gap vs Serena Calculation Fix\n');
  
  const calculator = demoCorrectCalculation();
  const report = calculator.generateGapReport();
  
  console.log(report);
  
  // Verify the critical fix
  const ndcgGap = calculator.getFixedNDCGGap();
  const expectedGap = 3.5; // +3.5pp as stated in TODO.md
  const actualGap = ndcgGap.percentage_points;
  
  console.log('\n🔍 VERIFICATION:');
  console.log(`Expected nDCG@10 gap: +${expectedGap}pp`);
  console.log(`Calculated gap: ${actualGap >= 0 ? '+' : ''}${actualGap.toFixed(1)}pp`);
  
  if (Math.abs(actualGap - expectedGap) < 0.1) {
    console.log('✅ CALCULATION FIXED: Gap correctly shows +3.5pp');
  } else {
    console.log('❌ CALCULATION ISSUE: Gap does not match expected value');
  }
}