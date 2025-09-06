/**
 * Slice-Wise Miscoverage Audit - Mondrian Conformal Prediction System
 * 
 * Implements comprehensive miscoverage auditing with:
 * 1. Mondrian slicing by {intent×lang×entropy_bin}
 * 2. Per-slice miscoverage ≤ target+1.5pp enforcement  
 * 3. Wilson Confidence Interval bounds checking
 * 4. Expected Calibration Error (ECE) monitoring
 * 5. Automated spend reduction and alerting on violations
 */

import { EventEmitter } from 'events';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';

export type Intent = 'NL' | 'symbol' | 'mixed';
export type Language = 'typescript' | 'python' | 'rust' | 'go' | 'javascript';  
export type EntropyBin = 'low' | 'medium' | 'high';

export interface MondrianSlice {
  intent: Intent;
  language: Language;
  entropy_bin: EntropyBin;
  slice_id: string; // "NL_typescript_high"
}

export interface ConformalPrediction {
  query_id: string;
  slice: MondrianSlice;
  confidence_score: number; // Model's confidence [0,1]
  prediction_set_size: number; // Size of conformal prediction set
  ground_truth_rank: number; // Actual rank of correct result
  is_covered: boolean; // Whether ground truth is in prediction set
  timestamp: Date;
}

export interface SliceCoverageMetrics {
  slice: MondrianSlice;
  sample_count: number;
  empirical_coverage: number; // Actual coverage rate
  target_coverage: number; // Desired coverage rate (e.g., 90%)
  miscoverage: number; // target - empirical
  wilson_ci_lower: number; // Wilson CI lower bound
  wilson_ci_upper: number; // Wilson CI upper bound
  ece_score: number; // Expected Calibration Error
  violation_detected: boolean; // miscoverage > target+1.5pp
  spend_reduction_triggered: boolean; // Auto-reduce spend by 25%
}

export interface MiscoverageAuditReport {
  timestamp: Date;
  total_slices: number;
  slices_with_violations: number;
  overall_miscoverage: number;
  overall_ece: number;
  slice_metrics: SliceCoverageMetrics[];
  violations: {
    slice_id: string;
    miscoverage: number;
    ece_score: number;
    recommended_action: string;
  }[];
  spend_adjustments: {
    slice_id: string;
    old_spend_percentage: number;
    new_spend_percentage: number;
    reduction_amount: number;
  }[];
  alerts: string[];
  recommendations: string[];
}

export interface MiscoverageAuditConfig {
  target_coverage: number; // e.g., 0.90 for 90% coverage
  miscoverage_tolerance: number; // +1.5pp tolerance
  ece_threshold: number; // ECE threshold (0.01)
  wilson_ci_alpha: number; // Wilson CI significance level (0.05)
  min_samples_per_slice: number; // Minimum samples for reliable estimates
  auto_spend_reduction: number; // 25% reduction on violation
  alert_threshold: number; // Alert if >X% of slices violate
}

export const DEFAULT_MISCOVERAGE_CONFIG: MiscoverageAuditConfig = {
  target_coverage: 0.90, // 90% target coverage
  miscoverage_tolerance: 1.5, // ±1.5pp tolerance
  ece_threshold: 0.01, // |ΔECE| ≤ 0.01
  wilson_ci_alpha: 0.05, // 95% confidence intervals
  min_samples_per_slice: 50, // Minimum samples for analysis
  auto_spend_reduction: 25, // 25% spend reduction
  alert_threshold: 0.15 // Alert if >15% slices violate
};

export class SliceWiseMiscoverageAuditor extends EventEmitter {
  private config: MiscoverageAuditConfig;
  private predictions: ConformalPrediction[] = [];
  private sliceSpendMap: Map<string, number> = new Map(); // Track spend per slice
  
  constructor(config: MiscoverageAuditConfig = DEFAULT_MISCOVERAGE_CONFIG) {
    super();
    this.config = config;
    
    // Initialize default spend percentages per slice
    this.initializeSliceSpending();
  }
  
  /**
   * Initialize default spending allocation across slices
   */
  private initializeSliceSpending(): void {
    const intents: Intent[] = ['NL', 'symbol', 'mixed'];
    const languages: Language[] = ['typescript', 'python', 'rust', 'go', 'javascript'];
    const entropyBins: EntropyBin[] = ['low', 'medium', 'high'];
    
    // Default 5% spend, adjusted by slice characteristics
    for (const intent of intents) {
      for (const language of languages) {
        for (const entropyBin of entropyBins) {
          const sliceId = `${intent}_${language}_${entropyBin}`;
          
          // Base spend allocation
          let spend = 5.0; // 5% base
          
          // Adjust by intent (NL gets more budget)
          if (intent === 'NL') spend += 1.0;
          else if (intent === 'symbol') spend -= 0.5;
          
          // Adjust by entropy (high entropy gets more budget) 
          if (entropyBin === 'high') spend += 2.0;
          else if (entropyBin === 'low') spend -= 1.0;
          
          // Language-specific adjustments
          if (language === 'typescript' || language === 'python') spend += 0.5;
          
          this.sliceSpendMap.set(sliceId, Math.max(1.0, spend)); // Min 1%
        }
      }
    }
  }
  
  /**
   * Ingest conformal predictions for analysis
   */
  async ingestPredictions(predictions: ConformalPrediction[]): Promise<void> {
    console.log(`📊 Ingesting ${predictions.length} conformal predictions for slice analysis...`);
    
    this.predictions = predictions;
    
    // Validate slice coverage
    const sliceDistribution = this.getSliceDistribution();
    const underrepresentedSlices = Object.entries(sliceDistribution)
      .filter(([_, count]) => count < this.config.min_samples_per_slice)
      .map(([slice, count]) => `${slice}: ${count} samples`);
    
    if (underrepresentedSlices.length > 0) {
      console.warn(`⚠️  Underrepresented slices detected: ${underrepresentedSlices.join(', ')}`);
    }
    
    console.log(`Slice distribution: ${Object.keys(sliceDistribution).length} unique slices`);
    this.emit('predictions_ingested', { total: predictions.length, slices: Object.keys(sliceDistribution).length });
  }
  
  /**
   * Execute comprehensive slice-wise miscoverage audit
   */
  async executeAudit(outputDir: string): Promise<MiscoverageAuditReport> {
    console.log('🔍 Executing slice-wise miscoverage audit with Mondrian analysis...');
    
    if (this.predictions.length === 0) {
      throw new Error('No predictions loaded. Call ingestPredictions() first.');
    }
    
    await mkdir(outputDir, { recursive: true });
    
    // Group predictions by Mondrian slices
    const sliceGroups = this.groupByMondrianSlices();
    console.log(`📋 Analyzing ${sliceGroups.size} Mondrian slices...`);
    
    const sliceMetrics: SliceCoverageMetrics[] = [];
    const violations: any[] = [];
    const spendAdjustments: any[] = [];
    const alerts: string[] = [];
    
    // Analyze each slice independently
    for (const [sliceId, slicePredictions] of sliceGroups) {
      const metrics = await this.analyzeSlice(sliceId, slicePredictions);
      sliceMetrics.push(metrics);
      
      // Check for violations
      if (metrics.violation_detected) {
        violations.push({
          slice_id: sliceId,
          miscoverage: metrics.miscoverage,
          ece_score: metrics.ece_score,
          recommended_action: this.getRecommendedAction(metrics)
        });
        
        // Auto-reduce spend
        if (metrics.spend_reduction_triggered) {
          const oldSpend = this.sliceSpendMap.get(sliceId) || 5.0;
          const newSpend = oldSpend * (1 - this.config.auto_spend_reduction / 100);
          const reduction = oldSpend - newSpend;
          
          this.sliceSpendMap.set(sliceId, newSpend);
          
          spendAdjustments.push({
            slice_id: sliceId,
            old_spend_percentage: oldSpend,
            new_spend_percentage: newSpend,
            reduction_amount: reduction
          });
          
          alerts.push(`SPEND REDUCED: ${sliceId} from ${oldSpend.toFixed(1)}% to ${newSpend.toFixed(1)}% (${reduction.toFixed(1)}pp reduction)`);
        }
      }
    }
    
    // Compute overall metrics
    const totalPredictions = this.predictions.length;
    const totalCovered = this.predictions.filter(p => p.is_covered).length;
    const overallCoverage = totalCovered / totalPredictions;
    const overallMiscoverage = (this.config.target_coverage - overallCoverage) * 100; // Convert to pp
    const overallECE = this.computeOverallECE();
    
    // Generate alerts for system-wide issues
    const violationRate = violations.length / sliceMetrics.length;
    if (violationRate > this.config.alert_threshold) {
      alerts.push(`CRITICAL: ${(violationRate * 100).toFixed(1)}% of slices exceed miscoverage tolerance`);
    }
    
    if (Math.abs(overallECE) > this.config.ece_threshold) {
      alerts.push(`ECE VIOLATION: Overall ECE = ${overallECE.toFixed(3)} exceeds threshold ${this.config.ece_threshold}`);
    }
    
    const report: MiscoverageAuditReport = {
      timestamp: new Date(),
      total_slices: sliceMetrics.length,
      slices_with_violations: violations.length,
      overall_miscoverage: overallMiscoverage,
      overall_ece: overallECE,
      slice_metrics: sliceMetrics,
      violations,
      spend_adjustments: spendAdjustments,
      alerts,
      recommendations: this.generateRecommendations(sliceMetrics, violations, overallMiscoverage, overallECE)
    };
    
    // Save audit results
    await this.saveAuditReport(report, outputDir);
    
    console.log(`✅ Miscoverage audit completed: ${violations.length}/${sliceMetrics.length} slices with violations`);
    console.log(`Overall miscoverage: ${overallMiscoverage.toFixed(2)}pp, ECE: ${overallECE.toFixed(3)}`);
    
    this.emit('audit_completed', report);
    return report;
  }
  
  /**
   * Group predictions by Mondrian slices
   */
  private groupByMondrianSlices(): Map<string, ConformalPrediction[]> {
    const groups = new Map<string, ConformalPrediction[]>();
    
    for (const prediction of this.predictions) {
      const sliceId = prediction.slice.slice_id;
      
      if (!groups.has(sliceId)) {
        groups.set(sliceId, []);
      }
      groups.get(sliceId)!.push(prediction);
    }
    
    return groups;
  }
  
  /**
   * Analyze individual Mondrian slice for miscoverage
   */
  private async analyzeSlice(sliceId: string, predictions: ConformalPrediction[]): Promise<SliceCoverageMetrics> {
    const sampleCount = predictions.length;
    
    // Skip analysis if insufficient samples
    if (sampleCount < this.config.min_samples_per_slice) {
      return this.createEmptySliceMetrics(sliceId, sampleCount);
    }
    
    // Compute empirical coverage
    const coveredPredictions = predictions.filter(p => p.is_covered);
    const empiricalCoverage = coveredPredictions.length / sampleCount;
    const miscoverage = (this.config.target_coverage - empiricalCoverage) * 100; // Convert to pp
    
    // Compute Wilson Confidence Interval
    const wilsonCI = this.computeWilsonCI(coveredPredictions.length, sampleCount, this.config.wilson_ci_alpha);
    
    // Compute Expected Calibration Error for this slice
    const eceScore = this.computeSliceECE(predictions);
    
    // Check for violations
    const miscoverageViolation = Math.abs(miscoverage) > this.config.miscoverage_tolerance;
    const eceViolation = Math.abs(eceScore) > this.config.ece_threshold;
    const violationDetected = miscoverageViolation || eceViolation;
    
    // Determine if spend reduction should be triggered
    const spendReductionTriggered = violationDetected && (
      miscoverage > this.config.miscoverage_tolerance || // Significant undercoverage
      Math.abs(eceScore) > this.config.ece_threshold * 2 // Severe calibration error
    );
    
    const slice = this.parseSliceId(sliceId);
    
    return {
      slice,
      sample_count: sampleCount,
      empirical_coverage: empiricalCoverage,
      target_coverage: this.config.target_coverage,
      miscoverage,
      wilson_ci_lower: wilsonCI.lower,
      wilson_ci_upper: wilsonCI.upper,
      ece_score: eceScore,
      violation_detected: violationDetected,
      spend_reduction_triggered: spendReductionTriggered
    };
  }
  
  /**
   * Compute Wilson Confidence Interval for binomial proportion
   */
  private computeWilsonCI(successes: number, n: number, alpha: number): { lower: number; upper: number } {
    const z = this.getZScore(alpha / 2); // Two-tailed
    const p = successes / n;
    
    const denominator = 1 + (z * z) / n;
    const center = p + (z * z) / (2 * n);
    const halfWidth = z * Math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n);
    
    return {
      lower: Math.max(0, (center - halfWidth) / denominator),
      upper: Math.min(1, (center + halfWidth) / denominator)
    };
  }
  
  /**
   * Get Z-score for given alpha (two-tailed)
   */
  private getZScore(alpha: number): number {
    // Approximate Z-scores for common alpha values
    const zTable: Record<number, number> = {
      0.025: 1.96,  // 95% CI
      0.05: 1.645,  // 90% CI
      0.01: 2.576,  // 99% CI
      0.005: 2.807  // 99.5% CI
    };
    
    return zTable[alpha] || 1.96; // Default to 95% CI
  }
  
  /**
   * Compute Expected Calibration Error for a slice
   */
  private computeSliceECE(predictions: ConformalPrediction[], numBins: number = 10): number {
    // Bin predictions by confidence score
    const bins: ConformalPrediction[][] = Array.from({ length: numBins }, () => []);
    
    for (const pred of predictions) {
      const binIndex = Math.min(Math.floor(pred.confidence_score * numBins), numBins - 1);
      bins[binIndex].push(pred);
    }
    
    let totalECE = 0;
    let totalWeight = 0;
    
    for (let i = 0; i < numBins; i++) {
      if (bins[i].length === 0) continue;
      
      const binPredictions = bins[i];
      const binWeight = binPredictions.length / predictions.length;
      
      // Average confidence in bin
      const avgConfidence = binPredictions.reduce((sum, p) => sum + p.confidence_score, 0) / binPredictions.length;
      
      // Accuracy in bin (coverage rate)
      const avgAccuracy = binPredictions.filter(p => p.is_covered).length / binPredictions.length;
      
      // Calibration error for this bin
      const binECE = Math.abs(avgConfidence - avgAccuracy);
      
      totalECE += binWeight * binECE;
      totalWeight += binWeight;
    }
    
    return totalWeight > 0 ? totalECE / totalWeight : 0;
  }
  
  /**
   * Compute overall ECE across all predictions
   */
  private computeOverallECE(): number {
    return this.computeSliceECE(this.predictions, 20); // More bins for overall analysis
  }
  
  /**
   * Create empty metrics for slices with insufficient samples
   */
  private createEmptySliceMetrics(sliceId: string, sampleCount: number): SliceCoverageMetrics {
    const slice = this.parseSliceId(sliceId);
    
    return {
      slice,
      sample_count: sampleCount,
      empirical_coverage: 0,
      target_coverage: this.config.target_coverage,
      miscoverage: 0,
      wilson_ci_lower: 0,
      wilson_ci_upper: 0,
      ece_score: 0,
      violation_detected: false,
      spend_reduction_triggered: false
    };
  }
  
  /**
   * Parse slice ID back into components
   */
  private parseSliceId(sliceId: string): MondrianSlice {
    const [intent, language, entropyBin] = sliceId.split('_');
    
    return {
      intent: intent as Intent,
      language: language as Language,
      entropy_bin: entropyBin as EntropyBin,
      slice_id: sliceId
    };
  }
  
  /**
   * Get recommended action for violation
   */
  private getRecommendedAction(metrics: SliceCoverageMetrics): string {
    const { miscoverage, ece_score } = metrics;
    
    if (miscoverage > this.config.miscoverage_tolerance) {
      return `Reduce spend by ${this.config.auto_spend_reduction}% - severe undercoverage (${miscoverage.toFixed(1)}pp)`;
    }
    
    if (miscoverage < -this.config.miscoverage_tolerance) {
      return `Consider increasing spend - overcoverage detected (${Math.abs(miscoverage).toFixed(1)}pp waste)`;
    }
    
    if (Math.abs(ece_score) > this.config.ece_threshold) {
      return `Recalibrate model - poor calibration (ECE=${ece_score.toFixed(3)})`;
    }
    
    return 'Monitor closely - marginal violation detected';
  }
  
  /**
   * Generate system-wide recommendations
   */
  private generateRecommendations(
    sliceMetrics: SliceCoverageMetrics[], 
    violations: any[], 
    overallMiscoverage: number, 
    overallECE: number
  ): string[] {
    const recommendations: string[] = [];
    
    // System-wide coverage issues
    if (Math.abs(overallMiscoverage) > this.config.miscoverage_tolerance) {
      if (overallMiscoverage > 0) {
        recommendations.push(`CRITICAL: System undercoverage of ${overallMiscoverage.toFixed(1)}pp - increase model conservatism`);
      } else {
        recommendations.push(`EFFICIENCY: System overcoverage of ${Math.abs(overallMiscoverage).toFixed(1)}pp - reduce conservatism to save compute`);
      }
    }
    
    // Calibration issues
    if (Math.abs(overallECE) > this.config.ece_threshold) {
      recommendations.push(`CALIBRATION: Overall ECE=${overallECE.toFixed(3)} exceeds threshold - retrain or recalibrate models`);
    }
    
    // Slice-specific patterns
    const highEntropyViolations = violations.filter(v => v.slice_id.includes('_high')).length;
    const nlViolations = violations.filter(v => v.slice_id.startsWith('NL_')).length;
    const symbolViolations = violations.filter(v => v.slice_id.startsWith('symbol_')).length;
    
    if (highEntropyViolations > 2) {
      recommendations.push(`PATTERN: High-entropy slices frequently violate - increase spend allocation for complex queries`);
    }
    
    if (nlViolations > symbolViolations * 2) {
      recommendations.push(`PATTERN: Natural language queries need more coverage - rebalance spend toward NL slices`);
    }
    
    // Spend efficiency
    const totalSpendReductions = sliceMetrics.filter(m => m.spend_reduction_triggered).length;
    if (totalSpendReductions > sliceMetrics.length * 0.2) {
      recommendations.push(`EFFICIENCY: ${totalSpendReductions} slices had spend reduced - significant over-allocation detected`);
    }
    
    // Data collection needs
    const insufficientDataSlices = sliceMetrics.filter(m => m.sample_count < this.config.min_samples_per_slice).length;
    if (insufficientDataSlices > 0) {
      recommendations.push(`DATA: ${insufficientDataSlices} slices have insufficient data - increase sample collection for reliable analysis`);
    }
    
    return recommendations;
  }
  
  /**
   * Get current slice distribution
   */
  private getSliceDistribution(): Record<string, number> {
    const distribution: Record<string, number> = {};
    
    for (const prediction of this.predictions) {
      const sliceId = prediction.slice.slice_id;
      distribution[sliceId] = (distribution[sliceId] || 0) + 1;
    }
    
    return distribution;
  }
  
  /**
   * Save comprehensive audit report
   */
  private async saveAuditReport(report: MiscoverageAuditReport, outputDir: string): Promise<void> {
    // Save full JSON report
    await writeFile(
      join(outputDir, 'slice-wise-miscoverage-audit.json'),
      JSON.stringify(report, null, 2)
    );
    
    // Save CSV for slice metrics
    const csvHeader = 'slice_id,intent,language,entropy_bin,sample_count,empirical_coverage,target_coverage,miscoverage,wilson_ci_lower,wilson_ci_upper,ece_score,violation_detected,spend_reduction_triggered';
    const csvRows = report.slice_metrics.map(m => 
      `${m.slice.slice_id},${m.slice.intent},${m.slice.language},${m.slice.entropy_bin},${m.sample_count},${m.empirical_coverage.toFixed(4)},${m.target_coverage.toFixed(4)},${m.miscoverage.toFixed(2)},${m.wilson_ci_lower.toFixed(4)},${m.wilson_ci_upper.toFixed(4)},${m.ece_score.toFixed(4)},${m.violation_detected},${m.spend_reduction_triggered}`
    );
    
    await writeFile(
      join(outputDir, 'slice-metrics.csv'),
      [csvHeader, ...csvRows].join('\n')
    );
    
    // Save markdown summary
    const summaryReport = this.generateSummaryMarkdown(report);
    await writeFile(join(outputDir, 'miscoverage-audit-summary.md'), summaryReport);
    
    console.log(`✅ Audit report saved to ${outputDir}/`);
  }
  
  /**
   * Generate markdown summary report
   */
  private generateSummaryMarkdown(report: MiscoverageAuditReport): string {
    let md = '# Slice-Wise Miscoverage Audit Report\n\n';
    
    md += `**Audit Date**: ${report.timestamp.toISOString()}\n`;
    md += `**Total Slices**: ${report.total_slices}\n`;
    md += `**Slices with Violations**: ${report.slices_with_violations} (${(report.slices_with_violations / report.total_slices * 100).toFixed(1)}%)\n`;
    md += `**Overall Miscoverage**: ${report.overall_miscoverage.toFixed(2)}pp\n`;
    md += `**Overall ECE**: ${report.overall_ece.toFixed(3)}\n\n`;
    
    // Status indicator
    if (report.slices_with_violations === 0) {
      md += '## 🟢 Status: ALL SLICES COMPLIANT\n\n';
    } else if (report.slices_with_violations / report.total_slices < 0.1) {
      md += '## 🟡 Status: MINOR VIOLATIONS DETECTED\n\n';
    } else {
      md += '## 🔴 Status: SIGNIFICANT VIOLATIONS DETECTED\n\n';
    }
    
    // Alerts
    if (report.alerts.length > 0) {
      md += '## 🚨 Critical Alerts\n\n';
      for (const alert of report.alerts) {
        md += `- **${alert}**\n`;
      }
      md += '\n';
    }
    
    // Violations breakdown
    if (report.violations.length > 0) {
      md += '## ❌ Slice Violations\n\n';
      md += '| Slice | Miscoverage | ECE | Action |\n';
      md += '|-------|-------------|-----|--------|\n';
      
      for (const violation of report.violations) {
        md += `| ${violation.slice_id} | ${violation.miscoverage.toFixed(2)}pp | ${violation.ece_score.toFixed(3)} | ${violation.recommended_action} |\n`;
      }
      md += '\n';
    }
    
    // Spend adjustments
    if (report.spend_adjustments.length > 0) {
      md += '## 💰 Automated Spend Adjustments\n\n';
      md += '| Slice | Old Spend | New Spend | Reduction |\n';
      md += '|-------|-----------|-----------|----------|\n';
      
      for (const adjustment of report.spend_adjustments) {
        md += `| ${adjustment.slice_id} | ${adjustment.old_spend_percentage.toFixed(1)}% | ${adjustment.new_spend_percentage.toFixed(1)}% | ${adjustment.reduction_amount.toFixed(1)}pp |\n`;
      }
      md += '\n';
    }
    
    // Top performing slices
    const topSlices = report.slice_metrics
      .filter(m => m.sample_count >= this.config.min_samples_per_slice)
      .sort((a, b) => Math.abs(a.miscoverage) - Math.abs(b.miscoverage))
      .slice(0, 5);
    
    md += '## 🏆 Best Calibrated Slices\n\n';
    md += '| Slice | Samples | Coverage | Miscoverage | ECE |\n';
    md += '|-------|---------|----------|-------------|-----|\n';
    
    for (const slice of topSlices) {
      md += `| ${slice.slice.slice_id} | ${slice.sample_count} | ${(slice.empirical_coverage * 100).toFixed(1)}% | ${slice.miscoverage.toFixed(2)}pp | ${slice.ece_score.toFixed(3)} |\n`;
    }
    md += '\n';
    
    // Recommendations
    md += '## 💡 Recommendations\n\n';
    for (const rec of report.recommendations) {
      md += `- ${rec}\n`;
    }
    
    return md;
  }
  
  /**
   * Generate synthetic conformal predictions for testing
   */
  static generateSyntheticPredictions(count: number = 5000): ConformalPrediction[] {
    const predictions: ConformalPrediction[] = [];
    const intents: Intent[] = ['NL', 'symbol', 'mixed'];
    const languages: Language[] = ['typescript', 'python', 'rust', 'go', 'javascript'];
    const entropyBins: EntropyBin[] = ['low', 'medium', 'high'];
    
    for (let i = 0; i < count; i++) {
      const intent = intents[Math.floor(Math.random() * intents.length)];
      const language = languages[Math.floor(Math.random() * languages.length)];
      const entropyBin = entropyBins[Math.floor(Math.random() * entropyBins.length)];
      
      const slice: MondrianSlice = {
        intent,
        language,
        entropy_bin: entropyBin,
        slice_id: `${intent}_${language}_${entropyBin}`
      };
      
      // Generate realistic confidence and coverage
      let baseConfidence = 0.7 + Math.random() * 0.25; // 0.7-0.95
      let coverageProbability = 0.85; // Base 85% coverage
      
      // Slice-specific adjustments
      if (entropyBin === 'high') {
        baseConfidence -= 0.1; // Less confident on high entropy
        coverageProbability -= 0.05;
      }
      
      if (intent === 'NL') {
        baseConfidence -= 0.05; // Less confident on NL
        coverageProbability -= 0.02;
      }
      
      // Add calibration noise (some slices are better calibrated)
      const calibrationNoise = (Math.random() - 0.5) * 0.1;
      const isCovered = Math.random() < (coverageProbability + calibrationNoise);
      
      predictions.push({
        query_id: `query_${i}`,
        slice,
        confidence_score: Math.max(0.1, Math.min(0.99, baseConfidence)),
        prediction_set_size: Math.floor(1 + Math.random() * 10), // 1-10 predictions
        ground_truth_rank: isCovered ? Math.floor(Math.random() * 3) + 1 : 99, // In top 3 or missed
        is_covered: isCovered,
        timestamp: new Date(Date.now() - Math.random() * 86400000)
      });
    }
    
    return predictions;
  }
  
  /**
   * Get current spend allocation for all slices
   */
  getSliceSpendAllocation(): Map<string, number> {
    return new Map(this.sliceSpendMap);
  }
  
  /**
   * Update spend allocation for a specific slice
   */
  updateSliceSpend(sliceId: string, newSpendPercentage: number): void {
    this.sliceSpendMap.set(sliceId, Math.max(0, newSpendPercentage));
    this.emit('spend_updated', { slice_id: sliceId, new_spend: newSpendPercentage });
  }
}

// Factory function
export function createSliceWiseMiscoverageAuditor(config?: Partial<MiscoverageAuditConfig>): SliceWiseMiscoverageAuditor {
  const fullConfig = { ...DEFAULT_MISCOVERAGE_CONFIG, ...config };
  return new SliceWiseMiscoverageAuditor(fullConfig);
}