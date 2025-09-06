/**
 * Canary Gates Validator
 * 
 * Comprehensive validation system for centrality canary gates with statistical
 * significance testing, operational SLA monitoring, and quality assurance.
 */

import { EventEmitter } from 'events';

interface CanaryGate {
  metric: string;
  operator: 'gte' | 'lte' | 'eq';
  threshold: number;
  pValue?: number;
  description: string;
  critical?: boolean;
  category: 'quality' | 'ops' | 'drift';
}

interface GateValidationResult {
  gate: CanaryGate;
  passed: boolean;
  actualValue: number;
  thresholdValue: number;
  pValue?: number;
  confidenceInterval?: [number, number];
  severity: 'info' | 'warning' | 'critical';
  message: string;
}

interface ValidationSummary {
  passed: boolean;
  totalGates: number;
  passedGates: number;
  failedGates: string[];
  criticalFailures: string[];
  warnings: string[];
  validationDetails: GateValidationResult[];
  timestamp: Date;
}

interface QualityGates {
  // Quality Gates (vs LSP+RAPTOR baseline)
  nDCG_delta_NL: CanaryGate;
  recall_at_50: CanaryGate;
  diversity_delta: CanaryGate;
  core_delta: CanaryGate;
}

interface OpsGates {
  // Operations Gates  
  stageA_p95_latency: CanaryGate;
  stageC_p95_latency: CanaryGate;
  p99_p95_ratio: CanaryGate;
  span_coverage: CanaryGate;
}

interface DriftGates {
  // Drift Detection Gates
  semantic_share_drift: CanaryGate;
  util_flag_hits: CanaryGate;
  router_upshift_rate: CanaryGate;
}

export class CanaryGatesValidator extends EventEmitter {
  private qualityGates: QualityGates;
  private opsGates: OpsGates;
  private driftGates: DriftGates;
  private validationHistory: ValidationSummary[] = [];

  constructor() {
    super();
    this.initializeGates();
  }

  private initializeGates(): void {
    // Quality Gates - Core metrics that determine canary success
    this.qualityGates = {
      nDCG_delta_NL: {
        metric: 'nDCG@10_delta',
        operator: 'gte',
        threshold: 1.0,
        pValue: 0.05,
        description: 'nDCG@10 (NL) improvement ≥+1.0pt (p<0.05)',
        critical: true,
        category: 'quality'
      },
      recall_at_50: {
        metric: 'recall@50',
        operator: 'gte', 
        threshold: 88.9,
        description: 'Maintain baseline recall@50 ≥88.9%',
        critical: true,
        category: 'quality'
      },
      diversity_delta: {
        metric: 'diversity@10_delta',
        operator: 'gte',
        threshold: 10,
        description: 'Diversity@10 improvement ≥+10%',
        critical: false,
        category: 'quality'
      },
      core_delta: {
        metric: 'core@10_delta', 
        operator: 'gte',
        threshold: 10,
        description: 'Core@10 improvement ≥+10pp',
        critical: false,
        category: 'quality'
      }
    };

    // Operations Gates - Performance and reliability SLAs
    this.opsGates = {
      stageA_p95_latency: {
        metric: 'stageA_p95_delta',
        operator: 'lte',
        threshold: 1.0,
        description: 'Stage-A p95 latency increase ≤+1ms',
        critical: true,
        category: 'ops'
      },
      stageC_p95_latency: {
        metric: 'stageC_p95_delta_pct',
        operator: 'lte',
        threshold: 5.0,
        description: 'Stage-C p95 latency increase ≤+5%',
        critical: true,
        category: 'ops'
      },
      p99_p95_ratio: {
        metric: 'p99_p95_ratio',
        operator: 'lte',
        threshold: 2.0,
        description: 'p99/p95 ratio ≤2.0 for SLA compliance',
        critical: true,
        category: 'ops'
      },
      span_coverage: {
        metric: 'span_coverage',
        operator: 'gte',
        threshold: 100,
        description: 'Complete span coverage',
        critical: false,
        category: 'ops'
      }
    };

    // Drift Detection Gates - Prevent degradation or unwanted side effects
    this.driftGates = {
      semantic_share_drift: {
        metric: 'semantic_share_delta',
        operator: 'lte',
        threshold: 15,
        description: 'Semantic share increase ≤+15pp unless nDCG rises ≥+1pt',
        critical: false,
        category: 'drift'
      },
      util_flag_hits: {
        metric: 'util_flag_hits_delta',
        operator: 'lte',
        threshold: 5,
        description: 'Utility flag hits increase ≤+5pp over baseline',
        critical: false,
        category: 'drift'
      },
      router_upshift_rate: {
        metric: 'router_upshift_rate',
        operator: 'lte',
        threshold: 7, // 5% + 2pp tolerance
        description: 'Router upshift rate ≤5%+2pp',
        critical: false,
        category: 'drift'
      }
    };
  }

  public async validateGates(
    customGates: CanaryGate[], 
    metrics: Record<string, number>
  ): Promise<ValidationSummary> {
    console.log('🔍 Validating canary gates...');
    
    // Combine all gates
    const allGates = [
      ...customGates,
      ...Object.values(this.qualityGates),
      ...Object.values(this.opsGates),
      ...Object.values(this.driftGates)
    ];
    
    const validationDetails: GateValidationResult[] = [];
    const failedGates: string[] = [];
    const criticalFailures: string[] = [];
    const warnings: string[] = [];
    let passedCount = 0;

    for (const gate of allGates) {
      const result = await this.validateSingleGate(gate, metrics);
      validationDetails.push(result);
      
      if (result.passed) {
        passedCount++;
      } else {
        failedGates.push(gate.metric);
        
        if (gate.critical) {
          criticalFailures.push(gate.metric);
        }
        
        if (result.severity === 'warning') {
          warnings.push(result.message);
        }
      }
    }

    const validationSummary: ValidationSummary = {
      passed: failedGates.length === 0,
      totalGates: allGates.length,
      passedGates: passedCount,
      failedGates,
      criticalFailures,
      warnings,
      validationDetails,
      timestamp: new Date()
    };

    // Store validation history
    this.validationHistory.push(validationSummary);
    if (this.validationHistory.length > 100) {
      this.validationHistory.shift(); // Keep only last 100 validations
    }

    // Emit events
    if (!validationSummary.passed) {
      console.error(`❌ Gate validation failed: ${failedGates.length}/${allGates.length} gates failed`);
      if (criticalFailures.length > 0) {
        console.error(`🚨 Critical failures: ${criticalFailures.join(', ')}`);
        this.emit('criticalGateFailures', { failures: criticalFailures, summary: validationSummary });
      }
      this.emit('gateValidationFailed', validationSummary);
    } else {
      console.log(`✅ All gates passed: ${passedCount}/${allGates.length}`);
      if (warnings.length > 0) {
        console.warn(`⚠️ ${warnings.length} warnings issued`);
        this.emit('gateValidationWarnings', { warnings, summary: validationSummary });
      }
      this.emit('gateValidationPassed', validationSummary);
    }

    return validationSummary;
  }

  private async validateSingleGate(
    gate: CanaryGate, 
    metrics: Record<string, number>
  ): Promise<GateValidationResult> {
    const actualValue = metrics[gate.metric];
    
    if (actualValue === undefined) {
      return {
        gate,
        passed: false,
        actualValue: NaN,
        thresholdValue: gate.threshold,
        severity: 'critical',
        message: `Metric ${gate.metric} not found in measurement data`
      };
    }

    // Perform statistical significance test if required
    let pValue: number | undefined;
    let confidenceInterval: [number, number] | undefined;
    
    if (gate.pValue !== undefined) {
      // For this implementation, we'll simulate p-value calculation
      // In production, this would use proper statistical testing
      pValue = await this.calculatePValue(gate.metric, actualValue, gate.threshold);
      confidenceInterval = await this.calculateConfidenceInterval(gate.metric, actualValue);
    }

    // Check if gate passes
    const passed = this.evaluateGateCondition(actualValue, gate.operator, gate.threshold, pValue, gate.pValue);
    
    // Determine severity
    const severity = this.determineSeverity(gate, passed, actualValue);
    
    // Generate descriptive message
    const message = this.generateGateMessage(gate, actualValue, passed, pValue);

    return {
      gate,
      passed,
      actualValue,
      thresholdValue: gate.threshold,
      pValue,
      confidenceInterval,
      severity,
      message
    };
  }

  private evaluateGateCondition(
    actualValue: number,
    operator: string,
    threshold: number,
    pValue?: number,
    requiredPValue?: number
  ): boolean {
    // Check statistical significance first if required
    if (requiredPValue !== undefined && pValue !== undefined) {
      if (pValue >= requiredPValue) {
        return false; // Not statistically significant
      }
    }

    // Check threshold condition
    switch (operator) {
      case 'gte':
        return actualValue >= threshold;
      case 'lte':
        return actualValue <= threshold;
      case 'eq':
        return Math.abs(actualValue - threshold) < 0.001; // Small tolerance for floating point
      default:
        return false;
    }
  }

  private async calculatePValue(metric: string, actualValue: number, threshold: number): Promise<number> {
    // Simulate statistical significance testing
    // In production, this would perform proper hypothesis testing using historical data
    
    // For improvement metrics (delta), simulate t-test against null hypothesis
    if (metric.includes('delta') || metric.includes('improvement')) {
      // Simulate based on effect size and sample size
      const effectSize = (actualValue - threshold) / Math.abs(threshold);
      const sampleSize = 1000; // Assumed sample size
      
      // Very rough approximation of p-value based on effect size
      const zScore = Math.abs(effectSize) * Math.sqrt(sampleSize);
      const pValue = 2 * (1 - this.normalCDF(zScore));
      
      return Math.min(Math.max(pValue, 0.001), 0.999); // Clamp to reasonable range
    }
    
    return 0.001; // Assume significant for non-delta metrics
  }

  private async calculateConfidenceInterval(metric: string, actualValue: number): Promise<[number, number]> {
    // Simulate 95% confidence interval
    const margin = Math.abs(actualValue * 0.05); // ±5% margin as rough approximation
    return [actualValue - margin, actualValue + margin];
  }

  private normalCDF(x: number): number {
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  private erf(x: number): number {
    // Approximation of error function
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }

  private determineSeverity(
    gate: CanaryGate, 
    passed: boolean, 
    actualValue: number
  ): 'info' | 'warning' | 'critical' {
    if (passed) {
      return 'info';
    }

    if (gate.critical) {
      return 'critical';
    }

    // Check how far from threshold
    const percentageOff = Math.abs((actualValue - gate.threshold) / gate.threshold) * 100;
    
    if (percentageOff > 20) {
      return 'critical';
    } else if (percentageOff > 10) {
      return 'warning';
    }
    
    return 'info';
  }

  private generateGateMessage(
    gate: CanaryGate, 
    actualValue: number, 
    passed: boolean, 
    pValue?: number
  ): string {
    const operatorText = {
      'gte': '≥',
      'lte': '≤', 
      'eq': '='
    }[gate.operator];

    const statusEmoji = passed ? '✅' : '❌';
    const pValueText = pValue ? ` (p=${pValue.toFixed(3)})` : '';
    
    return `${statusEmoji} ${gate.metric}: ${actualValue.toFixed(2)} ${operatorText} ${gate.threshold}${pValueText} - ${gate.description}`;
  }

  public async validateSemanticShareDrift(
    semanticShareDelta: number, 
    nDCGDelta: number
  ): Promise<GateValidationResult> {
    // Special validation for semantic share drift
    const gate: CanaryGate = {
      metric: 'semantic_share_conditional',
      operator: 'lte',
      threshold: 15,
      description: 'Semantic share increase ≤+15pp unless nDCG rises ≥+1pt',
      category: 'drift'
    };

    // If semantic share increased >15pp, nDCG must also increase ≥1pt
    const passed = semanticShareDelta <= 15 || nDCGDelta >= 1.0;
    
    let message: string;
    if (semanticShareDelta <= 15) {
      message = `✅ Semantic share increase ${semanticShareDelta.toFixed(1)}pp is within acceptable range (≤15pp)`;
    } else if (nDCGDelta >= 1.0) {
      message = `✅ Semantic share increase ${semanticShareDelta.toFixed(1)}pp compensated by nDCG improvement ${nDCGDelta.toFixed(1)}pt`;
    } else {
      message = `❌ Semantic share increased ${semanticShareDelta.toFixed(1)}pp (>15pp) without sufficient nDCG improvement (${nDCGDelta.toFixed(1)}pt < 1.0pt)`;
    }

    return {
      gate,
      passed,
      actualValue: semanticShareDelta,
      thresholdValue: 15,
      severity: passed ? 'info' : 'warning',
      message
    };
  }

  public getValidationHistory(): ValidationSummary[] {
    return [...this.validationHistory];
  }

  public getGatePerformanceReport(): {
    mostFailedGates: Array<{gate: string, failureRate: number}>;
    averagePassRate: number;
    criticalFailureRate: number;
    trendAnalysis: string;
  } {
    if (this.validationHistory.length === 0) {
      return {
        mostFailedGates: [],
        averagePassRate: 0,
        criticalFailureRate: 0,
        trendAnalysis: 'No validation history available'
      };
    }

    // Count gate failures
    const gateFailureCounts = new Map<string, number>();
    let totalValidations = this.validationHistory.length;
    let totalPassingValidations = 0;
    let totalCriticalFailures = 0;

    for (const validation of this.validationHistory) {
      if (validation.passed) {
        totalPassingValidations++;
      }
      
      totalCriticalFailures += validation.criticalFailures.length;
      
      for (const failedGate of validation.failedGates) {
        gateFailureCounts.set(failedGate, (gateFailureCounts.get(failedGate) || 0) + 1);
      }
    }

    // Calculate most failed gates
    const mostFailedGates = Array.from(gateFailureCounts.entries())
      .map(([gate, failures]) => ({
        gate,
        failureRate: failures / totalValidations
      }))
      .sort((a, b) => b.failureRate - a.failureRate)
      .slice(0, 5);

    // Calculate rates
    const averagePassRate = totalPassingValidations / totalValidations;
    const criticalFailureRate = totalCriticalFailures / totalValidations;

    // Simple trend analysis
    const recentValidations = this.validationHistory.slice(-10);
    const recentPassRate = recentValidations.filter(v => v.passed).length / recentValidations.length;
    
    let trendAnalysis: string;
    if (recentPassRate > averagePassRate + 0.1) {
      trendAnalysis = 'Improving trend - recent pass rate higher than average';
    } else if (recentPassRate < averagePassRate - 0.1) {
      trendAnalysis = 'Declining trend - recent pass rate lower than average';
    } else {
      trendAnalysis = 'Stable trend - consistent performance';
    }

    return {
      mostFailedGates,
      averagePassRate,
      criticalFailureRate,
      trendAnalysis
    };
  }

  public async runRegressionAnalysis(): Promise<{
    regressionDetected: boolean;
    affectedMetrics: string[];
    severity: 'low' | 'medium' | 'high';
    recommendation: string;
  }> {
    if (this.validationHistory.length < 10) {
      return {
        regressionDetected: false,
        affectedMetrics: [],
        severity: 'low',
        recommendation: 'Insufficient data for regression analysis'
      };
    }

    const recentValidations = this.validationHistory.slice(-5);
    const baselineValidations = this.validationHistory.slice(-15, -5);
    
    const recentFailureRate = recentValidations.filter(v => !v.passed).length / recentValidations.length;
    const baselineFailureRate = baselineValidations.filter(v => !v.passed).length / baselineValidations.length;
    
    const regressionThreshold = 0.3; // 30% increase in failure rate
    const regressionDetected = recentFailureRate > baselineFailureRate * (1 + regressionThreshold);
    
    // Find most affected metrics
    const recentFailures = recentValidations.flatMap(v => v.failedGates);
    const baselineFailures = baselineValidations.flatMap(v => v.failedGates);
    
    const affectedMetrics = [...new Set(recentFailures)].filter(metric => {
      const recentCount = recentFailures.filter(m => m === metric).length;
      const baselineCount = baselineFailures.filter(m => m === metric).length;
      return recentCount > baselineCount;
    });

    let severity: 'low' | 'medium' | 'high' = 'low';
    let recommendation = 'Continue monitoring';
    
    if (regressionDetected) {
      if (recentFailureRate > 0.5) {
        severity = 'high';
        recommendation = 'Immediate investigation required - high failure rate detected';
      } else if (recentFailureRate > 0.2) {
        severity = 'medium'; 
        recommendation = 'Investigation recommended - moderate regression detected';
      } else {
        severity = 'low';
        recommendation = 'Monitor closely - minor regression detected';
      }
    }

    return {
      regressionDetected,
      affectedMetrics,
      severity,
      recommendation
    };
  }
}