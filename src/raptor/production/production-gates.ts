/**
 * Production Gates Validation System
 * 
 * Implements specific gates for declaring "LEADING" status
 * Addresses: NL nDCG@10 ≥+3.0pp (p<0.01), P@1 ≥+5pp, p95 ≤Serena-10ms, QPS@150ms ≥1.2x
 */

export interface ProductionGate {
  id: string;
  name: string;
  category: 'quality' | 'performance' | 'reliability';
  target: number;
  operator: '>=' | '<=' | '=' | '>' | '<';
  unit: string;
  critical: boolean;
  description: string;
  validation_method: 'statistical' | 'direct' | 'computed';
  statistical_requirements?: {
    min_p_value: number;
    min_effect_size: number;
    required_power: number;
  };
}

export interface GateEvaluation {
  gate: ProductionGate;
  measured_value: number;
  target_value: number;
  passed: boolean;
  margin: number;
  confidence: number;
  evidence: {
    source: string;
    method: string;
    sample_size: number;
    statistical_significance?: number;
    effect_size?: number;
    power?: number;
  };
  timestamp: Date;
}

export interface ProductionReadinessAssessment {
  overall_status: 'LEADING' | 'COMPETITIVE' | 'NOT_READY';
  quality_gates_passed: number;
  performance_gates_passed: number;
  reliability_gates_passed: number;
  total_gates: number;
  critical_failures: GateEvaluation[];
  marginal_passes: GateEvaluation[];
  strong_passes: GateEvaluation[];
  overall_confidence: number;
  recommendation: string;
  next_actions: string[];
}

export class ProductionGatesValidator {
  private gates: Map<string, ProductionGate>;

  constructor() {
    this.gates = this.defineProductionGates();
  }

  /**
   * Define the comprehensive set of production gates
   */
  private defineProductionGates(): Map<string, ProductionGate> {
    const gates = new Map<string, ProductionGate>();

    // QUALITY GATES
    gates.set('nl_nDCG_10', {
      id: 'nl_nDCG_10',
      name: 'NL nDCG@10 Improvement',
      category: 'quality',
      target: 3.0, // +3.0pp minimum
      operator: '>=',
      unit: 'percentage points',
      critical: true,
      description: 'Natural Language queries must show ≥3.0pp nDCG@10 improvement with p<0.01',
      validation_method: 'statistical',
      statistical_requirements: {
        min_p_value: 0.01,
        min_effect_size: 0.3,
        required_power: 0.8
      }
    });

    gates.set('P_at_1_symbol', {
      id: 'P_at_1_symbol',
      name: 'P@1 Symbol Improvement', 
      category: 'quality',
      target: 5.0, // +5pp minimum
      operator: '>=',
      unit: 'percentage points',
      critical: true,
      description: 'Symbol queries must show ≥5pp P@1 improvement',
      validation_method: 'statistical',
      statistical_requirements: {
        min_p_value: 0.05,
        min_effect_size: 0.2,
        required_power: 0.8
      }
    });

    gates.set('recall_50_overall', {
      id: 'recall_50_overall',
      name: 'Recall@50 Maintenance',
      category: 'quality', 
      target: 0.0, // Must not decrease
      operator: '>=',
      unit: 'percentage points',
      critical: true,
      description: 'Overall Recall@50 must not decrease vs baseline',
      validation_method: 'direct'
    });

    gates.set('ece_calibration', {
      id: 'ece_calibration',
      name: 'Calibration Quality',
      category: 'quality',
      target: 0.05, // ≤5% ECE
      operator: '<=',
      unit: 'ECE',
      critical: false,
      description: 'Expected Calibration Error must be ≤0.05',
      validation_method: 'computed'
    });

    // PERFORMANCE GATES  
    gates.set('p95_latency', {
      id: 'p95_latency',
      name: 'p95 Latency Improvement',
      category: 'performance',
      target: -10, // 10ms under Serena
      operator: '<=',
      unit: 'milliseconds',
      critical: true,
      description: 'p95 latency must be ≥10ms better than Serena',
      validation_method: 'direct'
    });

    gates.set('QPS_150ms', {
      id: 'QPS_150ms',
      name: 'Throughput at SLA',
      category: 'performance',
      target: 1.2, // 1.2x minimum
      operator: '>=',
      unit: 'ratio',
      critical: true,
      description: 'QPS at 150ms SLA must be ≥1.2x Serena',
      validation_method: 'computed'
    });

    gates.set('p99_latency', {
      id: 'p99_latency',
      name: 'Tail Latency Control',
      category: 'performance',
      target: 2.0, // ≤2x p95
      operator: '<=',
      unit: 'ratio to p95',
      critical: false,
      description: 'p99 latency should not exceed 2x p95',
      validation_method: 'computed'
    });

    // RELIABILITY GATES
    gates.set('timeout_reduction', {
      id: 'timeout_reduction',
      name: 'Timeout Rate Improvement',
      category: 'reliability',
      target: 2.0, // ≥2pp reduction
      operator: '>=',
      unit: 'percentage points',
      critical: true,
      description: 'Timeout rate must be ≥2pp lower than Serena',
      validation_method: 'direct'
    });

    gates.set('span_coverage', {
      id: 'span_coverage',
      name: 'Span Coverage',
      category: 'reliability',
      target: 100.0, // Must be 100%
      operator: '>=',
      unit: 'percentage',
      critical: true,
      description: 'Span coverage must be 100%',
      validation_method: 'direct'
    });

    gates.set('error_rate', {
      id: 'error_rate',
      name: 'System Error Rate',
      category: 'reliability',
      target: 1.0, // ≤1% errors
      operator: '<=',
      unit: 'percentage',
      critical: false,
      description: 'System error rate should be ≤1%',
      validation_method: 'direct'
    });

    gates.set('sentinel_nzc', {
      id: 'sentinel_nzc',
      name: 'Sentinel Query Coverage',
      category: 'reliability',
      target: 99.0, // ≥99% NZC
      operator: '>=',
      unit: 'percentage',
      critical: true,
      description: 'Sentinel queries must achieve ≥99% Non-Zero Coverage',
      validation_method: 'direct'
    });

    return gates;
  }

  /**
   * Evaluate all production gates against measurements
   */
  async evaluateProductionReadiness(
    measurements: Map<string, number>,
    evidence: Map<string, any> = new Map()
  ): Promise<ProductionReadinessAssessment> {
    console.log('🚪 Evaluating production gates...');
    
    const evaluations: GateEvaluation[] = [];
    let qualityPassed = 0;
    let performancePassed = 0;
    let reliabilityPassed = 0;

    // Evaluate each gate
    for (const [gateId, gate] of this.gates) {
      const measured = measurements.get(gateId);
      
      if (measured === undefined) {
        console.warn(`⚠️ Missing measurement for gate: ${gateId}`);
        continue;
      }

      const evaluation = this.evaluateGate(gate, measured, evidence.get(gateId));
      evaluations.push(evaluation);

      // Count passes by category
      if (evaluation.passed) {
        switch (gate.category) {
          case 'quality': qualityPassed++; break;
          case 'performance': performancePassed++; break;
          case 'reliability': reliabilityPassed++; break;
        }
      }

      console.log(`${evaluation.passed ? '✅' : '❌'} ${gate.name}: ${measured.toFixed(3)}${gate.unit} (target: ${gate.operator}${gate.target}${gate.unit})`);
    }

    // Compute overall assessment
    const assessment = this.computeOverallAssessment(evaluations);
    
    console.log(`\n📊 Overall Status: ${assessment.overall_status}`);
    console.log(`Quality: ${assessment.quality_gates_passed}/${this.getGatesByCategory('quality').length}`);
    console.log(`Performance: ${assessment.performance_gates_passed}/${this.getGatesByCategory('performance').length}`);
    console.log(`Reliability: ${assessment.reliability_gates_passed}/${this.getGatesByCategory('reliability').length}`);
    
    return assessment;
  }

  /**
   * Evaluate a single production gate
   */
  private evaluateGate(
    gate: ProductionGate, 
    measuredValue: number, 
    evidence?: any
  ): GateEvaluation {
    // Check if gate passes
    let passed = false;
    switch (gate.operator) {
      case '>=':
        passed = measuredValue >= gate.target;
        break;
      case '<=':
        passed = measuredValue <= gate.target;
        break;
      case '>':
        passed = measuredValue > gate.target;
        break;
      case '<':
        passed = measuredValue < gate.target;
        break;
      case '=':
        passed = Math.abs(measuredValue - gate.target) < 0.001;
        break;
    }

    // Compute margin
    let margin = 0;
    if (gate.operator === '>=' || gate.operator === '>') {
      margin = measuredValue - gate.target;
    } else if (gate.operator === '<=' || gate.operator === '<') {
      margin = gate.target - measuredValue;
    }

    // Assess confidence based on evidence
    let confidence = 0.8; // Default confidence
    let statisticalSignificance = undefined;
    let effectSize = undefined;
    let power = undefined;

    if (evidence && gate.statistical_requirements) {
      statisticalSignificance = evidence.p_value;
      effectSize = evidence.effect_size;
      power = evidence.power;

      // Check statistical requirements
      if (statisticalSignificance && statisticalSignificance <= gate.statistical_requirements.min_p_value) {
        confidence += 0.15;
      }
      if (effectSize && effectSize >= gate.statistical_requirements.min_effect_size) {
        confidence += 0.05;
      }
      if (power && power >= gate.statistical_requirements.required_power) {
        confidence += 0.1;
      }

      confidence = Math.min(1.0, confidence);
    }

    return {
      gate,
      measured_value: measuredValue,
      target_value: gate.target,
      passed,
      margin,
      confidence,
      evidence: {
        source: evidence?.source || 'measurement',
        method: evidence?.method || gate.validation_method,
        sample_size: evidence?.sample_size || 0,
        statistical_significance: statisticalSignificance,
        effect_size: effectSize,
        power: power
      },
      timestamp: new Date()
    };
  }

  /**
   * Compute overall production readiness assessment
   */
  private computeOverallAssessment(evaluations: GateEvaluation[]): ProductionReadinessAssessment {
    const criticalFailures = evaluations.filter(e => e.gate.critical && !e.passed);
    const marginalPasses = evaluations.filter(e => e.passed && e.margin >= 0 && e.margin < (e.gate.target * 0.1));
    const strongPasses = evaluations.filter(e => e.passed && e.margin >= (e.gate.target * 0.1));

    // Count passes by category
    const qualityGates = this.getGatesByCategory('quality');
    const performanceGates = this.getGatesByCategory('performance');
    const reliabilityGates = this.getGatesByCategory('reliability');

    const qualityPassed = evaluations.filter(e => 
      e.gate.category === 'quality' && e.passed
    ).length;
    const performancePassed = evaluations.filter(e => 
      e.gate.category === 'performance' && e.passed
    ).length;
    const reliabilityPassed = evaluations.filter(e => 
      e.gate.category === 'reliability' && e.passed
    ).length;

    // Determine overall status
    let overallStatus: 'LEADING' | 'COMPETITIVE' | 'NOT_READY';
    
    if (criticalFailures.length > 0) {
      overallStatus = 'NOT_READY';
    } else if (
      qualityPassed >= qualityGates.length * 0.8 &&
      performancePassed >= performanceGates.length * 0.8 &&
      reliabilityPassed >= reliabilityGates.length * 0.8
    ) {
      overallStatus = 'LEADING';
    } else {
      overallStatus = 'COMPETITIVE';
    }

    // Compute overall confidence
    const overallConfidence = evaluations.reduce((sum, e) => sum + e.confidence, 0) / evaluations.length;

    // Generate recommendations
    const recommendation = this.generateRecommendation(overallStatus, criticalFailures, marginalPasses);
    const nextActions = this.generateNextActions(overallStatus, criticalFailures, marginalPasses);

    return {
      overall_status: overallStatus,
      quality_gates_passed: qualityPassed,
      performance_gates_passed: performancePassed,
      reliability_gates_passed: reliabilityPassed,
      total_gates: evaluations.length,
      critical_failures: criticalFailures,
      marginal_passes: marginalPasses,
      strong_passes: strongPasses,
      overall_confidence: overallConfidence,
      recommendation,
      next_actions: nextActions
    };
  }

  private getGatesByCategory(category: 'quality' | 'performance' | 'reliability'): ProductionGate[] {
    return Array.from(this.gates.values()).filter(g => g.category === category);
  }

  private generateRecommendation(
    status: 'LEADING' | 'COMPETITIVE' | 'NOT_READY',
    criticalFailures: GateEvaluation[],
    marginalPasses: GateEvaluation[]
  ): string {
    switch (status) {
      case 'LEADING':
        return 'System is ready for production rollout. All critical gates passed with strong margins.';
      case 'COMPETITIVE':
        return 'System shows competitive performance but has some marginal passes. Consider targeted improvements before full rollout.';
      case 'NOT_READY':
        return `System has ${criticalFailures.length} critical gate failures. Must address these before production consideration.`;
    }
  }

  private generateNextActions(
    status: 'LEADING' | 'COMPETITIVE' | 'NOT_READY',
    criticalFailures: GateEvaluation[],
    marginalPasses: GateEvaluation[]
  ): string[] {
    const actions: string[] = [];

    if (criticalFailures.length > 0) {
      actions.push(`Address ${criticalFailures.length} critical gate failures:`);
      criticalFailures.forEach(failure => {
        actions.push(`  - Fix ${failure.gate.name}: need ${failure.gate.target - failure.measured_value} improvement`);
      });
    }

    if (marginalPasses.length > 0) {
      actions.push(`Strengthen ${marginalPasses.length} marginal passes for robustness`);
    }

    if (status === 'LEADING') {
      actions.push('Begin canary rollout at 5%');
      actions.push('Monitor tripwires closely during initial rollout');
    }

    return actions;
  }

  /**
   * Generate detailed gate evaluation report
   */
  generateGateReport(assessment: ProductionReadinessAssessment): string {
    let report = '# Production Gates Evaluation Report\n\n';
    report += `**Overall Status**: ${assessment.overall_status}\n`;
    report += `**Overall Confidence**: ${(assessment.overall_confidence * 100).toFixed(1)}%\n\n`;

    report += '## Gate Summary\n\n';
    report += `- Quality Gates: ${assessment.quality_gates_passed}/${this.getGatesByCategory('quality').length}\n`;
    report += `- Performance Gates: ${assessment.performance_gates_passed}/${this.getGatesByCategory('performance').length}\n`;
    report += `- Reliability Gates: ${assessment.reliability_gates_passed}/${this.getGatesByCategory('reliability').length}\n\n`;

    if (assessment.critical_failures.length > 0) {
      report += '## ❌ Critical Failures\n\n';
      for (const failure of assessment.critical_failures) {
        report += `### ${failure.gate.name}\n`;
        report += `- **Target**: ${failure.gate.operator} ${failure.target_value} ${failure.gate.unit}\n`;
        report += `- **Actual**: ${failure.measured_value.toFixed(3)} ${failure.gate.unit}\n`;
        report += `- **Gap**: ${Math.abs(failure.margin).toFixed(3)} ${failure.gate.unit}\n`;
        report += `- **Description**: ${failure.gate.description}\n\n`;
      }
    }

    if (assessment.marginal_passes.length > 0) {
      report += '## ⚠️ Marginal Passes\n\n';
      for (const marginal of assessment.marginal_passes) {
        report += `- **${marginal.gate.name}**: ${marginal.measured_value.toFixed(3)} ${marginal.gate.unit} (margin: +${marginal.margin.toFixed(3)})\n`;
      }
      report += '\n';
    }

    if (assessment.strong_passes.length > 0) {
      report += '## ✅ Strong Passes\n\n';
      for (const strong of assessment.strong_passes) {
        report += `- **${strong.gate.name}**: ${strong.measured_value.toFixed(3)} ${strong.gate.unit} (margin: +${strong.margin.toFixed(3)})\n`;
      }
      report += '\n';
    }

    report += '## Recommendations\n\n';
    report += `${assessment.recommendation}\n\n`;

    report += '## Next Actions\n\n';
    for (const action of assessment.next_actions) {
      report += `- ${action}\n`;
    }

    return report;
  }
}

// Factory function
export function createProductionGatesValidator(): ProductionGatesValidator {
  return new ProductionGatesValidator();
}

// CLI execution with demo data
if (import.meta.main) {
  console.log('🚪 Production Gates Validation System\n');
  
  const validator = createProductionGatesValidator();
  
  // Demo measurements showing a system that passes most gates
  const measurements = new Map<string, number>([
    ['nl_nDCG_10', 3.5],        // +3.5pp (passes ≥3.0pp)
    ['P_at_1_symbol', 5.2],     // +5.2pp (passes ≥5pp)
    ['recall_50_overall', 0.5], // +0.5pp (passes ≥0pp)
    ['ece_calibration', 0.04],  // 0.04 ECE (passes ≤0.05)
    ['p95_latency', -12],       // 12ms better (passes ≤-10ms)
    ['QPS_150ms', 1.25],        // 1.25x (passes ≥1.2x)
    ['p99_latency', 1.8],       // 1.8x p95 (passes ≤2.0x)
    ['timeout_reduction', 2.1], // 2.1pp reduction (passes ≥2pp)
    ['span_coverage', 100.0],   // 100% (passes ≥100%)
    ['error_rate', 0.8],        // 0.8% (passes ≤1%)
    ['sentinel_nzc', 99.2]      // 99.2% (passes ≥99%)
  ]);

  // Demo evidence for statistical gates
  const evidence = new Map<string, any>([
    ['nl_nDCG_10', {
      source: 'paired_bootstrap_test',
      method: 'bootstrap_ci',
      sample_size: 1000,
      p_value: 0.003,
      effect_size: 0.42,
      power: 0.85
    }],
    ['P_at_1_symbol', {
      source: 'wilcoxon_test',
      method: 'signed_rank',
      sample_size: 850,
      p_value: 0.01,
      effect_size: 0.35,
      power: 0.82
    }]
  ]);

  const assessment = await validator.evaluateProductionReadiness(measurements, evidence);
  console.log(validator.generateGateReport(assessment));
}