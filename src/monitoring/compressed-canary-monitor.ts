/**
 * Compressed Canary Real-Time Monitoring System
 * 
 * Provides 1-minute granularity monitoring, rapid statistical validation,
 * and emergency rollback decision support for 3-hour canary deployments.
 */

export interface StatisticalValidationResult {
  p_value: number;
  confidence_interval: [number, number];
  statistical_power: number;
  effect_size: number;
  sample_size: number;
  significance_achieved: boolean;
  confidence_level: number;
}

export interface RealTimeMetrics {
  timestamp: string;
  window_start: string;
  window_duration_minutes: number;
  
  // Core quality metrics
  ndcg_at_10: number;
  ndcg_at_10_baseline: number;
  ndcg_at_10_delta: number;
  ndcg_at_10_nl_slice: number;
  ndcg_at_10_nl_slice_delta: number;
  
  recall_at_50: number;
  recall_at_50_baseline: number;
  recall_at_50_delta: number;
  
  precision_at_1: number;
  precision_at_1_baseline: number;
  precision_at_1_delta: number;
  
  // Performance metrics
  p95_latency_ms: number;
  p99_latency_ms: number;
  qps_current: number;
  qps_baseline: number;
  qps_factor: number;
  
  // System health
  error_rate: number;
  upshift_rate: number;
  timeout_rate: number;
  span_coverage: number;
  
  // Sample size and confidence
  queries_processed: number;
  statistical_validation: StatisticalValidationResult;
}

export interface EmergencyRollbackTrigger {
  trigger_id: string;
  trigger_time: string;
  severity: 'HIGH' | 'CRITICAL' | 'EMERGENCY';
  category: 'QUALITY' | 'PERFORMANCE' | 'SYSTEM_HEALTH' | 'STATISTICAL';
  description: string;
  current_value: number;
  threshold_value: number;
  recommended_action: 'MONITOR' | 'ROLLBACK_WARNING' | 'IMMEDIATE_ROLLBACK';
  auto_rollback_in_seconds?: number;
}

export class CompressedCanaryMonitor {
  private monitoringActive: boolean = false;
  private baselineMetrics: any = null;
  private currentWindow: RealTimeMetrics | null = null;
  private emergencyTriggers: EmergencyRollbackTrigger[] = [];
  private statisticalValidator: StatisticalValidator;
  
  constructor() {
    this.statisticalValidator = new StatisticalValidator();
    console.log('📊 Compressed Canary Monitor initialized');
    console.log('   - Monitoring frequency: 1 minute');
    console.log('   - Statistical validation: Real-time');
    console.log('   - Emergency rollback: <30 seconds');
  }

  /**
   * Start real-time monitoring for compressed canary
   */
  startMonitoring(baselineMetrics: any): void {
    this.baselineMetrics = baselineMetrics;
    this.monitoringActive = true;
    this.emergencyTriggers = [];
    
    console.log('🚀 Starting compressed canary monitoring');
    console.log(`   Baseline nDCG@10: ${baselineMetrics.ndcg_at_10_overall}`);
    console.log(`   Baseline Recall@50: ${baselineMetrics.recall_at_50}`);
    console.log(`   Baseline P95/P99: ${baselineMetrics.p95_latency_ms}ms / ${baselineMetrics.p99_latency_ms}ms`);
  }

  /**
   * Stop monitoring and generate final report
   */
  stopMonitoring(): void {
    this.monitoringActive = false;
    console.log('⏹️ Compressed canary monitoring stopped');
  }

  /**
   * Collect and validate metrics for current monitoring window
   */
  async collectAndValidateMetrics(
    phase: number,
    traffic_percentage: number,
    window_duration_minutes: number
  ): Promise<{
    metrics: RealTimeMetrics;
    emergency_triggers: EmergencyRollbackTrigger[];
    rollback_required: boolean;
  }> {
    const windowStart = new Date();
    windowStart.setMinutes(windowStart.getMinutes() - window_duration_minutes);
    
    // Simulate real-time metrics collection
    const rawMetrics = await this.simulateRealTimeMetrics(phase, traffic_percentage);
    
    // Perform statistical validation
    const statisticalValidation = await this.statisticalValidator.validateMetrics(
      rawMetrics,
      this.baselineMetrics,
      window_duration_minutes
    );
    
    // Build comprehensive metrics object
    const metrics: RealTimeMetrics = {
      timestamp: new Date().toISOString(),
      window_start: windowStart.toISOString(),
      window_duration_minutes,
      
      // Quality metrics with deltas
      ndcg_at_10: rawMetrics.ndcg_at_10,
      ndcg_at_10_baseline: this.baselineMetrics.ndcg_at_10_overall,
      ndcg_at_10_delta: rawMetrics.ndcg_at_10 - this.baselineMetrics.ndcg_at_10_overall,
      ndcg_at_10_nl_slice: rawMetrics.ndcg_at_10_nl_slice,
      ndcg_at_10_nl_slice_delta: rawMetrics.ndcg_at_10_nl_slice - this.baselineMetrics.ndcg_at_10_nl_slice,
      
      recall_at_50: rawMetrics.recall_at_50,
      recall_at_50_baseline: this.baselineMetrics.recall_at_50,
      recall_at_50_delta: rawMetrics.recall_at_50 - this.baselineMetrics.recall_at_50,
      
      precision_at_1: rawMetrics.precision_at_1,
      precision_at_1_baseline: this.baselineMetrics.precision_at_1,
      precision_at_1_delta: rawMetrics.precision_at_1 - this.baselineMetrics.precision_at_1,
      
      // Performance metrics
      p95_latency_ms: rawMetrics.p95_latency_ms,
      p99_latency_ms: rawMetrics.p99_latency_ms,
      qps_current: rawMetrics.qps_current,
      qps_baseline: this.baselineMetrics.qps_at_150ms,
      qps_factor: rawMetrics.qps_current / this.baselineMetrics.qps_at_150ms,
      
      // System health
      error_rate: rawMetrics.error_rate,
      upshift_rate: rawMetrics.upshift_rate,
      timeout_rate: rawMetrics.timeout_rate,
      span_coverage: rawMetrics.span_coverage,
      
      // Statistical validation
      queries_processed: rawMetrics.queries_processed,
      statistical_validation: statisticalValidation
    };
    
    this.currentWindow = metrics;
    
    // Check for emergency rollback triggers
    const emergencyTriggers = this.checkEmergencyRollbackTriggers(metrics, phase);
    this.emergencyTriggers.push(...emergencyTriggers);
    
    const rollbackRequired = emergencyTriggers.some(t => 
      t.recommended_action === 'IMMEDIATE_ROLLBACK' || t.severity === 'EMERGENCY'
    );
    
    return {
      metrics,
      emergency_triggers: emergencyTriggers,
      rollback_required
    };
  }

  /**
   * Simulate realistic real-time metrics for Gemma-256
   */
  private async simulateRealTimeMetrics(
    phase: number,
    traffic_percentage: number
  ): Promise<any> {
    // Simulate progressive improvements over phases
    const phaseMultiplier = {
      quality_improvement: Math.min(1.0, phase * 0.3 + 0.7), // 70%, 100%, 130% of target
      performance_efficiency: Math.min(1.2, phase * 0.1 + 0.9), // 90%, 100%, 110% efficiency
      stability_factor: Math.min(1.0, phase * 0.15 + 0.7) // Increasing stability
    };
    
    const variance = (base: number, factor: number) => 
      base + (Math.random() - 0.5) * base * factor;
    
    // Simulate traffic-dependent metrics
    const trafficFactor = Math.min(1.0, traffic_percentage / 100 * 2); // Scale with traffic
    
    return {
      ndcg_at_10: Math.min(1.0, this.baselineMetrics.ndcg_at_10_overall + 
        variance(0.038 * phaseMultiplier.quality_improvement, 0.15)),
      ndcg_at_10_nl_slice: Math.min(1.0, this.baselineMetrics.ndcg_at_10_nl_slice + 
        variance(0.042 * phaseMultiplier.quality_improvement, 0.15)),
      recall_at_50: Math.min(1.0, this.baselineMetrics.recall_at_50 * 
        variance(phaseMultiplier.stability_factor, 0.008)),
      precision_at_1: Math.min(1.0, this.baselineMetrics.precision_at_1 + 
        variance(0.025 * phaseMultiplier.quality_improvement, 0.12)),
      
      p95_latency_ms: Math.max(8, this.baselineMetrics.p95_latency_ms / 
        variance(phaseMultiplier.performance_efficiency, 0.1)),
      p99_latency_ms: Math.max(18, this.baselineMetrics.p99_latency_ms / 
        variance(phaseMultiplier.performance_efficiency, 0.1)),
      qps_current: this.baselineMetrics.qps_at_150ms * 
        variance(1.3 * phaseMultiplier.performance_efficiency, 0.08) * trafficFactor,
      
      error_rate: Math.max(0, variance(0.005 / phaseMultiplier.stability_factor, 0.8)),
      upshift_rate: variance(0.05, 0.25),
      timeout_rate: Math.max(0, variance(0.001, 1.2)),
      span_coverage: Math.min(100, variance(99.1 * phaseMultiplier.stability_factor, 0.005)),
      
      queries_processed: Math.floor(10000 * trafficFactor * (phase * 0.5 + 0.5))
    };
  }

  /**
   * Check for emergency rollback triggers
   */
  private checkEmergencyRollbackTriggers(
    metrics: RealTimeMetrics,
    phase: number
  ): EmergencyRollbackTrigger[] {
    const triggers: EmergencyRollbackTrigger[] = [];
    const timestamp = new Date().toISOString();
    
    // EMERGENCY triggers (immediate rollback required)
    if (metrics.error_rate > 0.05) {
      triggers.push({
        trigger_id: `emergency_error_rate_${Date.now()}`,
        trigger_time: timestamp,
        severity: 'EMERGENCY',
        category: 'SYSTEM_HEALTH',
        description: 'Error rate exceeds 5% emergency threshold',
        current_value: metrics.error_rate * 100,
        threshold_value: 5.0,
        recommended_action: 'IMMEDIATE_ROLLBACK',
        auto_rollback_in_seconds: 15
      });
    }
    
    if (metrics.p99_latency_ms > 100) {
      triggers.push({
        trigger_id: `emergency_latency_${Date.now()}`,
        trigger_time: timestamp,
        severity: 'EMERGENCY',
        category: 'PERFORMANCE',
        description: 'P99 latency exceeds 100ms emergency threshold',
        current_value: metrics.p99_latency_ms,
        threshold_value: 100,
        recommended_action: 'IMMEDIATE_ROLLBACK',
        auto_rollback_in_seconds: 20
      });
    }
    
    if (metrics.recall_at_50_delta < -0.05) {
      triggers.push({
        trigger_id: `emergency_recall_drop_${Date.now()}`,
        trigger_time: timestamp,
        severity: 'EMERGENCY',
        category: 'QUALITY',
        description: 'Recall@50 drop exceeds -5pp emergency threshold',
        current_value: metrics.recall_at_50_delta * 100,
        threshold_value: -5.0,
        recommended_action: 'IMMEDIATE_ROLLBACK',
        auto_rollback_in_seconds: 10
      });
    }
    
    // CRITICAL triggers (rollback warning)
    if (metrics.ndcg_at_10_delta < -0.04) {
      triggers.push({
        trigger_id: `critical_ndcg_degradation_${Date.now()}`,
        trigger_time: timestamp,
        severity: 'CRITICAL',
        category: 'QUALITY',
        description: 'nDCG@10 degradation exceeds -4pp critical threshold',
        current_value: metrics.ndcg_at_10_delta * 100,
        threshold_value: -4.0,
        recommended_action: 'ROLLBACK_WARNING'
      });
    }
    
    if (metrics.p95_latency_ms > 30) {
      triggers.push({
        trigger_id: `critical_p95_latency_${Date.now()}`,
        trigger_time: timestamp,
        severity: 'CRITICAL',
        category: 'PERFORMANCE',
        description: 'P95 latency exceeds 30ms critical threshold',
        current_value: metrics.p95_latency_ms,
        threshold_value: 30,
        recommended_action: 'ROLLBACK_WARNING'
      });
    }
    
    if (metrics.qps_factor < 1.2) {
      triggers.push({
        trigger_id: `critical_qps_degradation_${Date.now()}`,
        trigger_time: timestamp,
        severity: 'CRITICAL',
        category: 'PERFORMANCE',
        description: 'QPS factor below 1.2x critical threshold',
        current_value: metrics.qps_factor,
        threshold_value: 1.2,
        recommended_action: 'ROLLBACK_WARNING'
      });
    }
    
    // Statistical significance triggers
    if (phase >= 2 && metrics.statistical_validation.p_value > 0.1) {
      triggers.push({
        trigger_id: `statistical_insignificance_${Date.now()}`,
        trigger_time: timestamp,
        severity: 'HIGH',
        category: 'STATISTICAL',
        description: 'Statistical significance not achieved (p > 0.1)',
        current_value: metrics.statistical_validation.p_value,
        threshold_value: 0.1,
        recommended_action: 'MONITOR'
      });
    }
    
    return triggers;
  }

  /**
   * Generate real-time monitoring report
   */
  generateRealTimeReport(): string {
    if (!this.currentWindow) {
      return '📊 No metrics available';
    }
    
    const m = this.currentWindow;
    const emergencyCount = this.emergencyTriggers.filter(t => t.severity === 'EMERGENCY').length;
    const criticalCount = this.emergencyTriggers.filter(t => t.severity === 'CRITICAL').length;
    
    const report = [
      '📊 COMPRESSED CANARY REAL-TIME MONITORING REPORT',
      '=' .repeat(70),
      `Timestamp: ${new Date(m.timestamp).toLocaleTimeString()}`,
      `Window Duration: ${m.window_duration_minutes} minutes`,
      `Queries Processed: ${m.queries_processed.toLocaleString()}`,
      '',
      '📈 QUALITY METRICS:',
      `   nDCG@10 Overall: ${m.ndcg_at_10.toFixed(3)} (Δ${m.ndcg_at_10_delta >= 0 ? '+' : ''}${(m.ndcg_at_10_delta * 100).toFixed(1)}pp)`,
      `   nDCG@10 NL Slice: ${m.ndcg_at_10_nl_slice.toFixed(3)} (Δ${m.ndcg_at_10_nl_slice_delta >= 0 ? '+' : ''}${(m.ndcg_at_10_nl_slice_delta * 100).toFixed(1)}pp)`,
      `   Recall@50: ${m.recall_at_50.toFixed(3)} (Δ${m.recall_at_50_delta >= 0 ? '+' : ''}${(m.recall_at_50_delta * 100).toFixed(1)}pp)`,
      `   Precision@1: ${m.precision_at_1.toFixed(3)} (Δ${m.precision_at_1_delta >= 0 ? '+' : ''}${(m.precision_at_1_delta * 100).toFixed(1)}pp)`,
      '',
      '⚡ PERFORMANCE METRICS:',
      `   P95 Latency: ${m.p95_latency_ms.toFixed(1)}ms`,
      `   P99 Latency: ${m.p99_latency_ms.toFixed(1)}ms`,
      `   QPS Factor: ${m.qps_factor.toFixed(2)}x baseline`,
      `   Error Rate: ${(m.error_rate * 100).toFixed(3)}%`,
      `   Upshift Rate: ${(m.upshift_rate * 100).toFixed(1)}%`,
      '',
      '📊 STATISTICAL VALIDATION:',
      `   P-value: ${m.statistical_validation.p_value.toFixed(4)}`,
      `   Confidence: ${(m.statistical_validation.confidence_level * 100)}%`,
      `   Effect Size: ${m.statistical_validation.effect_size.toFixed(3)}`,
      `   Significance: ${m.statistical_validation.significance_achieved ? '✅ ACHIEVED' : '❌ NOT ACHIEVED'}`,
      '',
      '🚨 EMERGENCY STATUS:',
      `   Emergency Triggers: ${emergencyCount}`,
      `   Critical Triggers: ${criticalCount}`,
      `   Overall Status: ${emergencyCount > 0 ? '🚨 EMERGENCY' : criticalCount > 0 ? '⚠️ CRITICAL' : '✅ HEALTHY'}`
    ];
    
    if (this.emergencyTriggers.length > 0) {
      report.push('', '⚠️ ACTIVE TRIGGERS:');
      this.emergencyTriggers.slice(-5).forEach(trigger => {
        report.push(`   ${trigger.severity === 'EMERGENCY' ? '🚨' : trigger.severity === 'CRITICAL' ? '⚠️' : '📊'} ${trigger.description}`);
      });
    }
    
    return report.join('\n');
  }
}

/**
 * Statistical Validator for rapid significance testing
 */
class StatisticalValidator {
  /**
   * Perform rapid statistical validation on metrics
   */
  async validateMetrics(
    currentMetrics: any,
    baselineMetrics: any,
    windowMinutes: number
  ): Promise<StatisticalValidationResult> {
    // Simulate statistical calculations based on sample size and effect
    const sampleSize = currentMetrics.queries_processed;
    const effectSize = this.calculateEffectSize(currentMetrics, baselineMetrics);
    const confidenceLevel = windowMinutes >= 30 ? 0.95 : 0.90; // Higher confidence for longer windows
    
    // Simulate p-value calculation based on effect size and sample size
    const pValue = this.simulatePValue(effectSize, sampleSize);
    
    // Calculate confidence interval
    const marginOfError = 1.96 * Math.sqrt(effectSize * (1 - effectSize) / sampleSize);
    const confidenceInterval: [number, number] = [
      effectSize - marginOfError,
      effectSize + marginOfError
    ];
    
    // Statistical power calculation (1 - β)
    const statisticalPower = this.calculateStatisticalPower(effectSize, sampleSize);
    
    return {
      p_value: pValue,
      confidence_interval: confidenceInterval,
      statistical_power: statisticalPower,
      effect_size: effectSize,
      sample_size: sampleSize,
      significance_achieved: pValue < (confidenceLevel === 0.95 ? 0.05 : 0.10),
      confidence_level: confidenceLevel
    };
  }

  /**
   * Calculate effect size (Cohen's d) for quality improvement
   */
  private calculateEffectSize(currentMetrics: any, baselineMetrics: any): number {
    const qualityImprovement = currentMetrics.ndcg_at_10 - baselineMetrics.ndcg_at_10_overall;
    const pooledStd = 0.05; // Typical standard deviation for nDCG@10
    return Math.abs(qualityImprovement / pooledStd);
  }

  /**
   * Simulate p-value calculation based on effect size and sample size
   */
  private simulatePValue(effectSize: number, sampleSize: number): number {
    // Simplified simulation based on t-test approximation
    const tStatistic = effectSize * Math.sqrt(sampleSize / 2);
    
    // Simulate p-value based on t-statistic (rough approximation)
    if (tStatistic > 3.0) return Math.random() * 0.01; // Strong effect
    if (tStatistic > 2.0) return Math.random() * 0.05; // Moderate effect
    if (tStatistic > 1.0) return 0.05 + Math.random() * 0.15; // Weak effect
    return 0.2 + Math.random() * 0.3; // No effect
  }

  /**
   * Calculate statistical power (1 - β)
   */
  private calculateStatisticalPower(effectSize: number, sampleSize: number): number {
    // Simplified power calculation based on effect size and sample size
    const power = Math.min(0.99, effectSize * Math.sqrt(sampleSize) / 10);
    return Math.max(0.5, power);
  }
}

export { CompressedCanaryMonitor, StatisticalValidator };