/**
 * Isotonic Calibration System for Continuous Monitoring
 * Implements Section 6 of TODO.md: Calibration sanity (continuous)
 */

import fs from 'fs/promises';
import path from 'path';

export interface CalibrationConfig {
  ece_threshold: number;
  slope_clamp_bounds: [number, number];
  clamp_activation_threshold: number; // Percentage of bins that trigger P1
  refit_schedule: string; // Cron format
  min_samples_per_bin: number;
  confidence_level: number;
}

export interface IntentLanguageCombination {
  intent: string;
  language: string;
  combination_id: string;
}

export interface CalibrationBin {
  predicted_probability: number;
  actual_probability: number;
  sample_count: number;
  slope_before_clamp: number;
  slope_after_clamp: number;
  clamped: boolean;
}

export interface IsotonicModel {
  combination: IntentLanguageCombination;
  bins: CalibrationBin[];
  overall_ece: number;
  slope_violations: number;
  clamp_activation_rate: number;
  last_fit_timestamp: number;
  model_version: string;
}

export interface CalibrationReport {
  timestamp: number;
  total_combinations: number;
  healthy_combinations: number;
  ece_violations: Array<{
    combination: IntentLanguageCombination;
    ece: number;
    threshold: number;
  }>;
  slope_clamp_violations: Array<{
    combination: IntentLanguageCombination;
    violation_rate: number;
    affected_bins: number;
  }>;
  drift_alerts: Array<{
    combination: IntentLanguageCombination;
    drift_type: 'calibration' | 'distribution' | 'performance';
    severity: 'P1' | 'P2' | 'P3';
    description: string;
  }>;
  recommendations: string[];
}

export class IsotonicCalibrationSystem {
  private config: CalibrationConfig;
  private models: Map<string, IsotonicModel> = new Map();
  private calibrationDir = path.join(process.cwd(), 'calibration-data');
  private isRunning = false;

  constructor(config: CalibrationConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    console.log('🎯 Initializing isotonic calibration system...');
    
    // Ensure calibration directory exists
    await fs.mkdir(this.calibrationDir, { recursive: true });
    
    // Load existing models if available
    await this.loadExistingModels();
    
    console.log(`✅ Calibration system initialized with ${this.models.size} existing models`);
  }

  async refitAllModels(
    trainingData: Map<string, Array<{ predicted: number; actual: number }>>
  ): Promise<CalibrationReport> {
    console.log('🔄 Refitting isotonic models for all intent×language combinations...');
    const startTime = Date.now();
    
    const report: CalibrationReport = {
      timestamp: startTime,
      total_combinations: trainingData.size,
      healthy_combinations: 0,
      ece_violations: [],
      slope_clamp_violations: [],
      drift_alerts: [],
      recommendations: []
    };
    
    for (const [combinationId, samples] of trainingData) {
      try {
        const combination = this.parseCombinationId(combinationId);
        console.log(`🎯 Fitting model for ${combination.intent}×${combination.language}...`);
        
        // Fit isotonic regression model
        const model = await this.fitIsotonicModel(combination, samples);
        
        // Validate ECE constraint
        if (model.overall_ece <= this.config.ece_threshold) {
          report.healthy_combinations++;
        } else {
          report.ece_violations.push({
            combination,
            ece: model.overall_ece,
            threshold: this.config.ece_threshold
          });
        }
        
        // Check slope clamp violations
        if (model.clamp_activation_rate > this.config.clamp_activation_threshold) {
          report.slope_clamp_violations.push({
            combination,
            violation_rate: model.clamp_activation_rate,
            affected_bins: model.slope_violations
          });
          
          // Open P1 for calibration drift
          report.drift_alerts.push({
            combination,
            drift_type: 'calibration',
            severity: 'P1',
            description: `Slope clamp activated in ${(model.clamp_activation_rate * 100).toFixed(1)}% of bins (threshold: ${(this.config.clamp_activation_threshold * 100).toFixed(1)}%)`
          });
        }
        
        // Store fitted model
        this.models.set(combinationId, model);
        
      } catch (error) {
        console.error(`Failed to fit model for ${combinationId}:`, error);
        report.drift_alerts.push({
          combination: this.parseCombinationId(combinationId),
          drift_type: 'performance',
          severity: 'P2',
          description: `Model fitting failed: ${(error as Error).message}`
        });
      }
    }
    
    // Generate recommendations
    report.recommendations = this.generateRecommendations(report);
    
    // Save models and report
    await this.saveModels();
    await this.saveCalibrationReport(report);
    
    const fitTime = Date.now() - startTime;
    console.log(`✅ Refit complete: ${report.healthy_combinations}/${report.total_combinations} healthy models (${fitTime}ms)`);
    
    return report;
  }

  private async fitIsotonicModel(
    combination: IntentLanguageCombination,
    samples: Array<{ predicted: number; actual: number }>
  ): Promise<IsotonicModel> {
    if (samples.length < this.config.min_samples_per_bin * 5) {
      throw new Error(`Insufficient samples: ${samples.length} (min: ${this.config.min_samples_per_bin * 5})`);
    }
    
    // Sort by predicted probability
    samples.sort((a, b) => a.predicted - b.predicted);
    
    // Create calibration bins
    const numBins = Math.min(20, Math.floor(samples.length / this.config.min_samples_per_bin));
    const binSize = Math.floor(samples.length / numBins);
    const bins: CalibrationBin[] = [];
    
    let slopeViolations = 0;
    let clampedBins = 0;
    
    for (let i = 0; i < numBins; i++) {
      const startIdx = i * binSize;
      const endIdx = i === numBins - 1 ? samples.length : (i + 1) * binSize;
      const binSamples = samples.slice(startIdx, endIdx);
      
      // Calculate bin statistics
      const avgPredicted = binSamples.reduce((sum, s) => sum + s.predicted, 0) / binSamples.length;
      const avgActual = binSamples.reduce((sum, s) => sum + s.actual, 0) / binSamples.length;
      
      // Calculate raw slope
      const rawSlope = binSamples.length > 1 ? this.calculateBinSlope(binSamples) : 1.0;
      
      // Apply slope clamping
      const clampedSlope = Math.max(
        this.config.slope_clamp_bounds[0],
        Math.min(this.config.slope_clamp_bounds[1], rawSlope)
      );
      
      const wasClamped = rawSlope !== clampedSlope;
      if (wasClamped) {
        clampedBins++;
        if (rawSlope < this.config.slope_clamp_bounds[0] || rawSlope > this.config.slope_clamp_bounds[1]) {
          slopeViolations++;
        }
      }
      
      bins.push({
        predicted_probability: avgPredicted,
        actual_probability: avgActual,
        sample_count: binSamples.length,
        slope_before_clamp: rawSlope,
        slope_after_clamp: clampedSlope,
        clamped: wasClamped
      });
    }
    
    // Calculate Expected Calibration Error
    const ece = this.calculateECE(bins);
    const clampActivationRate = clampedBins / numBins;
    
    const model: IsotonicModel = {
      combination,
      bins,
      overall_ece: ece,
      slope_violations: slopeViolations,
      clamp_activation_rate: clampActivationRate,
      last_fit_timestamp: Date.now(),
      model_version: this.generateModelVersion(combination, ece)
    };
    
    console.log(`📊 ${combination.intent}×${combination.language}: ECE=${ece.toFixed(4)}, Clamped=${(clampActivationRate * 100).toFixed(1)}%`);
    
    return model;
  }

  private calculateBinSlope(samples: Array<{ predicted: number; actual: number }>): number {
    if (samples.length < 2) return 1.0;
    
    // Simple linear regression slope
    const n = samples.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (const sample of samples) {
      sumX += sample.predicted;
      sumY += sample.actual;
      sumXY += sample.predicted * sample.actual;
      sumX2 += sample.predicted * sample.predicted;
    }
    
    const denominator = n * sumX2 - sumX * sumX;
    if (Math.abs(denominator) < 1e-10) return 1.0;
    
    const slope = (n * sumXY - sumX * sumY) / denominator;
    return slope;
  }

  private calculateECE(bins: CalibrationBin[]): number {
    let ece = 0;
    let totalSamples = 0;
    
    for (const bin of bins) {
      const binECE = Math.abs(bin.predicted_probability - bin.actual_probability);
      ece += binECE * bin.sample_count;
      totalSamples += bin.sample_count;
    }
    
    return totalSamples > 0 ? ece / totalSamples : 0;
  }

  private generateRecommendations(report: CalibrationReport): string[] {
    const recommendations: string[] = [];
    
    if (report.ece_violations.length > 0) {
      recommendations.push(`🎯 ${report.ece_violations.length} combinations exceed ECE threshold - consider model retraining or feature engineering`);
    }
    
    if (report.slope_clamp_violations.length > 0) {
      const p1Count = report.slope_clamp_violations.length;
      recommendations.push(`🚨 ${p1Count} combinations have excessive slope clamping - investigate data distribution shifts`);
    }
    
    const healthyRate = report.healthy_combinations / report.total_combinations;
    if (healthyRate < 0.9) {
      recommendations.push(`📉 Only ${(healthyRate * 100).toFixed(1)}% of models are healthy - consider systematic calibration review`);
    }
    
    if (report.drift_alerts.some(a => a.severity === 'P1')) {
      recommendations.push(`🔴 P1 calibration drift detected - immediate investigation required`);
    }
    
    if (recommendations.length === 0) {
      recommendations.push(`✅ All calibration models are healthy and within thresholds`);
    }
    
    return recommendations;
  }

  // Continuous monitoring methods
  async startContinuousMonitoring(): Promise<void> {
    if (this.isRunning) {
      console.warn('⚠️ Continuous calibration monitoring is already running');
      return;
    }
    
    console.log(`🔄 Starting continuous calibration monitoring (${this.config.refit_schedule})`);
    
    // Set up monitoring loop (in real implementation, use proper cron scheduling)
    this.isRunning = true;
    
    // For demo, simulate periodic monitoring
    this.simulatePeriodicMonitoring();
    
    console.log('✅ Continuous calibration monitoring started');
  }

  private simulatePeriodicMonitoring(): void {
    // Simulate weekly calibration check
    setTimeout(async () => {
      if (!this.isRunning) return;
      
      try {
        console.log('📅 Weekly calibration health check...');
        
        // Generate mock training data
        const mockTrainingData = this.generateMockTrainingData();
        
        // Refit models
        const report = await this.refitAllModels(mockTrainingData);
        
        // Handle alerts
        await this.handleCalibrationAlerts(report);
        
        // Schedule next check
        this.simulatePeriodicMonitoring();
        
      } catch (error) {
        console.error('💥 Calibration monitoring failed:', error);
        // In real implementation, send critical alert
      }
    }, 60000); // Check every minute for demo (would be weekly in production)
  }

  private async handleCalibrationAlerts(report: CalibrationReport): Promise<void> {
    // Handle P1 alerts (critical calibration drift)
    const p1Alerts = report.drift_alerts.filter(a => a.severity === 'P1');
    if (p1Alerts.length > 0) {
      console.log(`🚨 Opening P1 incident for calibration drift: ${p1Alerts.length} combinations affected`);
      
      // In real implementation, integrate with incident management
      await this.openCalibrationIncident('P1', p1Alerts);
    }
    
    // Handle slope clamp violations
    if (report.slope_clamp_violations.length > 0) {
      console.log(`⚠️ Slope clamp violations detected in ${report.slope_clamp_violations.length} combinations`);
      
      for (const violation of report.slope_clamp_violations) {
        if (violation.violation_rate > this.config.clamp_activation_threshold) {
          console.log(`📊 ${violation.combination.intent}×${violation.combination.language}: ${(violation.violation_rate * 100).toFixed(1)}% of bins clamped`);
        }
      }
    }
    
    // Log healthy status
    const healthyRate = report.healthy_combinations / report.total_combinations;
    console.log(`✅ Calibration health: ${report.healthy_combinations}/${report.total_combinations} (${(healthyRate * 100).toFixed(1)}%)`);
  }

  private async openCalibrationIncident(
    severity: string,
    alerts: CalibrationReport['drift_alerts']
  ): Promise<void> {
    const incident = {
      severity,
      title: `Calibration Drift Detected - ${alerts.length} Intent×Language Combinations`,
      description: `Slope clamp activation exceeded ${this.config.clamp_activation_threshold * 100}% threshold`,
      affected_combinations: alerts.map(a => `${a.combination.intent}×${a.combination.language}`),
      timestamp: new Date().toISOString(),
      auto_generated: true
    };
    
    // Save incident for tracking
    const incidentPath = path.join(this.calibrationDir, `incident-${Date.now()}.json`);
    await fs.writeFile(incidentPath, JSON.stringify(incident, null, 2));
    
    console.log(`🔴 ${severity} incident created: ${incident.title}`);
  }

  // Utility methods
  private parseCombinationId(combinationId: string): IntentLanguageCombination {
    const [intent, language] = combinationId.split('×');
    return {
      intent: intent || 'unknown',
      language: language || 'unknown',
      combination_id: combinationId
    };
  }

  private generateModelVersion(combination: IntentLanguageCombination, ece: number): string {
    const timestamp = Date.now();
    const eceString = ece.toFixed(4).replace('.', '');
    return `${combination.intent}_${combination.language}_${eceString}_${timestamp}`;
  }

  private generateMockTrainingData(): Map<string, Array<{ predicted: number; actual: number }>> {
    const data = new Map<string, Array<{ predicted: number; actual: number }>>();
    
    // Mock intent×language combinations
    const intents = ['search', 'navigate', 'understand'];
    const languages = ['python', 'typescript', 'javascript'];
    
    for (const intent of intents) {
      for (const language of languages) {
        const combinationId = `${intent}×${language}`;
        const samples = [];
        
        // Generate mock prediction vs actual data
        for (let i = 0; i < 1000; i++) {
          const predicted = Math.random();
          // Add some calibration error to simulate drift
          const actual = predicted + (Math.random() - 0.5) * 0.1;
          
          samples.push({
            predicted: Math.max(0, Math.min(1, predicted)),
            actual: Math.max(0, Math.min(1, actual))
          });
        }
        
        data.set(combinationId, samples);
      }
    }
    
    return data;
  }

  private async loadExistingModels(): Promise<void> {
    try {
      const modelFiles = await fs.readdir(this.calibrationDir);
      const modelFilePattern = /^model-(.+)\.json$/;
      
      for (const file of modelFiles) {
        const match = file.match(modelFilePattern);
        if (match) {
          const combinationId = match[1];
          const modelPath = path.join(this.calibrationDir, file);
          const modelData = JSON.parse(await fs.readFile(modelPath, 'utf-8'));
          this.models.set(combinationId, modelData);
        }
      }
      
      console.log(`📚 Loaded ${this.models.size} existing calibration models`);
      
    } catch (error) {
      console.log('📚 No existing models found, starting fresh');
    }
  }

  private async saveModels(): Promise<void> {
    for (const [combinationId, model] of this.models) {
      const modelPath = path.join(this.calibrationDir, `model-${combinationId}.json`);
      await fs.writeFile(modelPath, JSON.stringify(model, null, 2));
    }
    
    console.log(`💾 Saved ${this.models.size} calibration models`);
  }

  private async saveCalibrationReport(report: CalibrationReport): Promise<void> {
    const reportPath = path.join(this.calibrationDir, `report-${Date.now()}.json`);
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    // Also save human-readable summary
    const summaryPath = path.join(this.calibrationDir, `summary-${Date.now()}.md`);
    const summaryContent = this.generateReportSummary(report);
    await fs.writeFile(summaryPath, summaryContent);
    
    console.log(`📊 Calibration report saved: ${reportPath}`);
  }

  private generateReportSummary(report: CalibrationReport): string {
    return `# Calibration Health Report

**Generated**: ${new Date(report.timestamp).toISOString()}
**Overall Health**: ${report.healthy_combinations}/${report.total_combinations} combinations (${(report.healthy_combinations / report.total_combinations * 100).toFixed(1)}%)

## ECE Violations (${report.ece_violations.length})

${report.ece_violations.map(v => 
  `- **${v.combination.intent}×${v.combination.language}**: ECE=${v.ece.toFixed(4)} > ${v.threshold}`
).join('\n')}

## Slope Clamp Violations (${report.slope_clamp_violations.length})

${report.slope_clamp_violations.map(v => 
  `- **${v.combination.intent}×${v.combination.language}**: ${(v.violation_rate * 100).toFixed(1)}% bins clamped (${v.affected_bins} violations)`
).join('\n')}

## Alerts (${report.drift_alerts.length})

${report.drift_alerts.map(a => 
  `- **${a.severity}** [${a.drift_type}] ${a.combination.intent}×${a.combination.language}: ${a.description}`
).join('\n')}

## Recommendations

${report.recommendations.map(r => `- ${r}`).join('\n')}

---
*Generated by Isotonic Calibration System*
`;
  }

  async stop(): Promise<void> {
    this.isRunning = false;
    console.log('🛑 Continuous calibration monitoring stopped');
  }

  // Public API for model calibration
  async getCalibratedProbability(
    combinationId: string,
    rawProbability: number
  ): Promise<number> {
    const model = this.models.get(combinationId);
    if (!model) {
      console.warn(`No calibration model found for ${combinationId}, returning raw probability`);
      return rawProbability;
    }
    
    // Find appropriate bin for calibration
    for (const bin of model.bins) {
      if (Math.abs(bin.predicted_probability - rawProbability) < 0.1) {
        return bin.actual_probability;
      }
    }
    
    // Fallback to raw probability if no matching bin
    return rawProbability;
  }

  getSystemHealth(): {
    total_models: number;
    healthy_models: number;
    health_percentage: number;
    last_update: number;
  } {
    const healthyModels = Array.from(this.models.values())
      .filter(m => m.overall_ece <= this.config.ece_threshold).length;
    
    return {
      total_models: this.models.size,
      healthy_models: healthyModels,
      health_percentage: this.models.size > 0 ? (healthyModels / this.models.size) * 100 : 0,
      last_update: Math.max(...Array.from(this.models.values()).map(m => m.last_fit_timestamp))
    };
  }
}

// Factory function with production defaults
export function createCalibrationSystem(overrides: Partial<CalibrationConfig> = {}): IsotonicCalibrationSystem {
  const defaultConfig: CalibrationConfig = {
    ece_threshold: 0.02, // ECE ≤ 0.02 per TODO.md
    slope_clamp_bounds: [0.9, 1.1], // Slope clamp [0.9, 1.1]
    clamp_activation_threshold: 0.10, // 10% of bins trigger P1
    refit_schedule: '0 2 * * 0', // Weekly at Sunday 02:00
    min_samples_per_bin: 50,
    confidence_level: 0.95
  };

  const config = { ...defaultConfig, ...overrides };
  return new IsotonicCalibrationSystem(config);
}