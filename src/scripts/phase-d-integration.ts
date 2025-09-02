#!/usr/bin/env node
/**
 * Phase D Integration Script - Complete release readiness validation
 * Orchestrates the entire Phase D rollout process for lens v1.0 release
 */

import { RCReleaseManager, RCBuildConfig } from '../core/rc-release-manager.js';
import { SignoffManager, SignoffConfigFactory } from '../core/signoff-manager.js';
import { TailLatencyMonitor, MonitoringConfigFactory, BenchmarkLatencyIntegration } from '../core/tail-latency-monitor.js';
import { checkCompatibility, SERVER_API_VERSION, SERVER_INDEX_VERSION } from '../core/version-manager.js';
import { MigrationManager } from '../core/migration-manager.js';

interface PhaseDConfig {
  rc_version: string;
  environment: 'staging' | 'production';
  validation_duration_hours: number;
  enable_monitoring: boolean;
  enable_signoff_process: boolean;
  stakeholder_approvals_required: string[];
  output_dir: string;
}

interface PhaseDResults {
  phase_d_success: boolean;
  timestamp: string;
  rc_build_result?: any;
  compatibility_result?: any;
  nightly_validation_results?: any[];
  signoff_status?: any;
  promotion_result?: any;
  monitoring_summary?: any;
  recommendations: string[];
  next_steps: string[];
}

/**
 * Phase D Integration Orchestrator
 * Manages the complete RC to production promotion workflow
 */
class PhaseDOrchestrator {
  private config: PhaseDConfig;
  private rcManager: RCReleaseManager;
  private signoffManager: SignoffManager;
  private tailLatencyMonitor: TailLatencyMonitor;
  private benchmarkIntegration: BenchmarkLatencyIntegration;

  constructor(config: PhaseDConfig) {
    this.config = config;
    
    // Initialize RC Release Manager
    const rcConfig: RCBuildConfig = {
      version: config.rc_version,
      target_env: config.environment === 'production' ? 'production' : 'rc',
      enable_sbom: true,
      enable_sast: true,
      enable_container: true,
      enable_provenance: true,
      output_dir: config.output_dir
    };
    this.rcManager = new RCReleaseManager(rcConfig);
    
    // Initialize Sign-off Manager
    const signoffConfig = config.environment === 'production' ?
      SignoffConfigFactory.createProductionConfig() :
      SignoffConfigFactory.createPhaseDBenchmarkConfig();
    this.signoffManager = new SignoffManager(signoffConfig, `${config.output_dir}/signoff-data`);
    
    // Initialize Tail Latency Monitor
    const monitorConfig = MonitoringConfigFactory.createPhaseDBenchmarkConfig();
    this.tailLatencyMonitor = new TailLatencyMonitor(monitorConfig);
    this.benchmarkIntegration = new BenchmarkLatencyIntegration(this.tailLatencyMonitor);
    
    if (config.enable_monitoring) {
      this.tailLatencyMonitor.start();
    }
  }

  /**
   * Execute complete Phase D workflow
   */
  async executePhaseD(): Promise<PhaseDResults> {
    console.log('🚀 Starting Phase D - RC Rollout & Production Promotion');
    console.log(`Version: ${this.config.rc_version}`);
    console.log(`Environment: ${this.config.environment}`);
    console.log(`Validation Duration: ${this.config.validation_duration_hours}h`);
    
    const results: PhaseDResults = {
      phase_d_success: false,
      timestamp: new Date().toISOString(),
      recommendations: [],
      next_steps: []
    };

    try {
      // Phase D.1: Cut RC Build
      console.log('\n📦 Phase D.1: Cutting RC Build...');
      results.rc_build_result = await this.rcManager.cutRC();
      
      if (!results.rc_build_result.success) {
        results.recommendations.push('Fix RC build issues before proceeding');
        results.next_steps.push('Review build artifacts and security scan results');
        return results;
      }
      console.log('✅ RC build completed successfully');

      // Phase D.2: Compatibility Drill
      console.log('\n🔄 Phase D.2: Running Compatibility Drill...');
      results.compatibility_result = await this.rcManager.runCompatibilityDrill(
        this.config.rc_version,
        ['v0.9.0', 'v0.9.1', 'v0.9.2'] // Previous versions
      );
      
      if (!results.compatibility_result.success) {
        results.recommendations.push('Address compatibility issues before nightly validation');
        results.next_steps.push('Review migration paths and API changes');
        return results;
      }
      console.log('✅ Compatibility drill passed');

      // Phase D.3: Nightly Validation Process
      if (this.config.enable_signoff_process) {
        console.log('\n🌙 Phase D.3: Starting Nightly Validation Process...');
        results.nightly_validation_results = await this.runNightlyValidationCycle();
        
        const allNightsPassed = results.nightly_validation_results.every(r => r.success);
        if (!allNightsPassed) {
          results.recommendations.push('Address failing nightly validations');
          results.next_steps.push('Review quality gates and performance metrics');
          return results;
        }
        console.log('✅ All nightly validations passed');
      }

      // Phase D.4: Sign-off Process
      console.log('\n📊 Phase D.4: Checking Sign-off Status...');
      results.signoff_status = this.signoffManager.checkPromotionReadiness();
      
      if (!results.signoff_status.ready_for_promotion) {
        results.recommendations.push(`Complete missing requirements: ${results.signoff_status.missing_requirements.join(', ')}`);
        results.next_steps.push('Wait for stakeholder approvals and quality gates');
        return results;
      }
      console.log('✅ Sign-off criteria satisfied');

      // Phase D.5: Production Promotion (if ready)
      if (this.config.environment === 'production') {
        console.log('\n🎯 Phase D.5: Executing Production Promotion...');
        results.promotion_result = await this.rcManager.promoteToProduction(this.config.rc_version);
        
        if (!results.promotion_result.success) {
          results.recommendations.push('Review promotion failure and initiate rollback if necessary');
          results.next_steps.push('Investigate promotion issues and adjust deployment strategy');
          return results;
        }
        console.log('✅ Production promotion completed');
      }

      // Phase D.6: Monitoring Summary
      if (this.config.enable_monitoring) {
        console.log('\n📊 Phase D.6: Generating Monitoring Summary...');
        results.monitoring_summary = this.tailLatencyMonitor.generateReport();
        
        const healthStatus = this.tailLatencyMonitor.isSystemHealthy();
        if (!healthStatus.healthy) {
          results.recommendations.push('Address tail-latency violations before declaring success');
          results.next_steps.push('Review performance optimization opportunities');
        }
      }

      // Success!
      results.phase_d_success = true;
      results.recommendations.push('Phase D completed successfully - production ready!');
      results.next_steps.push('Monitor production metrics for 24-48 hours');
      results.next_steps.push('Conduct post-deployment retrospective');

      console.log('\n🎉 Phase D completed successfully!');
      return results;

    } catch (error) {
      console.error('\n❌ Phase D failed:', error);
      results.recommendations.push(`Address critical error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      results.next_steps.push('Review error logs and system state');
      return results;
    }
  }

  /**
   * Run 3-night validation cycle
   */
  private async runNightlyValidationCycle(): Promise<any[]> {
    const validationResults = [];
    
    for (let night = 1; night <= 3; night++) {
      console.log(`🌙 Night ${night}/3 validation...`);
      
      // Run nightly validation
      const nightlyResult = await this.rcManager.runNightlyValidation({
        repo_types: ['backend', 'frontend', 'monorepo'],
        language_coverage: ['typescript', 'javascript', 'python', 'go', 'rust'],
        size_categories: ['small', 'medium', 'large'],
        test_duration_minutes: 60, // Shortened for demo
        quality_gates: {
          min_recall_at_50: 0.85,
          max_p99_latency_multiple: 2.0,
          min_span_coverage: 0.98
        }
      });
      
      // Record for sign-off process
      const signoffRecord = await this.signoffManager.recordNightlyValidation(nightlyResult);
      
      validationResults.push({
        night,
        success: nightlyResult.success,
        quality_score: signoffRecord.quality_metrics.quality_score_average,
        blocking_issues: signoffRecord.blocking_issues.length,
        stakeholder_approvals: signoffRecord.stakeholder_sign_offs.filter(s => s.approved).length
      });
      
      console.log(`  Quality Score: ${(signoffRecord.quality_metrics.quality_score_average * 100).toFixed(1)}%`);
      console.log(`  Blocking Issues: ${signoffRecord.blocking_issues.length}`);
      
      if (!nightlyResult.success) {
        console.log(`  ❌ Night ${night} failed - quality gates not met`);
        break;
      } else {
        console.log(`  ✅ Night ${night} passed`);
      }
      
      // Wait between nights (shortened for demo)
      if (night < 3) {
        console.log('    Waiting for next night...');
        await this.sleep(2000); // 2 seconds instead of 24 hours
      }
    }
    
    return validationResults;
  }

  /**
   * Generate comprehensive Phase D report
   */
  generateReport(results: PhaseDResults): string {
    const reportLines = [
      '# Phase D - RC Rollout & Production Promotion Report',
      '',
      `**Generated**: ${results.timestamp}`,
      `**RC Version**: ${this.config.rc_version}`,
      `**Environment**: ${this.config.environment}`,
      `**Overall Success**: ${results.phase_d_success ? '✅ SUCCESS' : '❌ FAILED'}`,
      '',
      '## 📋 Phase D Checklist',
      '',
      `- [${results.rc_build_result?.success ? 'x' : ' '}] **D.1: RC Build** - Container + artifacts + SBOM + security scanning`,
      `- [${results.compatibility_result?.success ? 'x' : ' '}] **D.2: Compatibility Drill** - Cross-version compatibility testing`,
      `- [${results.nightly_validation_results?.every((r: any) => r.success) ? 'x' : ' '}] **D.3: Nightly Validation** - 3-night validation cycle`,
      `- [${results.signoff_status?.ready_for_promotion ? 'x' : ' '}] **D.4: Sign-off Process** - Stakeholder approvals and quality gates`,
      `- [${results.promotion_result?.success ? 'x' : ' '}] **D.5: Production Promotion** - Safe deployment to production`,
      `- [${results.monitoring_summary?.system_health?.healthy ? 'x' : ' '}] **D.6: Post-Deployment Monitoring** - Health validation`,
      '',
      '## 🎯 Key Results',
      ''
    ];

    if (results.rc_build_result) {
      reportLines.push('### RC Build Results');
      reportLines.push(`- **Build Success**: ${results.rc_build_result.success}`);
      reportLines.push(`- **Security Issues**: ${results.rc_build_result.security_scan_results.critical_issues}`);
      reportLines.push(`- **Test Coverage**: ${results.rc_build_result.quality_metrics.test_coverage}%`);
      reportLines.push('');
    }

    if (results.nightly_validation_results) {
      reportLines.push('### Nightly Validation Summary');
      results.nightly_validation_results.forEach((night: any) => {
        reportLines.push(`- **Night ${night.night}**: ${night.success ? '✅' : '❌'} Quality: ${(night.quality_score * 100).toFixed(1)}%`);
      });
      reportLines.push('');
    }

    if (results.signoff_status) {
      reportLines.push('### Sign-off Status');
      reportLines.push(`- **Consecutive Nights**: ${results.signoff_status.consecutive_nights_passed}/3`);
      reportLines.push(`- **Stakeholder Approvals**: ${results.signoff_status.stakeholder_status.approved}/${results.signoff_status.stakeholder_status.total_required}`);
      reportLines.push(`- **Risk Level**: ${results.signoff_status.risk_assessment.level.toUpperCase()}`);
      reportLines.push('');
    }

    if (results.monitoring_summary) {
      reportLines.push('### System Health');
      reportLines.push(`- **Overall Health**: ${results.monitoring_summary.system_health.healthy ? '✅ Healthy' : '❌ Issues Detected'}`);
      reportLines.push(`- **Active Violations**: ${results.monitoring_summary.system_health.violations.length}`);
      reportLines.push(`- **Monitored Slices**: ${results.monitoring_summary.system_health.total_slices}`);
      reportLines.push('');
    }

    reportLines.push('## 💡 Recommendations');
    results.recommendations.forEach(rec => {
      reportLines.push(`- ${rec}`);
    });
    reportLines.push('');

    reportLines.push('## 🚀 Next Steps');
    results.next_steps.forEach(step => {
      reportLines.push(`- ${step}`);
    });

    return reportLines.join('\n');
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    if (this.config.enable_monitoring) {
      this.tailLatencyMonitor.stop();
    }
    console.log('🧹 Phase D resources cleaned up');
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * CLI interface for Phase D operations
 */
async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  const config: PhaseDConfig = {
    rc_version: process.env.RC_VERSION || '1.0.0-rc.1',
    environment: (process.env.ENVIRONMENT as 'staging' | 'production') || 'staging',
    validation_duration_hours: parseInt(process.env.VALIDATION_DURATION_HOURS || '72'),
    enable_monitoring: process.env.ENABLE_MONITORING !== 'false',
    enable_signoff_process: process.env.ENABLE_SIGNOFF_PROCESS !== 'false',
    stakeholder_approvals_required: (process.env.REQUIRED_APPROVALS || 'platform_team,security_team,qa_team,product_owner').split(','),
    output_dir: process.env.OUTPUT_DIR || './phase-d-output'
  };

  const orchestrator = new PhaseDOrchestrator(config);

  try {
    switch (command) {
      case 'execute':
        console.log('🚀 Executing complete Phase D workflow...');
        const results = await orchestrator.executePhaseD();
        
        // Generate and save report
        const report = orchestrator.generateReport(results);
        const fs = await import('fs');
        const path = await import('path');
        
        fs.mkdirSync(config.output_dir, { recursive: true });
        fs.writeFileSync(
          path.join(config.output_dir, `phase-d-report-${new Date().toISOString().replace(/[:.]/g, '-')}.md`),
          report
        );
        
        console.log('\n📊 Phase D Report Generated');
        console.log(report);
        
        process.exit(results.phase_d_success ? 0 : 1);
        
      case 'monitor':
        console.log('📊 Starting Phase D monitoring dashboard...');
        // Keep process alive for monitoring
        process.on('SIGINT', () => {
          console.log('\n🛑 Shutting down monitoring...');
          orchestrator.cleanup();
          process.exit(0);
        });
        
        // Monitor indefinitely
        while (true) {
          const report = orchestrator.tailLatencyMonitor.generateReport();
          console.log(`📊 ${new Date().toISOString()}: System Health = ${report.system_health.healthy ? 'Healthy' : 'Issues'}`);
          await new Promise(resolve => setTimeout(resolve, 30000)); // 30 seconds
        }
        
      case 'validate':
        console.log('🔍 Running Phase D validation checks...');
        const validationResults = await orchestrator.executePhaseD();
        console.log('Validation Results:', JSON.stringify(validationResults, null, 2));
        process.exit(validationResults.phase_d_success ? 0 : 1);
        
      case 'report':
        console.log('📊 Generating Phase D status report...');
        const statusResults = await orchestrator.executePhaseD();
        const statusReport = orchestrator.generateReport(statusResults);
        console.log(statusReport);
        break;
        
      default:
        console.log('Phase D - RC Rollout & Production Promotion');
        console.log('');
        console.log('Usage: phase-d-integration.ts <command>');
        console.log('');
        console.log('Commands:');
        console.log('  execute   - Execute complete Phase D workflow');
        console.log('  monitor   - Start continuous monitoring dashboard');
        console.log('  validate  - Run validation checks only');
        console.log('  report    - Generate status report');
        console.log('');
        console.log('Environment Variables:');
        console.log('  RC_VERSION                 - RC version (default: 1.0.0-rc.1)');
        console.log('  ENVIRONMENT               - staging|production (default: staging)');
        console.log('  VALIDATION_DURATION_HOURS - Validation duration (default: 72)');
        console.log('  ENABLE_MONITORING         - Enable tail-latency monitoring (default: true)');
        console.log('  ENABLE_SIGNOFF_PROCESS    - Enable 3-night sign-off (default: true)');
        console.log('  OUTPUT_DIR               - Output directory (default: ./phase-d-output)');
        process.exit(1);
    }
  } catch (error) {
    console.error('❌ Phase D execution failed:', error);
    orchestrator.cleanup();
    process.exit(1);
  } finally {
    if (command !== 'monitor') {
      orchestrator.cleanup();
    }
  }
}

// Execute if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { PhaseDOrchestrator, type PhaseDConfig, type PhaseDResults };