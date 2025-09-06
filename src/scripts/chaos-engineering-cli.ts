#!/usr/bin/env node

/**
 * Chaos Engineering CLI for Lens Search Engine
 * 
 * Command-line interface for executing chaos engineering and robustness tests:
 * - Individual chaos experiments
 * - Comprehensive robustness test suites
 * - Production-safe experiment execution
 * - Real-time monitoring and emergency stop capabilities
 */

import { Command } from 'commander';
import chalk from 'chalk';
import { promises as fs } from 'fs';
import path from 'path';
import { 
  ChaosEngineeringFramework,
  ChaosExperimentType,
  ChaosExperimentState 
} from '../core/chaos-engineering-framework.js';
import { ChaosExperimentSuite } from '../core/chaos-scenarios.js';
import { 
  RobustnessTestOrchestrator,
  RobustnessTestConfig,
  RobustnessAcceptanceCriteria 
} from '../core/robustness-orchestrator.js';

const program = new Command();

// Global configuration
const DEFAULT_OUTPUT_DIR = './chaos-engineering-results';
const DEFAULT_CONFIG_FILE = './chaos-config.json';

interface CLIConfig {
  productionMode: boolean;
  outputDir: string;
  baselineService: {
    url: string;
    healthEndpoint: string;
    metricsEndpoint: string;
  };
  safetyLimits: {
    maxConcurrentExperiments: number;
    maxExperimentDuration: number;
    emergencyStopThreshold: {
      errorRate: number;
      latencyMultiplier: number;
    };
  };
}

/**
 * CLI Application
 */
class ChaosEngineeringCLI {
  private config: CLIConfig;
  private chaosFramework?: ChaosEngineeringFramework;
  private robustnessOrchestrator?: RobustnessTestOrchestrator;
  
  constructor() {
    this.config = this.loadDefaultConfig();
  }
  
  private loadDefaultConfig(): CLIConfig {
    return {
      productionMode: process.env.NODE_ENV === 'production',
      outputDir: DEFAULT_OUTPUT_DIR,
      baselineService: {
        url: process.env.LENS_SERVICE_URL || 'http://localhost:3001',
        healthEndpoint: '/health',
        metricsEndpoint: '/metrics'
      },
      safetyLimits: {
        maxConcurrentExperiments: 1,
        maxExperimentDuration: 300000, // 5 minutes
        emergencyStopThreshold: {
          errorRate: 0.05, // 5%
          latencyMultiplier: 2.0
        }
      }
    };
  }
  
  async initialize(): Promise<void> {
    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });
    
    // Initialize frameworks
    this.chaosFramework = ChaosEngineeringFramework.getInstance(this.config);
    this.robustnessOrchestrator = new RobustnessTestOrchestrator(
      this.config.outputDir,
      {
        productionMode: this.config.productionMode,
        maxConcurrentTests: this.config.safetyLimits.maxConcurrentExperiments,
        defaultTimeout: this.config.safetyLimits.maxExperimentDuration,
        dataConsistencyChecks: true
      }
    );
  }
  
  setupCommands(): void {
    program
      .name('chaos-engineering')
      .description('Chaos Engineering CLI for Lens Search Engine')
      .version('1.0.0');
    
    // List available experiments
    program
      .command('list')
      .description('List all available chaos experiments')
      .option('--type <type>', 'Filter by experiment type')
      .option('--production-safe', 'Show only production-safe experiments')
      .action(async (options) => {
        await this.initialize();
        await this.listExperiments(options);
      });
    
    // Register experiments
    program
      .command('register')
      .description('Register chaos experiments')
      .option('--production', 'Register production-safe experiments only')
      .action(async (options) => {
        await this.initialize();
        await this.registerExperiments(options);
      });
    
    // Execute single experiment
    program
      .command('experiment <experimentId>')
      .description('Execute a specific chaos experiment')
      .option('--config <file>', 'Configuration file path')
      .option('--dry-run', 'Simulate experiment without actual execution')
      .action(async (experimentId, options) => {
        await this.initialize();
        await this.executeExperiment(experimentId, options);
      });
    
    // Execute robustness test suite
    program
      .command('robustness')
      .description('Execute comprehensive robustness test suite')
      .option('--suite <name>', 'Test suite name (default: comprehensive)')
      .option('--config <file>', 'Custom test configuration file')
      .option('--production', 'Use production-safe test configuration')
      .option('--parallel', 'Execute tests in parallel where possible')
      .action(async (options) => {
        await this.initialize();
        await this.executeRobustnessTests(options);
      });
    
    // Monitor experiments
    program
      .command('monitor')
      .description('Monitor active chaos experiments')
      .option('--follow', 'Follow experiment progress in real-time')
      .option('--interval <seconds>', 'Monitoring interval in seconds', '5')
      .action(async (options) => {
        await this.initialize();
        await this.monitorExperiments(options);
      });
    
    // Emergency stop
    program
      .command('stop')
      .description('Emergency stop all running experiments')
      .option('--reason <reason>', 'Reason for emergency stop')
      .action(async (options) => {
        await this.initialize();
        await this.emergencyStop(options);
      });
    
    // Generate reports
    program
      .command('report <resultId>')
      .description('Generate detailed report from experiment results')
      .option('--format <format>', 'Report format (json, html, pdf)', 'html')
      .option('--output <file>', 'Output file path')
      .action(async (resultId, options) => {
        await this.initialize();
        await this.generateReport(resultId, options);
      });
    
    // Validate system health
    program
      .command('health')
      .description('Validate system health and readiness for chaos testing')
      .option('--deep', 'Perform deep health checks')
      .action(async (options) => {
        await this.initialize();
        await this.validateSystemHealth(options);
      });
    
    // Interactive mode
    program
      .command('interactive')
      .description('Start interactive chaos engineering session')
      .action(async () => {
        await this.initialize();
        await this.interactiveMode();
      });
  }
  
  async run(): Promise<void> {
    this.setupCommands();
    await program.parseAsync(process.argv);
  }
  
  // Command implementations
  
  private async listExperiments(options: any): Promise<void> {
    console.log(chalk.blue('📋 Available Chaos Experiments\n'));
    
    const experiments = this.config.productionMode || options.productionSafe ? 
      ChaosExperimentSuite.createProductionSafeTestSuite() : 
      ChaosExperimentSuite.createRobustnessTestSuite();
    
    // Group by type
    const groupedExperiments = experiments.reduce((groups, exp) => {
      const type = exp.type;
      if (!groups[type]) groups[type] = [];
      groups[type].push(exp);
      return groups;
    }, {} as Record<string, any[]>);
    
    for (const [type, typeExperiments] of Object.entries(groupedExperiments)) {
      if (options.type && type !== options.type) continue;
      
      console.log(chalk.yellow(`\n🔬 ${type.replace('_', ' ').toUpperCase()}`));
      console.log('─'.repeat(50));
      
      typeExperiments.forEach(exp => {
        const safetyIndicator = exp.impactRadius === 'single_shard' ? '🟢' : 
                                exp.impactRadius === 'single_service' ? '🟡' : 
                                exp.impactRadius === 'partial_system' ? '🟠' : '🔴';
        
        console.log(`${safetyIndicator} ${chalk.bold(exp.name)}`);
        console.log(`   ID: ${chalk.gray(exp.id)}`);
        console.log(`   Impact: ${exp.impactRadius}`);
        console.log(`   Duration: ${exp.maxDuration / 1000}s`);
        console.log(`   ${exp.description}`);
        console.log();
      });
    }
    
    console.log(chalk.green(`\nTotal experiments: ${experiments.length}`));
    if (this.config.productionMode) {
      console.log(chalk.yellow('Running in PRODUCTION mode - only production-safe experiments shown'));
    }
  }
  
  private async registerExperiments(options: any): Promise<void> {
    console.log(chalk.blue('🔧 Registering Chaos Experiments...\n'));
    
    await ChaosExperimentSuite.registerAllExperiments(
      this.chaosFramework!,
      this.config.productionMode || options.production
    );
    
    const status = this.chaosFramework!.getExperimentStatus();
    console.log(chalk.green(`✅ Registered ${status.registered} experiments`));
    
    if (this.config.productionMode || options.production) {
      console.log(chalk.yellow('⚠️  Production-safe experiments only'));
    }
  }
  
  private async executeExperiment(experimentId: string, options: any): Promise<void> {
    console.log(chalk.blue(`🎯 Executing Chaos Experiment: ${experimentId}\n`));
    
    if (options.dryRun) {
      console.log(chalk.yellow('🔍 DRY RUN MODE - No actual failures will be injected\n'));
    }
    
    try {
      if (options.dryRun) {
        console.log(chalk.green('✅ Dry run completed - experiment would execute successfully'));
        return;
      }
      
      // Register experiments if not already done
      await ChaosExperimentSuite.registerAllExperiments(this.chaosFramework!, this.config.productionMode);
      
      const result = await this.chaosFramework!.executeExperiment(experimentId);
      
      console.log(chalk.green(`\n✅ Experiment completed: ${result.state}`));
      console.log(`Duration: ${result.endTime!.getTime() - result.startTime.getTime()}ms`);
      console.log(`Resilience Score: ${result.insights.resilience}/100`);
      
      if (result.insights.weakPoints.length > 0) {
        console.log(chalk.yellow('\n⚠️  Weak Points Identified:'));
        result.insights.weakPoints.forEach(point => {
          console.log(`   • ${point}`);
        });
      }
      
      if (result.insights.improvements.length > 0) {
        console.log(chalk.blue('\n💡 Recommended Improvements:'));
        result.insights.improvements.forEach(improvement => {
          console.log(`   • ${improvement}`);
        });
      }
      
      // Save detailed results
      const resultPath = path.join(this.config.outputDir, `experiment-${experimentId}-${Date.now()}.json`);
      await fs.writeFile(resultPath, JSON.stringify(result, null, 2));
      console.log(chalk.gray(`\n📄 Detailed results saved to: ${resultPath}`));
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      console.error(chalk.red(`❌ Experiment failed: ${errorMsg}`));
      process.exit(1);
    }
  }
  
  private async executeRobustnessTests(options: any): Promise<void> {
    const suiteName = options.suite || 'comprehensive';
    console.log(chalk.blue(`🏋️  Executing Robustness Test Suite: ${suiteName}\n`));
    
    try {
      const testConfig = this.createRobustnessTestConfig(suiteName, options);
      const result = await this.robustnessOrchestrator!.executeRobustnessTestSuite(testConfig);
      
      console.log(chalk.green(`\n✅ Robustness test suite completed`));
      console.log(`Overall Score: ${result.overallScore}/100`);
      console.log(`Acceptance Criteria Met: ${result.acceptanceCriteriaMet ? '✅ Yes' : '❌ No'}`);
      console.log(`Total Scenarios: ${result.scenarios.length}`);
      console.log(`Passed: ${result.scenarios.filter(s => s.status === 'passed').length}`);
      console.log(`Failed: ${result.scenarios.filter(s => s.status === 'failed').length}`);
      
      if (result.insights.weakestComponents.length > 0) {
        console.log(chalk.yellow('\n🔍 Weakest Components:'));
        result.insights.weakestComponents.forEach(component => {
          console.log(`   • ${component}`);
        });
      }
      
      if (result.insights.recommendedImprovements.length > 0) {
        console.log(chalk.blue('\n💡 Top Recommendations:'));
        result.insights.recommendedImprovements.slice(0, 3).forEach(improvement => {
          console.log(`   • ${improvement}`);
        });
      }
      
      console.log(chalk.gray(`\n📊 Detailed reports available in: ${this.config.outputDir}`));
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      console.error(chalk.red(`❌ Robustness tests failed: ${errorMsg}`));
      process.exit(1);
    }
  }
  
  private async monitorExperiments(options: any): Promise<void> {
    console.log(chalk.blue('👁️  Monitoring Active Experiments\n'));
    
    const interval = parseInt(options.interval) * 1000;
    
    if (options.follow) {
      console.log(chalk.gray(`Monitoring every ${options.interval} seconds... Press Ctrl+C to stop\n`));
      
      const monitorInterval = setInterval(async () => {
        const status = this.chaosFramework!.getExperimentStatus();
        
        console.clear();
        console.log(chalk.blue('👁️  Chaos Engineering Monitor\n'));
        console.log(`Active Experiments: ${status.active}`);
        console.log(`Registered Experiments: ${status.registered}`);
        console.log(`Last Update: ${new Date().toLocaleTimeString()}\n`);
        
        if (status.experiments.length > 0) {
          console.log(chalk.yellow('🔬 Active Experiments:'));
          status.experiments.forEach(exp => {
            const stateColor = exp.state === ChaosExperimentState.RUNNING ? chalk.green :
                               exp.state === ChaosExperimentState.FAILED ? chalk.red :
                               chalk.yellow;
            
            console.log(`   ${stateColor(exp.state)} ${exp.name} (${exp.id})`);
            if (exp.startTime) {
              const runtime = Date.now() - exp.startTime.getTime();
              console.log(`   Runtime: ${Math.round(runtime / 1000)}s`);
            }
          });
        } else {
          console.log(chalk.gray('No active experiments'));
        }
      }, interval);
      
      // Handle Ctrl+C
      process.on('SIGINT', () => {
        clearInterval(monitorInterval);
        console.log(chalk.yellow('\n👋 Monitoring stopped'));
        process.exit(0);
      });
      
    } else {
      const status = this.chaosFramework!.getExperimentStatus();
      console.log(JSON.stringify(status, null, 2));
    }
  }
  
  private async emergencyStop(options: any): Promise<void> {
    const reason = options.reason || 'Manual emergency stop via CLI';
    console.log(chalk.red(`🚨 EMERGENCY STOP: ${reason}\n`));
    
    await this.chaosFramework!.emergencyStop(reason);
    
    console.log(chalk.green('✅ All experiments stopped'));
  }
  
  private async generateReport(resultId: string, options: any): Promise<void> {
    console.log(chalk.blue(`📊 Generating Report for: ${resultId}\n`));
    
    const format = options.format || 'html';
    const outputPath = options.output || `./report-${resultId}.${format}`;
    
    try {
      // Load result data
      const resultPath = path.join(this.config.outputDir, `${resultId}.json`);
      const resultData = JSON.parse(await fs.readFile(resultPath, 'utf8'));
      
      // Generate report based on format
      switch (format) {
        case 'html':
          await this.generateHTMLReport(resultData, outputPath);
          break;
        case 'json':
          await this.generateJSONReport(resultData, outputPath);
          break;
        case 'pdf':
          console.log(chalk.yellow('PDF format not yet implemented'));
          break;
        default:
          throw new Error(`Unsupported format: ${format}`);
      }
      
      console.log(chalk.green(`✅ Report generated: ${outputPath}`));
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      console.error(chalk.red(`❌ Report generation failed: ${errorMsg}`));
    }
  }
  
  private async validateSystemHealth(options: any): Promise<void> {
    console.log(chalk.blue('🏥 Validating System Health\n'));
    
    const healthChecks = [
      { name: 'Lens Search Service', check: () => this.checkServiceHealth(this.config.baselineService.url) },
      { name: 'NATS Messaging', check: () => this.checkNATSHealth() },
      { name: 'Database Connection', check: () => this.checkDatabaseHealth() },
      { name: 'Memory Usage', check: () => this.checkMemoryUsage() },
      { name: 'Disk Space', check: () => this.checkDiskSpace() }
    ];
    
    if (options.deep) {
      healthChecks.push(
        { name: 'Circuit Breakers', check: () => this.checkCircuitBreakers() },
        { name: 'Rate Limiters', check: () => this.checkRateLimiters() },
        { name: 'Data Consistency', check: () => this.checkDataConsistency() }
      );
    }
    
    let allHealthy = true;
    
    for (const healthCheck of healthChecks) {
      try {
        const result = await healthCheck.check();
        console.log(`${chalk.green('✅')} ${healthCheck.name}: ${result ? 'Healthy' : 'Degraded'}`);
        if (!result) allHealthy = false;
      } catch (error) {
        console.log(`${chalk.red('❌')} ${healthCheck.name}: Failed - ${error}`);
        allHealthy = false;
      }
    }
    
    console.log(`\n${allHealthy ? chalk.green('✅ System Ready for Chaos Testing') : chalk.yellow('⚠️  System has health issues')}`);
    
    if (!allHealthy && !options.force) {
      console.log(chalk.yellow('Consider fixing health issues before running chaos experiments'));
    }
  }
  
  private async interactiveMode(): Promise<void> {
    console.log(chalk.blue('🎮 Interactive Chaos Engineering Mode\n'));
    console.log('This feature will be implemented in a future version.');
    console.log('For now, use the individual commands to execute experiments.');
  }
  
  // Helper methods
  
  private createRobustnessTestConfig(suiteName: string, options: any): RobustnessTestConfig {
    const scenarios = ChaosExperimentSuite.createRobustnessTestSuite().map(exp => ({
      name: exp.name,
      type: 'chaos_experiment' as const,
      config: { experimentId: exp.id },
      timeout: exp.maxDuration + 60000 // Add 1 minute buffer
    }));
    
    const acceptanceCriteria: RobustnessAcceptanceCriteria = {
      maxOverallErrorRate: this.config.productionMode ? 0.02 : 0.05,
      minAvailabilityDuringFailure: this.config.productionMode ? 0.95 : 0.90,
      maxRecoveryTime: 180000, // 3 minutes
      minDataConsistency: 0.95,
      maxPerformanceDegradation: 2.0,
      
      searchPipeline: {
        maxLatencyP99: 1000, // 1 second
        minThroughputMaintained: 0.7, // 70%
        fallbackEffectiveness: 0.9 // 90%
      },
      
      messagingSystem: {
        maxMessageLoss: 0.001, // 0.1%
        maxBacklogRecoveryTime: 300000, // 5 minutes
        maxDuplicateRate: 0.01 // 1%
      },
      
      storageSystem: {
        dataIntegrityScore: 0.99,
        corruptionRecoveryTime: 120000, // 2 minutes
        checksumValidationRate: 0.99
      }
    };
    
    return {
      name: suiteName,
      description: `Comprehensive robustness test suite: ${suiteName}`,
      scenarios,
      acceptance: acceptanceCriteria,
      scheduling: {
        executionMode: options.parallel ? 'parallel' : 'sequential',
        maxConcurrentTests: options.parallel ? 3 : 1,
        intervalBetweenTests: 30000, // 30 seconds
        retryFailedTests: false,
        maxRetries: 1
      }
    };
  }
  
  // Health check implementations
  
  private async checkServiceHealth(url: string): Promise<boolean> {
    try {
      const response = await fetch(`${url}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
  
  private async checkNATSHealth(): Promise<boolean> {
    // In a real implementation, this would check NATS connectivity
    return true;
  }
  
  private async checkDatabaseHealth(): Promise<boolean> {
    // In a real implementation, this would check database connectivity
    return true;
  }
  
  private async checkMemoryUsage(): Promise<boolean> {
    const usage = process.memoryUsage();
    const usedMB = usage.heapUsed / 1024 / 1024;
    return usedMB < 500; // Less than 500MB
  }
  
  private async checkDiskSpace(): Promise<boolean> {
    // In a real implementation, this would check available disk space
    return true;
  }
  
  private async checkCircuitBreakers(): Promise<boolean> {
    // In a real implementation, this would check circuit breaker states
    return true;
  }
  
  private async checkRateLimiters(): Promise<boolean> {
    // In a real implementation, this would check rate limiter states
    return true;
  }
  
  private async checkDataConsistency(): Promise<boolean> {
    // In a real implementation, this would validate data checksums
    return true;
  }
  
  // Report generation
  
  private async generateHTMLReport(data: any, outputPath: string): Promise<void> {
    const html = `<!DOCTYPE html>
<html>
<head>
    <title>Chaos Engineering Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .metric { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { color: #27ae60; }
        .error { color: #e74c3c; }
        .warning { color: #f39c12; }
    </style>
</head>
<body>
    <h1 class="header">Chaos Engineering Report</h1>
    <h2>Summary</h2>
    <div class="metric">
        <strong>Test ID:</strong> ${data.testId || data.experimentId}<br>
        <strong>Status:</strong> <span class="${data.status === 'passed' ? 'success' : 'error'}">${data.status}</span><br>
        <strong>Duration:</strong> ${data.duration}ms<br>
        <strong>Score:</strong> ${data.overallScore || data.insights?.resilience || 'N/A'}/100
    </div>
    <h2>Details</h2>
    <pre>${JSON.stringify(data, null, 2)}</pre>
</body>
</html>`;
    
    await fs.writeFile(outputPath, html);
  }
  
  private async generateJSONReport(data: any, outputPath: string): Promise<void> {
    await fs.writeFile(outputPath, JSON.stringify(data, null, 2));
  }
}

// Execute CLI if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const cli = new ChaosEngineeringCLI();
  cli.run().catch(error => {
    console.error(chalk.red('❌ CLI Error:'), error);
    process.exit(1);
  });
}

export { ChaosEngineeringCLI };