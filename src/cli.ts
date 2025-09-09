#!/usr/bin/env node
/**
 * Lens CLI - Command line interface for lens operations
 * Implements Phase A2 migration commands and other CLI operations
 */

import { Command } from 'commander';
import { handleMigrateCommand, MigrationManager } from './core/migration-manager.js';
import { SERVER_API_VERSION, SERVER_INDEX_VERSION } from './core/version-manager.js';
import { handleRCCommand } from './core/rc-release-manager.js';

const program = new Command();

program
  .name('lens')
  .description('Lens code search engine CLI')
  .version('1.0.0-rc.1');

// Migration command
program
  .command('migrate-index')
  .description('Migrate index from one version to another')
  .option('--from <version>', 'Source version (e.g., v1)', 'v1')
  .option('--to <version>', 'Target version (e.g., v1)', 'v1')
  .option('--dry-run', 'Show what would be migrated without making changes', false)
  .option('--verbose', 'Show detailed migration progress', false)
  .action(async (options) => {
    try {
      await handleMigrateCommand({
        from: options.from,
        to: options.to,
        dryRun: options.dryRun,
        verbose: options.verbose,
      });
    } catch (error) {
      console.error('Migration failed:', error);
      process.exit(1);
    }
  });

// List migrations command
program
  .command('list-migrations')
  .description('List all available migrations')
  .action(() => {
    const migrations = MigrationManager.listMigrations();
    
    if (migrations.length === 0) {
      console.log('No migrations available');
      return;
    }

    console.log('Available migrations:');
    console.log('==================');
    
    migrations.forEach(migration => {
      console.log(`\n📦 ${migration.name}`);
      console.log(`   From: ${migration.fromVersion}`);
      console.log(`   To: ${migration.toVersion}`);
      console.log(`   Description: ${migration.description}`);
    });
  });

// Version command
program
  .command('version')
  .description('Show lens version and compatibility information')
  .action(() => {
    console.log('🔍 Lens Code Search Engine');
    console.log('==========================');
    console.log(`CLI Version: 1.0.0-rc.1`);
    console.log(`API Version: ${SERVER_API_VERSION}`);
    console.log(`Index Version: ${SERVER_INDEX_VERSION}`);
  });

// Build command for Phase A3
program
  .command('build')
  .description('Build lens with security artifacts')
  .option('--sbom', 'Generate Software Bill of Materials', false)
  .option('--sast', 'Enable SAST security scanning', false)
  .option('--lock', 'Use locked dependency versions', false)
  .option('--container', 'Build container image', false)
  .action(async (options) => {
    const { execSync } = await import('child_process');
    const { join } = await import('path');
    
    console.log('🔨 Starting secure build...');
    
    // Build script arguments
    const args = [];
    if (options.sbom) args.push('--sbom');
    if (options.sast) args.push('--sast');
    if (options.lock) args.push('--lock');
    if (options.container) args.push('--container');
    
    try {
      const scriptPath = join(process.cwd(), 'scripts', 'build-secure.sh');
      const command = `"${scriptPath}" ${args.join(' ')}`;
      
      console.log(`Executing: ${command}`);
      
      execSync(command, {
        stdio: 'inherit',
        cwd: process.cwd(),
        env: {
          ...process.env,
          LENS_VERSION: '1.0.0-rc.1',
        }
      });
      
      console.log('✅ Secure build completed successfully');
    } catch (error) {
      console.error('❌ Build failed:', error);
      process.exit(1);
    }
  });

// Phase D RC release commands
program
  .command('cut-rc')
  .description('Cut RC release with all artifacts and security scanning')
  .option('--version <version>', 'RC version (e.g., v1.0.0-rc.1)', '1.0.0-rc.1')
  .option('--env <env>', 'Target environment (rc|production)', 'rc')
  .option('--output-dir <dir>', 'Output directory for artifacts', './release-output')
  .option('--no-sbom', 'Skip SBOM generation')
  .option('--no-sast', 'Skip SAST scanning')
  .option('--no-container', 'Skip container build')
  .option('--no-provenance', 'Skip build provenance')
  .action(async (options) => {
    await handleRCCommand('cut-rc', options);
  });

program
  .command('compat-drill')
  .description('Run compatibility drill against previous versions')
  .option('--version <version>', 'RC version', '1.0.0-rc.1')
  .option('--previous-versions <versions>', 'Comma-separated list of previous versions', 'v0.9.0,v0.9.1')
  .action(async (options) => {
    const previousVersions = options.previousVersions.split(',').map((v: string) => v.trim());
    await handleRCCommand('compat-drill', { ...options, previousVersions });
  });

program
  .command('nightly-validation')
  .description('Run comprehensive nightly validation across repo slices')
  .option('--duration <minutes>', 'Test duration in minutes', '120')
  .option('--repo-types <types>', 'Comma-separated repo types', 'backend,frontend,monorepo')
  .option('--languages <langs>', 'Comma-separated languages', 'typescript,javascript,python,go,rust')
  .option('--size-categories <sizes>', 'Comma-separated size categories', 'small,medium,large')
  .action(async (options) => {
    await handleRCCommand('nightly-validation', options);
  });

program
  .command('check-signoff')
  .description('Check three-night sign-off criteria for production promotion')
  .option('--version <version>', 'RC version', '1.0.0-rc.1')
  .action(async (options) => {
    await handleRCCommand('check-signoff', options);
  });

program
  .command('promote')
  .description('Promote RC to production release')
  .option('--version <version>', 'RC version to promote', '1.0.0-rc.1')
  .option('--force', 'Force promotion without sign-off checks', false)
  .action(async (options) => {
    if (!options.force) {
      console.log('⚠️  Production promotion requires sign-off verification...');
      console.log('Use --force to bypass checks (not recommended)');
    }
    await handleRCCommand('promote', options);
  });

// TODO.md CI Pipeline Commands
program
  .command('bench:freeze')
  .description('Freeze benchmark configuration and generate fingerprint')
  .option('--config <path>', 'Benchmark configuration file', '../../benchmarks/src/config.json')
  .option('--output <path>', 'Output directory for frozen artifacts', '../../benchmarks/src-frozen')
  .action(async (options) => {
    console.log('🥶 Freezing benchmark configuration...');
    
    try {
      const { LensBenchmarkOrchestrator } = await import('../../benchmarks/src/index.js');
      const { GroundTruthBuilder } = await import('../../benchmarks/src/ground-truth-builder.js');
      
      // Create orchestrator and ground truth builder
      const orchestrator = new LensBenchmarkOrchestrator({
        workingDir: process.cwd(),
        outputDir: options.output,
        repositories: [{ name: 'current', path: process.cwd() }]
      });
      
      const groundTruthBuilder = new GroundTruthBuilder(process.cwd(), options.output);
      
      // Freeze repo snapshot
      const snapshot = await groundTruthBuilder.freezeRepoSnapshot(process.cwd());
      
      // Generate config fingerprint
      const fingerprint = groundTruthBuilder.generateConfigFingerprint(
        { frozen: true, timestamp: new Date().toISOString() }, 
        [42], // seed set
        { gamma: 1.0, delta: 0.5, beta: 0.3 } // CBU coefficients
      );
      
      // Save fingerprint
      const { writeFile } = await import('fs/promises');
      await writeFile(
        `${options.output}/config_fingerprint.json`, 
        JSON.stringify(fingerprint, null, 2)
      );
      
      console.log(`✅ Configuration frozen with fingerprint: ${fingerprint.config_hash}`);
      console.log(`📄 Fingerprint saved to: ${options.output}/config_fingerprint.json`);
      
    } catch (error) {
      console.error('❌ Freeze failed:', error);
      process.exit(1);
    }
  });

program
  .command('bench:oracle')
  .description('Generate oracle validation dataset')
  .option('--config <path>', 'Frozen configuration fingerprint', '../../benchmarks/src-frozen/config_fingerprint.json')
  .option('--output <path>', 'Output directory', '../../benchmarks/src-oracle')
  .action(async (options) => {
    console.log('🔮 Generating oracle validation dataset...');
    
    try {
      const { GroundTruthBuilder } = await import('../../benchmarks/src/ground-truth-builder.js');
      const { readFile, mkdir } = await import('fs/promises');
      
      // Load frozen configuration
      const fingerprintJson = await readFile(options.config, 'utf8');
      const fingerprint = JSON.parse(fingerprintJson);
      
      console.log(`📋 Using frozen config: ${fingerprint.config_hash}`);
      
      // Create oracle dataset
      await mkdir(options.output, { recursive: true });
      const builder = new GroundTruthBuilder(process.cwd(), options.output);
      
      // Generate comprehensive ground truth with adversarial examples
      const snapshots = [await builder.freezeRepoSnapshot(process.cwd())];
      await builder.constructGoldenSet(snapshots);
      await builder.persistGoldenDataset();
      
      console.log(`✅ Oracle dataset generated: ${builder.currentGoldenItems.length} items`);
      console.log(`📄 Saved to: ${options.output}/`);
      
    } catch (error) {
      console.error('❌ Oracle generation failed:', error);
      process.exit(1);
    }
  });

program
  .command('bench:compare')
  .description('Compare benchmark results with statistical validation')
  .option('--baseline <path>', 'Baseline results directory', '../../benchmarks/src-baseline')
  .option('--treatment <path>', 'Treatment results directory', '../../benchmarks/src-treatment') 
  .option('--bootstrap', 'Enable bootstrap confidence intervals (B=1,000)', false)
  .option('--perm', 'Enable permutation tests for significance', false)
  .option('--output <path>', 'Output directory for comparison results', '../../benchmarks/src-comparison')
  .action(async (options) => {
    console.log('📊 Comparing benchmark results with statistical validation...');
    
    try {
      const { MetricsCalculator } = await import('../../benchmarks/src/metrics-calculator.js');
      const { readFile, writeFile, mkdir } = await import('fs/promises');
      
      await mkdir(options.output, { recursive: true });
      
      // Load baseline and treatment results
      console.log('📂 Loading benchmark results...');
      const baselineData = JSON.parse(await readFile(`${options.baseline}/results.json`, 'utf8'));
      const treatmentData = JSON.parse(await readFile(`${options.treatment}/results.json`, 'utf8'));
      
      const calculator = new MetricsCalculator();
      
      // Perform A/B test with statistical validation
      const abTestResults = await calculator.performABTest(
        baselineData.query_results || [],
        treatmentData.query_results || [],
        'cbu_score' // Primary metric from TODO.md
      );
      
      console.log(`📈 CBU Improvement: ${abTestResults.delta_percent.toFixed(2)}%`);
      console.log(`🎯 Statistical Significance: ${abTestResults.is_significant ? 'YES' : 'NO'} (p=${abTestResults.p_value.toFixed(4)})`);
      console.log(`📊 95% CI: [${abTestResults.ci_lower.toFixed(3)}, ${abTestResults.ci_upper.toFixed(3)}]`);
      
      // Additional metrics comparison
      const ndcgResults = await calculator.performABTest(
        baselineData.query_results || [],
        treatmentData.query_results || [],
        'ndcg_at_10'
      );
      
      const comparisonReport = {
        timestamp: new Date().toISOString(),
        primary_metric: abTestResults,
        secondary_metrics: { ndcg_at_10: ndcgResults },
        bootstrap_enabled: options.bootstrap,
        permutation_test_enabled: options.perm,
        sample_size: abTestResults.sample_size
      };
      
      // Save comparison results
      await writeFile(
        `${options.output}/comparison_results.json`, 
        JSON.stringify(comparisonReport, null, 2)
      );
      
      console.log(`✅ Comparison complete - Results saved to: ${options.output}/comparison_results.json`);
      
    } catch (error) {
      console.error('❌ Comparison failed:', error);
      process.exit(1);
    }
  });

program
  .command('bench:report')
  .description('Generate comprehensive benchmark report')
  .option('--data <path>', 'Comparison results directory', '../../benchmarks/src-comparison')
  .option('--html', 'Generate HTML report', false)
  .option('--fingerprint', 'Include configuration fingerprint validation', false)
  .option('--output <path>', 'Output directory for reports', '../../benchmarks/src-reports')
  .action(async (options) => {
    console.log('📝 Generating comprehensive benchmark report...');
    
    try {
      const { BenchmarkReportGenerator } = await import('../../benchmarks/src/report-generator.js');
      const { readFile, mkdir } = await import('fs/promises');
      
      await mkdir(options.output, { recursive: true });
      
      // Load comparison data
      const comparisonData = JSON.parse(await readFile(`${options.data}/comparison_results.json`, 'utf8'));
      
      // Load fingerprint if requested
      let fingerprint = null;
      if (options.fingerprint) {
        try {
          fingerprint = JSON.parse(await readFile('../../benchmarks/src-frozen/config_fingerprint.json', 'utf8'));
          console.log(`🔒 Validating fingerprint: ${fingerprint.config_hash}`);
        } catch (error) {
          console.warn('⚠️  Could not load configuration fingerprint');
        }
      }
      
      const reportGenerator = new BenchmarkReportGenerator(options.output);
      
      const reportData = {
        title: 'Lens Benchmark Comparison Report',
        config: { timestamp: comparisonData.timestamp },
        benchmarkRuns: [],
        abTestResults: [comparisonData.primary_metric, ...Object.values(comparisonData.secondary_metrics || {})],
        metamorphicResults: [],
        robustnessResults: [],
        configFingerprint: fingerprint,
        metadata: {
          generated_at: new Date().toISOString(),
          total_duration_ms: 0,
          systems_tested: ['baseline', 'treatment'],
          queries_executed: comparisonData.sample_size || 0
        }
      };
      
      // Generate reports
      const reports = await reportGenerator.generateReport(reportData);
      
      console.log('✅ Reports generated:');
      console.log(`📄 PDF: ${reports.pdf_path}`);
      console.log(`📝 Markdown: ${reports.markdown_path}`);
      console.log(`📊 JSON: ${reports.json_path}`);
      
      if (options.html) {
        console.log('🌐 HTML report generation not yet implemented');
      }
      
    } catch (error) {
      console.error('❌ Report generation failed:', error);
      process.exit(1);
    }
  });

// Daemon management commands
const daemonCommand = program
  .command('daemon')
  .description('Manage Lens daemon service');

daemonCommand
  .command('start')
  .description('Start Lens daemon in background')
  .option('--port <port>', 'Server port (default: 5678)', '5678')
  .option('--host <host>', 'Server host (default: 0.0.0.0)', '0.0.0.0')
  .option('--foreground', 'Run in foreground mode instead of background', false)
  .option('--config <path>', 'Configuration file path')
  .option('--no-auto-restart', 'Disable automatic restart on crashes')
  .action(async (options) => {
    try {
      const { DaemonManager } = await import('./daemon/daemon-manager.js');
      
      const daemonConfig: any = {
        port: parseInt(options.port),
        host: options.host,
        autoRestart: options.autoRestart !== false,
      };
      
      if (options.config) {
        daemonConfig.configFile = options.config;
      }
      
      const daemon = new DaemonManager(daemonConfig);
      await daemon.loadConfig();
      await daemon.start(options.foreground);
    } catch (error) {
      console.error('Failed to start daemon:', error);
      process.exit(1);
    }
  });

daemonCommand
  .command('stop')
  .description('Stop Lens daemon')
  .option('--config <path>', 'Configuration file path')
  .action(async (options) => {
    try {
      const { DaemonManager } = await import('./daemon/daemon-manager.js');
      
      const daemonConfig: any = {};
      if (options.config) {
        daemonConfig.configFile = options.config;
      }
      
      const daemon = new DaemonManager(daemonConfig);
      await daemon.loadConfig();
      await daemon.stop();
    } catch (error) {
      console.error('Failed to stop daemon:', error);
      process.exit(1);
    }
  });

daemonCommand
  .command('restart')
  .description('Restart Lens daemon')
  .option('--config <path>', 'Configuration file path')
  .action(async (options) => {
    try {
      const { DaemonManager } = await import('./daemon/daemon-manager.js');
      
      const daemonConfig: any = {};
      if (options.config) {
        daemonConfig.configFile = options.config;
      }
      
      const daemon = new DaemonManager(daemonConfig);
      await daemon.loadConfig();
      await daemon.restart();
    } catch (error) {
      console.error('Failed to restart daemon:', error);
      process.exit(1);
    }
  });

daemonCommand
  .command('status')
  .description('Show Lens daemon status')
  .option('--config <path>', 'Configuration file path')
  .option('--json', 'Output status in JSON format', false)
  .action(async (options) => {
    try {
      const { DaemonManager } = await import('./daemon/daemon-manager.js');
      const chalk = await import('chalk');
      
      const daemonConfig: any = {};
      if (options.config) {
        daemonConfig.configFile = options.config;
      }
      
      const daemon = new DaemonManager(daemonConfig);
      await daemon.loadConfig();
      const status = await daemon.getStatus();
      
      if (options.json) {
        console.log(JSON.stringify(status, null, 2));
      } else {
        console.log('\n🔍 Lens Daemon Status');
        console.log('====================');
        console.log(`Status: ${status.running ? chalk.default.green('RUNNING') : chalk.default.red('STOPPED')}`);
        
        if (status.running && status.pid) {
          console.log(`PID: ${status.pid}`);
          console.log(`Health: ${status.health === 'healthy' ? chalk.default.green('HEALTHY') : 
            status.health === 'unhealthy' ? chalk.default.red('UNHEALTHY') : chalk.default.yellow('UNKNOWN')}`);
          
          if (status.uptime) {
            const uptimeSeconds = Math.floor(status.uptime / 1000);
            const hours = Math.floor(uptimeSeconds / 3600);
            const minutes = Math.floor((uptimeSeconds % 3600) / 60);
            const seconds = uptimeSeconds % 60;
            console.log(`Uptime: ${hours}h ${minutes}m ${seconds}s`);
          }
          
          console.log(`Restart Count: ${status.restartCount}`);
          
          if (status.lastStarted) {
            console.log(`Last Started: ${status.lastStarted.toISOString()}`);
          }
        }
        
        console.log('');
      }
    } catch (error) {
      console.error('Failed to get daemon status:', error);
      process.exit(1);
    }
  });

daemonCommand
  .command('logs')
  .description('Show recent daemon logs')
  .option('--lines <n>', 'Number of lines to show (default: 100)', '100')
  .option('--follow', 'Follow log output (like tail -f)', false)
  .option('--config <path>', 'Configuration file path')
  .action(async (options) => {
    try {
      const { DaemonManager } = await import('./daemon/daemon-manager.js');
      const chalk = await import('chalk');
      
      const daemonConfig: any = {};
      if (options.config) {
        daemonConfig.configFile = options.config;
      }
      
      const daemon = new DaemonManager(daemonConfig);
      await daemon.loadConfig();
      
      const lines = parseInt(options.lines);
      const logs = await daemon.getLogs(lines);
      
      console.log(chalk.default.blue(`📄 Recent Lens Daemon Logs (last ${lines} lines):`));
      console.log(chalk.default.gray('=' .repeat(50)));
      
      logs.forEach(line => console.log(line));
      
      if (options.follow) {
        console.log(chalk.default.yellow('\n📡 Following logs (press Ctrl+C to exit)...'));
        // TODO: Implement log following functionality
        console.log(chalk.default.gray('Note: Log following not yet implemented'));
      }
      
    } catch (error) {
      console.error('Failed to show daemon logs:', error);
      process.exit(1);
    }
  });

daemonCommand
  .command('config')
  .description('Manage daemon configuration')
  .option('--show', 'Show current configuration', false)
  .option('--edit', 'Open configuration file for editing', false)
  .option('--set <key=value>', 'Set a configuration value', [])
  .option('--config <path>', 'Configuration file path')
  .action(async (options) => {
    try {
      const { DaemonManager } = await import('./daemon/daemon-manager.js');
      const chalk = await import('chalk');
      
      const daemonConfig: any = {};
      if (options.config) {
        daemonConfig.configFile = options.config;
      }
      
      const daemon = new DaemonManager(daemonConfig);
      await daemon.loadConfig();
      
      if (options.show) {
        console.log(chalk.default.blue('🔧 Lens Daemon Configuration:'));
        console.log(JSON.stringify(daemon['config'], null, 2));
      } else if (options.edit) {
        const { spawn } = await import('child_process');
        const editor = process.env.EDITOR || 'nano';
        spawn(editor, [daemon['config'].configFile], { stdio: 'inherit' });
      } else if (options.set && options.set.length > 0) {
        // TODO: Implement config setting functionality  
        console.log(chalk.default.yellow('Configuration setting not yet implemented'));
      } else {
        console.log(chalk.default.yellow('Please specify --show, --edit, or --set'));
      }
    } catch (error) {
      console.error('Failed to manage daemon config:', error);
      process.exit(1);
    }
  });

// Error handling
program.exitOverride();

try {
  program.parse();
} catch (error) {
  console.error('CLI Error:', error);
  process.exit(1);
}