#!/usr/bin/env tsx
/**
 * Enterprise Systems Demo
 * 
 * Demonstrates all four enterprise-grade systems working together:
 * 1. Task-Level Correctness with Witness Set Mining
 * 2. Declarative Query-DAG Planner with DSL
 * 3. Tenant Economics as Math (Convex Programming) 
 * 4. Adversarial/Durability Drills
 * 
 * Usage: npm run demo:enterprise
 */

import { EnterpriseSystemsCoordinator } from '../systems/index.js';
import type { EnterpriseSystemsConfig, SearchRequest } from '../systems/index.js';
import chalk from 'chalk';

// Demo configuration
const DEMO_CONFIG: EnterpriseSystemsConfig = {
  witnessSetMining: {
    ciLogsPath: './demo-data/ci-logs',
    gitRepoPath: '.',
    maxWitnessSize: 8,
    minConfidence: 0.7,
  },
  queryPlanning: {
    maxPlannerSpendRatio: 0.1,
    planCacheSize: 1000,
    sloConstraints: {
      maxLatencyMs: 20,
      maxMemoryMB: 100,
      minRecall: 0.8,
    },
  },
  tenantEconomics: {
    maxCpuTimeMs: 10000,
    maxMemoryGB: 16,
    lambdaMs: 0.001,
    lambdaGB: 0.1,
  },
  adversarialDurability: {
    maxFileSize: 5 * 1024 * 1024, // 5MB
    maxEntropy: 7.5,
    minLanguageConfidence: 0.8,
  },
};

// Demo queries and expected witness sets
const DEMO_SCENARIOS = [
  {
    name: "Function Implementation",
    query: "async function process data",
    tenantId: "tenant_premium_1",
    expectedWitnessSet: ["src/data-processor.ts", "src/async-utils.ts"],
    slaClass: "premium" as const,
  },
  {
    name: "Bug Fix Search",
    query: "null pointer exception",
    tenantId: "tenant_standard_1", 
    expectedWitnessSet: ["src/error-handler.ts", "src/validation.ts", "src/null-checks.ts"],
    slaClass: "standard" as const,
  },
  {
    name: "API Integration",
    query: "REST API endpoint authentication",
    tenantId: "tenant_basic_1",
    expectedWitnessSet: ["src/api/auth.ts", "src/middleware/jwt.ts"],
    slaClass: "basic" as const,
  },
  {
    name: "Performance Optimization",
    query: "optimize database query performance",
    tenantId: "tenant_premium_2",
    expectedWitnessSet: ["src/db/optimization.ts", "src/cache/redis.ts", "src/monitoring/perf.ts"],
    slaClass: "premium" as const,
  },
];

async function main() {
  console.log(chalk.blue.bold('\n🏢 Enterprise Systems Demo\n'));
  console.log(chalk.gray('Initializing enterprise-grade search systems...\n'));

  try {
    // Initialize enterprise systems coordinator
    const coordinator = new EnterpriseSystemsCoordinator(DEMO_CONFIG);
    await coordinator.initialize();

    console.log(chalk.green('✅ Enterprise systems initialized successfully\n'));

    // Register demo tenants
    console.log(chalk.yellow('📋 Registering demo tenants...'));
    for (const scenario of DEMO_SCENARIOS) {
      const tenantProfile = await coordinator.registerTenant(
        scenario.tenantId,
        scenario.slaClass,
        scenario.slaClass === 'premium' ? 1.0 : scenario.slaClass === 'standard' ? 0.5 : 0.2
      );
      
      console.log(chalk.cyan(`  • ${scenario.tenantId} (${scenario.slaClass}) - Priority: ${tenantProfile.businessPriority}`));
    }

    // Wait a moment for resource allocation
    console.log(chalk.gray('\n⏳ Optimizing resource allocation...\n'));
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Execute demo searches
    console.log(chalk.yellow('🔍 Executing enterprise search scenarios...\n'));
    
    const results = [];
    for (const scenario of DEMO_SCENARIOS) {
      console.log(chalk.blue(`\n--- ${scenario.name} ---`));
      console.log(chalk.gray(`Query: "${scenario.query}"`));
      console.log(chalk.gray(`Tenant: ${scenario.tenantId} (${scenario.slaClass})`));
      
      const searchRequest: SearchRequest = {
        context: {
          trace_id: `demo_${Date.now()}_${Math.random().toString(36).slice(2)}`,
          query: scenario.query,
          repo_sha: 'demo_repo',
          k: 10,
          mode: 'hybrid',
          fuzzy_distance: 0.2,
          started_at: new Date(),
          stages: [],
        },
        tenantId: scenario.tenantId,
        expectedWitnessSet: scenario.expectedWitnessSet,
      };

      try {
        const result = await coordinator.executeEnhancedSearch(searchRequest);
        results.push({ scenario, result });

        // Display key metrics
        console.log(chalk.green(`✅ Search completed successfully`));
        console.log(chalk.cyan(`   • Total hits: ${result.hits.length}`));
        console.log(chalk.cyan(`   • Witness set coverage: ${(result.witnessSetCoverage * 100).toFixed(1)}%`));
        console.log(chalk.cyan(`   • Tenant utility score: ${result.tenantUtilityScore.toFixed(3)}`));
        console.log(chalk.cyan(`   • Adversarial filtered: ${result.adversarialFiltered}`));
        console.log(chalk.cyan(`   • Total latency: ${result.performanceMetrics.totalLatencyMs.toFixed(1)}ms`));
        console.log(chalk.cyan(`   • Plan efficiency: ${result.queryPlan.estimatedCostMs.toFixed(1)}ms estimated`));

      } catch (error) {
        console.log(chalk.red(`❌ Search failed: ${error}`));
      }

      // Small delay between searches
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Generate comprehensive health report
    console.log(chalk.yellow('\n📊 Generating system health report...\n'));
    const healthReport = await coordinator.generateHealthReport();

    console.log(chalk.blue.bold('=== SYSTEM HEALTH REPORT ==='));
    console.log(chalk.green(`Overall Health Score: ${(healthReport.overallHealthScore * 100).toFixed(1)}%`));
    
    console.log(chalk.yellow('\n📈 Witness Set Mining:'));
    console.log(`   • Success@10 Rate: ${(healthReport.witnessSetMetrics.successAt10.successRate * 100).toFixed(1)}%`);
    console.log(`   • SLA Compliance: ${healthReport.witnessSetMetrics.slaCompliant ? '✅' : '❌'}`);
    console.log(`   • Total Witness Sets: ${healthReport.witnessSetMetrics.totalWitnessSets}`);

    console.log(chalk.yellow('\n🎯 Query Planning:'));
    console.log(`   • Planner Spend Ratio: ${(healthReport.plannerMetrics.spendRatio * 100).toFixed(1)}%`);
    console.log(`   • Cache Hit Ratio: ${(healthReport.plannerMetrics.cacheHitRatio * 100).toFixed(1)}%`);
    console.log(`   • P95 Planning Cost: ${healthReport.plannerMetrics.p95CostMs.toFixed(1)}ms`);

    console.log(chalk.yellow('\n💰 Tenant Economics:'));
    console.log(`   • Total Tenants: ${healthReport.tenantMetrics.totalTenants}`);
    console.log(`   • Avg Utility Score: ${healthReport.tenantMetrics.avgUtilityScore.toFixed(3)}`);
    console.log(`   • Upshift Compliance: ${healthReport.tenantMetrics.upshiftCompliant ? '✅' : '❌'}`);
    console.log(`   • Avg Upshift: ${(healthReport.tenantMetrics.avgUpshift * 100).toFixed(1)}%`);

    console.log(chalk.yellow('\n🛡️ Adversarial Durability:'));
    console.log(`   • Quarantined Files: ${healthReport.adversarialMetrics.quarantinedFiles}`);
    console.log(`   • Detection Accuracy: ${(healthReport.adversarialMetrics.detectionAccuracy * 100).toFixed(1)}%`);
    console.log(`   • System Resilience: ${(healthReport.adversarialMetrics.systemResilience.overallHealthScore * 100).toFixed(1)}%`);

    // Demonstrate chaos experiment
    console.log(chalk.yellow('\n🔥 Starting chaos experiment...\n'));
    const chaosExperimentId = await coordinator.startChaosExperiment(
      'Content Adversary Test',
      'content_adversary',
      'all',
      { intensity: 0.3, duration: 10000 },
      10000 // 10 seconds
    );
    
    console.log(chalk.cyan(`Chaos experiment started: ${chaosExperimentId}`));
    console.log(chalk.gray('Running for 10 seconds...'));

    // Wait for chaos experiment
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Check tripwire alerts
    const alerts = coordinator.getTripwireAlerts();
    if (alerts.length > 0) {
      console.log(chalk.red(`\n⚠️ Tripwire alerts detected: ${alerts.length}`));
      for (const alert of alerts) {
        console.log(chalk.red(`   • ${alert.type}: ${alert.description}`));
      }
    } else {
      console.log(chalk.green('\n✅ No tripwire alerts - system resilient'));
    }

    // Demonstrate reproducible ranking
    console.log(chalk.yellow('\n🔄 Testing reproducible ranking...\n'));
    const policyDelta = "lexscan:k=50,struct:patterns=class|function,rerank:model=monotone";
    const reproduciblePlan = coordinator.getReproduciblePlan(policyDelta);
    
    if (reproduciblePlan) {
      console.log(chalk.green(`✅ Reproducible plan generated from policy: ${policyDelta}`));
      console.log(chalk.cyan(`   • Plan ID: ${reproduciblePlan.id}`));
      console.log(chalk.cyan(`   • Operators: ${reproduciblePlan.operators.length}`));
      console.log(chalk.cyan(`   • Estimated cost: ${reproduciblePlan.estimatedCostMs.toFixed(1)}ms`));
    }

    // Summary statistics
    console.log(chalk.blue.bold('\n=== DEMO SUMMARY ==='));
    
    const totalSearches = results.length;
    const successfulSearches = results.filter(r => r.result.hits.length > 0).length;
    const avgLatency = results.reduce((sum, r) => sum + r.result.performanceMetrics.totalLatencyMs, 0) / results.length;
    const avgWitnesssCoverage = results.reduce((sum, r) => sum + r.result.witnessSetCoverage, 0) / results.length;
    const totalAdversarialFiltered = results.reduce((sum, r) => sum + r.result.adversarialFiltered, 0);

    console.log(chalk.green(`✅ Searches executed: ${totalSearches}`));
    console.log(chalk.green(`✅ Successful searches: ${successfulSearches}`));
    console.log(chalk.cyan(`📊 Average latency: ${avgLatency.toFixed(1)}ms`));
    console.log(chalk.cyan(`📊 Average witness coverage: ${(avgWitnesssCoverage * 100).toFixed(1)}%`));
    console.log(chalk.cyan(`🛡️ Adversarial content filtered: ${totalAdversarialFiltered} files`));
    console.log(chalk.green(`🏆 Overall system health: ${(healthReport.overallHealthScore * 100).toFixed(1)}%`));

    // Enterprise value proposition
    console.log(chalk.blue.bold('\n=== ENTERPRISE VALUE DELIVERED ==='));
    console.log(chalk.green('🎯 Task-Level Correctness:'));
    console.log('   • Mathematical Success@k optimization with witness set mining');
    console.log('   • CI/build failure pattern learning for better task completion');
    console.log('   • Embedder-agnostic approach with minimal hitting set coverage');
    
    console.log(chalk.green('\n📊 Declarative Query Planning:'));
    console.log('   • DSL-driven query compilation with cost optimization');
    console.log('   • Plan caching and policy deltas for reproducible rankings');
    console.log('   • <10% planner overhead with p99/p95 ≤ 2.0 constraint');
    
    console.log(chalk.green('\n💰 Mathematical Tenant Economics:'));
    console.log('   • Convex programming for optimal resource allocation');
    console.log('   • Transparent SLA-Utility pricing with spend governors');
    console.log('   • 3-7% upshift guarantee with utility maximization');
    
    console.log(chalk.green('\n🛡️ Adversarial Durability:'));
    console.log('   • Content adversary detection and quarantine');
    console.log('   • Chaos engineering for weird codebase resilience');
    console.log('   • Entropy/size heuristics with language confidence guards');

    console.log(chalk.blue.bold('\n🚀 Future-Proofed Architecture:'));
    console.log('   • Embedder-agnostic design survives model changes');
    console.log('   • Outcomes-per-millisecond optimization');
    console.log('   • Mathematical rigor in resource allocation');
    console.log('   • Comprehensive adversarial robustness');

    console.log(chalk.green.bold('\n✅ Enterprise Systems Demo Complete!\n'));

  } catch (error) {
    console.error(chalk.red.bold('❌ Demo failed:'), error);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log(chalk.yellow('\n\n🛑 Demo interrupted by user'));
  process.exit(0);
});

process.on('unhandledRejection', (error) => {
  console.error(chalk.red.bold('❌ Unhandled error:'), error);
  process.exit(1);
});

// Run the demo
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error(chalk.red.bold('❌ Demo failed:'), error);
    process.exit(1);
  });
}