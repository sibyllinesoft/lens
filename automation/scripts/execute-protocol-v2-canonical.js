#!/usr/bin/env node

/**
 * Protocol v2.0 Fast Execution - CANONICAL VERSION
 * 
 * This is the updated version that delegates all metrics calculation to @lens/metrics
 * instead of using its own implementation. This ensures single source of truth.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { DataMigrator, LensMetricsEngine, DEFAULT_CONFIG, DEFAULT_VALIDATION_GATES } from './packages/lens-metrics/dist/minimal-index.js';

const SCENARIOS = ['regex', 'substring', 'symbol', 'structural_pattern', 'nl_span', 'cross_repo', 'syntax_repair', 'temporal_change', 'multi_repo'];
const SYSTEMS = ['ripgrep', 'grep', 'find', 'opensearch', 'qdrant', 'lens'];

class CanonicalProtocolRunner {
  constructor() {
    this.runId = Math.random().toString(36).slice(2, 10);
    this.startTime = new Date().toISOString();
    this.corpusPath = '/media/nathan/Seagate Hub/Projects/lens/benchmark-corpus';
    this.sampleSize = 20;
    this.results = [];
    
    console.log('🚀 PROTOCOL V2.0 CANONICAL EXECUTION - SINGLE METRICS ENGINE');
    console.log(`📊 Run ID: ${this.runId}`);
    console.log(`🕐 Started: ${this.startTime}`);
    console.log(`⚡ Using @lens/metrics engine for all evaluation`);
  }

  async run() {
    try {
      // Quick infrastructure check
      console.log('🔍 Quick infrastructure check...');
      await this.checkInfrastructure();
      
      // Generate queries using canonical format
      console.log('📝 Generating canonical queries...');
      const queries = await this.generateCanonicalQueries();
      console.log(`📝 Generated ${queries.length} canonical queries across ${SCENARIOS.length} scenarios`);
      
      // Run evaluation using canonical metrics engine
      console.log('🏃 Running canonical evaluation...');
      await this.runCanonicalEvaluation(queries);
      
      // Generate outputs
      console.log('📊 Generating Protocol v2.0 outputs...');
      await this.generateCompleteOutputs();
      
      console.log(`✅ CANONICAL PROTOCOL V2.0 COMPLETE`);
      console.log(`📁 Results: ./protocol-v2-canonical-${this.runId}/`);
      
    } catch (error) {
      console.error(`❌ Protocol execution failed:`, error);
      process.exit(1);
    }
  }

  async checkInfrastructure() {
    // Check corpus
    const corpusFiles = await this.readFilesRecursively(this.corpusPath);
    console.log(`  ✅ Corpus: ${corpusFiles.length} files available`);
    
    // Quick OpenSearch check
    try {
      const response = await this.makeRequest('GET', 'http://localhost:9200/_cluster/health');
      if (response) {
        console.log('  ✅ OpenSearch ready');
      } else {
        console.log('  ⚠️ OpenSearch check failed');
      }
    } catch (error) {
      console.log('  ⚠️ OpenSearch unavailable');
    }
  }

  async readFilesRecursively(dirPath) {
    const files = [];
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        const subFiles = await this.readFilesRecursively(fullPath);
        files.push(...subFiles);
      } else {
        const relativePath = path.relative(this.corpusPath, fullPath);
        files.push(relativePath.replace(/\\/g, '/'));
      }
    }
    
    return files;
  }

  async generateCanonicalQueries() {
    // Read corpus files recursively
    const corpusFiles = await this.readFilesRecursively(this.corpusPath);
    const pythonFiles = corpusFiles.filter(f => f.endsWith('.py')).slice(0, this.sampleSize);
    const tsFiles = corpusFiles.filter(f => f.endsWith('.ts') || f.endsWith('.js')).slice(0, this.sampleSize);
    
    console.log(`🐛 DEBUG: Found ${pythonFiles.length} Python files, ${tsFiles.length} TS/JS files`);
    
    const queries = [];
    
    // Generate queries in canonical format using data migrator
    const legacyScenarioQueries = {
      regex: [
        { query: 'def\\s+\\w+', language: 'python', expected_files: pythonFiles.slice(0, 5), suite: 'protocol_v2_fast' },
        { query: 'class\\s+\\w+', language: 'python', expected_files: pythonFiles.slice(5, 10), suite: 'protocol_v2_fast' }
      ],
      substring: [
        { query: 'import', language: 'python', expected_files: pythonFiles.slice(0, 8), suite: 'protocol_v2_fast' },
        { query: 'function', language: 'typescript', expected_files: tsFiles.slice(0, 8), suite: 'protocol_v2_fast' }
      ],
      symbol: [
        { query: '__init__', language: 'python', expected_files: pythonFiles.slice(0, 6), suite: 'protocol_v2_fast' },
        { query: 'interface', language: 'typescript', expected_files: tsFiles.slice(0, 6), suite: 'protocol_v2_fast' }
      ],
      structural_pattern: [
        { query: 'try:', language: 'python', expected_files: pythonFiles.slice(0, 5), suite: 'protocol_v2_fast' },
        { query: 'if (', language: 'typescript', expected_files: tsFiles.slice(0, 5), suite: 'protocol_v2_fast' }
      ],
      nl_span: [
        { query: 'error handling function', language: 'python', expected_files: pythonFiles.slice(0, 4), suite: 'protocol_v2_fast' },
        { query: 'user authentication', language: 'typescript', expected_files: tsFiles.slice(0, 4), suite: 'protocol_v2_fast' }
      ],
      cross_repo: [
        { query: 'shared utilities', language: 'python', expected_files: pythonFiles.slice(0, 4), suite: 'protocol_v2_fast' },
        { query: 'common modules', language: 'typescript', expected_files: tsFiles.slice(0, 4), suite: 'protocol_v2_fast' }
      ],
      syntax_repair: [
        { query: 'deprecated', language: 'python', expected_files: pythonFiles.slice(0, 4), suite: 'protocol_v2_fast' },
        { query: 'legacy', language: 'typescript', expected_files: tsFiles.slice(0, 4), suite: 'protocol_v2_fast' }
      ],
      temporal_change: [
        { query: 'duplicate', language: 'python', expected_files: pythonFiles.slice(0, 4), suite: 'protocol_v2_fast' },
        { query: 'similar', language: 'typescript', expected_files: tsFiles.slice(0, 4), suite: 'protocol_v2_fast' }
      ],
      multi_repo: [
        { query: 'configuration', language: 'python', expected_files: pythonFiles.slice(0, 4), suite: 'protocol_v2_fast' },
        { query: 'settings', language: 'typescript', expected_files: tsFiles.slice(0, 4), suite: 'protocol_v2_fast' }
      ]
    };
    
    // Convert to canonical format using DataMigrator
    for (const scenario of SCENARIOS) {
      if (legacyScenarioQueries[scenario]) {
        for (const legacyQuery of legacyScenarioQueries[scenario]) {
          const canonicalQuery = DataMigrator.migrateQuery({
            ...legacyQuery,
            query_id: `${scenario}_${Math.random().toString(36).slice(2, 8)}`,
            expected_results: legacyQuery.expected_files?.map(file => ({ path: file }))
          }, 'swebench_sample');
          
          queries.push({
            scenario,
            canonical_query: canonicalQuery,
            query_text: legacyQuery.query
          });
        }
      }
    }
    
    return queries;
  }

  async runCanonicalEvaluation(queries) {
    const metricsEngine = new LensMetricsEngine(DEFAULT_CONFIG);
    
    for (const system of SYSTEMS) {
      console.log(`\n🎯 Evaluating system: ${system}`);
      
      // Collect results for this system
      const systemQueries = [];
      
      for (const queryData of queries) {
        console.log(`  📝 ${queryData.scenario}: "${queryData.query_text}"`);
        
        const startTime = process.hrtime.bigint();
        
        // Get search results (simulated for now)
        const searchResults = await this.getSearchResults(system, queryData.query_text, queryData.canonical_query);
        
        const endTime = process.hrtime.bigint();
        const latencyMs = Number(endTime - startTime) / 1_000_000;
        
        systemQueries.push({
          query: queryData.canonical_query,
          results: searchResults,
          latency_ms: latencyMs
        });
        
        console.log(`    ⏱️ Latency: ${latencyMs.toFixed(2)}ms, Results: ${searchResults.length}`);
      }
      
      // Evaluate using canonical metrics engine
      const evaluation = metricsEngine.evaluateSystem(
        { system_id: system, queries: systemQueries },
        undefined, // No pooled qrels for now
        DEFAULT_VALIDATION_GATES
      );
      
      console.log(`  📊 Mean nDCG@10: ${evaluation.aggregate_metrics.mean_ndcg_at_10.toFixed(3)}`);
      console.log(`  📊 SLA Compliance: ${(evaluation.aggregate_metrics.sla_compliance_rate * 100).toFixed(1)}%`);
      
      // Store results for output generation
      this.results.push({
        system_id: system,
        evaluation: evaluation
      });
      
      // Show validation status
      if (!evaluation.validation_report.gates_passed) {
        console.log(`  ⚠️ VALIDATION FAILED: ${evaluation.validation_report.errors.join(', ')}`);
      }
      
      if (evaluation.validation_report.warnings.length > 0) {
        console.log(`  ⚠️ Warnings: ${evaluation.validation_report.warnings.join(', ')}`);
      }
    }
  }

  async getSearchResults(system, query, canonicalQuery) {
    // Mock implementation that returns some realistic results
    // In real implementation, this would call the actual search systems
    
    const mockResults = [];
    const expectedFiles = canonicalQuery.expected_files || [];
    
    // Return some of the expected files as results (simulating real search)
    for (let i = 0; i < Math.min(expectedFiles.length, 10); i++) {
      const file = expectedFiles[i];
      mockResults.push({
        repo: file.repo,
        path: file.path,
        line: Math.floor(Math.random() * 100) + 1,
        col: Math.floor(Math.random() * 50),
        score: Math.random() * 0.8 + 0.2, // 0.2 to 1.0
        rank: i + 1,
        snippet: `Mock snippet for ${query} in ${file.path}`,
        why_tag: 'mock'
      });
    }
    
    // Add some noise (files that don't match)
    for (let i = 0; i < 5; i++) {
      mockResults.push({
        repo: 'swebench_sample',
        path: `noise/file_${i}.py`,
        line: Math.floor(Math.random() * 100) + 1,
        col: Math.floor(Math.random() * 50),
        score: Math.random() * 0.3, // Lower scores for noise
        rank: mockResults.length + 1,
        snippet: `Noise snippet ${i}`,
        why_tag: 'noise'
      });
    }
    
    return mockResults.sort((a, b) => b.score - a.score).map((result, index) => ({
      ...result,
      rank: index + 1
    }));
  }

  async generateCompleteOutputs() {
    const outputDir = `./protocol-v2-canonical-${this.runId}`;
    await fs.mkdir(outputDir, { recursive: true });
    
    // Generate CSV results
    const csvContent = this.generateProtocolCSV();
    await fs.writeFile(path.join(outputDir, 'protocol_v2_canonical_results.csv'), csvContent);
    
    // Generate detailed evaluation report
    const report = this.generateDetailedReport();
    await fs.writeFile(path.join(outputDir, 'canonical_evaluation_report.json'), JSON.stringify(report, null, 2));
    
    // Generate validation summary
    const validationSummary = this.generateValidationSummary();
    await fs.writeFile(path.join(outputDir, 'validation_summary.json'), JSON.stringify(validationSummary, null, 2));
    
    console.log(`📁 Generated outputs in ${outputDir}/`);
  }

  generateProtocolCSV() {
    const headers = [
      'run_id', 'suite', 'scenario', 'system', 'version', 'cfg_hash',
      'mean_ndcg_at_10', 'mean_success_at_10', 'sla_compliance_rate',
      'span_coverage_avg', 'file_credit_ratio', 'validation_passed'
    ];
    
    let csv = headers.join(',') + '\n';
    
    for (const result of this.results) {
      const metrics = result.evaluation.aggregate_metrics;
      const validation = result.evaluation.validation_report;
      
      const row = [
        this.runId,
        'protocol_v2_canonical',
        'all_scenarios',
        result.system_id,
        'canonical-v1.0.0',
        'canonical-engine',
        metrics.mean_ndcg_at_10.toFixed(6),
        metrics.mean_success_at_10.toFixed(6),
        metrics.sla_compliance_rate.toFixed(6),
        metrics.span_coverage_avg.toFixed(6),
        (validation.gate_results.file_credit_ratio || 0).toFixed(6),
        validation.gates_passed ? '1' : '0'
      ];
      
      csv += row.join(',') + '\n';
    }
    
    return csv;
  }

  generateDetailedReport() {
    return {
      run_id: this.runId,
      execution_time: this.startTime,
      metrics_engine: '@lens/metrics v1.0.0',
      configuration: DEFAULT_CONFIG,
      validation_gates: DEFAULT_VALIDATION_GATES,
      systems_evaluated: this.results.length,
      results: this.results.map(r => ({
        system_id: r.system_id,
        aggregate_metrics: r.evaluation.aggregate_metrics,
        validation_report: r.evaluation.validation_report,
        query_count: r.evaluation.query_metrics.length
      })),
      summary: {
        best_system_ndcg: Math.max(...this.results.map(r => r.evaluation.aggregate_metrics.mean_ndcg_at_10)),
        validation_pass_rate: this.results.filter(r => r.evaluation.validation_report.gates_passed).length / this.results.length,
        avg_span_coverage: this.results.reduce((sum, r) => sum + r.evaluation.aggregate_metrics.span_coverage_avg, 0) / this.results.length
      }
    };
  }

  generateValidationSummary() {
    return {
      run_id: this.runId,
      total_systems: this.results.length,
      systems_passed: this.results.filter(r => r.evaluation.validation_report.gates_passed).length,
      systems_failed: this.results.filter(r => !r.evaluation.validation_report.gates_passed).length,
      common_warnings: this.getCommonWarnings(),
      common_errors: this.getCommonErrors(),
      recommendations: this.getRecommendations()
    };
  }

  getCommonWarnings() {
    const allWarnings = this.results.flatMap(r => r.evaluation.validation_report.warnings);
    const warningCounts = {};
    allWarnings.forEach(warning => {
      const key = warning.split(':')[0]; // Get warning type
      warningCounts[key] = (warningCounts[key] || 0) + 1;
    });
    return warningCounts;
  }

  getCommonErrors() {
    const allErrors = this.results.flatMap(r => r.evaluation.validation_report.errors);
    const errorCounts = {};
    allErrors.forEach(error => {
      const key = error.split(':')[0]; // Get error type
      errorCounts[key] = (errorCounts[key] || 0) + 1;
    });
    return errorCounts;
  }

  getRecommendations() {
    const recommendations = [];
    
    // Check if file credit ratio is too high
    const avgFileRatio = this.results.reduce((sum, r) => 
      sum + (r.evaluation.validation_report.gate_results.file_credit_ratio || 0), 0
    ) / this.results.length;
    
    if (avgFileRatio > 0.7) {
      recommendations.push('Consider improving span coverage in labels to reduce file-level fallback');
    }
    
    // Check nDCG scores
    const avgNdcg = this.results.reduce((sum, r) => 
      sum + r.evaluation.aggregate_metrics.mean_ndcg_at_10, 0
    ) / this.results.length;
    
    if (avgNdcg < 0.1) {
      recommendations.push('Very low nDCG scores detected - check evaluation setup and data quality');
    }
    
    if (avgNdcg > 0.9) {
      recommendations.push('Very high nDCG scores detected - verify that evaluation is not too lenient');
    }
    
    return recommendations;
  }

  async makeRequest(method, url, body) {
    // Simple HTTP request implementation
    const urlObj = new URL(url);
    const headers = { 'Content-Type': 'application/json' };
    
    return new Promise((resolve) => {
      const options = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: method,
        headers: headers,
        timeout: 1000
      };

      const protocol = urlObj.protocol === 'https:' ? require('https') : require('http');
      const req = protocol.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => data += chunk);
        res.on('end', () => resolve(res.statusCode < 400 ? data : null));
      });

      req.on('error', () => resolve(null));
      req.on('timeout', () => {
        req.destroy();
        resolve(null);
      });
      
      if (body) req.write(body);
      req.end();
    });
  }
}

// Run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  const runner = new CanonicalProtocolRunner();
  runner.run().catch(console.error);
}

export { CanonicalProtocolRunner };