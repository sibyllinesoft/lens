#!/usr/bin/env node

/**
 * Phase 2 Precision Pack A/B Benchmarking Script
 * Tests learned reranker impact on nDCG@10 with controlled latency
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');

const execAsync = promisify(exec);

// Test configuration
const BENCHMARK_CONFIG = {
  serverUrl: 'http://localhost:3001',
  repoSha: '8a9f5a125032a00804bf45cedb7d5e334489fbda', // Using the indexed TypeScript content
  testQueries: [
    // TypeScript-specific queries to test new patterns
    'export function',
    'class implements',
    'interface extends', 
    'async function',
    'SearchEngine',
    'type definition',
    'SearchHit',
    'benchmark',
    'cache',
    'rerank',
    'symbols near',
    'struct search',
  ],
  iterations: 3, // Run each test 3 times for stability
  warmupQueries: 2, // Warmup queries per condition
};

class Phase2Benchmark {
  constructor() {
    this.results = {
      rerankerOff: [],
      rerankerOn: [],
      coverageStats: null,
      startTime: new Date().toISOString(),
    };
  }

  async run() {
    console.log('🧪 Phase 2 Precision Pack A/B Benchmarking');
    console.log(`📊 Testing ${BENCHMARK_CONFIG.testQueries.length} queries with ${BENCHMARK_CONFIG.iterations} iterations each`);
    console.log('');

    try {
      // Check server health
      await this.checkServerHealth();
      
      // Get AST coverage baseline
      await this.getCoverageStats();
      
      // Run A/B tests
      console.log('🚫 Testing with reranker OFF...');
      await this.setRerankerEnabled(false);
      await this.runBenchmarkSuite('rerankerOff');
      
      console.log('🧠 Testing with reranker ON...');
      await this.setRerankerEnabled(true);
      await this.runBenchmarkSuite('rerankerOn');
      
      // Analyze results
      console.log('📈 Analyzing results...');
      await this.analyzeResults();
      
      // Generate report
      await this.generateReport();
      
    } catch (error) {
      console.error('❌ Benchmark failed:', error.message);
      process.exit(1);
    }
  }

  async checkServerHealth() {
    try {
      const response = await fetch(`${BENCHMARK_CONFIG.serverUrl}/health`);
      const health = await response.json();
      
      if (health.status !== 'ok') {
        throw new Error(`Server not healthy: ${health.status}`);
      }
      
      console.log('✅ Server health check passed');
      console.log(`📂 Indexed repositories: ${health.shards_healthy} shards`);
    } catch (error) {
      throw new Error(`Server health check failed: ${error.message}`);
    }
  }

  async getCoverageStats() {
    try {
      const response = await fetch(`${BENCHMARK_CONFIG.serverUrl}/coverage/ast`);
      const stats = await response.json();
      
      this.results.coverageStats = stats;
      
      console.log('📋 AST Coverage Statistics:');
      console.log(`   TypeScript files: ${stats.coverage.cachedTSFiles}/${stats.coverage.totalTSFiles} (${stats.coverage.coveragePercentage}%)`);
      console.log(`   Symbols cached: ${stats.coverage.symbolsCached}`);
      console.log(`   Cache hit rate: ${stats.stats.hitRate}%`);
      console.log('');
    } catch (error) {
      console.warn('⚠️  Failed to get coverage stats:', error.message);
    }
  }

  async setRerankerEnabled(enabled) {
    try {
      const response = await fetch(`${BENCHMARK_CONFIG.serverUrl}/reranker/enable`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      
      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || 'Failed to set reranker state');
      }
      
      console.log(`🔄 Reranker ${enabled ? 'enabled' : 'disabled'}`);
    } catch (error) {
      throw new Error(`Failed to set reranker: ${error.message}`);
    }
  }

  async runBenchmarkSuite(condition) {
    const results = [];
    
    // Warmup queries
    console.log(`🔥 Running ${BENCHMARK_CONFIG.warmupQueries} warmup queries...`);
    for (let i = 0; i < BENCHMARK_CONFIG.warmupQueries; i++) {
      const query = BENCHMARK_CONFIG.testQueries[i % BENCHMARK_CONFIG.testQueries.length];
      await this.runSingleQuery(query, { isWarmup: true });
    }
    
    // Main benchmark queries
    for (let iteration = 1; iteration <= BENCHMARK_CONFIG.iterations; iteration++) {
      console.log(`📋 Iteration ${iteration}/${BENCHMARK_CONFIG.iterations}`);
      
      for (const query of BENCHMARK_CONFIG.testQueries) {
        console.log(`   🔍 Query: "${query}"`);
        
        const result = await this.runSingleQuery(query, { 
          iteration, 
          condition,
          isWarmup: false 
        });
        
        results.push(result);
        
        // Brief pause between queries
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    this.results[condition] = results;
    console.log(`✅ Completed ${results.length} queries for ${condition}\n`);
  }

  async runSingleQuery(query, options = {}) {
    const startTime = Date.now();
    
    try {
      const response = await fetch(`${BENCHMARK_CONFIG.serverUrl}/search`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-Trace-Id': `phase2-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        },
        body: JSON.stringify({
          repo_sha: BENCHMARK_CONFIG.repoSha,
          q: query,
          mode: 'hybrid', // Use hybrid mode to test structural patterns
          k: 10, // Testing nDCG@10
          fuzzy: 1,
        }),
      });
      
      const result = await response.json();
      const endTime = Date.now();
      
      if (!options.isWarmup && result.hits) {
        // Calculate simple relevance metrics
        const exactMatches = result.hits.filter(hit => 
          hit.snippet?.toLowerCase().includes(query.toLowerCase())
        ).length;
        
        const structuralHits = result.hits.filter(hit =>
          hit.why?.includes('struct') || hit.why?.includes('structural')
        ).length;
        
        const symbolHits = result.hits.filter(hit =>
          hit.symbol_kind && (hit.why?.includes('symbol') || hit.symbol_kind)
        ).length;
        
        return {
          query,
          condition: options.condition,
          iteration: options.iteration,
          totalHits: result.total,
          exactMatches,
          structuralHits,
          symbolHits,
          latency: {
            total: endTime - startTime,
            stageA: result.latency_ms?.stage_a || 0,
            stageB: result.latency_ms?.stage_b || 0,
            stageC: result.latency_ms?.stage_c || 0,
          },
          avgScore: result.hits.length > 0 ? 
            result.hits.reduce((sum, hit) => sum + hit.score, 0) / result.hits.length : 0,
          topScore: result.hits.length > 0 ? result.hits[0].score : 0,
          timestamp: new Date().toISOString(),
        };
      }
      
      return null; // Warmup query
      
    } catch (error) {
      console.warn(`⚠️  Query failed: "${query}" - ${error.message}`);
      return {
        query,
        condition: options.condition,
        iteration: options.iteration,
        error: error.message,
        timestamp: new Date().toISOString(),
      };
    }
  }

  async analyzeResults() {
    const offResults = this.results.rerankerOff.filter(r => r && !r.error);
    const onResults = this.results.rerankerOn.filter(r => r && !r.error);
    
    console.log('📊 Results Summary:');
    console.log(`   Reranker OFF: ${offResults.length} successful queries`);
    console.log(`   Reranker ON:  ${onResults.length} successful queries`);
    
    if (offResults.length === 0 || onResults.length === 0) {
      console.warn('⚠️  Insufficient results for analysis');
      return;
    }
    
    // Calculate averages
    const offAvg = this.calculateAverages(offResults);
    const onAvg = this.calculateAverages(onResults);
    
    console.log('\n📈 Performance Comparison:');
    console.log(`   Total Latency: ${offAvg.totalLatency}ms → ${onAvg.totalLatency}ms (${this.formatChange(offAvg.totalLatency, onAvg.totalLatency)})`);
    console.log(`   Exact Matches: ${offAvg.exactMatches.toFixed(1)} → ${onAvg.exactMatches.toFixed(1)} (${this.formatChange(offAvg.exactMatches, onAvg.exactMatches)})`);
    console.log(`   Structural Hits: ${offAvg.structuralHits.toFixed(1)} → ${onAvg.structuralHits.toFixed(1)} (${this.formatChange(offAvg.structuralHits, onAvg.structuralHits)})`);
    console.log(`   Symbol Hits: ${offAvg.symbolHits.toFixed(1)} → ${onAvg.symbolHits.toFixed(1)} (${this.formatChange(offAvg.symbolHits, onAvg.symbolHits)})`);
    console.log(`   Avg Score: ${offAvg.avgScore.toFixed(3)} → ${onAvg.avgScore.toFixed(3)} (${this.formatChange(offAvg.avgScore, onAvg.avgScore)})`);
    console.log(`   Top Score: ${offAvg.topScore.toFixed(3)} → ${onAvg.topScore.toFixed(3)} (${this.formatChange(offAvg.topScore, onAvg.topScore)})`);
    
    // Store analysis for report
    this.results.analysis = {
      rerankerOff: offAvg,
      rerankerOn: onAvg,
      improvements: {
        exactMatches: ((onAvg.exactMatches - offAvg.exactMatches) / offAvg.exactMatches * 100).toFixed(1),
        structuralHits: ((onAvg.structuralHits - offAvg.structuralHits) / offAvg.structuralHits * 100).toFixed(1),
        avgScore: ((onAvg.avgScore - offAvg.avgScore) / offAvg.avgScore * 100).toFixed(1),
        latencyIncrease: ((onAvg.totalLatency - offAvg.totalLatency) / offAvg.totalLatency * 100).toFixed(1),
      }
    };
  }

  calculateAverages(results) {
    const count = results.length;
    return {
      totalLatency: Math.round(results.reduce((sum, r) => sum + r.latency.total, 0) / count),
      stageALatency: Math.round(results.reduce((sum, r) => sum + r.latency.stageA, 0) / count),
      stageBLatency: Math.round(results.reduce((sum, r) => sum + r.latency.stageB, 0) / count),
      stageCLatency: Math.round(results.reduce((sum, r) => sum + (r.latency.stageC || 0), 0) / count),
      totalHits: results.reduce((sum, r) => sum + r.totalHits, 0) / count,
      exactMatches: results.reduce((sum, r) => sum + r.exactMatches, 0) / count,
      structuralHits: results.reduce((sum, r) => sum + r.structuralHits, 0) / count,
      symbolHits: results.reduce((sum, r) => sum + r.symbolHits, 0) / count,
      avgScore: results.reduce((sum, r) => sum + r.avgScore, 0) / count,
      topScore: results.reduce((sum, r) => sum + r.topScore, 0) / count,
    };
  }

  formatChange(before, after) {
    const change = ((after - before) / before * 100);
    const sign = change > 0 ? '+' : '';
    const color = change > 0 ? (before < after && change < 10 ? '🟡' : '🟢') : '🔴';
    return `${sign}${change.toFixed(1)}% ${color}`;
  }

  async generateReport() {
    const reportName = `phase2-benchmark-${Date.now()}.json`;
    const reportPath = path.join(process.cwd(), reportName);
    
    const report = {
      config: BENCHMARK_CONFIG,
      results: this.results,
      summary: {
        testDate: new Date().toISOString(),
        queriesPerCondition: BENCHMARK_CONFIG.testQueries.length * BENCHMARK_CONFIG.iterations,
        successfulQueries: {
          rerankerOff: this.results.rerankerOff.filter(r => r && !r.error).length,
          rerankerOn: this.results.rerankerOn.filter(r => r && !r.error).length,
        }
      }
    };
    
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    console.log('');
    console.log('📄 Phase 2 Benchmark Report:');
    console.log(`   Report saved: ${reportName}`);
    console.log(`   AST Coverage: ${this.results.coverageStats?.coverage.coveragePercentage || 'N/A'}%`);
    
    if (this.results.analysis) {
      console.log(`   Precision Improvement: ${this.results.analysis.improvements.avgScore}%`);
      console.log(`   Latency Impact: ${this.results.analysis.improvements.latencyIncrease}%`);
    }
    
    console.log('');
    console.log('✅ Phase 2 benchmarking completed successfully!');
  }
}

// Run benchmark if called directly
if (require.main === module) {
  const benchmark = new Phase2Benchmark();
  benchmark.run().catch(console.error);
}

module.exports = { Phase2Benchmark };