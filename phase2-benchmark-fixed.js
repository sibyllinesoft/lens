#!/usr/bin/env node

/**
 * Phase 2 Fixed Benchmark Script
 * Tests learned reranker with corrected settings and lexical mode
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');

const execAsync = promisify(exec);

// Test configuration
const BENCHMARK_CONFIG = {
  serverUrl: 'http://localhost:3001',
  repoSha: '8a9f5a125032a00804bf45cedb7d5e334489fbda',
  testQueries: [
    // Queries that should return varied results for meaningful reranking
    'search',           // Generic term with many matches
    'engine',          // Multiple engine-related matches  
    'SearchEngine',    // Specific class name
    'function',        // Common programming term
    'async',           // Async programming patterns
    'benchmark',       // Benchmark-related code
    'cache',           // Caching functionality
    'export function', // Structural pattern (lexical mode)
    'import',          // Import statements
    'interface',       // Interface definitions
    'class',           // Class definitions  
    'type',            // Type definitions
  ],
  iterations: 3,
  warmupQueries: 2,
};

class Phase2FixedBenchmark {
  constructor() {
    this.results = {
      rerankerOff: [],
      rerankerOn: [],
      coverageStats: null,
      startTime: new Date().toISOString(),
    };
  }

  async run() {
    console.log('🧪 Phase 2 Fixed Benchmark (Lexical Mode + Reranker)');
    console.log(`📊 Testing ${BENCHMARK_CONFIG.testQueries.length} queries with ${BENCHMARK_CONFIG.iterations} iterations each`);
    console.log('');

    try {
      // Check server health
      await this.checkServerHealth();
      
      // Get AST coverage baseline
      await this.getCoverageStats();
      
      // Configure reranker for code queries
      await this.configureReranker();
      
      // Run A/B tests with lexical mode
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

  async configureReranker() {
    try {
      const response = await fetch(`${BENCHMARK_CONFIG.serverUrl}/reranker/enable`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          enabled: true,
          nlThreshold: 0.1,  // Lower threshold for code queries
          minCandidates: 3   // Lower min candidates for testing
        }),
      });
      
      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || 'Failed to configure reranker');
      }
      
      console.log('🔧 Reranker configured: nlThreshold=0.1, minCandidates=3');
    } catch (error) {
      throw new Error(`Failed to configure reranker: ${error.message}`);
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
    console.log(`✅ Completed ${results.length} queries for ${condition}\\n`);
  }

  async runSingleQuery(query, options = {}) {
    const startTime = Date.now();
    
    try {
      const response = await fetch(`${BENCHMARK_CONFIG.serverUrl}/search`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-Trace-Id': `phase2-fixed-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        },
        body: JSON.stringify({
          repo_sha: BENCHMARK_CONFIG.repoSha,
          q: query,
          mode: 'lex', // Use lexical mode since hybrid is broken
          k: 10,
          fuzzy: 1, // Fixed fuzzy level
        }),
      });
      
      const result = await response.json();
      const endTime = Date.now();
      
      if (!options.isWarmup && result.hits) {
        // Calculate relevance metrics
        const exactMatches = result.hits.filter(hit => 
          hit.snippet?.toLowerCase().includes(query.toLowerCase())
        ).length;
        
        const symbolHits = result.hits.filter(hit =>
          hit.symbol_kind || hit.why?.includes('symbol')
        ).length;

        // Calculate score variance to measure reranker impact
        const scores = result.hits.map(hit => hit.score);
        const avgScore = scores.reduce((sum, s) => sum + s, 0) / scores.length;
        const scoreVariance = scores.reduce((sum, s) => sum + Math.pow(s - avgScore, 2), 0) / scores.length;
        
        return {
          query,
          condition: options.condition,
          iteration: options.iteration,
          totalHits: result.total,
          exactMatches,
          symbolHits,
          scoreVariance,
          latency: {
            total: endTime - startTime,
            stageA: result.latency_ms?.stage_a || 0,
            stageB: result.latency_ms?.stage_b || 0,
            stageC: result.latency_ms?.stage_c || 0, // Reranker stage
          },
          avgScore: avgScore,
          topScore: result.hits.length > 0 ? result.hits[0].score : 0,
          rerankerExecuted: !!(result.latency_ms?.stage_c),
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
    
    // Count how many queries had reranker executed
    const rerankerExecutedCount = onResults.filter(r => r.rerankerExecuted).length;
    
    console.log('\\n📈 Performance Comparison:');
    console.log(`   Reranker executed: ${rerankerExecutedCount}/${onResults.length} queries (${(rerankerExecutedCount/onResults.length*100).toFixed(1)}%)`);
    console.log(`   Total Latency: ${offAvg.totalLatency}ms → ${onAvg.totalLatency}ms (${this.formatChange(offAvg.totalLatency, onAvg.totalLatency)})`);
    console.log(`   Stage C Latency: ${offAvg.stageCLatency}ms → ${onAvg.stageCLatency}ms`);
    console.log(`   Exact Matches: ${offAvg.exactMatches.toFixed(1)} → ${onAvg.exactMatches.toFixed(1)} (${this.formatChange(offAvg.exactMatches, onAvg.exactMatches)})`);
    console.log(`   Symbol Hits: ${offAvg.symbolHits.toFixed(1)} → ${onAvg.symbolHits.toFixed(1)} (${this.formatChange(offAvg.symbolHits, onAvg.symbolHits)})`);
    console.log(`   Avg Score: ${offAvg.avgScore.toFixed(3)} → ${onAvg.avgScore.toFixed(3)} (${this.formatChange(offAvg.avgScore, onAvg.avgScore)})`);
    console.log(`   Top Score: ${offAvg.topScore.toFixed(3)} → ${onAvg.topScore.toFixed(3)} (${this.formatChange(offAvg.topScore, onAvg.topScore)})`);
    console.log(`   Score Variance: ${offAvg.scoreVariance.toFixed(4)} → ${onAvg.scoreVariance.toFixed(4)} (${this.formatChange(offAvg.scoreVariance, onAvg.scoreVariance)})`);
    
    // Store analysis for report
    this.results.analysis = {
      rerankerOff: offAvg,
      rerankerOn: onAvg,
      rerankerExecutedCount,
      improvements: {
        exactMatches: ((onAvg.exactMatches - offAvg.exactMatches) / offAvg.exactMatches * 100).toFixed(1),
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
      symbolHits: results.reduce((sum, r) => sum + (r.symbolHits || 0), 0) / count,
      avgScore: results.reduce((sum, r) => sum + r.avgScore, 0) / count,
      topScore: results.reduce((sum, r) => sum + r.topScore, 0) / count,
      scoreVariance: results.reduce((sum, r) => sum + (r.scoreVariance || 0), 0) / count,
    };
  }

  formatChange(before, after) {
    if (before === 0) return after > 0 ? '+∞%' : '0%';
    const change = ((after - before) / before * 100);
    const sign = change > 0 ? '+' : '';
    const color = change > 0 ? (before < after && change < 10 ? '🟡' : '🟢') : '🔴';
    return `${sign}${change.toFixed(1)}% ${color}`;
  }

  async generateReport() {
    const reportName = `phase2-fixed-benchmark-${Date.now()}.json`;
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
    console.log('📄 Phase 2 Fixed Benchmark Report:');
    console.log(`   Report saved: ${reportName}`);
    console.log(`   AST Coverage: ${this.results.coverageStats?.coverage.coveragePercentage || 'N/A'}%`);
    
    if (this.results.analysis) {
      console.log(`   Reranker Executed: ${this.results.analysis.rerankerExecutedCount} queries`);
      console.log(`   Score Improvement: ${this.results.analysis.improvements.avgScore}%`);
      console.log(`   Latency Impact: ${this.results.analysis.improvements.latencyIncrease}%`);
    }
    
    console.log('');
    console.log('✅ Phase 2 fixed benchmarking completed successfully!');
  }
}

// Run benchmark if called directly
if (require.main === module) {
  const benchmark = new Phase2FixedBenchmark();
  benchmark.run().catch(console.error);
}

module.exports = { Phase2FixedBenchmark };