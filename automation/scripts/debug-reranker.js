#!/usr/bin/env node

/**
 * Debug script for Phase 2 reranker implementation
 * Tests reranker with adjusted settings for TypeScript structural queries
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;

const execAsync = promisify(exec);

const SERVER_URL = 'http://localhost:3001';
const REPO_SHA = '8a9f5a125032a00804bf45cedb7d5e334489fbda';

// Test queries that should work with adjusted settings
const TEST_QUERIES = [
  'SearchEngine',      // Simple identifier - should work
  'export function',   // Structural pattern - needs fuzzy=2
  'class implements',  // Structural pattern - needs fuzzy=2
  'interface extends', // Structural pattern - needs fuzzy=2
];

class RerankerDebugger {
  
  async run() {
    console.log('🔬 Phase 2 Reranker Debugging');
    console.log('===============================\n');
    
    try {
      // Step 1: Lower NL threshold to 0.1 to allow TypeScript queries
      console.log('1. Adjusting reranker configuration...');
      await this.configureReranker();
      
      // Step 2: Test each query with reranker ON/OFF
      console.log('2. Testing queries with reranker ON vs OFF...\n');
      
      for (const query of TEST_QUERIES) {
        await this.testQuery(query);
      }
      
      // Step 3: Test AST cache coverage
      console.log('3. Testing AST cache coverage...');
      await this.testASTCoverage();
      
    } catch (error) {
      console.error('❌ Debug failed:', error.message);
    }
  }
  
  async configureReranker() {
    // Enable reranker with lower NL threshold
    const response = await fetch(`${SERVER_URL}/reranker/enable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        enabled: true,
        nlThreshold: 0.1,  // Lower threshold for code queries
        minCandidates: 3   // Lower min candidates
      }),
    });
    
    const result = await response.json();
    if (!result.success) {
      throw new Error(`Failed to configure reranker: ${result.error}`);
    }
    
    console.log('✅ Reranker configured: nlThreshold=0.1, minCandidates=3');
  }
  
  async testQuery(query) {
    console.log(`\n🔍 Testing: "${query}"`);
    console.log('  ' + '─'.repeat(40));
    
    // Test with reranker OFF
    await this.setReranker(false);
    const offResult = await this.searchQuery(query);
    
    // Test with reranker ON  
    await this.setReranker(true);
    const onResult = await this.searchQuery(query);
    
    console.log(`  Reranker OFF: ${offResult.hits.length} hits, ${offResult.latency_ms.total}ms total`);
    console.log(`  Reranker ON:  ${onResult.hits.length} hits, ${onResult.latency_ms.total}ms total`);
    
    if (onResult.latency_ms.stage_c) {
      console.log(`  ✅ Reranker executed: ${onResult.latency_ms.stage_c}ms stage C`);
    } else {
      console.log(`  ❌ Reranker skipped (no stage C latency)`);
    }
    
    // Compare scores if we have hits
    if (offResult.hits.length > 0 && onResult.hits.length > 0) {
      const offAvgScore = offResult.hits.reduce((sum, hit) => sum + hit.score, 0) / offResult.hits.length;
      const onAvgScore = onResult.hits.reduce((sum, hit) => sum + hit.score, 0) / onResult.hits.length;
      const improvement = ((onAvgScore - offAvgScore) / offAvgScore * 100).toFixed(1);
      console.log(`  Score improvement: ${improvement}% (${offAvgScore.toFixed(3)} → ${onAvgScore.toFixed(3)})`);
    }
  }
  
  async setReranker(enabled) {
    const response = await fetch(`${SERVER_URL}/reranker/enable`, {
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    
    const result = await response.json();
    if (!result.success) {
      throw new Error(`Failed to set reranker: ${result.error}`);
    }
  }
  
  async searchQuery(query) {
    const response = await fetch(`${SERVER_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        repo_sha: REPO_SHA,
        q: query,
        mode: 'lex',  // Use lexical mode since hybrid is broken
        k: 10,
        fuzzy: 1, // Use fuzzy 1 for better matches
      }),
    });
    
    const result = await response.json();
    if (!result.hits) {
      throw new Error(`Search failed for "${query}": ${result.error || 'Unknown error'}`);
    }
    
    return result;
  }
  
  async testASTCoverage() {
    const response = await fetch(`${SERVER_URL}/coverage/ast`);
    const stats = await response.json();
    
    console.log(`\n📊 AST Cache Coverage:`);
    console.log(`  TypeScript files: ${stats.coverage.cachedTSFiles}/${stats.coverage.totalTSFiles} (${stats.coverage.coveragePercentage}%)`);
    console.log(`  Cache hit rate: ${stats.stats.hitRate}%`);
    
    if (stats.coverage.coveragePercentage === 0) {
      console.log(`  ⚠️  No TypeScript files cached - AST cache not working`);
    } else {
      console.log(`  ✅ AST cache working`);
    }
  }
}

// Run if called directly
if (require.main === module) {
  const debug = new RerankerDebugger();
  debug.run().catch(console.error);
}

module.exports = { RerankerDebugger };