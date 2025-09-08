#!/usr/bin/env node

/**
 * Simple test to verify structural search basics
 */

const { LensSearchEngine } = require('./dist/api/search-engine.js');

async function quickTest() {
  console.log('🔍 Quick Structural Search Test');
  console.log('==============================\n');
  
  const engine = new LensSearchEngine('./indexed-content');
  
  try {
    await engine.initialize();
    const manifest = await engine.getManifest();
    const repoSha = Object.values(manifest)[0];
    
    console.log('Testing basic structural patterns...\n');
    
    // Test simple patterns that should exist in Python files
    const tests = [
      { query: 'def', mode: 'struct' },
      { query: 'class', mode: 'struct' },
      { query: 'import', mode: 'struct' }
    ];
    
    for (const test of tests) {
      console.log(`Testing "${test.query}" (${test.mode}):`);
      
      const searchContext = {
        repo_sha: repoSha,
        query: test.query,
        mode: test.mode,
        fuzzy_distance: 0,
        k: 5,
        language: 'python',
        search_intent: 'code_search',
        user_context: { skill_level: 'senior', domain_knowledge: 'high' }
      };
      
      try {
        const results = await engine.search(searchContext);
        console.log(`  Found ${results.hits.length} results`);
        
        if (results.hits.length > 0) {
          const hit = results.hits[0];
          console.log(`  First result: ${hit.file}:${hit.line}`);
          console.log(`  Snippet: "${hit.snippet}"`);
          console.log(`  Match reasons: [${hit.why.join(', ')}]`);
          if (hit.pattern_type) console.log(`  Pattern: ${hit.pattern_type}`);
          if (hit.symbol_name) console.log(`  Symbol: ${hit.symbol_name}`);
        }
      } catch (error) {
        console.log(`  Error: ${error.message}`);
      }
      console.log('');
    }
    
  } finally {
    try {
      await engine.shutdown();
    } catch (error) {
      // Ignore shutdown errors for this test
    }
  }
}

quickTest().catch(console.error);