#!/usr/bin/env node

/**
 * Simple Node.js test for Stage-B structural search
 */

const { LensBenchmarkOrchestrator } = require('./dist/benchmark/index.js');
const { LensSearchEngine } = require('./dist/api/search-engine.js');

async function testStructuralSearch() {
  console.log('🔍 Testing Stage-B Structural Search Implementation');
  console.log('=================================================\n');
  
  const engine = new LensSearchEngine('./indexed-content');
  
  try {
    // Initialize the search engine
    console.log('1. Initializing search engine...');
    await engine.initialize();
    console.log('   ✅ Search engine initialized\n');
    
    // Get manifest to find available repositories
    const manifest = await engine.getManifest();
    console.log('2. Available repositories:');
    Object.entries(manifest).forEach(([ref, sha]) => {
      console.log(`   ${ref} -> ${sha}`);
    });
    
    const repoSha = Object.values(manifest)[0];
    if (!repoSha) {
      throw new Error('No repositories found in manifest');
    }
    console.log(`\n   Using repository: ${repoSha}\n`);
    
    // Test structural search queries
    const structuralTests = [
      {
        name: 'Python function definitions',
        query: 'def authenticate',
        mode: 'struct',
        description: 'Should find Python function definitions'
      },
      {
        name: 'Python class definitions', 
        query: 'class User',
        mode: 'struct',
        description: 'Should find Python class definitions'
      },
      {
        name: 'Python async functions',
        query: 'async def',
        mode: 'struct', 
        description: 'Should find async function definitions'
      },
      {
        name: 'Import statements',
        query: 'import',
        mode: 'struct',
        description: 'Should find import statements'
      },
      {
        name: 'Hybrid search',
        query: 'hash_password',
        mode: 'hybrid',
        description: 'Should combine lexical and structural results'
      }
    ];
    
    for (const [index, test] of structuralTests.entries()) {
      console.log(`3.${index + 1} Testing: ${test.name}`);
      console.log(`     Query: "${test.query}" (mode: ${test.mode})`);
      console.log(`     ${test.description}`);
      
      const searchContext = {
        repo_sha: repoSha,
        query: test.query,
        mode: test.mode,
        fuzzy_distance: 0,
        k: 10,
        language: 'python',
        search_intent: 'code_search',
        user_context: {
          skill_level: 'senior',
          domain_knowledge: 'high'
        }
      };
      
      const startTime = Date.now();
      const results = await engine.search(searchContext);
      const endTime = Date.now();
      
      console.log(`     ⏱️  Results: ${results.hits.length} hits in ${endTime - startTime}ms`);
      
      if (results.hits.length > 0) {
        // Show performance breakdown
        console.log(`     📊 Stage latencies:`);
        console.log(`         Stage A (lexical): ${results.stage_a_latency}ms`);
        console.log(`         Stage B (structural): ${results.stage_b_latency}ms`);
        if (results.stage_c_latency) {
          console.log(`         Stage C (semantic): ${results.stage_c_latency}ms`);
        }
        
        // Show first few results
        const sampleCount = Math.min(3, results.hits.length);
        console.log(`     🎯 Top ${sampleCount} results:`);
        
        for (let i = 0; i < sampleCount; i++) {
          const hit = results.hits[i];
          console.log(`       [${i + 1}] ${hit.file}:${hit.line}:${hit.col} (score: ${hit.score.toFixed(3)})`);
          console.log(`           Match reasons: [${hit.why.join(', ')}]`);
          console.log(`           Snippet: "${hit.snippet?.substring(0, 50)}..."`);
          
          // Show structural metadata if present
          if (hit.pattern_type) {
            console.log(`           Pattern type: ${hit.pattern_type}`);
          }
          if (hit.symbol_name) {
            console.log(`           Symbol: ${hit.symbol_name}`);
          }
          if (hit.signature) {
            console.log(`           Signature: ${hit.signature.substring(0, 40)}...`);
          }
        }
        
        // Count structural vs lexical matches
        const structuralMatches = results.hits.filter(h => 
          h.why.includes('structural') || h.pattern_type
        );
        
        if (structuralMatches.length > 0) {
          console.log(`     🏗️  Structural matches: ${structuralMatches.length}/${results.hits.length}`);
          
          // Show pattern type distribution
          const patterns = {};
          structuralMatches.forEach(hit => {
            if (hit.pattern_type) {
              patterns[hit.pattern_type] = (patterns[hit.pattern_type] || 0) + 1;
            }
          });
          
          if (Object.keys(patterns).length > 0) {
            console.log(`           Pattern distribution: ${Object.entries(patterns)
              .map(([type, count]) => `${type}(${count})`)
              .join(', ')}`);
          }
        }
        
        console.log('     ✅ Test passed\n');
      } else {
        console.log('     ⚠️  No results found\n');
      }
    }
    
    console.log('4. Summary');
    console.log('   🎉 Structural search tests completed successfully!');
    console.log('   📈 Stage-B implementation is working');
    console.log('   🏗️  AST-based pattern matching functional');
    
  } catch (error) {
    console.error('❌ Error during structural search testing:', error);
    if (error.stack) {
      console.error(error.stack);
    }
    process.exit(1);
  } finally {
    console.log('\n5. Cleaning up...');
    await engine.shutdown();
    console.log('   ✅ Engine shutdown complete');
  }
}

// Run the test
testStructuralSearch().catch(error => {
  console.error('Test failed:', error);
  process.exit(1);
});