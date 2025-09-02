#!/usr/bin/env tsx

/**
 * Test script for Stage-B structural search implementation
 * Tests AST-based pattern matching for Python code structures
 */

import { LensSearchEngine } from './src/api/search-engine.js';
import { SearchContext } from './src/types/core.js';

async function testStructuralSearch() {
  console.log('🔍 Testing Stage-B Structural Search');
  console.log('=====================================\n');
  
  const engine = new LensSearchEngine('./indexed-content');
  
  try {
    // Initialize the search engine
    console.log('1. Initializing search engine...');
    await engine.initialize();
    console.log('   ✅ Initialized\n');
    
    // Get available repository
    const manifest = await engine.getManifest();
    const repoSha = Object.values(manifest)[0];
    if (!repoSha) {
      throw new Error('No repositories found in manifest');
    }
    console.log(`2. Using repository: ${repoSha}\n`);
    
    // Test cases for structural search
    const testCases = [
      {
        name: 'Function definitions',
        query: 'def',
        mode: 'struct' as const,
        expected: 'Should find function definitions with pattern_type="function_def"'
      },
      {
        name: 'Async function definitions', 
        query: 'async def',
        mode: 'struct' as const,
        expected: 'Should find async function definitions with pattern_type="async_def"'
      },
      {
        name: 'Class definitions',
        query: 'class',
        mode: 'struct' as const,
        expected: 'Should find class definitions with pattern_type="class_def"'
      },
      {
        name: 'Import statements',
        query: 'import',
        mode: 'struct' as const,
        expected: 'Should find import statements with pattern_type="import"'
      },
      {
        name: 'Hybrid search',
        query: 'authenticate',
        mode: 'hybrid' as const,
        expected: 'Should combine lexical and structural search results'
      }
    ];
    
    for (const [index, testCase] of testCases.entries()) {
      console.log(`3.${index + 1} Testing: ${testCase.name}`);
      console.log(`     Query: "${testCase.query}" (mode: ${testCase.mode})`);
      console.log(`     Expected: ${testCase.expected}`);
      
      const context: SearchContext = {
        repo_sha: repoSha,
        query: testCase.query,
        mode: testCase.mode,
        fuzzy_distance: 0.5,
        k: 10,
        language: 'python',
        search_intent: 'code_search',
        user_context: {
          skill_level: 'senior',
          domain_knowledge: 'high',
        },
      };
      
      const startTime = Date.now();
      const results = await engine.search(context);
      const endTime = Date.now();
      
      console.log(`     Results: ${results.hits.length} hits in ${endTime - startTime}ms`);
      
      if (results.hits.length > 0) {
        // Show first few results with details
        const samplesCount = Math.min(3, results.hits.length);
        console.log(`     First ${samplesCount} results:`);
        
        for (let i = 0; i < samplesCount; i++) {
          const hit = results.hits[i]!;
          console.log(`       [${i + 1}] ${hit.file}:${hit.line}:${hit.col}`);
          console.log(`           Score: ${hit.score.toFixed(3)}`);
          console.log(`           Why: [${hit.why.join(', ')}]`);
          console.log(`           Snippet: ${hit.snippet?.substring(0, 60)}...`);
          
          if (hit.pattern_type) {
            console.log(`           Pattern Type: ${hit.pattern_type}`);
          }
          if (hit.symbol_name) {
            console.log(`           Symbol: ${hit.symbol_name}`);
          }
          if (hit.signature) {
            console.log(`           Signature: ${hit.signature.substring(0, 50)}...`);
          }
        }
        
        // Analyze pattern types for structural searches
        if (testCase.mode === 'struct' || testCase.mode === 'hybrid') {
          const patternTypes = results.hits
            .filter(h => h.pattern_type)
            .map(h => h.pattern_type)
            .reduce((acc, type) => {
              acc[type!] = (acc[type!] || 0) + 1;
              return acc;
            }, {} as Record<string, number>);
            
          if (Object.keys(patternTypes).length > 0) {
            console.log('     Pattern type distribution:');
            Object.entries(patternTypes).forEach(([type, count]) => {
              console.log(`       ${type}: ${count}`);
            });
          }
        }
        
        console.log('     ✅ Test passed - results found');
      } else {
        console.log('     ⚠️  No results found');
      }
      
      console.log('');
    }
    
    // Test specific structural patterns
    console.log('4. Testing specific structural patterns...');
    
    const context: SearchContext = {
      repo_sha: repoSha,
      query: 'authentication function',
      mode: 'struct',
      fuzzy_distance: 0,
      k: 20,
      language: 'python',
      search_intent: 'code_search',
      user_context: {
        skill_level: 'senior',
        domain_knowledge: 'high',
      },
    };
    
    const structResults = await engine.search(context);
    console.log(`   Found ${structResults.hits.length} structural matches`);
    
    const structuralHits = structResults.hits.filter(h => h.why.includes('structural'));
    console.log(`   ${structuralHits.length} hits used structural patterns`);
    
    if (structuralHits.length > 0) {
      console.log('   Sample structural hits:');
      structuralHits.slice(0, 2).forEach((hit, i) => {
        console.log(`     [${i + 1}] ${hit.file}:${hit.line} (${hit.pattern_type})`);
        console.log(`         "${hit.snippet}"`);
      });
    }
    
    console.log('\n🎉 All structural search tests completed!');
    console.log('\n📊 Performance Summary:');
    console.log(`   Stage A: ${results.stage_a_latency}ms (lexical)`);
    console.log(`   Stage B: ${results.stage_b_latency}ms (structural)`);
    if (results.stage_c_latency) {
      console.log(`   Stage C: ${results.stage_c_latency}ms (semantic)`);
    }
    
  } catch (error) {
    console.error('❌ Error during testing:', error);
    throw error;
  } finally {
    console.log('\n5. Shutting down engine...');
    await engine.shutdown();
    console.log('   ✅ Shutdown complete');
  }
}

// Run the test
if (import.meta.url === `file://${process.argv[1]}`) {
  testStructuralSearch().catch(error => {
    console.error('Test failed:', error);
    process.exit(1);
  });
}