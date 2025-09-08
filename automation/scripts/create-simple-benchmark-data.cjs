#!/usr/bin/env node

/**
 * Create simplified benchmark data for testing the Benchmark Protocol v1.0
 */

const fs = require('fs').promises;
const path = require('path');

async function createBenchmarkData() {
  console.log('🔍 Creating simplified benchmark data...');

  // Create output directory
  await fs.mkdir('./benchmark-corpus', { recursive: true });

  // CoIR queries - Code Information Retrieval
  const coirQueries = [
    { id: 'coir_1', query: 'parse configuration file', suite: 'coir', intent: 'semantic', language: 'python', difficulty: 'easy' },
    { id: 'coir_2', query: 'astropy coordinates', suite: 'coir', intent: 'identifier', language: 'python', difficulty: 'easy' },
    { id: 'coir_3', query: 'class BaseFrame', suite: 'coir', intent: 'identifier', language: 'python', difficulty: 'easy' },
    { id: 'coir_4', query: 'def transform_coordinates', suite: 'coir', intent: 'structural', language: 'python', difficulty: 'medium' },
    { id: 'coir_5', query: 'import numpy', suite: 'coir', intent: 'structural', language: 'python', difficulty: 'easy' },
    { id: 'coir_6', query: 'error handling in validation', suite: 'coir', intent: 'semantic', language: 'python', difficulty: 'medium' },
    { id: 'coir_7', query: 'test_configuration', suite: 'coir', intent: 'identifier', language: 'python', difficulty: 'easy' },
    { id: 'coir_8', query: 'angle calculation', suite: 'coir', intent: 'semantic', language: 'python', difficulty: 'medium' },
    { id: 'coir_9', query: 'for * in *:', suite: 'coir', intent: 'structural', language: 'python', difficulty: 'medium' },
    { id: 'coir_10', query: 'coordinate transformation', suite: 'coir', intent: 'semantic', language: 'python', difficulty: 'hard' }
  ];

  // SWE-bench Verified queries - Software Engineering benchmarks
  const sweQueries = [
    { id: 'swe_1', query: 'fix null pointer exception', suite: 'swe_verified', intent: 'semantic', language: 'unknown', difficulty: 'hard' },
    { id: 'swe_2', query: 'resolve import error', suite: 'swe_verified', intent: 'semantic', language: 'python', difficulty: 'medium' },
    { id: 'swe_3', query: 'handle division by zero', suite: 'swe_verified', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'swe_4', query: 'test validation logic', suite: 'swe_verified', intent: 'semantic', language: 'python', difficulty: 'medium' },
    { id: 'swe_5', query: 'memory leak in loop', suite: 'swe_verified', intent: 'semantic', language: 'unknown', difficulty: 'hard' },
    { id: 'swe_6', query: 'configuration setup error', suite: 'swe_verified', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'swe_7', query: 'unit test failure', suite: 'swe_verified', intent: 'semantic', language: 'python', difficulty: 'medium' },
    { id: 'swe_8', query: 'deprecated function usage', suite: 'swe_verified', intent: 'semantic', language: 'unknown', difficulty: 'easy' }
  ];

  // CSN queries - CodeSearchNet
  const csnQueries = [
    { id: 'csn_1', query: 'read JSON file', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'easy' },
    { id: 'csn_2', query: 'validate email address', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'easy' },
    { id: 'csn_3', query: 'connect to database', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'csn_4', query: 'format date string', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'easy' },
    { id: 'csn_5', query: 'calculate hash value', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'csn_6', query: 'sort array elements', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'easy' },
    { id: 'csn_7', query: 'handle HTTP requests', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'csn_8', query: 'parse XML data', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'csn_9', query: 'log error messages', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'easy' },
    { id: 'csn_10', query: 'compress file data', suite: 'csn', intent: 'semantic', language: 'unknown', difficulty: 'medium' }
  ];

  // CoSQA queries - Code Search Question Answering
  const cosqaQueries = [
    { id: 'cosqa_1', query: 'How to implement authentication?', suite: 'cosqa', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'cosqa_2', query: 'What is the best error handling?', suite: 'cosqa', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'cosqa_3', query: 'How to parse configuration files?', suite: 'cosqa', intent: 'semantic', language: 'unknown', difficulty: 'easy' },
    { id: 'cosqa_4', query: 'What library for HTTP requests?', suite: 'cosqa', intent: 'semantic', language: 'unknown', difficulty: 'easy' },
    { id: 'cosqa_5', query: 'How to optimize database queries?', suite: 'cosqa', intent: 'semantic', language: 'unknown', difficulty: 'hard' },
    { id: 'cosqa_6', query: 'What is the purpose of this class?', suite: 'cosqa', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'cosqa_7', query: 'How does caching work here?', suite: 'cosqa', intent: 'semantic', language: 'unknown', difficulty: 'medium' },
    { id: 'cosqa_8', query: 'What are the configuration options?', suite: 'cosqa', intent: 'semantic', language: 'unknown', difficulty: 'easy' }
  ];

  // Save query suites
  await fs.writeFile('./benchmark-corpus/coir_queries.json', JSON.stringify(coirQueries, null, 2));
  await fs.writeFile('./benchmark-corpus/swe_verified_queries.json', JSON.stringify(sweQueries, null, 2));
  await fs.writeFile('./benchmark-corpus/csn_queries.json', JSON.stringify(csnQueries, null, 2));
  await fs.writeFile('./benchmark-corpus/cosqa_queries.json', JSON.stringify(cosqaQueries, null, 2));

  console.log(`✅ Created query suites:`);
  console.log(`   📋 CoIR: ${coirQueries.length} queries`);
  console.log(`   🔧 SWE-bench: ${sweQueries.length} queries`);
  console.log(`   🔍 CSN: ${csnQueries.length} queries`);
  console.log(`   💬 CoSQA: ${cosqaQueries.length} queries`);
  
  // Create summary
  const summary = {
    generated_at: new Date().toISOString(),
    total_queries: coirQueries.length + sweQueries.length + csnQueries.length + cosqaQueries.length,
    suites: [
      { name: 'coir', count: coirQueries.length },
      { name: 'swe_verified', count: sweQueries.length },
      { name: 'csn', count: csnQueries.length },
      { name: 'cosqa', count: cosqaQueries.length }
    ]
  };

  await fs.writeFile('./benchmark-corpus/query_generation_summary.json', JSON.stringify(summary, null, 2));
  console.log(`📊 Summary: ${summary.total_queries} total queries`);
  console.log('✅ Benchmark data creation complete');
}

// Execute
createBenchmarkData().catch(console.error);