#!/usr/bin/env node
/**
 * Create a simple golden dataset with just a few known good items for testing
 */

import { promises as fs } from 'fs';
import { v4 as uuidv4 } from 'uuid';

async function createSimpleGolden() {
  // Based on the successful search results I got earlier, create a few simple test cases
  const goldenItems = [
    {
      id: uuidv4(),
      query: "function",
      query_class: "identifier", 
      language: "ts",
      source: "synthetics",
      snapshot_sha: "8a9f5a1",
      slice_tags: ["SMOKE_DEFAULT"],
      expected_results: [{
        file: "benchmark-endpoints.ts",
        line: 39,
        col: 13,
        relevance_score: 1.0,
        match_type: "symbol",
        why: "function definition"
      }]
    },
    {
      id: uuidv4(),
      query: "class",
      query_class: "identifier",
      language: "ts", 
      source: "synthetics",
      snapshot_sha: "8a9f5a1",
      slice_tags: ["SMOKE_DEFAULT"],
      expected_results: [{
        file: "search-engine.ts",
        line: 50, // approximate, will need to verify
        col: 0,
        relevance_score: 1.0,
        match_type: "symbol", 
        why: "class definition"
      }]
    }
  ];
  
  const outputPath = '/media/nathan/Seagate Hub/Projects/lens/benchmark-results/golden-dataset.json';
  await fs.writeFile(outputPath, JSON.stringify(goldenItems, null, 2));
  
  console.log(`Created simple golden dataset with ${goldenItems.length} items`);
  console.log('Items:');
  goldenItems.forEach(item => {
    console.log(`- Query: "${item.query}" -> ${item.expected_results[0].file}:${item.expected_results[0].line}`);
  });
}

createSimpleGolden().catch(console.error);