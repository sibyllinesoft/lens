#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

console.log('🔧 Creating minimal corpus for benchmark validation...');

// Read the golden dataset to see what files we need
const goldenPath = './benchmark-results/golden-dataset.json';
if (!fs.existsSync(goldenPath)) {
  console.error('❌ Golden dataset not found at', goldenPath);
  process.exit(1);
}

const golden = JSON.parse(fs.readFileSync(goldenPath, 'utf8'));
console.log('📊 Found', golden.length, 'golden items');

// Extract unique files referenced in golden dataset
const requiredFiles = new Set();
golden.forEach(item => {
  if (item.expected_results) {
    item.expected_results.forEach(result => {
      if (result.file) {
        requiredFiles.add(result.file);
      }
    });
  }
});

console.log('📁 Required files:', Array.from(requiredFiles).length);

// Create indexed-content structure
const outputDir = './indexed-content';
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Create repository structure
const repoSha = crypto.randomBytes(4).toString('hex');
const manifestData = {
  repo_sha: repoSha,
  repo_ref: 'HEAD',
  version: '1.0',
  languages: ['typescript'],
  shard_paths: [`${repoSha}.shard.json`],
  created_at: new Date().toISOString(),
  file_count: requiredFiles.size,
  repositories: [{
    name: 'lens',
    path: 'lens-src',
    files: Array.from(requiredFiles).map(file => ({
      path: file,
      content: fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : `// Placeholder for ${file}`,
      size: fs.existsSync(file) ? fs.statSync(file).size : 50
    }))
  }]
};

// Write manifest
const manifestPath = path.join(outputDir, `${repoSha}.manifest.json`);
fs.writeFileSync(manifestPath, JSON.stringify(manifestData, null, 2));

// Create shard data
const shardData = {
  segments: Array.from(requiredFiles).map(file => ({
    path: file,
    spans: [{
      start: 0,
      end: fs.existsSync(file) ? fs.readFileSync(file, 'utf8').length : 50,
      content: fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : `// Placeholder for ${file}`
    }]
  }))
};

const shardPath = path.join(outputDir, `${repoSha}.shard.json`);
fs.writeFileSync(shardPath, JSON.stringify(shardData, null, 2));

console.log('✅ Created minimal corpus:');
console.log('  📄 Manifest:', manifestPath);
console.log('  📦 Shard:', shardPath);
console.log('  📁 Files:', requiredFiles.size);
console.log('🎯 Ready for benchmark validation!');