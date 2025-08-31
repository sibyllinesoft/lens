#!/usr/bin/env node
/**
 * Simple structure validation script
 * Validates project structure without TypeScript compilation
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 Validating Lens project structure...\n');

// Required files
const requiredFiles = [
  'package.json',
  'tsconfig.json',
  'src/types/config.ts',
  'src/types/api.ts', 
  'src/types/core.ts',
  'src/api/server.ts',
  'src/api/search-engine.ts',
  'src/indexer/lexical.ts',
  'src/storage/segments.ts',
  'src/telemetry/tracer.ts',
  'src/core/messaging.ts',
];

// Required directories
const requiredDirs = [
  'src',
  'src/types',
  'src/api',
  'src/indexer',
  'src/storage',
  'src/telemetry',
  'src/core',
];

let errors = 0;

// Check directories
console.log('📁 Checking directories:');
requiredDirs.forEach(dir => {
  if (fs.existsSync(dir)) {
    console.log(`  ✅ ${dir}`);
  } else {
    console.log(`  ❌ ${dir} - MISSING`);
    errors++;
  }
});

console.log();

// Check files  
console.log('📄 Checking files:');
requiredFiles.forEach(file => {
  if (fs.existsSync(file)) {
    const stats = fs.statSync(file);
    console.log(`  ✅ ${file} (${stats.size} bytes)`);
  } else {
    console.log(`  ❌ ${file} - MISSING`);
    errors++;
  }
});

console.log();

// Check package.json
console.log('📦 Validating package.json:');
try {
  const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  
  // Check key dependencies
  const requiredDeps = [
    'fastify',
    'zod',
    '@opentelemetry/api',
    'nats',
    'fast-fuzzy',
    'uuid'
  ];
  
  const requiredDevDeps = [
    'typescript',
    'vitest',
    '@types/node'
  ];
  
  requiredDeps.forEach(dep => {
    if (pkg.dependencies && pkg.dependencies[dep]) {
      console.log(`  ✅ ${dep}: ${pkg.dependencies[dep]}`);
    } else {
      console.log(`  ❌ ${dep} - MISSING from dependencies`);
      errors++;
    }
  });
  
  requiredDevDeps.forEach(dep => {
    if (pkg.devDependencies && pkg.devDependencies[dep]) {
      console.log(`  ✅ ${dep}: ${pkg.devDependencies[dep]}`);
    } else {
      console.log(`  ❌ ${dep} - MISSING from devDependencies`);
      errors++;
    }
  });
  
} catch (err) {
  console.log(`  ❌ Error reading package.json: ${err.message}`);
  errors++;
}

console.log();

// Summary
if (errors === 0) {
  console.log('🎉 Project structure validation PASSED!');
  console.log('✨ Ready for TypeScript compilation');
  process.exit(0);
} else {
  console.log(`❌ Found ${errors} errors in project structure`);
  process.exit(1);
}