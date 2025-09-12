#!/usr/bin/env node

/**
 * Quick test coverage analysis script for the Lens project
 * Analyzes both Rust and TypeScript test infrastructure
 */

import { spawn } from 'child_process';
import { readFileSync, readdirSync, statSync, existsSync } from 'fs';
import { join } from 'path';

const PROJECT_ROOT = process.cwd();

// Helper function to run commands
function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: PROJECT_ROOT,
      stdio: 'pipe',
      ...options
    });
    
    let stdout = '';
    let stderr = '';
    
    child.stdout?.on('data', (data) => {
      stdout += data.toString();
    });
    
    child.stderr?.on('data', (data) => {
      stderr += data.toString();
    });
    
    child.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });
    
    child.on('error', reject);
  });
}

// Analyze file structure
function analyzeFileStructure() {
  console.log('📊 ANALYZING PROJECT STRUCTURE\n');
  
  const rustFiles = [];
  const tsFiles = [];
  const testFiles = [];
  
  function walkDir(dir, files = []) {
    if (dir.includes('node_modules') || dir.includes('target')) return files;
    
    const items = readdirSync(dir);
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      
      if (stat.isDirectory()) {
        walkDir(fullPath, files);
      } else if (stat.isFile()) {
        const relativePath = fullPath.replace(PROJECT_ROOT + '/', '');
        
        if (fullPath.endsWith('.rs')) {
          rustFiles.push(relativePath);
        } else if (fullPath.endsWith('.ts') && !fullPath.endsWith('.d.ts')) {
          tsFiles.push(relativePath);
        }
        
        if (fullPath.includes('test') || fullPath.includes('spec')) {
          testFiles.push(relativePath);
        }
      }
    }
    return files;
  }
  
  walkDir('./src');
  walkDir('./tests');
  
  console.log(`🦀 Rust files found: ${rustFiles.length}`);
  console.log(`📝 TypeScript files found: ${tsFiles.length}`);
  console.log(`🧪 Test files found: ${testFiles.length}`);
  
  return { rustFiles, tsFiles, testFiles };
}

// Analyze test configuration
function analyzeTestConfigs() {
  console.log('\n🔧 ANALYZING TEST CONFIGURATIONS\n');
  
  // Check Cargo.toml for test configuration
  if (existsSync('./Cargo.toml')) {
    const cargoToml = readFileSync('./Cargo.toml', 'utf-8');
    console.log('✅ Cargo.toml found - Rust tests configured');
    
    // Check for dev dependencies
    if (cargoToml.includes('[dev-dependencies]')) {
      console.log('✅ Rust dev dependencies configured');
    }
  }
  
  // Check package.json for test scripts
  if (existsSync('./package.json')) {
    const packageJson = JSON.parse(readFileSync('./package.json', 'utf-8'));
    console.log('✅ package.json found - Node.js tests configured');
    
    if (packageJson.scripts && packageJson.scripts.test) {
      console.log(`✅ Test script: ${packageJson.scripts.test}`);
    }
    
    if (packageJson.devDependencies && packageJson.devDependencies.vitest) {
      console.log(`✅ Vitest configured: ${packageJson.devDependencies.vitest}`);
    }
  }
  
  // Check vitest.config.ts
  if (existsSync('./vitest.config.ts')) {
    console.log('✅ vitest.config.ts found - TypeScript test configuration present');
  }
}

// Run quick Rust compilation check
async function checkRustCompilation() {
  console.log('\n🦀 CHECKING RUST COMPILATION\n');
  
  try {
    const result = await runCommand('cargo', ['check', '--tests'], { 
      timeout: 30000 
    });
    
    if (result.code === 0) {
      console.log('✅ Rust tests compile successfully');
      return true;
    } else {
      console.log('❌ Rust compilation errors found:');
      console.log(result.stderr.split('\n').slice(0, 10).join('\n'));
      return false;
    }
  } catch (error) {
    console.log('❌ Failed to run Rust compilation check:', error.message);
    return false;
  }
}

// Check TypeScript test structure
function analyzeTypeScriptTests() {
  console.log('\n📝 ANALYZING TYPESCRIPT TESTS\n');
  
  const testDirs = ['tests', 'src/__tests__', 'src/*/__tests__'];
  let testCount = 0;
  
  function countTests(dir) {
    if (!existsSync(dir)) return 0;
    
    const items = readdirSync(dir);
    let count = 0;
    
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      
      if (stat.isFile() && (item.includes('.test.') || item.includes('.spec.'))) {
        count++;
        console.log(`  📄 ${fullPath}`);
      } else if (stat.isDirectory()) {
        count += countTests(fullPath);
      }
    }
    return count;
  }
  
  testCount += countTests('./tests');
  testCount += countTests('./src');
  
  console.log(`\n📊 Total TypeScript test files: ${testCount}`);
  return testCount;
}

// Generate coverage analysis report
async function generateCoverageReport() {
  console.log('\n📈 COVERAGE ANALYSIS RECOMMENDATIONS\n');
  
  const { rustFiles, tsFiles, testFiles } = analyzeFileStructure();
  
  console.log('🎯 CRITICAL AREAS REQUIRING TEST COVERAGE:');
  
  // Identify core modules that need coverage
  const coreModules = [
    'src/search.rs',
    'src/server.rs',
    'src/query.rs',
    'src/lsp/',
    'src/semantic/',
    'src/benchmark/',
    'src/calibration/',
    'src/pipeline/'
  ];
  
  for (const module of coreModules) {
    const exists = rustFiles.some(f => f.startsWith(module)) || 
                   tsFiles.some(f => f.startsWith(module.replace('.rs', '.ts')));
    
    if (exists) {
      console.log(`  🎯 ${module} - CRITICAL for coverage`);
    }
  }
  
  console.log('\n📋 RECOMMENDED TEST ADDITIONS:');
  console.log('  1. HTTP API integration tests');
  console.log('  2. Search engine unit tests');
  console.log('  3. LSP server integration tests');
  console.log('  4. Semantic processing pipeline tests');
  console.log('  5. Benchmarking infrastructure tests');
  console.log('  6. Dataset loading and validation tests');
  console.log('  7. Error handling and edge case tests');
  
  console.log('\n🎯 COVERAGE TARGETS:');
  console.log('  • Unit tests: >85% line coverage');
  console.log('  • Integration tests: Core workflows covered');
  console.log('  • End-to-end tests: Main user journeys');
  console.log('  • Performance tests: Latency and throughput validation');
}

// Main execution
async function main() {
  console.log('🔍 LENS PROJECT TEST COVERAGE ANALYSIS');
  console.log('=====================================\n');
  
  // Step 1: Analyze file structure
  analyzeFileStructure();
  
  // Step 2: Check configurations
  analyzeTestConfigs();
  
  // Step 3: Check Rust compilation
  const rustCompiles = await checkRustCompilation();
  
  // Step 4: Analyze TypeScript tests
  const tsTestCount = analyzeTypeScriptTests();
  
  // Step 5: Generate recommendations
  await generateCoverageReport();
  
  console.log('\n🏁 ANALYSIS COMPLETE');
  console.log(`Rust compilation: ${rustCompiles ? '✅' : '❌'}`);
  console.log(`TypeScript tests: ${tsTestCount} files found`);
  
  if (!rustCompiles) {
    console.log('\n⚠️  NEXT STEPS: Fix Rust compilation errors before running coverage');
  } else {
    console.log('\n🚀 READY FOR COVERAGE ANALYSIS');
  }
}

main().catch(console.error);