#!/usr/bin/env node

/**
 * Comprehensive Test Coverage Analysis for Lens Project
 * Analyzes both Rust and TypeScript code for coverage and creates actionable recommendations
 */

import { spawn } from 'child_process';
import { readFileSync, readdirSync, statSync, existsSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = process.cwd();

// Helper to run commands with timeout
function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const timeout = options.timeout || 30000;
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
    
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`Command timed out after ${timeout}ms`));
    }, timeout);
    
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ code, stdout, stderr });
    });
    
    child.on('error', (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

// Analyze codebase structure
function analyzeCodebaseStructure() {
  console.log('🔍 ANALYZING CODEBASE STRUCTURE\n');
  
  const analysis = {
    rustFiles: [],
    tsFiles: [],
    testFiles: [],
    coreModules: [],
    criticalPaths: []
  };
  
  function walkDirectory(dir, relativePath = '') {
    if (dir.includes('node_modules') || dir.includes('target') || dir.includes('.git')) {
      return;
    }
    
    try {
      const items = readdirSync(dir);
      
      for (const item of items) {
        const fullPath = join(dir, item);
        const relativeFullPath = join(relativePath, item);
        
        try {
          const stat = statSync(fullPath);
          
          if (stat.isDirectory()) {
            walkDirectory(fullPath, relativeFullPath);
          } else if (stat.isFile()) {
            if (fullPath.endsWith('.rs') && !fullPath.includes('target/')) {
              analysis.rustFiles.push(relativeFullPath);
              
              // Identify core modules
              if (fullPath.includes('src/') && !fullPath.includes('test') && !fullPath.includes('bin/')) {
                analysis.coreModules.push(relativeFullPath);
              }
            } else if (fullPath.endsWith('.ts') && !fullPath.endsWith('.d.ts')) {
              analysis.tsFiles.push(relativeFullPath);
            }
            
            if (fullPath.includes('test') || fullPath.includes('spec')) {
              analysis.testFiles.push(relativeFullPath);
            }
          }
        } catch (error) {
          // Skip files we can't access
        }
      }
    } catch (error) {
      // Skip directories we can't access
    }
  }
  
  walkDirectory('./src');
  walkDirectory('./tests');
  
  // Identify critical paths that need coverage
  const criticalPatterns = [
    'src/search.rs', 'src/server.rs', 'src/query.rs', 'src/lsp/', 
    'src/semantic/', 'src/benchmark/', 'src/cache.rs', 'src/pipeline/',
    'src/api/', 'src/calibration/', 'src/grpc/'
  ];
  
  for (const pattern of criticalPatterns) {
    const matches = analysis.rustFiles.filter(f => f.includes(pattern.replace('.rs', '')) || f.startsWith(pattern));
    if (matches.length > 0) {
      analysis.criticalPaths.push(...matches);
    }
  }
  
  console.log(`🦀 Rust files: ${analysis.rustFiles.length}`);
  console.log(`📝 TypeScript files: ${analysis.tsFiles.length}`);
  console.log(`🧪 Test files: ${analysis.testFiles.length}`);
  console.log(`🎯 Core modules: ${analysis.coreModules.length}`);
  console.log(`⚡ Critical paths: ${analysis.criticalPaths.length}`);
  
  return analysis;
}

// Test individual TypeScript test files to see which ones work
async function testTypeScriptFiles() {
  console.log('\\n📝 TESTING TYPESCRIPT TEST FILES\\n');
  
  const testResults = {
    working: [],
    failing: [],
    totalPassed: 0,
    totalFailed: 0
  };
  
  // Find all test files
  const testFiles = [];
  function findTestFiles(dir) {
    try {
      const items = readdirSync(dir);
      for (const item of items) {
        const fullPath = join(dir, item);
        const stat = statSync(fullPath);
        
        if (stat.isDirectory()) {
          findTestFiles(fullPath);
        } else if (item.endsWith('.test.ts')) {
          testFiles.push(fullPath);
        }
      }
    } catch (error) {
      // Skip directories we can't access
    }
  }
  
  findTestFiles('./tests');
  findTestFiles('./src');
  
  console.log(`Found ${testFiles.length} TypeScript test files`);
  
  // Test a sample of files to avoid timeouts
  const sampleFiles = testFiles.slice(0, 10);
  
  for (const testFile of sampleFiles) {
    try {
      console.log(`Testing: ${testFile}`);
      const result = await runCommand('node', [
        'node_modules/vitest/vitest.mjs', 
        'run', 
        '--reporter=json',
        testFile
      ], { timeout: 15000 });
      
      if (result.code === 0) {
        try {
          const output = JSON.parse(result.stdout);
          const passed = output.testResults?.[0]?.assertionResults?.filter(t => t.status === 'passed').length || 0;
          const failed = output.testResults?.[0]?.assertionResults?.filter(t => t.status === 'failed').length || 0;
          
          testResults.working.push({
            file: testFile,
            passed,
            failed
          });
          testResults.totalPassed += passed;
          testResults.totalFailed += failed;
          
          console.log(`  ✅ ${passed} passed, ${failed} failed`);
        } catch (parseError) {
          // Count as working if exit code 0 even if we can't parse JSON
          testResults.working.push({
            file: testFile,
            passed: 'unknown',
            failed: 'unknown'
          });
          console.log(`  ✅ Completed (parsing error)`);
        }
      } else {
        testResults.failing.push({
          file: testFile,
          error: result.stderr.split('\\n').slice(0, 3).join('\\n')
        });
        console.log(`  ❌ Failed: ${result.stderr.split('\\n')[0]}`);
      }
    } catch (error) {
      testResults.failing.push({
        file: testFile,
        error: error.message
      });
      console.log(`  ❌ Error: ${error.message}`);
    }
  }
  
  return testResults;
}

// Analyze Rust module structure
function analyzeRustModules() {
  console.log('\\n🦀 ANALYZING RUST MODULES\\n');
  
  const modules = [];
  
  try {
    const libContent = readFileSync('./src/lib.rs', 'utf-8');
    const moduleLines = libContent.split('\\n').filter(line => line.startsWith('pub mod '));
    
    for (const line of moduleLines) {
      const match = line.match(/pub mod (\\w+);/);
      if (match) {
        const moduleName = match[1];
        const modulePath = `src/${moduleName}/`;
        
        let hasTests = false;
        let testCount = 0;
        
        // Check if module directory exists and has tests
        if (existsSync(modulePath)) {
          try {
            const moduleFiles = readdirSync(modulePath);
            hasTests = moduleFiles.some(f => f.includes('test') || f === 'mod.rs');
            
            // Count test functions in module files
            for (const file of moduleFiles) {
              if (file.endsWith('.rs')) {
                try {
                  const content = readFileSync(join(modulePath, file), 'utf-8');
                  const testMatches = content.match(/#\\[test\\]|#\\[tokio::test\\]/g);
                  testCount += testMatches ? testMatches.length : 0;
                } catch (error) {
                  // Skip files we can't read
                }
              }
            }
          } catch (error) {
            // Skip modules we can't analyze
          }
        } else {
          // Check if it's a single file module
          const singleFilePath = `src/${moduleName}.rs`;
          if (existsSync(singleFilePath)) {
            try {
              const content = readFileSync(singleFilePath, 'utf-8');
              hasTests = content.includes('#[test]') || content.includes('#[tokio::test]');
              const testMatches = content.match(/#\\[test\\]|#\\[tokio::test\\]/g);
              testCount = testMatches ? testMatches.length : 0;
            } catch (error) {
              // Skip files we can't read
            }
          }
        }
        
        modules.push({
          name: moduleName,
          path: modulePath,
          hasTests,
          testCount,
          priority: ['search', 'server', 'query', 'lsp', 'semantic', 'benchmark'].includes(moduleName) ? 'HIGH' : 'NORMAL'
        });
      }
    }
  } catch (error) {
    console.log('Could not analyze lib.rs:', error.message);
  }
  
  // Sort by priority and test coverage
  modules.sort((a, b) => {
    if (a.priority === 'HIGH' && b.priority !== 'HIGH') return -1;
    if (b.priority === 'HIGH' && a.priority !== 'HIGH') return 1;
    return b.testCount - a.testCount;
  });
  
  console.log('Rust modules analysis:');
  for (const module of modules) {
    const status = module.hasTests ? '✅' : '❌';
    console.log(`  ${status} ${module.name} (${module.priority}) - ${module.testCount} tests`);
  }
  
  return modules;
}

// Generate coverage recommendations
function generateRecommendations(codebaseAnalysis, testResults, rustModules) {
  console.log('\\n📋 COVERAGE RECOMMENDATIONS\\n');
  
  const recommendations = {
    immediate: [],
    shortTerm: [],
    longTerm: []
  };
  
  // Immediate actions
  if (testResults.failing.length > 0) {
    recommendations.immediate.push({
      action: 'Fix failing TypeScript tests',
      priority: 'CRITICAL',
      impact: 'High',
      details: `${testResults.failing.length} test files are failing and need fixing`,
      files: testResults.failing.slice(0, 5).map(f => f.file)
    });
  }
  
  // Find Rust modules without tests
  const untestedRustModules = rustModules.filter(m => !m.hasTests && m.priority === 'HIGH');
  if (untestedRustModules.length > 0) {
    recommendations.immediate.push({
      action: 'Add tests for critical Rust modules',
      priority: 'HIGH',
      impact: 'High',
      details: `${untestedRustModules.length} critical modules lack tests`,
      files: untestedRustModules.map(m => m.name)
    });
  }
  
  // Short term actions
  recommendations.shortTerm.push({
    action: 'Create integration tests for HTTP API',
    priority: 'HIGH',
    impact: 'Medium',
    details: 'End-to-end API testing for search, LSP, and benchmarking endpoints'
  });
  
  recommendations.shortTerm.push({
    action: 'Add performance regression tests',
    priority: 'MEDIUM',
    impact: 'Medium',
    details: 'Automated tests to catch performance regressions in search and semantic processing'
  });
  
  // Long term actions
  recommendations.longTerm.push({
    action: 'Implement property-based testing',
    priority: 'MEDIUM',
    impact: 'Low',
    details: 'Use property-based testing for complex algorithms'
  });
  
  return recommendations;
}

// Generate coverage report
async function generateCoverageReport() {
  console.log('\\n📊 GENERATING COVERAGE REPORT\\n');
  
  const codebaseAnalysis = analyzeCodebaseStructure();
  const testResults = await testTypeScriptFiles();
  const rustModules = analyzeRustModules();
  const recommendations = generateRecommendations(codebaseAnalysis, testResults, rustModules);
  
  // Calculate coverage estimates
  const workingTests = testResults.working.length;
  const totalTests = testResults.working.length + testResults.failing.length;
  const testSuccessRate = totalTests > 0 ? Math.round((workingTests / totalTests) * 100) : 0;
  
  const rustModulesWithTests = rustModules.filter(m => m.hasTests).length;
  const totalRustModules = rustModules.length;
  const rustTestCoverage = totalRustModules > 0 ? Math.round((rustModulesWithTests / totalRustModules) * 100) : 0;
  
  const report = {
    summary: {
      totalRustFiles: codebaseAnalysis.rustFiles.length,
      totalTypeScriptFiles: codebaseAnalysis.tsFiles.length,
      totalTestFiles: codebaseAnalysis.testFiles.length,
      workingTests,
      failingTests: testResults.failing.length,
      testSuccessRate: `${testSuccessRate}%`,
      rustTestCoverage: `${rustTestCoverage}%`,
      criticalPathsCovered: codebaseAnalysis.criticalPaths.filter(path => {
        return codebaseAnalysis.testFiles.some(test => test.includes(path.replace('.rs', '').replace('src/', '')));
      }).length
    },
    workingTests: testResults.working,
    failingTests: testResults.failing.slice(0, 10), // Limit for readability
    rustModules: rustModules,
    recommendations
  };
  
  // Write detailed report to file
  writeFileSync('test_coverage_analysis_report.json', JSON.stringify(report, null, 2));
  
  // Display summary
  console.log('📊 COVERAGE SUMMARY');
  console.log('===================');
  console.log(`🦀 Rust files: ${report.summary.totalRustFiles}`);
  console.log(`📝 TypeScript files: ${report.summary.totalTypeScriptFiles}`);
  console.log(`🧪 Test files: ${report.summary.totalTestFiles}`);
  console.log(`✅ Working tests: ${report.summary.workingTests}`);
  console.log(`❌ Failing tests: ${report.summary.failingTests}`);
  console.log(`📈 Test success rate: ${report.summary.testSuccessRate}`);
  console.log(`🎯 Rust test coverage: ${report.summary.rustTestCoverage}`);
  
  console.log('\\n🎯 TOP RECOMMENDATIONS:');
  for (const rec of recommendations.immediate) {
    console.log(`  ${rec.priority}: ${rec.action}`);
    console.log(`    Impact: ${rec.impact} - ${rec.details}`);
  }
  
  console.log('\\n📁 Report saved to: test_coverage_analysis_report.json');
  
  return report;
}

// Main execution
async function main() {
  try {
    console.log('🔍 COMPREHENSIVE TEST COVERAGE ANALYSIS');
    console.log('========================================\\n');
    
    const report = await generateCoverageReport();
    
    console.log('\\n🏁 ANALYSIS COMPLETE');
    console.log(`\\nNEXT STEPS:`);
    console.log(`1. Fix ${report.summary.failingTests} failing tests`);
    console.log(`2. Add tests for ${report.rustModules.filter(m => !m.hasTests && m.priority === 'HIGH').length} critical Rust modules`);
    console.log(`3. Target >85% coverage through systematic test addition`);
    
    if (report.summary.testSuccessRate.replace('%', '') < 70) {
      console.log('\\n⚠️  WARNING: Test success rate below 70% - prioritize fixing failing tests');
      process.exit(1);
    }
    
  } catch (error) {
    console.error('Analysis failed:', error);
    process.exit(1);
  }
}

main();