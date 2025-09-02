#!/usr/bin/env node

/**
 * Index the Smith project for realistic code search benchmarking
 * 
 * Smith has 25,675+ files making it excellent for testing code search at scale.
 * This creates indexed content and golden queries for proper benchmark testing.
 */

import fs from 'fs/promises';
import path from 'path';
import { execSync } from 'child_process';

const SMITH_PROJECT_ROOT = '/media/nathan/Seagate Hub/Projects/smith';
const LENS_ROOT = '/media/nathan/Seagate Hub/Projects/lens';
const INDEXED_CONTENT_DIR = path.join(LENS_ROOT, 'indexed-content');
const SMITH_INDEXED_DIR = path.join(INDEXED_CONTENT_DIR, 'smith-src');

// Import our TypeScript indexer
async function importIndexer() {
  try {
    console.log('📦 Loading code indexer...');
    const indexerModule = await import('./dist/indexer.js');
    if (indexerModule.codeIndexer) {
      console.log('✅ Loaded compiled indexer from dist/');
      return indexerModule.codeIndexer;
    }
  } catch (error) {
    console.log('💡 Trying TypeScript source directly...');
  }
  
  try {
    const indexerModule = await import('./src/indexer.ts');
    if (indexerModule.codeIndexer) {
      console.log('✅ Loaded indexer from TypeScript source');
      return indexerModule.codeIndexer;
    }
  } catch (error) {
    console.error('❌ Failed to import indexer:', error.message);
    console.log('💡 Run: npm run build OR npx tsx index-smith-project.js');
    process.exit(1);
  }
}

async function getSmithGitInfo() {
  try {
    const originalDir = process.cwd();
    process.chdir(SMITH_PROJECT_ROOT);
    
    const sha = execSync('git rev-parse HEAD', { encoding: 'utf-8' }).trim();
    const ref = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf-8' }).trim();
    
    process.chdir(originalDir);
    
    console.log(`📍 Smith Git SHA: ${sha.slice(0, 8)}...`);
    console.log(`🌿 Smith Git Ref: ${ref}`);
    
    return { sha, ref };
  } catch (error) {
    console.error('❌ Failed to get Smith git info:', error.message);
    process.exit(1);
  }
}

async function copySmithFiles() {
  console.log(`📂 Copying Smith source files to ${SMITH_INDEXED_DIR}...`);
  
  // Ensure target directory exists
  await fs.mkdir(SMITH_INDEXED_DIR, { recursive: true });
  
  let fileCount = 0;
  const languages = new Set();
  const sampleFiles = [];
  
  async function copyRecursive(srcDir, destDir, relativePath = '') {
    try {
      const entries = await fs.readdir(srcDir, { withFileTypes: true });
      
      for (const entry of entries) {
        const srcPath = path.join(srcDir, entry.name);
        const destPath = path.join(destDir, entry.name);
        const currentRelative = path.join(relativePath, entry.name);
        
        if (entry.isDirectory()) {
          // Skip unwanted directories
          if (['node_modules', '.git', 'dist', 'build', '.next', 'target', 'coverage', '.pytest_cache', '__pycache__'].includes(entry.name)) {
            continue;
          }
          
          await fs.mkdir(destPath, { recursive: true });
          await copyRecursive(srcPath, destPath, currentRelative);
        } else if (entry.isFile()) {
          // Only copy code files
          const ext = path.extname(entry.name).toLowerCase();
          const codeExtensions = ['.rs', '.ts', '.js', '.tsx', '.jsx', '.py', '.go', '.java', '.cpp', '.c', '.h', '.hpp', '.cs'];
          
          if (codeExtensions.includes(ext)) {
            await fs.copyFile(srcPath, destPath);
            fileCount++;
            
            // Track language
            const langMap = {
              '.rs': 'rust',
              '.ts': 'typescript', 
              '.tsx': 'typescript',
              '.js': 'javascript',
              '.jsx': 'javascript',
              '.py': 'python',
              '.go': 'go',
              '.java': 'java',
              '.cpp': 'cpp',
              '.c': 'c',
              '.h': 'c',
              '.hpp': 'cpp',
              '.cs': 'csharp'
            };
            
            if (langMap[ext]) {
              languages.add(langMap[ext]);
            }
            
            // Collect sample files for golden data
            if (sampleFiles.length < 100 && (entry.name.includes('lib') || entry.name.includes('main') || entry.name.includes('api'))) {
              sampleFiles.push({
                file: currentRelative,
                language: langMap[ext] || 'unknown',
                ext
              });
            }
            
            if (fileCount % 1000 === 0) {
              console.log(`   📄 Copied ${fileCount} files...`);
            }
          }
        }
      }
    } catch (error) {
      console.warn(`⚠️ Skipping directory ${srcDir}: ${error.message}`);
    }
  }
  
  await copyRecursive(SMITH_PROJECT_ROOT, SMITH_INDEXED_DIR);
  
  console.log(`✅ Copied ${fileCount} Smith source files`);
  console.log(`🗣️ Languages detected: ${Array.from(languages).join(', ')}`);
  
  return { fileCount, languages: Array.from(languages), sampleFiles };
}

async function indexSmithFiles(codeIndexer) {
  console.log(`🔍 Indexing Smith files...`);
  
  try {
    await codeIndexer.indexDirectory(SMITH_INDEXED_DIR);
    
    const stats = codeIndexer.getIndexStats();
    console.log('📊 Index Statistics:', stats);
    
    return stats;
  } catch (error) {
    console.error('❌ Failed to index Smith files:', error);
    throw error;
  }
}

async function createSmithManifest(gitInfo, fileCount, languages) {
  const manifestPath = path.join(INDEXED_CONTENT_DIR, `${gitInfo.sha.slice(0, 8)}-smith.manifest.json`);
  
  console.log(`📋 Creating Smith manifest: ${manifestPath}`);
  
  // Get all code files in smith-src directory
  const sourceFiles = [];
  
  async function collectFiles(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        await collectFiles(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        const codeExtensions = ['.rs', '.ts', '.js', '.tsx', '.jsx', '.py', '.go', '.java', '.cpp', '.c', '.h', '.hpp', '.cs'];
        
        if (codeExtensions.includes(ext)) {
          sourceFiles.push(fullPath);
        }
      }
    }
  }
  
  await collectFiles(SMITH_INDEXED_DIR);
  
  // Create manifest for Smith project
  const manifest = {
    repo_sha: gitInfo.sha,
    repo_ref: gitInfo.ref,
    version: "1.0.0",
    languages: languages,
    shard_paths: sourceFiles,
    created_at: new Date().toISOString(),
    file_count: sourceFiles.length,
    source_project: "smith",
    description: "Smith project codebase for comprehensive code search testing"
  };
  
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log(`✅ Created Smith manifest with ${sourceFiles.length} source files and ${languages.length} languages`);
  return { manifestPath, sourceFiles };
}

async function createSmithGoldenData(sampleFiles, sourceFiles, gitInfo) {
  console.log('🏗️ Creating golden dataset from Smith codebase...');
  
  // Import UUID
  const { v4: uuidv4 } = await import('uuid');
  
  const goldenItems = [];
  
  // Sample a subset of files to analyze
  const filesToAnalyze = sampleFiles.slice(0, 50); // Analyze first 50 interesting files
  
  for (const fileInfo of filesToAnalyze) {
    try {
      const fullPath = path.join(SMITH_INDEXED_DIR, fileInfo.file);
      const content = await fs.readFile(fullPath, 'utf-8');
      const lines = content.split('\n');
      
      // Extract meaningful identifiers based on language
      const extractedItems = [];
      
      for (let i = 0; i < Math.min(lines.length, 200); i++) { // Limit to first 200 lines per file
        const line = lines[i].trim();
        
        if (fileInfo.language === 'rust') {
          // Rust patterns
          const structMatch = line.match(/^(?:pub\s+)?struct\s+(\w+)/);
          const implMatch = line.match(/^impl(?:\s*<[^>]*>)?\s+(\w+)/);
          const fnMatch = line.match(/^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)/);
          const traitMatch = line.match(/^(?:pub\s+)?trait\s+(\w+)/);
          const enumMatch = line.match(/^(?:pub\s+)?enum\s+(\w+)/);
          
          if (structMatch) extractedItems.push({ type: 'struct', name: structMatch[1], line: i + 1 });
          if (implMatch) extractedItems.push({ type: 'impl', name: implMatch[1], line: i + 1 });
          if (fnMatch) extractedItems.push({ type: 'function', name: fnMatch[1], line: i + 1 });
          if (traitMatch) extractedItems.push({ type: 'trait', name: traitMatch[1], line: i + 1 });
          if (enumMatch) extractedItems.push({ type: 'enum', name: enumMatch[1], line: i + 1 });
          
        } else if (fileInfo.language === 'typescript' || fileInfo.language === 'javascript') {
          // TypeScript/JavaScript patterns
          const classMatch = line.match(/^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)/);
          const interfaceMatch = line.match(/^(?:export\s+)?interface\s+(\w+)/);
          const functionMatch = line.match(/^(?:export\s+)?(?:async\s+)?function\s+(\w+)/);
          const constMatch = line.match(/^(?:export\s+)?const\s+(\w+)\s*=.*(?:=\s*>|\bfunction\b)/);
          const typeMatch = line.match(/^(?:export\s+)?type\s+(\w+)/);
          
          if (classMatch) extractedItems.push({ type: 'class', name: classMatch[1], line: i + 1 });
          if (interfaceMatch) extractedItems.push({ type: 'interface', name: interfaceMatch[1], line: i + 1 });
          if (functionMatch) extractedItems.push({ type: 'function', name: functionMatch[1], line: i + 1 });
          if (constMatch) extractedItems.push({ type: 'function', name: constMatch[1], line: i + 1 });
          if (typeMatch) extractedItems.push({ type: 'type', name: typeMatch[1], line: i + 1 });
          
        } else if (fileInfo.language === 'python') {
          // Python patterns
          const classMatch = line.match(/^class\s+(\w+)/);
          const defMatch = line.match(/^(?:async\s+)?def\s+(\w+)/);
          
          if (classMatch) extractedItems.push({ type: 'class', name: classMatch[1], line: i + 1 });
          if (defMatch) extractedItems.push({ type: 'function', name: defMatch[1], line: i + 1 });
        }
      }
      
      // Create golden items for the most interesting symbols (limit per file)
      const topItems = extractedItems.slice(0, 5); // Max 5 queries per file
      
      for (const item of topItems) {
        const goldenItem = {
          id: uuidv4(),
          query: item.name,
          query_class: 'identifier',
          language: fileInfo.language === 'rust' ? 'rs' : fileInfo.language === 'typescript' ? 'ts' : fileInfo.language === 'python' ? 'py' : 'js',
          source: 'smith_codebase',
          snapshot_sha: gitInfo.sha,
          slice_tags: ['SMOKE_DEFAULT', 'ALL'],
          expected_results: [{
            file: fileInfo.file,
            line: item.line,
            col: line.indexOf(item.name),
            relevance_score: 1.0,
            match_type: 'symbol',
            why: `${item.type} definition in Smith project`
          }]
        };
        
        goldenItems.push(goldenItem);
      }
      
    } catch (error) {
      console.warn(`⚠️ Skipping file analysis for ${fileInfo.file}: ${error.message}`);
    }
  }
  
  console.log(`✅ Generated ${goldenItems.length} golden test queries from Smith codebase`);
  
  // Save golden dataset
  const goldenPath = path.join(LENS_ROOT, 'benchmark-results', 'smith-golden-dataset.json');
  await fs.writeFile(goldenPath, JSON.stringify(goldenItems, null, 2));
  
  console.log(`📁 Saved Smith golden dataset to: ${goldenPath}`);
  
  return { goldenItems, goldenPath };
}

async function testSmithSearch(codeIndexer) {
  console.log('🔍 Testing search on Smith codebase...');
  
  const testQueries = ['struct', 'impl', 'function', 'class', 'interface', 'async'];
  
  for (const query of testQueries) {
    try {
      const results = codeIndexer.search(query);
      console.log(`   Query "${query}": ${results.length} results`);
      
      if (results.length > 0) {
        const first = results[0];
        const preview = first.text ? first.text.slice(0, 60) + '...' : 'No text';
        console.log(`     Sample: ${first.file}:${first.line} - "${preview}"`);
      }
    } catch (error) {
      console.warn(`   Query "${query}" failed: ${error.message}`);
    }
  }
}

async function main() {
  console.log('🚀 Indexing Smith project for comprehensive code search benchmarking...\n');
  
  try {
    // Check if Smith project exists
    try {
      await fs.access(SMITH_PROJECT_ROOT);
    } catch (error) {
      console.error(`❌ Smith project not found at: ${SMITH_PROJECT_ROOT}`);
      console.log('💡 Make sure the Smith project exists in the parent directory');
      process.exit(1);
    }
    
    // Get Smith git info
    const gitInfo = await getSmithGitInfo();
    
    // Import indexer
    const codeIndexer = await importIndexer();
    
    // Copy Smith files
    const { fileCount, languages, sampleFiles } = await copySmithFiles();
    
    // Index the copied files
    await indexSmithFiles(codeIndexer);
    
    // Create manifest
    const { manifestPath, sourceFiles } = await createSmithManifest(gitInfo, fileCount, languages);
    
    // Create golden test data
    const { goldenItems, goldenPath } = await createSmithGoldenData(sampleFiles, sourceFiles, gitInfo);
    
    // Test search functionality
    await testSmithSearch(codeIndexer);
    
    console.log('\n✅ Smith project indexing complete!');
    console.log(`📊 Indexed: ${fileCount} files across ${languages.length} languages`);
    console.log(`📋 Manifest: ${manifestPath}`);
    console.log(`🏆 Golden queries: ${goldenItems.length} test cases in ${goldenPath}`);
    console.log(`📂 Source: ${SMITH_INDEXED_DIR}`);
    console.log('');
    console.log('🎯 Ready for realistic code search benchmarking!');
    console.log('');
    console.log('💡 Next steps:');
    console.log('   1. Restart the Lens server to load the new indexed content');
    console.log('   2. Run benchmarks using the Smith golden dataset');
    console.log('   3. Compare performance against the current lens-based dataset');
    
  } catch (error) {
    console.error('❌ Failed to index Smith project:', error);
    process.exit(1);
  }
}

// Run if executed directly
import { fileURLToPath } from 'url';

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { main as indexSmithProject };