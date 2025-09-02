#!/usr/bin/env node

/**
 * Create an indexed version of the Lens TypeScript source code for benchmarking
 * 
 * This script:
 * 1. Gets the current git SHA and branch information
 * 2. Copies all TypeScript/JavaScript source files from src/ to indexed-content/lens-src/
 * 3. Uses the Lens codeIndexer to index the copied files
 * 4. Creates a proper .manifest.json file that the IndexRegistry can discover
 * 5. Tests the search functionality to verify everything works
 * 
 * Usage:
 *   node create-lens-index.js
 *   npx tsx create-lens-index.js  (if TypeScript isn't compiled)
 * 
 * Prerequisites:
 *   - Run `npm install` to install dependencies
 *   - Optionally run `npm run build` to compile TypeScript (faster)
 *   - Must be in a git repository with committed changes
 */

import fs from 'fs/promises';
import path from 'path';
import { execSync } from 'child_process';

// Import our TypeScript indexer
async function importIndexer() {
  try {
    // First try to import the compiled ES module version
    console.log('📦 Loading code indexer (trying compiled version first)...');
    const indexerModule = await import('./dist/indexer.js');
    if (indexerModule.codeIndexer) {
      console.log('✅ Loaded compiled indexer from dist/');
      return indexerModule.codeIndexer;
    }
  } catch (error) {
    console.log('💡 Compiled version not found, trying source directly...');
  }
  
  try {
    // Fall back to importing TypeScript source directly (works with tsx/ts-node)
    const indexerModule = await import('./src/indexer.ts');
    if (indexerModule.codeIndexer) {
      console.log('✅ Loaded indexer from TypeScript source');
      return indexerModule.codeIndexer;
    }
  } catch (error) {
    console.error('❌ Failed to import indexer:', error.message);
    console.log('💡 Solutions:');
    console.log('   1. Run: npm run build (to compile TypeScript to dist/)');
    console.log('   2. Or run this script with tsx: npx tsx create-lens-index.js');
    console.log('   3. Make sure dependencies are installed: npm install');
    process.exit(1);
  }
  
  throw new Error('Could not find codeIndexer export in any module');
}

async function getGitInfo() {
  try {
    const sha = execSync('git rev-parse HEAD', { encoding: 'utf-8' }).trim();
    const ref = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf-8' }).trim();
    
    console.log(`📍 Git SHA: ${sha.slice(0, 8)}...`);
    console.log(`🌿 Git Ref: ${ref}`);
    
    return { sha, ref };
  } catch (error) {
    console.error('❌ Failed to get git information:', error.message);
    console.log('💡 Make sure you\'re in a git repository');
    process.exit(1);
  }
}

async function copySourceFiles() {
  const sourceDir = path.join(process.cwd(), 'src');
  const targetDir = path.join(process.cwd(), 'indexed-content', 'lens-src');
  
  console.log(`📂 Copying source files from ${sourceDir} to ${targetDir}...`);
  
  // Ensure target directory exists
  await fs.mkdir(targetDir, { recursive: true });
  
  let fileCount = 0;
  const languages = new Set();
  
  async function copyRecursive(srcDir, destDir) {
    const entries = await fs.readdir(srcDir, { withFileTypes: true });
    
    for (const entry of entries) {
      const srcPath = path.join(srcDir, entry.name);
      const destPath = path.join(destDir, entry.name);
      
      if (entry.isDirectory()) {
        // Skip node_modules and other unwanted directories
        if (['node_modules', '.git', 'dist', 'build', '.next'].includes(entry.name)) {
          continue;
        }
        
        await fs.mkdir(destPath, { recursive: true });
        await copyRecursive(srcPath, destPath);
      } else if (entry.isFile()) {
        // Only copy code files
        const ext = path.extname(entry.name).toLowerCase();
        const codeExtensions = ['.ts', '.js', '.tsx', '.jsx', '.py', '.go', '.rs', '.java', '.cpp', '.c', '.h'];
        
        if (codeExtensions.includes(ext)) {
          await fs.copyFile(srcPath, destPath);
          fileCount++;
          
          // Track language based on extension
          const langMap = {
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.py': 'python',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c'
          };
          
          if (langMap[ext]) {
            languages.add(langMap[ext]);
          }
          
          if (fileCount % 50 === 0) {
            console.log(`   📄 Copied ${fileCount} files...`);
          }
        }
      }
    }
  }
  
  await copyRecursive(sourceDir, targetDir);
  
  console.log(`✅ Copied ${fileCount} source files`);
  console.log(`🗣️ Languages detected: ${Array.from(languages).join(', ')}`);
  
  return { fileCount, languages: Array.from(languages), targetDir };
}

async function indexFiles(targetDir, codeIndexer) {
  console.log(`🔍 Indexing files in ${targetDir}...`);
  
  try {
    await codeIndexer.indexDirectory(targetDir);
    
    const stats = codeIndexer.getIndexStats();
    console.log('📊 Index Statistics:', stats);
    
    return stats;
  } catch (error) {
    console.error('❌ Failed to index files:', error);
    throw error;
  }
}

async function createManifest(gitInfo, fileCount, languages) {
  const indexContentDir = path.join(process.cwd(), 'indexed-content');
  const manifestPath = path.join(indexContentDir, `${gitInfo.sha.slice(0, 8)}.manifest.json`);
  
  console.log(`📋 Creating manifest file: ${manifestPath}`);
  
  // Get all TypeScript/JavaScript files in the lens-src directory
  const lensSourceDir = path.join(indexContentDir, 'lens-src');
  const sourceFiles = [];
  
  async function collectFiles(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        await collectFiles(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        const codeExtensions = ['.ts', '.js', '.tsx', '.jsx'];
        
        if (codeExtensions.includes(ext)) {
          sourceFiles.push(fullPath);
        }
      }
    }
  }
  
  await collectFiles(lensSourceDir);
  
  // Create manifest structure expected by IndexRegistry
  const manifest = {
    repo_sha: gitInfo.sha,
    repo_ref: gitInfo.ref,
    version: "1.0.0",
    languages: languages,
    shard_paths: sourceFiles, // Individual source files that can be searched
    created_at: new Date().toISOString(),
    file_count: sourceFiles.length
  };
  
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log(`✅ Created manifest with ${sourceFiles.length} source files and ${languages.length} languages`);
  return manifestPath;
}

async function createLensIndex() {
  console.log('🚀 Creating Lens source code index for benchmarking...');
  console.log('');
  
  try {
    // Step 1: Get git information
    const gitInfo = await getGitInfo();
    
    // Step 2: Import the indexer
    console.log('📦 Loading code indexer...');
    const codeIndexer = await importIndexer();
    
    // Step 3: Copy source files
    const { fileCount, languages, targetDir } = await copySourceFiles();
    
    // Step 4: Index the copied files
    await indexFiles(targetDir, codeIndexer);
    
    // Step 5: Create manifest file
    const manifestPath = await createManifest(gitInfo, fileCount, languages);
    
    console.log('');
    console.log('✅ Lens source code indexing complete!');
    console.log(`📋 Manifest: ${manifestPath}`);
    console.log(`📂 Source files: ${targetDir}`);
    console.log(`📊 Total files indexed: ${fileCount}`);
    console.log(`🗣️ Languages: ${languages.join(', ')}`);
    console.log('');
    console.log('🎯 Ready for benchmarking! The indexed content can now be used with the Lens search engine.');
    console.log('');
    console.log('💡 To use this indexed content:');
    console.log('   • The IndexRegistry will automatically discover the manifest file');
    console.log('   • Benchmarks can now use the actual Lens source code');
    console.log('   • Search queries will return results from the real codebase');
    
    // Test a quick search to verify everything is working
    console.log('');
    console.log('🔍 Testing search functionality...');
    const testResults = codeIndexer.search('class');
    console.log(`   Found ${testResults.length} results for "class"`);
    
    if (testResults.length > 0) {
      const first = testResults[0];
      console.log(`   Sample: ${first.file}:${first.line} - "${first.text.slice(0, 60)}..."`);
    }
    
    // Also test a few more queries to show variety
    const queryTests = ['interface', 'async', 'export', 'function'];
    for (const query of queryTests) {
      const results = codeIndexer.search(query);
      if (results.length > 0) {
        console.log(`   Query "${query}": ${results.length} results`);
      }
    }
    
  } catch (error) {
    console.error('❌ Failed to create Lens index:', error);
    process.exit(1);
  }
}

// Run the script if executed directly
import { fileURLToPath } from 'url';

if (import.meta.url === `file://${process.argv[1]}`) {
  createLensIndex().catch(console.error);
}

export { createLensIndex };