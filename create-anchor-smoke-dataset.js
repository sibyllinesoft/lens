#!/usr/bin/env node

/**
 * AnchorSmoke Dataset Builder
 * Creates a small, high-quality, human-verifiable dataset following TODO.md specifications
 * 
 * Stratification:
 * - Intent: [lexical, symbol, structural, nl]  
 * - Language: [py, ts/js, go, rust, java, cpp]
 * - Zone: [src, test, vendor, config]
 * - Target: ~10 queries per stratum, ~100 total
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

// Configuration
const CORPUS_DIR = './indexed-content';
const OUTPUT_DIR = './anchor-datasets';
const TARGET_QUERIES_PER_STRATUM = 10;
const MAX_TOTAL_QUERIES = 100;

// Stratification dimensions
const STRATA = {
  intent: ['lexical', 'symbol', 'structural', 'nl'],
  lang: ['py', 'ts', 'js', 'go', 'rust', 'java', 'cpp'],
  zone: ['src', 'test', 'vendor', 'config']
};

class AnchorSmokeBuilder {
  constructor() {
    this.corpusFiles = [];
    this.anchorQueries = [];
    this.createdAt = new Date().toISOString();
    this.version = this.generateVersion();
  }

  generateVersion() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5) + 'Z';
    const hash = crypto.randomBytes(4).toString('hex');
    return `anchor-${hash}-${timestamp}`;
  }

  async loadCorpus() {
    console.log('📂 Loading corpus files...');
    
    if (!fs.existsSync(CORPUS_DIR)) {
      throw new Error(`Corpus directory not found: ${CORPUS_DIR}`);
    }

    const files = fs.readdirSync(CORPUS_DIR);
    this.corpusFiles = files
      .filter(f => this.isRelevantFile(f))
      .map(f => ({
        filename: f,
        path: path.join(CORPUS_DIR, f),
        language: this.extractLanguage(f),
        zone: this.extractZone(f),
        content: fs.readFileSync(path.join(CORPUS_DIR, f), 'utf-8')
      }));

    console.log(`✅ Loaded ${this.corpusFiles.length} corpus files`);
    console.log('   Languages:', [...new Set(this.corpusFiles.map(f => f.language))]);
    console.log('   Zones:', [...new Set(this.corpusFiles.map(f => f.zone))]);
  }

  isRelevantFile(filename) {
    const extensions = ['.py', '.ts', '.tsx', '.js', '.jsx', '.go', '.rs', '.java', '.cpp', '.hpp', '.c', '.h'];
    return extensions.some(ext => filename.endsWith(ext)) && 
           !filename.includes('.min.') &&
           !filename.includes('.d.ts');
  }

  extractLanguage(filename) {
    if (filename.endsWith('.py')) return 'py';
    if (filename.endsWith('.ts') || filename.endsWith('.tsx')) return 'ts';
    if (filename.endsWith('.js') || filename.endsWith('.jsx')) return 'js';
    if (filename.endsWith('.go')) return 'go';
    if (filename.endsWith('.rs')) return 'rust';
    if (filename.endsWith('.java')) return 'java';
    if (filename.endsWith('.cpp') || filename.endsWith('.hpp') || filename.endsWith('.c') || filename.endsWith('.h')) return 'cpp';
    return 'unknown';
  }

  extractZone(filename) {
    const path = filename.toLowerCase();
    if (path.includes('test') || path.includes('spec') || path.includes('__tests__')) return 'test';
    if (path.includes('vendor') || path.includes('node_modules') || path.includes('third_party')) return 'vendor';
    if (path.includes('config') || path.includes('.yml') || path.includes('.yaml') || path.includes('.json') || path.includes('settings')) return 'config';
    if (path.includes('src') || path.includes('lib') || path.includes('api') || path.includes('core')) return 'src';
    return 'src'; // Default to src
  }

  generateQueriesForFile(file, intentTypes = ['lexical', 'symbol', 'structural', 'nl']) {
    const queries = [];
    const lines = file.content.split('\n');
    
    // Generate queries for each requested intent type
    for (const intent of intentTypes) {
      let intentQueries = [];
      
      switch (intent) {
        case 'lexical':
          intentQueries = this.extractLexicalQueries(file, lines);
          break;
        case 'symbol':
          intentQueries = this.extractSymbolQueries(file, lines);
          break;
        case 'structural':
          intentQueries = this.extractStructuralQueries(file, lines);
          break;
        case 'nl':
          intentQueries = this.extractNLQueries(file, lines);
          break;
      }
      
      queries.push(...intentQueries.slice(0, 1)); // Take max 1 per intent type
    }

    return queries.map(q => ({
      ...q,
      language: file.language,
      zone: file.zone,
      source_file: file.filename
    }));
  }

  extractLexicalQueries(file, lines) {
    const queries = [];
    const keywords = new Set();
    
    // Extract meaningful identifiers
    const identifierRegex = /\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b/g;
    let match;
    
    for (let lineNum = 0; lineNum < lines.length && queries.length < 3; lineNum++) {
      const line = lines[lineNum];
      while ((match = identifierRegex.exec(line)) !== null) {
        const identifier = match[1];
        if (!keywords.has(identifier) && this.isInterestingIdentifier(identifier)) {
          keywords.add(identifier);
          queries.push({
            query: identifier,
            intent: 'lexical',
            golden_spans: [{
              file: file.filename,
              line: lineNum + 1,
              col: match.index + 1,
              relevance_score: 1.0,
              match_type: 'exact',
              why: 'identifier usage'
            }]
          });
          if (queries.length >= 3) break;
        }
      }
    }
    
    return queries;
  }

  extractSymbolQueries(file, lines) {
    const queries = [];
    
    if (file.language === 'py') {
      return this.extractPythonSymbols(file, lines);
    } else if (file.language === 'ts' || file.language === 'js') {
      return this.extractTypeScriptSymbols(file, lines);
    }
    
    return queries;
  }

  extractPythonSymbols(file, lines) {
    const queries = [];
    
    for (let i = 0; i < lines.length && queries.length < 3; i++) {
      const line = lines[i].trim();
      
      // Class definitions
      const classMatch = line.match(/^class\s+(\w+)/);
      if (classMatch) {
        queries.push({
          query: classMatch[1],
          intent: 'symbol',
          golden_spans: [{
            file: file.filename,
            line: i + 1,
            col: line.indexOf(classMatch[1]) + 1,
            relevance_score: 1.0,
            match_type: 'definition',
            why: 'class definition'
          }]
        });
      }
      
      // Function definitions
      const funcMatch = line.match(/^def\s+(\w+)/);
      if (funcMatch && queries.length < 3) {
        queries.push({
          query: funcMatch[1],
          intent: 'symbol',
          golden_spans: [{
            file: file.filename,
            line: i + 1,
            col: line.indexOf(funcMatch[1]) + 1,
            relevance_score: 1.0,
            match_type: 'definition',
            why: 'function definition'
          }]
        });
      }
    }
    
    return queries;
  }

  extractTypeScriptSymbols(file, lines) {
    const queries = [];
    
    for (let i = 0; i < lines.length && queries.length < 3; i++) {
      const line = lines[i].trim();
      
      // Interface definitions
      const interfaceMatch = line.match(/^(?:export\s+)?interface\s+(\w+)/);
      if (interfaceMatch) {
        queries.push({
          query: interfaceMatch[1],
          intent: 'symbol',
          golden_spans: [{
            file: file.filename,
            line: i + 1,
            col: line.indexOf(interfaceMatch[1]) + 1,
            relevance_score: 1.0,
            match_type: 'definition',
            why: 'interface definition'
          }]
        });
      }
      
      // Class definitions
      const classMatch = line.match(/^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)/);
      if (classMatch && queries.length < 3) {
        queries.push({
          query: classMatch[1],
          intent: 'symbol',
          golden_spans: [{
            file: file.filename,
            line: i + 1,
            col: line.indexOf(classMatch[1]) + 1,
            relevance_score: 1.0,
            match_type: 'definition',
            why: 'class definition'
          }]
        });
      }
      
      // Function definitions
      const funcMatch = line.match(/^(?:export\s+)?(?:async\s+)?function\s+(\w+)/);
      if (funcMatch && queries.length < 3) {
        queries.push({
          query: funcMatch[1],
          intent: 'symbol',
          golden_spans: [{
            file: file.filename,
            line: i + 1,
            col: line.indexOf(funcMatch[1]) + 1,
            relevance_score: 1.0,
            match_type: 'definition',
            why: 'function definition'
          }]
        });
      }
    }
    
    return queries;
  }

  extractStructuralQueries(file, lines) {
    const queries = [];
    
    if (file.language === 'py') {
      // Python structural patterns
      if (file.content.includes('class ') && file.content.includes('def __init__')) {
        queries.push({
          query: 'class with constructor',
          intent: 'structural',
          golden_spans: [{
            file: file.filename,
            line: 1,
            col: 1,
            relevance_score: 0.8,
            match_type: 'structural',
            why: 'class with __init__ method'
          }]
        });
      }
    } else if (file.language === 'ts') {
      // TypeScript structural patterns  
      if (file.content.includes('interface ') && file.content.includes('extends')) {
        queries.push({
          query: 'interface extending another',
          intent: 'structural',
          golden_spans: [{
            file: file.filename,
            line: 1,
            col: 1,
            relevance_score: 0.8,
            match_type: 'structural',
            why: 'interface inheritance'
          }]
        });
      }
    }
    
    return queries.slice(0, 1); // Limit structural queries
  }

  extractNLQueries(file, lines) {
    const queries = [];
    
    // Generate semantic queries based on file purpose
    if (file.filename.includes('database') || file.content.includes('db') || file.content.includes('query')) {
      queries.push({
        query: 'database operations',
        intent: 'nl',
        golden_spans: [{
          file: file.filename,
          line: 1,
          col: 1,
          relevance_score: 0.7,
          match_type: 'semantic',
          why: 'database-related functionality'
        }]
      });
    }
    
    if (file.filename.includes('api') || file.content.includes('endpoint') || file.content.includes('route')) {
      queries.push({
        query: 'API endpoint definition',
        intent: 'nl',
        golden_spans: [{
          file: file.filename,
          line: 1,
          col: 1,
          relevance_score: 0.7,
          match_type: 'semantic',
          why: 'API endpoint functionality'
        }]
      });
    }
    
    return queries.slice(0, 1); // Limit NL queries
  }

  isInterestingIdentifier(identifier) {
    // Filter out common words and short identifiers
    const boring = ['the', 'and', 'for', 'with', 'from', 'this', 'that', 'data', 'info', 'item', 'obj', 'val', 'var', 'tmp', 'temp'];
    return identifier.length > 3 && 
           !boring.includes(identifier.toLowerCase()) &&
           !/^[A-Z]+$/.test(identifier) && // Skip ALL_CAPS
           !/^\d+$/.test(identifier); // Skip numbers
  }

  generateAnchorDataset() {
    console.log('🎯 Generating AnchorSmoke dataset...');
    
    // Group files by language and zone
    const filesByStrata = {};
    
    for (const lang of STRATA.lang) {
      for (const zone of STRATA.zone) {
        const key = `${lang}-${zone}`;
        filesByStrata[key] = this.corpusFiles.filter(f => f.language === lang && f.zone === zone);
      }
    }
    
    // Generate queries for each stratum, ensuring intent diversity
    let totalQueries = 0;
    const queriesPerStrata = Math.max(4, Math.floor(MAX_TOTAL_QUERIES / Object.keys(filesByStrata).filter(key => filesByStrata[key].length > 0).length));
    
    for (const [stratum, files] of Object.entries(filesByStrata)) {
      if (files.length === 0) continue;
      
      console.log(`  📊 Processing stratum: ${stratum} (${files.length} files)`);
      
      // Sample files from this stratum - more files for more diverse queries
      const sampleSize = Math.min(files.length, Math.max(2, Math.floor(queriesPerStrata / 2)));
      const sampledFiles = this.sampleFiles(files, sampleSize);
      
      // Generate queries with intent diversity
      let stratumQueries = 0;
      const intentTypesNeeded = [...STRATA.intent]; // ['lexical', 'symbol', 'structural', 'nl']
      
      for (const file of sampledFiles) {
        if (stratumQueries >= queriesPerStrata) break;
        
        try {
          // Cycle through intent types to ensure diversity
          const intentsForThisFile = intentTypesNeeded.splice(0, Math.min(intentTypesNeeded.length, queriesPerStrata - stratumQueries));
          if (intentsForThisFile.length === 0) break;
          
          const fileQueries = this.generateQueriesForFile(file, intentsForThisFile);
          
          for (const query of fileQueries) {
            if (stratumQueries >= queriesPerStrata) break;
            
            this.anchorQueries.push({
              id: crypto.randomUUID(),
              ...query,
              stratum,
              created_at: this.createdAt
            });
            stratumQueries++;
            totalQueries++;
          }
        } catch (error) {
          console.warn(`    ⚠️  Error processing ${file.filename}: ${error.message}`);
        }
      }
      
      console.log(`    ✅ Generated ${stratumQueries} queries for ${stratum}`);
    }
    
    console.log(`🎯 Total anchor queries generated: ${totalQueries}`);
  }

  sampleFiles(files, sampleSize) {
    // Simple random sampling
    const shuffled = [...files].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, sampleSize);
  }

  generateCorpusHash() {
    // Generate hash of corpus state for fingerprinting
    const hasher = crypto.createHash('sha256');
    
    const sortedFiles = [...this.corpusFiles].sort((a, b) => a.filename.localeCompare(b.filename));
    for (const file of sortedFiles) {
      hasher.update(file.filename + file.content);
    }
    
    return hasher.digest('hex').slice(0, 16);
  }

  async saveAnchorDataset() {
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    const corpusHash = this.generateCorpusHash();
    const dataset = {
      version: this.version,
      created_at: this.createdAt,
      corpus_hash: corpusHash,
      git_sha: process.env.GIT_SHA || 'unknown',
      total_queries: this.anchorQueries.length,
      stratification: {
        intent_distribution: this.getDistribution('intent'),
        language_distribution: this.getDistribution('language'),
        zone_distribution: this.getDistribution('zone')
      },
      queries: this.anchorQueries
    };

    const filename = `anchor_${this.version}.json`;
    const filepath = path.join(OUTPUT_DIR, filename);
    
    fs.writeFileSync(filepath, JSON.stringify(dataset, null, 2));
    
    // Also create a 'current' symlink
    const currentPath = path.join(OUTPUT_DIR, 'anchor_current.json');
    if (fs.existsSync(currentPath)) {
      fs.unlinkSync(currentPath);
    }
    fs.writeFileSync(currentPath, JSON.stringify(dataset, null, 2));

    console.log(`💾 Saved AnchorSmoke dataset: ${filepath}`);
    console.log(`📊 Dataset stats:`);
    console.log(`   Total queries: ${dataset.total_queries}`);
    console.log(`   Intent distribution:`, dataset.stratification.intent_distribution);
    console.log(`   Language distribution:`, dataset.stratification.language_distribution);
    console.log(`   Zone distribution:`, dataset.stratification.zone_distribution);
    console.log(`   Corpus hash: ${corpusHash}`);
    
    return dataset;
  }

  getDistribution(field) {
    const dist = {};
    for (const query of this.anchorQueries) {
      const value = query[field] || 'unknown';
      dist[value] = (dist[value] || 0) + 1;
    }
    return dist;
  }
}

// Main execution
async function main() {
  try {
    console.log('🚀 Starting AnchorSmoke dataset generation...');
    
    const builder = new AnchorSmokeBuilder();
    await builder.loadCorpus();
    builder.generateAnchorDataset();
    const dataset = await builder.saveAnchorDataset();
    
    console.log('✅ AnchorSmoke dataset generation complete!');
    console.log(`📁 Output: ./anchor-datasets/anchor_${builder.version}.json`);
    
  } catch (error) {
    console.error('❌ Error creating AnchorSmoke dataset:', error);
    process.exit(1);
  }
}

console.log('Script loaded. import.meta.url:', import.meta.url);
console.log('process.argv[1]:', process.argv[1]);

main().catch(console.error);

export { AnchorSmokeBuilder };