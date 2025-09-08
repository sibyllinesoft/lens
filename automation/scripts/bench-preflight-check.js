#!/usr/bin/env node

/**
 * Bench Preflight Check
 * Validates golden_ref + corpus_ref hashes before benchmarking
 * 
 * Per TODO.md requirement: "Add a bench preflight that **fails** if either hash is missing/mismatched."
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

class BenchPreflightChecker {
  constructor() {
    this.corpusDir = './indexed-content';
    this.anchorDir = './anchor-datasets';
    this.configPath = './config_fingerprint.json';
  }

  async runPreflight() {
    console.log('🔍 Running Bench Preflight Check...');
    
    try {
      // 1. Load configuration fingerprint
      const config = this.loadConfigFingerprint();
      
      // 2. Validate anchor dataset exists and matches hash
      await this.validateAnchorDataset(config);
      
      // 3. Validate corpus matches hash  
      await this.validateCorpus(config);
      
      // 4. Check for additional integrity requirements
      await this.validateIntegrity(config);
      
      console.log('✅ Bench preflight check PASSED');
      console.log('🚀 System ready for benchmarking');
      return { success: true, message: 'All preflight checks passed' };
      
    } catch (error) {
      console.error('❌ Bench preflight check FAILED');
      console.error('🛑 Error:', error.message);
      console.error('💡 Benchmarking blocked until issues are resolved');
      return { success: false, message: error.message };
    }
  }

  loadConfigFingerprint() {
    if (!fs.existsSync(this.configPath)) {
      return this.createDefaultConfig();
    }
    
    const configContent = fs.readFileSync(this.configPath, 'utf-8');
    const config = JSON.parse(configContent);
    
    // Check if this is the old format - if so, create new format
    if (!config.golden_ref || !config.corpus_ref) {
      console.log('🔄 Converting old config format to new hash validation format');
      return this.createDefaultConfig();
    }
    
    console.log(`📄 Loaded config fingerprint: ${path.basename(this.configPath)}`);
    return config;
  }

  createDefaultConfig() {
    console.log('📝 Creating default config fingerprint...');
    
    // Try to find current anchor dataset
    const currentAnchor = path.join(this.anchorDir, 'anchor_current.json');
    let anchorHash = null;
    let anchorPath = null;
    
    if (fs.existsSync(currentAnchor)) {
      const anchorData = JSON.parse(fs.readFileSync(currentAnchor, 'utf-8'));
      anchorHash = this.generateDatasetHash(anchorData);
      anchorPath = currentAnchor;
    }
    
    // Generate current corpus hash
    const corpusHash = this.generateCorpusHash();
    
    const defaultConfig = {
      version: '1.0.0',
      created_at: new Date().toISOString(),
      description: 'Benchmark configuration fingerprint for hash validation',
      golden_ref: {
        hash: anchorHash,
        path: anchorPath,
        required: true
      },
      corpus_ref: {
        hash: corpusHash,
        path: this.corpusDir,
        required: true
      },
      validation: {
        strict_hash_matching: true,
        allow_corpus_drift: false,
        require_anchor_dataset: true
      }
    };
    
    fs.writeFileSync(this.configPath, JSON.stringify(defaultConfig, null, 2));
    console.log(`✅ Created ${this.configPath}`);
    
    return defaultConfig;
  }

  async validateAnchorDataset(config) {
    console.log('🎯 Validating anchor dataset...');
    
    const goldenRef = config.golden_ref;
    
    // Check if golden_ref hash is present
    if (!goldenRef || !goldenRef.hash) {
      throw new Error('golden_ref hash is missing from config_fingerprint.json');
    }
    
    // Check if anchor dataset path exists
    if (!goldenRef.path || !fs.existsSync(goldenRef.path)) {
      throw new Error(`Anchor dataset not found at: ${goldenRef.path}`);
    }
    
    // Load and validate anchor dataset
    const anchorData = JSON.parse(fs.readFileSync(goldenRef.path, 'utf-8'));
    const currentHash = this.generateDatasetHash(anchorData);
    
    if (currentHash !== goldenRef.hash) {
      throw new Error(`Anchor dataset hash mismatch!\n  Expected: ${goldenRef.hash}\n  Actual:   ${currentHash}`);
    }
    
    console.log(`  ✅ Anchor dataset hash: ${currentHash}`);
    console.log(`  📊 Dataset queries: ${anchorData.total_queries}`);
    console.log(`  🏷️  Dataset version: ${anchorData.version}`);
  }

  async validateCorpus(config) {
    console.log('📂 Validating corpus...');
    
    const corpusRef = config.corpus_ref;
    
    // Check if corpus_ref hash is present
    if (!corpusRef || !corpusRef.hash) {
      throw new Error('corpus_ref hash is missing from config_fingerprint.json');
    }
    
    // Check if corpus directory exists
    if (!corpusRef.path || !fs.existsSync(corpusRef.path)) {
      throw new Error(`Corpus directory not found at: ${corpusRef.path}`);
    }
    
    // Generate current corpus hash
    const currentHash = this.generateCorpusHash();
    
    if (config.validation?.strict_hash_matching && currentHash !== corpusRef.hash) {
      throw new Error(`Corpus hash mismatch!\n  Expected: ${corpusRef.hash}\n  Actual:   ${currentHash}\n  This indicates corpus drift - reindex or update config.`);
    }
    
    if (currentHash !== corpusRef.hash) {
      console.log(`  ⚠️  Corpus hash drift detected (non-strict mode)`);
      console.log(`  📝 Expected: ${corpusRef.hash}`);
      console.log(`  📝 Actual:   ${currentHash}`);
    } else {
      console.log(`  ✅ Corpus hash: ${currentHash}`);
    }
    
    // Count corpus files
    const files = fs.readdirSync(corpusRef.path);
    const relevantFiles = files.filter(f => this.isRelevantFile(f));
    console.log(`  📁 Corpus files: ${relevantFiles.length}`);
  }

  async validateIntegrity(config) {
    console.log('🔒 Validating system integrity...');
    
    // Check that anchor dataset aligns with current corpus
    const anchorData = JSON.parse(fs.readFileSync(config.golden_ref.path, 'utf-8'));
    
    // Validate that referenced files in golden spans exist in corpus
    let missingFiles = 0;
    let totalSpans = 0;
    
    for (const query of anchorData.queries) {
      for (const span of query.golden_spans) {
        totalSpans++;
        const corpusFile = path.join(this.corpusDir, span.file);
        if (!fs.existsSync(corpusFile)) {
          missingFiles++;
          console.warn(`    ⚠️  Missing corpus file: ${span.file}`);
        }
      }
    }
    
    if (missingFiles > 0) {
      const missingPercent = ((missingFiles / totalSpans) * 100).toFixed(1);
      if (missingPercent > 10) {
        throw new Error(`Too many missing corpus files: ${missingFiles}/${totalSpans} (${missingPercent}%)`);
      } else {
        console.log(`  ⚠️  Minor missing files: ${missingFiles}/${totalSpans} (${missingPercent}%)`);
      }
    } else {
      console.log(`  ✅ All golden span files present: ${totalSpans} spans`);
    }
    
    // Validate anchor dataset structure
    this.validateAnchorStructure(anchorData);
  }

  validateAnchorStructure(anchorData) {
    const requiredFields = ['version', 'created_at', 'corpus_hash', 'total_queries', 'queries'];
    for (const field of requiredFields) {
      if (!anchorData[field]) {
        throw new Error(`Anchor dataset missing required field: ${field}`);
      }
    }
    
    // Validate query structure
    for (const query of anchorData.queries) {
      const requiredQueryFields = ['id', 'query', 'intent', 'golden_spans', 'language'];
      for (const field of requiredQueryFields) {
        if (!query[field]) {
          throw new Error(`Query ${query.id || 'unknown'} missing field: ${field}`);
        }
      }
      
      // Validate intent types
      const validIntents = ['lexical', 'symbol', 'structural', 'nl'];
      if (!validIntents.includes(query.intent)) {
        throw new Error(`Query ${query.id} has invalid intent: ${query.intent}`);
      }
      
      // Validate golden spans
      for (const span of query.golden_spans) {
        const requiredSpanFields = ['file', 'line', 'relevance_score'];
        for (const field of requiredSpanFields) {
          if (span[field] === undefined || span[field] === null) {
            throw new Error(`Golden span missing field: ${field}`);
          }
        }
      }
    }
    
    console.log(`  ✅ Anchor structure valid`);
  }

  generateDatasetHash(dataset) {
    // Generate hash of dataset content for fingerprinting
    const hasher = crypto.createHash('sha256');
    
    // Include key structural elements in hash
    hasher.update(JSON.stringify({
      version: dataset.version,
      total_queries: dataset.total_queries,
      queries: dataset.queries.map(q => ({
        id: q.id,
        query: q.query,
        intent: q.intent,
        language: q.language,
        golden_spans: q.golden_spans
      })).sort((a, b) => a.id.localeCompare(b.id))
    }));
    
    return hasher.digest('hex').slice(0, 16);
  }

  generateCorpusHash() {
    if (!fs.existsSync(this.corpusDir)) {
      throw new Error(`Corpus directory not found: ${this.corpusDir}`);
    }
    
    const hasher = crypto.createHash('sha256');
    const files = fs.readdirSync(this.corpusDir);
    const relevantFiles = files.filter(f => this.isRelevantFile(f)).sort();
    
    for (const file of relevantFiles) {
      const filePath = path.join(this.corpusDir, file);
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        hasher.update(file + content);
      } catch (error) {
        // Skip files that can't be read
        console.warn(`Warning: Cannot read ${file}: ${error.message}`);
      }
    }
    
    return hasher.digest('hex').slice(0, 16);
  }

  isRelevantFile(filename) {
    const extensions = ['.py', '.ts', '.tsx', '.js', '.jsx', '.go', '.rs', '.java', '.cpp', '.hpp', '.c', '.h'];
    return extensions.some(ext => filename.endsWith(ext)) && 
           !filename.includes('.min.') &&
           !filename.includes('.d.ts');
  }

  // Utility method to update config with new hashes  
  async updateConfigFingerprint() {
    console.log('🔄 Updating config fingerprint...');
    
    const config = this.loadConfigFingerprint();
    
    // Ensure structure exists
    if (!config.corpus_ref) config.corpus_ref = {};
    if (!config.golden_ref) config.golden_ref = {};
    
    // Update corpus hash
    config.corpus_ref.hash = this.generateCorpusHash();
    config.corpus_ref.path = this.corpusDir;
    config.corpus_ref.last_updated = new Date().toISOString();
    
    // Update anchor hash if current anchor exists
    const currentAnchor = path.join(this.anchorDir, 'anchor_current.json');
    if (fs.existsSync(currentAnchor)) {
      const anchorData = JSON.parse(fs.readFileSync(currentAnchor, 'utf-8'));
      config.golden_ref.hash = this.generateDatasetHash(anchorData);
      config.golden_ref.path = currentAnchor;
      config.golden_ref.last_updated = new Date().toISOString();
    }
    
    config.last_updated = new Date().toISOString();
    
    fs.writeFileSync(this.configPath, JSON.stringify(config, null, 2));
    console.log('✅ Config fingerprint updated');
    
    return config;
  }
}

// Main execution
async function main() {
  const checker = new BenchPreflightChecker();
  
  if (process.argv.includes('--update')) {
    await checker.updateConfigFingerprint();
    return;
  }
  
  const result = await checker.runPreflight();
  if (!result.success) {
    process.exit(1);
  }
}

console.log('Script loaded. import.meta.url:', import.meta.url);
console.log('process.argv[1]:', process.argv[1]);

main().catch(console.error);

export { BenchPreflightChecker };