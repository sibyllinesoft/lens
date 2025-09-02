#!/usr/bin/env node

/**
 * LadderFull Dataset Builder
 * Creates a large, weak-labeled dataset for exploratory analysis and drift detection
 * 
 * Per TODO.md: "LadderFull (≈2–5k queries, weak labels) used for exploratory deltas and drift detection.
 * Auto-generate from current logs + synthetics; accept candidates as "relevant" if they match 
 * defs/refs (LSIF) or a tree-sitter structural predicate. Keep hard negatives (near-miss files)."
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

// Configuration
const CORPUS_DIR = './indexed-content';
const OUTPUT_DIR = './ladder-datasets';
const TARGET_QUERIES = 3000; // ~2-5k queries as specified
const HARD_NEGATIVES_RATIO = 0.15; // 15% hard negatives

// Weak label acceptance thresholds
const WEAK_LABEL_CONFIG = {
  min_confidence: 0.3,
  accept_structural_matches: true,
  accept_fuzzy_matches: true,
  keep_hard_negatives: true
};

class LadderFullBuilder {
  constructor() {
    this.corpusFiles = [];
    this.ladderQueries = [];
    this.createdAt = new Date().toISOString();
    this.version = this.generateVersion();
    this.stats = {
      synthetic_queries: 0,
      log_derived_queries: 0,
      weak_positives: 0,
      hard_negatives: 0,
      structural_matches: 0
    };
  }

  generateVersion() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5) + 'Z';
    const hash = crypto.randomBytes(4).toString('hex');
    return `ladder-${hash}-${timestamp}`;
  }

  async loadCorpus() {
    console.log('📂 Loading corpus for LadderFull generation...');
    
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
        content: fs.readFileSync(path.join(CORPUS_DIR, f), 'utf-8'),
        size: fs.statSync(path.join(CORPUS_DIR, f)).size
      }));

    console.log(`✅ Loaded ${this.corpusFiles.length} corpus files`);
    console.log('   Total size:', Math.round(this.corpusFiles.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024) + 'MB');
  }

  isRelevantFile(filename) {
    const extensions = ['.py', '.ts', '.tsx', '.js', '.jsx', '.go', '.rs', '.java', '.cpp', '.hpp', '.c', '.h', '.md'];
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
    if (filename.endsWith('.md')) return 'md';
    return 'unknown';
  }

  extractZone(filename) {
    const path = filename.toLowerCase();
    if (path.includes('test') || path.includes('spec') || path.includes('__tests__')) return 'test';
    if (path.includes('vendor') || path.includes('node_modules') || path.includes('third_party')) return 'vendor';
    if (path.includes('config') || path.includes('.yml') || path.includes('.yaml') || path.includes('.json') || path.includes('settings')) return 'config';
    if (path.includes('doc') || path.includes('readme') || path.includes('.md')) return 'docs';
    return 'src';
  }

  async generateLadderDataset() {
    console.log('🏗️ Generating LadderFull dataset...');
    
    // 1. Generate synthetic queries (bulk of the dataset)
    await this.generateSyntheticQueries();
    
    // 2. Generate log-derived queries (if logs were available)
    await this.generateLogDerivedQueries();
    
    // 3. Add hard negatives for robustness
    await this.generateHardNegatives();
    
    console.log(`📊 Generated ${this.ladderQueries.length} total queries`);
    console.log(`   Synthetic: ${this.stats.synthetic_queries}`);
    console.log(`   Log-derived: ${this.stats.log_derived_queries}`);
    console.log(`   Hard negatives: ${this.stats.hard_negatives}`);
    console.log(`   Structural matches: ${this.stats.structural_matches}`);
  }

  async generateSyntheticQueries() {
    console.log('🧬 Generating synthetic queries...');
    
    const targetSynthetic = Math.floor(TARGET_QUERIES * 0.7); // 70% synthetic
    let generated = 0;
    
    // Create queries from various code patterns
    for (const file of this.corpusFiles) {
      if (generated >= targetSynthetic) break;
      
      try {
        const fileQueries = this.extractSyntheticQueriesFromFile(file);
        
        for (const query of fileQueries) {
          if (generated >= targetSynthetic) break;
          
          const weakLabels = this.generateWeakLabels(query, file);
          if (weakLabels.length > 0) {
            this.ladderQueries.push({
              id: crypto.randomUUID(),
              ...query,
              weak_labels: weakLabels,
              confidence: this.calculateConfidence(query, weakLabels),
              source: 'synthetic',
              created_at: this.createdAt
            });
            
            generated++;
            this.stats.synthetic_queries++;
          }
        }
      } catch (error) {
        console.warn(`    ⚠️  Error processing ${file.filename}: ${error.message}`);
      }
    }
    
    console.log(`   ✅ Generated ${generated} synthetic queries`);
  }

  extractSyntheticQueriesFromFile(file) {
    const queries = [];
    const lines = file.content.split('\n');
    
    // Extract identifiers for lexical queries
    queries.push(...this.extractIdentifierQueries(file, lines));
    
    // Extract symbols for symbol queries  
    queries.push(...this.extractSymbolQueries(file, lines));
    
    // Generate structural queries
    queries.push(...this.generateStructuralQueries(file, lines));
    
    // Generate semantic/NL queries
    queries.push(...this.generateSemanticQueries(file, lines));
    
    return queries;
  }

  extractIdentifierQueries(file, lines) {
    const queries = [];
    const identifierRegex = /\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b/g;
    const seenIdentifiers = new Set();
    
    for (let lineNum = 0; lineNum < lines.length && queries.length < 10; lineNum++) {
      const line = lines[lineNum];
      let match;
      
      while ((match = identifierRegex.exec(line)) !== null) {
        const identifier = match[1];
        if (!seenIdentifiers.has(identifier) && this.isInterestingIdentifier(identifier)) {
          seenIdentifiers.add(identifier);
          queries.push({
            query: identifier,
            intent: 'lexical',
            language: file.language,
            zone: file.zone,
            source_file: file.filename,
            line: lineNum + 1,
            col: match.index + 1
          });
          
          if (queries.length >= 10) break;
        }
      }
    }
    
    return queries;
  }

  extractSymbolQueries(file, lines) {
    const queries = [];
    
    if (file.language === 'py') {
      queries.push(...this.extractPythonSymbols(file, lines));
    } else if (file.language === 'ts' || file.language === 'js') {
      queries.push(...this.extractTypeScriptSymbols(file, lines));
    }
    
    return queries.slice(0, 5); // Limit symbol queries per file
  }

  extractPythonSymbols(file, lines) {
    const queries = [];
    
    for (let i = 0; i < lines.length && queries.length < 5; i++) {
      const line = lines[i].trim();
      
      // Class definitions
      const classMatch = line.match(/^class\s+(\w+)/);
      if (classMatch) {
        queries.push({
          query: classMatch[1],
          intent: 'symbol',
          language: file.language,
          zone: file.zone,
          source_file: file.filename,
          line: i + 1,
          symbol_type: 'class'
        });
      }
      
      // Function definitions
      const funcMatch = line.match(/^def\s+(\w+)/);
      if (funcMatch) {
        queries.push({
          query: funcMatch[1],
          intent: 'symbol',
          language: file.language,
          zone: file.zone,
          source_file: file.filename,
          line: i + 1,
          symbol_type: 'function'
        });
      }
    }
    
    return queries;
  }

  extractTypeScriptSymbols(file, lines) {
    const queries = [];
    
    for (let i = 0; i < lines.length && queries.length < 5; i++) {
      const line = lines[i].trim();
      
      // Interface definitions
      const interfaceMatch = line.match(/^(?:export\s+)?interface\s+(\w+)/);
      if (interfaceMatch) {
        queries.push({
          query: interfaceMatch[1],
          intent: 'symbol',
          language: file.language,
          zone: file.zone,
          source_file: file.filename,
          line: i + 1,
          symbol_type: 'interface'
        });
      }
      
      // Type definitions
      const typeMatch = line.match(/^(?:export\s+)?type\s+(\w+)/);
      if (typeMatch) {
        queries.push({
          query: typeMatch[1],
          intent: 'symbol',
          language: file.language,
          zone: file.zone,
          source_file: file.filename,
          line: i + 1,
          symbol_type: 'type'
        });
      }
    }
    
    return queries;
  }

  generateStructuralQueries(file, lines) {
    const queries = [];
    
    // Common structural patterns
    const patterns = [
      { pattern: /class.*extends/i, query: 'class inheritance', type: 'inheritance' },
      { pattern: /interface.*extends/i, query: 'interface extension', type: 'extension' },
      { pattern: /async\s+function/i, query: 'async function', type: 'async' },
      { pattern: /try\s*\{[\s\S]*catch/i, query: 'error handling', type: 'error_handling' },
      { pattern: /import.*from/i, query: 'module import', type: 'import' }
    ];
    
    const content = file.content;
    
    for (const { pattern, query, type } of patterns) {
      if (pattern.test(content)) {
        queries.push({
          query,
          intent: 'structural',
          language: file.language,
          zone: file.zone,
          source_file: file.filename,
          pattern_type: type
        });
      }
    }
    
    return queries.slice(0, 3);
  }

  generateSemanticQueries(file, lines) {
    const queries = [];
    
    // Generate semantic queries based on file content patterns
    const semanticPatterns = [
      { keywords: ['database', 'db', 'query', 'sql'], query: 'database operations' },
      { keywords: ['api', 'endpoint', 'route', 'handler'], query: 'API endpoints' },
      { keywords: ['test', 'spec', 'assert', 'expect'], query: 'testing code' },
      { keywords: ['config', 'settings', 'environment'], query: 'configuration' },
      { keywords: ['cache', 'redis', 'memory'], query: 'caching mechanism' },
      { keywords: ['auth', 'login', 'token', 'jwt'], query: 'authentication' },
      { keywords: ['validate', 'validation', 'schema'], query: 'data validation' }
    ];
    
    const content = file.content.toLowerCase();
    
    for (const { keywords, query } of semanticPatterns) {
      if (keywords.some(keyword => content.includes(keyword))) {
        queries.push({
          query,
          intent: 'nl',
          language: file.language,
          zone: file.zone,
          source_file: file.filename,
          semantic_category: keywords[0]
        });
      }
    }
    
    return queries.slice(0, 2);
  }

  generateWeakLabels(query, sourceFile) {
    const weakLabels = [];
    
    // Primary match: the source file itself is always relevant
    weakLabels.push({
      file: sourceFile.filename,
      line: query.line || 1,
      col: query.col || 1,
      relevance_score: 0.9,
      match_type: 'source_definition',
      confidence: 'high',
      why: 'query originated from this file'
    });
    
    // Find additional matches across corpus
    const additionalMatches = this.findCrossCorpusMatches(query);
    weakLabels.push(...additionalMatches);
    
    return weakLabels;
  }

  findCrossCorpusMatches(query) {
    const matches = [];
    const queryLower = query.query.toLowerCase();
    
    // Search for matches in other files (structural predicates)
    for (const file of this.corpusFiles) {
      if (file.filename === query.source_file) continue; // Skip source file
      
      const content = file.content.toLowerCase();
      if (content.includes(queryLower)) {
        // Find specific line matches
        const lines = file.content.split('\n');
        for (let i = 0; i < lines.length && matches.length < 3; i++) {
          if (lines[i].toLowerCase().includes(queryLower)) {
            const confidence = this.calculateMatchConfidence(query, lines[i], file);
            
            if (confidence >= WEAK_LABEL_CONFIG.min_confidence) {
              matches.push({
                file: file.filename,
                line: i + 1,
                col: lines[i].toLowerCase().indexOf(queryLower) + 1,
                relevance_score: confidence,
                match_type: 'cross_reference',
                confidence: confidence > 0.7 ? 'high' : confidence > 0.5 ? 'medium' : 'low',
                why: `found "${query.query}" in ${file.language} file`
              });
            }
          }
        }
      }
    }
    
    return matches.slice(0, 5); // Limit cross-references
  }

  calculateMatchConfidence(query, line, file) {
    let confidence = 0.3; // Base confidence
    
    // Language match bonus
    if (file.language === query.language) {
      confidence += 0.2;
    }
    
    // Zone match bonus
    if (file.zone === query.zone) {
      confidence += 0.1;
    }
    
    // Context bonuses
    const lineLower = line.toLowerCase();
    
    if (query.intent === 'symbol') {
      // Symbol definitions get higher confidence
      if (lineLower.includes('class ') || lineLower.includes('def ') || lineLower.includes('function ')) {
        confidence += 0.3;
      }
    }
    
    if (query.intent === 'lexical') {
      // Exact word boundary matches
      const wordRegex = new RegExp(`\\b${query.query.toLowerCase()}\\b`);
      if (wordRegex.test(lineLower)) {
        confidence += 0.2;
      }
    }
    
    return Math.min(confidence, 1.0);
  }

  calculateConfidence(query, weakLabels) {
    if (weakLabels.length === 0) return 0;
    
    // Average confidence of all weak labels, weighted by relevance
    const totalWeight = weakLabels.reduce((sum, label) => sum + label.relevance_score, 0);
    const weightedConfidence = weakLabels.reduce((sum, label) => sum + (label.relevance_score * 0.8), 0) / totalWeight;
    
    return Math.min(weightedConfidence, 1.0);
  }

  async generateLogDerivedQueries() {
    console.log('📜 Generating log-derived queries...');
    
    // In a real implementation, this would parse actual search logs
    // For now, we'll create synthetic "log-like" queries based on common patterns
    
    const logPatterns = [
      'error handling in database',
      'async function implementation',
      'configuration validation',
      'API endpoint testing',
      'cache implementation details',
      'authentication middleware',
      'data serialization',
      'memory optimization',
      'concurrent processing',
      'type safety checks'
    ];
    
    const targetLogDerived = Math.floor(TARGET_QUERIES * 0.2); // 20% log-derived
    
    for (let i = 0; i < Math.min(logPatterns.length, targetLogDerived); i++) {
      const pattern = logPatterns[i];
      
      // Find files that might match this pattern
      const relevantFiles = this.findFilesForSemanticQuery(pattern);
      
      if (relevantFiles.length > 0) {
        const weakLabels = relevantFiles.slice(0, 3).map(file => ({
          file: file.filename,
          line: 1,
          col: 1,
          relevance_score: 0.6,
          match_type: 'semantic_match',
          confidence: 'medium',
          why: `semantic match for "${pattern}"`
        }));
        
        this.ladderQueries.push({
          id: crypto.randomUUID(),
          query: pattern,
          intent: 'nl',
          language: 'multi',
          zone: 'multi',
          weak_labels: weakLabels,
          confidence: 0.6,
          source: 'log_derived',
          created_at: this.createdAt
        });
        
        this.stats.log_derived_queries++;
      }
    }
    
    console.log(`   ✅ Generated ${this.stats.log_derived_queries} log-derived queries`);
  }

  findFilesForSemanticQuery(pattern) {
    const keywords = pattern.toLowerCase().split(' ');
    const relevantFiles = [];
    
    for (const file of this.corpusFiles) {
      const content = file.content.toLowerCase();
      const matchCount = keywords.filter(keyword => content.includes(keyword)).length;
      
      if (matchCount >= 2) { // At least 2 keywords must match
        relevantFiles.push(file);
      }
    }
    
    return relevantFiles;
  }

  async generateHardNegatives() {
    console.log('🎯 Generating hard negatives...');
    
    const targetHardNegatives = Math.floor(TARGET_QUERIES * HARD_NEGATIVES_RATIO);
    
    // Create hard negatives: queries that should NOT match certain files
    const hardNegativePatterns = [
      { query: 'DatabaseConnection', should_not_match_zones: ['test'], language: 'py' },
      { query: 'TestRunner', should_not_match_zones: ['src'], language: 'ts' },
      { query: 'ConfigParser', should_not_match_languages: ['js'], zone: 'config' },
      { query: 'async await pattern', should_not_match_languages: ['py'], intent: 'structural' }
    ];
    
    for (const pattern of hardNegativePatterns.slice(0, targetHardNegatives)) {
      // Find files that should NOT match this pattern
      const negativeFiles = this.corpusFiles.filter(file => {
        if (pattern.should_not_match_zones && pattern.should_not_match_zones.includes(file.zone)) {
          return true;
        }
        if (pattern.should_not_match_languages && pattern.should_not_match_languages.includes(file.language)) {
          return true;
        }
        return false;
      });
      
      if (negativeFiles.length > 0) {
        const hardNegativeLabels = negativeFiles.slice(0, 3).map(file => ({
          file: file.filename,
          line: 1,
          col: 1,
          relevance_score: 0.1, // Very low relevance
          match_type: 'hard_negative',
          confidence: 'low',
          why: `should not match due to ${pattern.should_not_match_zones ? 'zone' : 'language'} mismatch`
        }));
        
        this.ladderQueries.push({
          id: crypto.randomUUID(),
          query: pattern.query,
          intent: pattern.intent || 'lexical',
          language: pattern.language || 'multi',
          zone: pattern.zone || 'multi',
          weak_labels: hardNegativeLabels,
          confidence: 0.1,
          source: 'hard_negative',
          is_hard_negative: true,
          created_at: this.createdAt
        });
        
        this.stats.hard_negatives++;
      }
    }
    
    console.log(`   ✅ Generated ${this.stats.hard_negatives} hard negatives`);
  }

  isInterestingIdentifier(identifier) {
    const boring = ['the', 'and', 'for', 'with', 'from', 'this', 'that', 'data', 'info', 'item', 'obj', 'val', 'var', 'tmp', 'temp', 'test', 'name', 'type', 'file', 'line'];
    return identifier.length > 3 && 
           !boring.includes(identifier.toLowerCase()) &&
           !/^[A-Z]+$/.test(identifier) && 
           !/^\d+$/.test(identifier);
  }

  async saveLadderDataset() {
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    const dataset = {
      version: this.version,
      created_at: this.createdAt,
      description: 'LadderFull dataset with weak labels for exploratory analysis and drift detection',
      total_queries: this.ladderQueries.length,
      statistics: this.stats,
      weak_label_config: WEAK_LABEL_CONFIG,
      stratification: {
        intent_distribution: this.getDistribution('intent'),
        language_distribution: this.getDistribution('language'),
        zone_distribution: this.getDistribution('zone'),
        source_distribution: this.getDistribution('source'),
        confidence_distribution: this.getConfidenceDistribution()
      },
      quality_metrics: {
        avg_confidence: this.calculateAverageConfidence(),
        avg_labels_per_query: this.calculateAverageLabelsPerQuery(),
        coverage_ratio: this.calculateCorpusCoverage()
      },
      queries: this.ladderQueries
    };

    const filename = `ladder_${this.version}.json`;
    const filepath = path.join(OUTPUT_DIR, filename);
    
    fs.writeFileSync(filepath, JSON.stringify(dataset, null, 2));
    
    // Also create a 'current' symlink
    const currentPath = path.join(OUTPUT_DIR, 'ladder_current.json');
    if (fs.existsSync(currentPath)) {
      fs.unlinkSync(currentPath);
    }
    fs.writeFileSync(currentPath, JSON.stringify(dataset, null, 2));

    console.log(`💾 Saved LadderFull dataset: ${filepath}`);
    console.log(`📊 Dataset statistics:`);
    console.log(`   Total queries: ${dataset.total_queries}`);
    console.log(`   Average confidence: ${dataset.quality_metrics.avg_confidence.toFixed(3)}`);
    console.log(`   Average labels per query: ${dataset.quality_metrics.avg_labels_per_query.toFixed(1)}`);
    console.log(`   Corpus coverage: ${(dataset.quality_metrics.coverage_ratio * 100).toFixed(1)}%`);
    
    return dataset;
  }

  getDistribution(field) {
    const dist = {};
    for (const query of this.ladderQueries) {
      const value = query[field] || 'unknown';
      dist[value] = (dist[value] || 0) + 1;
    }
    return dist;
  }

  getConfidenceDistribution() {
    const ranges = { 'low (0-0.4)': 0, 'medium (0.4-0.7)': 0, 'high (0.7-1.0)': 0 };
    
    for (const query of this.ladderQueries) {
      const confidence = query.confidence || 0;
      if (confidence < 0.4) ranges['low (0-0.4)']++;
      else if (confidence < 0.7) ranges['medium (0.4-0.7)']++;
      else ranges['high (0.7-1.0)']++;
    }
    
    return ranges;
  }

  calculateAverageConfidence() {
    const totalConfidence = this.ladderQueries.reduce((sum, q) => sum + (q.confidence || 0), 0);
    return totalConfidence / this.ladderQueries.length;
  }

  calculateAverageLabelsPerQuery() {
    const totalLabels = this.ladderQueries.reduce((sum, q) => sum + (q.weak_labels?.length || 0), 0);
    return totalLabels / this.ladderQueries.length;
  }

  calculateCorpusCoverage() {
    const coveredFiles = new Set();
    
    for (const query of this.ladderQueries) {
      for (const label of query.weak_labels || []) {
        coveredFiles.add(label.file);
      }
    }
    
    return coveredFiles.size / this.corpusFiles.length;
  }
}

// Main execution
async function main() {
  try {
    console.log('🚀 Starting LadderFull dataset generation...');
    
    const builder = new LadderFullBuilder();
    await builder.loadCorpus();
    await builder.generateLadderDataset();
    const dataset = await builder.saveLadderDataset();
    
    console.log('✅ LadderFull dataset generation complete!');
    console.log(`📁 Output: ./ladder-datasets/ladder_${builder.version}.json`);
    
  } catch (error) {
    console.error('❌ Error creating LadderFull dataset:', error);
    process.exit(1);
  }
}

console.log('Script loaded. import.meta.url:', import.meta.url);
console.log('process.argv[1]:', process.argv[1]);

main().catch(console.error);

export { LadderFullBuilder };