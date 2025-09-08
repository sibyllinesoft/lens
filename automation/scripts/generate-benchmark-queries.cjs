#!/usr/bin/env node

/**
 * Generate realistic benchmark queries from the existing corpus
 */

const fs = require('fs').promises;
const path = require('path');
const { glob } = require('glob');

class BenchmarkQueryGenerator {
  constructor(corpusPath, outputPath) {
    this.corpusPath = corpusPath;
    this.outputPath = outputPath;
  }

  async generateAllSuites() {
    console.log('🔍 Generating benchmark query suites...');
    
    // Read corpus files
    const corpusFiles = await this.readCorpusFiles();
    console.log(`📁 Loaded ${corpusFiles.length} corpus files`);

    // Generate query suites
    const coirQueries = await this.generateCoirQueries(corpusFiles);
    const sweQueries = await this.generateSweQueries(corpusFiles);
    const csnQueries = await this.generateCsnQueries(corpusFiles);
    const cosqaQueries = await this.generateCosqaQueries(corpusFiles);

    // Save query suites
    await this.saveQuerySuite('coir', coirQueries);
    await this.saveQuerySuite('swe_verified', sweQueries);
    await this.saveQuerySuite('csn', csnQueries);
    await this.saveQuerySuite('cosqa', cosqaQueries);

    // Generate summary
    await this.generateSummary([
      { name: 'coir', queries: coirQueries, total_count: coirQueries.length },
      { name: 'swe_verified', queries: sweQueries, total_count: sweQueries.length },
      { name: 'csn', queries: csnQueries, total_count: csnQueries.length },
      { name: 'cosqa', queries: cosqaQueries, total_count: cosqaQueries.length }
    ]);

    console.log('✅ Benchmark query generation complete');
  }

  async readCorpusFiles() {
    const files = await glob(path.join(this.corpusPath, '**/*'), { nodir: true });
    const corpusFiles = [];

    for (const file of files.slice(0, 200)) { // Limit for performance
      try {
        const content = await fs.readFile(file, 'utf8');
        const language = this.detectLanguage(file);
        
        if (content.length > 100 && content.length < 50000) {
          corpusFiles.push({
            path: file,
            content,
            language
          });
        }
      } catch (error) {
        // Skip files that can't be read
      }
    }

    return corpusFiles;
  }

  detectLanguage(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const mapping = {
      '.py': 'python',
      '.js': 'javascript', 
      '.ts': 'typescript',
      '.java': 'java',
      '.cpp': 'cpp',
      '.c': 'c',
      '.rs': 'rust',
      '.go': 'go',
      '.rb': 'ruby',
      '.php': 'php'
    };
    return mapping[ext] || 'unknown';
  }

  async generateCoirQueries(corpusFiles) {
    console.log('  📋 Generating CoIR queries...');
    
    const queries = [];
    let queryId = 1;

    // Semantic queries
    for (const file of corpusFiles.slice(0, 30)) {
      if (file.language === 'python') {
        const classes = this.extractPythonClasses(file.content);
        for (const className of classes.slice(0, 2)) {
          queries.push({
            id: `coir_${queryId++}`,
            query: `${className} class implementation`,
            suite: 'coir',
            intent: 'semantic',
            language: file.language,
            expected_file: file.path,
            difficulty: 'medium'
          });
        }
      }
    }

    // Identifier queries
    for (const file of corpusFiles.slice(0, 20)) {
      const identifiers = this.extractIdentifiers(file.content, file.language);
      for (const identifier of identifiers.slice(0, 2)) {
        queries.push({
          id: `coir_${queryId++}`,
          query: identifier,
          suite: 'coir',
          intent: 'identifier',
          language: file.language,
          expected_file: file.path,
          difficulty: 'easy'
        });
      }
    }

    // Structural queries
    const structuralQueries = [
      'class * extends',
      'function * async',  
      'import * from',
      'def *(',
      'for * in'
    ];

    for (const pattern of structuralQueries) {
      queries.push({
        id: `coir_${queryId++}`,
        query: pattern,
        suite: 'coir',
        intent: 'structural',
        language: 'unknown',
        difficulty: 'medium'
      });
    }

    return queries.slice(0, 100);
  }

  async generateSweQueries(corpusFiles) {
    console.log('  🔧 Generating SWE-bench queries...');
    
    const queries = [];
    let queryId = 1;

    const bugPatterns = [
      'fix error handling',
      'resolve bug in class', 
      'handle exception',
      'validate input',
      'memory leak fix',
      'null pointer check'
    ];

    for (const pattern of bugPatterns) {
      queries.push({
        id: `swe_${queryId++}`,
        query: pattern,
        suite: 'swe_verified',
        intent: 'semantic',
        language: 'unknown',
        difficulty: 'hard'
      });
    }

    return queries.slice(0, 75);
  }

  async generateCsnQueries(corpusFiles) {
    console.log('  🔍 Generating CSN queries...');
    
    const queries = [];
    let queryId = 1;

    const taskQueries = [
      'parse JSON data',
      'validate input', 
      'connect database',
      'send HTTP request',
      'read file contents',
      'format date string',
      'calculate hash',
      'sort array',
      'handle errors',
      'log information'
    ];

    for (const taskQuery of taskQueries) {
      queries.push({
        id: `csn_${queryId++}`,
        query: taskQuery,
        suite: 'csn',
        intent: 'semantic',
        language: 'unknown',
        difficulty: 'easy'
      });
    }

    return queries.slice(0, 90);
  }

  async generateCosqaQueries(corpusFiles) {
    console.log('  💬 Generating CoSQA queries...');
    
    const queries = [];
    let queryId = 1;

    const qaPatterns = [
      'How to implement authentication?',
      'What is error handling?',
      'How to parse data?',
      'What library for HTTP?',
      'How to optimize queries?',
      'What is class purpose?',
      'How does caching work?',
      'What are config options?'
    ];

    for (const question of qaPatterns) {
      queries.push({
        id: `cosqa_${queryId++}`,
        query: question,
        suite: 'cosqa',
        intent: 'semantic',
        language: 'unknown',
        difficulty: 'medium'
      });
    }

    return queries.slice(0, 60);
  }

  extractPythonClasses(content) {
    const classRegex = /class\s+(\w+)/g;
    const classes = [];
    let match;
    while ((match = classRegex.exec(content)) !== null) {
      classes.push(match[1]);
    }
    return classes;
  }

  extractIdentifiers(content, language) {
    const identifiers = new Set();
    
    if (language === 'python') {
      const pythonIdRegex = /\b[a-zA-Z_][a-zA-Z0-9_]{3,15}\b/g;
      let match;
      while ((match = pythonIdRegex.exec(content)) !== null) {
        const identifier = match[0];
        if (!this.isPythonKeyword(identifier)) {
          identifiers.add(identifier);
        }
      }
    }
    
    return Array.from(identifiers).slice(0, 5);
  }

  isPythonKeyword(word) {
    const keywords = ['def', 'class', 'import', 'from', 'if', 'else', 'for', 'while', 'try', 'except', 'return', 'True', 'False', 'None'];
    return keywords.includes(word);
  }

  async saveQuerySuite(suiteName, queries) {
    await fs.mkdir(this.outputPath, { recursive: true });
    const filePath = path.join(this.outputPath, `${suiteName}_queries.json`);
    await fs.writeFile(filePath, JSON.stringify(queries, null, 2));
    console.log(`  ✅ ${suiteName}: ${queries.length} queries saved to ${filePath}`);
  }

  async generateSummary(suites) {
    const summary = {
      generated_at: new Date().toISOString(),
      total_queries: suites.reduce((sum, suite) => sum + suite.total_count, 0),
      suites: suites.map(suite => ({
        name: suite.name,
        count: suite.total_count,
        intents: this.getIntentDistribution(suite.queries),
        languages: this.getLanguageDistribution(suite.queries),
        difficulties: this.getDifficultyDistribution(suite.queries)
      }))
    };

    const summaryPath = path.join(this.outputPath, 'query_generation_summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2));
    
    console.log(`📊 Summary: ${summary.total_queries} total queries across ${suites.length} suites`);
    console.log(`📁 Saved to: ${summaryPath}`);
  }

  getIntentDistribution(queries) {
    const distribution = {};
    for (const query of queries) {
      distribution[query.intent] = (distribution[query.intent] || 0) + 1;
    }
    return distribution;
  }

  getLanguageDistribution(queries) {
    const distribution = {};
    for (const query of queries) {
      distribution[query.language] = (distribution[query.language] || 0) + 1;
    }
    return distribution;
  }

  getDifficultyDistribution(queries) {
    const distribution = {};
    for (const query of queries) {
      distribution[query.difficulty] = (distribution[query.difficulty] || 0) + 1;
    }
    return distribution;
  }
}

// Execute if run directly
if (require.main === module) {
  const generator = new BenchmarkQueryGenerator('./benchmark-corpus', './benchmark-corpus');
  generator.generateAllSuites().catch(console.error);
}

module.exports = BenchmarkQueryGenerator;