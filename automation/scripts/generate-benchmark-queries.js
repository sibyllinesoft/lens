#!/usr/bin/env node

/**
 * Generate realistic benchmark queries from the existing corpus
 * 
 * Creates query datasets for the benchmark protocol:
 * - CoIR (Code Information Retrieval) queries
 * - SWE-bench Verified queries  
 * - CSN (CodeSearchNet) queries
 * - CoSQA (Code Search Question Answering) queries
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

    // Semantic queries - find functions/classes by purpose
    for (const file of corpusFiles.slice(0, 50)) {
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

        const functions = this.extractPythonFunctions(file.content);
        for (const funcName of functions.slice(0, 2)) {
          queries.push({
            id: `coir_${queryId++}`,
            query: `function that ${funcName.replace('_', ' ')}`,
            suite: 'coir',
            intent: 'semantic', 
            language: file.language,
            expected_file: file.path,
            difficulty: 'easy'
          });
        }
      }
    }

    // Identifier queries - exact symbol matches
    for (const file of corpusFiles.slice(0, 30)) {
      const identifiers = this.extractIdentifiers(file.content, file.language);
      for (const identifier of identifiers.slice(0, 3)) {
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

    // Structural queries - code patterns
    const structuralQueries = [
      'class * extends *',
      'function * async',  
      'import * from *',
      'def *(self',
      'for * in *:',
      'if __name__ == "__main__":'
    ];

    for (const pattern of structuralQueries) {
      queries.push({
        id: `coir_${queryId++}`,
        query: pattern,
        suite: 'coir',
        intent: 'structural',
        language: pattern.includes('def') || pattern.includes('__name__') ? 'python' : 'unknown',
        difficulty: 'medium'
      });
    }

    return queries.slice(0, 200); // Limit to 200 queries
  }

  async generateSweQueries(corpusFiles) {
    console.log('  🔧 Generating SWE-bench Verified queries...');
    
    const queries = [];
    let queryId = 1;

    // Bug finding queries
    const bugPatterns = [
      'fix error in function',
      'resolve bug in class', 
      'handle exception properly',
      'memory leak fix',
      'null pointer check',
      'validate input parameters'
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

    // Test-related queries
    for (const file of corpusFiles.filter(f => f.path.includes('test')).slice(0, 30)) {
      const testFunctions = this.extractPythonFunctions(file.content).filter(f => f.startsWith('test_'));
      for (const testFunc of testFunctions.slice(0, 2)) {
        queries.push({
          id: `swe_${queryId++}`,
          query: `test case for ${testFunc.replace('test_', '').replace('_', ' ')}`,
          suite: 'swe_verified',
          intent: 'semantic',
          language: file.language,
          expected_file: file.path,
          difficulty: 'medium'
        });
      }
    }

    // Configuration and setup queries
    const configQueries = [
      'configuration setup',
      'environment variables',
      'initialization parameters',
      'default settings',
      'logging configuration'
    ];

    for (const configQuery of configQueries) {
      queries.push({
        id: `swe_${queryId++}`,
        query: configQuery,
        suite: 'swe_verified',
        intent: 'semantic',
        language: 'unknown',
        difficulty: 'medium'
      });
    }

    return queries.slice(0, 150);
  }

  async generateCsnQueries(corpusFiles) {
    console.log('  🔍 Generating CodeSearchNet queries...');
    
    const queries = [];
    let queryId = 1;

    // Natural language descriptions of code functionality  
    for (const file of corpusFiles.slice(0, 40)) {
      const docstrings = this.extractDocstrings(file.content, file.language);
      for (const docstring of docstrings.slice(0, 2)) {
        const query = this.convertDocstringToQuery(docstring);
        if (query) {
          queries.push({
            id: `csn_${queryId++}`,
            query,
            suite: 'csn',
            intent: 'semantic',
            language: file.language,
            expected_file: file.path,
            difficulty: 'medium'
          });
        }
      }
    }

    // Common programming tasks
    const taskQueries = [
      'parse JSON data',
      'validate user input', 
      'connect to database',
      'send HTTP request',
      'read file contents',
      'format date string',
      'calculate hash value',
      'sort list items',
      'handle errors gracefully',
      'log debug information'
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

    return queries.slice(0, 180);
  }

  async generateCosqaQueries(corpusFiles) {
    console.log('  💬 Generating CoSQA queries...');
    
    const queries = [];
    let queryId = 1;

    // Question-answer style queries
    const qaPatterns = [
      'How to implement authentication?',
      'What is the best way to handle errors?',
      'How do you parse XML data?',
      'What library is used for HTTP requests?',
      'How to optimize database queries?',
      'What is the purpose of this class?',
      'How does the caching mechanism work?',
      'What are the configuration options?'
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

    // API usage queries
    for (const file of corpusFiles.slice(0, 30)) {
      const imports = this.extractImports(file.content, file.language);
      for (const importName of imports.slice(0, 2)) {
        queries.push({
          id: `cosqa_${queryId++}`,
          query: `How to use ${importName}?`,
          suite: 'cosqa',
          intent: 'semantic',
          language: file.language,
          expected_file: file.path,
          difficulty: 'easy'
        });
      }
    }

    return queries.slice(0, 120);
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

  extractPythonFunctions(content) {
    const funcRegex = /def\s+(\w+)/g;
    const functions = [];
    let match;
    while ((match = funcRegex.exec(content)) !== null) {
      functions.push(match[1]);
    }
    return functions;
  }

  extractIdentifiers(content, language) {
    const identifiers = new Set();
    
    if (language === 'python') {
      const pythonIdRegex = /\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b/g;
      let match;
      while ((match = pythonIdRegex.exec(content)) !== null) {
        const identifier = match[0];
        if (!this.isPythonKeyword(identifier) && identifier.length > 3 && identifier.length < 20) {
          identifiers.add(identifier);
        }
      }
    }
    
    return Array.from(identifiers).slice(0, 10);
  }

  isPythonKeyword(word) {
    const keywords = ['def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'return', 'True', 'False', 'None'];
    return keywords.includes(word);
  }

  extractDocstrings(content, language) {
    if (language !== 'python') return [];
    
    const docstrings = [];
    const tripleQuoteRegex = /"""([\s\S]*?)"""/g;
    let match;
    while ((match = tripleQuoteRegex.exec(content)) !== null) {
      const docstring = match[1].trim();
      if (docstring.length > 20 && docstring.length < 200) {
        docstrings.push(docstring);
      }
    }
    return docstrings;
  }

  convertDocstringToQuery(docstring) {
    // Extract the first sentence as a query
    const sentences = docstring.split('.').map(s => s.trim()).filter(s => s.length > 10);
    if (sentences.length > 0) {
      let query = sentences[0];
      if (query.length > 100) {
        query = query.substring(0, 100) + '...';
      }
      return query;
    }
    return null;
  }

  extractImports(content, language) {
    if (language !== 'python') return [];
    
    const imports = [];
    const importRegex = /import\s+(\w+)|from\s+(\w+)\s+import/g;
    let match;
    while ((match = importRegex.exec(content)) !== null) {
      const importName = match[1] || match[2];
      if (importName && importName.length > 2) {
        imports.push(importName);
      }
    }
    return imports;
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
    
    console.log(`📊 Summary generated: ${summary.total_queries} total queries across ${suites.length} suites`);
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