/**
 * Create golden dataset from storyviz corpus
 * Generates test queries based on actual storyviz code patterns
 */

import { promises as fs } from 'fs';
import path from 'path';
import crypto from 'crypto';

class StoryVizGoldenGenerator {
  constructor() {
    this.indexedDir = path.resolve('./indexed-content');
    this.goldenFile = path.resolve('./validation-data/golden-storyviz.json');
    this.goldenItems = [];
  }

  async generateGoldenDataset() {
    console.log('🌟 Generating golden dataset from storyviz corpus...');

    await fs.mkdir(path.dirname(this.goldenFile), { recursive: true });

    // Read the indexing summary to understand our corpus
    const summaryPath = path.join(this.indexedDir, 'indexing-summary.json');
    const summary = JSON.parse(await fs.readFile(summaryPath, 'utf-8'));
    
    console.log(`📊 Corpus summary: ${summary.total_files} files, ${summary.total_lines.toLocaleString()} lines`);

    // Generate golden items by language
    await this.generatePythonGoldenItems();
    await this.generateTypeScriptGoldenItems();
    await this.generateStructuralQueries();
    await this.generateSemanticQueries();

    // Save the golden dataset
    const goldenDataset = {
      metadata: {
        generated_at: new Date().toISOString(),
        corpus_summary: summary,
        total_queries: this.goldenItems.length,
        query_types: this.countQueryTypes()
      },
      golden_items: this.goldenItems
    };

    await fs.writeFile(this.goldenFile, JSON.stringify(goldenDataset, null, 2));
    console.log(`✅ Generated ${this.goldenItems.length} golden items`);
    console.log(`📁 Golden dataset saved to: ${this.goldenFile}`);
    
    return goldenDataset;
  }

  async generatePythonGoldenItems() {
    console.log('🐍 Generating Python golden items...');
    
    const pythonFiles = await this.findFilesByPattern('*.py');
    
    // Sample some Python files to extract patterns
    for (const file of pythonFiles.slice(0, 20)) {
      try {
        const content = await fs.readFile(path.join(this.indexedDir, file), 'utf-8');
        const lines = content.split('\n');
        
        // Find class definitions
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i];
          
          // Class definitions
          const classMatch = line.match(/^class\s+(\w+)/);
          if (classMatch) {
            const className = classMatch[1];
            this.addGoldenItem({
              query: `class ${className}`,
              expected_file: file,
              expected_line: i + 1,
              query_class: 'exact_match',
              language: 'python',
              match_type: 'class_definition'
            });
            
            // Also add partial class name queries
            if (className.length > 4) {
              this.addGoldenItem({
                query: className.toLowerCase(),
                expected_file: file,
                expected_line: i + 1,
                query_class: 'identifier',
                language: 'python',
                match_type: 'class_name'
              });
            }
          }
          
          // Function definitions
          const funcMatch = line.match(/^\s*def\s+(\w+)/);
          if (funcMatch) {
            const funcName = funcMatch[1];
            // Skip dunder methods and very short names
            if (!funcName.startsWith('_') && funcName.length > 3) {
              this.addGoldenItem({
                query: `def ${funcName}`,
                expected_file: file,
                expected_line: i + 1,
                query_class: 'exact_match',
                language: 'python',
                match_type: 'function_definition'
              });
            }
          }
          
          // Import statements
          const importMatch = line.match(/^from\s+(\w+)/);
          if (importMatch) {
            const moduleName = importMatch[1];
            this.addGoldenItem({
              query: `from ${moduleName}`,
              expected_file: file,
              expected_line: i + 1,
              query_class: 'exact_match',
              language: 'python',
              match_type: 'import_statement'
            });
          }
        }
      } catch (error) {
        console.warn(`⚠️ Failed to process ${file}:`, error.message);
      }
    }
  }

  async generateTypeScriptGoldenItems() {
    console.log('📜 Generating TypeScript golden items...');
    
    const tsFiles = await this.findFilesByPattern(['*.ts', '*.tsx']);
    
    for (const file of tsFiles.slice(0, 15)) {
      try {
        const content = await fs.readFile(path.join(this.indexedDir, file), 'utf-8');
        const lines = content.split('\n');
        
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          
          // Interface definitions
          const interfaceMatch = line.match(/^export interface\s+(\w+)/);
          if (interfaceMatch) {
            const interfaceName = interfaceMatch[1];
            this.addGoldenItem({
              query: `interface ${interfaceName}`,
              expected_file: file,
              expected_line: i + 1,
              query_class: 'exact_match',
              language: 'typescript',
              match_type: 'interface_definition'
            });
          }
          
          // Class definitions
          const classMatch = line.match(/^export class\s+(\w+)/) || line.match(/^class\s+(\w+)/);
          if (classMatch) {
            const className = classMatch[1];
            this.addGoldenItem({
              query: `class ${className}`,
              expected_file: file,
              expected_line: i + 1,
              query_class: 'exact_match',
              language: 'typescript',
              match_type: 'class_definition'
            });
          }
          
          // Function definitions
          const funcMatch = line.match(/^export function\s+(\w+)/) || 
                           line.match(/^function\s+(\w+)/) ||
                           line.match(/^export const\s+(\w+)\s*=.*=>/);
          if (funcMatch) {
            const funcName = funcMatch[1];
            this.addGoldenItem({
              query: funcName,
              expected_file: file,
              expected_line: i + 1,
              query_class: 'identifier',
              language: 'typescript',
              match_type: 'function_definition'
            });
          }
          
          // Type definitions
          const typeMatch = line.match(/^export type\s+(\w+)/);
          if (typeMatch) {
            const typeName = typeMatch[1];
            this.addGoldenItem({
              query: `type ${typeName}`,
              expected_file: file,
              expected_line: i + 1,
              query_class: 'exact_match',
              language: 'typescript',
              match_type: 'type_definition'
            });
          }
        }
      } catch (error) {
        console.warn(`⚠️ Failed to process ${file}:`, error.message);
      }
    }
  }

  async generateStructuralQueries() {
    console.log('🏗️ Generating structural search queries...');

    // Common structural patterns
    const structuralPatterns = [
      {
        query: 'class * extends',
        description: 'Class inheritance patterns',
        query_class: 'structural'
      },
      {
        query: 'function * async',
        description: 'Async function patterns',
        query_class: 'structural'
      },
      {
        query: 'import * from',
        description: 'Import statement patterns',
        query_class: 'structural'
      },
      {
        query: '*.map(*)',
        description: 'Array map operations',
        query_class: 'structural'
      },
      {
        query: 'async def *',
        description: 'Python async functions',
        query_class: 'structural'
      }
    ];

    // For structural queries, we'll create synthetic expected results
    // In a real scenario, these would be validated against actual matches
    for (const pattern of structuralPatterns) {
      this.addGoldenItem({
        query: pattern.query,
        expected_file: 'synthetic',
        expected_line: 1,
        query_class: pattern.query_class,
        language: 'multi',
        match_type: 'structural_pattern',
        description: pattern.description,
        validation_type: 'pattern_match' // Special marker for structural queries
      });
    }
  }

  async generateSemanticQueries() {
    console.log('🧠 Generating semantic search queries...');

    // Common semantic queries based on storyviz domain
    const semanticQueries = [
      {
        query: 'cache implementation',
        expected_files: ['*cache*.py', '*cache*.ts'],
        description: 'Find caching related code'
      },
      {
        query: 'database connection',
        expected_files: ['*db*.py', '*model*.py'],
        description: 'Find database related code'
      },
      {
        query: 'error handling',
        expected_files: ['*exception*.py', '*error*.py'],
        description: 'Find error handling patterns'
      },
      {
        query: 'api endpoint',
        expected_files: ['*api*.py', '*route*.py', '*endpoint*.ts'],
        description: 'Find API endpoint definitions'
      },
      {
        query: 'text processing',
        expected_files: ['*nlp*.py', '*text*.py', '*process*.py'],
        description: 'Find text processing logic'
      }
    ];

    for (const semanticQuery of semanticQueries) {
      this.addGoldenItem({
        query: semanticQuery.query,
        expected_file: 'semantic_match',
        expected_line: 1,
        query_class: 'semantic',
        language: 'multi',
        match_type: 'semantic_search',
        description: semanticQuery.description,
        validation_type: 'semantic_relevance',
        expected_patterns: semanticQuery.expected_files
      });
    }
  }

  addGoldenItem(item) {
    const id = crypto.randomUUID();
    
    const goldenItem = {
      id,
      query: item.query,
      language: item.language,
      query_class: item.query_class,
      expected_results: [{
        file: item.expected_file,
        line: item.expected_line,
        col: item.expected_col || 1,
        relevance_score: 1.0,
        match_type: item.match_type
      }],
      metadata: {
        description: item.description,
        validation_type: item.validation_type || 'exact_match',
        expected_patterns: item.expected_patterns
      },
      snapshot_sha: 'storyviz-corpus-v1'
    };

    this.goldenItems.push(goldenItem);
  }

  async findFilesByPattern(patterns) {
    const files = await fs.readdir(this.indexedDir);
    const matchingFiles = [];
    
    const patternsArray = Array.isArray(patterns) ? patterns : [patterns];
    
    for (const file of files) {
      for (const pattern of patternsArray) {
        const regex = new RegExp(pattern.replace('*', '.*'));
        if (regex.test(file)) {
          matchingFiles.push(file);
          break; // Don't add the same file multiple times
        }
      }
    }
    
    return matchingFiles;
  }

  countQueryTypes() {
    const types = {};
    for (const item of this.goldenItems) {
      const type = item.query_class;
      types[type] = (types[type] || 0) + 1;
    }
    return types;
  }
}

// Main execution
async function main() {
  const generator = new StoryVizGoldenGenerator();
  
  try {
    const dataset = await generator.generateGoldenDataset();
    
    console.log('\n📈 Golden Dataset Summary:');
    console.log(`  Total queries: ${dataset.golden_items.length}`);
    console.log(`  Query types:`, dataset.metadata.query_types);
    console.log('\n✅ Golden dataset generation completed successfully!');
    
  } catch (error) {
    console.error('❌ Golden dataset generation failed:', error.message);
    process.exit(1);
  }
}

// Run if this is the main module
main();