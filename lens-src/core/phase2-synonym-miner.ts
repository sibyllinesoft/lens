/**
 * Phase 2 Synonym Miner - PMI-based synonym table generation
 * Mines synonyms from subtokens and docstrings using Point-wise Mutual Information
 */

import { LensTracer } from '../telemetry/tracer.js';
import { promises as fs } from 'fs';
import path from 'path';

interface PMIEntry {
  head_term: string;
  synonym: string;
  pmi_score: number;
  frequency: number;
  contexts: string[];
}

interface SynonymTable {
  version: string;
  generated_at: string;
  parameters: {
    tau_pmi: number;
    min_freq: number;
    k_synonyms: number;
  };
  entries: PMIEntry[];
}

export class Phase2SynonymMiner {
  private subtokenCounts = new Map<string, number>();
  private cooccurrenceCounts = new Map<string, Map<string, number>>();
  private totalPairs = 0;
  private docstringCache = new Map<string, string[]>();

  constructor(
    private readonly indexRoot: string,
    private readonly outputDir: string = './synonyms'
  ) {}

  /**
   * Mine synonyms using PMI from subtokens and docstrings
   */
  async mineSynonyms(params: {
    tau_pmi?: number;
    min_freq?: number;
    k_synonyms?: number;
  } = {}): Promise<SynonymTable> {
    const span = LensTracer.createChildSpan('mine_synonyms');
    const { tau_pmi = 3.0, min_freq = 20, k_synonyms = 8 } = params;

    try {
      console.log('🔍 Starting PMI-based synonym mining...');
      
      // Step 1: Extract subtokens and docstrings from indexed content
      await this.extractTokensAndDocstrings();
      
      // Step 2: Calculate PMI scores
      const pmiEntries = this.calculatePMIScores(tau_pmi, min_freq);
      
      // Step 3: Filter top-K synonyms per head term
      const filteredEntries = this.filterTopKSynonyms(pmiEntries, k_synonyms);
      
      // Step 4: Generate synonym table
      const synonymTable: SynonymTable = {
        version: 'pmi_subtokens_docstrings_v1',
        generated_at: new Date().toISOString(),
        parameters: { tau_pmi, min_freq, k_synonyms },
        entries: filteredEntries,
      };

      // Step 5: Save synonym table
      await this.saveSynonymTable(synonymTable);

      console.log(`✅ Generated ${filteredEntries.length} synonym entries`);
      
      span.setAttributes({
        success: true,
        total_entries: filteredEntries.length,
        tau_pmi,
        min_freq,
        k_synonyms,
      });

      return synonymTable;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Synonym mining failed: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Extract subtokens and docstrings from indexed content
   */
  private async extractTokensAndDocstrings(): Promise<void> {
    const span = LensTracer.createChildSpan('extract_tokens_docstrings');

    try {
      const indexedDir = path.join(this.indexRoot, 'indexed-content');
      const files = await fs.readdir(indexedDir);
      
      let processedFiles = 0;
      for (const file of files) {
        if (file.endsWith('.py') || file.endsWith('.ts') || file.endsWith('.js')) {
          const filePath = path.join(indexedDir, file);
          await this.processFile(filePath);
          processedFiles++;
        }
      }

      console.log(`📊 Processed ${processedFiles} files for token extraction`);
      console.log(`📊 Found ${this.subtokenCounts.size} unique subtokens`);

      span.setAttributes({
        success: true,
        processed_files: processedFiles,
        unique_subtokens: this.subtokenCounts.size,
      });

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Process a single file to extract subtokens and docstrings
   */
  private async processFile(filePath: string): Promise<void> {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n');
      
      // Extract docstrings for context
      const docstrings = this.extractDocstrings(content, path.extname(filePath));
      this.docstringCache.set(filePath, docstrings);
      
      for (let lineNum = 0; lineNum < lines.length; lineNum++) {
        const line = lines[lineNum];
        const tokens = this.tokenizeLine(line);
        
        // Process subtokens from each token
        for (const token of tokens) {
          const subtokens = this.extractSubtokens(token);
          
          // Update subtoken counts
          for (const subtoken of subtokens) {
            this.subtokenCounts.set(subtoken, (this.subtokenCounts.get(subtoken) || 0) + 1);
          }
          
          // Update co-occurrence counts (within same line/context)
          this.updateCooccurrence(subtokens);
          
          // Include docstring context if available
          for (const docstring of docstrings) {
            const docTokens = this.extractSubtokens(docstring);
            this.updateCooccurrence([...subtokens, ...docTokens]);
          }
        }
      }
      
    } catch (error) {
      console.warn(`Failed to process file ${filePath}:`, error);
    }
  }

  /**
   * Extract docstrings from source code
   */
  private extractDocstrings(content: string, fileExt: string): string[] {
    const docstrings: string[] = [];
    
    if (fileExt === '.py') {
      // Python docstring patterns
      const pythonDocRegex = /(?:"""([^"]*?)"""|'''([^']*?)''')/gs;
      let match;
      while ((match = pythonDocRegex.exec(content)) !== null) {
        const docstring = (match[1] || match[2]).trim();
        if (docstring.length > 10) {
          docstrings.push(docstring);
        }
      }
    } else if (fileExt === '.ts' || fileExt === '.js') {
      // TypeScript/JavaScript JSDoc patterns
      const jsDocRegex = /\/\*\*([^*]|\*(?!\/))*\*\//gs;
      let match;
      while ((match = jsDocRegex.exec(content)) !== null) {
        const docstring = match[0]
          .replace(/\/\*\*|\*\//g, '')
          .replace(/^\s*\*\s?/gm, '')
          .trim();
        if (docstring.length > 10) {
          docstrings.push(docstring);
        }
      }
    }
    
    return docstrings;
  }

  /**
   * Tokenize a line into meaningful tokens
   */
  private tokenizeLine(line: string): string[] {
    // Split on common delimiters while preserving identifier structure
    const tokens = line
      .replace(/[^\w\s._-]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 2 && /[a-zA-Z]/.test(token));
    
    return tokens;
  }

  /**
   * Extract subtokens from camelCase/snake_case identifiers
   */
  private extractSubtokens(token: string): string[] {
    const subtokens: string[] = [];
    
    // Handle camelCase
    const camelCaseTokens = token.split(/(?=[A-Z])/).filter(t => t.length > 0);
    subtokens.push(...camelCaseTokens.map(t => t.toLowerCase()));
    
    // Handle snake_case
    const snakeCaseTokens = token.split('_').filter(t => t.length > 0);
    subtokens.push(...snakeCaseTokens.map(t => t.toLowerCase()));
    
    // Handle dot notation
    const dotTokens = token.split('.').filter(t => t.length > 0);
    subtokens.push(...dotTokens.map(t => t.toLowerCase()));
    
    // Remove duplicates and filter short tokens
    const uniqueSubtokens = Array.from(new Set(subtokens)).filter(t => t.length > 2);
    
    return uniqueSubtokens;
  }

  /**
   * Update co-occurrence counts for subtoken pairs
   */
  private updateCooccurrence(subtokens: string[]): void {
    for (let i = 0; i < subtokens.length; i++) {
      for (let j = i + 1; j < subtokens.length; j++) {
        const token1 = subtokens[i];
        const token2 = subtokens[j];
        
        // Ensure consistent ordering for co-occurrence counting
        const [first, second] = token1 < token2 ? [token1, token2] : [token2, token1];
        
        if (!this.cooccurrenceCounts.has(first)) {
          this.cooccurrenceCounts.set(first, new Map());
        }
        
        const secondMap = this.cooccurrenceCounts.get(first)!;
        secondMap.set(second, (secondMap.get(second) || 0) + 1);
        this.totalPairs++;
      }
    }
  }

  /**
   * Calculate PMI scores for all token pairs
   */
  private calculatePMIScores(tauPMI: number, minFreq: number): PMIEntry[] {
    const entries: PMIEntry[] = [];
    const totalTokens = Array.from(this.subtokenCounts.values()).reduce((sum, count) => sum + count, 0);
    
    for (const [token1, cooccurrences] of this.cooccurrenceCounts) {
      const freq1 = this.subtokenCounts.get(token1) || 0;
      if (freq1 < minFreq) continue;
      
      for (const [token2, cooccurrenceCount] of cooccurrences) {
        const freq2 = this.subtokenCounts.get(token2) || 0;
        if (freq2 < minFreq) continue;
        
        // Calculate PMI: log(P(x,y) / (P(x) * P(y)))
        const pXY = cooccurrenceCount / this.totalPairs;
        const pX = freq1 / totalTokens;
        const pY = freq2 / totalTokens;
        
        const pmi = Math.log2(pXY / (pX * pY));
        
        if (pmi >= tauPMI) {
          entries.push({
            head_term: token1,
            synonym: token2,
            pmi_score: pmi,
            frequency: cooccurrenceCount,
            contexts: this.getContexts(token1, token2),
          });
          
          // Add reverse relationship
          entries.push({
            head_term: token2,
            synonym: token1,
            pmi_score: pmi,
            frequency: cooccurrenceCount,
            contexts: this.getContexts(token2, token1),
          });
        }
      }
    }
    
    return entries;
  }

  /**
   * Get context examples for a token pair
   */
  private getContexts(token1: string, token2: string): string[] {
    const contexts: string[] = [];
    
    // Sample from docstring cache
    for (const [, docstrings] of this.docstringCache) {
      for (const docstring of docstrings) {
        const lowerDoc = docstring.toLowerCase();
        if (lowerDoc.includes(token1) && lowerDoc.includes(token2)) {
          // Extract relevant sentence
          const sentences = docstring.split(/[.!?]/).filter(s => s.trim().length > 0);
          const relevantSentence = sentences.find(s => 
            s.toLowerCase().includes(token1) && s.toLowerCase().includes(token2)
          );
          if (relevantSentence && contexts.length < 3) {
            contexts.push(relevantSentence.trim());
          }
        }
      }
    }
    
    return contexts;
  }

  /**
   * Filter top-K synonyms per head term
   */
  private filterTopKSynonyms(entries: PMIEntry[], k: number): PMIEntry[] {
    const groupedEntries = new Map<string, PMIEntry[]>();
    
    // Group by head term
    for (const entry of entries) {
      if (!groupedEntries.has(entry.head_term)) {
        groupedEntries.set(entry.head_term, []);
      }
      groupedEntries.get(entry.head_term)!.push(entry);
    }
    
    const filtered: PMIEntry[] = [];
    
    // Keep top-K for each head term
    for (const [headTerm, synonyms] of groupedEntries) {
      const topK = synonyms
        .sort((a, b) => b.pmi_score - a.pmi_score)
        .slice(0, k);
      filtered.push(...topK);
    }
    
    return filtered;
  }

  /**
   * Save synonym table to disk
   */
  private async saveSynonymTable(table: SynonymTable): Promise<void> {
    const span = LensTracer.createChildSpan('save_synonym_table');

    try {
      // Ensure output directory exists
      await fs.mkdir(this.outputDir, { recursive: true });
      
      // Save as TSV format
      const tsvPath = path.join(this.outputDir, 'synonyms_v1.tsv');
      const tsvContent = [
        'head_term\tsynonym\tpmi_score\tfrequency\tcontexts',
        ...table.entries.map(entry => 
          `${entry.head_term}\t${entry.synonym}\t${entry.pmi_score.toFixed(4)}\t${entry.frequency}\t"${entry.contexts.join('; ')}"`
        )
      ].join('\n');
      
      await fs.writeFile(tsvPath, tsvContent);
      
      // Save as JSON for programmatic access
      const jsonPath = path.join(this.outputDir, 'synonyms_v1.json');
      await fs.writeFile(jsonPath, JSON.stringify(table, null, 2));

      console.log(`💾 Saved synonym table to ${tsvPath} and ${jsonPath}`);

      span.setAttributes({
        success: true,
        tsv_path: tsvPath,
        json_path: jsonPath,
        entries_count: table.entries.length,
      });

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Load existing synonym table
   */
  async loadSynonymTable(version: string = 'pmi_subtokens_docstrings_v1'): Promise<SynonymTable | null> {
    const span = LensTracer.createChildSpan('load_synonym_table');

    try {
      const jsonPath = path.join(this.outputDir, `synonyms_v1.json`);
      const content = await fs.readFile(jsonPath, 'utf-8');
      const table: SynonymTable = JSON.parse(content);
      
      if (table.version === version) {
        span.setAttributes({
          success: true,
          loaded_version: table.version,
          entries_count: table.entries.length,
        });
        return table;
      }
      
      return null;

    } catch (error) {
      span.setAttributes({ success: false, error: 'File not found or invalid' });
      return null;
    } finally {
      span.end();
    }
  }

  /**
   * Get synonym lookup map for fast access
   */
  createSynonymLookup(table: SynonymTable): Map<string, string[]> {
    const lookup = new Map<string, string[]>();
    
    for (const entry of table.entries) {
      if (!lookup.has(entry.head_term)) {
        lookup.set(entry.head_term, []);
      }
      lookup.get(entry.head_term)!.push(entry.synonym);
    }
    
    return lookup;
  }
}