import { promises as fs } from 'fs';
import * as path from 'path';
import { StageAAdapter, StageBAdapter, StageCAdapter } from './span_resolver/adapters';
// Simple SearchResult interface
interface SearchResult {
  file: string;
  line: number;
  col: number;
  text: string;
  context: string;
}

export interface IndexEntry {
  file: string;
  content: string;
  tokens: string[];
  lines: string[];
}

export class CodeIndexer {
  private index: Map<string, IndexEntry> = new Map();
  private stageAAdapter = new StageAAdapter();
  private stageBAdapter = new StageBAdapter();
  private stageCAdapter = new StageCAdapter();

  async indexFile(filePath: string): Promise<void> {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const tokens = this.tokenize(content);
      const lines = content.split('\n');

      const entry: IndexEntry = {
        file: filePath,
        content,
        tokens,
        lines
      };

      this.index.set(filePath, entry);
      console.log(`Indexed ${filePath} with ${tokens.length} tokens`);
    } catch (error) {
      console.error(`Failed to index ${filePath}:`, error);
    }
  }

  async indexDirectory(dirPath: string): Promise<void> {
    try {
      const entries = await fs.readdir(dirPath, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);
        
        if (entry.isDirectory()) {
          await this.indexDirectory(fullPath);
        } else if (this.shouldIndex(entry.name)) {
          await this.indexFile(fullPath);
        }
      }
    } catch (error) {
      console.error(`Failed to index directory ${dirPath}:`, error);
    }
  }

  private shouldIndex(filename: string): boolean {
    const extensions = ['.ts', '.js', '.tsx', '.jsx', '.py', '.java', '.cpp', '.c', '.h'];
    return extensions.some(ext => filename.endsWith(ext));
  }

  private tokenize(content: string): string[] {
    // Simple tokenization - split on word boundaries and filter
    return content
      .toLowerCase()
      .split(/\W+/)
      .filter(token => token.length > 2)
      .filter(token => !this.isStopWord(token));
  }

  private isStopWord(word: string): boolean {
    const stopWords = new Set([
      'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
      'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could'
    ]);
    return stopWords.has(word);
  }

  search(query: string, stage: 'A' | 'B' | 'C' = 'A'): SearchResult[] {
    const queryTokens = this.tokenize(query);
    const results: SearchResult[] = [];

    for (const [filePath, entry] of this.index) {
      const matches = this.findMatches(entry, queryTokens);
      
      for (const match of matches) {
        const resolver = this.getResolverForStage(stage, entry.content);
        
        // Find the byte position of the match
        const matchStart = entry.content.indexOf(match.text, match.byteOffset);
        const matchEnd = matchStart + match.text.length;
        
        const span = resolver.resolveSpan(matchStart, matchEnd);
        
        results.push({
          file: filePath,
          line: span.start.line,
          col: span.start.col,
          text: match.text,
          context: match.context
        });
      }
    }

    // Sort by relevance (number of matching tokens)
    return results.sort((a, b) => {
      const aMatches = queryTokens.filter(token => 
        a.text.toLowerCase().includes(token) || a.context.toLowerCase().includes(token)
      ).length;
      const bMatches = queryTokens.filter(token => 
        b.text.toLowerCase().includes(token) || b.context.toLowerCase().includes(token)
      ).length;
      return bMatches - aMatches;
    });
  }

  private getResolverForStage(stage: 'A' | 'B' | 'C', content: string) {
    switch (stage) {
      case 'A': return this.stageAAdapter.createResolver(content);
      case 'B': return this.stageBAdapter.createResolver(content);
      case 'C': return this.stageCAdapter.createResolver(content);
      default: return this.stageAAdapter.createResolver(content);
    }
  }

  private findMatches(entry: IndexEntry, queryTokens: string[]): Array<{
    text: string;
    context: string;
    byteOffset: number;
  }> {
    const matches: Array<{ text: string; context: string; byteOffset: number }> = [];
    const content = entry.content.toLowerCase();
    
    for (const token of queryTokens) {
      let index = 0;
      while ((index = content.indexOf(token, index)) !== -1) {
        // Get context around the match
        const lineStart = content.lastIndexOf('\n', index);
        const lineEnd = content.indexOf('\n', index);
        const lineContent = entry.content.substring(
          lineStart === -1 ? 0 : lineStart + 1,
          lineEnd === -1 ? entry.content.length : lineEnd
        );
        
        matches.push({
          text: token,
          context: lineContent.trim(),
          byteOffset: index
        });
        
        index += token.length;
      }
    }
    
    return matches;
  }

  getIndexStats(): { files: number; tokens: number; totalSize: number } {
    let totalTokens = 0;
    let totalSize = 0;
    
    for (const entry of this.index.values()) {
      totalTokens += entry.tokens.length;
      totalSize += entry.content.length;
    }
    
    return {
      files: this.index.size,
      tokens: totalTokens,
      totalSize
    };
  }

  clear(): void {
    this.index.clear();
  }
}

// Singleton instance
export const codeIndexer = new CodeIndexer();