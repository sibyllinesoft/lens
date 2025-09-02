/**
 * Stage B: Symbol/Structure Span Resolution
 * Uses tree-sitter AST nodes or LSIF/ctags for precise symbol locations
 */

import * as fs from 'fs/promises';
import { SearchHit, SymbolCandidate } from './types.js';
import { 
  extractSnippet, 
  extractContext,
  validateSpanBounds
} from './normalize.js';

// Mock tree-sitter types for now (would import from actual tree-sitter)
interface TreeSitterPoint {
  row: number;      // 0-based
  column: number;   // 0-based (code points)
}

interface TreeSitterNode {
  startPosition: TreeSitterPoint;
  endPosition: TreeSitterPoint;
  text: string;
  type: string;
}

/**
 * Resolve symbol matches to precise definition/reference positions
 */
export async function resolveSymbolMatches(
  candidates: SymbolCandidate[]
): Promise<SearchHit[]> {
  const results: SearchHit[] = [];
  
  for (const candidate of candidates) {
    try {
      const hit = await resolveSymbolCandidate(candidate);
      if (hit) {
        results.push(hit);
      }
    } catch (error) {
      console.warn(`Failed to resolve symbol spans for ${candidate.file_path}:`, error);
      
      // Fallback: use upstream coordinates or default to line 1
      results.push({
        file: candidate.file_path,
        line: candidate.upstream_line || 1,
        col: candidate.upstream_col || 0,
        score: candidate.score,
        why: candidate.match_reasons as any,
        symbol_kind: candidate.symbol_kind as any,
        ast_path: candidate.ast_path,
      });
    }
  }
  
  return results;
}

/**
 * Resolve a single symbol candidate
 */
async function resolveSymbolCandidate(
  candidate: SymbolCandidate
): Promise<SearchHit | null> {
  // Try tree-sitter AST approach first
  if (candidate.ast_path) {
    const astHit = await resolveWithTreeSitter(candidate);
    if (astHit) return astHit;
  }
  
  // Fallback to LSIF/ctags approach
  return await resolveWithSymbolIndex(candidate);
}

/**
 * Resolve using tree-sitter AST information
 */
async function resolveWithTreeSitter(
  candidate: SymbolCandidate
): Promise<SearchHit | null> {
  try {
    // In a real implementation, we would:
    // 1. Parse the file with tree-sitter
    // 2. Navigate to the AST node using ast_path
    // 3. Extract the start position
    
    // Mock implementation for now
    const fileContent = await fs.readFile(candidate.file_path, 'utf-8');
    const mockNode = await parseASTPath(fileContent, candidate.ast_path!);
    
    if (!mockNode) return null;
    
    // Tree-sitter provides 0-based coordinates, convert to our format
    const line = mockNode.startPosition.row + 1; // Convert to 1-based
    const col = mockNode.startPosition.column;   // Keep 0-based
    
    // Validate bounds
    const validation = validateSpanBounds(fileContent, line, col);
    if (!validation.valid) {
      console.warn(`Invalid AST span bounds for ${candidate.file_path}:${line}:${col}: ${validation.error}`);
      return null;
    }
    
    // Extract snippet and context
    const snippet = extractSnippet(fileContent, line, col);
    const context = extractContext(fileContent, line);
    
    return {
      file: candidate.file_path,
      line,
      col,
      snippet,
      score: candidate.score,
      why: candidate.match_reasons as any,
      symbol_kind: candidate.symbol_kind as any,
      ast_path: candidate.ast_path,
      context_before: context.context_before,
      context_after: context.context_after,
    };
    
  } catch (error) {
    console.warn(`Tree-sitter resolution failed for ${candidate.file_path}:`, error);
    return null;
  }
}

/**
 * Resolve using symbol index (LSIF/ctags)
 */
async function resolveWithSymbolIndex(
  candidate: SymbolCandidate
): Promise<SearchHit | null> {
  try {
    // In a real implementation, we would:
    // 1. Query the symbol index for definitions/references
    // 2. If multiple locations, prefer definition over reference
    // 3. If multiple definitions, pick nearest to upstream seed line
    
    const fileContent = await fs.readFile(candidate.file_path, 'utf-8');
    
    // Mock symbol lookup - in reality this would query an actual index
    const symbolLocation = await lookupSymbolLocation(
      candidate.file_path,
      candidate.symbol_kind,
      candidate.upstream_line
    );
    
    if (!symbolLocation) {
      // Fallback to upstream coordinates if available
      if (candidate.upstream_line && candidate.upstream_col !== undefined) {
        const line = candidate.upstream_line;
        const col = candidate.upstream_col;
        
        const validation = validateSpanBounds(fileContent, line, col);
        if (!validation.valid) {
          console.warn(`Invalid upstream span bounds for ${candidate.file_path}:${line}:${col}: ${validation.error}`);
          return null;
        }
        
        const snippet = extractSnippet(fileContent, line, col);
        const context = extractContext(fileContent, line);
        
        return {
          file: candidate.file_path,
          line,
          col,
          snippet,
          score: candidate.score,
          why: candidate.match_reasons as any,
          symbol_kind: candidate.symbol_kind as any,
          ast_path: candidate.ast_path,
          context_before: context.context_before,
          context_after: context.context_after,
        };
      }
      
      return null;
    }
    
    // Validate bounds
    const validation = validateSpanBounds(fileContent, symbolLocation.line, symbolLocation.col);
    if (!validation.valid) {
      console.warn(`Invalid symbol span bounds for ${candidate.file_path}:${symbolLocation.line}:${symbolLocation.col}: ${validation.error}`);
      return null;
    }
    
    // Extract snippet and context
    const snippet = extractSnippet(fileContent, symbolLocation.line, symbolLocation.col);
    const context = extractContext(fileContent, symbolLocation.line);
    
    return {
      file: candidate.file_path,
      line: symbolLocation.line,
      col: symbolLocation.col,
      snippet,
      score: candidate.score,
      why: candidate.match_reasons as any,
      symbol_kind: candidate.symbol_kind as any,
      ast_path: candidate.ast_path,
      context_before: context.context_before,
      context_after: context.context_after,
    };
    
  } catch (error) {
    console.warn(`Symbol index resolution failed for ${candidate.file_path}:`, error);
    return null;
  }
}

/**
 * Mock AST path parsing (would use real tree-sitter in production)
 */
async function parseASTPath(
  fileContent: string,
  astPath: string
): Promise<TreeSitterNode | null> {
  // Mock implementation - in reality this would:
  // 1. Parse file with tree-sitter for the detected language
  // 2. Navigate AST using the provided path
  // 3. Return the target node
  
  console.warn(`Mock AST path parsing for: ${astPath}`);
  
  // Return a mock node for demonstration
  return {
    startPosition: { row: 10, column: 5 }, // Mock position
    endPosition: { row: 10, column: 15 },
    text: 'mockFunction',
    type: 'function_declaration'
  };
}

/**
 * Mock symbol location lookup (would use real LSIF/ctags in production)
 */
async function lookupSymbolLocation(
  filePath: string,
  symbolKind: string | undefined,
  seedLine: number | undefined
): Promise<{ line: number; col: number } | null> {
  // Mock implementation - in reality this would:
  // 1. Query LSIF index or ctags file
  // 2. Find all definitions/references for symbols in this file
  // 3. Prefer definitions over references
  // 4. If multiple definitions, pick nearest to seedLine
  
  console.warn(`Mock symbol lookup for ${filePath}, kind: ${symbolKind}, seed: ${seedLine}`);
  
  // Return a mock location
  return {
    line: seedLine || 5,
    col: 0
  };
}