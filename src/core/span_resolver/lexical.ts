/**
 * Stage A: Lexical/Fuzzy Span Resolution
 * Uses in-process scanning or ripgrep to locate exact match positions
 */

import * as fs from 'fs/promises';
import { spawn } from 'child_process';
import { SearchHit, LexicalCandidate, SpanLocation } from './types.js';
import { 
  normalizeLineEndings, 
  extractSnippet, 
  extractContext,
  validateSpanBounds,
  byteOffsetToLineCol
} from './normalize.js';

interface RipgrepMatch {
  type: 'match';
  data: {
    path: { text: string };
    lines: {
      text: string;
    };
    line_number: number;
    absolute_offset: number;
    submatches: Array<{
      start: number;
      end: number;
    }>;
  };
}

/**
 * Resolve lexical matches to precise span coordinates
 */
export async function resolveLexicalMatches(
  candidates: LexicalCandidate[],
  query: string,
  fuzzyDistance: number = 0,
  maxCandidatesPerFile: number = 3
): Promise<SearchHit[]> {
  const results: SearchHit[] = [];
  
  // Cap candidates to avoid CPU explosion (performance constraint)
  const cappedCandidates = candidates.slice(0, 200);
  
  for (const candidate of cappedCandidates) {
    try {
      const hits = await resolveFileMatches(
        candidate.file_path,
        query,
        candidate.score,
        candidate.match_reasons,
        fuzzyDistance,
        maxCandidatesPerFile
      );
      
      results.push(...hits);
    } catch (error) {
      console.warn(`Failed to resolve spans for ${candidate.file_path}:`, error);
      
      // Fallback: create file-level hit (line 1, col 0)
      results.push({
        file: candidate.file_path,
        line: 1,
        col: 0,
        score: candidate.score,
        why: candidate.match_reasons,
      });
    }
  }
  
  return results;
}

/**
 * Resolve matches within a single file
 */
async function resolveFileMatches(
  filePath: string,
  query: string,
  score: number,
  matchReasons: string[],
  fuzzyDistance: number,
  maxMatches: number
): Promise<SearchHit[]> {
  // Try ripgrep first (faster), fallback to in-process
  try {
    return await resolveWithRipgrep(filePath, query, score, matchReasons, fuzzyDistance, maxMatches);
  } catch (error) {
    console.warn(`Ripgrep failed for ${filePath}, falling back to in-process:`, error);
    return await resolveInProcess(filePath, query, score, matchReasons, fuzzyDistance, maxMatches);
  }
}

/**
 * Use ripgrep for fast span location
 */
async function resolveWithRipgrep(
  filePath: string,
  query: string,
  score: number,
  matchReasons: string[],
  fuzzyDistance: number,
  maxMatches: number
): Promise<SearchHit[]> {
  return new Promise((resolve, reject) => {
    const args = [
      '--json',           // JSON output
      '-n',               // Show line numbers
      '-H',               // Show file names
      '-b',               // Show byte offsets
      '--no-heading',     // No file headers
      '-m', maxMatches.toString(), // Max matches per file
    ];
    
    // Add fuzzy search if requested
    if (fuzzyDistance > 0) {
      // Note: ripgrep doesn't have built-in fuzzy, so we'll search for exact
      // and handle fuzzy in post-processing if needed
      console.warn(`Fuzzy distance ${fuzzyDistance} not supported with ripgrep, using exact match`);
    }
    
    args.push('--', query, filePath);
    
    const rg = spawn('rg', args);
    let stdout = '';
    let stderr = '';
    
    rg.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    rg.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    rg.on('close', async (code) => {
      if (code !== 0 && code !== 1) { // 1 means no matches found
        reject(new Error(`ripgrep failed with code ${code}: ${stderr}`));
        return;
      }
      
      try {
        const hits = await parseRipgrepOutput(stdout, filePath, score, matchReasons, query);
        resolve(hits);
      } catch (error) {
        reject(error);
      }
    });
    
    rg.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * Parse ripgrep JSON output into SearchHits
 */
async function parseRipgrepOutput(
  output: string,
  filePath: string,
  score: number,
  matchReasons: string[],
  query: string
): Promise<SearchHit[]> {
  if (!output.trim()) {
    return [];
  }
  
  const lines = output.trim().split('\n');
  const hits: SearchHit[] = [];
  
  // Read file for context extraction
  const fileContent = await fs.readFile(filePath, 'utf-8');
  
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line) as RipgrepMatch;
      
      if (parsed.type !== 'match') continue;
      
      const { data } = parsed;
      const lineNumber = data.line_number;
      
      // Extract column from first submatch
      if (data.submatches && data.submatches.length > 0) {
        const submatch = data.submatches[0];
        if (!submatch) continue;
        const col = submatch.start; // Ripgrep provides 0-based column
        
        // Validate bounds
        const validation = validateSpanBounds(fileContent, lineNumber, col);
        if (!validation.valid) {
          console.warn(`Invalid span bounds for ${filePath}:${lineNumber}:${col}: ${validation.error}`);
          continue;
        }
        
        // Extract snippet and context
        const snippet = extractSnippet(fileContent, lineNumber, col);
        const context = extractContext(fileContent, lineNumber);
        
        hits.push({
          file: filePath,
          line: lineNumber,
          col,
          snippet,
          score,
          why: matchReasons as any,
          byte_offset: data.absolute_offset,
          span_len: submatch ? submatch.end - submatch.start : query.length,
          context_before: context.context_before,
          context_after: context.context_after,
        });
      }
    } catch (error) {
      console.warn(`Failed to parse ripgrep line: ${line}`, error);
    }
  }
  
  return hits;
}

/**
 * Fallback: in-process text scanning
 */
async function resolveInProcess(
  filePath: string,
  query: string,
  score: number,
  matchReasons: string[],
  fuzzyDistance: number,
  maxMatches: number
): Promise<SearchHit[]> {
  const fileContent = await fs.readFile(filePath, 'utf-8');
  const normalizedContent = normalizeLineEndings(fileContent);
  
  const hits: SearchHit[] = [];
  
  if (fuzzyDistance === 0) {
    // Exact match search
    let searchIndex = 0;
    let matchCount = 0;
    
    while (matchCount < maxMatches) {
      const matchIndex = normalizedContent.indexOf(query, searchIndex);
      if (matchIndex === -1) break;
      
      const location = byteOffsetToLineCol(normalizedContent, matchIndex);
      
      // Validate bounds
      const validation = validateSpanBounds(fileContent, location.line, location.col);
      if (!validation.valid) {
        console.warn(`Invalid span bounds for ${filePath}:${location.line}:${location.col}: ${validation.error}`);
        searchIndex = matchIndex + 1;
        continue;
      }
      
      const snippet = extractSnippet(fileContent, location.line, location.col);
      const context = extractContext(fileContent, location.line);
      
      hits.push({
        file: filePath,
        line: location.line,
        col: location.col,
        snippet,
        score,
        why: matchReasons as any,
        byte_offset: matchIndex,
        span_len: query.length,
        context_before: context.context_before,
        context_after: context.context_after,
      });
      
      matchCount++;
      searchIndex = matchIndex + 1;
    }
  } else {
    // Fuzzy search (simplified implementation)
    // For production, consider using a proper fuzzy string matching library
    console.warn(`Fuzzy search with distance ${fuzzyDistance} not fully implemented, using exact match`);
    return await resolveInProcess(filePath, query, score, matchReasons, 0, maxMatches);
  }
  
  return hits;
}