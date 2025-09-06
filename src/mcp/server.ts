#!/usr/bin/env node

/**
 * Lens MCP Server
 * Provides code search capabilities through the Model Context Protocol
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { LensSearchEngine } from '../api/search-engine.js';

// Schema definitions for MCP tool arguments
const SearchArgsSchema = z.object({
  repo_sha: z.string().describe('Repository SHA or identifier'),
  query: z.string().describe('Search query (supports natural language and fuzzy matching)'),
  mode: z.enum(['lexical', 'semantic', 'hybrid']).optional().default('hybrid').describe('Search mode'),
  fuzzy: z.number().int().min(0).max(2).optional().default(1).describe('Fuzzy matching level (0=exact, 1=medium, 2=high)'),
  k: z.number().int().min(1).max(100).optional().default(20).describe('Number of results to return'),
  scopes: z.array(z.string()).optional().describe('File path patterns to scope search (e.g., ["src/**/*.ts"])'),
  constraints: z.object({
    lang: z.string().optional(),
    symbol_kind: z.enum(['function', 'class', 'variable', 'type', 'interface', 'constant', 'enum', 'method', 'property']).optional(),
  }).optional().describe('Additional search constraints'),
  token_budget: z.number().int().min(100).max(20000).optional().default(10000).describe('Token budget for results'),
});

const ContextArgsSchema = z.object({
  refs: z.array(z.string()).describe('Array of lens:// references to resolve'),
  token_budget: z.number().int().min(100).max(20000).optional().default(10000).describe('Token budget for context'),
});

const ResolveArgsSchema = z.object({
  ref: z.string().describe('Lens reference to resolve (e.g., lens://repo-sha/file.ts@hash#L10:15)'),
  context_lines: z.number().int().min(0).max(50).optional().default(5).describe('Number of surrounding context lines'),
});

const SymbolsArgsSchema = z.object({
  repo_sha: z.string().describe('Repository SHA or identifier'),
  filter: z.object({
    kind: z.enum(['function', 'class', 'variable', 'type', 'interface', 'constant', 'enum', 'method', 'property']).optional(),
    lang: z.string().optional(),
    name_pattern: z.string().optional().describe('Regex pattern for symbol names'),
  }).optional().describe('Filters for symbol listing'),
  token_budget: z.number().int().min(100).max(20000).optional().default(5000).describe('Token budget for symbols'),
});

class LensMCPServer {
  private server: Server;
  private searchEngine: LensSearchEngine;

  constructor() {
    this.server = new Server(
      {
        name: 'lens-search',
        version: '1.0.0-rc.2',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    this.setupErrorHandling();
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'lens_search',
            description: 'Search for code across repositories with semantic understanding and fuzzy matching',
            inputSchema: {
              type: 'object',
              properties: SearchArgsSchema.shape,
              required: ['repo_sha', 'query'],
            },
          } as Tool,
          {
            name: 'lens_context',
            description: 'Batch resolve multiple lens:// references into readable context',
            inputSchema: {
              type: 'object',
              properties: ContextArgsSchema.shape,
              required: ['refs'],
            },
          } as Tool,
          {
            name: 'lens_resolve',
            description: 'Resolve a single lens:// reference to its full content with surrounding context',
            inputSchema: {
              type: 'object',
              properties: ResolveArgsSchema.shape,
              required: ['ref'],
            },
          } as Tool,
          {
            name: 'lens_symbols',
            description: 'List and filter symbols (functions, classes, etc.) in a repository',
            inputSchema: {
              type: 'object',
              properties: SymbolsArgsSchema.shape,
              required: ['repo_sha'],
            },
          } as Tool,
        ],
      };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'lens_search':
            return await this.handleSearch(args);
          
          case 'lens_context':
            return await this.handleContext(args);
          
          case 'lens_resolve':
            return await this.handleResolve(args);
          
          case 'lens_symbols':
            return await this.handleSymbols(args);

          default:
            throw new McpError(
              ErrorCode.MethodNotFound,
              `Tool "${name}" not found`
            );
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        throw new McpError(
          ErrorCode.InternalError,
          `Tool execution failed: ${errorMessage}`
        );
      }
    });
  }

  private async handleSearch(args: unknown) {
    const parsed = SearchArgsSchema.parse(args);
    
    // Initialize search engine if needed
    if (!this.searchEngine) {
      this.searchEngine = new LensSearchEngine();
    }

    // Perform search
    const result = await this.searchEngine.search({
      trace_id: `mcp-${Date.now()}`,
      repo_sha: parsed.repo_sha,
      query: parsed.query,
      mode: parsed.mode,
      k: parsed.k,
      fuzzy_distance: parsed.fuzzy,
      started_at: new Date(),
      stages: [],
    });

    // Transform results with stable references
    const hits = result.hits.map((hit: any) => {
      const sourceHash = this.calculateSourceHash(hit.snippet || '');
      const byteStart = hit.byte_offset || 0;
      const byteEnd = byteStart + (hit.span_len || hit.snippet?.length || 0);

      return {
        ref: this.generateStableRef(
          parsed.repo_sha,
          hit.file,
          sourceHash,
          hit.line,
          hit.line + (hit.snippet?.split('\n').length || 1) - 1,
          byteStart,
          byteEnd,
          hit.ast_path
        ),
        file: hit.file,
        line: hit.line,
        column: hit.col,
        snippet: hit.snippet,
        score: hit.score,
        symbol_kind: hit.symbol_kind,
        language: hit.lang,
        why: hit.why || [],
      };
    });

    // Apply token budget
    let totalTokens = 0;
    const budgetedHits = [];
    for (const hit of hits) {
      const hitTokens = this.estimateTokenCount(JSON.stringify(hit));
      if (totalTokens + hitTokens > parsed.token_budget) {
        break;
      }
      budgetedHits.push(hit);
      totalTokens += hitTokens;
    }

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            results: budgetedHits,
            total_found: result.hits.length,
            returned: budgetedHits.length,
            token_usage: {
              used: totalTokens,
              budget: parsed.token_budget,
              budget_exceeded: budgetedHits.length < hits.length,
            },
            search_info: {
              query: parsed.query,
              mode: parsed.mode,
              repo_sha: parsed.repo_sha,
              latency_ms: result.stage_a_latency + result.stage_b_latency + (result.stage_c_latency || 0),
            },
          }, null, 2),
        },
      ],
    };
  }

  private async handleContext(args: unknown) {
    const parsed = ContextArgsSchema.parse(args);
    
    const contexts = [];
    let totalTokens = 0;
    let budgetExceeded = false;

    for (const ref of parsed.refs) {
      if (totalTokens >= parsed.token_budget) {
        budgetExceeded = true;
        break;
      }

      // Mock context resolution - in production, would actually resolve lens:// refs
      const content = `// Resolved context for ${ref}\n// This would contain the actual code content\nfunction example() {\n  return "resolved content";\n}`;
      const tokenCount = this.estimateTokenCount(content);

      if (totalTokens + tokenCount <= parsed.token_budget) {
        contexts.push({
          ref,
          content,
          token_count: tokenCount,
          truncated: false,
        });
        totalTokens += tokenCount;
      } else {
        // Truncate to fit budget
        const remainingTokens = parsed.token_budget - totalTokens;
        if (remainingTokens > 50) {
          const truncatedContent = content.substring(0, remainingTokens * 4);
          contexts.push({
            ref,
            content: truncatedContent,
            token_count: remainingTokens,
            truncated: true,
          });
          totalTokens = parsed.token_budget;
        }
        budgetExceeded = true;
        break;
      }
    }

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            contexts,
            total_tokens: totalTokens,
            budget_exceeded: budgetExceeded,
            refs_omitted: parsed.refs.length - contexts.length,
          }, null, 2),
        },
      ],
    };
  }

  private async handleResolve(args: unknown) {
    const parsed = ResolveArgsSchema.parse(args);
    
    // Parse the lens:// reference
    const refMatch = parsed.ref.match(/^lens:\/\/([^\/]+)\/([^@]+)@([^#]+)#L(\d+):(\d+)(?:\|B(\d+):(\d+))?(?:\|AST:(.+))?$/);
    
    if (!refMatch) {
      throw new Error(`Invalid lens:// reference format: ${parsed.ref}`);
    }

    const [, repoSha, filePath, sourceHash, lineStart, lineEnd, byteStart, byteEnd, astPath] = refMatch;

    // Mock resolution - would actually read file content in production
    const content = `// Resolved content for ${filePath} lines ${lineStart}-${lineEnd}\nfunction resolvedFunction() {\n  // This would be the actual file content\n  return "example content";\n}`;

    const surroundingLines = {
      before: Array.from({ length: parsed.context_lines }, (_, i) => `// Context line ${i + 1} before`),
      after: Array.from({ length: parsed.context_lines }, (_, i) => `// Context line ${i + 1} after`),
    };

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            ref: parsed.ref,
            file_path: filePath,
            content,
            source_hash: sourceHash,
            line_start: parseInt(lineStart),
            line_end: parseInt(lineEnd),
            byte_start: byteStart ? parseInt(byteStart) : undefined,
            byte_end: byteEnd ? parseInt(byteEnd) : undefined,
            ast_path: astPath,
            surrounding_lines: surroundingLines,
            metadata: {
              language: this.inferLanguageFromPath(filePath),
              symbol_kind: 'function', // Would be derived from AST in production
            },
          }, null, 2),
        },
      ],
    };
  }

  private async handleSymbols(args: unknown) {
    const parsed = SymbolsArgsSchema.parse(args);
    
    // Mock symbol data - would query actual symbol index in production
    const allSymbols = Array.from({ length: 100 }, (_, i) => ({
      symbol_id: `symbol_${i}`,
      name: `function_${i}`,
      kind: 'function' as const,
      file_path: `src/file_${Math.floor(i / 10)}.ts`,
      line: i % 100 + 1,
      ref: `lens://${parsed.repo_sha}/src/file_${Math.floor(i / 10)}.ts@hash${i}#L${i % 100 + 1}:${i % 100 + 1}|B${i * 50}:${i * 50 + 20}`,
      language: 'typescript',
    }));

    // Apply filters
    let filteredSymbols = allSymbols;
    if (parsed.filter?.kind) {
      filteredSymbols = filteredSymbols.filter(s => s.kind === parsed.filter?.kind);
    }
    if (parsed.filter?.name_pattern) {
      const regex = new RegExp(parsed.filter.name_pattern, 'i');
      filteredSymbols = filteredSymbols.filter(s => regex.test(s.name));
    }

    // Apply token budget
    let totalTokens = 0;
    const budgetedSymbols = [];
    for (const symbol of filteredSymbols) {
      const symbolTokens = this.estimateTokenCount(JSON.stringify(symbol));
      if (totalTokens + symbolTokens > parsed.token_budget) {
        break;
      }
      budgetedSymbols.push(symbol);
      totalTokens += symbolTokens;
    }

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            symbols: budgetedSymbols,
            total_found: filteredSymbols.length,
            returned: budgetedSymbols.length,
            token_usage: {
              used: totalTokens,
              budget: parsed.token_budget,
              budget_exceeded: budgetedSymbols.length < filteredSymbols.length,
            },
          }, null, 2),
        },
      ],
    };
  }

  private generateStableRef(repoSha: string, filePath: string, sourceHash: string, lineStart: number, lineEnd: number, byteStart: number, byteEnd: number, astPath?: string): string {
    let ref = `lens://${repoSha}/${filePath}@${sourceHash}#L${lineStart}:${lineEnd}|B${byteStart}:${byteEnd}`;
    if (astPath) {
      ref += `|AST:${astPath}`;
    }
    return ref;
  }

  private calculateSourceHash(content: string): string {
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  }

  private estimateTokenCount(text: string): number {
    return Math.ceil(text.length / 4);
  }

  private inferLanguageFromPath(filePath: string): string {
    const ext = filePath.split('.').pop()?.toLowerCase();
    const langMap: Record<string, string> = {
      'ts': 'typescript',
      'js': 'javascript',
      'py': 'python',
      'rs': 'rust',
      'go': 'go',
      'java': 'java',
      'cpp': 'cpp',
      'c': 'c',
      'h': 'c',
      'hpp': 'cpp',
    };
    return langMap[ext || ''] || 'unknown';
  }

  private setupErrorHandling() {
    this.server.onerror = (error) => {
      console.error('[MCP Server Error]', error);
    };

    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Lens MCP Server running on stdio');
  }
}

// Run the server
if (import.meta.url === `file://${process.argv[1]}`) {
  const server = new LensMCPServer();
  server.run().catch((error) => {
    console.error('Failed to start MCP server:', error);
    process.exit(1);
  });
}

export { LensMCPServer };