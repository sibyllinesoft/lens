/**
 * LSP SPI Service Layer
 * Coordinates LSP clients, caching, and provides typed operations
 */

import { LSPClient, LSPPosition, LSPRange, LSPDiagnostic, LSPTextEdit, LSPWorkspaceEdit, LSPCodeAction } from './client.js';
import { LSPCacheManager, LSPCacheKey, globalLSPCache } from './cache.js';
import { globalLSPMetrics, withLSPMetrics } from './metrics.js';
import { LensTracer } from '../telemetry/tracer.js';
import { LensSearchEngine } from '../api/search-engine.js';
import { readFile } from 'fs/promises';
import { join, resolve } from 'path';
import { spawn } from 'child_process';
import { existsSync } from 'fs';
import { v4 as uuidv4 } from 'uuid';

import type {
  LSPCapabilitiesResponse,
  LSPDiagnosticsRequest,
  LSPDiagnosticsResponse,
  LSPFormatRequest,
  LSPFormatResponse,
  LSPSelectionRangesRequest,
  LSPSelectionRangesResponse,
  LSPFoldingRangesRequest,
  LSPFoldingRangesResponse,
  LSPPrepareRenameRequest,
  LSPPrepareRenameResponse,
  LSPRenameRequest,
  LSPRenameResponse,
  LSPCodeActionsRequest,
  LSPCodeActionsResponse,
  LSPHierarchyRequest,
  LSPHierarchyResponse,
  LensSymbol
} from '../types/api.js';

interface LSPLanguageConfig {
  language: string;
  extensions: string[];
  command: string;
  args: string[];
  enabled: boolean;
  initializationOptions?: any;
  workspaceDetection?: {
    files: string[];
    directories: string[];
  };
  installationGuide?: string;
}

export class LSPSPIService {
  private clients = new Map<string, LSPClient>(); // language -> client
  private symbolIdCounter = 0;
  private symbolRegistry = new Map<string, LensSymbol>(); // symbol_id -> LensSymbol
  
  // Language configurations
  private languageConfigs: LSPLanguageConfig[] = [
    {
      language: 'typescript',
      extensions: ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'],
      command: 'typescript-language-server',
      args: ['--stdio'],
      enabled: process.env.LSP_TYPESCRIPT_ENABLED !== 'false',
      workspaceDetection: {
        files: ['package.json', 'tsconfig.json', 'jsconfig.json'],
        directories: ['node_modules', '.git']
      },
      initializationOptions: {
        preferences: {
          disableSuggestions: false,
          quotePreference: 'auto',
          includeCompletionsForModuleExports: true,
          includeCompletionsForImportStatements: true
        }
      },
      installationGuide: 'npm install -g typescript-language-server typescript'
    },
    {
      language: 'python',
      extensions: ['.py', '.pyx', '.pyi'],
      command: 'pylsp',
      args: [],
      enabled: process.env.LSP_PYTHON_ENABLED !== 'false',
      workspaceDetection: {
        files: ['requirements.txt', 'pyproject.toml', 'setup.py', 'setup.cfg', 'Pipfile'],
        directories: ['venv', '.venv', '__pycache__', '.git']
      },
      initializationOptions: {
        settings: {
          pylsp: {
            plugins: {
              pycodestyle: {
                enabled: true,
                maxLineLength: 88
              },
              pyflakes: { enabled: true },
              pylint: { enabled: false },
              autopep8: { enabled: false },
              yapf: { enabled: false },
              black: { enabled: true },
              rope_completion: { enabled: true },
              jedi_completion: { 
                enabled: true,
                fuzzy: true,
                eager: true,
                include_params: true
              },
              jedi_hover: { enabled: true },
              jedi_references: { enabled: true },
              jedi_signature_help: { enabled: true },
              jedi_symbols: { enabled: true }
            }
          }
        }
      },
      installationGuide: 'pip install python-lsp-server[all] python-lsp-black'
    },
    {
      language: 'go',
      extensions: ['.go'],
      command: 'gopls',
      args: [],
      enabled: process.env.LSP_GO_ENABLED !== 'false',
      workspaceDetection: {
        files: ['go.mod', 'go.sum'],
        directories: ['vendor', '.git']
      },
      initializationOptions: {
        usePlaceholders: true,
        completionDocumentation: true,
        deepCompletion: true,
        completeUnimported: true,
        staticcheck: true,
        gofumpt: true,
        analyses: {
          unusedparams: true,
          shadow: true,
          nilness: true,
          unusedwrite: true,
          useany: true
        },
        codelenses: {
          gc_details: true,
          generate: true,
          regenerate_cgo: true,
          test: true,
          tidy: true,
          upgrade_dependency: true,
          vendor: true
        },
        hints: {
          assignVariableTypes: true,
          compositeLiteralFields: true,
          compositeLiteralTypes: true,
          constantValues: true,
          functionTypeParameters: true,
          parameterNames: true,
          rangeVariableTypes: true
        }
      },
      installationGuide: 'go install golang.org/x/tools/gopls@latest'
    },
    {
      language: 'rust',
      extensions: ['.rs'],
      command: 'rust-analyzer',
      args: [],
      enabled: process.env.LSP_RUST_ENABLED !== 'false',
      workspaceDetection: {
        files: ['Cargo.toml', 'Cargo.lock', 'rust-toolchain', 'rust-toolchain.toml'],
        directories: ['target', 'src', '.git']
      },
      initializationOptions: {
        cargo: {
          loadOutDirsFromCheck: true,
          runBuildScripts: true,
          buildScripts: {
            enable: true
          }
        },
        procMacro: {
          enable: true
        },
        diagnostics: {
          enable: true,
          experimental: {
            enable: true
          }
        },
        completion: {
          addCallArgumentSnippets: true,
          addCallParenthesis: true,
          postfix: {
            enable: true
          },
          privateEditable: {
            enable: true
          }
        },
        hover: {
          actions: {
            enable: true,
            implementations: {
              enable: true
            },
            references: {
              enable: true
            },
            run: {
              enable: true
            },
            debug: {
              enable: true
            }
          }
        },
        inlayHints: {
          bindingModeHints: {
            enable: false
          },
          chainingHints: {
            enable: true
          },
          closingBraceHints: {
            enable: true,
            minLines: 25
          },
          closureReturnTypeHints: {
            enable: "never"
          },
          lifetimeElisionHints: {
            enable: "never",
            useParameterNames: false
          },
          maxLength: 25,
          parameterHints: {
            enable: true
          },
          reborrowHints: {
            enable: "never"
          },
          renderColons: true,
          typeHints: {
            enable: true,
            hideClosureInitialization: false,
            hideNamedConstructor: false
          }
        },
        lens: {
          enable: true,
          implementations: {
            enable: true
          },
          references: {
            adt: {
              enable: true
            },
            enumVariant: {
              enable: true
            },
            method: {
              enable: true
            },
            trait: {
              enable: true
            }
          },
          run: {
            enable: true
          }
        }
      },
      installationGuide: 'rustup component add rust-analyzer'
    }
  ];

  constructor(private workspaceRoot: string = './indexed-content') {}

  /**
   * Get system status and available LSP servers
   */
  async getSystemStatus(): Promise<{
    languages: Array<{
      language: string;
      enabled: boolean;
      available: boolean;
      command: string;
      installationGuide: string;
      workspaceDetected?: boolean;
      workspaceRoot?: string;
    }>;
  }> {
    const languageStatuses = await Promise.all(
      this.languageConfigs.map(async (config) => {
        const available = config.enabled ? await this.checkLSPServerAvailable(config.command) : false;
        const workspaceRoot = config.enabled ? this.detectWorkspaceRoot(config.language, this.workspaceRoot) : undefined;
        const workspaceDetected = workspaceRoot !== this.workspaceRoot;

        return {
          language: config.language,
          enabled: config.enabled,
          available,
          command: config.command,
          installationGuide: config.installationGuide || 'Check language documentation',
          workspaceDetected,
          workspaceRoot
        };
      })
    );

    return { languages: languageStatuses };
  }

  /**
   * Get capabilities for all supported languages
   */
  async getCapabilities(repo_sha?: string): Promise<LSPCapabilitiesResponse> {
    const span = LensTracer.createChildSpan('lsp_get_capabilities');
    
    try {
      // Only include languages that are enabled and have available LSP servers
      const languagesWithStatus = await Promise.all(
        this.languageConfigs
          .filter(config => config.enabled)
          .map(async (config) => {
            const available = await this.checkLSPServerAvailable(config.command);
            if (available) {
              return {
                lang: config.language,
                features: [
                  'diagnostics',
                  'format',
                  'selectionRanges',
                  'foldingRanges',
                  'prepareRename',
                  'rename',
                  'codeActions',
                  'callHierarchy',
                  'typeHierarchy'
                ] as ('diagnostics' | 'format' | 'selectionRanges' | 'foldingRanges' | 'prepareRename' | 'rename' | 'codeActions' | 'callHierarchy' | 'typeHierarchy')[]
              };
            }
            return null;
          })
      );

      const languages = languagesWithStatus.filter((lang): lang is NonNullable<typeof lang> => lang !== null);

      span.setAttributes({
        success: true,
        languages_count: languages.length,
        enabled_languages: languages.map(l => l.lang).join(',')
      });

      return { languages };
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get diagnostics for files
   */
  getDiagnostics = withLSPMetrics('diagnostics', async (request: LSPDiagnosticsRequest): Promise<LSPDiagnosticsResponse> => {
    const startTime = Date.now();
    const span = LensTracer.createChildSpan('lsp_get_diagnostics');
    const budget = request.budget_ms || 5000;
    const timeoutAt = Date.now() + budget;

    try {
      const diagnosticsPromises = request.files.map(async (fileReq) => {
        // Check cache first
        const cacheKey: LSPCacheKey = {
          repo_sha: this.extractRepoSha(fileReq.path),
          path: fileReq.path,
          source_hash: fileReq.source_hash,
          operation: 'diagnostics'
        };

        const cached = await globalLSPCache.get<LSPDiagnostic[]>(cacheKey);
        if (cached) {
          return {
            path: fileReq.path,
            source_hash: fileReq.source_hash,
            items: cached.map(diag => this.convertDiagnostic(diag, ''))
          };
        }

        // Get language for file
        const language = this.detectLanguage(fileReq.path);
        if (!language) {
          return {
            path: fileReq.path,
            source_hash: fileReq.source_hash,
            items: []
          };
        }

        // Get or create LSP client
        const client = await this.getOrCreateClient(language);
        if (!client) {
          return {
            path: fileReq.path,
            source_hash: fileReq.source_hash,
            items: []
          };
        }

        // Read file content
        const filePath = resolve(this.workspaceRoot, fileReq.path);
        const content = await readFile(filePath, 'utf8');
        const uri = `file://${filePath}`;

        // Sync document with LSP server
        client.didOpen(uri, language, 1, content);

        // Get diagnostics (simplified - in real implementation would use publishDiagnostics)
        const diagnostics = await client.diagnostics(uri);
        
        // Convert and cache
        const convertedDiags = diagnostics.map(diag => this.convertDiagnostic(diag, ''));
        await globalLSPCache.set(cacheKey, diagnostics, Date.now() - startTime);

        return {
          path: fileReq.path,
          source_hash: fileReq.source_hash,
          items: convertedDiags
        };
      });

      // Wait for all diagnostics with timeout
      const diagnosticsResults = await Promise.all(diagnosticsPromises);
      
      const duration_ms = Date.now() - startTime;
      const timed_out = Date.now() >= timeoutAt;

      // Sort diagnostics deterministically
      for (const diag of diagnosticsResults) {
        diag.items.sort((a, b) => {
          // Sort by severity (error=4, warning=3, info=2, hint=1)
          const severityOrder = { error: 4, warning: 3, info: 2, hint: 1 };
          const severityDiff = (severityOrder[b.severity] || 0) - (severityOrder[a.severity] || 0);
          if (severityDiff !== 0) return severityDiff;

          // Then by position
          const posDiff = a.range.b0 - b.range.b0;
          if (posDiff !== 0) return posDiff;

          // Then by code
          return (a.code || '').localeCompare(b.code || '');
        });
      }

      span.setAttributes({
        success: true,
        files_count: request.files.length,
        total_diagnostics: diagnosticsResults.reduce((sum, d) => sum + d.items.length, 0),
        duration_ms,
        timed_out
      });

      return {
        diags: diagnosticsResults,
        duration_ms,
        timed_out
      };

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      const duration_ms = Date.now() - startTime;
      
      span.recordException(error);
      span.setAttributes({ 
        success: false, 
        error: errorMsg,
        duration_ms 
      });

      // Return partial results on timeout
      if (Date.now() >= timeoutAt) {
        return {
          diags: request.files.map(f => ({
            path: f.path,
            source_hash: f.source_hash,
            items: []
          })),
          duration_ms,
          timed_out: true
        };
      }

      throw error;
    } finally {
      span.end();
    }
  });

  /**
   * Format document or range
   */
  format = withLSPMetrics('format', async (request: LSPFormatRequest): Promise<LSPFormatResponse> => {
    const startTime = Date.now();
    const span = LensTracer.createChildSpan('lsp_format');
    const budget = request.budget_ms || 5000;

    try {
      let filePath: string;
      let repo_sha = 'unknown';

      // Extract path from ref or use direct path
      if (request.ref) {
        const parsed = LSPCacheManager.parseLensRef(request.ref);
        if (!parsed) {
          throw new Error('Invalid lens:// ref format');
        }
        filePath = parsed.path;
        repo_sha = parsed.repo_sha;
      } else if (request.path) {
        filePath = request.path;
      } else {
        throw new Error('Either ref or path must be provided');
      }

      // Read current file content for source hash
      const fullPath = resolve(this.workspaceRoot, filePath);
      const content = await readFile(fullPath, 'utf8');
      const source_hash = LSPCacheManager.computeSourceHash(content);

      // Check cache
      const cacheKey: LSPCacheKey = {
        repo_sha,
        path: filePath,
        source_hash,
        operation: 'format'
      };

      const cached = await globalLSPCache.get<LSPFormatResponse>(cacheKey);
      if (cached) {
        return cached;
      }

      // Get language and client
      const language = this.detectLanguage(filePath);
      if (!language) {
        throw new Error(`Unsupported file type: ${filePath}`);
      }

      const client = await this.getOrCreateClient(language);
      if (!client) {
        throw new Error(`LSP client not available for ${language}`);
      }

      const uri = `file://${fullPath}`;
      
      // Sync document
      client.didOpen(uri, language, 1, content);

      // Format document or range
      let edits: LSPTextEdit[];
      const options = request.options || this.getDefaultFormatOptions(language);
      
      if (request.range && request.range.b0 !== undefined && request.range.b1 !== undefined) {
        const lspRange = this.convertToLSPRange({ b0: request.range.b0, b1: request.range.b1 }, content);
        edits = await client.formatRange(uri, lspRange, options);
      } else {
        edits = await client.formatDocument(uri, options);
      }

      // Convert edits to our format
      const convertedEdits = edits.map(edit => ({
        path: filePath,
        range: this.convertFromLSPRange(edit.range, content),
        new_text: edit.newText
      }));

      // Test idempotence by applying edits and formatting again
      let isIdempotent = false;
      try {
        const editedContent = this.applyEdits(content, edits);
        client.didChange(uri, 2, [{ text: editedContent }]);
        
        const secondEdits = (request.range && request.range.b0 !== undefined && request.range.b1 !== undefined)
          ? await client.formatRange(uri, this.convertToLSPRange({ b0: request.range.b0, b1: request.range.b1 }, editedContent), options)
          : await client.formatDocument(uri, options);
          
        isIdempotent = secondEdits.length === 0;
      } catch (error) {
        console.warn('Failed to test format idempotence:', error);
      }

      const duration_ms = Date.now() - startTime;
      
      const response: LSPFormatResponse = {
        edits: convertedEdits,
        idempotent: isIdempotent,
        duration_ms
      };

      // Cache result
      await globalLSPCache.set(cacheKey, response, duration_ms);

      span.setAttributes({
        success: true,
        language,
        edits_count: convertedEdits.length,
        idempotent: isIdempotent,
        duration_ms
      });

      return response;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      const duration_ms = Date.now() - startTime;
      
      span.recordException(error);
      span.setAttributes({ 
        success: false, 
        error: errorMsg,
        duration_ms 
      });
      
      throw error;
    } finally {
      span.end();
    }
  }); // Close the withLSPMetrics arrow function

  /**
   * Get selection ranges
   */
  async getSelectionRanges(request: LSPSelectionRangesRequest): Promise<LSPSelectionRangesResponse> {
    const startTime = Date.now();
    const span = LensTracer.createChildSpan('lsp_selection_ranges');

    try {
      const chainsPromises = request.refs.map(async (ref) => {
        const parsed = LSPCacheManager.parseLensRef(ref);
        if (!parsed || parsed.start === undefined || parsed.end === undefined) {
          return [];
        }

        // Check cache
        const cacheKey: LSPCacheKey = {
          repo_sha: parsed.repo_sha,
          path: parsed.path,
          source_hash: parsed.source_hash,
          operation: 'selectionRanges'
        };

        const cached = await globalLSPCache.get<any[]>(cacheKey);
        if (cached) return cached;

        // Get language and client
        const language = this.detectLanguage(parsed.path);
        if (!language) return [];

        const client = await this.getOrCreateClient(language);
        if (!client) return [];

        // Read file and convert position
        const fullPath = resolve(this.workspaceRoot, parsed.path);
        const content = await readFile(fullPath, 'utf8');
        const position = this.byteOffsetToPosition(content, parsed.start);
        const uri = `file://${fullPath}`;

        // Sync document and get selection ranges
        client.didOpen(uri, language, 1, content);
        const selections = await client.selectionRanges(uri, [position]);

        // Convert to our format
        const chain = (selections[0] || []).map((sel: any, idx: number) => ({
          range: this.convertFromLSPRange(sel.range, content),
          parent_ix: sel.parent ? idx + 1 : undefined
        }));

        await globalLSPCache.set(cacheKey, chain, Date.now() - startTime);
        return chain;
      });

      const chains = await Promise.all(chainsPromises);
      const duration_ms = Date.now() - startTime;

      span.setAttributes({
        success: true,
        refs_count: request.refs.length,
        total_selections: chains.reduce((sum, chain) => sum + chain.length, 0),
        duration_ms
      });

      return { chains, duration_ms };

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      const duration_ms = Date.now() - startTime;
      
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg, duration_ms });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get folding ranges
   */
  async getFoldingRanges(request: LSPFoldingRangesRequest): Promise<LSPFoldingRangesResponse> {
    const startTime = Date.now();
    const span = LensTracer.createChildSpan('lsp_folding_ranges');

    try {
      const foldsPromises = request.files.map(async (fileReq) => {
        // Check cache
        const cacheKey: LSPCacheKey = {
          repo_sha: 'unknown',
          path: fileReq.path,
          source_hash: fileReq.source_hash,
          operation: 'foldingRanges'
        };

        const cached = await globalLSPCache.get<any[]>(cacheKey);
        if (cached) {
          return { path: fileReq.path, ranges: cached };
        }

        // Get language and client
        const language = this.detectLanguage(fileReq.path);
        if (!language) {
          return { path: fileReq.path, ranges: [] };
        }

        const client = await this.getOrCreateClient(language);
        if (!client) {
          return { path: fileReq.path, ranges: [] };
        }

        // Read file
        const fullPath = resolve(this.workspaceRoot, fileReq.path);
        const content = await readFile(fullPath, 'utf8');
        const uri = `file://${fullPath}`;

        // Sync document and get folding ranges
        client.didOpen(uri, language, 1, content);
        const folds = await client.foldingRanges(uri);

        // Convert to our format
        const ranges = folds.map((fold: any) => ({
          b0: this.positionToByteOffset(content, { line: fold.startLine, character: fold.startCharacter || 0 }),
          b1: this.positionToByteOffset(content, { line: fold.endLine, character: fold.endCharacter || 0 }),
          kind: fold.kind
        }));

        // Sort deterministically by b0, then b1
        ranges.sort((a, b) => a.b0 - b.b0 || a.b1 - b.b1);

        await globalLSPCache.set(cacheKey, ranges, Date.now() - startTime);
        return { path: fileReq.path, ranges };
      });

      const folds = await Promise.all(foldsPromises);
      const duration_ms = Date.now() - startTime;

      span.setAttributes({
        success: true,
        files_count: request.files.length,
        total_folds: folds.reduce((sum, f) => sum + f.ranges.length, 0),
        duration_ms
      });

      return { folds, duration_ms };

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      const duration_ms = Date.now() - startTime;
      
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg, duration_ms });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Prepare rename validation
   */
  async prepareRename(request: LSPPrepareRenameRequest): Promise<LSPPrepareRenameResponse> {
    const startTime = Date.now();
    const span = LensTracer.createChildSpan('lsp_prepare_rename');

    try {
      const parsed = LSPCacheManager.parseLensRef(request.ref);
      if (!parsed || parsed.start === undefined || parsed.end === undefined) {
        return {
          allowed: false,
          reason: 'Invalid lens:// ref format',
          duration_ms: Date.now() - startTime
        };
      }

      // Check cache
      const cacheKey: LSPCacheKey = {
        repo_sha: parsed.repo_sha,
        path: parsed.path,
        source_hash: parsed.source_hash,
        operation: 'prepareRename'
      };

      const cached = await globalLSPCache.get<LSPPrepareRenameResponse>(cacheKey);
      if (cached) return cached;

      // Get language and client
      const language = this.detectLanguage(parsed.path);
      if (!language) {
        return {
          allowed: false,
          reason: `Unsupported language for ${parsed.path}`,
          duration_ms: Date.now() - startTime
        };
      }

      const client = await this.getOrCreateClient(language);
      if (!client) {
        return {
          allowed: false,
          reason: `LSP client not available for ${language}`,
          duration_ms: Date.now() - startTime
        };
      }

      // Read file and prepare
      const fullPath = resolve(this.workspaceRoot, parsed.path);
      const content = await readFile(fullPath, 'utf8');
      const position = this.byteOffsetToPosition(content, parsed.start);
      const uri = `file://${fullPath}`;

      client.didOpen(uri, language, 1, content);
      const result = await client.prepareRename(uri, position);

      let response: LSPPrepareRenameResponse;
      
      if (result) {
        response = {
          allowed: true,
          placeholder: typeof result === 'string' ? result : result.placeholder,
          range: result.range ? this.convertFromLSPRange(result.range, content) : {
            b0: parsed.start,
            b1: parsed.end
          },
          duration_ms: Date.now() - startTime
        };
      } else {
        response = {
          allowed: false,
          reason: 'Rename not supported at this location',
          duration_ms: Date.now() - startTime
        };
      }

      await globalLSPCache.set(cacheKey, response, Date.now() - startTime);
      
      span.setAttributes({
        success: true,
        allowed: response.allowed,
        language,
        duration_ms: response.duration_ms
      });

      return response;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      const duration_ms = Date.now() - startTime;
      
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg, duration_ms });
      
      return {
        allowed: false,
        reason: `Error: ${errorMsg}`,
        duration_ms
      };
    } finally {
      span.end();
    }
  }

  /**
   * Execute rename operation
   */
  async rename(request: LSPRenameRequest): Promise<LSPRenameResponse> {
    const startTime = Date.now();
    const span = LensTracer.createChildSpan('lsp_rename');

    try {
      const parsed = LSPCacheManager.parseLensRef(request.ref);
      if (!parsed || parsed.start === undefined || parsed.end === undefined) {
        throw new Error('Invalid lens:// ref format');
      }

      // Check cache
      const cacheKey: LSPCacheKey = {
        repo_sha: parsed.repo_sha,
        path: parsed.path,
        source_hash: parsed.source_hash,
        operation: 'rename'
      };

      const cached = await globalLSPCache.get<LSPRenameResponse>(cacheKey);
      if (cached) return cached;

      // Get language and client
      const language = this.detectLanguage(parsed.path);
      if (!language) {
        throw new Error(`Unsupported language for ${parsed.path}`);
      }

      const client = await this.getOrCreateClient(language);
      if (!client) {
        throw new Error(`LSP client not available for ${language}`);
      }

      // Read file and execute rename
      const fullPath = resolve(this.workspaceRoot, parsed.path);
      const content = await readFile(fullPath, 'utf8');
      const position = this.byteOffsetToPosition(content, parsed.start);
      const uri = `file://${fullPath}`;

      client.didOpen(uri, language, 1, content);
      const workspaceEdit = await client.rename(uri, position, request.new_name);

      if (!workspaceEdit) {
        throw new Error('Rename operation failed');
      }

      // Convert to our format and sort deterministically
      const changes = await Promise.all(Object.entries(workspaceEdit.changes || {}).map(async ([uri, edits]) => {
        const path = uri.replace(/^file:\/\//, '');
        
        // Read actual file content for accurate byte offset conversion
        let fileContent = content; // Default to current file content
        try {
          if (path !== parsed.path) {
            fileContent = await readFile(path, 'utf-8');
          }
        } catch (error) {
          // Fall back to provided content if file can't be read
          fileContent = content;
        }
        
        return {
          path,
          source_hash: LSPCacheManager.computeSourceHash(fileContent),
          edits: edits.map(edit => ({
            b0: this.positionToByteOffset(fileContent, edit.range.start),
            b1: this.positionToByteOffset(fileContent, edit.range.end),
            new_text: edit.newText
          })).sort((a, b) => a.b0 - b.b0) // Sort edits by position
        };
      })).then(changes => changes.sort((a, b) => a.path.localeCompare(b.path))); // Sort changes by path

      const duration_ms = Date.now() - startTime;
      
      const response: LSPRenameResponse = {
        workspaceEdit: { changes },
        duration_ms
      };

      // Cache result
      await globalLSPCache.set(cacheKey, response, duration_ms);

      span.setAttributes({
        success: true,
        language,
        changes_count: changes.length,
        total_edits: changes.reduce((sum, c) => sum + c.edits.length, 0),
        duration_ms
      });

      return response;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      const duration_ms = Date.now() - startTime;
      
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg, duration_ms });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get code actions
   */
  async getCodeActions(request: LSPCodeActionsRequest): Promise<LSPCodeActionsResponse> {
    const startTime = Date.now();
    const span = LensTracer.createChildSpan('lsp_code_actions');

    try {
      const parsed = LSPCacheManager.parseLensRef(request.ref);
      if (!parsed || parsed.start === undefined || parsed.end === undefined) {
        throw new Error('Invalid lens:// ref format');
      }

      // Check cache
      const cacheKey: LSPCacheKey = {
        repo_sha: parsed.repo_sha,
        path: parsed.path,
        source_hash: parsed.source_hash,
        operation: 'codeActions'
      };

      const cached = await globalLSPCache.get<LSPCodeActionsResponse>(cacheKey);
      if (cached) return cached;

      // Get language and client
      const language = this.detectLanguage(parsed.path);
      if (!language) {
        throw new Error(`Unsupported language for ${parsed.path}`);
      }

      const client = await this.getOrCreateClient(language);
      if (!client) {
        throw new Error(`LSP client not available for ${language}`);
      }

      // Read file and get code actions
      const fullPath = resolve(this.workspaceRoot, parsed.path);
      const content = await readFile(fullPath, 'utf8');
      const range = {
        start: this.byteOffsetToPosition(content, parsed.start),
        end: this.byteOffsetToPosition(content, parsed.end)
      };
      const uri = `file://${fullPath}`;

      client.didOpen(uri, language, 1, content);
      
      const context = {
        diagnostics: request.diagnostics || [],
        only: request.kinds
      };
      
      const lspActions = await client.codeActions(uri, range, context);

      // Convert and filter actions (only text edits, no commands)
      const actions = lspActions
        .filter(action => action.edit && !action.data) // Only pure text edits
        .map(action => ({
          title: action.title,
          kind: action.kind || 'quickfix',
          workspaceEdit: action.edit ? {
            changes: Object.entries(action.edit.changes || {}).map(([uri, edits]) => ({
              path: uri.replace(/^file:\/\//, ''),
              source_hash: LSPCacheManager.computeSourceHash(content),
              edits: edits.map(edit => ({
                b0: this.positionToByteOffset(content, edit.range.start),
                b1: this.positionToByteOffset(content, edit.range.end),
                new_text: edit.newText
              })).sort((a, b) => a.b0 - b.b0)
            }))
          } : undefined
        }))
        .sort((a, b) => {
          // Sort by kind, then title, then first edit position
          const kindDiff = a.kind.localeCompare(b.kind);
          if (kindDiff !== 0) return kindDiff;
          
          const titleDiff = a.title.localeCompare(b.title);
          if (titleDiff !== 0) return titleDiff;
          
          const aFirstEdit = a.workspaceEdit?.changes[0]?.edits[0]?.b0 || 0;
          const bFirstEdit = b.workspaceEdit?.changes[0]?.edits[0]?.b0 || 0;
          return aFirstEdit - bFirstEdit;
        });

      const duration_ms = Date.now() - startTime;
      
      const response: LSPCodeActionsResponse = {
        actions,
        duration_ms
      };

      await globalLSPCache.set(cacheKey, response, duration_ms);

      span.setAttributes({
        success: true,
        language,
        actions_count: actions.length,
        duration_ms
      });

      return response;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      const duration_ms = Date.now() - startTime;
      
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg, duration_ms });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get hierarchy (call or type hierarchy)
   */
  async getHierarchy(request: LSPHierarchyRequest): Promise<LSPHierarchyResponse> {
    const startTime = Date.now();
    const span = LensTracer.createChildSpan('lsp_hierarchy');

    try {
      const parsed = LSPCacheManager.parseLensRef(request.ref);
      if (!parsed || parsed.start === undefined || parsed.end === undefined) {
        throw new Error('Invalid lens:// ref format');
      }

      // Check cache
      const cacheKey: LSPCacheKey = {
        repo_sha: parsed.repo_sha,
        path: parsed.path,
        source_hash: parsed.source_hash,
        operation: 'hierarchy'
      };

      const cached = await globalLSPCache.get<LSPHierarchyResponse>(cacheKey);
      if (cached) return cached;

      // Get language and client
      const language = this.detectLanguage(parsed.path);
      if (!language) {
        throw new Error(`Unsupported language for ${parsed.path}`);
      }

      const client = await this.getOrCreateClient(language);
      if (!client) {
        throw new Error(`LSP client not available for ${language}`);
      }

      // Read file and setup
      const fullPath = resolve(this.workspaceRoot, parsed.path);
      const content = await readFile(fullPath, 'utf8');
      const position = this.byteOffsetToPosition(content, parsed.start);
      const uri = `file://${fullPath}`;

      client.didOpen(uri, language, 1, content);

      let nodes: any[] = [];
      let edges: any[] = [];
      let truncated = false;
      
      if (request.kind === 'call') {
        // Prepare call hierarchy
        const items = await client.prepareCallHierarchy(uri, position);
        
        if (items.length > 0) {
          const depth = request.depth || 2;
          const fanoutCap = request.fanout_cap || 100;
          
          // Get incoming or outgoing calls
          let calls: any[] = [];
          if (request.dir === 'incoming') {
            calls = await client.callHierarchyIncoming(items[0]);
          } else {
            calls = await client.callHierarchyOutgoing(items[0]);
          }
          
          // Truncate if needed
          if (calls.length > fanoutCap) {
            calls = calls.slice(0, fanoutCap);
            truncated = true;
          }
          
          // Convert to our format
          const nodeMap = new Map<string, any>();
          
          // Add root node
          const rootId = this.generateSymbolId(items[0]);
          nodeMap.set(rootId, {
            symbol_id: rootId,
            name: items[0].name,
            kind: items[0].kind,
            def_ref: this.convertLocationToRef(items[0].uri, items[0].range, parsed.repo_sha, parsed.source_hash)
          });
          
          // Add connected nodes and edges
          for (const call of calls) {
            const callId = this.generateSymbolId(call.from || call.to);
            
            if (!nodeMap.has(callId)) {
              nodeMap.set(callId, {
                symbol_id: callId,
                name: (call.from || call.to).name,
                kind: (call.from || call.to).kind,
                def_ref: this.convertLocationToRef(
                  (call.from || call.to).uri, 
                  (call.from || call.to).range, 
                  parsed.repo_sha, 
                  parsed.source_hash
                )
              });
            }
            
            // Add edge
            edges.push({
              src: request.dir === 'incoming' ? callId : rootId,
              dst: request.dir === 'incoming' ? rootId : callId,
              role: 'calls'
            });
          }
          
          nodes = Array.from(nodeMap.values());
        }
      }

      const duration_ms = Date.now() - startTime;
      
      const response: LSPHierarchyResponse = {
        nodes,
        edges,
        truncated,
        duration_ms
      };

      await globalLSPCache.set(cacheKey, response, duration_ms);

      span.setAttributes({
        success: true,
        language,
        nodes_count: nodes.length,
        edges_count: edges.length,
        truncated,
        duration_ms
      });

      return response;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      const duration_ms = Date.now() - startTime;
      
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg, duration_ms });
      throw error;
    } finally {
      span.end();
    }
  }

  // Helper methods
  private detectLanguage(path: string): string | null {
    const lowerPath = path.toLowerCase();
    
    // Handle special cases first
    if (lowerPath.includes('cargo.toml') || lowerPath.includes('cargo.lock')) {
      return this.languageConfigs.find(c => c.language === 'rust' && c.enabled) ? 'rust' : null;
    }
    
    const ext = lowerPath.split('.').pop();
    if (!ext) return null;

    // Handle composite extensions
    const fullExt = `.${ext}`;
    for (const config of this.languageConfigs) {
      if (config.enabled && config.extensions.includes(fullExt)) {
        return config.language;
      }
    }
    
    return null;
  }

  /**
   * Check if LSP server is available on the system
   */
  private async checkLSPServerAvailable(command: string): Promise<boolean> {
    return new Promise((resolve) => {
      // Try different methods to check command availability
      const isWindows = process.platform === 'win32';
      const checkCommand = isWindows ? 'where' : 'which';
      
      const checkProcess = spawn(checkCommand, [command], { 
        stdio: 'ignore',
        shell: isWindows // Windows requires shell for 'where' command
      });
      
      let resolved = false;
      const timeout = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          checkProcess.kill();
          resolve(false);
        }
      }, 5000); // 5 second timeout

      checkProcess.on('close', (code) => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          resolve(code === 0);
        }
      });
      
      checkProcess.on('error', () => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          resolve(false);
        }
      });
    });
  }

  /**
   * Detect workspace root for a given language in a directory
   */
  private detectWorkspaceRoot(language: string, startPath: string): string {
    const config = this.languageConfigs.find(c => c.language === language);
    if (!config?.workspaceDetection) {
      return startPath;
    }

    let currentPath = resolve(startPath);
    const rootPath = resolve('/');

    // Traverse up the directory tree looking for workspace markers
    while (currentPath !== rootPath) {
      // Check for workspace files
      for (const file of config.workspaceDetection.files) {
        if (existsSync(join(currentPath, file))) {
          return currentPath;
        }
      }

      // Check for workspace directories
      for (const dir of config.workspaceDetection.directories) {
        if (existsSync(join(currentPath, dir))) {
          return currentPath;
        }
      }

      const parentPath = resolve(currentPath, '..');
      if (parentPath === currentPath) break;
      currentPath = parentPath;
    }

    return startPath;
  }

  /**
   * Get language-specific environment variables and settings
   */
  private getLanguageEnvironment(language: string): Record<string, string> {
    const env = { ...process.env };

    switch (language) {
      case 'python':
        // Detect virtual environment
        const venvPaths = [
          join(this.workspaceRoot, 'venv'),
          join(this.workspaceRoot, '.venv'),
          process.env.VIRTUAL_ENV
        ].filter(Boolean);

        for (const venvPath of venvPaths) {
          if (venvPath && existsSync(venvPath)) {
            env.VIRTUAL_ENV = venvPath;
            env.PATH = `${join(venvPath, 'bin')}:${env.PATH}`;
            break;
          }
        }
        break;

      case 'go':
        // Set Go module mode
        env.GO111MODULE = 'on';
        
        // Detect Go workspace
        const goWorkspace = this.detectWorkspaceRoot('go', this.workspaceRoot);
        if (goWorkspace !== this.workspaceRoot) {
          env.GOWORK = join(goWorkspace, 'go.work');
        }
        break;

      case 'rust':
        // Set Rust environment variables for better analysis
        env.RUST_LOG = env.RUST_LOG || 'warn';
        break;
    }

    return env;
  }

  private async getOrCreateClient(language: string): Promise<LSPClient | null> {
    const existing = this.clients.get(language);
    if (existing) return existing;

    const config = this.languageConfigs.find(c => c.language === language);
    if (!config || !config.enabled) {
      return null;
    }

    // Check if LSP server is available
    const isAvailable = await this.checkLSPServerAvailable(config.command);
    if (!isAvailable) {
      console.warn(
        `LSP server '${config.command}' not found for ${language}. ` +
        `Install with: ${config.installationGuide || 'Check language documentation'}`
      );
      return null;
    }

    try {
      // Detect appropriate workspace root for this language
      const workspaceRoot = this.detectWorkspaceRoot(language, this.workspaceRoot);
      
      // Get language-specific environment
      const env = this.getLanguageEnvironment(language);

      const client = new LSPClient({
        command: config.command,
        args: config.args,
        workspaceRoot,
        initializationOptions: config.initializationOptions,
        env
      });

      await client.initialize();
      this.clients.set(language, client);
      
      console.log(
        `Successfully initialized ${language} LSP server (${config.command}) ` +
        `with workspace root: ${workspaceRoot}`
      );
      
      return client;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(
        `Failed to create LSP client for ${language}:`, errorMessage,
        `\nTry installing with: ${config.installationGuide || 'Check language documentation'}`
      );
      return null;
    }
  }

  // This method is now replaced by convertDiagnostic helper method

  private convertToLSPRange(range: { b0: number; b1: number }, content: string): LSPRange {
    return {
      start: this.byteOffsetToPosition(content, range.b0),
      end: this.byteOffsetToPosition(content, range.b1)
    };
  }

  private convertFromLSPRange(range: LSPRange, content: string): { b0: number; b1: number } {
    return {
      b0: this.positionToByteOffset(content, range.start),
      b1: this.positionToByteOffset(content, range.end)
    };
  }

  private byteOffsetToPosition(content: string, offset: number): LSPPosition {
    const lines = content.slice(0, offset).split('\n');
    return {
      line: lines.length - 1,
      character: lines[lines.length - 1].length
    };
  }

  private positionToByteOffset(content: string, position: LSPPosition): number {
    const lines = content.split('\n');
    let offset = 0;
    
    for (let i = 0; i < position.line && i < lines.length; i++) {
      offset += lines[i].length + 1; // +1 for newline
    }
    
    offset += Math.min(position.character, lines[position.line]?.length || 0);
    return offset;
  }

  private applyEdits(content: string, edits: LSPTextEdit[]): string {
    // Sort edits by position (descending) to apply from end to start
    const sortedEdits = [...edits].sort((a, b) => 
      this.positionToByteOffset(content, b.range.start) - this.positionToByteOffset(content, a.range.start)
    );

    let result = content;
    for (const edit of sortedEdits) {
      const startOffset = this.positionToByteOffset(result, edit.range.start);
      const endOffset = this.positionToByteOffset(result, edit.range.end);
      result = result.slice(0, startOffset) + edit.newText + result.slice(endOffset);
    }
    
    return result;
  }

  private getDefaultFormatOptions(language: string): any {
    const baseOptions = { 
      insertFinalNewline: true,
      trimTrailingWhitespace: true,
      trimFinalNewlines: true
    };

    const languageDefaults = {
      typescript: { 
        tabSize: 2, 
        insertSpaces: true,
        ...baseOptions
      },
      python: { 
        tabSize: 4, 
        insertSpaces: true,
        ...baseOptions
      },
      rust: { 
        tabSize: 4, 
        insertSpaces: true,
        ...baseOptions
      },
      go: { 
        tabSize: 4, 
        insertSpaces: false, // Go uses tabs by convention
        ...baseOptions
      }
    };
    
    return languageDefaults[language] || { tabSize: 2, insertSpaces: true, ...baseOptions };
  }

  /**
   * Generate a stable, deterministic symbol ID from LSP symbol information
   */
  private generateSymbolId(symbolItem: any): string {
    // Create a stable identifier based on symbol properties
    const name = symbolItem.name || 'unknown';
    const kind = symbolItem.kind || 'unknown';
    const uri = symbolItem.uri || '';
    const range = symbolItem.range;
    
    // Use file path + position + name + kind for stable ID
    const path = uri.replace(/^file:\/\//, '');
    const position = range ? `${range.start.line}:${range.start.character}` : '0:0';
    const composite = `${path}#${position}#${name}#${kind}`;
    
    // Generate a hash for consistent length
    let hash = 0;
    for (let i = 0; i < composite.length; i++) {
      const char = composite.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    return `sym_${Math.abs(hash).toString(16)}`;
  }

  /**
   * Convert LSP location to lens:// reference format
   */
  private convertLocationToRef(uri: string, range: LSPRange, repo_sha: string, source_hash: string): string {
    const path = uri.replace(/^file:\/\//, '');
    
    // Convert LSP line/character positions to byte offsets
    // For now, approximate using line/character (would need file content for exact byte offsets)
    const start = range.start.line * 100 + range.start.character; // Rough approximation
    const end = range.end.line * 100 + range.end.character;
    
    return LSPCacheManager.generateLensRef(repo_sha, path, source_hash, start, end);
  }

  /**
   * Convert diagnostic to our format with byte offset conversion
   */
  private convertDiagnostic(diag: LSPDiagnostic, content: string): any {
    const severityMap = { 1: 'error', 2: 'warning', 3: 'info', 4: 'hint' };
    return {
      range: {
        b0: this.positionToByteOffset(content, diag.range.start),
        b1: this.positionToByteOffset(content, diag.range.end)
      },
      severity: severityMap[diag.severity] || 'info',
      code: diag.code?.toString() || '',
      message: diag.message,
      source: 'lsp'
    };
  }

  /**
   * Extract repo_sha from file path or context
   */
  private extractRepoSha(path: string): string {
    // Try to extract from git context or use placeholder
    // In a real implementation, this would interface with git
    return 'unknown_repo_sha';
  }

  /**
   * Shutdown all LSP clients
   */
  async shutdown(): Promise<void> {
    const shutdownPromises = Array.from(this.clients.values()).map(client => 
      client.shutdown().catch(err => console.warn('LSP client shutdown error:', err))
    );
    
    await Promise.all(shutdownPromises);
    this.clients.clear();
  }
}

// Global LSP SPI service instance
export const globalLSPService = new LSPSPIService();