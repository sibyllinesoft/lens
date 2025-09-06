/**
 * LSP-Assist Sidecar System
 * Local LSP servers per language running in no-editor mode
 * Harvests: definitions, resolved imports/aliases, type info, project graph
 * Emits compact Hints.ndjson per shard with TTL management
 */

import { spawn, ChildProcess } from 'child_process';
import { existsSync, mkdirSync, writeFileSync, readFileSync } from 'fs';
import { join, dirname } from 'path';
import type { 
  LSPSidecarConfig, 
  LSPHint, 
  WorkspaceConfig,
  SupportedLanguage,
  LSPCapabilities
} from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

interface LSPMessage {
  jsonrpc: string;
  id?: number;
  method: string;
  params?: any;
  result?: any;
  error?: any;
}

interface LSPSymbol {
  name: string;
  kind: number;
  location: {
    uri: string;
    range: {
      start: { line: number; character: number };
      end: { line: number; character: number };
    };
  };
  containerName?: string;
}

export class LSPSidecar {
  private lspProcess?: ChildProcess;
  private messageId = 0;
  private pendingRequests = new Map<number, { resolve: Function; reject: Function }>();
  private initialized = false;
  private harvestCache = new Map<string, { hints: LSPHint[]; timestamp: Date }>();

  constructor(
    private config: LSPSidecarConfig,
    private repoSha: string,
    private workspaceRoot: string
  ) {}

  /**
   * Initialize LSP sidecar with workspace configuration
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('lsp_sidecar_init', {
      'lsp.language': this.config.language,
      'lsp.server': this.config.lsp_server,
      'workspace.root': this.workspaceRoot,
    });

    try {
      await this.startLSPServer();
      await this.performInitialization();
      this.initialized = true;
      
      span.setAttributes({ success: true });
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw new Error(`Failed to initialize LSP sidecar: ${error}`);
    } finally {
      span.end();
    }
  }

  /**
   * Start the appropriate LSP server for the language
   */
  private async startLSPServer(): Promise<void> {
    const serverCommands = this.getServerCommand(this.config.language);
    
    this.lspProcess = spawn(serverCommands.command, serverCommands.args, {
      cwd: this.workspaceRoot,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, ...serverCommands.env },
    });

    if (!this.lspProcess.stdout || !this.lspProcess.stdin) {
      throw new Error('Failed to create LSP process stdio');
    }

    // Handle LSP messages
    this.setupMessageHandlers();
  }

  /**
   * Get LSP server command and arguments for each language
   */
  private getServerCommand(language: SupportedLanguage): {
    command: string;
    args: string[];
    env: Record<string, string>;
  } {
    const commands = {
      typescript: {
        command: 'typescript-language-server',
        args: ['--stdio', '--tsserver-path', 'tsserver'],
        env: { NODE_OPTIONS: '--max-old-space-size=4096' },
      },
      python: {
        command: 'pyright-langserver',
        args: ['--stdio'],
        env: {},
      },
      rust: {
        command: 'rust-analyzer',
        args: [],
        env: {},
      },
      go: {
        command: 'gopls',
        args: ['serve'],
        env: {},
      },
      java: {
        command: 'jdtls',
        args: ['-data', join(this.workspaceRoot, '.jdt-workspace')],
        env: {},
      },
      bash: {
        command: 'bash-language-server',
        args: ['start'],
        env: {},
      },
    };

    return commands[language] || commands.typescript;
  }

  /**
   * Setup message handlers for LSP communication
   */
  private setupMessageHandlers(): void {
    if (!this.lspProcess?.stdout || !this.lspProcess?.stdin) return;

    let buffer = '';
    this.lspProcess.stdout.on('data', (data: Buffer) => {
      buffer += data.toString();
      
      while (true) {
        const headerEndIndex = buffer.indexOf('\r\n\r\n');
        if (headerEndIndex === -1) break;

        const header = buffer.substring(0, headerEndIndex);
        const contentLengthMatch = header.match(/Content-Length: (\d+)/);
        
        if (!contentLengthMatch) break;
        
        const contentLength = parseInt(contentLengthMatch[1]);
        const messageStart = headerEndIndex + 4;
        
        if (buffer.length < messageStart + contentLength) break;
        
        const messageContent = buffer.substring(messageStart, messageStart + contentLength);
        buffer = buffer.substring(messageStart + contentLength);
        
        try {
          const message: LSPMessage = JSON.parse(messageContent);
          this.handleLSPMessage(message);
        } catch (error) {
          console.error('Failed to parse LSP message:', error);
        }
      }
    });

    this.lspProcess.stderr?.on('data', (data: Buffer) => {
      console.error('LSP stderr:', data.toString());
    });

    this.lspProcess.on('exit', (code, signal) => {
      console.log(`LSP process exited with code ${code}, signal ${signal}`);
    });
  }

  /**
   * Handle incoming LSP messages
   */
  private handleLSPMessage(message: LSPMessage): void {
    if (message.id !== undefined) {
      // Response to our request
      const pending = this.pendingRequests.get(message.id);
      if (pending) {
        this.pendingRequests.delete(message.id);
        if (message.error) {
          pending.reject(new Error(message.error.message));
        } else {
          pending.resolve(message.result);
        }
      }
    } else if (message.method) {
      // Notification from server
      this.handleNotification(message);
    }
  }

  /**
   * Handle LSP notifications
   */
  private handleNotification(message: LSPMessage): void {
    switch (message.method) {
      case 'window/logMessage':
        console.log(`LSP Log: ${message.params?.message}`);
        break;
      case 'textDocument/publishDiagnostics':
        // Could be used for error tracking
        break;
    }
  }

  /**
   * Send LSP request and wait for response
   */
  private sendRequest(method: string, params?: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.lspProcess?.stdin) {
        reject(new Error('LSP process not available'));
        return;
      }

      const id = ++this.messageId;
      const message: LSPMessage = {
        jsonrpc: '2.0',
        id,
        method,
        params,
      };

      this.pendingRequests.set(id, { resolve, reject });
      
      const content = JSON.stringify(message);
      const header = `Content-Length: ${Buffer.byteLength(content, 'utf8')}\r\n\r\n`;
      
      this.lspProcess.stdin.write(header + content);

      // Timeout after 30 seconds
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`LSP request timeout: ${method}`));
        }
      }, 30000);
    });
  }

  /**
   * Perform LSP initialization handshake
   */
  private async performInitialization(): Promise<void> {
    // Initialize request
    const initResult = await this.sendRequest('initialize', {
      processId: process.pid,
      clientInfo: { name: 'lens-lsp-sidecar', version: '1.0.0' },
      rootUri: `file://${this.workspaceRoot}`,
      capabilities: {
        textDocument: {
          definition: { linkSupport: true },
          references: { includeDeclaration: true },
          hover: { contentFormat: ['plaintext', 'markdown'] },
          documentSymbol: { hierarchicalDocumentSymbolSupport: true },
        },
        workspace: {
          symbol: {},
          didChangeConfiguration: {},
        },
      },
      workspaceFolders: [{
        uri: `file://${this.workspaceRoot}`,
        name: 'lens-workspace',
      }],
    });

    // Store server capabilities
    this.config.capabilities = {
      definition: initResult.capabilities?.definitionProvider || false,
      references: initResult.capabilities?.referencesProvider || false,
      hover: initResult.capabilities?.hoverProvider || false,
      completion: initResult.capabilities?.completionProvider || false,
      rename: initResult.capabilities?.renameProvider || false,
      workspace_symbols: initResult.capabilities?.workspaceSymbolProvider || false,
    };

    // Send initialized notification
    this.sendNotification('initialized', {});

    console.log(`LSP sidecar initialized for ${this.config.language}`);
  }

  /**
   * Send LSP notification (no response expected)
   */
  private sendNotification(method: string, params?: any): void {
    if (!this.lspProcess?.stdin) return;

    const message: LSPMessage = {
      jsonrpc: '2.0',
      method,
      params,
    };

    const content = JSON.stringify(message);
    const header = `Content-Length: ${Buffer.byteLength(content, 'utf8')}\r\n\r\n`;
    
    this.lspProcess.stdin.write(header + content);
  }

  /**
   * Harvest LSP hints for a repository snapshot
   * *** STEP 2: COMPLETE HINT HARVESTING IMPLEMENTATION ***
   */
  async harvestHints(filePaths: string[], forceRefresh = false): Promise<LSPHint[]> {
    const span = LensTracer.createChildSpan('lsp_harvest', {
      'repo.sha': this.repoSha,
      'files.count': filePaths.length,
      'force_refresh': forceRefresh,
    });

    try {
      if (!this.initialized) {
        console.warn('LSP sidecar not initialized, cannot harvest hints');
        return [];
      }

      const cacheKey = `${this.repoSha}:${filePaths.slice(0, 10).join(':')}`;
      
      // Check cache first (unless forcing refresh)
      if (!forceRefresh && this.harvestCache.has(cacheKey)) {
        const cached = this.harvestCache.get(cacheKey)!;
        const ageHours = (Date.now() - cached.timestamp.getTime()) / (1000 * 60 * 60);
        
        if (ageHours < this.config.harvest_ttl_hours) {
          console.log(`📋 Using cached LSP hints: ${cached.hints.length} hints`);
          span.setAttributes({ 
            success: true, 
            cache_hit: true,
            hints_count: cached.hints.length,
          });
          return cached.hints;
        }
      }

      console.log(`🌾 Harvesting LSP hints for ${filePaths.length} files...`);
      const allHints: LSPHint[] = [];

      // Step 1: Get workspace symbols if supported
      if (this.config.capabilities.workspace_symbols) {
        try {
          console.log('📡 Fetching workspace symbols...');
          const workspaceSymbols = await this.getWorkspaceSymbols();
          const workspaceHints = this.convertWorkspaceSymbolsToHints(workspaceSymbols);
          allHints.push(...workspaceHints);
          console.log(`📊 Found ${workspaceHints.length} workspace symbols`);
        } catch (error) {
          console.warn('Failed to get workspace symbols:', error);
        }
      }

      // Step 2: Process files in smaller batches to avoid overwhelming LSP
      const batchSize = 50;
      const fileBatches = [];
      for (let i = 0; i < filePaths.length; i += batchSize) {
        fileBatches.push(filePaths.slice(i, i + batchSize));
      }

      let processedFiles = 0;
      for (const batch of fileBatches) {
        for (const filePath of batch) {
          if (!this.shouldProcessFile(filePath)) continue;

          try {
            const fileHints = await this.getFileHints(filePath);
            allHints.push(...fileHints);
            processedFiles++;
            
            // Log progress every 20 files
            if (processedFiles % 20 === 0) {
              console.log(`📄 Processed ${processedFiles}/${filePaths.length} files, ${allHints.length} hints so far`);
            }
          } catch (error) {
            console.warn(`Failed to get hints for ${filePath}:`, error);
          }
        }
        
        // Small delay between batches to prevent LSP overload
        if (fileBatches.length > 1) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }

      // Step 3: Deduplicate and enrich hints
      console.log(`🔄 Processing ${allHints.length} raw hints...`);
      const uniqueHints = this.deduplicateHints(allHints);
      console.log(`✂️ After deduplication: ${uniqueHints.length} unique hints`);
      
      const enrichedHints = await this.enrichHints(uniqueHints);
      console.log(`🔧 After enrichment: ${enrichedHints.length} enriched hints`);

      // Step 4: Cache results
      this.harvestCache.set(cacheKey, {
        hints: enrichedHints,
        timestamp: new Date(),
      });

      // Step 5: Write to shard file for persistence
      await this.writeHintsToShard(enrichedHints);

      console.log(`✅ LSP harvest complete: ${enrichedHints.length} hints saved to shard`);

      span.setAttributes({
        success: true,
        cache_hit: false,
        hints_count: enrichedHints.length,
        files_processed: processedFiles,
      });

      return enrichedHints;
      
    } catch (error) {
      console.error('LSP harvest failed:', error);
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      
      // Return empty array instead of throwing - LSP is optional
      return [];
    } finally {
      span.end();
    }
  }

  /**
   * Get workspace symbols from LSP server
   */
  private async getWorkspaceSymbols(query = ''): Promise<LSPSymbol[]> {
    if (!this.config.capabilities.workspace_symbols) return [];

    try {
      const result = await this.sendRequest('workspace/symbol', { query });
      return Array.isArray(result) ? result : [];
    } catch (error) {
      console.warn('Failed to get workspace symbols:', error);
      return [];
    }
  }

  /**
   * Get hints for a specific file
   */
  private async getFileHints(filePath: string): Promise<LSPHint[]> {
    const hints: LSPHint[] = [];
    const fileUri = `file://${filePath}`;

    // Get document symbols
    try {
      const symbols = await this.sendRequest('textDocument/documentSymbol', {
        textDocument: { uri: fileUri },
      });
      
      if (Array.isArray(symbols)) {
        hints.push(...this.convertDocumentSymbolsToHints(symbols, filePath));
      }
    } catch (error) {
      console.warn(`Failed to get document symbols for ${filePath}:`, error);
    }

    return hints;
  }

  /**
   * Convert LSP workspace symbols to hints
   */
  private convertWorkspaceSymbolsToHints(symbols: LSPSymbol[]): LSPHint[] {
    return symbols.map((symbol, index) => ({
      symbol_id: `ws_${this.repoSha}_${index}`,
      name: symbol.name,
      kind: this.convertLSPSymbolKind(symbol.kind),
      file_path: symbol.location.uri.replace('file://', ''),
      line: symbol.location.range.start.line + 1,
      col: symbol.location.range.start.character,
      definition_uri: symbol.location.uri,
      signature: `${this.getSymbolKindName(symbol.kind)} ${symbol.name}`,
      aliases: [],
      resolved_imports: [],
      references_count: 0, // Will be enriched later
    }));
  }

  /**
   * Convert LSP document symbols to hints
   */
  private convertDocumentSymbolsToHints(symbols: any[], filePath: string): LSPHint[] {
    const hints: LSPHint[] = [];
    let symbolIndex = 0;

    const processSymbol = (symbol: any, prefix = ''): void => {
      hints.push({
        symbol_id: `doc_${this.repoSha}_${symbolIndex++}`,
        name: symbol.name,
        kind: this.convertLSPSymbolKind(symbol.kind),
        file_path: filePath,
        line: symbol.range?.start?.line + 1 || symbol.selectionRange?.start?.line + 1 || 1,
        col: symbol.range?.start?.character || symbol.selectionRange?.start?.character || 0,
        signature: symbol.detail || `${this.getSymbolKindName(symbol.kind)} ${symbol.name}`,
        aliases: [],
        resolved_imports: [],
        references_count: 0,
      });

      // Process children recursively
      if (symbol.children) {
        symbol.children.forEach((child: any) => {
          processSymbol(child, `${prefix}${symbol.name}.`);
        });
      }
    };

    symbols.forEach(processSymbol);
    return hints;
  }

  /**
   * Convert LSP symbol kind number to our SymbolKind
   */
  private convertLSPSymbolKind(lspKind: number): any {
    const kindMap: { [key: number]: string } = {
      1: 'file',
      2: 'module', 
      3: 'namespace',
      4: 'package',
      5: 'class',
      6: 'method',
      7: 'property',
      8: 'field',
      9: 'constructor',
      10: 'enum',
      11: 'interface',
      12: 'function',
      13: 'variable',
      14: 'constant',
      15: 'string',
      16: 'number',
      17: 'boolean',
      18: 'array',
      19: 'object',
      20: 'key',
      21: 'null',
      22: 'enumMember',
      23: 'struct',
      24: 'event',
      25: 'operator',
      26: 'typeParameter',
    };

    const lspKindName = kindMap[lspKind] || 'unknown';
    
    // Map to our simplified SymbolKind
    const mapping: { [key: string]: string } = {
      'class': 'class',
      'method': 'method',
      'function': 'function',
      'variable': 'variable',
      'constant': 'constant',
      'property': 'property',
      'field': 'property',
      'interface': 'interface',
      'enum': 'enum',
      'type': 'type',
    };

    return mapping[lspKindName] || 'variable';
  }

  /**
   * Get human-readable symbol kind name
   */
  private getSymbolKindName(kind: number): string {
    const names: { [key: number]: string } = {
      5: 'class',
      6: 'method',
      11: 'interface',
      12: 'function',
      13: 'variable',
      14: 'constant',
    };
    return names[kind] || 'symbol';
  }

  /**
   * Check if file should be processed based on workspace config
   */
  private shouldProcessFile(filePath: string): boolean {
    const relativePath = filePath.replace(this.workspaceRoot, '').replace(/^\//, '');
    
    // Check exclude patterns
    for (const pattern of this.config.workspace_config.exclude_patterns) {
      if (this.matchesPattern(relativePath, pattern)) {
        return false;
      }
    }

    // Check include patterns
    if (this.config.workspace_config.include_patterns.length > 0) {
      return this.config.workspace_config.include_patterns.some(pattern =>
        this.matchesPattern(relativePath, pattern)
      );
    }

    return true;
  }

  /**
   * Simple pattern matching for include/exclude
   */
  private matchesPattern(path: string, pattern: string): boolean {
    const regex = new RegExp(
      pattern
        .replace(/\./g, '\\.')
        .replace(/\*/g, '.*')
        .replace(/\?/g, '.')
    );
    return regex.test(path);
  }

  /**
   * Deduplicate hints by symbol name and location
   */
  private deduplicateHints(hints: LSPHint[]): LSPHint[] {
    const seen = new Set<string>();
    return hints.filter(hint => {
      const key = `${hint.name}:${hint.file_path}:${hint.line}:${hint.col}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  /**
   * Enrich hints with additional information
   */
  private async enrichHints(hints: LSPHint[]): Promise<LSPHint[]> {
    // Add reference counts, resolve aliases, etc.
    for (const hint of hints) {
      try {
        // Get references count
        if (this.config.capabilities.references) {
          const refs = await this.getReferences(hint.file_path, hint.line - 1, hint.col);
          hint.references_count = refs.length;
        }

        // Resolve imports and aliases (simplified)
        hint.aliases = await this.resolveAliases(hint);
        hint.resolved_imports = await this.resolveImports(hint.file_path);
      } catch (error) {
        console.warn(`Failed to enrich hint ${hint.name}:`, error);
      }
    }

    return hints;
  }

  /**
   * Get references for a symbol at a specific location
   */
  private async getReferences(filePath: string, line: number, character: number): Promise<any[]> {
    try {
      const result = await this.sendRequest('textDocument/references', {
        textDocument: { uri: `file://${filePath}` },
        position: { line, character },
        context: { includeDeclaration: true },
      });
      
      return Array.isArray(result) ? result : [];
    } catch (error) {
      return [];
    }
  }

  /**
   * Resolve aliases for a symbol
   */
  private async resolveAliases(hint: LSPHint): Promise<string[]> {
    // Simplified alias resolution - would need language-specific logic
    const aliases: string[] = [];
    
    // Common alias patterns
    if (hint.name.includes('_')) {
      aliases.push(hint.name.replace(/_/g, ''));
    }
    
    return aliases;
  }

  /**
   * Resolve imports for a file
   */
  private async resolveImports(filePath: string): Promise<string[]> {
    try {
      const content = readFileSync(filePath, 'utf8');
      const imports: string[] = [];
      
      // Simple import extraction - would need proper AST parsing
      const importRegex = /import\s+.*?from\s+['"]([^'"]*)['"]/g;
      let match;
      while ((match = importRegex.exec(content)) !== null) {
        imports.push(match[1]);
      }
      
      return imports;
    } catch (error) {
      return [];
    }
  }

  /**
   * Write hints to shard NDJSON file
   */
  private async writeHintsToShard(hints: LSPHint[]): Promise<void> {
    const hintsDir = join(this.workspaceRoot, '.lens', 'hints');
    if (!existsSync(hintsDir)) {
      mkdirSync(hintsDir, { recursive: true });
    }

    const shardFile = join(hintsDir, `${this.repoSha}_${this.config.language}.ndjson`);
    const ndjsonContent = hints.map(hint => JSON.stringify(hint)).join('\n');
    
    writeFileSync(shardFile, ndjsonContent, 'utf8');
    
    console.log(`Wrote ${hints.length} LSP hints to ${shardFile}`);
  }

  /**
   * Load hints from shard file
   */
  async loadHintsFromShard(): Promise<LSPHint[]> {
    const hintsDir = join(this.workspaceRoot, '.lens', 'hints');
    const shardFile = join(hintsDir, `${this.repoSha}_${this.config.language}.ndjson`);
    
    if (!existsSync(shardFile)) return [];

    try {
      const content = readFileSync(shardFile, 'utf8');
      return content.split('\n')
        .filter(line => line.trim())
        .map(line => JSON.parse(line) as LSPHint);
    } catch (error) {
      console.error(`Failed to load hints from ${shardFile}:`, error);
      return [];
    }
  }

  /**
   * Check if hints need refreshing based on TTL and pressure
   */
  shouldRefreshHints(): boolean {
    const cacheEntries = Array.from(this.harvestCache.values());
    if (cacheEntries.length === 0) return true;

    const avgAgeHours = cacheEntries.reduce((sum, entry) => {
      const ageHours = (Date.now() - entry.timestamp.getTime()) / (1000 * 60 * 60);
      return sum + ageHours;
    }, 0) / cacheEntries.length;

    const memoryPressure = process.memoryUsage().heapUsed / 1024 / 1024; // MB

    return avgAgeHours > this.config.harvest_ttl_hours || 
           memoryPressure > this.config.pressure_threshold;
  }

  /**
   * Cleanup cache entries older than TTL
   */
  cleanupCache(): void {
    const now = Date.now();
    const ttlMs = this.config.harvest_ttl_hours * 60 * 60 * 1000;

    for (const [key, entry] of this.harvestCache) {
      if (now - entry.timestamp.getTime() > ttlMs) {
        this.harvestCache.delete(key);
      }
    }
  }

  /**
   * Get sidecar statistics
   */
  getStats(): {
    initialized: boolean;
    cache_entries: number;
    total_hints: number;
    capabilities: LSPCapabilities;
  } {
    const totalHints = Array.from(this.harvestCache.values())
      .reduce((sum, entry) => sum + entry.hints.length, 0);

    return {
      initialized: this.initialized,
      cache_entries: this.harvestCache.size,
      total_hints: totalHints,
      capabilities: this.config.capabilities,
    };
  }

  /**
   * Shutdown LSP sidecar
   */
  async shutdown(): Promise<void> {
    if (this.lspProcess) {
      this.lspProcess.kill('SIGTERM');
      
      // Wait for graceful shutdown
      await new Promise<void>((resolve) => {
        if (!this.lspProcess) {
          resolve();
          return;
        }

        const timeout = setTimeout(() => {
          this.lspProcess?.kill('SIGKILL');
          resolve();
        }, 5000);

        this.lspProcess.on('exit', () => {
          clearTimeout(timeout);
          resolve();
        });
      });
    }

    this.harvestCache.clear();
    this.pendingRequests.clear();
    this.initialized = false;
    
    console.log(`LSP sidecar shutdown for ${this.config.language}`);
  }
}