/**
 * LSP Client for Lens SPI system
 * Manages LSP server connections and provides typed API
 */

import { spawn, ChildProcess } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import { LensTracer } from '../telemetry/tracer.js';

interface LSPServerConfig {
  command: string;
  args: string[];
  workspaceRoot: string;
  initializationOptions?: any;
  capabilities?: any;
  env?: Record<string, string>;
}

export interface LSPPosition {
  line: number;
  character: number;
}

export interface LSPRange {
  start: LSPPosition;
  end: LSPPosition;
}

export interface LSPDiagnostic {
  range: LSPRange;
  severity: 1 | 2 | 3 | 4; // Error, Warning, Information, Hint
  code?: string | number;
  message: string;
}

export interface LSPTextEdit {
  range: LSPRange;
  newText: string;
}

export interface LSPWorkspaceEdit {
  changes?: { [uri: string]: LSPTextEdit[] };
}

export interface LSPCodeAction {
  title: string;
  kind?: string;
  edit?: LSPWorkspaceEdit;
  data?: any;
}

export class LSPClient {
  private process: ChildProcess | null = null;
  private messageId = 0;
  private pendingRequests = new Map<number, {
    resolve: (value: any) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }>();
  private buffer = '';
  private initialized = false;
  private shutdownRequested = false;

  constructor(private config: LSPServerConfig) {}

  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('lsp_client_initialize');
    try {
      // Start LSP server process
      this.process = spawn(this.config.command, this.config.args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: this.config.workspaceRoot,
        env: this.config.env || process.env
      });

      if (!this.process.stdout || !this.process.stdin || !this.process.stderr) {
        throw new Error('Failed to create LSP server process');
      }

      // Handle process output
      this.process.stdout.on('data', (data) => {
        this.handleMessage(data.toString());
      });

      this.process.stderr.on('data', (data) => {
        console.warn('LSP server stderr:', data.toString());
      });

      this.process.on('exit', (code) => {
        if (!this.shutdownRequested && code !== 0) {
          console.error(`LSP server exited with code ${code}`);
        }
      });

      // Send initialize request
      const initializeParams = {
        processId: process.pid,
        rootUri: `file://${this.config.workspaceRoot}`,
        capabilities: {
          textDocument: {
            diagnostics: { dynamicRegistration: false },
            formatting: { dynamicRegistration: false },
            rangeFormatting: { dynamicRegistration: false },
            selectionRange: { dynamicRegistration: false },
            foldingRange: { dynamicRegistration: false },
            rename: { dynamicRegistration: false, prepareSupport: true },
            codeAction: { dynamicRegistration: false },
            callHierarchy: { dynamicRegistration: false },
            typeHierarchy: { dynamicRegistration: false }
          },
          workspace: {
            workspaceEdit: {
              documentChanges: true,
              resourceOperations: ['create', 'rename', 'delete']
            }
          }
        },
        initializationOptions: this.config.initializationOptions
      };

      const initResponse = await this.sendRequest('initialize', initializeParams, 10000);
      
      // Send initialized notification
      this.sendNotification('initialized', {});
      
      this.initialized = true;
      
      span.setAttributes({
        success: true,
        server_capabilities: JSON.stringify(initResponse.capabilities || {}),
        server_info: initResponse.serverInfo?.name || 'unknown'
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }

  async shutdown(): Promise<void> {
    if (!this.initialized || this.shutdownRequested) return;
    
    this.shutdownRequested = true;
    
    try {
      // Send shutdown request
      await this.sendRequest('shutdown', null, 5000);
      
      // Send exit notification
      this.sendNotification('exit', null);
      
      // Clean up pending requests
      for (const [id, request] of this.pendingRequests) {
        clearTimeout(request.timeout);
        request.reject(new Error('LSP client shutting down'));
      }
      this.pendingRequests.clear();
      
      // Kill process if still running
      if (this.process && !this.process.killed) {
        this.process.kill('SIGTERM');
        
        // Force kill after timeout
        setTimeout(() => {
          if (this.process && !this.process.killed) {
            this.process.kill('SIGKILL');
          }
        }, 2000);
      }
      
    } catch (error) {
      console.warn('Error during LSP shutdown:', error);
    }
  }

  // Diagnostic methods
  async diagnostics(uri: string): Promise<LSPDiagnostic[]> {
    // For simplicity, we'll return diagnostics through document synchronization
    // Real implementation would use publishDiagnostics notifications
    return [];
  }

  // Formatting methods
  async formatDocument(uri: string, options: any): Promise<LSPTextEdit[]> {
    const params = {
      textDocument: { uri },
      options
    };
    
    const result = await this.sendRequest('textDocument/formatting', params);
    return result || [];
  }

  async formatRange(uri: string, range: LSPRange, options: any): Promise<LSPTextEdit[]> {
    const params = {
      textDocument: { uri },
      range,
      options
    };
    
    const result = await this.sendRequest('textDocument/rangeFormatting', params);
    return result || [];
  }

  // Selection range methods
  async selectionRanges(uri: string, positions: LSPPosition[]): Promise<any[]> {
    const params = {
      textDocument: { uri },
      positions
    };
    
    const result = await this.sendRequest('textDocument/selectionRange', params);
    return result || [];
  }

  // Folding range methods
  async foldingRanges(uri: string): Promise<any[]> {
    const params = {
      textDocument: { uri }
    };
    
    const result = await this.sendRequest('textDocument/foldingRange', params);
    return result || [];
  }

  // Rename methods
  async prepareRename(uri: string, position: LSPPosition): Promise<any> {
    const params = {
      textDocument: { uri },
      position
    };
    
    try {
      const result = await this.sendRequest('textDocument/prepareRename', params);
      return result;
    } catch (error) {
      return null; // Rename not supported at this position
    }
  }

  async rename(uri: string, position: LSPPosition, newName: string): Promise<LSPWorkspaceEdit | null> {
    const params = {
      textDocument: { uri },
      position,
      newName
    };
    
    const result = await this.sendRequest('textDocument/rename', params);
    return result || null;
  }

  // Code actions methods
  async codeActions(uri: string, range: LSPRange, context: any): Promise<LSPCodeAction[]> {
    const params = {
      textDocument: { uri },
      range,
      context
    };
    
    const result = await this.sendRequest('textDocument/codeAction', params);
    return result || [];
  }

  // Hierarchy methods
  async prepareCallHierarchy(uri: string, position: LSPPosition): Promise<any[]> {
    const params = {
      textDocument: { uri },
      position
    };
    
    const result = await this.sendRequest('textDocument/prepareCallHierarchy', params);
    return result || [];
  }

  async callHierarchyIncoming(item: any): Promise<any[]> {
    const result = await this.sendRequest('callHierarchy/incomingCalls', { item });
    return result || [];
  }

  async callHierarchyOutgoing(item: any): Promise<any[]> {
    const result = await this.sendRequest('callHierarchy/outgoingCalls', { item });
    return result || [];
  }

  // Document synchronization
  didOpen(uri: string, languageId: string, version: number, text: string): void {
    const params = {
      textDocument: {
        uri,
        languageId,
        version,
        text
      }
    };
    
    this.sendNotification('textDocument/didOpen', params);
  }

  didChange(uri: string, version: number, changes: any[]): void {
    const params = {
      textDocument: {
        uri,
        version
      },
      contentChanges: changes
    };
    
    this.sendNotification('textDocument/didChange', params);
  }

  didClose(uri: string): void {
    const params = {
      textDocument: { uri }
    };
    
    this.sendNotification('textDocument/didClose', params);
  }

  private async sendRequest(method: string, params: any, timeoutMs: number = 30000): Promise<any> {
    if (!this.initialized && method !== 'initialize') {
      throw new Error('LSP client not initialized');
    }

    const id = ++this.messageId;
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Request ${method} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pendingRequests.set(id, {
        resolve,
        reject,
        timeout
      });

      const message = JSON.stringify(request);
      const content = `Content-Length: ${Buffer.byteLength(message, 'utf8')}\r\n\r\n${message}`;
      
      if (this.process?.stdin) {
        this.process.stdin.write(content);
      } else {
        clearTimeout(timeout);
        this.pendingRequests.delete(id);
        reject(new Error('LSP server process not available'));
      }
    });
  }

  private sendNotification(method: string, params: any): void {
    if (!this.initialized && method !== 'initialized' && method !== 'exit') {
      return;
    }

    const notification = {
      jsonrpc: '2.0',
      method,
      params
    };

    const message = JSON.stringify(notification);
    const content = `Content-Length: ${Buffer.byteLength(message, 'utf8')}\r\n\r\n${message}`;
    
    if (this.process?.stdin) {
      this.process.stdin.write(content);
    }
  }

  private handleMessage(data: string): void {
    this.buffer += data;
    
    while (true) {
      const headerEnd = this.buffer.indexOf('\r\n\r\n');
      if (headerEnd === -1) break;
      
      const header = this.buffer.slice(0, headerEnd);
      const contentLengthMatch = header.match(/Content-Length: (\d+)/);
      
      if (!contentLengthMatch) {
        this.buffer = this.buffer.slice(headerEnd + 4);
        continue;
      }
      
      const contentLength = parseInt(contentLengthMatch[1], 10);
      const messageStart = headerEnd + 4;
      const messageEnd = messageStart + contentLength;
      
      if (this.buffer.length < messageEnd) break;
      
      const messageContent = this.buffer.slice(messageStart, messageEnd);
      this.buffer = this.buffer.slice(messageEnd);
      
      try {
        const message = JSON.parse(messageContent);
        this.handleParsedMessage(message);
      } catch (error) {
        console.error('Failed to parse LSP message:', error, messageContent);
      }
    }
  }

  private handleParsedMessage(message: any): void {
    if (message.id !== undefined && this.pendingRequests.has(message.id)) {
      // Response to a request
      const request = this.pendingRequests.get(message.id)!;
      this.pendingRequests.delete(message.id);
      
      clearTimeout(request.timeout);
      
      if (message.error) {
        request.reject(new Error(`LSP error ${message.error.code}: ${message.error.message}`));
      } else {
        request.resolve(message.result);
      }
    } else if (message.method) {
      // Notification or request from server
      this.handleNotification(message.method, message.params);
    }
  }

  private handleNotification(method: string, params: any): void {
    // Handle server notifications (like publishDiagnostics)
    switch (method) {
      case 'textDocument/publishDiagnostics':
        // Store diagnostics for later retrieval
        break;
      default:
        // Ignore unknown notifications
        break;
    }
  }
}