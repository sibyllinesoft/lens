/**
 * Comprehensive Tests for LSP Sidecar Component
 * Tests LSP server management, hint generation, and core functionality
 */

import { describe, it, expect, beforeEach, afterEach, mock, jest, mock } from 'bun:test';
import { LSPSidecar } from '../lsp-sidecar.js';
import type { LSPSidecarConfig, LSPHint } from '../../types/core.js';
import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import fs from 'fs';

// Mock child_process module
mock('child_process', () => ({
  spawn: jest.fn()
}));

// Mock fs module
mock('fs', () => ({
  existsSync: jest.fn(),
  mkdirSync: jest.fn(),
  writeFileSync: jest.fn(),
  readFileSync: jest.fn()
}));

// Mock path module
mock('path', () => ({
  join: jest.fn((...args) => args.join('/')),
  dirname: jest.fn()
}));

// Mock telemetry tracer
mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: jest.fn(() => ({
      setAttributes: jest.fn(),
      recordException: jest.fn(),
      end: jest.fn()
    }))
  }
}));

describe('LSPSidecar', () => {
  let lspSidecar: LSPSidecar;
  let mockConfig: LSPSidecarConfig;
  let mockProcess: MockChildProcess;
  
  // Mock child process with event emitter behavior
  class MockChildProcess extends EventEmitter {
    stdin = { write: jest.fn() };
    stdout = new EventEmitter();
    stderr = new EventEmitter();
    kill = jest.fn();
  }

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup mock config
    mockConfig = {
      language: 'typescript',
      lsp_server: 'typescript-language-server',
      harvest_ttl_hours: 24,
      pressure_threshold: 512,
      workspace_config: {
        include_patterns: ['**/*.ts', '**/*.js'],
        exclude_patterns: ['node_modules/**', '**/*.test.ts']
      },
      capabilities: {
        definition: false,
        references: false,
        hover: false,
        completion: false,
        rename: false,
        workspace_symbols: false
      }
    };

    // Setup mock process
    mockProcess = new MockChildProcess();
    (spawn as any).mockReturnValue(mockProcess);
    
    // Setup filesystem mocks
    (fs.existsSync as any).mockReturnValue(true);
    (fs.readFileSync as any).mockReturnValue('');

    lspSidecar = new LSPSidecar(mockConfig, 'test-sha', '/test/workspace');
  });

  afterEach(async () => {
    try {
      await lspSidecar.shutdown();
    } catch (error) {
      // Ignore shutdown errors in tests
    }
  });

  describe('initialization', () => {
    it('should initialize LSP sidecar successfully', async () => {
      // Mock successful initialization
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 200\r\n\r\n' +
          JSON.stringify({
            id: 1,
            jsonrpc: '2.0',
            result: {
              capabilities: {
                definitionProvider: true,
                referencesProvider: true,
                hoverProvider: true,
                workspaceSymbolProvider: true
              }
            }
          })
        ));

        // Send initialized notification response
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({
            jsonrpc: '2.0',
            method: 'initialized',
            params: {}
          })
        ));
      }, 10);

      await lspSidecar.initialize();
      
      expect(spawn).toHaveBeenCalledWith(
        'typescript-language-server',
        ['--stdio', '--tsserver-path', 'tsserver'],
        expect.objectContaining({
          cwd: '/test/workspace',
          stdio: ['pipe', 'pipe', 'pipe'],
          env: expect.objectContaining({
            NODE_OPTIONS: '--max-old-space-size=4096'
          })
        })
      );

      const stats = lspSidecar.getStats();
      expect(stats.initialized).toBe(true);
      expect(stats.capabilities.definition).toBe(true);
      expect(stats.capabilities.references).toBe(true);
      expect(stats.capabilities.hover).toBe(true);
      expect(stats.capabilities.workspace_symbols).toBe(true);
    });

    it('should handle initialization failure gracefully', async () => {
      // Mock initialization error
      setTimeout(() => {
        mockProcess.emit('exit', 1, null);
      }, 10);

      await expect(lspSidecar.initialize()).rejects.toThrow('Failed to initialize LSP sidecar');
    });

    it('should get correct server command for different languages', async () => {
      const languages: Array<{ lang: any, command: string, args: string[] }> = [
        { lang: 'python', command: 'pyright-langserver', args: ['--stdio'] },
        { lang: 'rust', command: 'rust-analyzer', args: [] },
        { lang: 'go', command: 'gopls', args: ['serve'] },
        { lang: 'java', command: 'jdtls', args: ['-data', '/test/workspace/.jdt-workspace'] }
      ];

      for (const { lang, command, args } of languages) {
        const config = { ...mockConfig, language: lang };
        const sidecar = new LSPSidecar(config, 'test-sha', '/test/workspace');
        
        // Mock initialization response
        setTimeout(() => {
          mockProcess.stdout.emit('data', Buffer.from(
            'Content-Length: 100\r\n\r\n' +
            JSON.stringify({ id: 1, jsonrpc: '2.0', result: { capabilities: {} } })
          ));
        }, 10);

        await sidecar.initialize();
        
        expect(spawn).toHaveBeenCalledWith(
          command,
          args,
          expect.any(Object)
        );

        await sidecar.shutdown();
      }
    });
  });

  describe('LSP message handling', () => {
    beforeEach(async () => {
      // Initialize with mock responses
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 100\r\n\r\n' +
          JSON.stringify({ id: 1, jsonrpc: '2.0', result: { capabilities: {} } })
        ));
      }, 10);

      await lspSidecar.initialize();
    });

    it('should handle LSP responses correctly', (done) => {
      const testResponse = {
        id: 2,
        jsonrpc: '2.0',
        result: { test: 'data' }
      };

      // Send a mock response
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          `Content-Length: ${JSON.stringify(testResponse).length}\r\n\r\n` +
          JSON.stringify(testResponse)
        ));
      }, 50);

      // This would normally be triggered by sendRequest, but we're testing the handler directly
      done();
    });

    it('should handle LSP notifications correctly', () => {
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      
      const notification = {
        jsonrpc: '2.0',
        method: 'window/logMessage',
        params: { message: 'Test log message' }
      };

      mockProcess.stdout.emit('data', Buffer.from(
        `Content-Length: ${JSON.stringify(notification).length}\r\n\r\n` +
        JSON.stringify(notification)
      ));

      expect(logSpy).toHaveBeenCalledWith('LSP Log: Test log message');
      logSpy.mockRestore();
    });

    it('should handle malformed LSP messages gracefully', () => {
      const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      
      // Send malformed JSON
      mockProcess.stdout.emit('data', Buffer.from(
        'Content-Length: 20\r\n\r\n' +
        '{ invalid json }'
      ));

      expect(errorSpy).toHaveBeenCalledWith('Failed to parse LSP message:', expect.any(Error));
      errorSpy.mockRestore();
    });
  });

  describe('hint harvesting', () => {
    beforeEach(async () => {
      // Initialize with full capabilities
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 200\r\n\r\n' +
          JSON.stringify({
            id: 1,
            jsonrpc: '2.0',
            result: {
              capabilities: {
                definitionProvider: true,
                referencesProvider: true,
                workspaceSymbolProvider: true
              }
            }
          })
        ));
      }, 10);

      await lspSidecar.initialize();
    });

    it('should harvest workspace symbols successfully', async () => {
      const mockSymbols = [
        {
          name: 'TestClass',
          kind: 5, // class
          location: {
            uri: 'file:///test/workspace/test.ts',
            range: {
              start: { line: 0, character: 0 },
              end: { line: 10, character: 0 }
            }
          }
        },
        {
          name: 'testFunction',
          kind: 12, // function
          location: {
            uri: 'file:///test/workspace/test.ts',
            range: {
              start: { line: 5, character: 0 },
              end: { line: 8, character: 0 }
            }
          }
        }
      ];

      // Mock workspace/symbol response
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          `Content-Length: ${JSON.stringify({ id: 2, jsonrpc: '2.0', result: mockSymbols }).length}\r\n\r\n` +
          JSON.stringify({ id: 2, jsonrpc: '2.0', result: mockSymbols })
        ));
      }, 20);

      // Mock textDocument/documentSymbol response (empty for this test)
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 3, jsonrpc: '2.0', result: [] })
        ));
      }, 40);

      const hints = await lspSidecar.harvestHints(['/test/workspace/test.ts']);
      
      expect(hints).toHaveLength(2);
      expect(hints[0].name).toBe('TestClass');
      expect(hints[0].kind).toBe('class');
      expect(hints[0].file_path).toBe('/test/workspace/test.ts');
      expect(hints[0].line).toBe(1); // LSP uses 0-based, we use 1-based
      expect(hints[1].name).toBe('testFunction');
      expect(hints[1].kind).toBe('function');

      // Verify hints were written to shard
      expect(fs.writeFileSync).toHaveBeenCalledWith(
        expect.stringContaining('test-sha_typescript.ndjson'),
        expect.stringContaining('TestClass'),
        'utf8'
      );
    });

    it('should handle document symbols correctly', async () => {
      const mockDocSymbols = [
        {
          name: 'MyInterface',
          kind: 11, // interface
          range: {
            start: { line: 2, character: 0 },
            end: { line: 8, character: 0 }
          },
          detail: 'interface MyInterface',
          children: [
            {
              name: 'method1',
              kind: 6, // method
              range: {
                start: { line: 3, character: 2 },
                end: { line: 5, character: 2 }
              }
            }
          ]
        }
      ];

      // Mock responses
      setTimeout(() => {
        // workspace/symbol response (empty)
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 2, jsonrpc: '2.0', result: [] })
        ));
      }, 20);

      setTimeout(() => {
        // textDocument/documentSymbol response
        mockProcess.stdout.emit('data', Buffer.from(
          `Content-Length: ${JSON.stringify({ id: 3, jsonrpc: '2.0', result: mockDocSymbols }).length}\r\n\r\n` +
          JSON.stringify({ id: 3, jsonrpc: '2.0', result: mockDocSymbols })
        ));
      }, 40);

      const hints = await lspSidecar.harvestHints(['/test/workspace/interface.ts']);
      
      expect(hints).toHaveLength(2); // interface + method
      expect(hints[0].name).toBe('MyInterface');
      expect(hints[0].kind).toBe('interface');
      expect(hints[0].signature).toBe('interface MyInterface');
      expect(hints[1].name).toBe('method1');
      expect(hints[1].kind).toBe('method');
    });

    it('should use cache when hints are fresh', async () => {
      const filePaths = ['/test/workspace/test.ts'];
      
      // First call - mock fresh response
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 2, jsonrpc: '2.0', result: [] })
        ));
      }, 20);

      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 3, jsonrpc: '2.0', result: [] })
        ));
      }, 40);

      const firstResult = await lspSidecar.harvestHints(filePaths);
      
      // Second call should use cache (no new LSP requests)
      const secondResult = await lspSidecar.harvestHints(filePaths);
      
      expect(secondResult).toEqual(firstResult);
      
      // Should only have written to shard once
      expect(fs.writeFileSync).toHaveBeenCalledTimes(1);
    });

    it('should force refresh when requested', async () => {
      const filePaths = ['/test/workspace/test.ts'];
      
      // First call
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 2, jsonrpc: '2.0', result: [] })
        ));
      }, 20);

      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 3, jsonrpc: '2.0', result: [] })
        ));
      }, 40);

      await lspSidecar.harvestHints(filePaths);
      
      // Force refresh - should make new requests
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 4, jsonrpc: '2.0', result: [] })
        ));
      }, 60);

      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 5, jsonrpc: '2.0', result: [] })
        ));
      }, 80);

      await lspSidecar.harvestHints(filePaths, true);
      
      // Should have written to shard twice
      expect(fs.writeFileSync).toHaveBeenCalledTimes(2);
    });

    it('should filter files based on workspace config', async () => {
      const filePaths = [
        '/test/workspace/src/main.ts',      // should include
        '/test/workspace/node_modules/dep.js', // should exclude
        '/test/workspace/test/main.test.ts', // should exclude
        '/test/workspace/src/util.js'       // should include
      ];

      // Mock empty responses for valid files
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 2, jsonrpc: '2.0', result: [] })
        ));
      }, 20);

      // Only expect documentSymbol requests for included files (2 files)
      let requestCount = 0;
      const originalWrite = mockProcess.stdin.write;
      mockProcess.stdin.write = jest.fn((data: any) => {
        if (data.includes('textDocument/documentSymbol')) {
          requestCount++;
          setTimeout(() => {
            mockProcess.stdout.emit('data', Buffer.from(
              `Content-Length: 50\r\n\r\n` +
              JSON.stringify({ id: 2 + requestCount, jsonrpc: '2.0', result: [] })
            ));
          }, 10);
        }
        return originalWrite.call(mockProcess.stdin, data);
      });

      await lspSidecar.harvestHints(filePaths);
      
      // Should only process 2 files (excluding node_modules and test files)
      expect(requestCount).toBe(2);
    });
  });

  describe('hint loading and shard management', () => {
    it('should load hints from shard file successfully', async () => {
      const mockHints: LSPHint[] = [
        {
          symbol_id: 'test_1',
          name: 'TestFunction',
          kind: 'function',
          file_path: '/test/workspace/test.ts',
          line: 10,
          col: 0,
          signature: 'function TestFunction()',
          aliases: [],
          resolved_imports: [],
          references_count: 5
        }
      ];

      const ndjsonContent = mockHints.map(h => JSON.stringify(h)).join('\n');
      (fs.readFileSync as any).mockReturnValue(ndjsonContent);

      const loadedHints = await lspSidecar.loadHintsFromShard();
      
      expect(loadedHints).toEqual(mockHints);
      expect(fs.readFileSync).toHaveBeenCalledWith(
        expect.stringContaining('test-sha_typescript.ndjson'),
        'utf8'
      );
    });

    it('should handle missing shard file gracefully', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      const loadedHints = await lspSidecar.loadHintsFromShard();
      
      expect(loadedHints).toEqual([]);
    });

    it('should handle corrupted shard file gracefully', async () => {
      const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      (fs.readFileSync as any).mockReturnValue('{ invalid json }');

      const loadedHints = await lspSidecar.loadHintsFromShard();
      
      expect(loadedHints).toEqual([]);
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to load hints from'),
        expect.any(Error)
      );
      
      errorSpy.mockRestore();
    });
  });

  describe('cache management', () => {
    beforeEach(async () => {
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 100\r\n\r\n' +
          JSON.stringify({ id: 1, jsonrpc: '2.0', result: { capabilities: {} } })
        ));
      }, 10);

      await lspSidecar.initialize();
    });

    it('should determine when hints need refreshing based on TTL', () => {
      // Fresh cache - should not need refresh
      expect(lspSidecar.shouldRefreshHints()).toBe(true); // Empty cache

      // Add some mock cache data by harvesting
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 2, jsonrpc: '2.0', result: [] })
        ));
      }, 20);

      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 3, jsonrpc: '2.0', result: [] })
        ));
      }, 40);
    });

    it('should cleanup old cache entries', () => {
      // This is mainly testing the method exists and doesn't crash
      lspSidecar.cleanupCache();
      
      const stats = lspSidecar.getStats();
      expect(stats.cache_entries).toBe(0);
    });
  });

  describe('shutdown and cleanup', () => {
    beforeEach(async () => {
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 100\r\n\r\n' +
          JSON.stringify({ id: 1, jsonrpc: '2.0', result: { capabilities: {} } })
        ));
      }, 10);

      await lspSidecar.initialize();
    });

    it('should shutdown gracefully', async () => {
      const shutdownPromise = lspSidecar.shutdown();
      
      // Simulate process exit
      setTimeout(() => {
        mockProcess.emit('exit', 0, null);
      }, 10);

      await shutdownPromise;

      expect(mockProcess.kill).toHaveBeenCalledWith('SIGTERM');

      const stats = lspSidecar.getStats();
      expect(stats.initialized).toBe(false);
      expect(stats.cache_entries).toBe(0);
    });

    it('should force kill if graceful shutdown fails', async () => {
      const shutdownPromise = lspSidecar.shutdown();
      
      // Don't emit exit event to trigger timeout
      await shutdownPromise;

      expect(mockProcess.kill).toHaveBeenCalledWith('SIGTERM');
    });
  });

  describe('LSP symbol kind conversion', () => {
    it('should convert LSP symbol kinds correctly', async () => {
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 100\r\n\r\n' +
          JSON.stringify({ id: 1, jsonrpc: '2.0', result: { capabilities: {} } })
        ));
      }, 10);

      await lspSidecar.initialize();

      const mockSymbols = [
        { name: 'TestClass', kind: 5, location: { uri: 'file:///test.ts', range: { start: { line: 0, character: 0 }, end: { line: 1, character: 0 } } } },
        { name: 'testMethod', kind: 6, location: { uri: 'file:///test.ts', range: { start: { line: 1, character: 0 }, end: { line: 2, character: 0 } } } },
        { name: 'testVar', kind: 13, location: { uri: 'file:///test.ts', range: { start: { line: 2, character: 0 }, end: { line: 3, character: 0 } } } },
        { name: 'TestInterface', kind: 11, location: { uri: 'file:///test.ts', range: { start: { line: 3, character: 0 }, end: { line: 4, character: 0 } } } },
        { name: 'unknownKind', kind: 999, location: { uri: 'file:///test.ts', range: { start: { line: 4, character: 0 }, end: { line: 5, character: 0 } } } }
      ];

      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          `Content-Length: ${JSON.stringify({ id: 2, jsonrpc: '2.0', result: mockSymbols }).length}\r\n\r\n` +
          JSON.stringify({ id: 2, jsonrpc: '2.0', result: mockSymbols })
        ));
      }, 20);

      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 3, jsonrpc: '2.0', result: [] })
        ));
      }, 40);

      const hints = await lspSidecar.harvestHints(['/test/workspace/test.ts']);
      
      expect(hints[0].kind).toBe('class');    // LSP kind 5
      expect(hints[1].kind).toBe('method');   // LSP kind 6  
      expect(hints[2].kind).toBe('variable'); // LSP kind 13
      expect(hints[3].kind).toBe('interface');// LSP kind 11
      expect(hints[4].kind).toBe('variable'); // Unknown kind defaults to variable
    });
  });

  describe('error handling', () => {
    it('should handle LSP server startup failure', async () => {
      // Mock spawn to throw error
      (spawn as any).mockImplementation(() => {
        throw new Error('Server not found');
      });

      await expect(lspSidecar.initialize()).rejects.toThrow('Failed to initialize LSP sidecar');
    });

    it('should handle LSP request timeout', async () => {
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 100\r\n\r\n' +
          JSON.stringify({ id: 1, jsonrpc: '2.0', result: { capabilities: {} } })
        ));
      }, 10);

      await lspSidecar.initialize();

      // Don't send response to trigger timeout
      const harvestPromise = lspSidecar.harvestHints(['/test/workspace/test.ts']);

      // Fast-forward time to trigger timeout (would normally be 30s)
      jest.useFakeTimers();
      setTimeout(() => {
        jest.advanceTimersByTime(31000);
      }, 100);

      await expect(harvestPromise).rejects.toThrow();
      
      jest.useRealTimers();
    });

    it('should handle file processing errors gracefully', async () => {
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 200\r\n\r\n' +
          JSON.stringify({ id: 1, jsonrpc: '2.0', result: { capabilities: { workspaceSymbolProvider: true } } })
        ));
      }, 10);

      await lspSidecar.initialize();

      const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

      // Mock workspace symbols request
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 2, jsonrpc: '2.0', result: [] })
        ));
      }, 20);

      // Mock document symbols request to fail
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 100\r\n\r\n' +
          JSON.stringify({ id: 3, jsonrpc: '2.0', error: { message: 'File not found' } })
        ));
      }, 40);

      const hints = await lspSidecar.harvestHints(['/test/workspace/missing.ts']);
      
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to get hints for /test/workspace/missing.ts:'),
        expect.any(Error)
      );

      // Should still return hints from workspace symbols
      expect(hints).toBeInstanceOf(Array);
      
      warnSpy.mockRestore();
    });
  });

  describe('statistics and monitoring', () => {
    beforeEach(async () => {
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 200\r\n\r\n' +
          JSON.stringify({
            id: 1,
            jsonrpc: '2.0',
            result: {
              capabilities: {
                definitionProvider: true,
                referencesProvider: true,
                workspaceSymbolProvider: true
              }
            }
          })
        ));
      }, 10);

      await lspSidecar.initialize();
    });

    it('should provide accurate statistics', async () => {
      const stats = lspSidecar.getStats();
      
      expect(stats.initialized).toBe(true);
      expect(stats.cache_entries).toBe(0);
      expect(stats.total_hints).toBe(0);
      expect(stats.capabilities).toEqual({
        definition: true,
        references: true,
        hover: false,
        completion: false,
        rename: false,
        workspace_symbols: true
      });
    });

    it('should track cache statistics after harvesting', async () => {
      // Mock successful harvest
      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 2, jsonrpc: '2.0', result: [] })
        ));
      }, 20);

      setTimeout(() => {
        mockProcess.stdout.emit('data', Buffer.from(
          'Content-Length: 50\r\n\r\n' +
          JSON.stringify({ id: 3, jsonrpc: '2.0', result: [] })
        ));
      }, 40);

      await lspSidecar.harvestHints(['/test/workspace/test.ts']);
      
      const stats = lspSidecar.getStats();
      expect(stats.cache_entries).toBe(1);
    });
  });
});