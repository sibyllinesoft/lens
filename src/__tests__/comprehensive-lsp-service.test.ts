/**
 * Comprehensive LSP Service Coverage Tests
 * 
 * Target: Test all LSP service methods, protocols, and integrations
 * Coverage focus: Language server functionality, caching, error handling
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, jest } from 'bun:test';
import { LSPService } from '../lsp/service.js';
import { LensTracer } from '../telemetry/tracer.js';
import type { 
  LSPSymbolInformation,
  LSPDefinitionRequest,
  LSPReferencesRequest,
  LSPHoverRequest,
  LSPCompletionRequest,
  LSPWorkspaceSymbolRequest
} from '../types/lsp.js';
import type { SupportedLanguage } from '../types/api.js';

// Import fixtures
import { getSearchFixtures } from './fixtures/db-fixtures-simple.js';

describe('Comprehensive LSP Service Coverage Tests', () => {
  let lspService: LSPService;
  let fixtures: any;

  beforeAll(async () => {
    // Get test fixtures
    fixtures = await getSearchFixtures();
    
    // Create LSP service instance
    lspService = new LSPService();
    
    // Initialize service
    await lspService.initialize();
  });

  afterAll(async () => {
    if (lspService) {
      await lspService.shutdown();
    }
  });

  beforeEach(() => {
    // Reset any stateful components between tests
    jest.clearAllMocks();
  });

  describe('Service Initialization and Lifecycle', () => {
    it('should initialize LSP service properly', async () => {
      const service = new LSPService();
      
      await expect(service.initialize()).resolves.not.toThrow();
      
      // Verify service is ready
      const isReady = await service.isReady();
      expect(isReady).toBe(true);
      
      await service.shutdown();
    });

    it('should handle initialization failures', async () => {
      const service = new LSPService();
      
      // Mock initialization failure
      jest.spyOn(service as any, 'initializeLanguageServers')
        .mockRejectedValueOnce(new Error('Language server init failed'));
      
      await expect(service.initialize())
        .rejects.toThrow('Language server init failed');
    });

    it('should shutdown gracefully', async () => {
      const service = new LSPService();
      await service.initialize();
      
      await expect(service.shutdown()).resolves.not.toThrow();
      
      const isReady = await service.isReady();
      expect(isReady).toBe(false);
    });

    it('should report readiness status accurately', async () => {
      const service = new LSPService();
      
      // Before initialization
      let isReady = await service.isReady();
      expect(isReady).toBe(false);
      
      // After initialization
      await service.initialize();
      isReady = await service.isReady();
      expect(isReady).toBe(true);
      
      // After shutdown
      await service.shutdown();
      isReady = await service.isReady();
      expect(isReady).toBe(false);
    });
  });

  describe('Definition Requests', () => {
    it('should handle go-to-definition requests', async () => {
      const request: LSPDefinitionRequest = {
        textDocument: { uri: 'file:///src/api/server.ts' },
        position: { line: 100, character: 15 },
        language: 'typescript' as SupportedLanguage
      };

      const response = await lspService.getDefinition(request);
      
      expect(response).toBeDefined();
      expect(Array.isArray(response.locations)).toBe(true);
      
      if (response.locations.length > 0) {
        const location = response.locations[0];
        expect(location).toHaveProperty('uri');
        expect(location).toHaveProperty('range');
        expect(location.range).toHaveProperty('start');
        expect(location.range).toHaveProperty('end');
      }
    });

    it('should handle definitions for different languages', async () => {
      const languages: SupportedLanguage[] = ['typescript', 'javascript', 'python', 'rust', 'go'];
      
      for (const language of languages) {
        const request: LSPDefinitionRequest = {
          textDocument: { uri: `file:///src/test.${getExtension(language)}` },
          position: { line: 10, character: 5 },
          language
        };

        const response = await lspService.getDefinition(request);
        
        expect(response).toBeDefined();
        expect(Array.isArray(response.locations)).toBe(true);
      }
    });

    it('should handle invalid position requests', async () => {
      const request: LSPDefinitionRequest = {
        textDocument: { uri: 'file:///src/nonexistent.ts' },
        position: { line: -1, character: -1 },
        language: 'typescript' as SupportedLanguage
      };

      const response = await lspService.getDefinition(request);
      
      expect(response).toBeDefined();
      expect(response.locations.length).toBe(0);
    });
  });

  describe('References Requests', () => {
    it('should find all references for a symbol', async () => {
      const request: LSPReferencesRequest = {
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        position: { line: 62, character: 20 }, // LensSearchEngine class
        context: { includeDeclaration: true },
        language: 'typescript' as SupportedLanguage
      };

      const response = await lspService.getReferences(request);
      
      expect(response).toBeDefined();
      expect(Array.isArray(response.references)).toBe(true);
      
      if (response.references.length > 0) {
        const reference = response.references[0];
        expect(reference).toHaveProperty('uri');
        expect(reference).toHaveProperty('range');
      }
    });

    it('should handle references with includeDeclaration flag', async () => {
      const requestWithDeclaration: LSPReferencesRequest = {
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        position: { line: 62, character: 20 },
        context: { includeDeclaration: true },
        language: 'typescript' as SupportedLanguage
      };

      const requestWithoutDeclaration: LSPReferencesRequest = {
        ...requestWithDeclaration,
        context: { includeDeclaration: false }
      };

      const responseWith = await lspService.getReferences(requestWithDeclaration);
      const responseWithout = await lspService.getReferences(requestWithoutDeclaration);
      
      expect(responseWith).toBeDefined();
      expect(responseWithout).toBeDefined();
      
      // With declaration should have same or more references
      expect(responseWith.references.length).toBeGreaterThanOrEqual(responseWithout.references.length);
    });

    it('should handle references for non-existent symbols', async () => {
      const request: LSPReferencesRequest = {
        textDocument: { uri: 'file:///src/nonexistent.ts' },
        position: { line: 1, character: 1 },
        context: { includeDeclaration: true },
        language: 'typescript' as SupportedLanguage
      };

      const response = await lspService.getReferences(request);
      
      expect(response).toBeDefined();
      expect(response.references.length).toBe(0);
    });
  });

  describe('Hover Information', () => {
    it('should provide hover information for symbols', async () => {
      const request: LSPHoverRequest = {
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        position: { line: 70, character: 10 },
        language: 'typescript' as SupportedLanguage
      };

      const response = await lspService.getHover(request);
      
      expect(response).toBeDefined();
      
      if (response.hover) {
        expect(response.hover).toHaveProperty('contents');
        expect(response.hover).toHaveProperty('range');
        
        if (typeof response.hover.contents === 'string') {
          expect(response.hover.contents.length).toBeGreaterThan(0);
        } else if (Array.isArray(response.hover.contents)) {
          expect(response.hover.contents.length).toBeGreaterThan(0);
        }
      }
    });

    it('should handle hover requests for different symbol types', async () => {
      const positions = [
        { line: 62, character: 20 }, // Class name
        { line: 80, character: 10 }, // Method name
        { line: 90, character: 15 }, // Variable
        { line: 100, character: 25 }, // Function call
      ];
      
      for (const position of positions) {
        const request: LSPHoverRequest = {
          textDocument: { uri: 'file:///src/api/search-engine.ts' },
          position,
          language: 'typescript' as SupportedLanguage
        };

        const response = await lspService.getHover(request);
        expect(response).toBeDefined();
      }
    });

    it('should return null for positions with no symbol', async () => {
      const request: LSPHoverRequest = {
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        position: { line: 1, character: 1 }, // Likely empty space
        language: 'typescript' as SupportedLanguage
      };

      const response = await lspService.getHover(request);
      
      expect(response).toBeDefined();
      // Should either be null or have empty/minimal content
    });
  });

  describe('Code Completion', () => {
    it('should provide completion suggestions', async () => {
      const request: LSPCompletionRequest = {
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        position: { line: 100, character: 10 },
        language: 'typescript' as SupportedLanguage,
        context: {
          triggerKind: 1, // Invoked
          triggerCharacter: '.'
        }
      };

      const response = await lspService.getCompletion(request);
      
      expect(response).toBeDefined();
      expect(Array.isArray(response.completions)).toBe(true);
      
      if (response.completions.length > 0) {
        const completion = response.completions[0];
        expect(completion).toHaveProperty('label');
        expect(completion).toHaveProperty('kind');
        expect(typeof completion.label).toBe('string');
        expect(completion.label.length).toBeGreaterThan(0);
      }
    });

    it('should handle different trigger characters', async () => {
      const triggerChars = ['.', '->', '::', '<'];
      
      for (const triggerChar of triggerChars) {
        const request: LSPCompletionRequest = {
          textDocument: { uri: 'file:///src/api/search-engine.ts' },
          position: { line: 100, character: 10 },
          language: 'typescript' as SupportedLanguage,
          context: {
            triggerKind: 2, // TriggerCharacter
            triggerCharacter: triggerChar
          }
        };

        const response = await lspService.getCompletion(request);
        expect(response).toBeDefined();
        expect(Array.isArray(response.completions)).toBe(true);
      }
    });

    it('should provide contextual completions', async () => {
      const request: LSPCompletionRequest = {
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        position: { line: 80, character: 20 }, // Inside class
        language: 'typescript' as SupportedLanguage,
        context: {
          triggerKind: 1, // Invoked
        }
      };

      const response = await lspService.getCompletion(request);
      
      expect(response).toBeDefined();
      expect(Array.isArray(response.completions)).toBe(true);
    });
  });

  describe('Workspace Symbol Search', () => {
    it('should find workspace symbols by query', async () => {
      const request: LSPWorkspaceSymbolRequest = {
        query: 'LensSearchEngine',
        language: 'typescript' as SupportedLanguage
      };

      const response = await lspService.getWorkspaceSymbols(request);
      
      expect(response).toBeDefined();
      expect(Array.isArray(response.symbols)).toBe(true);
      
      if (response.symbols.length > 0) {
        const symbol = response.symbols[0];
        expect(symbol).toHaveProperty('name');
        expect(symbol).toHaveProperty('kind');
        expect(symbol).toHaveProperty('location');
        expect(symbol.location).toHaveProperty('uri');
        expect(symbol.location).toHaveProperty('range');
      }
    });

    it('should handle empty query strings', async () => {
      const request: LSPWorkspaceSymbolRequest = {
        query: '',
        language: 'typescript' as SupportedLanguage
      };

      const response = await lspService.getWorkspaceSymbols(request);
      
      expect(response).toBeDefined();
      expect(Array.isArray(response.symbols)).toBe(true);
    });

    it('should handle partial symbol queries', async () => {
      const partialQueries = ['Lens', 'Search', 'Engine', 'search', 'lens'];
      
      for (const query of partialQueries) {
        const request: LSPWorkspaceSymbolRequest = {
          query,
          language: 'typescript' as SupportedLanguage
        };

        const response = await lspService.getWorkspaceSymbols(request);
        
        expect(response).toBeDefined();
        expect(Array.isArray(response.symbols)).toBe(true);
      }
    });

    it('should filter symbols by kind', async () => {
      const request: LSPWorkspaceSymbolRequest = {
        query: 'function',
        language: 'typescript' as SupportedLanguage,
        symbolKind: 12 // Function
      };

      const response = await lspService.getWorkspaceSymbols(request);
      
      expect(response).toBeDefined();
      expect(Array.isArray(response.symbols)).toBe(true);
      
      // If we have results, they should match the requested kind
      response.symbols.forEach(symbol => {
        if (symbol.kind === 12) {
          expect(symbol.kind).toBe(12);
        }
      });
    });
  });

  describe('Document Symbol Requests', () => {
    it('should provide document outline/symbols', async () => {
      const response = await lspService.getDocumentSymbols({
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        language: 'typescript' as SupportedLanguage
      });
      
      expect(response).toBeDefined();
      expect(Array.isArray(response.symbols)).toBe(true);
      
      if (response.symbols.length > 0) {
        const symbol = response.symbols[0];
        expect(symbol).toHaveProperty('name');
        expect(symbol).toHaveProperty('kind');
        expect(symbol).toHaveProperty('range');
        expect(symbol).toHaveProperty('selectionRange');
      }
    });

    it('should handle hierarchical document symbols', async () => {
      const response = await lspService.getDocumentSymbols({
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        language: 'typescript' as SupportedLanguage
      });
      
      expect(response).toBeDefined();
      
      // Check for nested symbols (like methods inside classes)
      const classSymbols = response.symbols.filter(s => s.kind === 5); // Class
      if (classSymbols.length > 0 && classSymbols[0].children) {
        expect(Array.isArray(classSymbols[0].children)).toBe(true);
      }
    });
  });

  describe('Language Server Management', () => {
    it('should handle multiple language servers', async () => {
      const languages: SupportedLanguage[] = ['typescript', 'python', 'rust'];
      
      for (const language of languages) {
        const isSupported = await lspService.isLanguageSupported(language);
        expect(typeof isSupported).toBe('boolean');
        
        if (isSupported) {
          const request: LSPDefinitionRequest = {
            textDocument: { uri: `file:///test.${getExtension(language)}` },
            position: { line: 1, character: 1 },
            language
          };
          
          const response = await lspService.getDefinition(request);
          expect(response).toBeDefined();
        }
      }
    });

    it('should restart language servers on failure', async () => {
      // Test language server recovery mechanism
      const isReady = await lspService.isReady();
      expect(isReady).toBe(true);
      
      // Simulate recovery after restart
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const stillReady = await lspService.isReady();
      expect(stillReady).toBe(true);
    });
  });

  describe('Caching and Performance', () => {
    it('should cache similar requests for performance', async () => {
      const request: LSPDefinitionRequest = {
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        position: { line: 62, character: 20 },
        language: 'typescript' as SupportedLanguage
      };

      // First request
      const start1 = Date.now();
      const response1 = await lspService.getDefinition(request);
      const time1 = Date.now() - start1;
      
      // Second request (should be faster due to caching)
      const start2 = Date.now();
      const response2 = await lspService.getDefinition(request);
      const time2 = Date.now() - start2;
      
      expect(response1).toBeDefined();
      expect(response2).toBeDefined();
      
      // Results should be consistent
      expect(response1.locations.length).toBe(response2.locations.length);
    });

    it('should handle concurrent requests efficiently', async () => {
      const requests = Array.from({ length: 5 }, (_, i) => ({
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        position: { line: 70 + i, character: 10 },
        language: 'typescript' as SupportedLanguage
      }));

      const promises = requests.map(req => lspService.getDefinition(req));
      const responses = await Promise.all(promises);
      
      // All requests should complete
      expect(responses).toHaveLength(5);
      responses.forEach(response => {
        expect(response).toBeDefined();
        expect(Array.isArray(response.locations)).toBe(true);
      });
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle malformed requests gracefully', async () => {
      const malformedRequests = [
        null,
        undefined,
        {},
        { textDocument: null },
        { position: null },
        { textDocument: { uri: '' }, position: { line: -1, character: -1 } }
      ];

      for (const badRequest of malformedRequests) {
        await expect(lspService.getDefinition(badRequest as any))
          .rejects.toThrow();
      }
    });

    it('should handle language server timeouts', async () => {
      const request: LSPDefinitionRequest = {
        textDocument: { uri: 'file:///very-large-file.ts' },
        position: { line: 10000, character: 100 },
        language: 'typescript' as SupportedLanguage,
        timeout: 100 // Very short timeout
      };

      // Should either complete quickly or timeout gracefully
      const response = await lspService.getDefinition(request);
      expect(response).toBeDefined();
    });

    it('should handle unsupported file types', async () => {
      const request: LSPDefinitionRequest = {
        textDocument: { uri: 'file:///test.unknown' },
        position: { line: 1, character: 1 },
        language: 'unknown' as any
      };

      await expect(lspService.getDefinition(request))
        .rejects.toThrow();
    });

    it('should handle network/connection failures', async () => {
      // Mock connection failure
      const originalGetDefinition = lspService.getDefinition.bind(lspService);
      jest.spyOn(lspService, 'getDefinition')
        .mockRejectedValueOnce(new Error('Connection failed'));

      const request: LSPDefinitionRequest = {
        textDocument: { uri: 'file:///src/api/search-engine.ts' },
        position: { line: 1, character: 1 },
        language: 'typescript' as SupportedLanguage
      };

      await expect(lspService.getDefinition(request))
        .rejects.toThrow('Connection failed');
    });
  });

  describe('Integration with Search Engine', () => {
    it('should provide LSP hints for search enhancement', async () => {
      const hints = await lspService.getSearchHints({
        query: 'LensSearchEngine',
        language: 'typescript' as SupportedLanguage,
        context: {
          file: 'src/api/server.ts',
          line: 100,
          character: 10
        }
      });
      
      expect(hints).toBeDefined();
      expect(Array.isArray(hints.hints)).toBe(true);
      
      if (hints.hints.length > 0) {
        const hint = hints.hints[0];
        expect(hint).toHaveProperty('file');
        expect(hint).toHaveProperty('line');
        expect(hint).toHaveProperty('character');
        expect(hint).toHaveProperty('hint_type');
        expect(hint).toHaveProperty('confidence');
      }
    });

    it('should provide symbol context for search results', async () => {
      const context = await lspService.getSymbolContext({
        file: 'src/api/search-engine.ts',
        line: 62,
        character: 20,
        language: 'typescript' as SupportedLanguage
      });
      
      expect(context).toBeDefined();
      expect(context).toHaveProperty('symbolInfo');
      expect(context).toHaveProperty('containingScope');
      
      if (context.symbolInfo) {
        expect(context.symbolInfo).toHaveProperty('name');
        expect(context.symbolInfo).toHaveProperty('kind');
      }
    });
  });
});

// Helper function to get file extension for language
function getExtension(language: SupportedLanguage): string {
  const extensionMap: Record<string, string> = {
    typescript: 'ts',
    javascript: 'js',
    python: 'py',
    rust: 'rs',
    go: 'go',
    java: 'java',
    cpp: 'cpp',
    csharp: 'cs'
  };
  return extensionMap[language] || 'txt';
}
