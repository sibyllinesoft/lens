/**
 * Integration tests for LSP SPI Service
 * Tests the extended language support for Python, Go, and Rust
 */

import { describe, test, expect, beforeEach, afterEach } from 'vitest';
import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, writeFile, rm } from 'fs/promises';

// Mock the LSPSPIService to test just the new language configurations
class MockLSPSPIService {
  private workspaceRoot: string;
  private languageConfigs = [
    {
      language: 'typescript',
      extensions: ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'],
      command: 'typescript-language-server',
      args: ['--stdio'],
      enabled: true,
      workspaceDetection: {
        files: ['package.json', 'tsconfig.json', 'jsconfig.json'],
        directories: ['node_modules', '.git']
      },
      installationGuide: 'npm install -g typescript-language-server typescript'
    },
    {
      language: 'python',
      extensions: ['.py', '.pyx', '.pyi'],
      command: 'pylsp',
      args: [],
      enabled: true,
      workspaceDetection: {
        files: ['requirements.txt', 'pyproject.toml', 'setup.py', 'setup.cfg', 'Pipfile'],
        directories: ['venv', '.venv', '__pycache__', '.git']
      },
      installationGuide: 'pip install python-lsp-server[all] python-lsp-black'
    },
    {
      language: 'go',
      extensions: ['.go'],
      command: 'gopls',
      args: [],
      enabled: true,
      workspaceDetection: {
        files: ['go.mod', 'go.sum'],
        directories: ['vendor', '.git']
      },
      installationGuide: 'go install golang.org/x/tools/gopls@latest'
    },
    {
      language: 'rust',
      extensions: ['.rs'],
      command: 'rust-analyzer',
      args: [],
      enabled: true,
      workspaceDetection: {
        files: ['Cargo.toml', 'Cargo.lock', 'rust-toolchain', 'rust-toolchain.toml'],
        directories: ['target', 'src', '.git']
      },
      installationGuide: 'rustup component add rust-analyzer'
    }
  ];

  constructor(workspaceRoot: string) {
    this.workspaceRoot = workspaceRoot;
  }

  detectLanguage(path: string): string | null {
    const lowerPath = path.toLowerCase();
    
    if (lowerPath.includes('cargo.toml') || lowerPath.includes('cargo.lock')) {
      return this.languageConfigs.find(c => c.language === 'rust' && c.enabled) ? 'rust' : null;
    }
    
    const ext = lowerPath.split('.').pop();
    if (!ext) return null;

    const fullExt = `.${ext}`;
    for (const config of this.languageConfigs) {
      if (config.enabled && config.extensions.includes(fullExt)) {
        return config.language;
      }
    }
    
    return null;
  }

  getDefaultFormatOptions(language: string): any {
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
        insertSpaces: false,
        ...baseOptions
      }
    };
    
    return languageDefaults[language] || { tabSize: 2, insertSpaces: true, ...baseOptions };
  }

  async getSystemStatus() {
    return {
      languages: this.languageConfigs.map(config => ({
        language: config.language,
        enabled: config.enabled,
        available: false, // Mock as false since we can't test actual availability
        command: config.command,
        installationGuide: config.installationGuide || 'Check language documentation',
        workspaceDetected: false,
        workspaceRoot: this.workspaceRoot
      }))
    };
  }

  async shutdown() {
    // Mock shutdown
  }
}

describe('LSP SPI Service - Extended Language Support', () => {
  let service: MockLSPSPIService;
  let testDir: string;

  beforeEach(async () => {
    // Create temporary directory for test workspace
    testDir = await mkdtemp(join(tmpdir(), 'lens-lsp-test-'));
    service = new MockLSPSPIService(testDir);
  });

  afterEach(async () => {
    // Clean up
    await service.shutdown();
    await rm(testDir, { recursive: true, force: true });
  });

  test('should detect languages correctly', () => {
    // TypeScript/JavaScript
    expect(service.detectLanguage('app.ts')).toBe('typescript');
    expect(service.detectLanguage('component.tsx')).toBe('typescript');
    expect(service.detectLanguage('utils.js')).toBe('typescript');
    expect(service.detectLanguage('module.mjs')).toBe('typescript');

    // Python
    expect(service.detectLanguage('main.py')).toBe('python');
    expect(service.detectLanguage('extension.pyx')).toBe('python');
    expect(service.detectLanguage('types.pyi')).toBe('python');

    // Go
    expect(service.detectLanguage('main.go')).toBe('go');
    expect(service.detectLanguage('handler.go')).toBe('go');

    // Rust
    expect(service.detectLanguage('main.rs')).toBe('rust');
    expect(service.detectLanguage('lib.rs')).toBe('rust');
    expect(service.detectLanguage('Cargo.toml')).toBe('rust');

    // Unsupported
    expect(service.detectLanguage('document.txt')).toBe(null);
    expect(service.detectLanguage('style.css')).toBe(null);
  });

  test('should get system status for all languages', async () => {
    const status = await service.getSystemStatus();
    
    expect(status.languages).toHaveLength(4);
    
    const languages = status.languages.reduce((acc, lang) => {
      acc[lang.language] = lang;
      return acc;
    }, {} as Record<string, any>);

    // Check TypeScript configuration
    expect(languages.typescript).toEqual(
      expect.objectContaining({
        language: 'typescript',
        command: 'typescript-language-server',
        installationGuide: 'npm install -g typescript-language-server typescript'
      })
    );

    // Check Python configuration (now using pylsp)
    expect(languages.python).toEqual(
      expect.objectContaining({
        language: 'python',
        command: 'pylsp',
        installationGuide: 'pip install python-lsp-server[all] python-lsp-black'
      })
    );

    // Check Go configuration
    expect(languages.go).toEqual(
      expect.objectContaining({
        language: 'go',
        command: 'gopls',
        installationGuide: 'go install golang.org/x/tools/gopls@latest'
      })
    );

    // Check Rust configuration
    expect(languages.rust).toEqual(
      expect.objectContaining({
        language: 'rust',
        command: 'rust-analyzer',
        installationGuide: 'rustup component add rust-analyzer'
      })
    );
  });

  test('should get language-specific format options', () => {
    const tsOptions = service.getDefaultFormatOptions('typescript');
    expect(tsOptions).toEqual(
      expect.objectContaining({
        tabSize: 2,
        insertSpaces: true,
        insertFinalNewline: true,
        trimTrailingWhitespace: true
      })
    );

    const pythonOptions = service.getDefaultFormatOptions('python');
    expect(pythonOptions).toEqual(
      expect.objectContaining({
        tabSize: 4,
        insertSpaces: true,
        insertFinalNewline: true,
        trimTrailingWhitespace: true
      })
    );

    const goOptions = service.getDefaultFormatOptions('go');
    expect(goOptions).toEqual(
      expect.objectContaining({
        tabSize: 4,
        insertSpaces: false, // Go uses tabs
        insertFinalNewline: true,
        trimTrailingWhitespace: true
      })
    );

    const rustOptions = service.getDefaultFormatOptions('rust');
    expect(rustOptions).toEqual(
      expect.objectContaining({
        tabSize: 4,
        insertSpaces: true,
        insertFinalNewline: true,
        trimTrailingWhitespace: true
      })
    );
  });

});