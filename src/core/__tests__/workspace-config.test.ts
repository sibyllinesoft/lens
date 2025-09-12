/**
 * Comprehensive Tests for Workspace Config Component
 * Tests config parsing, path mapping, and language-specific configurations
 */

import { describe, it, expect, beforeEach, afterEach, mock, jest, mock } from 'bun:test';
import { WorkspaceConfigParser } from '../workspace-config.js';
import type { WorkspaceConfig, SupportedLanguage } from '../../types/core.js';
import fs from 'fs';

// Mock fs module
mock('fs', () => ({
  existsSync: jest.fn(),
  readFileSync: jest.fn()
}));

// Mock path module
mock('path', () => ({
  join: jest.fn((...args) => args.join('/')),
  resolve: jest.fn((...args) => args.join('/')),
  dirname: jest.fn((path) => path.split('/').slice(0, -1).join('/')),
  relative: jest.fn((from, to) => to.replace(from, '').replace(/^\//, ''))
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

describe('WorkspaceConfigParser', () => {
  let parser: WorkspaceConfigParser;
  const mockWorkspaceRoot = '/test/workspace';

  beforeEach(() => {
    jest.clearAllMocks();
    parser = new WorkspaceConfigParser();
  });

  afterEach(() => {
    parser.clearCache();
  });

  describe('TypeScript configuration parsing', () => {
    it('should parse basic tsconfig.json correctly', async () => {
      const mockTsConfig = {
        compilerOptions: {
          baseUrl: '.',
          paths: {
            '@/*': ['src/*'],
            '@components/*': ['src/components/*'],
            '@utils/*': ['src/utils/*']
          },
          rootDirs: ['src', 'test']
        },
        include: ['src/**/*', 'test/**/*'],
        exclude: ['node_modules', 'dist', 'build']
      };

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('tsconfig.json');
      });

      (fs.readFileSync as any).mockImplementation((path: string) => {
        if (path.endsWith('tsconfig.json')) {
          return JSON.stringify(mockTsConfig);
        }
        return '{}';
      });

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');

      expect(config.root_path).toBe(mockWorkspaceRoot);
      expect(config.include_patterns).toContain('src/**/*');
      expect(config.include_patterns).toContain('test/**/*');
      expect(config.exclude_patterns).toContain('node_modules');
      expect(config.exclude_patterns).toContain('dist');
      expect(config.exclude_patterns).toContain('build');

      // Check path mappings
      expect(config.path_mappings.get('@')).toBe('src');
      expect(config.path_mappings.get('@components')).toBe('src/components');
      expect(config.path_mappings.get('@utils')).toBe('src/utils');

      expect(config.config_files).toContain('/test/workspace/tsconfig.json');
    });

    it('should handle tsconfig.json with comments', async () => {
      const mockTsConfigWithComments = `{
        // TypeScript configuration
        "compilerOptions": {
          "baseUrl": ".",
          /* Path mappings for cleaner imports */
          "paths": {
            "@/*": ["src/*"]
          }
        },
        // Include patterns
        "include": ["src/**/*"]
      }`;

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('tsconfig.json');
      });

      (fs.readFileSync as any).mockReturnValue(mockTsConfigWithComments);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');

      expect(config.include_patterns).toContain('src/**/*');
      expect(config.path_mappings.get('@')).toBe('src');
    });

    it('should handle project references in tsconfig', async () => {
      const mockTsConfig = {
        references: [
          { path: './packages/core' },
          { path: './packages/ui' }
        ]
      };

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('tsconfig.json') || 
               path.includes('/packages/core') || 
               path.includes('/packages/ui');
      });

      (fs.readFileSync as any).mockImplementation((path: string) => {
        if (path.endsWith('tsconfig.json')) {
          return JSON.stringify(mockTsConfig);
        }
        return '{}';
      });

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');

      expect(config.config_files).toContain('/test/workspace/tsconfig.json');
      expect(config.config_files).toContain('/test/workspace/packages/core');
      expect(config.config_files).toContain('/test/workspace/packages/ui');
    });

    it('should parse package.json workspaces', async () => {
      const mockPackageJson = {
        workspaces: ['packages/*', 'apps/*']
      };

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('package.json');
      });

      (fs.readFileSync as any).mockImplementation((path: string) => {
        if (path.endsWith('package.json')) {
          return JSON.stringify(mockPackageJson);
        }
        return '{}';
      });

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');

      expect(config.include_patterns).toContain('packages/*/**/*');
      expect(config.include_patterns).toContain('apps/*/**/*');
    });

    it('should provide default TypeScript patterns when none specified', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');

      expect(config.include_patterns).toContain('**/*.ts');
      expect(config.include_patterns).toContain('**/*.tsx');
      expect(config.include_patterns).toContain('**/*.js');
      expect(config.include_patterns).toContain('**/*.jsx');
      expect(config.exclude_patterns).toContain('node_modules/**');
      expect(config.exclude_patterns).toContain('dist/**');
    });
  });

  describe('Python configuration parsing', () => {
    it('should parse pyproject.toml with pyright config', async () => {
      const mockPyprojectToml = `
[tool.pyright]
include = ["src", "tests"]
exclude = ["build", "dist"]
extraPaths = ["./lib", "./vendor"]

[tool.mypy]
files = ["src/**/*.py"]
exclude = ["tests/**/*"]
mypy_path = ["./stubs", "./typings"]
`;

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('pyproject.toml');
      });

      (fs.readFileSync as any).mockReturnValue(mockPyprojectToml);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'python');

      expect(config.include_patterns).toContain('src');
      expect(config.include_patterns).toContain('tests');
      expect(config.include_patterns).toContain('src/**/*.py');
      expect(config.exclude_patterns).toContain('build');
      expect(config.exclude_patterns).toContain('dist');
      expect(config.exclude_patterns).toContain('tests/**/*');

      // Check path mappings
      expect(config.path_mappings.get('./lib')).toBe('./lib');
      expect(config.path_mappings.get('./stubs')).toBe('./stubs');
    });

    it('should handle setup.py and setup.cfg files', async () => {
      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('setup.py') || path.endsWith('setup.cfg');
      });

      (fs.readFileSync as any).mockReturnValue('# setup file');

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'python');

      expect(config.config_files).toContain('/test/workspace/setup.py');
      expect(config.config_files).toContain('/test/workspace/setup.cfg');
    });

    it('should provide default Python patterns when none specified', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'python');

      expect(config.include_patterns).toContain('**/*.py');
      expect(config.include_patterns).toContain('**/*.pyi');
      expect(config.exclude_patterns).toContain('**/__pycache__/**');
      expect(config.exclude_patterns).toContain('.venv/**');
    });
  });

  describe('Rust configuration parsing', () => {
    it('should parse Cargo.toml workspace configuration', async () => {
      const mockCargoToml = `
[workspace]
members = ["crates/*", "tools/cli"]
exclude = ["target", "deprecated"]

[lib]
path = "src/lib.rs"

[package]
name = "my-project"
`;

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('Cargo.toml');
      });

      (fs.readFileSync as any).mockReturnValue(mockCargoToml);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'rust');

      expect(config.include_patterns).toContain('crates/*/**/*.rs');
      expect(config.include_patterns).toContain('tools/cli/**/*.rs');
      expect(config.include_patterns).toContain('src/**/*.rs'); // from lib.path
      expect(config.exclude_patterns).toContain('target');
      expect(config.exclude_patterns).toContain('deprecated');
    });

    it('should provide default Rust patterns when none specified', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'rust');

      expect(config.include_patterns).toContain('src/**/*.rs');
      expect(config.include_patterns).toContain('tests/**/*.rs');
      expect(config.include_patterns).toContain('examples/**/*.rs');
      expect(config.include_patterns).toContain('benches/**/*.rs');
      expect(config.exclude_patterns).toContain('target/**');
      expect(config.exclude_patterns).toContain('Cargo.lock');
    });
  });

  describe('Go configuration parsing', () => {
    it('should parse go.mod configuration', async () => {
      const mockGoMod = `
module github.com/example/project

go 1.21

replace github.com/old/pkg => github.com/new/pkg v1.0.0
replace ./local => ../local-pkg

exclude github.com/bad/pkg v1.0.0
`;

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('go.mod');
      });

      (fs.readFileSync as any).mockReturnValue(mockGoMod);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'go');

      expect(config.path_mappings.get('github.com/old/pkg')).toBe('github.com/new/pkg v1.0.0');
      expect(config.path_mappings.get('./local')).toBe('../local-pkg');
      expect(config.exclude_patterns).toContain('github.com/bad/pkg v1.0.0');
      expect(config.config_files).toContain('/test/workspace/go.mod');
    });

    it('should handle go.work files', async () => {
      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('go.work');
      });

      (fs.readFileSync as any).mockReturnValue('go 1.21\nuse ./module1\nuse ./module2');

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'go');

      expect(config.config_files).toContain('/test/workspace/go.work');
    });

    it('should provide default Go patterns when none specified', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'go');

      expect(config.include_patterns).toContain('**/*.go');
      expect(config.exclude_patterns).toContain('vendor/**');
    });
  });

  describe('Java configuration parsing', () => {
    it('should detect Java build files', async () => {
      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('pom.xml') || 
               path.endsWith('build.gradle') || 
               path.endsWith('settings.gradle');
      });

      (fs.readFileSync as any).mockReturnValue('<xml>build file</xml>');

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'java');

      expect(config.config_files).toContain('/test/workspace/pom.xml');
      expect(config.config_files).toContain('/test/workspace/build.gradle');
      expect(config.config_files).toContain('/test/workspace/settings.gradle');
      expect(config.include_patterns).toContain('**/src/main/java/**/*.java');
      expect(config.include_patterns).toContain('**/src/test/java/**/*.java');
    });

    it('should provide default Java patterns', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'java');

      expect(config.include_patterns).toContain('**/src/main/java/**/*.java');
      expect(config.include_patterns).toContain('**/src/test/java/**/*.java');
      expect(config.include_patterns).toContain('**/*.java');
      expect(config.exclude_patterns).toContain('target/**');
      expect(config.exclude_patterns).toContain('**/*.class');
    });
  });

  describe('Bash configuration parsing', () => {
    it('should provide default Bash patterns', async () => {
      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'bash');

      expect(config.include_patterns).toContain('**/*.sh');
      expect(config.include_patterns).toContain('**/*.bash');
      expect(config.include_patterns).toContain('**/.*rc');
      expect(config.include_patterns).toContain('**/.*profile');
    });
  });

  describe('TOML parsing', () => {
    it('should parse TOML format correctly', async () => {
      const mockToml = `
# Configuration
[tool.test]
name = "value"
number = 42
float = 3.14
bool = true
array = ["item1", "item2", "item3"]

[tool.nested.section]
key = "nested-value"
`;

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('pyproject.toml');
      });

      (fs.readFileSync as any).mockReturnValue(mockToml);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'python');

      // TOML parsing is tested indirectly through pyproject.toml parsing
      expect(config).toBeDefined();
      expect(config.config_files).toContain('/test/workspace/pyproject.toml');
    });

    it('should handle TOML arrays correctly', async () => {
      const mockToml = `
[tool.pyright]
include = ["src", "tests", "examples"]
exclude = ["build"]
`;

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('pyproject.toml');
      });

      (fs.readFileSync as any).mockReturnValue(mockToml);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'python');

      expect(config.include_patterns).toContain('src');
      expect(config.include_patterns).toContain('tests');
      expect(config.include_patterns).toContain('examples');
      expect(config.exclude_patterns).toContain('build');
    });
  });

  describe('caching behavior', () => {
    it('should cache parsed configurations', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      expect(parser.getCacheSize()).toBe(0);

      // First call should parse and cache
      const config1 = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');
      expect(parser.getCacheSize()).toBe(1);

      // Second call should use cache
      const config2 = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');
      expect(parser.getCacheSize()).toBe(1);
      expect(config2).toBe(config1); // Same object reference from cache
    });

    it('should cache different configurations separately', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');
      await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'python');
      await parser.parseWorkspaceConfig('/different/workspace', 'typescript');

      expect(parser.getCacheSize()).toBe(3);
    });

    it('should clear cache when requested', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');
      expect(parser.getCacheSize()).toBe(1);

      parser.clearCache();
      expect(parser.getCacheSize()).toBe(0);
    });
  });

  describe('error handling', () => {
    it('should handle malformed JSON in tsconfig gracefully', async () => {
      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('tsconfig.json');
      });

      (fs.readFileSync as any).mockReturnValue('{ invalid json }');

      // Should not throw
      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');
      
      expect(config.root_path).toBe(mockWorkspaceRoot);
      expect(config.include_patterns).toContain('**/*.ts'); // Should use defaults
    });

    it('should handle file read errors gracefully', async () => {
      (fs.existsSync as any).mockReturnValue(true);
      (fs.readFileSync as any).mockImplementation(() => {
        throw new Error('Permission denied');
      });

      // Should handle gracefully and continue with defaults
      expect(async () => {
        await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');
      }).not.toThrow();
    });

    it('should handle unknown languages', async () => {
      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'unknown' as SupportedLanguage);
      
      expect(config.root_path).toBe(mockWorkspaceRoot);
      expect(config.include_patterns).toEqual([]);
      expect(config.exclude_patterns).toContain('node_modules/**');
      expect(config.exclude_patterns).toContain('.git/**');
    });

    it('should handle missing workspace root gracefully', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      const config = await parser.parseWorkspaceConfig('', 'typescript');
      
      expect(config.root_path).toBe('');
      expect(config.include_patterns).toContain('**/*.ts');
    });
  });

  describe('default excludes', () => {
    it('should include common exclude patterns for all languages', async () => {
      const languages: SupportedLanguage[] = ['typescript', 'python', 'rust', 'go', 'java', 'bash'];
      
      for (const language of languages) {
        (fs.existsSync as any).mockReturnValue(false);
        
        const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, language);
        
        expect(config.exclude_patterns).toContain('node_modules/**');
        expect(config.exclude_patterns).toContain('.git/**');
        expect(config.exclude_patterns).toContain('.svn/**');
        expect(config.exclude_patterns).toContain('**/.DS_Store');
        expect(config.exclude_patterns).toContain('**/Thumbs.db');
      }
    });

    it('should include language-specific exclude patterns', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      const typescriptConfig = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');
      expect(typescriptConfig.exclude_patterns).toContain('dist/**');
      expect(typescriptConfig.exclude_patterns).toContain('**/*.d.ts');

      const pythonConfig = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'python');
      expect(pythonConfig.exclude_patterns).toContain('**/__pycache__/**');
      expect(pythonConfig.exclude_patterns).toContain('**/*.pyc');

      const rustConfig = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'rust');
      expect(rustConfig.exclude_patterns).toContain('target/**');
      expect(rustConfig.exclude_patterns).toContain('Cargo.lock');

      const goConfig = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'go');
      expect(goConfig.exclude_patterns).toContain('vendor/**');

      const javaConfig = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'java');
      expect(javaConfig.exclude_patterns).toContain('target/**');
      expect(javaConfig.exclude_patterns).toContain('**/*.class');
    });
  });

  describe('complex workspace scenarios', () => {
    it('should handle monorepo TypeScript configuration', async () => {
      const mockRootTsConfig = {
        references: [
          { path: './packages/core' },
          { path: './packages/ui' },
          { path: './apps/web' }
        ]
      };

      const mockPackageJson = {
        workspaces: {
          packages: ['packages/*', 'apps/*']
        }
      };

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('tsconfig.json') || 
               path.endsWith('package.json') ||
               path.includes('/packages/') ||
               path.includes('/apps/');
      });

      (fs.readFileSync as any).mockImplementation((path: string) => {
        if (path.endsWith('tsconfig.json')) {
          return JSON.stringify(mockRootTsConfig);
        }
        if (path.endsWith('package.json')) {
          return JSON.stringify(mockPackageJson);
        }
        return '{}';
      });

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');

      expect(config.config_files).toContain('/test/workspace/tsconfig.json');
      expect(config.config_files).toContain('/test/workspace/package.json');
      expect(config.config_files).toContain('/test/workspace/packages/core');
      expect(config.config_files).toContain('/test/workspace/packages/ui');
      expect(config.config_files).toContain('/test/workspace/apps/web');
      
      expect(config.include_patterns).toContain('packages/*/**/*');
      expect(config.include_patterns).toContain('apps/*/**/*');
    });

    it('should handle complex path mappings in TypeScript', async () => {
      const mockTsConfig = {
        compilerOptions: {
          baseUrl: './src',
          paths: {
            '@core/*': ['../packages/core/src/*'],
            '@ui/*': ['../packages/ui/src/*'],
            '@shared/*': ['../shared/*'],
            '@/*': ['./*']
          }
        }
      };

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('tsconfig.json');
      });

      (fs.readFileSync as any).mockReturnValue(JSON.stringify(mockTsConfig));

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');

      expect(config.path_mappings.get('@core')).toBe('packages/core/src');
      expect(config.path_mappings.get('@ui')).toBe('packages/ui/src');
      expect(config.path_mappings.get('@shared')).toBe('shared');
      expect(config.path_mappings.get('@')).toBe('src');
    });

    it('should handle Rust workspace with multiple crates', async () => {
      const mockCargoToml = `
[workspace]
members = [
    "crates/core",
    "crates/utils", 
    "crates/cli",
    "examples/*"
]
exclude = [
    "archived/*",
    "experiments/*"
]
`;

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('Cargo.toml');
      });

      (fs.readFileSync as any).mockReturnValue(mockCargoToml);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'rust');

      expect(config.include_patterns).toContain('crates/core/**/*.rs');
      expect(config.include_patterns).toContain('crates/utils/**/*.rs');
      expect(config.include_patterns).toContain('crates/cli/**/*.rs');
      expect(config.include_patterns).toContain('examples/*/**/*.rs');
      expect(config.exclude_patterns).toContain('archived/*');
      expect(config.exclude_patterns).toContain('experiments/*');
    });
  });

  describe('performance and edge cases', () => {
    it('should handle very large configuration files', async () => {
      const largeConfig = {
        compilerOptions: {
          baseUrl: '.',
          paths: Object.fromEntries(
            Array.from({ length: 100 }, (_, i) => [`@lib${i}/*`, [`src/lib${i}/*`]])
          )
        },
        include: Array.from({ length: 50 }, (_, i) => `src/module${i}/**/*`),
        exclude: Array.from({ length: 30 }, (_, i) => `build${i}/**`)
      };

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('tsconfig.json');
      });

      (fs.readFileSync as any).mockReturnValue(JSON.stringify(largeConfig));

      const start = Date.now();
      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(100); // Should be fast even with large configs
      expect(config.path_mappings.size).toBe(100);
      expect(config.include_patterns.length).toBeGreaterThanOrEqual(50);
      expect(config.exclude_patterns.length).toBeGreaterThanOrEqual(30);
    });

    it('should handle empty configuration files', async () => {
      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('tsconfig.json');
      });

      (fs.readFileSync as any).mockReturnValue('{}');

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');

      expect(config.root_path).toBe(mockWorkspaceRoot);
      expect(config.include_patterns).toContain('**/*.ts'); // Should use defaults
    });

    it('should handle configuration files with only comments', async () => {
      const commentOnlyConfig = `
      // This is a comment file
      /* 
       * Multi-line comment
       */
      // Another comment
      `;

      (fs.existsSync as any).mockImplementation((path: string) => {
        return path.endsWith('tsconfig.json');
      });

      (fs.readFileSync as any).mockReturnValue(commentOnlyConfig);

      const config = await parser.parseWorkspaceConfig(mockWorkspaceRoot, 'typescript');

      expect(config.root_path).toBe(mockWorkspaceRoot);
      expect(config.include_patterns).toContain('**/*.ts'); // Should use defaults
    });
  });
});