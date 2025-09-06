/**
 * Workspace Configuration Import System
 * Parses tsconfig.json/pyproject.toml/go.mod/Cargo.toml
 * Replicates LSP path mapping & excludes in Stage-A path priors and indexer
 */

import { existsSync, readFileSync } from 'fs';
import { join, resolve, dirname, relative } from 'path';
import type { WorkspaceConfig, SupportedLanguage } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

interface TSConfig {
  compilerOptions?: {
    baseUrl?: string;
    paths?: { [key: string]: string[] };
    rootDirs?: string[];
    include?: string[];
    exclude?: string[];
  };
  include?: string[];
  exclude?: string[];
  references?: Array<{ path: string }>;
}

interface PyProject {
  tool?: {
    pyright?: {
      include?: string[];
      exclude?: string[];
      extraPaths?: string[];
    };
    mypy?: {
      files?: string[];
      exclude?: string[];
      mypy_path?: string[];
    };
  };
}

interface CargoToml {
  workspace?: {
    members?: string[];
    exclude?: string[];
  };
  package?: {
    name?: string;
  };
  lib?: {
    path?: string;
  };
}

interface GoMod {
  module?: string;
  replace?: { [key: string]: string };
  exclude?: string[];
}

export class WorkspaceConfigParser {
  private configCache = new Map<string, WorkspaceConfig>();

  /**
   * Parse workspace configuration for a given language and root path
   */
  async parseWorkspaceConfig(
    workspaceRoot: string, 
    language: SupportedLanguage
  ): Promise<WorkspaceConfig> {
    const span = LensTracer.createChildSpan('parse_workspace_config', {
      'workspace.root': workspaceRoot,
      'workspace.language': language,
    });

    try {
      const cacheKey = `${workspaceRoot}:${language}`;
      
      // Check cache first
      if (this.configCache.has(cacheKey)) {
        const cached = this.configCache.get(cacheKey)!;
        span.setAttributes({ success: true, cache_hit: true });
        return cached;
      }

      const config = await this.parseConfigForLanguage(workspaceRoot, language);
      
      // Cache the result
      this.configCache.set(cacheKey, config);
      
      span.setAttributes({
        success: true,
        cache_hit: false,
        include_patterns: config.include_patterns.length,
        exclude_patterns: config.exclude_patterns.length,
        path_mappings: config.path_mappings.size,
        config_files: config.config_files.length,
      });

      return config;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Parse configuration based on language
   */
  private async parseConfigForLanguage(
    workspaceRoot: string,
    language: SupportedLanguage
  ): Promise<WorkspaceConfig> {
    const baseConfig: WorkspaceConfig = {
      root_path: workspaceRoot,
      include_patterns: [],
      exclude_patterns: this.getDefaultExcludes(language),
      path_mappings: new Map(),
      config_files: [],
    };

    switch (language) {
      case 'typescript':
        return await this.parseTypeScriptConfig(workspaceRoot, baseConfig);
      case 'python':
        return await this.parsePythonConfig(workspaceRoot, baseConfig);
      case 'rust':
        return await this.parseRustConfig(workspaceRoot, baseConfig);
      case 'go':
        return await this.parseGoConfig(workspaceRoot, baseConfig);
      case 'java':
        return await this.parseJavaConfig(workspaceRoot, baseConfig);
      case 'bash':
        return await this.parseBashConfig(workspaceRoot, baseConfig);
      default:
        return baseConfig;
    }
  }

  /**
   * Parse TypeScript configuration
   */
  private async parseTypeScriptConfig(
    workspaceRoot: string, 
    baseConfig: WorkspaceConfig
  ): Promise<WorkspaceConfig> {
    const config = { ...baseConfig };
    
    // Look for tsconfig.json files
    const tsconfigPaths = [
      join(workspaceRoot, 'tsconfig.json'),
      join(workspaceRoot, 'tsconfig.base.json'),
      join(workspaceRoot, 'jsconfig.json'),
    ];

    for (const tsconfigPath of tsconfigPaths) {
      if (existsSync(tsconfigPath)) {
        config.config_files.push(tsconfigPath);
        const tsconfig = this.parseTSConfigFile(tsconfigPath);
        
        // Process compiler options
        if (tsconfig.compilerOptions) {
          this.processTSCompilerOptions(tsconfig.compilerOptions, config, workspaceRoot);
        }

        // Process includes and excludes
        if (tsconfig.include) {
          config.include_patterns.push(...tsconfig.include);
        }
        if (tsconfig.exclude) {
          config.exclude_patterns.push(...tsconfig.exclude);
        }

        // Process project references
        if (tsconfig.references) {
          for (const ref of tsconfig.references) {
            const refPath = resolve(dirname(tsconfigPath), ref.path);
            if (existsSync(refPath)) {
              config.config_files.push(refPath);
            }
          }
        }
      }
    }

    // Look for package.json with TypeScript config
    const packageJsonPath = join(workspaceRoot, 'package.json');
    if (existsSync(packageJsonPath)) {
      config.config_files.push(packageJsonPath);
      const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8'));
      
      if (packageJson.workspaces) {
        // Handle workspace patterns
        const workspaces = Array.isArray(packageJson.workspaces) 
          ? packageJson.workspaces 
          : packageJson.workspaces.packages || [];
        
        for (const workspace of workspaces) {
          config.include_patterns.push(join(workspace, '**/*'));
        }
      }
    }

    // Default TypeScript includes if none specified
    if (config.include_patterns.length === 0) {
      config.include_patterns.push(
        '**/*.ts',
        '**/*.tsx',
        '**/*.js',
        '**/*.jsx'
      );
    }

    return config;
  }

  /**
   * Parse TypeScript compiler options
   */
  private processTSCompilerOptions(
    compilerOptions: NonNullable<TSConfig['compilerOptions']>,
    config: WorkspaceConfig,
    workspaceRoot: string
  ): void {
    // Process path mappings
    if (compilerOptions.paths && compilerOptions.baseUrl) {
      const baseUrl = resolve(workspaceRoot, compilerOptions.baseUrl);
      
      for (const [alias, paths] of Object.entries(compilerOptions.paths)) {
        const resolvedPaths = paths.map(p => 
          resolve(baseUrl, p.replace('*', ''))
        );
        
        // Store the first path as the primary mapping
        if (resolvedPaths.length > 0) {
          config.path_mappings.set(
            alias.replace('/*', ''),
            relative(workspaceRoot, resolvedPaths[0])
          );
        }
      }
    }

    // Process root directories
    if (compilerOptions.rootDirs) {
      for (const rootDir of compilerOptions.rootDirs) {
        const fullPath = resolve(workspaceRoot, rootDir);
        config.include_patterns.push(join(relative(workspaceRoot, fullPath), '**/*'));
      }
    }
  }

  /**
   * Parse Python configuration
   */
  private async parsePythonConfig(
    workspaceRoot: string,
    baseConfig: WorkspaceConfig
  ): Promise<WorkspaceConfig> {
    const config = { ...baseConfig };

    // Look for pyproject.toml
    const pyprojectPath = join(workspaceRoot, 'pyproject.toml');
    if (existsSync(pyprojectPath)) {
      config.config_files.push(pyprojectPath);
      const pyprojectContent = readFileSync(pyprojectPath, 'utf8');
      const pyproject = this.parseToml(pyprojectContent) as PyProject;
      
      // Process pyright configuration
      if (pyproject.tool?.pyright) {
        const pyrightConfig = pyproject.tool.pyright;
        
        if (pyrightConfig.include) {
          config.include_patterns.push(...pyrightConfig.include);
        }
        if (pyrightConfig.exclude) {
          config.exclude_patterns.push(...pyrightConfig.exclude);
        }
        if (pyrightConfig.extraPaths) {
          for (const extraPath of pyrightConfig.extraPaths) {
            config.path_mappings.set(extraPath, extraPath);
          }
        }
      }

      // Process mypy configuration
      if (pyproject.tool?.mypy) {
        const mypyConfig = pyproject.tool.mypy;
        
        if (mypyConfig.files) {
          config.include_patterns.push(...mypyConfig.files);
        }
        if (mypyConfig.exclude) {
          config.exclude_patterns.push(...mypyConfig.exclude);
        }
        if (mypyConfig.mypy_path) {
          for (const path of mypyConfig.mypy_path) {
            config.path_mappings.set(path, path);
          }
        }
      }
    }

    // Look for setup.py, setup.cfg
    const setupFiles = [
      join(workspaceRoot, 'setup.py'),
      join(workspaceRoot, 'setup.cfg'),
    ];

    for (const setupFile of setupFiles) {
      if (existsSync(setupFile)) {
        config.config_files.push(setupFile);
      }
    }

    // Default Python includes
    if (config.include_patterns.length === 0) {
      config.include_patterns.push('**/*.py', '**/*.pyi');
    }

    return config;
  }

  /**
   * Parse Rust configuration
   */
  private async parseRustConfig(
    workspaceRoot: string,
    baseConfig: WorkspaceConfig
  ): Promise<WorkspaceConfig> {
    const config = { ...baseConfig };

    // Look for Cargo.toml
    const cargoTomlPath = join(workspaceRoot, 'Cargo.toml');
    if (existsSync(cargoTomlPath)) {
      config.config_files.push(cargoTomlPath);
      const cargoContent = readFileSync(cargoTomlPath, 'utf8');
      const cargo = this.parseToml(cargoContent) as CargoToml;

      // Process workspace members
      if (cargo.workspace?.members) {
        for (const member of cargo.workspace.members) {
          config.include_patterns.push(join(member, '**/*.rs'));
        }
      }

      // Process workspace excludes
      if (cargo.workspace?.exclude) {
        config.exclude_patterns.push(...cargo.workspace.exclude);
      }

      // Process library path
      if (cargo.lib?.path) {
        const libPath = dirname(cargo.lib.path);
        config.include_patterns.push(join(libPath, '**/*.rs'));
      }
    }

    // Default Rust includes
    if (config.include_patterns.length === 0) {
      config.include_patterns.push(
        'src/**/*.rs',
        'tests/**/*.rs',
        'examples/**/*.rs',
        'benches/**/*.rs'
      );
    }

    return config;
  }

  /**
   * Parse Go configuration
   */
  private async parseGoConfig(
    workspaceRoot: string,
    baseConfig: WorkspaceConfig
  ): Promise<WorkspaceConfig> {
    const config = { ...baseConfig };

    // Look for go.mod
    const goModPath = join(workspaceRoot, 'go.mod');
    if (existsSync(goModPath)) {
      config.config_files.push(goModPath);
      const goModContent = readFileSync(goModPath, 'utf8');
      const goMod = this.parseGoMod(goModContent);

      // Process replace directives
      if (goMod.replace) {
        for (const [from, to] of Object.entries(goMod.replace)) {
          config.path_mappings.set(from, to);
        }
      }

      // Process excludes
      if (goMod.exclude) {
        config.exclude_patterns.push(...goMod.exclude);
      }
    }

    // Look for go.work
    const goWorkPath = join(workspaceRoot, 'go.work');
    if (existsSync(goWorkPath)) {
      config.config_files.push(goWorkPath);
      // Process workspace configuration
    }

    // Default Go includes
    if (config.include_patterns.length === 0) {
      config.include_patterns.push('**/*.go');
    }

    return config;
  }

  /**
   * Parse Java configuration
   */
  private async parseJavaConfig(
    workspaceRoot: string,
    baseConfig: WorkspaceConfig
  ): Promise<WorkspaceConfig> {
    const config = { ...baseConfig };

    // Look for build files
    const buildFiles = [
      join(workspaceRoot, 'pom.xml'),
      join(workspaceRoot, 'build.gradle'),
      join(workspaceRoot, 'build.gradle.kts'),
      join(workspaceRoot, 'settings.gradle'),
    ];

    for (const buildFile of buildFiles) {
      if (existsSync(buildFile)) {
        config.config_files.push(buildFile);
      }
    }

    // Default Java includes
    config.include_patterns.push(
      '**/src/main/java/**/*.java',
      '**/src/test/java/**/*.java',
      '**/*.java'
    );

    return config;
  }

  /**
   * Parse Bash configuration
   */
  private async parseBashConfig(
    workspaceRoot: string,
    baseConfig: WorkspaceConfig
  ): Promise<WorkspaceConfig> {
    const config = { ...baseConfig };

    // Default Bash includes
    config.include_patterns.push(
      '**/*.sh',
      '**/*.bash',
      '**/.*rc',
      '**/.*profile'
    );

    return config;
  }

  /**
   * Parse tsconfig.json file
   */
  private parseTSConfigFile(tsconfigPath: string): TSConfig {
    try {
      const content = readFileSync(tsconfigPath, 'utf8');
      // Handle JSON with comments
      const cleanContent = content.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
      return JSON.parse(cleanContent);
    } catch (error) {
      console.warn(`Failed to parse ${tsconfigPath}:`, error);
      return {};
    }
  }

  /**
   * Simple TOML parser (simplified implementation)
   */
  private parseToml(content: string): any {
    const result: any = {};
    let currentSection: any = result;
    let currentPath: string[] = [];

    const lines = content.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;

      // Section headers
      if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
        const sectionName = trimmed.slice(1, -1);
        currentPath = sectionName.split('.');
        currentSection = result;
        
        for (const part of currentPath) {
          if (!currentSection[part]) {
            currentSection[part] = {};
          }
          currentSection = currentSection[part];
        }
        continue;
      }

      // Key-value pairs
      const equalIndex = trimmed.indexOf('=');
      if (equalIndex > 0) {
        const key = trimmed.slice(0, equalIndex).trim();
        const value = trimmed.slice(equalIndex + 1).trim();
        
        currentSection[key] = this.parseTomlValue(value);
      }
    }

    return result;
  }

  /**
   * Parse TOML value
   */
  private parseTomlValue(value: string): any {
    value = value.trim();

    // String values
    if (value.startsWith('"') && value.endsWith('"')) {
      return value.slice(1, -1);
    }
    if (value.startsWith("'") && value.endsWith("'")) {
      return value.slice(1, -1);
    }

    // Array values
    if (value.startsWith('[') && value.endsWith(']')) {
      const arrayContent = value.slice(1, -1).trim();
      if (!arrayContent) return [];
      
      return arrayContent.split(',').map(item => this.parseTomlValue(item.trim()));
    }

    // Boolean values
    if (value === 'true') return true;
    if (value === 'false') return false;

    // Number values
    if (/^\d+$/.test(value)) return parseInt(value, 10);
    if (/^\d*\.\d+$/.test(value)) return parseFloat(value);

    return value;
  }

  /**
   * Parse go.mod file
   */
  private parseGoMod(content: string): GoMod {
    const result: GoMod = {};
    const lines = content.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('//')) continue;

      if (trimmed.startsWith('module ')) {
        result.module = trimmed.slice(7).trim();
      } else if (trimmed.startsWith('replace ')) {
        if (!result.replace) result.replace = {};
        const replaceContent = trimmed.slice(8).trim();
        const arrowIndex = replaceContent.indexOf(' => ');
        if (arrowIndex > 0) {
          const from = replaceContent.slice(0, arrowIndex).trim();
          const to = replaceContent.slice(arrowIndex + 4).trim();
          result.replace[from] = to;
        }
      }
    }

    return result;
  }

  /**
   * Get default exclude patterns for a language
   */
  private getDefaultExcludes(language: SupportedLanguage): string[] {
    const common = [
      'node_modules/**',
      '.git/**',
      '.svn/**',
      '.hg/**',
      '**/.DS_Store',
      '**/Thumbs.db',
    ];

    const languageSpecific = {
      typescript: ['dist/**', 'build/**', '**/*.d.ts'],
      python: ['**/__pycache__/**', '**/*.pyc', '.venv/**', 'venv/**'],
      rust: ['target/**', 'Cargo.lock'],
      go: ['vendor/**'],
      java: ['target/**', 'build/**', '**/*.class'],
      bash: [],
    };

    return [...common, ...languageSpecific[language]];
  }

  /**
   * Clear configuration cache
   */
  clearCache(): void {
    this.configCache.clear();
  }

  /**
   * Get cached configuration count
   */
  getCacheSize(): number {
    return this.configCache.size;
  }
}