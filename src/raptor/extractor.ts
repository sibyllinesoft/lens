/**
 * Semantic Card Extractor
 * 
 * Extracts semantic features from Python, TypeScript, and JavaScript files
 * using AST analysis and pattern matching to identify roles, resources,
 * shapes, domain tokens, and effects.
 */

import { SemanticCard, Language, Role, Effect, Resources, Shapes, PathHints, SemanticCardUtils, SemanticCardCacheKey } from './semantic-card.js';
import * as crypto from 'crypto';
import * as path from 'path';

export interface ExtractorConfig {
  utilityPackages: string[];
  domainStopWords: string[];
  pmiThreshold: number;
  embeddingDim: number;
}

export interface ParsedFile {
  file_id: string;
  file_path: string;
  content: string;
  file_sha: string;
  lang: Language;
}

export interface ExtractionContext {
  imports: string[];
  exports: string[];
  decorators: string[];
  functionCalls: string[];
  stringLiterals: string[];
  comments: string[];
  typeNames: string[];
  identifiers: string[];
}

/**
 * Main semantic card extractor
 */
export class SemanticCardExtractor {
  private config: ExtractorConfig;
  private cache: Map<string, SemanticCard>;

  constructor(config: ExtractorConfig) {
    this.config = config;
    this.cache = new Map();
  }

  static createDefaultConfig(): ExtractorConfig {
    return {
      utilityPackages: [
        // Python
        "logging", "sys", "os", "json", "time", "datetime", "uuid", "re", "math",
        "collections", "itertools", "functools", "typing", "pathlib", "asyncio",
        "pytest", "unittest", "mock", "requests", "httpx", "pydantic", "sqlalchemy",
        
        // TypeScript/JavaScript
        "lodash", "ramda", "moment", "dayjs", "axios", "fetch", "express", "koa",
        "react", "vue", "angular", "next", "nuxt", "webpack", "vite", "rollup",
        "jest", "mocha", "chai", "sinon", "cypress", "playwright", "puppeteer",
        
        // General utilities
        "utils", "util", "helpers", "common", "shared", "lib", "libs", "core"
      ],
      domainStopWords: [
        "data", "info", "item", "value", "object", "result", "response", "request",
        "config", "settings", "options", "params", "args", "props", "state", "context"
      ],
      pmiThreshold: 2.0,
      embeddingDim: 384
    };
  }

  async extractCard(file: ParsedFile): Promise<SemanticCard> {
    const cacheKey = SemanticCardUtils.getCacheKey(file.file_sha, file.content);
    const cacheKeyStr = `${cacheKey.file_sha}-${cacheKey.strings_hash}`;

    if (this.cache.has(cacheKeyStr)) {
      return this.cache.get(cacheKeyStr)!;
    }

    const context = this.parseFile(file);
    const card = await this.buildCard(file, context);
    
    this.cache.set(cacheKeyStr, card);
    return card;
  }

  private parseFile(file: ParsedFile): ExtractionContext {
    const context: ExtractionContext = {
      imports: [],
      exports: [],
      decorators: [],
      functionCalls: [],
      stringLiterals: [],
      comments: [],
      typeNames: [],
      identifiers: []
    };

    switch (file.lang) {
      case "py":
        return this.parsePython(file.content, context);
      case "ts":
      case "js":
        return this.parseTypeScript(file.content, context);
      default:
        return context;
    }
  }

  private parsePython(content: string, context: ExtractionContext): ExtractionContext {
    const lines = content.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();

      // Imports
      if (trimmed.startsWith('import ') || trimmed.startsWith('from ')) {
        context.imports.push(trimmed);
      }

      // Decorators
      if (trimmed.startsWith('@')) {
        context.decorators.push(trimmed);
      }

      // String literals
      const stringMatches = line.match(/['"]{1,3}([^'"]*)['"]{1,3}/g);
      if (stringMatches) {
        context.stringLiterals.push(...stringMatches.map(s => s.slice(1, -1)));
      }

      // Comments
      if (trimmed.startsWith('#')) {
        context.comments.push(trimmed.slice(1).trim());
      }

      // Function calls (simple pattern)
      const callMatches = line.match(/(\w+)\s*\(/g);
      if (callMatches) {
        context.functionCalls.push(...callMatches.map(m => m.replace('(', '')));
      }

      // Class definitions and type hints
      const classMatch = trimmed.match(/^class\s+(\w+)/);
      if (classMatch) {
        context.typeNames.push(classMatch[1]);
      }

      // Type annotations
      const typeMatches = line.match(/:\s*(\w+)/g);
      if (typeMatches) {
        context.typeNames.push(...typeMatches.map(m => m.replace(':', '').trim()));
      }
    }

    return context;
  }

  private parseTypeScript(content: string, context: ExtractionContext): ExtractionContext {
    const lines = content.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();

      // Imports/exports
      if (trimmed.startsWith('import ') || trimmed.startsWith('export ')) {
        if (trimmed.startsWith('import ')) context.imports.push(trimmed);
        if (trimmed.startsWith('export ')) context.exports.push(trimmed);
      }

      // Decorators
      if (trimmed.startsWith('@')) {
        context.decorators.push(trimmed);
      }

      // String literals
      const stringMatches = line.match(/['"]{1}([^'"]*)['"]{1}/g);
      if (stringMatches) {
        context.stringLiterals.push(...stringMatches.map(s => s.slice(1, -1)));
      }

      // Comments
      if (trimmed.startsWith('//')) {
        context.comments.push(trimmed.slice(2).trim());
      }

      // Function calls
      const callMatches = line.match(/(\w+)\s*\(/g);
      if (callMatches) {
        context.functionCalls.push(...callMatches.map(m => m.replace('(', '')));
      }

      // Type definitions
      const interfaceMatch = trimmed.match(/^interface\s+(\w+)/);
      const typeMatch = trimmed.match(/^type\s+(\w+)/);
      const classMatch = trimmed.match(/^class\s+(\w+)/);
      
      if (interfaceMatch) context.typeNames.push(interfaceMatch[1]);
      if (typeMatch) context.typeNames.push(typeMatch[1]);
      if (classMatch) context.typeNames.push(classMatch[1]);

      // Generic types and annotations
      const genericMatches = line.match(/<(\w+)>/g);
      if (genericMatches) {
        context.typeNames.push(...genericMatches.map(m => m.slice(1, -1)));
      }
    }

    return context;
  }

  private async buildCard(file: ParsedFile, context: ExtractionContext): Promise<SemanticCard> {
    const roles = this.extractRoles(context);
    const resources = this.extractResources(context);
    const shapes = this.extractShapes(context);
    const domainTokens = this.extractDomainTokens(context);
    const effects = this.extractEffects(context);
    const utilAffinity = this.computeUtilAffinity(context);
    const pathHints = this.extractPathHints(file.file_path);

    // Placeholder for actual embedding computation
    const e_sem = new Array(this.config.embeddingDim).fill(0);
    
    // Compute businessness score (simplified for now)
    const resourceCount = Object.values(resources).reduce((sum, arr) => sum + arr.length, 0);
    const shapeSpecificity = SemanticCardUtils.computeShapeSpecificity(shapes);
    const hasBusinessRole = roles.some(role => SemanticCardUtils.isBusinessRole(role));
    
    // Using simplified stats for now
    const stats = {
      pmiMean: 1.0, pmiStd: 0.5,
      resourceMean: 2.0, resourceStd: 1.0,
      shapeMean: 3.0, shapeStd: 2.0,
      utilMean: 0.3, utilStd: 0.2
    };

    const pmiDomain = this.computePMI(domainTokens);
    const B = SemanticCardUtils.computeBusinessnessScore(
      pmiDomain, resourceCount, shapeSpecificity, hasBusinessRole, utilAffinity, stats
    );

    return {
      file_id: file.file_id,
      file_sha: file.file_sha,
      lang: file.lang,
      roles,
      resources,
      shapes,
      domainTokens,
      effects,
      utilAffinity,
      pathHints,
      e_sem,
      B
    };
  }

  private extractRoles(context: ExtractionContext): Role[] {
    const roles = new Set<Role>();

    // Handler detection
    const handlerPatterns = [
      /@app\.route/, /@router\./, /@get/, /@post/, /@put/, /@delete/,
      /express\.Router/, /router\.get/, /router\.post/,
      /def.*handler/, /async def.*handler/, /Handler/, /Controller/
    ];

    for (const pattern of handlerPatterns) {
      const content = context.decorators.concat(context.functionCalls).join(' ');
      if (pattern.test(content)) {
        roles.add("handler");
        break;
      }
    }

    // Service detection
    const servicePatterns = [
      /Service/, /@service/, /@injectable/, /class.*Service/,
      /async def.*service/, /def.*service/
    ];

    for (const pattern of servicePatterns) {
      const content = context.typeNames.concat(context.decorators).join(' ');
      if (pattern.test(content)) {
        roles.add("service");
        break;
      }
    }

    // Repository detection
    const repoPatterns = [
      /Repository/, /@repository/, /class.*Repo/, /DAO/,
      /query/, /select/, /insert/, /update/, /delete/
    ];

    for (const pattern of repoPatterns) {
      const content = context.typeNames.concat(context.functionCalls).join(' ');
      if (pattern.test(content)) {
        roles.add("repo");
        break;
      }
    }

    // Job detection
    const jobPatterns = [
      /@task/, /@job/, /@celery/, /@cron/, /scheduler/,
      /async def.*job/, /def.*task/, /Worker/, /Job/
    ];

    for (const pattern of jobPatterns) {
      const content = context.decorators.concat(context.typeNames).join(' ');
      if (pattern.test(content)) {
        roles.add("job");
        break;
      }
    }

    // Validator detection
    const validatorPatterns = [
      /Validator/, /@validate/, /validate/, /Schema/,
      /pydantic/, /joi/, /yup/, /zod/
    ];

    for (const pattern of validatorPatterns) {
      const content = context.imports.concat(context.typeNames).join(' ');
      if (pattern.test(content)) {
        roles.add("validator");
        break;
      }
    }

    // Adapter detection
    const adapterPatterns = [
      /Adapter/, /Client/, /Wrapper/, /Proxy/,
      /requests/, /axios/, /fetch/, /httpx/
    ];

    for (const pattern of adapterPatterns) {
      const content = context.imports.concat(context.typeNames).join(' ');
      if (pattern.test(content)) {
        roles.add("adapter");
        break;
      }
    }

    return Array.from(roles);
  }

  private extractResources(context: ExtractionContext): Resources {
    const resources: Resources = {
      routes: [],
      sql: [],
      topics: [],
      buckets: [],
      featureFlags: []
    };

    // Extract routes from string literals and decorators
    const routePatterns = [
      /['"]\/([\w\-\/{}:]+)['"]/g,  // "/api/users/{id}"
      /@app\.route\s*\(\s*['"]([\w\-\/{}:]+)['"]/g,
      /router\.\w+\s*\(\s*['"]([\w\-\/{}:]+)['"]/g
    ];

    const allText = context.stringLiterals.concat(context.decorators).join(' ');
    for (const pattern of routePatterns) {
      const matches = allText.matchAll(pattern);
      for (const match of matches) {
        if (match[1] && match[1].startsWith('/')) {
          resources.routes.push(match[1]);
        }
      }
    }

    // Extract SQL table names
    const sqlPatterns = [
      /FROM\s+(\w+)/gi,
      /INTO\s+(\w+)/gi,
      /UPDATE\s+(\w+)/gi,
      /TABLE\s+(\w+)/gi,
      /table[_\s]*name\s*[=:]\s*['"'](\w+)['"']/gi
    ];

    for (const pattern of sqlPatterns) {
      const matches = allText.matchAll(pattern);
      for (const match of matches) {
        if (match[1]) {
          resources.sql.push(match[1]);
        }
      }
    }

    // Extract message topics
    const topicPatterns = [
      /topic[_\s]*name\s*[=:]\s*['"'](\w+)['"']/gi,
      /publish\s*\(\s*['"'](\w+)['"']/gi,
      /subscribe\s*\(\s*['"'](\w+)['"']/gi
    ];

    for (const pattern of topicPatterns) {
      const matches = allText.matchAll(pattern);
      for (const match of matches) {
        if (match[1]) {
          resources.topics.push(match[1]);
        }
      }
    }

    // Extract S3 buckets and prefixes
    const bucketPatterns = [
      /bucket[_\s]*name\s*[=:]\s*['"'](\w+)['"']/gi,
      /s3:\/\/(\w+)/gi
    ];

    for (const pattern of bucketPatterns) {
      const matches = allText.matchAll(pattern);
      for (const match of matches) {
        if (match[1]) {
          resources.buckets.push(match[1]);
        }
      }
    }

    // Extract feature flags
    const flagPatterns = [
      /feature[_\s]*flag\s*[=:]\s*['"'](\w+)['"']/gi,
      /is[_\s]*enabled\s*\(\s*['"'](\w+)['"']/gi,
      /flag[_\s]*enabled\s*[=:]\s*['"'](\w+)['"']/gi
    ];

    for (const pattern of flagPatterns) {
      const matches = allText.matchAll(pattern);
      for (const match of matches) {
        if (match[1]) {
          resources.featureFlags.push(match[1]);
        }
      }
    }

    return resources;
  }

  private extractShapes(context: ExtractionContext): Shapes {
    const shapes: Shapes = {
      typeNames: [...new Set(context.typeNames)],
      jsonKeys: []
    };

    // Extract JSON keys from object literals and string patterns
    const keyPatterns = [
      /['"'](\w+)['"']\s*:/g,  // "key": value
      /(\w+)\s*:/g             // key: value
    ];

    const allText = context.stringLiterals.join(' ');
    for (const pattern of keyPatterns) {
      const matches = allText.matchAll(pattern);
      for (const match of matches) {
        if (match[1] && !this.config.domainStopWords.includes(match[1].toLowerCase())) {
          shapes.jsonKeys.push(match[1]);
        }
      }
    }

    shapes.jsonKeys = [...new Set(shapes.jsonKeys)];
    return shapes;
  }

  private extractDomainTokens(context: ExtractionContext): string[] {
    const tokens = new Set<string>();
    
    // Extract from comments and string literals
    const text = context.comments.concat(context.stringLiterals).join(' ');
    const words = text
      .toLowerCase()
      .replace(/[^a-zA-Z\s]/g, ' ')
      .split(/\s+/)
      .filter(word => 
        word.length > 3 &&
        !this.config.domainStopWords.includes(word) &&
        !this.config.utilityPackages.includes(word)
      );

    for (const word of words) {
      if (this.computePMI([word]) > this.config.pmiThreshold) {
        tokens.add(word);
      }
    }

    return Array.from(tokens).slice(0, 20); // Limit to top 20 tokens
  }

  private extractEffects(context: ExtractionContext): Effect[] {
    const effects = new Set<Effect>();

    // File system effects
    if (context.imports.some(imp => /\b(fs|os|pathlib|file)\b/.test(imp)) ||
        context.functionCalls.some(call => /\b(open|read|write|unlink|mkdir)\b/.test(call))) {
      effects.add("fs");
    }

    // Network effects
    if (context.imports.some(imp => /\b(requests|axios|fetch|httpx|urllib|http)\b/.test(imp))) {
      effects.add("net");
    }

    // Database effects
    if (context.imports.some(imp => /\b(sqlalchemy|django|sequelize|mongoose|redis)\b/.test(imp)) ||
        context.functionCalls.some(call => /\b(query|execute|commit|rollback)\b/.test(call))) {
      effects.add("db");
    }

    // Cache effects
    if (context.imports.some(imp => /\b(redis|memcached|cache)\b/.test(imp)) ||
        context.functionCalls.some(call => /\b(get|set|delete).*cache\b/.test(call))) {
      effects.add("cache");
    }

    // Auth effects
    if (context.imports.some(imp => /\b(auth|jwt|oauth|passport)\b/.test(imp)) ||
        context.functionCalls.some(call => /\b(login|logout|authenticate|authorize)\b/.test(call))) {
      effects.add("auth");
    }

    // Email effects
    if (context.imports.some(imp => /\b(email|smtp|sendgrid|mailgun)\b/.test(imp)) ||
        context.functionCalls.some(call => /\b(send.*mail|email)\b/.test(call))) {
      effects.add("email");
    }

    // Crypto effects
    if (context.imports.some(imp => /\b(crypto|hashlib|bcrypt|jwt)\b/.test(imp)) ||
        context.functionCalls.some(call => /\b(hash|encrypt|decrypt|sign)\b/.test(call))) {
      effects.add("crypto");
    }

    return Array.from(effects);
  }

  private computeUtilAffinity(context: ExtractionContext): number {
    const totalImports = context.imports.length;
    if (totalImports === 0) return 0;

    const utilImports = context.imports.filter(imp => 
      this.config.utilityPackages.some(util => imp.includes(util))
    ).length;

    return utilImports / totalImports;
  }

  private extractPathHints(filePath: string): PathHints {
    const parts = filePath.split('/');
    const depth = parts.length;
    
    // Generate n-grams from path components
    const ngrams: string[] = [];
    for (let i = 0; i < parts.length; i++) {
      for (let j = i + 1; j <= Math.min(i + 3, parts.length); j++) {
        ngrams.push(parts.slice(i, j).join('/'));
      }
    }

    return {
      ngrams,
      depth,
      recentlyTouched: false // This would be set based on git history
    };
  }

  private computePMI(tokens: string[]): number {
    // Simplified PMI computation - in production, this would use
    // actual corpus statistics
    return tokens.length > 0 ? Math.random() * 5 : 0;
  }

  clearCache(): void {
    this.cache.clear();
  }

  getCacheSize(): number {
    return this.cache.size;
  }
}

export default SemanticCardExtractor;