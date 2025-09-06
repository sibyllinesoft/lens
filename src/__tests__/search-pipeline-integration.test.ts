/**
 * Integration Tests for 3-Stage Search Pipeline
 * Tests the complete lexical → symbol → semantic search flow with realistic data
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  SearchEngine,
  SearchRequest,
  SearchResponse,
  SearchStageResult,
} from '../api/search-engine.js';
import { LexicalIndex } from '../indexer/lexical.js';
import { SymbolIndex } from '../indexer/enhanced-symbols.js';
import { SemanticIndex } from '../indexer/enhanced-semantic.js';
import { QueryClassifier } from '../core/query-classifier.js';
import { LearnedReranker } from '../core/learned-reranker.js';
import type { SearchHit, SearchContext } from '../types/core.js';

// Mock telemetry
vi.mock('../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      setStatus: vi.fn(),
      end: vi.fn(),
    })),
    startSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      setStatus: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

describe('3-Stage Search Pipeline Integration', () => {
  let searchEngine: SearchEngine;
  let testCodebase: Array<{
    file: string;
    content: string;
    symbols: Array<{ name: string; type: string; line: number }>;
  }>;

  beforeEach(async () => {
    // Set up test codebase with realistic code samples
    testCodebase = [
      {
        file: 'src/auth/authentication.ts',
        content: `
export interface User {
  id: string;
  email: string;
  isActive: boolean;
}

export class AuthenticationService {
  private users: Map<string, User> = new Map();

  async authenticate(email: string, password: string): Promise<User | null> {
    const user = await this.findUserByEmail(email);
    if (!user || !this.validatePassword(password, user)) {
      return null;
    }
    return user;
  }

  private async findUserByEmail(email: string): Promise<User | null> {
    // Find user by email implementation
    return this.users.get(email) || null;
  }

  private validatePassword(password: string, user: User): boolean {
    // Password validation logic
    return password.length >= 8;
  }
}`,
        symbols: [
          { name: 'User', type: 'interface', line: 2 },
          { name: 'AuthenticationService', type: 'class', line: 7 },
          { name: 'authenticate', type: 'method', line: 10 },
          { name: 'findUserByEmail', type: 'method', line: 17 },
          { name: 'validatePassword', type: 'method', line: 22 },
        ],
      },
      {
        file: 'src/api/user-controller.ts',
        content: `
import { Request, Response } from 'express';
import { AuthenticationService } from '../auth/authentication.js';
import { UserService } from './user-service.js';

export class UserController {
  constructor(
    private authService: AuthenticationService,
    private userService: UserService
  ) {}

  async login(req: Request, res: Response) {
    const { email, password } = req.body;
    
    const user = await this.authService.authenticate(email, password);
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    return res.json({ user, token: this.generateToken(user) });
  }

  async register(req: Request, res: Response) {
    const userData = req.body;
    const user = await this.userService.createUser(userData);
    return res.status(201).json({ user });
  }

  private generateToken(user: any): string {
    return \`token-\${user.id}\`;
  }
}`,
        symbols: [
          { name: 'UserController', type: 'class', line: 6 },
          { name: 'login', type: 'method', line: 12 },
          { name: 'register', type: 'method', line: 23 },
          { name: 'generateToken', type: 'method', line: 28 },
        ],
      },
      {
        file: 'src/api/user-service.ts',
        content: `
import { User } from '../auth/authentication.js';

export interface CreateUserRequest {
  email: string;
  password: string;
  firstName?: string;
  lastName?: string;
}

export class UserService {
  private users: User[] = [];

  async createUser(request: CreateUserRequest): Promise<User> {
    const user: User = {
      id: this.generateId(),
      email: request.email,
      isActive: true,
    };
    
    this.users.push(user);
    return user;
  }

  async getUserById(id: string): Promise<User | null> {
    return this.users.find(user => user.id === id) || null;
  }

  async getAllUsers(): Promise<User[]> {
    return this.users.filter(user => user.isActive);
  }

  private generateId(): string {
    return Math.random().toString(36).substr(2, 9);
  }
}`,
        symbols: [
          { name: 'CreateUserRequest', type: 'interface', line: 4 },
          { name: 'UserService', type: 'class', line: 10 },
          { name: 'createUser', type: 'method', line: 13 },
          { name: 'getUserById', type: 'method', line: 24 },
          { name: 'getAllUsers', type: 'method', line: 28 },
          { name: 'generateId', type: 'method', line: 32 },
        ],
      },
      {
        file: 'tests/auth.test.ts',
        content: `
import { describe, it, expect } from 'vitest';
import { AuthenticationService, User } from '../src/auth/authentication.js';

describe('AuthenticationService', () => {
  let authService: AuthenticationService;

  beforeEach(() => {
    authService = new AuthenticationService();
  });

  it('should authenticate valid user', async () => {
    const user: User = {
      id: '123',
      email: 'test@example.com',
      isActive: true,
    };

    const result = await authService.authenticate('test@example.com', 'password123');
    expect(result).toBeDefined();
    expect(result?.email).toBe('test@example.com');
  });

  it('should reject invalid credentials', async () => {
    const result = await authService.authenticate('invalid@example.com', 'wrong');
    expect(result).toBeNull();
  });
});`,
        symbols: [
          { name: 'describe', type: 'function', line: 5 },
          { name: 'it', type: 'function', line: 12 },
          { name: 'it', type: 'function', line: 22 },
        ],
      },
      {
        file: 'README.md',
        content: `
# User Authentication System

This project implements a comprehensive user authentication system with the following features:

## Features

- User registration and login
- Password validation
- JWT token generation  
- User management API

## Authentication Flow

1. User submits email and password
2. System validates credentials using AuthenticationService
3. If valid, returns user data and JWT token
4. Token is used for subsequent authenticated requests

## API Endpoints

- \`POST /api/auth/login\` - User login
- \`POST /api/auth/register\` - User registration
- \`GET /api/users/:id\` - Get user by ID
- \`GET /api/users\` - Get all active users

## Testing

Run tests with \`npm test\`. The test suite covers:
- Authentication service functionality
- User controller endpoints
- User service operations
`,
        symbols: [],
      },
    ];

    // Initialize search engine with test data
    searchEngine = new SearchEngine({
      indexPath: './test-index',
      maxResults: 50,
      timeoutMs: 30000,
    });

    await searchEngine.initialize();

    // Index the test codebase
    for (const file of testCodebase) {
      await searchEngine.indexFile(file.file, file.content, file.symbols);
    }
  });

  afterEach(async () => {
    await searchEngine.destroy();
  });

  describe('Stage 1: Lexical Search', () => {
    it('should perform exact keyword matching', async () => {
      const request: SearchRequest = {
        query: 'authenticate',
        mode: 'lex',
        max_results: 10,
      };

      const response = await searchEngine.search(request);

      expect(response.hits).toBeDefined();
      expect(response.hits.length).toBeGreaterThan(0);

      // Should find exact matches for "authenticate"
      const authenticateHits = response.hits.filter(hit =>
        hit.snippet.toLowerCase().includes('authenticate')
      );
      
      expect(authenticateHits.length).toBeGreaterThan(0);
      expect(response.processing_time_ms).toBeLessThan(100); // Fast lexical search
      expect(response.stages.lexical).toBeDefined();
      expect(response.stages.lexical.candidates).toBeGreaterThan(0);
    });

    it('should handle fuzzy matching for typos', async () => {
      const request: SearchRequest = {
        query: 'authentcate', // Missing 'i'
        mode: 'lex',
        max_results: 10,
      };

      const response = await searchEngine.search(request);

      expect(response.hits.length).toBeGreaterThan(0);
      
      // Should still find authenticate-related results
      const relevantHits = response.hits.filter(hit =>
        hit.snippet.toLowerCase().includes('authenticate') ||
        hit.file.includes('auth')
      );
      
      expect(relevantHits.length).toBeGreaterThan(0);
    });

    it('should score matches by relevance', async () => {
      const request: SearchRequest = {
        query: 'user authentication',
        mode: 'lex',
        max_results: 10,
      };

      const response = await searchEngine.search(request);

      expect(response.hits.length).toBeGreaterThan(0);
      
      // Results should be ordered by score (descending)
      for (let i = 1; i < response.hits.length; i++) {
        expect(response.hits[i-1].score).toBeGreaterThanOrEqual(response.hits[i].score);
      }
      
      // Top result should be highly relevant
      expect(response.hits[0].score).toBeGreaterThan(0.5);
    });
  });

  describe('Stage 2: Symbol-Enhanced Search', () => {
    it('should enhance results with symbol information', async () => {
      const request: SearchRequest = {
        query: 'AuthenticationService',
        mode: 'sym',
        max_results: 10,
      };

      const response = await searchEngine.search(request);

      expect(response.hits.length).toBeGreaterThan(0);
      expect(response.stages.symbol).toBeDefined();
      
      // Should find symbol matches
      const symbolHits = response.hits.filter(hit =>
        hit.symbol_name === 'AuthenticationService'
      );
      
      expect(symbolHits.length).toBeGreaterThan(0);
      
      // Symbol hits should be scored highly
      expect(symbolHits[0].score).toBeGreaterThan(0.8);
    });

    it('should find method and property references', async () => {
      const request: SearchRequest = {
        query: 'login method',
        mode: 'sym',
        max_results: 10,
      };

      const response = await searchEngine.search(request);

      expect(response.hits.length).toBeGreaterThan(0);
      
      // Should find login method
      const loginHits = response.hits.filter(hit =>
        hit.symbol_name === 'login' || 
        hit.snippet.includes('login(')
      );
      
      expect(loginHits.length).toBeGreaterThan(0);
    });

    it('should understand symbol relationships and imports', async () => {
      const request: SearchRequest = {
        query: 'User interface import',
        mode: 'sym',
        max_results: 10,
      };

      const response = await searchEngine.search(request);

      expect(response.hits.length).toBeGreaterThan(0);
      
      // Should find User interface definition and imports
      const userHits = response.hits.filter(hit =>
        hit.snippet.includes('User') && 
        (hit.snippet.includes('interface') || hit.snippet.includes('import'))
      );
      
      expect(userHits.length).toBeGreaterThan(0);
    });
  });

  describe('Stage 3: Semantic Search (Hybrid Mode)', () => {
    it('should perform semantic understanding of natural language queries', async () => {
      const request: SearchRequest = {
        query: 'how to validate user password',
        mode: 'hybrid',
        max_results: 10,
      };

      const response = await searchEngine.search(request);

      expect(response.hits.length).toBeGreaterThan(0);
      expect(response.stages.semantic).toBeDefined();
      
      // Should find password validation related code
      const passwordHits = response.hits.filter(hit =>
        hit.snippet.toLowerCase().includes('password') ||
        hit.snippet.toLowerCase().includes('validate')
      );
      
      expect(passwordHits.length).toBeGreaterThan(0);
      
      // Should specifically find the validatePassword method
      const validateMethod = response.hits.find(hit =>
        hit.symbol_name === 'validatePassword'
      );
      
      expect(validateMethod).toBeDefined();
    });

    it('should combine lexical, symbol, and semantic signals', async () => {
      const request: SearchRequest = {
        query: 'create new user account',
        mode: 'hybrid',
        max_results: 10,
      };

      const response = await searchEngine.search(request);

      expect(response.hits.length).toBeGreaterThan(0);
      
      // Should combine multiple search strategies
      expect(response.stages.lexical.candidates).toBeGreaterThan(0);
      expect(response.stages.symbol.candidates).toBeGreaterThan(0);
      expect(response.stages.semantic.candidates).toBeGreaterThan(0);
      
      // Should find createUser method
      const createUserHit = response.hits.find(hit =>
        hit.symbol_name === 'createUser'
      );
      
      expect(createUserHit).toBeDefined();
      expect(createUserHit?.score).toBeGreaterThan(0.7);
    });

    it('should handle complex conceptual queries', async () => {
      const request: SearchRequest = {
        query: 'authentication flow with token generation',
        mode: 'hybrid',
        max_results: 15,
      };

      const response = await searchEngine.search(request);

      expect(response.hits.length).toBeGreaterThan(0);
      
      // Should find authentication, login, and token generation code
      const relevantFiles = new Set(response.hits.map(hit => hit.file));
      
      expect(relevantFiles.has('src/auth/authentication.ts')).toBe(true);
      expect(relevantFiles.has('src/api/user-controller.ts')).toBe(true);
      
      // Should include generateToken method
      const tokenHit = response.hits.find(hit =>
        hit.symbol_name === 'generateToken'
      );
      
      expect(tokenHit).toBeDefined();
    });

    it('should apply learned reranking for better results', async () => {
      const request: SearchRequest = {
        query: 'user login authentication endpoint',
        mode: 'hybrid',
        max_results: 10,
        apply_reranking: true,
      };

      const response = await searchEngine.search(request);

      expect(response.hits.length).toBeGreaterThan(0);
      expect(response.reranking_applied).toBe(true);
      
      // Should prioritize implementation over tests/docs
      const topHits = response.hits.slice(0, 3);
      const implementationHits = topHits.filter(hit =>
        hit.file.includes('src/') && !hit.file.includes('test')
      );
      
      expect(implementationHits.length).toBeGreaterThan(0);
    });
  });

  describe('End-to-End Pipeline Performance', () => {
    it('should complete searches within latency targets', async () => {
      const queries = [
        'authenticate user',
        'createUser method',
        'how to register new user',
        'password validation logic',
        'user login endpoint',
      ];

      for (const query of queries) {
        const start = performance.now();
        
        const response = await searchEngine.search({
          query,
          mode: 'hybrid',
          max_results: 10,
        });
        
        const duration = performance.now() - start;
        
        // Should meet sub-20ms target
        expect(duration).toBeLessThan(20);
        expect(response.hits.length).toBeGreaterThan(0);
        expect(response.processing_time_ms).toBeLessThan(20);
      }
    });

    it('should handle concurrent search requests', async () => {
      const requests = Array.from({ length: 10 }, (_, i) => ({
        query: `search query ${i}`,
        mode: 'hybrid' as const,
        max_results: 10,
      }));

      const start = performance.now();
      const responses = await Promise.all(
        requests.map(req => searchEngine.search(req))
      );
      const totalDuration = performance.now() - start;

      // All requests should complete
      expect(responses).toHaveLength(10);
      responses.forEach(response => {
        expect(response).toBeDefined();
        expect(response.processing_time_ms).toBeLessThan(50);
      });

      // Total time should be reasonable (parallelization should help)
      expect(totalDuration).toBeLessThan(200);
    });

    it('should maintain quality with increasing index size', async () => {
      // Add more files to increase index size
      const additionalFiles = Array.from({ length: 50 }, (_, i) => ({
        file: `src/module${i}.ts`,
        content: `
export class Module${i} {
  process(): void {
    console.log('Processing module ${i}');
  }

  getData(): any {
    return { id: ${i}, data: 'module data' };
  }
}`,
        symbols: [
          { name: `Module${i}`, type: 'class', line: 2 },
          { name: 'process', type: 'method', line: 3 },
          { name: 'getData', type: 'method', line: 7 },
        ],
      }));

      // Index additional files
      for (const file of additionalFiles) {
        await searchEngine.indexFile(file.file, file.content, file.symbols);
      }

      const response = await searchEngine.search({
        query: 'authenticate user',
        mode: 'hybrid',
        max_results: 10,
      });

      // Should still find relevant authentication results
      expect(response.hits.length).toBeGreaterThan(0);
      expect(response.processing_time_ms).toBeLessThan(30); // Slight increase acceptable
      
      // Top results should still be authentication-related
      const authHits = response.hits.slice(0, 5).filter(hit =>
        hit.file.includes('auth') || hit.snippet.toLowerCase().includes('auth')
      );
      
      expect(authHits.length).toBeGreaterThan(0);
    });
  });

  describe('Query Classification and Routing', () => {
    it('should correctly classify and route natural language queries', async () => {
      const naturalQueries = [
        'how to authenticate users',
        'show me user registration logic',
        'find password validation functions',
      ];

      for (const query of naturalQueries) {
        const response = await searchEngine.search({
          query,
          mode: 'hybrid',
          max_results: 10,
        });

        expect(response.query_classification).toBeDefined();
        expect(response.query_classification.is_natural_language).toBe(true);
        expect(response.query_classification.confidence).toBeGreaterThan(0.5);
        
        // Should have applied semantic processing
        expect(response.stages.semantic).toBeDefined();
        expect(response.stages.semantic.applied).toBe(true);
      }
    });

    it('should correctly classify and route code-specific queries', async () => {
      const codeQueries = [
        'AuthenticationService.authenticate',
        'createUser()',
        'user.id',
      ];

      for (const query of codeQueries) {
        const response = await searchEngine.search({
          query,
          mode: 'hybrid',
          max_results: 10,
        });

        expect(response.query_classification).toBeDefined();
        expect(response.query_classification.is_natural_language).toBe(false);
        
        // Should emphasize lexical and symbol search
        expect(response.stages.lexical.candidates).toBeGreaterThan(0);
        expect(response.stages.symbol.candidates).toBeGreaterThan(0);
      }
    });
  });

  describe('Result Quality and Relevance', () => {
    it('should return highly relevant results for specific method searches', async () => {
      const response = await searchEngine.search({
        query: 'authenticate method implementation',
        mode: 'hybrid',
        max_results: 5,
      });

      expect(response.hits.length).toBeGreaterThan(0);
      
      // Top result should be the authenticate method
      const topHit = response.hits[0];
      expect(topHit.symbol_name).toBe('authenticate');
      expect(topHit.score).toBeGreaterThan(0.8);
      expect(topHit.file).toBe('src/auth/authentication.ts');
    });

    it('should provide diverse results for broad queries', async () => {
      const response = await searchEngine.search({
        query: 'user management',
        mode: 'hybrid',
        max_results: 10,
      });

      expect(response.hits.length).toBeGreaterThan(0);
      
      // Should include results from multiple files
      const fileCount = new Set(response.hits.map(hit => hit.file)).size;
      expect(fileCount).toBeGreaterThan(2);
      
      // Should include different types of symbols
      const symbolTypes = new Set(
        response.hits.map(hit => hit.symbol_type).filter(Boolean)
      );
      expect(symbolTypes.size).toBeGreaterThan(1);
    });

    it('should handle edge cases gracefully', async () => {
      const edgeCases = [
        '', // Empty query
        'nonexistentfunction', // No matches
        'a'.repeat(1000), // Very long query
        '🚀💻🔍', // Emoji query
        'SELECT * FROM users', // SQL query
      ];

      for (const query of edgeCases) {
        const response = await searchEngine.search({
          query,
          mode: 'hybrid',
          max_results: 10,
        });

        expect(response).toBeDefined();
        expect(response.processing_time_ms).toBeLessThan(100);
        
        if (response.hits.length === 0) {
          expect(response.total_hits).toBe(0);
        } else {
          expect(response.hits.every(hit => hit.score >= 0)).toBe(true);
        }
      }
    });
  });

  describe('Filter and Context Support', () => {
    it('should respect file type filters', async () => {
      const response = await searchEngine.search({
        query: 'user authentication',
        mode: 'hybrid',
        max_results: 10,
        filters: {
          file_extensions: ['.ts'],
        },
      });

      expect(response.hits.length).toBeGreaterThan(0);
      
      // All results should be TypeScript files
      expect(response.hits.every(hit => hit.file.endsWith('.ts'))).toBe(true);
    });

    it('should respect path filters', async () => {
      const response = await searchEngine.search({
        query: 'authentication',
        mode: 'hybrid',
        max_results: 10,
        filters: {
          paths: ['src/auth/'],
        },
      });

      expect(response.hits.length).toBeGreaterThan(0);
      
      // All results should be from auth directory
      expect(response.hits.every(hit => hit.file.includes('src/auth/'))).toBe(true);
    });

    it('should handle symbol type filters', async () => {
      const response = await searchEngine.search({
        query: 'user',
        mode: 'hybrid',
        max_results: 10,
        filters: {
          symbol_types: ['class', 'interface'],
        },
      });

      expect(response.hits.length).toBeGreaterThan(0);
      
      // Results should only include classes and interfaces
      const validTypes = response.hits.filter(hit =>
        !hit.symbol_type || ['class', 'interface'].includes(hit.symbol_type)
      );
      
      expect(validTypes.length).toBe(response.hits.length);
    });
  });

  describe('Detailed Stage Analysis', () => {
    it('should provide comprehensive stage diagnostics', async () => {
      const response = await searchEngine.search({
        query: 'authentication service user login',
        mode: 'hybrid',
        max_results: 10,
        include_diagnostics: true,
      });

      expect(response.stages).toBeDefined();
      
      // Lexical stage
      expect(response.stages.lexical).toBeDefined();
      expect(response.stages.lexical.candidates).toBeGreaterThan(0);
      expect(response.stages.lexical.processing_time_ms).toBeLessThan(50);
      
      // Symbol stage
      expect(response.stages.symbol).toBeDefined();
      expect(response.stages.symbol.candidates).toBeGreaterThan(0);
      expect(response.stages.symbol.processing_time_ms).toBeLessThan(50);
      
      // Semantic stage
      expect(response.stages.semantic).toBeDefined();
      expect(response.stages.semantic.applied).toBe(true);
      expect(response.stages.semantic.processing_time_ms).toBeLessThan(100);
    });

    it('should show stage fusion and candidate filtering', async () => {
      const response = await searchEngine.search({
        query: 'user authentication workflow',
        mode: 'hybrid',
        max_results: 10,
        include_diagnostics: true,
      });

      expect(response.fusion_details).toBeDefined();
      
      // Should show how candidates were merged and filtered
      expect(response.fusion_details.total_candidates_before_fusion).toBeDefined();
      expect(response.fusion_details.candidates_after_deduplication).toBeDefined();
      expect(response.fusion_details.final_candidates_after_filtering).toBeDefined();
      
      // Should show stage contributions
      expect(response.fusion_details.stage_contributions).toBeDefined();
      expect(response.fusion_details.stage_contributions.lexical).toBeGreaterThanOrEqual(0);
      expect(response.fusion_details.stage_contributions.symbol).toBeGreaterThanOrEqual(0);
      expect(response.fusion_details.stage_contributions.semantic).toBeGreaterThanOrEqual(0);
    });
  });
});