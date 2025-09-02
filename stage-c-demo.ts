#!/usr/bin/env tsx

/**
 * Stage-C Semantic Reranking Demo
 * Demonstrates the enhanced three-stage search pipeline with intelligent query classification
 */

import { SemanticRerankEngine } from './src/indexer/semantic.js';
import { SegmentStorage } from './src/storage/segments.js';
import { classifyQuery, shouldApplySemanticReranking, explainSemanticDecision } from './src/core/query-classifier.js';
import type { Candidate, SearchContext } from './src/types/core.js';

async function demonstrateStageC() {
  console.log('🔍 Stage-C Semantic Reranking Demo\n');

  // Initialize semantic engine
  const segmentStorage = new SegmentStorage('./demo-segments');
  const semanticEngine = new SemanticRerankEngine(segmentStorage);
  await semanticEngine.initialize();

  // Index sample code documents
  console.log('📚 Indexing sample code documents...');
  const documents = [
    {
      id: 'auth_service',
      content: `
        class AuthService {
          async authenticate(username, password) {
            const user = await this.findUser(username);
            if (!user || !this.verifyPassword(password, user.hashedPassword)) {
              throw new Error('Invalid credentials');
            }
            return this.generateToken(user);
          }
          
          async findUser(username) {
            return await this.database.query('SELECT * FROM users WHERE username = ?', [username]);
          }
          
          verifyPassword(password, hashedPassword) {
            return bcrypt.compare(password, hashedPassword);
          }
        }
      `,
      path: '/src/services/auth.js'
    },
    {
      id: 'math_utils',
      content: `
        const MathUtils = {
          calculateSum(numbers) {
            return numbers.reduce((sum, num) => sum + num, 0);
          },
          
          calculateAverage(numbers) {
            return this.calculateSum(numbers) / numbers.length;
          },
          
          multiplyNumbers(a, b) {
            return a * b;
          },
          
          divideNumbers(a, b) {
            if (b === 0) throw new Error('Cannot divide by zero');
            return a / b;
          }
        };
      `,
      path: '/src/utils/math.js'
    },
    {
      id: 'user_controller',
      content: `
        class UserController {
          async registerUser(userData) {
            const hashedPassword = await bcrypt.hash(userData.password, 10);
            const user = await this.userService.createUser({
              ...userData,
              password: hashedPassword
            });
            return { success: true, user: { id: user.id, username: user.username } };
          }
          
          async loginUser(credentials) {
            try {
              const token = await this.authService.authenticate(credentials.username, credentials.password);
              return { success: true, token };
            } catch (error) {
              return { success: false, error: error.message };
            }
          }
        }
      `,
      path: '/src/controllers/user.js'
    },
    {
      id: 'file_utils',
      content: `
        const FileUtils = {
          async readFile(filePath) {
            try {
              const data = await fs.readFile(filePath, 'utf8');
              return { success: true, data };
            } catch (error) {
              return { success: false, error: error.message };
            }
          },
          
          async writeFile(filePath, content) {
            try {
              await fs.writeFile(filePath, content, 'utf8');
              return { success: true };
            } catch (error) {
              return { success: false, error: error.message };
            }
          }
        };
      `,
      path: '/src/utils/file.js'
    },
    {
      id: 'api_client',
      content: `
        class ApiClient {
          async get(url, headers = {}) {
            const response = await fetch(url, {
              method: 'GET',
              headers: { 'Content-Type': 'application/json', ...headers }
            });
            return await response.json();
          }
          
          async post(url, data, headers = {}) {
            const response = await fetch(url, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', ...headers },
              body: JSON.stringify(data)
            });
            return await response.json();
          }
        }
      `,
      path: '/src/services/api.js'
    }
  ];

  // Index documents
  for (const doc of documents) {
    await semanticEngine.indexDocument(doc.id, doc.content, doc.path);
  }

  const stats = semanticEngine.getStats();
  console.log(`✅ Indexed ${stats.vectors} documents with ${stats.avg_dim}-dimensional embeddings\n`);

  // Test queries - mix of natural language and keyword queries
  const testQueries = [
    // Natural language queries (should use semantic reranking)
    {
      query: 'find authentication logic for user login',
      type: 'Natural Language',
      expectedDocs: ['auth_service', 'user_controller']
    },
    {
      query: 'show me functions that calculate mathematical operations',
      type: 'Natural Language', 
      expectedDocs: ['math_utils']
    },
    {
      query: 'locate error handling for file operations',
      type: 'Natural Language',
      expectedDocs: ['file_utils']
    },
    {
      query: 'get all HTTP request methods',
      type: 'Natural Language',
      expectedDocs: ['api_client']
    },
    
    // Keyword queries (should skip semantic reranking)
    {
      query: 'def authenticate',
      type: 'Keyword',
      expectedDocs: ['auth_service']
    },
    {
      query: 'class UserController',
      type: 'Keyword',
      expectedDocs: ['user_controller']
    },
    {
      query: 'calculateSum()',
      type: 'Keyword',
      expectedDocs: ['math_utils']
    },
    {
      query: 'fetch(url',
      type: 'Keyword',
      expectedDocs: ['api_client']
    }
  ];

  console.log('🧪 Testing Query Classification and Semantic Reranking\n');

  for (const testCase of testQueries) {
    console.log(`Query: "${testCase.query}"`);
    console.log(`Type: ${testCase.type}`);
    
    // Test query classification
    const classification = classifyQuery(testCase.query);
    console.log(`Classification: ${classification.isNaturalLanguage ? 'Natural Language' : 'Keyword'} (confidence: ${(classification.confidence * 100).toFixed(1)}%)`);
    console.log(`Characteristics: ${classification.characteristics.join(', ')}`);
    
    // Create mock candidates for testing
    const mockCandidates: Candidate[] = documents.map((doc, index) => ({
      doc_id: `${doc.id}:1:1`,
      file_path: doc.path,
      line: 1,
      col: 1,
      score: 0.8 - (index * 0.1), // Decreasing relevance
      match_reasons: ['exact'],
      context: doc.content.substring(0, 100) + '...'
    }));

    // Test semantic reranking decision
    const shouldRerank = shouldApplySemanticReranking(testCase.query, mockCandidates.length, 'hybrid');
    const decision = explainSemanticDecision(testCase.query, mockCandidates.length, 'hybrid');
    console.log(`Semantic Reranking: ${shouldRerank ? 'Applied' : 'Skipped'}`);
    console.log(`Reason: ${decision}`);

    if (shouldRerank) {
      // Test actual reranking
      const searchContext: SearchContext = {
        trace_id: `demo-${Date.now()}`,
        query: testCase.query,
        mode: 'hybrid',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: []
      };

      const startTime = Date.now();
      const reranked = await semanticEngine.rerankCandidates(mockCandidates, searchContext, 5);
      const rerankTime = Date.now() - startTime;
      
      console.log(`Reranking Performance: ${rerankTime}ms (target: <10ms)`);
      console.log('Top Results:');
      reranked.slice(0, 3).forEach((candidate, index) => {
        const docId = candidate.doc_id.split(':')[0];
        const isExpected = testCase.expectedDocs?.includes(docId || '') ? '✅' : '  ';
        console.log(`  ${index + 1}. ${isExpected} ${candidate.file_path} (score: ${candidate.score.toFixed(3)})`);
      });
    } else {
      console.log('Skipped reranking - using original candidate order');
    }
    
    console.log('─'.repeat(80));
  }

  // Performance test
  console.log('\n⚡ Performance Testing');
  console.log('Testing query embedding caching...');
  
  const repeatQuery = 'find authentication logic for user login';
  const mockSearchContext: SearchContext = {
    trace_id: 'perf-test',
    query: repeatQuery,
    mode: 'hybrid',
    k: 10,
    fuzzy_distance: 0,
    started_at: new Date(),
    stages: []
  };
  
  const mockCandidates: Candidate[] = documents.map((doc, index) => ({
    doc_id: `${doc.id}:1:1`,
    file_path: doc.path,
    line: 1,
    col: 1,
    score: 0.8 - (index * 0.1),
    match_reasons: ['exact'],
    context: doc.content.substring(0, 100) + '...'
  }));

  // First run (cache miss)
  const firstRunStart = Date.now();
  await semanticEngine.rerankCandidates(mockCandidates, mockSearchContext, 5);
  const firstRunTime = Date.now() - firstRunStart;
  
  // Second run (cache hit)
  const secondRunStart = Date.now();
  await semanticEngine.rerankCandidates(mockCandidates, mockSearchContext, 5);
  const secondRunTime = Date.now() - secondRunStart;
  
  console.log(`First run (cache miss): ${firstRunTime}ms`);
  console.log(`Second run (cache hit): ${secondRunTime}ms`);
  console.log(`Cache speedup: ${((firstRunTime - secondRunTime) / firstRunTime * 100).toFixed(1)}%`);

  // Cleanup
  await semanticEngine.shutdown();
  await segmentStorage.shutdown();
  
  console.log('\n✅ Stage-C Demo completed successfully!');
  console.log('\nKey improvements implemented:');
  console.log('• ✅ Query classification (Natural Language vs Keyword)');
  console.log('• ✅ Intelligent semantic reranking gating (≥10 candidates, hybrid mode, NL queries)');
  console.log('• ✅ Query embedding caching (1000 query LRU cache)');
  console.log('• ✅ Performance optimization (<10ms additional latency target)');
  console.log('• ✅ Comprehensive error handling and fallbacks');
}

// Run demo
if (import.meta.url === `file://${process.argv[1]}`) {
  demonstrateStageC().catch(console.error);
}