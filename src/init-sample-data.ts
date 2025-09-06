#!/usr/bin/env node

/**
 * Initialize sample data for the Lens search engine
 * This script creates sample content and indexes it for demonstration purposes
 */

import { promises as fs } from 'fs';
import * as path from 'path';
import { codeIndexer } from './indexer.js';

async function createSampleFiles() {
  const sampleDir = path.join(process.cwd(), 'sample-code');
  
  // Ensure sample directory exists
  await fs.mkdir(sampleDir, { recursive: true });
  
  // Sample TypeScript file
  const userServiceContent = `// User management service
export interface User {
  id: string;
  name: string;
  email: string;
}

export class UserService {
  private users: User[] = [];

  async createUser(userData: Omit<User, 'id'>): Promise<User> {
    const user: User = {
      id: Math.random().toString(36).substr(2, 9),
      ...userData
    };
    this.users.push(user);
    return user;
  }

  async findUserById(id: string): Promise<User | null> {
    return this.users.find(u => u.id === id) || null;
  }

  async searchUsers(query: string): Promise<User[]> {
    const q = query.toLowerCase();
    return this.users.filter(user => 
      user.name.toLowerCase().includes(q) ||
      user.email.toLowerCase().includes(q)
    );
  }
}`;

  // Sample JavaScript file
  const utilsContent = `// Utility functions
export function validateEmail(email) {
  const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
  return emailRegex.test(email);
}

export function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

export class Logger {
  constructor(name) {
    this.name = name;
  }

  info(message) {
    console.log('[' + this.name + '] INFO: ' + message);
  }

  error(message) {
    console.error('[' + this.name + '] ERROR: ' + message);
  }
}`;

  // Sample Python file
  const analyticContent = `# Data analytics module
import json
from typing import List, Dict, Any

class DataAnalyzer:
    def __init__(self):
        self.data = []
    
    def add_data_point(self, point: Dict[str, Any]) -> None:
        self.data.append(point)
    
    def analyze_trends(self, field: str) -> Dict[str, float]:
        values = [point.get(field) for point in self.data if field in point]
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        if not numeric_values:
            return {}
        
        return {
            'avg': sum(numeric_values) / len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values)
        }
    
    def search_data(self, query: str) -> List[Dict[str, Any]]:
        results = []
        query_lower = query.lower()
        
        for point in self.data:
            for key, value in point.items():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(point)
                    break
        
        return results`;

  // Write sample files
  await fs.writeFile(path.join(sampleDir, 'user-service.ts'), userServiceContent);
  await fs.writeFile(path.join(sampleDir, 'utils.js'), utilsContent);
  await fs.writeFile(path.join(sampleDir, 'analytics.py'), analyticContent);
  
  console.log('✅ Created sample files in:', sampleDir);
  return sampleDir;
}

async function initializeSampleData() {
  console.log('🚀 Initializing sample data for Lens search engine...');
  
  try {
    // Use real indexed content instead of creating sample files
    const contentDir = path.join(process.cwd(), 'indexed-content');
    console.log('📚 Indexing repository content for integration testing...');
    
    // Check if indexed content exists, otherwise create sample files
    try {
      await fs.access(contentDir);
      console.log('✅ Found real indexed content, using for integration testing');
      await codeIndexer.indexDirectory(contentDir);
    } catch {
      console.log('ℹ️  No indexed content found, creating sample files...');
      const sampleDir = await createSampleFiles();
      await codeIndexer.indexDirectory(sampleDir);
    }
    
    // Also index the src/example.ts file
    const exampleFile = path.join(process.cwd(), 'src', 'example.ts');
    try {
      await codeIndexer.indexFile(exampleFile);
    } catch (error) {
      console.warn('Could not index src/example.ts:', error);
    }
    
    // Show index stats
    const stats = codeIndexer.getIndexStats();
    console.log('📊 Index Statistics:', stats);
    
    // Test search functionality with Python/integration-focused queries
    console.log('\\n🔍 Testing search functionality...');
    const testQueries = ['class', 'async def', 'FastAPI', 'middleware', 'auth', 'security', 'logging', 'database', 'api', 'function'];
    
    for (const query of testQueries) {
      const results = codeIndexer.search(query);
      console.log('Query "' + query + '": ' + results.length + ' results');
      
      if (results.length > 0) {
        const first = results[0];
        if (first) {
          console.log('  Sample: ' + first.file + ':' + first.line + ':' + first.col + ' - "' + first.text + '"');
        }
      }
    }
    
    console.log('\\n✅ Sample data initialization complete!');
    console.log('🌟 You can now run queries against the search engine and get actual results.');
    
  } catch (error) {
    console.error('❌ Failed to initialize sample data:', error);
    process.exit(1);
  }
}

// Run initialization if this script is executed directly
if (require.main === module) {
  initializeSampleData().catch(console.error);
}

export { initializeSampleData, createSampleFiles };