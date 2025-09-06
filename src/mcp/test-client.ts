#!/usr/bin/env node

/**
 * Test client for the Lens MCP Server
 * Demonstrates how to interact with the MCP server tools
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { spawn } from 'child_process';

class LensMCPTestClient {
  private client: Client;
  private serverProcess: any;

  constructor() {
    this.client = new Client(
      {
        name: 'lens-test-client',
        version: '1.0.0',
      },
      {
        capabilities: {},
      }
    );
  }

  async connect() {
    // Start the MCP server process
    this.serverProcess = spawn('tsx', ['src/mcp/server.ts'], {
      stdio: ['pipe', 'pipe', 'inherit'],
      cwd: process.cwd(),
    });

    // Connect client to server
    const transport = new StdioClientTransport({
      reader: this.serverProcess.stdout,
      writer: this.serverProcess.stdin,
    });

    await this.client.connect(transport);
    console.log('✅ Connected to Lens MCP Server');
  }

  async testListTools() {
    console.log('\n🔧 Testing tool listing...');
    
    const response = await this.client.request(
      { method: 'tools/list' },
      {}
    );

    console.log(`Found ${response.tools.length} tools:`);
    response.tools.forEach((tool: any) => {
      console.log(`  • ${tool.name}: ${tool.description}`);
    });

    return response.tools;
  }

  async testSearch() {
    console.log('\n🔍 Testing lens_search tool...');
    
    const response = await this.client.request(
      { method: 'tools/call' },
      {
        name: 'lens_search',
        arguments: {
          repo_sha: 'test-repo-123',
          query: 'function authentication',
          mode: 'hybrid',
          k: 5,
          token_budget: 2000,
        },
      }
    );

    const result = JSON.parse(response.content[0].text);
    console.log(`Search found ${result.total_found} results, returned ${result.returned}`);
    console.log(`Token usage: ${result.token_usage.used}/${result.token_usage.budget}`);
    
    if (result.results.length > 0) {
      console.log('Sample result:');
      const sample = result.results[0];
      console.log(`  • File: ${sample.file}:${sample.line}`);
      console.log(`  • Ref: ${sample.ref}`);
      console.log(`  • Score: ${sample.score}`);
      console.log(`  • Snippet: ${sample.snippet.substring(0, 50)}...`);
    }

    return result;
  }

  async testContext() {
    console.log('\n📄 Testing lens_context tool...');
    
    const testRefs = [
      'lens://test-repo/src/auth.ts@abc123#L10:15|B200:250',
      'lens://test-repo/src/utils.ts@def456#L25:30|B500:550',
    ];

    const response = await this.client.request(
      { method: 'tools/call' },
      {
        name: 'lens_context',
        arguments: {
          refs: testRefs,
          token_budget: 1500,
        },
      }
    );

    const result = JSON.parse(response.content[0].text);
    console.log(`Resolved ${result.contexts.length}/${testRefs.length} references`);
    console.log(`Total tokens: ${result.total_tokens}`);
    console.log(`Budget exceeded: ${result.budget_exceeded}`);

    if (result.contexts.length > 0) {
      console.log('Sample context:');
      const sample = result.contexts[0];
      console.log(`  • Ref: ${sample.ref}`);
      console.log(`  • Tokens: ${sample.token_count}`);
      console.log(`  • Truncated: ${sample.truncated}`);
    }

    return result;
  }

  async testResolve() {
    console.log('\n🎯 Testing lens_resolve tool...');
    
    const response = await this.client.request(
      { method: 'tools/call' },
      {
        name: 'lens_resolve',
        arguments: {
          ref: 'lens://test-repo/src/example.ts@abc123#L42:45|B1024:1100|AST:function_declaration',
          context_lines: 3,
        },
      }
    );

    const result = JSON.parse(response.content[0].text);
    console.log(`Resolved reference for ${result.file_path}`);
    console.log(`Lines: ${result.line_start}-${result.line_end}`);
    console.log(`Language: ${result.metadata.language}`);
    console.log(`Context lines: ${result.surrounding_lines.before.length} before, ${result.surrounding_lines.after.length} after`);

    return result;
  }

  async testSymbols() {
    console.log('\n🏗️  Testing lens_symbols tool...');
    
    const response = await this.client.request(
      { method: 'tools/call' },
      {
        name: 'lens_symbols',
        arguments: {
          repo_sha: 'test-repo-123',
          filter: {
            kind: 'function',
            name_pattern: '.*test.*',
          },
          token_budget: 2000,
        },
      }
    );

    const result = JSON.parse(response.content[0].text);
    console.log(`Found ${result.total_found} symbols, returned ${result.returned}`);
    console.log(`Token usage: ${result.token_usage.used}/${result.token_usage.budget}`);

    if (result.symbols.length > 0) {
      console.log('Sample symbols:');
      result.symbols.slice(0, 3).forEach((symbol: any) => {
        console.log(`  • ${symbol.name} (${symbol.kind}) in ${symbol.file_path}:${symbol.line}`);
        console.log(`    Ref: ${symbol.ref}`);
      });
    }

    return result;
  }

  async runAllTests() {
    try {
      console.log('🚀 Starting Lens MCP Server Tests\n');

      await this.connect();

      const tools = await this.testListTools();
      const expectedTools = ['lens_search', 'lens_context', 'lens_resolve', 'lens_symbols'];
      
      const foundTools = tools.map((t: any) => t.name);
      const missingTools = expectedTools.filter(name => !foundTools.includes(name));
      
      if (missingTools.length > 0) {
        throw new Error(`Missing tools: ${missingTools.join(', ')}`);
      }

      await this.testSearch();
      await this.testContext();
      await this.testResolve();
      await this.testSymbols();

      console.log('\n🎉 All MCP tests passed successfully!');
      console.log('\n📋 Available MCP Tools:');
      console.log('• lens_search - Semantic code search with fuzzy matching');
      console.log('• lens_context - Batch resolve lens:// references');
      console.log('• lens_resolve - Resolve single reference with context');
      console.log('• lens_symbols - List and filter repository symbols');
      console.log('\n💡 Usage: Connect any MCP-compatible client to use Lens for code search');

      return true;

    } catch (error) {
      console.error('\n❌ MCP test failed:', error);
      return false;
    } finally {
      await this.disconnect();
    }
  }

  async disconnect() {
    try {
      await this.client.close();
    } catch (error) {
      console.error('Error closing client:', error);
    }

    if (this.serverProcess) {
      this.serverProcess.kill();
    }
  }
}

// Run tests if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const testClient = new LensMCPTestClient();
  
  testClient.runAllTests().then((success) => {
    process.exit(success ? 0 : 1);
  }).catch((error) => {
    console.error('Test runner failed:', error);
    process.exit(1);
  });
}

export { LensMCPTestClient };