/**
 * Index storyviz repository for lens benchmarking
 * Creates a comprehensive indexed corpus from ../storyviz
 */

import { promises as fs } from 'fs';
import path from 'path';

class StoryVizIndexer {
  constructor() {
    this.stats = [];
    this.storyVizPath = path.resolve('../storyviz');
    this.outputDir = path.resolve('./indexed-content');
    
    console.log(`📁 StoryViz path: ${this.storyVizPath}`);
    console.log(`📁 Output directory: ${this.outputDir}`);
  }

  async indexStoryVizRepository() {
    console.log('🚀 Starting storyviz repository indexing...');

    // Ensure output directory exists
    await fs.mkdir(this.outputDir, { recursive: true });

    // Check if storyviz exists
    try {
      await fs.access(this.storyVizPath);
      console.log('✅ StoryViz repository found');
    } catch (error) {
      throw new Error(`❌ StoryViz repository not found at ${this.storyVizPath}`);
    }

    // Index the repository
    await this.walkDirectory(this.storyVizPath);

    // Generate stats
    await this.generateStats();

    console.log(`✅ Successfully indexed ${this.stats.length} files from storyviz`);
  }

  async walkDirectory(dirPath) {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      
      // Skip hidden files, node_modules, and other irrelevant directories
      if (this.shouldSkipPath(entry.name, fullPath)) {
        continue;
      }

      if (entry.isDirectory()) {
        await this.walkDirectory(fullPath);
      } else if (entry.isFile() && this.shouldIndexFile(entry.name)) {
        await this.processFile(fullPath);
      }
    }
  }

  shouldSkipPath(name, fullPath) {
    const skipPatterns = [
      /^\./,              // Hidden files/directories
      /^__pycache__$/,    // Python cache
      /^node_modules$/,   // Node modules
      /^\.git$/,          // Git directory
      /^\.vscode$/,       // VS Code settings
      /^cache$/,          // Cache directories
      /^logs$/,           // Log directories
      /^\.pytest_cache$/, // Pytest cache
      /^\.mypy_cache$/,   // MyPy cache
      /^\.ruff_cache$/,   // Ruff cache
      /^migrations$/,     // Database migrations
      /^\.serena$/,       // Serena cache
      /^\.claude-temp$/,  // Claude temp files
    ];

    return skipPatterns.some(pattern => pattern.test(name));
  }

  shouldIndexFile(filename) {
    const extensions = [
      '.py',    // Python
      '.ts',    // TypeScript
      '.js',    // JavaScript
      '.tsx',   // TypeScript React
      '.jsx',   // JavaScript React
      '.java',  // Java
      '.cpp',   // C++
      '.c',     // C
      '.h',     // C/C++ headers
      '.rs',    // Rust
      '.go',    // Go
      '.sql',   // SQL
      '.yaml',  // YAML
      '.yml',   // YAML
      '.json',  // JSON (selective)
      '.md',    // Markdown (documentation)
    ];

    // Skip certain JSON files that are likely config/generated
    if (filename.endsWith('.json')) {
      const skipJsonPatterns = [
        /package-lock\.json$/,
        /yarn\.lock$/,
        /composer\.lock$/,
        /poetry\.lock$/,
        /uv\.lock$/,
      ];
      if (skipJsonPatterns.some(pattern => pattern.test(filename))) {
        return false;
      }
    }

    return extensions.some(ext => filename.endsWith(ext));
  }

  async processFile(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n');
      const extension = path.extname(filePath);
      
      // Create stats entry
      const stats = {
        path: filePath,
        extension,
        size: content.length,
        lines: lines.length,
        language: this.getLanguageFromExtension(extension)
      };
      
      this.stats.push(stats);

      // Copy file to indexed-content directory for corpus validation
      const relativePath = path.relative(this.storyVizPath, filePath);
      const flattenedName = this.flattenPath(relativePath);
      const outputPath = path.join(this.outputDir, flattenedName);
      
      // Ensure directory exists
      const outputDir = path.dirname(outputPath);
      await fs.mkdir(outputDir, { recursive: true });
      
      // Copy the file
      await fs.copyFile(filePath, outputPath);

      if (this.stats.length % 50 === 0) {
        console.log(`📊 Processed ${this.stats.length} files...`);
      }
    } catch (error) {
      console.warn(`⚠️ Failed to process ${filePath}:`, error.message);
    }
  }

  flattenPath(relativePath) {
    // Convert directory separators to underscores to avoid nesting
    return relativePath.replace(/[\/\\]/g, '_');
  }

  getLanguageFromExtension(extension) {
    const languageMap = {
      '.py': 'python',
      '.ts': 'typescript',
      '.js': 'javascript', 
      '.tsx': 'typescript',
      '.jsx': 'javascript',
      '.java': 'java',
      '.cpp': 'cpp',
      '.c': 'c',
      '.h': 'c',
      '.rs': 'rust',
      '.go': 'go',
      '.sql': 'sql',
      '.yaml': 'yaml',
      '.yml': 'yaml', 
      '.json': 'json',
      '.md': 'markdown'
    };

    return languageMap[extension] || 'unknown';
  }

  async generateStats() {
    const languageStats = new Map();
    
    for (const stat of this.stats) {
      const current = languageStats.get(stat.language) || { count: 0, lines: 0, size: 0 };
      current.count++;
      current.lines += stat.lines;
      current.size += stat.size;
      languageStats.set(stat.language, current);
    }

    const summary = {
      total_files: this.stats.length,
      total_lines: this.stats.reduce((sum, s) => sum + s.lines, 0),
      total_size: this.stats.reduce((sum, s) => sum + s.size, 0),
      languages: Object.fromEntries(languageStats),
      timestamp: new Date().toISOString(),
      storyviz_path: this.storyVizPath,
      indexed_content_path: this.outputDir
    };

    // Save summary
    const summaryPath = path.join(this.outputDir, 'indexing-summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2));

    // Save detailed stats
    const statsPath = path.join(this.outputDir, 'file-stats.json'); 
    await fs.writeFile(statsPath, JSON.stringify(this.stats, null, 2));

    console.log(`📊 Indexing Summary:`);
    console.log(`  Total files: ${summary.total_files}`);
    console.log(`  Total lines: ${summary.total_lines.toLocaleString()}`);
    console.log(`  Total size: ${(summary.total_size / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Languages:`);
    
    for (const [lang, stats] of languageStats) {
      console.log(`    ${lang}: ${stats.count} files, ${stats.lines.toLocaleString()} lines`);
    }
  }
}

// Main execution
async function main() {
  const indexer = new StoryVizIndexer();
  
  try {
    await indexer.indexStoryVizRepository();
    
    console.log('\n✅ StoryViz indexing completed successfully!');
    console.log('📁 Indexed content available in ./indexed-content/');
    console.log('📊 Check indexing-summary.json for detailed statistics');
    
  } catch (error) {
    console.error('❌ Indexing failed:', error.message);
    process.exit(1);
  }
}

// Run if this is the main module
main();