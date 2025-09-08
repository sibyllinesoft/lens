#!/usr/bin/env node

/**
 * Index SWE-Bench Corpus for Search
 * 
 * This script copies the extracted SWE-bench repository files from benchmark-corpus/
 * to indexed-content/ so the search engine can process them.
 */

import fs from 'fs/promises';
import path from 'path';

const BENCHMARK_CORPUS_DIR = './benchmark-corpus';
const INDEXED_CONTENT_DIR = './indexed-content';

async function main() {
    console.log('🚀 Indexing SWE-Bench corpus for search...');
    
    try {
        // Ensure indexed-content exists
        await fs.mkdir(INDEXED_CONTENT_DIR, { recursive: true });
        
        // Clear existing content
        console.log('🧹 Clearing existing indexed content...');
        try {
            const existingFiles = await fs.readdir(INDEXED_CONTENT_DIR);
            for (const file of existingFiles) {
                const filePath = path.join(INDEXED_CONTENT_DIR, file);
                const stat = await fs.lstat(filePath);
                if (stat.isFile()) {
                    await fs.unlink(filePath);
                } else if (stat.isDirectory()) {
                    await fs.rmdir(filePath, { recursive: true });
                }
            }
        } catch (clearError) {
            console.log('Note: No existing content to clear');
        }
        
        // Copy all Python files from benchmark-corpus to indexed-content
        console.log('📋 Copying SWE-bench corpus files...');
        const corpusFiles = await fs.readdir(BENCHMARK_CORPUS_DIR);
        
        let copiedCount = 0;
        for (const file of corpusFiles) {
            if (file.endsWith('.py')) {
                const sourcePath = path.join(BENCHMARK_CORPUS_DIR, file);
                const targetPath = path.join(INDEXED_CONTENT_DIR, file);
                
                await fs.copyFile(sourcePath, targetPath);
                copiedCount++;
                
                if (copiedCount % 1000 === 0) {
                    console.log(`   📁 ${copiedCount} files copied...`);
                }
            }
        }
        
        // Create metadata file
        const metadata = {
            index_type: 'swe_bench_corpus',
            created_at: new Date().toISOString(),
            total_files: copiedCount,
            source_directory: BENCHMARK_CORPUS_DIR
        };
        
        await fs.writeFile(
            path.join(INDEXED_CONTENT_DIR, 'meta.json'),
            JSON.stringify(metadata, null, 2)
        );
        
        console.log('📊 INDEXING COMPLETE');
        console.log('=====================');
        console.log(`Files indexed: ${copiedCount}`);
        console.log(`Target directory: ${INDEXED_CONTENT_DIR}`);
        console.log(`Metadata: ${path.join(INDEXED_CONTENT_DIR, 'meta.json')}`);
        console.log('');
        console.log('✅ SWE-Bench corpus is now ready for search!');
        
    } catch (error) {
        console.error('❌ Error indexing corpus:', error);
        process.exit(1);
    }
}

main();