#!/usr/bin/env node

/**
 * SWE-Bench Corpus Extractor
 * 
 * This script extracts real repository code from SWE-bench verified dataset
 * and populates the benchmark-corpus directory with actual source files.
 * 
 * Repositories to extract from SWE-bench:
 * - django/django (231 entries) - Web framework
 * - sympy/sympy (75 entries) - Symbolic math library  
 * - sphinx-doc/sphinx (44 entries) - Documentation generator
 * - matplotlib/matplotlib (34 entries) - Plotting library
 * - scikit-learn/scikit-learn (32 entries) - Machine learning
 * - astropy/astropy (22 entries) - Astronomy library
 * - pydata/xarray (22 entries) - Data analysis
 * - pytest-dev/pytest (19 entries) - Testing framework
 * - pylint-dev/pylint (10 entries) - Code linter
 * - psf/requests (8 entries) - HTTP library
 * - mwaskom/seaborn (2 entries) - Statistical visualization
 * - pallets/flask (1 entry) - Web framework
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const REPOS_DIR = './swe-bench-repos';
const CORPUS_DIR = './benchmark-corpus';
const MAX_FILES_PER_REPO = 2000; // Prevent overwhelming corpus
const MAX_FILE_SIZE = 500000; // 500KB max per file
const MIN_FILE_SIZE = 50; // Skip tiny files

// Repository configurations
const REPOSITORIES = [
    {
        name: 'django/django',
        url: 'https://github.com/django/django.git',
        branch: 'main',
        paths: ['django/'],
        priority: 1 // Highest priority - most entries
    },
    {
        name: 'sympy/sympy', 
        url: 'https://github.com/sympy/sympy.git',
        branch: 'master',
        paths: ['sympy/'],
        priority: 2
    },
    {
        name: 'sphinx-doc/sphinx',
        url: 'https://github.com/sphinx-doc/sphinx.git', 
        branch: 'master',
        paths: ['sphinx/'],
        priority: 3
    },
    {
        name: 'matplotlib/matplotlib',
        url: 'https://github.com/matplotlib/matplotlib.git',
        branch: 'main', 
        paths: ['lib/matplotlib/'],
        priority: 4
    },
    {
        name: 'scikit-learn/scikit-learn',
        url: 'https://github.com/scikit-learn/scikit-learn.git',
        branch: 'main',
        paths: ['sklearn/'],
        priority: 5
    },
    {
        name: 'astropy/astropy',
        url: 'https://github.com/astropy/astropy.git',
        branch: 'main',
        paths: ['astropy/'],
        priority: 6
    },
    {
        name: 'pydata/xarray',
        url: 'https://github.com/pydata/xarray.git',
        branch: 'main', 
        paths: ['xarray/'],
        priority: 7
    },
    {
        name: 'pytest-dev/pytest',
        url: 'https://github.com/pytest-dev/pytest.git',
        branch: 'main',
        paths: ['src/pytest/', 'src/_pytest/'],
        priority: 8
    },
    {
        name: 'pylint-dev/pylint',
        url: 'https://github.com/pylint-dev/pylint.git',
        branch: 'main',
        paths: ['pylint/'],
        priority: 9
    },
    {
        name: 'psf/requests',
        url: 'https://github.com/psf/requests.git',
        branch: 'main',
        paths: ['src/requests/', 'requests/'],
        priority: 10
    }
];

function shouldSkipFile(filePath, stats) {
    // Skip if too large or too small
    if (stats.size > MAX_FILE_SIZE || stats.size < MIN_FILE_SIZE) {
        return true;
    }
    
    // Skip build artifacts and cache files
    const skipPatterns = [
        /\/__pycache__\//,
        /\.pyc$/,
        /\.pyo$/,
        /\.egg-info\//,
        /\/build\//,
        /\/dist\//,
        /\/\.git\//,
        /\/node_modules\//,
        /\/venv\//,
        /\/\.pytest_cache\//,
        /\/\.tox\//,
        /\/docs\/build\//,
        /\/htmlcov\//,
        /\.coverage$/,
        /\.DS_Store$/
    ];
    
    return skipPatterns.some(pattern => pattern.test(filePath));
}

function isPythonFile(filePath) {
    return filePath.endsWith('.py');
}

function cloneOrUpdateRepository(repo) {
    const repoPath = path.join(REPOS_DIR, repo.name.replace('/', '_'));
    
    try {
        if (fs.existsSync(repoPath)) {
            console.log(`📥 Updating ${repo.name}...`);
            execSync(`cd "${repoPath}" && git fetch origin && git checkout ${repo.branch} && git reset --hard origin/${repo.branch}`, 
                    { stdio: 'pipe' });
        } else {
            console.log(`🔄 Cloning ${repo.name}...`);
            fs.mkdirSync(path.dirname(repoPath), { recursive: true });
            execSync(`git clone --depth 1 --branch ${repo.branch} "${repo.url}" "${repoPath}"`, 
                    { stdio: 'pipe' });
        }
        return repoPath;
    } catch (error) {
        console.error(`❌ Failed to clone/update ${repo.name}:`, error.message);
        return null;
    }
}

function extractFilesFromPath(repoPath, sourcePath, targetPrefix, maxFiles) {
    let fileCount = 0;
    const fullSourcePath = path.join(repoPath, sourcePath);
    
    if (!fs.existsSync(fullSourcePath)) {
        console.warn(`⚠️ Path not found: ${fullSourcePath}`);
        return 0;
    }
    
    function walkDirectory(dir, relativePath = '') {
        if (fileCount >= maxFiles) return;
        
        try {
            const entries = fs.readdirSync(dir);
            
            for (const entry of entries) {
                if (fileCount >= maxFiles) break;
                
                const fullPath = path.join(dir, entry);
                const relativeFilePath = path.join(relativePath, entry);
                
                try {
                    const stats = fs.statSync(fullPath);
                    
                    if (shouldSkipFile(fullPath, stats)) {
                        continue;
                    }
                    
                    if (stats.isDirectory()) {
                        walkDirectory(fullPath, relativeFilePath);
                    } else if (stats.isFile() && isPythonFile(fullPath)) {
                        // Create flattened filename
                        const flattenedName = `${targetPrefix}_${relativeFilePath.replace(/[\/\\]/g, '_')}`;
                        const targetPath = path.join(CORPUS_DIR, flattenedName);
                        
                        // Copy file to corpus
                        try {
                            const content = fs.readFileSync(fullPath, 'utf8');
                            fs.writeFileSync(targetPath, content);
                            fileCount++;
                            
                            if (fileCount % 100 === 0) {
                                console.log(`   📁 ${fileCount} files extracted...`);
                            }
                        } catch (copyError) {
                            console.warn(`⚠️ Failed to copy ${fullPath}:`, copyError.message);
                        }
                    }
                } catch (statError) {
                    console.warn(`⚠️ Failed to stat ${fullPath}:`, statError.message);
                }
            }
        } catch (readError) {
            console.warn(`⚠️ Failed to read directory ${dir}:`, readError.message);
        }
    }
    
    walkDirectory(fullSourcePath);
    return fileCount;
}

function extractRepositoryFiles(repo) {
    const repoPath = cloneOrUpdateRepository(repo);
    if (!repoPath) {
        return 0;
    }
    
    console.log(`🔍 Extracting files from ${repo.name}...`);
    
    let totalFiles = 0;
    const filesPerPath = Math.floor(MAX_FILES_PER_REPO / repo.paths.length);
    
    for (const sourcePath of repo.paths) {
        const targetPrefix = `${repo.name.replace('/', '_')}_${sourcePath.replace(/[\/\\]/g, '_')}`.replace(/_+$/, '');
        const fileCount = extractFilesFromPath(repoPath, sourcePath, targetPrefix, filesPerPath);
        totalFiles += fileCount;
        console.log(`   ✅ ${fileCount} files from ${sourcePath}`);
    }
    
    return totalFiles;
}

async function main() {
    console.log('🚀 SWE-Bench Corpus Extraction Starting...');
    console.log('');
    
    // Ensure directories exist
    fs.mkdirSync(REPOS_DIR, { recursive: true });
    fs.mkdirSync(CORPUS_DIR, { recursive: true });
    
    // Clear existing corpus
    console.log('🧹 Clearing existing corpus...');
    const existingFiles = fs.readdirSync(CORPUS_DIR);
    for (const file of existingFiles) {
        if (file.endsWith('.py')) {
            fs.unlinkSync(path.join(CORPUS_DIR, file));
        }
    }
    
    let totalFiles = 0;
    const results = {};
    
    // Sort repositories by priority
    const sortedRepos = REPOSITORIES.sort((a, b) => a.priority - b.priority);
    
    for (const repo of sortedRepos) {
        console.log(`\n📦 Processing ${repo.name}...`);
        const fileCount = extractRepositoryFiles(repo);
        results[repo.name] = fileCount;
        totalFiles += fileCount;
        console.log(`✅ ${repo.name}: ${fileCount} files extracted`);
    }
    
    // Generate summary
    console.log('\n📊 EXTRACTION SUMMARY');
    console.log('======================');
    for (const [repo, count] of Object.entries(results)) {
        console.log(`${repo.padEnd(30)} ${count.toString().padStart(6)} files`);
    }
    console.log('======================');
    console.log(`${'TOTAL'.padEnd(30)} ${totalFiles.toString().padStart(6)} files`);
    
    // Generate extraction metadata
    const metadata = {
        extraction_time: new Date().toISOString(),
        total_files: totalFiles,
        repositories: results,
        extraction_config: {
            max_files_per_repo: MAX_FILES_PER_REPO,
            max_file_size: MAX_FILE_SIZE,
            min_file_size: MIN_FILE_SIZE
        }
    };
    
    fs.writeFileSync(path.join(CORPUS_DIR, 'extraction_metadata.json'), 
                     JSON.stringify(metadata, null, 2));
    
    console.log('\n🎉 SWE-Bench corpus extraction completed!');
    console.log(`📁 ${totalFiles} Python files extracted to ${CORPUS_DIR}/`);
    console.log(`📋 Metadata saved to ${CORPUS_DIR}/extraction_metadata.json`);
    
    if (totalFiles === 0) {
        console.error('\n❌ No files were extracted! Check repository access and paths.');
        process.exit(1);
    }
}

main().catch(error => {
    console.error('💥 Fatal error:', error);
    process.exit(1);
});