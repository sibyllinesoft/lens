#!/usr/bin/env python3
"""
SWE-bench Corpus Extractor

This script extracts actual repository content for SWE-bench queries, creating a proper 
corpus that matches what the queries are actually looking for. The key insight is that
each benchmark needs its own corpus that aligns with the query domain.

SWE-bench queries are about real GitHub issues/PRs, so they need actual repository code.
"""

import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_unique_repositories(swe_bench_data):
    """Extract unique repositories from SWE-bench dataset"""
    repositories = set()
    repo_counts = Counter()
    
    for entry in swe_bench_data:
        repo_name = entry['repository']
        repositories.add(repo_name)
        repo_counts[repo_name] += 1
    
    logger.info(f"Found {len(repositories)} unique repositories across {len(swe_bench_data)} entries")
    logger.info(f"Top 10 repositories: {repo_counts.most_common(10)}")
    
    return repositories, repo_counts

def clone_repository(repo_name, base_commit, clone_dir):
    """Clone a repository at a specific commit"""
    repo_url = f"https://github.com/{repo_name}.git"
    repo_path = clone_dir / repo_name.replace('/', '_')
    
    try:
        # Clone the repository
        subprocess.run(['git', 'clone', repo_url, str(repo_path)], 
                      check=True, capture_output=True)
        
        # Checkout specific commit
        subprocess.run(['git', 'checkout', base_commit], 
                      cwd=repo_path, check=True, capture_output=True)
        
        logger.info(f"Successfully cloned {repo_name} at commit {base_commit[:8]}")
        return repo_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone {repo_name}: {e}")
        return None

def extract_source_files(repo_path, output_dir, repo_name):
    """Extract relevant source files from cloned repository"""
    if not repo_path or not repo_path.exists():
        return []
    
    # File extensions to include
    source_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.cc', '.cxx', '.c', 
        '.h', '.hpp', '.cs', '.rb', '.go', '.rs', '.php', '.scala', '.kt', '.swift',
        '.m', '.mm', '.sh', '.sql', '.html', '.css', '.scss', '.less', '.vue',
        '.md', '.rst', '.txt', '.yaml', '.yml', '.json', '.xml', '.toml', '.ini'
    }
    
    # Directories to skip
    skip_dirs = {
        '.git', '.svn', '.hg', '__pycache__', 'node_modules', '.tox', 'venv', 
        'env', '.env', 'build', 'dist', '.pytest_cache', '.coverage', 
        'htmlcov', '.idea', '.vscode', 'target', 'bin', 'obj'
    }
    
    extracted_files = []
    file_counter = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = Path(root) / file
            
            # Check if file has a relevant extension
            if file_path.suffix.lower() in source_extensions:
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Skip empty files or files that are too large
                    if 10 <= len(content) <= 100000:  # Between 10 chars and 100KB
                        # Create output filename
                        safe_repo_name = repo_name.replace('/', '_')
                        relative_path = file_path.relative_to(repo_path)
                        safe_relative_path = str(relative_path).replace('/', '_').replace('\\', '_')
                        output_filename = f"{safe_repo_name}_{safe_relative_path}"
                        
                        # Write to corpus
                        output_path = output_dir / output_filename
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        extracted_files.append(output_filename)
                        file_counter += 1
                        
                        if file_counter % 100 == 0:
                            logger.info(f"Extracted {file_counter} files from {repo_name}")
                
                except (IOError, UnicodeDecodeError):
                    continue  # Skip files that can't be read
    
    logger.info(f"Extracted {len(extracted_files)} files from {repo_name}")
    return extracted_files

def process_repository_sample(repo_name, entries, clone_dir, output_dir, max_commits=3):
    """Process a sample of commits from a repository"""
    # Sort entries by creation date and take latest commits
    entries_sorted = sorted(entries, key=lambda x: x['created_at'], reverse=True)
    sample_entries = entries_sorted[:max_commits]
    
    all_extracted_files = []
    
    for entry in sample_entries:
        base_commit = entry['base_commit']
        
        # Clone repository at this commit
        repo_path = clone_repository(repo_name, base_commit, clone_dir)
        
        if repo_path:
            # Extract source files
            extracted_files = extract_source_files(repo_path, output_dir, repo_name)
            all_extracted_files.extend(extracted_files)
            
            # Clean up cloned repo
            shutil.rmtree(repo_path)
        
        # Don't extract from too many commits of the same repo
        if len(all_extracted_files) > 500:  # Reasonable limit per repository
            break
    
    return all_extracted_files

def create_corpus_directories():
    """Create corpus directory structure"""
    base_dir = Path("benchmark-corpus")
    
    corpus_dirs = {
        'swe-bench': base_dir / "swe-bench",
        'codesearchnet': base_dir / "codesearchnet", 
        'coir': base_dir / "coir",
        'cosqa': base_dir / "cosqa"
    }
    
    # Create directories
    for name, path in corpus_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return corpus_dirs

def main():
    parser = argparse.ArgumentParser(description='Extract SWE-bench corpus from repository content')
    parser.add_argument('--dataset', default='datasets/swe-bench-verified.json', 
                       help='Path to SWE-bench dataset')
    parser.add_argument('--max-repos', type=int, default=20,
                       help='Maximum number of repositories to process')
    parser.add_argument('--max-commits-per-repo', type=int, default=3,
                       help='Maximum commits to sample per repository')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # Load SWE-bench dataset
    logger.info(f"Loading SWE-bench dataset from {args.dataset}")
    with open(args.dataset, 'r') as f:
        swe_bench_data = json.load(f)
    
    # Get unique repositories
    repositories, repo_counts = get_unique_repositories(swe_bench_data)
    
    if args.dry_run:
        logger.info("DRY RUN - Would process:")
        for repo, count in repo_counts.most_common(args.max_repos):
            logger.info(f"  {repo}: {count} entries")
        return
    
    # Create corpus directories
    corpus_dirs = create_corpus_directories()
    
    # Group entries by repository
    entries_by_repo = defaultdict(list)
    for entry in swe_bench_data:
        entries_by_repo[entry['repository']].append(entry)
    
    # Process top repositories
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_dir = Path(temp_dir)
        
        total_files = 0
        processed_repos = 0
        
        for repo_name, count in repo_counts.most_common(args.max_repos):
            if processed_repos >= args.max_repos:
                break
                
            logger.info(f"Processing repository {repo_name} ({count} entries)")
            
            entries = entries_by_repo[repo_name]
            extracted_files = process_repository_sample(
                repo_name, entries, clone_dir, corpus_dirs['swe-bench'], 
                args.max_commits_per_repo
            )
            
            total_files += len(extracted_files)
            processed_repos += 1
            
            logger.info(f"Progress: {processed_repos}/{args.max_repos} repos, {total_files} total files")
    
    # Write corpus summary
    summary = {
        'total_repositories_processed': processed_repos,
        'total_files_extracted': total_files,
        'corpus_type': 'swe-bench',
        'source_dataset': args.dataset,
        'extraction_params': {
            'max_repos': args.max_repos,
            'max_commits_per_repo': args.max_commits_per_repo
        }
    }
    
    summary_path = corpus_dirs['swe-bench'] / 'corpus_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Extraction complete! Summary written to {summary_path}")
    logger.info(f"Total: {processed_repos} repositories, {total_files} files")

if __name__ == '__main__':
    main()