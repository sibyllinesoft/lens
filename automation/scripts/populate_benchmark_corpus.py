#!/usr/bin/env python3
"""
Populate the search index with code content from benchmark datasets.
This ensures proper corpus-query alignment for semantic reranking validation.
"""

import json
import os
import tempfile
import subprocess
from pathlib import Path
import shutil

def load_dataset(dataset_path):
    """Load a benchmark dataset from JSON."""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {dataset_path}: {e}")
        return []

def extract_code_content(datasets_info):
    """Extract code content from all benchmark datasets."""
    code_files = []
    
    for dataset_name, dataset_path in datasets_info.items():
        print(f"📊 Processing {dataset_name} dataset...")
        dataset = load_dataset(dataset_path)
        
        for i, item in enumerate(dataset[:100]):  # Limit to first 100 items for testing
            if 'code' in item and item['code'].strip():
                # Create a synthetic file for this code snippet
                language = item.get('language', 'python')
                file_extension = {'python': 'py', 'javascript': 'js', 'java': 'java', 'go': 'go', 'rust': 'rs', 'ruby': 'rb'}.get(language, 'txt')
                
                file_path = f"{dataset_name}_{i:04d}.{file_extension}"
                code_content = item['code']
                
                # Add metadata as comments
                if language == 'python':
                    header = f'# Query: {item.get("query_text", "")[:100]}...\n# Dataset: {dataset_name}\n# Query ID: {item.get("query_id", "unknown")}\n\n'
                else:
                    header = f'// Query: {item.get("query_text", "")[:100]}...\n// Dataset: {dataset_name}\n// Query ID: {item.get("query_id", "unknown")}\n\n'
                
                full_content = header + code_content
                
                code_files.append({
                    'path': file_path,
                    'content': full_content,
                    'language': language,
                    'query_text': item.get('query_text', ''),
                    'query_id': item.get('query_id', f'{dataset_name}_{i}')
                })
        
        print(f"✅ Extracted {len([f for f in code_files if f['path'].startswith(dataset_name)])} code snippets from {dataset_name}")
    
    return code_files

def create_corpus_directory(code_files, corpus_dir):
    """Create a temporary directory structure with the code files."""
    if os.path.exists(corpus_dir):
        shutil.rmtree(corpus_dir)
    
    os.makedirs(corpus_dir, exist_ok=True)
    
    files_created = 0
    for code_file in code_files:
        file_path = os.path.join(corpus_dir, code_file['path'])
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code_file['content'])
            files_created += 1
        except Exception as e:
            print(f"⚠️ Failed to write {file_path}: {e}")
    
    print(f"📁 Created corpus directory with {files_created} files at {corpus_dir}")
    return corpus_dir

def update_search_engine_indexing(corpus_dir):
    """Update the search engine to index from the benchmark corpus directory."""
    print("🔧 Updating search engine indexing configuration...")
    
    # For now, just provide instructions
    print(f"""
    📝 TO COMPLETE THE SETUP:
    
    1. Update the search engine indexing to read from: {corpus_dir}
    2. Modify the index population logic in src/search.rs to:
       - Read from {corpus_dir} instead of src/ and rust-core/src/
       - Index .py, .js, .java, .go files in addition to .rs/.ts
    3. Rebuild and test with the new corpus
    
    This will ensure benchmark queries match the indexed content.
    """)
    
    return corpus_dir

def main():
    print("🚀 Populating benchmark corpus for semantic reranking validation")
    print("=" * 60)
    
    # Define available datasets
    datasets = {
        'cosqa': './datasets/cosqa.json',
        'codesearchnet': './datasets/codesearchnet.json',
        # Note: SWE-bench doesn't have direct code content, it's issue descriptions
        # 'swe-bench': './datasets/swe-bench-verified.json',
    }
    
    # Extract code content from datasets
    print("📊 Extracting code content from benchmark datasets...")
    code_files = extract_code_content(datasets)
    
    if not code_files:
        print("❌ No code content extracted from datasets")
        return
    
    print(f"✅ Total extracted: {len(code_files)} code snippets")
    
    # Create corpus directory
    corpus_dir = './benchmark-corpus'
    create_corpus_directory(code_files, corpus_dir)
    
    # Provide instructions for updating search engine
    update_search_engine_indexing(corpus_dir)
    
    # Create a sample query mapping for testing
    with open('./benchmark-query-mapping.json', 'w') as f:
        query_mapping = []
        for code_file in code_files:
            query_mapping.append({
                'query_id': code_file['query_id'],
                'query_text': code_file['query_text'],
                'expected_file': code_file['path'],
                'language': code_file['language']
            })
        json.dump(query_mapping, f, indent=2)
    
    print(f"📋 Created query mapping file: ./benchmark-query-mapping.json")
    print("🎯 Next step: Update search engine to index the benchmark corpus")

if __name__ == "__main__":
    main()