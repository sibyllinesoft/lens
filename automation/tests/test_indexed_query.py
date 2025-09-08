#!/usr/bin/env python3
"""
Test script to verify that queries matching the indexed content return results
and trigger semantic reranking correctly.
"""

import subprocess
import json
import tempfile
import sys

def test_relevant_query():
    """Test with a query that should match the Rust code in the index."""
    
    print("🧪 Testing query that matches indexed Rust content...")
    
    # Create a test query that should match the Rust search engine code
    test_query = {
        "instance_id": "test-rust-001",
        "query": "struct SearchEngine impl search function",
        "golden_patch": "// Expected result from Rust search engine implementation"
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([test_query], f)
        temp_file = f.name
    
    try:
        # Run a single test with this query
        cmd = [
            './target/release/todo_validation_runner',
            '--single-query', temp_file
        ]
        
        print(f"🔍 Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd='/media/nathan/Seagate Hub/Projects/lens',
            capture_output=True,
            text=True,
            timeout=30,
            env={'RUST_LOG': 'info', 'NODE_ENV': 'benchmark'}
        )
        
        print(f"✅ Return code: {result.returncode}")
        if "Semantic reranking applied" in result.stdout:
            # Look for the number after "applied:"
            lines = result.stdout.split('\n')
            for line in lines:
                if "Semantic reranking applied" in line:
                    print(f"📊 {line.strip()}")
        
        if "results processed" in result.stdout and "0 results processed" not in result.stdout:
            print("🎯 SUCCESS: Semantic reranking processed non-zero results!")
            return True
        else:
            print("⚠️  Still processing 0 results with relevant query")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        import os
        try:
            os.unlink(temp_file)
        except:
            pass

def check_index_content():
    """Check what's actually in the index by looking at a few documents."""
    print("\n🔍 Checking index content...")
    
    # Simple search for common Rust keywords
    test_queries = [
        "struct",
        "impl", 
        "fn main",
        "SearchEngine",
        "pub fn search"
    ]
    
    for query in test_queries:
        try:
            # Quick search to see if we get results
            cmd = ['grep', '-r', '-l', query, 'src/']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/media/nathan/Seagate Hub/Projects/lens')
            files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            print(f"📄 Query '{query}' matches {len(files)} files in src/")
        except:
            pass

if __name__ == "__main__":
    check_index_content()
    success = test_relevant_query()
    
    if success:
        print("\n🎉 DIAGNOSIS CONFIRMED: Infrastructure works with relevant queries!")
        print("🔧 ROOT CAUSE: Benchmark queries don't match indexed content (corpus-query mismatch)")
        sys.exit(0)
    else:
        print("\n🤔 Need further investigation...")
        sys.exit(1)