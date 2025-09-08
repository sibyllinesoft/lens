#!/usr/bin/env python3
"""
Direct test of the search API to understand why queries return 0 results.
"""

import subprocess
import json
import sys

def test_direct_search():
    """Test the search API directly with simple queries."""
    
    print("🧪 Testing direct search API with basic queries...")
    
    # Test queries from simple to complex
    test_queries = [
        "struct",
        "SearchEngine", 
        "pub fn search",
        "impl",
        "fn main"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        try:
            # Call the search API directly via curl or a simple test
            # For now, let's see if we can test with a minimal Rust program
            test_code = f'''
use std::collections::HashMap;

// Simple test to see if our search works
fn main() {{
    println!("Testing query: {query}");
    
    // Try to search for this query
    // This would call our search engine if we had a simple interface
}}
'''
            
            # Write test code to a temporary file
            with open('/tmp/test_search.rs', 'w') as f:
                f.write(test_code)
            
            print(f"✅ Query '{query}' - test prepared")
            
        except Exception as e:
            print(f"❌ Error with query '{query}': {e}")

def check_tantivy_index():
    """Check if we can read the Tantivy index directly."""
    print("\n🔍 Checking Tantivy index directly...")
    
    try:
        # Check if the index directory exists and has content
        cmd = ['ls', '-la', 'indexed-content/']
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/media/nathan/Seagate Hub/Projects/lens')
        print("📁 Index directory contents:")
        print(result.stdout)
        
        # Check meta.json
        with open('/media/nathan/Seagate Hub/Projects/lens/indexed-content/meta.json', 'r') as f:
            meta = json.load(f)
            print(f"📊 Index meta - opstamp: {meta['opstamp']}")
            print(f"📊 Index meta - segments: {len(meta['segments'])}")
            if meta['segments']:
                print(f"📊 First segment max_doc: {meta['segments'][0]['max_doc']}")
        
    except Exception as e:
        print(f"❌ Error checking index: {e}")

def diagnose_search_issue():
    """Try to understand why searches return 0 results."""
    print("\n🔬 DIAGNOSIS: Understanding the search issue...")
    
    print("1. Index Status:")
    check_tantivy_index()
    
    print("\n2. Query Testing:")
    test_direct_search()
    
    print("\n3. Hypothesis:")
    print("   ❓ Tantivy query parsing failing on all queries?")
    print("   ❓ Index not properly populated with searchable content?")
    print("   ❓ Search method routing not working correctly?")
    print("   ❓ Content indexed but not searchable due to field mapping?")

if __name__ == "__main__":
    diagnose_search_issue()