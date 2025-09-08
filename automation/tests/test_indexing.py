#!/usr/bin/env python3
"""
Simple test to manually create documents in the index to verify the index works.
Since the Rust index is empty, let's add a few test documents directly.
"""

import json
import os

def main():
    print("🔧 Creating test documents to populate empty index...")
    
    # Create simple test files in the project that can be indexed
    test_files = [
        {
            "path": "src/test_file_1.rs",
            "content": "fn test_function() {\n    println!(\"Hello search!\");\n}\n\nstruct TestStruct {\n    field: i32,\n}"
        },
        {
            "path": "src/test_file_2.rs", 
            "content": "impl TestStruct {\n    pub fn new() -> Self {\n        Self { field: 42 }\n    }\n}\n\nfn another_function() {\n    let x = 10;\n}"
        },
        {
            "path": "src/test_file_3.ts",
            "content": "interface TestInterface {\n    name: string;\n    value: number;\n}\n\nclass TestClass implements TestInterface {\n    name = 'test';\n    value = 100;\n}"
        }
    ]
    
    # Create the files
    os.makedirs("src", exist_ok=True)
    for file_info in test_files:
        with open(file_info["path"], "w") as f:
            f.write(file_info["content"])
        print(f"📄 Created test file: {file_info['path']}")
    
    print("✅ Test files created. Now run the indexing process to populate the search index.")
    print("🔍 You can now test searches for terms like: 'function', 'test', 'struct', 'interface'")

if __name__ == "__main__":
    main()