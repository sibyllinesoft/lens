#!/usr/bin/env python3
"""
Debug script to understand query sanitization behavior
"""

def simulate_rust_sanitization(query):
    """Simulate the Rust sanitization function"""
    
    # Remove markdown syntax
    query = query.replace("```", " ")
    query = query.replace("**", " ")
    query = query.replace("__", " ")
    query = query.replace("<!--", " ")
    query = query.replace("-->", " ")
    query = query.replace("###", " ")
    
    # Remove special characters that cause Tantivy issues
    special_chars = ['(', ')', '[', ']', '{', '}', '"', "'", '+', '-', '!', '?', 
                     ':', ';', '#', '@', '$', '%', '^', '&', '*', '=', '|', 
                     '\\', '/', '<', '>', '.', ',', '~', '`']
    
    for char in special_chars:
        query = query.replace(char, " ")
    
    # Clean up whitespace
    return " ".join(query.split()).strip()

def test_queries():
    test_cases = [
        "struct SearchEngine",
        "fn main", 
        "impl search",
        "pub fn search",
        "Consider removing auto-transform of structured column into NdarrayMixin",
        "error handling",
        "async function",
        "rust implementation",
    ]
    
    print("🔍 Testing query sanitization behavior:")
    print("=" * 60)
    
    for original in test_cases:
        sanitized = simulate_rust_sanitization(original)
        print(f"Original:  '{original}'")
        print(f"Sanitized: '{sanitized}'")
        
        if len(sanitized.strip()) == 0:
            print("❌ COMPLETELY EMPTY - WILL RETURN NO RESULTS")
        elif len(sanitized.split()) < len(original.split()) // 2:
            print("⚠️  SIGNIFICANT LOSS - POOR SEARCH QUALITY")
        else:
            print("✅ PRESERVED CORE TERMS")
        print("-" * 40)

if __name__ == "__main__":
    test_queries()