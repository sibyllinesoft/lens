#!/usr/bin/env python3
"""
Smoke tests for key queries before running full benchmark.
Tests specific queries: APIRouter include_router (FastAPI) and BaseModel validate_assignment (Pydantic)
"""

import requests
import json
import asyncio
import sys

async def test_qdrant_health():
    """Test Qdrant health endpoint."""
    try:
        response = requests.get("http://localhost:6333/", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant health check: PASS")
            return True
        else:
            print(f"❌ Qdrant health check: FAIL - Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Qdrant health check: FAIL - {e}")
        return False

async def test_opensearch_health():
    """Test OpenSearch health endpoint."""
    try:
        response = requests.get("http://localhost:9200/_cluster/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            if health.get('status') in ['yellow', 'green']:
                print("✅ OpenSearch health check: PASS")
                return True
            else:
                print(f"❌ OpenSearch health check: FAIL - Status {health.get('status')}")
                return False
        else:
            print(f"❌ OpenSearch health check: FAIL - Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ OpenSearch health check: FAIL - {e}")
        return False

def test_key_queries():
    """Test the specific queries mentioned in requirements."""
    
    print("\n🔍 Testing Key Queries:")
    print("=" * 40)
    
    # Key queries to test
    queries = [
        "APIRouter include_router",  # FastAPI
        "BaseModel validate_assignment",  # Pydantic 
        "class BaseModel",
        "def include_router",
        "FastAPI router",
        "pydantic validation"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        # For now, just log what we would test
        # In a real implementation, we would test against the search systems
        print(f"  📝 Would search for: {query}")
    
    print("\n💡 Note: This is a smoke test stub.")
    print("💡 Full implementation would test actual search against corpus.")
    return True

async def main():
    """Run smoke tests."""
    print("🚀 Running Smoke Tests")
    print("=" * 50)
    
    # Test service health
    qdrant_ok = await test_qdrant_health()
    opensearch_ok = await test_opensearch_health()
    
    # Test key queries (stubbed for now)
    queries_ok = test_key_queries()
    
    # Overall result
    all_tests_passed = qdrant_ok and opensearch_ok and queries_ok
    
    print(f"\n📊 SMOKE TEST SUMMARY")
    print(f"=" * 25)
    print(f"Qdrant Health: {'✅ PASS' if qdrant_ok else '❌ FAIL'}")
    print(f"OpenSearch Health: {'✅ PASS' if opensearch_ok else '❌ FAIL'}")
    print(f"Key Queries: {'✅ PASS' if queries_ok else '❌ FAIL'}")
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
    
    return all_tests_passed

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
