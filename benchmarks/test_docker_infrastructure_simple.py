#!/usr/bin/env python3
"""
Simple Docker Infrastructure Test

Quick test to verify Docker services are running and responding.
"""

import requests
import sys

# Service URLs from docker-compose.yml
SERVICES = {
    'zoekt': 'http://localhost:6070',
    'livegrep': 'http://localhost:9898', 
    'ripgrep': 'http://localhost:8080/health',
    'comby': 'http://localhost:8081/health',
    'opensearch': 'http://localhost:9200/_cluster/health',
    'qdrant': 'http://localhost:6333/health',
    'faiss': 'http://localhost:8084/health',
    'milvus': 'http://localhost:9091/healthz',
    'ctags': 'http://localhost:8083/health'
}

def test_service(name, url):
    """Test if a service is responding."""
    try:
        print(f"Testing {name}... ", end="")
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("✅ OK")
            return True
        else:
            print(f"❌ HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {e}")
        return False

def main():
    """Test all services."""
    print("🔍 Testing Docker Infrastructure")
    print("=" * 40)
    
    working = 0
    total = len(SERVICES)
    
    for name, url in SERVICES.items():
        if test_service(name, url):
            working += 1
    
    print(f"\n📊 Results: {working}/{total} services working")
    
    if working == total:
        print("🎉 All services are ready!")
        return 0
    elif working >= total // 2:
        print("⚠️  Most services are working")
        return 0
    else:
        print("❌ Too many services failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())