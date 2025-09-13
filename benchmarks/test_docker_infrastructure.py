#!/usr/bin/env python3
"""
Test Docker Infrastructure for Real Competitor Systems

Verifies that all Docker services are running and can respond to health checks.
Tests basic connectivity and API endpoints for each competitor system.
"""

import asyncio
import requests
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Import the service URLs
from real_competitor_systems import SERVICE_URLS


class DockerInfrastructureTester:
    """Test Docker infrastructure and service connectivity."""
    
    def __init__(self):
        self.service_urls = SERVICE_URLS
        self.test_results = {}
        
    async def test_all_services(self) -> Dict[str, bool]:
        """Test all Docker services for availability and basic functionality."""
        print("🔍 Testing Docker Infrastructure for Competitor Systems")
        print("=" * 60)
        
        # Test each service
        for service_name, service_url in self.service_urls.items():
            print(f"\n📊 Testing {service_name.upper()} ({service_url})")
            success = await self.test_service(service_name, service_url)
            self.test_results[service_name] = success
            
            if success:
                print(f"✅ {service_name} is working")
            else:
                print(f"❌ {service_name} failed")
                
        # Print summary
        self.print_summary()
        return self.test_results
        
    async def test_service(self, service_name: str, service_url: str) -> bool:
        """Test individual service connectivity and basic API."""
        try:
            if service_name == 'zoekt':
                return await self.test_zoekt(service_url)
            elif service_name == 'livegrep':
                return await self.test_livegrep(service_url)
            elif service_name == 'ripgrep':
                return await self.test_ripgrep(service_url)
            elif service_name == 'comby':
                return await self.test_comby(service_url)
            elif service_name == 'ast_grep':
                return await self.test_ast_grep(service_url)
            elif service_name == 'opensearch':
                return await self.test_opensearch(service_url)
            elif service_name == 'qdrant':
                return await self.test_qdrant(service_url)
            elif service_name == 'vespa':
                return await self.test_vespa(service_url)
            elif service_name == 'faiss':
                return await self.test_faiss(service_url)
            elif service_name == 'milvus':
                return await self.test_milvus(service_url)
            elif service_name == 'ctags':
                return await self.test_ctags(service_url)
            else:
                print(f"⚠️  No test implemented for {service_name}")
                return False
                
        except Exception as e:
            print(f"❌ Exception testing {service_name}: {e}")
            return False
            
    async def test_zoekt(self, service_url: str) -> bool:
        """Test Zoekt service."""
        try:
            # Test main page
            response = requests.get(f"{service_url}/", timeout=10)
            if response.status_code != 200:
                print(f"   Health check failed: {response.status_code}")
                return False
                
            # Test search API
            response = requests.get(f"{service_url}/search", params={"q": "function"}, timeout=10)
            print(f"   Search test: {response.status_code}")
            return response.status_code in [200, 404]  # 404 is OK if no index yet
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
            
    async def test_livegrep(self, service_url: str) -> bool:
        """Test Livegrep service."""
        try:
            response = requests.get(f"{service_url}/", timeout=10)
            print(f"   Health check: {response.status_code}")
            return response.status_code == 200
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
            
    async def test_ripgrep(self, service_url: str) -> bool:
        """Test Ripgrep service."""
        try:
            # Test health endpoint
            response = requests.get(f"{service_url}/health", timeout=10)
            if response.status_code != 200:
                print(f"   Health check failed: {response.status_code}")
                return False
                
            # Test search API
            response = requests.post(\n                f\"{service_url}/search\",\n                json={\"query\": \"function\"},\n                timeout=10\n            )\n            print(f\"   Search test: {response.status_code}\")\n            return response.status_code == 200\n            \n        except Exception as e:\n            print(f\"   Error: {e}\")\n            return False\n            \n    async def test_comby(self, service_url: str) -> bool:\n        \"\"\"Test Comby service.\"\"\"\n        try:\n            response = requests.get(f\"{service_url}/health\", timeout=10)\n            if response.status_code != 200:\n                print(f\"   Health check failed: {response.status_code}\")\n                return False\n                \n            # Test search API\n            response = requests.post(\n                f\"{service_url}/search\",\n                json={\"pattern\": \"def :[name](...)\", \"language\": \"python\"},\n                timeout=10\n            )\n            print(f\"   Search test: {response.status_code}\")\n            return response.status_code == 200\n            \n        except Exception as e:\n            print(f\"   Error: {e}\")\n            return False\n            \n    async def test_ast_grep(self, service_url: str) -> bool:\n        \"\"\"Test AST-grep service.\"\"\"\n        try:\n            response = requests.get(f\"{service_url}/health\", timeout=10)\n            print(f\"   Health check: {response.status_code}\")\n            return response.status_code == 200\n            \n        except Exception as e:\n            print(f\"   Error: {e}\")\n            return False\n            \n    async def test_opensearch(self, service_url: str) -> bool:\n        \"\"\"Test OpenSearch service.\"\"\"\n        try:\n            response = requests.get(f\"{service_url}/_cluster/health\", timeout=10)\n            if response.status_code == 200:\n                health = response.json()\n                status = health.get('status', 'unknown')\n                print(f\"   Cluster status: {status}\")\n                return status in ['green', 'yellow']\n            else:\n                print(f\"   Health check failed: {response.status_code}\")\n                return False\n                \n        except Exception as e:\n            print(f\"   Error: {e}\")\n            return False\n            \n    async def test_qdrant(self, service_url: str) -> bool:\n        \"\"\"Test Qdrant service.\"\"\"\n        try:\n            response = requests.get(f\"{service_url}/health\", timeout=10)\n            print(f\"   Health check: {response.status_code}\")\n            return response.status_code == 200\n            \n        except Exception as e:\n            print(f\"   Error: {e}\")\n            return False\n            \n    async def test_vespa(self, service_url: str) -> bool:\n        \"\"\"Test Vespa service.\"\"\"\n        try:\n            response = requests.get(f\"{service_url}/ApplicationStatus\", timeout=10)\n            print(f\"   Health check: {response.status_code}\")\n            return response.status_code == 200\n            \n        except Exception as e:\n            print(f\"   Error: {e}\")\n            return False\n            \n    async def test_faiss(self, service_url: str) -> bool:\n        \"\"\"Test FAISS service.\"\"\"\n        try:\n            response = requests.get(f\"{service_url}/health\", timeout=10)\n            if response.status_code != 200:\n                print(f\"   Health check failed: {response.status_code}\")\n                return False\n                \n            # Test search API\n            response = requests.post(\n                f\"{service_url}/search\",\n                json={\"query\": \"function definition\", \"k\": 5},\n                timeout=10\n            )\n            print(f\"   Search test: {response.status_code}\")\n            return response.status_code == 200\n            \n        except Exception as e:\n            print(f\"   Error: {e}\")\n            return False\n            \n    async def test_milvus(self, service_url: str) -> bool:\n        \"\"\"Test Milvus service.\"\"\"\n        try:\n            # Use the health endpoint on port 9091\n            health_url = service_url.replace(':19530', ':9091')\n            response = requests.get(f\"{health_url}/healthz\", timeout=10)\n            print(f\"   Health check: {response.status_code}\")\n            return response.status_code == 200\n            \n        except Exception as e:\n            print(f\"   Error: {e}\")\n            return False\n            \n    async def test_ctags(self, service_url: str) -> bool:\n        \"\"\"Test ctags service.\"\"\"\n        try:\n            response = requests.get(f\"{service_url}/health\", timeout=10)\n            if response.status_code == 200:\n                data = response.json()\n                tags_loaded = data.get('tags_loaded', 0)\n                print(f\"   Health check: {response.status_code}, Tags loaded: {tags_loaded}\")\n                return True\n            else:\n                print(f\"   Health check failed: {response.status_code}\")\n                return False\n                \n        except Exception as e:\n            print(f\"   Error: {e}\")\n            return False\n            \n    def print_summary(self):\n        \"\"\"Print test summary.\"\"\"\n        print(\"\\n\" + \"=\" * 60)\n        print(\"📊 DOCKER INFRASTRUCTURE TEST SUMMARY\")\n        print(\"=\" * 60)\n        \n        working_services = [name for name, success in self.test_results.items() if success]\n        failed_services = [name for name, success in self.test_results.items() if not success]\n        \n        print(f\"\\n✅ Working Services ({len(working_services)}/{len(self.test_results)}):\")\n        for service in working_services:\n            print(f\"   • {service}\")\n            \n        if failed_services:\n            print(f\"\\n❌ Failed Services ({len(failed_services)}/{len(self.test_results)}):\")\n            for service in failed_services:\n                print(f\"   • {service}\")\n                \n        success_rate = len(working_services) / len(self.test_results) * 100\n        print(f\"\\n📈 Overall Success Rate: {success_rate:.1f}%\")\n        \n        if success_rate >= 80:\n            print(\"🎉 Infrastructure is ready for benchmarking!\")\n        elif success_rate >= 50:\n            print(\"⚠️  Infrastructure is partially ready. Some services need attention.\")\n        else:\n            print(\"❌ Infrastructure needs significant work before benchmarking.\")\n            \n        print(\"\\n💡 To start Docker services: docker-compose up -d\")\n        print(\"💡 To view logs: docker-compose logs -f [service_name]\")\n        print(\"💡 To stop services: docker-compose down\")\n\n\nasync def main():\n    \"\"\"Main test function.\"\"\"\n    tester = DockerInfrastructureTester()\n    results = await tester.test_all_services()\n    \n    # Exit with error code if too many services failed\n    failed_count = sum(1 for success in results.values() if not success)\n    if failed_count > len(results) // 2:\n        sys.exit(1)\n        \n\nif __name__ == \"__main__\":\n    asyncio.run(main())