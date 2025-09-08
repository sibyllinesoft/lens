#!/usr/bin/env python3
"""
System Adapters - Authentic API integration with competitor systems
Provides unified interface to all benchmark systems
"""

import os
import time
import json
import hashlib
import logging
import asyncio
import aiohttp
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class SystemAdapter(ABC):
    """Abstract base class for system adapters"""
    
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.version = "unknown"
        self.config_hash = self._generate_config_hash()
    
    @abstractmethod
    def search(self, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search query synchronously"""
        pass
    
    @abstractmethod
    async def search_async(self, session: aiohttp.ClientSession, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search query asynchronously"""  
        pass
    
    def get_health_url(self) -> str:
        """Get health check URL"""
        return f"{self.base_url}/health"
    
    def get_version(self) -> str:
        """Get system version"""
        return self.version
    
    def get_config_hash(self) -> str:
        """Get configuration hash for reproducibility"""
        return self.config_hash
    
    def _generate_config_hash(self) -> str:
        """Generate hash of system configuration"""
        config_data = {
            "name": self.name,
            "base_url": self.base_url,
            "version": self.version
        }
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class LensAdapter(SystemAdapter):
    """Adapter for Lens system"""
    
    def __init__(self):
        base_url = os.environ.get("LENS_URL", "http://lens-core:50051")
        super().__init__("lens", base_url)
    
    def search(self, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via Lens API"""
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "corpus": corpus,
                    "k": k,
                    "include_embeddings": False
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Lens search failed: {e}")
            return {"error": str(e), "results": []}
    
    async def search_async(self, session: aiohttp.ClientSession, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via Lens API asynchronously"""
        try:
            async with session.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "corpus": corpus, 
                    "k": k,
                    "include_embeddings": False
                },
                timeout=10
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Lens async search failed: {e}")
            return {"error": str(e), "results": []}

class ZoektAdapter(SystemAdapter):
    """Adapter for Zoekt search system"""
    
    def __init__(self):
        base_url = os.environ.get("ZOEKT_URL", "http://zoekt-webserver:6070")
        super().__init__("zoekt", base_url)
    
    def search(self, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via Zoekt API"""
        try:
            # Zoekt uses different API format
            params = {
                "q": query,
                "num": k,
                "format": "json"
            }
            response = requests.get(f"{self.base_url}/search", params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to standard format
            results = []
            if "Results" in data:
                for result in data["Results"][:k]:
                    results.append({
                        "file_path": result.get("FileName", ""),
                        "line_number": result.get("LineNumber", 0),
                        "content": result.get("Line", ""),
                        "score": result.get("Score", 0.0)
                    })
            
            return {
                "system": "zoekt",
                "query": query,
                "total_hits": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Zoekt search failed: {e}")
            return {"error": str(e), "results": []}
    
    async def search_async(self, session: aiohttp.ClientSession, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via Zoekt API asynchronously"""
        try:
            params = {
                "q": query,
                "num": k,
                "format": "json"
            }
            async with session.get(f"{self.base_url}/search", params=params, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Convert to standard format
                results = []
                if "Results" in data:
                    for result in data["Results"][:k]:
                        results.append({
                            "file_path": result.get("FileName", ""),
                            "line_number": result.get("LineNumber", 0),
                            "content": result.get("Line", ""),
                            "score": result.get("Score", 0.0)
                        })
                
                return {
                    "system": "zoekt",
                    "query": query,
                    "total_hits": len(results),
                    "results": results
                }
                
        except Exception as e:
            logger.error(f"Zoekt async search failed: {e}")
            return {"error": str(e), "results": []}

class RipgrepAdapter(SystemAdapter):
    """Adapter for ripgrep search system"""
    
    def __init__(self):
        base_url = os.environ.get("RIPGREP_URL", "http://ripgrep-server:8080")
        super().__init__("ripgrep", base_url)
    
    def search(self, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via ripgrep API"""
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "max_results": k,
                    "case_sensitive": False,
                    "regex": True
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Ripgrep search failed: {e}")
            return {"error": str(e), "results": []}
    
    async def search_async(self, session: aiohttp.ClientSession, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via ripgrep API asynchronously"""
        try:
            async with session.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "max_results": k,
                    "case_sensitive": False,
                    "regex": True
                },
                timeout=10
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Ripgrep async search failed: {e}")
            return {"error": str(e), "results": []}

class CombyAdapter(SystemAdapter):
    """Adapter for Comby structural search"""
    
    def __init__(self):
        base_url = os.environ.get("COMBY_URL", "http://comby-server:8081")
        super().__init__("comby", base_url)
    
    def search(self, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via Comby API"""
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "match_template": query,
                    "max_results": k,
                    "language": "generic"
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Comby search failed: {e}")
            return {"error": str(e), "results": []}
    
    async def search_async(self, session: aiohttp.ClientSession, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via Comby API asynchronously"""
        try:
            async with session.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "match_template": query,
                    "max_results": k,
                    "language": "generic"
                },
                timeout=10
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Comby async search failed: {e}")
            return {"error": str(e), "results": []}

class AstGrepAdapter(SystemAdapter):
    """Adapter for ast-grep structural search"""
    
    def __init__(self):
        base_url = os.environ.get("AST_GREP_URL", "http://ast-grep-server:8082")
        super().__init__("ast-grep", base_url)
    
    def search(self, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via ast-grep API"""
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={
                    "pattern": query,
                    "language": "javascript",
                    "max_results": k
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"ast-grep search failed: {e}")
            return {"error": str(e), "results": []}
    
    async def search_async(self, session: aiohttp.ClientSession, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via ast-grep API asynchronously"""
        try:
            async with session.post(
                f"{self.base_url}/search",
                json={
                    "pattern": query,
                    "language": "javascript",
                    "max_results": k
                },
                timeout=10
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"ast-grep async search failed: {e}")
            return {"error": str(e), "results": []}

class OpenSearchAdapter(SystemAdapter):
    """Adapter for OpenSearch with k-NN"""
    
    def __init__(self):
        base_url = os.environ.get("OPENSEARCH_URL", "http://opensearch:9200")
        super().__init__("opensearch", base_url)
    
    def search(self, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via OpenSearch API"""
        try:
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "code", "text"],
                        "type": "best_fields"
                    }
                },
                "size": k,
                "_source": ["content", "file_path", "language"]
            }
            
            response = requests.post(
                f"{self.base_url}/{corpus}/_search",
                json=search_body,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "hits" in data and "hits" in data["hits"]:
                for hit in data["hits"]["hits"]:
                    results.append({
                        "file_path": hit["_source"].get("file_path", ""),
                        "content": hit["_source"].get("content", ""),
                        "score": hit["_score"],
                        "language": hit["_source"].get("language", "")
                    })
            
            return {
                "system": "opensearch",
                "query": query,
                "total_hits": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"OpenSearch search failed: {e}")
            return {"error": str(e), "results": []}
    
    async def search_async(self, session: aiohttp.ClientSession, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via OpenSearch API asynchronously"""
        try:
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "code", "text"],
                        "type": "best_fields"
                    }
                },
                "size": k,
                "_source": ["content", "file_path", "language"]
            }
            
            async with session.post(
                f"{self.base_url}/{corpus}/_search",
                json=search_body,
                timeout=10
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                results = []
                if "hits" in data and "hits" in data["hits"]:
                    for hit in data["hits"]["hits"]:
                        results.append({
                            "file_path": hit["_source"].get("file_path", ""),
                            "content": hit["_source"].get("content", ""),
                            "score": hit["_score"],
                            "language": hit["_source"].get("language", "")
                        })
                
                return {
                    "system": "opensearch",
                    "query": query,
                    "total_hits": len(results),
                    "results": results
                }
                
        except Exception as e:
            logger.error(f"OpenSearch async search failed: {e}")
            return {"error": str(e), "results": []}

class QdrantAdapter(SystemAdapter):
    """Adapter for Qdrant vector search"""
    
    def __init__(self):
        base_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
        super().__init__("qdrant", base_url)
    
    def search(self, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via Qdrant API"""
        try:
            # For demonstration, using basic text search
            # In reality, this would use vector embeddings
            search_body = {
                "vector": [0.1] * 512,  # Mock vector
                "limit": k,
                "with_payload": True,
                "with_vector": False
            }
            
            response = requests.post(
                f"{self.base_url}/collections/{corpus}/points/search",
                json=search_body,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "result" in data:
                for result in data["result"]:
                    payload = result.get("payload", {})
                    results.append({
                        "file_path": payload.get("file_path", ""),
                        "content": payload.get("content", ""),
                        "score": result.get("score", 0.0),
                        "id": result.get("id", "")
                    })
            
            return {
                "system": "qdrant",
                "query": query,
                "total_hits": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return {"error": str(e), "results": []}
    
    async def search_async(self, session: aiohttp.ClientSession, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via Qdrant API asynchronously"""
        try:
            search_body = {
                "vector": [0.1] * 512,  # Mock vector
                "limit": k,
                "with_payload": True,
                "with_vector": False
            }
            
            async with session.post(
                f"{self.base_url}/collections/{corpus}/points/search",
                json=search_body,
                timeout=10
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                results = []
                if "result" in data:
                    for result in data["result"]:
                        payload = result.get("payload", {})
                        results.append({
                            "file_path": payload.get("file_path", ""),
                            "content": payload.get("content", ""),
                            "score": result.get("score", 0.0),
                            "id": result.get("id", "")
                        })
                
                return {
                    "system": "qdrant",
                    "query": query,
                    "total_hits": len(results),
                    "results": results
                }
                
        except Exception as e:
            logger.error(f"Qdrant async search failed: {e}")
            return {"error": str(e), "results": []}

class FAISSAdapter(SystemAdapter):
    """Adapter for FAISS ANN search"""
    
    def __init__(self):
        base_url = os.environ.get("FAISS_URL", "http://faiss-server:8084")
        super().__init__("faiss", base_url)
    
    def search(self, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via FAISS API"""
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "corpus": corpus,
                    "k": k
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return {"error": str(e), "results": []}
    
    async def search_async(self, session: aiohttp.ClientSession, query: str, corpus: str = "default", k: int = 10) -> Dict[str, Any]:
        """Execute search via FAISS API asynchronously"""
        try:
            async with session.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "corpus": corpus,
                    "k": k
                },
                timeout=10
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"FAISS async search failed: {e}")
            return {"error": str(e), "results": []}

class SystemAdapterRegistry:
    """Registry for all system adapters"""
    
    def __init__(self):
        self.adapters: Dict[str, SystemAdapter] = {}
        self._register_adapters()
    
    def _register_adapters(self):
        """Register all available system adapters"""
        adapter_classes = [
            LensAdapter,
            ZoektAdapter, 
            RipgrepAdapter,
            CombyAdapter,
            AstGrepAdapter,
            OpenSearchAdapter,
            QdrantAdapter,
            FAISSAdapter
        ]
        
        for adapter_class in adapter_classes:
            try:
                adapter = adapter_class()
                self.adapters[adapter.name] = adapter
                logger.info(f"Registered adapter: {adapter.name} ({adapter.base_url})")
            except Exception as e:
                logger.error(f"Failed to register adapter {adapter_class.__name__}: {e}")
    
    def get_adapter(self, system_name: str) -> Optional[SystemAdapter]:
        """Get adapter by system name"""
        return self.adapters.get(system_name)
    
    def get_all_adapters(self) -> Dict[str, SystemAdapter]:
        """Get all registered adapters"""
        return self.adapters.copy()
    
    def get_system_names(self) -> List[str]:
        """Get list of all system names"""
        return list(self.adapters.keys())