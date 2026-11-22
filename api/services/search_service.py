"""
Search service that wraps the existing search engine for FastAPI.
"""

import time
import logging
from typing import List, Optional
from datetime import datetime
import uuid

from core.config.settings import Settings
from core.database.qdrant_client import QdrantManager
from core.services.embedding_service import EmbeddingService
from core.services.search_engine import HybridSearchEngine
from core.models.search_result import SearchResult, SearchResultType
from core.models.document import Document, DocumentMetadata

logger = logging.getLogger(__name__)


class SearchService:
    """FastAPI-compatible search service."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.qdrant_manager = QdrantManager(settings)
        self.embedding_service = EmbeddingService(settings)
        self.search_engine = HybridSearchEngine(
            self.qdrant_manager,
            self.embedding_service,
            settings
        )
        self._search_count = 0
        self._total_search_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._start_time = time.time()

    async def search(
        self,
        query: str,
        limit: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        min_score: Optional[float] = None,
        include_chunks: bool = True,
    ) -> List[SearchResult]:
        """Perform search and return results."""
        start_time = time.time()

        try:
            results = await self.search_engine.search(
                query=query,
                limit=limit,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )

            # Apply min_score filter if provided
            if min_score is not None:
                results = [r for r in results if r.combined_score >= min_score]

            # Filter chunks if not included
            if not include_chunks:
                results = [r for r in results if r.result_type != SearchResultType.CHUNK]

            search_time = time.time() - start_time
            self._search_count += 1
            self._total_search_time += search_time

            logger.info(
                f"Search completed: query='{query[:50]}...', "
                f"results={len(results)}, time={search_time:.3f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

    async def health_check(self) -> dict:
        """Check health of all components."""
        checks = {}

        # Check Qdrant
        try:
            qdrant_healthy = self.qdrant_manager.health_check()
            checks["qdrant"] = {
                "status": "healthy" if qdrant_healthy else "unhealthy",
                "details": "Connected" if qdrant_healthy else "Connection failed"
            }
        except Exception as e:
            checks["qdrant"] = {"status": "unhealthy", "details": str(e)}

        # Check embedding service (try a simple embedding)
        try:
            test_result = await self.embedding_service.create_embedding("test")
            checks["embeddings"] = {
                "status": "healthy" if test_result.embedding else "unhealthy",
                "details": "Embedding generation working"
            }
        except Exception as e:
            checks["embeddings"] = {"status": "unhealthy", "details": str(e)}

        # Overall status
        all_healthy = all(c["status"] == "healthy" for c in checks.values())
        overall = "healthy" if all_healthy else "degraded" if any(c["status"] == "healthy" for c in checks.values()) else "unhealthy"

        return {
            "status": overall,
            "checks": checks
        }

    def get_metrics(self) -> dict:
        """Get service metrics."""
        uptime = time.time() - self._start_time
        avg_time = self._total_search_time / self._search_count if self._search_count > 0 else 0.0

        # Get cache stats from embedding service
        cache_stats = self.embedding_service.get_cache_stats()
        cache_hits = cache_stats.get("size", 0)  # Using cache size as proxy for hits

        return {
            "total_searches": self._search_count,
            "avg_search_time_ms": avg_time * 1000,
            "cache_hits": cache_hits,
            "cache_misses": self._cache_misses,
            "active_connections": 0,  # Would track if using websockets
            "uptime_seconds": uptime,
        }

    def get_stats(self) -> dict:
        """Get detailed statistics."""
        cache_stats = self.embedding_service.get_cache_stats()
        return {
            "searches": self._search_count,
            "total_search_time_ms": self._total_search_time * 1000,
            "avg_search_time_ms": (self._total_search_time / self._search_count * 1000) if self._search_count > 0 else 0,
            "cache": cache_stats,
            "uptime_seconds": time.time() - self._start_time,
        }


# Global service instance (will be initialized in lifespan)
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get the global search service instance."""
    if _search_service is None:
        raise RuntimeError("Search service not initialized")
    return _search_service


def set_search_service(service: SearchService) -> None:
    """Set the global search service instance."""
    global _search_service
    _search_service = service