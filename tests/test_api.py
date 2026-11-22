# -*- coding: utf-8 -*-
"""
Integration tests for the Greek Law Case RAG API.

These tests verify that the FastAPI endpoints work correctly
when running in a Docker environment with Qdrant.
"""

import os
import pytest
import httpx
import asyncio
from typing import AsyncGenerator

# Base URL for the API (can be overridden via environment variable)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client for testing the API."""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        yield client


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    async def test_health_check_returns_healthy(self, http_client: httpx.AsyncClient):
        """Test that the health endpoint returns a healthy status."""
        response = await http_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "timestamp" in data
        assert "checks" in data

    async def test_health_check_includes_component_checks(self, http_client: httpx.AsyncClient):
        """Test that health check includes Qdrant and embeddings checks."""
        response = await http_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        checks = data["checks"]

        # Should have at least Qdrant and embeddings checks
        assert "qdrant" in checks
        assert "embeddings" in checks

        # Each check should have status and details
        for check_name, check_data in checks.items():
            assert "status" in check_data
            assert check_data["status"] in ["healthy", "unhealthy"]
            assert "details" in check_data


@pytest.mark.integration
class TestSearchEndpoint:
    """Tests for the search endpoint."""

    async def test_search_post_valid_request(self, http_client: httpx.AsyncClient):
        """Test POST /search with a valid request."""
        payload = {
            "query": "κατοχή ναρκωτικών ουσιών για προσωπική χρήση",
            "limit": 5,
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
            "min_score": 0.3,
            "include_chunks": True
        }

        response = await http_client.post("/api/v1/search", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "query" in data
        assert data["query"] == payload["query"]
        assert "results" in data
        assert "total_results" in data
        assert "search_time_ms" in data
        assert "timestamp" in data

        # Verify result structure if results exist
        if data["total_results"] > 0:
            result = data["results"][0]
            assert "id" in result
            assert "description" in result
            assert "metadata" in result
            assert "vector_score" in result
            assert "keyword_score" in result
            assert "combined_score" in result
            assert "rank" in result
            assert "result_type" in result

    async def test_search_post_invalid_weights(self, http_client: httpx.AsyncClient):
        """Test POST /search with invalid weights (don't sum to 1.0)."""
        payload = {
            "query": "test query",
            "limit": 5,
            "vector_weight": 0.8,
            "keyword_weight": 0.1,  # Sums to 0.9, not 1.0
        }

        response = await http_client.post("/api/v1/search", json=payload)
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "vector_weight and keyword_weight must sum to 1.0" in str(data["detail"])

    async def test_search_get_valid_request(self, http_client: httpx.AsyncClient):
        """Test GET /search with query parameters."""
        params = {
            "query": "απάτη μέσω ηλεκτρονικών μέσων",
            "limit": 3,
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }

        response = await http_client.get("/api/v1/search", params=params)
        assert response.status_code == 200

        data = response.json()
        assert "query" in data
        assert data["query"] == params["query"]
        assert "results" in data
        assert data["total_results"] <= params["limit"]

    async def test_search_get_missing_query(self, http_client: httpx.AsyncClient):
        """Test GET /search without required query parameter."""
        response = await http_client.get("/api/v1/search")
        assert response.status_code == 422  # Validation error

    async def test_search_empty_query(self, http_client: httpx.AsyncClient):
        """Test search with empty query string."""
        payload = {
            "query": "",
            "limit": 5,
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }

        response = await http_client.post("/api/v1/search", json=payload)
        assert response.status_code == 422  # Validation error for min_length

    async def test_search_limit_validation(self, http_client: httpx.AsyncClient):
        """Test search with invalid limit values."""
        # Test limit too high
        payload = {
            "query": "test",
            "limit": 100,  # Max is 50
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }

        response = await http_client.post("/api/v1/search", json=payload)
        assert response.status_code == 422

        # Test limit too low
        payload["limit"] = 0
        response = await http_client.post("/api/v1/search", json=payload)
        assert response.status_code == 422


@pytest.mark.integration
class TestMetricsEndpoint:
    """Tests for the metrics endpoint."""

    async def test_metrics_endpoint(self, http_client: httpx.AsyncClient):
        """Test that metrics endpoint returns expected fields."""
        response = await http_client.get("/api/v1/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "total_searches" in data
        assert "avg_search_time_ms" in data
        assert "cache_hits" in data
        assert "cache_misses" in data
        assert "active_connections" in data
        assert "uptime_seconds" in data


@pytest.mark.integration
class TestRootEndpoint:
    """Tests for the root endpoint."""

    async def test_root_endpoint(self, http_client: httpx.AsyncClient):
        """Test that root endpoint returns API info."""
        response = await http_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "docs" in data
        assert "health" in data
        assert "metrics" in data


@pytest.mark.integration
class TestStatsEndpoint:
    """Tests for the stats endpoint."""

    async def test_stats_endpoint(self, http_client: httpx.AsyncClient):
        """Test that stats endpoint returns detailed statistics."""
        response = await http_client.get("/api/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert "searches" in data
        assert "total_search_time_ms" in data
        assert "avg_search_time_ms" in data
        assert "cache" in data
        assert "uptime_seconds" in data


@pytest.mark.integration
class TestCORSHeaders:
    """Tests for CORS headers."""

    async def test_cors_headers_present(self, http_client: httpx.AsyncClient):
        """Test that CORS headers are present in responses."""
        response = await http_client.options("/api/v1/health")
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers


# Standalone test runner for manual execution
async def run_basic_health_check():
    """Run a basic health check against the API."""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        try:
            response = await client.get("/api/v1/health")
            print(f"Health check status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"API Status: {data['status']}")
                print(f"Version: {data['version']}")
                print("Component checks:")
                for check, details in data.get("checks", {}).items():
                    print(f"  {check}: {details['status']} - {details['details']}")
            else:
                print(f"Health check failed: {response.text}")
        except Exception as e:
            print(f"Error connecting to API: {e}")


if __name__ == "__main__":
    asyncio.run(run_basic_health_check())