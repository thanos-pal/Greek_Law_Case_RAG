"""
Search API routes.
"""

import time
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from api.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
    SearchResultType,
)
from api.services.search_service import get_search_service, SearchService

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_documents(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
    http_request: Request = None,
):
    """
    Search for similar Greek law cases using hybrid vector and keyword search.

    - **query**: The search query text in Greek
    - **limit**: Maximum number of results to return (1-50)
    - **vector_weight**: Weight for vector similarity search (0.0-1.0)
    - **keyword_weight**: Weight for keyword matching (0.0-1.0)
    - **min_score**: Minimum combined score threshold (optional)
    - **include_chunks**: Whether to include chunk-level results
    """
    start_time = time.time()
    request_id = getattr(http_request.state, "request_id", None) if http_request else None

    try:
        # Validate weights sum
        if abs(request.vector_weight + request.keyword_weight - 1.0) > 0.001:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="vector_weight and keyword_weight must sum to 1.0"
            )

        # Perform search
        results = await search_service.search(
            query=request.query,
            limit=request.limit,
            vector_weight=request.vector_weight,
            keyword_weight=request.keyword_weight,
            min_score=request.min_score,
            include_chunks=request.include_chunks,
        )

        search_time_ms = (time.time() - start_time) * 1000

        # Convert results to response format
        result_items = []
        for result in results:
            # result_type might be an enum or a string (due to use_enum_values=True)
            if hasattr(result.result_type, 'value'):
                result_type_value = result.result_type.value
            else:
                result_type_value = result.result_type

            result_items.append(SearchResultItem(
                id=result.id,
                description=result.description,
                metadata=result.metadata,
                vector_score=result.vector_score,
                keyword_score=result.keyword_score,
                combined_score=result.combined_score,
                explanation=result.explanation,
                rank=result.rank,
                result_type=SearchResultType(result_type_value),
                parent_document_id=result.parent_document_id,
                chunk_index=result.chunk_index,
                retrieved_at=result.retrieved_at.isoformat() if result.retrieved_at else datetime.now().isoformat(),
            ))

        return SearchResponse(
            query=request.query,
            results=result_items,
            total_results=len(result_items),
            search_time_ms=round(search_time_ms, 2),
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        search_time_ms = (time.time() - start_time) * 1000
        # Log the error with request ID for tracing
        if request_id:
            import logging
            logger = logging.getLogger("api")
            logger.error(
                "Search failed",
                extra={
                    "request_id": request_id,
                    "query": request.query[:100],
                    "error": str(e),
                    "search_time_ms": round(search_time_ms, 2),
                },
                exc_info=True
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Search failed",
                "message": str(e),
                "request_id": request_id,
                "search_time_ms": round(search_time_ms, 2),
            }
        )


@router.get("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_documents_get(
    query: str = Query(..., min_length=1, max_length=5000, description="Search query text"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of results"),
    vector_weight: float = Query(default=0.7, ge=0.0, le=1.0, description="Vector search weight"),
    keyword_weight: float = Query(default=0.3, ge=0.0, le=1.0, description="Keyword search weight"),
    min_score: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Minimum score threshold"),
    include_chunks: bool = Query(default=True, description="Include chunk results"),
    search_service: SearchService = Depends(get_search_service),
    http_request: Request = None,
):
    """
    Search for similar Greek law cases using GET method.

    Useful for simple queries and caching scenarios.
    """
    # Convert to POST request model and delegate
    search_request = SearchRequest(
        query=query,
        limit=limit,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        min_score=min_score,
        include_chunks=include_chunks,
    )
    return await search_documents(search_request, search_service, http_request)


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(search_service: SearchService = Depends(get_search_service)):
    """
    Health check endpoint for monitoring and load balancers.

    Returns the health status of all system components.
    """
    health_data = await search_service.health_check()

    return HealthResponse(
        status=health_data["status"],
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        checks=health_data["checks"],
    )


@router.get("/metrics", response_model=MetricsResponse, status_code=status.HTTP_200_OK)
async def get_metrics(search_service: SearchService = Depends(get_search_service)):
    """
    Get service metrics for monitoring and observability.

    Returns search counts, timing, cache statistics, and uptime.
    """
    metrics = search_service.get_metrics()

    return MetricsResponse(**metrics)


@router.get("/stats", response_model=dict, status_code=status.HTTP_200_OK)
async def get_stats(search_service: SearchService = Depends(get_search_service)):
    """
    Get detailed service statistics.

    Returns comprehensive stats including embedding cache and search performance.
    """
    return search_service.get_stats()