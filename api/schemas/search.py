"""
Search API schemas for request/response validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class SearchResultType(str, Enum):
    """Type of search result."""
    DOCUMENT = "document"
    CHUNK = "chunk"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, max_length=5000, description="Search query text")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for vector search")
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for keyword search")
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum combined score threshold")
    include_chunks: bool = Field(default=True, description="Include chunk results in search")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchResultItem(BaseModel):
    """Individual search result item."""
    id: str = Field(..., description="Unique identifier")
    description: str = Field(..., description="Law case description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata")
    vector_score: float = Field(..., ge=0.0, le=1.0, description="Vector similarity score")
    keyword_score: float = Field(..., ge=0.0, le=1.0, description="Keyword matching score")
    combined_score: float = Field(..., ge=0.0, le=1.0, description="Final combined score")
    explanation: str = Field(default="", description="Explanation of scoring calculation")
    rank: Optional[int] = Field(default=None, description="Rank in search results")
    result_type: SearchResultType = Field(default=SearchResultType.DOCUMENT, description="Type of search result")
    parent_document_id: Optional[str] = Field(default=None, description="Parent document ID for chunks")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index within parent document")
    retrieved_at: str = Field(..., description="Retrieval timestamp (ISO format)")


class SearchResponse(BaseModel):
    """Search response model."""
    query: str = Field(..., description="Original search query")
    results: List[SearchResultItem] = Field(default_factory=list, description="Search results")
    total_results: int = Field(default=0, description="Total number of results found")
    search_time_ms: float = Field(..., description="Search processing time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp (ISO format)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: str = Field(..., description="Check timestamp (ISO format)")
    checks: Dict[str, Any] = Field(default_factory=dict, description="Individual health checks")


class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_searches: int = Field(default=0, description="Total number of searches performed")
    avg_search_time_ms: float = Field(default=0.0, description="Average search time in milliseconds")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    active_connections: int = Field(default=0, description="Number of active connections")
    uptime_seconds: float = Field(default=0.0, description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp (ISO format)")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracing")