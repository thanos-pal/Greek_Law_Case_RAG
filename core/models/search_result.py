"""
Search result models.

This module defines the data structures for search results, response analysis
and response generation.

"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class SearchResultType(str, Enum):
    """Type of search result."""

    DOCUMENT = "document"
    CHUNK = "chunk"
    HYBRID = "hybrid"


class SearchResult(BaseModel):
    """Search result with scoring details."""

    # Basic information
    id: str = Field(..., description="Unique identifier of the result")
    description: str = Field(..., description="Law case description")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Associated metadata"
    )

    # Scoring information
    vector_score: float = Field(..., description="Vector similarity score")
    keyword_score: float = Field(0.0, description="Keyword matching score")
    combined_score: float = Field(..., description="Final combined score")

    # Additional scoring details
    explanation: str = Field("", description="Explanation of scoring calculation")
    rank: Optional[int] = Field(None, description="Rank in search results")

    # Result type and source
    result_type: SearchResultType = Field(
        SearchResultType.DOCUMENT, description="Type of search result"
    )
    parent_document_id: Optional[str] = Field(
        None, description="Parent document ID for chunks"
    )
    chunk_index: Optional[int] = Field(
        None, description="Chunk index within parent document"
    )

    # Processing metadata
    retrieved_at: datetime = Field(
        default_factory=datetime.now, description="Retrieval timestamp"
    )
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )

    @field_validator("vector_score", "keyword_score", "combined_score")
    @classmethod
    def validate_scores(cls, v):
        """Ensure scores are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Scores must be between 0 and 1")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        """Ensure description is not empty."""
        if not v or not v.strip():
            raise ValueError("Law case description cannot be empty")
        return v.strip()

    @property
    def is_chunk(self) -> bool:
        """Check if this is a chunk result."""
        return self.result_type == SearchResultType.CHUNK

    @property
    def is_document(self) -> bool:
        """Check if this is a document result."""
        return self.result_type == SearchResultType.DOCUMENT

    def get_source_info(self) -> Dict[str, Any]:
        """Get source information for citation."""
        source_info = {
            "id": self.id,
            "type": self.result_type,
            "score": self.combined_score,
        }

        if self.parent_document_id:
            source_info["parent_document_id"] = self.parent_document_id

        if self.chunk_index is not None:
            source_info["chunk_index"] = self.chunk_index

        # Add relevant metadata
        if "case_number" in self.metadata:
            source_info["case_number"] = self.metadata["case_number"]
        if "year" in self.metadata:
            source_info["year"] = self.metadata["year"]

        return source_info

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}
