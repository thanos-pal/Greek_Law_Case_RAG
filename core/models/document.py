"""
Document models.

This module defines the data structures for documents and their metadata.

"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""

    JSON = "json"


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""

    # Basic metadata
    case_number: Optional[str] = Field(None, description="Number of the case")
    year: Optional[str] = Field(None, description="Year of the case")
    link: Optional[str] = Field(None, description="URL of the case")

    # Classification
    court_type: Optional[str] = Field(None, description="Court type the case belongs")
    tags: List[str] = Field(default_factory=list, description="Case tags")

    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Document creation time")
    modified_at: Optional[datetime] = Field(None, description="Last modification time")
    ingested_at: Optional[datetime] = Field(None, description="Ingestion timestamp")

    # Processing metadata
    status: DocumentStatus = Field(
        DocumentStatus.PENDING, description="Processing status"
    )
    processing_version: str = Field("1.0", description="Processing pipeline version")

    # Custom metadata
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata fields"
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Ensure tags are non-empty strings."""
        return [tag.strip() for tag in v if tag.strip()]

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


class Document(BaseModel):
    """Main document model for the RAG system."""

    id: Optional[str] = Field(None, description="Unique document identifier")
    description: str = Field(..., description="Law case description")
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata, description="Document metadata"
    )

    # Chunking information
    chunks: Optional[List[str]] = Field(
        None, description="Document chunks for processing"
    )
    chunk_metadata: Optional[List[Dict[str, Any]]] = Field(
        None, description="Metadata for each chunk"
    )

    # Embedding information
    embedding: Optional[List[float]] = Field(
        None, description="Document embedding vector"
    )
    chunk_embeddings: Optional[List[List[float]]] = Field(
        None, description="Chunk embedding vectors"
    )

    # Processing information
    token_count: Optional[int] = Field(None, description="Total token count")
    chunk_token_counts: Optional[List[int]] = Field(
        None, description="Token count per chunk"
    )

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        """Ensure description is not empty."""
        if not v or not v.strip():
            raise ValueError("Law case description cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_chunks_consistency(self):
        """Validate chunks consistency."""
        if self.chunks is not None:
            # Ensure chunks are non-empty
            self.chunks = [chunk.strip() for chunk in self.chunks if chunk.strip()]
            if not self.chunks:
                self.chunks = None

        # Validate chunk embeddings match chunks
        if self.chunk_embeddings is not None and self.chunks is not None:
            if len(self.chunk_embeddings) != len(self.chunks):
                raise ValueError(
                    "Number of chunk embeddings must match number of chunks"
                )

        # Validate chunk token counts match chunks
        if self.chunk_token_counts is not None and self.chunks is not None:
            if len(self.chunk_token_counts) != len(self.chunks):
                raise ValueError(
                    "Number of chunk token counts must match number of chunks"
                )

        return self

    @property
    def has_chunks(self) -> bool:
        """Check if document has chunks."""
        return self.chunks is not None and len(self.chunks) > 0

    @property
    def has_embeddings(self) -> bool:
        """Check if document has embeddings."""
        return self.embedding is not None

    @property
    def has_chunk_embeddings(self) -> bool:
        """Check if document has chunk embeddings."""
        return self.chunk_embeddings is not None and len(self.chunk_embeddings) > 0

    def get_chunk_count(self) -> int:
        """Get the number of chunks."""
        return len(self.chunks) if self.chunks else 0

    def get_total_tokens(self) -> int:
        """Get total token count."""
        if self.token_count:
            return self.token_count
        elif self.chunk_token_counts:
            return sum(self.chunk_token_counts)
        else:
            # Rough estimate
            return len(self.description.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        return cls(**data)

    @classmethod
    def from_text(
        cls,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> "Document":
        """Create document from plain text."""
        doc_metadata = DocumentMetadata(**(metadata or {}))
        return cls(id=document_id, description=description, metadata=doc_metadata)

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}
