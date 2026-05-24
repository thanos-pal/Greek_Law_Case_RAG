"""
Document storage and management.

This module provides high-level document operations including ingestion,
retrieval, and management with automatic embedding generation and metadata handling.
"""

from typing import List, Optional
from qdrant_client.models import PointStruct
import uuid
import logging
import hashlib
from datetime import datetime
from dataclasses import dataclass
from .qdrant_client import QdrantManager, SearchPoint
from ..models.document import Document, DocumentMetadata
from ..config.settings import Settings


@dataclass
class IngestionResult:
    """Result of document ingestion operation."""

    success: bool
    document_id: str
    message: str
    processing_time: float
    token_count: Optional[int] = None
    chunk_count: Optional[int] = None


class DocumentStore:
    """High-level document storage and retrieval system."""

    def __init__(self, qdrant_manager: QdrantManager, settings: Settings):
        """Initialize document store."""
        self.qdrant = qdrant_manager
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    def ingest_document(
        self,
        document: Document,
        embedding: List[float],
        chunk_embeddings: Optional[List[List[float]]] = None,
    ) -> IngestionResult:
        """Ingest a single document with its embedding."""
        start_time = datetime.now()

        try:
            # Generate document ID if not provided
            if not document.id:
                document.id = self.generate_document_id(document.description)

            # Create main document point
            main_point = PointStruct(
                id=document.id,
                vector=embedding,
                payload={
                    "description": document.description,
                    "metadata": document.metadata.model_dump(),
                    "document_type": "main",
                    "created_at": datetime.now().isoformat(),
                    "description_hash": self.hash_content(document.description),
                    "token_count": len(document.description.split()),  # Rough estimate
                },
            )

            points_to_upsert = [main_point]
            chunk_count = 0

            # Add chunk points if provided
            if chunk_embeddings and document.chunks:
                for i, (chunk, chunk_embedding) in enumerate(
                    zip(document.chunks, chunk_embeddings)
                ):
                    chunk_id = f"{document.id}_chunk_{i}"
                    chunk_point = PointStruct(
                        id=chunk_id,
                        vector=chunk_embedding,
                        payload={
                            "description": chunk,
                            "metadata": document.metadata.model_dump(),
                            "document_type": "chunk",
                            "parent_document_id": document.id,
                            "chunk_index": i,
                            "created_at": datetime.now().isoformat(),
                            "description_hash": self.hash_content(chunk),
                            "token_count": len(chunk.split()),
                        },
                    )
                    points_to_upsert.append(chunk_point)
                    chunk_count += 1

            # Upsert all points
            success = self.qdrant.upsert_points(points_to_upsert)

            processing_time = (datetime.now() - start_time).total_seconds()

            if success:
                self.logger.info(
                    f"Successfully ingested document {document.id} with {chunk_count} chunks"
                )
                return IngestionResult(
                    success=True,
                    document_id=document.id,
                    message=f"Document ingested successfully with {chunk_count} chunks",
                    processing_time=processing_time,
                    token_count=len(document.description.split()),
                    chunk_count=chunk_count,
                )
            else:
                return IngestionResult(
                    success=False,
                    document_id=document.id,
                    message="Failed to upsert document points",
                    processing_time=processing_time,
                )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error ingesting document: {e}")
            return IngestionResult(
                success=False,
                document_id=document.id or "unknown",
                message=f"Ingestion failed: {str(e)}",
                processing_time=processing_time,
            )

    def ingest_documents_batch(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        chunk_embeddings: Optional[List[List[List[float]]]] = None,
    ) -> List[IngestionResult]:
        """Ingest multiple documents in batch."""
        results = []

        for i, (document, embedding) in enumerate(zip(documents, embeddings)):
            chunk_emb = chunk_embeddings[i] if chunk_embeddings else None
            result = self.ingest_document(document, embedding, chunk_emb)
            results.append(result)

        successful = sum(1 for r in results if r.success)
        self.logger.info(
            f"Batch ingestion completed: {successful}/{len(results)} documents successful"
        )

        return results

    def get_document(
        self, document_id: str, include_chunks: bool = False
    ) -> Optional[Document]:
        """Retrieve a document by ID."""
        try:
            # Get main document
            points, _ = self.qdrant.scroll_points(
                limit=1, offset=None, with_payload=True
            )

            main_doc = None
            for point in points:
                if (
                    point.id == document_id
                    and point.payload.get("document_type") == "main"
                ):
                    main_doc = point
                    break

            if not main_doc:
                return None

            # Reconstruct document
            metadata = DocumentMetadata(**main_doc.payload["metadata"])
            chunks = []

            if include_chunks:
                # Get all chunks for this document
                chunk_points, _ = self.qdrant.scroll_points(
                    limit=1000, with_payload=True  # Assume max 1000 chunks per document
                )

                document_chunks = [
                    p
                    for p in chunk_points
                    if p.payload.get("parent_document_id") == document_id
                ]

                # Sort chunks by index
                document_chunks.sort(key=lambda x: x.payload.get("chunk_index", 0))
                chunks = [chunk.payload["description"] for chunk in document_chunks]

            return Document(
                id=document_id,
                description=main_doc.payload["description"],
                metadata=metadata,
                chunks=chunks if chunks else None,
            )

        except Exception as e:
            self.logger.error(f"Error retrieving document {document_id}: {e}")
            return None

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            # Find all points related to this document
            points_to_delete = [document_id]

            # Find chunk IDs
            chunk_points, _ = self.qdrant.scroll_points(limit=1000, with_payload=True)

            for point in chunk_points:
                if point.payload.get("parent_document_id") == document_id:
                    points_to_delete.append(point.id)

            # Delete all related points
            success = self.qdrant.delete_points(points_to_delete)

            if success:
                self.logger.info(
                    f"Deleted document {document_id} and {len(points_to_delete)-1} chunks"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error deleting document {document_id}: {e}")
            return False

    def search_documents(
        self,
        query_vector: List[float],
        limit: int = 10,
        include_chunks: bool = True,
        score_threshold: Optional[float] = None,
    ) -> List[SearchPoint]:
        """Search documents using vector similarity."""
        try:

            # Perform search
            results = self.qdrant.search(
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold,
            )

            # Filter results based on document type preference
            if not include_chunks:
                results = [
                    r for r in results if r.payload.get("document_type") == "main"
                ]

            return results

        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []

    def generate_document_id(self, description: str) -> str:
        """Generate a unique document ID as a valid UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, description))

    def hash_content(self, description: str) -> str:
        """Generate a hash of the content for deduplication."""
        return hashlib.sha256(description.encode()).hexdigest()
