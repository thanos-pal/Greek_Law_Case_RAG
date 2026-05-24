"""
Advanced search engine combining vector and keyword search with intelligent ranking.

This module implements hybrid search capabilities that combine semantic vector search
with traditional keyword matching for superior retrieval accuracy.

"""

from typing import List, Dict, Optional
import logging
import time
import re
import math
from collections import Counter
from dataclasses import dataclass
from ..models.search_result import SearchResult, SearchResultType
from ..database.qdrant_client import QdrantManager, SearchPoint
from .embedding_service import EmbeddingService
from ..config.settings import Settings


@dataclass
class SearchQuery:
    """Structured search query with analysis."""

    original_query: str
    processed_query: str
    query_terms: List[str]


class QueryAnalyzer:
    """Analyzes queries to determine optimal search strategy."""

    def __init__(self):
        self.technical_patterns = [
            r"\b[A-Z]{2,}\b",  # Acronyms
            r"\b\d+\.\d+\b",  # Version numbers
            r"\b[a-zA-Z]+\d+\b",  # Product codes
            r"\b[A-Z][a-z]+[A-Z][a-z]+\b",  # CamelCase
            r"\b\w+\.\w+\b",  # Dotted notation
        ]

    def analyze_query(self, query: str) -> SearchQuery:
        """Analyze query."""
        processed_query = self.preprocess_query(query)
        query_terms = processed_query.lower().split()

        return SearchQuery(
            original_query=query,
            processed_query=processed_query,
            query_terms=query_terms,
        )

    def preprocess_query(self, query: str) -> str:
        """Clean and normalize query text."""
        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query.strip())

        # Preserve important punctuation but remove noise
        query = re.sub(r"[^\w\s\.\-_]", " ", query)

        return query


class HybridSearchEngine:
    """Advanced search engine combining vector and keyword search with intelligent ranking."""

    def __init__(
        self,
        qdrant_manager: QdrantManager,
        embedding_service: EmbeddingService,
        settings: Settings,
    ):
        """Initialize hybrid search engine."""
        self.qdrant = qdrant_manager
        self.embedder = embedding_service
        self.settings = settings
        self.query_analyzer = QueryAnalyzer()
        self.logger = logging.getLogger(__name__)

        # Search configuration
        self.default_vector_weight = settings.default_vector_weight
        self.default_keyword_weight = settings.default_keyword_weight
        self.min_score = settings.min_search_score

    async def search(
        self,
        query: str,
        limit: int = 10,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        """Perform hybrid search with weighted score combination."""
        start_time = time.time()

        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(query)

        # Determine weights
        vector_weight = vector_weight or self.default_vector_weight
        keyword_weight = keyword_weight or self.default_keyword_weight

        min_score = self.min_score

        # Generate query embedding
        query_embedding_result = await self.embedder.create_embedding(
            query_analysis.processed_query
        )
        query_vector = query_embedding_result.embedding

        # Perform vector search with expanded limit for reranking
        vector_results = self.qdrant.search(
            query_vector=query_vector,
            limit=limit * 3,  # Get more results for reranking
            with_payload=True,
            with_vectors=False,
            score_threshold=0.1,  # Low threshold for initial retrieval
        )

        if not vector_results:
            self.logger.warning("No vector search results found")
            return []

        # Calculate keyword scores
        keyword_scores = self.calculate_keyword_scores(query_analysis, vector_results)

        # Combine and rank results
        hybrid_results = self.combine_scores(
            vector_results, keyword_scores, vector_weight, keyword_weight
        )

        # Filter by minimum score and return top results
        filtered_results = [r for r in hybrid_results if r.combined_score >= min_score]
        final_results = sorted(
            filtered_results, key=lambda x: x.combined_score, reverse=True
        )[:limit]

        search_time = time.time() - start_time
        self.logger.info(
            f"Search completed in {search_time:.3f}s: "
            f"{len(vector_results)} initial → {len(filtered_results)} filtered → {len(final_results)} final"
        )

        # Add rank information
        for i, result in enumerate(final_results):
            result.rank = i + 1

        return final_results

    def calculate_keyword_scores(
        self, query_analysis: SearchQuery, vector_results: List[SearchPoint]
    ) -> Dict[str, float]:
        """Calculate keyword relevance scores using TF-IDF principles."""
        query_terms = set(term.lower() for term in query_analysis.query_terms)
        scores = {}

        # Calculate document frequencies for IDF
        doc_frequencies = Counter()
        all_documents = []

        for result in vector_results:
            description = result.payload.get("description", "").lower()
            description_terms = set(description.split())
            all_documents.append(description_terms)

            for term in query_terms:
                if term in description_terms:
                    doc_frequencies[term] += 1

        total_docs = len(all_documents)

        # Calculate scores for each document
        for i, result in enumerate(vector_results):
            description = result.payload.get("description", "").lower()
            description_terms = description.split()
            description_term_counts = Counter(description_terms)

            tf_idf_score = 0.0

            for term in query_terms:
                if term in description_term_counts:
                    # Term frequency
                    tf = description_term_counts[term] / len(description_terms)

                    # Inverse document frequency
                    df = doc_frequencies[term]
                    idf = math.log(total_docs / (df + 1)) if df > 0 else 0

                    tf_idf_score += tf * idf

            # Normalize by query length
            normalized_score = tf_idf_score / len(query_terms) if query_terms else 0

            # Apply additional scoring factors
            exact_matches = sum(
                1 for term in query_terms if term in description.lower()
            )
            exact_match_bonus = exact_matches / len(query_terms) if query_terms else 0

            # Combine TF-IDF with exact match bonus
            final_score = (normalized_score * 0.7) + (exact_match_bonus * 0.3)

            scores[result.id] = min(final_score, 1.0)  # Cap at 1.0

        return scores

    def combine_scores(
        self,
        vector_results: List[SearchPoint],
        keyword_scores: Dict[str, float],
        vector_weight: float,
        keyword_weight: float,
    ) -> List[SearchResult]:
        """Combine vector and keyword scores with weighted ranking."""
        combined_results = []

        for result in vector_results:
            vector_score = result.score
            keyword_score = keyword_scores.get(result.id, 0.0)

            # Weighted combination
            combined_score = (vector_score * vector_weight) + (
                keyword_score * keyword_weight
            )

            explanation = (
                f"Vector: {vector_score:.3f} (w={vector_weight:.2f}) + "
                f"Keyword: {keyword_score:.3f} (w={keyword_weight:.2f}) = "
                f"{combined_score:.3f}"
            )

            # Determine result type
            result_type = (
                SearchResultType.CHUNK
                if result.payload.get("document_type") == "chunk"
                else SearchResultType.DOCUMENT
            )

            combined_results.append(
                SearchResult(
                    id=result.id,
                    description=result.payload.get("description", ""),
                    metadata=result.payload.get("metadata", {}),
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    combined_score=min(combined_score, 1.0),  # Cap at 1.0
                    explanation=explanation,
                    result_type=result_type,
                    parent_document_id=result.payload.get("parent_document_id"),
                    chunk_index=result.payload.get("chunk_index"),
                )
            )

        return combined_results
