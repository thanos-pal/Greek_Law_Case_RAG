"""
Dimosthenis AI: Production-Ready Greek Law Case Recommendation System with Qdrant Vector Database

A Retrieval-Augmented Generation system that combines Qdrant's
vector database with advanced search capabilities, hybrid search,
metadata filtering, and intelligent response generation.

"""

__version__ = "1.0.0"
__author__ = "T. Palantzas"
__email__ = "thanospalantzas@gmail.com"

from .config.settings import Settings
from .database.qdrant_client import QdrantManager
from .database.document_store import DocumentStore
from .services.embedding_service import EmbeddingService
from .services.search_engine import HybridSearchEngine
from .services.response_generator import ResponseGenerator

__all__ = [
    "Settings",
    "QdrantManager",
    "DocumentStore",
    "EmbeddingService",
    "HybridSearchEngine",
    "ResponseGenerator",
]
