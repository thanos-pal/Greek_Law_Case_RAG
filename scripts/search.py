# -*- coding: utf-8 -*-
"""
This script provides document retrieval

"""

import sys
import asyncio
from pathlib import Path
from core.config.settings import Settings
from core.database.qdrant_client import QdrantManager
from core.database.document_store import DocumentStore
from core.services.embedding_service import EmbeddingService
from core.services.search_engine import HybridSearchEngine

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Initialize system components
settings = Settings()
qdrant_manager = QdrantManager(settings)
embedding_service = EmbeddingService(settings)
document_store = DocumentStore(qdrant_manager, settings)
search_engine = HybridSearchEngine(qdrant_manager, embedding_service, settings)


# Retrieve documents
async def retrieve_documents(query: str):

    print(f"Processing: {query} ...")

    # Search for relevant documents
    search_results = await search_engine.search(query, limit=5)

    return search_results


# test_queries = [
#     "κατοχή ναρκωτικών ουσιών για προσωπική χρήση",
#     "απάτη μέσω ηλεκτρονικών μέσων",
#     "απρόσεκτη οδήγηση που προκάλεσε σωματική βλάβη"
# ]

query = input()

response_df = asyncio.run(retrieve_documents(query))
print(f"Response: {response_df}")
