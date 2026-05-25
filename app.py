"""
Streamlit frontend for RAG-based legal document retrieval
"""

import streamlit as st
import asyncio
from core.config.settings import Settings
from core.database.qdrant_client import QdrantManager
from core.services.embedding_service import EmbeddingService
from core.services.search_engine import HybridSearchEngine

# Page config
st.set_page_config(
    page_title="Greek Law Case Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .result-card {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #4F8BF9;
    }
    .score-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #4F8BF9 0%, #4F8BF9 var(--score));
    }
    .metadata-tag {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        margin: 0.25rem;
    }
    .rank-badge {
        background-color: #4F8BF9;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_components():
    """Initialize system components with caching"""
    settings = Settings()
    qdrant_manager = QdrantManager(settings)
    embedding_service = EmbeddingService(settings)
    search_engine = HybridSearchEngine(qdrant_manager, embedding_service, settings)
    return search_engine


def run_async_search(search_engine, query: str, limit: int = 5):
    """Wrapper to run async search in Streamlit"""
    return asyncio.run(search_engine.search(query, limit=limit))


def display_search_result(result, rank: int):
    """Display a single search result with styling"""

    with st.container():
        st.markdown(f'<div class="rank-badge">#{rank}</div>', unsafe_allow_html=True)

        # Description/Content
        st.markdown("### 📄 Summary")
        st.info(result.description)

        # External link
        link = result.metadata.get("link")
        if link:
            st.markdown(f"🔗 Case link: ({link})", unsafe_allow_html=True)

        st.divider()


def main():
    st.title("⚖️ Greek Legal Document Search")
    st.markdown(
        """
    Find similar Greek law cases based on your input.
    """
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Search Settings")

        limit = st.slider("Number of results", min_value=1, max_value=10, value=5)

    # Main search input
    query = st.text_input(
        "🔍 Enter your query",
        placeholder="π.χ. κατοχή ναρκωτικών ουσιών για προσωπική χρήση",
        key="search_query",
    )

    # Search button
    if st.button("🔎 Search Documents", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("🔍 Searching legal documents..."):
                try:
                    # Initialize components (cached)
                    search_engine = initialize_components()

                    # Run search
                    results = run_async_search(search_engine, query, limit=limit)

                    # Display results
                    if results:
                        st.success(f"Found {len(results)} relevant document(s)")

                        for idx, result in enumerate(results, 1):
                            display_search_result(result, idx)
                    else:
                        st.info("No documents found. Try adjusting your query.")

                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    st.exception(e)  # Show full traceback in development


if __name__ == "__main__":
    main()
