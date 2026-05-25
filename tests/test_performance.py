# -*- coding: utf-8 -*-
"""
This script provides performance analysis and
comparative analysis of different configurations.

"""

import sys
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config.settings import Settings
from core.database.qdrant_client import QdrantManager
from core.services.embedding_service import EmbeddingService
from core.services.search_engine import HybridSearchEngine

# Initialize system components
settings = Settings()
qdrant_manager = QdrantManager(settings)
embedding_service = EmbeddingService(settings)
search_engine = HybridSearchEngine(qdrant_manager, embedding_service, settings)

# Define test queries for analysis
# test_queries = [
#     "αθέτηση συμβατικών υποχρεώσεων",
#     "αθέτηση υποχρέωσης αποζημίωσης για πλημμέλημα",
#     "απρόσεκτη οδήγηση που προκάλεσε σωματική βλάβη",
#     "ένσταση δυσφήμισης κατά ιδιώτη",
#     "κατοχή ναρκωτικών ουσιών για προσωπική χρήση",
#     "απάτη μέσω ηλεκτρονικών μέσων",
#     "ακύρωση δημοπρασίας δημόσιων έργων λόγω κακοτεχνίας",
#     "αποζημίωση για υπερωριακή εργασία"
# ]

test_queries = [
    "κατοχή ναρκωτικών ουσιών για προσωπική χρήση",
    "απάτη μέσω ηλεκτρονικών μέσων",
    "απρόσεκτη οδήγηση που προκάλεσε σωματική βλάβη",
]

print(f"Prepared {len(test_queries)} test queries for analysis")


# Perform search performance analysis
async def analyze_search_performance(
    queries: List[str], weight_configs: List[Dict[str, float]]
):
    results = []

    for config in weight_configs:
        config_name = f"V{config['vector']:.1f}_K{config['keyword']:.1f}"
        print(f"Testing configuration: {config_name}")

        for query in queries:
            start_time = time.time()

            search_results = await search_engine.search(
                query=query,
                limit=10,
                vector_weight=config["vector"],
                keyword_weight=config["keyword"],
            )

            search_time = time.time() - start_time

            # Calculate metrics
            avg_score = (
                np.mean([r.combined_score for r in search_results])
                if search_results
                else 0
            )
            max_score = (
                max([r.combined_score for r in search_results]) if search_results else 0
            )
            result_count = len(search_results)

            results.append(
                {
                    "config": config_name,
                    "vector_weight": config["vector"],
                    "keyword_weight": config["keyword"],
                    "query": query,
                    "search_time": search_time,
                    "result_count": result_count,
                    "avg_score": avg_score,
                    "max_score": max_score,
                }
            )

    return pd.DataFrame(results)


# Generate system recommendations based on analysis
def generate_recommendations(performance_df):
    recommendations = []

    # Performance recommendations
    best_config = performance_df.loc[performance_df["avg_score"].idxmax()]
    recommendations.append(
        f"Optimal search configuration: Vector weight {best_config['vector_weight']:.1f}, "
        f"Keyword weight {best_config['keyword_weight']:.1f}"
    )

    avg_search_time = performance_df["search_time"].mean()
    if avg_search_time > 1.0:
        recommendations.append(
            f"Consider optimizing search performance (current avg: {avg_search_time:.3f}s)"
        )

    return recommendations


# Define weight configurations to test
weight_configs = [
    {"vector": 1.0, "keyword": 0.0},  # Pure vector search
    {"vector": 0.8, "keyword": 0.2},  # Vector-heavy
    {"vector": 0.7, "keyword": 0.3},  # Default
    {"vector": 0.5, "keyword": 0.5},  # Balanced
    {"vector": 0.3, "keyword": 0.7},  # Keyword-heavy
    {"vector": 0.0, "keyword": 1.0},  # Pure keyword search
]

# Run analysis
performance_df = asyncio.run(
    analyze_search_performance(test_queries[:5], weight_configs)
)
print(f"✅ Completed performance analysis with {len(performance_df)} data points")

# Visualize search performance
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Average search time by configuration
time_by_config = performance_df.groupby("config")["search_time"].mean()
axes[0, 0].bar(time_by_config.index, time_by_config.values)
axes[0, 0].set_title("Average Search Time by Configuration")
axes[0, 0].set_ylabel("Time (seconds)")
axes[0, 0].tick_params(axis="x", rotation=45)

# Average score by configuration
score_by_config = performance_df.groupby("config")["avg_score"].mean()
axes[0, 1].bar(score_by_config.index, score_by_config.values)
axes[0, 1].set_title("Average Search Score by Configuration")
axes[0, 1].set_ylabel("Average Score")
axes[0, 1].tick_params(axis="x", rotation=45)

# Result count by configuration
count_by_config = performance_df.groupby("config")["result_count"].mean()
axes[1, 0].bar(count_by_config.index, count_by_config.values)
axes[1, 0].set_title("Average Result Count by Configuration")
axes[1, 0].set_ylabel("Result Count")
axes[1, 0].tick_params(axis="x", rotation=45)

# Score distribution
axes[1, 1].boxplot(
    [
        performance_df[performance_df["config"] == config]["avg_score"].values
        for config in performance_df["config"].unique()
    ],
    labels=performance_df["config"].unique(),
)
axes[1, 1].set_title("Score Distribution by Configuration")
axes[1, 1].set_ylabel("Average Score")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

recommendations = generate_recommendations(performance_df)

# Export analysis results
output_dir = Path("data/processed")
output_dir.mkdir(exist_ok=True)

# Save performance analysis
performance_df.to_csv(output_dir / "search_performance_analysis.csv", index=False)
print(
    f"Saved search performance analysis to {output_dir / 'search_performance_analysis.csv'}"
)
