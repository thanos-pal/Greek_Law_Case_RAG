# -*- coding: utf-8 -*-
"""
RAGAS Evaluation Script for Greek Law Case RAG System

This script evaluates the retrieval performance using RAGAS metrics:
- Context Precision: How many of the retrieved contexts are relevant
- Context Recall: How many of the relevant contexts were retrieved
- Context Relevance: How relevant the retrieved contexts are to the query

Evaluation Dataset Format (JSON):
[
  {
    "query": "κατοχή ναρκωτικών ουσιών για προσωπική χρήση",
    "relevant_cases": ["76/2015", "142/2018", "89/2020"],
    "expected_top_k": 5
  },
  ...
]

Usage:
    python tests/ragas_evaluation.py                    # Uses default dataset
    python tests/ragas_evaluation.py --dataset my_eval.json  # Uses custom dataset
    python tests/ragas_evaluation.py --create-template        # Creates template dataset
"""

import sys
import asyncio
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config.settings import Settings
from core.database.qdrant_client import QdrantManager
from core.services.embedding_service import EmbeddingService
from core.services.search_engine import HybridSearchEngine
from core.models.search_result import SearchResult

# RAGAS imports (new API)
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import ContextPrecision, ContextRecall, ContextRelevance
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI
from instructor import patch as instructor_patch, Mode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """Evaluates RAG retrieval performance using RAGAS metrics."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.qdrant_manager = QdrantManager(settings)
        self.embedding_service = EmbeddingService(settings)
        self.search_engine = HybridSearchEngine(
            self.qdrant_manager,
            self.embedding_service,
            settings
        )

        # Initialize RAGAS LLM and embeddings using new API
        # Patch the OpenAI client with instructor
        openai_client = OpenAI(api_key=settings.openai_api_key)
        patched_client = instructor_patch(openai_client, mode=Mode.JSON)
        self.llm = llm_factory("gpt-4o-mini", client=patched_client)

        # Embeddings - use unpatched client
        embedding_client = OpenAI(api_key=settings.openai_api_key)
        self.embeddings = OpenAIEmbeddings(
            client=embedding_client,
            model="text-embedding-3-small"
        )

        # Cache for case lookup by "case_number/year"
        self._case_lookup: Dict[str, Dict[str, Any]] = {}

    def load_case_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load case data from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def build_case_lookup(self, case_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build a lookup dictionary mapping 'case_number/year' to case data."""
        lookup = {}
        for case in case_data:
            metadata = case.get("metadata", {})
            case_number = metadata.get("case_number", "")
            year = metadata.get("year", "")
            if case_number and year:
                key = f"{case_number}/{year}"
                lookup[key] = case
        return lookup

    def load_evaluation_dataset(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Load evaluation dataset from JSON file."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Evaluation dataset must be a JSON array")

        # Validate each entry
        for i, item in enumerate(data):
            if "query" not in item:
                raise ValueError(f"Entry {i}: missing 'query' field")
            if "relevant_cases" not in item:
                raise ValueError(f"Entry {i}: missing 'relevant_cases' field")
            if not isinstance(item["relevant_cases"], list):
                raise ValueError(f"Entry {i}: 'relevant_cases' must be a list")
            if "expected_top_k" not in item:
                item["expected_top_k"] = 5  # default

        logger.info(f"Loaded {len(data)} evaluation queries from {dataset_path}")
        return data

    def create_template_dataset(self, output_path: Path, num_examples: int = 5):
        """Create a template evaluation dataset for the user to fill in."""
        template = [
            {
                "query": "κατοχή ναρκωτικών ουσιών για προσωπική χρήση",
                "relevant_cases": ["76/2015", "142/2018", "89/2020"],
                "expected_top_k": 5
            },
            {
                "query": "απάτη μέσω ηλεκτρονικών μέσων",
                "relevant_cases": ["234/2019", "567/2020", "123/2021"],
                "expected_top_k": 5
            },
            {
                "query": "απρόσεκτη οδήγηση που προκάλεσε σωματική βλάβη",
                "relevant_cases": ["45/2017", "289/2018", "678/2019"],
                "expected_top_k": 5
            },
            {
                "query": "κλοπή μεσών κοινής χρήσης",
                "relevant_cases": ["321/2016", "654/2017", "987/2018"],
                "expected_top_k": 5
            },
            {
                "query": "δυσφήμιση και ύβρις μέσω διαδικτύου",
                "relevant_cases": ["111/2020", "222/2021", "333/2022"],
                "expected_top_k": 5
            },
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(template[:num_examples], f, ensure_ascii=False, indent=2)

        logger.info(f"Template dataset created at {output_path}")
        print(f"Template created at {output_path}")
        print("Edit this file with your actual case numbers (format: 'case_number/year')")

    def resolve_relevant_cases(
        self,
        ground_truth: List[Dict[str, Any]],
        case_lookup: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Resolve 'case_number/year' references to full case descriptions."""
        resolved = []

        for gt in ground_truth:
            relevant_descriptions = []
            missing_cases = []

            for case_ref in gt.get("relevant_cases", []):
                if case_ref in case_lookup:
                    relevant_descriptions.append(case_lookup[case_ref]["description"])
                else:
                    missing_cases.append(case_ref)

            if missing_cases:
                logger.warning(f"Query '{gt['query'][:50]}...': Could not find cases: {missing_cases}")

            resolved.append({
                "query": gt["query"],
                "reference_contexts": relevant_descriptions,
                "expected_top_k": gt.get("expected_top_k", 5),
                "original_case_refs": gt.get("relevant_cases", [])
            })

        return resolved

    async def retrieve_contexts(
        self,
        query: str,
        top_k: int = 5
    ) -> List[str]:
        """Retrieve top-k contexts for a query using the search engine."""
        try:
            results = await self.search_engine.search(
                query=query,
                limit=top_k,
                vector_weight=0.7,
                keyword_weight=0.3
            )
            # Extract descriptions as contexts
            contexts = [result.description for result in results]
            return contexts
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return []

    def create_evaluation_dataset(
        self,
        resolved_ground_truth: List[Dict[str, Any]],
        retrieved_contexts: List[List[str]]
    ) -> EvaluationDataset:
        """Create RAGAS EvaluationDataset from ground truth and retrieved contexts."""
        samples = []

        for i, (gt, contexts) in enumerate(zip(resolved_ground_truth, retrieved_contexts)):
            sample = SingleTurnSample(
                user_input=gt["query"],
                retrieved_contexts=contexts,
                # Reference contexts are the known relevant cases
                reference_contexts=gt.get("reference_contexts", [])
            )
            samples.append(sample)

        return EvaluationDataset(samples=samples)

    async def run_evaluation(
        self,
        evaluation_dataset: List[Dict[str, Any]],
        top_k: int = 5,
        case_data_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run complete RAGAS evaluation.

        Args:
            evaluation_dataset: List of dicts with query, relevant_cases, expected_top_k
            top_k: Number of top documents to retrieve (default 5)
            case_data_path: Path to case data JSON file

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting RAGAS evaluation with {len(evaluation_dataset)} queries, top_k={top_k}")

        # Load case data
        if case_data_path is None:
            case_data_path = project_root / "data" / "case_data.json"

        case_data = self.load_case_data(case_data_path)
        logger.info(f"Loaded {len(case_data)} cases from {case_data_path}")

        # Build case lookup
        self._case_lookup = self.build_case_lookup(case_data)
        logger.info(f"Built case lookup with {len(self._case_lookup)} entries")

        # Resolve relevant cases to descriptions
        resolved_ground_truth = self.resolve_relevant_cases(evaluation_dataset, self._case_lookup)

        # Retrieve contexts for each query
        logger.info("Retrieving contexts for each query...")
        retrieved_contexts = []
        queries = []

        for gt in resolved_ground_truth:
            query = gt["query"]
            queries.append(query)
            contexts = await self.retrieve_contexts(query, top_k=top_k)
            retrieved_contexts.append(contexts)
            logger.info(f"Query: {query[:50]}... -> Retrieved {len(contexts)} contexts")

        # Create evaluation dataset
        eval_dataset = self.create_evaluation_dataset(resolved_ground_truth, retrieved_contexts)

        # Initialize RAGAS metrics
        context_precision = ContextPrecision(llm=self.llm)
        context_recall = ContextRecall(llm=self.llm)
        context_relevance = ContextRelevance(llm=self.llm)

        # Run evaluation
        logger.info("Running RAGAS evaluation...")

        # Evaluate each metric
        precision_scores = []
        recall_scores = []
        relevance_scores = []

        for i, sample in enumerate(eval_dataset):
            try:
                # Context Precision - use reference contexts
                reference_contexts = sample.reference_contexts

                if reference_contexts:
                    # Use the first reference context for precision calculation
                    ref = reference_contexts[0]
                    precision_result = await context_precision.ascore(
                        user_input=sample.user_input,
                        reference=ref,
                        retrieved_contexts=sample.retrieved_contexts
                    )
                    precision_scores.append(precision_result.value)

                    # Context Recall - use reference contexts
                    recall_result = await context_recall.ascore(
                        user_input=sample.user_input,
                        reference=ref,
                        retrieved_contexts=sample.retrieved_contexts
                    )
                    recall_scores.append(recall_result.value)

                # Context Relevance
                relevance_result = await context_relevance.ascore(
                    user_input=sample.user_input,
                    retrieved_contexts=sample.retrieved_contexts
                )
                relevance_scores.append(relevance_result.value)

                logger.info(f"Sample {i+1}: Precision={precision_scores[-1]:.4f}, "
                          f"Recall={recall_scores[-1] if recall_scores else 'N/A':.4f}, "
                          f"Relevance={relevance_scores[-1]:.4f}")

            except Exception as e:
                logger.error(f"Evaluation failed for sample {i}: {e}")
                precision_scores.append(0.0)
                relevance_scores.append(0.0)
                if len(recall_scores) < len(precision_scores):
                    recall_scores.append(0.0)

        # Calculate averages
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

        results = {
            "num_queries": len(evaluation_dataset),
            "top_k": top_k,
            "context_precision": {
                "mean": avg_precision,
                "scores": precision_scores
            },
            "context_recall": {
                "mean": avg_recall,
                "scores": recall_scores
            },
            "context_relevance": {
                "mean": avg_relevance,
                "scores": relevance_scores
            },
            "queries": queries,
            "ground_truth": evaluation_dataset,
            "resolved_ground_truth": resolved_ground_truth,
            "retrieved_contexts": retrieved_contexts
        }

        logger.info(f"Evaluation complete!")
        logger.info(f"Context Precision: {avg_precision:.4f}")
        logger.info(f"Context Recall: {avg_recall:.4f}")
        logger.info(f"Context Relevance: {avg_relevance:.4f}")

        return results

    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save evaluation results to JSON file."""
        # Convert any non-serializable objects
        serializable_results = {
            "num_queries": results["num_queries"],
            "top_k": results["top_k"],
            "context_precision": results["context_precision"],
            "context_recall": results["context_recall"],
            "context_relevance": results["context_relevance"],
            "queries": results["queries"],
            "ground_truth": results["ground_truth"],
            "retrieved_contexts_count": [len(ctx) for ctx in results["retrieved_contexts"]],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_path}")


async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for Greek Law Case RAG")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to evaluation dataset JSON file"
    )
    parser.add_argument(
        "--create-template",
        type=str,
        help="Create a template evaluation dataset at the specified path"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--case-data",
        type=str,
        help="Path to case data JSON file (default: data/case_data.json)"
    )

    args = parser.parse_args()

    # Load settings
    settings = Settings()

    # Initialize evaluator
    evaluator = RAGASEvaluator(settings)

    # Handle template creation
    if args.create_template:
        output_path = Path(args.create_template)
        evaluator.create_template_dataset(output_path)
        return

    # Load evaluation dataset
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            sys.exit(1)
        evaluation_dataset = evaluator.load_evaluation_dataset(dataset_path)
    else:
        # Default: use fallback (for backward compatibility)
        logger.warning("No dataset specified, using built-in fallback queries")
        case_data_path = Path(args.case_data) if args.case_data else project_root / "data" / "case_data.json"
        case_data = evaluator.load_case_data(case_data_path)
        # Create minimal fallback
        evaluation_dataset = [
            {
                "query": "κατοχή ναρκωτικών ουσιών για προσωπική χρήση",
                "relevant_cases": [],  # Will be resolved from first few cases
                "expected_top_k": 5
            },
        ]
        # Try to use first 3 cases as fallback
        for i, case in enumerate(case_data[:3]):
            metadata = case.get("metadata", {})
            case_number = metadata.get("case_number", "")
            year = metadata.get("year", "")
            if case_number and year:
                evaluation_dataset[0]["relevant_cases"].append(f"{case_number}/{year}")

    # Run evaluation
    case_data_path = Path(args.case_data) if args.case_data else None
    results = await evaluator.run_evaluation(
        evaluation_dataset=evaluation_dataset,
        top_k=args.top_k,
        case_data_path=case_data_path
    )

    # Save results
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "ragas_evaluation_results.json"
    evaluator.save_results(results, output_path)

    # Print summary
    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"Number of queries: {results['num_queries']}")
    print(f"Top-K retrieved: {results['top_k']}")
    print(f"\nContext Precision:  {results['context_precision']['mean']:.4f}")
    print(f"Context Recall:     {results['context_recall']['mean']:.4f}")
    print(f"Context Relevance:  {results['context_relevance']['mean']:.4f}")
    print(f"\nResults saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())