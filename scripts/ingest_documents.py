# -*- coding: utf-8 -*-
"""
Document ingestion.

This script processes and ingests documents into the Qdrant vector database,
generating embeddings.

"""

import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config.settings import Settings
from core.database.qdrant_client import QdrantManager
from core.database.document_store import DocumentStore
from core.services.embedding_service import EmbeddingService
from core.models.document import Document, DocumentMetadata
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("ingest_documents.log")],
    )


def load_documents_from_json(file_path: Path) -> List[Dict[str, Any]]:
    """Load documents from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both single document and array of documents
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(
            "JSON file must contain a document object or array of documents"
        )


def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for testing."""
    return [
        {
            "description": "Εκουσία δικαιοδοσία. Ανακοπή κατά κληρονομητηρίου 824ΚΠολΔ, η οποία κάνει δεκτή εν μέρει την ανακοπή ως προς τον δεύτερο θάνατο (πατέρα), ενώ για θάνατο του παππού την απέρριψε ως μη νόμιμη λόγω ότι ο θάνατος επήλθε το 1918 και εφαρμογή του βυζαντινορωμαϊκού δικαίου (άρθρ 92 ΕισΑκ).",
            "metadata": {
                "court_type": "Πρωτοδικείο",
                "case_number": "809",
                "year": "2026",
                "link": "https://www.dsanet.gr/Epikairothta/Nomologia/ΜΠρΑθ%20809.2026.htm",
                "tags": [],
            },
        },
        {
            "description": "Τηλεφωνικές κλήσεις για σκοπούς απευθείας προώθησης προϊόντων ή υπηρεσιών και για διαφημιστικούς σκοπούς. Αποζημίωση ποσού 4.000,00 ευρώ, ως χρηματική ικανοποίηση λόγω ηθικής βλάβης νομιμότοκα ήτοι από το 2022. Παράνομη επεξεργασία προσωπικών δεδομένων. Προσωρινά εκτελεστή η απόφαση.",
            "metadata": {
                "court_type": "Πρωτοδικείο",
                "case_number": "805",
                "year": "2026",
                "link": "https://www.dsanet.gr/Epikairothta/Nomologia/ΜΠρΑθηνών%20805.2026.htm",
                "tags": ["Μη ζητηθείσες τηλεφωνικές κλήσεις", "Προσωπικά δεδομένα"],
            },
        },
        {
            "description": "Όταν η Διοίκηση ενεργεί, συμμορφούμενη προς ακυρωτική απόφαση του Συμβουλίου της Επικρατείας ή του Διοικητικού Εφετείου, υποχρεούται, όχι μόνο να θεωρήσει ως ανίσχυρη και μη υφιστάμενη νομικά την ακυρωθείσα διοικητική πράξη, αλλά και να προβεί, με θετικές ενέργειες, στην αναμόρφωση της εν τω μεταξύ δημιουργηθείσας, βάσει της ακυρωθείσας πράξης, νομικής κατάστασης, ανακαλώντας ή τροποποιώντας τις σχετικές, εν τω μεταξύ, εκδοθείσες πράξεις ή εκδίδοντας άλλες με αναδρομική ισχύ, προκειμένου να αποκαταστήσει τα πράγματα στη θέση στην οποία θα ευρίσκονταν, εάν εξαρχής δεν είχε εκδοθεί η ακυρωθείσα πράξη ή δεν είχε λάβει χώρα η ακυρωθείσα παράλειψη. Εάν το Δημόσιο αρνηθεί να τοποθετήσει ορισμένο υποψήφιο σε δημόσια θέση και η άρνηση αυτή ακυρωθεί, στη συνέχεια, με απόφαση του αρμόδιου διοικητικού δικαστηρίου, ακολούθως δε η Διοίκηση, σε συμμόρφωση προς την ακυρωτική αυτή απόφαση, διορίσει τον υποψήφιο αναδρομικώς στην επίμαχη θέση, αυτός δικαιούται να ζητήσει αποζημίωση για την αποκατάσταση της ζημίας που υπέστη εκ του ότι κατά το χρονικό διάστημα από την ημερομηνία του αναδρομικού διορισμού του έως την ημερομηνία κατά την οποία ανέλαβε, πράγματι, υπηρεσία δεν εισέπραξε το σύνολο των αποδοχών που θα είχε εισπράξει, αν είχε αναλάβει, πράγματι, υπηρεσία από την ημερομηνία στην οποία ανατρέχει αναδρομικώς ο διορισμός του. Απορρίπτει την έφεση του Ελληνικού Δημοσίου.",
            "metadata": {
                "court_type": "Εφετείο",
                "case_number": "3976",
                "year": "2025",
                "link": "https://www.dsanet.gr/Epikairothta/Nomologia/ΜονΔΕφΑθ%203976.2025.htm",
                "tags": [
                    "Αδικοπρακτική ευθύνη δημοσίου",
                    "Συμμόρφωση διοίκησης σε ακυρωτικές αποφάσεις",
                    "Αναδρομική τοποθέτηση σε θέση Προϊσταμένου οργανικής μονάδος σε εφαρμογή δικαστικής αποφάσεως",
                ],
            },
        },
        {
            "description": "Επιβολή ποινής φυλάκισης 3 ετών για την κλοπή και χρηματικής ποινής 90 ημερησίων μονάδων προς 3 ευρώ εκάστη για την χρήση πλαστού. Μετατροπή της ποινής φυλάκισης σε παροχή κοινωφελούς εργασίας με εφαρμογή της ευμενέστερης διάταξης που ίσχυσε από την τέλεση της πράξης μέχρι την αμετάκλητη εκδίκασή της.",
            "metadata": {
                "court_type": "Πρωτοδικείο",
                "case_number": "1689",
                "year": "2025",
                "link": "https://www.dsanet.gr/Epikairothta/Nomologia/ΤρΕφΠλημΑθ%201689.2025.htm",
                "tags": [
                    "Κλοπή κατ' εξακολούθηση και χρήση πλαστού πιστοποιητικού κατ' εξακολούθηση"
                ],
            },
        },
        {
            "description": "Το Γενικό Πολεοδομικό Σχέδιο, το οποίο συγκροτεί το πρώτο επίπεδο πολεοδομικού σχεδιασμού, έχει, καταρχήν, ως πεδίο αναφοράς την κτηματική περιφέρεια ορισμένου Οργανισμού Τοπικής Αυτοδιοίκησης, δηλαδή υποδιαίρεση του χώρου που καθορίζεται ως πεδίο άσκησης των αρμοδιοτήτων των πρωτοβάθμιων τοπικών αρχών, κατά τα ειδικότερα οριζόμενα στο Σύνταγμα και τον νόμο. Ο ορισμός, όμως, από τον νόμο της περιφέρειας ενός πρωτοβαθμίου ΟΤΑ ως περιοχής αναφοράς του ΓΠΣ δεν έχει την αντίστροφη έννοια του ακριβούς καθορισμού ή τροποποίησης των ορίων του ΟΤΑ μέσω της έγκρισης του ΓΠΣ, διότι ο καθορισμός των ορίων και η τροποποίηση αυτών επιχειρείται, σύμφωνα με τον νόμο, από άλλα διοικητικά και δικαστικά όργανα και με άλλη ειδικώς οριζόμενη διαδικασία.",
            "metadata": {
                "court_type": "Συμβούλιο της Επικράτειας",
                "case_number": "931",
                "year": "2025",
                "link": "https://www.dsanet.gr/Epikairothta/Nomologia/ΣτΕ%20931.2025.htm",
                "tags": [
                    "Δήμοι",
                    "Γενικά Πολεοδομικά Σχέδια",
                    "Χωροταξικός/Πολεοδομικός Σχεδιασμός",
                    "Διοικητικά όρια δήμων",
                ],
            },
        },
    ]


async def ingest_documents_batch(
    documents: List[Document],
    document_store: DocumentStore,
    embedding_service: EmbeddingService,
    console: Console,
    progress: Progress,
    task_id: TaskID,
) -> Dict[str, Any]:
    """Ingest a batch of documents."""
    start_time = time.time()

    # Generate embeddings for all documents
    descriptions = [doc.description for doc in documents]
    embedding_results = await embedding_service.create_embeddings_batch(descriptions)

    # Process documents with chunking if needed
    ingestion_results = []
    total_tokens = 0
    total_chunks = 0

    for i, (document, embedding_result) in enumerate(zip(documents, embedding_results)):
        # Generate chunks and chunk embeddings if document is large
        chunk_embeddings = None
        if len(document.description.split()) > 500:  # Chunk if more than 500 words
            chunks = embedding_service.chunk_text(document.description)
            if len(chunks) > 1:
                document.chunks = chunks
                chunk_embedding_results = (
                    await embedding_service.create_embeddings_batch(chunks)
                )
                chunk_embeddings = [
                    result.embedding for result in chunk_embedding_results
                ]

        # Ingest document
        result = document_store.ingest_document(
            document=document,
            embedding=embedding_result.embedding,
            chunk_embeddings=chunk_embeddings,
        )

        ingestion_results.append(result)
        total_tokens += embedding_result.token_count
        total_chunks += result.chunk_count or 0

        # Update progress
        progress.update(task_id, advance=1)

        if result.success:
            console.print(f"  ✅ {document.metadata.court_type or document.id}")
        else:
            console.print(
                f"  ❌ {document.metadata.court_type or document.id}: {result.message}"
            )

    processing_time = time.time() - start_time
    successful = sum(1 for r in ingestion_results if r.success)

    return {
        "total": len(documents),
        "successful": successful,
        "failed": len(documents) - successful,
        "total_tokens": total_tokens,
        "total_chunks": total_chunks,
        "processing_time": processing_time,
        "results": ingestion_results,
    }


async def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest documents")
    parser.add_argument(
        "--data-path", type=Path, help="Path to documents (file or directory)"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample documents for testing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of documents to process in each batch",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Initialize console
    console = Console()

    console.print(
        Panel.fit(
            "Document Ingestion\n"
            "Processing and ingesting documents into vector database...",
            border_style="blue",
        )
    )

    try:
        # Load settings
        console.print("Loading configuration...")
        settings = Settings()

        # Initialize services
        console.print("Initializing services...")
        qdrant_manager = QdrantManager(settings)
        embedding_service = EmbeddingService(settings)
        document_store = DocumentStore(qdrant_manager, settings)

        # Health check
        if not qdrant_manager.health_check():
            console.print("❌ Cannot connect to Qdrant database!")
            console.print("Please run: python scripts/setup_database.py")
            return 1

        # Load documents
        console.print("Loading documents...")

        if args.create_sample:
            console.print("Creating sample documents...")
            raw_documents = create_sample_documents()
        elif args.data_path:
            if args.data_path.is_file():
                raw_documents = load_documents_from_json(args.data_path)
            else:
                console.print("❌ Path not found")
                return 1
        else:
            console.print("❌ Please specify --data-path or --create-sample")
            return 1

        if not raw_documents:
            console.print("⚠️  No documents found to ingest")
            return 0

        # Convert to Document objects
        documents = []
        for doc_data in raw_documents:
            # Skip documents with empty content
            if not doc_data.get("description", "").strip():
                console.print(
                    f"⚠️  Skipping document with empty content: {doc_data.get('metadata', {}).get('case_number', 'unknown')}"
                )
                continue
            metadata = DocumentMetadata(**doc_data.get("metadata", {}))
            document = Document(description=doc_data["description"], metadata=metadata)
            documents.append(document)

        console.print(f"Found {len(documents)} documents to process")

        # Estimate costs
        descriptions = [doc.description for doc in documents]
        cost_estimate = embedding_service.estimate_cost(descriptions)

        cost_table = Table(title="Processing Estimate")
        cost_table.add_column("Metric", style="cyan")
        cost_table.add_column("Value", style="green")

        cost_table.add_row("Documents", str(len(documents)))
        cost_table.add_row("Total Tokens", f"{cost_estimate['total_tokens']:,}")
        cost_table.add_row(
            "Estimated Cost", f"${cost_estimate['estimated_cost_usd']:.4f}"
        )
        cost_table.add_row("Batch Count", str(cost_estimate["batch_count"]))

        console.print(cost_table)

        # Process documents
        console.print("\nStarting document ingestion...")

        with Progress(console=console) as progress:
            task = progress.add_task("Processing documents...", total=len(documents))

            # Process in batches
            batch_size = args.batch_size
            all_results = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                console.print(
                    f"\nProcessing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}"
                )

                batch_results = await ingest_documents_batch(
                    batch, document_store, embedding_service, console, progress, task
                )
                all_results.append(batch_results)

        # Summary
        total_processed = sum(r["total"] for r in all_results)
        total_successful = sum(r["successful"] for r in all_results)
        total_failed = sum(r["failed"] for r in all_results)
        total_tokens = sum(r["total_tokens"] for r in all_results)
        total_chunks = sum(r["total_chunks"] for r in all_results)
        total_time = sum(r["processing_time"] for r in all_results)

        summary_table = Table(title="Ingestion Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Documents", str(total_processed))
        summary_table.add_row("Successful", str(total_successful))
        summary_table.add_row("Failed", str(total_failed))
        summary_table.add_row(
            "Success Rate", f"{(total_successful/total_processed)*100:.1f}%"
        )
        summary_table.add_row("Total Tokens", f"{total_tokens:,}")
        summary_table.add_row("Total Chunks", str(total_chunks))
        summary_table.add_row("Processing Time", f"{total_time:.2f}s")

        console.print(summary_table)

        if total_successful > 0:
            console.print(
                Panel.fit(
                    "Document ingestion completed!\n" "You can now search documents.",
                    border_style="green",
                )
            )

        logger.info(
            f"Ingestion completed: {total_successful}/{total_processed} documents successful"
        )
        return 0 if total_failed == 0 else 1

    except Exception as e:
        console.print(f"\n❌ Ingestion failed: {e}")
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
