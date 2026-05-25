"""
Database setup.

This script initializes the Qdrant database, creates collections with optimized
configurations, and sets up necessary indexes for efficient searching.

"""

import sys
import logging
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config.settings import Settings
from core.database.qdrant_client import QdrantManager
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("setup_database.log")],
    )


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Qdrant database")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing collection (WARNING: This will delete all data)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check database status without making changes",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    console = Console()

    console.print(
        Panel.fit(
            "Qdrant Database Setup\n" "Initializing vector database and collections...",
            border_style="blue",
        )
    )

    try:
        # Load settings
        console.print("Loading configuration...")
        settings = Settings()

        # Display configuration
        config_table = Table(title="Database Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Qdrant Host", settings.qdrant_host)
        config_table.add_row("Qdrant Port", str(settings.qdrant_port))
        config_table.add_row("Collection Name", settings.qdrant_collection_name)
        config_table.add_row("Environment", settings.environment)

        console.print(config_table)

        # Initialize Qdrant manager
        console.print("\nConnecting to Qdrant...")
        qdrant_manager = QdrantManager(settings)

        # Health check
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Checking Qdrant connection...", total=None)

            if not qdrant_manager.health_check():
                console.print("❌ Failed to connect to Qdrant!")
                console.print("Please ensure Qdrant is running and accessible.")
                return 1

            progress.update(task, description="✅ Qdrant connection successful!")

        # Check if collection exists
        collection_info = qdrant_manager.get_collection_info()
        collection_exists = collection_info is not None

        if args.check_only:
            status_table = Table(title="Database Status")
            status_table.add_column("Component", style="cyan")
            status_table.add_column("Status", style="green")

            status_table.add_row("Qdrant Connection", "✅ Connected")
            status_table.add_row(
                "Collection",
                (
                    f"✅ Exists ({collection_info.indexed_vectors_count} vectors)"
                    if collection_exists
                    else "❌ Not found"
                ),
            )

            console.print(status_table)
            return 0

        # Handle collection reset
        if args.reset and collection_exists:
            console.print("\n⚠️  Reset requested - this will delete all existing data!")
            confirm = console.input("Are you sure? Type 'yes' to confirm: ")

            if confirm.lower() == "yes":
                console.print("Deleting existing collection...")
                if qdrant_manager.delete_collection():
                    console.print("✅ Collection deleted successfully")
                    collection_exists = False
                else:
                    console.print("❌ Failed to delete collection")
                    return 1
            else:
                console.print("Reset cancelled.")
                return 0

        # Create collection if it doesn't exist
        if not collection_exists:
            console.print("\nCreating new collection...")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Creating collection with optimized settings...", total=None
                )

                qdrant_manager.initialize_collection()

                progress.update(task, description="✅ Collection created successfully!")

            # Create payload indexes for common metadata fields
            console.print("Creating payload indexes...")

            indexes_to_create = [
                ("metadata.court_type", "keyword"),
                ("metadata.case_number", "keyword"),
                ("metadata.year", "keyword"),
                ("metadata.link", "keyword"),
                ("metadata.created_at", "datetime"),
                ("document_type", "keyword"),
                ("parent_document_id", "keyword"),
            ]

            for field_name, field_type in indexes_to_create:
                try:
                    qdrant_manager.create_payload_index(field_name, field_type)
                    console.print(f"  ✅ Created index for {field_name}")
                except Exception as e:
                    console.print(f"  ⚠️  Failed to create index for {field_name}: {e}")

        else:
            console.print(
                f"\n✅ Collection already exists with {collection_info.indexed_vectors_count} vectors"
            )

        # Final status check
        final_info = qdrant_manager.get_collection_info()

        summary_table = Table(title="Setup Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Collection Name", settings.qdrant_collection_name)
        summary_table.add_row("Vector Count", str(final_info.indexed_vectors_count))
        summary_table.add_row("Vector Size", str(final_info.config.params.vectors.size))
        summary_table.add_row(
            "Distance Metric", str(final_info.config.params.vectors.distance)
        )
        summary_table.add_row("Status", "✅ Ready for use")

        console.print(summary_table)

        console.print(
            Panel.fit(
                "Database setup completed successfully!\n"
                "You can now start ingesting documents.",
                border_style="green",
            )
        )

        logger.info("Database setup completed successfully")
        return 0

    except Exception as e:
        console.print(f"\n❌ Setup failed: {e}")
        logger.error(f"Setup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
