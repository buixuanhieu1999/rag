from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli_logging import configure_logger
from rag_app.config import AppConfig
from rag_app.services import RagService


def log_ingest_progress(logger):
    def callback(event: dict[str, object]) -> None:
        event_name = event.get("event")
        if event_name == "fetch_start":
            logger.info(
                "Fetching knowledge-export page %s with limit %s.",
                event.get("page"),
                event.get("limit"),
            )
        elif event_name == "fetch_done":
            logger.info(
                "Fetched page %s: %s raw records.",
                event.get("page"),
                event.get("records"),
            )
        elif event_name == "page_done":
            logger.info(
                "Cleaned page %s: added %s documents; total documents %s.",
                event.get("page"),
                event.get("documents_added"),
                event.get("total_documents"),
            )
        elif event_name == "stop_empty":
            logger.info("Stopping fetch: page %s returned no records.", event.get("page"))
        elif event_name == "stop_short_page":
            logger.info(
                "Stopping fetch: page %s returned %s records, below limit %s.",
                event.get("page"),
                event.get("records"),
                event.get("limit"),
            )
        elif event_name == "documents_loaded":
            logger.info("Loaded %s source knowledge records.", event.get("documents"))
        elif event_name == "chunks_prepared":
            logger.info("Prepared %s chunks.", event.get("chunks"))
        elif event_name == "reset_start":
            logger.info("Resetting Chroma collection before indexing.")
        elif event_name == "reset_done":
            logger.info("Chroma collection reset completed.")
        elif event_name == "batch_start":
            logger.info(
                "Indexing chunk batch %s-%s of %s.",
                event.get("start"),
                event.get("end"),
                event.get("total"),
            )
        elif event_name == "batch_done":
            logger.info(
                "Indexed chunk batch %s-%s of %s.",
                event.get("start"),
                event.get("end"),
                event.get("total"),
            )

    return callback


def parse_args() -> argparse.Namespace:
    config = AppConfig()
    parser = argparse.ArgumentParser(description="Build or refresh the Chroma RAG index.")
    parser.add_argument(
        "--source",
        choices=["knowledge-export"],
        default="knowledge-export",
        help="Data source to ingest.",
    )
    parser.add_argument("--knowledge-url", default=config.knowledge_export_url)
    parser.add_argument("--knowledge-token", default=config.knowledge_export_token)
    parser.add_argument("--knowledge-os", default=config.knowledge_export_os)
    parser.add_argument("--knowledge-version", default=config.knowledge_export_version)
    parser.add_argument("--knowledge-limit", type=int, default=config.knowledge_export_limit)
    parser.add_argument("--knowledge-timeout", type=float, default=config.knowledge_export_timeout)
    parser.add_argument(
        "--knowledge-max-pages",
        type=int,
        default=None,
        help="Optional page cap for testing. Each page contains at most --knowledge-limit records.",
    )
    parser.add_argument("--chroma-dir", default=str(config.chroma_dir))
    parser.add_argument("--collection-name", default=config.collection_name)
    parser.add_argument("--embedding-provider", default=config.embedding_provider)
    parser.add_argument("--embedding-model", default=config.embedding_model)
    parser.add_argument("--embedding-host", default=config.local_ollama_host)
    parser.add_argument("--keep-alive", default=config.ollama_keep_alive)
    parser.add_argument("--chunk-size", type=int, default=config.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=config.chunk_overlap)
    parser.add_argument(
        "--log-file",
        default=str(ROOT / "logs" / "ingest.log"),
        help="Progress log file path. Use an empty value to disable file logging.",
    )
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger("ingest", args.log_file or None)
    logger.info("Starting ingest. source=%s", args.source)

    service_config = AppConfig().with_overrides(
        knowledge_export_url=args.knowledge_url,
        knowledge_export_token=args.knowledge_token,
        knowledge_export_os=args.knowledge_os,
        knowledge_export_version=args.knowledge_version,
        knowledge_export_limit=args.knowledge_limit,
        knowledge_export_timeout=args.knowledge_timeout,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        local_ollama_host=args.embedding_host,
        ollama_keep_alive=args.keep_alive,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    logger.info(
        "Opening Chroma collection '%s' at %s. reset=%s",
        args.collection_name,
        args.chroma_dir,
        args.reset,
    )
    service = RagService(service_config)
    result = service.ingest_knowledge(
        reset=args.reset,
        max_pages=args.knowledge_max_pages,
        progress_callback=log_ingest_progress(logger),
    )
    logger.info("Indexed %s chunks in collection '%s'.", result.chunks_indexed, args.collection_name)
    logger.info("Chroma collection count: %s", result.collection_count)
    logger.info("Finished ingest.")


if __name__ == "__main__":
    main()
