from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli_logging import configure_logger
from rag_app.config import AppConfig
from rag_app.data_loader import (
    build_page_content_from_knowledge_record,
    clean_knowledge_export_record,
    fetch_knowledge_export_page,
)


def parse_args() -> argparse.Namespace:
    config = AppConfig()
    parser = argparse.ArgumentParser(
        description="Export OMS knowledge-export rows to clean JSON without indexing."
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "exports" / "knowledge_export_sample.json"),
        help="JSON output path.",
    )
    parser.add_argument("--knowledge-url", default=config.knowledge_export_url)
    parser.add_argument("--knowledge-token", default=config.knowledge_export_token)
    parser.add_argument("--knowledge-os", default=config.knowledge_export_os)
    parser.add_argument("--knowledge-version", default=config.knowledge_export_version)
    parser.add_argument(
        "--knowledge-limit",
        type=int,
        default=5,
        help="Records per API page. The API caps this at 30.",
    )
    parser.add_argument("--knowledge-timeout", type=float, default=config.knowledge_export_timeout)
    parser.add_argument(
        "--knowledge-max-pages",
        type=int,
        default=1,
        help="Number of API pages to export unless --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all pages until the API returns an empty or short page.",
    )
    parser.add_argument(
        "--knowledge-start-page",
        type=int,
        default=0,
        help="First API page number to fetch.",
    )
    parser.add_argument(
        "--log-file",
        default=str(ROOT / "logs" / "export_knowledge.log"),
        help="Progress log file path. Use an empty value to disable file logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger("export_knowledge", args.log_file or None)
    limit = min(max(1, int(args.knowledge_limit)), 30)
    max_pages = None if args.all else max(1, int(args.knowledge_max_pages))
    start_page = max(0, int(args.knowledge_start_page))

    records: list[dict[str, object]] = []
    fetched_pages = 0
    logger.info(
        "Starting knowledge export. start_page=%s limit=%s max_pages=%s output=%s",
        start_page,
        limit,
        "all" if max_pages is None else max_pages,
        args.output,
    )

    page_offset = 0
    while max_pages is None or page_offset < max_pages:
        page = start_page + page_offset
        logger.info("Fetching knowledge-export page %s with limit %s.", page, limit)
        api_records = fetch_knowledge_export_page(
            url=args.knowledge_url,
            token=args.knowledge_token,
            limit=limit,
            skip=page,
            timeout=args.knowledge_timeout,
            access_os=args.knowledge_os,
            access_version=args.knowledge_version,
        )
        logger.info("Fetched page %s: %s raw records.", page, len(api_records))
        if not api_records:
            logger.info("Stopping export: page %s returned no records.", page)
            break

        fetched_pages += 1
        before_count = len(records)
        for api_index, api_record in enumerate(api_records):
            clean_record = clean_knowledge_export_record(api_record)
            clean_record.update(
                {
                    "page_content": build_page_content_from_knowledge_record(api_record),
                    "source_page": page,
                    "source_page_index": api_index,
                }
            )
            records.append(clean_record)
        logger.info(
            "Cleaned page %s: added %s records; total records %s.",
            page,
            len(records) - before_count,
            len(records),
        )

        if len(api_records) < limit:
            logger.info(
                "Stopping export: page %s returned %s records, below limit %s.",
                page,
                len(api_records),
                limit,
            )
            break
        page_offset += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_url": args.knowledge_url,
        "limit": limit,
        "start_page": start_page,
        "fetched_pages": fetched_pages,
        "count": len(records),
        "data": records,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Exported %s records from %s page(s).", len(records), fetched_pages)
    logger.info("Wrote %s", output_path)
    logger.info("Finished export.")


if __name__ == "__main__":
    main()
