#!/usr/bin/env python3
"""INDmoney Google Play review extractor for Layer 1."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import pathlib
import time
from typing import Dict, List, Tuple

import requests
from google_play_scraper import Sort, reviews
from tenacity import retry, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger("layer1.extractor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect structured Google Play reviews for INDmoney."
    )
    parser.add_argument(
        "--app-id",
        default="in.indwealth",
        help="Google Play application id (default: in.indwealth)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language/locale code for Play Store (default: en)",
    )
    parser.add_argument(
        "--country",
        default="in",
        help="Country/region code for Play Store (default: in)",
    )
    parser.add_argument(
        "--count-per-rating",
        type=int,
        default=30,
        help="How many reviews to fetch for each star rating bucket.",
    )
    parser.add_argument(
        "--output-dir",
        default="layer-1-data-extraction/layer-1",
        help="Directory where extracted data will be written.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Retry attempts for network calls.",
    )
    parser.add_argument(
        "--sleep-between-retries",
        type=float,
        default=0.5,
        help="Base sleep (seconds) between retries.",
    )
    return parser.parse_args()


def hashed_pseudo_id(text: str, timestamp: dt.datetime) -> str:
    normalized_text = (text or "").strip()
    ts_str = timestamp.isoformat()
    payload = f"{normalized_text}::{ts_str}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_run_folder(base_dir: pathlib.Path) -> pathlib.Path:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def save_jsonl(path: pathlib.Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_reviews_for_rating(
    app_id: str,
    rating: int,
    lang: str,
    country: str,
    count_target: int,
    max_attempts: int,
    sleep_between: float,
    locale: str,
    platform: str,
    seen_ids: set[str],
) -> Tuple[List[Dict], Dict]:
    attempts = 0
    collected: List[Dict] = []
    continuation_token = None
    stats = {"rating": rating, "batches": 0, "http_status": []}

    while len(collected) < count_target:
        attempts += 1
        LOGGER.debug(
            "Fetching rating=%s batch #%s (current total=%s)",
            rating,
            attempts,
            len(collected),
        )
        batch = fetch_reviews_page(
            app_id=app_id,
            lang=lang,
            country=country,
            rating=rating,
            continuation_token=continuation_token,
            max_attempts=max_attempts,
            sleep_between=sleep_between,
        )
        continuation_token = batch["next_token"]
        stats["batches"] += 1
        stats["http_status"].append(batch["http_status"])

        for raw_review in batch["rows"]:
            text = (raw_review.get("content") or "").strip()
            if len(text) < 10:
                continue
            normalized = normalize(raw_review, locale=locale, platform=platform)
            pseudo_id = normalized["pseudo_id"]
            if pseudo_id in seen_ids:
                continue
            seen_ids.add(pseudo_id)
            collected.append(normalized)
            if len(collected) >= count_target:
                break

        if not continuation_token or len(collected) >= count_target:
            break

    if len(collected) < count_target:
        LOGGER.warning(
            "Only gathered %s/%s reviews for rating %s after exhausting pages.",
            len(collected),
            count_target,
            rating,
        )

    return collected[:count_target], stats


@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=8), stop=stop_after_attempt(5))
def _reviews_request(
    app_id: str,
    lang: str,
    country: str,
    rating: int,
    continuation_token: Dict | None,
) -> Tuple[List[Dict], Dict | None]:
    return reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=200,
        filter_score_with=rating,
        continuation_token=continuation_token,
    )


def fetch_reviews_page(
    app_id: str,
    lang: str,
    country: str,
    rating: int,
    continuation_token: Dict | None,
    max_attempts: int,
    sleep_between: float,
) -> Dict:
    retries = 0
    start = time.time()
    while True:
        try:
            rows, next_token = _reviews_request(
                app_id=app_id,
                lang=lang,
                country=country,
                rating=rating,
                continuation_token=continuation_token,
            )
            duration = time.time() - start
            LOGGER.info(
                "Fetched %s reviews for %s-star bucket in %.2fs",
                len(rows),
                rating,
                duration,
            )
            return {
                "rows": rows,
                "next_token": next_token,
                "http_status": 200,
                "duration_seconds": duration,
            }
        except Exception as exc:  # pylint: disable=broad-except
            retries += 1
            if retries >= max_attempts:
                raise
            wait_time = sleep_between * (2 ** (retries - 1))
            LOGGER.warning(
                "Retrying fetch for rating %s after error: %s (retry %s/%s)",
                rating,
                exc,
                retries,
                max_attempts,
            )
            time.sleep(wait_time)


def normalize(review: Dict, locale: str, platform: str) -> Dict:
    timestamp = review["at"]
    text = (review.get("content") or "").strip()
    return {
        "pseudo_id": hashed_pseudo_id(text, timestamp),
        "review_id": review.get("reviewId"),
        "text": text,
        "rating": review.get("score"),
        "timestamp": timestamp.isoformat(),
        "platform": platform,
        "locale": locale,
        "app_version": review.get("appVersion"),
        "helpful_count": review.get("thumbsUpCount"),
        "source": "google_play_popup",
    }


@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
def fetch_raw_html(app_id: str, lang: str, country: str) -> str:
    url = (
        "https://play.google.com/store/apps/details"
        f"?id={app_id}&hl={lang}&gl={country}"
    )
    headers = {
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    base_dir = pathlib.Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_folder(base_dir)

    locale = f"{args.lang}-{args.country.upper()}"
    platform_name = "google_play"
    LOGGER.info("Starting extraction for %s into %s", args.app_id, run_dir)

    normalized_rows: List[Dict] = []
    seen_ids: set[str] = set()
    per_bucket_stats: List[Dict] = []
    for rating in range(1, 6):
        rows, stats = collect_reviews_for_rating(
            app_id=args.app_id,
            rating=rating,
            lang=args.lang,
            country=args.country,
            count_target=args.count_per_rating,
            max_attempts=args.max_attempts,
            sleep_between=args.sleep_between_retries,
            locale=locale,
            platform=platform_name,
            seen_ids=seen_ids,
        )
        per_bucket_stats.append({**stats, "fetched": len(rows)})
        normalized_rows.extend(rows)

    normalized_path = run_dir / "normalized.jsonl"
    save_jsonl(normalized_path, normalized_rows)
    LOGGER.info("Wrote %s normalized rows to %s", len(normalized_rows), normalized_path)

    metadata = {
        "app_id": args.app_id,
        "fetched_at": dt.datetime.utcnow().isoformat(),
        "count_per_rating": args.count_per_rating,
        "locale": locale,
        "platform": platform_name,
        "stats": per_bucket_stats,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    try:
        raw_html = fetch_raw_html(args.app_id, args.lang, args.country)
        (run_dir / "raw.html").write_text(raw_html, encoding="utf-8")
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Unable to fetch raw HTML snapshot: %s", exc)

    LOGGER.info("Extraction complete. Output available in %s", run_dir)


if __name__ == "__main__":
    main()

