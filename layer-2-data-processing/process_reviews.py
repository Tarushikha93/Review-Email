#!/usr/bin/env python3
"""Layer 2 processing: clean, segment, score, and persist reviews."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import spacy
from langdetect import LangDetectException, detect_langs

LOGGER = logging.getLogger("layer2.processor")


@dataclass
class ReviewSegment:
    segment_id: str
    review_id: str
    pseudo_id: str
    rating: int
    star_bucket: str
    platform: str
    locale: str
    country: str | None
    submitted_at: dt.datetime
    submission_week: str
    text: str
    snapshot_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process normalized reviews for Layer 2")
    parser.add_argument(
        "--input-dir",
        default="layer-1-data-extraction/layer-1",
        help="Directory containing Layer 1 run folders.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Specific run folder name (e.g., run-20251125T155927Z). Defaults to latest.",
    )
    parser.add_argument(
        "--output-dir",
        default="layer-2-data-processing/output",
        help="Destination root for processed artifacts.",
    )
    parser.add_argument(
        "--lang-threshold",
        type=float,
        default=0.8,
        help="Minimum probability for language detection to keep a segment.",
    )
    parser.add_argument(
        "--min-segment-chars",
        type=int,
        default=30,
        help="Minimum length for sentence segments.",
    )
    return parser.parse_args()


def latest_run_folder(base_dir: Path) -> Path:
    runs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    )
    if not runs:
        raise FileNotFoundError(f"No run folders found under {base_dir}")
    return runs[-1]


def load_reviews(run_dir: Path) -> List[Dict]:
    normalized_path = run_dir / "normalized.jsonl"
    if not normalized_path.exists():
        raise FileNotFoundError(f"Missing normalized.jsonl in {run_dir}")
    rows: List[Dict] = []
    with normalized_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    # Remove any lingering HTML tags
    cleaned_chars: List[str] = []
    in_tag = False
    for char in text:
        if char == "<":
            in_tag = True
            continue
        if char == ">":
            in_tag = False
            continue
        if not in_tag:
            cleaned_chars.append(char)
    cleaned = "".join(cleaned_chars)
    normalized = " ".join(cleaned.split())
    return normalized.strip()


def detect_language(text: str) -> tuple[str, float]:
    if not text:
        return "unknown", 0.0
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return "unknown", 0.0
    best = max(candidates, key=lambda lang: lang.prob, default=None)
    if best is None:
        return "unknown", 0.0
    return best.lang, best.prob


def load_snapshot_date(run_dir: Path) -> str:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fetched_at = metadata.get("fetched_at")
        if fetched_at:
            return fetched_at.split("T")[0]
    # fallback to run directory timestamp if metadata missing
    return run_dir.name.replace("run-", "")[:8]


def week_bucket(ts: dt.datetime) -> str:
    monday = ts - dt.timedelta(days=ts.weekday())
    return monday.strftime("%Y-%m-%d")


def segment_reviews(
    reviews: List[Dict],
    nlp,
    lang_threshold: float,
    min_segment_chars: int,
    snapshot_date: str,
) -> tuple[List[ReviewSegment], Dict]:
    segments: List[ReviewSegment] = []
    issues = {
        "empty_reviews": 0,
        "non_english": 0,
        "no_segments": 0,
    }

    for review in reviews:
        raw_text = review.get("text") or ""
        cleaned = clean_text(raw_text)
        if not cleaned:
            issues["empty_reviews"] += 1
            continue

        lang_code, lang_prob = detect_language(cleaned)
        if lang_code != "en" or lang_prob < lang_threshold:
            issues["non_english"] += 1
            continue

        doc = nlp(cleaned)
        review_segments = [
            sent.text.strip()
            for sent in doc.sents
            if len(sent.text.strip()) >= min_segment_chars
        ]
        if not review_segments:
            issues["no_segments"] += 1
            continue

        submitted_at = dt.datetime.fromisoformat(review["timestamp"])
        platform = review.get("platform", "unknown")
        locale = review.get("locale", "en")
        country = locale.split("-")[1] if "-" in locale else None
        bucket = f"{review.get('rating', 0)}-star"
        for idx, segment_text in enumerate(review_segments, start=1):
            segments.append(
                ReviewSegment(
                    segment_id=f"{review['review_id']}::{idx}",
                    review_id=review["review_id"],
                    pseudo_id=review["pseudo_id"],
                    rating=review.get("rating", 0),
                    star_bucket=bucket,
                    platform=platform,
                    locale=locale,
                    country=country,
                    submitted_at=submitted_at,
                    submission_week=week_bucket(submitted_at),
                    text=segment_text,
                    snapshot_date=snapshot_date,
                )
            )
    qc_stats = {
        "total_reviews": len(reviews),
        "segments_produced": len(segments),
        **issues,
    }
    return segments, qc_stats


def segments_to_df(segments: List[ReviewSegment]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_id": s.segment_id,
                "review_id": s.review_id,
                "pseudo_id": s.pseudo_id,
                "rating": s.rating,
                "star_bucket": s.star_bucket,
                "platform": s.platform,
                "locale": s.locale,
                "country": s.country,
                "submitted_at": s.submitted_at.isoformat(),
                "submission_week": s.submission_week,
                "text": s.text,
                "snapshot_date": s.snapshot_date,
            }
            for s in segments
        ]
    )


def persist_to_sqlite(df: pd.DataFrame, sqlite_path: Path) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        df.to_sql("review_segments", conn, if_exists="replace", index=False)
    finally:
        conn.close()


def persist_to_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
    df.to_parquet(parquet_path, index=False)


def update_latest_symlink(latest_path: Path, run_dir: Path) -> None:
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.resolve())


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_root = Path(args.input_dir)
    run_dir = (
        input_root / args.run_id
        if args.run_id
        else latest_run_folder(input_root)
    )
    run_id = run_dir.name

    LOGGER.info("Loading reviews from %s", run_dir)
    reviews = load_reviews(run_dir)
    if not reviews:
        raise ValueError(f"No reviews found in {run_dir}")

    LOGGER.info("Initializing NLP pipeline")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:  # model missing
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. Run `python -m spacy download en_core_web_sm`."
        ) from exc

    snapshot_date = load_snapshot_date(run_dir)
    segments, qc_stats = segment_reviews(
        reviews,
        nlp=nlp,
        lang_threshold=args.lang_threshold,
        min_segment_chars=args.min_segment_chars,
        snapshot_date=snapshot_date,
    )

    if not segments:
        raise ValueError("No segments produced; check quality filters.")

    df = segments_to_df(segments)
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_dir / "processed.db"
    parquet_path = output_dir / "segments.parquet"

    LOGGER.info("Writing %s segments to SQLite and Parquet", len(df))
    persist_to_sqlite(df, sqlite_path)
    persist_to_parquet(df, parquet_path)

    qc_path = output_dir / "quality_report.json"
    qc_path.write_text(json.dumps(qc_stats, indent=2), encoding="utf-8")
    latest_link = Path(args.output_dir) / "latest"
    update_latest_symlink(latest_link, output_dir)
    LOGGER.info("Quality stats: %s", qc_stats)
    LOGGER.info("Layer 2 processing complete. Outputs stored in %s", output_dir)


if __name__ == "__main__":
    main()

