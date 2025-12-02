#!/usr/bin/env python3
"""Layer 3: cluster review segments into <=5 actionable themes."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import google.generativeai as genai  # type: ignore

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

LOGGER = logging.getLogger("layer3.cluster")


@dataclass
class Theme:
    theme_id: str
    label: str
    description: str
    segment_count: int
    avg_rating: float
    sample_segment_ids: List[str]
    keywords: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster Layer-2 segments into themes.")
    parser.add_argument(
        "--segments-db",
        default="layer-2-data-processing/output/latest/processed.db",
        help="Path to Layer-2 SQLite database (processed.db).",
    )
    parser.add_argument(
        "--max-themes",
        type=int,
        default=5,
        help="Maximum number of themes to produce.",
    )
    parser.add_argument(
        "--min-theme-size",
        type=int,
        default=5,
        help="Minimum segments per theme; smaller clusters collapse into 'other'.",
    )
    parser.add_argument(
        "--output-dir",
        default="layer-3-theme-grouping/output",
        help="Directory for clustered outputs.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for embeddings.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-1.5-flash",
        help="Gemini model used for theme labeling.",
    )
    parser.add_argument(
        "--gemini-api-key-env",
        default="AIzaSyDMkOAFz8VtY00I9B57qdFzPujqmh8qBR0",
        help="Environment variable holding the Gemini API key.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Direct Gemini API key (overrides env lookup).",
    )
    return parser.parse_args()


def resolve_run_id(db_path: Path) -> str:
    run_dir = db_path.resolve().parent
    return run_dir.name


def load_segments(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM review_segments", conn)
    finally:
        conn.close()
    if df.empty:
        raise ValueError("No segments found in Layer-2 database.")
    return df


def compute_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    return np.asarray(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray, max_themes: int
) -> Tuple[np.ndarray, int]:
    count = embeddings.shape[0]
    n_clusters = min(max_themes, count)
    # Use KMeans to force exactly n_clusters; better for ensuring distinct themes
    from sklearn.preprocessing import normalize
    embeddings_norm = normalize(embeddings, norm="l2")
    clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clustering.fit_predict(embeddings_norm)
    return labels, n_clusters


def label_themes(
    df: pd.DataFrame, labels: np.ndarray, min_theme_size: int
) -> Tuple[pd.DataFrame, Dict[str, Theme]]:
    df = df.copy()
    df["theme_raw"] = labels
    theme_map: Dict[str, Theme] = {}

    # Build fallback bucket
    df["theme_id"] = "other"
    df["theme_label"] = "Other"

    grouped = df.groupby("theme_raw")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30)
    vectorizer.fit(df["text"])

    for raw_id, group in grouped:
        if len(group) < min_theme_size:
            continue
        texts = group["text"].tolist()
        tfidf = vectorizer.transform(texts).toarray().sum(axis=0)
        feature_names = vectorizer.get_feature_names_out()
        top_indices = np.argsort(tfidf)[::-1][:3]
        keywords = [feature_names[idx] for idx in top_indices if tfidf[idx] > 0]
        if not keywords:
            keywords = ["general"]
        theme_id = f"theme_{raw_id}"
        df.loc[group.index, "theme_id"] = theme_id
        df.loc[group.index, "theme_label"] = "Pending"
        theme_map[theme_id] = Theme(
            theme_id=theme_id,
            label="Pending",
            description="Awaiting Gemini labeling",
            segment_count=len(group),
            avg_rating=float(group["rating"].mean()),
            sample_segment_ids=group["segment_id"].head(3).tolist(),
            keywords=keywords[:5],
        )

    # Add fallback theme stats
    other_df = df[df["theme_id"] == "other"]
    if not other_df.empty:
        theme_map["other"] = Theme(
            theme_id="other",
            label="Other / Low Volume",
            description="Segments that did not meet clustering thresholds.",
            segment_count=len(other_df),
            avg_rating=float(other_df["rating"].mean()),
            sample_segment_ids=other_df["segment_id"].head(3).tolist(),
            keywords=[],
        )

    return df, theme_map


def configure_gemini(api_key: str) -> None:
    genai.configure(api_key=api_key)


def clean_json_response(raw_text: str) -> Dict[str, str]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]
    return json.loads(raw_text.strip())


def apply_gemini_labels(
    themes: Dict[str, Theme],
    assignments: pd.DataFrame,
    gemini_model: str,
) -> Dict[str, Theme]:
    text_lookup = dict(zip(assignments["segment_id"], assignments["text"]))
    model = genai.GenerativeModel(gemini_model)

    for theme_id, theme in themes.items():
        if theme_id == "other":
            continue
        sample_texts = [
            text_lookup.get(segment_id, "")
            for segment_id in theme.sample_segment_ids
            if text_lookup.get(segment_id)
        ]
        if not sample_texts:
            continue

        prompt = (
            "You are a product insights analyst for a finance app. "
            "Given customer review segments, identify the PRODUCT theme (e.g., UPI payments, SIP investments, account linking, trading features, app performance). "
            "Respond with JSON ONLY containing `label` (3-4 word product-focused title) and "
            "`description` (exactly one actionable sentence <=160 characters that summarizes the theme). "
            f"Keywords: {', '.join(theme.keywords) or 'n/a'}\n"
            "Review segments:\n"
        )
        for idx, text in enumerate(sample_texts, start=1):
            prompt += f"{idx}. {text}\n"
        prompt += "\nJSON:"

        try:
            response = model.generate_content(prompt)
            data = clean_json_response(response.text)
            theme.label = data.get("label", theme.label)
            theme.description = data.get("description", theme.description)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Gemini labeling failed for %s: %s", theme_id, exc)
            if not theme.label or theme.label == "Pending":
                # Create product-focused label from keywords
                if theme.keywords:
                    top_keywords = [k.title() for k in theme.keywords[:2]]
                    theme.label = " / ".join(top_keywords) if len(top_keywords) > 1 else f"{top_keywords[0]} Features"
                    theme.description = (
                        f"Focus on {', '.join(theme.keywords[:3])} issues impacting customers."
                    )
                else:
                    theme.label = "App Insights"
                    theme.description = "Focus on general app experience issues affecting customers."
    return themes


def resolve_gemini_key(args: argparse.Namespace) -> str:
    if args.gemini_api_key:
        return args.gemini_api_key
    env_target = args.gemini_api_key_env
    if env_target.startswith("AIza"):
        return env_target
    api_key = os.environ.get(env_target)
    if not api_key:
        raise EnvironmentError(
            f"Gemini API key not found. Set {env_target} or pass --gemini-api-key."
        )
    return api_key


def persist_outputs(
    assignments: pd.DataFrame,
    themes: Dict[str, Theme],
    output_dir: Path,
    root_output: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "segment_themes.parquet"
    themes_path = output_dir / "themes.parquet"
    summary_path = output_dir / "themes.json"

    assignments.to_parquet(assignments_path, index=False)
    themes_df = pd.DataFrame(
        [
            {
                "theme_id": theme.theme_id,
                "label": theme.label,
                "description": theme.description,
                "number_of_reviews": theme.segment_count,
                "sample_segment_ids": theme.sample_segment_ids,
            }
            for theme in themes.values()
        ]
    )
    themes_df.to_parquet(themes_path, index=False)
    summary = {
        "theme_count": len(themes),
        "total_segments": int(assignments.shape[0]),
        "themes": [
            {
                "theme_id": theme.theme_id,
                "label": theme.label,
                "description": theme.description,
                "number_of_reviews": theme.segment_count,
                "keywords": theme.keywords,
                "sample_segment_ids": theme.sample_segment_ids,
            }
            for theme in themes.values()
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_link = root_output / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.resolve())
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path = Path(args.segments_db)
    run_id = resolve_run_id(db_path)
    LOGGER.info("Loading segments from %s", db_path)
    segments = load_segments(db_path)

    LOGGER.info("Encoding %s segments with %s", len(segments), args.model_name)
    embeddings = compute_embeddings(segments["text"].tolist(), args.model_name)

    LOGGER.info("Clustering embeddings into <=%s themes", args.max_themes)
    labels, n_clusters = cluster_embeddings(embeddings, args.max_themes)
    LOGGER.info("Initial clusters: %s", n_clusters)

    assignments, themes = label_themes(
        segments,
        labels,
        min_theme_size=args.min_theme_size,
    )
    api_key = resolve_gemini_key(args)
    configure_gemini(api_key)
    themes = apply_gemini_labels(
        themes,
        assignments,
        gemini_model=args.gemini_model,
    )

    root_output = Path(args.output_dir)
    output_dir = root_output / run_id
    LOGGER.info("Persisting outputs to %s", output_dir)
    persist_outputs(assignments, themes, output_dir, root_output)
    LOGGER.info(
        "Theme counts: %s",
        {theme.theme_id: theme.segment_count for theme in themes.values()},
    )


if __name__ == "__main__":
    main()

