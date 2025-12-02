#!/usr/bin/env python3
"""Layer 3: Group similar review segments into meaningful themes."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("theme_grouping.log")
    ]
)
logger = logging.getLogger("theme_grouper")


def load_segments(db_path: str) -> pd.DataFrame:
    """Load segmented reviews from the SQLite database."""
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT 
                id, review_id, segment_text, sentiment, 
                rating, app_version, created_at
            FROM segments
            """
            return pd.read_sql_query(query, conn)
    except Exception as e:
        logger.error(f"Error loading segments from {db_path}: {e}")
        raise


def extract_keywords(texts: List[str], max_features: int = 100) -> Tuple[Any, np.ndarray]:
    """Extract TF-IDF features from segment texts."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )
    features = vectorizer.fit_transform(texts)
    return vectorizer, features


def group_into_themes(
    df: pd.DataFrame, 
    n_clusters: int = 5,
    max_features: int = 100
) -> Tuple[pd.DataFrame, Dict]:
    """Group segments into themes using K-means clustering."""
    # Extract features
    vectorizer, features = extract_keywords(df['segment_text'].tolist(), max_features)
    
    # Cluster segments
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['theme_id'] = kmeans.fit_predict(features)
    
    # Get top keywords for each theme
    feature_names = vectorizer.get_feature_names_out()
    theme_keywords = {}
    
    for i in range(n_clusters):
        # Get feature importance for this cluster
        cluster_center = kmeans.cluster_centers_[i]
        top_keyword_indices = cluster_center.argsort()[-5:][::-1]
        theme_keywords[i] = [feature_names[idx] for idx in top_keyword_indices]
    
    return df, theme_keywords


def save_results(df: pd.DataFrame, theme_keywords: Dict, output_dir: Path) -> None:
    """Save theme grouping results."""
    # Create output directory with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save theme assignments
    df[['id', 'review_id', 'theme_id']].to_csv(run_dir / 'theme_assignments.csv', index=False)
    
    # Save theme keywords
    with open(run_dir / 'theme_keywords.json', 'w') as f:
        json.dump(theme_keywords, f, indent=2)
    
    # Create a summary of themes
    theme_summary = []
    for theme_id, keywords in theme_keywords.items():
        theme_segments = df[df['theme_id'] == theme_id]
        theme_summary.append({
            'theme_id': int(theme_id),
            'segment_count': len(theme_segments),
            'avg_rating': theme_segments['rating'].mean(),
            'top_keywords': keywords,
            'example_segments': theme_segments['segment_text'].head(3).tolist()
        })
    
    with open(run_dir / 'theme_summary.json', 'w') as f:
        json.dump(theme_summary, f, indent=2)
    
    # Update latest symlink
    latest_link = output_dir / 'latest'
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name, target_is_directory=True)
    
    logger.info(f"Results saved to {run_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Group review segments into themes.')
    parser.add_argument(
        '--segments-db',
        type=str,
        default='layer-2-data-processing/output/latest/processed.db',
        help='Path to SQLite database with segmented reviews',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='layer-3-theme-grouping/output',
        help='Directory to save theme grouping results',
    )
    parser.add_argument(
        '--max-themes',
        type=int,
        default=5,
        help='Maximum number of themes to create',
    )
    return parser.parse_args()


def main() -> int:
    """Main function to run theme grouping."""
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load segments
        logger.info(f"Loading segments from {args.segments_db}")
        segments_df = load_segments(args.segments_db)
        logger.info(f"Loaded {len(segments_df)} segments")
        
        if len(segments_df) == 0:
            logger.error("No segments found in the database")
            return 1
        
        # Group into themes
        logger.info(f"Grouping segments into {args.max_themes} themes")
        segments_df, theme_keywords = group_into_themes(
            segments_df, 
            n_clusters=args.max_themes
        )
        
        # Save results
        logger.info("Saving theme grouping results")
        save_results(segments_df, theme_keywords, output_dir)
        
        logger.info("Theme grouping completed successfully")
        return 0
        
    except Exception as e:
        logger.exception("Error in theme grouping")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())