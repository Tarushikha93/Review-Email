#!/usr/bin/env python3
"""Layer 4: turn clustered themes into insights, quotes, and actions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import google.generativeai as genai  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate insights from Layer-3 themes.")
    parser.add_argument(
        "--themes-json",
        default="layer-3-theme-grouping/output/latest/themes.json",
        help="Path to Layer-3 themes.json (default: latest run).",
    )
    parser.add_argument(
        "--segment-parquet",
        default="layer-3-theme-grouping/output/latest/segment_themes.parquet",
        help="Path to Layer-3 segment assignment parquet.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-1.5-flash",
        help="Gemini model for insight generation.",
    )
    parser.add_argument(
        "--gemini-api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable name containing the Gemini API key.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Direct Gemini API key (overrides env lookup).",
    )
    parser.add_argument(
        "--max-quotes",
        type=int,
        default=2,
        help="Maximum quotes per theme.",
    )
    parser.add_argument(
        "--output-dir",
        default="layer-4-insight-extraction/output",
        help="Directory to store generated insights.",
    )
    return parser.parse_args()


def configure_gemini(api_key: str) -> None:
    genai.configure(api_key=api_key)


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


def load_inputs(themes_path: Path, segments_path: Path) -> tuple[List[Dict], pd.DataFrame]:
    themes = json.loads(themes_path.read_text(encoding="utf-8"))["themes"]
    segments = pd.read_parquet(segments_path)
    return themes, segments


def select_quotes(theme_id: str, df: pd.DataFrame, max_quotes: int) -> List[Dict]:
    subset = df[df["theme_id"] == theme_id].copy()
    subset = subset.sort_values("submitted_at").head(max_quotes)
    quotes = []
    for _, row in subset.iterrows():
        text = (row["text"] or "").strip()
        if not text or len(text) < 30:
            continue
        quotes.append(
            {
                "segment_id": row["segment_id"],
                "rating": int(row.get("rating", 0)),
                "text": text[:200],
            }
        )
    return quotes


def build_prompt(theme: Dict, quotes: List[Dict]) -> str:
    prompt = (
        "You are a product insights strategist for a finance super-app. "
        "Given the theme metadata and supporting quotes, return JSON with keys: "
        "`insight` (2 sentences), `quotes` (array of objects with `text` and `why_it_matters`), "
        "`actions` (array with `description` and `owner`), and `priority_reason`. "
        "Do not include markdown fences. Quotes must stay under 200 characters and remove PII. "
        "Owners should be functions like 'Product', 'Engineering', 'Support', or 'Marketing'.\n\n"
        f"Theme Label: {theme['label']}\n"
        f"Description: {theme['description']}\n"
        f"Segments: {theme['segment_count']}\n"
        f"Average Rating: {theme['avg_rating']:.2f}\n"
        f"Keywords: {', '.join(theme.get('keywords', [])) or 'n/a'}\n"
        "Quotes:\n"
    )
    for quote in quotes:
        prompt += f"- ({quote['rating']}★) {quote['text']}\n"
    prompt += "\nJSON:"
    return prompt


def call_gemini(model_name: str, prompt: str) -> Dict:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def compute_impact(theme: Dict) -> float:
    return float(theme["segment_count"]) * max(0.1, 5 - float(theme["avg_rating"]))


def main() -> None:
    args = parse_args()
    themes_path = Path(args.themes_json)
    segments_path = Path(args.segment_parquet)
    themes, segments = load_inputs(themes_path, segments_path)

    api_key = resolve_gemini_key(args)
    configure_gemini(api_key)

    enriched = []
    for theme in themes:
        quotes = select_quotes(theme["theme_id"], segments, args.max_quotes)
        if not quotes:
            continue
        prompt = build_prompt(theme, quotes)
        try:
            ai_summary = call_gemini(args.gemini_model, prompt)
        except Exception as exc:  # pylint: disable=broad-except
            ai_summary = {
                "insight": f"{theme['label']}: {theme['description']}",
                "quotes": [
                    {
                        "text": q["text"],
                        "why_it_matters": "Representative customer voice.",
                    }
                    for q in quotes
                ],
                "actions": [
                    {
                        "description": "Investigate recurring customer complaints in this theme.",
                        "owner": "Product",
                    }
                ],
                "priority_reason": str(exc),
            }
        impact = compute_impact(theme)
        enriched.append(
            {
                "theme_id": theme["theme_id"],
                "label": theme["label"],
                "segment_count": theme["segment_count"],
                "avg_rating": theme["avg_rating"],
                "impact_score": impact,
                "insight": ai_summary.get("insight", ""),
                "quotes": ai_summary.get("quotes", []),
                "actions": ai_summary.get("actions", []),
                "priority_reason": ai_summary.get("priority_reason", ""),
            }
        )

    enriched.sort(key=lambda row: row["impact_score"], reverse=True)
    for idx, row in enumerate(enriched, start=1):
        row["rank"] = idx

    run_id = themes_path.parent.name
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "model": args.gemini_model,
        "theme_count": len(enriched),
        "themes": enriched,
    }
    (output_dir / "insights.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(enriched).to_parquet(output_dir / "insights.parquet", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

