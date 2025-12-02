#!/usr/bin/env python3
"""Layer 5: Generate weekly one-page pulse email from Layer-4 insights."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai  # type: ignore

LOGGER = logging.getLogger("layer5.email")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weekly review pulse email.")
    parser.add_argument(
        "--insights-json",
        default="layer-4-insight-extraction/output/latest/insights.json",
        help="Path to Layer-4 insights JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="layer-5-email-generation/output",
        help="Directory for email outputs.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-1.5-flash",
        help="Gemini model for email drafting.",
    )
    parser.add_argument(
        "--gemini-api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable name containing the Gemini API key.",
    )
    parser.add_argument(
        "--recipient",
        default="product-team@indmoney.com",
        help="Primary email recipient.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Snapshot date override (YYYY-MM-DD). Defaults to latest from insights.",
    )
    return parser.parse_args()


def load_insights(insights_path: Path) -> Dict:
    with insights_path.open(encoding="utf-8") as f:
        return json.load(f)


def build_email_payload(insights: Dict, snapshot_date: str) -> Dict:
    """Build structured JSON payload for email template."""
    themes = insights.get("themes", [])
    total_segments = sum(t.get("segment_count", 0) for t in themes)
    avg_rating = sum(t.get("avg_rating", 0) * t.get("segment_count", 0) for t in themes)
    if total_segments > 0:
        avg_rating /= total_segments

    return {
        "snapshot_date": snapshot_date,
        "total_segments": total_segments,
        "theme_count": len(themes),
        "avg_rating": round(avg_rating, 2),
        "themes": [
            {
                "rank": t.get("rank", idx + 1),
                "label": t.get("label", "Unknown"),
                "segment_count": t.get("segment_count", 0),
                "avg_rating": round(t.get("avg_rating", 0), 2),
                "insight": t.get("insight", ""),
                "top_quote": t.get("quotes", [{}])[0].get("text", "") if t.get("quotes") else "",
                "action": t.get("actions", [{}])[0].get("description", "") if t.get("actions") else "",
                "owner": t.get("actions", [{}])[0].get("owner", "TBD") if t.get("actions") else "TBD",
            }
            for idx, t in enumerate(themes)
        ],
    }


def draft_email_with_llm(payload: Dict, gemini_model: str, api_key: str) -> str:
    """Use Gemini to draft email content."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(gemini_model)

    prompt = f"""You are a product insights analyst drafting a weekly one-page pulse email for a finance app product team.

Write a concise, professional email (max 500 words) that:
- Opens with a brief snapshot summary (date, total reviews analyzed, average rating)
- Presents top themes in a clear table format with: Rank, Theme, Segments, Avg Rating
- Includes 1-2 representative quotes per theme (truncate to <200 chars, no PII)
- Lists actionable next steps per theme with owner assignments
- Closes with a brief call-to-action

Tone: Professional, data-driven, actionable. Audience: Product managers, engineers, designers.

Data:
{json.dumps(payload, indent=2)}

Generate the email body in HTML format suitable for email clients. Use simple HTML tags: <h2>, <h3>, <p>, <table>, <tr>, <td>, <strong>, <em>. Include inline styles for basic formatting."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Gemini email drafting failed: %s", exc)
        return generate_fallback_email(payload)


def generate_fallback_email(payload: Dict) -> str:
    """Generate fallback email template if LLM fails."""
    themes_html = ""
    themes_detail = ""
    for theme in payload["themes"]:
        themes_html += f"""
        <tr>
            <td><strong>{theme['rank']}</strong></td>
            <td>{theme['label']}</td>
            <td>{theme['segment_count']}</td>
            <td>{theme['avg_rating']}★</td>
        </tr>"""
        
        quote_text = theme.get('top_quote', '')[:200] if theme.get('top_quote') else 'No quote available'
        action_text = theme.get('action', 'TBD')
        owner = theme.get('owner', 'TBD')
        
        themes_detail += f"""
    <div style="margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; background-color: #f9f9f9;">
        <h3>{theme['rank']}. {theme['label']}</h3>
        <p><strong>Insight:</strong> {theme.get('insight', 'N/A')}</p>
        <div class="quote">"{quote_text}"</div>
        <div class="action">
            <strong>Action:</strong> {action_text}<br>
            <strong>Owner:</strong> {owner}
        </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .quote {{ font-style: italic; color: #666; margin: 10px 0; padding-left: 20px; border-left: 3px solid #4CAF50; }}
        .action {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Weekly Review Pulse - {payload['snapshot_date']}</h1>
        <p><strong>Total Segments:</strong> {payload['total_segments']} | 
           <strong>Average Rating:</strong> {payload['avg_rating']}★ | 
           <strong>Themes Identified:</strong> {payload['theme_count']}</p>
    </div>

    <h2>Top Themes Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Theme</th>
                <th>Segments</th>
                <th>Avg Rating</th>
            </tr>
        </thead>
        <tbody>
            {themes_html}
        </tbody>
    </table>

    <h2>Theme Details & Actions</h2>
    {themes_detail}
    
    <div style="margin-top: 30px; padding: 15px; background-color: #e7f3ff; border-radius: 5px;">
        <p><strong>Next Steps:</strong> Review themes above and prioritize based on segment count and ratings. Assign owners and track progress in next week's pulse.</p>
    </div>
</body>
</html>
"""


def extract_plaintext(html_content: str) -> str:
    """Extract plaintext version from HTML (simple approach)."""
    import re
    text = re.sub(r"<[^>]+>", "", html_content)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_email(html_content: str, payload: Dict) -> List[str]:
    """Basic QA checks."""
    issues = []
    if "PII" in html_content.upper() or "@" in html_content:
        issues.append("Potential PII detected")
    if len(html_content) > 50000:
        issues.append("Email exceeds recommended length")
    if payload["snapshot_date"] not in html_content:
        issues.append("Snapshot date missing")
    return issues


def persist_email(
    html_content: str,
    plaintext: str,
    payload: Dict,
    output_dir: Path,
    recipient: str,
) -> None:
    """Save email outputs with metadata. Overwrites same-day outputs to keep only latest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.utcnow()
    date_dir = now.strftime("%Y%m%d")  # One directory per day
    run_id = now.strftime("%Y%m%dT%H%M%SZ")  # Full timestamp for metadata

    # Use date-based directory, overwrite files if they exist
    run_dir = output_dir / date_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    
    html_path = run_dir / "email.html"
    txt_path = run_dir / "email.txt"
    metadata_path = run_dir / "metadata.json"

    # Overwrite existing files for the same day
    html_path.write_text(html_content, encoding="utf-8")
    txt_path.write_text(plaintext, encoding="utf-8")

    metadata = {
        "generated_at": now.isoformat(),
        "run_id": run_id,
        "recipient": recipient,
        "snapshot_date": payload["snapshot_date"],
        "total_segments": payload["total_segments"],
        "theme_count": payload["theme_count"],
        "avg_rating": payload["avg_rating"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latest_link = output_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.resolve())

    LOGGER.info("Email saved to %s (overwrites same-day outputs)", html_path)
    LOGGER.info("Plaintext saved to %s", txt_path)
    LOGGER.info("Metadata saved to %s", metadata_path)


def resolve_gemini_key(args: argparse.Namespace) -> str:
    if args.gemini_api_key_env.startswith("AIza"):
        return args.gemini_api_key_env
    api_key = os.environ.get(args.gemini_api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Gemini API key not found. Set {args.gemini_api_key_env} or pass --gemini-api-key."
        )
    return api_key


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    insights_path = Path(args.insights_json)
    if not insights_path.exists():
        raise FileNotFoundError(f"Insights file not found: {insights_path}")

    LOGGER.info("Loading insights from %s", insights_path)
    insights = load_insights(insights_path)

    snapshot_date = args.snapshot_date
    if not snapshot_date:
        snapshot_date = insights.get("generated_at", dt.datetime.utcnow().isoformat())[:10]

    payload = build_email_payload(insights, snapshot_date)
    LOGGER.info("Built email payload: %s themes, %s segments", payload["theme_count"], payload["total_segments"])

    api_key = resolve_gemini_key(args)
    LOGGER.info("Drafting email with %s", args.gemini_model)
    html_content = draft_email_with_llm(payload, args.gemini_model, api_key)

    plaintext = extract_plaintext(html_content)
    issues = validate_email(html_content, payload)
    if issues:
        LOGGER.warning("Email validation issues: %s", issues)

    output_dir = Path(args.output_dir)
    persist_email(html_content, plaintext, payload, output_dir, args.recipient)

    print("\n" + "=" * 80)
    print("EMAIL GENERATED")
    print("=" * 80)
    print(f"HTML: {output_dir}/latest/email.html")
    print(f"Plaintext: {output_dir}/latest/email.txt")
    print(f"Metadata: {output_dir}/latest/metadata.json")
    if issues:
        print(f"\nWarnings: {', '.join(issues)}")


if __name__ == "__main__":
    main()

