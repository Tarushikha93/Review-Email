#!/usr/bin/env python3
"""Layer 6: Orchestrate weekly review pipeline from extraction to email generation."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional
export SMTP_USER='shikha.taru93@gmail.com'
export SMTP_PASSWORD='jlao wswx qngq wmxs'

LOGGER = logging.getLogger("layer6.orchestrator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate weekly review pipeline.")
    parser.add_argument(
        "--skip-layer",
        action="append",
        default=[],
        help="Skip specific layers (e.g., --skip-layer 1 --skip-layer 2)",
    )
    parser.add_argument(
        "--count-per-rating",
        type=int,
        default=30,
        help="Reviews per rating bucket for Layer 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    return parser.parse_args()


def run_command(
    cmd: list[str], cwd: Optional[Path] = None, dry_run: bool = False
) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    cmd_str = " ".join(cmd)
    LOGGER.info("Running: %s", cmd_str)
    
    if dry_run:
        print(f"[DRY RUN] Would execute: {cmd_str}")
        return 0, "", ""
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            LOGGER.error("Command failed: %s", cmd_str)
            LOGGER.error("STDERR: %s", result.stderr)
        else:
            LOGGER.info("Command succeeded")
        return result.returncode, result.stdout, result.stderr
    except Exception as exc:
        LOGGER.error("Exception running command: %s", exc)
        return 1, "", str(exc)


def run_layer_1(count_per_rating: int, dry_run: bool) -> tuple[bool, Optional[str]]:
    """Run Layer 1: Data extraction."""
    LOGGER.info("=" * 80)
    LOGGER.info("LAYER 1: Data Extraction")
    LOGGER.info("=" * 80)
    
    cmd = [
        "python3",
        "layer-1-data-extraction/run_extraction.py",
        "--count-per-rating",
        str(count_per_rating),
    ]
    
    exit_code, stdout, stderr = run_command(cmd, dry_run=dry_run)
    if exit_code != 0:
        return False, stderr
    
    # Extract run ID from output
    for line in stdout.split("\n"):
        if "run-" in line and "INFO" in line:
            parts = line.split("run-")
            if len(parts) > 1:
                run_id = "run-" + parts[1].split()[0].rstrip("/")
                return True, run_id
    
    return True, None


def run_layer_2(run_id: Optional[str], dry_run: bool) -> tuple[bool, Optional[str]]:
    """Run Layer 2: Data processing."""
    LOGGER.info("=" * 80)
    LOGGER.info("LAYER 2: Data Processing")
    LOGGER.info("=" * 80)
    
    if run_id:
        cmd = [
            "python3",
            "layer-2-data-processing/process_reviews.py",
            "--input-dir",
            "layer-1-data-extraction/layer-1",
            "--run-id",
            run_id,
        ]
    else:
        # Use latest
        cmd = [
            "python3",
            "layer-2-data-processing/process_reviews.py",
            "--input-dir",
            "layer-1-data-extraction/layer-1",
        ]
    
    exit_code, stdout, stderr = run_command(cmd, dry_run=dry_run)
    if exit_code != 0:
        return False, stderr
    
    return True, None


def run_layer_3(dry_run: bool) -> tuple[bool, Optional[str]]:
    """Run Layer 3: Theme grouping."""
    LOGGER.info("=" * 80)
    LOGGER.info("LAYER 3: Theme Grouping")
    LOGGER.info("=" * 80)
    
    cmd = [
        "python3",
        "layer-3-theme-grouping/group_themes.py",
        "--segments-db",
        "layer-2-data-processing/output/latest/processed.db",
        "--max-themes",
        "5",
    ]
    
    exit_code, stdout, stderr = run_command(cmd, dry_run=dry_run)
    if exit_code != 0:
        return False, stderr
    
    return True, None


def run_layer_4(dry_run: bool) -> tuple[bool, Optional[str]]:
    """Run Layer 4: Insight extraction."""
    LOGGER.info("=" * 80)
    LOGGER.info("LAYER 4: Insight Extraction")
    LOGGER.info("=" * 80)
    
    cmd = [
        "python3",
        "layer-4-insight-extraction/generate_insights.py",
    ]
    
    exit_code, stdout, stderr = run_command(cmd, dry_run=dry_run)
    if exit_code != 0:
        return False, stderr
    
    return True, None


def run_layer_5(dry_run: bool) -> tuple[bool, Optional[str]]:
    """Run Layer 5: Email generation."""
    LOGGER.info("=" * 80)
    LOGGER.info("LAYER 5: Email Generation")
    LOGGER.info("=" * 80)
    
    cmd = [
        "python3",
        "layer-5-email-generation/generate_email.py",
    ]
    
    exit_code, stdout, stderr = run_command(cmd, dry_run=dry_run)
    if exit_code != 0:
        return False, stderr
    
    return True, None


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    
    start_time = dt.datetime.utcnow()
    LOGGER.info("Starting weekly pipeline orchestration at %s", start_time.isoformat())
    
    skipped = set(args.skip_layer)
    errors = []
    
    # Layer 1: Extraction
    run_id = None
    if "1" not in skipped:
        success, error = run_layer_1(args.count_per_rating, args.dry_run)
        if not success:
            errors.append(f"Layer 1 failed: {error}")
            LOGGER.error("Pipeline failed at Layer 1")
            return 1
    else:
        LOGGER.info("Skipping Layer 1")
    
    # Layer 2: Processing
    if "2" not in skipped:
        success, error = run_layer_2(run_id, args.dry_run)
        if not success:
            errors.append(f"Layer 2 failed: {error}")
            LOGGER.error("Pipeline failed at Layer 2")
            return 1
    else:
        LOGGER.info("Skipping Layer 2")
    
    # Layer 3: Theme Grouping
    if "3" not in skipped:
        success, error = run_layer_3(args.dry_run)
        if not success:
            errors.append(f"Layer 3 failed: {error}")
            LOGGER.error("Pipeline failed at Layer 3")
            return 1
    else:
        LOGGER.info("Skipping Layer 3")
    
    # Layer 4: Insight Extraction
    if "4" not in skipped:
        success, error = run_layer_4(args.dry_run)
        if not success:
            errors.append(f"Layer 4 failed: {error}")
            LOGGER.error("Pipeline failed at Layer 4")
            return 1
    else:
        LOGGER.info("Skipping Layer 4")
    
    # Layer 5: Email Generation
    if "5" not in skipped:
        success, error = run_layer_5(args.dry_run)
        if not success:
            errors.append(f"Layer 5 failed: {error}")
            LOGGER.error("Pipeline failed at Layer 5")
            return 1
    else:
        LOGGER.info("Skipping Layer 5")
    
    end_time = dt.datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    
    LOGGER.info("=" * 80)
    LOGGER.info("PIPELINE COMPLETED SUCCESSFULLY")
    LOGGER.info("=" * 80)
    LOGGER.info("Start time: %s", start_time.isoformat())
    LOGGER.info("End time: %s", end_time.isoformat())
    LOGGER.info("Duration: %.2f seconds", duration)
    LOGGER.info("Email available at: layer-5-email-generation/output/latest/email.html")
    
    if errors:
        LOGGER.warning("Errors encountered: %s", errors)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

