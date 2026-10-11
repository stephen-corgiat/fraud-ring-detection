"""Detector. Reads ONLY the event stream (never ground truth)."""

from fraudrings.detector.engine import run_investigation
from fraudrings.detector.ingest import build_db

__all__ = ["build_db", "run_investigation"]
