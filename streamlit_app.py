"""Entrypoint for the public hosted demo (Streamlit Community Cloud).

Locally, use `fraudrings view`. This file exists so a host can point at one script: it switches the
viewer into hosted mode (per-session labels, bounded world cache) and runs the same app.
"""

import os
import sys

os.environ["FRAUDRINGS_HOSTED"] = "1"
# Keep native thread pools small on a shared, memory-capped host; must be set before polars loads.
os.environ.setdefault("POLARS_MAX_THREADS", "2")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from fraudrings.viewer.main import main  # noqa: E402

main()
