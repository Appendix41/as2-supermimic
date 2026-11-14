"""Project-local runtime outputs, including when launched outside the project."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("AS2_OUTPUT_ROOT", PROJECT_ROOT / "outputs/as2-supermimic"))
