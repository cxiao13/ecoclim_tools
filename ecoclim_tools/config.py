import os
from pathlib import Path

# Try to get the path from the system environment, 
# otherwise default to a folder named 'scratch' in the current directory.
SCRATCH_PATH = os.getenv("ECO_SCRATCH", "./scratch")

# Convert to a Path object (easier to use than strings)
scratch_dir = Path(SCRATCH_PATH)

# Ensure they exist
if not scratch_dir.exists():
    SCRATCH_PATH = '.'

# --- Other Constants ---
# You can also store default plotting dictionaries here
# to keep your plot.py clean.
DEFAULT_FIG_SIZE = (12, 8)
DEFAULT_DPI = 300