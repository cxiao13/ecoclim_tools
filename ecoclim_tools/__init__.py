"""
eco_tools: A lightweight package for eco-climate data preprocessing and plotting.
"""

# 1. Define the version
__version__ = "0.1.0"

# 2. Expose the configuration module
# Usage: et.config.scratch_dir
from . import config

# 3. Expose the Plotting module as a namespace
# Usage: et.plot.plot_global_var()
from . import plot

# 4. Expose Preprocessing functions directly (The "Shortcut")
# Usage: et.detrend() instead of et.preprocess.detrend()
# This makes common math functions faster to type.
from .preprocess import (
    format_coords,
    detrend,
    deseasonalize,
    area_weighted_mean,
    normalize,
    mask_ocean
)

# 5. (Optional) Define what gets imported if 'from eco_tools import *'
__all__ = [
    "config",
    "plot",
    "format_coords",
    "detrend",
    "deseasonalize",
    "area_weighted_mean",
    "normalize",
    "mask_ocean",
]