````markdown
# eco_tools: Eco-Resilience Analysis Toolkit

A lightweight Python package containing shared utilities for data preprocessing (detrending, deseasonalizing) and visualization (maps, boxplots).

This package is designed to be modular and installable, following the best practices for scientific computing.

## 📦 Installation

This package is designed to be installed in **editable mode**. This means you can modify the code in `eco_tools/` and the changes will immediately be reflected in your scripts without reinstalling.

1. **Clone or download** this repository.
2. Open your terminal and navigate to the root folder (where `pyproject.toml` is).
3. Run:

```bash
pip install -e .
````

### Dependencies

The installation will automatically fetch the required libraries, including:

  * `xarray` & `dask` (Data handling)
  * `matplotlib` & `seaborn` (Plotting)
  * `cartopy` & `regionmask` (Geospatial tools)
  * `scipy` (Statistics)

## 🚀 Usage

### 1\. Preprocessing Data

The package exposes common climate data operations at the top level.

```python
import eco_tools as et
import xarray as xr

# Load your raw data
ds = xr.open_dataset('raw_data.nc')

# 1. Fix Coordinates (Longitude 0-360 -> -180-180, sort Lat/Lon)
ds = et.standardize_coords(ds)

# 2. Mask Ocean (Keep Land only)
ds_land = et.mask_ocean(ds)

# 3. Remove Linear Trend
ds_detrended = et.detrend(ds_land, dim='time')

# 4. Remove Seasonal Cycle
ds_anom = et.deseasonalize(ds_detrended)
```

### 2\. Visualization

Plotting functions are grouped under the `et.plot` module.

```python
# Plot a global map with country borders and colorbar
et.plot.plot_global_var(
    ds_anom['temp'].isel(time=0), 
    label="Temperature Anomaly (K)",
    save_fig="global_temp_map"  # Saves to configured scratch path
)

# Plot boxplots
et.plot.plot_boxplot([data_array1, data_array2], xlabels=['Region A', 'Region B'])
```

## ⚙️ Configuration

The package looks for a configuration file to determine where to save figures and intermediate data.

You can modify `eco_tools/config.py` directly, or set the environment variable `ECO_SCRATCH` on your system.

```python
# Check your current config paths
print(et.config.SCRATCH_PATH)
```

## 📂 Project Structure

```text
eco_tools_project/
├── pyproject.toml       # Installer configuration
├── eco_tools/           # Source code
│   ├── __init__.py      # Exports functions
│   ├── config.py        # Paths and settings
│   ├── preprocess.py    # Math & Xarray wrappers (detrend, etc.)
│   └── plot.py          # Visualization wrappers
└── tests/               # Unit tests
```

## 📝 License

[MIT](LICENSE) - Free to use and modify.

```

```