import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtickers
import string
import xarray as xr
import os

# Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Seaborn
import seaborn as sns

# Colorbar utilities
import mplotutils as mpu

# Import config (optional, handles if config is missing)
try:
    from . import config
    DEFAULT_SAVE_PATH = getattr(config, 'SCRATCH_PATH', '.')
    DEFAULT_DPI = getattr(config, 'DEFAULT_DPI', 300)
except ImportError:
    DEFAULT_SAVE_PATH = '.'
    DEFAULT_DPI = 300


# --- Helpers ---

def _get_lat_name(da):
    """Finds latitude name (lat or latitude)."""
    for name in ['lat', 'latitude']:
        if name in da.coords: 
            return name
    return 'lat' # Default fallback

def plot_save(save_fig='figure', path=None):
    """
    Saves the figure.
    """
    if save_fig is not None:
        if path is None:
            # defaults to . if config not set
            base_dir = os.path.join(DEFAULT_SAVE_PATH) 
        else:
            base_dir = path
            
        full_path = os.path.join(base_dir, f"{save_fig}.png")
        plt.savefig(full_path, dpi=DEFAULT_DPI, bbox_inches='tight')

# --- Plotting Functions ---

def plot_add_subplot_label(axes, location='upper left', adjust_wi=0., adjust_hi=0., 
                           rm_title=True, defined_label=None, weight=None, 
                           prefix='(', suffix=')', size=20):
    """
    Adds labels (a, b, c...) to subplots.
    """
    if location == 'upper left':
        x, y = 0 + adjust_wi, 1 + adjust_hi
    elif location == 'upper right':
        x, y = 0.95 + adjust_wi, 1 + adjust_hi
    else:
        raise ValueError("location should be only upper left or right")

    # Handle single axis vs list of axes
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    else:
        # Flatten if it's a numpy array of axes
        if hasattr(axes, 'flat'):
            axes = axes.flat

    for i, ax in enumerate(axes):
        if rm_title:
            ax.set_title('')

        label_text = defined_label[i] if defined_label else string.ascii_lowercase[i]
        final_text = f"{prefix}{label_text}{suffix}"
        
        ax.text(x, y, final_text, transform=ax.transAxes,
                size=size, weight=weight)

    return axes


def plot_feature_grids(ax, draw_xlabels=True, draw_ylabels=True, 
                       xticks=np.arange(-180, 180, 90), 
                       yticks=np.arange(-90, 90, 30)):
    """
    Draws coastal lines, borders, and gridlines.
    Fixed for Cartopy >= 0.20.
    """
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    # NEW CARTOPY SYNTAX (Old gl.xlabels_top is deprecated)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = draw_xlabels
    gl.left_labels = draw_ylabels
    
    gl.xlocator = mtickers.FixedLocator(xticks)
    gl.ylocator = mtickers.FixedLocator(yticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    return ax


def plot_global_var(data, ax=None, cmap='RdBu_r', levels=None, extend='neither', 
                    add_colorbar=True, label='', save_fig=None, 
                    vmin=None, vmax=None, label_size=18,
                    projection=ccrs.PlateCarree()):
    """
    Plots a 2D variable on a map.
    """
    # 1. Handle Time Dimension (grab last step if exists)
    if 'time' in data.dims and data.ndim == 3:
        data = data.isel(time=-1)

    # 2. Setup Axis
    if ax is None:
        # Avoid hard resetting sns unless necessary
        # sns.reset_orig() 
        plt.rcParams.update({'font.size': 24})
        fig = plt.figure(figsize=(20, 13))
        ax = plt.axes(projection=projection)

    plot_feature_grids(ax, True, True)

    # 3. Handle vmin/vmax Logic
    kwargs = {
        'ax': ax, 
        'transform': ccrs.PlateCarree(), 
        'cmap': cmap, 
        'add_colorbar': False
    }
    
    if levels is not None:
        kwargs['levels'] = levels
        kwargs['extend'] = extend if extend else 'neither'
    else:
        if vmin is not None: 
            kwargs['vmin'] = vmin
        if vmax is not None: 
            kwargs['vmax'] = vmax
        # If both are None and no levels, xarray handles it auto

    # 4. Plot
    im = data.plot(**kwargs)
    ax.set_title('')

    # 5. Manual Colorbar (Replaces mplotutils)
    if add_colorbar:
        # Create a colorbar axis at the bottom
        cbar = mpu.colorbar(im, ax=ax, orientation='horizontal', 
                            shrink=0.6, pad=0.05, extend=kwargs.get('extend', 'neither'))
        cbar.set_label(label, fontsize=label_size)
        if levels is not None:
            cbar.set_ticks(levels)

    # 6. Save
    plot_save(save_fig)

    return ax


def plot_boxplot(data_list, xlabels=None, ylabel='Correlation'):
    """
    Plots boxplots from a list of DataArrays.
    """
    # Flatten and clean data for seaborn
    data_remove_na_list = []
    
    for data in data_list:
        # .values (or .to_numpy()) is safer than .data for xarray
        # flatten() ensures 1D array
        d = data.values.flatten()
        d = d[~np.isnan(d)]
        data_remove_na_list.append(d)

    custom_params = {"axes.spines.right": False,
                     "axes.spines.top": False, 'figure.figsize': (10, 8)}
    sns.set_theme(style="ticks", rc=custom_params, font_scale=2)

    ax = sns.boxplot(data=data_remove_na_list)
    ax.axhline(y=0, ls='--')
    
    # Ticks
    if xlabels is None:
        xlabels = np.arange(len(data_remove_na_list))
    
    ax.set_xticks(np.arange(len(data_remove_na_list)))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel, fontsize=20)

    return ax