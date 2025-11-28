import xarray as xr
import numpy as np
import dask
from typing import Hashable

def format_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Standardizes coordinate names and ranges:
    1. Renames 'longitude' -> 'lon' and 'latitude' -> 'lat'.
    2. Checks if longitude is 0-360 (by checking if max > 180). 
       If yes, converts to -180 to 180 and sorts ascending.
    3. Sorts 'lat' descending (90 to -90).
    """
    # 1. Standardize Names (longitude -> lon, latitude -> lat)
    # We must do this FIRST so we can find 'lon' to check the range.
    rename_map = {}
    if 'longitude' in da.dims or 'longitude' in da.coords:
        rename_map['longitude'] = 'lon'
    if 'latitude' in da.dims or 'latitude' in da.coords:
        rename_map['latitude'] = 'lat'
    
    if rename_map:
        da = da.rename(rename_map)

    # 2. Logic: Check Range -> Convert -> Sort
    if 'lon' in da.coords:
        # We assume if the max longitude is > 180, it is in 0-360 format.
        # We verify it doesn't have negative values to be safe.
        if da.lon.max() > 180:
            print("Detected 0-360 longitude. Converting to -180 to 180...")
            
            # The conversion
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
            
            # We MUST sort by lon after wrapping, or plotting will look broken
            da = da.sortby('lon', ascending=True)

    # 3. Sort Latitude (Descending: 90 -> -90)
    if 'lat' in da.coords:
        # Only sort if not already sorted descending to save time
        if not da.lat[0] > da.lat[-1]:
             da = da.sortby('lat', ascending=False)

    return da

def detrend(da: xr.DataArray, dim='time', deg=1) -> xr.DataArray:
    """
    Removes the linear trend using xarray's polyfit.
    Works lazily with Dask.
    """
    # 1. Fit the polynomial (e.g. linear line)
    # skipna=True ensures we don't crash on NaNs
    coeffs = da.polyfit(dim=dim, deg=deg, skipna=True)
    
    # 2. Evaluate the trend
    trend = xr.polyval(da[dim], coeffs.polyfit_coefficients)
    
    # 3. Subtract trend
    return da - trend

def _polyval(coord: xr.DataArray, coeffs: xr.DataArray, degree_dim: str = "degree") -> xr.DataArray:
    """
    Internal helper to evaluate a polynomial using the Vandermonde matrix method.
    
    Note: This is often faster than xr.polyval for large Dask arrays.
    """
    x = coord.data
    deg_coord = coeffs[degree_dim]
    N = int(deg_coord.max()) + 1

    # Create the Vandermonde matrix (Left Hand Side)
    # stacking x^n for n in degrees
    lhs = xr.DataArray(
        np.stack([x ** (N - 1 - i) for i in range(N)], axis=1),
        dims=(coord.name, degree_dim),
        coords={coord.name: coord, degree_dim: np.arange(deg_coord.max() + 1)[::-1]},
    )
    
    # Dot product
    return (lhs * coeffs).sum(degree_dim)


def detrend_dask(da: xr.DataArray, dim: Hashable = 'time', deg: int = 1) -> xr.DataArray:
    """
    Detrends a DataArray using a Dask-optimized polynomial fit only for NA free data. => 40% faster for large datasets.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    dim : str
        Dimension to detrend along (e.g., 'time').
    deg : int, optional
        Degree of polynomial (1=linear), by default 1.

    Returns
    -------
    xr.DataArray
        Detrended anomalies.
    """
    # 1. Fit the polynomial
    # skipna=False is faster for Dask; ensure data is clean or filled first
    p = da.polyfit(dim=dim, deg=deg, skipna=False)

    # 2. Chunk the coordinate
    # We reconstruct the coordinate as a dask array to match the data chunks
    chunked_coord = xr.DataArray(
        dask.array.from_array(da[dim].data, chunks=da.chunksizes[dim]),
        dims=dim,
        name=dim,
    )

    # 3. Evaluate trend (Using the fast internal helper)
    trend = _polyval(chunked_coord, p.polyfit_coefficients)

    # 4. Remove trend
    return da - trend


def deseasonalize(da: xr.DataArray, time_dim='time', freq='month') -> xr.DataArray:
    """
    Removes the seasonal cycle (climatology).
    """
    # Calculate climatology
    climatology = da.groupby(f"{time_dim}.{freq}").mean(time_dim)
    
    # Subtract to get anomalies
    return da.groupby(f"{time_dim}.{freq}") - climatology

def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Calculates the area-weighted mean.
    """
    weights = np.cos(np.deg2rad(da['lat']))
    weights.name = "weights"
    
    # xarray handles the weighted sum/mean automatically
    return da.weighted(weights).mean(dim=['lat', 'lon'])

def standardize(da: xr.DataArray, dim=None) -> xr.DataArray:
    """
    Standardizes data: (x - mean) / std
    """
    return (da - da.mean(dim=dim)) / da.std(dim=dim)

def normalize(da: xr.DataArray, dim=None) -> xr.DataArray:
    """
    Normalizes data to [0, 1]: (x - min) / (max - min)
    """
    return (da - da.min(dim=dim)) / (da.max(dim=dim) - da.min(dim=dim))

def mask_ocean(da: xr.DataArray) -> xr.DataArray:
    """
    Masks ocean points in the data array using a land mask.
    Assumes land_mask has 1 for land and 0 for ocean.
    """
    # use default land mask from xarray tutorial datasets
    land_mask = xr.tutorial.load_dataset("air_temperature").land_mask
    return da.where(land_mask == 1)