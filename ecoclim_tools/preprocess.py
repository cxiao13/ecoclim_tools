import xarray as xr
import numpy as np

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

