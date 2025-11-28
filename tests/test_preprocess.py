# create test file for all functions in ecoclimate_tools
import ecoclim_tools as et
import numpy as np
import pandas as pd
import xarray as xr
import pytest

def test_format_coords():
    # Create a sample DataArray with longitude 0-360 and latitude -90 to 90
    lon = np.array([0, 90, 180, 270, 360])
    lat = np.array([-90, -45, 0, 45, 90])
    data = xr.DataArray(np.random.rand(5, 5), coords=[('lat', lat), ('lon', lon)])
    
    # Apply format_coords
    formatted = et.format_coords(data)
    
    # Check if lon is now -180 to 180
    assert formatted.lon.min() >= -180 and formatted.lon.max() <= 180
    # Check if lat is sorted descending
    assert formatted.lat[0] > formatted.lat[-1]

def test_detrend():
    # Create a sample DataArray with a linear trend
    time = np.arange(10)
    data = xr.DataArray(2 * time + np.random.rand(10), coords=[('time', time)])
    
    # Apply detrend
    detrended = et.detrend(data)
    
    # Check if the trend is removed (slope should be close to 0)
    slope = np.polyfit(time, detrended.values, 1)[0]
    assert abs(slope) < 1e-5

def test_detrend_dask():
    # Create a sample DataArray with a linear trend
    time = np.arange(10)
    data = xr.DataArray(
        2 * time + np.random.rand(10),
        coords=[('time', time)],
        dims=('time')
    )
    dask_data = data.chunk({'time': -1})

    # Apply detrend
    detrended = et.detrend_dask(dask_data).compute()
    
    # Check if the trend is removed (slope should be close to 0)
    slope = np.polyfit(time, detrended.values, 1)[0]
    assert abs(slope) < 1e-5


def test_deseasonalize():
    # Create a sample DataArray with seasonal cycle
    time = pd.date_range('2000-01-01', periods=24, freq='ME')
    seasonal_cycle = 10 * np.sin(2 * np.pi * (time.month - 1) / 12)
    data = xr.DataArray(seasonal_cycle + np.random.rand(24), coords=[('time', time)])
    
    # Apply deseasonalize
    deseasonalized = et.deseasonalize(data)
    
    # Check if the seasonal cycle is removed (mean of each month should be close to 0)
    monthly_means = deseasonalized.groupby('time.month').mean()
    assert np.all(np.abs(monthly_means.values) < 1e-5)

def test_area_weighted_mean():
    # Create a sample DataArray with lat/lon
    lat = np.array([-90, -45, 0, 45, 90])
    lon = np.array([0, 90, 180, 270, 360])
    data = xr.DataArray(np.random.rand(5, 5), coords=[('lat', lat), ('lon', lon)])
    
    # Apply area_weighted_mean
    mean = et.area_weighted_mean(data)
    
    # Check if the result is a scalar
    assert np.isscalar(mean.values.item())

def test_standardize():
    # Create a sample DataArray
    data = xr.DataArray(np.random.rand(100) * 10 + 5)  # Random data between 5 and 15
    
    # Apply standardize
    standardized = et.standardize(data)
    
    # Check if mean is approximately 0 and std is approximately 1
    assert abs(standardized.mean().values) < 1e-5
    assert abs(standardized.std().values - 1) < 1e-5

def test_normalize():
    # Create a sample DataArray
    data = xr.DataArray(np.random.rand(100) * 20 + 10)  # Random data between 10 and 30
    
    # Apply normalize
    normalized = et.normalize(data)
    
    # Check if min is approximately 0 and max is approximately 1
    assert abs(normalized.min().values) < 1e-5
    assert abs(normalized.max().values - 1) < 1e-5

if __name__ == "__main__":
    pytest.main()