#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:01:35 2023

reading raster .tif and then interpolating

@author: zelong
"""

import rasterio
import numpy as np
from scipy.interpolate import RegularGridInterpolator

filename = '/misc/zs7/Zelong/shared/zdt.tif'
resample_factor = 3
output_file = "/misc/zs7/Zelong/shared/new_zdt.tif"

with rasterio.open(filename) as ds:
    print("The raster information:")
    print(f"Data type: {ds.driver}")
    print(f"Bind number: {ds.count}")
    print(f"Width: {ds.width}")
    print(f"Height: {ds.height}")
    print(f"Geopgysical range: {ds.bounds}")
    print(f"Parameters: {ds.transform}")
    print(f"Projection: {ds.crs}")
    
    data = ds.read(1) # read the 1st bind
    transform = ds.transform
    crs = ds.crs
    
    lats = [];
    lons = [];
    for row in range(ds.height):
        for col in range(ds.width):
            lon, lat = ds.xy(row, col, offset='center')
            lats.append(lat)
            lons.append(lon)

lon_grid, lat_grid = np.meshgrid(np.unique(lons),np.unique(lats))

interp_func = RegularGridInterpolator((np.unique(lats), np.unique(lons)), data)

new_lats = np.linspace(min(lats), max(lats), data.shape[1] * resample_factor)
new_lons = np.linspace(min(lons), max(lons), data.shape[0] * resample_factor)
new_lon_grid, new_lat_grid = np.meshgrid(new_lons, new_lats)

new_data = interp_func((new_lat_grid, new_lon_grid))    

# # interpolatation

# interp_func = interp2d(np.arange(data.shape[1]), np.arange(data.shape[0]), data, kind = 'cubic')

# new_x = np.linspace(0, data.shape[1], data.shape[1] * resample_factor)
# new_y = np.linspace(0, data.shape[0], data.shape[0] * resample_factor)

# new_data = interp_func(new_x, new_y)

# output a new tif
with rasterio.open(output_file, 'w', driver = 'GTiff', width = new_data.shape[1], \
                    height = new_data.shape[0], count = 1, dtype = new_data.dtype, crs = crs, \
                        transform = transform) as dst:
    dst.write(new_data, 1)
    
print(f"The new data is saved as {output_file}")
    