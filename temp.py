#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used for reading the InSAR iamges.
Reading the images output by unwrapping (like mcf) of GAMMA:
    1. big endian
    2. phase data
The InSAR images processed by Gamma are binary files.

Created on Thu May 18 15:32:47 2023

@author: zelong
"""
# import struct
# import numpy as np
# import os


import parse_files
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

# # -----------------------------------------------------------------------------------
# # the image is unwrapped phase
# file_name = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T72A/UNW/mcf/20171111_20171117.unw_utm"
# dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T72A/SIM/20171111_20171117.utm.dem.par"

file_name = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T79D/UNW/mcf/20171112_20171124.unw_utm"
dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T79D/SIM/20171112_20171124.utm.dem.par"


# file_name = "20171111_20171117.unw_utm"
# width = 7758
# height = 6044
# range_samples = 7758
# azimuth_lines = 6044

#dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T174A/FINAL/20171106_20171118.utm.dem.par"
# -----------------------------------------------------------------------------------



#width, nlines, corner_lat, corner_lon, post_lat, post_lon = parse_files.get_image_para(dem_par)

paras = parse_files.get_image_para(dem_par)
data, para2 = parse_files.get_image_data(file_name, paras, 1, 1)
los = parse_files.phase2los(data, para2, 'sentinel', 1)

#data, para2 = parse_files.get_image_data(file_path + file_name, paras)

#-------------------------------------------------------------------------------------

# range_samples = para2[0] # width
# azimuth_lines = para2[1] # nlines
# corner_lat = para2[2] 
# corner_lon = para2[3]
# post_lat = para2[4]
# post_lon = para2[5]
# post_arc = para2[6]
# post_utm = para2[7]
    
# lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1) * post_lat, azimuth_lines)
# lons = np.linspace(corner_lon, corner_lon + (range_samples - 1) * post_lon, range_samples)

# # make the 0 vlaues to be nan to better plotting
# data = np.where(data == 0, np.nan, data) 

# plt.imshow(data, cmap = 'jet', vmin = np.nanmin(data), vmax = np.nanmax(data), origin = 'upper', \
#             extent= [np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha = 1.0)
# plt.colorbar(label = 'Deformation (phase)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()    
