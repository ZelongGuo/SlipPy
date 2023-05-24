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
import image_preprocessing

# # -----------------------------------------------------------------------------------
# # the image is unwrapped phase
file_name = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T72A/UNW/mcf/20171111_20171117.unw_utm"
dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T72A/SIM/20171111_20171117.utm.dem.par"
dem_name = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T72A/SIM/20171111_20171117.utm.dem"

# file_name = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T79D/UNW/mcf/20171112_20171124.unw_utm"
# dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T79D/SIM/20171112_20171124.utm.dem.par"


# file_name = "20171111_20171117.unw_utm"
# width = 7758
# height = 6044
# range_samples = 7758
# azimuth_lines = 6044

#dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T174A/FINAL/20171106_20171118.utm.dem.par"
# -----------------------------------------------------------------------------------





# get the data and parameters firstly
#paras = parse_files.get_image_para(dem_par)
unw, paras = parse_files.get_image_data(file_name, dem_par)
dem, _ = parse_files.get_image_data(dem_name, dem_par)

# deramp and remove dem-raleted error if needed


# resampling and los2phase









# #-------------------------------------------------------------------------------------
# unw = los
# x, y = np.meshgrid(lons, lats)

# unw_mask = unw

# # mask = [[45.2, 47],
# #         [34.2, 35.2]] # min_lon, max_lon, min_lat, max_lat

# # --------------------
# mask = [[250, 1500],
#         [1200, 2500]]
# # --------------------

# unw_mask[mask[0][0]:mask[0][1], mask[1][0]:mask[1][1]] = np.nan

# xx, yy = np.meshgrid(np.arange(0, unw.shape[1]), np.arange(0, unw.shape[0]))

# plt.imshow(unw_mask, cmap = 'jet') #, vmin = np.nanmin(los), vmax = np.nanmax(los), \
#                     #origin = 'upper', extent= [np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha = 1.0)
# plt.colorbar(label = 'Los Deformation (m)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()



















