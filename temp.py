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
file_path = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T174A/UNW/mcf/"
file_name = "20171106_20171118.unw_utm"
# width = 7758
# height = 6044
# range_samples = 7758
# azimuth_lines = 6044

#dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T174A/FINAL/20171106_20171118.utm.dem.par"
# -----------------------------------------------------------------------------------


dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T174A/SIM/20171106_20171118.utm.dem.par"
#width, nlines, corner_lat, corner_lon, post_lat, post_lon = parse_files.get_image_para(dem_par)

paras = parse_files.get_image_para(dem_par)
data, para2 = parse_files.get_image_data(file_path + file_name, paras, 3, 1)

#data, para2 = parse_files.get_image_data(file_path + file_name, paras)


# fa = 3
# range_samples, azimuth_lines = 8213, 6132
# x = range_samples // fa
# y = azimuth_lines // fa

# rows, cols = np.arange(0, range_samples, 1), np.arange(0, azimuth_lines, 1)

# interp_func = interpolate.interp2d(cols, rows, data, kind='cubic')

# new_rows, new_cols = np.arange(0, range_samples, fa), np.arange(0, azimuth_lines, fa)

# new_data = interp_func(new_cols, new_rows)



# plt.imshow(data, cmap = 'jet')
# plt.show()

# plt.imshow(new_data, cmap='jet')
# plt.show()
