#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used for reading the InSAR images.
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
import matplotlib.gridspec as gridspec
from copy import copy
import image_preprocessing

# # -----------------------------------------------------------------------------------
# # the image is unwrapped phase
file_name = "/Users/zelong/testspace/2017_Iran_Coseismic/EQ_20171112_T72A/UNW/mcf/20171111_20171117.unw_utm"
dem_par = "/Users/zelong/testspace/2017_Iran_Coseismic/EQ_20171112_T72A/SIM/20171111_20171117.utm.dem.par"
dem_name = "/Users/zelong/testspace/2017_Iran_Coseismic/EQ_20171112_T72A/SIM/20171111_20171117.utm.dem"

# file_name = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T79D/UNW/mcf/20171112_20171124.unw_utm"
# dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T79D/SIM/20171112_20171124.utm.dem.par"


# file_name = "20171111_20171117.unw_utm"
# width = 7758
# height = 6044
# range_samples = 7758
# azimuth_lines = 6044

# dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T174A/FINAL/20171106_20171118.utm.dem.par"
# -----------------------------------------------------------------------------------



# get the data and parameters firstly
# paras = parse_files.get_image_para(dem_par)
unw, paras = parse_files.get_image_data(file_name, dem_par)
dem, _ = parse_files.get_image_data(dem_name, dem_par)

# parse_files.plot_image(unw, paras, "insar_phase")
# parse_files.plot_image(dem, paras, 'dem')

# parse_files.plot_image_geo(unw, paras, 'insar_phase')
# parse_files.plot_dem_geo(unw, dem, paras)

# deramp and remove dem-raleted error if needed, it returns los (m)
# --------------------
mask = [[1200, 4500],
        [4000, 7200]]  # height, width

deramp_method = 3
# --------------------
los = image_preprocessing.deramp_dem(unw, paras, dem, mask, 6, 1, "sentinel")


# if no need, then phase to los: phase2los
# phase to LOS
unw_los = image_preprocessing.phase2los(unw, paras, 'sentinel', 1)  # unit m

# resampling if needed
resample_los, resample_paras = image_preprocessing.resample_image(los, paras, 3, 1, "insar_los")
parse_files.plot_image_geo(resample_los, resample_paras, 'insar_los')

