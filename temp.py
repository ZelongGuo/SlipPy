#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:32:47 2023
@author: zelong
"""
# import struct
# import numpy as np
# import os

from data import parse_files, deramp_dem

# # -----------------------------------------------------------------------------------
# # the image is unwrapped phase
file_name = "/Users/zelong/testspace/2017_Iran_Coseismic/EQ_20171112_T72A/UNW/mcf/20171111_20171117.unw_utm"
dem_par = "/Users/zelong/testspace/2017_Iran_Coseismic/EQ_20171112_T72A/SIM/20171111_20171117.utm.dem.par"
dem_name = "/Users/zelong/testspace/2017_Iran_Coseismic/EQ_20171112_T72A/SIM/20171111_20171117.utm.dem"

# file_name = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T79D/UNW/mcf/20171112_20171124.unw_utm"
# dem_par = "/misc/zs7/Zelong/EQ_DInSAR/EQ_20171112_T79D/SIM/20171112_20171124.utm.dem.par"
# -----------------------------------------------------------------------------------

# get the data and parameters firstly
# paras = parse_files.get_image_para(dem_par)
unw, paras = parse_files.get_image_data(file_name, dem_par)
dem, _ = parse_files.get_image_data(dem_name, dem_par)

# figures plotting and saving
parse_files.plot_image(unw, paras, "umw_insar_phase_full", "insar_phase")
parse_files.plot_image(dem, paras, 'dem_full', 'dem')
parse_files.plot_image_geo(unw, paras, 'geo_unw_insar_phase_full', 'insar_phase')
parse_files.plot_dem_geo(unw, dem, paras, 'dem_geo')

# ###################### re-check the deramp_dem, you should remove the 0 vaules to nan before deramping
# # deramp and remove dem-raleted error if needed, it returns los (m)
# # --------------------
# mask = [[1200, 4500],
#         [4000, 7200]]  # height, width
#
# deramp_method = 3
# # --------------------
# los = image_preprocessing.deramp_dem(unw, paras, dem, mask, 6, 1, "sentinel")
#
#
# # if no need, then phase to los: phase2los
# # phase to LOS
# unw_los = image_preprocessing.phase2los(unw, paras, 'sentinel', 1)  # unit m
#
# # resampling if needed
# resample_los, resample_paras = image_preprocessing.resample_image(unw_los, paras, 3, 1, "insar_los")
# parse_files.plot_image_geo(resample_los, resample_paras, 'insar_los')

