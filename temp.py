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
import matplotlib.gridspec as gridspec
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

parse_files.plot_image(unw, paras, "insar_phase")
parse_files.plot_image(dem, paras, 'dem')

parse_files.plot_image_geo(unw, paras, 'insar_phase')
parse_files.plot_dem_geo(unw, dem, paras)

# deramp and remove dem-raleted error if needed

unw_mask = unw
# --------------------
mask = [[1200, 4500],
        [4000, 7200]] # height, width

deramp_method = 1
# --------------------
unw_mask[mask[0][0]:mask[0][1], mask[1][0]:mask[1][1]] = np.nan

parse_files.plot_image(unw_mask, paras, "insar_phase")


x = unw.shape[1] # width
y = unw.shape[0] # height
xx, yy = np.meshgrid(np.arange(0, unw.shape[1]), np.arange(0, unw.shape[0]))

xx_mask = np.where(np.isnan(unw_mask), np.nan, xx)
yy_mask = np.where(np.isnan(unw_mask), np.nan, yy)
dem_mask = np.where(np.isnan(unw_mask), np.nan, dem)

unw_mask2 = unw_mask.flatten()
unw_mask2 = unw_mask2[~np.isnan(unw_mask2)]
xx_mask2, yy_mask2 = xx_mask.flatten(), yy_mask.flatten()
xx_mask2, yy_mask2 = xx_mask2[~np.isnan(xx_mask2)], yy_mask2[~np.isnan(yy_mask2)]
dem_mask2 = dem_mask.flatten()
dem_mask2 = dem_mask2[~np.isnan(dem_mask2)]

if deramp_method == 1:
    A = np.column_stack(( np.ones(len(xx_mask2)), xx_mask2, yy_mask2, dem_mask2 ))
elif deramp_method == 2:
    A = np.column_stack(( np.ones(len(xx_mask2)), xx_mask2, yy_mask2, xx_mask2 * yy_mask2, dem_mask2))
elif deramp_method == 3:
    A = np.column_stack(( np.ones(len(xx_mask2)), xx_mask2, yy_mask2, xx_mask2 * yy_mask2, \
                        xx_mask2 ** 2, yy_mask2 ** 2, dem_mask2))

print("The iterations needed:")        
while True:
    params = np.linalg.lstsq(A, unw_mask2, rcond = None)[0]
    #[params, v] = np.linalg.lstsq(A, unw_mask2, rcond = None)[0:1]
    v = np.dot(A, params) - unw_mask2
    sigma = np.sqrt(np.dot(v.T, v) / len(unw_mask2))
    index = np.where(np.abs(v) > 4 * sigma)[0]
    print(f"{len(index)} ", end = "\r")
    if len(index) == 0:
        break
    A = np.delete(A, index, axis = 0)
    unw_mask2 = np.delete(unw_mask2, index)

if deramp_method == 1:
    unw_flat = unw - params[0]*np.ones((y, x)) - params[1]*xx - params[2]*yy - params[3]*dem
    aps = params[3]*dem
    ramp = params[0]*np.ones((y, x)) + params[1]*xx + params[2]*yy
elif deramp_method == 2:
    unw_flat = unw - params[0]*np.ones((y, x)) - params[1]*xx - params[2]*yy - params[3]*xx*yy - params[4]*dem
    aps = params[4]*dem
    ramp = params[0]*np.ones((y, x)) + params[1]*xx + params[2]*yy + params[3]*xx*yy
else:
    unw_flat = unw - params[0]*np.ones((y, x)) - params[1]*xx - params[2]*yy - params[3]*xx*yy - params[4]*xx**2 - params[5]*yy**2 - params[6]*dem
    aps = params[6]*dem
    ramp = params[0]*np.ones((y, x)) + params[1]*xx + params[2]*yy + params[3]*xx*yy + params[4]*xx**2 + params[5]*yy**2

# unw_flat[np.isnan(unw)] = np.nan
# aps[np.isnan(unw)] = np.nan
# ramp[np.isnan(unw)] = np.nan

parse_files.plot_image(aps, paras, "insar_phase")

fig, axs = plt.subplots(2, 3, figsize = (12, 8))


im1 = axs[0, 0].imshow(unw_mask, cmap = 'jet', vmin = np.nanmin(unw), vmax = np.nanmax(unw))
axs[0, 0].set_title("Original Masked Image")
cbar1 = plt.colorbar(im1, ax=axs[0, 0])
cbar1.set_label('Phase (radians)')
print("Plotting image 1...")

im2 = axs[0, 1].imshow(unw, cmap = 'jet', vmin = np.nanmin(unw), vmax = np.nanmax(unw))
axs[0, 1].set_title("Original Image")
cbar2 = plt.colorbar(im2, ax=axs[0, 1])
cbar2.set_label('Phase (radians)')
print("Plotting image 2...")

im3 = axs[0, 2].imshow(unw_flat, cmap = 'jet', vmin = np.nanmin(unw), vmax = np.nanmax(unw))
axs[0, 2].set_title("Flatted Image")
cbar3 = plt.colorbar(im3, ax=axs[0, 2])
cbar3.set_label('Phase (radians)')
print("Plotting image 3...")

im4 = axs[1, 0].imshow(ramp, cmap = 'jet', vmin = np.nanmin(ramp), vmax = np.nanmax(ramp))
axs[1, 0].set_title("Ramp")
cbar4 = plt.colorbar(im4, ax=axs[1, 0])
cbar4.set_label('Phase (radians)')
print("Plotting image 4...")

im5 = axs[1, 1].imshow(aps, cmap = 'jet', vmin = np.nanmin(aps), vmax = np.nanmax(aps))
axs[1, 1].set_title("Flatted Image")
cbar5 = plt.colorbar(im5, ax=axs[1, 1])
cbar5.set_label('Phase (radians)')
print("Plotting image 5...")

axs[1, 2].axis('off')

plt.tight_layout()
plt.show()

# fig = plt.figure(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.2], height_ratios=[1, 1])


# ax1 = plt.subplot(gs[0, 0])
# im1 = ax1.imshow(unw_mask, cmap='jet', vmin=np.nanmin(unw), vmax=np.nanmax(unw))
# ax1.set_title("Original Masked Image")

# ax2 = plt.subplot(gs[0, 1])
# im2 = ax2.imshow(unw, cmap='jet', vmin=np.nanmin(unw), vmax=np.nanmax(unw))
# ax2.set_title("Original Image")

# ax3 = plt.subplot(gs[0, 2])
# im3 = ax3.imshow(unw_flat, cmap='jet', vmin=np.nanmin(unw), vmax=np.nanmax(unw))
# ax3.set_title("Flatted Image")

# ax4 = plt.subplot(gs[1, 0])
# im4 = ax4.imshow(ramp, cmap='jet', vmin=np.nanmin(ramp), vmax=np.nanmax(ramp))
# ax4.set_title("Ramp")

# ax5 = plt.subplot(gs[1, 1])
# im5 = ax5.imshow(aps, cmap='jet', vmin=np.nanmin(aps), vmax=np.nanmax(aps))
# ax5.set_title("Flatted Image")


# cax = plt.subplot(gs[:, 2])
# cbar = plt.colorbar(im5, cax=cax)
# cbar.ax.set_ylabel('Phase (radians)')


# ax6 = plt.subplot(gs[1, 2])
# ax6.axis('off')


# plt.tight_layout()

# plt.show()


# plt.imshow(unw_mask, cmap = 'jet') #, vmin = np.nanmin(los), vmax = np.nanmax(los), \
#                     #origin = 'upper', extent= [np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha = 1.0)
# #plt.colorbar(label = 'Los Deformation (m)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# resampling and los2phase




# #-------------------------------------------------------------------------------------
# unw = los
# x, y = np.meshgrid(lons, lats)

# unw_mask = unw

# # mask = [[45.2, 47],
# #         [34.2, 35.2]] # min_lon, max_lon, min_lat, max_lat























