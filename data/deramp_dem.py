#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Zelong Guo, @ GFZ, Potsdam
Email: zelong.guo@outlook.com
Created on Sat May 20 20:29:30 2023

#---------------------------------------------------------------------------------------------
This script defines some functions for pre-processing of InSAR images, including:
    deramp_dem (deramp and remove dem-related errors)

#### NEED TO BE UPDATED ###
"""

# import utm
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
from copy import copy
# import parse_files
from scipy import interpolate


def deramp_dem(phase_data, parameters, dem_data, mask, sig_factor=4, deramp_method=1, satellite="sentinel"):
    """
    Deramping and remove dem-related errors from unwrapping phase data.
    Remove orbit errors by applying a best-fitting polynomial to individual interferogram.
    Return LOS image.

    Parameters
    ----------
    phase_data : insar phase data from get_image_data*
    parameters :
        related info from get_image_para*.
    dem_data : dem data from get_image_data
    mask : mask matrix. Height, Width
        [[azimuth_begin, azimuth_end],
         [range_begin, range_end]]
    sig_factor : you may need trial and error.
                The default is 4.
    deramp_method :
        1: a + bx + cy + d*dem (default)
        2: a + bx + cy + dxy + e*dem
        3: a + bx + cy + dxy + ex^2 + fy^2 + g*dem
    satellite : to calculate the los deformation
        The default is sentinel
        satellite type: sentinel, alos ...

    Returns
    -------
    Flattened LOS image (unit m)

    """
    unw_los = phase2los(phase_data, parameters, satellite, 0)  # unit m
    unw_mask = copy(unw_los)
    dem_mask = copy(dem_data)

    # -----------------------------------------------
    unw_mask = np.where(unw_mask == 0, np.nan, unw_mask)
    dem_mask = np.where(unw_mask == 0, np.nan, dem_mask)
    # -----------------------------------------------

    unw_mask[mask[0][0]:mask[0][1], mask[1][0]:mask[1][1]] = np.nan

    x = unw_los.shape[1]  # width
    y = unw_los.shape[0]  # height
    xx, yy = np.meshgrid(np.arange(0, unw_los.shape[1]), np.arange(0, unw_los.shape[0]))

    xx_mask = np.where(np.isnan(unw_mask), np.nan, xx)
    yy_mask = np.where(np.isnan(unw_mask), np.nan, yy)
    dem_mask = np.where(np.isnan(unw_mask), np.nan, dem_data)

    unw_mask2 = unw_mask.flatten()
    unw_mask2 = unw_mask2[~np.isnan(unw_mask2)]
    xx_mask2, yy_mask2 = xx_mask.flatten(), yy_mask.flatten()
    xx_mask2, yy_mask2 = xx_mask2[~np.isnan(xx_mask2)], yy_mask2[~np.isnan(yy_mask2)]
    dem_mask2 = dem_mask.flatten()
    dem_mask2 = dem_mask2[~np.isnan(dem_mask2)]

    if deramp_method == 1:
        A = np.column_stack((np.ones(len(xx_mask2)), xx_mask2, yy_mask2, dem_mask2))
    elif deramp_method == 2:
        A = np.column_stack((np.ones(len(xx_mask2)), xx_mask2, yy_mask2, xx_mask2 * yy_mask2, dem_mask2))
    elif deramp_method == 3:
        A = np.column_stack((np.ones(len(xx_mask2)), xx_mask2, yy_mask2, xx_mask2 * yy_mask2, \
                             xx_mask2 ** 2, yy_mask2 ** 2, dem_mask2))

    print("The iterations needed:")
    while True:
        params = np.linalg.lstsq(A, unw_mask2, rcond=None)[0]
        # [params, v] = np.linalg.lstsq(A, unw_mask2, rcond = None)[0:1]
        v = np.dot(A, params) - unw_mask2
        sigma = np.sqrt(np.dot(v.T, v) / len(unw_mask2))
        index = np.where(np.abs(v) > sig_factor * sigma)[0]

        print(f"{len(index)} ", end="\r")
        if len(index) == 0:
            break
        else:
            A = np.delete(A, index, axis=0)
            unw_mask2 = np.delete(unw_mask2, index)

    if deramp_method == 1:
        unw_flat = unw_los - params[0] * np.ones((y, x)) - params[1] * xx - params[2] * yy - params[3] * dem_data
        aps = params[3] * dem_data
        ramp = params[0] * np.ones((y, x)) + params[1] * xx + params[2] * yy
    elif deramp_method == 2:
        unw_flat = unw_los - params[0] * np.ones((y, x)) - params[1] * xx - \
                   params[2] * yy - params[3] * xx * yy - params[4] * dem_data
        aps = params[4] * dem_data
        ramp = params[0] * np.ones((y, x)) + params[1] * xx + params[2] * yy + \
               params[3] * xx * yy
    else:
        unw_flat = unw_los - params[0] * np.ones((y, x)) - params[1] * xx - \
                   params[2] * yy - params[3] * xx * yy - params[4] * xx ** 2 - params[5] * yy ** 2 - \
                   params[6] * dem_data
        aps = params[6] * dem_data
        ramp = params[0] * np.ones((y, x)) + params[1] * xx + params[2] * yy + \
               params[3] * xx * yy + params[4] * xx ** 2 + params[5] * yy ** 2

    # parse_files.plot_image(aps, paras, "insar_phase")
    plt.figure()
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # fig, axs = plt.subplots(2, 3, figsize = (16, 10))

    im1 = axs[0, 0].imshow(unw_mask, cmap='jet', vmin=np.nanmin(unw_mask), vmax=np.nanmax(unw_mask))
    axs[0, 0].set_title("Original Masked Image")
    cbar1 = plt.colorbar(im1, ax=axs[0, 0])
    cbar1.set_label('Defornation (m)')
    print("Plotting image 1...")

    im2 = axs[0, 1].imshow(unw_los, cmap='jet', vmin=np.nanmin(unw_los), vmax=np.nanmax(unw_los))
    axs[0, 1].set_title("Original Image")
    cbar2 = plt.colorbar(im2, ax=axs[0, 1])
    cbar2.set_label('Defornation (m)')
    print("Plotting image 2...")

    im3 = axs[0, 2].imshow(unw_flat, cmap='jet', vmin=np.nanmin(unw_los), vmax=np.nanmax(unw_los))
    axs[0, 2].set_title("Flatted Image")
    cbar3 = plt.colorbar(im3, ax=axs[0, 2])
    cbar3.set_label('Defornation (m)')
    print("Plotting image 3...")

    im4 = axs[1, 0].imshow(ramp, cmap='jet', vmin=np.nanmin(ramp), vmax=np.nanmax(ramp))
    axs[1, 0].set_title("Ramp")
    cbar4 = plt.colorbar(im4, ax=axs[1, 0])
    cbar4.set_label('Deformation (m)')
    print("Plotting image 4...")

    im5 = axs[1, 1].imshow(aps, cmap='jet', vmin=np.nanmin(aps), vmax=np.nanmax(aps))
    axs[1, 1].set_title("DEM-related Errors")
    cbar5 = plt.colorbar(im5, ax=axs[1, 1])
    cbar5.set_label('Deformation (m)')
    print("Plotting image 5...")

    im6 = axs[1, 2].imshow(dem_data, cmap='jet', vmin=np.nanmin(dem_data), vmax=np.nanmax(dem_data))
    axs[1, 2].set_title("DEM")
    cbar6 = plt.colorbar(im6, ax=axs[1, 2])
    cbar6.set_label('Elevation (m)')
    print("Plotting image 6...")

    # axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    unw_flat = np.where(phase_data == 0, 0, unw_flat)
    return unw_flat
