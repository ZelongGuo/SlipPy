#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Zelong Guo
Email: zelong.guo@outlook.com
Created on Sat May 20 20:29:30 2023

#---------------------------------------------------------------------------------------------
This script defines some functions for pre-processing of InSAR images, including phase to los or
los to phase as well as resampling the images if needed.

"""

# import utm
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
from copy import copy
from data import parse_files
from scipy import interpolate


# ----------------------------------------------------------------------------------------------------------
def phase2los(phase_data, satellite="sentinel"):
    """
    Converting InSAR phase to InSAR line-of-sight (los) disp.

    Parameters
    ----------
    phase_data : 
        the phase data from get_image_data*.

    satellite : TYPE
        The default is sentinel
        satellite type: sentinel, alos ...

    Returns 
    -------
    InSAR LOS deformation field (unit m).

    """

    if satellite == "sentinel" or satellite == "sentinel-1" or satellite == "s1":
        radar_freq = 5.40500045433e9  # Hz
        # wavelength = 0.0555041577 # m
    elif satellite == "ALOS" or satellite == "alos":
        radar_freq = 1.27e9  # Hz
    elif satellite == "ALOS-2/U":
        radar_freq = 1.2575e9
    elif satellite == "ALOS-2/{F,W}":
        radar_freq = 1.2365e9
    else:
        pass

    wavelength = c / radar_freq  # m
    los = - (phase_data / 2 / np.pi * wavelength / 2)

    # the unit is "m"    
    return los


# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
def resample_image(image_data, parameters, resample_factor=1, plot_flag=0, data_flag="insar_phase"):
    """
    Resampling the insar images (phase or los).

    Parameters
    ----------
    image_data : phase or LOS insar images.
    parameters : the image parameters from get_image_para
    resample_factor : optional
        resampling factor, >1 is downsampling, <1 is upsampling. The default is 1.
    plot_flag : TYPE, optional
        plot (1) or not (0). The default is 0.
    data_flag: "insar_phase" or "insar_los".

    Returns
    -------
    TYPE
        InSAR image data.
    list
        related info of the image.

    """

    range_samples = parameters['width']  # width
    azimuth_lines = parameters['nlines']  # nlines
    corner_lat = parameters['corner_lat']
    corner_lon = parameters['corner_lon']
    post_lat = parameters['post_lat']
    post_lon = parameters['post_lon']
    post_arc = parameters['post_arc']
    post_utm = parameters['post_utm']

    # resample the image or not, if not, do nothing
    if resample_factor == 1:
        print("-------------------------------------------------------------")
        print("Here you choose do not to resample the images.")
        print("The InSAR pixel resolution is %f arc-second, ~%f meters." % (post_arc, post_utm))
        print("-------------------------------------------------------------")

        # make the 0 values to be nan
        # image_data2 = np.where(image_data == 0, np.nan, image_data)
        if plot_flag != 0:
            print("Quick preview image is generated ...")
            parse_files.plot_image(image_data, parameters, data_flag)

        return image_data, [range_samples, azimuth_lines, corner_lat, corner_lon, post_lat, \
                            post_lon, post_arc, post_utm]

    else:
        # new_range_samples = range_samples // resample_factor
        # new_azimuth_lines = azimuth_lines // resample_factor
        new_post_lat, new_post_lon = post_lat * resample_factor, post_lon * resample_factor
        new_post_arc, new_post_utm = post_arc * resample_factor, post_utm * resample_factor

        print("-------------------------------------------------------------")
        print("Here you choose resample the image with a factor of %d." % resample_factor)
        print("The pixel resoluton of resampled InSAR image is %f arc-second, ~%f meters." \
              % (new_post_arc, new_post_utm))
        print("-------------------------------------------------------------")

        # create the rows and cols
        rows, cols = np.arange(0, azimuth_lines, 1), np.arange(0, range_samples, 1)
        # linear interpolation
        interp_func = interpolate.interp2d(cols, rows, image_data, kind='linear')

        new_rows, new_cols = np.arange(0, azimuth_lines, resample_factor), \
            np.arange(0, range_samples, resample_factor)

        new_image_arry = interp_func(new_cols, new_rows)

        new_azimuth_lines = new_image_arry.shape[0]
        new_range_samples = new_image_arry.shape[1]

        new_parameters = [new_range_samples, new_azimuth_lines, corner_lat, corner_lon, new_post_lat, \
                          new_post_lon, new_post_arc, new_post_utm]
        new_image_arry2 = np.where(new_image_arry == 0, np.nan, new_image_arry)
        if plot_flag != 0:
            print("Quick preview image is generated ...")
            parse_files.plot_image(new_image_arry, new_parameters, data_flag)

            # lats = np.linspace(corner_lat, corner_lat + (new_azimuth_lines - 1 ) * new_post_lat, new_azimuth_lines)
            # lons = np.linspace(corner_lon, corner_lon + (new_range_samples - 1 ) * new_post_lon, new_range_samples)

            # # make the 0 vlaues to be nan to better plotting
            # new_image_arry2 = np.where(new_image_arry == 0, np.nan, new_image_arry)
            # #new_image_arry2 = new_image_arry
            # plt.imshow(new_image_arry2, cmap = 'jet', vmin = np.nanmin(new_image_arry2), vmax = np.nanmax(new_image_arry2), \
            #            origin = 'upper', extent= [np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha = 1.0)
            # plt.colorbar(label = 'Phase (radians)')
            # plt.xlabel('Longitude')
            # plt.ylabel('Latitude')
            # plt.show()    

        return new_image_arry2, new_parameters


# --------------------------------------------------------------------------------------------
def get_ll(parameters):
    range_samples = parameters[0]  # width
    azimuth_lines = parameters[1]  # nlines
    corner_lat = parameters[2]
    corner_lon = parameters[3]
    post_lat = parameters[4]
    post_lon = parameters[5]
    # post_arc = parameters[6]
    # post_utm = parameters[7]

    lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1) * post_lat, azimuth_lines)
    lons = np.linspace(corner_lon, corner_lon + (range_samples - 1) * post_lon, range_samples)

    return lats, lons

# def ll2utm(lats, lons):
#     """
#     The default is WGS84 system and its zone number.
#
#     """
#     # lats and lons shouls have the same dimensions
#     utm_easting, utm_northing, utm_zone, _ = utm.from_latlon(lats.flatten(), lons.flatten())
#     # _, _, _, utm_zone_letter = utm.from_latlon(latitude, longitude)
#     utm_easting = utm_easting.reshape(lats.shape)
#     utm_northing = utm_northing.reshape(lons.shape)
#     utm_zone = utm_zone.reshape(utm_zone.shape)
#     return utm_easting, utm_northing, utm_zone
#
# # -----------------------------------------------------------------------------
# # need to futher modification
# def utm2ll(utm_easting, utm_northing, utm_zone, utm_zone_letter):
#     lats, lons = utm.to_latlon(utm_easting, utm_northing, utm_zone, utm_zone_letter)
#     return lats, lons
#     pass
# # -----------------------------------------------------------------------------
