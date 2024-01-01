#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Zelong Guo, @ GFZ, Potsdam
Email: zelong.guo@outlook.com
Created on 05.11.23

#---------------------------------------------------------------------------------------------
This script consists of several functions which are used for visualization, including plotting
the InSAR images and DEM.

"""
import os.path
import numpy as np
import matplotlib.pyplot as plt


def check_folder4figs(fig_name, folder_name='img'):
    """
    This function will return a path for figures saving.
    The default path is the /img of current folder.

    :param fig_name to save
    :param folder_name: default is img
    :return: fig_path for figures saving
    """
    # get the path of current folder
    current_path = os.getcwd()
    # check the folder's existence
    folder_path = os.path.join(current_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fig_path = os.path.join(folder_path, fig_name)
    print(f"The figures would be saved at {fig_path}...")

    return fig_path


# --------------------------------------------------------------------------------------------
def plot_image_geo(image_data, parameters, fig_name, data_flag="insar_phase"):
    """
    Plot geocoded InSAR images.

    Parameters
    ----------
    image_data : Image matrix (phase or los) reading from get_image_data.

    parameters : parameter from get_image_para

    fig_name : figure name for saving

    data_flag : insar_phase or insar_los (the unit should be m)
    The default is "insar_phase".

    Returns
    -------
    None.

    """
    range_samples = parameters['width']  # width
    azimuth_lines = parameters['nlines']  # nlines
    corner_lat = parameters['corner_lat']
    corner_lon = parameters['corner_lon']
    post_lat = parameters['post_lat']
    post_lon = parameters['post_lon']
    post_arc = parameters['post_arc']
    post_utm = parameters['post_utm']

    print("Quick preview image with geocoding ...")
    print(f"Width: {range_samples}, Height: {azimuth_lines}")
    print("The InSAR pixel resoluton is %f arc-second, ~%f meters." % (post_arc, post_utm))

    lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1) * post_lat, azimuth_lines)
    lons = np.linspace(corner_lon, corner_lon + (range_samples - 1) * post_lon, range_samples)

    # make the 0 vlaues to be nan to better plotting
    data2 = np.where(image_data == 0, np.nan, image_data)
    # new_image_arry2 = new_image_arry

    # plt.figure()
    plt.ioff()
    plt.imshow(data2, cmap='jet', vmin=np.nanmin(data2), vmax=np.nanmax(data2), \
               origin='upper', extent=[np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha=1.0)
    if data_flag == "insar_phase":
        plt.colorbar(label='Phase (radians)')
    elif data_flag == "insar_los":
        plt.colorbar(label='LOS (m)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # plt.show()
    # save figs to /img
    fig_path = check_folder4figs(fig_name) + '.png'
    plt.savefig(fig_path)
    plt.close()
    print(f'Now you can check the figure at {fig_path}')


# -------------------------------------------------------------------------------------------
def plot_dem_geo(image_data, dem, parameters, fig_name):
    """
    Plot geocoded DEM.

    Parameters
    ----------
    image_data : Image matrix reading from get_image_data.

    dem: dem matrix reading from get_image_data.

    parameters : parameter from get_image_para

    fig_name : figure name for saving

    Returns
    -------
    None.

    """

    range_samples = parameters['width']  # width
    azimuth_lines = parameters['nlines']  # nlines
    corner_lat = parameters['corner_lat']
    corner_lon = parameters['corner_lon']
    post_lat = parameters['post_lat']
    post_lon = parameters['post_lon']
    post_arc = parameters['post_arc']
    post_utm = parameters['post_utm']

    print("Quick preview DEM with geocoding ...")
    print(f"Width: {range_samples}, Height: {azimuth_lines}")
    print("The InSAR pixel resoluton is %f arc-second, ~%f meters." % (post_arc, post_utm))

    lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1) * post_lat, azimuth_lines)
    lons = np.linspace(corner_lon, corner_lon + (range_samples - 1) * post_lon, range_samples)

    # make the 0 vlaues to be nan to better plotting
    dem2 = np.where(image_data == 0, np.nan, dem)
    # new_image_arry2 = new_image_arry

    # plt.figure()
    plt.ioff()
    plt.imshow(dem2, cmap='jet', vmin=np.nanmin(dem2), vmax=np.nanmax(dem2), \
               origin='upper', extent=[np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha=1.0)

    plt.colorbar(label='Elevation (m)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # plt.show()
    fig_path = check_folder4figs(fig_name) + '.png'
    plt.savefig(fig_path)
    plt.close()
    print(f'Now you can check the figure at {fig_path}')


def plot_image(data, parameters, fig_name, data_flag="insar_phase"):
    """
    Plot InSAR images (phase or los) or DEM (no geocoding).

    Parameters
    ----------
    data : Image/dem matrix reading from get_image_data.

    parameters : parameter from get_image_para

    fig_name : figure name for saving

    data_flag : insar_phase, insar_los (the unit should be m) or dem
    The default is "insar_phase".

    Returns
    -------
    None.

    """
    range_samples = parameters['width']  # width
    azimuth_lines = parameters['nlines']  # nlines
    post_arc = parameters['post_arc']
    post_utm = parameters['post_utm']

    print("Quick preview image without geocoding ...")
    print(f"Width: {range_samples}, Height: {azimuth_lines}")
    print("The InSAR pixel resoluton is %f arc-second, ~%f meters." % (post_arc, post_utm))

    # lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1 ) * post_lat, azimuth_lines)
    # lons = np.linspace(corner_lon, corner_lon + (range_samples - 1 ) * post_lon, range_samples)

    # make the 0 vlaues to be nan to better plotting
    # data2 = np.where(data == 0, np.nan, data)
    # new_image_arry2 = new_image_arry

    # plt.figure()
    plt.ioff()
    plt.imshow(data, cmap='jet', vmin=np.nanmin(data), vmax=np.nanmax(data), origin='upper')
    if data_flag == "insar_phase":
        plt.colorbar(label='Phase (radians)')
    elif data_flag == "insar_los":
        plt.colorbar(label='LOS (m)')
    elif data_flag == "dem":
        plt.colorbar(label='Elevation (m)')
    plt.xlabel('X (pixel number)')
    plt.ylabel('Y (pixel number)')
    # plt.show()
    fig_path = check_folder4figs(fig_name) + '.png'
    plt.savefig(fig_path)
    plt.close()
    print(f'Now you can check the figure at {fig_path}')
