#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions for parsing files and visualization:
    get_image_para (.par files, e.g., .dem.par to get the image parameters from GAMMA)
    get image data (deformation images to get the deformation phase from GAMMA)
    plot_image_geo
    plot_dem_geo
    plot_image
    
Created on Tue May 18 20:08:23 2023

@author: zelong
"""
import sys
import numpy as np
import struct
import matplotlib.pyplot as plt



#--------------------------------------------------------------------------------------------
def get_image_para(para_file):
    """
    function: read the parameter files.
    Parameters
    ----------
    para_file : text file 
        parameter files like *.utm.dem.par

    Returns: the width, nlines and lon lat info of the InSAR image
    -------

    Update log:
    12.06.2023, update the return values as dic
    """
    try:
        with open(para_file, 'r') as file:
            print("-------------------------------------------------------------")
            print("Now we are reading the parameter file...")
            print("-------------------------------------------------------------")
            for line in file:
                
               # line = line.strip()
               # print(line)
               
                if line.startswith('width:'):
                    width = int(line.strip().split(':')[1])
                    print(line)
                elif line.startswith('nlines:'):
                    nlines = int(line.strip().split(':')[1])
                    print(line)
                elif line.startswith('corner_lat:'):
                    corner_lat = float(line.strip().split(':')[1].split()[0])
                    print(line)
                elif line.startswith('corner_lon:'):
                    corner_lon = float(line.strip().split(':')[1].split()[0])
                    print(line)
                elif line.startswith('post_lat:'):
                    post_lat = float(line.strip().split(':')[1].split()[0])
                    print(line) 
                elif line.startswith('post_lon:'):
                    post_lon = float(line.strip().split(':')[1].split()[0])
                    print(line)
                else:
                    pass
        post_arc = post_lon * 3600 # to arcsecond
        post_utm = post_arc * 40075017 / 360 / 3600  # earth circumference to ground resolution, meter
        # post_arc2 = "{:.2f}".format(post_arc)
        # post_utm2 = "{:.2f}".format(post_utm)
        print("-------------------------------------------------------------")
        # print("The InSAR pixel resoluton is {} arc-second, ~{} meters." .format(post_arc2, post_utm2))
        print("The InSAR pixel resoluton is %f arc-second, ~%f meters." %(post_arc, post_utm))
        print("-------------------------------------------------------------")

        # return [width, nlines, corner_lat, corner_lon, post_lat, post_lon, post_arc, post_utm]
        return {
            'width': width,
            'nlines': nlines,
            'corner_lat': corner_lat,
            'corner_lon': corner_lon,
            'post_lat': post_lat,
            'post_lon': post_lon,
            'post_arc': post_arc,
            'post_utm': post_utm
        }
                    
    except IOError:
        print("Error: cannot open the parameter file, please check the file path!")

#--------------------------------------------------------------------------------------------
def get_image_data(image_file, para_file, swap_bytes = "big-endian"):
    """
    Read the InSAR images (and DEM files if needed) From GAMMA.

    Parameters
    ----------
    image_file : big-endian files of insar images or dem from GAMMA

    para_file : the image parameters from get_image_para
    
    swap_bytes : The default is "big-endian" (only big-endian is supported now) .

    Returns
    -------
    image_arry : array of imges for the follwing processing.
    
    parameters : parameter from get_image_para

    """
    
    parameters = get_image_para(para_file)
    
    range_samples = parameters['width']  # width
    azimuth_lines = parameters['nlines']  # nlines
   
    
    try:
        with open(image_file, 'rb') as file:
            image_arry = np.zeros([range_samples, azimuth_lines])
            # need read in column
            print("-------------------------------------------------------------")
            print("Now we are reading the phase images (binary files)...")
            print(f"Total {azimuth_lines}:")

            for i in range(azimuth_lines):
                if i % 500 == 0:
                    # print(f"{i} ", end = '\r')
                    sys.stdout.write(f"{i} ")
                    sys.stdout.flush()
                for j in range(range_samples):
                    # >f, big-endian, 4 bytes float
                    chunk = file.read(4)
                    image_arry[j][i] = struct.unpack('>f', chunk)[0]
            
        image_arry = image_arry.transpose()
        
                
        return image_arry, parameters
        
    except IOError: 
       print("Error: cannot open the image file, please check the file path or the parameters!")        

#--------------------------------------------------------------------------------------------
def plot_image_geo(image_data, parameters, data_flag = "insar_phase"):
    """
    Plot geocoded InSAR images.

    Parameters
    ----------
    image_data : Image matrix (phase or los) reading from get_image_data.
        
    parameters : parameter from get_image_para
        
    data_flag : insar_phase or insar_los (the unit should be m)
    The default is "insar_phase".

    Returns
    -------
    None.

    """
    range_samples = parameters['width'] # width
    azimuth_lines = parameters['nlines'] # nlines
    corner_lat = parameters['corner_lat']
    corner_lon = parameters['corner_lon']
    post_lat = parameters['post_lat']
    post_lon = parameters['post_lon']
    post_arc = parameters['post_arc']
    post_utm = parameters['post_utm']
    
    print("Quick preview image with geocoding ...")
    print(f"Width: {range_samples}, Height: {azimuth_lines}")
    print("The InSAR pixel resoluton is %f arc-second, ~%f meters." %(post_arc, post_utm))
        
    lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1 ) * post_lat, azimuth_lines)
    lons = np.linspace(corner_lon, corner_lon + (range_samples - 1 ) * post_lon, range_samples)
    
    # make the 0 vlaues to be nan to better plotting
    data2 = np.where(image_data == 0, np.nan, image_data)
    #new_image_arry2 = new_image_arry
    plt.figure()
    plt.imshow(data2, cmap = 'jet', vmin = np.nanmin(data2), vmax = np.nanmax(data2), \
               origin = 'upper', extent= [np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha = 1.0)
    if data_flag == "insar_phase":
        plt.colorbar(label = 'Phase (radians)')
    elif data_flag == "insar_los":
        plt.colorbar(label = 'LOS (m)')
   
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()    
    
def plot_dem_geo(image_data, dem, parameters):
    """
    Plot geocoded DEM.

    Parameters
    ----------
    image_data : Image matrix reading from get_image_data.
    
    dem: dem matrix reading from get_image_data.
        
    parameters : parameter from get_image_para
         
    Returns
    -------
    None.

    """

    range_samples = parameters['width'] # width
    azimuth_lines = parameters['nlines'] # nlines
    corner_lat = parameters['corner_lat']
    corner_lon = parameters['corner_lon']
    post_lat = parameters['post_lat']
    post_lon = parameters['post_lon']
    post_arc = parameters['post_arc']
    post_utm = parameters['post_utm']
    
    print("Quick preview DEM with geocoding ...")
    print(f"Width: {range_samples}, Height: {azimuth_lines}")
    print("The InSAR pixel resoluton is %f arc-second, ~%f meters." %(post_arc, post_utm))
        
    lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1 ) * post_lat, azimuth_lines)
    lons = np.linspace(corner_lon, corner_lon + (range_samples - 1 ) * post_lon, range_samples)
    
    # make the 0 vlaues to be nan to better plotting
    dem2 = np.where(image_data == 0, np.nan, dem)
    #new_image_arry2 = new_image_arry
    plt.figure()
    plt.imshow(dem2, cmap = 'jet', vmin = np.nanmin(dem2), vmax = np.nanmax(dem2), \
               origin = 'upper', extent= [np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha = 1.0)
       
    plt.colorbar(label = 'Elevation (m)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()    
    
        
    

def plot_image(data, parameters, data_flag = "insar_phase"):
    """
    Plot InSAR images (phase or los) or DEM (no geocoding).

    Parameters
    ----------
    data : Image/dem matrix reading from get_image_data.
        
    parameters : parameter from get_image_para
        
    data_flag : insar_phase, insar_los (the unit should be m) or dem
    The default is "insar_phase".

    Returns
    -------
    None.

    """
    range_samples = parameters['width'] # width
    azimuth_lines = parameters['nlines'] # nlines
    post_arc = parameters['post_arc']
    post_utm = parameters['post_utm']
    
    print("Quick preview image without geocoding ...")
    print(f"Width: {range_samples}, Height: {azimuth_lines}")
    print("The InSAR pixel resoluton is %f arc-second, ~%f meters." %(post_arc, post_utm))
    
    # lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1 ) * post_lat, azimuth_lines)
    # lons = np.linspace(corner_lon, corner_lon + (range_samples - 1 ) * post_lon, range_samples)
    
    # make the 0 vlaues to be nan to better plotting
    # data2 = np.where(data == 0, np.nan, data)
    #new_image_arry2 = new_image_arry
    plt.figure()
    plt.imshow(data, cmap = 'jet', vmin = np.nanmin(data), vmax = np.nanmax(data), \
               origin = 'upper')
    if data_flag == "insar_phase":
        plt.colorbar(label = 'Phase (radians)')
    elif data_flag == "insar_los":
        plt.colorbar(label = 'LOS (m)')
    elif data_flag == "dem":
        plt.colorbar(label = 'Elevation (m)')
    plt.xlabel('X (pixel number)')
    plt.ylabel('Y (pixel number)')
    plt.show()        
    
  

#--------------------------------------------------------------------------------------------




if __name__ == "__main__":
    print("Only functions are defined here.")