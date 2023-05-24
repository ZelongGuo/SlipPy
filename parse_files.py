#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions for parsing files:
    (1) .par files, e.g., .dem.par to get the image parameters
    (2) deformation images to get the deformation phase
    (3) change the phase to LOS direction
    
Created on Tue May 18 20:08:23 2023

@author: zelong
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.constants import c

def get_image_para(par_file):
    """
    function: read the parameter files.
    Parameters
    ----------
    par_file : text file 
        parameter files like *.utm.dem.par

    Returns: the width, nlines and lon lat info of the InSAR image
    -------
    """
    try:
        with open(par_file, 'r') as file:
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

        return [width, nlines, corner_lat, corner_lon, post_lat, post_lon, post_arc, post_utm]
                    
    except IOError:
        print("Error: cannot open the parameter file, please check the file path!")


def get_image_data(image_file, para_file, swap_bytes = "big-endian"):
    """
    Read the InSAR images (and DEM files if needed).

    Parameters
    ----------
    image_file : big-endian files of insar images or dem from GAMMA

    para_file : the image parameters from get_image_para
        
    swap_bytes : The default is "big-endian" (only big-endian is supported now) .

    Returns
    -------
    image_arry : array of imges for the follwing processing.

    """
    
    parameters = get_image_para(para_file)
    
    range_samples = parameters[0] # width
    azimuth_lines = parameters[1] # nlines
    # corner_lat = parameters[2] 
    # corner_lon = parameters[3]
    # post_lat = parameters[4]
    # post_lon = parameters[5]
    # post_arc = parameters[6]
    # post_utm = parameters[7]
    
    try:
        with open(image_file, 'rb') as file:
            image_arry = np.zeros([range_samples, azimuth_lines])
            # need read in column
            print("-------------------------------------------------------------")
            print("Now we are reading the phase images (binary files)...")
            print(f"Total {azimuth_lines}:")

            for i in range(azimuth_lines):
                print(f"{i} ", end = '\r')
                for j in range(range_samples):
                    # >f, big-endian, 4 bytes float
                    chunk = file.read(4)
                    image_arry[j][i] = struct.unpack('>f', chunk)[0]
            
        image_arry = image_arry.transpose()
        
        return image_arry, parameters
        
    except IOError: 
       print("Error: cannot open the image file, please check the file path or the parameters!")        
#--------------------------------------------------------------------------------------------


def get_image_data2(image_file, parameters, resample_factor = 1, plot_flag = 0, swap_bytes = "big-endian"):
    """

    Read the insar images.

    Parameters
    ----------
    image_file : 
        the image data from GAMMA, note it is binary file
    parameters :
        the image parameters from get_image_para
    resample_factor : optional
        resampling factor, >1 is downsampling, <1 is upsampling. The default is 1.
    plot_flag : TYPE, optional
        plot (1) or not (0). The default is 0.
    swap_bytes : optional
        the unpack method of the input data. The default is "big-endian" of GAMMA interferograms.

    Returns
    -------
    TYPE
        InSAR image data (phase).
    list
        ralated info of the image.

 
    """
  
    range_samples = parameters[0] # width
    azimuth_lines = parameters[1] # nlines
    corner_lat = parameters[2] 
    corner_lon = parameters[3]
    post_lat = parameters[4]
    post_lon = parameters[5]
    post_arc = parameters[6]
    post_utm = parameters[7]
    
    
    # read the binary file firstly
    try:
        with open(image_file, 'rb') as file:
            image_arry = np.zeros([range_samples, azimuth_lines])
            # need read in column
            print("-------------------------------------------------------------")
            print("Now we are reading the phase images (binary files)...")

            for i in range(azimuth_lines):
                print(f"{i} of {azimuth_lines} ...")
                for j in range(range_samples):
                    # >f, big-endian, 4 bytes float
                    chunk = file.read(4)
                    image_arry[j][i] = struct.unpack('>f', chunk)[0]
            
            image_arry = image_arry.transpose()
            
         
        # resample the image or not
        if resample_factor == 1:
            
            print("Done readind!")
            print("Here you choose do not to resample the images.")
            print("The InSAR pixel resoluton is %f arc-second, ~%f meters." %(post_arc, post_utm))
            print("-------------------------------------------------------------")
            
            
            # make the 0 vlaues to be nan to better plotting
            image_arry2 = np.where(image_arry == 0, np.nan, image_arry)
            if plot_flag != 0:
                print("Quick preview image is generated ...")
                lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1) * post_lat, azimuth_lines)
                lons = np.linspace(corner_lon, corner_lon + (range_samples - 1 ) * post_lon, range_samples)
                
                plt.imshow(image_arry2, cmap = 'jet', vmin = np.nanmin(image_arry2), vmax = np.nanmax(image_arry2), \
                           origin = 'upper', extent= [np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha = 1.0)
                plt.colorbar(label = 'Phase (radians)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.show()    
                
                                  
            return image_arry2, lats, lons, [range_samples, azimuth_lines, corner_lat, corner_lon, post_lat, \
                            post_lon, post_arc,post_utm]
            
        else:
            # new_range_samples = range_samples // resample_factor
            # new_azimuth_lines = azimuth_lines // resample_factor
            new_post_lat, new_post_lon = post_lat * resample_factor, post_lon * resample_factor
            new_post_arc, new_post_utm = post_arc * resample_factor, post_utm * resample_factor
           
            
            print("Done readind!")
            print("Here you choose resample the image with a factor of %d." %resample_factor)
            print("The pixel resoluton of resampled InSAR image is %f arc-second, ~%f meters." \
                  %(new_post_arc, new_post_utm))
            print("-------------------------------------------------------------")
            
            
            # create the rows and cols
            rows, cols = np.arange(0, azimuth_lines, 1), np.arange(0, range_samples, 1)
            # linear interpolation
            interp_func = interpolate.interp2d(cols, rows, image_arry, kind='linear')
        
            new_rows, new_cols = np.arange(0, azimuth_lines, resample_factor), \
                np.arange(0, range_samples, resample_factor)                               
            
            new_image_arry = interp_func(new_cols, new_rows)
            
            new_azimuth_lines = new_image_arry.shape[0]
            new_range_samples = new_image_arry.shape[1]
            
            if plot_flag != 0:
                print("Quick preview image (phase) is generated ...")
                
                lats = np.linspace(corner_lat, corner_lat + (new_azimuth_lines - 1 ) * new_post_lat, new_azimuth_lines)
                lons = np.linspace(corner_lon, corner_lon + (new_range_samples - 1 ) * new_post_lon, new_range_samples)
                
                # make the 0 vlaues to be nan to better plotting
                new_image_arry2 = np.where(new_image_arry == 0, np.nan, new_image_arry)
                #new_image_arry2 = new_image_arry
                plt.imshow(new_image_arry2, cmap = 'jet', vmin = np.nanmin(new_image_arry2), vmax = np.nanmax(new_image_arry2), \
                           origin = 'upper', extent= [np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha = 1.0)
                plt.colorbar(label = 'Phase (radians)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.show()    
                
            
            return new_image_arry2, lats, lons, [new_range_samples, new_azimuth_lines, corner_lat, corner_lon, new_post_lat, \
                                new_post_lon, new_post_arc, new_post_utm]
              
    
    except IOError: 
        print("Error: cannot open the image file, please check the file path or the parameters!")
    
    
      

def phase2los(phase_data, parameters, satellite = "sentinel", plot_flag = 0):
    """
    after get_image data to get the los deformation

    Parameters
    ----------
    phase_data : 
        the phase data from get_image_data.
    parameters : 
        related info from get_image_data.
    satellite : TYPE
        The default is sentinel
        satellite type: sentinel, alos ...

    Returns 
    -------
    InSAR LOS deformation feiled.

    """

    range_samples = parameters[0] # width
    azimuth_lines = parameters[1] # nlines
    corner_lat = parameters[2] 
    corner_lon = parameters[3]
    post_lat = parameters[4]
    post_lon = parameters[5]
    
    if satellite == "sentinel" or satellite == "sentinel-1" or  satellite == "s1":
        radar_freq = 5.40500045433e9 # Hz
        wavelength = c / radar_freq # m
        # wavelength = 0.0555041577 # m
    elif satellite == "ALOS" or satellite == "alos":
        radar_freq = 1.27e9 # Hz
        pass
    elif satellite == "ALOS-2/U":
        radar_freq = 1.2575e9
    elif satellite == "ALOS-2/{F,W}":
        radar_freq = 1.2365e9
        
    los = - (phase_data / 2 / np.pi * wavelength / 2)
    
    if plot_flag != 0:
        print("Quick preview image (LOS) is generated ...")
                
        lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1) * post_lat, azimuth_lines)
        lons = np.linspace(corner_lon, corner_lon + (range_samples - 1) * post_lon, azimuth_lines)
                
        plt.imshow(los, cmap = 'jet', vmin = np.nanmin(los), vmax = np.nanmax(los), \
                   origin = 'upper', extent= [np.min(lons), np.max(lons), np.min(lats), np.max(lats)], alpha = 1.0)
        plt.colorbar(label = 'Los Deformation (m)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
        
    return los
    


if __name__ == "__main__":
    print("Only functions are defined here.")