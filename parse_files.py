#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions for parsing files:
    (1) .par files, e.g., .dem.par to get the image parameters
    
Created on Tue May 18 20:08:23 2023

@author: zelong
"""
import numpy as np
import struct

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
        post_arc2 = "{:.2f}".format(post_arc)
        post_utm2 = "{:.2f}".format(post_utm)
        print("-------------------------------------------------------------")
        print("The InSAR pixel resoluton is {} arc-second, ~{} meters." .format(post_arc2, post_utm2))
        print("-------------------------------------------------------------")

        return [width, nlines, corner_lat, corner_lon, post_lat, post_lon, post_arc, post_utm]
                    
    except IOError:
        print("Error: cannot open the parameter file, please check the file path!")





def get_image_data(image_file, parameters, resample_factor = None, swap_bytes = "big-endian"):
    """
    Read the insar images.
    ----------
    image_file : the image data from GAMMA, note it is binary file
    parameters : the image parameters from get_image_para

    -------
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
        # with open(image_file, 'rb') as file:
        #     image_arry = np.zeros([range_samples, azimuth_lines])
        #     # need read in column
        #     print("-------------------------------------------------------------")
        #     print("Now we are reading the phase images (binary files)...")
        #     print("-------------------------------------------------------------")
        #     for i in range(azimuth_lines):
        #         for j in range(range_samples):
        #             # >f, big-endian, 4 bytes float
        #             chunk = file.read(4)
        #             image_arry[j][i] = struct.unpack('>f', chunk)[0]
                    
         
        # resample the image or not
        if resample_factor is None:
            print("-------------------------------------------------------------")
            print("Done readind!")
            print("Here you choose do not to resample the images.")
            print("The InSAR pixel resoluton is %f arc-second, ~%f meters." %(post_arc, post_utm))
            print("-------------------------------------------------------------")
        else:
            range_samples = range_samples // resample_factor
            azimuth_lines = azimuth_lines // resample_factor
            post_lat = post_lat * resample_factor
            post_lon = post_lon * resample_factor
            post_arc = post_arc * resample_factor
            post_utm = post_utm * resample_factor
            print("-------------------------------------------------------------")
            print("Done readind!")
            print("Here you choose resample the image with a factor of %d." %resample_factor)
            print("The pixel resoluton of resampled InSAR image is %f arc-second, ~%f meters." \
                  %(post_arc, post_utm))
            print("-------------------------------------------------------------")
            
        #     resampled_arry = np.zeros([range_samples, azimuth_lines])
        #     for x in range(azimuth_lines):
        #         for y in range(range_samples):
        #             window_data = image_arry[x * resample_factor:(x + 1) * resample_factor, y * resample_factor \
        #                                      :(y + 1) *resample_factor]
        #             window_data = np.mean(window_data)
        #             resampled_arry[x, y] = window_data
        #     image_arry = resampled_arry
            
            
            
        # return image_arry, [range_samples, azimuth_lines, corner_lat, corner_lon, post_lat, \
        #                     post_lon, post_arc,post_utm]
        
    
    except IOError:
        print("Error: cannot open the image file, please check the file path or the parameters!")
    
    
       
    
    #pass           




if __name__ == "__main__":
    print("Only functions are defined here.")