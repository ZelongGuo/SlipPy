#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InSAR data and related operations.

Author: Zelong Guo, @ GFZ, Potsdam
Email: zelong.guo@outlook.com
Created on Tue May 18 20:08:23 2023

#---------------------------------------------------------------------------------------------
Some functions for parsing files from some processing software like GAMMA etc.:
    get_image_para_gamma (.par files, e.g., .dem.par to get the image parameters from GAMMA)
    get_image_data_gamma (deformation images to get the deformation phase from GAMMA)

"""

__author__ = "Zelong Guo"

# Standard and third-party libs
import sys
import numpy as np
import struct

# SlipPy libs
from ..slipbase import Slip


# Insar Class
class InSAR(Slip):
    """
    Insar class for handling InSAR data.
    """

    def __init__(self, name, notes="InSAR"):
        # init properties from parent class
        super(Insar, self).__init__(name)

        # Initialization
        self.notes = notes

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # | G | A | M | M | A |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    def read_from_gamma(self, para_file):
        """Read InSAR interferograms from GAMMA with the parameter files.

        Args:
            para_file :     para_files, parameter files like *.utm.dem.par which aligns with the sar images after
                            co-registration of dem and mli, resampling or oversampling dem file to proper resolution cell

        Return: the width, nlines and lon lat info of the InSAR image
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


    # --------------------------------------------------------------------------------------------
    def get_image_data_gamma(image_file, para_file, swap_bytes="big-endian"):
        """
        Read the InSAR images (observations, azimuth and incidence files) and DEM files if needed From GAMMA.

        Parameters
        ----------
        image_file : big-endian files of insar images or dem from GAMMA

        para_file : the image parameters from get_image_para, should *dem.par

        swap_bytes : The default is "big-endian" which is also default setting from GAMMA files (only big-endian
         is supported now).

        Returns
        -------
        image_arry : array of images for the following processing.

        parameters : parameter from get_image_para

        """

        parameters = get_image_para_gamma(para_file)

        range_samples = parameters['width']  # width
        azimuth_lines = parameters['nlines']  # nlines

        try:
            with open(image_file, 'rb') as file:
                image_arry = np.zeros([range_samples, azimuth_lines])
                # need read in column
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


# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Only functions are defined here.")
