#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InSAR data and related operations.

Author: Zelong Guo, @ GFZ, Potsdam
Email: zelong.guo@outlook.com
Created on Tue May 18 20:08:23 2023

"""

__author__ = "Zelong Guo"

# Standard and third-party libs
import sys
import numpy as np
import struct

# SlipPy libs
from ..slippy import SlipPy


# Insar Class
class InSAR(SlipPy):
    """Insar class for handling InSAR data.

    Args:
        - name:     instance name
        - lon0:     longitude of the UTM zone
        - lat0:     latitude of the UTM zone
        - ellps:    ellipsoid, default = "WGS84"

    Return:
        None.

    """

    def __init__(self, name, lon0=None, lat0=None, ellps="WGS84", utmzone=None):
        # call init function of the parent class to initialize
        super().__init__(name, lon0, lat0, ellps, utmzone)

        print("-------------------------------------------------------------")
        print(f"Now we initialize the InSAR instance {self.name}...")

        # Internal initialization
        # format of data_ori: lon lat x y phase los azimuth incidence Ue Un Uu
        self.data_ori = None
        # InSAR images' parameters like width, length, corner coordinates etc.
        self.parameters = None
        # self.azi = None
        # self.inc = None


    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # | G | A | M | M | A |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    def read_from_gamma(self, para_file, phase_file, azi_file, inc_file):
        """Read InSAR files (including phase, incidence, azimuth and DEM files) processed by GAMMA software
        with the parameter files.

        This method will assign values to the instance attribute, data_ori.

        Args:
            - para_file:        parameter files like *.utm.dem.par which aligns with the sar images after
            co-registration of dem and mli, resampling or oversampling dem file to proper resolution cell
            - phase_file:       filename of InSAR phase data
            - azi_file:         filename of azimuth file
            - inc_file:         filename of incidence file

        Return:
            None.
        -------

        Update log:
        12.06.2023, update the return values as dic
        """

        # Firstly we read the para_file to get the porameters
        try:
            with open(para_file, 'r') as file:
                print("-------------------------------------------------------------")
                print("Now we are reading the parameter file...")
                for line in file:

                   # line = line.strip()
                   # print(line)

                    if line.startswith('width:'):
                        range_samples = int(line.strip().split(':')[1])
                        print(line)
                    elif line.startswith('nlines:'):
                        azimuth_lines = int(line.strip().split(':')[1])
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
            print("-------------------------------------------------------------")
            print(f"The InSAR pixel resoluton is {post_arc:.3f} arc-second, ~{post_utm:.3f} meters.")

            # parameters: {width, nlines, corner_lat, corner_lon, post_lat, post_lon, post_arc, post_utm}
            self.parameters = {
                'range_samples(width)': range_samples,
                'azimuth_lines(nlines)': azimuth_lines,
                'corner_lat': corner_lat,
                'corner_lon': corner_lon,
                'post_lat': post_lat,
                'post_lon': post_lon,
                'post_arc': post_arc,
                'post_utm': post_utm
            }

        except IOError:
            print("Error: cannot open the parameter file, please check the file path!")

        # Then we read the phase, azimuth and the incidence files to get the original data. All the files
        # processed by GAMMA are big-endian (swap_bytes="big-endian").
        try:
            with open(phase_file, 'rb') as f1, open(azi_file, 'rb') as f2, open(inc_file, 'rb') as f3:
                phase = np.zeros([range_samples, azimuth_lines])
                azimuth = np.zeros([range_samples, azimuth_lines])
                incidence = np.zeros([range_samples, azimuth_lines])

                # need read in column
                print("-------------------------------------------------------------")
                print("Now we are reading the phase, azimuth and incidence images (binary files)...")
                print(f"Total {azimuth_lines}:")

                for i in range(azimuth_lines):
                    if i % 500 == 0:
                        # print(f"{i} ", end = '\r')
                        sys.stdout.write(f"{i} ")
                        sys.stdout.flush()
                    for j in range(range_samples):
                        # >f, big-endian, 4 bytes float
                        chunk = f1.read(4)
                        phase[j][i] = struct.unpack('>f', chunk)[0]
                        chunk = f2.read(4)
                        azimuth[j][i] = struct.unpack('>f', chunk)[0]
                        chunk = f3.read(4)
                        incidence[j][i] = struct.unpack('>f', chunk)[0]

            phase = phase.transpose().reshape(-1, 1)
            azimuth = azimuth.transpose().reshape(-1, 1)
            incidence = incidence.transpose().reshape(-1, 1)
            lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1) * post_lat, azimuth_lines)
            lons = np.linspace(corner_lon, corner_lon + (range_samples - 1) * post_lon, range_samples)
            Lons,Lats = np.meshgrid(lons, lats)
            Lons = Lons.reshape(-1, 1)
            Lats = Lats.reshape(-1, 1)

            self.data_ori = np.hstack([Lons, Lats, phase, azimuth, incidence])

            filename = ("/Users/zelong/Desktop/test.txt")
            np.savetxt(filename, self.data_ori)

            # TO DO:
            # converting to UTM, and the unit vectors of InSAR data


        except IOError:
            print("Error: cannot open the image file, please check the file path or the parameters!")


    def deramp(self, dem_file):
        pass




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
