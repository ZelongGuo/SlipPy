#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InSAR data and related operations.

Author: Zelong Guo, @ GFZ, Potsdam
Email: zelong.guo@outlook.com
Created on Tue May 18 20:08:23 2023

"""

__author__ = "Zelong Guo"
__version__ = "1.0.0"

# Standard and third-party libs
import sys
from typing import Optional, Union, Tuple
import struct
import numpy as np
from scipy.constants import c


# SlipPy libs
from ..slippy import GeoTrans


# Insar Class
class InSAR(GeoTrans):
    """Insar class for handling InSAR data.

    Args:
        - name:     instance name
        - lon0:     longitude of the UTM zone
        - lat0:     latitude of the UTM zone
        - ellps:    ellipsoid, default = "WGS84"

    Return:
        None.

    """

    def __init__(self,
                 name: str,
                 lon0: Optional[float] = None,
                 lat0: Optional[float] = None,
                 ellps: str = "WGS84",
                 utmzone: Optional[str] = None) -> None:
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

    def read_from_gamma(self,
                        para_file: str,
                        phase_file: str,
                        azi_file: str,
                        inc_file: str,
                        satellite: str,
                        downsample: int = 3) -> None:
        """Read InSAR files (including phase, incidence, azimuth and DEM files) processed by GAMMA software
        with the parameter files, and preliminary downsampling the data if needed.

        This method will assign values to the instance attribute, self.data.

        Args:
            - para_file:        parameter files like *.utm.dem.par which aligns with the sar images after
            co-registration of dem and mli, resampling or oversampling dem file to proper resolution cell
            - phase_file:       filename of InSAR phase data
            - azi_file:         filename of azimuth file
            - inc_file:         filename of incidence file
            - satellite:        Satellite type, "Sentinel-1", "ALOS" ...
            - downsample:       downsample factor to reduce the number of the points

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
                    # Here we also read the geodetic datum, usually it would be WGS 84.
                    elif line.startswith('ellipsoid_name:'):
                        ellps_name = line.strip().split(':')[1].strip()

                    else:
                        pass
            post_arc = post_lon * 3600 # to arcsecond
            post_utm = post_arc * 40075017 / 360 / 3600  # earth circumference to ground resolution, meter
            # post_arc2 = "{:.2f}".format(post_arc)
            # post_utm2 = "{:.2f}".format(post_utm)
            # print("The InSAR pixel resoluton is {} arc-second, ~{} meters." .format(post_arc2, post_utm2))
            print("-------------------------------------------------------------")
            print(f"The InSAR pixel resoluton is {post_arc:.3f} arc-second, ~{post_utm:.3f} meters.")

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

            # phase and los (unit in m)
            phase = phase.transpose().reshape(-1, 1)[::downsample]
            los = self.__phase2los(phase=phase, satellite=satellite)
            # azi and inc
            azimuth = azimuth.transpose().reshape(-1, 1)[::downsample]
            incidence = incidence.transpose().reshape(-1, 1)[::downsample]
            # lon and lat
            lats = np.linspace(corner_lat, corner_lat + (azimuth_lines - 1) * post_lat, azimuth_lines)
            lons = np.linspace(corner_lon, corner_lon + (range_samples - 1) * post_lon, range_samples)
            Lons, Lats = np.meshgrid(lons, lats)
            Lons = Lons.reshape(-1, 1)[::downsample]
            Lats = Lats.reshape(-1, 1)[::downsample]
            # utm
            utm_x, utm_y = self.ll2xy(Lons, Lats)

            self.data = np.hstack([Lons, Lats, utm_x, utm_y, phase, los, azimuth, incidence])

            # filename = ("/Users/zelong/Desktop/test.txt")
            # np.savetxt(filename, self.data_ori)

            # parameters: {width, nlines, corner_lat, corner_lon, post_lat, post_lon, post_arc, post_utm}
            self.parameters = {
                'range_samples(width)':     range_samples,
                'azimuth_lines(nlines)':    azimuth_lines,
                'corner_lat':               corner_lat,
                'corner_lon':               corner_lon,
                'post_lat':                 post_lat,
                'post_lon':                 post_lon,
                'post_arc':                 post_arc,
                'post_utm':                 post_utm,
                'ellps_name':               ellps_name,
                'satellite':                satellite,
                'los_unit':                 'm'
            }

        except IOError:
            print("Error: cannot open the image file, please check the file path or the parameters!")

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def __phase2los(self,
                    phase: Union[float, np.ndarray],
                    satellite: str) -> Union[float, np.ndarray]:
        """Converting InSAR phase to InSAR line-of-sight (los) disp.

        Args:
            - phase:       InSAR phase data.
            - satellite:        Satellite type, "Sentinel-1", "ALOS", "ALOS-2/U" ...

        Returns:
             - los:             InSAR LOS (unit in m)
        """
        if satellite in ("sentinel-1", "Sentinel-1", "s1", "S1"):
            radar_freq = 5.40500045433e9  # Hz
        elif satellite in ("ALOS", "alos"):
            radar_freq = 1.27e9  # Hz
        elif satellite == "ALOS-2/U":
            radar_freq = 1.2575e9
        elif satellite == "ALOS-2/{F,W}":
            radar_freq = 1.2365e9
        else:
            raise ValueError("The radar frequency of this satellite is not yet specified!")

        wavelength = c / radar_freq  # m
        los = - (phase / 2 / np.pi * wavelength / 2)

        # the unit is "m"
        return los

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def __get_coner_coor(self):
        pass

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def deramp(self, dem_file):
        pass

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def write2file(self, dem_file):
        pass


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":
    pass

