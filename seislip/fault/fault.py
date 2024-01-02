#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31.12.23

@author: Zelong Guo
@GFZ, Potsdam
"""

import sys
import numpy as np
import math

sys.path.append("../")
from seislip.seislip import GeoTrans

# slippy libs
# from ..seislip import GeoTrans

class Fault(GeoTrans):
    def __init__(self, name, lon0, lat0, ellps="WGS84", utmzone=None):
        super().__init__(name, lon0, lat0, ellps, utmzone)

        # fault parameters
        self.flon0 = None
        self.flat0 = None
        self.fdepth = None
        self.fstrike = None
        self.fdip = None
        self.flength = None
        self.fwidth = None

        self.patch = {}


    def initialize_planar_fault(self, lon_uc, lat_uc, verdepth_uc, strike, dip, length, width):
        """Initialize a rectangle fault with one patch.

        Args:
            - lon_uc:         longitude of the central point on the upper edge of the fault, degree
            - lat_uc:         latitude of the central point on the upper edge of the fault, degree
            - verdepth_uc:    vertical depth of the central point on the upper edge of the fault, km
            - strike:         strike angle of the fault, degree
            - dip:            dip angle of the fault, degree
            - width:          width along the fault dip direction, km
            - length:         length along the fault strike direction, km

        Return:
            - None.
        """
        lon_uc, lat_uc, verdepth_uc = lon_uc, lat_uc, verdepth_uc  # central point on upper edge
        strike_comp = np.radians(90 - strike)
        dip = np.radians(dip)
        length = length
        width = width
        projwidth = width * np.cos(dip)
        verdepth_bc = -verdepth_uc - width * np.cos(dip)

        # upper center point
        x_uc, y_uc = self.ll2xy(lon_uc, lat_uc)
        x_uc, y_uc, z_uc = x_uc, y_uc, -verdepth_uc
        # upper left point
        ul = (-0.5 * length + 0j) * np.exp(strike_comp * 1j)
        xy_ul = (x_uc + y_uc * 1j) + ul
        x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, -verdepth_uc
        # upper right point
        ur = (0.5 * length + 0j) * np.exp(strike_comp * 1j)
        xy_ur = (x_uc + y_uc * 1j) + ur
        x_ur, y_ur, z_ur = xy_ur.real, xy_ur.imag, -verdepth_uc

        # bottom left point
        bl = (0 + projwidth * 1j) * np.exp(strike_comp * 1j)
        xy_bl = (x_ul + y_ul * 1j) + bl
        x_bl, y_bl, z_bl = xy_bl.real, xy_bl.imag, verdepth_bc
        # bottom right point
        xy_br = (x_ur + y_ur * 1j) + bl
        x_br, y_br, z_br = xy_br.real, xy_br.imag, verdepth_bc
        # bottom center point
        x_bc, y_bc, z_bc = (x_bl + x_br) / 2, (y_bl + y_br) /2, verdepth_bc

        # fault center point
        x_cc, y_cc, z_cc = (x_uc + x_bc) / 2, (y_uc + y_bc) /2, (z_uc + z_bc) / 2

        patch = {
            "upper left":       (x_ul, y_ul, z_ul),
            "upper center":     (x_uc, y_uc, z_uc),
            "upper right":      (x_ur, y_ur, z_ur),
            "bottom left":      (x_bl, y_bl, z_bl),
            "bottom center":    (x_bc, y_bc, z_bc),
            "bottom right":     (x_br, y_br, z_br),
            "centroid center":  (x_cc, y_cc, z_cc)
        }

        return patch


    def construct_fault(self):
        """Construct a uniform fault plane."""
        strike_rad, dip_rad = np.radians(self.fstrike), np.radians(self.fdip)

        # coordinates of the 4 fault corner points
        half_flength = math.floor(self.flength / 2)
        dx = half_flength * np.sin(strike_rad)
        dy = half_flength * np.cos(strike_rad)
        dz = self.fwidth * np.sin(dip_rad)
        dw = self.fwidth * np.cos(dip_rad)

        x1, y1, z1 = self.futmx + dx, self.futmy + dy, self.fdepth
        x1, y2, z2 = self.futmx - dx, self.futmy - dy, self.fdepth


        pass


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":

    fault = Fault("flt", 44.28, 35.47)
    # fault.get_fault_pameters(lon=44.344, lat=35.603, depth=3, strike=350, dip=15, length=80, width=50)
    patch = fault.initialize_planar_fault(lon_uc=44.344, lat_uc=35.603, verdepth_uc=3, strike=350, dip=15, length=80, width=50)