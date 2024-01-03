#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31.12.23

@author: Zelong Guo
@GFZ, Potsdam
"""

import sys

import matplotlib.pyplot as plt
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

    def _calc_plfault(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
        """Initialize a rectangle fault with one patch.

        Args:
            - pointpos:       the position of the specified point:
                              "ul": upper left point,  "uc": upper center point,  "ur": upper right point
                              "bl": bottom left point, "bc": bottom center point, "br": bottom right point
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
        # complementary angle of strike
        strike_comp = np.radians(90 - strike)
        dip = np.radians(dip)
        projwidth = width * np.cos(dip)

        cpl_length = (length + 0j) * np.exp(strike_comp * 1j)
        cpl_width = (0 - projwidth * 1j) * np.exp(strike_comp * 1j)
        x, y = self.ll2xy(lon, lat)

        if pointpos in ("ul", "UL", "upper left", "upper_left"):
            x_ul, y_ul, z_ul = x, y, -verdepth
        elif pointpos in ("uc", "UC", "upper center", "upper_center"):
            x_uc, y_uc, z_uc = x, y, -verdepth
            xy_ul = (x_uc + y_uc * 1j) - cpl_length / 2
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, -verdepth
        elif pointpos in ("ur", "UR", "upper right", "upper_right"):
            x_ur, y_ur, z_ur = x, y, -verdepth
            xy_ul = (x_ur + y_ur * 1j) - cpl_length
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, -verdepth
        elif pointpos in ("br", "BR", "bottom right", "bottom_right"):
            x_br, y_br, z_br = x, y, -verdepth
            xy_ul = (x_br + y_br * 1j) - cpl_length - cpl_width
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, -verdepth + width * np.sin(dip)
        elif pointpos in ("bc", "BC", "bottom center", "bottom_center"):
            x_bc, y_bc, z_bc = x, y, -verdepth
            xy_ul = (x_bc + y_bc * 1j) - cpl_length / 2 - cpl_width
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, -verdepth + width * np.sin(dip)
        elif pointpos in ("bl", "BL", "bottom left", "bottom_left"):
            x_bl, y_bl, z_bl = x, y, -verdepth
            xy_ul = (x_bl + y_bl * 1j) - cpl_width
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, -verdepth + width * np.sin(dip)
        elif pointpos in ("cc", "CC", "centroid center", "centroid_center"):
            x_cc, y_cc, z_cc = x, y, -verdepth
            xy_ul = (x_cc + y_cc * 1j) - cpl_width / 2 - cpl_length / 2
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, -verdepth + width * np.sin(dip) / 2
        else:
            raise ValueError("Please specify a correct point position VALUE!")

        # upper center point
        xy_uc = (x_ul + y_ul * 1j) + cpl_length / 2
        x_uc, y_uc, z_uc = xy_uc.real, xy_uc.imag, z_ul
        # upper right point
        xy_ur = (x_ul + y_ul * 1j) + cpl_length
        x_ur, y_ur, z_ur = xy_ur.real, xy_ur.imag, z_ul
        # bottom right point
        xy_br = (x_ur + y_ur * 1j) + cpl_width
        x_br, y_br, z_br = xy_br.real, xy_br.imag, z_ul - width * np.sin(dip)
        # bottom center point
        xy_bc = (x_br + y_br * 1j) - cpl_length / 2
        x_bc, y_bc, z_bc = xy_bc.real, xy_bc.imag, z_ul - width * np.sin(dip)
        # bottom left point
        xy_bl = (x_bc + y_bc * 1j) - cpl_length / 2
        x_bl, y_bl, z_bl = xy_bl.real, xy_bl.imag, z_ul - width * np.sin(dip)
        # fault center point
        x_cc, y_cc, z_cc = (x_uc + x_bc) / 2, (y_uc + y_bc) / 2, (z_uc + z_bc) / 2

        fault = {
            "upper left":       (x_ul, y_ul, z_ul),
            "upper center":     (x_uc, y_uc, z_uc),
            "upper right":      (x_ur, y_ur, z_ur),
            "bottom left":      (x_bl, y_bl, z_bl),
            "bottom center":    (x_bc, y_bc, z_bc),
            "bottom right":     (x_br, y_br, z_br),
            "centroid center":  (x_cc, y_cc, z_cc),
            "length":           length,
            "width":            width
        }

        fault_corner = {
            "x":        np.array([[x_ul, x_ur], [x_bl, x_br]]),
            "y":        np.array([[y_ul, y_ur], [y_bl, y_br]]),
            "z":        np.array([[z_ul, z_ur], [z_bl, z_br]]),
            "length":   length,
            "width":    width
        }

        return fault, fault_corner

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
        verdepth_bc = -verdepth_uc - width * np.sin(dip)

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
        bl = (0 - projwidth * 1j) * np.exp(strike_comp * 1j)
        xy_bl = (x_ul + y_ul * 1j) + bl
        x_bl, y_bl, z_bl = xy_bl.real, xy_bl.imag, verdepth_bc
        # bottom right point
        xy_br = (x_ur + y_ur * 1j) + bl
        x_br, y_br, z_br = xy_br.real, xy_br.imag, verdepth_bc
        # bottom center point
        x_bc, y_bc, z_bc = (x_bl + x_br) / 2, (y_bl + y_br) / 2, verdepth_bc

        # fault center point
        x_cc, y_cc, z_cc = (x_uc + x_bc) / 2, (y_uc + y_bc) / 2, (z_uc + z_bc) / 2

        fault = {
            "upper left":       (x_ul, y_ul, z_ul),
            "upper center":     (x_uc, y_uc, z_uc),
            "upper right":      (x_ur, y_ur, z_ur),
            "bottom left":      (x_bl, y_bl, z_bl),
            "bottom center":    (x_bc, y_bc, z_bc),
            "bottom right":     (x_br, y_br, z_br),
            "centroid center":  (x_cc, y_cc, z_cc),
            "length":           length,
            "width":            width
        }

        fault_corner = {
            "x":        np.array([[x_ul, x_ur], [x_bl, x_br]]),
            "y":        np.array([[y_ul, y_ur], [y_bl, y_br]]),
            "z":        np.array([[z_ul, z_ur], [z_bl, z_br]]),
            "length":   length,
            "width":    width
        }

        return fault, fault_corner

    def plot(self, x, y, z):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(z.min(), 0)

        plt.show()


    def mesh_planar_fault(self, fault, patchlen, patchwid):
        """Meshing a uniform fault plane to tons of fault patches.

        Args:
            - fault:         fault rectangle plane, a dict given by initialize_palar_fault
            - patchlen:             the length of fault patch
            - patchwid:             the width of fault patch
        Return:
            - None.
        """
        ul = fault["upper left"]
        ur = fault["upper right"]
        bl = fault["bottom left"]
        length = fault["length"]
        width = fault["width"]

        m = math.floor(length / patchlen)
        n = math.floor(width / patchwid)

        xn = np.linspace(ul[0], ur[0], m)
        # yn = np.linspace(ul[1], )








# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":

    fault = Fault("flt", 44.28, 35.47)
    # fault.get_fault_pameters(lon=44.344, lat=35.603, depth=3, strike=350, dip=15, length=80, width=50)
    patch1, patch_corner1 = fault.initialize_planar_fault(lon_uc=44.344, lat_uc=35.603, verdepth_uc=3, strike=10, dip=45, length=80, width=50)
    patch2, patch_corner2 = fault._calc_plfault(pointpos="upper center", lon=44.344, lat=35.603, verdepth=3, strike=10, dip=45, length=80, width=50)

    # fault.plot(patch_corner1["x"], patch_corner1["y"], patch_corner1["z"])
    fault.plot(patch_corner2["x"], patch_corner2["y"], patch_corner2["z"])

