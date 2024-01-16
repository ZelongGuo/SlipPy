#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planar fault class for single or multiple faults.

Created on 31.12.23

@author: Zelong Guo
"""

__author__ = "Zelong Guo"

import sys

import matplotlib.pyplot as plt
import numpy as np
import math

# seislip libs
if __name__ == "__main__":
    sys.path.append("../")
    from seislip.seislip import GeoTrans
else:
    from ..seislip import GeoTrans


class Fault(GeoTrans):
    """Constructing a Fault object with four corner coordinates (and central/centroid point coordinates).

    The object of this class would be used for mesh the fault plane into rectangle or triangle patches.

    Args:
        - name:                 Fault instance name
        - lon0:                 longitude used for specifying the utm zone
        - lat0:                 lattitude used for specifying the utm zone
        - ellps:                Optional, reference ellipsoid, defatult = "WGS84"
        - utmzone:              Optional, if not specify lon0, lat0 and ellps, default = None.

    Return:
        - None.
    """
    def __init__(self, name, lon0, lat0, ellps="WGS84", utmzone=None):
        super().__init__(name, lon0, lat0, ellps, utmzone)

        # fault parameters
        self.origin = None
        self.strike = None
        self.dip = None
        self.length = None
        self.width = None

        # read from trace and etc...
        self.mutifaults = None  # TODO: multifaults is a list contain multiple fault objects,
        # TODO: if there is multiple faults, then delete the above attributes

    def initialize_fault(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
        """Initialize a rectangle fault plane by specifying a corner or central point of the fault.

        The definition of the fault coordinate system:
        This coordinate system is right-hand:
        X:      along strike
        Y:      along opposite direction of dip
        Z:      normal direction the fault
        Point position on fault plane, the origin of the fault coordinate system is bottom original point.
        "uo": upper original point,     "uc": upper center point,       "ue": upper end point
        "bo": bottom original point,    "bc": bottom center point,      "be": bottom end point
        "cc": centroid center point

        The definition of UTM system:
        X:      easting, km
        Y:      northing, km
        Z:      zenith direction, km

        Args:
            - pointpos:       the position of the specified fault point in the fault coordinate system:
                              "uo", "uc", "ue", "bo", "bc", "be", "cc". "uc" is strongly recommended.
            - lon:            longitude of the specified point, degree
            - lat:            latitude of the specified point, degree
            - verdepth:       vertical depth of the specified point, km. Depth should
                              specified as Negative value.
            - strike:         strike angle of the fault, degree
            - dip:            dip angle of the fault, degree
            - width:          width along the fault dip direction, km
            - length:         length along the fault strike direction, km

        Return:
            - None.
        """
        # pointpos_list = ("uo",  "UO",  "upper origin",    "upper_origin",
        #                  "uc",  "UC",  "upper center",    "upper_center",
        #                  "ue",  "UE",  "upper end",       "upper_end",
        #                  "bo",  "BO",  "bottom origin",   "bottom_origin"
        #                  "bc",  "BC",  "bottom center",   "bottom_center",
        #                  "be",  "BE",  "bottom end",      "bottom_end",
        #                  "cc",  "CC",  "centroid center", "centroid_center")

        # For now only support "uc"
        pointpos_list = ("uc",  "UC",  "upper center",    "upper_center")
        if pointpos not in pointpos_list:
            raise ValueError("Please specify a right point position!")
        else:
            self.origin = (pointpos, (lon, lat, verdepth))
            self.strike = strike
            self.dip = dip
            self.length = length
            self.width = width

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def _initialize_fault(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
        """Initialize a rectangle fault plane by calculating the corner or central points of the fault, based
        on one point given.

        Args:
            - pointpos:       the position of the specified/original point:
                              "ul": upper left point,  "uc": upper center point,  "ur": upper right point
                              "bl": bottom left point, "bc": bottom center point, "br": bottom right point
            - lon:            longitude of the central point on the upper edge of the fault, degree
            - lat:            latitude of the central point on the upper edge of the fault, degree
            - verdepth:       vertical depth of the central point on the upper edge of the fault, km. Depth should
                              specified as Negative value.
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
            x_ul, y_ul, z_ul = x, y, verdepth
        elif pointpos in ("uc", "UC", "upper center", "upper_center"):
            x_uc, y_uc, z_uc = x, y, verdepth
            xy_ul = (x_uc + y_uc * 1j) - cpl_length / 2
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, verdepth
        elif pointpos in ("ur", "UR", "upper right", "upper_right"):
            x_ur, y_ur, z_ur = x, y, verdepth
            xy_ul = (x_ur + y_ur * 1j) - cpl_length
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, verdepth
        elif pointpos in ("br", "BR", "bottom right", "bottom_right"):
            x_br, y_br, z_br = x, y, verdepth
            xy_ul = (x_br + y_br * 1j) - cpl_length - cpl_width
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("bc", "BC", "bottom center", "bottom_center"):
            x_bc, y_bc, z_bc = x, y, verdepth
            xy_ul = (x_bc + y_bc * 1j) - cpl_length / 2 - cpl_width
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("bl", "BL", "bottom left", "bottom_left"):
            x_bl, y_bl, z_bl = x, y, verdepth
            xy_ul = (x_bl + y_bl * 1j) - cpl_width
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("cc", "CC", "centroid center", "centroid_center"):
            x_cc, y_cc, z_cc = x, y, verdepth
            xy_ul = (x_cc + y_cc * 1j) - cpl_width / 2 - cpl_length / 2
            x_ul, y_ul, z_ul = xy_ul.real, xy_ul.imag, verdepth + width * np.sin(dip) / 2
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
            "upper left":       [x_ul, y_ul, z_ul],
            "upper center":     [x_uc, y_uc, z_uc],
            "upper right":      [x_ur, y_ur, z_ur],
            "bottom left":      [x_bl, y_bl, z_bl],
            "bottom center":    [x_bc, y_bc, z_bc],
            "bottom right":     [x_br, y_br, z_br],
            "centroid center":  [x_cc, y_cc, z_cc],
            "length":           length,
            "width":            width
        }

        fault_corner = [
            [fault["upper left"], fault["upper right"], fault["bottom right"], fault["bottom left"]]
        ]
        # fault_corner = np.array(fault_corner)

        # fault_corner = {
        #     "x":        [[x_ul, x_ur], [x_bl, x_br]],
        #     "y":        [[y_ul, y_ur], [y_bl, y_br]],
        #     "z":        [[z_ul, z_ur], [z_bl, z_br]],
        # }

        # temp = self.xy2ll(x_cc, y_cc)

        return fault, fault_corner

    def read_from_trace(self):
        pass

    def mesh_planar_fault(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width, patchlen, patchwid):
        """Meshing a uniform fault plane to tons of fault patches.

        Args:
            - fault:                fault rectangle plane, a dict given by initialize_palar_fault
            - patchlen:             the length of fault patch
            - patchwid:             the width of fault patch
        Return:
            - None.
        """
        verts = []
        x = []
        y = []
        z = []
        m = math.floor(length / patchlen)
        n = math.floor(width / patchwid)

        for i in range(m):
            if i == 0:
                _, temp = self._initialize_fault(pointpos, lon, lat, verdepth, strike, dip, patchlen, patchwid)
                x0, y0 = self.xy2ll(temp[-1][3][0], temp[-1][3][1])
                z0 = temp[-1][3][2]

            else:
                if pointpos in ("ul", "UL", "upper left", "upper_left"):
                    lon, lat = self.xy2ll(verts[-1][1][0], verts[-1][1][1])
                    verdepth = verts[-1][1][2]
                    x0, y0 = self.xy2ll(verts[-1][3][0], verts[-1][3][1])
                    z0 = verts[-1][3][2]
                _, temp = self._initialize_fault(pointpos, lon, lat, verdepth, strike, dip, patchlen, patchwid)
            verts.append(temp[0])
            x.append(x0)
            y.append(y0)
            z.append(z0)

        for lon, lat, verdepth in zip(x, y, z):
            for i in range(n-1):
                if i == 1:
                    _, temp = self._initialize_fault(pointpos, lon, lat, verdepth, strike, dip, patchlen, patchwid)
                else:
                    if pointpos in ("ul", "UL", "upper left", "upper_left"):
                        lon, lat = self.xy2ll(verts[-1][3][0], verts[-1][3][1])
                        verdepth = verts[-1][3][2]
                    _, temp = self._initialize_fault(pointpos, lon, lat, verdepth, strike, dip, patchlen,
                                                             patchwid)
                verts.append(temp[0])



        return verts


        # for i in range(m):
        #     for j in range(n):
        #         if i == 0 and j == 0:
        #             _, temp = self._initialize_fault_corners(pointpos, lon, lat, verdepth, strike, dip, patchlen, patchwid)
        #         else:
        #             if pointpos in ("ul", "UL", "upper left", "upper_left"):
        #                 lon, lat = self.xy2ll(x[-1][0][-1], y[-1][0][-1])
        #                 verdepth = z[-1][0][-1]
        #             # elif pointpos in ("ur", "UR"):
        #             #     lon, lat = self.xy2ll(x[-1][1], y[-1][1])
        #             _, temp = self._initialize_fault_corners(pointpos, lon, lat, verdepth, strike, dip, patchlen, patchwid)
        #         x.append(temp["x"])
        #         y.append(temp["y"])
        #         z.append(temp["z"])
        #         # TODO
        #
        # x, y, z = np.array(x), np.array(y), np.array(z)
        #
        # x = np.reshape(x, (m * 2, n * 2))
        # y = np.reshape(y, (m * 2, n * 2))
        # z = np.reshape(z, (m * 2, n * 2))
        #
        # return x, y, z


        # xn = np.linspace(ul[0], ur[0], m)
        # yn = np.linspace(ul[1], )

    def plot(self, verts):
        # x, y, z = np.array(x), np.array(y), np.array(z)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(x, y, z)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_zlim(z.min(), 0)
        #
        # plt.show()

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        poly3d = Poly3DCollection(verts, edgecolor='k', alpha=0.7)

        ax.add_collection3d(poly3d)

        verts = np.array(verts)
        xmin = verts[:, :, 0].min()
        xmax = verts[:, :, 0].max()
        ymin = verts[:, :, 1].min()
        ymax = verts[:, :, 1].max()
        zmin = verts[:, :, 2].min()
        zmax = verts[:, :, 2].max()


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        plt.show()





# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
if __name__ == "__main__":

    fault = Fault("flt", 44.28, 35.47)
    # patch1, patch_corner1 = fault.initialize_planar_fault(lon_uc=44.344, lat_uc=35.603, verdepth_uc=3, strike=10, dip=45, length=80, width=50)
    patch1, patch_corner1 = fault._initialize_fault(pointpos="upper center", lon=44.344, lat=35.603, verdepth=-3, strike=10, dip=45, length=80, width=50)
    fault.plot(patch_corner1)
    # verts = fault.mesh_planar_fault(pointpos="upper left", lon=44.344, lat=35.603, verdepth=-3, strike=10, dip=45, length=80, width=50, patchlen=2, patchwid=2)

    #
    #

    # fault.plot(patch_corner1)
    # fault.plot(verts)


