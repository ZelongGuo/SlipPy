#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planar fault class for single or multiple faults.

Created on 31.12.23

@author: Zelong Guo
"""

__author__ = "Zelong Guo"

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import math

# seislip libs
if __name__ == "__main__":
    sys.path.append("../")
    from seislip.seislip import GeoTrans
else:
    from ..seislip import GeoTrans

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
class Fault(GeoTrans):
    """Constructing a Fault object with four corner coordinates (and central/centroid point coordinates).

    The object of this class would be used for meshing the fault plane into rectangle or triangle patches.

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

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def initialize_fault(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
        """Initialize a rectangle fault plane by specifying a corner or central point of the fault.

        The definition of the fault coordinate system:
        This coordinate system is right-hand:
        X:      along strike
        Y:      along opposite direction of dip
        Z:      normal direction the fault
        Point position on fault plane.
        "uo": upper original point,     "uc": upper center point,       "ue": upper end point
        "bo": bottom original point,    "bc": bottom center point,      "be": bottom end point
        "cc": centroid center point

        The definition of UTM system:
        X:      easting, km
        Y:      northing, km
        Z:      zenith direction, km

        Args:
            - pointpos:       the position of the specified fault point in the fault coordinate system:
                              "uo", "uc", "ue", "bo", "bc", "be", "cc".
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

        pointpos_list = ("uo",  "UO",  "upper origin",    "upper_origin",
                         "uc",  "UC",  "upper center",    "upper_center",
                         "ue",  "UE",  "upper end",       "upper_end",
                         "bo",  "BO",  "bottom origin",   "bottom_origin",
                         "bc",  "BC",  "bottom center",   "bottom_center",
                         "be",  "BE",  "bottom end",      "bottom_end",
                         "cc",  "CC",  "centroid center", "centroid_center")

        if pointpos not in pointpos_list:
            raise ValueError("Please specify a right point position!")
        else:
            # calculate fault corner coordinates
            fault_corners = self.__calc_corner_coordinates(pointpos, lon, lat, verdepth, strike, dip, length, width)
            # check if the fault expose to surface, and return the upper center point
            lon_uc, lat_uc, depth_uc = self.__check_breach_surface(fault_corners, strike, dip)
            self.origin = ("upper center", (lon_uc, lat_uc, depth_uc))
            self.strike = strike
            self.dip = dip
            self.length = length
            self.width = width

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    def extend_to_surface(self):
        pass

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def __check_breach_surface(self, fault_corners, strike, dip):
        """Check the fault breach to the surface or not. If it does, then assign the depth of upper edge
        to 0 (surface) automatically. If it does not, do nothing. Finally, we return the lon, lat and depth of the
        upper center point of the fault.
        The depth should be negative values.

        Args:
            - fault_corners:            a dict containing fault corners coordinates

        Return:
            - lon, lat and depth of center point on the fault upper edge
        """

        # uo = fault_corners["upper origin"]
        uc = fault_corners["upper center"]
        # ue = fault_corners["upper end"]

        x_uc, y_uc, z_uc = uc[0], uc[1], uc[2]
        # if the upper fault edge breaches to the surface:
        if z_uc > 0:
            warnings.warn("Warning: The fault has breached to the surface! "
                          "We are now setting the upper edge depth to 0.")
            # complementary angle of strike
            strike_comp = np.radians(90 - strike)
            dip = np.radians(dip)
            exposed_surface_wid = z_uc / np.tan(dip)
            cpl_exposed_surface_wid = (0 - exposed_surface_wid * 1j) * np.exp(strike_comp * 1j)

            surface_xy_uc = (x_uc + y_uc * 1j) + cpl_exposed_surface_wid
            surface_x_uc, surface_y_uc, surface_z_uc = surface_xy_uc.real, surface_xy_uc.imag, 0
            surface_lon_uc, surface_lat_uc = self.xy2ll(surface_x_uc, surface_y_uc)

            return surface_lon_uc, surface_lat_uc, surface_z_uc
        else:
            lon_uc, lat_uc = self.xy2ll(x_uc, y_uc)
            return lon_uc, lat_uc, z_uc

    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

    def __calc_corner_coordinates(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
        """Calculate the 3D UTM coordinates of corners or central points on fault, based on one point given.

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
            - fault_corners:    3D UTM coordinates of corners anf central points of the fault plane
        """
        # complementary angle of strike
        strike_comp = np.radians(90 - strike)
        dip = np.radians(dip)
        projwidth = width * np.cos(dip)

        cpl_length = (length + 0j) * np.exp(strike_comp * 1j)
        cpl_width = (0 - projwidth * 1j) * np.exp(strike_comp * 1j)
        x, y = self.ll2xy(lon, lat)

        # if pointpos in ("ul", "UL", "upper left", "upper_left"):
        if pointpos in ("uo", "UO", "upper origin", "upper_origin"):
            x_uo, y_uo, z_uo = x, y, verdepth
        elif pointpos in ("uc", "UC", "upper center", "upper_center"):
            x_uc, y_uc, z_uc = x, y, verdepth
            xy_uo = (x_uc + y_uc * 1j) - cpl_length / 2
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth
        elif pointpos in ("ue", "UE", "upper end", "upper_end"):
            x_ue, y_ue, z_ue = x, y, verdepth
            xy_uo = (x_ue + y_ue * 1j) - cpl_length
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth
        elif pointpos in ("be", "BE", "bottom end", "bottom_end"):
            x_be, y_be, z_be = x, y, verdepth
            xy_uo = (x_be + y_be * 1j) - cpl_length - cpl_width
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("bc", "BC", "bottom center", "bottom_center"):
            x_bc, y_bc, z_bc = x, y, verdepth
            xy_uo = (x_bc + y_bc * 1j) - cpl_length / 2 - cpl_width
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("bo", "BO", "bottom origin", "bottom_origin"):
            x_bo, y_bo, z_bo = x, y, verdepth
            xy_uo = (x_bo + y_bo * 1j) - cpl_width
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("cc", "CC", "centroid center", "centroid_center"):
            x_cc, y_cc, z_cc = x, y, verdepth
            xy_uo = (x_cc + y_cc * 1j) - cpl_width / 2 - cpl_length / 2
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip) / 2
        else:
            raise ValueError("Please specify a correct point position VALUE!")

        # upper center point
        xy_uc = (x_uo + y_uo * 1j) + cpl_length / 2
        x_uc, y_uc, z_uc = xy_uc.real, xy_uc.imag, z_uo
        # upper right point
        xy_ue = (x_uo + y_uo * 1j) + cpl_length
        x_ue, y_ue, z_ue = xy_ue.real, xy_ue.imag, z_uo
        # bottom right point
        xy_be = (x_ue + y_ue * 1j) + cpl_width
        x_be, y_be, z_be = xy_be.real, xy_be.imag, z_uo - width * np.sin(dip)
        # bottom center point
        xy_bc = (x_be + y_be * 1j) - cpl_length / 2
        x_bc, y_bc, z_bc = xy_bc.real, xy_bc.imag, z_uo - width * np.sin(dip)
        # bottom left point
        xy_bo = (x_bc + y_bc * 1j) - cpl_length / 2
        x_bo, y_bo, z_bo = xy_bo.real, xy_bo.imag, z_uo - width * np.sin(dip)
        # fault center point
        x_cc, y_cc, z_cc = (x_uc + x_bc) / 2, (y_uc + y_bc) / 2, (z_uc + z_bc) / 2

        fault_corners = {
            "upper origin":       (x_uo, y_uo, z_uo),
            "upper center":     (x_uc, y_uc, z_uc),
            "upper end":      (x_ue, y_ue, z_ue),
            "bottom origin":      (x_bo, y_bo, z_bo),
            "bottom center":    (x_bc, y_bc, z_bc),
            "bottom end":     (x_be, y_be, z_be),
            "centroid center":  (x_cc, y_cc, z_cc),
        }

        return fault_corners



    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-


    def _initialize_fault(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
        """Initialize a rectangle fault plane by calculating the corner or central points of the fault, based
        on one point given.

        Args:
            - pointpos:       the position of the specified/original point:
                              "uo": upper origin point,  "uc": upper center point,  "ue": upper end point
                              "bo": bottom origin point, "bc": bottom center point, "be": bottom end point
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

        # if pointpos in ("ul", "UL", "upper left", "upper_left"):
        if pointpos in ("uo", "UO", "upper origin", "upper_origin"):
            x_uo, y_uo, z_uo = x, y, verdepth
        elif pointpos in ("uc", "UC", "upper center", "upper_center"):
            x_uc, y_uc, z_uc = x, y, verdepth
            xy_uo = (x_uc + y_uc * 1j) - cpl_length / 2
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth
        elif pointpos in ("ue", "UE", "upper end", "upper_end"):
            x_ue, y_ue, z_ue = x, y, verdepth
            xy_uo = (x_ue + y_ue * 1j) - cpl_length
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth
        elif pointpos in ("be", "BE", "bottom end", "bottom_end"):
            x_be, y_be, z_be = x, y, verdepth
            xy_uo = (x_be + y_be * 1j) - cpl_length - cpl_width
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("bc", "BC", "bottom center", "bottom_center"):
            x_bc, y_bc, z_bc = x, y, verdepth
            xy_uo = (x_bc + y_bc * 1j) - cpl_length / 2 - cpl_width
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("bo", "BO", "bottom origin", "bottom_origin"):
            x_bo, y_bo, z_bo = x, y, verdepth
            xy_uo = (x_bo + y_bo * 1j) - cpl_width
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip)
        elif pointpos in ("cc", "CC", "centroid center", "centroid_center"):
            x_cc, y_cc, z_cc = x, y, verdepth
            xy_uo = (x_cc + y_cc * 1j) - cpl_width / 2 - cpl_length / 2
            x_uo, y_uo, z_uo = xy_uo.real, xy_uo.imag, verdepth + width * np.sin(dip) / 2
        else:
            raise ValueError("Please specify a correct point position VALUE!")

        # upper center point
        xy_uc = (x_uo + y_uo * 1j) + cpl_length / 2
        x_uc, y_uc, z_uc = xy_uc.real, xy_uc.imag, z_uo
        # upper right point
        xy_ue = (x_uo + y_uo * 1j) + cpl_length
        x_ue, y_ue, z_ue = xy_ue.real, xy_ue.imag, z_uo
        # bottom right point
        xy_be = (x_ue + y_ue * 1j) + cpl_width
        x_be, y_be, z_be = xy_be.real, xy_be.imag, z_uo - width * np.sin(dip)
        # bottom center point
        xy_bc = (x_be + y_be * 1j) - cpl_length / 2
        x_bc, y_bc, z_bc = xy_bc.real, xy_bc.imag, z_uo - width * np.sin(dip)
        # bottom left point
        xy_bo = (x_bc + y_bc * 1j) - cpl_length / 2
        x_bo, y_bo, z_bo = xy_bo.real, xy_bo.imag, z_uo - width * np.sin(dip)
        # fault center point
        x_cc, y_cc, z_cc = (x_uc + x_bc) / 2, (y_uc + y_bc) / 2, (z_uc + z_bc) / 2

        fault = {
            "upper left":       (x_uo, y_uo, z_uo),
            "upper center":     (x_uc, y_uc, z_uc),
            "upper right":      (x_ue, y_ue, z_ue),
            "bottom left":      (x_bo, y_bo, z_bo),
            "bottom center":    (x_bc, y_bc, z_bc),
            "bottom right":     (x_be, y_be, z_be),
            "centroid center":  (x_cc, y_cc, z_cc),
            "length":           length,
            "width":            width
        }

        fault_corner = [
            [fault["upper left"], fault["upper right"], fault["bottom right"], fault["bottom left"]]
        ]


        return fault, fault_corner



    def read_from_trace(self):
        pass


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
        poly3d = Poly3DCollection(verts, edgecolor='black', alpha=0.7)

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
    # fault.plot(patch_corner1)




