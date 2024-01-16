#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rectangle fault Patch Construction.

Created on 15.01.24

@author: Zelong Guo
"""
__author__ = "Zelong Guo"

# standard libs
import sys
import numpy as np
import math


if __name__ == "__main__":
    sys.path.append("../")
    from seislip.fault.fault import Fault
    from seislip.utils.transformation import Transformation
else:
    from .fault import Fault
    from ..utils.transformation import Transformation


# -----------------------------------------------------------------------------------------
class RectPatch(Fault):
    """Planar/rectangle fault grid generation with rectangle patches.

    Args:
        - fault:        Fault object

    """
    def __init__(self, name, lon0, lat0, ellps="WGS84", utmzone=None):
        super().__init__(name, lon0, lat0, ellps, utmzone)

        self.trans = None


    def initialize_rectangle_patch(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
        # initialize the fault parameters
        self.initialize_fault(pointpos, lon, lat, verdepth, strike, dip, length, width)

        # origin to UTM
        utm_x, utm_y = self.ll2xy(self.origin[1][0], self.origin[1][1])
        utm_z = self.origin[1][2]
        origin = (utm_x, utm_y, utm_z)

        # constructing transformation between UTM and fault coordinate systems
        trans = Transformation()
        trans.translation(T=origin)
        trans.rotation_x(np.radians(self.dip))
        trans.rotation_z(np.radians(90-self.strike))
        trans.inverse()
        self.trans = trans


        # if not hasattr(self.fault, "origin") or getattr(self.fault, "origin") is None:
        #     raise AttributeError(f"The fault object {fault.name} does not specify attribute 'origin' yet!")

    def utm2fault(self, points):
        """UTM to fault coordinate system.

        Args:
            - points:           point lists in UTM ccordinates, m x 3 array
        Return:
            - point list in fault coodinates, m x 3 array
        """
        return self.trans.forwars_trans(points)

    def fault2utm(self, points):
        """fault coordinate system to UTM."""
        return self.trans.inverse_trans(points)

    def discretize(self, patch_length, patch_width):
        if self.origin[0] in ("uc", "UC", "upper center", "upper_center"):
            x_num = math.ceil(self.length/patch_length)
            y_num = math.ceil(self.width/patch_width)
            x = np.linspace(-self.length/2, self.length/2, x_num+1)
            y = np.linspace(-self.width, 0, y_num+1)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros((X.shape[0], X.shape[1]))
            X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
            patch_points = np.array([X, Y, Z]).transpose()

            patch_points = np.array2list(patch_points)

            # patch_points = [
            #     [patch_points]
            # ]

            return patch_points







# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # fault = Fault("flt", 44.28, 35.47)
    # fault.initialize_fault(pointpos="upper center", lon=44.34, lat=35.603, verdepth=-3,
    #                        strike=10, dip=45, length=80, width=50)

    patch = RectPatch("pat", 44.28, 35.47)
    patch.initialize_rectangle_patch(pointpos="upper center", lon=44.34, lat=35.603, verdepth=-3,
                           strike=10, dip=45, length=80, width=50)

    patch_points = patch.discretize(2, 2)
    pputm = patch.fault2utm(patch_points)
    patch.plot(pputm)







