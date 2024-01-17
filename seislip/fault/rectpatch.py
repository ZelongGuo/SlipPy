#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rectangle Fault Patch Construction.

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

    """
    def __init__(self, name, lon0, lat0, ellps="WGS84", utmzone=None):
        super().__init__(name, lon0, lat0, ellps, utmzone)

        # fault parameters
        self.origin = None
        self.strike = None
        self.dip = None
        self.length = None
        self.width = None

        self.trans = None


    def initialize_rectangle_patch(self, pointpos: str, lon, lat, verdepth, strike, dip, length, width):
        """Initialize the fault parameters of the whole fault need to be meshed, and construct the
        transformation matrix between Cartesian/UTM and local fault coordinate system.

        Args:
            - pointpos:         point position you specified, "uo", "uc", "ue" ...
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

        # initialize the fault parameters
        self.initialize_fault(pointpos, lon, lat, verdepth, strike, dip, length, width)

        # origin to UTM
        utm_x, utm_y = self.ll2xy(self.origin[1][0], self.origin[1][1])
        utm_z = self.origin[1][2]

        # constructing transformation between UTM and fault coordinate systems
        trans = Transformation()
        trans.rotation_x(np.radians(self.dip))
        trans.rotation_z(np.radians(90-self.strike))
        trans.translation(T=(utm_x, utm_y, utm_z))
        trans.inverse()
        self.trans = trans

        # if not hasattr(self.fault, "origin") or getattr(self.fault, "origin") is None:
        #     raise AttributeError(f"The fault object {fault.name} does not specify attribute 'origin' yet!")

    def utm2fault(self, points):
        """UTM to fault coordinate system.

        Args:
            - points:           point lists in UTM ccordinates, m x 3 list/array
        Return:
            - point list in fault coodinates, m x 3 list/array
        """
        return self.trans.inverse_trans(points)

    def fault2utm(self, points):
        """fault coordinate system to UTM."""
        return self.trans.forwars_trans(points)

    def discretize(self, patch_length, patch_width):
        x_num = math.ceil(self.length / patch_length)
        y_num = math.ceil(self.width / patch_width)
        patch = []

        if self.origin[0] in ("uo", "UO", "upper origin", "upper_origin"):
            x = np.linspace(0, self.length, x_num + 1)
            y = np.linspace(-self.width, 0, y_num + 1)
        elif self.origin[0] in ("uc", "UC", "upper center", "upper_center"):
            x = np.linspace(-self.length/2, self.length/2, x_num+1)
            y = np.linspace(-self.width, 0, y_num+1)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((X.shape[0], X.shape[1]))
        # X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        # patch_points = np.array([X, Y, Z]).transpose()

        # patch_points = [
        #     [patch_points]
        # ]
        for i in range(y_num):
            for j in range(x_num):
                x1, y1, z1 = X[i, j],       Y[i, j],        Z[i, j]
                x2, y2, z2 = X[i, j+1],     Y[i, j+1],      Z[i, j+1]
                x3, y3, z3 = X[i+1, j+1],   Y[i+1, j+1],    Z[i+1, j+1]
                x4, y4, z4 = X[i+1, j],     Y[i+1, j],      Z[i+1, j]
                rectangle = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
                rectangle = self.fault2utm(rectangle)
                patch.append(rectangle)

        return patch



    # def plot(self, verts):
    #     self.plot(verts)



# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    fault = Fault("flt", 44.28, 35.47)
    # fault.initialize_fault(pointpos="upper center", lon=44.34, lat=35.603, verdepth=-3,
    #                        strike=10, dip=45, length=80, width=50)
    _, fcorner = fault._initialize_fault(pointpos="upper origin", lon=44.34, lat=35.603, verdepth=-3,
                           strike=255, dip=45, length=80, width=50)
    fault.plot(fcorner)

    patch = RectPatch("pat", 44.28, 35.47)
    patch.initialize_rectangle_patch(pointpos="upper origin", lon=44.34, lat=35.603, verdepth=-3,
                           strike=255, dip=45, length=80, width=50)

    patch_points = patch.discretize(3, 2)
    # pputm = patch.fault2utm(patch_points)
    patch.plot(patch_points)







