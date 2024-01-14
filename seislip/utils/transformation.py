#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14.01.24

@author: Zelong Guo
@GFZ, Potsdam
"""
import numpy as np

# ---------------------------------------------------------------------------------
class Transformation(object):
    """ Transformation between Cartesian (UTM) and fault coordinate systems using homogeneous equations.
    Construct the transformation matrix between them.

    The definition of the fault plane coordinate system:
    This coordinate system is right-hand:
    X:      along strike
    Y:      along opposite direction of dip
    Z:      normal direction the fault
    The origin of the fault coordinate system is bottom left/right corner point.

    """
    def __init__(self):
        """ Initialize the transformation matrix M."""
        self.M = np.identity(4)

    def translation(self, T):
        M = np.array([[1.0, 0.0, 0.0, T[0]],
                      [0.0, 1.0, 0.0, T[1]],
                      [0.0, 0.0, 1.0, T[2]],
                      [0.0, 0.0, 0.0, 1.0]])
        self.M = M.dot(self.M)

    def rotation_x(self, theta):
        M = np.array([[1.0, 0.0,           0.0,            0.0],
                      [0.0, np.cos(theta), -np.sin(theta), 0.0],
                      [0.0, np.sin(theta), np.cos(theta),  0.0],
                      [0.0, 0.0,           0.0,            1.0]])

        self.M = M.dot(self.M)

    def rotation_y(self, alpha):
        M = np.array([[np.cos(alpha),   0.0, np.sin(alpha), 0.0],
                      [0.0,             1.0, 0.0,           0.0],
                      [-np.sin(alpha),  0.0, np.cos(alpha), 0.0],
                      [0.0,             0.0, 0.0,           1.0]])

        self.M = M.dot(self.M)

    def rotation_z(self, beta):
        M = np.array([[np.cos(beta), -np.sin(beta), 0.0, 0.0],
                      [np.sin(beta), np.cos(beta),  0.0, 0.0],
                      [0.0,          0.0,           1.0, 0.0],
                      [0.0,          0.0,           0.0, 1.0]])

        self.M = M.dot(self.M)

    def inverse(self):
        """Inverse of transformation matrix."""
        Minv = np.linalg.inv(self.M)
        self.Minv = Minv

    def _homogenous(self, points):
        """The points should be an array of mx3, indicating the point number is m and their three
        position components.

        Args:
            - points:           array of m x 3

        Return:
            Homogenous matrix.
        """
        points = np.array(points, dtype=float, copy=True)
        m, n = np.shape(points)
        homo = np.ones((m, 1))
        return np.concatenate((points, homo), axis=-1).transpose()

    def _inhomogenous(self, points):
        """Return the inhomogenous results. """
        points = np.array(points, dtype=float, copy=True)
        return points[:-1, ...].transpose()

    def forwars_trans(self, points):
        points = self._homogenous(points)
        new_points = np.dot(self.M, points)
        return self._inhomogenous(new_points)

    def inverse_trans(self, points):
        points = self._homogenous(points)
        new_points = np.dot(self.Minv, points)
        return self._inhomogenous(new_points)

# ---------------------------------------------------------------------------------

if __name__ == "__main__":

    trans = Transformation()
    trans.translation(T=[1, 1, 1])
    trans.rotation_z(np.radians(30))
    trans.rotation_x(np.radians(60))
    trans.rotation_y(np.radians(45))
    trans.inverse()

    point = [[3, 4, 5],
             [6, 7, 8],
             [9, 10, 11],
             [12, 13, 14]]

    new_points = trans.forwars_trans(point)
    old_point = trans.inverse_trans(new_points)

    # homo_points = homogenous(point)
    # new_points = np.dot(trans.M, homo_points)
    # new_points = inhomogenous(new_points)
    #
    # new_points = homogenous(new_points)
    # old_points = np.dot(trans.Minv, new_points)
    # old_points = inhomogenous(old_points)
