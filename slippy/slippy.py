#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the base class for SLipPy.

Created on Tue Nov. 21 2023
@author: Zelong Guo, GFZ, Potsdam
"""


class SlipPy(object):
    """A parent class to SOME modules of SlipPy.

    In this parent class, we have defined properties and methods that SOME subclasses need to share.

    Args:
        name:

    Return:

    """
    def __init__(self, name, lon0=None, lat0=None, ellps="WGS84", utmzone=None):
        self.name = name
        self.lon0 = lon0
        self.lat0 = lat0
        self.ellps = ellps
        self.utmzone = utmzone

    def test_stuff(self):
        print("This is a test of the parent class SlipPy.")
        print(__file__)
 
