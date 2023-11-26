#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the base class for SLipPy.

Created on Tue Nov. 21 2023
@author: Zelong Guo, GFZ, Potsdam
"""


class SlipPy(object):
    """
    A parent class to other modules of SlipPy.
    """
    def __init__(self, name):
        self.name = name

    def test_stuff(self):
        print("This is a test of the parent class SlipPy.")
        print(__file__)
 
