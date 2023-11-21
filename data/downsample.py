#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10.11.23

Author: Zelong Guo, @ GFZ, Potsdam
Email: zelong.guo@outlook.com
Created on Sat May 20 20:29:30 2023

"""

# downsampling and also extract the azimuth and incidence for every point
# Down-sampling images could be writen as a class

class Dsampler(object):
    # class attribute

    def __init__(self, block_size):
