#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    (1) deg2utm and utm2deg
    (2) deramp and remove dem-related errors
    
Created on Sat May 20 20:29:30 2023

@author: zelong
"""

import utm
import numpy as np

def deg2utm(lats, lons):
    """
    The default is WGS84 system and its zone number.

    """
    # lats and lons shouls have the same dimensions
    utm_easting, utm_northing, utm_zone, _ = utm.from_latlon(lats.flatten(), lons.flatten())
    # _, _, _, utm_zone_letter = utm.from_latlon(latitude, longitude)
    utm_easting = utm_easting.reshape(lats.shape)
    utm_northing = utm_northing.reshape(lons.shape)
    utm_zone = utm_zone.reshape(utm_zone.shape)
    return utm_easting, utm_northing, utm_zone

# -----------------------------------------------------------------------------
# need to futher modification
def utm2deg(utm_easting, utm_northing, utm_zone, utm_zone_letter):
    lats, lons = utm.to_latlon(utm_easting, utm_northing, utm_zone, utm_zone_letter)
    return lats, lons
    pass 
# -----------------------------------------------------------------------------


def deramp_dem(unw, mask, dem, deramp_method):
    pass





