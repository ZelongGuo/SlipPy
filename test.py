#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21.07.23

@author: Zelong Guo
@GFZ, Potsdam
"""
from data import plot_data
import os
fig_path = plot_data.check_folder4figs('test.png')
fig_path = fig_path + 'a'
print(f'Now the bs is {fig_path}')

print(os.path.basename(__file__))
# os.path.