#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21.07.23

@author: Zelong Guo
@GFZ, Potsdam
"""
# from data import plot_data
# import os
# fig_path = plot_data.check_folder4figs('test.png')
# fig_path = fig_path + 'a'
# print(f'Now the bs is {fig_path}')
#
# print(os.path.basename(__file__))
# # os.path.

# class Student(object):
#     def __init__(self, name="anoy", age=None):
#         self.name = name
#         self.age = age
#
#     def print_s(self):
#         print(f"NAME: {self.name}, AGE: {self.age}")
#
# # class inheritance
#
# class UniStudent(Student):
#
#     def __init__(self, name=None, age=None, score=100):
#         super().__init__(name, age)
#
#         self.score = score
#         self.sex = None
#
# bart = Student('barnhat', 12)
# # tim = UniStudent('timit', 15, 99)
# a = UniStudent()
#
# bart.print_s()

# # -------------------------------------------------------------
# def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
#     print("-- This parrot wouldn't", action, end=' ')
#     print("if you put", voltage, "volts through it.")
#     print("-- Lovely plumage, the", type)
#     print("-- It's", state, "!")
#
# parrot(voltage=5.0, 'dead')  # non-keyword argument after a keyword argument

# # -------------------------------------------------------------
# import numpy as np
#
# x = np.array([1,2,3])
# y = np.array([4,5,6,7])
#
# xv,yv = np.meshgrid(x,y,indexing = 'xy')
#
# xv2,yv2 = np.meshgrid(x,y,indexing = 'ij')
#
# print('-----向量的形状-----')
# print(x.shape)
# print(y.shape)
#
# print('-----xy-----')
# print(xv.shape)
# print(yv.shape)
#
# print('-----ij-----')
# print(xv2.shape)
# print(yv2.shape)
#
# print(f"xv: {xv}")
# print(f"yv: {yv}")

# # -------------------------------------------------------------
#
# from pyproj import CRS
# from pyproj.aoi import AreaOfInterest
# from pyproj.database import query_utm_crs_info
#
# utm_crs_list = query_utm_crs_info(
#     datum_name="WGS 84",
#     area_of_interest=AreaOfInterest(
#         west_lon_degree=-93.581543,
#         south_lat_degree=42.032974,
#         east_lon_degree=-93.581543,
#         north_lat_degree=42.032974,
#     ),
# )
#
# print(utm_crs_list)
#
# utm_crs = CRS.from_epsg(utm_crs_list[0].code)
#
# print(utm_crs)

# # -------------------------------------------------------------
import numpy as np

a = np.arange(3*4).reshape(3, 4).flatten().reshape(-1, 1)
b = np.arange(3*4).reshape(3, 4).flatten().reshape(-1, 1)
c = np.hstack([a, b])

print("---------------")
print(a)
print("---------------")
print(b)
print("---------------")
print(c)