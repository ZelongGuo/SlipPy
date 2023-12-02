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
# from pyproj import CRS, Transformer
# from pyproj.aoi import AreaOfInterest
# from pyproj.database import query_utm_crs_info
# import numpy as np
#
# crs1 = CRS("WGS84")
# print("----- crs1 -------")
# print(crs1)
#
# utm_crs_list = query_utm_crs_info(
#     datum_name= "WGS84", #"NAD83", #"WGS 84",
#     area_of_interest=AreaOfInterest(
#         west_lon_degree=-95,
#         south_lat_degree=41,
#         east_lon_degree=-91,
#         north_lat_degree=45,
#     ),
# )
#
#
# # print(utm_crs_list)
#
# utm_crs = CRS.from_epsg(utm_crs_list[0].code)
# print("----- utm_crs -------")
# print(utm_crs)
#
# lonlat = np.array([[-90.2897635, 40.1467463],
#                    [-91.4456356, 43.5353664],
#                    [-94.7463463, 44.8363636],
#                    [-94.9236646, 42.9463463]])
#
#
# transformer1 = Transformer.from_crs(crs1, utm_crs, always_xy=True)
# b1, b2 = transformer1.transform(lonlat[:, 0], lonlat[:, 1])
#
# transformer2 = Transformer.from_crs(utm_crs,utm_crs.geodetic_crs, always_xy=True)
# c = transformer2.transform(b1, b2)
#
# transformer3 = Transformer.from_crs(utm_crs, crs1, always_xy=True)
# d = transformer3.transform(b1, b2)
#
# utm_zone_crs = CRS(proj="utm", zone=22, ellps="WGS84")
# transformer4 = Transformer.from_crs(crs1, utm_zone_crs, always_xy=True)
# e = transformer4.transform(lonlat[:, 0], lonlat[:, 1])

# # # -------------------------------------------------------------
# import numpy as np
#
# a = np.arange(3*4).reshape(3, 4).flatten().reshape(-1, 1)
# b = np.arange(3*4).reshape(3, 4).flatten().reshape(-1, 1)
# c = np.hstack([a, b])
#
# print("---------------")
# print(a)
# print("---------------")
# print(b)
# print("---------------")
# print(c)

# #-------------------------------------------------------------
#
# import numpy as np
#
# a = np.arange(1, 100, 1)
# b = a[::3]
#
# matrix_list = [[1, 2, 3, 3], [4, 5, 6, 6], [7, 8, 9,9], [10, 11, 12, 13]]
# c = matrix_list[::2]

# class A(object):
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#     def add_age(self, *args):
#         for i in args:
#             if i == "name":
#                 print(f"My name is {self.name}")
#             elif i == "age":
#                 print(f"Add age: {self.age +1}.")
#             else:
#                 raise ValueError("Wrong Value!!!")
#
# a = A("a", 20)
# print("+-" * 20)
# a.add_age("name")
# print("+-" * 20)
# a.add_age("age")
# print("+-" * 20)
# a.add_age("age", "name")

# define dict

a = {
    "lon":  {"value": [1, 2, 3], "unit": "deg"},
    "lat":  {"value": [4, 5, 6], "unit": "deg"},
    "los":  {"value": [0.1, 0.2, 0.3], "unit": "m"}
}

def test(key):
    if key not in a:
        raise ValueError(f"key {key} is not in a!")
    print(a[key]["value"])
def test2(key):
    match key:
        case "lon" | "LON" | "longi":
            print(a["lon"]["value"])
        case "lat" | "Lat":
            print(a["lat"]["value"])
        case _:
            raise ValueError("Error!")

test2('lon')
print("+-" * 50)
test2("Lat")

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
#
# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#
# # plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.scatter(x, y, c=area, cmap="hsv")
# plt.colorbar(label="label")
#
# plt.show()