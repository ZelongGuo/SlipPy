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

# a = {
#     "lon":  {"value": [1, 2, 3], "unit": "deg"},
#     "lat":  {"value": [4, 5, 6], "unit": "deg"},
#     "los":  {"value": [0.1, 0.2, 0.3], "unit": "m"}
# }
#
# def test(key):
#     if key not in a:
#         raise ValueError(f"key {key} is not in a!")
#     print(a[key]["value"])
# def test2(key):
#     match key:
#         case "lon" | "LON" | "longi":
#             print(a["lon"]["value"])
#         case "lat" | "Lat":
#             print(a["lat"]["value"])
#         case _:
#             raise ValueError("Error!")
#
# test2('lon')
# print("+-" * 50)
# test2("Lat")

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
# down_factor = 4
#
# a = [i for i in range(1, 99, 1)]
# b = a[::down_factor]
#
# import math
# import numpy as np
# b_num = math.ceil(len(a)/down_factor)
#
# print(f"a width is {len(a)}")
# print(f"real b width is {len(b)}, calculated b width is {b_num}")
#
# c = np.array(a)
# d = c.reshape((2,-1))

# import numpy as np
# import matplotlib.pyplot as plt
#
# class InSARQuadtree:
#     def __init__(self, data):
#         self.data = data
#         self.result = []
#
#     def quadtree_downsampling(self, block, mindim, maxdim, std_threshold, fraction=0.3):
#         split, m, n = self.should_split(block, mindim, maxdim, std_threshold, fraction)
#
#         if split == 2:
#             print("忽略一个块，因为非NaN元素的比例太低。")
#             return None
#         elif split == 0:
#             mean_value, middle_i, middle_j = self.get_block_info(block)
#             print(f"块大小 ({m}, {n}) 已达到最小限制，降采样块的均值为 {mean_value:.4f}。")
#             self.result.append((middle_i, middle_j, mean_value))
#             return block
#         elif split == 1:
#             print(f"分裂一个块，块大小为 ({m}, {n})。")
#             middle_m, middle_n = m // 2, n // 2
#             upper_left = block[:middle_m, :middle_n]
#             upper_right = block[:middle_m, middle_n:]
#             lower_left = block[middle_m:, :middle_n]
#             lower_right = block[middle_m:, middle_n:]
#
#             # 递归调用，将子块的返回值赋给相应的位置
#             upper_left = self.quadtree_downsampling(upper_left, mindim, maxdim, std_threshold, fraction)
#             upper_right = self.quadtree_downsampling(upper_right, mindim, maxdim, std_threshold, fraction)
#             lower_left = self.quadtree_downsampling(lower_left, mindim, maxdim, std_threshold, fraction)
#             lower_right = self.quadtree_downsampling(lower_right, mindim, maxdim, std_threshold, fraction)
#
#             # 将子块的返回值合并成一个新的块
#             new_block = np.vstack([np.hstack([upper_left, upper_right]),
#                                    np.hstack([lower_left, lower_right])])
#             return new_block
#
#     def should_split(self, block, mindim, maxdim, std_threshold, fraction=0.3):
#         m, n = np.shape(block)
#         nonnan_num = np.count_nonzero(~np.isnan(block))
#         nonnan_fraction = nonnan_num / (m * n)
#
#         if nonnan_fraction <= fraction:
#             split = 2
#         elif m > maxdim or n > maxdim:
#             split = 1
#         elif m <= mindim or n <= mindim:
#             split = 0
#         else:
#             nonnan = block[~np.isnan(block)]
#             std = np.std(nonnan)
#
#             if std > std_threshold:
#                 split = 1
#             else:
#                 split = 0
#
#         return split, m, n
#
#     def get_block_info(self, block):
#         m, n = block.shape
#         nonnan = block[~np.isnan(block)]
#         mean_value = np.mean(nonnan)
#         middle_i, middle_j = self.get_coord_quadtree(block)
#         return mean_value, middle_i, middle_j
#
#     def get_coord_quadtree(self, block):
#         m, n = block.shape
#         data = self.data
#         m_limit, n_limit = data.shape[0] - m + 1, data.shape[1] - n + 1
#         tolerance = 1e-8
#
#         for i in range(m_limit):
#             for j in range(n_limit):
#                 if np.allclose(data[i:i+m, j:j+n], block, atol=tolerance):
#                     middle_i = i + m // 2
#                     middle_j = j + n // 2
#                     return middle_i, middle_j
#         raise ValueError("在原始数据中未找到块的索引！")
#
# # 示例使用
# # 创建一个示例 InSAR 数据（假设为二维数组）
# x = y = np.arange(-1, 1.0, 0.025)
# X, Y = np.meshgrid(x, y)
# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# insar_data = (Z1 - Z2) * 2
# # insar_data = np.random.rand(64, 64)
#
# # 创建 InSARQuadtree 实例
# quadtree = InSARQuadtree(insar_data)
#
# # 进行四叉树降采样
# std = np.std(insar_data)
# quadtree.quadtree_downsampling(insar_data, mindim=2, maxdim=16, std_threshold=std, fraction=0.2)
#
# # # 输出结果
# # print("降采样后的结果:")
# # print(quadtree.result)
#
# # 可视化原始数据和降采样结果
# fig, ax = plt.subplots(figsize=(8, 8))
#
# # ax.imshow(insar_data, cmap='viridis')
# # ax.set_title("original")
#
# # 绘制降采样结果
# result_array = np.array(quadtree.result)
# ax.scatter(result_array[:, 1], result_array[:, 0], c=result_array[:, 2], cmap='viridis', marker='s', s=50)
#
# plt.show()

# from pyqtree import Index
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 生成模拟的 InSAR 数据
# np.random.seed(42)
# image_size = 512
# insar_data = np.random.rand(image_size, image_size)
#
# # 定义四叉树的边界框
# bbox = (0, 0, image_size, image_size)
#
# # 创建四叉树索引
# spatial_index = Index(bbox=bbox)
#
# # 将数据插入四叉树
# for row in range(image_size):
#     for col in range(image_size):
#         point_bbox = (row, col, row + 1, col + 1)
#         spatial_index.insert(item=(row, col), bbox=point_bbox)
#
# # 定义降采样的目标分辨率
# target_resolution = 64
#
# # 初始化降采样后的结果数组
# downsampled_data = np.zeros((image_size // target_resolution, image_size // target_resolution))
#
# # 遍历降采样后的网格
# for row in range(0, image_size, target_resolution):
#     for col in range(0, image_size, target_resolution):
#         # 查询四叉树，获取与当前网格相交的原始数据点
#         intersected_points = spatial_index.intersect((row, col, row + target_resolution, col + target_resolution))
#
#         # 计算当前网格的平均值，并填充降采样后的数组
#         if intersected_points:
#             values = [insar_data[row, col] for (row, col) in intersected_points]
#             average_value = np.mean(values)
#             downsampled_data[row // target_resolution, col // target_resolution] = average_value
#
# # 显示原始数据和降采样后的结果
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(insar_data, cmap='viridis')
# plt.title('Original InSAR Data')
#
# plt.subplot(1, 2, 2)
# plt.imshow(downsampled_data, cmap='viridis')
# plt.title('Downsampled InSAR Data')
#
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# def quadtree_downsample(matrix, threshold=0.5):
#     """
#     对一个128x128的二维矩阵进行四叉树降采样。
#
#     参数：
#     - matrix: 输入的128x128矩阵
#     - threshold: 降采样的阈值，小于该值的单元将被合并
#
#     返回：
#     - 降采样后的均值矩阵
#     - 降采样后的中心点坐标
#     """
#
#     def recursive_downsample(submatrix, row_start, col_start, size):
#         # 检查是否满足降采样条件
#         if np.all(submatrix <= threshold):
#             # 降采样：计算均值和中心点坐标
#             avg_value = np.mean(submatrix)
#             center_row = row_start + size // 2
#             center_col = col_start + size // 2
#             return [(avg_value, (center_row, center_col))]
#
#         # 不满足条件，继续分割为四个子矩阵
#         size //= 2
#         half_size = size // 2
#         quadrants = [
#             (row_start, col_start, size),
#             (row_start, col_start + half_size, size),
#             (row_start + half_size, col_start, size),
#             (row_start + half_size, col_start + half_size, size)
#         ]
#
#         result = []
#         for quadrant in quadrants:
#             r_start, c_start, quad_size = quadrant
#             submatrix = matrix[r_start:r_start + quad_size, c_start:c_start + quad_size]
#             result.extend(recursive_downsample(submatrix, r_start, c_start, quad_size))
#
#         return result
#
#     # 开始递归降采样
#     result = recursive_downsample(matrix, 0, 0, len(matrix))
#     avg_matrix = np.zeros_like(matrix, dtype=float)
#
#     # 更新降采样后的均值矩阵
#     for avg_value, (center_row, center_col) in result:
#         avg_matrix[center_row, center_col] = avg_value
#
#     return avg_matrix, result
#
# # 示例用法
# # 生成一个随机的128x128矩阵作为形变场数据
# delta = 0.025
# x = y = np.arange(-3.0, 3.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# Z = (Z1 - Z2) * 2
# random_matrix = Z
#
# # 进行四叉树降采样
# downsampled_matrix, cell_info = quadtree_downsample(random_matrix, np.std(random_matrix)*0.005)
#
# # # 打印结果
# # print("降采样后的均值矩阵:")
# # print(downsampled_matrix)
# # print("\n降采样后的中心点坐标和均值:")
# # for avg_value, (center_row, center_col) in cell_info:
# #     print(f"中心点坐标: ({center_row}, {center_col}), 均值: {avg_value}")
#
# # 绘制原始图像
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('原始图像')
# plt.imshow(random_matrix, cmap='viridis', origin='upper')
#
# # 绘制降采样后的结果
# plt.subplot(1, 2, 2)
# plt.title('降采样后的结果')
# plt.imshow(downsampled_matrix, cmap='viridis', origin='upper')
# plt.colorbar()
#
#
# plt.show()

import matplotlib.colorbar as cbar
from matplotlib import pyplot as plt
import numpy as np

N = 2
xs = np.random.randint(0, 100, N)
ys = np.random.randint(0, 100, N)
ws = np.random.randint(10, 20, N)
hs = np.random.randint(10, 20, N)
vs = np.random.randn(N)
normal = plt.Normalize(vs.min(), vs.max())
colors = plt.cm.jet(normal(vs))

ax = plt.subplot(111)
for x,y,w,h,c in zip(xs,ys,ws,hs,colors):
    rect = plt.Rectangle((x,y),w,h,color=c)
    ax.add_patch(rect)

cax, _ = cbar.make_axes(ax)
cb2 = cbar.ColorbarBase(cax, cmap=plt.cm.jet,norm=normal)

ax.set_xlim(0,120)
ax.set_ylim(0,120)
plt.show()