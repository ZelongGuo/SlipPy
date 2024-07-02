#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21.07.23

@author: Zelong Guo, Potsdam

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

# import matplotlib.colorbar as cbar
# from matplotlib import pyplot as plt
# import numpy as np
#
# N = 2
# xs = np.random.randint(0, 100, N)
# ys = np.random.randint(0, 100, N)
# ws = np.random.randint(10, 20, N)
# hs = np.random.randint(10, 20, N)
# vs = np.random.randn(N)
# normal = plt.Normalize(vs.min(), vs.max())
# colors = plt.cm.jet(normal(vs))
#
# ax = plt.subplot(111)
# for x,y,w,h,c in zip(xs,ys,ws,hs,colors):
#     rect = plt.Rectangle((x,y),w,h,color=c)
#     ax.add_patch(rect)
#
# cax, _ = cbar.make_axes(ax)
# cb2 = cbar.ColorbarBase(cax, cmap=plt.cm.jet,norm=normal)
#
# ax.set_xlim(0,120)
# ax.set_ylim(0,120)
# plt.show()

# import numpy as np
#
# a = np.random.rand(5, 4)
# a[1:3, -1] = 0
# for i in range(a.shape[0]):
#
#     print(f"m: {a[i, 0]}, n: {a[i, 1]}, p: {a[i,2]}, q: {a[i, 3]}")
#
# print("all done.")

# import sys
# sys.path.append("../slippy/utils/")
#
# from quadtree import QTree
# import numpy as np
#
# delta = 0.025
# x = y = np.arange(-3.0, 5.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = np.exp(-X ** 2 - Y ** 2)
# Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
# Z = (Z1 - Z2) * 20
# img = Z
# # set up nan
# img[:, 0:120] = 0
# img[:, -1] = 0
# img[0, :] = 0
# img[-1, :] = 0
#
# qtTemp = QTree(X, Y, img)  # contrast threshold, min cell size, img
# qtTemp.subdivide(16, 64, np.std(img) - 2)  # recursively generates quad tree
# qtTemp.qtresults(0.3)
# qtTemp.show_qtresults()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# def plot_3d_fault(xc, yc, zc, length, width, strike, dip):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 转换走向和倾向为弧度
#     strike_rad = np.radians(strike)
#     dip_rad = np.radians(dip)
#
#     # 生成断层的坐标
#     x = np.linspace(xc - length / 2, xc + length / 2, 100)
#     y = np.linspace(yc - width / 2, yc + width / 2, 100)
#     x, y = np.meshgrid(x, y)
#
#     # 根据走向和倾向计算断层坐标
#     z = zc + (x - xc) * np.tan(dip_rad) * np.cos(strike_rad) + (y - yc) * np.tan(dip_rad) * np.sin(strike_rad)
#
#     # 绘制3D断层
#     ax.plot_surface(x, y, z, color='b', alpha=0.6)
#
#     # 设置坐标轴标签
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     # 显示图形
#     plt.show()
#
# # 例子：中心位置坐标为 (1, 2, 3)，长度为 5，宽度为 2，走向为 45 度，倾向为 30 度
# plot_3d_fault(xc=1, yc=2, zc=3, length=5, width=2, strike=45, dip=30)
#

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
#
#
# def create_fault_patch(center_x, center_y, depth, length, width, strike, dip):
#     # 计算矩形的角度
#     angle_rad = np.radians(strike)
#
#     # 计算矩形的四个顶点坐标
#     dx = 0.5 * length * np.cos(angle_rad)
#     dy = 0.5 * length * np.sin(angle_rad)
#     x = np.array([center_x - dx, center_x + dx, center_x + dx, center_x - dx, center_x - dx])
#     y = np.array([center_y - dy, center_y - dy, center_y + dy, center_y + dy, center_y - dy])
#
#     # 构建矩形断层片
#     fault_patch = Rectangle((x[0], y[0]), length, width, angle=np.degrees(angle_rad), fill=None, edgecolor='r')
#
#     return fault_patch
#
#
# # 断层参数
# center_x = 10.0  # 中心点 x 坐标
# center_y = 10.0  # 中心点 y 坐标
# depth = 5.0  # 深度
# length = 20.0  # 长度
# width = 5.0  # 宽度
# strike = 45.0  # 走向
# dip = 30.0  # 倾向
#
# # 创建矩形断层片
# fault_patch = create_fault_patch(center_x, center_y, depth, length, width, strike, dip)
#
# # 绘制图形
# fig, ax = plt.subplots()
# ax.add_patch(fault_patch)
# ax.set_xlim(center_x - length, center_x + length)
# ax.set_ylim(center_y - length, center_y + length)
# ax.set_aspect('equal', adjustable='datalim')
# plt.title('Fault Patch')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# def create_fault(x0, y0, z0, azimuth, dip, length, width):
#     # 生成网格坐标
#     x = np.linspace(x0 - length / 2, x0 + length / 2, 2)
#     y = np.linspace(y0 - width / 2, y0 + width / 2, 2)
#     x, y = np.meshgrid(x, y)
#
#     # 根据断层的走向和倾向计算z坐标
#     z = z0 + (x - x0) * np.tan(np.radians(dip)) + (y - y0) * np.tan(np.radians(azimuth))
#
#     return x, y, z
#
# # 已知的断层参数
# x0, y0, z0 = 0, 0, 0  # 断层上边缘中点的坐标
# azimuth = 45  # 断层走向
# dip = 30  # 断层倾向
# length = 20  # 断层长度
# width = 10  # 断层宽度
#
# # 创建3D矩形断层片
# x, y, z = create_fault(x0, y0, z0, azimuth, dip, length, width)
#
# # 可视化
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
#
# # 设置坐标轴标签
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
#
# # 显示图形
# plt.show()

# z1 = 1 + 2j
# z2 = 2 + 3j
# z3 = z1 + z2
# z4 = z1 * z2

# -------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import numpy as np
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 定义立方体的八个顶点
# vertices = [
#     [1, 1, 1],
#     [1, 1, 0],
#     [1, 0, 1],
#     [1, 0, 0],
#     [0, 1, 1],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 0, 0]
# ]
#
# # 定义立方体的六个面，每个面由四个顶点组成
# faces = [
#     [vertices[0], vertices[1], vertices[3], vertices[2]],
#     [vertices[4], vertices[5], vertices[7], vertices[6]],
#     [vertices[0], vertices[1], vertices[5], vertices[4]],
#     [vertices[2], vertices[3], vertices[7], vertices[6]],
#     [vertices[0], vertices[2], vertices[6], vertices[4]],
#     [vertices[1], vertices[3], vertices[7], vertices[5]]
#     ]
#
#
#
# # 创建 Poly3DCollection 对象
# poly3d = Poly3DCollection(faces, edgecolor='k')
#
# # 将 Poly3DCollection 添加到三维坐标轴上
# ax.add_collection3d(poly3d)
#
# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# plt.show()

# -------------------------------------------------------------------------------

# import math
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# from matplotlib.collections import PolyCollection
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
#
# def polygon_under_graph(x, y):
#     """
#     Construct the vertex list which defines the polygon filling the space under
#     the (x, y) line graph. This assumes x is in ascending order.
#     """
#     return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]
#
#
# ax = plt.figure().add_subplot(projection='3d')
#
# x = np.linspace(0., 10., 31)
# lambdas = range(1, 9)
#
# # verts[i] is a list of (x, y) pairs defining polygon i.
# gamma = np.vectorize(math.gamma)
# verts = [polygon_under_graph(x, l**x * np.exp(-l) / gamma(x + 1))
#          for l in lambdas]
# facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))
#
# poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
# ax.add_collection3d(poly, zs=lambdas, zdir='y')
#
# ax.set(xlim=(0, 10), ylim=(1, 9), zlim=(0, 0.35),
#        xlabel='x', ylabel=r'$\lambda$', zlabel='probability')
#
# plt.show()

# -------------------------------------------------------------------------------
# x = [1, 2,3 ,4]
# y = [4, 5, 6, 7]
# z = [8, 9, 10, 11]
#
# for m, n, q in zip(x, y ,z):
#     print(f"Line:{m}, {n}, {q}")


# import matplotlib.pyplot as plt
# import numpy as np
#
# from matplotlib import cbook, cm
# from matplotlib.colors import LightSource
#
# # Load and format data
# # dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
# # z = dem['elevation']
# nrows, ncols = 1000, 1000
# x = np.linspace(0, 2000, ncols)
# y = np.linspace(0, 2000, nrows)
# x, y = np.meshgrid(x, y)
# z = x * x + y * x
#
# region = np.s_[5:50, 5:50]
# x, y, z = x[region], y[region], z[region]
#
# # Set up plot
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#
# ls = LightSource(270, 45)
# # To use a custom hillshading mode, override the built-in shading and pass
# # in the rgb colors of the shaded surface calculated from "shade".
# rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
#                        linewidth=0, antialiased=False, shade=False)
#
# plt.show()


#
# import math
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# from matplotlib.collections import PolyCollection
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
#
# def polygon_under_graph(x, y):
#     """
#     Construct the vertex list which defines the polygon filling the space under
#     the (x, y) line graph. This assumes x is in ascending order.
#     """
#     return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]
#
#
# ax = plt.figure().add_subplot(projection='3d')
#
# x = np.linspace(0., 10., 31)
# lambdas = range(1, 9)
#
# # verts[i] is a list of (x, y) pairs defining polygon i.
# gamma = np.vectorize(math.gamma)
# verts = [polygon_under_graph(x, l**x * np.exp(-l) / gamma(x + 1)) for l in lambdas]
# facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))
#
# poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
# ax.add_collection3d(poly, zs=lambdas, zdir='y')
#
# ax.set(xlim=(0, 10), ylim=(1, 9), zlim=(0, 0.35),
#        xlabel='x', ylabel=r'$\lambda$', zlabel='probability')
#
# plt.show()


# pointpos_list = (("uo", "UO", "upper origin", "upper_origin"),
#                  ("uc", "UC", "upper center", "upper_center"),
#                  ("ue", "UE", "upper end", "upper_end"),
#                  ("bo", "BO", "bottom origin", "bottom_origin"),
#                  ("bc", "BC", "bottom center", "bottom_center"),
#                  ("be", "BE", "bottom end", "bottom_end"),
#                  ("cc", "CC", "centroid center", "centroid_center"))
#
# # For now only support "uc"
# # pointpos_list = ("uc",  "UC",  "upper center",    "upper_center")
# a = "uo"
# if a not in pointpos_list:
#     raise ValueError("Please specify a right point position!")
# else:
#     print("a is in the list!")

#
# a = (1, 2, 3)
# b = (4, 5, 6)
# c = (7, 8, 9)
#
# d = [a, b, c]
# e = [d]
#
# def split_line_segment(length, sublen, ratio, increasing=True):
#     lengths = [sublen * ratio**i for i in range(int(length / sublen))]
#     if not increasing:
#         lengths = lengths[::-1]
#
#     starting_coordinates = [(sum(lengths[:i]), 0) for i in range(len(lengths))]
#
#     return lengths, starting_coordinates
#
# # 示例调用
# length_of_line = 20
# subsegment_length = 2
# growth_ratio = 1.5
# is_increasing = True
#
# lengths, starting_coords = split_line_segment(length_of_line, subsegment_length, growth_ratio, is_increasing)
#
# for i, (length, coords) in enumerate(zip(lengths, starting_coords), 1):
#     print(f"Subsegment {i}: Length = {length}, Starting Coordinates = {coords}")

import numpy as np



# def line_plane_intersect(x, y, z, sx, sy, sz):
#     '''
#     Calculate the intersection of a line and a plane using a parametric
#     representation of the plane. This is hardcoded for a vertical line.
#     '''
#
#     numerator = np.array([[1., 1., 1., 1.],
#                           [x[0], x[1], x[2], sx],
#                           [y[0], y[1], y[2], sy],
#                           [z[0], z[1], z[2], sz]])
#
#     numerator = np.linalg.det(numerator)
#
#     denominator = np.array([[1., 1., 1., 1.],
#                             [x[0], x[1], x[2], 0],
#                             [y[0], y[1], y[2], 0],
#                             [z[0], z[1], z[2], -sz]])
#
#     denominator = np.linalg.det(denominator)
#
#     if denominator == 0:
#         denominator = np.spacing(1)
#
#     t = numerator / denominator
#     d = np.array([sx, sy, sz]) - t * (np.array([sx, sy, 0])-
#                                       np.array([sx, sy, sz]))
#     return d

from numpy import (pi,cross,dot,sin,cos,tan,arctan2,log,
                   sqrt,array,asarray,zeros,empty,copy,
                   nonzero)
from numpy.linalg import norm,det
def line_plane_intersect(x, y, z, sx, sy, sz):
  # Calculate the intersection of a line and a plane using a parametric
  # representation of the plane.  This is hardcoded for a vertical line.
  numerator = array([[   1,    1,    1,  1],
                     [x[0], x[1], x[2], sx],
                     [y[0], y[1], y[2], sy],
                     [z[0], z[1], z[2], sz]])
  numerator = det(numerator)
  denominator = array([[   1,    1,    1,   0],
                       [x[0], x[1], x[2],   0],
                       [y[0], y[1], y[2],   0],
                       [z[0], z[1], z[2], -sz]])
  denominator = det(denominator)
  if denominator == 0:
    print('hey')
    denominator = 1e-10

  t = numerator/denominator # parametric curve parameter
  d = array([sx,sy,sz])-array([0,0,-sz])*t
  return d

# import numpy as np
# def line_plane_intersect(p1, p2, p3, sx, sy, sz):
#     """
#     Calculate the intersection of a line and a plane using a parametric
#     representation of the plane.  This is hardcoded for a vertical line.
#
#     Args:
#         * sx        : x coordinates of ground points
#         * sy        : y coordinates of ground points
#         * sz        : z coordinates of ground points
#         * p1        : xyz tuple or list of first triangle vertex
#         * p2        : xyz tuple or list of second triangle vertex
#         * p3        : xyz tuple or list of third triangle vertex
#     """
#     # Extract the separate x,y,z values
#     x1, y1, z1 = p1
#     x2, y2, z2 = p2
#     x3, y3, z3 = p3
#
#     numerator = np.array([[1.0, 1.0, 1.0, 1.0],
#                           [x1, x2, x3, sx],
#                           [y1, y2, y3, sy],
#                           [z1, z2, z3, sz]])
#     numerator = np.linalg.det(numerator)
#     denominator = np.array([[1.0, 1.0, 1.0, 0.0],
#                             [x1, x2, x3, 0.0],
#                             [y1, y2, y3, 0.0],
#                             [z1, z2, z3, -sz]])
#     denominator = np.linalg.det(denominator)
#     if denominator == 0:
#         denominator = 1.0e-15
#     # Parameteric curve parameter
#     t = numerator / denominator
#     d = np.array([sx, sy, sz]) - np.array([0.0, 0.0, -sz]) * t
#
#     return d


x = [-50, 52, 45]
y = [-80, 80, -80]
z = [-3, 59, 34]
sx, sy, sz = 10, 11, 56

d = line_plane_intersect(x, y, z, sx, sy, sz)
print(f"result: {d}")
