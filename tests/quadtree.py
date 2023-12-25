# matplotlib for displaying the images
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import random
import math
import numpy as np


class Node(object):
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.children = []

    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_points(self, img):
        return img[self.x0:self.x0 + self.get_width(), self.y0:self.y0+self.get_height()]
    
    def get_error(self, img):
        pixels = self.get_points(img)
        return np.std(pixels)



class QTree():
    def __init__(self, stdThreshold, minPixelSize, img):
        self.threshold = stdThreshold
        self.min_size = minPixelSize
        self.minPixelSize = minPixelSize
        self.img = img
        self.root = Node(0, 0, img.shape[0], img.shape[1])
        self.qtimg = None

    
    def subdivide(self):
        recursive_subdivide(self.root, self.threshold, self.minPixelSize, self.img)

    def qtresults(self):
        c = find_children(self.root)
        results = []
        for n in c:
            pixels = n.get_points(self.img)
            mean_value = np.mean(pixels)
            new_x = (n.x0 + n.x0 + n.width) // 2
            new_y = (n.y0 + n.y0 + n.height) // 2
            results.append([new_x, new_y, mean_value])
        self.qtimg = np.array(results)

    def show_qtresults(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Data")
        plt.imshow(self.img, cmap="rainbow", origin="lower")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("QuadTree")
        plt.scatter(self.qtimg[:,0], self.qtimg[:,1], c=self.qtimg[:, 2], cmap="rainbow")
        plt.colorbar()


        import matplotlib.colorbar as cbar
        normal = plt.Normalize(self.img.min(), self.img.max())
        # color = plt.cm.rainbow(normal(self.img.reshape(-1, 1)))
        ax = plt.subplot(1, 3, 3)
        c = find_children(self.root)
        for n in c:
            mean_value = np.mean(n.get_points(self.img))
            color = plt.cm.rainbow(normal(mean_value))
            rect = patches.Rectangle((n.x0, n.y0), n.width, n.height, facecolor=color,
                                     edgecolor="black", fill=True)
            ax.add_patch(rect)
        cax, _ = cbar.make_axes(ax)
        cb2 = cbar.ColorbarBase(cax, cmap=plt.cm.rainbow, norm=normal)
        ax.set_xlim(0, self.img.shape[1])
        ax.set_ylim(0, self.img.shape[0])
        plt.show()
        plt.close()


    
    # def graph_tree(self):
    #     fig = plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.title('orignal')
    #     plt.imshow(img, cmap='rainbow')
    #     plt.colorbar()
    #
    #     plt.subplot(1, 2, 2)
    #     plt.title("Quadtree")
    #     c = find_children(self.root)
    #     print(f"Number of segments: {len(c)}")
    #     for n in c:
    #         pixels = n.get_points(self.img)
    #         mean_value = np.mean(pixels)
    #         rect = patches.Rectangle((n.y0, n.x0), n.height, n.width, edgecolor="black", fill=True,
    #                                                     facecolor=plt.cm.rainbow(mean_value))
    #         plt.gcf().gca().add_patch(rect)
    #     plt.gcf().gca().set_xlim(0, img.shape[1])
    #     plt.gcf().gca().set_ylim(img.shape[0], 0)
    #     # Add colorbar to the second subplot
    #     cax = plt.axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    #     cb = plt.colorbar(cax=cax)
    #     cb.set_label('Mean Value')  # Set colorbar label
    #
    #     # plt.axis('equal')
    #     plt.show()
    #     return



def recursive_subdivide(node, k, minPixelSize, img):

    # if node.get_error(img) <= k:
    #     return
    # w_1 = int(math.floor(node.width/2))
    # w_2 = int(math.ceil(node.width/2))
    # h_1 = int(math.floor(node.height/2))
    # h_2 = int(math.ceil(node.height/2))
    #
    # if w_1 <= minPixelSize or h_1 <= minPixelSize:
    #     return
    #
    # x1 = Node(node.x0, node.y0, w_1, h_1) # top left
    # recursive_subdivide(x1, k, minPixelSize, img)
    # x2 = Node(node.x0, node.y0 + h_1, w_1, h_2) # btm left
    # recursive_subdivide(x2, k, minPixelSize, img)
    # x3 = Node(node.x0 + w_1, node.y0, w_2, h_1)# top right
    # recursive_subdivide(x3, k, minPixelSize, img)
    # x4 = Node(node.x0 + w_1, node.y0 + h_1, w_2, h_2) # btm right
    # recursive_subdivide(x4, k, minPixelSize, img)
    # node.children = [x1, x2, x3, x4]

    if node.get_error(img) <= k:
        return
    middle_w1, middle_h1 = math.floor(node.width / 2), math.floor(node.height / 2)
    middle_w2, middle_h2 = math.ceil(node.width / 2), math.ceil(node.height / 2)

    if middle_w1 <= minPixelSize or middle_h1 <= minPixelSize:
        return

    x1 = Node(node.x0, node.y0, middle_w1, middle_h1)  # top left
    recursive_subdivide(x1, k, minPixelSize, img)
    x2 = Node(node.x0 + middle_w1, node.y0, middle_w2, middle_h1)  # topo right
    recursive_subdivide(x2, k, minPixelSize, img)
    x3 = Node(node.x0, node.y0 + middle_h1, middle_w1, middle_h2)  # btm left
    recursive_subdivide(x3, k, minPixelSize, img)
    x4 = Node(node.x0 + middle_w1, node.y0 + middle_h1, middle_w2, middle_h2) # btm right
    recursive_subdivide(x4, k, minPixelSize, img)
    node.children = [x1, x2, x3, x4]
   

def find_children(node):
   if not node.children:
       return [node]
   else:
       children = []
       for child in node.children:
           children += (find_children(child))
   return children



if __name__ == '__main__':
    # img = cv2.imread('test.jpg')
    # printI(img)

    delta = 0.025
    x = y = np.arange(-3.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 20
    img = Z

    qtTemp = QTree(np.std(img)-3, 4, img)  #contrast threshold, min cell size, img
    qtTemp.subdivide() # recursively generates quad tree
    # qtTemp.graph_tree()
    qtTemp.qtresults()
    qtTemp.show_qtresults()




# ========================================================================================================
