from math import hypot, inf, pi, sqrt, fabs

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import ndimage, signal
from skimage import io
from os import listdir
from os.path import isfile, join
import time

def read_data_set(dataset):
    data_matrix = []
    img_sum = np.zeros_like(io.imread(dataset[0]), dtype='int32')
    for image in dataset:
        img = io.imread(image)
        img_sum = np.add(img_sum, img)
        data_matrix.append(img.flatten())
    img_aver = np.divide(img_sum, len(dataset)).astype('uint8')
    # io.imshow(img_aver)
    # plt.show()
    data_matrix = np.array(data_matrix)
    img_aver_flatten = img_aver.flatten()
    # do I need to substract img_aver??? What if it is negative value??
    # for i in data_matrix:
    #     print(i - img_aver_flatten)
    data_matrix = np.transpose(data_matrix)
    return data_matrix, img_aver

def method_one(data_matrix):
    covariance_matrix = np.dot(data_matrix, np.transpose(data_matrix))
    w, v = LA.eig(covariance_matrix)
    return v

def method_second(data_matrix):
    y_mat = np.divide(np.transpose(data_matrix), sqrt(data_matrix[0]-1))
    U,s,Vh = LA.svd(y_mat)
    return Vh

def method_third(data_matrix):
    xtx_mat = np.dot(np.transpose(data_matrix), data_matrix)
    w, v = LA.eig(xtx_mat)
    v = data_matrix * v
    return v

if __name__ == '__main__':
    mypath = 'yalefaces_centered_small'
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    data_matrix, img_aver = read_data_set(onlyfiles)
    t1 = time.time()
    v1 = method_one(data_matrix)
    dt1 = time.time() - t1
    print(dt1)
    print(v1)
    # v2 = method_second(data_matrix)
    # print(v2)
    # v3 = method_third(data_matrix)
    # print(v3)