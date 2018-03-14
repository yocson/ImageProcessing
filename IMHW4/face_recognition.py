from math import hypot, inf, pi, sqrt, fabs

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import ndimage, signal
from skimage import io
from os import listdir
from os.path import isfile, join
import time

class Train:
    def __init__(self, mypath):
        dataset = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        self.data_matrix_, self.img_aver_ = self.read_data_set(dataset)
        
    def read_data_set(self, dataset):
        data_matrix = []
        img_sum = np.zeros_like(io.imread(dataset[0]), dtype='int32')
        for image in dataset:
            img = io.imread(image)
            img_sum = np.add(img_sum, img)
            data_matrix.append(img.flatten())
        img_aver = np.divide(img_sum, len(dataset)).astype('uint8')

        data_matrix = np.array(data_matrix, dtype='int32')
        img_aver_flatten = img_aver.flatten()
        for index, _ in enumerate(data_matrix):
            data_matrix[index] = np.subtract(data_matrix[index], img_aver_flatten)
        data_matrix = np.transpose(data_matrix)
        return data_matrix, img_aver
    
    def show_img_average(self):
        io.imshow(self.img_aver_)
        plt.show()

    def method_one(self):
        t = time.time()
        covariance_matrix = np.dot(self.data_matrix_, np.transpose(self.data_matrix_))
        eigenvalue, eigenvector = LA.eig(covariance_matrix)
        dt = time.time() - t
        return eigenvector, dt
        
    def method_second(self):
        t = time.time()
        y_mat = np.divide(np.transpose(self.data_matrix_), sqrt(self.data_matrix_.shape[0]-1))
        U, s, Vh = LA.svd(y_mat)
        dt = time.time() - t
        return np.transpose(Vh), dt

    def method_third(self):
        t = time.time()
        xtx_mat = np.dot(np.transpose(self.data_matrix_), self.data_matrix_)
        eigenvalue, eigenvector = LA.eig(xtx_mat)
        eigenvector = np.dot(self.data_matrix_, eigenvector)
        dt = time.time() - t
        return eigenvalue, eigenvector

    def find_n_eigenvector(self, n, eigenvalue, eigenvector):
        varray = np.array(eigenvalue,dtype='int32')
        ind = np.argpartition(varray, -n)[-n:]
        self.eigenfaces_ = eigenvector[:,ind]


class TestImg:
    def __init__(self, eigenfaces, img_aver, img):
        self.eigenfaces_ = np.transpose(eigenfaces)
        self.img_ = io.imread(img)
        self.img_aver_ = img_aver
        
    def projection(self):
        self.weightvec = []
        for vec in self.eigenfaces_:
            self.weightvec.append(np.dot(np.transpose(vec), np.subtract(self.img_, self.img_aver_).flatten()))

    def reconstruct(self):
        self.reimg_ = np.zeros_like(self.eigenfaces_[0])
        for index, item in enumerate(self.weightvec):
            self.reimg_ = np.add(self.reimg_, np.multiply(item, np.array(self.eigenfaces_[index])))
        self.reimg_ = np.transpose(self.reimg_)
        self.reimg_ = np.reshape(self.reimg_, (self.img_.shape[0], self.img_.shape[1]))
        # print(self.reimg_.shape)
        self.reimg_ = np.add(self.reimg_, self.img_aver_)
        io.imshow(self.reimg_, cmap='gray')
        plt.show()

if __name__ == '__main__':
    mypath = 'TrainSet'
    f = Train(mypath)
    # f.show_img_average()
    # print(f.data_matrix_)
    eigenvalue, eigenvector = f.method_third()
    f.find_n_eigenvector(100, eigenvalue, eigenvector)
    t = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet/subject03.normal')
    t.projection()
    t.reconstruct()
    # t1 = time.time()
    # v1 = method_one(data_matrix)
    # dt1 = time.time() - t1
    # print(dt1)
    # print(v1)
    # v2 = method_second(data_matrix)
    # print(v2)
    # v3 = method_third(data_matrix)
    # print(v3)
    # test_img = np.array([[1,2], [3, 4]])
    # print(np.multiply(3, test_img))
    