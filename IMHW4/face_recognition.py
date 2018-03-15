import time
from math import fabs, hypot, inf, pi, sqrt
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import ndimage, signal
from scipy.spatial import distance
from skimage import io
from sklearn import preprocessing


class Train:
    def __init__(self, mypath):
        self.dataset_ = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        self.dataset_.sort()
        self.data_matrix_, self.img_aver_ = self.read_data_set(self.dataset_)
        
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
        np.savetxt('average.txt', img_aver, delimiter=',', fmt='%i')
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
        return np.transpose(Vh)

    def method_third(self):
        t = time.time()
        xtx_mat = np.dot(np.transpose(self.data_matrix_), self.data_matrix_)
        eigenvalue, eigenvector = LA.eig(xtx_mat)
        eigenvector = np.dot(self.data_matrix_, eigenvector)
        dt = time.time() - t
        np.savetxt('eigenvalue.txt', eigenvalue, delimiter=',', fmt='%i')
        np.savetxt('eigenvector.txt', eigenvector, delimiter=',', fmt='%i')
        return eigenvalue, eigenvector

    def find_n_eigenvector(self, n, eigenvalue, eigenvector):
        varray = np.array(eigenvalue,dtype='int32')
        ind = np.argpartition(varray, -n)[-n:]
        self.eigenfaces_ = eigenvector[:,ind]

        self.eigenfaces_ = preprocessing.normalize(self.eigenfaces_, norm='l1')

        self.eigenfaces_ = np.transpose(self.eigenfaces_)

        print(self.eigenfaces_.shape)
        np.savetxt('eigenfaces.txt', self.eigenfaces_, delimiter=',')

    
    def calssify(self, n):
        self.face_class_ = []

        weightvec_img = np.zeros((1,n))

        count = 0
        
        for img in self.dataset_:
            im = io.imread(img)
            weightvec = []
            for vec in self.eigenfaces_:
                weightvec.append(np.dot(np.transpose(vec), np.subtract(im, self.img_aver_).flatten()))
            weightvec_img = np.add(weightvec_img, np.array(weightvec))
            count += 1
            if (count == 9):
                count = 0
                self.face_class_.append(np.divide(weightvec, 9))
                weightvec_img = np.zeros((1,n))
        # print(len(self.dataset_))
        # print(len(self.face_class_))

    def show_eigenfaces(self):
        for face in self.eigenfaces_:
             face = np.reshape(face, (io.imread(self.dataset_[0]).shape[0], io.imread(self.dataset_[0]).shape[1]))
             io.imshow(face)
             plt.show()


class TestImg:
    def __init__(self, eigenfaces, img_aver, img, face_class):
        self.eigenfaces_ = eigenfaces
        self.img_ = io.imread(img)
        self.img_aver_ = img_aver
        self.face_class_ = face_class
        
    def projection(self):
        self.weightvec_ = np.dot(self.eigenfaces_, np.subtract(self.img_, self.img_aver_).flatten())
        print(self.eigenfaces_.shape)
        print(self.weightvec_)
        # self.weightvec_ = []
        # for vec in self.eigenfaces_:
        #     print(vec)
        #     self.weightvec_.append(np.dot(np.transpose(vec), np.subtract(self.img_, self.img_aver_).flatten()))
        # self.weightvec_ = preprocessing.normalize(np.array(self.weightvec_).reshape(-1, 1), axis=0, norm='l2')
        print(self.weightvec_.shape)
        np.savetxt('weight.txt', self.weightvec_, delimiter=',')

    def reconstruct(self):
        self.reimg_ = np.zeros_like(self.eigenfaces_[0])
        for index, item in enumerate(self.weightvec_):
            self.reimg_ = self.reimg_ + np.multiply(item, np.array(self.eigenfaces_[index]))
        self.reimg_ = np.transpose(self.reimg_)
        self.reimg_ = np.reshape(self.reimg_, (self.img_.shape[0], self.img_.shape[1]))
        # print(self.reimg_.shape)
        io.imshow(self.reimg_, cmap='gray')
        plt.show()
        self.reimg_ = self.reimg_ + self.img_aver_
        print(self.img_aver_)

        print(self.reimg_)
        io.imshow(self.reimg_, cmap='gray')
        plt.show()

    def find_class(self, th1):
        i_min = 0
        dis_min = inf
        for index, face in enumerate(self.face_class_):
            dis = distance.euclidean(self.weightvec_, face)
            if  dis < th1 and dis < dis_min:
                i_min = index
                dis_min = dis
        match = io.imread('TrainSet/subject' + str(i_min).zfill(2) +'.normal')
        io.imshow(match)
        plt.show()
                

if __name__ == '__main__':
    mypath = 'TrainSet'
    f = Train(mypath)
    # f.show_img_average()
    # print(f.data_matrix_)
    eigenvalue, eigenvector = f.method_third()
    f.find_n_eigenvector(10, eigenvalue, eigenvector)
    f.calssify(10)
    f.show_eigenfaces()
    t = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet/subject09.wink', f.face_class_)
    t.projection()
    t.reconstruct()
    t.find_class(100)
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
