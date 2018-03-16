import time
from math import fabs, hypot, inf, pi, sqrt
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import ndimage, signal
from scipy.linalg import eigh as largest_eigh
from scipy.spatial import distance
from skimage import io
from sklearn import preprocessing


class Train:
    def __init__(self, mypath):
        # readin all images
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
        # get mean average
        img_aver = np.divide(img_sum, len(dataset))

        data_matrix = np.array(data_matrix, dtype='int32')
        img_aver_flatten = img_aver.flatten()
        for index, _ in enumerate(data_matrix):
            # substract every image by the mean image
            data_matrix[index] = np.subtract(data_matrix[index], img_aver_flatten)
        # transpose, then every column is a image
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
        print(self.data_matrix_.shape)
        # AtA
        xtx_mat = np.transpose(self.data_matrix_) @ self.data_matrix_
        eigenvalue, eigenvector = LA.eig(xtx_mat)
        print(eigenvector.shape)
        eigenvector = self.data_matrix_ @ eigenvector
        print(eigenvector.shape)
        # every row is an eigen face
        eigenvector = eigenvector.T

        # print(eigenvector.shape)

        # evals_large, evecs_large = largest_eigh(xtx_mat, eigvals=(self.data_matrix_.shape[1]-50,self.data_matrix_.shape[1]-1))
        # self.eigenfaces_ = self.data_matrix_ @ evecs_large
        # self.eigenfaces_ = preprocessing.normalize(self.eigenfaces_, axis=0, norm='l2')
        # self.eigenfaces_ = self.eigenfaces_.T
        

        dt = time.time() - t
        np.savetxt('eigenvalue.txt', eigenvalue, delimiter=',', fmt='%i')
        np.savetxt('eigenvector.txt', eigenvector, delimiter=',', fmt='%i')
        return eigenvalue, eigenvector

    def find_n_eigenvector(self, n, eigenvalue, eigenvector):
        varray = np.array(eigenvalue, dtype='int32')
        # get n largest indices
        ind = np.argpartition(varray, -n)[-n:]
        print(eigenvector.shape)
        self.eigenfaces_ = eigenvector[ind, :]
        print(self.eigenfaces_.shape)
        # normalize the eigenfaces
        self.eigenfaces_ = preprocessing.normalize(self.eigenfaces_, axis=1, norm='l2')
        # eigenfaces becomes coloums
        # self.eigenfaces_ = self.eigenfaces_.T


        print(self.eigenfaces_.shape)
        np.savetxt('eigenfaces.txt', self.eigenfaces_, delimiter=',')
    
    def calssify(self, n):
        self.face_class_ = []

        weightvec_img = np.zeros((1,n))

        count = 0
        
        for img in self.dataset_:
            im = io.imread(img)
            weightvec = self.eigenfaces_ @ np.transpose(np.subtract(im, self.img_aver_).flatten())

            weightvec_img = weightvec_img + np.array(weightvec)
            count += 1
            if (count == 9):
                count = 0
                self.face_class_.append(np.divide(weightvec_img, 9))
                weightvec_img = np.zeros((1,n))
        # print(len(self.dataset_))
        # print(len(self.face_class_))

    def show_eigenfaces(self):
        for face in self.eigenfaces_:
             face = np.reshape(face, (io.imread(self.dataset_[0]).shape[0], io.imread(self.dataset_[0]).shape[1]))
             io.imshow(face, cmap='gray')
             plt.show()
        
    def show_faces_class(self):
        for vec in self.face_class_:
            img = vec @ self.eigenfaces_
            img = np.reshape(img, (self.img_aver_.shape[0], self.img_aver_.shape[1]))
            plt.imshow(img)
            plt.show()


class TestImg:
    def __init__(self, eigenfaces, img_aver, img, face_class):
        self.eigenfaces_ = eigenfaces
        self.img_ = io.imread(img)
        self.img_aver_ = img_aver
        self.face_class_ = face_class
        
    def projection(self):
        # eigenfaces * image
        self.weightvec_ = self.eigenfaces_ @ np.transpose(np.subtract(self.img_, self.img_aver_).flatten())
        # self.weightvec_ = []
        # for vec in self.eigenfaces_:
        #     print(vec)
        #     self.weightvec_.append(np.dot(np.transpose(vec), np.subtract(self.img_, self.img_aver_).flatten()))
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        # self.weightvec_ = min_max_scaler.fit_transform(np.array(self.weightvec_).reshape(-1, 1))
        # self.weightvec_ = preprocessing.normalize(np.array(self.weightvec_).reshape(-1, 1))
        np.savetxt('weight.txt', self.weightvec_, delimiter=',')

    def reconstruct(self):
        # self.reimg_ = np.zeros_like(self.eigenfaces_[0])
        # for index, item in enumerate(self.weightvec_):
        #     print(item)
        #     self.reimg_ = self.reimg_ + np.multiply(item, np.array(self.eigenfaces_[index]))
        self.reimg_ = np.transpose(self.weightvec_) @ self.eigenfaces_
        # self.reimg_ = np.transpose(self.reimg_)
        # self.reimg_ = preprocessing.normalize(self.reimg_.reshape(1, -1), norm='l2')
        self.reimg_ = np.reshape(self.reimg_, (self.img_.shape[0], self.img_.shape[1]))
        # print(self.reimg_.shape)

        self.reimg_f = self.reimg_ + self.img_aver_
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,256))
        # self.reimg_f = min_max_scaler.fit_transform(self.reimg_f)
        print(self.img_aver_)

        
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,255))
        # self.reimg_f = min_max_scaler.fit_transform(self.reimg_f)
        print(self.reimg_f)
        np.savetxt('re.txt', self.reimg_f, delimiter=',')
        show_face(self.img_, self.reimg_, self.reimg_f)

    def find_class(self, th1):
        i_min = 0
        dis_min = inf
        for index, face in enumerate(self.face_class_):
            dis = distance.euclidean(self.weightvec_, face)
            print(dis)
            if  dis < th1 and dis < dis_min:
                print(i_min)
                i_min = index + 1
                dis_min = dis
        match = io.imread('TrainSet/subject' + str(i_min).zfill(2) +'.normal')
        print(i_min)
        io.imshow(match)
        plt.show()
                
def show_face(img1, img2, img3):
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img1, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(img3, cmap='gray')
    plt.show()

if __name__ == '__main__':
    mypath = 'TrainSet'
    f = Train(mypath)
    # f.show_img_average()
    # print(f.data_matrix_)
    eigenvalue, eigenvector = f.method_third()
    f.find_n_eigenvector(100, eigenvalue, eigenvector)
    f.calssify(100)
    # f.show_faces_class()
    # f.show_eigenfaces()
    # plt.imshow(f.img_aver_)
    # plt.show()
    t1 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet/subject02.glasses', f.face_class_)
    t1.projection()
    t1.reconstruct()
    t2 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet/subject09.surprised', f.face_class_)
    t2.projection()
    t2.reconstruct()
    t3 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet/subject15.centerlight', f.face_class_)
    t3.projection()
    t3.reconstruct()
    t1.find_class(10000)
    t2.find_class(10000)
    t3.find_class(10000)

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
