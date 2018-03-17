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
        io.imshow(self.img_aver_, cmap='gray')
        plt.show()

    def method_one(self):
        t = time.time()
        covariance_matrix = np.dot(self.data_matrix_, np.transpose(self.data_matrix_))
        eigenvalue, eigenvector = LA.eig(covariance_matrix)
        print("Method 1 takes ", time.time() - t)
        return eigenvector
        
    def method_second(self):
        t = time.time()
        y_mat = np.divide(self.data_matrix_.T, sqrt(self.data_matrix_.shape[0]-1))
        U, s, Vh = LA.svd(y_mat)
        print("Method 2 takes ", time.time() - t, " s")
        np.savetxt("vec2.txt", U.T, delimiter=',')
        return U.T

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
        print("Method 3 takes ", time.time() - t, " s")
        return eigenvalue, eigenvector

    def find_n_eigenvector(self, n, eigenvalue, eigenvector):
        varray = np.array(eigenvalue, dtype='int32')
        # get n largest indices
        ind = np.argpartition(varray, -n)[-n:]
        self.eigenfaces_ = eigenvector[ind, :]
        # normalize the eigenfaces
        self.eigenfaces_ = preprocessing.normalize(self.eigenfaces_, axis=1, norm='l2')
    
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
        self.img_ = io.imread(img, as_grey=True)
        self.img_.reshape((img_aver.shape[0], -1))
        self.img_aver_ = img_aver
        self.face_class_ = face_class
        
    def projection(self):
        # eigenfaces * image
        self.weightvec_ = self.eigenfaces_ @ np.transpose(np.subtract(self.img_, self.img_aver_).flatten())

    def reconstruct(self):
        self.reimg_ = np.transpose(self.weightvec_) @ self.eigenfaces_
        self.reimg_ = np.reshape(self.reimg_, (self.img_.shape[0], self.img_.shape[1]))
        self.reimg_f = self.reimg_ + self.img_aver_
        # show_face(self.img_, self.reimg_f)

    def find_class(self, th1):
        i_min = 0
        dis_min = inf
        for index, face in enumerate(self.face_class_):
            dis = distance.euclidean(self.weightvec_, face)
            if  dis < th1 and dis < dis_min:
                i_min = index + 1
                dis_min = dis
        match = self.face_class_[i_min - 1] @ self.eigenfaces_
        match = np.reshape(match, (self.img_aver_.shape[0], self.img_aver_.shape[1]))
        match = match + self.img_aver_
        print("match")
        show_face(self.img_, match)
    
    def image_diff(self):
        self.diff = self.img_ - self.reimg_f
        show_face3(self.img_, self.reimg_f, self.diff)
        diff = LA.norm(self.diff)
        global diff_norm
        diff_norm.append(diff)

diff_norm = []
                
def show_face(img1, img3):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(img3, cmap='gray')
    # plt.draw()
    # plt.pause(0.001)
    plt.show(block=False)

def show_face3(img1, img2, img3):
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img1, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(img3, cmap='gray')
    # plt.draw()
    # plt.pause(0.001)
    plt.show(block=False)

if __name__ == '__main__':
    mypath = 'TrainSet2'
    f = Train(mypath)
    eigenvalue, eigenvector = f.method_third()
    ei2 = f.method_second()
    np.savetxt("vec3.txt", preprocessing.normalize(eigenvector, axis=0, norm='l2'), delimiter=',')
    f.find_n_eigenvector(15, eigenvalue, eigenvector)
    f.calssify(15)
    # f.show_img_average()
    # t1 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject02.glasses', f.face_class_)
    # t1.projection()
    # t1.reconstruct()
    # t1.image_diff()
    # t2 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject09.surprised', f.face_class_)
    # t2.projection()
    # t2.reconstruct()
    # t2.image_diff()
    # t3 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject15.centerlight', f.face_class_)
    # t3.projection()
    # t3.reconstruct()
    # t3.image_diff()
    # t4 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject01.glasses', f.face_class_)
    # t4.projection()
    # t4.reconstruct()
    # t4.image_diff()
    # t5 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject03.noglasses', f.face_class_)
    # t5.projection()
    # t5.reconstruct()
    # t5.image_diff()
    # t6 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject04.centerlight', f.face_class_)
    # t6.projection()
    # t6.reconstruct()
    # t6.image_diff()
    # t7 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject05.glasses', f.face_class_)
    # t7.projection()
    # t7.reconstruct()
    # t7.image_diff()
    # t8 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject06.normal', f.face_class_)
    # t8.projection()
    # t8.reconstruct()
    # t8.image_diff()
    # t9 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject07.surprised', f.face_class_)
    # t9.projection()
    # t9.reconstruct()
    # t9.image_diff()
    # t10 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject13.centerlight', f.face_class_)
    # t10.projection()
    # t10.reconstruct()
    # t10.image_diff()
    # t11 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject01.surprised', f.face_class_)
    # t11.projection()
    # t11.reconstruct()
    # t11.image_diff()
    # t12 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject02.centerlight', f.face_class_)
    # t12.projection()
    # t12.reconstruct()
    # t12.image_diff()
    # t13 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject13.glasses', f.face_class_)
    # t13.projection()
    # t13.reconstruct()
    # t13.image_diff()
    # t14 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject03.glasses', f.face_class_)
    # t14.projection()
    # t14.reconstruct()
    # t14.image_diff()
    # t15 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject05.surprised', f.face_class_)
    # t15.projection()
    # t15.reconstruct()
    # t15.image_diff()
    # t16 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject06.glasses', f.face_class_)
    # t16.projection()
    # t16.reconstruct()
    # t16.image_diff()
    # t17 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject08.centerlight', f.face_class_)
    # t17.projection()
    # t17.reconstruct()
    # t17.image_diff()
    # t18 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject09.glasses', f.face_class_)
    # t18.projection()
    # t18.reconstruct()
    # t18.image_diff()
    # t19 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject10.centerlight', f.face_class_)
    # t19.projection()
    # t19.reconstruct()
    # t19.image_diff()
    # t20 = TestImg(f.eigenfaces_, f.img_aver_, 'TestSet2/subject11.centerlight', f.face_class_)
    # t20.projection()
    # t20.reconstruct()
    # t20.image_diff()

    # t1.find_class(10000)
    # t2.find_class(10000)
    # t3.find_class(10000)
    # t4.find_class(10000)
    # t5.find_class(10000)
    # t6.find_class(10000)
    # t7.find_class(10000)
    # t8.find_class(10000)
    # t9.find_class(10000)
    # t10.find_class(10000)
    # t11.find_class(10000)
    # t12.find_class(10000)
    # t13.find_class(10000)
    # t14.find_class(10000)
    # t15.find_class(10000)
    # t16.find_class(10000)
    # t17.find_class(10000)
    # t18.find_class(10000)
    # t19.find_class(10000)
    # t20.find_class(10000)
    # testpath = 'NonfaceImages'
    # nonface = [join(testpath, f) for f in listdir(testpath) if isfile(join(testpath, f))]
    # print(nonface)
    # t = [TestImg(f.eigenfaces_, f.img_aver_, nonface[0], f.face_class_)] * len(nonface)
    # for index, image in enumerate(nonface):
    #     t[index] = TestImg(f.eigenfaces_, f.img_aver_, image, f.face_class_)
    #     t[index].projection()
    #     t[index].reconstruct()
    #     t[index].image_diff()

    # plt.figure()
    # x = np.array(range(1,21, 1))
    # plt.plot(x, np.array(diff_norm).reshape(-1, 1), 'bo')

    plt.show()

