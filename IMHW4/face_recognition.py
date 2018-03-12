from math import hypot, inf, pi, sqrt, fabs

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
from skimage import io
from os import listdir
from os.path import isfile, join

def read_data_set(dataset):
    data_matrix = []
    img_sum = np.zeros_like(io.imread(dataset[0]), dtype='int32')
    for image in dataset:
        img = io.imread(image)
        img_sum = np.add(img_sum, img)
        data_matrix.append(img.flatten())
    img_aver = np.divide(img_sum, len(dataset)).astype('uint8')
    io.imshow(img_aver)
    plt.show()
    data_matrix = np.array(data_matrix)
    data_matrix = np.transpose(data_matrix)
    return data_matrix, img_aver

if __name__ == '__main__':
    mypath = 'yalefaces_centered_small'
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    print(read_data_set(onlyfiles))

