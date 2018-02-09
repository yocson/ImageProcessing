#!/usr/bin/env python

# Cheng Wang(cwang76@syr.edu)
# CSE691 Spring 2018
# Code for HW1: Noise filtering

# you need to install the following packages to run the code

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from math import sqrt, pi

def mean_filter(image, size):
    kernel = np.divide(np.ones((size,size)), size*size)
    res = ndimage.convolve(image, kernel, mode='constant');
    return res

def gaussian(x, mu, sig):
    return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def generate_gaussian_kernel(sigma):
    gau = [gaussian(x, 0, sigma) for x in [-2, -1, 0, 1, 2]]
    normal_gau = [x/gau[2] for x in gau]
    # make the smallest one 1
    gau_kernel = [int(x/normal_gau[0]) for x in normal_gau]
    return np.array(gau_kernel)

def gaussian_filter(img, sigma):
    kernel = generate_gaussian_kernel(sigma).reshape(1, 5)
    # SEPAR FILTER Algorithm
    res = ndimage.convolve(img, np.divide(kernel,np.sum(kernel)), mode='constant')
    kernel = np.transpose(kernel)
    res = ndimage.convolve(res, np.divide(kernel,np.sum(kernel)), mode='constant')

    return res

def median_filter(img, size):
    margin = int(size/2)
    kernel = np.zeros((size, size))
    res = np.copy(img)
    for i in range(margin, img.shape[0]-margin+1):
        for j in range(margin, img.shape[1]-margin+1):
            kernel = img[i-margin:i+margin+1, j-margin:j+margin+1]
            res[i,j] = np.median(kernel)
    
    return res

def show_medianf(image):
    res1 = median_filter(image, 3)
    res2 = median_filter(image, 5)

    plt.figure('Median Filter')
    plt.subplot(1,3,1)
    plt.title('Before Median filter')
    io.imshow(image)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('After Median filter with size 3')
    io.imshow(res1)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('After Median filter with size 5')
    io.imshow(res2)
    plt.axis('off')
    plt.show()

def show_meanf(image):
    res1 = mean_filter(image, 3)
    res2 = mean_filter(image, 5)

    plt.figure('Mean Filter')
    plt.subplot(1,3,1)
    plt.title('Before mean filter')
    io.imshow(image)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('After mean filter with kernel size 3*3')
    io.imshow(res1)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('After mean filter with kernel size 5*5')
    io.imshow(res2)
    plt.axis('off')
    plt.show()

def show_gaussianf(image):
    res1 = gaussian_filter(image, 0.9)
    res2 = gaussian_filter(image, 2)

    plt.figure('Gaussian Filter')
    plt.subplot(1,3,1)
    plt.title('Before Gaussian filter')
    io.imshow(image)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('After Gaussian filter with sigma 0.9')
    io.imshow(res1)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('After Gaussian filter with sigma 2')
    io.imshow(res2)
    plt.axis('off')
    plt.show()    

if __name__ == "__main__":
    image1 = io.imread("NoisyImage1.jpg")
    image2 = io.imread("NoisyImage2.jpg")

    # show_medianf(image1)
    # show_meanf(image1)
    # show_gaussianf(image1)

    # show_medianf(image2)
    # show_meanf(image2)
    show_gaussianf(image2)