from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from math import sqrt, pi, isnan, atan

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