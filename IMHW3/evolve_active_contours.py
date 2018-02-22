from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from math import sqrt, pi, isnan, atan
import gaussian as gs

def gradient_image(img, direction='x'):
    """Calculate iamge gradient

    Args:
        img: image to get edge
        direction: along which direction

    Returns:
        image with gradient value along direction
    """
    ker = np.array([-1, 0, 1]).reshape(1, 3)
    if (direction == 'y'):
        ker = np.transpose(ker)

    res = signal.convolve2d(img, ker, boundary='fill', mode='same')
    np.set_printoptions(threshold=np.nan)
    return res

def get_strength_img(img, sigma):
    """Operate CNNNY_ENHACER algorithm

    Args:
        img: image to get edge
        sigma: std for gaussian filter

    Returns:
        Es: strength image
    """
    img_after_gau = gs.gaussian_filter(img, sigma)
    img_of_grad_x = gradient_image(img_after_gau)
    img_of_grad_y = gradient_image(img_after_gau, 'y')
    Es = np.sqrt(np.square(img_of_grad_x) + np.square(img_of_grad_y))
    return Es

def evolve_active_contours(img):
    image = io.imread(img)
    


if __name__ == '__main__':
    
