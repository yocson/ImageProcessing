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

coords = []

def onclick(event):
    global ix, iy
    ix, iy = int(event.xdata), int(event.ydata)
    print(ix, iy)
    plt.scatter([ix], [iy])
    plt.draw()
    # global coords
    coords.append((ix, iy))

def get_points(image):
    fig = plt.figure('test')
    io.imshow(image)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

def show_contours(image):
    io.imshow(image)
    for t in coords:
        plt.scatter([t[0]], [t[1]])
    plt.show() 

def evolve_active_contours(img):
    image = io.imread(img)
    get_points(image)
    print('after')
    show_contours(image)

if __name__ == '__main__':
    image = io.imread('image1.jpg')
    get_points(image)
    print('after')
    show_contours(image)
