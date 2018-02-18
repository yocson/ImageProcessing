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

def canny_enhancer(img, sigma):
    """Operate CNNNY_ENHACER algorithm

    Args:
        img: image to get edge
        sigma: std for gaussian filter

    Returns:
        Es: strength image
        Eo: orientation image
    """
    img_after_gau = gaussian_filter(img, sigma)
    img_of_grad_x = gradient_image(img_after_gau)
    img_of_grad_y = gradient_image(img_after_gau, 'y')
    Es = np.sqrt(np.square(img_of_grad_x) + np.square(img_of_grad_y))
    Eo = np.arctan(img_of_grad_y / img_of_grad_x)
    return Es, Eo

def get_direction(Eo):
    """from orientation image to 0, 45, 90, 135

    Args:
        Eo: orientation image

    Returns:
        direction image
    """
    dir_img = np.zeros_like(Eo)
    it = np.nditer(Eo, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index[0]
        j = it.multi_index[1]
        if (isnan(it[0])):
            direction = 90
            dir_img[i, j] = direction
            it.iternext()
            continue
        value = it[0] / pi * 180
        if (value < 0):
            value = value + 180
        if ((value >= 0 and value < 22.5) or (value > 157.5 and value <= 180)): 
            direction = 0
        elif (value >= 22.5 and value < 67.5):
            direction = 45
        elif (value >= 67.6 and value < 112.5):
            direction = 90
        else:
            direction = 135
        dir_img[i, j] = direction
        it.iternext()

    return dir_img.astype(int).reshape(Eo.shape[0], Eo.shape[1])

def checkBoundary(im, jm, ip, jp, Es):
    if (im < 0 or ip >= Es.shape[0] or jm < 0 or jm >= Es.shape[1] or jp < 0 or jp >= Es.shape[1]):
        return False
    else:
        return True

def nonmax_suppression(Es, Eo, dir_img):
    """Operate algorithm of nonmax_suppression

    To produce 1-pixel wide edge

    Args:
        Es: strength iamge
        Eo: orientation image
        dir_img: direction with 0 45 90 135

    Returns:
        suppressed image
    """

    suppressed_img = np.zeros_like(Es)
    it = np.nditer(Es, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index[0]
        j = it.multi_index[1]
        im, jm, ip, jp = get_neighbor_of_index(i, j, dir_img)
        if (checkBoundary(im, jm, ip, jp, Es)):
            if (it[0] >= Es[im, jm] and it[0] >= Es[ip, jp]):
                suppressed_img[i, j] = it[0]
        it.iternext()

    return suppressed_img
        
def get_neighbor_of_index(i, j, dir_img, direct = 0):
    """get neighbors' indices

    Find neighbors of i, j along the edge direction

    Args:
        i, j: current index
        dir_img: with direction of degree 0, 45, 90, 135

    Returns:
        Two neighbors's indices, [im, jm] and [jp, jp]
    """
    direction = dir_img[i, j]
    if (direct == 0):
        if (direction == 0):
            im, jm = i, j-1
            ip, jp = i, j+1
        elif (direction == 45):
            im, jm = i-1, j-1
            ip, jp = i+1, j+1
        elif (direction == 90):
            im, jm = i-1, j
            ip, jp = i+1, j
        else:
            im, jm = i-1, j+1
            ip, jp = i+1, j-1
    else:
        if (direction == 0):
            im, jm = i-1, j
            ip, jp = i+1, j
        elif (direction == 45):
            im, jm = i-1, j+1
            ip, jp = i+1, j-1
        elif (direction == 90):
            im, jm = i, j-1
            ip, jp = i, j+1
        else:
            im, jm = i-1, j-1
            ip, jp = i+1, j+1

    return im, jm, ip, jp

def track_edge(i, j, In, dir_img, visited_img, edge_img, ti):
    """chain all edges

    From point i, j, check whether it is greater than ti,
    if so, track its neighbors along the edge direction.

    Args:
        i, j: current index
        In: suppressed image
        dir_img: with direction of degree 0, 45, 90, 135
        ti: low thresholds
    """
    im, jm, ip, jp = get_neighbor_of_index(i, j, dir_img, 1)
    visited_img[i, j] = 1
    if (im >= 0 and jm < In.shape[1] and jm >= 0 and In[im, jm] > ti and not visited_img[im, jm]):
        edge_img[im, jm] = In[im, jm]
        visited_img, edge_img = track_edge(im, jm, In, dir_img, visited_img, edge_img, ti)
    if (ip < In.shape[0] and jp >= 0 and jp < In.shape[1] and In[ip, jp] > ti and not visited_img[ip, jp]):  
        edge_img[ip, jp] = In[ip, jp]
        visited_img, edge_img = track_edge(ip, jp, In, dir_img, visited_img, edge_img, ti)
    return visited_img, edge_img
    
def hystersis_threshold(In, dir_img, ti, th):
    """Operate HYSTERIS_THRESHOLD

    Find the index where value is greater than th, than
    track edge from this point

    Args:
        In: suppressed image
        dir_img: with direction of degree 0, 45, 90, 135
        ti, th: two thresholds

    Returns:
        images with edge chains
    """
    visited_img = np.zeros_like(In)
    edge_img = np.zeros_like(In)

    it = np.nditer(In, flags=['multi_index'])
    while not it.finished:
        if (it[0] > th): 
            i = it.multi_index[0]
            j = it.multi_index[1]
            if (visited_img[i, j] == 0): 
                edge_img[i, j] = it[0]
                visited_img, edge_img = track_edge(i, j, In, dir_img, visited_img, edge_img, ti)
        it.iternext()
    return edge_img

def canny_edge_detection(img, sigma, thresL, thresH):
    """Operate canny edge detection

    1) CANNY_ENHANCER
    2) NONMAX_SUPPRESSION
    3) HYSTERESIS_THRESH

    Args:
        img: input image
        sigma: standard deviation for gaussian filter
        thresL, thresH: two thresholds

    Returns:
        images with edge chains
    """
    image = io.imread(img)
    Es, Eo = canny_enhancer(image, sigma)
    dir_img = get_direction(Eo)
    In = nonmax_suppression(Es, Eo, dir_img)
    edge_img = hystersis_threshold(In, dir_img, thresL ,thresH)

    # plt.figure('CANNY EDGE DETECTION')
    # plt.subplot(1,3,1)
    # plt.title('CANNY_ENHANCER')
    # io.imshow(Es, cmap='gray')
    # io.imsave('CANNY_ENHANCER.jpg', Es.astype(np.uint8))
    # plt.axis('off')
    # plt.subplot(1,3,2)
    # plt.title('NONMAX_SUPPRESSION')
    # io.imshow(In, cmap='gray')
    # io.imsave('NONMAX_SUPPRESSION.jpg', In.astype(np.uint8))
    # plt.axis('off')
    # plt.subplot(1,3,3)
    # plt.title('HYSTERESIS_THRESH')
    # io.imshow(edge_img, cmap='gray')
    # io.imsave('HYSTERESIS_THRESH.jpg', edge_img.astype(np.uint8))
    # plt.axis('off')
    # plt.show()

    return Es, In, edge_img

if __name__ == '__main__':
    Es1, In1, edge_img1 = canny_edge_detection('Syracuse_01.jpg', 1, 10, 60)
    io.imsave('CANNY_ENHANCER_6.jpg', Es1.astype(np.uint8))
    io.imsave('NONMAX_SUPPRESSION_6.jpg', In1.astype(np.uint8))
    io.imsave('HYSTERESIS_THRESH_6.jpg', edge_img1.astype(np.uint8))
    # Es2, In2, edge_img2 = canny_edge_detection('Syracuse_01.jpg', 1, 10, 80)
    # io.imsave('CANNY_ENHANCER_2.jpg', Es2.astype(np.uint8))
    # io.imsave('NONMAX_SUPPRESSION_2.jpg', In2.astype(np.uint8))
    # io.imsave('HYSTERESIS_THRESH_7.jpg', edge_img2.astype(np.uint8))
    # Es3, In3, edge_img3 = canny_edge_detection('Syracuse_01.jpg', 1, 30, 60)
    # io.imsave('HYSTERESIS_THRESH_8.jpg', edge_img3.astype(np.uint8))
    
