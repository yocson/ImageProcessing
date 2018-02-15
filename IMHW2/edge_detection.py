from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

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
    ker = np.array([-1, 0, 1])
    if (direction == 'y'):
        ker = np.transpose(ker)

    res = ndimage.convolve(img, ker)

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
    Es = np.sqrt(np.square(img_of_grad_x) + np.square(img_of_grad_y), dtype=int)
    Eo = np.arctan(img_of_grad_x / img_of_grad_y)
    return Es, Eo

def get_detector(Eo):
    """from orientation image to 0, 45, 90, 135

    Args:
        Eo: orientation image

    Returns:
        direction image
    """
    dir_img = np.array([])
    for x in np.nditer(Eo):
        if ((x >= 0 and x < 22.5) or (x > 157.5 and x <= 180)): 
            direction = 0
        elif (x >= 22.5 and x < 67.5):
            direction = 45
        elif (x >= 67.6 and x < 112.5):
            direction = 90
        else:
            direction = 135
        dir_img = np.append(dir_img, dir)
    return dir_img.astype(int).reshape(Eo.shape[0], Eo.shape[1])

def nonmax_suppression(Es, Eo):
    """Operate algorithm of nonmax_suppression

    To produce 1-pixel wide edge

    Args:
        Es: strength iamge
        Eo: orientation image

    Returns:
        suppressed image
    """
    dir_img = get_detector(Eo)

    suppressed_img = np.zeros_like(Es)
    it = np.nditer(Es, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index[0]
        j = it.multi_index[1]
        im, jm, ip, jp = get_neighbor_of_index(i, j, dir_img)
        if (it[0] >= Es[im, jm] and it[0] >= Es[ip, jp]):
            suppressed_img[i, j] = it[0]
        it.iternext()
    return suppressed_img
        
def get_neighbor_of_index(i, j, dir_img):
    """get neighbors' indices

    Find neighbors of i, j along the edge direction

    Args:
        i, j: current index
        dir_img: with direction of degree 0, 45, 90, 135

    Returns:
        Two neighbors's indices, [im, jm] and [jp, jp]
    """
    direction = dir_img[i, j]
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
        ip, jp = i+1, i+1
    return im, jm, ip, jp

def track_edge(i, j, In, dir_img, visited_img, ti):
    """chain all edges

    From point i, j, check whether it is greater than ti,
    if so, track its neighbors along the edge direction.

    Args:
        i, j: current index
        In: suppressed image
        dir_img: with direction of degree 0, 45, 90, 135
        ti: low thresholds
    """
    im, jm, ip, jp = get_neighbor_of_index(i, j, dir_img)
    if (In[im, jm] > ti):
        visited_img[im, jm] = 1
        track_edge(im, jm, In, dir_img, visited_img, ti)
    if (In[ip, jp] > ti):
        visited_img[ip, jp] = 1
        track_edge(ip, jp, In, dir_img, visited_img, ti)
    return
    
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

    it = np.nditer(In, flags=['multi_index'])
    while not it.finished:
        if (it[0] < th): 
            continue
        else:
            i = it.multi_index[0]
            j = it.multi_index[1]
            if (visited_img[i, j] == 1):
                continue
            visited_img[i, j] == 1
            track_edge(i, j, dir_img, visited_img, ti)
        it.iternext()
    return visited_img

if __name__ == '__main__':
    image = io.imread('Syracuse_01.jpg')
    img_after_gau = gaussian_filter(image, 0.9)
    io.imshow(img_after_gau)
    plt.show()
    

    


    
