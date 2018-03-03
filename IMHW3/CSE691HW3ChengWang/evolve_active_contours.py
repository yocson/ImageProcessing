from math import hypot, inf, pi, sqrt, fabs

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
from skimage import io

import gaussian as gs

coords = []
corners = []
d = 0

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
    return res


def get_strength_img(img, sigma):
    """Operate CNNNY_ENHACER algorithm

    Args:
        img: image to get edge
        sigma: std for gaussian filter

    Returns:
        mag: strength image
    """
    img_after_gau = ndimage.gaussian_filter(img.astype(float), sigma)

    img_of_grad_x, img_of_grad_y = np.gradient(img_after_gau)

    mag=np.zeros(img_after_gau.shape)
    mag = np.sqrt(np.square(img_of_grad_x) + np.square(img_of_grad_y))

    return mag

def onclick(event):
    """click event, create new points
    
    Arguments:
        event, event binded to click
    """
    global ix, iy
    ix, iy = int(event.xdata), int(event.ydata)
    # print(ix, iy)
    plt.scatter([ix], [iy], s = 2, c='r')
    plt.draw()
    global coords
    coords.append((iy, ix))

def get_points(image):
    """get initial points
    
    Arguments:
        image {[image array]} -- input image
    """
    fig = plt.figure('ACTIVE CONTOURS')
    io.imshow(image)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

def show_contours(image):
    """show image with contour points
    
    Arguments:
        image {[image array]} -- input image
    """
    global coords
    global corners
    io.imshow(image)
    for point in coords:
        plt.scatter([point[1]], [point[0]], s=2, c='r')
    for point in corners:
        plt.scatter([point[1]], [point[0]], s=2, c='w')
    plt.show() 

def interpolate():
    """interpolate points to make distance between points < 5px
    """
    coords_temp = []
    global coords
    global corners
    for index, item in enumerate(coords):
        next_item = coords[0] if (index == len(coords) - 1) else coords[index + 1]
        interval = hypot(next_item[0] - item[0], next_item[1] - item[1]) 
        coords_temp.append(item) 
        if (interval > 5):
            sub = int(interval / 5)
            disx = (next_item[0] - item[0]) / (sub + 1)
            disy = (next_item[1] - item[1]) / (sub + 1)
            for i in range(0, sub):
                ex, ey = item[0] + disx * (i + 1), item[1] + disy * (i + 1)
                coords_temp.append((int(ex), int(ey)))
    coords = coords_temp
    

def norm_square(pt1, pt2):
    return (pt1[0] - pt2[0])**2+(pt1[1] - pt2[1])**2

def distance(pt1, pt2):
    return sqrt((pt1[0] - pt2[0])**2+(pt1[1] - pt2[1])**2)


def cal_cont(index, nei):
    """calculate the continuity term
    
    Arguments:
        index {int} -- current index of points
        nei {list of tuple} -- neighbour points
    
    Returns:
        [list] -- normalized continuity term list of one neighourhood
    """
    global d
    global coords
    prev_item = coords[index-1] if index != 0 else coords[-1]
    nei_cont = list(fabs((d - distance(x, prev_item))) for x in nei)
    return nei_cont


def cal_curv(index, nei):
    """calculate the curvature term
    
    Arguments:
        index {int} -- index of current point
        nei {list of tuple} -- [description]
    
    Returns:
        list -- normalized curvature term list of one neighbourhood
    """
    global coords
    n = len(coords)
    nei_curv = []
    pre = coords[index-1] if (index != 0) else coords[-1]
    nex = coords[index+1] if (index != n-1) else coords[0]
    for cur in nei:
        x = pre[0] + nex[0] - 2 * cur[0]
        y = pre[1] + nex[1] - 2 * cur[1]
        ter = x**2 + y**2
        nei_curv.append(ter)
    
    return nei_curv

def get_neighbor(index, size_of_neigh):
    """get neighbour points pos
    
    Arguments:
        index {int} -- index of point in coords
        size_of_neigh {int} -- size of neighborhood, eg, 3*3=9
    
    Returns:
        [list of tuple] -- all neighbour points with point position
    """
    global coords
    center = coords[index]
    rang = sqrt(size_of_neigh)
    nei = []
    delta = int(rang/2)
    for di in range(-delta, delta + 1):
        for dj in range(-delta, delta + 1):
            nei.append(tuple((center[0] + di, center[1] + dj)))
    return nei


def cal_image(mag, nei):
    """calculate image edge attraction term
    
    Arguments:
        mag {array} -- gradient magnitude
        nei {list} -- neighbour points

    Returns:
        edge attraction term list of one neighborhood
    """
    nei_mag = list(mag[x[0], x[1]] for x in nei)
    max_neigh = max(nei_mag)
    min_neigh = min(nei_mag)
    if (max_neigh - min_neigh < 5):
        min_neigh = max_neigh - 5
    return list((min_neigh - mag[pos[0],pos[1]]) / (max_neigh - min_neigh) for pos in nei)

def average_dis():
    """average distance between points
    """
    global coords
    global d
    n = len(coords)
    s = 0
    for index, value in enumerate(coords):
        if (index != (n - 1)):
            next_item = coords[index + 1]
        else:
            next_item = coords[0]
        s += sqrt((next_item[0] - value[0])**2 + (next_item[1] - value[1])**2)
    d = s / n

def cal_curvature(i):
    global coords

    prev_item = coords[i-1] if (i != 0) else coords[-1]
    cur_item = coords[i]
    next_item = coords[i+1] if (i != len(coords)-1) else coords[0]

    u_1   = tuple((cur_item[0] - prev_item[0], cur_item[1] - prev_item[1]))
    u_2   = tuple((next_item[0] - cur_item[0], next_item[1] - cur_item[1]))
    len_1 = distance(u_1, (0, 0))
    len_2 = distance(u_2, (0, 0))

    one = tuple(t/len_1 if (len_1 != 0) else 0 for t in u_1)
    two = tuple(t/len_2 if (len_2 != 0) else 0 for t in u_2)

    return norm_square(one, two)
    

def greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, alphap, betap, gammap):
    global coords
    global corners
    global d
    n = len(coords)
    alpha = [alphap] * n
    beta = [betap] * n
    gamma = [gammap] * n
    curvature = [0] * n
    count = 0
    print(n)
    while True:
        ptsmoved = 0
        for i in range(0, n):
            average_dis()
            e_min = inf
            nei = get_neighbor(i, size_of_neigh)
            nei_cont = cal_cont(i, nei)
            nei_curv = cal_curv(i, nei)
            nei_imag = cal_image(mag, nei)


            for j in range(0, size_of_neigh):
                e_j = alpha[i] * nei_cont[j] / max(nei_cont) + beta[i] * nei_curv[j] / max(nei_curv)+ gamma[i] * nei_imag[j]
                if (j == int(size_of_neigh / 2)):
                    e_cur = e_j
                if (e_j < e_min):
                    e_min = e_j
                    j_min = j
            if (j_min != int(size_of_neigh / 2) and e_cur != e_min):
                coords[i] = nei[j_min]
                ptsmoved += 1
        count += 1
        for i in range(0, n):
            curvature[i] = cal_curvature(i)

        for i in range(0, n):
            prev_item = curvature[i-1] if (i != 0) else curvature[-1]
            cur_item = curvature[i]
            next_item = curvature[i+1] if (i != n-1) else curvature[0]
            if (cur_item > prev_item and cur_item > next_item and cur_item > th1 and mag[coords[i][0], coords[i][1]] > th2):
                beta[i] = 0
                corners.append(coords[i])
        if (count == 10 or count == 30):
            show_contours(image)
        if (ptsmoved < th3 * n):
            break
    print("finish")
    show_contours(image)


def evolve_active_contours(th1, th2, th3, size_of_neigh, img, sigma, alpha, beta, gamma):
    image = plt.imread(img)
    mag = get_strength_img(image, sigma)
    # io.imshow(mag, cmap='gray')
    # plt.show()
    get_points(image)
    interpolate()
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, alpha, beta, gamma)

def evolve_active_contours_sigma(th1, th2, th3, size_of_neigh, img, sigma1, sigma2, sigma3):
    image = plt.imread(img)
    mag1 = get_strength_img(image, sigma1)
    mag2 = get_strength_img(image, sigma2)
    mag3 = get_strength_img(image, sigma3)
    get_points(image)
    interpolate()
    show_contours(image)
    global coords
    global corners

    temp = np.zeros_like(coords)
    temp = np.copy(coords)
    print("first")
    greedy_evolve(th1, th2, th3, size_of_neigh, mag1, image, 1, 1, 1)

    corners = []
    coords = np.copy(temp)
    print("second")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh, mag2, image, 1, 1, 1)
    
    corners = []
    coords = np.copy(temp)
    print("third")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh, mag3, image, 1, 1, 1)


def evolve_active_contours_nei(th1, th2, th3, size_of_neigh1, size_of_neigh2, size_of_neigh3, img, sigma):
    image = plt.imread(img)
    mag = get_strength_img(image, sigma)
    get_points(image)
    interpolate()
    show_contours(image)
    global coords
    global corners

    temp = np.zeros_like(coords)
    temp = np.copy(coords)
    print("first")
    greedy_evolve(th1, th2, th3, size_of_neigh1, mag, image, 1, 1, 1)

    corners = []
    coords = np.copy(temp)
    print("second")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh2, mag, image, 1, 1, 1)
    
    corners = []
    coords = np.copy(temp)
    print("third")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh3, mag, image, 1, 1, 1)

def evolve_active_contours_alpha(th1, th2, th3, size_of_neigh, img, sigma, alpha1, alpha2, alpha3):
    image = plt.imread(img)
    mag = get_strength_img(image, sigma)
    get_points(image)
    interpolate()
    show_contours(image)
    global coords
    global corners

    temp = np.zeros_like(coords)
    temp = np.copy(coords)
    print("first")
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, alpha1, 1, 1)

    corners = []
    coords = np.copy(temp)
    print("second")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, alpha2, 1, 1)
    
    corners = []
    coords = np.copy(temp)
    print("third")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, alpha3, 1, 1)

def evolve_active_contours_beta(th1, th2, th3, size_of_neigh, img, sigma, beta1, beta2, beta3):
    image = plt.imread(img)
    mag = get_strength_img(image, sigma)
    get_points(image)
    interpolate()
    show_contours(image)
    global coords
    global corners

    temp = np.zeros_like(coords)
    temp = np.copy(coords)
    print("first")
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, 1, beta1, 1)

    corners = []
    coords = np.copy(temp)
    print("second")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, 1, beta2, 1)
    
    corners = []
    coords = np.copy(temp)
    print("third")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, 1, beta3, 1)

def evolve_active_contours_gamma(th1, th2, th3, size_of_neigh, img, sigma, gamma1, gamma2, gamma3):
    image = plt.imread(img)
    mag = get_strength_img(image, sigma)
    get_points(image)
    interpolate()
    show_contours(image)
    global coords
    global corners

    temp = np.zeros_like(coords)
    temp = np.copy(coords)
    print("first")
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, 1, 1, gamma1)

    corners = []
    coords = np.copy(temp)
    print("second")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, 1, 1, gamma2)
    
    corners = []
    coords = np.copy(temp)
    print("third")
    show_contours(image)
    greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, 1, 1, gamma3)

def evolve_active_contours_th3(th1, th2, th31, th32, th33, size_of_neigh, img, sigma):
    image = plt.imread(img)
    mag = get_strength_img(image, sigma)
    get_points(image)
    interpolate()
    show_contours(image)
    global coords
    global corners

    temp = np.zeros_like(coords)
    temp = np.copy(coords)
    print("first")
    greedy_evolve(th1, th2, th31, size_of_neigh, mag, image, 1, 1, 1)

    corners = []
    coords = np.copy(temp)
    print("second")
    show_contours(image)
    greedy_evolve(th1, th2, th32, size_of_neigh, mag, image, 1, 1, 1)
    
    corners = []
    coords = np.copy(temp)
    print("third")
    show_contours(image)
    greedy_evolve(th1, th2, th33, size_of_neigh, mag, image, 1, 1, 1)

def evolve_active_contours_sequence(th1, th2, th3, size_of_neigh, img, sigma):
    for i in range(0, 180, 15):
        image = plt.imread(img + str(i).zfill(3) + ".jpg")
        print(img + str(i).zfill(3) + ".jpg")
        if (i == 0):
            get_points(image)
            interpolate()
            show_contours(image)
        global corners
        corners = []
        mag = get_strength_img(image, sigma)
        greedy_evolve(th1, th2, th3, size_of_neigh, mag, image, 1, 1, 1)
    

if __name__ == '__main__':
    # part a
    # evolve_active_contours(0.5, 50, 0.1, 9, 'Images1through8/image1.jpg', 3.0, 1, 1, 1)
    # evolve_active_contours(1, 10, 0.1, 9, 'Images1through8/image2.jpg', 3.0, 1, 1, 1)
    evolve_active_contours(0.3, 10, 0.1, 9, 'Images1through8/image3.jpg', 3.0, 1, 1, 1)
    # evolve_active_contours(0.5, 50, 0.1, 9, 'Images1through8/image4.jpg', 3.0, 1, 1, 1)
    # evolve_active_contours(0.5, 50, 0.1, 9, 'Images1through8/image5.jpg', 3.0, 1, 1, 1)
    # evolve_active_contours(0.3, 10, 0.1, 9, 'Images1through8/image6.jpg', 3.0, 1, 1, 1)
    # evolve_active_contours(0.2, 10, 0.1, 9, 'Images1through8/image7.jpg', 3.0, 1, 1, 1)
    # evolve_active_contours(0.1, 10, 0.1, 9, 'Images1through8/image8.jpg', 3.0, 1, 1, 1)

    # evolve_active_contours_sigma(0.3, 30, 0.1, 9, 'Images1through8/image1.jpg', 1.0, 2.0, 3.0)
    # evolve_active_contours_sigma(0.3, 30, 0.1, 9, 'Images1through8/image3.jpg', 1.0, 2.0, 3.0)
    # evolve_active_contours_sigma(0.5, 50, 0.1, 9, 'Images1through8/image6.jpg', 1.0, 2.0, 3.0)

    # evolve_active_contours_nei(0.5, 50, 0.1, 9, 25, 49, 'Images1through8/image1.jpg', 1.0)
    # evolve_active_contours_nei(0.5, 50, 0.1, 9, 25, 49, 'Images1through8/image3.jpg', 1.0)
    # evolve_active_contours_nei(0.5, 50, 0.1, 9, 25, 49, 'Images1through8/image6.jpg', 1.0)

    # evolve_active_contours_alpha(0.5, 50, 0.1, 25, 'Images1through8/image1.jpg', 1.0, 0.5, 1, 3)
    # evolve_active_contours_alpha(0.5, 50, 0.1, 25, 'Images1through8/image3.jpg', 1.0, 0.5, 1, 3)
    # evolve_active_contours_alpha(0.5, 50, 0.1, 25, 'Images1through8/image6.jpg', 1.0, 0.5, 1, 3)

    # evolve_active_contours_beta(0.5, 50, 0.1, 25, 'Images1through8/image1.jpg', 1.0, 0.5, 1, 3)
    # evolve_active_contours_beta(0.5, 50, 0.1, 25, 'Images1through8/image3.jpg', 1.0, 0.5, 1, 3)
    # evolve_active_contours_beta(0.5, 50, 0.1, 25, 'Images1through8/image6.jpg', 1.0, 0.5, 1, 3)

    # evolve_active_contours_gamma(0.5, 50, 0.1, 9, 'Images1through8/image1.jpg', 1.0, 0.5, 1, 3)
    # evolve_active_contours_gamma(0.5, 50, 0.1, 25, 'Images1through8/image3.jpg', 1.0, 0.5, 1, 3)
    # evolve_active_contours_gamma(0.5, 50, 0.1, 9, 'Images1through8/image6.jpg', 1.0, 0.5, 1, 3)

    # evolve_active_contours_th3(0.5, 50, 0.1, 0.01, 0.3, 9, 'Images1through8/image1.jpg', 1.0)
    # evolve_active_contours_th3(0.5, 50, 0.1, 0.01, 0.3, 9, 'Images1through8/image3.jpg', 1.0)
    # evolve_active_contours_th3(0.5, 50, 0.1, 0.01, 0.3, 9, 'Images1through8/image6.jpg', 1.0)

    # evolve_active_contours_sequence(0.5, 50, 0.1, 25, "Sequence2/deg", 1.0)

