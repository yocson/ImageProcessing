from math import hypot, inf, pi, sqrt, fabs

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
from skimage import io

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
    return res

def get_strength_img(img, sigma):
    """Operate CNNNY_ENHACER algorithm

    Args:
        img: image to get edge
        sigma: std for gaussian filter

    Returns:
        mag: strength image
    """
    img_after_gau = gs.gaussian_filter(img, sigma)
    img_of_grad_x = gradient_image(img_after_gau)
    img_of_grad_y = gradient_image(img_after_gau, 'y')
    mag = np.sqrt(np.square(img_of_grad_x) + np.square(img_of_grad_y))
    return mag

coords = []
d = 0

def onclick(event):
    """click event, create new points
    
    Arguments:
        event, event binded to click
    """
    global ix, iy
    ix, iy = int(event.xdata), int(event.ydata)
    print(ix, iy)
    plt.scatter([ix], [iy])
    plt.draw()
    global coords
    coords.append((iy, ix))

def get_points(image):
    """get initial points
    
    Arguments:
        image {[image array]} -- input image
    """
    fig = plt.figure('test')
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
    io.imshow(image)
    for t in coords:
        plt.scatter([t[1]], [t[0]])
    plt.show() 

def interpolate():
    """interpolate points to make distance between points < 5px
    """
    coords_temp = []
    global coords
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
    return hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]) ** 2

def distance(pt1, pt2):
    return hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

# def e_cont(index, pos, nei):
#     """calculate the continuity term
    
#     Arguments:
#         index {int} -- current index of points
#         pos {tuple} -- point position in image
#         nei {list of tuple} -- neighbour points
    
#     Returns:
#         [float] -- normalized continuity term
#     """
#     global d
#     global coords
#     prev_item = coords[index-1] if index != 0 else coords[-1]
#     max_cont = max(list(fabs(d - distance(x, prev_item)) for x in nei))
#     return fabs((d - distance(pos, prev_item))) / max_cont

def cal_cont(index, nei):
    global d
    global coords
    prev_item = coords[index-1] if index != 0 else coords[-1]
    nei_cont = list(fabs((d - distance(x, prev_item))) for x in nei)
    return nei_cont

# def e_curv(index, pos, nei):
#     """calculate the curvature term
    
#     Arguments:
#         index {int} -- index of current point
#         pos {tuple} -- neighbor point position in image
#         nei {list of tuple} -- [description]
    
#     Returns:
#         [float] -- normalized curvature term
#     """
#     global coords
#     n = len(coords)
#     pre = coords[index-1] if (index != 0) else coords[-1]
#     nex = coords[index+1] if (index != n-1) else coords[0]
#     cur = pos
#     max_curv = - inf
#     for item in nei:
#         ter = tuple((pre[0] + nex[0] - 2 * item[0], pre[1] + nex[1] - 2 * item[1]))
#         res = norm_square(ter, (0, 0))
#         if (res > max_curv):
#             max_curv = res
#     ter = tuple((pre[0] + nex[0] - 2 * cur[0], pre[1] + nex[1] - 2 * cur[1]))
#     return norm_square(ter, (0, 0)) / max_curv

def cal_curv(index, nei):
    global coords
    n = len(coords)
    pre = coords[index-1] if (index != 0) else coords[-1]
    nex = coords[index+1] if (index != n-1) else coords[0]
    ter_list = list(tuple((pre[0] + nex[0] - 2 * cur[0], pre[1] + nex[1] - 2 * cur[1])) for cur in nei)
    nei_curv = list(norm_square(ter, (0, 0)) for ter in ter_list)
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

# def grad_neigh(index, size_of_neigh, mag, mode):
#     global coords
#     nei = get_neighbor(index, size_of_neigh)
#     nei_mag = list(mag[coords[x][0], coords[x][1]] for x in nei)
#     if (mode == 'max'):
#         return max(nei_mag)
#     else:
#         return min(nei_mag)

# def e_image(pos, mag, nei):
#     """calculate image edge attraction term
    
#     Arguments:
#         pos {tuple} -- current postion in image
#         mag {array} -- gradient magnitude
#         nei {list} -- neighbour points

#     Returns:
#         edge attraction term
#     """

#     nei_mag = list(mag[x[0], x[1]] for x in nei)
#     max_neigh = max(nei_mag)
#     min_neigh = min(nei_mag)
#     if (max_neigh - min_neigh < 5):
#         min_neigh = max_neigh - 5
#     return (min_neigh - mag[pos[0],pos[1]]) / (max_neigh - min_neigh)

def cal_image(mag, nei):
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
        s += distance(next_item, value)
    d = s / n

def cal_curvature(i):
    global coords

    prev_item = coords[i-1] if (i != 0) else coords[-1]
    cur_item = coords[i]
    next_item = coords[i+1] if (i != len(coords)-1) else coords[0]

    u_1   = (cur_item[0] - prev_item[0], cur_item[1] - prev_item[1])
    u_2   = (next_item[0] - cur_item[0], next_item[1] - cur_item[1])
    len_1 = distance(u_1, (0, 0))
    len_2 = distance(u_2, (0, 0))

    one = tuple(t/len_1 if (len_1 != 0) else 0 for t in u_1)
    two = tuple(t/len_2 if (len_2 != 0) else 0 for t in u_2)

    return norm_square(one, two)
    

def greedy_evolve(th1, th2, th3, size_of_neigh, mag, image):
    global coords
    global d
    n = len(coords)
    alpha = beta = gamma =[1] * n
    curvature = [0] * n
    while True:
        average_dis()
        ptsmoved = 0
        for i in range(0, n):
            e_min = inf
            nei = get_neighbor(i, size_of_neigh)
            nei_cont = cal_cont(i, nei)
            nei_curv = cal_curv(i, nei)
            nei_imag = cal_image(mag, nei)
            for j in range(0, size_of_neigh):
                # e_j = alpha[i] * e_cont(i, pos, nei) + beta[i] * e_curv(i, pos, nei) + gamma[i] * e_image(pos, mag, nei)
                e_j = alpha[i] * nei_cont[j] / max(nei_cont) + beta[i] * nei_curv[j] / max(nei_curv) + gamma[i] * nei_imag[j]
                if (e_j < e_min):
                    e_min = e_j
                    j_min = j
            if (j_min != int(size_of_neigh / 2)):
                coords[i] = nei[j_min]
                ptsmoved += 1

        for i in range(0, n):
            curvature[i] = cal_curvature(i)

        for i in range(0, n):
            prev_item = curvature[i-1] if (i != 0) else curvature[-1]
            cur_item = curvature[i]
            next_item = curvature[i+1] if (i != n-1) else curvature[0]
            if (cur_item > prev_item and cur_item > next_item and cur_item > th1 and mag[coords[i][0], coords[i][1]] > th2):
                beta[i] = 0

        # show_contours(image)
        if (ptsmoved < th3 * n):
            break
    show_contours(image)

    

def evolve_active_contours(img):
    image = io.imread(img)
    mag = get_strength_img(image, 1)
    get_points(image)
    print(coords)
    print('after')
    show_contours(image)
    interpolate()
    show_contours(image)
    greedy_evolve(10, 10, 0.2, 49, mag, image)


if __name__ == '__main__':
    # image = io.imread('image1.jpg')
    # get_points(image)
    # print(coords)
    # print('after')
    # show_contours(image)
    # interpolate()
    # show_contours(image)
    # print(coords)
    evolve_active_contours('image1.jpg')