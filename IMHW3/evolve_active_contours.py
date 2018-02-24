from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from math import sqrt, pi, hypot, inf
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
d = 0

def onclick(event):
    global ix, iy
    ix, iy = int(event.xdata), int(event.ydata)
    print(ix, iy)
    plt.scatter([ix], [iy])
    plt.draw()
    global coords
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

def interpolate():
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
    return hypot(pt1[0]-pt2[0], pt1[1]-pt2[1]) ** 2

def distance(pt1, pt2):
    return hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])

def e_cont(index):
    global d
    return d - distance(coords[index], coords[index-1])

def e_curv(index):
   pre = coords[index-1]
   nex = coords[index+1]
   cur = coords[index]
   ter = tuple(pre[0] + nex[0] - 2 * cur[0], pre[1] + nex[1] - 2 * cur[1])
   return norm_square(ter, (0, 0))

def get_neighbor(index, size_of_neigh):
    center = coords[index]
    ran = sqrt(size_of_neigh)
    nei = []
    for di in range(-ran, ran+1):
        for dj in range(-ran, ran+1):
            nei.append(tuple(center[0] + di, center[1] + dj))
    return nei


def grad_neigh(index, size_of_neigh, mag, mode):
    nei = get_neighbor(index, size_of_neigh)
    nei_mag = list(mag[x[0], x[1]] for x in nei)
    if (mode == 'max'):
        return max(nei_mag)
    else:
        return min(nei_mag)

def e_image(index, size_of_neigh, mag):
    max_neigh = grad_neigh(index, size_of_neigh, mag, 'max')
    min_neigh = grad_neigh(index, size_of_neigh, mag, 'min')
    if (max_neigh - min_neigh < 5):
        min_neigh = max_neigh - 5
    return (min_neigh - mag[index[0],index[1]]) / (max_neigh - min_neigh)

def average_dis():
    global coords
    global d
    length = len(coords)
    s = 0
    for index, value in enumerate(coords):
        if (index != length-1):
            next_item = coords[index + 1]
        else:
            next_item = coords[0]
        s += distance(next_item, value)
    d = s / (length - 1)

def greedy_evolve(th1, th2, th3, size_of_neigh, mag):
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
            for j in range(0, size_of_neigh):
                e_j = alpha[i] * e_cont(i) + beta[i] * e_curv(i) + gamma[i] * e_image(i, size_of_neigh, mag)
                if (e_j < e_min):
                    j_min = j
            if (j_min != 5):
                ptsmoved += 1
        for i in range(0, n):
            u_i = (coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
            u_i_1 = (coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1])
            u_i_len = hypot(u_i[0], u_i[1])
            u_i_1_len = hypot(u_i_1[0], u_i_1[1])
            curvature[i] = norm_square(tuple(t/u_i_len for t in u_i), tuple(t/u_i_1_len for t in u_i_1))
        for i in range(0, n):
            if (curvature[i] > curvature[i-1] and curvature[i] > curvature[i+1] and curvature[i] > th1 and mag[coords[i][0], coords[i][1]] > th2):
                beta[i] = 0
        if (ptsmoved < th3):
            break
    

def evolve_active_contours(img):
    image = io.imread(img)
    get_points(image)
    print('after')
    show_contours(image)
    interpolate()
    show_contours(image)

if __name__ == '__main__':
    image = io.imread('image1.jpg')
    get_points(image)
    print(coords)
    print('after')
    show_contours(image)
    interpolate()
    show_contours(image)
    print(coords)

