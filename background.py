import time
from math import fabs, hypot, inf, pi, sqrt
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import ndimage, signal, misc
from scipy.linalg import eigh as largest_eigh
from scipy.spatial import distance
from skimage import io
from skimage.morphology import square

class Codeword:
    def __init__(self, vec, aux):
        self.vec = vec
        self.aux = aux


def colordis(xt, vi):
    # norm_1 = xt[0]**2 + xt[1]**2 + xt[2]**2
    # norm_2 = vi[0]**2 + vi[1]**2 + vi[2]**2
    # norm_1 = np.linalg.norm(xt)**2
    # norm_2 = np.linalg.norm(vi)**2
    norm_1 = np.sum(np.array(xt)**2)
    norm_2 = np.sum(np.array(vi)**2)
    p2 = np.dot(xt, vi)**2 
    p2 = p2 / norm_2
    if (norm_1 - p2 < 0):
        return 0
    return np.sqrt(norm_1 - p2)

def brightness(I, aux, alpha, beta):
    Im = aux[0]
    IM = aux[1]
    Ilo = IM * alpha
    Ihi = min(beta*IM, Im / alpha)
    if (I >= Ilo and I <= Ihi):
        return True
    return False
    
def create_codebook(videopath, e1, alpha, beta):
    print('Start creating codebook...')
    # read all video frames in
    frames = [join(videopath, f) for f in listdir(videopath) if isfile(join(videopath, f))]
    # sort frames
    frames.sort()
    # create codebook for the whole theme
    code_book = np.zeros_like(misc.imread(frames[0], mode='L'), dtype=list)
    match = False
    # for every frame
    # note: index start from 0, if we use t, we need to add one to it.
    for index, fr in enumerate(frames):
        t = index + 1
        print(t)
        img = misc.imread(fr, mode='RGB')
        assert len(img.shape) == 3 and img.shape[2] == 3
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (t == 1):
                    code_book[i, j] = []
                # set x = (R, G, B), I = sqrt(R^2+G^2+B^2)
                R = img[i, j, 0]
                G = img[i, j, 1]
                B = img[i, j, 2]
                # print((R,G,B))
                xt = [R, G, B]
                I = sqrt(R**2 + G**2 + B**2)
                # iterate over the codewords at this pixel
                for cw in code_book[i, j]:
                    # if (len(code_book[i, j]) == 0):
                    #     break
                    # match
                    if (colordis(xt, cw.vec) <= e1 and brightness(I, cw.aux, alpha, beta)):
                        auxm = cw.aux
                        fm = auxm[2]
                        vm = cw.vec
                        # update vector of RGB
                        cw.vec = [(fm * vm[0] + xt[0])/(fm + 1), (fm * vm[1] + xt[1])/(fm + 1), (fm * vm[2] + xt[2])/(fm + 1)]
                        # update aux (Im, IM, fm, lm, pm, qm)
                        cw.aux = [min(I, auxm[0]), max(I,auxm[1]), fm + 1, max(auxm[3], t - auxm[5]), auxm[4], t]
                        match = True
                        break

                # if not matched
                if (match == False):
                    # create new code word
                    vl = xt
                    auxl = [I, I, 1, t - 1, t, t]
                    code_book[i, j].append(Codeword(vl, auxl))
                    
                # if the current frame is the last one, we need to wrap lambda for current pixel
                if (t == len(frames)):
                    # for every codeword in this codebook
                    for cw in code_book[i, j]:
                        auxi = cw.aux
                        cw.aux[3] = max(auxi[3], len(frames)-auxi[5]+auxi[4]-1)
                    
                match = False
        
        # # for every pixel
        # while not it.finished:
        #     i = it.multi_index[0]
        #     j = it.multi_index[1]
        #     # first frame, set every pixel's codebook to an empty list
        #     if (t == 1):
        #         code_book[i, j] = []
        #     # set x = (R, G, B), I = sqrt(R^2+G^2+B^2)
        #     R = img[i, j, 0]
        #     G = img[i, j, 1]
        #     B = img[i, j, 2]
        #     # print((R,G,B))
        #     xt = [R, G, B]
        #     I = sqrt(R**2 + G**2 + B**2)
        #     # iterate over the codewords at this pixel
        #     for cw in code_book[i, j]:
        #         # if (len(code_book[i, j]) == 0):
        #         #     break
        #         # match
        #         if (colordis(xt, cw.vec) <= e1 and brightness(I, cw.aux, alpha, beta)):
        #             auxm = cw.aux
        #             fm = auxm[2]
        #             vm = cw.vec
        #             # update vector of RGB
        #             cw.vec = [(fm * vm[0] + xt[0])/(fm + 1), (fm * vm[1] + xt[1])/(fm + 1), (fm * vm[2] + xt[2])/(fm + 1)]
        #             # update aux (Im, IM, fm, lm, pm, qm)
        #             cw.aux = [min(I, auxm[0]), max(I,auxm[1]), fm + 1, max(auxm[3], t - auxm[5]), auxm[4], t]
        #             match = True
        #             break

        #     # if not matched
        #     if (match == False):
        #         # create new code word
        #         vl = xt
        #         auxl = [I, I, 1, t - 1, t, t]
        #         code_book[i, j].append(Codeword(vl, auxl))
                
        #     # if the current frame is the last one, we need to wrap lambda for current pixel
        #     if (t == len(frames)):
        #         # for every codeword in this codebook
        #         for cw in code_book[i, j]:
        #             auxi = cw.aux
        #             cw.aux[3] = max(auxi[3], len(frames)-auxi[5]+auxi[4]-1)
                
        #     match = False
        #     it.iternext()
    return code_book, len(frames)

def compress_codebook(code_book, N):
    print('Start compressing codebook...')
    Tm = N / 2
    for i in range(code_book.shape[0]):
        for j in range(code_book.shape[1]):
            new_code_words = []
            # iterate over all codeword in current pixel
            for cw in code_book[i, j]:
                if (cw.aux[3] <= Tm):
                    new_code_words.append(cw)
            code_book[i, j] = new_code_words

    # it = np.nditer(code_book, flags=['multi_index'])
    # # iterate over the whole codebook
    # while not it.finished:
    #     i = it.multi_index[0]
    #     j = it.multi_index[1]
    #     new_code_words = []
    #     # iterate over all codeword in current pixel
    #     for cw in code_book[i, j]:
    #         if (cw.aux[3] <= Tm):
    #             new_code_words.append(cw)
    #     code_book[i, j] = new_code_words
    #     it.iternext()

    return code_book

def detect_foreground(image, code_book, e2, alpha, beta):
    print('Start detecting backgroud...')
    img = misc.imread(image, mode='RGB')
    it = np.nditer(img, flags=['multi_index'])
    BGS = np.zeros_like(misc.imread(image, mode='L'))
    match = False

    for i in range(code_book.shape[0]):
        for j in range(code_book.shape[1]):
            R = img[i, j, 0]
            G = img[i, j, 1]
            B = img[i, j, 2]
            xt = [R, G, B]
            I = sqrt(R**2 + G**2 + B**2)

            for cw in code_book[i, j]:
                if (colordis(xt, cw.vec) < e2 and brightness(I, cw.aux, alpha, beta)):
                    match = True
                    break
            # if is background, set to black, else white
            if (match):
                BGS[i, j] = 0
            else:
                BGS[i, j] = 255
            match = False

    # # iterate all pixel
    # while not it.finished:
    #     i = it.multi_index[0]
    #     j = it.multi_index[1]
    #     # set x = (R, G, B), I = sqrt(R^2+G^2+B^2)
    #     R = img[i, j, 0]
    #     G = img[i, j, 1]
    #     B = img[i, j, 2]
    #     xt = [R, G, B]
    #     I = sqrt(R**2 + G**2 + B**2)

    #     for cw in code_book[i, j]:
    #         if (colordis(xt, cw.vec) < e2 and brightness(I, cw.aux, alpha, beta)):
    #             match = True
    #             break
    #     # if is background, set to black, else white
    #     if (match):
    #         BGS[i, j] = 0
    #     else:
    #         BGS[i, j] = 255
    #     match = False
    #     it.iternext()
    return BGS

def morphological(BGS):
    after_mor = ndimage.grey_opening(BGS, size=(3,3))
    after_mor = ndimage.grey_closing(after_mor, size=(2,2))
    return after_mor

def show_img(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show(block=False)

if __name__ == '__main__':
    alpha = 0.7
    beta = 1.2
    eh1 = 300
    eh2 = 300
    fat_codebook, N = create_codebook('V1-1', eh1, alpha, beta)
    code_book = compress_codebook(fat_codebook, N)
    background1 = detect_foreground('Video1_1/PetsD2TeC1_00700.jpg', code_book, eh2, alpha, beta)
    background2 = detect_foreground('Video1_2/PetsD2TeC1_00939.jpg', code_book, eh2, alpha, beta)
    background3 = detect_foreground('Video1_3/PetsD2TeC1_01522.jpg', code_book, eh2, alpha, beta)
    background4 = detect_foreground('Video1_3/PetsD2TeC1_01675.jpg', code_book, eh2, alpha, beta)
    background5 = detect_foreground('Video1_3/PetsD2TeC1_01815.jpg', code_book, eh2, alpha, beta)
    show_img(background1)
    show_img(background2)
    show_img(background3)
    show_img(background4)
    show_img(background5)
    background1 = morphological(background1)
    background2 = morphological(background2)
    background3 = morphological(background3)
    background4 = morphological(background4)
    background5 = morphological(background5)
    show_img(background1)
    show_img(background2)
    show_img(background3)
    show_img(background4)
    show_img(background5)
    plt.show()
