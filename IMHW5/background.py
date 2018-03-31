import time
from math import fabs, hypot, inf, pi, sqrt
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import ndimage, signal
from scipy.linalg import eigh as largest_eigh
from scipy.spatial import distance
from skimage import io
from skimage.morphology import square, opening, closing

class Codeword:
    def __init__(self, vec, aux):
        self.vec = vec
        self.aux = aux

def colordis(xt, vi):
    norm_1 = xt[0]**2 + xt[1]**2 + xt[2]**2
    norm_2 = vi[0]**2 + vi[1]**2 + vi[2]**2
    p2 = (xt @ vi)**2/ norm_2
    return sqrt(norm_1 - p2)

def brightness(I, aux, alpha, beta):
    Im = aux[0]
    IM = aux[1]
    Ilo = Im * alpha
    Ihi = min(beta*IM, Im / alpha)
    if (I >= Ilo and I <= Ihi):
        return True
    return False
    

def create_codebook(videopath, e1, alpha, beta):
    # read all video frames in
    frames = [join(videopath, f) for f in listdir(videopath) if isfile(join(videopath, f))]
    # sort frames
    frames.sort()
    # create codebook for the whole theme
    code_book = np.zeros_like(io.imread(frames[0]))
    updated = False
    # for every frame
    # note: index start from 0, if we use t, we need to add one to it.
    for index, fr in enumerate(frames):
        t = index + 1
        img = io.imread(fr)
        it = np.nditer(img, flags=['multi_index'])
        # for every pixel
        while not it.finished:
            i = it.multi_index[0]
            j = it.multi_index[1]
            # first frame, set every pixel's codebook to an empty list
            if (t == 0):
                code_book[i, j] = []
            # set x = (R, G, B), I = sqrt(R^2+G^2+B^2)
            xt = [it[0], it[1], it[2]]
            I = sqrt(it[0]**2 + it[1]**2 + it[2]**2)
            # iterate over the codewords at this pixel
            for ix, cw in enumerate(code_book[i, j]):
                # match
                if (colordis(xt, cw.vec) <= e1 and brightness(I, cw.aux, alpha, beta)):
                    auxm = code_book[i, j][ix].aux
                    fm = auxm[2]
                    vm = code_book[i, j][ix].vec
                    # update vector of RGB
                    code_book[i, j][ix].vec = [(fm * vm[0] + xt[0])/(fm + 1), (fm * vm[1] + xt[1])/(fm + 1), (fm * vm[2] + xt[2])/(fm + 1)]
                    # update aux (Im, IM, fm, lm, pm, qm)
                    code_book[i, j][ix].aux = [min(I, auxm[0]), max(I,auxm[1]), fm + 1, max(auxm[3], t - auxm[5]), auxm[4], t]
                    updated = True
                    break

            # if not matched
            if (updated == False):
                # create new code word
                vl = xt
                auxl = [I, I, 1, t - 1, t, t]
                code_book[i, j].append(Codeword(vl, auxl))
                
            # if the current frame is the last one, we need to wrap lambda for current pixel
            if (t == len(frames)):
                # for every codeword in this codebook
                for ix, cw in enumerate(code_book[i, j]):
                    auxi = code_book[i, j][ix].aux
                    code_book[i, j][ix].aux[3] = max(auxi[3], len(frames)-auxi[5]+auxi[4]-1)
                
            updated = False
            it.iternext()
    return code_book, len(frames)

def compress_codebook(code_book, N):
    it = np.nditer(code_book, flags=['multi_index'])
    Tm = N / 2
    # iterate over the whole codebook
    while not it.finished:
        i = it.multi_index[0]
        j = it.multi_index[1]
        new_code_words = []
        # iterate over all codeword in current pixel
        for cw in code_book[i, j]:
            if (cw.aux[3] <= Tm):
                new_code_words.append(cw)
        code_book[i, j] = new_code_words
        it.iternext()
    return code_book

def detect_foreground(image, code_book, e2, alpha, beta):
    img = io.imread(image)
    it = np.nditer(img, flags=['multi_index'])
    BGS = np.zeros_like(img)
    match = False
    # iterate all pixel
    while not it.finished:
        i = it.multi_index[0]
        j = it.multi_index[1]
        # set x = (R, G, B), I = sqrt(R^2+G^2+B^2)
        xt = [it[0], it[1], it[2]]
        I = sqrt(it[0]**2 + it[1]**2 + it[2]**2)

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
    return BGS

def morphological(BGS):
    after_mor = opening(BGS, square(3))
    after_mor = closing(BGS, square(3))
    return after_mor

if __name__ == '__main__':
    print(1)