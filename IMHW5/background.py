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
from sklearn import preprocessing

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
    frames = [join(videopath, f) for f in listdir(videopath) if isfile(join(videopath, f))]
    code_book = np.zeros_like(io.imread(frames[0]))
    updated = False
    for t, fr in enumerate(frames):
        img = io.imread(fr)
        it = np.nditer(img, flags=['multi_index'])
        while not it.finished:
            i = it.multi_index[0]
            j = it.multi_index[1]
            if (t == 0):
                code_book[i, j] = []

            xt = [it[0], it[1], it[2]]
            I = sqrt(it[0]**2 + it[1]**2 + it[2]**2)

            for ix, cw in enumerate(code_book[i, j]):
                if (colordis(xt, cw.vec) < e1 and brightness(I, cw.aux, alpha, beta)):
                    auxm = code_book[i, j][ix].aux
                    fm = auxm[2]
                    vm = code_book[i, j][ix].vec
                    code_book[i, j][ix].vec = [(fm*vm[0] + xt[0])/(fm + 1), (fm*vm[1] + xt[1])/(fm + 1), (fm*vm[2] + xt[2])/(fm + 1)]
                    code_book[i, j][ix].aux = [min(I, auxm[0]), max(I,auxm[1]), fm+1, max(auxm[3], t-auxm[5]), auxm[4], t + 1]
                    updated = True
                    break

            if (updated == False):
                # create new code word
                vl = xt
                auxl = [I, I, 1, t, t + 1, t + 1]
                code_book[i, j].append(Codeword(vl, auxl))
                
            if (t == len(frames) - 1):
                for ix, cw in enumerate(code_book[i, j]):
                    auxi = code_book[i, j][ix].aux
                    code_book[i, j][ix].aux[3] = max(auxi[3], len(frames)-auxi[5]+auxi[4]-1)
                
            updated = False
            it.iternext()
    return code_book, len(frames)

def compress_codebook(code_book, N):
    it = np.nditer(code_book, flags=['multi_index'])
    Tm = N / 2
    while not it.finished:
        i = it.multi_index[0]
        j = it.multi_index[1]
        new_code_words = []
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
    while not it.finished:
        i = it.multi_index[0]
        j = it.multi_index[1]
    
        xt = [it[0], it[1], it[2]]
        I = sqrt(it[0]**2 + it[1]**2 + it[2]**2)

        for cw in code_book[i, j]:
            if (colordis(xt, cw.vec) < e2 and brightness(I, cw.aux, alpha, beta)):
                match = True
                break

        if (match):
            BGS[i, j] = 0
        else:
            BGS[i, j] = 255
        match = False
    return BGS

if __name__ == '__main__':
    print(1)