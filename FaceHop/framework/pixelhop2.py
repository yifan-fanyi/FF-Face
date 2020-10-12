# 2020.06.04
import numpy as np
import cv2
import os
from skimage.util import view_as_windows

from framework.saab import Saab

def Shrink(X, win, stride):
    X = view_as_windows(X, (1,win,win,1), (1,stride,stride,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def PixelHop_Unit(X, num_kernels, saab=None, window=5, stride=1, train=True):
    X = Shrink(X, window, stride)
    S = list(X.shape)
    X = X.reshape(-1, S[-1])
    if(train==True):
        saab = Saab(num_kernels=num_kernels, useDC=True, needBias=True)
    saab.fit(X)
    transformed = saab.transform(X).reshape(S[0],S[1],S[2],-1)
    return saab,transformed

def faltten(listoflists, kernel_filter):
    flattened = []
    for i in range(len(listoflists)):
        for j in range(len(listoflists[i][kernel_filter[i]])):
            flattened.append(listoflists[i][j])
    return flattened

def PixelHopPP_Unit(X, num_kernels, saab=None, window=5, stride=1, train=True, energy_th=0, ch_decoupling=True, ch_energies=None, kernel_filter=[]):
    #for taining specify X, num_kernels, window, stride, train, energy_th, ch_decoupling, ch_energies(only if it is not the first layer)
    #for testing specift X, num_kernels, saab, window, stride, train, ch_decoupling, kernel_filter
    N, L, W, D = X.shape
    if ch_energies == None:
        ch_energies = np.ones((D)).tolist()
    out_ch_energies = []
    output = None
    if ch_decoupling == True:
        for i in range(D):
            saab, transformed = (PixelHop_Unit(X[:,:,:,i].reshape(N,L,W,1), num_kernels=num_kernels, saab=saab, window=window, stride=stride, train=train))
            if train==True:
                out_ch_energies.append(ch_energies[i] * saab.Energy)
                kernel_filter.append(out_ch_energies[i] > energy_th)
            transformed = transformed[:, :, :, kernel_filter[i]]
            if i == 0:
                output = transformed
            else:
                output = np.concatenate((output,transformed),axis=3)
    else:
        saab, transformed = (PixelHop_Unit(X, num_kernels=num_kernels, saab=saab ,window=window, stride=stride, train=train))
        if train == True:
            out_ch_energies.append(saab.Energy)
            kernel_filter.append(out_ch_energies[0] > energy_th)
        transformed = transformed[:, :, :, kernel_filter[0]]
        output = transformed
    return saab, output, kernel_filter, faltten(out_ch_energies, kernel_filter)