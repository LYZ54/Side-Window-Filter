from scipy.signal import convolve2d as conv2
import scipy.signal as signal
import numpy as np
import cv2
import math
from normal_filter import gaussian


def get_kernels(radius, type):
    r = radius
    len = 2 * r + 1
    kernel = np.zeros((len, len, 8), dtype='double')
    if type == 'box':
        k = np.ones((len, 1))
        k_L = k.copy()
        k_L[r + 1: len] = 0
        k_R = np.flipud(k_L)
        kernel[:, :, 0] = np.dot(k, k_L.T)
        kernel[:, :, 1] = np.dot(k, k_R.T)
        kernel[:, :, 2] = np.dot(k_L, k.T)
        kernel[:, :, 3] = np.dot(k_R, k.T)
        kernel[:, :, 4] = np.dot(k_L, k_L.T)
        kernel[:, :, 5] = np.dot(k_L, k_R.T)
        kernel[:, :, 6] = np.dot(k_R, k_L.T)
        kernel[:, :, 7] = np.dot(k_R, k_R.T)
    elif type == 'mean':
        k = np.ones((len, 1)) / len
        k_L = k.copy()
        k_L[r + 1: len] = 0
        k_L = k_L / np.sum(k_L)
        k_R = np.flipud(k_L)
        kernel[:, :, 0] = np.dot(k, k_L.T)
        kernel[:, :, 1] = np.dot(k, k_R.T)
        kernel[:, :, 2] = np.dot(k_L, k.T)
        kernel[:, :, 3] = np.dot(k_R, k.T)
        kernel[:, :, 4] = np.dot(k_L, k_L.T)
        kernel[:, :, 5] = np.dot(k_L, k_R.T)
        kernel[:, :, 6] = np.dot(k_R, k_L.T)
        kernel[:, :, 7] = np.dot(k_R, k_R.T)
    elif type == 'median':
        k = np.ones((len, 1))
        k_L = k.copy()
        k_L[r + 1: len] = np.nan
        k_R = np.flipud(k_L)
        kernel[:, :, 0] = np.dot(k, k_L.T)
        kernel[:, :, 1] = np.dot(k, k_R.T)
        kernel[:, :, 2] = np.dot(k_L, k.T)
        kernel[:, :, 3] = np.dot(k_R, k.T)
        kernel[:, :, 4] = np.dot(k_L, k_L.T)
        kernel[:, :, 5] = np.dot(k_L, k_R.T)
        kernel[:, :, 6] = np.dot(k_R, k_L.T)
        kernel[:, :, 7] = np.dot(k_R, k_R.T)
    elif type == 'gaussian':
        gauss_k = gaussian(1.0, r)
        # gauss_k = np.dot(gauss_k, gauss_k.T)
        k_half = gauss_k[:, 0:r + 1]
        k_L = np.hstack((k_half, np.zeros((len, r)))) / np.sum(k_half)
        k_R = np.fliplr(k_L)
        k_U = k_L.T
        k_D = np.flipud(k_U)
        k_quad = gauss_k[0:r + 1, 0:r + 1]
        k_NW = np.pad(k_quad, ((0, r), (0, r))) / np.sum(k_quad)
        k_NE = np.fliplr(k_NW)
        k_SW = np.flipud(k_NW)
        k_SE = np.flipud(k_NE)
        kernel[:, :, 0] = k_L
        kernel[:, :, 1] = k_R
        kernel[:, :, 2] = k_U
        kernel[:, :, 3] = k_D
        kernel[:, :, 4] = k_NW
        kernel[:, :, 5] = k_NE
        kernel[:, :, 6] = k_SW
        kernel[:, :, 7] = k_SE
    else:
        print('Type error!!')

    return kernel


if __name__ == '__main__':
    list = ['mean', 'median', 'box', 'gaussian']
    kernel = get_kernels(1, list[1])
    for i in range(8):
        print(kernel[:, :, i])

