from scipy.signal import convolve2d as conv2
import scipy.signal as signal
import numpy as np
import cv2
import math
from show import show, compare_show
from get_filter import get_kernels
import time


def SWF(img, type, radius, iteration):
    r = radius
    len = 2 * r + 1
    h, w, chs = img.shape
    img = img.astype(np.float32)
    result = img.copy()
    kernels = get_kernels(r, type).astype(np.float32)
    m = h + 2 * r
    n = w + 2 * r
    diff = np.zeros((m, n, 8), 'float')
    dm = np.zeros((m, n), 'float')

    for ch in range(chs):
        U = np.pad(img[:, :, ch], ((r, r), (r, r)), mode='edge')
        for it in range(iteration):
            if type == 'median':
                for k_i in range(8):
                    for i in range(m - len + 1):
                        for j in range(n - len + 1):
                            roi = U[i:i + len, j:j + len]
                            roi = roi * kernels[:, :, k_i]
                            med = np.nanmedian(roi)
                            diff[i + r, j + r] = med
                    diff[:, :, k_i] = diff[:, :, k_i] - U
            else:
                for k_i in range(8):
                    diff[:, :, k_i] = conv2(U, kernels[:, :, k_i], 'same') - U
            tmp = abs(diff)
            inds = np.argmin(tmp, 2)
            for i in range(m):
                for j in range(n):
                    ind = inds[i, j]
                    dm[i, j] = diff[i, j, ind]
            U = dm + U
        result[:, :, ch] = U[r:m - r, r:n - r]
    result = result.astype(np.uint8)
    return result


if __name__ == '__main__':
    start_time = time.time()
    type_list = ['mean', 'median', 'box', 'gaussian']
    img_list = ['image/img1.jpeg', 'image/img2.jpeg', 'image/img3.jpeg', 'image/img4.JPG']

    in_name = img_list[1]
    type = type_list[0]
    radius = 1
    iteration = 2

    img_in = cv2.imread(in_name)
    result = SWF(img_in, type, radius, iteration)
    end_time = time.time()
    print('side_window_filter run time:', end_time - start_time)
    compare_show(img_in, result)
