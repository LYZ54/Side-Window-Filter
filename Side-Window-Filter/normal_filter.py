from scipy.signal import convolve2d as conv2
import scipy.signal as signal
import numpy as np
import cv2
import math
from show import show, compare_show


def gaussian(sigma, radius):
    len = 2 * radius + 1
    kernel = np.zeros((len, len))
    for i in range(-radius, -radius + len):
        for j in range(-radius, -radius + len):
            kernel[i + radius][j + radius] = np.exp(-(i * i + j * j) / (2 * (sigma ** 2)))
    kernel /= (sigma * np.sqrt(2 * np.pi))
    kernel /= kernel.sum()
    return kernel


def normal_filter(img, type, radius, iteration):
    img = img.astype(np.float32)
    h, w, chs = img.shape
    result = img.copy()
    r = radius  # 滤波器半径
    len = 2 * r + 1  # 滤波器长度
    m = h + 2 * r  # 边缘填充后的行数
    n = w + 2 * r  # 边缘填充后的列数
    for ch in range(chs):
        U = np.pad(img[:, :, ch], ((r, r), (r, r)), mode='edge')  # 用边缘填充
        U_out = U.copy()
        for it in range(iteration):
            if type == 'median':
                result = cv2.medianBlur(img, len)
                result = result.astype(np.uint8)
                return result
                '''
                for i in range(h):
                    for j in range(w):
                        roi = U[i:i + len, j:j + len]
                        med = np.median(roi)
                        U_out[i+r, j+r] = med
                '''
            elif type == 'box':
                kernel = np.ones((len, len))
                U_out = conv2(U, kernel, 'same')
            elif type == 'mean':
                kernel = np.ones((len, len)) / (len * len)
                U_out = conv2(U, kernel, 'same')
            elif type == 'gaussian':
                kernel = gaussian(1.0, r)
                U_out = conv2(U, kernel, 'same')
            else:
                print("Type error!!")
        result[:, :, ch] = U_out[r:m-r, r:n-r]
    result = result.astype(np.uint8)
    return result


if __name__ == '__main__':
    type_list = ['mean', 'median', 'box', 'gaussian']
    img_list = ['image/img1.jpeg', 'image/img2.jpeg', 'image/img3.jpeg', 'image/img4.jpeg']

    in_name = img_list[3]
    type = type_list[0]
    radius = 1
    iteration = 2

    img_in = cv2.imread(in_name)
    result = normal_filter(img_in, type, radius, iteration)
    compare_show(img_in, result)
