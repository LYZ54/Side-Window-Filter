from scipy.signal import convolve2d as conv2
import scipy.signal as signal
import numpy as np
import cv2
import math
from show import show, compare_show
from get_filter import get_kernels
import time
import math


def conv(array1, array2):
    h1, w1 = array1.shape
    h2, w2 = array2.shape
    if h1 != h2 or w1 != w2:
        print('Error!!')
    else:
        result = 0
        for i in range(h1):
            for j in range(w1):
                result += array1[i, j] * array2[i, j]
    return result


# def index(x, y):
#     if x == 0:
#         if y >= 0:
#             return 3
#         else:
#             return 2
#     else:
#         tang = y / x
#         if (x >= 0 and y >= 0 and tang <= math.tan(math.pi / 8)) or (
#                 x >= 0 and y <= 0 and tang >= math.tan(-math.pi / 8)):
#             return 1
#         elif (x <= 0 and y <= 0 and tang <= math.tan(math.pi / 8)) or (
#                 x <= 0 and y >= 0 and tang >= math.tan(-math.pi / 8)):
#             return 0
#         elif (x >= 0 and y <= 0 and tang <= math.tan(-math.pi * 3 / 8)) or (
#                 x <= 0 and y <= 0 and tang >= math.tan(math.pi * 3 / 8)):
#             return 2
#         elif (x >= 0 and y >= 0 and tang >= math.tan(math.pi * 3 / 8)) or (
#                 x <= 0 and y >= 0 and tang <= math.tan(-math.pi * 3 / 8)):
#             return 3
#         elif x < 0 and y < 0 and math.tan(math.pi / 8) <= tang <= math.tan(math.pi * 3 / 8):
#             return 4
#         elif x > 0 and y < 0 and math.tan(-math.pi * 3 / 8) <= tang <= math.tan(-math.pi / 8):
#             return 5
#         elif x < 0 and y > 0 and math.tan(-math.pi * 3 / 8) <= tang <= math.tan(-math.pi / 8):
#             return 6
#         elif x > 0 and y > 0 and math.tan(math.pi / 8) <= tang <= math.tan(math.pi * 3 / 8):
#             return 7

def choose_best(list, a):
    diff = 10000
    index = 0
    for i in range(len(list)):
        if abs(list[i] - a) < diff:
            diff = abs(list[i] - a)
            index = i
    return list[index]


def SWF_normal(img, type, radius, iteration):
    r = radius
    len = 2 * r + 1
    h, w, chs = img.shape
    img = img.astype(np.float32)
    result = img.copy()
    kernels = get_kernels(r, type).astype(np.float32)
    m = h + 2 * r
    n = w + 2 * r

    for ch in range(chs):
        U = np.pad(img[:, :, ch], ((r, r), (r, r)), mode='edge')
        for it in range(iteration):
            if type == 'median':
                for i in range(m - len + 1):
                    for j in range(n - len + 1):
                        list = []
                        roi = U[i:i + len, j:j + len]
                        for k_i in range(8):
                            tmp = roi * kernels[:, :, k_i]
                            med = np.nanmedian(tmp)
                            list.append(med)
                        result[i, j, ch] = choose_best(list, img[i, j, ch])
            else:
                for i in range(m - len + 1):
                    for j in range(n - len + 1):
                        roi = U[i:i + len, j:j + len]
                        list = []
                        for k_i in range(8):
                            tmp = conv(roi, kernels[:, :, k_i])
                            list.append(tmp)
                        result[i, j, ch] = choose_best(list, img[i, j, ch])
    return result / 255


if __name__ == '__main__':
    start_time = time.time()
    type_list = ['mean', 'median', 'box', 'gaussian']
    img_list = ['image/img1.jpeg', 'image/img2.jpeg', 'image/img3.jpeg', 'image/img4.jpeg']

    in_name = img_list[3]
    type = type_list[0]
    radius = 1
    iteration = 2

    img_in = cv2.imread(in_name)
    result = SWF_normal(img_in, type, radius, iteration)
    end_time = time.time()
    print('side_window_filter_normal run time:', end_time - start_time)
    compare_show(img_in, result)
