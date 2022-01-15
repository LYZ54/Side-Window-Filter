from scipy.signal import convolve2d as conv2
import scipy.signal as signal
import numpy as np
import cv2
import math
from normal_filter import gaussian

if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    b = np.array([[2, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(a*b)
