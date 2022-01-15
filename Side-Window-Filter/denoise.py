from side_window_filter import SWF
from normal_filter import normal_filter
import cv2
from show import compare_show, show
import time
from skimage import util
import numpy as np

if __name__ == '__main__':
    list = ['mean', 'median', 'box', 'gaussian']
    in_name = 'image/img2.jpeg'
    img_in = cv2.imread(in_name)
    img_in_noise = util.random_noise(img_in, mode='s&p')
    img_in_noise = img_in_noise*255
    img_in_noise = img_in_noise.astype(np.uint8)
    # compare_show(img_in, img_in_noise)
    radius = 1
    iteration = 1
    i = 1
    start_time = time.time()
    result1 = normal_filter(img_in_noise, list[i], radius, iteration)
    end_time = time.time()
    print(end_time - start_time)
    # show(result1)
    start_time = time.time()
    result2 = SWF(img_in_noise, list[i], radius, iteration)
    end_time = time.time()
    print(end_time - start_time)
    compare_show(result1, result2)
