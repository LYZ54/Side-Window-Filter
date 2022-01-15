import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def juanji(kernel, img, i, j):
    res = img[i - 1][j - 1] * kernel[0][0] + img[i - 1][j] * kernel[0][1] + img[i - 1][j + 1] * kernel[0][2] + \
          img[i][j - 1] * kernel[1][0] + img[i][j] * kernel[1][1] + img[i][j + 1] * kernel[1][2] + \
          img[i + 1][j - 1] * kernel[2][0] + img[i + 1][j] * kernel[2][1] + img[i + 1][j + 1] * kernel[2][2]
    return res


# (3,3), sigma = 1.3, 整数形式
def swf_gauss(img):
    kernel_l = np.array([[1, 2, 0], [2, 4, 0], [1, 2, 0]])
    kernel_r = np.array([[0, 2, 1], [0, 4, 2], [0, 2, 1]])
    kernel_u = np.array([[1, 2, 1], [2, 4, 2], [0, 0, 0]])
    kernel_d = np.array([[0, 0, 0], [2, 4, 2], [1, 2, 1]])
    kernel_nw = np.array([[1, 2, 0], [2, 4, 0], [0, 0, 0]])
    kernel_ne = np.array([[0, 2, 1], [0, 4, 2], [0, 0, 0]])
    kernel_sw = np.array([[0, 0, 0], [2, 4, 0], [1, 2, 0]])
    kernel_se = np.array([[0, 0, 0], [0, 4, 2], [0, 2, 1]])

    h = img.shape[0]  # 行
    w = img.shape[1]  # 列
    res = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if (i == 0 or j == 0 or i == h - 1 or j == w - 1):
                res[i][j] = img[i][j]
                continue

            x = np.zeros((8, 1))
            x[0] = juanji(kernel_l, img, i, j) / 16
            x[1] = juanji(kernel_r, img, i, j) / 16
            x[2] = juanji(kernel_u, img, i, j) / 16
            x[3] = juanji(kernel_d, img, i, j) / 16
            x[4] = juanji(kernel_nw, img, i, j) / 9
            x[5] = juanji(kernel_ne, img, i, j) / 9
            x[6] = juanji(kernel_sw, img, i, j) / 9
            x[7] = juanji(kernel_se, img, i, j) / 9

            min_xi = 0
            min_xerror = abs(x[0] - img[i][j])
            for l in range(1, 8):
                now = abs(x[l] - img[i][j])
                if (now < min_xerror):
                    min_xerror = now
                    min_xi = l

            res[i][j] = x[min_xi]
    return res


def Canny(img):
    # 去掉高斯滤波的canny，高斯滤波提到外面去做

    h = img.shape[0]
    w = img.shape[1]
    edge = np.zeros((h, w))

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)

    GX = np.zeros((h, w))
    GY = np.zeros((h, w))
    # 卷积
    for i in range(h):
        for j in range(w):
            if (i == 0 or j == 0 or i == h - 1 or j == w - 1):
                GX[i][j] = img[i][j]
                GY[i][j] = img[i][j]
                continue

            GX[i][j] = img[i - 1][j - 1] * kernel_x[0][0] + img[i - 1][j] * kernel_x[0][1] + img[i - 1][j + 1] * \
                       kernel_x[0][2] + \
                       img[i][j - 1] * kernel_x[1][0] + img[i][j] * kernel_x[1][1] + img[i][j + 1] * kernel_x[1][2] + \
                       img[i + 1][j - 1] * kernel_x[2][0] + img[i + 1][j] * kernel_x[2][1] + img[i + 1][j + 1] * \
                       kernel_x[2][2]

            GY[i][j] = img[i - 1][j - 1] * kernel_y[0][0] + img[i - 1][j] * kernel_y[0][1] + img[i - 1][j + 1] * \
                       kernel_y[0][2] + \
                       img[i][j - 1] * kernel_y[1][0] + img[i][j] * kernel_y[1][1] + img[i][j + 1] * kernel_y[1][2] + \
                       img[i + 1][j - 1] * kernel_y[2][0] + img[i + 1][j] * kernel_y[2][1] + img[i + 1][j + 1] * \
                       kernel_y[2][2]

    absx = cv2.convertScaleAbs(GX)
    absy = cv2.convertScaleAbs(GY)

    tan = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            edge[i][j] = math.sqrt(absx[i][j] ** 2 + absy[i][j] ** 2)
            tan[i][j] = math.atan(GY[i][j] / GX[i][j])
    edge = cv2.convertScaleAbs(edge)

    # 非极大值抑制
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            t = tan[i][j]
            if (t > (-0.4142) and t <= 0.4142):  # 0
                if (max(edge[i][j - 1], edge[i][j + 1]) > edge[i][j]): edge[i][j] = 0
            elif (t > 0.4142 and t < 2.4142):  # 45
                if (max(edge[i - 1][j + 1], edge[i + 1][j - 1]) > edge[i][j]): edge[i][j] = 0
            elif (t > (-2.4142) and t < (-0.4142)):  # 135
                if (max(edge[i - 1][j - 1], edge[i + 1][j + 1]) > edge[i][j]): edge[i][j] = 0
            else:  # 90
                if (max(edge[i - 1][j], edge[i + 1][j]) > edge[i][j]): edge[i][j] = 0

    th1 = 48
    th2 = 100
    res = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            t = tan[i][j]
            if (edge[i][j] >= th2):
                res[i][j] = 255
            elif (edge[i][j] < th1):
                res[i][j] = 0
            else:
                if (i == 0 or i == h - 1 or j == 0 or j == w - 1):
                    res[i][j] = 0
                else:
                    if (edge[i - 1][j - 1] >= th2 or edge[i - 1][j] >= th2 or edge[i - 1][j + 1] >= th2 or \
                            edge[i][j - 1] >= th2 or edge[i][j + 1] >= th2 or \
                            edge[i + 1][j - 1] >= th2 or edge[i + 1][j] >= th2 or edge[i + 1][j + 1] >= th2):
                        res[i][j] = 255
                    else:
                        res[i][j] = 0

    return res


if __name__ == '__main__':
    img = cv2.imread('images/img_4.jpeg', flags=cv2.IMREAD_GRAYSCALE)
    # res = Canny(img)
    img1 = cv2.GaussianBlur(img, (3, 3), 1.3)
    img2 = swf_gauss(img)

    res1 = Canny(img1)
    res2 = Canny(img2)
    # '''
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Origin')
    plt.subplot(1, 3, 2)
    plt.imshow(res1, cmap='gray')
    plt.title('Gauss')
    plt.subplot(1, 3, 3)
    plt.imshow(res2, cmap='gray')
    plt.title('swf_Gauss')
    plt.show()
    # '''
