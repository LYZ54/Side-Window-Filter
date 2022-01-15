import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

src_gray = cv2.imread('imges/img1_gray.bmp')
local_img = cv2.imread('imges/img1_local.bmp')  # 局部着色图
h = src_gray.shape[0]
w = src_gray.shape[1]
img = cv2.cvtColor(src_gray, cv2.COLOR_BGR2YUV)
local = cv2.cvtColor(local_img, cv2.COLOR_BGR2YUV)
# b,g,r = cv2.split(src)
# src_img_toshow = cv2.merge([r,g,b])
# img_toshow = cv2.cvtColor(img,cv2.COLOR_YUV2RGB)
src_y, src_u, src_v = cv2.split(img)


def choose_best(row, col, SW_weight):
    Min = 10000
    Min_flag = 0
    for i in range(0, 8):
        dis_Y = 0.0
        for j in range(0, 8):
            if j < 4:
                tmprow = int(j / 3) - 1 + row
                tmpcol = (j % 3) - 1 + col
            else:
                tmprow = int((j + 1) / 3) - 1 + row
                tmpcol = ((j + 1) % 3) - 1 + col

            dis_Y += SW_weight[i, j] * src_y[tmprow, tmpcol]

        dis_Y = (dis_Y - src_y[row, col]) ** 2

        if Min > dis_Y:
            Min = dis_Y
            Min_flag = i

    return Min_flag


def get_w(row, col):
    weight = np.empty(8, dtype=np.float64)
    SW_sum = np.empty(8, dtype=np.float64)
    dis = np.empty(8, dtype=np.float64)
    for j in range(0, 8):
        weight[j] = 0.0
        SW_sum[j] = 0.0
        dis[j] = 0.0

    dert_sum = 0.0
    if (row - 1 >= 0 and col - 1 >= 0):
        dis[0] = (int(src_y[row, col]) - int(local[row - 1, col - 1, 0])) ** 2
    if (row - 1 >= 0):
        dis[1] = (int(src_y[row, col]) - int(local[row - 1, col, 0])) ** 2
    if (row - 1 >= 0 and col + 1 <= w - 1):
        dis[2] = (int(src_y[row, col]) - int(local[row - 1, col + 1, 0])) ** 2
    if (col - 1 >= 0):
        dis[3] = (int(src_y[row, col]) - int(local[row, col - 1, 0])) ** 2
    if (col + 1 <= w - 1):
        dis[4] = (int(src_y[row, col]) - int(local[row, col + 1, 0])) ** 2
    if (row + 1 <= h - 1 and col - 1 >= 0):
        dis[5] = (int(src_y[row, col]) - int(local[row + 1, col - 1, 0])) ** 2
    if (row + 1 <= h - 1):
        dis[6] = (int(src_y[row, col]) - int(local[row + 1, col, 0])) ** 2
    if (row + 1 <= h - 1 and col + 1 <= w - 1):
        dis[7] = (int(src_y[row, col]) - int(local[row + 1, col + 1, 0])) ** 2

    flag = 0

    if row == 0 and col == 0:  # 左上角像素邻域
        dert_sum = dis[4] + dis[6] + dis[7]

    elif row == 0 and col == w - 1:  # 右上角像素邻域
        dert_sum = dis[3] + dis[5] + dis[6]

    elif row == h - 1 and col == 0:  # 左下角像素邻域
        dert_sum = dis[1] + dis[2] + dis[4]

    elif row == h - 1 and col == w - 1:  # 右下角像素邻域
        dert_sum = dis[0] + dis[1] + dis[3]

    elif row == 0:  # 上边界像素邻域
        dert_sum = dis[3] + dis[4] + dis[5] + dis[6] + dis[7]

    elif row == h - 1:  # 下边界像素邻域
        dert_sum = dis[0] + dis[1] + dis[2] + dis[3] + dis[4]

    elif col == 0:  # 左边界像素邻域
        dert_sum = dis[1] + dis[2] + dis[4] + dis[6] + dis[7]

    elif col == w - 1:  # 右边界像素邻域
        dert_sum = dis[0] + dis[1] + dis[3] + dis[5] + dis[6]

    else:
        flag = 1

        SW_sum[0] = dis[0] + dis[1] + dis[3] + dis[5] + dis[6]

        SW_sum[1] = dis[1] + dis[2] + dis[4] + dis[6] + dis[7]

        SW_sum[2] = dis[0] + dis[1] + dis[2] + dis[3] + dis[4]

        SW_sum[3] = dis[3] + dis[4] + dis[5] + dis[6] + dis[7]

        SW_sum[4] = dis[0] + dis[1] + dis[3]

        SW_sum[5] = dis[1] + dis[2] + dis[4]

        SW_sum[6] = dis[3] + dis[5] + dis[6]

        SW_sum[7] = dis[4] + dis[6] + dis[7]

        for j in range(0, 8):
            if SW_sum[j] == 0:
                SW_sum[j] = 0.00000002
            else:
                SW_sum[j] = 1 * SW_sum[j]

    if flag == 0:
        if dert_sum < 0.0001:
            sigma = 0.0001
        else:
            sigma = 0.8 * dert_sum
    else:

        SW_mask = np.zeros((3, 3, 8), dtype=int)
        SW_mask[0, 0, 0] = SW_mask[0, 1, 0] = SW_mask[1, 0, 0] = SW_mask[2, 0, 0] = SW_mask[2, 1, 0] = 1
        SW_mask[0, 1, 1] = SW_mask[0, 2, 1] = SW_mask[1, 2, 1] = SW_mask[2, 1, 1] = SW_mask[2, 2, 1] = 1
        SW_mask[0, 0, 2] = SW_mask[0, 1, 2] = SW_mask[0, 2, 2] = SW_mask[1, 0, 2] = SW_mask[1, 2, 2] = 1
        SW_mask[2, 0, 3] = SW_mask[2, 1, 3] = SW_mask[2, 2, 3] = SW_mask[1, 0, 3] = SW_mask[1, 2, 3] = 1
        SW_mask[0, 0, 4] = SW_mask[0, 1, 4] = SW_mask[1, 0, 4] = 1
        SW_mask[0, 1, 5] = SW_mask[0, 2, 5] = SW_mask[1, 2, 5] = 1
        SW_mask[1, 0, 6] = SW_mask[2, 0, 6] = SW_mask[2, 1, 6] = 1
        SW_mask[1, 2, 7] = SW_mask[2, 1, 7] = SW_mask[2, 2, 7] = 1

    if row == 0 and col == 0:  # 左上角像素邻域
        weight[4] = math.exp((-(dis[4])) / (2 * (sigma ** 2)))
        weight[6] = math.exp((-(dis[6])) / (2 * (sigma ** 2)))
        weight[7] = math.exp((-(dis[7])) / (2 * (sigma ** 2)))
    elif row == 0 and col == w - 1:  # 右上角像素邻域
        weight[3] = math.exp((-(dis[3])) / (2 * (sigma ** 2)))
        weight[5] = math.exp((-(dis[5])) / (2 * (sigma ** 2)))
        weight[6] = math.exp((-(dis[6])) / (2 * (sigma ** 2)))
    elif row == h - 1 and col == 0:  # 左下角像素邻域
        weight[1] = math.exp((-(dis[1])) / (2 * (sigma ** 2)))
        weight[2] = math.exp((-(dis[2])) / (2 * (sigma ** 2)))
        weight[4] = math.exp((-(dis[4])) / (2 * (sigma ** 2)))
    elif row == h - 1 and col == w - 1:  # 右下角像素邻域
        weight[0] = math.exp((-(dis[0])) / (2 * (sigma ** 2)))
        weight[1] = math.exp((-(dis[1])) / (2 * (sigma ** 2)))
        weight[3] = math.exp((-(dis[3])) / (2 * (sigma ** 2)))
    elif row == 0:  # 上边界像素邻域
        weight[3] = math.exp((-(dis[3])) / (2 * (sigma ** 2)))
        weight[4] = math.exp((-(dis[4])) / (2 * (sigma ** 2)))
        weight[5] = math.exp((-(dis[5])) / (2 * (sigma ** 2)))
        weight[6] = math.exp((-(dis[6])) / (2 * (sigma ** 2)))
        weight[7] = math.exp((-(dis[7])) / (2 * (sigma ** 2)))
    elif row == h - 1:  # 下边界像素邻域
        weight[0] = math.exp((-(dis[0])) / (2 * (sigma ** 2)))
        weight[1] = math.exp((-(dis[1])) / (2 * (sigma ** 2)))
        weight[2] = math.exp((-(dis[2])) / (2 * (sigma ** 2)))
        weight[3] = math.exp((-(dis[3])) / (2 * (sigma ** 2)))
        weight[4] = math.exp((-(dis[4])) / (2 * (sigma ** 2)))
    elif col == 0:  # 左边界像素邻域
        weight[1] = math.exp((-(dis[1])) / (2 * (sigma ** 2)))
        weight[2] = math.exp((-(dis[2])) / (2 * (sigma ** 2)))
        weight[4] = math.exp((-(dis[4])) / (2 * (sigma ** 2)))
        weight[6] = math.exp((-(dis[6])) / (2 * (sigma ** 2)))
        weight[7] = math.exp((-(dis[7])) / (2 * (sigma ** 2)))
    elif col == w - 1:  # 右边界像素邻域
        weight[0] = math.exp((-(dis[0])) / (2 * (sigma ** 2)))
        weight[1] = math.exp((-(dis[1])) / (2 * (sigma ** 2)))
        weight[3] = math.exp((-(dis[3])) / (2 * (sigma ** 2)))
        weight[5] = math.exp((-(dis[5])) / (2 * (sigma ** 2)))
        weight[6] = math.exp((-(dis[6])) / (2 * (sigma ** 2)))
    else:
        SW_weight = np.zeros((8, 8), dtype=np.float64)
        for i in range(0, 8):
            SW_weight_sum = 0.0
            for j in range(0, 8):

                if j < 4:
                    tmprow = int(j / 3)
                    tmpcol = j % 3
                else:
                    tmprow = int((j + 1) / 3)
                    tmpcol = (j + 1) % 3

                if SW_mask[tmprow, tmpcol, i] == 1:
                    SW_weight[i, j] = math.exp((-(dis[j])) / (2 * (SW_sum[i] ** 2)))
                    SW_weight_sum += SW_weight[i, j]
                # else:
                # SW_weight[i,j] = 0.00000000001

                # SW_weight_sum += SW_weight[i,j]

            for j in range(0, 8):
                if SW_weight[i, j] != 0:
                    SW_weight[i, j] = float(SW_weight[i, j] / SW_weight_sum)  # 权值归一化

        # 选最佳侧窗
        SW_I = choose_best(row, col, SW_weight)
        wsum = 0.0
        for j in range(0, 8):
            if SW_weight[SW_I, j] == 0:
                # SW_weight[SW_I,j] = 0.0000001
                SW_weight[SW_I, j] = 0.0000001
            wsum += SW_weight[SW_I, j]

        for j in range(0, 8):
            SW_weight[SW_I, j] = SW_weight[SW_I, j] / wsum

        for j in range(0, 8):
            weight[j] = SW_weight[SW_I, j]

    if flag == 0:
        sum_w = 0.0
        for j in range(0, 8):
            sum_w += weight[j]

        for j in range(0, 8):
            if weight[j] != 0:
                weight[j] = float(weight[j] / sum_w)

    # print(weight)
    return weight


def getColor():
    p_num = h * w

    A_row = list()  # 非零元素行
    A_col = list()  # 非零元素列
    A_data = list()  # 非零元素值
    len = 0  # 非零元素个数

    Bu = np.empty(p_num, dtype=np.float64)
    Bv = np.empty(p_num, dtype=np.float64)

    for i in range(0, p_num):
        Bu[i] = 0
        Bv[i] = 0

    for i in range(0, p_num):
        # 初始化稀疏矩阵A的对角线为1
        A_row.append(i)
        A_col.append(i)
        A_data.append(1.0)
        len += 1

        # 计算行列
        row = int(i / w)
        col = int(i % w)
        # 如果自己是着色点
        if src_y[row, col] != local[row, col, 0] or src_u[row, col] != local[row, col, 1] or src_v[row, col] != local[
            row, col, 2]:
            Bu[i] += local[row, col, 1]
            Bv[i] += local[row, col, 2]
            # print(Bu[i])
            continue  # 跳过这次循环

        # 计算邻域权重
        weight = get_w(row, col)
        for j in range(0, 8):
            if weight[j] != 0:
                # print(weight[j])
                index = 0  # 初始化下标
                if j == 0:
                    index = (row - 1) * w + col - 1
                elif j == 1:
                    index = (row - 1) * w + col
                elif j == 2:
                    index = (row - 1) * w + col + 1
                elif j == 3:
                    index = row * w + col - 1
                elif j == 4:
                    index = row * w + col + 1
                elif j == 5:
                    index = (row + 1) * w + col - 1
                elif j == 6:
                    index = (row + 1) * w + col
                else:
                    index = (row + 1) * w + col + 1

                nei_row = int(index / w)
                nei_col = int(index % w)
                # 如果邻居是着色点
                if src_y[row, col] != local[row, col, 0] or src_u[nei_row, nei_col] != local[nei_row, nei_col, 1] or \
                        src_v[nei_row, nei_col] != local[nei_row, nei_col, 2]:
                    Bu[i] += weight[j] * local[nei_row, nei_col, 1]
                    Bv[i] += weight[j] * local[nei_row, nei_col, 2]

                else:
                    # 添加稀疏矩阵A第i行的非零元素
                    A_row.append(i)
                    A_col.append(index)
                    A_data.append(-weight[j])
                    # print(-weight[j],i)

                    len += 1
        # if i % 100 == 0:
        # print(i)

    A_row_array = np.empty(len, dtype=int)
    A_col_array = np.empty(len, dtype=int)
    A_data_array = np.empty(len, dtype=np.float64)

    for i in range(0, len):
        A_row_array[i] = A_row[i]
        A_col_array[i] = A_col[i]
        A_data_array[i] = A_data[i]

    # 构造稀疏矩阵A
    sp_A = csc_matrix((A_data_array, (A_row_array, A_col_array)), shape=(p_num, p_num), dtype=np.float64)

    # 解u和v
    Xu = spsolve(sp_A, Bu)
    Xv = spsolve(sp_A, Bv)
    print(Xu)

    coloriza_img = np.zeros((h, w, 3), dtype='u1')

    for i in range(0, h):
        for j in range(0, w):
            tmpindex = i * w + j
            coloriza_img[i, j, 0] = src_y[i, j]

            tmpu = int(Xu[tmpindex])
            tmpv = int(Xv[tmpindex])

            # 防止越界
            if tmpu < 16:
                coloriza_img[i, j, 1] = 16
            elif tmpu > 240:
                coloriza_img[i, j, 1] = 240
            else:
                coloriza_img[i, j, 1] = tmpu

            if tmpv < 16:
                coloriza_img[i, j, 2] = 16
            elif tmpv > 240:
                coloriza_img[i, j, 2] = 240
            else:
                coloriza_img[i, j, 2] = tmpv

    return coloriza_img


if __name__ == '__main__':
    res_img = getColor()

    res_img_show = cv2.cvtColor(res_img, cv2.COLOR_YUV2RGB)

    plt.subplot(1, 2, 1)
    plt.imshow(src_gray)
    plt.subplot(1, 2, 2)
    plt.imshow(res_img_show)

    plt.show()
