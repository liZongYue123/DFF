import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

####### 对数函数 #################################################
def log(x):
    """
    """
    if x == 0:
        return -(10 ** 30)
    return np.log(x)


def allmax(fm_list):
    all_idx = [0]
    max_fm = fm_list[0]
    for i in range(1, len(fm_list)):
        if fm_list[i] > max_fm:
            all_idx = [i]
            max_fm = fm_list[i]
        elif fm_list[i] == max_fm:
            all_idx.append(i)
    return np.array(all_idx)


def imgraysc(img):
    """ 0-255的缩放."""
    img_min = np.min(img)
    img_max = np.max(img)
    a = 255 / (img_max - img_min)
    b = -255 * img_min / (img_max - img_min)
    return a * img + b


def imshow(title, img):
    print(":::::Displaying", title)
    img = np.array(img, np.uint8)
    cv.imshow(title, img)
    cv.waitKey(0)


####### 对焦算子模块 #############################################
def sum_modified_laplacian(focus_stack, display_sml):
    sml_focus_stack = []
    kernel_fxx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
    kernel_fyy = kernel_fxx.T
    kernel_avg = np.ones((3, 3))
    for img in focus_stack:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        r, c = img.shape
        ml_img = np.zeros_like(img)

        fxx_img = cv.filter2D(img, cv.CV_64F, kernel_fxx)
        fyy_img = cv.filter2D(img, cv.CV_64F, kernel_fyy)
        ml_img = np.abs(fxx_img) + np.abs(fyy_img)
        sml_img = cv.filter2D(ml_img, cv.CV_64F, np.ones((3, 3)))

        if display_sml == "y":
            imshow("fxx", imgraysc(fxx_img))
            imshow("fxx Magnitude", imgraysc(np.abs(fxx_img)))
            imshow("fyy", imgraysc(fyy_img))
            imshow("fyy Magnitude", imgraysc(np.abs(fyy_img)))
            imshow("Laplacian", imgraysc(fxx_img + fyy_img))
            imshow("Modified Laplacian", imgraysc(ml_img))
            imshow("Sum Modified Laplacian", imgraysc(sml_img))
            cv.destroyAllWindows()

        sml_focus_stack.append(sml_img)

    return sml_focus_stack


####### 基于高斯插值进行深度估计 ##########################
def get_neighbors(fm_list, m):
    F_m_prev = fm_list[m - 1]
    F_m = fm_list[m]
    F_m_next = fm_list[m + 1]
    return F_m_prev, F_m, F_m_next


def adjusted_neighbor_triplet(fm_list, m):
    """
    保证Fm每次不相等，多尝试几次，可能迭代次数有点少，结果不是很好
    """
    F_m_prev, F_m, F_m_next = get_neighbors(fm_list, m)
    if F_m == F_m_prev:
        F_m_prev = 0.99 * F_m_prev
    if F_m == F_m_next:
        F_m_next = 0.99 * F_m_next
    return F_m_prev, F_m, F_m_next


def gaussian_interpolation(F_m_prev, F_m, F_m_next, m):
    if F_m == 0:
        depth = m
    else:
        depth = m
        depth += (log(F_m_next) - log(F_m_prev)) / (
            2 * (2 * log(F_m) - log(F_m_next) - log(F_m_prev))
        )
    return depth


def estimate_depth_per_pixel(sml_focus_stack):
    img_size = sml_focus_stack[0].size
    stack_size = len(sml_focus_stack)
    # 将图片stack转为矩阵
    focus_meas_matrix = np.zeros((stack_size, img_size))
    for i, sml_img in enumerate(sml_focus_stack):
        sml_img_vec = sml_img.reshape((img_size,))
        for j, pixel_focus_meas in enumerate(sml_img_vec):
            focus_meas_matrix[i, j] = pixel_focus_meas

    # 使用高斯插值获得深度
    depth_map = np.zeros((img_size,))
    zero_map = np.ones((img_size,))
    for i, focus_meas_vec in enumerate(focus_meas_matrix.T):
        all_max = allmax(focus_meas_vec) + 1
        focus_meas_vec = np.array([0] + focus_meas_vec.tolist() + [0])
        # 均值化
        depth_sum = 0
        for m in all_max:
            F_m_prev, F_m, F_m_next = adjusted_neighbor_triplet(focus_meas_vec, m)
            depth_sum += gaussian_interpolation(F_m_prev, F_m, F_m_next, m)

        avg_depth = depth_sum / len(all_max)
        depth_map[i] = avg_depth - 1  # 填充深度值

    return depth_map.reshape(sml_focus_stack[0].shape)
