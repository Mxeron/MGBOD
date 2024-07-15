from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D, axes3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from sklearn.cluster import k_means
import numpy.linalg as la
import datetime
import time
import warnings
import os
from scipy.spatial.distance import cdist


warnings.filterwarnings('ignore')


def get_radius(gb):
    real_gb = gb[:,:-1]
    center = real_gb.mean(0)
    radius = max(np.sqrt(np.sum((real_gb - center) ** 2, axis=1)))
    return radius


# 2-means
def spilt_ball(data):
    m = data.shape[1]
    cluster = k_means(X=data[:,:m - 1], init='k-means++', n_clusters=2, n_init=5)[1]
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]


# 根据距离快速划分粒球
def spilt_ball_2(data):
    real_data = data[:,:-1]
    ball1 = []
    ball2 = []
    D = cdist(real_data,real_data)
    r, c = np.where(D == np.max(D))
    r1 = r[1]
    c1 = c[1]
    # 上面找到粒球中距离最远的两个样本
    for j in range(0, len(data)):
        if D[j, r1] < D[j, c1]:  # 当前样本与样本r1、c1的距离
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_density_volume(gb):
    real_gb = gb[:,:-1]
    num = len(real_gb)
    center = real_gb.mean(0)
    sum_radius = np.sum(np.sqrt(np.sum((real_gb - center) ** 2, axis=1)))
    mean_radius = sum_radius / num
    if mean_radius != 0:
        density_volume = num / sum_radius
    else:
        density_volume = num

    return density_volume


def division_ball(gb_list):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) >= 4:  # 8 or 10?
            # ball_1, ball_2 = spilt_ball(gb)
            ball_1, ball_2 = spilt_ball_2(gb)
            # 标称数据存在问题
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb)
                continue
            density_parent = get_density_volume(gb)
            density_child_1 = get_density_volume(ball_1)
            density_child_2 = get_density_volume(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = (w1 * density_child_1 + w2 * density_child_2)
            # t1 = ((density_child_1 > density_parent) & (density_child_2 > density_parent))
            t2 = (w_child > density_parent)  # 大于父球的密度才划分，否则不划分
            # t3 = ((len(ball_1) > 4) & (len(ball_2) > 4))
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)

    return gb_list_new


def normalized_ball(gb_list, radius_detect):
    gb_list_temp = []
    for gb in gb_list:
        if len(gb) < 2:
            gb_list_temp.append(gb)
        else:
            # ball_1, ball_2 = spilt_ball(gb)
            ball_1, ball_2 = spilt_ball_2(gb)
            if get_radius(gb) <= 2 * radius_detect:
                gb_list_temp.append(gb)
            else:
                gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp


def get_GB(data):
    data_num = data.shape[0]
    index = np.array(range(data_num)).reshape(data_num, 1)  # 创建一个索引列, 从0开始的
    data = np.hstack((data, index))  # 在数据的最后一列添加索引列
    gb_list_temp = [data]

    # row = np.shape(gb_list_temp)[0]
    # col = np.shape(gb_list_temp)[1]
    # n = row * col

    # plot_dot(data)

    '''
    先生成粒球
    '''
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_ball(gb_list_temp)  # 质量轮
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break
    radius = []
    for gb in gb_list_temp:
        if len(gb) >= 2:  # 粒球中至少要有2个点
            radius.append(get_radius(gb))

    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median, radius_mean)
    
    '''
    再去除过大的粒球
    '''
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalized_ball(gb_list_temp, radius_detect)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    # plot_dot(data)
    # draw_ball(gb_list_temp)

    gb_list_final = gb_list_temp
    return gb_list_final

if __name__ == '__main__':
    load_data = loadmat(r"D:\outlier_datasets\Numerical\diabetes_tested_positive_26_variant1.mat")  # 加载数据集
    # load_data = loadmat('Lymphography.mat')
    data = load_data['trandata']
    n, m = data.shape
    print('样本个数：', n, '总属性个数', m)
    label = data[:, m - 1]
    print('异常点个数:', label.sum())
    trandata = data[:, :m - 1]
    scaler = MinMaxScaler()
    trandata = scaler.fit_transform(trandata)

    start_time = time.time()
    # 还没有把index带上！
    gb_final = get_GB(trandata)
    end_time = time.time()
    print("运行时间：", end_time - start_time)
    print("粒球个数：", len(gb_final))

    # idx_check = []
    # for i in gb_final:
    #     idx_check.extend(i[:,-1])
    #     print(i[:,-1])
    #     print('******')
    # idx_check = sorted(idx_check)
    # print(idx_check)
    # print(gb_final[0])
    # # gb center
    # print(np.mean(gb_final[0][:,:-1], axis=0))