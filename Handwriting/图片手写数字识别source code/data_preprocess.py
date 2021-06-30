'''
@author: Guangji Huang
Created on Dec. 1 , 2020
'''
from mnist import MNIST
import numpy as np
import copy
import os
import cv2 as cv


def preprocess_MNIST():
    """
    对mnist数据集进行预处理
    :return:返回一个元组，分别是训练的数据、训练的标签、测试的数据和测试标签
    """
    mndata = MNIST("sample")
    # ##处理训练集
    images_train_list, labels_train = mndata.load_training()
    # 将图像数据的列表形式转化成数组形式
    images_train_array = [np.array(i_list).reshape((28, 28)) for i_list in images_train_list]
    # print(type(images_train_array[0]), len(images_train_array[0]))
    # ##处理测试集
    images_test_list, labels_test = mndata.load_training()
    images_test_array = [np.array(j_list).reshape((28, 28)) for j_list in images_test_list]
    # 返回一个元组，分别是训练的数据、训练的标签、测试的数据和测试标签
    # return images_train_array[:1000], labels_train[:1000], images_test_array[:1000], labels_test[:1000]
    return images_train_array, labels_train, images_test_array, labels_test


def preprocess_image():
    dir_train = r"./dataset/train_dataset"
    # dir_test = r"./dataset/test_dataset"
    dir_test = r"./dataset/test_dataset"
    list_train = os.listdir(dir_train)
    list_test = os.listdir(dir_test)
    labels_train = [int(i[0]) for i in list_train]
    labels_test = [int(i[0]) for i in list_test]
    images_train = [cv.imread(rf"{dir_train}/{i}", 0) for i in list_train]
    images_test = [cv.imread(rf"{dir_test}/{i}", 0) for i in list_test]
    return images_train, labels_train, images_test, labels_test

# 对训练或者测试集的数据进行处理，转化为特征数组的形式
def c2feature(img, label):
    """
    用于提取训练集合测试集的特征，以及转化成数组的形式
    :param img:读取到的图像数据
    :param label:标签
    :return:返回一个特征的集合
    """
    img_feature = []
    for interation in range(len(img)):
        # 遍历每一个样本，调用函数求取特征
        feature_i = get_feature(img[interation], 5, 0.1, 0)  # 这是每一个样本的特征
        # feature_i = get_feature(img[interation], 4, 0.1, 1)  # 这是每一个样本的特征
        feature = np.array(feature_i)  # 将列表转化为数组，方便计算
        img_feature.append(feature)
    img_feature =np.array(img_feature)
    label = np.array(label)
    return img_feature, label

# 对手写板的图片进行剪切
def cut_image(img, reverse=0):
    # 这个size是一个标志，用于找到手写数字的边缘地方
    color = reverse
    size = [-1]*4
    row,col = img.shape
    for i in range(row):
        line_a = (color in img[i,:])
        line_b = (color in img[row-1-i,:])
        if line_a and size[0] == -1:
            size[0] = i
        if line_b and size[1]==-1:
            size[1] = row-1-i
        if -1 not in size[0:2]:
            break
    for j in range(col):
        line_l = (color in img[:,j])
        line_r = (color in img[:,col - 1 - j])
        if line_l and size[2] == -1:
            size[2] = j
        if line_r and size[3] == -1:
            size[3] = col - 1 - j
        if -1 not in size[2:4]:
            break
    # 获取数字像素点的上，下，左，右的位置
    a, b, l, r = size
    # 获得剪切后的图片
    img_cut = img[a:b + 1, l:r + 1]
    # 使用插opencv中的插值法对图片进行放大成原来的尺寸，缩放成原来的尺寸，采用双峰插值的方法
    img_scale = cv.resize(img_cut, dsize=(row, col), interpolation=cv.INTER_LINEAR)
    # img_scale = cv.resize(img_cut, dsize=(row, col), interpolation=cv.INTER_NEAREST)
    # 返回切割并且放大成原来尺寸的图片
    return img_scale


def get_feature(img, split_num, rate, reverse=0, show=0):
    """
    输入图像的灰度图和分块以及比率判断，返回的就是一个图像的特征
    :param img: 输入单通道的图像数组
    :param split_num: 分成多少块，其中特征的个数就是分块值的平方
    :param rate: 黑字像素占的比例
    :param reverse: 是否需要将0和1反转
    :param show: 将cut和feature的图片写入到存储中，写入到show文件夹中，以便在界面中显示出来
    :return:返回的就是一个列表，含有所有的特征，全部都是0或者1，即是二值的意思
    """
    # 对二值化之后的图片进行切割处理，获得主要的手写数字________________________________这个步骤很重要
    img_cut = cut_image(img,reverse)
    # 看下输入数据的直方图
    # plt.hist(img)
    # plt.show()
    # 设置分块的初值，原则上是方阵
    row = column = split_num
    # 分块后的每个小块的行列数
    k = int(img.shape[0]/row)
    # 复制一份数据，用来画网格线
    img_mesh = copy.copy(img_cut)
    # 这个可选
    # 只画中间的几条线，侧边的不画
    for i in range(1, row):
        for j in range(1, column):
            img_mesh[j*k-1:j*k+1] = 0
        img_mesh[:, i*k-1:i*k+1] = 0
    # 二值化，黑字为0，白字为1
    # ret, img_bin = np.threshold(img, 150, 1, cv.THRESH_BINARY)
    # 大于150就会被视为1否则为0
    if reverse == 0:
        img_bin = np.where(img_cut > 150, 1, 0)
    if reverse == 1:
        img_bin = np.where(img_cut > 150, 0, 1)

    #   设置特征，将特征放入大列表中去
    img_feature = []
    for i in range(row):
        for j in range(column):
            # 获得小份快，每个小份快就是一个特征
            feature_group = img_bin[i*k:(i+1)*k, j*k:(j+1)*k]
            # 统计小份快的0的个数
            num_zeroes = k ** 2 - np.count_nonzero(feature_group)
            # 确定写的字的像素占整个小份快的比例为0或者1
            feature_single = 0 if num_zeroes >= rate * k ** 2 else 1
            img_feature.append(feature_single)
    # 可选，测试使用
    #   将原图化成特征图
    #       使用特征变形的矩阵和255的矩阵相乘可以得到结果，顺便画一下网格线
    img_feature_xishu = np.reshape(img_feature, (row, column))
    img_feature_show = 255*np.ones((img.shape), dtype="uint8")
    for i in range(row):
        for j in range(column):
            # 两个矩阵相乘
            img_feature_show[i*k:(i+1)*k, j*k:(j+1)*k] = img_feature_xishu[i, j]*img_feature_show[i*k:(i+1)*k, j*k:(j+1)*k]
            # 设置网格线
            img_feature_show[:, (j+1)*k-1:(j+1)*k+1] = 0
        img_feature_show[(i+1)*k-1:(i+1)*k+1, :] = 0
    # 返回的就是一个列表存储的特征
    # print(img_feature, "\t特征的个数为：", len(img_feature))
    ###将两个图片写进去
    if show:
        cv.imwrite(r"./show/cut.bmp", img_cut)
        cv.imwrite(r"./show/feature.bmp", img_feature_show)

    return img_feature


def img_show(img_orginal, split_num, img_feature):
    k = split_num
    img_oandm = copy.copy(img_orginal)
    num = int(img_orginal.shape[0]/k)
    # 将特征转化成k行和k列作为矩阵系数，用于相乘
    img_xishu = np.array(img_feature).reshape((k, k))
    img_fandm = 255*np.ones(img_orginal.shape)
    for i in range(k):
        for j in range(k):
            img_fandm[i*num:(i+1)*num, j*num:(j+1)*num] *= img_xishu[i,j]
    for i in range(k):
        img_oandm[i*num-1:i*num+1, :] = img_fandm[i*num-1:i*num+1, :] = 0
        img_oandm[:, i*num-1:i*num+1] = img_fandm[:, i*num-1:i*num+1] = 0
    cv.imshow("img_orginal", img_orginal)
    cv.imshow("img_oandm", img_oandm)
    cv.imshow("img_fandm", img_fandm)
    cv.waitKey(0)