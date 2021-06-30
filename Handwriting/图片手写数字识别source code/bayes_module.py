'''
@author: Guangji Huang
Created on Dec. 1 , 2020
'''
import numpy as np
import data_preprocess


# 计算模型参数
def cal_pwi(X, y, split_num, rate_f, reverse=0):
    """
    输入训练集，获取模型的参数，主要先验概率和类条件概率这两个
    :param X:训练样本的特征
    :param y:训练的标签
    :param split_num:特征的个数的开平方
    :param rate_f:判断特征为0的像素的比率
    :param reverse:手写数字是否为黑色
    :return:返回的是pwi和pxwi，都是列表的形式
    """
    # 求先验概率PWi
    #   统计样本中的0-9的个数，存储在列表num_sample中
    num_sample = [0]*10
    for i in y:
        num_sample[i] += 1
    #   计算每个类的先验概率，存储在列表PWi中
    pwi = [i/sum(num_sample) for i in num_sample]
    # print(pwi)
    # 对于X中的每一个样本（可能属于任何类，混乱），先求取特征，然后再同类的相加，放tem_list上
    tem_list = [np.zeros((len(X[0], )))]*10 # 先对每一类的所有特征之和进行初始化为0，为列表，列表中的元素为数组
    # 遍历X
    for i in range(len(X)):
        # 先检查标签，标签用作索引，然后更新tem_list中的和值
        index = y[i] # index就是类，对应于0-9
        tem_array = tem_list[index] + X[i]    # 这里采用相加的方法，所以就是1的个数占的比率
        # 更新tem_list
        tem_list[index] = tem_array
    # print("各类求和之后的特征列表：", "\n", tem_list)
    # 由tem_list计算每个类的每个特征的概率，遍历即可
    # 避免特征为1出现次数为零的情况，最后会导致相乘为零。但是分母还是要比分子大1，所以要加上2
    # 使用1-数组（），得到的是特征为0时的概率。就是类条件概率分布
    pxwi_fenbu = [1-(tem_list[i]+1)/(num_sample[i]+2) for i in range (10)]   # i就是tem_list其中的一个数组，注意在同一类的所有样本中每一特征，0出现的概率
    # 因为对于一个类中的所有特征，每一个特征都是独立的，所以类条件概率可以是各个特征为0的概率的相乘
    # pxwi = [np.prod(i) for i in pxwi_fenbu] # 其中i就是上面训练类的对应的所有特征概率的数组，这里是将一个数组相乘
    # print("每一个类的特征的概率", "\n", pxwi)
    return pwi, pxwi_fenbu

# 训练模型
def train_module(img_train, label_train):
    X, y = data_preprocess.c2feature(img_train, label_train)
    # 调用贝叶斯模型，计算模型参数
    pwi, pxwi = cal_pwi(X, y, 5, 0.1, 0)
    # 把模型参数写进文件中
    np.savetxt(r".\module parameter\pwi.txt", pwi, fmt='%f', delimiter=',')
    np.savetxt(r".\module parameter\pxwi.txt", pxwi, fmt='%f', delimiter=',')
    return pwi, pxwi
# 测试模型
def test_module(img_test, label_test):
    # 获取模型的参数，直接从文件中读取
    pwi = np.loadtxt(r'./module parameter/pwi.txt', delimiter=',')
    pxwi = np.loadtxt(r'./module parameter/pxwi.txt', delimiter=',')
    # 对测试集进行特征提取和处理
    X , y = data_preprocess.c2feature(img_test, label_test)
    # 进行预测
    pre_labels = []
    pwix = []
    for i in range(len(X)):
        tem_pxwi = [1] * 10  # 列表存放这个样本对于每一个类的后验概率
        # 对于第j个类
        for j in range(10):
            # 对于第k个特征
            for k in range(len(X[0])):
                if X[i][k] == 0:
                    tem_pxwi[j] *= pxwi[j][k]
                else:
                    tem_pxwi[j] *= (1 - pxwi[j][k])
        tem_pxwicwi = [tem_pxwi[i] * pwi[i] for i in range(10)]
        # 如果出现除数为零的情况，则需要修改
        tem_pwix = [tem_pxwicwi[i] / sum(tem_pxwicwi) for i in range(10)]
        pwix.append(np.array(tem_pwix))
        # print(tem_pwix)
        # 获取最大的概率对应的数字
        pre_num = np.argmax(np.array(tem_pwix))
        pre_labels.append(pre_num)
    # print(pwix)
    # print(pre_labels)
    # print(len(pre_labels))
    # 计算模型的准确率
    num_acc = 0
    num_acc_list = [0]*10
    num_saple_list = [0]*10
    for i in range(len(y)):
        num_saple_list[y[i]] += 1
        if pre_labels[i] == y[i]:
            # print(y[i])
            num_acc += 1
            num_acc_list[y[i]] +=1
        else:
            if __name__ == "__main__":
                print("\t", "真实的数字为: ", y[i], f"({pwix[i][y[i]] * 100:0.2f}%)",
                  "预测的数字为：", pre_labels[i], f"({pwix[i][pre_labels[i]] * 100:0.2f}%)", )
            # data_preprocess.img_show(img_orginal[i], 5, X[i])
    # acc = num_acc / len(pre_labels) * 100
    # print(num_acc)
    acc = num_acc / len(pre_labels)
    num_test = len(img_test)
    num_labels = len(pre_labels)
    print(f'模型训练总数为：{num_test}\t测试总数为：{num_labels}\t准确率为：{acc}')
    print(num_acc_list,num_saple_list)
    print([num_acc_list[i]/num_saple_list[i] for i in range(len(num_acc_list))])
    return acc

if __name__ == "__main__":
    img_train, label_train, img_test, label_test = data_preprocess.preprocess_image()
    # 训练模型
    # train_module(img_train, label_train)
    # 测试模型
    # test_module(img_test, label_test)
    test_module(img_train, label_train)
#
