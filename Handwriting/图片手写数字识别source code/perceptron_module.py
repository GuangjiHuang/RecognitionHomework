'''
@author: Guangji Huang
Created on Dec. 1 , 2020
'''
import numpy as np
import data_preprocess
import numpy as np


def preceptron_pack(X, y, itera_max, acc_min):
    """
    感知器模型，二分类器。使用袋式算法，对于线性不可分的情况，训练数据分类正确率满足大于acc_min或者超过了迭代次数就会停止训练。
    并且挑选其中效果最好的分类模型参数，就是最大分类准确率。
    :param X: 训练样本，二维数组：样本x特征
    :param y: 标签，-1和+1，数组：一维数组
    :param itera_n: 最大的迭代次数
    :param acc: 预规定的训练准确率
    :return: 返回二分类模型的参数w，以及最终的迭代次数和acc
    """
    # 获取样本的数量和特征数
    n_sample, n_feature = X.shape
    # 初始化w和X，主要是随机赋值w以及给X映射到高一维
    X = np.hstack((np.ones((n_sample, 1)), X))
    # 权重初始化，标准正态
    # w = np.random.randn(n_feature+1,1)
    w = np.zeros((n_feature+1,1))    # 相当于扩充了一个维度
    # 定义一个口袋，放置最优的w_pack和acc_pack
    w_pack = w
    acc_pack = 0
    # 进行迭代，根据梯度下降法，不断更新超平面的参数w，同时更新口袋中的两个参数
    for interation in range(itera_max):
        s = np.dot(X, w)
        y_pred = np.ones_like(y)
        loc_n = np.where(s < 0)[0]
        y_pred[loc_n] = -1
        num_fault = len(np.where(y != y_pred)[0])
        acc = 1 - num_fault/n_sample
        # print(f"第{interation+1}次迭代，正确率为：{acc:.4f} \t 错误个数为：{num_fault}/{n_sample}")
        # 准确率和袋子中的准确率相比较，如果好的话就更新袋子中的两个参数
        if acc > acc_pack:
            acc_pack = acc
            w_pack = w
        if  acc >= acc_min: # 当准确率大于预定的结果时，则停止迭代
            break
        else: # 否则更新w参数，就是更新分类超平面
            # 所有的错误的点参与w的修正过程，这个是累加的，遍历所有的错误的点就可以了
            for t in np.where(y!=y_pred)[0]:    # t就是其中一个错误的索引
                # 选择学习系数为0.5，这个数会影响收敛的速度和精度。
                w = w + 0.5 * y[t] * X[t, :].reshape((n_feature + 1, 1))
    # 最终的模型参数是放在袋子中的w_pack和acc_pack
    # print(f"训练模型的迭代次数为：{interation+1},准确率为：{acc_pack}")
    # 将w_pack进行转置，变为列向量
    w_pack = w_pack.reshape((w_pack.shape[1],w_pack.shape[0]))
    # 返回口袋中的w和acc以及最终迭代的次数。最终迭代的次数用于观察收敛的情况。
    return  w_pack, acc_pack, interation

def mutipleclass(X,y, iter_max, acc_min):
    """
    感知器袋式算法，实现多分类。主要是调用上面的二分类器perceptrom_pack
    :param X:二维数组，样本*特征
    :param y:对应的样本的标签，必须是数字。可以是0-9中任意类。
    :param iter_max:二分类器的最大的迭代次数
    :param acc_min:分类器的最小的分类正确率
    :return:返回每个分类器的类别，正确率以及分类超平面的参数w，是一个二维的数组：分类器*【类别1，类别2，准确率，w】
    """
    # 找出y中的标签的种类，通过集合去掉重复的。
    y_class = set(y)
    y_class = list(y_class)
    n_label = len(y_class)
    # 获取训练样本的个数，特征的个数
    n_s, n_f = X.shape
    # 采用一对一的分类方法，分类器的个数为：
    n_classfier = int(n_label*(n_label-1)/2)
    # 创建一个数组来存储模型的参数，采用的是一对一的分类方法
    module_p = np.zeros((n_classfier, n_f+4))  # 0:3，这个3个位置分别是n_min,n_max,acc_pack,n_f+4的意思就是3:n_f+4是扩维之后的超平面w
    # 进行一一的分类,采用双循环的形式
    c_index = 0 # 分类器的索引
    for i in range(n_label):
        for j in range(i+1,n_label):
            # 对标签进行比较，制定的规则是，小标签是-1，大标签是+1
            n_min, n_max = (y_class[i],y_class[j])  if y_class[i]<y_class[j] else (y_class[j],y_class[i])
            index1 = np.where(y == n_min)[0]
            index2 = np.where(y == n_max)[0]
            y_part = np.ones((len(index1) + len(index2)))
            y_part[0:len(index1)] = -1
            # 将两类的索引连接起来
            index = np.concatenate((index1, index2), axis=0)
            # 获取对应的X，就是两类的训练样本的索引连接起来，用于提取对应两类样本的特征
            X_part = X[index, :]
            # 调用二分类模型参数，返回w和acc_pack
            module_w ,acc_pack = preceptron_pack(X_part, y_part, iter_max, acc_min)[0:2]
            # 将对应的参数存放入到module_p中去
            module_p[c_index, :3] = np.array([n_min, n_max, acc_pack])
            module_p[c_index, 3:] = module_w
            # 更新分类器，下一个分类器，主要是用来作为上面的索引。
            c_index += 1
    # 返回模型训练的所有的结果，分类器个数*【类别1，类别2，准确率，w】
    return module_p

def train_module(img_train, label_train, iter_max, acc_min):
    """
    主要是先对图片进行处理，然后再调用多分类器，并且把模型的参数写进到module parameter这个文件夹中
    :param img_train:
    :param label_train:
    :param iter_max:
    :param acc_min:
    :return:
    """
    # 首先对图像进行提取特征
    X, y = data_preprocess.c2feature(img_train, label_train)
    # 调用多分类器进行分类
    module_p = mutipleclass(X, y, iter_max, acc_min)
    # 将模型参数写进到文件中保存
    np.savetxt(r".\module parameter\perceptron.txt", module_p, fmt='%f', delimiter=',')
    return module_p

def test_module(img_test, label_test):
    # 先是获取模型的参数。模型参数是训练后写入到文件中去了。
    module_p = np.loadtxt(r'./module parameter/perceptron.txt', delimiter=',')
    module_w = module_p[:,3:]
    module_c = module_p[:,0:2]
    # 测试模型
    # 获取测试集的特征
    test_X, test_y = data_preprocess.c2feature(img_test, label_test)
    # 很重要，还需要对test_X进行扩维
    test_X = np.hstack((np.ones((test_X.shape[0], 1)), test_X))
    # 使用一个列表来存放结果
    pre_test = []
    for x in test_X:
        # print(x)
        s = [np.dot(x, module_w[j, :]) for j in range(module_w.shape[0])]
        s_flag = [0 if i <= 0 else 1 for i in s]
        # 将每一个分类器的分类结果标签存放在vote_result中，
        vote_result = [int(module_c[i, s_flag[i]]) for i in range(len(s_flag))]
        # 转化成数组
        vote_result = np.array(vote_result)
        pre_num = np.argmax(np.bincount(vote_result))
        pre_test.append(pre_num)
    # 统计测试集的结果
    fault_index = (np.where(pre_test != test_y)[0])
    # print(fault_index)
    num_right_list = [0]*10
    num_sample_list = [0]*10
    for i in range(len(test_y)):
        num_sample_list[test_y[i]] += 1
        if test_y[i] == pre_test[i]:
            num_right_list[test_y[i]] +=1
    acc_list = [num_right_list[j]/num_sample_list[j] for j in range(len(num_sample_list)) ]
    print(f"各类的识别准确率为：{acc_list}")
    fault_num = len(fault_index)
    acc_test = 1 - fault_num / len(test_y)
    # 打印分类错误的情况
    print("模型的错分结果统计为：")
    for i in fault_index:
        print(f"{test_y[i]}错分为{pre_test[i]}")
    # 显示测试的结果
    print(f"测试模型，测试样本{len(test_y)},分类的正确率为：{acc_test * 100:.2f}%")
    return acc_test

if __name__ =="__main__":
    # 首先是选取训练和测试的数据集
    img_train, lable_train, img_test, label_test = data_preprocess.preprocess_image()
    # 对模型进行训练。这里不接受返回的值，测试的时候直接读取文件就可以了。
    train_module(img_train, lable_train, 1000, 1)
    # 对模型进行测试
    # test_module(img_train, lable_train)
    test_module(img_test, label_test)
