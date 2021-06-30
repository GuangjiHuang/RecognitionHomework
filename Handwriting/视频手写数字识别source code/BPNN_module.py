'''
@author: Guangji Huang
Created on Dec. 1 , 2020
'''
import numpy as np
import data_preprocess
import time


# 使用单个神经元作为一个类
class  NeuralNetwork:
    # 创建一实例，初始化网络的参数，网络的结构和初始化的权重
    def __init__(self, num_inputs, layers, init_weight = None):
        # 网络的结构，隐藏层的层数
        self.num_inputs = num_inputs
        # 输入为列表就可以了，比如【2,2,2】表示的是有2个隐藏层和一个输出层，隐藏层有2,2个神经元，输出层有2个神经元
        self.layers = layers
        self.L = len(self.layers)
        # 初始化网络的参数,使用列表哦来存储每一层的参数，每一层使用数组进行存储，行数表示神经元对应的参数
        if init_weight:
            self.weight = init_weight
        else:
            # 如果是空的话，则全部初始化为0即可,偏置都要设置爱为0。注意需要增加一维
            self.weight = self.rand_w()
    # 定义一个初始化的权重函数
    def rand_w(self):
        weight = []
        for r in range(self.L):
            if r == 0:
                # weight_rj = np.random.rand(self.layers[r], self.num_inputs+1)
                weight_rj = np.random.rand(self.layers[r], self.num_inputs) * 0.2 - .2
                # weight_rj[:,-1] = 0
                # weight_rj[:,-1] = np.random.rand()
            else:
                # weight_rj = np.random.rand(self.layers[r], self.layers[r-1]+1)
                weight_rj = np.random.rand(self.layers[r], self.layers[r - 1]) * .2 - .2
                # weight_rj[:, -1] = 0
                # weight_rj[:, -1] = np.random.rand()
            weight.append(weight_rj)
        return weight
    ######## 对于单个神经元的相关的计算方法 #####
    # 向前计算
    # 对于一个样本，输入计算到输出的最终的结果
    # 输入样本定义为X，使用点乘的方法来实现
    def feed_forward(self, X, weight):
        # 先定义输出和激活输出,将各个层的输出和激活输出都放在一个列表中
        v_rj = [None]*self.L
        y_rj = [None]*self.L
        # 遍历每一层，并且遍历每一层中的神经元；
        # r = 1,2,……L表示神经网络的层数，j表示r层的第j个神经元。
        for r in range(self.L):
                # 计算第0层的神经网络的输出
                if r == 0:
                    # 进行矩阵乘法运算，得到的是第r列的神经元的数据
                    # 这里调用矩阵乘法，需要对权重的矩阵进行转置。那么输出的结果就是列向量
                    v_rj[r] = np.matmul(X,  np.transpose(weight[r]))
                else:
                    v_rj[r] = np.matmul(y_rj[r-1], np.transpose(weight[r]))
                # 然后计算y_rj
                y_rj[r] = 1 / (1 + np.exp(-v_rj[r]))
                # 进行扩维，在后面增加1
                # y_rj[r] = np.hstack((y_rj[r], 1))
        return v_rj,y_rj

    # 计算最后的MES，就是均方误差
    def errors(self, y_predict, y_target):
        # 误差的话，可能只是用来判断损失情况，并且决定是否结束迭代循环
        # 前面要注意对y_target进行扩维处理
        errors = 0.5* np.sum((y_predict - y_target)**2)
        return errors

    # 向后计算
    def calculate_delta_rj(self,X,y_target,weight,y_rj):
        # 获得网络的层数
        L = len(self.layers)
        # 将计算结果放在一个列表中，类似于输出一样
        delta_rj = [None] * L
        e_rj = [None] * L
        # 计算输出的偏导
        pd_v_rj =[i[:-1]*(1-i[:-1]) for i in y_rj] # 注意通过：-1去掉最后一个维度
        # 进行反向迭代，计算前面几层的e_rj
        for r_reverse in range(L-1, -1, -1):
            # 先计算最后一层的参数
            if r_reverse == L-1:
                # e_rj[r_reverse] = np.sum(y_rj[r_reverse] - y_target) * np.ones_like(y_target)[:-1] # 最后一维是没有用的，去掉
                e_rj[r_reverse] = (y_rj[r_reverse] - y_target) [:-1] # 最后一维是没有用的，去掉
            else:
                e_rj[r_reverse] = np.matmul(delta_rj[r_reverse + 1], weight[r_reverse + 1])[:-1]
            delta_rj[r_reverse] = e_rj[r_reverse] * pd_v_rj[r_reverse]

        return delta_rj

    def caculate_single_delat_w_rj(self, delta_rj, X, y_rj):
        # y_rj就是一个行数组，delta_rj就是一个行数组，所以首先是将其转置
        delta_w_rj = [None]*self.L
        for r in range(self.L):
            if r == 0:
                delta_w_rj[r] = delta_rj[r].reshape((delta_rj[r].shape[0], 1)) * X # 这里可能需要改一下
            else:
                # 乘以前一层
                delta_w_rj[r] = delta_rj[r].reshape((delta_rj[r].shape[0], 1)) * y_rj[r-1]
        # 一步更新所有的神经元的权值
        return delta_w_rj

    # 训练模型
    def train_module(self, X ,y, e_max, iteration_max, learning_rate):
        # 训练开始时间
        t_start = time.time()
        # 定义一个学习的速率
        rate = learning_rate

        # 网络的初始的权重
        weight = self.weight
        # 首先规定X表示训练集，列表形式，列表的元素就是一个特征数组
        # y就是标签集，列表形式，列表的元素就是一个多维的数组，对于手写数字，还是使用十个输出吧，所以一个数组就是1X10的情况
        # 进行扩维处理，新扩的那一个维度放在最后，固定为1,y是不用进行扩维的，因为是房子啊最后一层输出的
        # X = [np.hstack((x, 1)) for x in X]
        # 为了后面的计算方便，y也好扩维
        # y = [np.hstack((y_i, 1)) for y_i in y]
        num_sample = len(X)
        J_errors_pack = [1] * 3
        iter = 0
        J_errors_record = []
        while iter < iteration_max:
            if abs(J_errors_pack[1] - J_errors_pack[0]) <= 0.03 and abs(J_errors_pack[2] - J_errors_pack[0]) <= 0.03 and J_errors_pack[2]>20\
                and iter>3:
                # rate =  0.9*rate if rate > 0.01 else rate
                rate =  0.6*rate
                weight = self.rand_w()
                iter = 0
            if rate <= 0.001:
                rate = learning_rate
                iter = 0
                weight = self.weight
            # 如果误差在持续性地衰减，证明w的参数在正确的方向上，所以，可以适当增大一下rate，以提高收敛的速度
            # if iter == 20:
            #     rate = 5*rate
            # if iter == 100:
            #     rate = 0.8 * rate
            # if iter == 1000:
            #     rate = 0.8*rate
            print(f"学习率为：{rate}")
            # 对于一个特定的网络的权重，首先是判断该网络是否满足要求。看损失函数的值是否满足要求。
            # 所以，求出损失函数的值，以及计算好修正的权值
            J_erros = 0 # 最理想的结果就是为0
            delta_weight = [0] * self.L  # 最理想的结果就是不用更新了

            # 进行所有的样本遍历
            for num in range(num_sample-1,-1,-1):
                # 对于一个训练样本X[num]来说，首先是初始化。
                ######向前计算，求出输出v_rj和y_rj#######
                # 先定义输出和激活输出,将各个层的输出和激活输出都放在一个列表中
                v_rj = [None] * self.L
                y_rj = [None] * self.L
                # 遍历每一层，并且遍历每一层中的神经元；
                # r = 1,2,……L表示神经网络的层数，j表示r层的第j个神经元。
                for r in range(self.L):
                    # 计算第0层的神经网络的输出
                    if r == 0:
                        # 进行矩阵乘法运算，得到的是第r列的神经元的数据
                        # 这里调用矩阵乘法，需要对权重的矩阵进行转置。那么输出的结果就是列向量
                        v_rj[r] = np.matmul(X[num], np.transpose(weight[r]))
                    else:
                        v_rj[r] = np.matmul(y_rj[r - 1], np.transpose(weight[r]))
                    # 然后计算y_rj
                    y_rj[r] = 1 / (1 + np.exp(-v_rj[r]))
                    # 进行扩维，在后面增加1
                    # y_rj[r] = np.hstack((y_rj[r], 1))

                # 求出在当前的权重下的误差
                errors = 0.5 * np.sum((y_rj[-1] - y[num]) ** 2)

                if iter == iteration_max-1:
                    print(f"样本{num+1}的输出值为：{y_rj[-1]},误差为：{errors}")

                #### 向后计算，求出每一个样本对应的delta_rj
                # 将计算结果放在一个列表中，类似于输出一样
                delta_rj = [None] * self.L
                e_rj = [None] * self.L
                # 计算输出的偏导
                pd_v_rj = [i[:-1] * (1 - i[:-1]) for i in y_rj]  # 注意通过：-1去掉最后一个维度
                pd_v_rj = [i * (1 - i) for i in y_rj]  # 注意通过：-1去掉最后一个维度
                # 进行反向迭代，计算前面几层的e_rj
                for r_reverse in range(self.L - 1, -1, -1):
                    # 先计算最后一层的参数
                    if r_reverse == self.L - 1:
                        # e_rj[r_reverse] = np.sum(y_rj[r_reverse] - y_target) * np.ones_like(y_target)[:-1] # 最后一维是没有用的，去掉
                        e_rj[r_reverse] = (y_rj[r_reverse] - y[num])  # 最后一维是没有用的，去掉
                    else:
                        e_rj[r_reverse] = np.matmul(delta_rj[r_reverse + 1], weight[r_reverse + 1])
                    delta_rj[r_reverse] = e_rj[r_reverse] * pd_v_rj[r_reverse]

                # 计算每一个样本的权重改变贡献值
                # y_rj就是一个行数组，delta_rj就是一个行数组，所以首先是将其转置
                delta_w_rj = [None] * self.L
                for r in range(self.L):
                    if r == 0:
                        delta_w_rj[r] = delta_rj[r].reshape((delta_rj[r].shape[0], 1)) * X[num]  # 这里可能需要改一下
                    else:
                        # 乘以前一层
                        delta_w_rj[r] = delta_rj[r].reshape((delta_rj[r].shape[0], 1)) * y_rj[r - 1]
                # 一步更新所有的神经元的权值
                # 遍历叠加每一个样本产生的误差,以及需要修正的权重
                J_erros += errors

                # print(J_erros)
                # 注意下面的数组的运算方式
                # 先计算-u，就是学习率的情况
                tem_delta_weight = [-(rate*j) for j in delta_w_rj]
                delta_weight_temp = [delta_weight[i]+tem_delta_weight[i] for i in range(self.L)]
                delta_weight = delta_weight_temp
                # print(delta_weight)

            # 如果损失函数的值小于设定值，那么久终止迭代。或者迭代次数超过了最大的值，也停止了迭代。
            J_errors_record.append(J_erros)
            print(f"第{iter + 1}次迭代，损失函数的值为：{J_erros}")
            # 将错误放入到J_errors_pack袋中
            J_errors_pack[iter % 3] = J_erros
            if J_erros <= e_max:
                break
            # 如果没有满足，则继续更新，并且进行下一次迭代
            weight =[weight[i]+delta_weight[i] for i in range(self.L)]
            iter += 1
            #
            # print(weight)
        # 返回最终的网路的权重
        if __name__ != "__main__":
            np.save(r".\module parameter\BPNN_Wrj", np.array(weight))
        # figure_x = [i+1 for i in range(len(J_errors_record))]
        # 训练结束时间
        t_end = time.time()
        print(f"训练总共使用的时间：{t_end - t_start:.1f}s")
        return weight, J_erros

    def test_module(self, X_test, y_test, weight):
        y_label = [np.argmax(i) for i in y_test]
        y_out_list = []
        for num in range(len(X_test)):
            y_output = self.feed_forward(X_test[num], weight)[1][-1]
            y_predict = np.argmax(y_output)
            y_out_list.append(y_predict)
        # print(y_out_list)
        # print(y_label)
        num_fault = len(np.where(np.array(y_label) != np.array(y_out_list))[0])
        acc = 1-num_fault/len(X_test)
        print(f"测试样本数：{len(X_test)},错误样本数：{num_fault},准确率：{1-num_fault/len(X_test)}")
        num_right_list = [0] * 10
        num_sample_list = [0] * 10
        for i in range(len(y_label)):
            num_sample_list[y_label[i]] += 1
            if y_label[i] == y_out_list[i]:
                num_right_list[y_label[i]] += 1
        acc_list = [num_right_list[j] / num_sample_list[j] for j in range(len(num_sample_list))]
        print(f"各类的识别准确率为：{acc_list}")
        return  acc


    # 更新每一个神经元的权重值

if __name__ == "__main__":
    # 测试模型
    # 构造数据集
    img_train, label_train, img_test, label_test = data_preprocess.preprocess_image()
    X_test, y_test = data_preprocess.c2feature(img_test, label_test)
    X, y = data_preprocess.c2feature(img_train, label_train)
    X = list(X)
    y = list(y)
    y_tmp = []
    for i in range(len(y)):
        index = y[i]
        a_i = np.zeros((10,))
        a_i[index] = 1
        y_tmp.append(a_i)
    y = y_tmp
    y_tmp_test = []
    for i in range(len(y_test)):
        index = y_test[i]
        a_i = np.zeros((10,))
        a_i[index] = 1
        y_tmp_test.append(a_i)
    y_test = y_tmp_test
    # 训练模型
    nn = NeuralNetwork(25,[60,40,10])
    # nn.train_module(X,y,1.00,1000,10)[0]
    weight = np.load(r"./module parameter/BPNN_Wrj.npy",allow_pickle = True)
    nn.test_module(X_test,y_test,weight)
    nn.test_module(X,y,weight)



