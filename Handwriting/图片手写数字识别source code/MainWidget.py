'''
@author: Guangji Huang
Created on Dec. 1 , 2020
'''

from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox, QRect, QImage
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter,\
    QComboBox, QLabel, QSpinBox, QFileDialog
import qimage2ndarray as q2n
import cv2 as cv
import numpy as np
import bayes_module
import data_preprocess
import perceptron_module
from PaintBoard import PaintBoard
import random
import os
from BPNN_module import NeuralNetwork

# 将预测的数字定义为全局变量
pre_num = None

class MainWidget(QWidget):
    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)
        
        self.__InitData() #先初始化数据，再初始化界面
        self.__InitView()
    
    def __InitData(self):
        '''
                  初始化成员变量
        '''
        self.__paintBoard = PaintBoard(self)
        #获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames()  # 本来的类中就有这个
        
    def __InitView(self):
        '''
                  初始化界面
        '''
        # self.setFixedSize(640,480)
        self.setFixedSize(900,650)
        # self.setFixedSize(1130,700)
        self.setWindowTitle("手写数字识别系统")
        self.setStyleSheet("MainWidget{background:rgb(222,236,249,255);}")

        #新建一个水平布局作为本窗体的主布局,这里的main_layout其实就是一个对象，
        #这个对象规定了一些子部件的布局的问题，或者说是布局的位置的问题
        main_layout = QHBoxLayout(self)
        #设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)
        #########最左边,放置关于画板的画图的控件，提示表现，画板，画笔的粗细##################
        # subl_layout表示的是左边的子布局
        subl_layout = QVBoxLayout(self)
        subl_layout.setContentsMargins(10, 10, 10, 10)   # 设置边距
        self.__label_hint = QLabel(self)
        self.__label_hint.setText("数字手写板")
        self.__label_hint.setFixedSize(240,30)
        # self.__label_hint.setFixedHeight(30)
        self.__label_hint.setStyleSheet("QLabel{color:rgb(300,300,300,120);font-size:25px;font-weight:bold;font-family:宋体;}")
        subl_layout.addWidget(self.__label_hint)

        #subl_layout第二层放置画板
        subl_layout.addWidget(self.__paintBoard)

        # subl_layout第三层放置画笔粗细和复选框
        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        # self.__label_penThickness.setFixedHeight(20)
        self.__label_penThickness.setFixedSize(240, 30)
        subl_layout.addWidget(self.__label_penThickness)
        # 复选框
        self.__spinBox_penThickness = QSpinBox(self) # 好像是下拉列表
        self.__spinBox_penThickness.setMaximum(20)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setFixedSize(240, 30)
        self.__spinBox_penThickness.setValue(6) #默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2) #最小变化值为2
        self.__spinBox_penThickness.valueChanged.connect(self.on_PenThicknessChange)#关联spinBox值变化信号和函数on_PenThicknessChange
        subl_layout.addWidget(self.__spinBox_penThickness)
        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setFixedSize(240, 30)
        self.__btn_Clear.setParent(self) #设置父对象为本界面,因为self就是这个类所对应的实例，也就是这个总的界面
        # __btn_Clear是QPushButton的一个对象，这个对象是有一些方法的，比如下面的click方法，以及connect方法
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear) #__paintBoard是一个画板的对象，Clear明明就是一个方法函数，为什么不用（）呢？
        self.__btn_Clear.clicked.connect(self.__ctrltest) #__paintBoard是一个画板的对象，Clear明明就是一个方法函数，为什么不用（）呢？
        subl_layout.addWidget(self.__btn_Clear) # 将这个小控件按照垂直的布局的方法放到主界面中
        main_layout.addLayout(subl_layout)
        ###################创建一个subr_layout,主要是用于显示结果和使用分类方法###############
        ###### subr_layout = subra_layout(上面那层放置两个qpixmap，用来显示切割后和特征的图像)+subrb_layout(下面那层用来放置分类的方法和纠错的问题)
        subr_layout = QVBoxLayout(self)
        subra_layout= QHBoxLayout(self)
        subrb_layout = QVBoxLayout(self)
        ####subra_layout 左边放置切割之后的图像
        ####subra_layout再进行分割，= subra1_layout+ subra2_layout
        subra1_layout = QVBoxLayout(self)
        subra2_layout = QVBoxLayout(self)
        # subl_layout 先放置一个标签
        self.__label_hint = QLabel(self)
        self.__label_hint.setText("切割后的图片")
        self.__label_hint.setFixedHeight(30)
        self.__label_hint.setStyleSheet("QLabel{color:rgb(300,300,300,120);font-size:25px;font-weight:bold;font-family:宋体;}")
        subra1_layout.addWidget(self.__label_hint)
        # label用来放置qpixmap
        self.__label_cut = QLabel(self)
        self.__label_cut.setFixedSize(240, 240)
        self.__label_cut.setStyleSheet("QLabel{background:rgb(230,230,230,255);}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:50px;font-weight:bold;font-family:宋体;}")
        subra1_layout.addWidget(self.__label_cut)
        #################################   占位符组件############
        splitter = QSplitter(self) #占位符
        subra1_layout.addWidget(splitter)
        ####subra_layout 右边放置特征的图像
        # 先放置一个标签
        self.__label_hint = QLabel(self)
        self.__label_hint.setText("特征提取后的图片")
        # self.__label_showresult.setFixedSize(100,100)
        self.__label_hint.setFixedHeight(30)
        self.__label_hint.setStyleSheet("QLabel{color:rgb(300,300,300,120);font-size:25px;font-weight:bold;font-family:宋体;}")
        subra2_layout.addWidget(self.__label_hint)
        # label用来放置qpixmap
        self.__label_feature = QLabel(self)
        # self.__label_feature.setText("特征提取的手写数字")
        self.__label_feature.setFixedSize(240, 240)
        self.__label_feature.setStyleSheet("QLabel{background:rgb(230,230,230,255);}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:50px;font-weight:bold;font-family:宋体;}")
        subra2_layout.addWidget(self.__label_feature)
        #### 将subra_layout添加到subr_layout中去
        subra_layout.addLayout(subra1_layout)
        subra_layout.addLayout(subra2_layout)
        subr_layout.addLayout(subra_layout)
        ####subrb_layout第一层放置显示的分类结果###
        self.__label_showresult = QLabel(self)
        self.__label_showresult.setText("识别的结果为：")
        # self.__label_showresult.setFixedSize(100,100)
        self.__label_showresult.setFixedHeight(50)
        self.__label_showresult.setStyleSheet("QLabel{background:yellow;}"
                                              "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}")
        subrb_layout.addWidget(self.__label_showresult)
        # 在识别的结果后面再创建一个水平的布局
        subrb3_layout = QHBoxLayout(self)
        subrb3l_layout = QVBoxLayout(self)
        subrb3r_layout = QVBoxLayout(self)
        ####subrb_layout第二层放置分类器###
        self.__runbys = QPushButton("Bayes分类器")
        self.__runbys.setParent(self)  # 设置父对象为本界面
        self.__runbys.clicked.connect(self.__run_bayes)  # 将按键按下信号与画板清空函数相关联
        subrb3l_layout.addWidget(self.__runbys)
        # 贝叶斯分类结果
        self.__rungzq = QPushButton("感知器")
        self.__rungzq.setParent(self)  # 设置父对象为本界面
        self.__rungzq.clicked.connect(self.__run_preceptron)
        subrb3l_layout.addWidget(self.__rungzq)
        # 线性分类器分类结果
        self.__runcnn = QPushButton("BPNN分类器")
        self.__runcnn.setParent(self)  # 设置父对象为本界面
        self.__runcnn.clicked.connect(self.__run_BPNN)
        subrb3l_layout.addWidget(self.__runcnn)
        # 最底层的右半边的添加训练相关的按钮
        # 图片加入到巡行模型中
        self.__AddImage = QPushButton("正确识别，添加到训练模型")
        self.__AddImage.setParent(self)  # 设置父对象为本界面
        self.__AddImage.clicked.connect(self.Add_img)  # 将按键按下信号与画板清空函数相关联
        subrb3r_layout.addWidget(self.__AddImage)
        # 图片识别错误，选择正确的数字并加入到训练中去
        self.__AddCorect = QPushButton("错误识别，添加到训练模型")
        self.__AddCorect.setParent(self)  # 设置父对象为本界面
        self.__AddCorect.clicked.connect(self.Add_correct)  # 将按键按下信号与画板清空函数相关联
        subrb3r_layout.addWidget(self.__AddCorect)
        # 纠正后的数字复选框
        self.__spinBox_correct = QSpinBox(self) # 好像是下拉列表
        self.__spinBox_correct.setMaximum(9)
        self.__spinBox_correct.setMinimum(0)
        self.__spinBox_correct.setFixedSize(240, 30)
        self.__spinBox_correct.setValue(5) #默认粗细为10
        self.__spinBox_correct.setSingleStep(1) #最小变化值为2
        # self.__spinBox_correct.valueChanged.connect(self.on_PenThicknessChange)#关联spinBox值变化信号和函数on_PenThicknessChange
        subrb3r_layout.addWidget(self.__spinBox_correct)
        # 重新训练模型
        self.__Train = QPushButton("重新训练模型")
        self.__Train.setParent(self)  # 设置父对象为本界面
        self.__Train.clicked.connect(self.Train)  # 将按键按下信号与画板清空函数相关联
        subrb3r_layout.addWidget(self.__Train)
        # 下拉复选框，用于训练模型的选择
        self.__traincb = QComboBox(self)
        self.__traincb.addItems(["贝叶斯模型", "感知器模型", "BPNN模型"])
        # self.__traincb.currentIndexChanged[str].connect(self.Train)
        subrb3r_layout.addWidget(self.__traincb)
        ####将subrb_layout放置到subr_layout中去
        subrb3_layout.addLayout(subrb3l_layout)
        subrb3_layout.addLayout(subrb3r_layout)
        subrb_layout.addLayout(subrb3_layout)
        subr_layout.addLayout(subrb_layout)
        #####总的，将subr_layout放置到main_layout中去
        main_layout.addLayout(subr_layout)

    def __ctrltest(self):
        num = random.randint(0,100)
        self.__label_showresult.setText(f"识别的结果为：无输入")
        self.__label_cut.setPixmap(QPixmap(""))
        self.__label_feature.setPixmap(QPixmap(""))

    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList: 
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70,20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix),None)
            comboBox.setIconSize(QSize(70,20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)
        
    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)
    
    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()  # 这个是保存图片最关键的部分,这里的img是Qimg对象类型，并不是一个多维的数组
        image.save(savePath[0]) # 这里执行写入操作，将Qimg写入到磁盘中

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True #进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False #退出橡皮擦模式
        
    def Quit(self):
        self.close()

    def Add_img(self):
        # 如果画板是空的话，不执行
        flags = self.__paintBoard.IsEmpty()
        # 当画板为空的时候，就跳出这个循环
        while flags:
            # 显示那里提示没有输入
            self.__label_showresult.setText(f"画板为空，不保存！")
            return
        dir_train = r"./dataset/train_dataset"
        list_train = os.listdir(dir_train)
        list_label = [int(i[0]) for i in list_train]
        if pre_num == None:
            return
        num_pre_num = list_label.count(pre_num)
        print(f"现有{pre_num}的训练样本数目为：{num_pre_num}")
        # 路径和图片命名
        name = f"./dataset/train_dataset/{pre_num}_{num_pre_num+1}.bmp"
        # 将图片写入去
        img = self.__paintBoard.getImg()
        img = q2n.rgb_view(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imwrite(name, img)
        print(f"写入的图片为：{name}")
        self.__label_showresult.setText(f"添加的图片为：{pre_num}_{num_pre_num+1}.bmp")

    def Add_correct(self):
        # 如果画板是空的话，不执行
        flags = self.__paintBoard.IsEmpty()
        # 当画板为空的时候，就跳出这个循环
        while flags:
            # 显示那里提示没有输入
            self.__label_showresult.setText(f"画板为空，不保存！")
            return
        dir_train = r"./dataset/train_dataset"
        list_train = os.listdir(dir_train)
        list_label = [int(i[0]) for i in list_train]
        correct_num = self.__spinBox_correct.value()
        if correct_num == pre_num:
            return
        num_correct_num = list_label.count(correct_num)
        print(f"现有{correct_num}的训练样本数目为：{num_correct_num}")
        # 路径和图片命名
        name = f"./dataset/train_dataset/{correct_num}_{num_correct_num+1}.bmp"
        # 将图片写入去
        img = self.__paintBoard.getImg()
        img = q2n.rgb_view(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imwrite(name, img)
        print(f"写入的图片为：{name}")
        self.__label_showresult.setText(f"添加的图片为：{correct_num}_{num_correct_num+1}.bmp")

    def Train(self):

        # 选择要训练的模型的类型，贝叶斯，感知器，BPNN
        module_type = self.__traincb.currentText()
        # 模型列表
        module_l = ["贝叶斯模型", "感知器模型", "BPNN模型"]
        # 提示正在训练模型
        print(f"{module_type}正在训练中，waiting......")
        # 获取手写数字的训练集
        img_train, label_train = data_preprocess.preprocess_image()[0:2]
        #  调用训练模型，将训练后的模型参数写进到文件中
        if module_type == module_l[0]:
            bayes_module.train_module(img_train, label_train)
            acc = bayes_module.test_module(img_train, label_train)
        elif module_type == module_l[1]:
            perceptron_module.train_module(img_train, label_train, 100, 1)
            acc = perceptron_module.test_module(img_train,label_train)
        else:
            # BPNN模型待更新！
            # 先建立一个神经网络的对象
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
            # 训练模型
            # 训练模型
            # nn = NeuralNetwork(25, [80, 20, 20, 10])
            # weight = nn.train_module(X, y, 1.00, 2000, 0.1)[0]
            nn = NeuralNetwork(25, [60, 40, 10])
            weight = nn.train_module(X, y, 1.00, 2000, 1)[0]
            acc = nn.test_module(X, y, weight)
        self.__label_showresult.setText(f"{module_type}训练完成！ 训练集准确率：{acc:.3f}。")
        print(f"{module_type}训练完成，模型参数与更新！")

    def __run_bayes(self):
        flags = self.__paintBoard.IsEmpty()
        # print(flags)
        # 当画板为空的时候，就跳出这个循环
        while flags:
            # 显示那里提示没有输入
            self.__label_showresult.setText(f"识别的结果为：无输入\t\t（贝叶斯模型）")
            return
        img = self.__paintBoard.getImg()
        # img_s2 = self.__paintBoard.getQpixmap()
        img = q2n.rgb_view(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        pwi = np.loadtxt(r'./module parameter/pwi.txt', delimiter=',')
        pxwi = np.loadtxt(r'./module parameter/pxwi.txt', delimiter=',')
        feature_i = data_preprocess.get_feature(img, 5, 0.1, 0, 1)  # 这是每一个样本的特征
        feature = np.array(feature_i)  # 将列表转化为数组，方便计算
        tem_pxwi = [1]*10
        # 对于第j个类
        for j in range(10):
            # 对于第k个特征
            for k in range(len(feature)):
                if feature[k] == 0:
                    tem_pxwi[j] *= pxwi[j][k]
                else:
                    tem_pxwi[j] *= 1-pxwi[j][k]
        tem_pxwicwi = [tem_pxwi[i] * pwi[i] for i in range(10)]
        # 如果出现除数为零的情况，则需要修改
        tem_pwix = [tem_pxwicwi[i] / sum(tem_pxwicwi) for i in range(10)]
        global  pre_num
        pre_num = np.argmax(np.array(tem_pwix))
        # 更新显示识别的结果并且输出到label中，包括数字和图像
        # 显示识别数字
        self.__label_showresult.setText(f"识别的结果为：\t{pre_num}\t\t（贝叶斯模型）")
        # 显示切割后的图像
        pass
        img_s1 = QPixmap(r"./show/cut.bmp")
        self.__label_cut.setPixmap(img_s1)
        # 显示特征图像
        img_s2 = QPixmap(r"./show/feature.bmp")
        self.__label_feature.setPixmap(img_s2)

    def __run_preceptron(self):
        flags = self.__paintBoard.IsEmpty()
        # 当画板为空的时候，就跳出这个循环
        while flags:
            # 显示那里提示没有输入
            self.__label_showresult.setText(f"识别的结果为：无输入\t\t（感知器模型）")
            return
        img = self.__paintBoard.getImg()
        img = q2n.rgb_view(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 读取感知器模型的参数
        module_p = np.loadtxt(r'./module parameter/perceptron.txt', delimiter=',')
        module_w = module_p[:, 3:]
        module_c = module_p[:, 0:2]
        # 获取画板数字的特征参数
        feature_i = data_preprocess.get_feature(img, 5, 0.1, 0, 1)  # 这是每一个样本的特征
        feature = np.array(feature_i)  # 将列表转化为数组，方便计算
        # 增加一个维度，否则会出错
        x = np.hstack((np.array([1]),feature))
        s = [np.dot(x, module_w[j,:]) for j in range(module_w.shape[0])]
        s_flag = [0 if i <= 0 else 1 for i in s]
        # 将每一个分类器的分类结果标签存放在vote_result中，
        vote_result = [int(module_c[i,s_flag[i]]) for i in range(len(s_flag))]
        # 转化成数组
        vote_result = np.array(vote_result)
        # 这里的pre_num定义为全局的变量
        global pre_num
        pre_num = np.argmax(np.bincount(vote_result))
        # 更新显示识别的结果并且输出到label中，包括数字和图像
        # 显示识别数字
        self.__label_showresult.setText(f"识别的结果为：\t{pre_num}\t\t（感知器模型）")
        # 显示切割后的图像
        img_s1 = QPixmap(r"./show/cut.bmp")
        self.__label_cut.setPixmap(img_s1)
        # 显示特征图像
        img_s2 = QPixmap(r"./show/feature.bmp")
        self.__label_feature.setPixmap(img_s2)

    def __run_BPNN(self):
        flags = self.__paintBoard.IsEmpty()
        # 当画板为空的时候，就跳出这个循环
        while flags:
            # 显示那里提示没有输入
            self.__label_showresult.setText(f"识别的结果为：无输入\t\t（BPNN模型）")
            return
        img = self.__paintBoard.getImg()
        img = q2n.rgb_view(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 读取感知器模型的参数
        weight = np.load(r"./module parameter/BPNN_Wrj.npy", allow_pickle=True)
        # 获取画板数字的特征参数
        feature_i = data_preprocess.get_feature(img, 5, 0.1, 0, 1)  # 这是每一个样本的特征
        feature = np.array(feature_i)  # 将列表转化为数组，方便计算
        ### 创建一个神经网络的实例####
        ## 通过weight来确定layer的参数
        layers = [None] * len(weight)
        for i in range(len(weight)):
            layers[i] = weight[i].shape[0]
        nn = NeuralNetwork(len(feature), layers)
        y_output = nn.feed_forward(feature, weight)[1][-1] # 输出是一个一维的10数组
        # 这里的pre_num定义为全局的变量
        global pre_num
        pre_num = np.argmax(y_output)
        # print(pre_num)
        # 更新显示识别的结果并且输出到label中，包括数字和图像
        # 显示识别数字
        self.__label_showresult.setText(f"识别的结果为：\t{pre_num}\t\t（BPNN模型）")
        # 显示切割后的图像
        img_s1 = QPixmap(r"./show/cut.bmp")
        self.__label_cut.setPixmap(img_s1)
        # 显示特征图像
        img_s2 = QPixmap(r"./show/feature.bmp")
        self.__label_feature.setPixmap(img_s2)