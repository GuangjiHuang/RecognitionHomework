import time
import cv2 as cv
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from util import *

class Gui(QWidget):

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.init_data()
        self.init_wiget()

    def init_data(self):
        # the information that need to be shown
        self.status = "Running Status: "
        self.feature_text = []
        # the run time
        self.run_time = ""
        # the format of the StyleSheet
        self.ss_lb_img = "QLabel" \
                     "{" \
                     "background:#FFFFFF;" \
                     "font-size:15px;" \
                     "font-weight:bold;" \
                     "font-family:Time New Roman;" \
                     "}"
        self.ss_lb_title = "QLabel" \
                      "{" \
                      "background:#CDCDC1;" \
                      "font-size:18px;" \
                      "font-weight:bold;" \
                      "font-family:Time New Roman;" \
                      "}"
        self.ss_lb_status = "QLabel" \
                      "{" \
                      "background:#8EE5EE;" \
                      "color: red;" \
                      "font-size:18px;" \
                      "font-weight:bold;" \
                      "font-family:Time New Roman;" \
                      "}"
        self.ss_te = "QTextEdit" \
                           "{" \
                           "background:#E0EEEE;" \
                           "color:;" \
                           "font-size:14px" \
                           "font-weight:;" \
                           "font-family:Time New Roman;" \
                           "}"
        self.ss_bt = "QPushButton" \
                     "{" \
                     "background:#F7D674;" \
                     "font-size:14px;" \
                     "font-weight:;" \
                     "font-family:Time New Roman;" \
                     "}"

    def init_wiget(self):
        self.setWindowTitle("fingerprint recognition system")
        self.setGeometry(50, 50, 1000, 700)
        #self.setStyleSheet("MainWidget{background:rgb(222,236,249,255);}")
        self.setStyleSheet("MainWidget{background:#000000;}")
        """ deal with the main layout """
        main_layout = QHBoxLayout(self)
        l_layout = QVBoxLayout(self)
        r_layout = QVBoxLayout(self)
        main_layout.addLayout(l_layout)
        main_layout.addLayout(r_layout)
        """ deal with the left layout """
        # the line1: lb_1 and lb_2
        self.lb_1 = QLabel(self)
        self.lb_1.setText(f"原始图像")
        self.lb_1.setStyleSheet(self.ss_lb_title)
        self.lb_2 = QLabel(self)
        self.lb_2.setText(f"图像增强")
        self.lb_2.setStyleSheet(self.ss_lb_title)
        "add the widget to the l1_layout"
        l1_layout = QHBoxLayout(self)
        l1_layout.addWidget(self.lb_1)
        l1_layout.addWidget(self.lb_2)
        # the line2: lb_3 and lb_4
        self.lb_3 = QLabel(self)
        self.lb_3.setFixedSize(300, 300)
        self.lb_3.setStyleSheet(self.ss_lb_img)
        self.lb_4 = QLabel(self)
        self.lb_4.setFixedSize(300, 300)
        self.lb_4.setStyleSheet(self.ss_lb_img)
        "add the widget to the l2_layout"
        l2_layout = QHBoxLayout(self)
        l2_layout.addWidget(self.lb_3)
        l2_layout.addWidget(self.lb_4)
        # the line3: bt_1 and bt_2
        self.bt_1 = QPushButton("选择图像")
        self.bt_1.setStyleSheet(self.ss_bt)
        self.bt_1.clicked.connect(self.selectImage)
        self.bt_2 = QPushButton("图像增强")
        self.bt_2.setStyleSheet(self.ss_bt)
        self.bt_2.clicked.connect(self.showEnhancer)
        "add the widget to the l3_layout"
        l3_layout = QHBoxLayout(self)
        l3_layout.addWidget(self.bt_1)
        l3_layout.addWidget(self.bt_2)
        # the line4: lb_5 and lb_6
        self.lb_5 = QLabel(self)
        self.lb_5.setText(f"图像细化")
        self.lb_5.setStyleSheet(self.ss_lb_title)
        self.lb_6 = QLabel(self)
        self.lb_6.setText(f"特征提取")
        self.lb_6.setStyleSheet(self.ss_lb_title)
        "add the widget to the l4_layout"
        l4_layout = QHBoxLayout(self)
        l4_layout.addWidget(self.lb_5)
        l4_layout.addWidget(self.lb_6)
        # the line5: lb_7 and lb_8
        self.lb_7 = QLabel(self)
        self.lb_7.setFixedSize(300, 300)
        self.lb_7.setStyleSheet(self.ss_lb_img)
        self.lb_8 = QLabel(self)
        self.lb_8.setFixedSize(300, 300)
        self.lb_8.setStyleSheet(self.ss_lb_img)
        "add the widget to the l5_layout"
        l5_layout = QHBoxLayout(self)
        l5_layout.addWidget(self.lb_7)
        l5_layout.addWidget(self.lb_8)
        # the line6: bt_3 and bt_4
        self.bt_3 = QPushButton("图像细化")
        self.bt_3.setStyleSheet(self.ss_bt)
        self.bt_3.clicked.connect(self.showThinner)
        self.bt_4 = QPushButton("特征提取")
        self.bt_4.setStyleSheet(self.ss_bt)
        self.bt_4.clicked.connect(self.showFeature)
        "add the widget to the l6_layout"
        l6_layout = QHBoxLayout(self)
        l6_layout.addWidget(self.bt_3)
        l6_layout.addWidget(self.bt_4)
        # the line7: lb_status to show the running state
        self.lb_status = QLabel(self)
        self.lb_status.setFixedHeight(50)
        self.lb_status.setText(self.status)
        self.lb_status.setStyleSheet(self.ss_lb_status)
        "the line7 is the widget"
        "add the ln_layout to the l_layout"
        l_layout.addLayout(l1_layout)
        l_layout.addLayout(l2_layout)
        l_layout.addLayout(l3_layout)
        l_layout.addLayout(l4_layout)
        l_layout.addLayout(l5_layout)
        l_layout.addLayout(l6_layout)
        l_layout.addWidget(self.lb_status)
        """ deal with the right layout """
        # the line1: lb_9 to show the information of the feature
        self.lb_9 = QLabel(self)
        self.lb_9.setFixedHeight(20)
        self.lb_9.setText("     ** Feature Information **")
        self.lb_9.setStyleSheet(self.ss_lb_title)
        # the line2: te_1
        self.te_1 = QTextEdit(self)
        self.te_1.setFixedHeight(600)
        self.te_1.setStyleSheet(self.ss_te)
        self.te_1.setText("** Feature Infortamion ** ")
        # the line3: lb_10, to show the count of the feature
        self.lb_10 = QLabel(self)
        self.lb_10.setText("      ** Feature Count **")
        self.lb_10.setFixedHeight(20)
        self.lb_10.setStyleSheet(self.ss_lb_title)
        # the line4: lb_11, to show the count information
        self.lb_11 = QLabel(self)
        text_count = f"(1) 端点个数：\n\n" \
                     f"(2) 三叉点个数："
        self.lb_11.setText(f"{text_count}")
        self.lb_11.setStyleSheet(self.ss_lb_img)
        "add the widget to the r_layout"
        r_layout.addWidget(self.lb_9)
        r_layout.addWidget(self.te_1)
        r_layout.addWidget(self.lb_10)
        r_layout.addWidget(self.lb_11)

    def showStatus(self, m, n):
        action_list = ["加载图像", "图像增强", "图像细化", "特征提取"]
        if m == 0:
            self.status = f"Running Status: 正在进行{action_list[n]}, waiting..."
        else:
            self.status = f"Running Status: 完成{action_list[n]}! 用时：{self.run_time}"
        self.lb_status.setText(self.status)

    def selectImage(self):
        t_s = time.time()
        self.showStatus(0,0)
        # set the image as the global variable
        global img
        select_path = QFileDialog.getOpenFileName(self, "Select source file", r"./dataset")[0]
        self.lb_3.setPixmap(QPixmap(select_path))
        img = cv.imread(select_path)
        if len(img.shape) > 2:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # clear the lb_4, lb_7 and lb_8
        self.lb_4.clear()
        self.lb_7.clear()
        self.lb_8.clear()
        self.lb_status.setText(f"Running Status: 图像加载完成！")

    def showEnhancer(self):
        t_s = time.time()
        self.showStatus(0,1)
        global img
        blksze = 16
        thresh = 0.1
        normim, mask = ridge_segment(img, blksze, thresh)  # normalise the image and find a ROI
        gradientsigma = 1
        blocksigma = 7
        orientsmoothsigma = 7
        orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma)  # find orientation of every pixel
        blksze = 38
        windsze = 5
        minWaveLength = 5
        maxWaveLength = 15
        freq, medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,
                                   maxWaveLength)  # find the overall frequency of ridges
        freq = medfreq * mask
        kx = 0.65
        ky = 0.65
        newim = ridge_filter(normim, orientim, freq, kx, ky)  # create gabor filter and do the actual filtering
        img = 255 * (newim >= -3)
        cv.imwrite("temp/enhance.jpg", img)
        self.lb_4.setPixmap(QPixmap("temp/enhance.jpg"))
        # show the status
        self.run_time = f"{time.time() - t_s:.3f}s"
        self.showStatus(1,1)

    def showThinner(self):
        t_s = time.time()
        self.showStatus(0,2)
        global img
        array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
                 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
                 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
                 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
                 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

        num = 10
        for i in range(num):
            VThin(img, array)
            HThin(img, array)
        cv.imwrite("temp/thinner.jpg", img)
        self.lb_7.setPixmap(QPixmap("temp/thinner.jpg"))
        # show the status
        self.run_time = f"{time.time() - t_s:.3f}s"
        self.showStatus(1,2)

    def showFeature(self):
        t_s = time.time()
        self.showStatus(0,3)
        global img
        features = []
        h, w = img.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if img[i, j] == 0:  # 像素点为黑
                    m = i
                    n = j
                    eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1], img[m, n + 1],
                                  img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                    if sum(eightField) / 255 == 7:  # 黑色块1个，端点
                        # 判断是否为指纹图像边缘
                        if sum(img[:i, j]) == 255 * i or sum(img[i + 1:, j]) == 255 * (w - i - 1) or sum(
                                img[i, :j]) == 255 * j or sum(img[i, j + 1:]) == 255 * (h - j - 1):
                            continue
                        canContinue = True
                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                      [m + 1, n - 1],
                                      [m + 1, n], [m + 1, n + 1]]
                        for o in range(8):  # 寻找相连接的下一个点
                            if eightField[o] == 0:
                                index = o
                                m = coordinate[o][0]
                                n = coordinate[o][1]
                                break
                        for k in range(4):
                            coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                          [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                            eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1],
                                          img[m, n + 1],
                                          img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                            if sum(eightField) / 255 == 6:  # 连接点
                                for o in range(8):
                                    if eightField[o] == 0 and o != 7 - index:
                                        index = o
                                        m = coordinate[o][0]
                                        n = coordinate[o][1]
                                        break
                            else:
                                canContinue = False
                        if canContinue:

                            if n - j != 0:
                                if i - m >= 0 and j - n > 0:
                                    direction = atan((i - m) / (n - j)) + pi
                                elif i - m < 0 and j - n > 0:
                                    direction = atan((i - m) / (n - j)) - pi
                                else:
                                    direction = atan((i - m) / (n - j))
                            else:
                                if i - m >= 0:
                                    direction = pi / 2
                                else:
                                    direction = -pi / 2
                            feature = []
                            feature.append(i)
                            feature.append(j)
                            feature.append("endpoint")
                            feature.append(direction)
                            features.append(feature)
                            self.feature_text.append(f"端点,坐标({i},{j}),角度:{direction}")

                    elif sum(eightField) / 255 == 5:  # 黑色块3个，分叉点
                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                      [m + 1, n - 1],
                                      [m + 1, n], [m + 1, n + 1]]
                        junctionCoordinates = []
                        junctions = []
                        canContinue = True
                        for o in range(8):  # 寻找相连接的下一个点
                            if eightField[o] == 0:
                                junctions.append(o)
                                junctionCoordinates.append(coordinate[o])
                        for k in range(3):
                            if k == 0:
                                a = junctions[0]
                                b = junctions[1]
                            elif k == 1:
                                a = junctions[1]
                                b = junctions[2]
                            else:
                                a = junctions[0]
                                b = junctions[2]
                            if (a == 0 and b == 1) or (a == 1 and b == 2) or (a == 2 and b == 4) or (
                                    a == 4 and b == 7) or (
                                    a == 6 and b == 7) or (a == 5 and b == 6) or (a == 3 and b == 5) or (
                                    a == 0 and b == 3):
                                canContinue = False
                                break
                        if canContinue:  # 合格分叉点
                            directions = []
                            canContinue = True
                            for k in range(3):  # 分三路进行
                                if canContinue:
                                    junctionCoordinate = junctionCoordinates[k]
                                    m = junctionCoordinate[0]
                                    n = junctionCoordinate[1]
                                    eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1],
                                                  img[m, n + 1],
                                                  img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                    coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                                  [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                    canContinue = False
                                    for o in range(8):
                                        if eightField[o] == 0:
                                            a = coordinate[o][0]
                                            b = coordinate[o][1]
                                            if (a != i or b != j) and (
                                                    a != junctionCoordinates[0][0] or b != junctionCoordinates[0][
                                                1]) and (
                                                    a != junctionCoordinates[1][0] or b != junctionCoordinates[1][
                                                1]) and (
                                                    a != junctionCoordinates[2][0] or b != junctionCoordinates[2][1]):
                                                index = o
                                                m = a
                                                n = b
                                                canContinue = True
                                                break
                                    if canContinue:  # 能够找到第二个支路点
                                        for p in range(3):
                                            coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1],
                                                          [m, n + 1],
                                                          [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                            eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1],
                                                          img[m, n - 1],
                                                          img[m, n + 1],
                                                          img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                            if sum(eightField) / 255 == 6:  # 连接点
                                                for o in range(8):
                                                    if eightField[o] == 0 and o != 7 - index:
                                                        index = o
                                                        m = coordinate[o][0]
                                                        n = coordinate[o][1]
                                                        break
                                            else:
                                                canContinue = False
                                    if canContinue:  # 能够找到3个连接点

                                        if n - j != 0:
                                            if i - m >= 0 and j - n > 0:
                                                direction = atan((i - m) / (n - j)) + pi
                                            elif i - m < 0 and j - n > 0:
                                                direction = atan((i - m) / (n - j)) - pi
                                            else:
                                                direction = atan((i - m) / (n - j))
                                        else:
                                            if i - m >= 0:
                                                direction = pi / 2
                                            else:
                                                direction = -pi / 2
                                        directions.append(direction)
                            if canContinue:
                                feature = []
                                feature.append(i)
                                feature.append(j)
                                feature.append("bifurcation")
                                feature.append(directions)
                                features.append(feature)
                                features.append(feature)
                                self.feature_text.append(f"三叉点,坐标({i},{j}),角度:{tuple(directions)}")
        # write the te_1
        text = '\n\n'.join(self.feature_text)
        self.te_1.setText(text)
        # count the number of the endpoint and the bifurcation
        num_endpoint = 0
        # convert the img to the rgb
        img_show = cv.cvtColor(img.astype(np.uint8), cv.COLOR_GRAY2BGR)
        for m in range(len(features)):
            if features[m][2] == "endpoint":
                cv2.circle(img_show, (features[m][1], features[m][0]), 3, (0, 255, 0), 1)
                num_endpoint += 1
            else:
                cv2.circle(img_show, (features[m][1], features[m][0]), 3, (0, 0, 255), -1)
        cv.imwrite("temp/feature.jpg", img_show)
        self.lb_8.setPixmap(QPixmap("temp/feature.jpg"))
        # show  the count information in the label
        text_count = f"(1) 端点个数：{num_endpoint}\n\n" \
                     f"(2) 三叉点个数：{int((len(features) - num_endpoint)/2)}"
        self.lb_11.setText(text_count)
        # show the status
        self.run_time = f"{time.time() - t_s:.3f}s"
        self.showStatus(1,3)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    my_gui = Gui()
    my_gui.show()
    app.exec_()
