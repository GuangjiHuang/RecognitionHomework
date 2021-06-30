'''
@author: Guangji Huang
Created on May. 29 , 2021
'''
import sys
import os
import copy
import time
import cv2 as cv
import numpy as np
import data_preprocess
from BPNN_module import NeuralNetwork
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QTimer


class Gui(QWidget):


    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.init_data()
        self.init_wiget()

    def init_data(self):
        # the attribute:
        # the dealing type: the video from you disk or the camera that you are open, you can chose.
        self.deal_type = "Video" # the default value is the video, and this can be activated by the QComboBox
        # about the source video: path, file_name, shape, frames
        self.video_src = {"path": None,
                                "file_name": None,
                                "shape": None,
                                "frames": None,
                                "fps": None}
        # get the source video's information, every time you select the path, you should
        # renew the information of the vidoe_src
        # the default path
        self.init_video_path = r"testVideo/test.mp4"
        self.get_video_src()
        #print("-------inti the video_src--------")
        #print(self.video_src)

        # about the runtime video: frame_id
        ## use the Timer to control the output time
        self.timer_camera = QTimer() # creat the timer object
        # you have to set the configuration here, the bg and the bn's step; and the timer
        ## the timer is the gap of the image that will show in the lb2
        self.video_runtime_static = {"c_step": 24,
                                        "time_gap": 30} # it means that 25ms
        # about the video_runtime_cfg
        self.video_runtime_cfg = {"frame_id": 0,
                                  "classify_method": None}

        # about the video_runtime_record, including the digit's number and so on
        self.video_runtime_record = {"digit_nu": 0}

        # this is the run_time status: there are three status: run, pause, stop
        ## use the push_bt_num to control the run_time_status
        self.runtime_status = "stop"
        self.push_bt_num = {"start": 0, "stop": 1}
        # deal_img setting. it is about the some threshold of the image dealing:
        # thres_binary: 100
        # thres_area_min : 20
        # tres_area_max : 3000
        # thres_contour_length : 40
        # flag_rverse: False (it means that no need to flip the image)
        self.thres = {"binary": 130,
                        "area_min": 20,
                        "area_max": 3000,
                        "contour_length": 40,
                        "flag_reverse": False}

        # about the text that the label shows
        self.text_lb_gf = f"Author: Guangji && Shuqiao\t\tDate:May 28th, 2021\t\t\t\t\t\t\t\t\t{time.strftime('%H:%M:%S    %Y-%m-%d')}"
        # this is the output information:* the deal type; *the video's size, the video digit's  number
        self.output_info = f"1) You are dealing with the: {self.deal_type}\n\n\n" \
                           f"2) The image's shape: {self.video_src['shape']}\n\n\n" \
                           f"3) The total digit in the video now is: {self.video_runtime_record['digit_nu']}\n\n\n"

        # this is the editor_cfg, including the : threshold: binary_threshold, area_min, area_max, contour_length
        self.text_editor_info  = f"Note: you can modify the arguments shown as below.\n\n" \
                                 f"binary: {self.thres['binary']}\n\n" \
                                 f"area_min: {self.thres['area_min']}\n\n" \
                                 f"area_max: {self.thres['area_max']}\n\n" \
                                 f"contour_length: {self.thres['contour_length']}\n"


        # about the class attribute and it is used for the format the widgets
        # ss_bt -> ss_bt: the PushButton Style Sheet
        # self.ss_lb -> self.ss_lb: the label Style Sheet
        self.ss_bt = "QPushButton" \
                     "{" \
                     "background:#F7D674;" \
                     "font-size:14px;" \
                     "font-weight:;" \
                     "font-family:Time New Roman;" \
                     "}"
        self.ss_lb = "QLabel" \
                     "{" \
                     "background:#FFFFFF;" \
                     "font-size:14px;" \
                     "font-weight:;" \
                     "font-family:Time New Roman;" \
                     "}"
        self.ss_lb1 = "QLabel" \
                     "{" \
                     "background:#000000;" \
                     "font-size:14px;" \
                     "font-weight:;" \
                     "font-family:Time New Roman;" \
                     "}"
        self.ss_lb2_status = "QLabel" \
                      "{" \
                      "background:#8EE5EE;" \
                      "color: red;" \
                      "font-size:18px;" \
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
        self.ss_lb_title_r = "QLabel" \
                           "{" \
                           "background:#CDCDC1;" \
                             "color:red;" \
                           "font-size:18px;" \
                           "font-weight:bold;" \
                           "font-family:Time New Roman;" \
                           "}"
        self.ss_lb_ccb = "QLabel" \
                      "{" \
                      "background:#98FB98;" \
                      "font-size:12px;" \
                      "font-weight:bold;" \
                      "font-family:Time New Roman;" \
                      "}"
        self.ss_lb_text1 = "QLabel" \
                         "{" \
                         "background:#E0EEEE;" \
                         "font-size:14px;" \
                         "font-weight:bold;" \
                         "font-family:Time New Roman;" \
                         "}"
        self.ss_lb_text2= "QLabel" \
                           "{" \
                           "background:#C1FFC1;" \
                           "font-size:14px;" \
                           "font-weight:;" \
                           "font-family:Time New Roman;" \
                           "}"
        self.ss_text_editor = "QTextEdit" \
                           "{" \
                           "background:#E0EEEE;" \
                           "color:green;" \
                           "font-size:16px;" \
                           "font-weight:bold;" \
                           "font-family:Time New Roman;" \
                           "}"
        self.ss_lb_gf = "QLabel" \
                        "{" \
                        "background:#A8A8A8;" \
                        "font-size:13px;" \
                        "font-weight:bold;" \
                        "font-family:Time New Roman;" \
                        "}"
        self.ss_cb = "QCheckBox" \
                        "{" \
                        "background:#BFFF00;" \
                        "font-size:16;" \
                        "font-weight:bold;" \
                        "font-family:Time New Roman;" \
                        "}"
        self.ss_cbb_s = "QComboBox" \
                     "{" \
                     "font-size:16;" \
                        "color:red;" \
                     "font-weight:bold;" \
                     "font-family:Time New Roman;" \
                     "}"


    def init_wiget(self):
        self.setWindowTitle("HnadWriting Recognition video")
        self.setGeometry(50, 50, 1200, 675)
        #self.setStyleSheet("MainWidget{background:rgb(222,236,249,255);}")
        self.setStyleSheet("MainWidget{background:#000000;}")
        """ deal with the main layout """
        main_layout = QVBoxLayout(self)
        main_bottom_layout = QHBoxLayout(self)
        main_top_layout = QHBoxLayout(self)
        main_layout.addLayout(main_top_layout)
        main_layout.addLayout(main_bottom_layout)
        l_layout = QVBoxLayout(self)
        r_layout = QVBoxLayout(self)
        main_top_layout.addLayout(l_layout)
        main_top_layout.addLayout(r_layout)
        """ deal with the left layout """
        # the line1 : lb1_file_name
        self.lb1_file_name = QLabel(self)
        self.lb1_file_name.setFixedHeight(25)
        self.lb1_file_name.setText(f"(VIDEO) {self.video_src['file_name']}")
        self.lb1_file_name.setStyleSheet(self.ss_lb_title_r)
        # the line2 : lb2_video
        self.lb2_video = QLabel(self)
        self.lb2_video.setFixedSize(900, 600)
        # set the size the same as the video
        #self.lb2_video.setFixedSize(960, 544)
        self.lb2_video.setText("hello the world")
        self.lb2_video.setStyleSheet(self.ss_lb1)
        # the line2-2 : lb2_status
        self.lb2_status = QLabel(self)
        self.lb2_status.setFixedHeight(25)
        self.lb2_status.setText("Running Status: ")
        self.lb2_status.setStyleSheet(self.ss_lb2_status)
        # the line3 : the button: select, start, stop, <, and >(all 5 buttons)
        ## the bt2_select
        self.bt2_start = QPushButton("Start/Pause")
        self.bt2_start.setStyleSheet(self.ss_bt)
        self.bt2_start.clicked.connect(self.startVideo)
        ## the bt3_stop
        self.bt3_stop = QPushButton("Stop")
        self.bt3_stop.setStyleSheet(self.ss_bt)
        self.bt3_stop.clicked.connect(self.stopVideo)
        ## the bt4_bf
        self.bt4_bf = QPushButton("< <")
        self.bt4_bf.setStyleSheet(self.ss_bt)
        self.bt4_bf.clicked.connect(self.bfVideo)
        ## the bt5_bn
        self.bt5_bn = QPushButton("> >")
        self.bt5_bn.setStyleSheet(self.ss_bt)
        self.bt5_bn.clicked.connect(self.bnVideo)
        ## the checkbox
        self.cb1 = QCheckBox("flip Video", self)
        self.cb1.setStyleSheet(self.ss_cb)
        self.cb1.stateChanged.connect(self.selectVideoOrCamera)
        """ add the widget to the l1_layout """
        l1_layout = QHBoxLayout(self)
        l1_layout.addWidget(self.bt4_bf)
        l1_layout.addWidget(self.bt2_start)
        l1_layout.addWidget(self.bt5_bn)
        l1_layout.addWidget(self.bt3_stop)
        l1_layout.addWidget(self.cb1)
        # the line4: *lb7_type, *lb8_method, *cbb1_type, *cbb2_method
        ## the bt1_select
        self.bt1_select = QPushButton("Load Video")
        self.bt1_select.setStyleSheet(self.ss_bt)
        self.bt1_select.clicked.connect(self.selectVideoPath)
        ## the lb7_type
        self.lb7_type = QLabel("Video|Camera", self)
        self.lb7_type.setStyleSheet(self.ss_lb_ccb)
        ## the lb8_method
        self.lb8_method = QLabel("Methods", self)
        self.lb8_method.setStyleSheet(self.ss_lb_ccb)
        ## the cbb1_type
        self.cbb1_type = QComboBox(self)
        self.cbb1_type.addItems(["Video", "Camera"])
        self.cbb1_type.setStyleSheet(self.ss_cbb_s)
        self.cbb1_type.currentTextChanged.connect(self.setDealType)
        ## the cbb1_type
        self.cbb2_method = QComboBox(self)
        self.cbb2_method.addItems(["BPNN", "Bayes", "Perceptron"])
        self.cbb2_method.setStyleSheet(self.ss_cbb_s)
        """ add the widget to the l2_layout"""
        l2_layout = QHBoxLayout(self)
        l2_layout.addWidget(self.lb7_type)
        l2_layout.addWidget(self.cbb1_type)
        l2_layout.addWidget(self.bt1_select)
        l2_layout.addWidget(self.lb8_method)
        l2_layout.addWidget(self.cbb2_method)
        """ add the widget and the l1_layout to the l_layout"""
        l_layout.addWidget(self.lb1_file_name)
        l_layout.addWidget(self.lb2_video)
        l_layout.addWidget(self.lb2_status)
        l_layout.addLayout(l1_layout)
        l_layout.addLayout(l2_layout)
        """ deal with the right layout """
        # line 1: the lb3_title of the video's information
        self.lb3_cfg = QLabel(self)
        self.lb3_cfg.setFixedHeight(25)
        self.lb3_cfg.setStyleSheet(self.ss_lb_title)
        self.lb3_cfg.setText("** Configuration **")
        # line2 :creat the TextEdit
        self.text_editor = QTextEdit(self)
        self.text_editor.setFixedHeight(350)
        self.text_editor.setStyleSheet(self.ss_text_editor)
        self.text_editor.setText(self.text_editor_info)
        # line 3: the lb4_cfg_info
        self.lb4_cfg_info = QLabel(self)
        self.lb4_cfg_info.setFixedHeight(20)
        self.lb4_cfg_info.setStyleSheet(self.ss_lb_text2)
        # line4 : add two button to change the cfg
        self.bt6_show_cfg = QPushButton("Show")
        self.bt6_show_cfg.clicked.connect(self.showCfg)
        self.bt6_show_cfg.setStyleSheet(self.ss_bt)
        self.bt7_modify_cfg = QPushButton("Modify")
        self.bt7_modify_cfg.clicked.connect(self.modifyCfg)
        self.bt7_modify_cfg.setStyleSheet(self.ss_bt)
        """ creat the r1_layout to include the bt6 and the bt7"""
        r1_layout = QHBoxLayout(self)
        r1_layout.addWidget(self.bt6_show_cfg)
        r1_layout.addWidget(self.bt7_modify_cfg)
        # line5 : the lb5_dst_title
        self.lb5_dst_title = QLabel(self)
        self.lb5_dst_title.setFixedHeight(25)
        self.lb5_dst_title.setText("** Output information **")
        self.lb5_dst_title.setStyleSheet(self.ss_lb_title)
        # line6 : the lb6_dst_info
        self.lb6_dst_info = QLabel(self)
        self.lb6_dst_info.setStyleSheet(self.ss_lb_text1)
        self.lb6_dst_info.setText(self.output_info)
        """ add the widgets to the r_layout """
        r_layout.addWidget(self.lb3_cfg)
        r_layout.addWidget(self.text_editor)
        r_layout.addWidget(self.lb4_cfg_info)
        r_layout.addLayout(r1_layout)
        r_layout.addWidget(self.lb5_dst_title)
        r_layout.addWidget(self.lb6_dst_info)
        """ deal with the bottom line """
        self.lb_gf = QLabel(self)
        self.lb_gf.setFixedHeight(20)
        self.lb_gf.setStyleSheet(self.ss_lb_gf)
        self.lb_gf.setText(self.text_lb_gf)
        """ add the bottom label to the main_layout """
        main_bottom_layout.addWidget(self.lb_gf)
        ###  the others: show time , timer
        self.show_timer = QTimer(self)
        self.show_timer.timeout.connect(self.showTime)    # show the time nwo
        self.show_timer.start()

    def showTime(self):
        self.text_lb_gf = f"Author: Guangji && Shuqiao\t\tDate:May 28th, 2021\t\t\t\t\t\t\t\t\t{time.strftime('%H:%M:%S    %Y-%m-%d')}"
        self.lb_gf.setText(self.text_lb_gf)
    def setDealType(self):
        self.deal_type = self.cbb1_type.currentText()
        # and then, renew the self.output's info
        self.changeDstInfo()
        # and then change the way to the camera, and init the video_src
        self.get_video_src()
    def changeDstInfo(self):
        # because the f-string's content need to renew, so you have to assign the value to the output_info again
        # and then set the content to the lb6_dst_info
        self.output_info = f"1) You are dealing with the: {self.deal_type}\n\n\n" \
                           f"2) The image's shape: {self.video_src['shape']}\n\n\n" \
                           f"3) The total digit in the video now is: {self.video_runtime_record['digit_nu']}\n\n\n"
        self.lb6_dst_info.setText(self.output_info)

    def changeTextEditorInfo(self):
        # renew the info
        self.text_editor_info  = f"binary: {self.thres['binary']}\n\n" \
                                 f"area_min: {self.thres['area_min']}\n\n" \
                                 f"area_max: {self.thres['area_max']}\n\n" \
                                 f"contour_length: {self.thres['contour_length']}\n\n"

    def showCfg(self):
        # renew the text_editor's content
        self.changeTextEditorInfo()
        # put the text in the editor
        self.text_editor.setText(self.text_editor_info)

    def modifyCfg(self ):
        # get the text first
        get_cfg = self.text_editor.toPlainText()
        # delete all the space
        get_cfg = get_cfg.replace(" ", "")
        # then deal pharase the text
        # get the key-value first
        get_cfg_list = get_cfg.split("\n")
        get_cfg_list = [i for i in get_cfg_list if ":" in i]
        for key_value in get_cfg_list:
            key, value = key_value.split(":")
            if key in self.thres.keys():
                ## check if the value is the number
                if value.isnumeric():
                    self.thres[key] = int(value)
                else:
                    print(f"** cfg setting for the {key}: {value} is not the number!")
                    self.lb4_cfg_info.setText(f"** cfg setting for the {key}: {value} is not the number!")
                    continue

    def get_video_src(self):
        # creat the video_capture object
        ## if the path is not empty, creat the object, otherwise, it weill make wrong
        #*****************************************************************************
        # dealing with the video
        #*****************************************************************************
        if self.deal_type == "Video":
            #if self.video_src["path"]:
            # None or the last sate(the camera: just need to check if is the digit)
            if (self.video_src["path"] == None):
                self.video_src["path"] = self.init_video_path
                file_name = os.path.basename(self.video_src["path"])
                self.video_src["file_name"] = file_name
            elif (type(self.video_src["path"]) is  int):
                self.video_src["path"] = self.init_video_path
                file_name = os.path.basename(self.video_src["path"])
                self.video_src["file_name"] = file_name
                self.lb1_file_name.setText(f"(VIDOE) {self.video_src['file_name']}")
            else:
                file_name = os.path.basename(self.video_src["path"])
                self.video_src["file_name"] = file_name
                self.lb1_file_name.setText(f"(VIDOE) {self.video_src['file_name']}")
        #*****************************************************************************
        # dealing with the camera
        #*****************************************************************************
        else:
            self.video_src["path"] = 0
            self.video_src["file_name"] = "(CAMERA) LIVE"
            # reset the lb1
            self.lb1_file_name.setText(self.video_src["file_name"])
        #*****************************************************************************
        # here is the same
        #*****************************************************************************
        # renew the shape, frames and so on.
        self.cap = cv.VideoCapture(self.video_src["path"])
        img_width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        img_height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        frames_num = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        video_fps = self.cap.get(cv.CAP_PROP_FPS)
        self.video_src["shape"] = (img_height, img_width)
        self.video_src["fps"] = video_fps
        self.video_src["frames"] = frames_num

    def selectVideoPath(self):
        select_path = QFileDialog.getOpenFileName(self, "Select source file", r"./testVideo")[0]
        self.video_src["path"] = select_path
        # remember to renew the information of the video_src
        self.get_video_src()

    def startVideo(self):
        # push the button, so the push num add 1
        # this just valid when the status not in the stop
        if self.push_bt_num["stop"] == 0:
            # add 1
            self.push_bt_num["start"] = (self.push_bt_num["start"] + 1) % 2
            if self.push_bt_num["start"] == 1:
                # the running status is the start
                self.runtime_status = "start"
                # set the status label
                self.lb2_status.setText("Running Status:  Running now...")
                # continue to run
                self.timer_camera.start(self.video_runtime_static["time_gap"])
                self.timer_camera.timeout.connect(self.img_deal)
            else:
                # the running status is the pause
                self.runtime_status = "pause"
                # set the status label
                self.lb2_status.setText("Running Status: Pause!")
                # let the timer to stop
                self.timer_camera.stop()
        else:
            # now the status is in the stop, so it is invalid to push the start and the pause
            # there are two situation:
            # 1) the cap is created(when open the gui), start the video
            if (self.cap.isOpened()):
                # if can open the cap, that start
                self.push_bt_num["start"] = (self.push_bt_num["start"] + 1) % 2
                self.lb2_status.setText("Running Status: Running now ...")
                # set the stop state to no, assign the it's value to 0
                self.push_bt_num["stop"] = 0
                self.timer_camera.start(self.video_runtime_static["time_gap"])
                self.timer_camera.timeout.connect(self.img_deal)
            # 2) the cap is released after push the stop button, so you should select the path to laod the video
            else:
                print("There is no video, please select the path!")
                self.lb2_status.setText("Fail to start, please load the video firstly!")

    def stopVideo(self):
        # push the button, so the push num add 1
        ## only when the push num == 0, and then add 1
        if self.push_bt_num["stop"] == 0:
            self.push_bt_num["stop"] += 1
            # and then stop the timer and release the cap
            self.timer_camera.stop()
            self.cap.release()
            # and then reset the lb_video
            self.lb2_video.clear()
            # set the status stop
            self.lb2_status.setText("Running Status: Stop!")
            # and reset the start_num
            self.push_bt_num["start"] = 0
            # init the video path, so you can start it next time
            self.get_video_src()

    def selectVideoOrCamera(self, state):
        if state == Qt.Checked:
           self.thres["flag_reverse"] = True
        else:
            self.thres["flag_reverse"] = False

    def bfVideo(self):
    # sub the frames
        self.video_runtime_cfg["frame_id"] -= self.video_runtime_static["c_step"]
        # delimit the frame_id
        if self.video_runtime_cfg["frame_id"] < 0:
            self.video_runtime_cfg["frame_id"] = 0
        # and then change the object cap:
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.video_runtime_cfg["frame_id"] )
        #self.lb2_status.setText("<- forward")

    def bnVideo(self):
        # add the frames
        self.video_runtime_cfg["frame_id"] += self.video_runtime_static["c_step"]
        # delimit the frame_id
        if self.video_runtime_cfg['frame_id'] >= self.video_src["frames"]:
            self.video_runtime_cfg["frame_id"] = self.video_src["frames"]
        # and then change the object cap:
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.video_runtime_cfg["frame_id"] )


    def img_deal(self):
        # read each image
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.timer_camera.stop()
                self.cap.release()
                # set the running status to the Finish!
                self.lb2_status.setText('Running Status: Finished!')
                # clear the video show label
                self.lb2_video.clear()
                # and set the stop state
                self.push_bt_num["stop"] = 1
                self.push_bt_num["start"] = 0
                # if you push stop or the video is running out, init the path, so you can read it next time
                self.get_video_src()
                return
            # *************************************************************
            # call the function to deal with every image
            frame_deal = self.img_num_recg(frame)

            # *************************************************************
            h, w, c = frame_deal.shape
            frame_deal = cv.cvtColor(frame_deal, cv.COLOR_BGR2RGB)
            bytesPerLine = c * w
            q_frame = QImage(frame_deal.data, w, h, bytesPerLine,
                            QImage.Format_RGB888).scaled(self.lb2_video.width(), self.lb2_video.height())
            self.lb2_video.setPixmap(QPixmap.fromImage(q_frame))

    def img_num_recg(self, image):
        if self.runtime_status == "Video":
            self.video_runtime_cfg["frame_id"] += 1
        #print(f"the runtime or the frame id is : {self.video_runtime_cfg['frame_id']}")
        # *************************************************************
        # get the video_src's attribute
        img_height, img_width = image.shape[:2]
        thres_binary = self.thres["binary"]
        thres_area_min = self.thres["area_min"]
        thres_area_max = self.thres["area_max"]
        thres_contour_length = self.thres["contour_length"]
        flag_reverse = self.thres["flag_reverse"]
        # *************************************************************
        img = copy.copy(image)
        # try to transpose the arry, exchange the widthe and the length
        if flag_reverse:
            img = np.transpose(img, (1, 0, 2))
            img = cv.flip(img, 0)
        # binary
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        _, img_binary = cv.threshold(img_gray, thres_binary, 1, cv.THRESH_BINARY_INV)
        _, img_binary_nr = cv.threshold(img_gray, thres_binary, 255, cv.THRESH_BINARY)
        # find the contours
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # filter the contour
        # set the initial digit's number that has been detected
        self.video_runtime_record["digit_nu"] = 0
        for j, contour in enumerate(contours):
            #print(f"contour's area is", cv.contourArea(contour))
            if thres_area_min < cv.contourArea(contour) < thres_area_max:
                x_min, x_max = np.min(contour[:, :, 0]), np.max(contour[:, :, 0])
                y_min, y_max = np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
                # filter the rect. 
                # the width and the length may be long enough
                rec_width = x_max - x_min
                rec_height = y_max - y_min
                if (rec_width < thres_contour_length) and (rec_height < thres_contour_length):
                    continue  # it means that don't include the rectangle
                # that means there has the digit, so count the digit's number
                self.video_runtime_record["digit_nu"] += 1
                self.changeDstInfo() # because the digit_nu is renew!
                # else, add the rect to the contours_rec list
                # draw the rectangle
                p1, p2 = (x_min, y_min), (x_max, y_max)
                cv.rectangle(img, p1, p2, (0, 0, 255), 2, 1)
                img_cut = img_binary_nr[y_min:y_max, x_min:x_max]
                # call the function of the disk
                weight = np.load(r"./module parameter/BPNN_Wrj.npy", allow_pickle=True)
                # get the feature
                feature_i = data_preprocess.get_feature(img_cut, 5, 0.1, 0, 1, cut_or_not=0)
                feature = np.array(feature_i)
                ### about the nn
                layers = [None] * len(weight)
                for i in range(len(weight)):
                    layers[i] = weight[i].shape[0]
                nn = NeuralNetwork(len(feature), layers)
                y_output = nn.feed_forward(feature, weight)[1][-1]
                ### get the pre_num
                pre_num = np.argmax(y_output)
                cv.putText(img, f"{pre_num}", (x_max, y_max), cv.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0))

        #img = cv.cvtColor(img_binary_nr, cv.COLOR_GRAY2BGR)
        #return image
        return img


if __name__ == "__main__":
    app = QApplication(sys.argv)
    my_gui = Gui()
    my_gui.show()
    app.exec_()
