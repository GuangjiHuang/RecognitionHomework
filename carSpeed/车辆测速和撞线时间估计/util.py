"""
@author Guangji Huang
Creat on Jun. 9, 2021
Note: define the class and the function
"""
import cv2 as cv
import json
import os
import numpy as np


class ShowImage:
    """
    the interaction with the mouse to set the position of the virtual coil
    """
    # class attribute
    object_num = 0 # count how many object has been created
    # init the object
    def __init__(self, img, name=None, tm=0, flag_win=0, flag_cb=0):
        """
        initialize the object
        :param img: the image that need to set the virtual coil
        :param name: the window's name
        :param tm: the wait time of the function waitKey
        :param flag_win: if true, show the window, otherwise, not show
        :param flag_cb: if true, use the mouse callback function
        """
        self.img = img
        if name == None:
            ShowImage.object_num += 1
            self.name = f"unnamed_{ShowImage.object_num}"
        else:
            self.name = name
        self.tm = tm
        self.flag_win = flag_win
        self.flag_cb = flag_cb
        self.point_list = [] # use the dic to store the points
        self.polygon_dict= {"num": 0}

    def __call__(self):
        # just show the image, call hte object as the function
        cv.imshow(self.name, self.img)
        cv.waitKey(self.tm)
        cv.destroyAllWindows()

    def onEventLButton(self, event, x, y, flags, param):
        # click the left button and draw the point in the image
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img, (x, y), 1, (255, 0, 0), -1)
            cv.putText(self.img, f"{x}, {y}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), thickness=1)
            self.point_list.append((x, y))
            cv.imshow(self.name, self.img)

        # click the right button to save the polygon to the file: virtual_coil_position.json
        if event == cv.EVENT_RBUTTONDOWN:
            if len(self.point_list) < 4:
                print(f"The polygon list now is {self.point_list}, too less points, don't write.")
                return
            # load the dict from the disk
            try:
                with open(r"virtual_coil_position.json", "r") as f:
                    polygon_dict = json.load(f)
            # if no  content
            except:
                polygon_dict = {"num": 0}
            # renew the points num
            polygon_dict["num"] += 1
            # add the key and the polygon
            key = f"polygon_{polygon_dict['num']}"
            value = self.point_list
            polygon_dict[key] = value
            print(f"the current_dict is {polygon_dict} ")
            # write the data to the disk]
            with open(r"virtual_coil_position.json", "w") as f:
                json.dump(polygon_dict, f)
            # renew the self.polygon_dict
            # check if the file exists
            if not os.path.exists(r"virtual_coil_position.json"):
                # touch it
                os.system("touch virtual_coil_position.json")
            self.point_list.clear()
            cv.imshow(self.name, self.img)

    def showImage(self):
        # the mouse event
        if self.flag_win:
            cv.namedWindow(self.name)
        if self.flag_cb:
            # in the show window, show set the call back function
            cv.setMouseCallback(self.name, self.onEventLButton)
        cv.imshow(self.name, self.img)
        cv.waitKey()
        cv.destroyAllWindows()

def subImg(src1, src2, mask, thres=12, rgb_num=3):
    """
    src1 - src2, and then count the result
    :param src1: source image 1
    :param src2: source image 2
    :param mask: mask
    :param thres: the threshold of the change, if larger than the threshold, the pixel will be set to the 1
    :param rgb_num: depend the number of the rgb to set the pixel to 1
    :return: change_rate of the src1 and the src2, and the sub_img_bin(this is the binary)
    """
    # the src1, src2, and the mask
    # convert the dtype
    if (src1.dtype != np.uint8) or (src2.dtype != np.uint8):
        src1 = src1.astype(np.uint8)
        src2 = src1.astype(np.uint8)
    # the mask must be the single 8 channel
    if mask.ndim >= 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    # creat the sub_img to store the sub result
    # the datatype of the sub_img is the np.int16 ( can be the -)
    sub_img = src1.astype(np.int16) - src2.astype(np.int16)
    sub_img = np.abs(sub_img).astype(np.uint8)
    sub_img = cv.bitwise_and(sub_img, sub_img, mask=mask)
    # get the sub result and then filter
    _, sub_img_filter = cv.threshold(sub_img, thres, 1, cv.THRESH_BINARY)
    # add the r, g, b
    sub_img_add = sub_img_filter[:, :, 0] + sub_img_filter[:, :, 1] + sub_img_filter[:, :, 2]
    _, sub_img_bin = cv.threshold(sub_img_add, rgb_num-1, 255, cv.THRESH_BINARY)
    # calculate the change rate
    change_rate = np.count_nonzero(sub_img_bin) / (np.count_nonzero(mask))
    return change_rate, sub_img_bin

def showImage(img, name="show", tm=None):
    """
    use this function to show the image
    :param img: the image that you want to show
    :param name: the window's name you want to show
    :param tm: the wait time, if None, don't wait
    :return:
    """
    cv.namedWindow(name)
    cv.imshow(name, img)
    if tm:
        cv.waitKey(tm)

def showText(img, state, frame_id, pos=None, pos_step=None):
    """
    What the text should be put in the image that you deal, oh, this function will solve your problem.
    it will show three kind of different text in your image just according to the state(for example: passing the line1
    , passing the line2, or show the frame_id and show the velocity and the time and so on.
    :param img: the image
    :param state: 3 type of state: 1) show nothing; 2) show line1 3) show line2
    :param frame_id: the frame id that you are dealing with
    :param pos: the position of the text you want to put into
    :param pos_step: the offset of the vertical orientation
    :return: None. But, it will change your image, it will put the text in your image
    """
    status = state[0]
    if pos == None:
       pos = [20, 20]
    if pos_step == None:
        pos_step = 30
    frame_id += 1 # correct the frame_id
    # return the text according to the status
    show_constant = f"frame_id: {frame_id}"
    if status == "show nothing":
        cv.putText(img, show_constant, tuple(pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), thickness=2)
    elif status == "show line1":
        car_id = state[1]
        frame_id_l1 = state[2]
        show_pass_line1 = f"Car_{car_id} is passing the virtual coil 1 in frame_id {frame_id_l1}"
        cv.putText(img, show_constant, tuple(pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), thickness=2)
        cv.putText(img, show_pass_line1, tuple([pos[0], pos[1]+1*pos_step]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), thickness=2)
    elif status == "show line2":
        # get the arguments
        car_id = state[1]
        frame_id_l2 = state[2]
        v = state[-2]
        tm = state[-1]
        # set the text
        show_pass_line2 = f"Car_{car_id} is passing the virtual coil 2 in frame_id {frame_id_l2}"
        show_v = f"Car_{car_id}'s speed is {v:.3f}km/h."
        show_tm = f"Car_{car_id} will hit the stop line in {tm:.3f}s."
        # put the text in the image
        cv.putText(img, show_constant, tuple(pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), thickness=2)
        cv.putText(img, show_pass_line2, tuple([pos[0], pos[1]+1*pos_step]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), thickness=2)
        cv.putText(img, show_v, tuple([pos[0], pos[1]+2*pos_step]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), thickness=2)
        cv.putText(img, show_tm, tuple([pos[0], pos[1]+3*pos_step]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), thickness=2)

def showVirtualCoilText(img, org_list, flag_show_stop_line=1, color="green", fontweight=2):
    """
    I think it is necessary to put the text to denote where the virtual coil and the stop line is, so adding this as the auxiliary
    :param img: the image you want to put the text in
    :param org_list: the position of them, and just as this: [p1, p2, p3]
    :param flag_show_stop_line: determine to show the stop line or not
    :param color: you can chose your own color
    :param fontweight: the scale of the font
    """
    if not (type(org_list) == list and len(org_list) == 3):
        print("your format is wrong!")
        return
    color_list = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255), "purple": (255, 0, 255), "cyan": (255, 255, 0)}
    # check your color if in the color_list
    if not (color in color_list.keys()):
        color = "red"
    name_list = ["virtual_coil_1", "virtual_coil_2", "stop line"]
    end = 0 if flag_show_stop_line else 1
    for i in range(len(name_list)-end):
        cv.putText(img, name_list[i], org_list[i], cv.FONT_HERSHEY_SIMPLEX, 0.5, color_list[color], fontweight)

def checkPreviousValue(input_list, num, thres):
    """
    check the list's the last n values if less then the threshold, not include the input_list[-1]
    :param input_list: the list object
    :param num: the number of the last n value
    :param thres: the threshold
    :return : bool
    """
    if len(input_list) - 1 < num:
        print("the length of the list is too short!")
        return False
    object_list = input_list[-(num+1):-1]
    bool_list = list(map(lambda x: x <= thres, object_list))
    return all(bool_list)