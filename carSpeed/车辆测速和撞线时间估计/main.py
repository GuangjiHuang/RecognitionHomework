"""
@author Guangji Huang
Created on Jun. 9, 2021
Note : the main function
"""
from util import *
from glob import glob
import copy


# ------ set the variable, the cfg ------
video_dir = r"video/" # the directory of the video
dis = {"l1_l2": 2.5/76*140, "l2_stop": 2.5/76*80} # the distant between the l1 and l2, l2 and stop line
change_rate_thres = 0.12

# open the video, and creat the video object
path_video_list = glob(f"{video_dir}/*.mp4")
path_video = path_video_list[-1]
cap = cv.VideoCapture(path_video)

# the attribute variable of the video object, use them to describe the process or record the information
frame_id = 0 # which frame you are dealing with
pt_list = []
car_flag = {"car_num": 0} # use this to judge if the car enter the line 1 and line2
car_info = {"car_num": 0} # all the car's information, including line1's id, line2's id, velocity, except hit stop line's time
change_rates = [] # record the zones(line1 and line2)'s rgb value change rates, for every frame
show_text_status = "show nothing" # the value determine what the text will be put in the image(see the funciont showText)

# loop, dealing with every frame, and show the dealing result
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        if frame_id == 0:
            print("Can not read!") # means that read nothing, check your video!
        else:
            print("--Finish---") # read out the video, means that finishing the dealing process
        break
    # in the first frame, 1) get the cap's attribute 2) initialize some variables
    if frame_id == 0:
        # get the information
        fps = cap.get(cv.CAP_PROP_FPS) # get the fps

        """
        load the position information of the virtual coil. There are two way to set the position
        1) you can set it manual in the file virtual_coil_position.json
        2) or you can use mouse to click the point and then save them to the file automatically through using the 
           ShowImage class, note that this will only trigger if the file virtual_coil_position.json not exists or it is empty
        What is more, polygon_dict is the dictionary that store the two virtual coil(the polygon) information
        """
        try:
            with open("virtual_coil_position.json", "r") as f:
                polygon_dict = json.load(f)
                print(f"load the data successfully, {polygon_dict}")
        except:
            # creat the ShowImage object, and then mark the virtual_coil_position using the mouse and save them
            init_pos = ShowImage(copy.copy(frame), name="set_pos", flag_win=1, flag_cb=1)
            init_pos.showImage() # save the virtual_coil_position to the file
            # read them
            with open("virtual_coil_position.json", "r") as f:
                polygon_dict = json.load(f)
                print(f"load the data successfully, {polygon_dict}")

        # initialize the variables, mainly the virtual coil's zone information, or we can call then the ROI
        frame_ref = frame # use the first frame as the reference frame or the background
        polygon_contours = [] # convert the format of the point to the OpenCV's contours
        mask_contours = [] # full fil the contours, as the mask
        for i in range(1, len(polygon_dict)):
            mask_bg = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8) # note that the dtype must be the uint8
            polygon_contour = np.array(polygon_dict[f"polygon_{i}"], np.int32)
            polygon_contour = polygon_contour.reshape((-1, 1, 2)) # cvt the format
            polygon_contours.append(polygon_contour)
            cv.drawContours(mask_bg, [polygon_contour], -1, (255, 255, 255), -1) # the thickness means that fullfill the contours
            mask_contours.append(mask_bg)
        # by the way initialize the org_list(where to show the virtual coil line and the stop line)
        basic_x = 636
        org_list = [(basic_x, 66), (basic_x, 216), (basic_x, 300)]
    sub_img_show = 0 # to store the sub result
    change_rate = [] # change_rate[0]: the line1's change rate, change_rate[1]: the line2's change rate
    for mask_contour in mask_contours:
        # call the subImg function, the frame - frame+_ref
        change_rate_i, sub_img_bin = subImg(frame, frame_ref, mask=mask_contour, thres=30, rgb_num=3)
        sub_img_show += sub_img_bin
        change_rate.append(change_rate_i)
    # add the change_rate to the change_rates
    change_rates.append(change_rate)
    # renew the background, if the change_rate less then the threshold, renew the frame_ref
    #    pass
    # draw the contours and show the result
    sub_img_show = cv.cvtColor(sub_img_show, cv.COLOR_GRAY2BGR)
    frame_show = copy.copy(frame)
    cv.drawContours(frame_show, polygon_contours, -1, (0, 0, 255), 1)
    cv.drawContours(sub_img_show, polygon_contours, -1, (0, 0, 255), 1)
    # and then put the text to the image
    text = f"frame_id: {frame_id+1}\n" \
           f"virtual coil 1 rgb change: {change_rate[0]:.3f}\n" \
           f"virtual coil 2 rgb change: {change_rate[1]:.3f}"
    org = [20, 20]
    text_list = text.split("\n")
    for i, text_i in enumerate(text_list):
        org = [20, 20 + 25*i]
        cv.putText(sub_img_show, text_i, tuple(org), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # see the line1's change
    if change_rate[0] > change_rate_thres:
        # if the last three frame's change_rate less than change_rate_thres, this is the new car that enter the line
        change_rates_l1 = [i[0] for i in change_rates]
        change_rate_pre_flag_l1 = checkPreviousValue(change_rates_l1, 3, change_rate_thres)
        if change_rate_pre_flag_l1:
            # renew the car's number in the car_flag and the car_info
            car_flag["car_num"] += 1
            car_info["car_num"] += 1
            flag = 0 # if the flag=0, means the car is passing the line1
            # there may be more than one car enter the line1, so set the flag to the corresponding car
            car_flag[f"car_{car_flag['car_num']}"] =  flag
            car_info[f"car_{car_info['car_num']}"] = [frame_id] # add the line's frame id to the car
            # set the show_status, if there is the car passing the line1, show the text that promote the information
            show_text_status = ["show line1", car_flag["car_num"], frame_id]
            # stop
            #cv.imshow("frame", frame_show)
            #cv.waitKey()
    # see the line2's change
    if change_rate[1] > change_rate_thres:
        # check if there are the car passing the line2
        change_rates_l2 = [i[0] for i in change_rates]
        change_rate_pre_flag_l2 = checkPreviousValue(change_rates_l2, 3, change_rate_thres)

        # if there is the car enter the line1
        if change_rate_pre_flag_l2 and (car_flag["car_num"] >= 1):
            for i in range(car_flag["car_num"]):  # if there are 3 car in the line1 and the line2, i= 0, 1, 2
                if car_flag[f"car_{i+1}"]  == 0:  # find the flag in order that 1, 2, 3...
                    # renew the car_flag and the car_info
                    car_flag[f"car_{i+1}"] = 1 # the flag=1, means that the car has been passed the line2
                    car_info[f"car_{i+1}"].append(frame_id) # add the frame_id to the car_info
                    # calculate the velocity
                    last_frame_id = car_info[f"car_{i+1}"][0]
                    car_v = 3.6 * dis["l1_l2"] / (frame_id - last_frame_id) * fps
                    time2stop = dis["l2_stop"] / car_v * 3.6
                    car_info[f"car_{i+1}"].append(car_v)
                    car_info[f"car_{i+1}"].append(time2stop)
                    # set the show_status
                    show_text_status = ["show line2", i+1, frame_id, car_v, time2stop]
                    #cv.imshow("frame", frame_show)
                    #cv.waitKey()
                    # and then jump out of the loop
                    break
    # put the text in the img_show to show the result
    showText(frame_show, show_text_status,frame_id)
    showVirtualCoilText(frame_show, org_list, 1, color="red")
    showVirtualCoilText(sub_img_show, org_list, 0, fontweight=1)
    # show the frame
    showImage(sub_img_show, "sub")
    showImage(frame_show, "frame")
    key_val = cv.waitKey(8) & 0xFF
    # set the key_val to control the video's play
    """
    the key's function:
    q: quit the video
    s: stop the video, and then you can press any key to continue
    h: backward the 24 frames of the video, and then you can press any key to continue
    l: forward the 24 frames of the video, and then you can press any key to continue
    """
    if key_val == ord('q'):
        break
    elif key_val == ord('s'):
        cv.waitKey()
    elif key_val == ord('h'):
        frame_id -= 24
        if frame_id < 0:
            frame_id = 0
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        cv.waitKey()
    elif key_val == ord('l'):
        frame_id += 24
        max_frame_id = cap.get(cv.CAP_PROP_FRAME_COUNT)
        if frame_id > max_frame_id:
            frame_id = max_frame_id
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        cv.waitKey()
    # renew the frame_id
    frame_id += 1
