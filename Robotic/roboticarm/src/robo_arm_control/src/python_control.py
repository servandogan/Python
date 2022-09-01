#!/usr/bin/env python3
from enum import Enum
from typing import Dict
from warnings import WarningMessage
import rospy
from rospy.timer import sleep
from std_msgs.msg import String
import math
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64

global callback_output
callback_output = None

global joint_data
joint_data = None

global mode
mode = 0

_1_DEGREE = 0.0174533 # 1 degree
MOVE_AMOUNT = 0.5 * _1_DEGREE
DELAY = 0.25

class Dir(Enum):
    POS = "POS"
    NEG = "NEG"
    RESET = "RESET"
    RESET1 = "RESET1"

def publish_movement(joint, direction: Dir):
    
    cont, poss = joint
    new_pos = 0
    if direction == Dir.POS:
        new_pos = poss + MOVE_AMOUNT
        cont.publish(new_pos)
    elif direction == Dir.NEG:
        new_pos = poss - MOVE_AMOUNT
        cont.publish(new_pos)
    elif direction == Dir.RESET:
        cont.publish(0)
    elif direction == Dir.RESET1:
        new_pos = 1.5707
        cont.publish(new_pos)
    else:
        raise Exception("Unknown direction") 
    return (cont,new_pos)
    
def publish_movement_fast(joint, direction: Dir):
    
    cont, poss = joint
    new_pos = 0
    if direction == Dir.POS:
        new_pos = poss + 10*MOVE_AMOUNT
        cont.publish(new_pos)
    elif direction == Dir.NEG:
        new_pos = poss - 10*MOVE_AMOUNT
        cont.publish(new_pos)
    elif direction == Dir.RESET:
        cont.publish(0)
    elif direction == Dir.RESET1:
        new_pos = 1.5707
        cont.publish(new_pos)
    else:
        raise Exception("Unknown direction") 
    return (cont,new_pos)

def close_claw(finger_control):
    if(finger_control[0][1] < -0.3):
            return
    finger_control[0] = publish_movement_fast(finger_control[0], Dir.NEG)
    finger_control[1] = publish_movement_fast(finger_control[1], Dir.POS)
    finger_control[2] = publish_movement_fast(finger_control[2], Dir.NEG)
    finger_control[3] = publish_movement_fast(finger_control[3], Dir.POS)
def open_claw(finger_control):  
    if(finger_control[0][1] > 3):
            return
    finger_control[0] = publish_movement_fast(finger_control[0], Dir.POS)
    finger_control[1] = publish_movement_fast(finger_control[1], Dir.NEG)
    finger_control[2] = publish_movement_fast(finger_control[2], Dir.POS)
    finger_control[3] = publish_movement_fast(finger_control[3], Dir.NEG)

def update_joints(data):
    global joint_data
    joint_data = data

def callback(data):
    np_arr = np.frombuffer(data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
    #gray_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    gray_np = image_np
        
    (h, w) = image_np.shape[:2]

    blurred = cv2.GaussianBlur(gray_np, (5, 5), 0)
    wide = cv2.Canny(blurred, 10, 250)

    _, binary = cv2.threshold(wide, 200, 255, 0)
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_list = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image_np,(x,y),(x+w,y+h),(200,0,0),2)
        temp =  x + w//2, y + h// 2, (w,h), image_np[y + h// 2,x + w//2]
        temp_list.append(temp)
    cv2.imshow("dd", image_np)
    cv2.waitKey(1)

    global callback_output
    if contours is not None:
        callback_output = temp_list
    else: 
        callback_output = None
        
def listener():
    rospy.init_node("listener")
   
    rospy.Subscriber("/robo_arm/camera1/image_raw/compressed", CompressedImage, callback, queue_size = 1)
    rospy.Subscriber("/robo_arm/joint_states", JointState, update_joints, queue_size = 10)
        
    rate = rospy.Rate(10) # 10hz
    global callback_output
        
    body_control = (rospy.Publisher("/robo_arm/joint_body_position_controller/command", Float64, queue_size = 1),0)
    shoulder_control = (rospy.Publisher("/robo_arm/joint_shoulder_position_controller/command", Float64, queue_size = 1),0)
    elbow_control = (rospy.Publisher("/robo_arm/joint_elbow_position_controller/command", Float64, queue_size = 1),0)
    finger_control = []
    finger_control.append((rospy.Publisher("/robo_arm/joint_finger1_position_controller/command", Float64, queue_size = 1),0))
    finger_control.append((rospy.Publisher("/robo_arm/joint_finger2_position_controller/command", Float64, queue_size = 1),0))
    finger_control.append((rospy.Publisher("/robo_arm/joint_finger3_position_controller/command", Float64, queue_size = 1),0))
    finger_control.append((rospy.Publisher("/robo_arm/joint_finger4_position_controller/command", Float64, queue_size = 1),0))
    shoulder_control = publish_movement(shoulder_control,Dir.RESET1)
    elbow_control = publish_movement(elbow_control,Dir.RESET1)
    
    while not rospy.is_shutdown():
        # to prevent overloading sleep for a small amount of time
        sleep(DELAY)
        
        #elbow_control = publish_movement(elbow_control,Dir.POS)
        #shoulder_control = publish_movement(shoulder_control,Dir.NEG)
        #body_control = publish_movement(body_control,Dir.POS)
        
        #Everything below here is main logic.
        global joint_data
        global mode
        
        if joint_data == None:
            continue
        
        if(mode == 0):
            open_claw(finger_control)
            shoulder_control = publish_movement(shoulder_control,Dir.RESET1)
            elbow_control = publish_movement(elbow_control,Dir.RESET1)
            body_control = publish_movement_fast(body_control, Dir.POS)
            
            if(callback_output != None and len(callback_output) > 0):
                mode = 1
                print("Found")
        if(mode == 1):
            w = 800
            if(len(callback_output) == 0):
                mode = 0
                continue
            
            x = callback_output[0][0]
            y = callback_output[0][1]
            
            x_true = False
            y_true = False
            
            
            if((x - w//2) <= -50):
                #print("left")
                body_control = publish_movement(body_control,Dir.POS)
                    
            elif((x - w//2) >= 50):
                #print("right")
                body_control = publish_movement(body_control,Dir.NEG)
                    
            else:
                #print("middle")
                x_true = True
                    
            if((y - w//2) <= -50):
                #print("down")
                elbow_control = publish_movement(elbow_control,Dir.NEG)
                    
            elif((y - w//2) >= 50):
                #print("up")
                elbow_control = publish_movement(elbow_control,Dir.POS)
            else:
                #print("middle")
                y_true = True
                    
            if(x_true and y_true):
                if(callback_output[0][2][0] > 300 and callback_output[0][2][1] > 300):
                    close_claw(finger_control)
                else:
                    shoulder_control = publish_movement(shoulder_control,Dir.POS)
                    elbow_control = publish_movement(elbow_control,Dir.NEG)

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
