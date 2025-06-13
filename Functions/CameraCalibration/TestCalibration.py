#!/usr/bin/env python3
# encoding:utf-8
import cv2
import time
import numpy as np
from CalibrationConfig import *
import hiwonder.Camera as Camera
import hiwonder.yaml_handle as yaml_handle

open_once = yaml_handle.get_yaml_data('/boot/camera_setting.yaml')['open_once']
if open_once:
    cap = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream?dummy=param.mjpg')
else:
    cap = Camera.Camera()
    cap.camera_open()  

#加载参数
param_data = np.load(calibration_param_path + '.npz')

#获取参数
mtx = param_data['mtx_array']
dist = param_data['dist_array']

while True:
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        break
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    ret, Frame = cap.read()
    if ret:
        frame = Frame.copy()
        
        dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        img = dst.copy()
        
        cv2.imshow('frame', frame)
        cv2.imshow('dst',dst)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        time.sleep(0.01)
cap.release()
cv2.destroyAllWindows()
