#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
import cv2
import datetime
import time
# cap = cv2.VideoCapture('rtsp://admin:admin12345@192.168.1.71:554/h264/ch1/main/av_stream')
cap = cv2.VideoCapture('rtsp://admin:hpws12345@192.168.1.38:554/h264/ch1/main/av_stream')
#rtsp://admin:hpws12345@192.168.1.38:554/h264/ch1/main/av_stream
index=0
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    if frame is None:
        break
    hight=frame.shape[0]
    width=frame.shape[1]
    # print 'width = %d   height = %d'%(width,hight)
    if width>1000:
        frame=cv2.resize(frame,(int(width/2),int(hight/2)),interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame)
    # print'width'
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
    # index+=1
    # if index%25==0:
    #     print 'detect time :'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # time.sleep(0.04)

cap.release()
cv2.destroyAllWindows()