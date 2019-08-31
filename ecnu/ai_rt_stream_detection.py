#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import Queue
import sys
import threading
import time
import traceback
import cv2

from infrared_pumps_calculate import *
from infrared_cps_calculate import *
from faster_rcnn_calculate import *
from common import my_logging

class ai_rt_stream_detection():

    #巡检区参数
    alarm_area_info = []
    # 巡检区抽油机参数
    pumps_info = []  # 巡检区参数
    #视频流路径
    read_frame_rate=25
    #calculate 休眠时间
    calculate_sleep_time=0.1
    #是否保存中间结果图片
    is_save_result_image=True
    #显示窗口状态
    disply_video_win=False

    def __init__(self,logger_name):
        #初始化日志模块
        self.log_name=logger_name
        self.logger=my_logging.get_logger(self.log_name)
        #
        self.is_alive_recognition_thread = False  # 识别线程状态
        # 处理线程队列
        self.light_queue = Queue.Queue()
        #线程句柄
        self.recognition_thread_id=None
        self.faster_rcnn_model = None
        self.video_stream_path = None
        #读取数据帧线程句柄
        self.read_frame_thread_id=None
        #帧率转sleep时间
        self.read_frame_time_interval=float(1)/self.read_frame_rate
        #实例化处理对象，可考虑需要时再做
        #  初始化faster_rcnn模型
        self.faster_rcnn_model=faster_rcnn_calculate(self.is_save_result_image,self.log_name)
        self.logger.info('ecnu 模块初始化成功')
        #打开一次视频流句柄
        self.cap=None
        #
        self.read_frame_thread_status=True
        #数据帧队列
        self.frame_queue = Queue.Queue()

    def __del__(self):
        self.read_frame_thread_status=False

    def stop_calculate_thread(self):
        # 1 若识别线程正在运行，先停止
        if self.is_alive_recognition_thread != False:
            self.is_alive_recognition_thread = False
            #
            self.recognition_thread_id.join()  # 等待识别线程退出
            #将最后一次结果输出
            self.faster_rcnn_model.stop_inspection_out_result()
            # print '2新的巡检区线程'
    # 切换巡检区
    def change_alarm_zone(self,alarm_area_para, pumps_para,rt_stream_path):

        self.alarm_area_info = alarm_area_para
        self.pumps_info = pumps_para
        #更新实时视频流路径
        self.video_stream_path = rt_stream_path
        # 如果读取视频线程没有运行,则创立并运行
        if self.read_frame_thread_id is None:
            self.read_frame_thread_id = threading.Thread(name="read_frame_thread",target=self.thread_read_frames)
            self.read_frame_thread_id.start()
        #
        # 3 在此更新模型参数
        self.faster_rcnn_model.update_alarm_area_para(self.alarm_area_info, self.pumps_info)
        #4 清空数据帧队列
        self.logger.info('下次巡检清空队列前,队列中的数据帧数:'+str(self.frame_queue.qsize()))
        self.frame_queue.queue.clear()
        #5 重新启动识别线程
        self.recognition_thread_id = threading.Thread(name="recognition_thread", target=self.start_recognition_thread)
        self.recognition_thread_id.setDaemon(True)
        self.is_alive_recognition_thread = True  # 改变线程状态为running
        self.recognition_thread_id.start()
        #
        self.logger.info('更新巡检区成功')

    def thread_read_frames(self):
        try:
            while(self.read_frame_thread_status):#读取线程一直保持运行
                #判断视频是否初始化
                if self.cap is None:
                    self.cap = cv2.VideoCapture(self.video_stream_path)
                    self.logger.info('第一次巡检打开!')
                else:
                    if self.cap.isOpened():#如果没有打开  #去掉了not判断，每次读取不到视频帧时，重新打开视频流
                        self.cap.release()#先释放,再打开
                        self.cap = cv2.VideoCapture(self.video_stream_path)
                        self.logger.info('其它次巡检,视频流关闭了,重新打开!')
                    else:
                        self.logger.info('其它次巡检,直接读取!')
                #如果处理线程在运行,则向队列插入数据帧
                self.logger.info("frame count = "+str(self.frame_queue.qsize()))
                while(True):
                    #如果读取线程要关闭
                    if not self.read_frame_thread_status:
                        break
                    #判断视频流是否准备好
                    if self.cap.isOpened():
                        ret, frame = self.cap.read()
                    else:
                        self.logger.info( "视频流处于关闭状态！")
                        break
                    # 判断是否还有数据帧
                    if not ret:
                        self.logger.info( "读取不到视频帧,待重新建立连接,再次读取！")
                        break
                    #将读取到的视频帧,存储到数据队列
                    if frame is not None:
                        self.frame_queue.put(frame)
                #休眠一秒
                time.sleep(1)
        except:
            self.logger.error('读取视频帧线程错误!')
        finally:
            self.logger.error('读取视频帧线程结束!')

    def start_recognition_thread(self):
        try:
            # self.cap = cv2.VideoCapture(self.video_stream_path)
            while self.is_alive_recognition_thread:#若视频读完，视频还未处理完，继续处理
                while self.is_alive_recognition_thread:
                    #从队列中读取数据帧
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get()
                        # 启动判断算法
                        try:
                            self.faster_rcnn_model.faster_rcnn_detection(frame)
                        except:
                            self.logger.error("start faster_rcnn thread failed, failed msg:" + traceback.format_exc())
                    else:
                        time.sleep(self.read_frame_time_interval)
                time.sleep(1)  # 此时视频已经读完，等待数据线程处理完数据自然结束
        except:
            self.logger.error( "视频流读取错误！")
            self.is_alive_recognition_thread = False
            self.logger.error("read stream exec failed, failed msg:" + traceback.format_exc())
        finally:
            self.logger.info('本次巡检结束!')
            # self.cap.release()
            pass