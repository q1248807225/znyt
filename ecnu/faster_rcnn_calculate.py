#!/usr/bin/env python
# coding=utf-8
import Queue

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, re
import argparse
from shape import rectangle
from shape import point
from dqsy.result import ResultHandler
import sys
from common import my_logging
import traceback
import copy
import datetime
import random

class faster_rcnn_calculate():
    #全局变量
    CLASSES = ('__background__','machine1_lvtou','machine1_pingheng','machine2_peizhong','car','person')
    NETS = {'vgg16': ('VGG16',
                      'vgg16_faster_rcnn_iter_1206.caffemodel'),
            'vgg1024': ('VGG_CNN_M_1024',
                        'vgg_cnn_m_1024_faster_rcnn_iter_1023.caffemodel'),
            'zf': ('ZF',
                   'ZF_faster_rcnn_final_1030.caffemodel')}
    #score置信度阈值
    CONF_THRESH = 0.65
    CONF_THRESH_INIT = 0.6
    CONF_THRESH_PERSON=0.7
    CONF_THRESH_CAR = 0.7
    CONF_THRESH_LVTOU = 0.6
    CONF_THRESH_PINGHENG = 0.55
    CONF_THRESH_PEIZHONG = 0.55
    #非极大值抑制处理阈值
    NMS_THRESH = 0.15
    #输出路径
    results_path='/media/anruida/F447DF5D9ED16F85/dq_test_output/'
    #计算帧数
    cal_frame_num=0
    #经过200帧，判别依次抽油机状态
    status_judge_count=200
    #结果输出帧数
    output_status_count=600
    #抽油机垂直运动阈值
    move_percent=0.2
    unmove_percent=0.2
    #保存视频包含的帧数
    save_video_frame_num=200
    #计算跳帧数
    faster_rcnn_jump_frame = 20
    #人车输出跳帧数
    cps_out_jump_cal=3
    #上一阶段结果集
    pumps_status = []
    #
    alarm_zone_change_times=0
    #图像匹配参数
    first_second_distance_rate=0.65
    #小图裁剪参数参数704*576
    small_crop_min_x = 10
    small_crop_min_y = 120
    small_crop_max_x = 700
    small_crop_max_y = 510
    #大图裁剪区域参数1920*1080
    big_crop_min_x = 200
    big_crop_min_y = 130
    big_crop_max_x = 1800
    big_crop_max_y = 850  #950
    #
    filter_car_size_max=2500
    filter_person_size_max=2500

    filter_car_size_min=2500
    filter_person_size_min=2500
    #当前计算帧的人车结果集合
    area_cps_type_count = []
    #第零部分start### ----------------------初始化操作---------------------###
    #类初始化
    def __init__(self,is_save_image,logger_name):
        #初始化日志模块
        self.log_name=logger_name
        self.logger=my_logging.get_logger(self.log_name)
        #是否保存识别结果图片到磁盘
        self.is_save_results=is_save_image
        self.init_caffe_model()
        self.result_handler = ResultHandler()
        #解决UnicodeDecodeError: 'ascii' codec can't decode byte 0xe6 in position 0 问题
        reload(sys)
        sys.setdefaultencoding('utf8')
        #屏幕输出结果
        self.dict_status_labels={'1':'Normal','0':'Abnormal','2':'Unknown','-1':'Nodetect'}
        #
        self.logger.info('faster rcnn init success!')
    #faster_rcnn参数
    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Faster R-CNN demo')
        parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',default=0, type=int)
        parser.add_argument('--cpu', dest='cpu_mode',help='Use CPU mode (overrides --gpu)',action='store_true')
        parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',choices=self.NETS.keys(), default='vgg16')#zf vgg16 vgg1024
        args = parser.parse_args()
        return args
    #初始化faster_rcnn模型
    def init_caffe_model(self):
        # 初始化faster_rcnn 模型
        cfg.TEST.HAS_RPN = True
        # Parse input arguments40
        # args = self.parse_args()
        class Args():
            def __init__(self):
                self.demo_net = 'vgg16'# zf vgg16 vgg1024
                self.cpu_mode = 0
                self.gpu_id = 0
        args = Args()
        #zf模型参数
        # prototxt = os.path.join(cfg.MODELS_DIR, self.NETS[args.demo_net][0], 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
        # caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', self.NETS[args.demo_net][1])

        #vgg16 vgg1024参数
        prototxt = os.path.join(cfg.MODELS_DIR, self.NETS[args.demo_net][0], 'faster_rcnn_end2end', 'test.prototxt')
        caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', self.NETS[args.demo_net][1])

        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                           'fetch_faster_rcnn_models.sh?').format(caffemodel))

        if args.cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
            cfg.GPU_ID = args.gpu_id
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        self.logger.info('\n\nLoaded network {:s}'.format(caffemodel))
        #print'\n\nLoaded network {:s}'.format(caffemodel)

        # Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _ = im_detect(self.net, im)
    #第零部分end### ----------------------初始化操作---------------------###
    #
    #
    #
    #第一部分start### ----------------------更新巡检区---------------------###
    #
    def update_alarm_area_para(self,area_para,pumps_para):
        self.logger.info('开始更新faster rcnn巡检区参数')
        #alarm_area_para=[ri_id,fct_id,ri_zone_id,leftupX,leftupY,rightdownX,rightdownY]
        #更新帧号
        self.cal_frame_num=0
        #背景帧
        self.gray_background = None
        #更新模型判断参数
        self.alarm_area_para_source=area_para
        self.alarm_area_para=copy.deepcopy(self.alarm_area_para_source)
        self.alarm_area_rect2 = rectangle(int(area_para[3]), int(area_para[4]),
                                              int(area_para[5]), int(area_para[6]))  # 抽油机矩形框集合
        #判断当前为红外还是可见光,F003为空外，F001为可见光
        self.fct_id=area_para[1]
        if self.fct_id=='F001':#可见光
            self.filter_car_size_min=6000
            self.filter_person_size_max=15000
            self.filter_person_size_min=2000
        else:
            self.filter_car_size_min=1000
            self.filter_person_size_max=2200
            self.filter_person_size_min = 150
        #判断用大图还是小图,1标识1920*1080  0表示704×576
        self.image_size_type=int(area_para[8])
        if self.image_size_type==0:
            self.crop_min_x = self.small_crop_min_x
            self.crop_min_y = self.small_crop_min_y
            self.crop_max_x = self.small_crop_max_x
            self.crop_max_y = self.small_crop_max_y
        elif self.image_size_type==1:
            self.crop_min_x = self.big_crop_min_x
            self.crop_min_y = self.big_crop_min_y
            self.crop_max_x = self.big_crop_max_x
            self.crop_max_y = self.big_crop_max_y
        else:
            self.logger.error('image_size_type error!')
        #
        # 先对left x从小到大排序，然后对right x排序
        pumps_para=sorted(pumps_para,key=lambda x:(x[2],x[4]))
        self.pumps_alarm_area_para_source = pumps_para  #
        self.pumps_alarm_area_para=copy.deepcopy(self.pumps_alarm_area_para_source)
        self.pumps_alarm_area_rects2 = []
        for one_pump in pumps_para:
            pump_rect = rectangle(int(one_pump[2]), int(one_pump[3]), int(one_pump[4]), int(one_pump[5]))
            self.pumps_alarm_area_rects2.append(pump_rect)  # pumps_alarm_area_rects saves all pump rectangles
        #输出图像前缀
        nowTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.alarm_zone_change_times+=1
        if self.is_save_results:
            self.out_images_path = self.alarm_area_para[0] + '_' + self.alarm_area_para[1] + '_' + self.alarm_area_para[
                2]+ '_' + nowTime+'_'+str(self.alarm_zone_change_times)
            self.full_path_images = self.results_path +self.out_images_path+'/'
            if not os.path.exists(self.full_path_images):
                os.makedirs(self.full_path_images)
        #存储所有驴头的框
        self.all_lvtou_rects=[]
        #存储所有平衡铁的框
        self.all_pingheng_rects=[]
        #清空缓存队列
        self.logger.info('更新faster rcnn巡检区参数完成')
    #第一部分end### ----------------------更新巡检区---------------------###
    #
    #
    #
    # 第二部分start### ----------------------行人检测---------------------###
    # 判断两个矩形是否相交
    def mat_inter(self,box1, box2,enlarge_rate):
        # box=(xA,yA,xB,yB)
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        #对于行人不同的速度可以设定宽度放大率,速度快扩大率大,速度慢扩大率小
        width0 = x02 - x01
        width1 = x12 - x11
        x01 -= width0*enlarge_rate
        x02 += width0*enlarge_rate
        x11 -= width1*enlarge_rate
        x12 += width1*enlarge_rate
        #
        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)
        sbx = abs(x11 - x12)
        say = abs(y01 - y02)
        sby = abs(y11 - y12)
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            return True
        else:
            return False
    #计算两个矩形框的重合度
    def solve_coincide(self,box1, box2):
        # box=(xA,yA,xB,yB)
        if self.mat_inter(box1, box2,0) == True:
            x01, y01, x02, y02 = box1
            x11, y11, x12, y12 = box2
            col = min(x02, x12) - max(x01, x11)
            row = min(y02, y12) - max(y01, y11)
            intersection = col * row
            area1 = (x02 - x01) * (y02 - y01)
            area2 = (x12 - x11) * (y12 - y11)
            coincide = float(intersection) / (area1 + area2 - intersection)
            return coincide
        else:
            return 0
    #计算两个矩形框的距离
    def dis_two_rect(self,box1, box2):
        # 判断两个矩形中心点的距离
        # box=(xA,yA,xB,yB)
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        center1_x = (x01 + x02) / 2
        center1_y = (y01 + y02) / 2
        center2_x = (x11 + x12) / 2
        center2_y = (y11 + y12) / 2
        # distance=np.sqrt(pow(center1_x-center2_x,2)+pow(center1_y-center2_y,2))
        x_distance = abs(center1_x - center2_x)
        y_distance = abs(center1_y-center2_y)
        return x_distance,y_distance
    #两个矩形框的宽度之比
    def w_ratio_two_rect(self,box1, box2):
        # 计算两个矩形宽度比
        # box=(xA,yA,xB,yB)
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        w_ratio = float(x02 - x01) / (x12 - x11)
        return w_ratio
    #将目标插入已有轨迹
    def insert_infrared_object_to_trajectory(self,crect):
        # 表示当前crect是否能够添加到当前的列表中，若不能，则添加新对象
        is_insert = False
        for cur_trajectory in self.trajectorys:
            last_rect = cur_trajectory[-1]
            if self.mat_inter(last_rect, crect,0):
                if cur_trajectory[0] or not cur_trajectory[1]:  # 如果本轮插入过，或者上一轮没有插入过，进行下次循环
                    continue  # 已经有出口
                wr = self.w_ratio_two_rect(last_rect, crect)
                # print 'wr = ' + str(wr)
                if wr < 0.82 or wr > 1.18:  # 假定，移动目标的下一个位置框，宽度不会变化太大，太大意味着合并了不同目标
                    continue
                rect_inter = self.solve_coincide(last_rect, crect)
                # print 'rect_inter' + str(rect_inter)
                if rect_inter < 0.2 or rect_inter > 0.95:  # 把前后两对象移动较远的过滤掉，过近的也过滤掉
                    continue
                cur_dis,y_dis = self.dis_two_rect(last_rect, crect)
                cur_trajectory[0] = True  ##若有数据插入，则置为True，表示本轮已经插入过数据，不能够再插入
                # cur_trajectory[1] = True#更新状态
                cur_trajectory[2] = cur_dis
                cur_trajectory[3] += cur_dis
                cur_trajectory.append(crect)
                is_insert = True
                break  #

        if not is_insert:  # 添加新对象
            one_trajectory = [True, False, 0, 0, crect]  # 本轮状态，上轮状态，本次距离，累积距离，矩形框
            self.trajectorys.append(one_trajectory)
    #红外行人探测
    def infrarec_person_detect(self,cur_frame):
        try:
            # 输出的行人矩形框
            person_rects = []
            # 膨胀kernel
            es = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5))  # .MORPH_ELLIPSE   MORPH_RECT  MORPH_CROSS
            #处理当前帧
            frame_lwpCV = cur_frame[self.crop_min_y:self.crop_max_y, self.crop_min_x:self.crop_max_x]
            # cv2.imwrite('/home/xu/mypython/test2000m/30.jpg',frame_lwpCV2)
            # 对帧进行预处理，先转灰度图，再进行高斯滤波。
            # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
            gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
            gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)
            # 将第一帧设置为整个输入的背景
            if self.gray_background is None:
                self.logger.error('背景帧为空')
                raise RuntimeError('背景帧为空!')
            # 对于每个从背景之后读取的帧都会计算其与北京之间的差异，并得到一个差分图（different map）。
            # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
            diff = cv2.absdiff(self.gray_background, gray_lwpCV)
            diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
            diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀
            # 显示矩形框
            image, contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
            # 添加新对象
            area_count = 0
            for c in contours:

                (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
                wh_rate = w / float(h + 0.01)
                wh_area = w * h
                # 针对找到的边界框,做过滤判断
                # print 'wh_area = '+str(wh_area)
                if wh_area < 30 or wh_area > 1500:  # 面积过大或者过小都不作为 行人 目标
                    continue
                if wh_rate < 0.3 or wh_rate > 1.0:  # 宽度与高度之比，在0.3和1.1范围内才可能认为是 行人
                    continue
                # 插入
                x += self.crop_min_x
                y += self.crop_min_y
                cur_rect = (x, y, x + w, y + h)
                copy_cur_rect = copy.deepcopy(cur_rect)
                self.insert_infrared_object_to_trajectory(copy_cur_rect)
            # 判别新对象
            for index, cur_trajectory in enumerate(self.trajectorys):
                if cur_trajectory[0]:  # 本次计算没有目标更新
                    # 首先更新上轮状态
                    cur_trajectory[1] = True  # 赋值上一轮的状态，表示是否插入过数据
                    #
                    # print 'move_distance = '+str(cur_trajectory[2])+'      '+str(cur_trajectory[3])
                    if cur_trajectory[2] > 3 or cur_trajectory[3] / float(
                            len(cur_trajectory) - 4) > 2:  # cur_trajectory[2]>3 or
                        # 显示序列中所有的框
                        (x1, y1, x2, y2) = cur_trajectory[-1]
                        cv2.rectangle(cur_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        person_rects.append(rectangle(x1, y1, x2, y2))  # 转化为shape rect
                else:
                    cur_trajectory[1] = False  # 把没有更新的目标置为false
                # 恢复为未插入状态
                cur_trajectory[0] = False
            # 删除不在可能是轨迹的记录
            self.trajectorys = filter(lambda x: x[1] == True, self.trajectorys)  # 筛选符合条件的记录，不符合条件的删除
        except:
            self.logger.error('计算红外行人出错'+traceback.format_exc())
        finally:
            return person_rects
    #添加可见光行人轨迹
    def insert_light_object_to_trajectory(self,crect):
        # 表示当前crect是否能够添加到当前的列表中，若不能，则添加新对象
        is_insert = False
        for cur_trajectory in self.trajectorys_light:
            last_rect = cur_trajectory[-1]
            if self.mat_inter(last_rect, crect,0.6):
                if cur_trajectory[0]:  # or not cur_trajectory[1]:  # 如果本轮插入过，或者上一轮没有插入过，进行下次循环
                    continue  # 已经有出口
                cur_dis,y_dis = self.dis_two_rect(last_rect, crect)
                cur_trajectory[0] = True  ##若有数据插入，则置为True，表示本轮已经插入过数据，不能够再插入
                cur_trajectory[2] = cur_dis
                cur_trajectory[3] += cur_dis
                cur_trajectory.append(crect)
                is_insert = True
                break  #
        if not is_insert:  # 添加新对象
            one_trajectory = [True, False, 0, 0, crect]  # 本轮状态，上轮状态，本次距离，累积距离，矩形框
            self.trajectorys_light.append(one_trajectory)
    #添加可见光行人判断
    def lightrec_person_detect(self,cur_frame):
        try:
            # 输出的行人矩形框
            person_rects = []
            # 判别新对象
            for index, cur_trajectory in enumerate(self.trajectorys_light):#遍历每一条轨迹
                if cur_trajectory[0]:  # 本次计算没有目标更新
                    # 首先更新上轮状态
                    cur_trajectory[1] = True  # 赋值上一轮的状态，表示是否插入过数据
                    #判断水平或者垂直位置变化
                    sum_widht=0
                    sum_height=0
                    avg_width=0
                    avg_heigth=0
                    min_center_x=2000#首先赋值一个大值
                    max_center_x=0
                    cur_tra_count=0
                    for prect in cur_trajectory[4:]:
                        (x1, y1, x2, y2) = prect
                        sum_widht += (x2 - x1)
                        sum_height += (y2 - y1)
                        center_x=(x1+x2)/2
                        center_y=(y1+y2)/2
                        cur_tra_count+=1
                        if center_x<min_center_x:
                            min_center_x=center_x
                        if center_x>max_center_x:
                            max_center_x=center_x
                    if cur_tra_count>0:
                        avg_width=sum_widht/cur_tra_count
                        avg_heigth=sum_height/cur_tra_count
                    #结果判断
                    print 'max_center_x-min_center_x = ' + str(max_center_x-min_center_x) + '   avg_width = ' + str(avg_width)
                    if max_center_x-min_center_x>avg_width*1:#符合条件输出
                        (x1, y1, x2, y2) = cur_trajectory[-1]
                        cv2.rectangle(cur_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        person_rects.append(rectangle(x1, y1, x2, y2))  # 转化为shape rect
                else:
                    cur_trajectory[1] = False  # 把没有更新的目标置为false
                # 恢复为未插入状态
                cur_trajectory[0] = False
            # 删除不在可能是轨迹的记录
            self.trajectorys_light = filter(lambda x: x[1] == True, self.trajectorys_light)  # 筛选符合条件的记录，不符合条件的删除
        except:
            print '计算可见光行人出错' + traceback.format_exc()
        finally:
            return person_rects


    # 第二部分end### ----------------------行人检测---------------------###
    #
    #
    #
    #第三部分start### ----------------------图片偏离纠正---------------------###
    # 霍夫变化，用于基于图像的匹配
    def homo_graphy(self,kp1,kp2,good):
        if len(good) >= 5:
            src_points = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_points = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)  # .RANSAC  LMEDS
            good_refine = []
            if mask is None:
                return good
            else:
                matchesMask = mask.ravel().tolist()
                for index in range(len(matchesMask)):
                    if matchesMask[index]:
                        good_refine.append(good[index])
                return good_refine
        else:
            return good
    # 基于图像匹配的纠偏计算
    def calculate_offset(self,new_frame):
        try:
            avg_x=0
            avg_y=0
            #标定图片报警区边框
            alarm_rect=self.alarm_area_para[3:7]
            # 获取标定图片路径
            self.alarm_area_image_path = self.alarm_area_para[7]  # 标定时的图片路径
            cali_frame=cv2.imread(self.alarm_area_image_path)
            if cali_frame is None:
                raise RuntimeError('标定图片为空!')
            #标定图片报警区截图
            rows1 = cali_frame.shape[0]#高
            cols1 = cali_frame.shape[1]#宽
            rows2 = new_frame.shape[0]
            cols2 = new_frame.shape[1]
            # 过滤视频中的时间和类型标注
            # crop_cali_frame = cali_frame[alarm_rect[1]:alarm_rect[3],alarm_rect[0]:alarm_rect[2]]
            crop_cali_frame = cali_frame[self.crop_min_y:self.crop_max_y, self.crop_min_x:self.crop_max_x]
            #匹配计算
            sift = cv2.xfeatures2d.SIFT_create()#.SIFT_create()#
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(crop_cali_frame, None)
            #copy frame
            copy_frame=new_frame.copy()
            kp2, des2 = sift.detectAndCompute(copy_frame, None)
            # self.logger.info('detect time :'+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # self.logger.info('matche time :'+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            # Apply ratio test
            sorted_matches=sorted(matches,key=lambda x:(x[0].distance/x[1].distance))
            good_matches = [[m] for m, n in sorted_matches if m.distance < self.first_second_distance_rate * n.distance]
            if len(good_matches)==0:#设置小一点的比值
                good_matches = [[m] for m, n in matches if m.distance < (self.first_second_distance_rate+0.05) * n.distance]
            #霍夫变换
            good_matches_refine=self.homo_graphy(kp1,kp2,good_matches)
            # self.logger.info('homo time :'+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            #合并图片
            out_image = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
            # Place the first image to the left
            out_image[:rows1, :cols1] = cali_frame
            # Place the next image to the right of it
            out_image[:rows2, cols1:] = copy_frame

            #计算总偏移
            acc_x=0
            acc_y=0
            index=0
            for one_match in good_matches_refine:
                img1_idx = one_match[0].queryIdx
                img2_idx = one_match[0].trainIdx
                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt
                offsetx=x2-(x1+self.crop_min_x)
                offsety=y2-(y1+self.crop_min_y)
                # print 'x1= ' + str(x1+self.crop_min_x) + ' y1=' + str(y1+self.crop_min_y) + '           x2=' + str(x2) + ' y2=' + str(y2)
                acc_x+=offsetx
                acc_y+=offsety
                #绘制匹配点
                a = np.random.randint(0, 256)
                b = np.random.randint(0, 256)
                c = np.random.randint(0, 256)
                real_cali_x=int(np.round(x1+self.crop_min_x))
                real_cali_y=int(np.round(y1+self.crop_min_y))
                cv2.circle(out_image, (real_cali_x,real_cali_y), 10, (a, b, c), 4)
                cv2.circle(out_image, (int(np.round(x2))+cols1, int(np.round(y2))), 10, (a, b, c), 4)
                # 将匹配个数输出
                cv2.line(out_image, (int(np.round(real_cali_x)), int(np.round(real_cali_y))), (int(np.round(x2) + cols1), int(np.round(y2))),
                         (a, b, c), 2)  # 画线，cv2.line()参考官方文档
            #求平均偏移
            matches_count = len(good_matches_refine)
            if not True:#计算平均偏移
                if matches_count>0:
                    avg_x=int(acc_x/matches_count)
                    avg_y=int(acc_y/matches_count)
            else:#只计算第一个值的偏移
                if matches_count>0:
                    img1_idx = good_matches_refine[0][0].queryIdx
                    img2_idx = good_matches_refine[0][0].trainIdx
                    (x1, y1) = kp1[img1_idx].pt
                    (x2, y2) = kp2[img2_idx].pt
                    avg_x = x2 - (x1 + self.crop_min_x)
                    avg_y = y2 - (y1 + self.crop_min_y)
            #保存标定图片到磁盘
            cv2.imwrite(self.full_path_images+'0cali_image.jpg',cali_frame)
            cv2.imwrite(self.full_path_images+'0copy_image.jpg',copy_frame)
            cv2.imwrite(self.full_path_images + '0cali_copy_image.jpg', out_image)

        except:
            self.logger.info('计算偏移失败：'+traceback.format_exc())
        finally:
            return avg_x,avg_y
    # 基于拓扑的纠偏计算
    def topo_cal_offset(self, filter_candidate_results, avg_width, cali_pump_centers):
        #
        offset_x = 0
        offset_y = 0
        cali_pumps_count = len(cali_pump_centers)
        filter_pums_count = len(filter_candidate_results)
        if cali_pumps_count == 0 or filter_pums_count == 0:
            return offset_x, offset_y
        #
        cpc_left_center = cali_pump_centers[0]  # 最左边标定驴头中心点
        fcr_left_center = filter_candidate_results[0][1]  # 最左边识别驴头中心点
        #
        cpc_right_center = cali_pump_centers[cali_pumps_count - 1]  # 最右边标定驴头中心点
        fcr_right_center = filter_candidate_results[filter_pums_count - 1][1]  # 最右边识别驴头中心点
        #
        cali_width = cpc_right_center._x - cpc_left_center._x
        #
        if filter_pums_count <= cali_pumps_count:
            offset_x = fcr_left_center._x - cpc_left_center._x
            offset_y = fcr_left_center._y - cpc_left_center._y
        else:
            filter_width = fcr_right_center._x - fcr_left_center._x
            if cali_width > filter_width - avg_width:  # 如果标定框的总宽度大于计算驴头框的总宽度
                offset_x = fcr_left_center._x - cpc_left_center._x
                offset_y = fcr_left_center._y - cpc_left_center._y
            else:  # 宽度不一致的情况,比标定的要宽
                is_find = False
                for one_candidate in filter_candidate_results:
                    if one_candidate[0] >1:  # 抛除异常点，特别是lvtou比较少的点
                        one_candidate_center = one_candidate[1]
                        filter_width = fcr_right_center._x - one_candidate_center._x
                        if cali_width > filter_width - avg_width:  # 如果宽度都在范围内
                            offset_x = one_candidate_center._x - cpc_left_center._x
                            offset_y = one_candidate_center._y - cpc_left_center._y
                            is_find = True
                            break
                if not is_find:  # 若没有匹配找到，直接用第一个匹配
                    offset_x = fcr_left_center._x - cpc_left_center._x
                    offset_y = fcr_left_center._y - cpc_left_center._y
        return offset_x, offset_y
    #从左向右匹配
    def left_right_offset(self,filter_candidate_results,cali_pump_centers):
        cali_pumps_count = len(cali_pump_centers)
        filter_pums_count = len(filter_candidate_results)
        match_count=min(cali_pumps_count,filter_pums_count)
        alarm_offset_x=0
        alarm_offset_y=0
        for index in range(match_count):
            #聚类中心点
            one_cluster_center=filter_candidate_results[index][1]  # 识别到的驴头框的中心点
            #标定框的中心点
            one_cali_center=cali_pump_centers[index]
            #计算偏移
            cur_offset_x=one_cluster_center._x-one_cali_center._x
            cur_offset_y = one_cluster_center._y - one_cali_center._y
            if index==0:
                alarm_offset_x = cur_offset_x
                alarm_offset_y = cur_offset_y
            #修正当前的标定驴头框
            cur_cali_rect=self.pumps_alarm_area_para[index]
            cur_cali_rect[2]+=cur_offset_x
            cur_cali_rect[4]+=cur_offset_x
            cur_cali_rect[3]+=cur_offset_y
            cur_cali_rect[5]+=cur_offset_y
            #
        #报警区矩形框
        self.alarm_area_para[3] += alarm_offset_x
        self.alarm_area_para[5] += alarm_offset_x
        self.alarm_area_para[4] += alarm_offset_y
        self.alarm_area_para[6] += alarm_offset_y
        self.alarm_area_rect = rectangle(int(self.alarm_area_para[3]), int(self.alarm_area_para[4]),
                                         int(self.alarm_area_para[5]), int(self.alarm_area_para[6]))
        # 抽油机矩形框集合
        self.pumps_alarm_area_rects = []
        for pump_para in self.pumps_alarm_area_para:
            pump_rect = rectangle(int(pump_para[2]), int(pump_para[3]), int(pump_para[4]), int(pump_para[5]))
            self.pumps_alarm_area_rects.append(pump_rect)  # pumps_alarm_area_rects saves all pump rectangles
        #计算驴头的相对位置关系
        self.cali_pumps_spatial_relation()

    #计算两个抽油机标定框的位置关系
    def two_rect_relation(self,rect1,rect2):
        #
        center_left=rect1.getCenter()
        center_right=rect2.getCenter()
        if rect1.getxmax() <= rect2.getxmin():  # x不相交
            return False
        else:
            if abs(center_left._y - center_right._y) > rect1.getHeight() / 4+rect2.getHeight() / 4:  # y方向相交小于rect1+rect2高度的1/4
                return False
            else:
                return True
    #修正标定框的像素偏移
    def modify_rects_offset(self,offsetx,offsety):
        self.logger.info('offsetx = '+str(offsetx)+'      offsety = '+str(offsety))
        # 修正报警区矩形框坐标
        self.alarm_area_para[3] += offsetx
        self.alarm_area_para[5] += offsetx
        self.alarm_area_para[4] += offsety
        self.alarm_area_para[6] += offsety
        self.alarm_area_rect = rectangle(int(self.alarm_area_para[3]), int(self.alarm_area_para[4]),
                                         int(self.alarm_area_para[5]), int(self.alarm_area_para[6]))  # 抽油机矩形框集合
        #修正抽油机矩形框
        for one_pump in self.pumps_alarm_area_para:
            one_pump[2] += offsetx
            one_pump[4] += offsetx
            one_pump[3] += offsety
            one_pump[5] += offsety
        #排序
        # self.pumps_alarm_area_para=sorted(self.pumps_alarm_area_para,key=lambda x:(x[2],x[4]))
        self.pumps_alarm_area_rects = []
        for pump_para in self.pumps_alarm_area_para:
            pump_rect = rectangle(int(pump_para[2]), int(pump_para[3]), int(pump_para[4]), int(pump_para[5]))
            self.pumps_alarm_area_rects.append(pump_rect)  # pumps_alarm_area_rects saves all pump rectangles
        #计算驴头的相对位置关系
        self.cali_pumps_spatial_relation()
        #
    # 对于两个以上抽油机,计算相对位置关系
    def cali_pumps_spatial_relation(self):
        # 初始化相对关系数组为0
        self.pumps_overlay_relt = [0 for i in range(len(self.pumps_alarm_area_para))]  # 0左右都不交叉，1左边交叉，2右边交叉，3左右都有交叉
        if len(self.pumps_alarm_area_rects) >= 2:
            for one_index in range(len(self.pumps_alarm_area_rects) - 1):  # 右交叉
                pump_type = self.pumps_alarm_area_para[one_index][1]
                if pump_type == 1 or pump_type==5:  # 塔架抽油机或者本轮判断抽油机
                    self.pumps_overlay_relt[one_index] = 0
                    self.pumps_overlay_relt[one_index + 1] = 0
                elif pump_type == 0 or pump_type==2:#油梁抽油机和蜗牛抽油机
                    if self.two_rect_relation(self.pumps_alarm_area_rects[one_index],
                                              self.pumps_alarm_area_rects[one_index + 1]):
                        if self.pumps_overlay_relt[one_index] != 0:
                            self.pumps_overlay_relt[one_index] = 3  # 左右都交叉
                        else:
                            self.pumps_overlay_relt[one_index] = 2  # 仅仅右交叉
                        self.pumps_overlay_relt[one_index + 1] = 1
                    else:
                        self.pumps_overlay_relt[one_index + 1] = 0
                else:
                    self.logger.info('抽油机类型错误！')
        #抽油机机驴头中间结果集合
        self.lvtou_rects = []
        for k in range(len(self.pumps_alarm_area_rects)):
            lvtou_rect = []
            self.lvtou_rects.append(lvtou_rect)
    #第三部分end### ----------------------图片偏离纠正---------------------###
    #
    #
    #
    #第四部分start### ----------------------faster_rcnn检测---------------------###
    #计算点与矩形框的关系
    def pcontainbyrect(self,pt,rect):
        x01 = rect._xmin
        y01 = rect._ymin
        x02 = rect._xmax
        y02 = rect._ymax

        px=pt._x
        py=pt._y

        if px<x02 and px>x01 and py<y02 and py>y01:
            return True
        else:
            return False
    #计算两个矩形框是否相交
    def overlap(self,rect1, rect2):
        x01 = rect1._xmin
        y01 = rect1._ymin
        x02 = rect1._xmax
        y02 = rect1._ymax

        x11 = rect2._xmin
        y11 = rect2._ymin
        x12 = rect2._xmax
        y12 = rect2._ymax

        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)
        sbx = abs(x11 - x12)
        say = abs(y01 - y02)
        sby = abs(y11 - y12)
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            return True
        else:
            return False
    #检测对象类型的判断
    def object_classes_process(self,frame, inds, class_name, dets):
        """Draw detected bounding boxes."""
        object_type=class_name
        object_count=0
        #循环有效记录的序列号
        for i in inds:  # score满足条件的记录
            bbox = dets[i, :4]   #获取边界框
            score = round(dets[i, -1], 2)  # 获取置信得分列
            #依次判断对象类别
            if class_name == 'machine1_lvtou':
                lvtouRect = rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                #添加所有驴头的框到
                self.all_lvtou_rects.append(lvtouRect)
                #
                flag = False
                for k,area in enumerate(self.pumps_alarm_area_rects):  # 按照抽油机位置分配识别框
                    if self.pumps_alarm_area_para[k][1] == 1:  # 塔架抽油机不包含驴头
                        continue
                    #print 'pumps id = %s'%(self.pumps_alarm_area_para[k][0])
                    if self.pcontainbyrect(lvtouRect.getCenter(), area):
                        flag = True #
                        self.lvtou_rects[k].append(lvtouRect)#把lv_tou便捷框添加到对应抽油机的列表中
                        break#对于处于重叠交叉的对象，仅将其放入左侧即可
                if flag == True:#lv_tou在抽油机框范围内
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    cv2.putText(frame, 'lv_tou', (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 0, 255), 2)
            elif class_name=='machine1_pingheng':
                pinghengRect = rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                #添加所有平衡框到集合
                self.all_pingheng_rects.append(pinghengRect)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, 'ph', (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (255, 0, 0), 2)
                #
                pass
            elif class_name == 'machine2_peizhong':
                peizhongRect = rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                flag = False
                for pindex, area in enumerate(self.pumps_alarm_area_rects):
                    if self.pumps_alarm_area_para[pindex][1]==1:#塔架
                        if flag:
                            self.logger.info("塔架抽油机也有重合？")
                        if self.pcontainbyrect(peizhongRect.getCenter(), area):
                            flag = True  #
                            self.lvtou_rects[pindex].append(peizhongRect)  # 把lv_tou便捷框添加到对应抽油机的列表中
                if flag == True:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
                    cv2.putText(frame, 'pz', (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (255, 0, 0), 2)

            elif class_name == 'person':
                personRect = rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                personRectCenter=personRect.getCenter()
                #面积滤除
                personRectArea=personRect.getArea()
                if personRectArea>self.filter_person_size_max or personRectArea<self.filter_person_size_min:
                    self.logger.error('检测到的人面积太大或者太小!')
                    continue
                #宽高比滤除
                wh_rate=float(personRect.getWidth())/personRect.getHeight()
                if wh_rate<0.2 or wh_rate>0.7:
                    print self.logger.error('检测到人宽高比不合适！')
                    continue
                flag = False
                #中心点被平衡铁包含滤除
                is_under_pingheng=False
                for one_rect in self.all_pingheng_rects:
                    # is_contained=self.pcontainbyrect(personRectCenter,one_rect)
                    is_overlay=self.overlap(personRect,one_rect)
                    if is_overlay:
                        self.logger.info('行人在平衡铁附近滤除')
                        is_under_pingheng = True
                        break
                if is_under_pingheng:#如果行人在平衡铁下,不输出
                    continue
                # #过滤井口的人
                # is_under_pumps=False
                # personRectCenter_x=personRectCenter._x
                # for one_lvtou_rect in self.pumps_alarm_area_rects:
                #     x_left=one_lvtou_rect.getxmin()
                #     x_right=one_lvtou_rect.getxmax()
                #     print 'personRectCenter_x = '+str(personRectCenter_x)+'     x_left = '+str(x_left)+'    x_right = '+str(x_right)
                #     if personRectCenter_x>x_left- personRect.getWidth() and personRectCenter_x<x_right+personRect.getWidth():
                #         self.logger.info('有人在井口的位置!')
                #         is_under_pumps = True
                #         break
                # if is_under_pumps:#如果井口有人,不输出
                #     continue
                #
                if self.overlap(self.alarm_area_rect, personRect):
                    #判断人中心点是否在报警区框的上方(上边缘),这种情况不考虑
                    if personRectCenter._y<(self.alarm_area_rect.getymin()+self.alarm_area_rect.getHeight()/8):
                        continue

                    object_count+=1#增加对象数量
                    # self.logger.info('Attention! There are people in the dangerous area !')
                    flag=True
                    # 将新的检测人矩形加入轨迹,只添加在报警区的对象
                    cur_rect = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    copy_cur_rect = copy.deepcopy(cur_rect)
                    self.insert_light_object_to_trajectory(copy_cur_rect)
                #
                # if flag:
                #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 153), 2)
                #     cv2.putText(frame, str(score)+'_'+class_name, (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                #                 (255, 0, 153), 2)

            elif class_name == 'car':
                carRect = rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                carRectCenter=carRect.getCenter()
                #面积滤除
                carRectArea=carRect.getArea()
                if carRectArea<self.filter_car_size_min:
                    self.logger.error('检测到的车辆面积不足!')
                    continue
                #宽高比滤除
                wh_rate=float(carRect.getWidth())/carRect.getHeight()
                if wh_rate<0.8 or wh_rate>4:
                    self.logger.error('检测到车辆宽高比不合适！')
                    continue
                #中心点被平衡铁包含滤除
                is_under_pingheng = False
                for one_rect in self.all_pingheng_rects:#contain 变为  overlay
                    is_contained=self.pcontainbyrect(carRectCenter,one_rect)
                    # is_overlay=self.overlap(carRect,one_rect)
                    if is_contained:#is_overlay:
                        self.logger.error('平衡铁与检测车辆重叠,要滤除')
                        is_under_pingheng = True
                        break
                if is_under_pingheng:#如果车在平衡铁下,不输出
                    continue
                #
                #过滤井口的比较小的车
                is_under_pumps=False
                carRectCenter_x=carRectCenter._x
                for one_lvtou_rect in self.pumps_alarm_area_rects:
                    x_left=one_lvtou_rect.getxmin()
                    x_right=one_lvtou_rect.getxmax()
                    #过滤井口
                    if carRectCenter_x>x_left- carRect.getWidth()/2 and carRectCenter_x<x_right+carRect.getWidth()/2:
                        if carRect.getWidth()<2*one_lvtou_rect.getWidth():#车辆宽度要大于标定驴头框宽度的2倍数
                            self.logger.info('有车在井口的位置,且宽度不足!')
                            is_under_pumps = True
                        break
                    #过滤过小
                    if carRect.getWidth()<one_lvtou_rect.getWidth()*1.5:
                        self.logger.info('车辆宽度不足1.5个驴头!')
                        is_under_pumps = True
                        break
                    #
                if is_under_pumps:
                    continue

                flag=False
                if self.overlap(self.alarm_area_rect, carRect):
                    #判断车辆中心点是否在报警区框的上方(上边缘),这种情况不考虑
                    if carRectCenter._y<(self.alarm_area_rect.getymin()+self.alarm_area_rect.getHeight()/8):
                        continue
                    #
                    object_count+=1#增加对象数量
                    self.logger.info('Attention! There are cars in the dangerous area !')
                    flag=True
                if flag:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2)
                    cv2.putText(frame, str(score)+'_'+class_name, (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)
            else:
                self.logger.info("未知的对象类型，可能出现错误！")

        return object_type,object_count
    #基于faster_rcnn的计算
    def faster_rcnn_detection(self,frame):  # 巡检区框，抽油机框，驴头框
        #多线程添加GPU模式
        caffe.set_mode_gpu()
        #针对第一帧的计算，纠偏和作为背景
        if self.cal_frame_num==0:
            # 用图像匹配方法计算偏离并修正
            # self.logger.info('start time :'+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            offsetx, offsety = self.calculate_offset(frame)
            self.modify_rects_offset(offsetx, offsety)
            # self.logger.info('stop time :'+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            # 第一帧图片拷贝，用于红外行人识别
            crop_first_frame = frame[self.crop_min_y:self.crop_max_y, self.crop_min_x:self.crop_max_x]
            gray_lwpCV = cv2.cvtColor(crop_first_frame, cv2.COLOR_BGR2GRAY)

            self.gray_background = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)
            #存储目标轨迹
            self.trajectorys = []
            #存储可见光目标轨迹
            self.trajectorys_light=[]
            self.person_max_count=0
            #初始化人车输出结果
            self.area_cps_type_count=[]
            person_result=[self.alarm_area_para[0],self.alarm_area_para[2],'person',0]
            car_result=[self.alarm_area_para[0],self.alarm_area_para[2],'car',0]
            self.area_cps_type_count.append(person_result)
            self.area_cps_type_count.append(car_result)

        #跳帧处理
        if self.cal_frame_num%self.faster_rcnn_jump_frame!=0:#20帧计算一次
            self.cal_frame_num += 1  # 增加有效计算帧
            return
        #针对帧开始计算
        scores, boxes = im_detect(self.net, frame)
        self.logger.info('识别当前帧 ： '+str(self.cal_frame_num))
        # #检测到当前计算帧的人车集合
        # self.area_cps_type_count=[]
        #标识本次检测，是否发现人了人
        exist_person=False
        exist_car=False
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]  # 当前帧所有驴头
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)#合并cls_boxes和cls_scores
            keep = nms(dets, self.NMS_THRESH)  # 非极大值抑制，去掉重复的位置框
            dets = dets[keep, :]
            #针对不同的目标类型和光源类型,设置不同的置信度阈值
            if self.fct_id=='F003':#若为红外光
                if cls=='machine1_lvtou':
                    self.CONF_THRESH=self.CONF_THRESH_LVTOU #0.6
                elif cls=='person':
                    self.CONF_THRESH=self.CONF_THRESH_PERSON-0.1 #0.7
                elif cls=='car':
                    self.CONF_THRESH=self.CONF_THRESH_CAR-0.1 #0.7
                elif cls == 'machine1_pingheng':
                    self.CONF_THRESH = self.CONF_THRESH_PINGHENG#0.55
                elif cls == 'machine2_peizhong':
                    self.CONF_THRESH = self.CONF_THRESH_PEIZHONG#0.55
                else:
                    self.CONF_THRESH=self.CONF_THRESH_INIT
            elif self.fct_id=='F001':#若为可见光
                if cls=='machine1_lvtou':
                    self.CONF_THRESH=self.CONF_THRESH_LVTOU #0.6
                elif cls=='person':
                    self.CONF_THRESH=self.CONF_THRESH_PERSON #0.7
                elif cls=='car':
                    self.CONF_THRESH=self.CONF_THRESH_CAR #0.7
                elif cls == 'machine1_pingheng':
                    self.CONF_THRESH = self.CONF_THRESH_PINGHENG#0.55
                elif cls == 'machine2_peizhong':
                    self.CONF_THRESH = self.CONF_THRESH_PEIZHONG#0.55
                else:
                    self.CONF_THRESH=self.CONF_THRESH_INIT# 0.6
            #
            inds = np.where(dets[:, -1] >= self.CONF_THRESH)[0]  # score 得分大于阈值的记录序号
            if len(inds) != 0:
                non = False
            cps_type,cps_count = self.object_classes_process(frame, inds, cls, dets) # 处理识别到的对象
            #构造数组
            if cps_count>0:
                if cps_type=='person':
                    exist_person=True
                if cps_type=='car':#仅仅添加车即可
                    #更新检测车辆数目
                    if cps_count > self.area_cps_type_count[1][3]:  # 修改结果中的值
                        self.area_cps_type_count[1][3] = cps_count
                    exist_car=True
        #如果faster_rcnn检测到人车目标,输出一张图片
        if exist_car:# or exist_person:
            cv2.imwrite(self.full_path_images + 'ALL_car_YES_' + str(self.cal_frame_num) + '.jpg', frame)
        #只有在检测到人的情况下判断有移动目标
        if exist_person:
            person_cl_result=self.lightrec_person_detect(frame)
            person_cl_count=len(person_cl_result)
            if person_cl_count>self.area_cps_type_count[0][3]:#修改结果中的值
                self.area_cps_type_count[0][3]=person_cl_count
            #将判断行人结果插入结果数据集合
            if person_cl_count>0:
                cv2.imwrite(self.full_path_images + 'RCNN_person_detect_YES_' + str(self.cal_frame_num) + '.jpg', frame)#检测有人，判断也有人
            else:
                exist_person=False#排除疑似目标,仅仅计数实际目标
                cv2.imwrite(self.full_path_images + 'RCNN_person_detect_NO_' + str(self.cal_frame_num) + '.jpg', frame)#检测有人，判断没有人
            self.logger.info('检测到当前计算帧的行人数量为：'+str(person_cl_count))


        #启动红外行人判断的条件
        if self.fct_id=='F003' and not exist_person and not self.gray_background is None:#启用红外检测人的条件是，一是红外光，二是faster_rcnn没有检测到人
            person_dt_result=self.infrarec_person_detect(frame)
            #如果检测到行人,进行处理和判断,主要判断与平衡铁的关系,后续可以考虑平衡铁上方的也取消
            is_contain=False
            person_valid_count=0
            for one_person_rect in person_dt_result:
                personRect_center=one_person_rect.getCenter()
                #中心点被平衡铁包含滤除
                is_under_pingheng = False
                for one_rect in self.all_pingheng_rects:#contain 变为  overlay
                    is_contained=self.pcontainbyrect(personRect_center,one_rect)
                    # is_overlay=self.overlap(carRect,one_rect)
                    if is_contained:#is_overlay:
                        self.logger.error('平衡铁与检测车辆重叠,要滤除')
                        is_under_pingheng = True
                        break
                if is_under_pingheng:#如果车在平衡铁下,不输出
                    continue
                #
                if self.overlap(self.alarm_area_rect, one_person_rect):
                    person_valid_count+=1
                    is_contain = True

            if is_contain>0:
                #更新结果
                if person_valid_count > self.area_cps_type_count[0][3]:  # 输出红外帧的最大数量
                    self.area_cps_type_count[0][3] = person_valid_count
                cv2.imwrite(self.full_path_images + 'infrared_person_YES_' + str(self.cal_frame_num) + '.jpg', frame)
            self.logger.info('检测到红外下行人数量为：'+str(len(person_dt_result)))
        #更新数据库
        if self.cal_frame_num%self.status_judge_count==0 and self.cal_frame_num!=0:
            self.cps_state_judgement(False)
            # print 'area_cps_type_count'+str(len(self.area_cps_type_count))
        #
        #判断是否满足判断抽油机状态的条件
        #根据当前计算的帧数，判断抽油机的状态
        if self.cal_frame_num % self.status_judge_count == 0  and self.cal_frame_num!=0:#200帧
            if self.cal_frame_num==self.status_judge_count:#仅仅在第一次判断结果的时候，验证标定框匹配是否正确，若不正确，则用拓扑匹配方法
                in_rects_count=0#所有在标定框内的驴头框
                for one_pump in self.lvtou_rects:
                    in_rects_count+= len(one_pump)
                all_rects_count=len(self.all_lvtou_rects)#搜索驴头框
                if in_rects_count>all_rects_count*0.7:
                    self.pump_state_judgement_new(False)
                else:
                    #基于拓扑的方法计算偏离
                    filter_candidate_results,avg_width,avg_height = self.pumps_into_group(self.all_lvtou_rects)#已经排序
                    cali_pump_centers=[x.getCenter() for x in self.pumps_alarm_area_rects2]#获取所有标定驴头的中心点位置，已经排序
                    #标定框纠偏和计算偏移
                    x_offset,y_offset=self.topo_cal_offset(filter_candidate_results,avg_width,cali_pump_centers)
                    #切换到原始的标定框
                    self.alarm_area_para=copy.deepcopy(self.alarm_area_para_source)
                    self.pumps_alarm_area_para=copy.deepcopy(self.pumps_alarm_area_para_source)
                    #纠正偏移
                    self.modify_rects_offset(x_offset,y_offset)
                    #重新分配驴头列表
                    match_lvtou_count=self.re_dist_lvtous_into_rects()
                    #直接从左向右排列标定框
                    if match_lvtou_count > all_rects_count * 0.8:#基于拓扑的计算偏离方法
                        # self.pump_state_judgement_new(False)
                        pass
                    else:#匹配不成功
                        # 再次清空分配到标定驴头框的驴头
                        for k in range(len(self.lvtou_rects)):
                            self.lvtou_rects[k] = []
                        #从左向右分配驴头框
                        # 切换到原始的标定框
                        self.alarm_area_para = copy.deepcopy(self.alarm_area_para_source)
                        self.pumps_alarm_area_para = copy.deepcopy(self.pumps_alarm_area_para_source)
                        self.left_right_offset(filter_candidate_results,cali_pump_centers)
                        # 重新分配驴头列表
                        match_lvtou_count = self.re_dist_lvtous_into_rects()
                        self.logger.info('match_lvtou_count = '+str(match_lvtou_count)   +'all_rects_count = '+str(all_rects_count))
                    #重新执行
                    self.pump_state_judgement_new(False)
                #清空临时校正时用的临时驴头数组
                self.all_lvtou_rects=[]
            else:#后面判断就不再考虑标定框的移动
                self.pump_state_judgement_new(False)
            # 将报警区和抽油机矩形框添叠加到图像上面 area_para
            # 标定图矩形框位置
            cv2.rectangle(frame, (int(self.alarm_area_rect2._xmin), int(self.alarm_area_rect2._ymin)),
                          (int(self.alarm_area_rect2._xmax), int(self.alarm_area_rect2._ymax)), (255, 0, 0), 2)
            # 校正矩形框位置
            cv2.rectangle(frame, (int(self.alarm_area_rect._xmin), int(self.alarm_area_rect._ymin)),
                          (int(self.alarm_area_rect._xmax), int(self.alarm_area_rect._ymax)), (0, 0, 0), 2)
            # 校正驴头框位置
            for rect in self.pumps_alarm_area_rects:
                cv2.rectangle(frame, (int(rect._xmin), int(rect._ymin)), (int(rect._xmax), int(rect._ymax)), (0, 0, 0),
                              2)
            # 标定驴头框位置
            for rect in self.pumps_alarm_area_rects2:
                cv2.rectangle(frame, (int(rect._xmin), int(rect._ymin)), (int(rect._xmax), int(rect._ymax)),
                              (255, 0, 0), 2)
            # 将上一阶段结算结果标注到图像上
            for one_pump_rt in self.pumps_status:
                self.label_pump_status(one_pump_rt[2], one_pump_rt[3], frame)
            # 若保存识别结果框图片
            if self.is_save_results:
                cv2.imwrite(self.full_path_images + str(self.cal_frame_num) + '.jpg', frame)
                # self.out_alarm_video()
        #计算结束更新一次平衡框
        pingheng_rects_count=len(self.all_pingheng_rects)
        if pingheng_rects_count>200:
            self.all_pingheng_rects = self.all_pingheng_rects[:pingheng_rects_count/2]
        #当前计算的帧数
        self.cal_frame_num+=1
    #重新分配驴头列表
    def re_dist_lvtous_into_rects(self):
        # 重新分配驴头列表
        match_lvtou_count = 0
        for arc_rect in self.all_lvtou_rects:
            arc_center = arc_rect.getCenter()
            for ecpr_index, each_cali_pump_rect in enumerate(self.pumps_alarm_area_rects):
                if self.pcontainbyrect(arc_center, each_cali_pump_rect):
                    self.lvtou_rects[ecpr_index].append(arc_rect)  # 把lv_tou便捷框添加到对应抽油机的列表中
                    match_lvtou_count += 1
                    break
        return match_lvtou_count
    #将抽油机状态计算结果显示到图片上
    def label_pump_status(self,pump_id,pump_status,_frame):
        for index,one_pump_para in enumerate(self.pumps_alarm_area_para):
            if pump_id==one_pump_para[0] and one_pump_para[0]!=5:#有效的抽油机
                cur_paa_rect=self.pumps_alarm_area_rects[index]
                cts_point=cur_paa_rect.getCenter()
                label_pos_y=random.randint(cur_paa_rect.getymin(),cur_paa_rect.getymax())
                cv2.putText(_frame, self.dict_status_labels[pump_status], (int(cts_point._x), int(label_pos_y)),cv2.FONT_HERSHEY_COMPLEX, 0.8,(255, 0, 0), 2)
                break  # 不会有两个同名的抽油机
    #第四部分end### ----------------------faster_rcnn检测---------------------###
    #
    #
    #
    #第五部分start### ----------------------目标状态判断---------------------###
    #本次巡检结束,输出最后一个结果
    def stop_inspection_out_result(self):
        if self.cal_frame_num%self.output_status_count>=self.status_judge_count:#当输出最后一个结果时,帧数必须大于最小计算帧200帧
            #输出抽油机状态
            self.pump_state_judgement_new(True)
            #输出人车状态
            self.cps_state_judgement(True)
    #每200帧更新一次人车状态,每600秒输出一次结果
    def cps_state_judgement(self,is_end):
        self.logger.info('———————————————判断人车状态——————————')

        for cps_type1 in self.area_cps_type_count:
            self.logger.info('output200 ' + cps_type1[2] + ' = ' + str(cps_type1[3]))

        if self.cal_frame_num%self.output_status_count==0 or is_end:#600帧输出,或者结束时200帧以上结束

            for cps_type in self.area_cps_type_count:
                self.logger.info('output600 '+cps_type[2]+' = '+str(cps_type[3]))

            self.result_handler.update_area_status(self.area_cps_type_count,self.log_name)
            #每30秒更新一次结果
            self.area_cps_type_count[0][3]=0
            self.area_cps_type_count[1][3]=0

    # 帧率25帧每秒，每200帧判断一次抽油机状态,每600秒输出一次结果
    def pump_state_judgement_new(self,is_end):
        self.logger.info('———————————————判断抽油机状态——————————')
        self.pumps_status = []
        group_lvtou_list=[]
        group_id_list=[]
        for index, lvtou_list in enumerate(self.lvtou_rects):  # 对应抽油机数量
            if self.pumps_overlay_relt[index]==0:#没有任何的交叉，直接计算
                self.pumps_status.append(self.judg_one_pump_status(lvtou_list,index))
                #考虑到0210的情况
                if len(group_lvtou_list)>0:#考虑2开始时，前面为空的情况
                    self.pumps_status.extend(self.judg_group_pumps_status(group_lvtou_list,group_id_list))
                #清空group列表
                group_lvtou_list = []
                group_id_list = []
            elif self.pumps_overlay_relt[index]==1 or self.pumps_overlay_relt[index]==3:
                group_lvtou_list.extend(lvtou_list)
                group_id_list.append(index)
            elif self.pumps_overlay_relt[index]==2:#右边有，左边没有，重新开始
                if len(group_lvtou_list)>0:#考虑2开始时，前面为空的情况
                    self.pumps_status.extend(self.judg_group_pumps_status(group_lvtou_list,group_id_list))
                #清空group列表
                group_lvtou_list = []
                group_id_list = []
                group_lvtou_list.extend(lvtou_list)
                group_id_list.append(index)
        #对于最后一个左边有交叉的抽油机
        if len(group_lvtou_list) > 0:
            self.pumps_status.extend(self.judg_group_pumps_status(group_lvtou_list, group_id_list))
        # add code 将抽油机状态结果写入数据库,满足600输出条件,或者本次巡检结束,最后一次输出
        if len(self.pumps_status) > 0 and self.cal_frame_num%self.output_status_count==0 or is_end:
            self.pumps_status=filter(lambda x: x[4]!=5, self.pumps_status)#只筛选有效的抽油机,即类型不是5的
            self.result_handler.update_pumps_curresult(self.pumps_status,self.log_name)
            # 清空判断抽油机状态的列表,每过600帧输出一次结果
            for k in range(len(self.lvtou_rects)):
                self.lvtou_rects[k] = []
    #识别驴头框分组
    def pumps_into_group(self,g_lvtou_list):
        filter_candidate_results = []
        if len(g_lvtou_list)==0:
            return filter_candidate_results,0,0
        # 2.判断指标计算
        centers = []
        sum_height = 0
        sum_width = 0
        for lvtou in g_lvtou_list:
            centers.append(lvtou.getCenter())
            sum_height += lvtou.getHeight()
            sum_width += lvtou.getWidth()
        avg_height = sum_height / len(g_lvtou_list)
        avg_widht = sum_width / len(g_lvtou_list)
        # 3.对所有中心点按照x排序
        sorted_centers = sorted(centers, key=lambda x: x._x)
        pre_count = 0  # 前一center相邻域结果数量
        pre_index = 0  # 前一center位置
        pre_center = None  # 前一center
        pre_sub_centers = []  # 前一center相邻结果集
        candidate_results = []
        have_process = False
        #
        pre_center_x = 0
        # 4找出所有局部极值位置
        # 最后补0,即为有效区右边界X值
        sorted_centers.append(point(self.alarm_area_rect.getxmax(), 0))
        for index, one_center in enumerate(sorted_centers):
            ocx = one_center._x
            cur_count, sub_centers = self.filter_nn(sorted_centers, ocx - avg_widht / 3, ocx + avg_widht / 3)
            # self.logger.info('cur_count = '+str(cur_count)+'  ocx = '+str(ocx))
            # 判断当前子集相邻域的中心点数,如果比之前大
            if cur_count > pre_count:  # 爬山
                have_process = False

            elif cur_count == pre_count:
                if ocx - pre_center_x > avg_widht:  # and always_equ_count:
                    one_candidate_result = []
                    one_candidate_result.append(pre_count)
                    one_candidate_result.append(pre_center)
                    one_candidate_result.append(pre_sub_centers)
                    candidate_results.append(one_candidate_result)

            else:  # 若开始下降，则处理当前最大值
                # 找到局部最大值,或者
                if not have_process:  # or #进行阈值判断，不满足阈值条件的局部极大值忽略
                    # 与抽油机边框进行比较，最终判断是否属于该抽油机
                    one_candidate_result = []
                    one_candidate_result.append(pre_count)
                    one_candidate_result.append(pre_center)
                    one_candidate_result.append(pre_sub_centers)
                    candidate_results.append(one_candidate_result)
                    have_process = True
            #
            pre_center_x = ocx
            # 更新指标量,无论如何都会更新
            pre_count = cur_count
            pre_index = index
            pre_center = one_center
            pre_sub_centers = sub_centers
        # 5对所有局部极值进行过滤，合并距离较近的结果
        if len(candidate_results) == 0:
            # return group_pumps_status
            pass
        elif len(candidate_results) == 1:
            filter_candidate_results.extend(candidate_results)  # 添加到过滤候选结果集
        else:
            pre_ocr = None
            for idx, ocr in enumerate(candidate_results):
                if not pre_ocr:  # 第一个候选
                    pre_ocr = ocr
                    continue
                if ocr[1]._x - pre_ocr[1]._x < avg_widht / 2:  # 两个极值间距小于阈值，则合并
                    pre_ocr[0] += ocr[0]
                    pre_ocr[2].extend(ocr[2])
                else:
                    # 处理上一个
                    filter_candidate_results.append(pre_ocr)
                    pre_ocr = ocr
                if idx == len(candidate_results) - 1:  # 如果为最后一个结果集
                    filter_candidate_results.append(pre_ocr)
        return filter_candidate_results,avg_widht,avg_height

    def judg_group_pumps_status(self,g_lvtou_list,g_id_list):
        #1.初始化返回结果
        group_pumps_status=[]
        for idx in g_id_list:
            one_pump_status = []
            one_pump_status.append(self.alarm_area_para[0])  # 巡检记录ID
            one_pump_status.append(self.alarm_area_para[2])  # 报警区ID
            one_pump_status.append(self.pumps_alarm_area_para[idx][0])  # 抽油机ID
            one_pump_status.append('-1')
            one_pump_status.append(self.pumps_alarm_area_para[idx][1])
            group_pumps_status.append(one_pump_status)
        if len(g_lvtou_list)<=1:
            return group_pumps_status
        #所有未知驴头分组
        filter_candidate_results,avg_width,avg_height=self.pumps_into_group(g_lvtou_list)
        #6判断结果
        cur_pumps_idx=0
        for one_candidate in filter_candidate_results:
            one_center=one_candidate[1]
            if cur_pumps_idx==len(g_id_list):#所有抽油机都赋予了值，多余的候选位置抛弃
                self.logger.info('候选极大值太多！')
                break

            for opidx, one_pump in enumerate(g_id_list):
                if opidx<cur_pumps_idx:
                    continue
                if self.pcontainbyrect(one_center,self.pumps_alarm_area_rects[one_pump]):
                    # if cur_pumps_idx==2 and opidx==1:
                    #     pass
                    #print 'group_pumps_status = %d    cur_pumps_idx = %d   opidx = %d   g_id_list = %d'%(len(group_pumps_status),cur_pumps_idx,opidx,len(g_id_list))
                    group_pumps_status[opidx]=self.judge_group_one_pump_status(one_candidate[2],avg_height,one_pump)
                    cur_pumps_idx += 1  # 无论是否找到，都要后移
                    break
                else:
                    cur_pumps_idx += 1  # 无论是否找到，都要后移
        self.logger.info('处理完毕！')
        return group_pumps_status
    #截图相邻的中心点
    def filter_nn(self,data_centers,low_v,hight_v):
        nn_data = filter(lambda x: x._x >= low_v and x._x <= hight_v, data_centers)
        return len(nn_data),list(nn_data)

    def judg_one_pump_status(self,lvtou_list,index):
        centerys = []  # 仅需要判别Y轴坐标
        heights = []  # 识别边框高度
        for lvtou in lvtou_list:
            center = rectangle.getCenter(lvtou)
            centerys.append(center._y)
            heights.append(lvtou.getHeight())
        if len(heights)>0:
            height_avg = sum(heights) / len(heights)
        else:
            height_avg=0
        return self.judge_status(centerys,height_avg,index)

    def judge_group_one_pump_status(self,_centers,_height,_index):
        _centerys=[]
        for cts in _centers:
            _centerys.append(cts._y)
        return self.judge_status(_centerys,_height,_index)
    #单台抽油机状态判断
    def judge_status(self,_centerys,_height,_index):
        # 单台抽油机状态
        one_pump_status = []
        one_pump_status.append(self.alarm_area_para[0])  # 巡检记录ID
        one_pump_status.append(self.alarm_area_para[2])  # 报警区ID
        one_pump_status.append(self.pumps_alarm_area_para[_index][0])  # 抽油机ID
        one_pump_status.append('-1')
        one_pump_status.append(self.pumps_alarm_area_para[_index][1])  # 抽油机类型
        # 针对每台抽油机进行判断并输出结果
        if len(_centerys) <= 1:  # 如果检测对象数目不大于1个
            self.logger.info('检测抽油机对象过少 '+self.pumps_alarm_area_para[_index][0])
            return one_pump_status
        # 中心点的最大最下值
        ymin = min(_centerys)
        ymax = max(_centerys)
        # 识别框的高度的平均值
        self.logger.info('ymin = ' +str(ymin)+' ymax = '+str(ymax)+' height = '+str(_height))
        if (ymax - ymin) >= _height * self.move_percent:  # self.move_threshold
            one_pump_status[3]='1'
            self.logger.info('抽油机正在工作 '+self.pumps_alarm_area_para[_index][0])
        elif (ymax - ymin) < _height * self.unmove_percent:  # self.unmove_threshold
            one_pump_status[3]='0'
            self.logger.info('抽油机工作异常 '+ self.pumps_alarm_area_para[_index][0])
        else:
            one_pump_status[3]='2'
            self.logger.info('抽油机状态待定，待下一周期判断 '+ self.pumps_alarm_area_para[_index][0])

        #添加抽油机类型
        return one_pump_status
    #第五部分end### ----------------------目标状态判断---------------------###
