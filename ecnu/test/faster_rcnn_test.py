#!/usr/bin/env python
# -*- coding:utf-8 -*-

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import caffe, os, sys, cv2, re
import argparse
import traceback
import datetime
import time
from shape import rectangle
from shape import point
import copy
# from common import my_logging

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]', choices=NETS.keys(), default='vgg16')#zf vgg16 vgg1024  VGG_CNN_M_1024
    args = parser.parse_args()
    return args


def init_caffe_model():
    # 初始化faster_rcnn 模型
    cfg.TEST.HAS_RPN = True
    # Parse input arguments40
    args = parse_args()

    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0], 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    # caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', NETS[args.demo_net][1])

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)


    print'\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    return net


def object_classes_process(frame, inds, class_name, dets,f_tv,file_name):
    object_type = class_name
    object_count = 0
    # 循环有效记录的序列号
    for i in inds:  # score满足条件的记录
        bbox = dets[i, :4]  # 获取边界框
        score =round(dets[i, -1],2)  #获取置信得分列
        # print"The object is " + class_name + " xmin=" + str(int(bbox[0])) + " ymin" + str(int(bbox[1])) + " xmax" + str(
        #     int(bbox[2])) + " ymax" + str(int(bbox[3]))
        # 依次判断对象类别
        if class_name == 'machine1_lvtou':
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(frame, 'lvtou'+str(score), (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 0, 255), 1)
            # f_tv.write(file_name+','+str(int(bbox[2]-int(bbox[0])))+','+str(int(bbox[3]-int(bbox[1]))) +'\n')
        elif class_name == 'machine1_pingheng':
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
            cv2.putText(frame, 'pingheng'+str(score), (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (255, 0, 0), 1)
            pass

        elif class_name == 'person':
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 153), 1)
            # cv2.putText(frame, str(score)+class_name, (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
            #             (255, 0, 153), 1)
            #
            cur_rect = (int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3]))
            copy_cur_rect = copy.deepcopy(cur_rect)
            insert_object_to_trajectory(copy_cur_rect)
            object_count+=1

        elif class_name == 'car':
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 1)
            # cv2.putText(frame, str(score)+class_name, (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
            #             (0, 255, 255), 1)
            pass
        elif class_name == 'machine2_peizhong':
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 1)
            cv2.putText(frame, 'peizhong'+str(score), (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                        (0, 255, 255), 1)
        else:
            print"未知的对象类型，可能出现错误！"
    return object_type, object_count


def faster_rcnn_detection(frame,net,f_tv,file_name):  # 巡检区框，抽油机框，驴头框

    # 针对帧开始计算
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, frame)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    # 识别框筛选
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]  # 当前帧所有驴头
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)  # 合并cls_boxes和cls_scores
        keep = nms(dets, NMS_THRESH)  # 非极大值抑制，去掉重复的位置框
        dets = dets[keep, :]

        if cls == 'machine1_lvtou':
            CONF_THRESH =0.6   # 0.6
        elif cls == 'person':
            CONF_THRESH =0.6  # 0.7
        elif cls == 'car':
            CONF_THRESH =0.7 # 0.65
        elif cls=='machine1_pingheng':
            CONF_THRESH=0.55
        else:
            CONF_THRESH = 0.6  # 0.6

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]  # score 得分大于阈值的记录序号
        if len(inds) > 1:
            non = False
            # print 'inds count = %d'%(len(inds))
        cps_type, cps_count = object_classes_process(frame, inds, cls, dets,f_tv,file_name)  # 处理识别到的对象
        if cps_count>0:
            print cps_type +'     '+str(cps_count)
        # 构造数组


def test_image_result():

    inpath1='/home/hc/myworkspace/sourcecode/test_video/zhao20181009/'
    outpath1='/home/hc/myworkspace/sourcecode/test_video/zhao20181118_vgg16/'

    inpath2='/media/hc/46763B95763B84A9/myworkspace/train_dataset_prepare/CVC-CER-01/pedestrian/'
    outpath2 = '/media/hc/46763B95763B84A9/myworkspace/train_dataset_prepare/CVC-CER-01/result/'

    inpath3='/media/hc/46763B95763B84A9/myworkspace/train_dataset_prepare/CVC-14/Night/FIR/NewTest/FramesPos/'
    outpath3='/media/hc/46763B95763B84A9/myworkspace/train_dataset_prepare/CVC-14/Night/FIR/NewTest/result/'

    inpath4='/root/图片/P/ARD巡检录像/1/'
    outpath4='/root/图片/detection_result/00/'

    inpath=inpath1
    outpath=outpath1

    list = os.listdir(inpath)  # 列出文件夹下所有的目录与文件

    #驴头大小
    path = os.path.join(outpath, '1lvtou_wh.txt')
    f_tv = open(path, 'w')
    #
    for i in range(0, len(list)):
        path = os.path.join(inpath, list[i])
        if os.path.isfile(path):
            frame=cv2.imread(path)
            faster_rcnn_detection(frame,net,f_tv,list[i])
            #
            # time.sleep(0.8)
            #
            print 'current image index = '+str(i)
            cv2.imwrite(outpath + list[i], frame)
    #关闭文件
    f_tv.close()

def out_alarm_video(frame,full_path_name,width,hight,video_writer):
    # 输出视频
    if video_writer==None:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv2.VideoWriter(full_path_name, fourcc, 25.0,(width, hight))
    video_writer.write(frame)
    return video_writer

#test moving person
# 判断两个矩形是否相交
def mat_inter(box1, box2):
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    width0=x02-x01
    width1=x12-x11
    x01-=width0
    x02+=width0
    x11-=width1
    x12+=width1
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
def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    if mat_inter(box1, box2) == True:
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
def dis_two_rect(box1, box2):
    # 判断两个矩形中心点的距离
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    center1_x = (x01 + x02) / 2
    center1_y = (y01 + y02) / 2
    center2_x = (x11 + x12) / 2
    center2_y = (y11 + y12) / 2
    # distance=np.sqrt(pow(center1_x-center2_x,2)+pow(center1_y-center2_y,2))
    distance = abs(center1_x - center2_x)
    return distance
#两个矩形框的宽度之比
def w_ratio_two_rect(box1, box2):
    # 计算两个矩形宽度比
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    w_ratio = float(x02 - x01) / (x12 - x11)
    return w_ratio
#将目标插入已有轨迹
def insert_object_to_trajectory(crect):
    # 表示当前crect是否能够添加到当前的列表中，若不能，则添加新对象
    is_insert = False
    for cur_trajectory in trajectorys:
        last_rect = cur_trajectory[-1]
        if mat_inter(last_rect, crect):
            if cur_trajectory[0]:# or not cur_trajectory[1]:  # 如果本轮插入过，或者上一轮没有插入过，进行下次循环
                continue  # 已经有出口
            cur_dis = dis_two_rect(last_rect, crect)
            cur_trajectory[0] = True  ##若有数据插入，则置为True，表示本轮已经插入过数据，不能够再插入
            # cur_trajectory[1] = True#更新状态
            cur_trajectory[2] = cur_dis
            cur_trajectory[3] += cur_dis
            cur_trajectory.append(crect)
            is_insert = True
            break  #
    if not is_insert:  # 添加新对象
        one_trajectory = [True, False, 0, 0, crect]  # 本轮状态，上轮状态，本次距离，累积距离，矩形框
        trajectorys.append(one_trajectory)
def lightrec_person_detect(cur_frame):
    global trajectorys
    try:
        # 输出的行人矩形框
        person_rects = []
        # print 'person_count = '+str(len(trajectorys))
        # 判别新对象
        for index, cur_trajectory in enumerate(trajectorys):
            if cur_trajectory[0]:  # 本次计算没有目标更新
                # # 首先更新上轮状态
                # cur_trajectory[1] = True  # 赋值上一轮的状态，表示是否插入过数据
                # #
                # print 'move_distance = '+str(cur_trajectory[2])+'      '+str(cur_trajectory[3])
                # if cur_trajectory[2] > 3 or cur_trajectory[3] / float(
                #         len(cur_trajectory) - 4) > 2:  # cur_trajectory[2]>3 or
                #     # 显示序列中所有的框
                #     # (x1, y1, x2, y2) = cur_trajectory[-1]
                #     for prect in cur_trajectory[4:]:
                #         (x1, y1, x2, y2) = prect
                #         cv2.rectangle(cur_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                #     # person_rects.append(rectangle(x1, y1, x2, y2))  # 转化为shape rect

                # 首先更新上轮状态
                cur_trajectory[1] = True  # 赋值上一轮的状态，表示是否插入过数据
                # 判断水平或者垂直位置变化
                sum_widht = 0
                sum_height = 0
                avg_width = 0
                avg_heigth = 0
                min_center_x = 2000  # 首先赋值一个大值
                max_center_x = 0
                cur_tra_count = 0
                for prect in cur_trajectory[4:]:
                    (x1, y1, x2, y2) = prect
                    sum_widht += (x2 - x1)
                    sum_height += (y2 - y1)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    cur_tra_count += 1
                    if center_x < min_center_x:
                        min_center_x = center_x
                    if center_x > max_center_x:
                        max_center_x = center_x
                if cur_tra_count > 0:
                    avg_width = sum_widht / cur_tra_count
                    avg_heigth = sum_height / cur_tra_count
                # 结果判断
                print 'max_center_x-min_center_x = ' + str(max_center_x - min_center_x) + '   avg_width = ' + str(
                    avg_width)
                if max_center_x - min_center_x > avg_width * 1:  # 符合条件输出
                    (x1, y1, x2, y2) = cur_trajectory[-1]
                    cv2.rectangle(cur_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    person_rects.append(rectangle(x1, y1, x2, y2))  # 转化为shape rect

            else:
                cur_trajectory[1] = False  # 把没有更新的目标置为false
                pass
            # 恢复为未插入状态
            cur_trajectory[0] = False
        # 删除不在可能是轨迹的记录
        trajectorys = filter(lambda x: x[1] == True, trajectorys)  # 筛选符合条件的记录，不符合条件的删除
    except:
        print '计算可见光行人出错'+traceback.format_exc()
    finally:
        return person_rects
#
def test_video_result():
    inpath1 = '/home/hc/myworkspace/sourcecode/test_video/test_sample_1010/20181009/'
    inpath2='/home/hc/myworkspace/sourcecode/test_video/test_sample_1010/20181011/'
    outpath1='/home/hc/myworkspace/sourcecode/test_video/test_sample_1010/test20181009/'
    outpath2='/home/hc/myworkspace/sourcecode/test_video/test_sample_1010/test20181012/'

    inpath=inpath2
    outpath=outpath2

    list = os.listdir(inpath)  # 列出文件夹下所有的目录与文件

    for i in range(0, len(list)):
        path_in = os.path.join(inpath, list[i])
        print path_in
        path_out=os.path.join(outpath, list[i])
        if os.path.isfile(path_in):
            index = 0
            cap = cv2.VideoCapture(path_in)
            try:
                is_first=True
                video_writer=None
                index=0
                while cap.isOpened():
                    index+=1
                    ret, frame = cap.read()
                    if not ret:
                        print '视频流读取结束'
                        break
                    hight = frame.shape[0]
                    width = frame.shape[1]
                    if index % 20 == 0:
                        print str(index)
                        faster_rcnn_detection(frame, net,'dd','ddd')
                        # cv2.imshow('frame', frame)
                        # video_writer=out_alarm_video(frame,path_out,width,hight,video_writer)
                        # cv2.imwrite(os.path.join(outpath,str(index)+'.jpg'), frame)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                        lightrec_person_detect(frame)
                        height, width = frame.shape[:2]
                        # 缩小图像
                        size = (int(width * 0.5), int(height * 0.5))
                        shrink = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                        cv2.imshow('frame', shrink)
                        if cv2.waitKey(500) & 0xFF == ord('q'):
                            break

                    # index += 1
                    # #test 行人

                    # time.sleep(0.02)
            except:
                print traceback.print_exc()
            finally:
                if video_writer!=None:
                    video_writer.release()






if __name__ == '__main__':
    #
    # logger=my_logging.get_logger()
    # logger.debug('ddfddf')
    #全局变量
    CLASSES = ('__background__','machine1_lvtou','machine1_pingheng','machine2_peizhong','car','person')
    # NETS = {'zf': ('ZF','ZF_faster_rcnn_final_0918.caffemodel')}  vgg16_faster_rcnn_iter_1007.caffemodel
    NETS = {'vgg16': ('VGG16',
                      'vgg16_faster_rcnn_iter_1111.caffemodel'),
            'vgg1024': ('VGG_CNN_M_1024',
                        'vgg_cnn_m_1024_faster_rcnn_iter_1117.caffemodel'),
            'zf': ('ZF',
                   'ZF_faster_rcnn_final_1117.caffemodel')}
    #score置信度阈值
    CONF_THRESH = 0.8
    #非极大值抑制处理阈值
    NMS_THRESH = 0.15

    trajectorys = []

    # nowTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # print nowTime

    net = init_caffe_model()

    #
    test_video_result()

    # test_image_result()





