# -*- coding:utf-8 -*-
import threading
from ecnu.ai_rt_stream_detection import ai_rt_stream_detection
from common.my_dao import *
from common import my_logging
import time
import cv2
import inspect
import ctypes
import thread
import os


# 此类查询巡检区与抽油机参数
class ParameterGained:
    def __init__(self, fct_id, aas_obs_id, aas_dept_id, rt_stream_path):
        # 巡检区因素编号
        self.fct_id = fct_id
        # 巡检区部门编号
        self.aas_dept_id = aas_dept_id
        # 巡检区光电编号
        self.aas_obs_id = aas_obs_id
        # 多少秒一查询巡检记录表
        self.second = 2
        # 视频地址
        self.rt_stream_path = rt_stream_path
        # 当前巡检id
        self.current_ri_id = None
        # self.test_nums = 122
        self.aas_zone_id = None
        # 日志对象声明
        self.logger = my_logging.get_logger(aas_dept_id + "_" + aas_obs_id + "_" + fct_id)
        # 数据库工具
        self.MysqldbHelper = MysqldbHelper(self.logger)
        self.video_flag = False
        self.judge = False
        # 此处初始化识别程序
        self.detection = ai_rt_stream_detection(aas_dept_id + "_" + aas_obs_id + "_" + fct_id)
        self.pumpunit_list = []
        self.alarmarea_list = []
        self.testing_video_sum = True
        # self.videoPath()
        self.azimuth = 0.0
        self.deviation_x = 0.0
        self.tvfocus = 0.0
        self.tvfoci = 0.0
        self.irfocus = 0.0
        self.irfoci = 0.0

    def select_alarm_params(self, ri_id):
        """
        获取巡检区参数
        :param ri_id:
        :return:
        """

        search_alarm_sql = "select * from ldata_alarmarea_scope " \
                           "where fct_id='%s' and aas_dept_id='%s' " \
                           "and aas_obs_id='%s' and aas_zone_id='%s' " % \
                           (self.fct_id, self.aas_dept_id, self.aas_obs_id, self.aas_zone_id)
        results = self.MysqldbHelper.select(search_alarm_sql)

        var_list = ['aas_leftupx', 'aas_leftupy', 'aas_rightdownx', 'aas_rightdowny']
        ass_list = [ri_id, self.fct_id, self.aas_zone_id]
        if results is not None and len(results) != 0:
            pic_path = "/media/anruida/F447DF5D9ED16F85/cali_picture"
            for temp in results:
                pic = str(temp['aas_picturename']).split(':')
                p = pic[1].replace('\\', '/')
                for nums in range(0, 4, 1):
                    ass_list.append(temp[var_list[nums]])
                pic_path = pic_path + p
                ass_list.append(pic_path)
                ass_list.append(temp['aas_picturetype'])
        return ass_list

    def select_pump_params(self):
        """
        获取抽油机参数
        :return: pump_list
        """
        search_pump_sql = "select * from ldata_pumpunit_square where fct_id='%s' and aas_dept_id='%s'" \
                          "and aas_obs_id='%s' and aas_zone_id='%s'" % \
                          (self.fct_id, self.aas_dept_id, self.aas_obs_id, self.aas_zone_id)
        result = self.MysqldbHelper.select(search_pump_sql)
        pump_list = []

        var_pump_list = ['pump_id', 'pump_type', 'pump_leftupx', 'pump_leftupy', 'pump_rightdownx', 'pump_rightdowny']
        if result is not None and len(result) != 0:
            for temp in result:
                var_pumps_list = []
                for nums in range(0, len(var_pump_list), 1):
                    var_pumps_list.append(temp[var_pump_list[nums]])

                pump_list.append(var_pumps_list)
        return pump_list

    def update_ri_handle(self, temp):
        """
        更新巡检区处理记录
        :param temp:
        :return:
        """
        ld_sql = "UPDATE ldata_routing_inspection " \
                 "SET ri_handle = '%d' where ri_id = '%s' and ri_time = '%s' and ri_fct_id = '%s' and" \
                 " ri_dept_id = '%s' and ri_obs_id = '%s' and ri_zone_id = '%s'" % \
                 (1, temp['ri_id'], temp['ri_time'], temp['ri_fct_id'], temp['ri_dept_id'], temp['ri_obs_id'],
                  temp['ri_zone_id'])
        self.MysqldbHelper.update(ld_sql)

    def get_main_alarm(self):
        """
        获取报警区视场参数
        :return:
        """
        main_alarm_sql = "select zone_azimuth,zone_devation, " \
                         "zone_tvfocus, zone_tvfoci, zone_irfocus, zone_irfoci from main_alarm_zone" \
                         " where dept_id = '%s' and radar_id = '%s' and zone_id = '%s'" % \
                         (self.aas_dept_id, self.aas_obs_id, self.aas_zone_id)
        # print("main_alarm_sql :" + main_alarm_sql)
        rs = self.MysqldbHelper.select(main_alarm_sql)
        pd = 0
        if rs is not None and len(rs) != 0:
            for tes in rs:
                pd += self.get_obs_curview(tes)
        return pd

    @property
    def get_deviation(self):
        """
        获取报警区视场参数
       :return:
        """
        curview_sql = "SELECT * FROM ldata_obs_curview where ocv_obs_id = '%s';" % (self.aas_obs_id)
        alarm_sql = "SELECT * FROM main_alarm_zone where dept_id = '%s' and radar_id = '%s' and zone_id = '%s';" % \
                    (self.aas_dept_id, self.aas_obs_id, self.aas_zone_id)

        # print("main_alarm_sql :" + main_alarm_sql)

        rs2 = self.MysqldbHelper.select(curview_sql)
        rs3 = self.MysqldbHelper.select(alarm_sql)
        # azimuth = abs(rs2[0]['ocv_azimuth'] - rs3[0]['zone_azimuth'])
        # deviation_x = abs(rs2[0]['ocv_devation'] - rs3[0]['zone_devation'])
        azimuth = rs2[0]['ocv_azimuth'] - rs3[0]['zone_azimuth']
        deviation_x = rs2[0]['ocv_devation'] - rs3[0]['zone_devation']
        result = []
        result.append(azimuth)
        result.append(deviation_x)
        return result

    def get_error(self):
        """
        获取报警区视场参数
       :return:
        """
        curview_sql = "SELECT * FROM ldata_obs_curview where ocv_obs_id = '%s';" % (self.aas_obs_id)
        alarm_sql = "SELECT * FROM main_alarm_zone where dept_id = '%s' and radar_id = '%s' and zone_id = '%s';" % \
                    (self.aas_dept_id, self.aas_obs_id, self.aas_zone_id)

        # print("main_alarm_sql :" + main_alarm_sql)

        rs2 = self.MysqldbHelper.select(curview_sql)
        rs3 = self.MysqldbHelper.select(alarm_sql)
        # azimuth = abs(rs2[0]['ocv_azimuth'] - rs3[0]['zone_azimuth'])
        # deviation_x = abs(rs2[0]['ocv_devation'] - rs3[0]['zone_devation'])
        azimuth = rs2[0]['ocv_azimuth'] - rs3[0]['zone_azimuth']
        deviation_x = rs2[0]['ocv_devation'] - rs3[0]['zone_devation']
        tvfocus = rs2[0]['ocv_tvfocus'] - rs3[0]['zone_tvfocus']
        tvfoci = rs2[0]['ocv_tvfoci'] - rs3[0]['zone_tvfoci']
        irfocus = rs2[0]['ocv_irfocus'] - rs3[0]['zone_irfocus']
        irfoci = rs2[0]['ocv_irfoci'] - rs3[0]['zone_irfoci']
        result = []

        if abs(azimuth) > abs(self.azimuth):
            self.azimuth = azimuth
        if abs(deviation_x) > abs(self.deviation_x):
            self.deviation_x = deviation_x
        if abs(tvfocus) > abs(self.tvfocus):
            self.tvfocus = tvfocus
        if abs(tvfoci) > abs(self.tvfoci):
            self.tvfoci = tvfoci
        if abs(irfoci) > abs(self.irfoci):
            self.azimuth = azimuth
        if abs(irfocus) > abs(self.irfocus):
            self.azimuth = azimuth
        result.append(tvfocus)
        result.append(tvfocus)
        result.append(azimuth)
        result.append(deviation_x)
        result.append(irfocus)
        result.append(irfoci)
        return result

    def get_method2(self, y1, y2):

        """
        获取报警区视场参数
        :return:
        """
        arr = self.get_deviation
        azimuth = arr[0]
        deviation = arr[1]
        high = abs(y1 - y2)
        x = 300
        deviation_x = float((float(azimuth) * x)) / (float(high) / float(576))
        deviation_y = float((float(deviation) * x)) / (float(high) / float(576))
        print 'azimuth = %f   deviation = %f    high = %d' % (azimuth, deviation, high)
        result = []
        result.append(deviation_x)
        result.append(deviation_y)
        return result

    def begin_video(self):
        ocv_azimuth = 0.0
        ocv_devation = 0.0
        t = 1
        sh = 5
        change = 40
        if (self.aas_obs_id == '5'):
            change = 25
            sh = 6
        ocv_tvfocus = 0.0
        ocv_tvfoci = 0.0
        ocv_irfocus = 0.0
        ocv_irfoci = 0.0
        while True:

            szone_devationql = "SELECT * FROM ldata_obs_curview where ocv_obs_id='"
            szone_devationql += self.aas_obs_id + "'"
            result = self.MysqldbHelper.select(szone_devationql)

            if (ocv_azimuth == result[0]['ocv_azimuth']
                and ocv_devation == result[0]['ocv_devation'] and (
                            abs(ocv_irfoci - result[0]['ocv_irfoci']) < change and
                            abs(ocv_irfocus - result[0]['ocv_irfocus']) < change

            )) and t != 1:
                ocv_azimuth = result[0]['ocv_azimuth']
                ocv_devation = result[0]['ocv_devation']
                ocv_irfoci = result[0]['ocv_irfoci']
                ocv_irfocus = result[0]['ocv_irfocus']
                ocv_tvfocus = result[0]['ocv_tvfocus']
                ocv_tvfoci = result[0]['ocv_tvfoci']
                break
            else:
                t = 0
                ocv_irfoci = result[0]['ocv_irfoci']
                ocv_irfocus = result[0]['ocv_irfocus']
                ocv_tvfocus = result[0]['ocv_tvfocus']
                ocv_tvfoci = result[0]['ocv_tvfoci']
                ocv_azimuth = result[0]['ocv_azimuth']
                ocv_devation = result[0]['ocv_devation']
                time.sleep(sh)

    def testing_video(self):
        ri_time = 0.0
        t = 1;
        while True:

            szone_devationql = "SELECT * FROM ldata_routing_inspection  where ri_obs_id='"
            szone_devationql += self.aas_obs_id + "' order by ri_time desc"
            result = self.MysqldbHelper.select(szone_devationql)
            if ri_time != result[0]['ri_time'] and t != 1:
                self.detection.stop_calculate_thread()
                break
            else:
                t = 0
                ri_time = result[0]['ri_time']
            time.sleep(1)

    def test_alarm(self):
        ocv_azimuth = 0.0
        ocv_devation = 0.0
        t = 1
        ocv_tvfocus = 0.0
        ocv_tvfoci = 0.0
        ocv_irfocus = 0.0
        ocv_irfoci = 0.0
        while True:
            szone_devationql = "SELECT * FROM ldata_obs_curview where ocv_obs_id='"
            szone_devationql += self.aas_obs_id + "'"
            result = self.MysqldbHelper.select(szone_devationql)

            if (ocv_azimuth != result[0]['ocv_azimuth']
                or ocv_devation != result[0]['ocv_devation']) and t != 1:

                ocv_azimuth = result[0]['ocv_azimuth']
                ocv_devation = result[0]['ocv_devation']
                ocv_irfoci = result[0]['ocv_irfoci']
                ocv_irfocus = result[0]['ocv_irfocus']
                ocv_tvfocus = result[0]['ocv_tvfocus']
                ocv_tvfoci = result[0]['ocv_tvfoci']
                print "end"
                print ""
            else:
                t = 0
                print str(ocv_tvfocus) + " " + str(ocv_tvfocus) + " " + str(ocv_irfocus) + " " + str(ocv_irfoci)
                ocv_irfoci = result[0]['ocv_irfoci']
                ocv_irfocus = result[0]['ocv_irfocus']
                ocv_tvfocus = result[0]['ocv_tvfocus']
                ocv_tvfoci = result[0]['ocv_tvfoci']
                ocv_azimuth = result[0]['ocv_azimuth']
                ocv_devation = result[0]['ocv_devation']
            time.sleep(1)

    def judge__alarm(self):
        """
        判断巡检区是否稳定
        :return:
        """
        state = self.fct_id
        pd = True
        ftc = 0

        if state == "F001" or state == "F002":
            ftc = 1
        t = 0
        ocv_azimuth = ""
        ocv_devation = ""
        ocv_focus = 0
        ocv_foci = 0
        while True:
            szone_devationql = "SELECT * FROM ldata_obs_curview where ocv_obs_id='"
            szone_devationql += self.aas_obs_id + "'"
            result = self.MysqldbHelper.select(szone_devationql)
            if ocv_azimuth != result[0]['ocv_azimuth'] and ocv_devation != result[0]['ocv_devation']:
                ocv_azimuth = result[0]['ocv_azimuth']
                ocv_devation = result[0]['ocv_devation']
            else:
                now_focus = 0
                now_foci = 0
                if ftc == 0:
                    now_focus = int(result[0]['ocv_irfocus'])
                    now_foci = int(result[0]['ocv_irfoci'])
                else:
                    now_focus = int(result[0]['ocv_tvfocus'])
                    now_foci = int(result[0]['ocv_tvfoci'])
                if ocv_focus < now_focus + 100 and ocv_focus > now_focus - 100 and ocv_foci < now_foci + 100 and ocv_foci > now_foci - 100:
                    return pd
                    break
                else:
                    ocv_focus = now_focus
                    ocv_foci = now_foci
            t += 1
            if t == 15:
                pd = False
                break
            time.sleep(1)
        return pd

    def get_obs_curview(self, rs):
        """
        比对视场参数
        :param rs:
        :return:
        """
        obs_curview_sql = "select ocv_azimuth, ocv_devation, ocv_tvfocus, ocv_tvfoci, ocv_irfocus, ocv_irfoci from " \
                          "ldata_obs_curview where ocv_obs_id = '%s'" % self.aas_obs_id
        # print("obs_curview_sql :"+obs_curview_sql)
        re = self.MysqldbHelper.select(obs_curview_sql)
        sums = 0
        if rs is not None and len(rs) != 0:
            for temps in re:
                if temps['ocv_azimuth'] - 0.2 <= rs['zone_azimuth'] <= temps['ocv_azimuth'] + 0.2 and temps[
                    'ocv_devation'] - 0.2 <= rs['zone_devation'] <= temps['ocv_devation'] + 0.2 and temps[
                    'ocv_tvfocus'] - 24 <= rs['zone_tvfocus'] <= temps['ocv_tvfocus'] + 24 and temps[
                    'ocv_tvfoci'] - 34 <= rs['zone_tvfoci'] <= temps['ocv_tvfoci'] + 34 and temps[
                    'ocv_irfocus'] - 35 <= rs['zone_irfocus'] <= temps['ocv_irfocus'] + 35 and temps[
                    'ocv_irfoci'] - 30 <= rs['zone_irfoci'] <= temps['ocv_irfoci'] + 30:
                    sums += 1
        return sums

    def video_draw(self):
        # result = self.get_method2(pumpunit_list[0][3], pumpunit_list[0][5])
        # result = self.get_op(pumpunit_list[0][3], pumpunit_list[0][5])
        # if len(self.pumpunit_list)!=0:
        cap = cv2.VideoCapture(self.rt_stream_path)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if self.video_flag == False:
                cv2.destroyAllWindows()
            else:
                for temp in self.pumpunit_list:
                    cv2.rectangle(frame, (int(temp[2]), int(temp[3])),
                                  (int(temp[4]), int(temp[5])), (255, 255, 255), 2)
                    # cv2.putText(frame,'before', (int(temp[2]), int(temp[3])),cv2.FONT_HERSHEY_COMPLEX, 0.8,(0, 255, 0), 1)
                # Display the resulting frame
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

    def get_videos(self):
        # v_change = threading.Thread(target=self.video_draw)
        # v_change.start()
        pass

    def get_op(self, aas_leftupy, aas_rightdowny):
        pump_height = aas_rightdowny - aas_leftupy
        height = float(pump_height) / float(576)
        x = 0
        if height > 0.8:
            x = 10
        elif height > 0.5:
            x = 20
        elif height > 0.4:
            x = 40
        elif height > 0.3:
            x = 60
        elif height > 0.2:
            x = 80
        elif height > 0.1:
            x = 100
        deviation = self.get_deviation
        result = []
        if x != 0:
            result.append(deviation[0] * x)
            result.append(deviation[1] * x)
        return result

    def get_img(self):
        """
        获取巡检区的截图
        :return:
        """
        cap = cv2.VideoCapture(self.rt_stream_path)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        if cap.isOpened():
            # Capture frame-by-frame
            for i in self.pumpunit_list:
                ret, frame = cap.read()
                if ret:
                    if not os.path.exists(i[0]):
                        os.makedirs(i[0])
                    cv2.rectangle(frame, (int(i[2]), int(i[3])), (int(i[4]), int(i[5])), (255, 255, 255), 4)
                    # Display the resulting frame
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    cv2.imwrite(i[0] + '/' + i[0] + time_str + '.png', frame)
                    cv2.waitKey(10)

    def saw_data(self, temp):
        count = 1
        sql = "UPDATE ldata_routing_inspection  SET ri_clarity = '%d' where ri_id = '%s' and ri_time = '%s' " \
              "and ri_fct_id = '%s' and" \
              " ri_dept_id = '%s' and ri_obs_id = '%s' and ri_zone_id = '%s'" % \
              (count, temp['ri_id'], temp['ri_time'], temp['ri_fct_id'], temp['ri_dept_id'], temp['ri_obs_id'],
               temp['ri_zone_id'])
        try:
            self.MysqldbHelper.insert(sql=sql)
            self.logger.info("清晰度写入成功")
        except:
            self.logger.info("清晰度写入失败")

    def get_new_data(self):
        """
        判断数据库是否有新的数据
        :return:
        """

        self.get_videos()
        t = 0
        while True:
            sql = "select * from ldata_routing_inspection where ri_fct_id='%s' and ri_dept_id='%s'" \
                  "and ri_obs_id='%s' order by ri_time desc limit 1" \
                  % (self.fct_id, self.aas_dept_id, self.aas_obs_id)
            self.logger.info("扫描新的巡检区...")
            self.logger.debug(sql)
            result_ri = self.MysqldbHelper.select(sql)

            # 判断巡检区是否有新数据

            for temp in result_ri:
                # 新的巡检区是否与原巡检区一致
                # if self.current_ri_id == '' or cmp(self.current_ri_id, temp['ri_id']) == -1:
                if temp['ri_handle'] == 0:
                    self.begin_video()
                    self.current_ri_id = temp['ri_id']
                    self.aas_zone_id = temp['ri_zone_id']
                    alarmarea_list = self.select_alarm_params(temp['ri_id'])
                    pumpunit_list = self.select_pump_params()
                    self.logger.info(
                        "巡检发现新的报警区: 报警区坐标: " + str(alarmarea_list) + "  *******  抽油机坐标:" + str(pumpunit_list))
                    self.video_flag = False
                    self.saw_data(temp)
                    # self.get_error()

                    # print("误差为")
                    # print(self.azimuth)
                    # print(self.get_error())

                    pd_obs = self.get_main_alarm()
                    self.logger.info("视场参数匹配成功个数：" + str(pd_obs))
                    # if pd_obs > 0 and self.judge__alarm():
                    if len(alarmarea_list) > 6:
                        # self.flag = False
                        # if not self.judge:
                        #     self.judge = True
                        # else:
                        #     self.judge = False
                        # 更新报警区，调用徐先瑞的程序

                        # self.get_img()
                        self.logger.debug(
                            "报警区坐标参数" + str(alarmarea_list) + "  *******  抽油机坐标参数" + str(pumpunit_list))
                        self.update_ri_handle(temp)
                        self.detection.change_alarm_zone(alarmarea_list, pumpunit_list, self.rt_stream_path)
                        time.sleep(5)
                        self.testing_video()

                        # if (self.judge == True):
                        #     self.flag = True
                        self.video_flag = True
                    else:
                        self.logger.info("当前巡检区没有报警区坐标或抽油机坐标")
                        # else:
                        #     self.logger.info("视场参数不匹配")
            try:
                time.sleep(self.second)
            except KeyboardInterrupt as e:
                pass
