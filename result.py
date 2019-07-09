# -*- coding:utf-8 -*-
from common.my_dao import *
import time
import random
from common import my_logging


# 处理结果更新数据库类
class ResultHandler:

    def __init__(self):
        # 日志对象声明
        self.logger = None
        # 数据库工具
        self.MysqldbHelper = None
        #记录人的处理结果总数
        self.peopleCount = 0
        #记录车的处理结果总数
        self.carCount = 0
        #记录抽油机状态处理结果总数
        self.pumpCount = 0
        #记录处理结果总数
        self.count = 0

    # 增加巡检处理记录
    def update_area_status(self, alarms, logger_name):
        """
        增加巡检处理记录
        :param alarms:
        :param logger_name:
        :return:
        """
        if self.logger is None:
            self.logger = my_logging.get_logger(logger_name)
            self.MysqldbHelper = MysqldbHelper(self.logger)
        alarms_car_count = 0
        alarms_person_count = 0
        ri_id = ""
        for temp in alarms:
            if temp[2] == 'person':
                ri_id = temp[0]
                alarms_person_count = alarms_person_count + temp[3]
                self.peopleCount += 1
                self.logger.info("人的处理结果总和:"+ str(self.peopleCount))
            elif temp[2] == 'car':
                alarms_car_count = alarms_car_count + temp[3]
                self.carCount += 1
                self.logger.info("车的处理结果总和："+str(self.carCount))
        self.count = self.carCount + self.peopleCount + self.pumpCount
        self.logger.info('处理结果总和:'+str(self.count))
        if alarms_car_count != 0 or alarms_person_count != 0:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            times = random.uniform(10, 20)

            ip_sql = "INSERT INTO ldata_inspection_process_record (ri_id, ipr_id, ipr_datetime, ipr_people, ipr_car) " \
                     "VALUES ('%s','%s', '%s', '%d', '%d')" % (
                     alarms[0][0], times, time_str, alarms_person_count, alarms_car_count)
            self.logger.debug("对巡检区结果表进行插入人车数量 :"+ip_sql)
            self.MysqldbHelper.insert(ip_sql)
        if len(alarms) != 0:
            self.update_zone_curresult(alarms[0][1], alarms_person_count, alarms_car_count, ri_id)

    def update_zone_curresult(self, zcr_zone_id, alarms_person_count, alarms_car_count, ri_id):
        """
        更新当前结果人车数量
        :param zcr_zone_id:
        :param alarms_person_count:
        :param alarms_car_count:
        :param logger_name:
        :return:
        """

        zcr_sql = "select * from ldata_zone_curresult where zcr_zone_id = '%s'" % zcr_zone_id
        # print("attention"+zcr_sql)
        time_strs = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

        res = self.MysqldbHelper.select(zcr_sql)
        if len(res) != 0:
            ld_sql = "UPDATE ldata_zone_curresult SET zcr_updatetm = '%s' ,zcr_people = '%d' " \
                     ", zcr_car = '%d' ,zcr_flag = '%d' ,zcr_ri_id = '%s' WHERE zcr_zone_id = '%s' " % \
                     (time_strs, alarms_person_count, alarms_car_count, 1, ri_id, zcr_zone_id)
            print(ld_sql)
            self.logger.debug("更新当前报警区结果表人车时间 ："+ld_sql)
            self.MysqldbHelper.update(ld_sql)
        else:
            times = time.time()
            ld_sql = "INSERT INTO ldata_zone_curresult (zcr_zone_id, zcr_updatetm, zcr_people, zcr_car, zcr_flag, zcr_ri_id) " \
                     "VALUES ('%s', '%s', '%d', '%d', '%d', '%s')" % (zcr_zone_id, time_strs, alarms_person_count, alarms_car_count, 1, ri_id)
            self.logger.debug("增加当前报警区结果表人车时间 ：" + ld_sql)
            self.MysqldbHelper.insert(ld_sql)

    def update_pumps_curresult(self, pumps, logger_name):
        """
        更新抽油机状态记录
        :param pumps:
        :param logger_name:
        :return:
        """
        if self.logger is None:
            self.logger = my_logging.get_logger(logger_name)
            self.MysqldbHelper = MysqldbHelper(self.logger)
        global pump_sql, ld_sql, time_str
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        for temp in pumps:
            if len(temp) != 0:
                pump_sql = "select * from ldata_pumpunit_curresult " \
                           "where zcr_zone_id = '%s' and pcr_pump_id = '%s' " % (temp[1], temp[2])
                res = self.MysqldbHelper.select(pump_sql)
                if len(res) != 0:
                    ld_sql = "UPDATE ldata_pumpunit_curresult " \
                             "SET pcr_status = '%s' , pcr_updatetm = '%s', pcr_flag = '%s', pcr_ri_id = '%s' where zcr_zone_id = '%s' and pcr_pump_id = '%s' " % \
                             (temp[3], time_str, 1, temp[0], temp[1], temp[2])
                    print("update_pumps_curresult:" + ld_sql)
                    self.logger.info("更新当前巡检区结果表人车时间 ："+ld_sql)
                    self.MysqldbHelper.update(ld_sql)
                    if temp[3] == 0:
                        self.insert_inspection_process_record(temp, time_str)
                else:
                    ldata_zone_curresult_isnull_sql = "select * from ldata_zone_curresult where zcr_zone_id = '%s'" % \
                                                      temp[1]
                    isnull = self.MysqldbHelper.select(ldata_zone_curresult_isnull_sql)
                    print(ldata_zone_curresult_isnull_sql)
                    if len(isnull) == 0:
                        time_strs = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                        ld_sql = "INSERT INTO ldata_zone_curresult (zcr_zone_id, zcr_updatetm, zcr_people, zcr_car, zcr_flag, zcr_ri_id) " \
                                 "VALUES ('%s', '%s', '%d', '%d', '%d', '%s')" % (temp[1], time_strs, 0, 0, 1, temp[0])
                        print(ld_sql)
                        self.MysqldbHelper.insert(ld_sql)
                    ld_sql = "INSERT INTO ldata_pumpunit_curresult (zcr_zone_id,pcr_updatetm, pcr_pump_id, pcr_status, pcr_flag, pcr_ri_id)" \
                             " VALUES('%s', '%s', '%s', '%d', '%d', '%s')" % (temp[1], time_str, temp[2], int(temp[3]), 1, temp[0])
                    print("update_pumps_curresult:" + ld_sql)
                    self.logger.debug("增加当前巡检区结果表人车时间 ：" + ld_sql)
                    self.MysqldbHelper.insert(ld_sql)
                    if temp[3] == 0:
                        self.insert_inspection_process_record(temp, time_str)

    def insert_inspection_process_record(self, temp, time_str):
        times = time.time()
        self.pumpCount += 1
        self.logger.info('抽油机处理结果总和：'+str(self.pumpCount))
        cd_sql = "INSERT INTO ldata_inspection_process_record " \
                 "(ri_id, ipr_id, ipr_datetime, ipr_pumpid, ipr_pumpstatus) " \
                 "VALUES ('%s','%s', '%s', '%s', '%d')" % \
                 (temp[0], times, time_str, temp[2], int(temp[3]))
        self.logger.debug("对巡检区结果表进行插入抽油机状态 :" + cd_sql)
        self.MysqldbHelper.insert(cd_sql)

