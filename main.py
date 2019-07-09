# -*- coding:utf-8 -*-

import os.path as osp
import sys
import subprocess


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

add_path(osp.join(this_dir, '..'))
from common.my_dao import MysqldbHelper
from common.my_config import MyConfig
import argparse
from common import my_logging
import time
import threading



class start_main:
    """
    启动程序类
    """
    def __init__    (self):
        # 调用日志类
        self.logger = my_logging.get_logger(None)
        # 数据库类
        self.mysqldbHelper = MysqldbHelper(self.logger)
        # 存入当前执行的配置参数，是个字典，格式例如:fct_id + obs_id + dept_id , F001110101
        self.list = {}

        # 当前机器编号

        self.med_id = int(self.set_med_id())
        # 检测开关
        self.flag = True

    def implement(self):
        """
        无限检测数据库的变化
        :return:
        """
        while self.flag:
            self.check_start()
            time.sleep(1)


    def check_start(self):
        """
        检查是否新的巡检需要计算，当存在时，调用启动程序函数
        :return:
        """
        check_sql = "select * from ldata_config where program_type = '%d' and machine_id = '%d'" % (1, self.med_id)
        result = self.mysqldbHelper.select(check_sql)
        for temp in result:
            strings = ""
            strings += temp['fct_id']
            strings += temp['obs_id']
            strings += temp['dept_id']
            # 判断查询是否有新的配置参数
            if len(self.list) < 3:
                 if strings not in self.list.keys():
                    self.list[strings] = 1
                    self.start(temp, strings)
                    self.update_handle(temp)
                    # command = "python ../dqsy/main_thread.py --aas_dept_id %s --aas_obs_id %s --fct_id %s --stream %s" \
                    # % (aas_dept_id, aas_obs_id, fct_id, stream)

    def start(self, params, strings):
        """
        启动程序
        :param params:
        :return:
        """
        command = "python ../dqsy/main_thread.py --aas_dept_id %s --aas_obs_id %s --fct_id %s --stream %s" \
            % (params['dept_id'], params['obs_id'], params['fct_id'], params['video_url'])
        sub_process = subprocess.Popen(command.split())

        log = my_logging.get_logger(params['dept_id'] + "_" + params['obs_id'] + "_" + params['fct_id'])
        fct = params['fct_id']
        dept = params['dept_id']
        obs = params['obs_id']
        # log.info("部门[%s],光电[%s],因素[%s]的视频自动分析程序已启动,进程号:%d，关闭进程使用：kill -9 %d"
        #             % (params['dept_id'], params['obs_id'], params['fct_id'], sub_process.pid, sub_process.pid))
        self.list[strings] = sub_process.pid
        self.update_pid(sub_process.pid, params)
        # log.debug("启动命令:%s" % command)

    def update_pid(self, pid , params):
        """
        当程序启动后，程序数据库的进程号
        :param pid:
        :param params:
        :return:
        """

        pid_sql = "update ldata_config set pid = '%s' where fct_id = '%s' and obs_id = '%s' and dept_id = '%s'" % (str(pid), params['fct_id'], params['obs_id'], params['dept_id'])
        self.mysqldbHelper.update(pid_sql)

    def update_handle(self, params):
        """
        当程序启动后，对数据库进行反馈
        :param params:
        :return:
        """
        handle_sql = "update ldata_config set program_handle = '%d' where fct_id = '%s' and obs_id = '%s' and dept_id = " \
                     "'%s'" % (1, params['fct_id'], params['obs_id'], params['dept_id'])
        self.mysqldbHelper.update(handle_sql)

    def implement_1(self):
        """
        无限检测数据库数据
        :return:
        """
        while self.flag:
            self.check_close()
            time.sleep(1)

    def check_close(self):
        """
        检测是否有需要关闭的程序
        :return:
        """
        check_sql = "select * from ldata_config where program_type = '%d' and machine_id = '%d'" % (0, self.med_id)
        result = self.mysqldbHelper.select(check_sql)
        for temp in result: # rtsp://admin:hpws12345@192.168.1.33:8000/h264/ch1/main/av_stream
            strings = ""
            strings += temp['fct_id']
            strings += temp['obs_id']
            strings += temp['dept_id']
            if strings in self.list.keys():
                self.close(temp, strings)
                self.list.pop(strings)

    def close(self, params, strings):
        """
        关闭程序函数

        :param params:
        :return:
        """
        pid = int(self.list[strings])
        command = "kill -9 %d" % int(self.list[strings])
        sub_process = subprocess.Popen(command.split())
        print (command)
        log = my_logging.get_logger(params['dept_id'] + "_" + params['obs_id'] + "_" + params['fct_id'])
        # log.info("部门[%s],光电[%s],因素[%s]的视频自动分析程序已关闭,进程号:%s"
        #          % (params['dept_id'], params['obs_id'], params['fct_id'], sub_process.pid))pid
        self.update_close_handle(params)
        # log.debug("关闭命令:%s" % command)

    def update_close_handle(self, params):
        """
        将数据库中的处理字段更改
        :param params:
        :return:
        """
        handle_sql = "update ldata_config set program_handle = '%d' where fct_id = '%s' and obs_id = '%s' and dept_id = '%s'" % (0, params['fct_id'] ,params['obs_id'], params['dept_id'])
        self.mysqldbHelper.update(handle_sql)

    def update_close_pid(self, params):
        pid_sql = "update ldate_config set pid = '%s' where fct_id = '%s' and obs_id = '%s' and dept_id = '%s'" % (0, params['fct_id'], params['obs_id'], params['dept_id'])
        self.mysqldbHelper.update(pid_sql)
    def set_med_id(self):
        c = MyConfig()
        return c.get_med_id()

if __name__ == '__main__':

    start = start_main()

    t1 = threading.Thread(target=start.implement)
    t1.start()
    t2 = threading.Thread(target=start.implement_1)
    t2.start()
