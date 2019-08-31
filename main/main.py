# -*- coding:utf-8 -*-


import init_path
from common.my_config import MyConfig
import argparse
from common import my_logging
import subprocess


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Based on Deep Learning Automatic Data Analysis '
                                                 'Platform for Oilfield Monitoring Video')
    parser.add_argument('--channels', dest='channels', help='(e.g.)1010_1_F001,1010_2_F003')
    return parser.parse_args()


if __name__ == '__main__':
    config = MyConfig("../conf/my.ini")

    args = parse_args()
    channels_str = args.channels

    # 判断是否是用命令行参数形式进行启动
    if channels_str is not None:
        channels_str_list = list(channels_str.split(","))
        channels = config.get_channels(channels_str_list)
    # 按配置中配置启动
    else:
        channels = config.get_channels()

    # 子进程列表
    sub_process_list = []

    # 每个channel分配一路处理程序
    for channel in channels:
        # 获取部门编号
        aas_dept_id = channel.get_aas_dept_id()
        # 获取光电编号
        aas_obs_id = channel.get_aas_obs_id()
        # 获取影响因素编号
        fct_id = channel.get_fct_id()
        # 获取视频流地址
        stream = channel.get_stream()

        logger = my_logging.get_logger(aas_dept_id + "_" + aas_obs_id + "_" + fct_id)

        # 调用dqsy/main_thread.py，来处理一路
        command = "python ../dqsy/main_thread.py --aas_dept_id %s --aas_obs_id %s --fct_id %s --stream %s"\
                  % (aas_dept_id, aas_obs_id, fct_id, stream)

        # 以非阻塞方式启动子进程
        sub_process = subprocess.Popen(command.split())
        sub_process_list.append(sub_process)
        logger.info("部门[%s],光电[%s],因素[%s]的视频自动分析程序已启动,进程号:%s，关闭进程使用：kill -9 %d"
                    % (aas_dept_id, aas_obs_id, fct_id, sub_process.pid, sub_process.pid))
        logger.debug("启动命令:%s" % command)

    # 必须所有子进程结束，主进程才可以结束
    for sub_process in sub_process_list:
        #体面的推出
        try:
            sub_process.wait()
        except KeyboardInterrupt as e:
            pass

    my_logging.get_logger().info("主程序结束")
