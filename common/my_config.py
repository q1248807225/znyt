# -*- coding:utf-8 -*-
import ConfigParser


class MyConfig(ConfigParser.ConfigParser):
    """
    解析本项目的所有配置信息，所有变化的参数请从此类读取
    """

    conf_file = "../conf/my.ini"

    def __init__(self, filename=None, defaults=None):
        ConfigParser.ConfigParser.__init__(self, defaults=defaults)
        # 读取配置文件
        if filename is not None:
            MyConfig.conf_file = filename
        self.read(MyConfig.conf_file)

    def optionxform(self, optionstr):
        """
        重写ConfigParser的方法，原始返回的内容都是小写lower()，重写后就原样返回
        :param optionstr:
        :return:
        """
        return optionstr

    def get_med_id(self):
        """
        获取机器编号
        :return:
        """
        config = ConfigParser.ConfigParser()
        config.readfp(open('../conf/my.ini'))
        a = config.get("med_id", "med")

        return a

    def get_db_ip(self):
        """
        获取数据库ip地址
        :return:
        """
        return self.get("db", "ip")

    def get_db_port(self):
        """
        获取数据库端口
        :return:
        """
        return int(self.get("db", "port"))

    def get_db_username(self):
        """
        获取数据库用户名
        :return:
        """
        return self.get("db", "username")

    def get_db_password(self):
        """
        获取数据库密码
        :return:
        """
        return self.get("db", "password")

    def get_db_name(self):
        """
        获取数据库名
        :return:
        """
        return self.get("db", "db_name")

    def get_db_charset(self):
        """
        获取数据库字符集
        :return:
        """
        return self.get("db", "charset")

    # 内部类定义channel，进行channel的简单封装
    class Channel:
        def __init__(self, aas_dept_id, aas_obs_id, fct_id, stream):
            self.aas_dept_id = aas_dept_id
            self.aas_obs_id = aas_obs_id
            self.fct_id = fct_id
            self.stream = stream

        def get_aas_dept_id(self):
            return self.aas_dept_id

        def get_aas_obs_id(self):
            return self.aas_obs_id

        def get_fct_id(self):
            return self.fct_id

        def get_stream(self):
            return self.stream

        def __repr__(self):
            return "(%s, %s, %s, %s)" % (self.aas_dept_id, self.aas_obs_id, self.fct_id, self.stream)

    def get_channels(self, channel_str_list=None):
        """
        获取channels的信息，
        :param channel_str_list:如果为None，则返回所有channels；如果为一个列表，则返回指定key(s)的channel(s)信息
        :return: 型如：Channel对象列表
        """
        channels = []
        # 获取channel部分中的所有配置的“路”参数
        for channel in self.options('channels'):
            if channel_str_list is None or (channel_str_list is not None and channel in channel_str_list):
                stream = self.get('channels', channel)
                if stream is None:
                    print("配置文件出错:[channels]片段中不存在%s" % channel)
                    exit(-1)
                else:
                    # 部门编号
                    aas_dept_id = channel.split("_")[0]
                    # 光电编号
                    aas_obs_id = channel.split("_")[1]
                    # 影响因素编号
                    fct_id = channel.split("_")[2]
                # 创建自定义类Channel对象，对4个参数进行封装
                channels.append(MyConfig.Channel(aas_dept_id, aas_obs_id, fct_id, stream))
        return channels

def get_stream(self, channel_str):
        """
        获取某路channel对应的stream
        :param channel_str:型如 '1010_1_F001'
        :return:
        """
        return self.get("channels", channel_str)

