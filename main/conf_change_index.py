
# -*- coding: utf-8 -*-

import ConfigParser
from configobj import ConfigObj
from common import my_logging
from common.my_dao import MysqldbHelper
import cv2
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
class ConfChange:
    logger = my_logging.get_logger()
    MysqldbHelper = MysqldbHelper(logger)
    arr0 = [  "rtsp://admin:hpws12345@192.168.1.33:554/h264/ch1/main/av_stream"
            , "rtsp://admin:hpws12345@192.168.1.31:554/h264/ch1/main/av_stream"
            , "rtsp://admin:hpws12345@192.168.1.36:554/h264/ch1/main/av_stream"
            , "rtsp://admin:hpws12345@192.168.1.35:554/h264/ch1/main/av_stream"
            , "rtsp://admin:hpws12345@192.168.1.39:554/h264/ch1/main/av_stream"
            , "rtsp://admin:admin12345@192.168.1.70:554/h264/ch1/main/av_stream"
            , "rtsp://admin:admin12345@192.168.1.73:554/h264/ch1/main/av_stream"
            ]
    arr1 = [  "rtsp://admin:hpws12345@192.168.1.32:554/h264/ch1/main/av_stream"
            , "rtsp://admin:hpws12345@192.168.1.30:554/h264/ch1/main/av_stream"
            , "rtsp://admin:hpws12345@192.168.1.37:554/h264/ch1/main/av_stream"
            , "rtsp://admin:hpws12345@192.168.1.34:554/h264/ch1/main/av_stream"
            , "rtsp://admin:hpws12345@192.168.1.38:554/h264/ch1/main/av_stream"
            , "rtsp://admin:admin12345@192.168.1.71:554/h264/ch1/main/av_stream"
            , "rtsp://admin:admin12345@192.168.1.72:554/h264/ch1/main/av_stream"
            ]
    def get_ini(self):
        szone_devationql = "SELECT * FROM ldata_effect_factor;"
        result = self.MysqldbHelper.select(szone_devationql)
        list1 = []
        for index in range(len(result)):
           list1.append(result[index]['fct_id'])
        szone_devationql = "SELECT * FROM ldata_obs_curview;"
        result = self.MysqldbHelper.select(szone_devationql)
        list2 = []
        for index in range(len(result)):
            list2.append(result[index]['ocv_obs_id'])
        sum=[]
        sum.append(list1)
        sum.append(list2)
        return sum
    def get_list(self):
        list = []
        config = ConfigParser.ConfigParser()
        config.readfp(open('../conf/my.ini'))
        a = config.options("channels")
        for index in range(len(a)):
            b = config.get("channels", a[index])
            shu = b.split("/");
            end = shu[2].find(":")
            username = shu[2][0:end];
            shu2 = shu[2].split("@")
            password = shu2[0][end + 1:len(shu2[0])]
            ip0 = shu2[1]
            temp = ip0.split(":")
            ip = temp[0]
            code = shu[5]
            arr = a[index].split('_')
            arr[2] = arr[2].upper()
            strs = "部门编号:"+str(arr[0])+",光电编号:"+str(arr[1])+",因素编号:"+str(arr[2])+",用户名："+str(username)+",密码:"+str(password)+",ip:"+str(ip)
            list.append(strs)
        return list


    def get_section(self,id):
        list = []
        config = ConfigParser.ConfigParser()
        config.readfp(open('../conf/my.ini'))
        a = config.options("channels")
        for index in range(len(a)):
            b = config.get("channels", a[index])
            shu=b.split("/");
            end=shu[2].find(":")
            username=shu[2][0:end];
            shu2=shu[2].split("@")

            password=shu2[0][end+1:len(shu2[0])]

            ip0=shu2[1]
            temp=ip0.split(":")
            ip=temp[0]
            code=shu[5]
            arr = a[index].split('_')
            arr[2]=arr[2].upper()
            list.append({
                'dept': arr[0],
                'obs': arr[1],
                'fct': arr[2],
                'username':username,
                'ip':ip,
                'password':password,
                'code':code
            });
        if(len(a)==0):
            list.append({
                'dept': "",
                'obs': "",
                'fct': "",
                'username': "",
                'ip': "",
                'password': "",
                'code':""
            });
            return list[0]
        return list[id]

    def edge(self,id):
        config2 = ConfigParser.ConfigParser()
        config2.readfp(open('../conf/my.ini'))
        a = config2.options("channels")
        if(id<0 or len(a)==0):
            return -1
        elif(id>=len(a)):
            return 1
        else:
            return 0

    def video_test(self, string):
        cap = cv2.VideoCapture(string)

        if cap.isOpened():
            return True
        return False


    def delete(self,id):
        list = []
        config2 = ConfigParser.ConfigParser()
        config2.readfp(open('../conf/my.ini'))
        a = config2.options("channels")
        if(id<0 or id>=len(a)):
            list.append({
                'dept': "",
                'obs': "",
                'fct': "",
                'username': "",
                'ip': "",
                'password': "",
                'code': "",
                'id': 0
            });
            return list
        b=a[id].upper()
        conf_ini="../conf/my.ini"
        config=ConfigObj(conf_ini, encoding='UTF8')
        del config['channels'][b]
        config.write()
        edge=self.edge(id);
        if(edge==-1):
            list.append({
                'dept':"",
                'obs':"",
                'fct':"",
                'username':"",
                'ip': "",
                'password':"",
                'code': "",
                'id':0
            });
            return list[0];
        elif(edge==0):
            list = self.get_section(id)
            list.setdefault('id',id)
            return list
        else:
            list=self.get_section((id-1))
            list.setdefault('id',(id-1))
            return list

    def insert(self,arr):
        obs_id = arr[0]
        fct_id = arr[1]
        key=""
        value=""
        if fct_id=='F001':
            key="1010_"+str(obs_id)+"_F001"
            value=self.arr0[int(obs_id)-1]
        else:
            key = "1010_" + str(obs_id) + "_F003"
            value = self.arr1[int(obs_id)-1]

        conf_ini="../conf/my.ini"
        config=ConfigObj(conf_ini, encoding='UTF8')

        config['channels'][key]=value
        config.write()
        return True

    def update(self, arr, id):
        obs_id = arr[0]
        fct_id = arr[1]
        key = ""
        value = ""
        if fct_id == 'F001':
            key = "1010_" + str(obs_id) + "_F001"
            value = self.arr0[int(obs_id)-1]
        else:
            key = "1010_" + str(obs_id) + "_F003"
            value = self.arr1[int(obs_id)-1]

        self.delete(id)
        conf_ini="../conf/my.ini"
        config=ConfigObj(conf_ini, encoding='UTF8')
        config['channels'][key]=value
        config.write()
        config2 = ConfigParser.ConfigParser()
        config2.readfp(open('../conf/my.ini'))
        a = config2.options("channels")
        return (len(a)-1)
