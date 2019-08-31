# -*- coding:utf-8 -*-
import MySQLdb
from common.my_config import MyConfig
import traceback
from MySQLdb.cursors import DictCursor
from common import my_logging
from DBUtils.PooledDB import PooledDB


class MysqldbHelper:
    """
    数据库工具类
    """
    config = MyConfig("../conf/my.ini")
    ip = config.get_db_ip()
    port = config.get_db_port()
    username = config.get_db_username()
    password = config.get_db_password()
    db_name = config.get_db_name()
    charset = config.get_db_charset()

    __pool = None

    def __init__(self, logger):
        # 数据库构造函数，从连接池中取出连接，并生成操作游标
        self.logger = logger
        logger.info("数据库连接池创建中...：（%s:%s@%s:%s/%s?charset=%s）" %
                    (MysqldbHelper.username, MysqldbHelper.password, MysqldbHelper.ip,
                     MysqldbHelper.port, MysqldbHelper.db_name, MysqldbHelper.charset))


    @staticmethod
    def __get_conn():
        """
        @summary: 静态方法，从连接池中取出连接
        @return MySQLdb.connection
        """
        if MysqldbHelper.__pool is None:
            MysqldbHelper.__pool = PooledDB(creator=MySQLdb, mincached=1, maxcached=20, host=MysqldbHelper.ip, user=MysqldbHelper.username,
                               passwd=MysqldbHelper.password, db=MysqldbHelper.db_name, port=MysqldbHelper.port, charset=MysqldbHelper.charset, cursorclass=DictCursor)
        return MysqldbHelper.__pool.connection()

    def select(self, sql):
        try:
            conn = MysqldbHelper.__get_conn()
            cursor = conn.cursor()
            count = cursor.execute(sql)
            fc = cursor.fetchall()
            conn.commit()
            return fc
        except MySQLdb.Error, e:
            self.logger.error("Mysqldb Error:%s" % e)
            self.logger.error(traceback.print_exc())
        finally:
            cursor.close()
            conn.close()

    # 不带参数的更新方法
    def update(self, sql):
        try:
            conn = MysqldbHelper.__get_conn()
            cursor = conn.cursor()
            count = cursor.execute(sql)
            conn.commit()
            return count
        except MySQLdb.Error, e:
            self.logger.error("Mysqldb Error:%s" % e)
            self.logger.error(traceback.print_exc())
        finally:
            cursor.close()
            conn.close()

    def insert(self, sql):
        try:
            conn = MysqldbHelper.__get_conn()
            cursor = conn.cursor()
            count = cursor.execute(sql)
            conn.commit()
            return count
        except MySQLdb.Error, e:
            self.logger.error("Mysqldb Error:%s" % e)
            self.logger.error(traceback.print_exc())
        finally:
            cursor.close()
            conn.close()

