# -*- coding:utf-8 -*-
import os
import init_path
from tkinter import *
import tkinter.messagebox
from conf_change_index import ConfChange
import tkinter.messagebox as messagebox

from subprocess import call,Popen
from  tkinter  import ttk

class Application(Frame):   #从Frame派生出Application类，它是所有widget的父容器
    def __init__(self, master=None):
        self.fct_list = ("可见光","红外")
        self.ass_list = []
        self.con = ConfChange()
        self.index = 0
        list = self.con.get_section(self.index)
        if list['fct'] == 'F001':
            self.fct = '可见光'
        elif list['fct'] == 'F003':
            self.fct = '红外'
        else:
            self.fct = ''
        self.obs = list['obs']
        Frame.__init__(self, master)
        self.mylist = tkinter.Listbox(self, width=150,height=30)  # 列表框
        self.show_list()
        self.mylist.pack(pady=100)
        self.lists = self.con.get_ini()

          # 将widget加入到父容器中并实现布局
        self.createWidget0()
        # self.createWidget1()
        self.pack()
        self.fct_C.set(self.fct)
        self.quitButton = Button(self, text='退出', command=self.quit)
        self.quitButton.pack()

    def show_list(self):
        list = self.con.get_list()
        for item in list:  # 插入内容
            self.mylist.insert(tkinter.END, item)  # 从尾部插入

    def createWidget0(self):


        self.fctLabel = Label(self, text='因素编号')
        self.fctLabel.pack(side=LEFT)
        comvalue = tkinter.StringVar()  # 窗体自带的文本，新建一个值
        self.fct_C = ttk.Combobox(self, textvariable=comvalue)  # 初始化
        self.fct_C["values"] = self.fct_list
        self.fct_C.pack(side=LEFT)

        self.obsLabel = Label(self, text='光电参数')
        self.obsLabel.pack(side=LEFT)
        self.obs_input = Entry(self)
        self.obs_input.pack(side=LEFT)
        self.obs_input.insert(0, self.obs)

        fButton = Button(self, text='上一条', command=lambda: self.sx(self.index - 1))
        fButton.pack(side=LEFT)
        eButton = Button(self, text='下一条', command=lambda: self.sx(self.index + 1))
        eButton.pack(side=LEFT)
        gButton = Button(self, text='更改', command=lambda:self.change())
        gButton.pack(side=LEFT)
        dButton = Button(self, text='删除', command=lambda: self.delete(self.index))
        dButton.pack(side=LEFT)
        sButton = Button(self, text='添加', command=lambda: self.inserts())
        sButton.pack(side=LEFT)
        pButton=Button(self, text='打开主程序', command=lambda:self.callback())
        pButton.pack(side=LEFT)
        kButton = Button(self, text='关闭主程序', command=lambda: self.kill())
        kButton.pack(side=LEFT)
    def callback(self):
        cmd="python ../main/main.py"
        Popen(cmd.split())
        print('Document start working....')

    def kill(self):
        out = os.popen("nvidia-smi").read()
        arr = [];

        for line in out.splitlines():
            print(line)
            if 'python' in line:
                arr = line.split(" ")
                print (arr)
                os.popen("kill " + arr[9]).read()
    def inserts(self):

        top = Toplevel()
        top.title('添加')
        v1 = StringVar()
        self.fctLabel1 = Label(top, text='因素编号')
        self.fctLabel1.pack()
        comvalue1 = tkinter.StringVar()  # 窗体自带的文本，新建一个值
        self.fct_input1 = ttk.Combobox(top, textvariable=comvalue1)  # 初始化
        self.fct_input1["values"] = self.lists[0]
        self.fct_input1.current(0)  # 选择第一个
        self.fct_input1.bind("<<ComboboxSelected>>")  # 绑定事件,(下拉列表框被选中时，绑定go()函数)
        self.fct_input1.pack()
        self.obsLabel1 = Label(top, text='光电编号')
        self.obsLabel1.pack()
        comvalue2 = tkinter.StringVar()  # 窗体自带的文本，新建一个值
        self.obs_input1 = ttk.Combobox(top, textvariable=comvalue2)  # 初始化
        self.obs_input1["values"] = self.lists[1]
        self.obs_input1.current(0)  # 选择第一个
        self.obs_input1.bind("<<ComboboxSelected>>")  # 绑定事件,(下拉列表框被选中时，绑定go()函数)
        self.obs_input1.pack()


        fButton = Button(top, text='添加', command=lambda: self.insert())
        fButton.pack()


    def insert(self):
        arr = []

        arr.append(self.obs_input1.get())
        if self.fct_C.get() == '可见光':
            arr.append('F001')
        elif self.fct_C.get() == '红外' :
            arr.append('F003')

        if self.con.insert(arr):

            tkinter.messagebox.showinfo('提示', '保存成功')
            self.mylist.delete(0, END)
            self.show_list()
        else :
            tkinter.messagebox.showinfo('提示', '保存失败')


    def sx(self, num):
        # 获取数据接口
        if self.con.edge(num) == 0:
            list  = self.con.get_section(num)
            self.updata(list)
            self.update()
            self.index = num

    def updata(self, list):

        self.obs = list['obs']
        if list['fct'] == 'F001':
            self.fct = '可见光'
        else :
            self.fct = '红外'


    def update(self):
        self.obs_input.delete(0, END)
        self.obs_input.insert(0, self.obs)
        self.fct_C.set(self.fct)



    def delete(self, num):
        if self.con.edge(num) == 0:
            list = self.con.delete(num)
            self.updata(list)
            self.update()
            self.index = list['id']
            tkinter.messagebox.showinfo('提示', '删除成功')
            self.mylist.delete(0, END)
            self.show_list()


    def change(self):
        arr = []

        arr.append(self.obs_input.get())
        if self.fct_C.get() == '可见光':
            arr.append('F001')
        elif self.fct_C.get() == '红外' :
            arr.append('F003')
        arr.append(self.fct_C.get())
        flag = self.con.update(arr, self.index)
        if flag != -1:
            self.index = flag
            tkinter.messagebox.showinfo('提示', '修改成功')
            self.mylist.delete(0, END)
            self.show_list()
        else:
            tkinter.messagebox.showinfo('提示', '修改失败')
        # 更新接口

app = Application()
app.master.title("配置文件窗口")#窗口标题

app.mainloop()#主消息循环
