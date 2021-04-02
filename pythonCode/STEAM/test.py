from threading import Thread,Event,Condition
import threading
import os
import struct,time
import wx
from concurrent.futures import ThreadPoolExecutor
def bin_data_read(file_path):
        t = time.time()
        tid = threading.get_ident()
        print("load thread id : %d"%tid)
        data_bin = open(file_path, 'rb+')
        data_size = os.path.getsize(file_path)
        
        data_total = data_bin.read(data_size)

        data_tuple = struct.unpack(str(int(data_size/4))+'f', data_total)
        # print(len(data_tuple))
        print("load time %.3f" %(time.time()-t))
        return data_tuple
class bThread(Thread):
    def __init__(self,name):
        super(bThread,self).__init__()
        self.daemon = True
        self.name = name
        
        tid = threading.get_ident()
        print("b thread id : %d"%tid)

    def run(self):
        file2 = r"E:\Tianjin\data\123\temp\100001.bin"
        file1 = r"E:\Tianjin\data\123\wave20210316_T103657_160_1.Wfm.bin"
        file3 = r"G:\天津\原始数据\第二次原始数据\208\50\wave20201123_T112443_50_1.Wfm.bin"
        task1 = RecoverThread("1",file3)
        task1.setDaemon(True)
        task1.start()

def concurrent_test():  #异步进程
   with ThreadPoolExecutor(max_workers=100) as e:
        file2 = r"E:\Tianjin\data\123\temp\100001.bin"
        file1 = r"E:\Tianjin\data\123\wave20210316_T103657_160_1.Wfm.bin"
        file3 = r"G:\天津\原始数据\第二次原始数据\208\50\wave20201123_T112443_50_1.Wfm.bin"
        e.submit(bin_data_read,file3)
        # data = bin_data_read(file3)
        # print(len(data))

class RecoverThread(Thread):
    def __init__(self,name,filePath):
        super(RecoverThread,self).__init__()
        self.filePath = filePath
        self.pauseEvent = Event()
        self.stopEvent = Event()
        self.daemon = True
        self.name = name
        
        tid = threading.get_ident()
        print("recover thread id : %d"%tid)

    def run(self):
        # time.sleep(10)
        # print("end time")
        
        data = bin_data_read(self.filePath)
        print(len(data))
        print("%s end run"% self.name)
        while(True):
            time.sleep(0.5)
            print(self.name)

class MyFrame(wx.Frame):
    def __init__(self,parent,title):
        super(MyFrame,self).__init__(parent,title=title)
        panel = wx.Panel(self)
        btn = wx.Button(panel,size=(90,30),pos = (50,50))
        btn.SetLabel("button")
        btn.Bind(wx.EVT_BUTTON,self.Onload)
        btn2 = wx.Button(panel,size=(90,30),pos = (150,50))
        btn2.SetLabel("print")
        btn2.Bind(wx.EVT_BUTTON,self.print)

    def print(self,e):
        print("ydc %d" % threading.active_count ())
    def Onload(self,e):
        print("OnLoad")
        concurrent_test()

def  main():
    
    app = wx.App()
    ex = MyFrame(None, title='Vision Speed')
    # ex= RecoveryFrame(None,RecoveryParmas())

    ex.Show()
    app.MainLoop()



if __name__ == '__main__':
    main()
    # file2 = r"E:\Tianjin\data\123\temp\100001.bin"
    # file1 = r"E:\Tianjin\data\123\wave20210316_T103657_160_1.Wfm.bin"
    # task1 = RecoverThread("1",file1)
    # task1.setDaemon(True)
    # task1.start()
    # task2 = RecoverThread("2",file2)
    # task2.setDaemon(True)
    # task2.start()

    
    # while(True):
    #     time.sleep(1)
    #     print("loop")
