# import myFunction as MF
from threading import Thread
import wx
import time

print("ydc")

global GRecoverFrameNum
GRecoverFrameNum = 0
class MyThread(Thread):
    def __init__(self, name="Python"):
        super().__init__()
        self.name=name
        self.daemon = True
    def run(self):
        for i in range(5):
            print("hello", self.name)         
            time.sleep(2)

class RecoverFrame(wx.Frame):
    def __init__(self,parent,title):
        super(RecoverFrame,self).__init__(parent,title=title)

        
        self.CreatePanel()
        self.SetWindowsInfo()

    def CreatePanel(self):
        panel = wx.Panel(self)

        sizer = wx.GridBagSizer(10,)

        line = wx.StaticLine(panel)
        sizer.Add(line, pos=(1, 0), span=(1, 5),
            flag=wx.TOP|wx.EXPAND|wx.BOTTOM, border=10)

        binFilePath_ST = wx.StaticText(panel,label = "binFilePath")
        sizer.Add(binFilePath_ST,pos=(2,0),flag = wx.LEFT|wx.TOP,border = 10)

        binFilePath_TC = wx.TextCtrl(panel)
        sizer.Add(binFilePath_TC, pos=(2, 1), span=(1, 3), flag=wx.DOWN|wx.TOP|wx.EXPAND,border = 3)

        binFilePath_B = wx.Button(panel, label="Browse...")
        sizer.Add(binFilePath_B, pos=(2, 4), flag=wx.DOWN|wx.TOP|wx.RIGHT,border = 5)

        moveDesPath_ST = wx.StaticText(panel,label = "moveDesPath")
        sizer.Add(moveDesPath_ST,pos=(3,0),flag = wx.LEFT|wx.TOP,border = 10)

        moveDesPath_TC = wx.TextCtrl(panel)
        sizer.Add(moveDesPath_TC, pos=(3, 1), span=(1, 3), flag=wx.DOWN|wx.TOP|wx.EXPAND,border = 3)

        moveDesPath_B = wx.Button(panel, label="Browse...")
        sizer.Add(moveDesPath_B, pos=(3, 4), flag=wx.DOWN|wx.TOP|wx.RIGHT,border = 5)

        imgSavePath_ST = wx.StaticText(panel,label = "imgSavePath")
        sizer.Add(imgSavePath_ST,pos=(4,0),flag = wx.LEFT|wx.TOP,border = 10)

        imgSavePath_TC = wx.TextCtrl(panel)
        sizer.Add(imgSavePath_TC, pos=(4, 1), span=(1, 3), flag=wx.DOWN|wx.TOP|wx.EXPAND,border = 3)

        imgSavePath_B = wx.Button(panel, label="Browse...")
        sizer.Add(imgSavePath_B, pos=(4, 4), flag=wx.DOWN|wx.TOP|wx.RIGHT,border = 5)

        line = wx.StaticLine(panel)
        sizer.Add(line, pos=(5, 0), span=(1, 5),
            flag=wx.TOP|wx.EXPAND|wx.BOTTOM, border=10)

        begin_B = wx.Button(panel, label="Begin")
        sizer.Add(begin_B, pos=(6, 0), flag=wx.LEFT|wx.DOWN, border=10)
        self.Bind(wx.EVT_BUTTON,self.OnTask,begin_B)

        panel.SetSizer(sizer)
        sizer.Fit(self)

    def OnTask(self,e):
        global GRecoverFrameNum
        GRecoverFrameNum = GRecoverFrameNum + 1
        t1 = MyThread("thread %d" % GRecoverFrameNum)
        t1.start()

    def SetWindowsInfo(self):
        # self.SetSize((400, 300))
        # self.SetTitle('Simple menu')
        self.Centre()
    

class MyFrame(wx.Frame):
    def __init__(self,parent,title):
        super(MyFrame,self).__init__(parent,title=title)

        self.SetWindowsInfo()
        self.CreateMenuBar()


    def CreateMenuBar(self):
        # fileMenu
        fileMenu = wx.Menu()

        newItem = fileMenu.Append(wx.ID_NEW, '&New')
        self.Bind(wx.EVT_MENU, self.NewRecover, newItem)
        fileMenu.Append(wx.ID_OPEN, '&Open')
        fileMenu.Append(wx.ID_SAVE, '&Save')
        fileMenu.AppendSeparator() # add a separted line

        quitItem = fileMenu.Append(wx.ID_EXIT, 'Quit',"Quit App") # Auto create MenuItem
        self.Bind(wx.EVT_MENU, self.OnQuit, quitItem)

        # settingMenu
        settingMenu = wx.Menu()
        menu1 = settingMenu.Append(wx.ID_ANY,'menu1')
        self.Bind(wx.EVT_MENU, self.OnTaskOne, menu1)
        menu2 = settingMenu.Append(wx.ID_ANY,'menu2')
        self.Bind(wx.EVT_MENU, self.OnTaskTwo, menu2)

        # menubar
        menubar = wx.MenuBar()
        menubar.Append(fileMenu, '&File')
        menubar.Append(settingMenu, '&Setting')
        self.SetMenuBar(menubar)

    def SetWindowsInfo(self):
        self.SetSize((800, 600))
        # self.SetTitle('Simple menu')
        self.Centre()

    def OnTaskOne(self,e):
        t1 = MyThread("thread 1")
        t2 = MyThread("thread 2")
        t1.start()
        t2.start()
    def OnTaskTwo(self,e):
        print("ydc")
        
    def NewRecover(self,e):
        newFrame = RecoverFrame(None, title='  图像恢复')
        newFrame.Show()


    def OnQuit(self, e):
        self.Close()

def  main():
    
    app = wx.App()
    ex = MyFrame(None, title='  图像恢复')
    ex.Show()
    
    app.MainLoop()

if __name__ == '__main__':
    
    main()


