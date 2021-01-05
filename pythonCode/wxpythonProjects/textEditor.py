#!/usr/bin/env python
import wx
import os
class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        self.dirname = ''   # 文件路径定义

        wx.Frame.__init__(self, parent, title=title, size=(800,-1)) # 窗口frame初始化，size设为-1则为默认
        self.control = wx.TextCtrl(self, style=wx.TE_MULTILINE) # 创建一个control，指定为TextCtrl
        self.CreateStatusBar()  # 创建底部的状态栏

        # 创建菜单栏中的一个菜单，并加入子选项
        filemenu = wx.Menu()
        menuOpen = filemenu.Append(wx.ID_OPEN,"&Open"," Open a file to edit")
        menuAbout = filemenu.Append(wx.ID_ABOUT,"&About"," Information about this program")
        menuExit = filemenu.Append(wx.ID_EXIT,"&Exit"," Terminate the program")

        # 创建菜单栏的第二个菜单，并加入子选项
        editmenu = wx.Menu()
        menuCopy = editmenu.Append(wx.ID_COPY,"&copy"," copy what you select")
        menuPaste = editmenu.Append(wx.ID_PASTE,"&paste"," paste what you copy or cut")

        # 创建菜单栏，并把上面两个菜单加进去，且设置为当前菜单栏
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File")
        menuBar.Append(editmenu,"&Edit")
        self.SetMenuBar(menuBar)

        # 将菜单的子选项与事件绑定
        self.Bind(wx.EVT_MENU,self.OnOpen,menuOpen)
        self.Bind(wx.EVT_MENU,self.OnAbout,menuAbout)
        self.Bind(wx.EVT_MENU,self.OnExit,menuExit)

        # 创建名为sizer2的sizer用来放六个按钮
        self.sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.buttons = []
        for i in range(0,6):
            self.buttons.append(wx.Button(self,-1," Button &"+str(i)))
            self.sizer2.Add(self.buttons[i],1,wx.EXPAND)
        
        # 创建名为sizer的sizer，里面放一个control和sizer2
        self.sizer=wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.control,1,wx.EXPAND)
        self.sizer.Add(self.sizer2,0,wx.EXPAND)

        # 设置当前sizer
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)

        # 显示这个frame
        self.Show(True)
    
    # 以下是响应菜单子选项的几个函数
    def OnAbout(self,e):
        dlg = wx.MessageDialog(self," A small text editor"," About sample Editor",wx.CANCEL)
        dlg.ShowModal()
        dlg.Destroy()
    def OnExit(self,e):
        self.Close(True)
    def OnOpen(self,e):
        """open a file"""
        dlg = wx.FileDialog(self," Choose a file",self.dirname,"","*.*",wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = open(os.path.join(self.dirname,self.dirname),'r')
            self.control.SetValue(f.read())
            f.close()
        dlg.Destroy()

app = wx.App(False) # 创建应用程序实例
frame = MainWindow(None, 'Small editor') # 创建窗口实例
app.MainLoop() # 开始循环