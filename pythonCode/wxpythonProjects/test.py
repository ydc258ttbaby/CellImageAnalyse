#!/usr/bin/env python

# moving.py

import wx

APP_EXIT = 1

class MyPopupMenu(wx.Menu):
    def __init__(self,parent):
        super(MyPopupMenu,self).__init__()

        self.parent = parent 

        MinimizeMenuItem = wx.MenuItem(self,wx.NewId(),"Minimize")
        self.Append(MinimizeMenuItem)
        self.Bind(wx.EVT_MENU,self.OnMinimize,MinimizeMenuItem)

        CloseMenuItem = wx.MenuItem(self,wx.NewId(),"Close")
        self.Append(CloseMenuItem)
        self.Bind(wx.EVT_MENU,self.OnClose,CloseMenuItem)
    
    def OnMinimize(self,e):
        self.parent.Iconize()

    def OnClose(self,e):
        self.parent.Close()


class Example(wx.Frame):

    def __init__(self, parent, title):
        super(Example, self).__init__(parent, title=title)
        self.CreateMenu()
        self.CreateTool()
        self.CreateStat()
        self.SetWindowsInfo()
        self.InitUI()
        
        # self.GoToClass()
        self.CreatePanel()
    def GoToClass(self):
        panel = wx.Panel(self)
        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(9)

        vbox = wx.BoxSizer(wx.VERTICAL)
        
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        st1 = wx.StaticText(panel,label = "Class Name ")
        st1.SetFont(font)
        hbox1.Add(st1,flag=wx.RIGHT,border = 8)
        tc = wx.TextCtrl(panel)
        hbox1.Add(tc,proportion =1)
        vbox.Add(hbox1,flag = wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP,border = 10 )

        panel.SetSizer(vbox)

    def CreatePanel(self):

        panel = wx.Panel(self)
        panel.SetBackgroundColour('#595959')

        midPanel = wx.Panel(panel)
        midPanel.SetBackgroundColour('#ffeded')

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        st1 = wx.StaticText(panel,label = "Class Name ")
        hbox1.Add(st1,flag=wx.RIGHT,border = 8)
        tc = wx.TextCtrl(panel)
        hbox1.Add(tc,proportion =1)

        midPanel.SetSizer(hbox1)

        vBox = wx.BoxSizer(wx.VERTICAL)
        vBox.Add(midPanel,wx.ID_ANY,wx.EXPAND | wx.ALL,border = 60) # the goal is to Add Border to midPanel

        panel.SetSizer(vBox)

        
    def InitUI(self):
        self.Bind(wx.EVT_RIGHT_DOWN,self.OnRightDown)

    def OnRightDown(self,e):
        self.PopupMenu(MyPopupMenu(self),e.GetPosition())

    def SetWindowsInfo(self):
        self.SetSize((800, 600))
        # self.SetTitle('Simple menu')
        self.Centre()

    def CreateTool(self):
        self.toolbar = self.CreateToolBar()
        quitTool = self.toolbar.AddTool(wx.ID_ANY, 'Quit', wx.Bitmap(r'source\code_70x70.png'))
        self.toolbar.Realize()
        self.Bind(wx.EVT_TOOL,self.Print,quitTool) # bind tool a action
        

    def CreateStat(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText('Ready')

    def Print(self,e):
        print("ydc")

    def CreateMenu(self):
        # fileMenu
        fileMenu = wx.Menu()

        fileMenu.Append(wx.ID_NEW, '&New')
        fileMenu.Append(wx.ID_OPEN, '&Open')
        fileMenu.Append(wx.ID_SAVE, '&Save')
        fileMenu.AppendSeparator() # add a separted line

        imp = wx.Menu()
        imp.Append(wx.ID_ANY, 'Import newsfeed list...')
        imp.Append(wx.ID_ANY, 'Import bookmarks...')
        imp.Append(wx.ID_ANY, 'Import mail...')
        fileMenu.AppendMenu(wx.ID_ANY, 'Import', imp) # append Menu to Menu which can create submenu

        fileItem = fileMenu.Append(wx.ID_EXIT, 'Quit1',"Quit App") # Auto create MenuItem
        qmi = wx.MenuItem(fileMenu, APP_EXIT, '&Quit2\tCtrl+Q') # manu create MenuItem
        qmi.SetBitmap(wx.Bitmap(r'source\code_70x70.png')) # load png as quit icon
        self.Bind(wx.EVT_MENU, self.OnQuit, fileItem)
        self.Bind(wx.EVT_MENU, self.OnQuit, id=APP_EXIT)
        fileMenu.Append(qmi)

        # viewMenu
        viewMenu = wx.Menu()
        self.ShowStatMenuItem = viewMenu.Append(wx.ID_ANY,'Show Stat',kind=wx.ITEM_CHECK)
        self.ShowToolMenuItem = viewMenu.Append(wx.ID_ANY,'Show Tool',kind=wx.ITEM_CHECK)
        
        viewMenu.Check(self.ShowStatMenuItem.GetId(),True) # change CheckMenuItem state 
        viewMenu.Check(self.ShowToolMenuItem.GetId(),True)

        self.Bind(wx.EVT_MENU, self.ToggleStat, self.ShowStatMenuItem) # bind checkMenuItem to decide if show statBar
        self.Bind(wx.EVT_MENU, self.ToggleTool, self.ShowToolMenuItem)

        


        # menubar
        menubar = wx.MenuBar()
        menubar.Append(viewMenu, '&View')
        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)


    def ToggleStat(self,e): # note here input is two parmas
        if self.ShowStatMenuItem.IsChecked():
            self.statusbar.Show()
        else:
            self.statusbar.Hide()

    def ToggleTool(self,e):
        if self.ShowToolMenuItem.IsChecked():
            self.toolbar.Show()
        else:
            self.toolbar.Hide()

    def OnQuit(self, e):
        self.Close()

class LayoutExample(wx.Frame):

    def __init__(self, parent, title):
        super(LayoutExample, self).__init__(parent, title=title)

        self.InitUI()
        self.Centre()

    def InitUI(self):

        panel = wx.Panel(self)

        sizer = wx.GridBagSizer(5, 5)

        text1 = wx.StaticText(panel, label="Java Class Java Class Java Class")
        sizer.Add(text1, pos=(0, 0),span=(1, 4), flag=wx.TOP|wx.LEFT|wx.BOTTOM,
            border=15)

        icon = wx.StaticBitmap(panel, bitmap=wx.Bitmap(r'source\code_70x70.png'))
        sizer.Add(icon, pos=(0, 4), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,
            border=5)

        line = wx.StaticLine(panel)
        sizer.Add(line, pos=(1, 0), span=(1, 4),
            flag=wx.EXPAND|wx.BOTTOM, border=10)

        text2 = wx.StaticText(panel, label="Name")
        sizer.Add(text2, pos=(2, 0), flag=wx.LEFT, border=10)

        tc1 = wx.TextCtrl(panel)
        sizer.Add(tc1, pos=(2, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND)

        text3 = wx.StaticText(panel, label="Package")
        sizer.Add(text3, pos=(3, 0), flag=wx.LEFT|wx.TOP, border=10)

        tc2 = wx.TextCtrl(panel)
        sizer.Add(tc2, pos=(3, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND,
            border=5)

        button1 = wx.Button(panel, label="Browse...")
        sizer.Add(button1, pos=(3, 4), flag=wx.TOP|wx.RIGHT, border=5)

        text4 = wx.StaticText(panel, label="Extends")
        sizer.Add(text4, pos=(4, 0), flag=wx.TOP|wx.LEFT, border=10)

        combo = wx.ComboBox(panel)
        sizer.Add(combo, pos=(4, 1), span=(1, 3),
            flag=wx.TOP|wx.EXPAND, border=5)

        button2 = wx.Button(panel, label="Browse...")
        sizer.Add(button2, pos=(4, 4), flag=wx.TOP|wx.RIGHT, border=5)

        sb = wx.StaticBox(panel, label="Optional Attributes")

        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)
        boxsizer.Add(wx.CheckBox(panel, label="Public"),
            flag=wx.LEFT|wx.TOP, border=5)
        boxsizer.Add(wx.CheckBox(panel, label="Generate Default Constructor"),
            flag=wx.LEFT, border=5)
        boxsizer.Add(wx.CheckBox(panel, label="Generate Main Method"),
            flag=wx.LEFT|wx.BOTTOM, border=5)
        sizer.Add(boxsizer, pos=(5, 0), span=(1, 5),
            flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT , border=10)

        button3 = wx.Button(panel, label='Help')
        sizer.Add(button3, pos=(7, 0), flag=wx.LEFT, border=10)

        button4 = wx.Button(panel, label="Ok")
        sizer.Add(button4, pos=(7, 3))

        button5 = wx.Button(panel, label="Cancel")
        sizer.Add(button5, pos=(7, 4), span=(1, 1),
            flag=wx.BOTTOM|wx.RIGHT, border=10)

        sizer.AddGrowableCol(2)

        panel.SetSizer(sizer)
        sizer.Fit(self)

def  main():

    app = wx.App()
    ex = LayoutExample(None, title='Moving')
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()