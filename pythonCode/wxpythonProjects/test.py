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
        
        
    def InitUI(self):
        self.Bind(wx.EVT_RIGHT_DOWN,self.OnRightDown)

    def OnRightDown(self,e):
        self.PopupMenu(MyPopupMenu(self),e.GetPosition())

    def SetWindowsInfo(self):
        self.SetSize((300, 200))
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


def  main():

    app = wx.App()
    ex = Example(None, title='Moving')
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()