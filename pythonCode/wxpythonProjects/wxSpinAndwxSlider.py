import wx
class MyFrame(wx.Frame):
    def __init__(self,parent,title):
        wx.Frame.__init__(self,parent,title=title,size=(400,200))

        mainSizer = wx.BoxSizer(wx.VERTICAL)

        self.Silder = wx.Slider(self)
        self.Bind(wx.EVT_SCROLL,self.OnSroll,self.Silder)
        mainSizer.Add(self.Silder,0,wx.EXPAND)

        self.SpinCtr = wx.SpinCtrl(self)
        self.Bind(wx.EVT_SPINCTRL,self.OnSpin,self.SpinCtr)
        mainSizer.Add(self.SpinCtr,0,wx.EXPAND)

        self.control = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        mainSizer.Add(self.control,1,wx.EXPAND)

        self.SetSizerAndFit(mainSizer)
        self.Show(True)
    def OnSroll(self,event):
        self.control.AppendText('Silder Value: %d\n' % event.GetInt())
    def OnSpin(self,event):
        self.control.AppendText('Spin Value: %d\n' % event.GetInt())

app = wx.App(False)
Frame = MyFrame(None," Spin and Slider")
app.MainLoop()

