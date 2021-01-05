import wx
class ExamplePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        grid = wx.GridBagSizer(hgap = 3,vgap =10)
        hSizer = wx.BoxSizer(wx.HORIZONTAL)



        # 创建一个用于显示的标签
        self.quote = wx.StaticText(self, label="Your quote :")
        grid.Add(self.quote,pos=(0,0))
 
        # 创建一个多行的文本框，用于打印
        self.logger = wx.TextCtrl(self, size=(-1,-1), style=wx.TE_MULTILINE | wx.TE_READONLY)
 
        # 创建一个按钮，并绑定事件
        self.button =wx.Button(self, label="Save")
        self.Bind(wx.EVT_BUTTON, self.OnClick,self.button)
 
        
        self.lblname = wx.StaticText(self, label="Your name :")
        grid.Add(self.lblname,pos=(1,0))
        # 创建单行文本框，用于输入，注意两种不同的事件
        self.editname = wx.TextCtrl(self, value="Enter here your name", size=(140,-1))
        grid.Add(self.editname,pos=(1,1))
        self.Bind(wx.EVT_TEXT, self.EvtText, self.editname)
        self.Bind(wx.EVT_CHAR, self.EvtChar, self.editname)
 
        # ComboBox额外有一个事件wx.EVT_COMBOBOX
        self.sampleList = ['friends', 'advertising', 'web search', 'Yellow Pages']
        self.lblhear = wx.StaticText(self, label="How did you hear from us ?")
        grid.Add(self.lblhear,pos=(3,0))
        self.edithear = wx.ComboBox(self, size=(95, -1), choices=self.sampleList, style=wx.CB_DROPDOWN)
        grid.Add(self.edithear,pos = (3,1))
        self.Bind(wx.EVT_COMBOBOX, self.EvtComboBox, self.edithear)
        self.Bind(wx.EVT_TEXT, self.EvtText,self.edithear)

        # grid.Add((10,40),pos=(2,0))

        # 检查是否选中，类似多选
        self.insure = wx.CheckBox(self, label="Do you want Insured Shipment ?")
        grid.Add(self.insure,pos = (4,0),span=(1,2),flag=wx.BOTTOM,border = 5)
        self.Bind(wx.EVT_CHECKBOX, self.EvtCheckBox, self.insure)
 
        # 检查选择了哪一个，类似单选
        radioList = ['blue', 'red', 'yellow', 'orange', 'green', 'purple', 'navy blue', 'black', 'gray']
        rb = wx.RadioBox(self, label="What color would you like ?", choices=radioList,  majorDimension=3,
                         style=wx.RA_SPECIFY_COLS)
        grid.Add(rb,pos = (5,0))
        self.Bind(wx.EVT_RADIOBOX, self.EvtRadioBox, rb)

        hSizer.Add(grid,1,wx.ALL,5)
        hSizer.Add(self.logger,1,wx.EXPAND)
        mainSizer.Add(hSizer,1,wx.ALL,5)
        mainSizer.Add(self.button,0,wx.CENTER)
        self.SetSizerAndFit(mainSizer)
 
    def EvtRadioBox(self, event):
        self.logger.AppendText('EvtRadioBox: %d\n' % event.GetInt())
    def EvtComboBox(self, event):
        self.logger.AppendText('EvtComboBox: %s\n' % event.GetString())
    def OnClick(self,event):
        self.logger.AppendText(" Click on object with Id %d\n" %event.GetId())
    def EvtText(self, event):
        self.logger.AppendText('EvtText: %s\n' % event.GetString())
    def EvtChar(self, event):
        self.logger.AppendText('EvtChar: %d\n' % event.GetKeyCode())
        event.Skip()
    def EvtCheckBox(self, event):
        self.logger.AppendText('EvtCheckBox: %d\n' % event.IsChecked())
 
app = wx.App(False)
frame = wx.Frame(None, title="Demo without Notebook",size=(800,400))
panel = ExamplePanel(frame)
frame.Show()
app.MainLoop()


"""
# 用notebook功能，可以创建多个页面
app = wx.App(False)
frame = wx.Frame(None, title="Demo with Notebook")
nb = wx.Notebook(frame)
nb.AddPage(ExamplePanel(nb), "Absolute Positioning")
nb.AddPage(ExamplePanel(nb), "Page Two")
nb.AddPage(ExamplePanel(nb), "Page Three")
frame.Show()
app.MainLoop()
"""