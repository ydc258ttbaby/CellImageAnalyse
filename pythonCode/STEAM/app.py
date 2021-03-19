# import myFunction as MF
from threading import Thread,Event,Condition
import threading
import wx
import time

print("ydc")

global GRecoveryTaskNum
GRecoveryTaskNum = 0

ID_binFilePath_B = wx.NewId()
ID_moveDesPath_B = wx.NewId()
ID_imgSavePath_B = wx.NewId()
ID_newCollect_B = wx.NewId()
ID_newRecover_B = wx.NewId()
ID_newAnalyze_B = wx.NewId()

class RecoveryParmas():
    def __init__(   self,\
                    bValid = False ,\
                    binFilePath = r"E:\Tianjin\temp",\
                    moveDesPath = r"E:\Tianjin\原始数据",\
                    imgSavePath = r"E:\Tianjin\图像数据",\
                    bMove = True ,\
                    bLinearNor = False ,\
                    bPreCrop = True, \
                    midTargetValue = "160",\
                    cropH = "32",\
                    fullH = "56" \
                ):
        self.binFilePath = binFilePath
        self.moveDesPath = moveDesPath
        self.imgSavePath = imgSavePath
        self.bValid = bValid
        self.bMove = bMove 
        self.bLinearNor = bLinearNor 
        self.bPreCrop = bPreCrop
        self.midTargetValue = midTargetValue
        self.cropH = cropH
        self.fullH = fullH
        self.serialNumList = []
    def print(self):
        print("-----------------------------")
        print("源文件路径：  %s" % self.binFilePath)
        print("转移目标路径：%s" % self.moveDesPath)
        print("图像保存路径: %s" % self.imgSavePath)
        print("是否移动源文件：%d" % self.bMove)
        print("是否线性归一化：%d" % self.bLinearNor)
        print("是否预裁剪：%d" % self.bPreCrop)
        print("非线性归一化阈值：%s" % self.midTargetValue)
        print("纵向裁剪尺寸：%s" % self.cropH)
        print("纵向总尺寸: %s" % self.fullH)
        print("样本编号：    " ,end="")
        print(self.serialNumList)
        print("-----------------------------")

class RecoverThread(Thread):
    def __init__(self,NewRecoverPara,name):
        super(RecoverThread,self).__init__()
        self.paused = False
        self.end = False
        self.pauseEvent = Event()
        self.stopEvent = Event()
        self.daemon = True
        self.name = name
        self.NewRecoverPara = NewRecoverPara
        self.count = 0
    def run(self):
        # self.resume()
        while(not(self.stopEvent.isSet())):
            time.sleep(0.5)
            if self.paused:
                print("paused ...")
                self.pauseEvent.wait()
            else:
                print(self.name)  
                self.count += 5
        print("end run")
                
    def pause(self):
        self.paused = True
        self.pauseEvent.clear()
    def resume(self):
        self.paused = False
        self.pauseEvent.set()
    def stop(self):
        self.stopEvent.set()
    def GetProcess(self):
        return self.count


class RecoverInfoDislog(wx.Dialog):
    def __init__(self,parent):
        super(RecoverInfoDislog, self).__init__(parent)
        self.CreatePanel()
        self.SetWindowsInfo()
        self.rParmas = RecoveryParmas()
        
    def SetWindowsInfo(self):
        self.SetTitle("设置恢复参数")
        # self.SetSize((400, 300))
        self.Centre()

    def GetParmas(self):
        return self.rParmas

    def CreatePanel(self):
        
        font1 = wx.Font(10, family = wx.MODERN,style = wx.NORMAL,weight = wx.NORMAL,underline = False,faceName = '微软雅黑 Light')
        font2 = wx.Font(10, family = wx.MODERN,style = wx.NORMAL,weight = wx.NORMAL,underline = False,faceName = '微软雅黑 Light')

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.listbox = wx.ListBox(panel,size = (300,100))
        hbox.Add(self.listbox, wx.ID_ANY,wx.TOP|wx.LEFT|wx.DOWN, 10)
        
        btnPanel = wx.Panel(panel)
        btnvbox = wx.BoxSizer(wx.VERTICAL)
        newBtn = wx.Button(btnPanel, wx.ID_ANY, '新建', size=(60, 30))
        delBtn = wx.Button(btnPanel, wx.ID_ANY, '删除', size=(60, 30))
        clrBtn = wx.Button(btnPanel, wx.ID_ANY, '清除', size=(60, 30))

        newBtn.SetFont(font2)
        delBtn.SetFont(font2)
        clrBtn.SetFont(font2)

        self.Bind(wx.EVT_BUTTON, self.NewItem, id=newBtn.GetId())
        self.Bind(wx.EVT_BUTTON, self.OnDelete, id=delBtn.GetId())
        self.Bind(wx.EVT_BUTTON, self.OnClear, id=clrBtn.GetId())

        btnvbox.Add((-1, 10))
        btnvbox.Add(newBtn)
        btnvbox.Add(delBtn, 0, wx.TOP, 5)
        btnvbox.Add(clrBtn, 0, wx.TOP, 5)

        btnPanel.SetSizer(btnvbox)
        hbox.Add(btnPanel, 0, wx.EXPAND | wx.RIGHT|wx.LEFT, 10)

        vbox.Add(hbox,0,wx.EXPAND |wx.LEFT|wx.RIGHT, 5)
        line = wx.StaticLine(panel)
        vbox.Add(line,0,wx.EXPAND |wx.TOP|wx.DOWN, 15)

        pathPanel = wx.Panel(panel)

        binFilePath_ST = wx.StaticText(pathPanel,pos = (8,0),label = "源文件路径")
        self.binFilePath_TC = wx.TextCtrl(pathPanel,pos = (100,0),size=(196, 25),value = r"E:\Tianjin\temp")
        binFilePath_B = wx.Button(pathPanel,pos = (300,0), label="...",size=(25, 25),id=ID_binFilePath_B)

        moveDesPath_ST = wx.StaticText(pathPanel,pos = (8,30),label = "目标转移路径")
        self.moveDesPath_TC = wx.TextCtrl(pathPanel,pos = (100,30),size=(196, 25),value = r"E:\Tianjin\原始数据")
        moveDesPath_B = wx.Button(pathPanel,pos = (300,30), label="...",size=(25, 25),id=ID_moveDesPath_B)
        
        imgSavePath_ST = wx.StaticText(pathPanel,pos = (8,60),label = "图像保存路径")
        self.imgSavePath_TC = wx.TextCtrl(pathPanel,pos = (100,60),size=(196, 25),value = r"E:\Tianjin\图像数据")
        imgSavePath_B = wx.Button(pathPanel,pos = (300,60), label="...",size=(25, 25),id=ID_imgSavePath_B)

        self.Bind(wx.EVT_BUTTON,self.ChooseFileDir,binFilePath_B)
        self.Bind(wx.EVT_BUTTON,self.ChooseFileDir,moveDesPath_B)
        self.Bind(wx.EVT_BUTTON,self.ChooseFileDir,imgSavePath_B)

        binFilePath_ST.SetFont(font2)
        self.binFilePath_TC.SetFont(font2)
        binFilePath_B.SetFont(font2)
        moveDesPath_ST.SetFont(font2)
        self.moveDesPath_TC.SetFont(font2)
        moveDesPath_B.SetFont(font2)
        imgSavePath_ST.SetFont(font2)
        self.imgSavePath_TC.SetFont(font2)
        imgSavePath_B.SetFont(font2)

        vbox.Add(pathPanel,0,wx.EXPAND |wx.LEFT|wx.RIGHT, 15)


        line = wx.StaticLine(panel)        
        vbox.Add(line,0,wx.EXPAND |wx.TOP|wx.DOWN, 15)


        setting_Panel = wx.Panel(panel)

        midTargetValue_ST = wx.StaticText(setting_Panel,pos = (8,0),label = "归一化阈值：")
        cropH_ST = wx.StaticText(setting_Panel,pos = (8,30),label = "纵向裁剪尺寸：")
        fullH_ST = wx.StaticText(setting_Panel,pos = (8,60),label = "纵向总尺寸：")

        self.midTargetValue_TC = wx.TextCtrl(setting_Panel,pos = (120,0),size = (30,25),value = "160")
        self.cropH_TC = wx.TextCtrl(setting_Panel,pos = (120,30),size = (30,25),value = "32")
        self.fullH_TC = wx.TextCtrl(setting_Panel,pos = (120,60),size = (30,25),value = "56")

        self.bMoved_CB = wx.CheckBox(setting_Panel,pos = (170,3),label = "是否转移源文件")
        self.bMoved_CB.SetValue(True)
        self.bLinearNor_CB = wx.CheckBox(setting_Panel,pos = (170,33),label = "是否线性归一化") 
        self.bLinearNor_CB.SetValue(False)
        self.bPreCrop_CB = wx.CheckBox(setting_Panel,pos = (170,63),label = "是否预裁剪")
        self.bPreCrop_CB.SetValue(True)

        midTargetValue_ST.SetFont(font2)
        cropH_ST.SetFont(font2)
        fullH_ST.SetFont(font2)
        self.midTargetValue_TC.SetFont(font2)
        self.cropH_TC.SetFont(font2)
        self.fullH_TC.SetFont(font2)
        self.bMoved_CB.SetFont(font2)
        self.bLinearNor_CB.SetFont(font2)
        self.bPreCrop_CB.SetFont(font2)


        vbox.Add(setting_Panel,0,wx.EXPAND |wx.LEFT|wx.RIGHT, 15)



        line = wx.StaticLine(panel)        
        vbox.Add(line,0,wx.EXPAND |wx.TOP|wx.DOWN, 15)

        save_B = wx.Button(panel, label="保存")
        self.Bind(wx.EVT_BUTTON,self.OnSave,save_B)


        vbox.Add(save_B, 0, flag=wx.CENTER|wx.DOWN, border=10)

        panel.SetSizer(vbox)
        
        vbox.Fit(self)

    def NewItem(self, event):

        text = wx.GetTextFromUser('请输入样本编号：', '新建样本')
        if text != '':
            self.listbox.Append(text)

    def OnDelete(self, event):

        sel = self.listbox.GetSelection()
        if sel != -1:
            self.listbox.Delete(sel)

    def OnClear(self, event):
        self.listbox.Clear()

    def ChooseFileDir(self,e):
        dlg = wx.DirDialog(self,u"选择文件夹",style=wx.DD_DEFAULT_STYLE,defaultPath=r"E:\Tianjin")
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            if (e.GetId() == ID_binFilePath_B):
                self.binFilePath_TC.SetValue(path)
            if (e.GetId() == ID_moveDesPath_B):
                self.moveDesPath_TC.SetValue(path)
            if (e.GetId() == ID_imgSavePath_B):
                self.imgSavePath_TC.SetValue(path)
        dlg.Destroy()
    
    def OnSave(self,e):
        if (len(self.listbox.GetStrings()) != 0):
            self.rParmas.bValid = True
            self.rParmas.serialNumList = self.listbox.GetStrings()
            self.rParmas.binFilePath = self.binFilePath_TC.GetValue()
            self.rParmas.imgSavePath = self.imgSavePath_TC.GetValue()
            self.rParmas.moveDesPath = self.moveDesPath_TC.GetValue()
            
            self.rParmas.midTargetValue = self.midTargetValue_TC.GetValue()
            self.rParmas.cropH = self.cropH_TC.GetValue()
            self.rParmas.fullH = self.fullH_TC.GetValue()
            self.rParmas.bMove = self.bMoved_CB.GetValue()
            self.rParmas.bLinearNor = self.bLinearNor_CB.GetValue()
            self.rParmas.bPreCrop = self.bPreCrop_CB.GetValue()
            
        self.Destroy()
        
class FontAsset(wx.Font):
    def __init__(self,fontSize,family = wx.MODERN,style = wx.NORMAL,weight = wx.NORMAL,underline = False):
        super(FontAsset,self).__init__()
        self.SetFaceName = '微软雅黑 Light'
        self.SetPointSize(fontSize)
        # self.font = wx.Font(fontSize, family,style , weight, underline, faceName = '微软雅黑 Light')
class IconAsset():
    def __init__(self,name):
        self.name = name
    def Get(self):
        path = "%s.png" % self.name
        return wx.Bitmap(path)

class RecoveryPanel(wx.Panel):
    def __init__(self,parent,NewRecoverPara,topVbox):
        super(RecoveryPanel,self).__init__(parent)
        # self.ShowWithEffect(wx.SHOW_EFFECT_SLIDE_TO_LEFT)
        # self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Para = NewRecoverPara
        self.vbox = topVbox

        self.Init()
        self.CreatePanel()
        self.CreateTimer()
        
    def Init(self):

        
        self.bRunning = False
        self.bPaused = False
        self.bUnBegin = True
        self.TaskProcess = 0


        global GRecoveryTaskNum
        GRecoveryTaskNum += 1
        self.task = RecoverThread(name = "thread %d" % GRecoveryTaskNum,NewRecoverPara = self.Para )
        self.task.setDaemon(True)
        self.BGcolor = (79,93,112)
        self.FColor = (235,235,235)
        self.font = FontAsset(13)  
        
        self.StartIcon = IconAsset("begin").Get()
        self.StopIcon = IconAsset("stop").Get()
        self.ResumeIcon = IconAsset("resume").Get()
        self.InfoIcon = IconAsset("info").Get()
        self.RecoverIcon = IconAsset("recover").Get()
        self.CollectIcon = IconAsset("collect").Get()
        self.AnalyzeIcon = IconAsset("analyze").Get()
        
        self.StartIconFocus = IconAsset("beginFocus").Get()
        self.StopIconFocus = IconAsset("stopFocus").Get()
        self.ResumeIconFocus = IconAsset("resumeFocus").Get()
        self.InfoIconFocus = IconAsset("infoFocus").Get()
        self.RecoverIconFocus = IconAsset("recoverFocus").Get()
        self.CollectIconFocus = IconAsset("collectFocus").Get()
        self.AnalyzeIconFocus = IconAsset("analyzeFocus").Get()

    def CreateTimer(self):
        self.timer = wx.Timer(self,1)
        self.Bind(wx.EVT_TIMER,self.OnTimer,self.timer)
    def OnTimer(self,e):
        if(self.TaskProcess > 100):
            self.timer.Stop()
            
        self.TaskProcess = self.task.GetProcess()
        self.gauge.SetValue(self.TaskProcess)

        print("timer: %d"%self.TaskProcess)

    def CreatePanel(self):
        self.SetBackgroundColour(self.FColor)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        TextAndGaugevBox = wx.BoxSizer(wx.VERTICAL)

        TextBGPanel = wx.Panel(self)
        TextBGPanel.SetBackgroundColour((68,86,106))
        TextBGPanelBox = wx.BoxSizer(wx.VERTICAL)
        TextBGPanel.SetSizer(TextBGPanelBox)

        TextForePanel = wx.Panel(TextBGPanel)
        TextForePanel.SetBackgroundColour((255,255,255))
        
        SI_Label = ""
        bFirst = True
        for item in self.Para.serialNumList:
            if bFirst:
                bFirst = False
                SI_Label = item
            else:
                SI_Label = SI_Label + "-" + item

        SI_ST = wx.StaticText(TextForePanel,label = SI_Label,size = (-1,25),pos = (5,3))
        SI_ST.SetFont(self.font)

        self.gauge = wx.Gauge(self,range = 100,size=(-1,5))
        # self.gauge.SetValue(50)

        TextBGPanelBox.Add(TextForePanel,1,wx.EXPAND|wx.ALL,1)

        TextAndGaugevBox.Add(TextBGPanel,0,wx.EXPAND)
        TextAndGaugevBox.Add(self.gauge,proportion = 0,flag = wx.EXPAND)
        
        self.sBtn = wx.Button(self, wx.ID_ANY,  size=(32, 32))
        self.pBtn = wx.Button(self, wx.ID_ANY,  size=(32, 32))
        self.mBtn = wx.Button(self, wx.ID_ANY,  size=(32, 32))

        self.sBtn.SetBitmap(self.StartIcon)
        self.sBtn.SetBitmapCurrent(self.StartIconFocus)

        self.pBtn.SetBitmap(self.StopIcon)
        self.pBtn.SetBitmapCurrent(self.StopIconFocus)

        self.mBtn.SetBitmap(self.RecoverIcon)
        self.mBtn.SetBitmapCurrent(self.RecoverIconFocus)

        hbox.Add(TextAndGaugevBox, 1, wx.EXPAND|wx.ALL, 4)
        hbox.Add(self.sBtn, 0, wx.RIGHT|wx.TOP|wx.DOWN, 4)
        hbox.Add(self.pBtn, 0, wx.RIGHT|wx.TOP|wx.DOWN, 4)
        hbox.Add(self.mBtn, 0, wx.RIGHT|wx.TOP|wx.DOWN, 4)

        self.sBtn.Bind(wx.EVT_BUTTON,self.SwitchRunPause)
        self.pBtn.Bind(wx.EVT_BUTTON,self.CancelTask)
        self.mBtn.Bind(wx.EVT_BUTTON,self.ShowMoreInfo)

        self.SetSizer(hbox)

    def ShowMoreInfo(self,e):
        print("new Frame")
        moreInfoFrame = RecoveryFrame(None,self.Para)
        moreInfoFrame.Show()
    
        
    def CancelTask(self,e):
        global GRecoveryTaskNum
        GRecoveryTaskNum -= 1
        self.task.stop()
        self.Destroy()
        self.vbox.Layout()
        self.timer.Stop()
        
    def SwitchRunPause(self,e):
        if(self.bUnBegin):
            self.bUnBegin = False
            self.task.start()
            # self.sBtn.SetLabel("暂停")
            self.sBtn.SetBitmap(self.ResumeIcon)
            self.sBtn.SetBitmapCurrent(self.ResumeIconFocus)
            self.timer.Start(100)
            self.bRunning = True
        else:
            if(self.bRunning):
                self.bRunning = False
                self.task.pause()
                # self.sBtn.SetLabel("继续")
                self.sBtn.SetBitmap(self.StartIcon)
                self.sBtn.SetBitmapCurrent(self.StartIconFocus)
                self.bPaused = True
            else:
                if(self.bPaused):
                    self.bPaused = False
                    self.task.resume()
                    # self.sBtn.SetLabel("暂停")
                    self.sBtn.SetBitmap(self.ResumeIcon)
                    self.sBtn.SetBitmapCurrent(self.ResumeIconFocus)

                    self.bRunning = True

class RecoveryFrame(wx.Frame):
    def __init__(self,parent,NewRecoverPara):
        super(RecoveryFrame,self).__init__(parent)
        self.NewRecoverPara = NewRecoverPara
        self.Init()
        self.CreatePanel()
        # font1 = wx.Font(17, family = wx.MODERN,style = wx.NORMAL,weight = wx.NORMAL,underline = False,faceName = '微软雅黑 Light')
        # self.SetFont(font1)
    
    def Init(self):
        self.SetSize((800, 600))
        # self.SetTitle('Simple menu')
        self.Centre()
        # self.SetBackgroundStyle(wx.BG_STYLE_TRANSPARENT)
        # self.SetWindowStyleFlag(style = wx.CAPTION | wx.MINIMIZE_BOX | wx.MAXIMIZE_BOX | wx.RESIZE_BORDER)
        icon = wx.Icon(r"recover.png")
        self.SetIcon(icon)
        self.SetMinSize((800,600))

        self.SetTitle("图像恢复 - 详细信息")

    def CreatePanel(self):
        panel = wx.Panel(self)

        font1 = wx.Font(10, family = wx.MODERN,style = wx.NORMAL,weight = wx.NORMAL,underline = False,faceName = '微软雅黑 Light')
        font2 = wx.Font(10, family = wx.MODERN,style = wx.NORMAL,weight = wx.NORMAL,underline = False,faceName = '微软雅黑 Light')

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        leftInfoPanel = wx.Panel(panel)
        leftInfoVbox = wx.BoxSizer(wx.VERTICAL)
        leftInfoPanel.SetSizer(leftInfoVbox)

        path_Panel = wx.Panel(leftInfoPanel)
        path_SB = wx.StaticBox(path_Panel,size = (300,130),label = "路径设置")
        binFilePath_Info_ST = wx.StaticText(path_Panel,pos = (8,30),label = "源文件路径: ")
        moveDesPath_Info_ST = wx.StaticText(path_Panel,pos = (8,60),label = "目标转移路径: ")
        imgSavePath_Info_ST = wx.StaticText(path_Panel,pos = (8,90),label = "图像保存路径: ")
        binFilePath_Info_TC = wx.TextCtrl(path_Panel,pos = (120,26),size = (170,25),value =self.NewRecoverPara.binFilePath)
        moveDesPath_Info_TC = wx.TextCtrl(path_Panel,pos = (120,56),size = (170,25),value = self.NewRecoverPara.moveDesPath)
        imgSavePath_Info_TC = wx.TextCtrl(path_Panel,pos = (120,86),size = (170,25),value = self.NewRecoverPara.imgSavePath)
        
        path_SB.SetFont(font2)
        path_SB.SetForegroundColour((100,100,100))
        binFilePath_Info_ST.SetFont(font2)
        moveDesPath_Info_ST.SetFont(font2)
        imgSavePath_Info_ST.SetFont(font2)
        binFilePath_Info_TC.SetFont(font2)
        moveDesPath_Info_TC.SetFont(font2)
        imgSavePath_Info_TC.SetFont(font2)
        

        leftInfoVbox.Add(path_Panel,flag = wx.LEFT|wx.EXPAND|wx.TOP,border = 10)

        
        progress_Panel = wx.Panel(leftInfoPanel)
        # self.NewRecoverPara.serialNumList = ["12321","123214"]
        serialNum = len(self.NewRecoverPara.serialNumList)
        progress_SB = wx.StaticBox(progress_Panel,size = (300,20*serialNum+40),label = "恢复进度")
        
        progress_SB.SetFont(font2)
        progress_SB.SetForegroundColour((100,100,100))

        count = 0
        for item in self.NewRecoverPara.serialNumList:
            count += 1
            num_ST = wx.StaticText(progress_Panel,pos = (8,30 + 20*(count-1)),label = item)
            num_ST.SetFont(font2)
            res_ST = wx.StaticText(progress_Panel,pos = (120,30 + 20*(count-1)),label = "已完成")
            res_ST.SetFont(font2)


        leftInfoVbox.Add(progress_Panel,flag = wx.LEFT|wx.EXPAND|wx.TOP,border = 10)


        setting_Panel = wx.Panel(leftInfoPanel)
        setting_SB = wx.StaticBox(setting_Panel,size = (300,130),label = "其他设置")

        setting_SB.SetFont(font2)
        setting_SB.SetForegroundColour((100,100,100))

        midTargetValue_ST = wx.StaticText(setting_Panel,pos = (8,30),label = "归一化阈值：")
        cropH_ST = wx.StaticText(setting_Panel,pos = (8,60),label = "纵向裁剪尺寸：")
        fullH_ST = wx.StaticText(setting_Panel,pos = (8,90),label = "纵向总尺寸：")

        midTargetValue_TC = wx.TextCtrl(setting_Panel,pos = (120,30),size = (30,25),value = self.NewRecoverPara.midTargetValue)
        cropH_TC = wx.TextCtrl(setting_Panel,pos = (120,60),size = (30,25),value = self.NewRecoverPara.cropH)
        fullH_TC = wx.TextCtrl(setting_Panel,pos = (120,90),size = (30,25),value = self.NewRecoverPara.fullH)

        bMoved_CB = wx.CheckBox(setting_Panel,pos = (170,33),label = "是否转移源文件")
        bMoved_CB.SetValue(self.NewRecoverPara.bMove)
        bLinearNor_CB = wx.CheckBox(setting_Panel,pos = (170,63),label = "是否线性归一化") 
        bLinearNor_CB.SetValue(self.NewRecoverPara.bLinearNor)
        bPreCrop_CB = wx.CheckBox(setting_Panel,pos = (170,93),label = "是否预裁剪")
        bPreCrop_CB.SetValue(self.NewRecoverPara.bPreCrop)

        midTargetValue_ST.SetFont(font2)
        cropH_ST.SetFont(font2)
        fullH_ST.SetFont(font2)
        midTargetValue_TC.SetFont(font2)
        cropH_TC.SetFont(font2)
        fullH_TC.SetFont(font2)
        bMoved_CB.SetFont(font2)
        bLinearNor_CB.SetFont(font2)
        bPreCrop_CB.SetFont(font2)

        leftInfoVbox.Add(setting_Panel,flag = wx.LEFT|wx.EXPAND|wx.TOP,border = 10)

        hbox.Add(leftInfoPanel,proportion = 0,flag = wx.EXPAND|wx.ALL,border = 5)
        
        panel.SetSizer(hbox)

class MyFrame(wx.Frame):
    def __init__(self,parent,title):
        super(MyFrame,self).__init__(parent,title=title)
        self.Init()
        self.SetWindowsInfo()
        self.CreateMenuBar()
        self.CreatePanel()
        # self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
    def Init(self):
        self.SetOwnBackgroundColour("#ff8888")
        self.RecoverParasList = []
        
        self.RecoverIcon = IconAsset("recoverText").Get()
        self.CollectIcon = IconAsset("collectText").Get()
        self.AnalyzeIcon = IconAsset("analyzeText").Get()
        self.RecoverIconFocus = IconAsset("recoverTextFocus").Get()
        self.CollectIconFocus = IconAsset("collectTextFocus").Get()
        self.AnalyzeIconFocus = IconAsset("analyzeTextFocus").Get()

    def CreatePanel(self):
        self.BGcolor = (79,93,112)
        self.FColor = (235,235,235)
        # panel = RecoverPanel(self)
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour(self.BGcolor)

        font1 = FontAsset(20,style = wx.BOLD )
        font1 = wx.Font(24, family = wx.MODERN,style = wx.NORMAL,weight = wx.BOLD,underline = False,faceName = '微软雅黑 Light')


        hox = wx.BoxSizer(wx.HORIZONTAL)

        self.MenuVbox = wx.BoxSizer(wx.VERTICAL)
        self.MenuPanel = wx.Panel(self.panel)
        self.MenuPanel.SetMaxSize((200,1000))
        self.MenuPanel.SetMinSize((190,100))

        # self.MenuPanel.SetMinSize((200,0))
        self.MenuPanel.SetSizer(self.MenuVbox)
        self.MenuPanel.SetBackgroundColour(self.FColor)
        newCollect_B = wx.Button(self.MenuPanel, ID_newCollect_B,size=(200, 46))
        newRecover_B = wx.Button(self.MenuPanel, ID_newRecover_B, size=(200, 46))
        newAnalyze_B = wx.Button(self.MenuPanel, ID_newAnalyze_B,  size=(200, 46))

        newCollect_B.SetBitmap(self.CollectIcon)
        newRecover_B.SetBitmap(self.RecoverIcon)
        newAnalyze_B.SetBitmap(self.AnalyzeIcon)

        newCollect_B.SetBitmapCurrent(self.CollectIconFocus)
        newRecover_B.SetBitmapCurrent(self.RecoverIconFocus)
        newAnalyze_B.SetBitmapCurrent(self.AnalyzeIconFocus)

        newCollect_B.Bind(wx.EVT_BUTTON,self.NewTask)
        newRecover_B.Bind(wx.EVT_BUTTON,self.NewTask)
        newAnalyze_B.Bind(wx.EVT_BUTTON,self.NewTask)

        # self.MenuVbox.Add((-1, 46))

        self.MenuVbox.Add(newCollect_B, 0, wx.TOP, 0)
        self.MenuVbox.Add(newRecover_B, 0, wx.TOP, 0)
        self.MenuVbox.Add(newAnalyze_B, 0, wx.TOP, 0)

        self.TaskVbox = wx.BoxSizer(wx.VERTICAL)
        self.TaskPanel = wx.Panel(self.panel)
        self.TaskPanel.SetSizer(self.TaskVbox)
        self.TaskPanel.SetBackgroundColour(self.BGcolor)
        taskTitle = wx.StaticText(self.TaskPanel,label="任务列表" )
        taskTitle.SetFont(font1)
        taskTitle.SetForegroundColour((252,252,252))
        self.TaskVbox.Add(taskTitle,flag=wx.TOP|wx.DOWN,border = 15)
    
        hox.Add(self.MenuPanel,proportion = 7,flag = wx.TOP|wx.LEFT|wx.DOWN|wx.EXPAND,border = 15)
        hox.Add(self.TaskPanel,proportion = 17,flag = wx.LEFT|wx.RIGHT|wx.EXPAND,border = 15)

        self.panel.SetSizer(hox)
        # self.SetSizer(hox)

    def Print(self,e):
        print("print: GRecoveryTaskNum: %d" %GRecoveryTaskNum)
        self.TaskVbox.Layout()

    def CreateMenuBar(self):
        # fileMenu
        fileMenu = wx.Menu()

        newItem = fileMenu.Append(wx.ID_NEW, '&New')
        fileMenu.Append(wx.ID_OPEN, '&Open')
        fileMenu.Append(wx.ID_SAVE, '&Save')
        fileMenu.AppendSeparator() # add a separted line

        quitItem = fileMenu.Append(wx.ID_EXIT, 'Quit',"Quit App") # Auto create MenuItem
        self.Bind(wx.EVT_MENU, self.OnQuit, quitItem)

        # settingMenu
        settingMenu = wx.Menu()
        menu1 = settingMenu.Append(wx.ID_ANY,'LOG')
        self.Bind(wx.EVT_MENU, self.Print, menu1)

        # menubar
        menubar = wx.MenuBar()
        menubar.Append(fileMenu, '&文件')
        menubar.Append(settingMenu, '&工具')
        self.SetMenuBar(menubar)

    def SetWindowsInfo(self):
        self.SetSize((1000, 618))
        # self.SetTitle('Simple menu')
        self.Centre()
        # self.SetBackgroundStyle(wx.BG_STYLE_TRANSPARENT)
        # self.SetWindowStyleFlag(style = wx.CAPTION | wx.MINIMIZE_BOX | wx.MAXIMIZE_BOX | wx.RESIZE_BORDER)
        icon = wx.Icon(r"icon.png")
        self.SetIcon(icon)
        self.SetMinSize((1000,618))
   
    def NewTask(self,e):
        # self.newFrame = RecoverFrame(None, title='  图像恢复')
        # self.newFrame.Show()
        print(e.GetId())
        if(e.GetId() == ID_newRecover_B):
            dislog = RecoverInfoDislog(None)
            dislog.ShowModal()
            NewRecoverPara = dislog.GetParmas()
            if True or NewRecoverPara.bValid:
                self.RecoverParasList.append(dislog.GetParmas())
                NewRecoverPara.print()
                print("Paras Len: %d" % len(self.RecoverParasList))
                
                tPanel = RecoveryPanel(self.TaskPanel,NewRecoverPara,topVbox = self.TaskVbox)
                global GRecoveryTaskNum
                
                self.TaskVbox.Insert(GRecoveryTaskNum,tPanel,flag = wx.EXPAND|wx.TOP,border = 4)
                # self.rVbox.Detach(self.recovery_ST)
                self.TaskVbox.Layout()
            else:
                print("Paras inValid !!")
            dislog.Destroy()
        

    def OnQuit(self, e):
        self.Destroy()
        self.Close()
        # self.newFrame.Destroy()

def  main():
    
    app = wx.App()
    ex = MyFrame(None, title='Vision Speed')
    # ex= RecoveryFrame(None,RecoveryParmas())

    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    
    main()


