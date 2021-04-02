# import myFunction as MF
from threading import Thread,Event,Condition
import threading
import wx
import time

import matplotlib.pyplot as plt
import numpy as np
import os
import struct
from scipy.fftpack import fft,ifft
import myFunction as MF
import time
from functools import reduce
import shutil


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
                    fullH = "56",\
                    bSingleRecover = True \
                ):
        self.bSingleRecover = bSingleRecover
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


from skimage.transform import resize as skiResize
from skimage.io import imsave as skiImsave
from skimage.transform import resize as skiResize
import math
class RecoverOpera():
    def __init__(self,RecoverPara):
        self.Para = RecoverPara
        self.Init()

    def Init(self):
        self.progress =0
        self.paused = False
        self.stoped = False
        self.pauseEvent = Event()
        self.stopEvent = Event()


    def run(self,bPaused,bStoped,pauseEvent,stopEvent):
        if(self.Para.bSingleRecover):
            listDir = os.listdir(self.Para.binFilePath)
            for item in listDir:
                if(item[-7:] == "Wfm.bin"):
                    self.BinDataToCropImage(os.path.join(self.Para.binFilePath,item),\
                                            crop_H = self.str2int(self.Para.cropH),\
                                            full_H = self.str2int(self.Para.fullH),\
                                            imgSavePath = os.path.join(self.Para.imgSavePath,self.Para.serialNumList[0]),\
                                            bPreCrop = self.Para.bPreCrop,\
                                            bLinearNor = self.Para.bLinearNor,\
                                            midTargetValue = self.str2int(self.Para.midTargetValue)\
                                            
                                                )
                    if(self.Para.bMove):
                        if not os.path.exists(self.Para.moveDesPath):
                                os.makedirs(self.Para.moveDesPath)
                        shutil.move(os.path.join(self.Para.binFilePath,item),\
                                    os.path.join(self.Para.moveDesPath,item))
        else:
            for serialNum in self.Para.serialNumList:
                listDir = os.listdir(os.path.join(self.Para.binFilePath,serialNum))
                for item in listDir:
                    if(item[-7:] == "Wfm.bin"):
                        self.BinDataToCropImage(\
                                            file_path = os.path.join(self.Para.binFilePath,serialNum,item),\
                                            crop_H = self.str2int(self.Para.cropH),\
                                            full_H = self.str2int(self.Para.fullH),\
                                            imgSavePath = os.path.join(self.Para.imgSavePath,serialNum),\
                                            bPreCrop = self.Para.bPreCrop,\
                                            bLinearNor = self.Para.bLinearNor,\
                                            midTargetValue = self.str2int(self.Para.midTargetValue)\
                                                )
    def pause(self):
        print("call pause")
        self.paused = True
        self.pauseEvent.clear()
    def resume(self):
        print("call resume")

        self.paused = False
        self.pauseEvent.set()
    def stop(self):
        print("call stop")
        self.stoped = True

    def GetProgress(self):
        return progress

    # 以下函数封装起来，不改动

    def BinDataToCropImage(self,file_path,crop_H,full_H,imgSavePath,bPreCrop = True,bLinearNor = False,midTargetValue = 160,widPar=0.5,passBandPar=1.0,display=False,displayImage=False,useNet = False):
        print(file_path)
        lastPath = os.path.abspath(os.path.join(file_path,"..")) # 获取上级目录
        file_name = file_path[len(lastPath)+1:-8] # 获取文件名
        print(file_name)
        t = time.time()
        rawData = self.bin_data_read(file_path)
        print('    Load time: %.2f s' % (time.time()-t))
        L = len(rawData)
        dataNum = round(L/2000000)
        print("     Data num: %d" % dataNum)
        for i in np.arange(dataNum):
            if i > 10:
                break
            if(not(self.stopEvent.isSet())):
                print(self.stoped)
                print(self.paused)
                tid = threading.get_ident()
                print("recover opera id : %d"%tid)
                if self.paused: # check pause
                    print("paused ...")
                    self.pauseEvent.wait()
                else: # run
                    t1 = time.time()
                    data = rawData[round(i/dataNum*L):round((i+1)/dataNum*L)]
                    image = self.image_recovery(\
                                                data=data,\
                                                widPar=widPar,\
                                                passBandPar=passBandPar,\
                                                display=display,\
                                                coord = [0.5,0.5,0.5,1.0],\
                                                bPreCrop = bPreCrop\
                                                    )
                    # print('%d recovery time: %.2f s' % (i,time.time()-t1))
                    [row,col] = np.shape(image)
                    image = skiResize(image,[row,round(col*0.35*2.5)])
                    image = self.f_imgCrop(image,crop_H,full_H,6)
                    # image = skiResize(image,[500,500])
                    minSrcValue = np.min(image)
                    maxSrcValue = np.max(image)
                    if bLinearNor:
                        image = 255*(image-minSrcValue)/(maxSrcValue-minSrcValue)
                    else:
                        image = np.where(image>0,(255-midTargetValue)*(image-0)/(maxSrcValue-0)+midTargetValue,(midTargetValue-0)*(image-minSrcValue)/(0-minSrcValue)+0)
                    image = image.astype(np.uint8)
                    if not os.path.exists(imgSavePath):
                        os.makedirs(imgSavePath)

                    imageName = imgSavePath + '\\' + file_name + '_' + str(i+1) +'.png'
                    skiImsave(imageName,image)
                    print(' Process time: %.2f s' % (time.time()-t1))

    def bin_data_read(self,file_path):
        tid = threading.get_ident()
        print("load thread id : %d"%tid)
        data_bin = open(file_path, 'rb+')
        data_size = os.path.getsize(file_path)
        
        data_total = data_bin.read(data_size)

        data_tuple = struct.unpack(str(int(data_size/4))+'f', data_total)
        # print(len(data_tuple))
        return data_tuple
    def str2int(self,s):
        return reduce(lambda x,y:x*10+y, map(lambda s:{'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}[s], s))

    def image_recovery(self,data,widPar,passBandPar,display,bPreCrop = False,coord = [0.5,0.5,1.0,1.0],offset = [0.0,0.0]):
        if display:
            MF.plot_1D_single(data[:3000])
        if bPreCrop:
            RawL = len(data)
            HalfIndexRange = pow(2,len(bin(round((coord[2]+2*offset[0])*RawL-1)))-3)
            data = data[round(coord[0]*RawL)-HalfIndexRange:round(coord[0]*RawL)+HalfIndexRange]
        L = len(data)
        # print(L)
        data = data - np.mean(data)
        data_f = (fft(data))

        if display: 
            MF.plot_1D_single(abs(data_f[:10000]))
        RepRate = np.argmax(abs(data_f[100:4500]))+100
        # print('RepRate: %d'%RepRate)
        Width = round(widPar*L/RepRate)*2
        PassBand = round(passBandPar*RepRate)
        filter =  np.hstack([0,np.ones(PassBand),np.zeros(L-1-2*PassBand),np.ones(PassBand)])
        FilterData = np.real(ifft(data_f*filter))
        if display:
            MF.plot_1D_single(FilterData[:3000])

        SubData = FilterData[:4*Width]
        diffData = np.diff(np.sign(np.diff(SubData)))
        Locs = np.where(diffData==-2)[0]

        if Locs[0]-Width/2 > 0:
            FirstPulsePos = Locs[0]-Width/2
        else:
            FirstPulsePos = Locs[1]-Width/2
        FirstPulsePos = int(FirstPulsePos)
        FirstPulse = data[FirstPulsePos:FirstPulsePos+Width]
        # if display:
        #     MF.plot_1D_single(FirstPulse)

        CrossCor = np.zeros((Width,),dtype = float)
        partdata = data[L - 2*Width:]
        for i in np.arange(Width):
            cc = np.sum(FirstPulse*partdata[i:Width + i])
            CrossCor[i] = cc
        LastPulsePos = np.argmax(CrossCor)+L-2*Width+1


        FilterData = FilterData[FirstPulsePos:LastPulsePos]

        ColNum = 1
        diffData = np.diff(np.sign(np.diff(FilterData)))

        ColIndex = np.where(diffData==-2)[0]
        ColNum = len(ColIndex)+1

        Duration = (LastPulsePos-FirstPulsePos)/(ColNum-1)
        StartPoint = np.round(np.arange(ColNum)*Duration)+FirstPulsePos

        ImgIndex = (np.tile(np.arange(Width+1).reshape(Width+1,1)-1,(1,ColNum))+np.tile(StartPoint,(Width+1,1)))
        [row,col] = np.shape(ImgIndex)
        if ~bPreCrop:
            ImgIndex = ImgIndex[round(max((coord[1])-coord[3]/2-offset[1],0)*row):round(min((coord[1])+coord[3]/2+offset[1],1)*row),:]
        else:
            ImgIndex = ImgIndex[:,round((HalfIndexRange-(coord[2]/2+offset[0])*RawL)/row):round((HalfIndexRange+(coord[2]+2*offset[0])*RawL)/row)]
        [row,col] = np.shape(ImgIndex)
        ImgIndex = ImgIndex.reshape(1,-1)


        ImgIndex = ImgIndex.astype(int)
        res = data[ImgIndex]
        res = res.reshape(row,-1)

        backGround = np.mean(res[:,0:10],axis=1)
        res = res - np.tile(backGround.reshape(-1,1),(1,col))
        return res

    def f_imgCrop(self,image,crop_H,full_H,sigma):
        avg = np.mean(image)
        [m,n] = np.shape(image)
        cropRangeR = round(m/4)
        cropRangeC = cropRangeR

        extendImg = np.full((m+cropRangeR,n+cropRangeC),avg)
        extendImg[round(cropRangeR/2):round(cropRangeR/2)+m,round(cropRangeC/2):round(cropRangeC/2)+n] = image
        
        smallSize = 100
        smallImg = extendImg[0:(m+cropRangeR):round((m+cropRangeR)/smallSize),0:(n+cropRangeC):round((n+cropRangeC)/smallSize)]
        [smallImgRow,smallImgCol] = np.shape(smallImg)
        ImgMatrix = np.power(abs(smallImg - avg),0.25)
        leftTopR = round(cropRangeR/m/2*smallImgRow)
        leftTopC = round(cropRangeC/n/2*smallImgCol)
        
        

        fK1=1.0/(2*sigma*sigma)
        fK2=fK1/math.pi
        iSize = leftTopR+1
        out = np.zeros([1,iSize])
        step = math.floor(iSize/2 + 0.5)
        for j in np.arange(iSize):
            x = j-step
            out[0,j] = fK2 * math.exp(-x*x*fK1)
        
        model = out / np.sum(out)
        model2 = np.dot(model.T,model)

        halfRangeR = round(leftTopR/2)
        halfRangeC = round(leftTopC/2)
        sumMatrix = np.zeros([smallImgRow,smallImgCol])
        for r in range(leftTopR,(smallImgRow-leftTopR+1)):
            for c in range(leftTopC,(smallImgCol-leftTopC+1)):
                sumMatrix[r,c] = np.sum(model2*ImgMatrix[r-halfRangeR:r+halfRangeR+1,c-halfRangeC:c+halfRangeC+1])
            
        [row,col] = np.unravel_index(sumMatrix.argmax(), sumMatrix.shape)
        # row = np.mean(row)
        # col = np.mean(col)
        row = round(row * ((m+cropRangeR)/smallImgRow))
        col = round(col * ((n+cropRangeC)/smallImgCol))
        resHeight = round(m*crop_H/full_H)
        resWidth = resHeight
        row = max(min(row,round(m+cropRangeR/2-resHeight/2)),round(cropRangeR/2+resHeight/2))
        col = max(min(col,round(n+cropRangeC/2-resWidth/2)),round(cropRangeC/2+resWidth/2))
        return extendImg[round(row-resHeight/2):round(row+resHeight/2),round(col-resWidth/2):round(col+resWidth/2)]


class RecoverThread(Thread):
    def __init__(self,NewRecoverPara,name):
        super(RecoverThread,self).__init__()
        self.paused = False
        self.stoped = False
        self.pauseEvent = Event()
        self.stopEvent = Event()
        self.daemon = True
        self.name = name
        self.NewRecoverPara = NewRecoverPara
        self.count = 0
        self.RecoverInstance = RecoverOpera(NewRecoverPara)
        
        tid = threading.get_ident()
        print("recover thread id : %d"%tid)
    def run(self):
        # self.resume()
        
        self.RecoverInstance.run(self.paused,self.stoped,self.pauseEvent,self.stopEvent)
        
        print("end run")
                
    def pause(self):
        self.RecoverInstance.pause()

    def resume(self):
        self.RecoverInstance.resume()
        
    def stop(self):        
        self.RecoverInstance.stop()


    def GetProcess(self):
        return self.count
    def GetParas(self):
        return self.NewRecoverPara


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

    def GetParas(self):
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
        self.bSingleRecover_CB = wx.CheckBox(setting_Panel,pos = (170,93),label = "是否单一样本恢复")
        self.bSingleRecover_CB.SetValue(True)

        midTargetValue_ST.SetFont(font2)
        cropH_ST.SetFont(font2)
        fullH_ST.SetFont(font2)
        self.midTargetValue_TC.SetFont(font2)
        self.cropH_TC.SetFont(font2)
        self.fullH_TC.SetFont(font2)
        self.bMoved_CB.SetFont(font2)
        self.bLinearNor_CB.SetFont(font2)
        self.bPreCrop_CB.SetFont(font2)
        self.bSingleRecover_CB.SetFont(font2)


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
            self.rParmas.bSingleRecover = self.bSingleRecover_CB.GetValue()
            
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
        tid = threading.get_ident()
        print("recover panel id : %d"%tid)
        
        
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

        self.moreInfoFrame = RecoveryFrame(None,NewRecoverPara = self.Para,task = self.task)


    def CreateTimer(self):
        self.timer = wx.Timer(self,1)
        self.Bind(wx.EVT_TIMER,self.OnTimer,self.timer)
    def OnTimer(self,e):
        if(self.TaskProcess > 100):
            self.timer.Stop()
            
        self.TaskProcess = self.task.GetProcess()
        self.gauge.SetValue(self.TaskProcess)

        # print("timer: %d"%self.TaskProcess)

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
        self.moreInfoFrame.Show()
    
        
    def CancelTask(self,e):
        global GRecoveryTaskNum
        GRecoveryTaskNum -= 1
        self.task.stop()
        self.timer.Stop()
        if(self.moreInfoFrame):
            self.moreInfoFrame.Destroy()
        self.Destroy()
        self.vbox.Layout()
        
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
    def __init__(self,parent,NewRecoverPara,task):
        super(RecoveryFrame,self).__init__(parent)
        self.task = task
        self.NewRecoverPara = NewRecoverPara
        self.Init()
        self.CreatePanel()
        # font1 = wx.Font(17, family = wx.MODERN,style = wx.NORMAL,weight = wx.NORMAL,underline = False,faceName = '微软雅黑 Light')
        # self.SetFont(font1)
        tid = threading.get_ident()
        print("recover Frame id : %d"%tid)
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
        setting_SB = wx.StaticBox(setting_Panel,size = (300,190),label = "其他设置")

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
        bSingleRecover_CB = wx.CheckBox(setting_Panel,pos = (170,123),label = "是否单一样本恢复")
        bSingleRecover_CB.SetValue(self.NewRecoverPara.bSingleRecover)

        
        bShowImage_CB = wx.CheckBox(setting_Panel,pos = (170,153),label = "是否预览图像")
        bShowImage_CB.SetValue(True)


        midTargetValue_ST.SetFont(font2)
        cropH_ST.SetFont(font2)
        fullH_ST.SetFont(font2)
        midTargetValue_TC.SetFont(font2)
        cropH_TC.SetFont(font2)
        fullH_TC.SetFont(font2)
        bMoved_CB.SetFont(font2)
        bLinearNor_CB.SetFont(font2)
        bPreCrop_CB.SetFont(font2)
        bShowImage_CB.SetFont(font2)
        bSingleRecover_CB.SetFont(font2)

        leftInfoVbox.Add(setting_Panel,flag = wx.LEFT|wx.EXPAND|wx.TOP,border = 10)

        hbox.Add(leftInfoPanel,proportion = 0,flag = wx.EXPAND|wx.ALL,border = 5)

        rightImgPanel = wx.Panel(panel)
        rightBagSizer = wx.GridBagSizer(3,3)

        
        self.UpateImgPathList()

        index = 0
        for item in self.ImgPathList:
            img = wx.Image(item, wx.BITMAP_TYPE_ANY)
            img = img.Scale(160,160)
            temp = wx.StaticBitmap(rightImgPanel, wx.ID_ANY,wx.BitmapFromImage(img))
            xPos = 180*(index//3)
            yPos = 180*(index%3)
            print("x: %d y: %d index: %d" %(xPos,yPos,index))
            temp.SetPosition((xPos,yPos))
            index += 1

        hbox.Add(rightImgPanel,proportion = 0,flag = wx.EXPAND|wx.ALL,border = 20)
        panel.SetSizer(hbox)
        hbox.Fit(self)
    def UpateImgPathList(self):
        ImgRootPath = self.NewRecoverPara.imgSavePath

        self.ImgPathList = [r"F:\tianjin\ImageData\Sixth\210694\bigcell\wave20210224_T140059_50_1.Wfm.bin_2000002_5397.png_wave20210224_T140059_50_1.Wfm.bin_2000002_5397.png_wave20210224_T140059_50_1.Wfm.bin_2000002_5397.png_wave20210224_T140059_50_1.Wfm.bin_2000002_5397.png",\
                            r"F:\tianjin\ImageData\Sixth\210694\bigcell\wave20210303_T102448_160_1.Wfm.bin_2000002_7251.png_wave20210303_T102448_160_1.Wfm.bin_2000002_7251.png_wave20210303_T102448_160_1.Wfm.bin_2000002_7251.png_wave20210303_T102448_160_1.Wfm.bin_2000002_7251.png",\
                            r"F:\tianjin\ImageData\Sixth\210694\bigcell\wave20210303_T134716_160_1.Wfm.bin_2000002_2890.png_wave20210303_T134716_160_1.Wfm.bin_2000002_2890.png_wave20210303_T134716_160_1.Wfm.bin_2000002_2890.png_wave20210303_T134716_160_1.Wfm.bin_2000002_2890.png",\
                            r"F:\tianjin\ImageData\Sixth\210694\bigcell\wave20210303_T134842_160_1.Wfm.bin_2000002_3181.png_wave20210303_T134842_160_1.Wfm.bin_2000002_3181.png_wave20210303_T134842_160_1.Wfm.bin_2000002_3181.png_wave20210303_T134842_160_1.Wfm.bin_2000002_3181.png",\
                            r"F:\tianjin\ImageData\Sixth\210694\bigcell\wave20210303_T135611_160_2.Wfm.bin_2000002_3203.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3203.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3203.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3203.png",\
                            r"F:\tianjin\ImageData\Sixth\210694\bigcell\wave20210303_T135611_160_2.Wfm.bin_2000002_3214.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3214.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3214.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3214.png",\
                            r"F:\tianjin\ImageData\Sixth\210694\bigcell\wave20210303_T135611_160_2.Wfm.bin_2000002_3215.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3215.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3215.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3215.png",\
                            r"F:\tianjin\ImageData\Sixth\210694\bigcell\wave20210303_T135611_160_2.Wfm.bin_2000002_3223.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3223.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3223.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3223.png",\
                            r"F:\tianjin\ImageData\Sixth\210694\bigcell\wave20210303_T135611_160_2.Wfm.bin_2000002_3227.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3227.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3227.png_wave20210303_T135611_160_2.Wfm.bin_2000002_3227.png",\
                            ]


class MyFrame(wx.Frame):
    def __init__(self,parent,title):
        super(MyFrame,self).__init__(parent,title=title)
        self.Init()
        self.SetWindowsInfo()
        self.CreateMenuBar()
        self.CreatePanel()
        tid = threading.get_ident()
        print("My Frame id : %d"%tid)
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

        thcount = threading.active_count()
        print("thread Num: %d" %thcount )

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
            NewRecoverPara = dislog.GetParas()
            if NewRecoverPara.bValid:
                self.RecoverParasList.append(dislog.GetParas())
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
    # para = RecoveryParmas()
    # para.serialNumList = ["12321"]
    # para.binFilePath = r"E:\Tianjin\temp\temp"
    # recover = RecoverOpera(para)
    # recover.Init()
    # recover.run()


