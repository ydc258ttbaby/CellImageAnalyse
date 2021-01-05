import matplotlib.pyplot as plt
import numpy as np
import os
import struct
from scipy.fftpack import fft,ifft
import myFunction as MF
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from myClass import Net,Net2
from functools import reduce
def str2int(s):
    return reduce(lambda x,y:x*10+y, map(lambda s:{'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}[s], s))
def plot_1D_single(data):
    x = np.arange(np.shape(data)[0])
    # print(x)
    plt.plot(x,data)
    plt.show()
# @profile
def data_read(file_path):
    data_bin = open(file_path, 'rb+')
    data_size = os.path.getsize(file_path)
    data_list = []
    for i in range(int(data_size/4)):
        data_i = data_bin.read(4) # 每次输出一个字节
        #print(data_i)
        #print(i)
        num = struct.unpack('f', data_i) # B 无符号整数，b 有符号整数
        data_list.append(num[0])
        #print(num)
    data_bin.close()
    data = np.array(data_list[10:])
    return data
# @profile
def image_recovery(data,widPar,passBandPar,display,coord = [0.5,0.5,1.0,1.0],offset = [0.0,0.0]):
    if display:
        MF.plot_1D_single(data[:3000])
    
    RawL = len(data)
    HalfIndexRange = pow(2,len(bin(round((coord[2]+2*offset[0])*RawL-1)))-3)
    data = data[round(coord[0]*RawL)-HalfIndexRange:round(coord[0]*RawL)+HalfIndexRange]
    L = len(data)
    print(L)
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
    # ImgIndex = ImgIndex[round(max((coord[1])-coord[3]/2-offset[1],0)*row):round(min((coord[1])+coord[3]/2+offset[1],1)*row),:]
    ImgIndex = ImgIndex[:,round((HalfIndexRange-(coord[2]/2+offset[0])*RawL)/row):round((HalfIndexRange+(coord[2]+2*offset[0])*RawL)/row)]
    [row,col] = np.shape(ImgIndex)
    ImgIndex = ImgIndex.reshape(1,-1)


    ImgIndex = ImgIndex.astype(int)
    Image = data[ImgIndex]
    Image = Image.reshape(row,-1)

    backGround = np.mean(Image[:,0:10],axis=1)
    Image = Image - np.tile(backGround.reshape(-1,1),(1,col))
    return Image
# @profile
def ImageTransRecoverySave(file_path,netPATH,imgName='C:\\Users\\86133\\data\\testImage\\test',dataNum=1,widPar=0.5,passBandPar=1.0,display=False,displayImage=False,useNet = False):
    print(file_path)
    t1 = time.time()
    rawData = MF.bin_data_read(file_path)
    print('load completed !!')
    print(time.time()-t1)
    L = len(rawData)
    for i in np.arange(dataNum):
        data = rawData[round(i/dataNum*L):round((i+1)/dataNum*L)]
        if useNet:
            data = np.array([[data]])
            coord = MF.bin_data_to_coord(netPATH,data)
            print(coord)
            ImageMatrix = MF.image_recovery(data[0,0,:],widPar,passBandPar,display,coord=[coord[0,0],0.5,coord[0,1],1.0])
        else:
            ImageMatrix = MF.image_recovery(data,widPar,passBandPar,display,coord = [0.5,0.5,0.5,1.0])
        print('recovery completed %d' % i)
        # print(time.time()-t1)
        # print(np.shape(ImageMatrix))
        ImageSize = np.shape(ImageMatrix)
        ImageMatrix = (255*(ImageMatrix-np.min(ImageMatrix))/(np.max(ImageMatrix)-np.min(ImageMatrix))).astype(np.int32)
        image = Image.fromarray(ImageMatrix)
        image = image.convert('L')
        image = image.resize((round(ImageSize[1]*0.35*5),ImageSize[0]),Image.ANTIALIAS)
        # print(np.shape(image))
        imageName = imgName+'_'+str(i)+'.png'
        image.save(imageName)
        if displayImage:
            plt.imshow(image)
            plt.ion()
            plt.pause(1)
# @profile
def bin_data_read(file_path):
    t1 = time.time()
    data_bin = open(file_path, 'rb+')
    data_size = os.path.getsize(file_path)
    
    data_total = data_bin.read(data_size)

    data_tuple = struct.unpack(str(int(data_size/4))+'f', data_total)
    # print(len(data_tuple))
    return data_tuple
# @profile
def bin_data_to_coord(netPATH,data):
    net = Net2()
    net.load_state_dict(torch.load(netPATH,map_location='cpu'))

    with torch.no_grad():
        data = data - np.mean(data)
        data = data/np.std(data)
        # data = np.array([data])
        data = torch.from_numpy(data).float()
        outputs = net(data)
        output = outputs.numpy()
    return output

