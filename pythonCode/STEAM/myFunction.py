import matplotlib.pyplot as plt
import numpy as np
import os
import struct
from scipy.fftpack import fft,ifft
import myFunction as MF
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
def image_recovery(data,widPar,passBandPar,display,bPreCrop = False,coord = [0.5,0.5,1.0,1.0],offset = [0.0,0.0]):
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
# @profile
from skimage.transform import resize as skiResize
from skimage.io import imsave as skiImsave
def BinDataToCropImage(file_path,crop_H,full_H,imgSavePath,bPreCrop = True,bLinearNor = False,midTargetValue = 160,widPar=0.5,passBandPar=1.0,display=False,displayImage=False,useNet = False):
    print(file_path)
    lastPath = os.path.abspath(os.path.join(file_path,"..")) # 获取上级目录
    file_name = file_path[len(lastPath)+1:-8] # 获取文件名
    print(file_name)
    t = time.time()
    rawData = MF.bin_data_read(file_path)
    print('    Load time: %.2f s' % (time.time()-t))
    L = len(rawData)
    dataNum = round(L/2000000)
    print("     Data num: %d" % dataNum)
    for i in np.arange(dataNum):
        t1 = time.time()
        data = rawData[round(i/dataNum*L):round((i+1)/dataNum*L)]
        image = MF.image_recovery(\
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
        image = f_imgCrop(image,crop_H,full_H,6)
        # image = skiResize(image,[500,500])
        minSrcValue = np.min(image)
        maxSrcValue = np.max(image)
        if bLinearNor:
            image = 255*(image-minSrcValue)/(maxSrcValue-minSrcValue)
        else:
            image = np.where(image>0,(255-midTargetValue)*(image-0)/(maxSrcValue-0)+midTargetValue,(midTargetValue-0)*(image-minSrcValue)/(0-minSrcValue)+0)
        image = image.astype(np.uint8)
        imageName = imgSavePath + '\\' + file_name + '_' + str(i+1) +'.png'
        skiImsave(imageName,image)
        print(' Process time: %.2f s' % (time.time()-t1))
        break
# @profile
def bin_data_read(file_path):
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

from skimage.transform import resize as skiResize
import math
def f_imgCrop(image,crop_H,full_H,sigma):
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
    