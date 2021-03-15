from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
from skimage import transform as skiTransform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import struct
import csv
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


plt.ion()   # interactive mode

from YDC_DL_utility import Rescale,RandomCrop,ToTensor,grayToRGB,show_image,CellsLabelDataset,VGGNet,Conv3x3BNReLU

import random
import shutil

import config
fileList = config.fileList
TestFilePath = config.taskRootDir
pthName = config.pthName
moveDesName = config.moveDesName
TestImgFilePath = config.TestImgFilePath
PATH = "%s\\%s" %(TestFilePath,pthName)
classify_num = 2

def rename(csvfile,imgfile):
    with open(csvfile, 'r', newline='') as res_csvfile:
        print("rename")
        res_csv_reader = csv.reader(res_csvfile)
        rows = []
        for row in res_csv_reader:
            rows.append(row)
        i = 0
        file_dir = imgfile
        for root, dirs, files in os.walk(file_dir):
            for fileName in files:
                print(i)
                i+=1
                # if(rows[i][0] == '0'):
                #     os.remove(file_dir+'\\'+fileName)
                #     print("remove")
                #     continue
                
                newFileName = (os.path.abspath(os.path.join(file_dir,"..")))+'\\%s' %moveDesName
                if os.path.exists(newFileName) == False:
                    os.mkdir(newFileName)
                if (rows[i][0] == '1'):
                    shutil.move(file_dir+'\\'+fileName,newFileName+'\\'+fileName)
                    print("move")
                # newName = (rows[i][0]+'_'+fileName)
                # os.rename(file_dir+'\\'+fileName,file_dir+'\\'+newName)          
def csv_writer(file_dir,test_csv_writer,label):
    for root, dirs, files in os.walk(file_dir):
        for fileName in files:
            imageNameAndLabel = [os.path.join(file_dir,fileName)] + [label]
            test_csv_writer.writerow(imageNameAndLabel)
def create_dataset(csvfile,imgfile):
    with open(csvfile, 'w', newline='') as test_csvfile:
        print('create dataset')
        test_csv_writer = csv.writer(test_csvfile)
        header = ['image_name']+['label']
        test_csv_writer.writerow(header)
        csv_writer(imgfile,test_csv_writer,0)

if __name__ == '__main__':

    for file in fileList:
        
        csvfile = "%s\\test.csv" % (TestFilePath)
        imgfile = TestImgFilePath %file
        
        if os.path.exists(csvfile):
            os.remove(csvfile)
        create_dataset(csvfile,imgfile)

        bTest = 1
        if (bTest == 1):

            

            transformed_dataset_test = CellsLabelDataset(csv_file=csvfile,
                                                        root_dir_list=[imgfile],
                                                        transform=transforms.Compose([
                                                            grayToRGB(),
                                                            Rescale(250),
                                                            RandomCrop(224),
                                                            ToTensor()
                                                        ]))
            testloader = DataLoader(transformed_dataset_test, batch_size=1,
                                    shuffle=False, num_workers=24)
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # device = "cpu"
            print(device)

            def VGG16():
                block_nums = [2, 2, 3, 3, 3]
                model = VGGNet(block_nums,classify_num)
                return model

            net = VGG16()
            net.load_state_dict(torch.load(PATH))
            if(torch.cuda.is_available):
                net.to(device)

            correct = 0
            total = 0
            MSEloss = 0
            count = 0
            startTime = time.time()
            with torch.no_grad():
                with open(csvfile, 'w', newline='') as res_csvfile:
                    res_csv_writer = csv.writer(res_csvfile)
                    header = ['label']
                    res_csv_writer.writerow(header)
                    for data in testloader:
                        images = data['image']
                        if(torch.cuda.is_available):
                            images= data['image'].to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        res_csv_writer.writerow([predicted.item()])
                        count += 1
                        if count%100 == 0:
                            print(file+"-"+str(count))
            print('time: %.3f count: %d' % (time.time()-startTime,count))

            rename(csvfile,imgfile)
            print('test completed !!!')