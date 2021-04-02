from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
from skimage import transform as skiTransform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
from YDC_DL_utility import Rescale,RandomCrop,ToTensor,grayToRGB,show_image,CellsLabelDataset,VGGNet,Conv3x3BNReLU

import config

dir_list = config.DatasetImgDirList
CSVPathName = config.taskRootDir
pthName = config.pthName
TestLabelCSVFileName = config.TestLabelCSVFileName
PATH = '%s\\%s' % (CSVPathName,pthName)
print(PATH)
classify_num = 2
test_csv_file = "%s\\%s" % (CSVPathName,TestLabelCSVFileName)
transformed_dataset_test = CellsLabelDataset(csv_file=test_csv_file,\
                                             root_dir_list=dir_list,\
                                             transform=transforms.Compose([\
                                               grayToRGB(),\
                                               Rescale(250),\
                                               RandomCrop(224),\
                                               ToTensor()\
                                            ]))

testloader = DataLoader(transformed_dataset_test, batch_size=1, shuffle=False, num_workers=1)

if __name__ == '__main__':
    
    #------------------------------------------------------------------
    # Define a CUDA device
    #------------------------------------------------------------------
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    #------------------------------------------------------------------
    # Define a Convolutional Neural Network
    #------------------------------------------------------------------
    
    def VGG16():
        block_nums = [2, 2, 3, 3, 3]
        model = VGGNet(block_nums,classify_num)
        return model

    
    net = VGG16()
    net.load_state_dict(torch.load(PATH))
    if(torch.cuda.is_available):
        net.to(device)
    #------------------------------------------------------------------
    # Test the network on the test data
    #------------------------------------------------------------------
    dataiter = iter(testloader)
    images= dataiter.next()['image']
    labels =  dataiter.next()['label']
    

    correct = 0
    total = 0
    count_cancer = 0
    count_wrongiscancer = 0
    count_canceriswrong = 0  
    TP = 0
    NP = 0
    TN = 0
    NN = 0
    with torch.no_grad():
        for data in testloader:
            images = data['image']
            labels = data['label']
            if(torch.cuda.is_available):
                images, labels = data['image'].to(device), data['label'].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            if(predicted.item()==1 and labels.item()==1):
                TP += 1
            if(predicted.item()==1 and labels.item()==0):
                NP += 1
            if(predicted.item()==0 and labels.item()==1):
                NN += 1
            if(predicted.item()==0 and labels.item()==0):
                TN += 1
            
            total += labels.size(0)
            if total % 10 == 0 :
                print(total)
            correct += (predicted == labels).sum().item()
    print("TP: %d NP: %d TN: %d NN: %d"%(TP,NP,TN,NN))
    # print('wrongiscacer: %d canceriswrong: %d total: %d' % (count_wrongiscancer,count_canceriswrong,total))
    print('Accuracy : %d %%' % (
        100 * correct / total))
    # print('Cancer Accuracy : %d %%' % (
    # 100 * (1-(count_wrongiscancer+count_canceriswrong) / total)))
