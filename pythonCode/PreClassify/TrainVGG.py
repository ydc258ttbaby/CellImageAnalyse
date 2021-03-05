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
import random
# Ignore warnings
import warnings
import csv
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

from YDC_DL_utility import Rescale,RandomCrop,ToTensor,grayToRGB,show_image,CellsLabelDataset,VGGNet,Conv3x3BNReLU

dir_list = ["F:\\天津\\图像数据\\天津第六次图像数据\\剪裁后\\210694\\空图",\
            "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\大细胞",\
            "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\空图",\
            "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\淋巴",\
            "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\其他",\
            "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\杂质"]

csvName = 'lskj'
CSVPathName = "F:\\DeepLearningRes\\TwoPreClassify"


transformed_dataset_train = CellsLabelDataset(csv_file="%s\\lskj_train.csv" % CSVPathName,
                                            root_dir_list=dir_list,
                                            transform=transforms.Compose([
                                                grayToRGB(),
                                                Rescale(250),
                                               RandomCrop(224),
                                               ToTensor()
                                            ]))
transformed_dataset_test = CellsLabelDataset(csv_file="%s\\lskj_verify.csv" % CSVPathName,
                                            root_dir_list=dir_list,
                                            transform=transforms.Compose([
                                                grayToRGB(),
                                                Rescale(250),
                                               RandomCrop(224),
                                               ToTensor()
                                            ]))


trainloader = DataLoader(transformed_dataset_train, batch_size=4,
                        shuffle=True, num_workers=2)
testloader = DataLoader(transformed_dataset_test, batch_size=1,
                        shuffle=False, num_workers=2)

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
        model = VGGNet(block_nums,2)
        return model

    net = VGG16()
    bLoad = 1
    if bLoad == 1:
        PATH = '%s\\two_pre_classify_0305.pth' % CSVPathName
        if os.path.exists(PATH):
            net.load_state_dict(torch.load(PATH))
    if(torch.cuda.is_available):
        net.to(device)
    
    #------------------------------------------------------------------
    # Define a Loss function and optimizer
    #------------------------------------------------------------------
   

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9) # four classify up to 73% and 91%
    # optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)

    #------------------------------------------------------------------
    # Train the network
    #------------------------------------------------------------------
    print('Start Training')
    startTime = time.time()
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image']
            labels = data['label']
            if(torch.cuda.is_available):
                inputs, labels = data['image'].to(device), data['label'].to(device)
            labels = labels.flatten()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0
    print('time: %.3f' % (time.time()-startTime))
    # print('Finished Training')

    #------------------------------------------------------------------
    # quickly save our trained model
    #------------------------------------------------------------------
    PATH = '%s\\five_pre_classify_0305.pth' % CSVPathName
    torch.save(net.state_dict(), PATH)

    #------------------------------------------------------------------
    # Test the network on the test data
    #------------------------------------------------------------------
    dataiter = iter(testloader)
    images= dataiter.next()['image']
    labels =  dataiter.next()['label']
    #------------------------------------------------------------------
    # load the model
    #------------------------------------------------------------------
    net = VGG16()
    net.load_state_dict(torch.load(PATH))
    if(torch.cuda.is_available):
        net.to(device)

    #------------------------------------------------------------------
    # the network performs on the whole dataset
    #------------------------------------------------------------------
    correct = 0
    total = 0
    count_cancer = 0
    count_wrongiscancer = 0
    count_canceriswrong = 0
    with torch.no_grad():
        for data in testloader:
            images = data['image']
            labels = data['label']
            if(torch.cuda.is_available):
                images, labels = data['image'].to(device), data['label'].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            if(predicted.item()==0 and labels.item()!=0):
                count_wrongiscancer += 1
            if(predicted.item()!=0 and labels.item()==0):
                count_canceriswrong += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print('wrongiscacer: %d canceriswrong: %d total: %d' % (count_wrongiscancer,count_canceriswrong,total))
    print('Accuracy : %d %%' % (
        100 * correct / total))
    # print('Cancer Accuracy : %d %%' % (
    # 100 * (1-(count_wrongiscancer+count_canceriswrong) / total)))
