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

from create_cells_dataset import create_dataset
from renameByClass import rename

plt.ion()   # interactive mode

class VGGNet(nn.Module):
    def __init__(self, block_nums,num_classes=1000):
        super(VGGNet, self).__init__()

        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels,out_channels))
        for i in range(1,block_num):
            layers.append(Conv3x3BNReLU(out_channels,out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0),-1)
        out = self.classifier(x)
        return out
def Conv3x3BNReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )   

def VGG16():
    block_nums = [2, 2, 3, 3, 3]
    model = VGGNet(block_nums,7)
    return model

def show_image(image, label):
    """Show image with label"""
    plt.imshow(image)
    if(label[0] == 0):
        plt.scatter([100], [100], s=10, marker='.', c='r')
    if(label[0] == 1):
        plt.scatter([100], [100], s=10, marker='.', c='b')
    plt.pause(1)  # pause a bit so that plots are updated

class CellsLabelDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.label_frame.iloc[idx, 0])
        image = io.imread(img_name)
        # print(np.shape(image))
        # image = np.array([image])
        # print(np.shape(image))
        # image = image.transpose((1,2,0))
        label = self.label_frame.iloc[idx, 1]
        label = np.array([label])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        # print(np.shape(image))
        image = skiTransform.resize(image, (new_h, new_w))
        # print(np.shape(image))
        
        return {'image': image, 'label': label}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose(2,0,1)
        image = np.array(image)
        #image = image[0,:]
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).long()}
class grayToRGB(object):
    """convert gray to rgb"""
    def __call__(self,sample):
        image, label = sample['image'], sample['label']
        # print(np.shape(image))
        image = np.array([image])
        image = np.repeat(image,3,axis=0)
        image = image.transpose(1,2,0)
        # print(np.shape(image))
       
        return {'image': image, 'label': label}



if __name__ == '__main__':

    fileList = ['208894','209116','209172']
    for file in fileList:
        # file = '208331'
        create_dataset(file)
        print('create dataset completed !!!')

        ''' create dataset with-transform and imshow '''
        transformed_dataset_test = CellsLabelDataset(csv_file="D:\\神经网络\\csv12281514\\%s.csv" %file,
                                                    root_dir="E:\\清华\\32um\\%s\\" %file,
                                                    transform=transforms.Compose([
                                                        grayToRGB(),
                                                        Rescale(250),
                                                        RandomCrop(224),
                                                        ToTensor()
                                                    ]))

        testloader = DataLoader(transformed_dataset_test, batch_size=1,
                                shuffle=False, num_workers=2)
        
        #------------------------------------------------------------------
        # Define a CUDA device
        #------------------------------------------------------------------
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)

        #------------------------------------------------------------------
        # load the model
        #------------------------------------------------------------------
        net = VGG16()
        PATH = "D:\\神经网络\\1202_1743_7_94_2_99_cifar_net.pth"
        net.load_state_dict(torch.load(PATH,map_location='cpu'))
        # net.load_state_dict(torch.load(PATH))
        openCUDA =True
        if(torch.cuda.is_available and openCUDA):
            net.to(device)

        # outputs = net(images)
        # _, predicted = torch.max(outputs, 1)

        # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(1)))

        #------------------------------------------------------------------
        # the network performs on the whole dataset
        #------------------------------------------------------------------
        correct = 0
        total = 0
        MSEloss = 0
        count = 0
        startTime = time.time()
        with torch.no_grad():
            with open("D:\\神经网络\\csv12281514\\%s_res.csv" %file, 'w', newline='') as res_csvfile:
                res_csv_writer = csv.writer(res_csvfile)
                header = ['label']
                res_csv_writer.writerow(header)
                for data in testloader:
                    images = data['image']
                    if(torch.cuda.is_available and openCUDA):
                        images= data['image'].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    res_csv_writer.writerow([predicted.item()])
                    count += 1
                    print(count)
                    
                    

        # print('MSE : %.3f' % (MSEloss/count))
        print('time: %.3f count: %d' % (time.time()-startTime,count))

        print('test and save completed !!!')
        rename(file)
        print('rename completed !!!')