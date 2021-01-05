from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
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

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 200, 100)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 100,50)
        self.fc1 = nn.Linear(1248, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1248)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
        file_path = os.path.join(self.root_dir,
                                self.label_frame.iloc[idx, 0])
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
        data = np.array([data_list])
        mean = np.mean(data)
        data = data - mean
        std = np.std(data)
        data = data/std
        label = self.label_frame.iloc[idx, 1:5]
        label = label.astype(np.float64)
        label = np.array([label])
        sample = {'bindata': data, 'label': label}
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

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}

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
        data, label = sample['bindata'], sample['label']
        
        return {'bindata': torch.from_numpy(data).float(),
                'label': torch.from_numpy(label).float()}

''' create dataset with-transform and imshow '''
transformed_dataset_test = CellsLabelDataset(csv_file='data/580/cells_label_test.csv',
                                            root_dir='data/580/Rawdata/',
                                            transform=transforms.Compose([
                                               ToTensor()
                                            ]))

testloader = DataLoader(transformed_dataset_test, batch_size=1,
                        shuffle=False, num_workers=2)
                       
classes = ('523','576')

if __name__ == '__main__':
    
    #------------------------------------------------------------------
    # Define a CUDA device
    #------------------------------------------------------------------
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    #------------------------------------------------------------------
    # Define a Convolutional Neural Network
    #------------------------------------------------------------------
    

     
    #------------------------------------------------------------------
    # Define a Loss function and optimizer
    #------------------------------------------------------------------
   
    criterion = nn.MSELoss()

    #------------------------------------------------------------------
    # load the model
    #------------------------------------------------------------------
    net = Net2()
    PATH = './two_cifar_net.pth'
    net.load_state_dict(torch.load(PATH))
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
        with open('data/580/cells_label_res2.csv', 'w', newline='') as res_csvfile:
            res_csv_writer = csv.writer(res_csvfile)
            header = ['norleft']+['norright']+['left']+['right']
            res_csv_writer.writerow(header)
            for data in testloader:
                
                images = data['bindata']
                print(images)
                print(type(images))
                labels = data['label']
                # if(torch.cuda.is_available and openCUDA):
                #     images, labels = data['bindata'].to(device), data['label'].to(device)
                
                outputs = net(images)
                # print(type(outputs.numpy))
                
                output = outputs.numpy()
                label = labels.numpy()
                res_csv_writer.writerow(output[0,:])
                res_csv_writer.writerow(label[0,0,:])
                # print(outputs)
                # print(labels)
                # MSEloss += criterion(outputs,labels).item()
                count += 1
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (abs(predicted - labels)<10000).sum().item()
                break
                

    # print('MSE : %.3f' % (MSEloss/count))
    print('time: %.3f count: %d' % (time.time()-startTime,count))