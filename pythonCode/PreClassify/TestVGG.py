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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

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
    ''' create dataset with-transform and imshow '''
    # testList = ['wh_test','wh_extra_total','tj_331_total','tj_209116_total','bj_209172_total','bj_209116_total']
    testList = ['wh_test']
    for testFile in testList:
        print(testFile)
        transformed_dataset_test = CellsLabelDataset(csv_file='F:/PythonCode/tmp/TotalData/label/%s.csv' % testFile,
                                                    root_dir='F:/PythonCode/tmp/TotalData/total/',
                                                    transform=transforms.Compose([
                                                        grayToRGB(),
                                                        Rescale(250),
                                                    RandomCrop(224),
                                                    ToTensor()
                                                    ]))


        testloader = DataLoader(transformed_dataset_test, batch_size=1,
                                shuffle=True, num_workers=2)


            
        #------------------------------------------------------------------
        # Define a CUDA device
        #------------------------------------------------------------------

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)

        #------------------------------------------------------------------
        # Loading and normalizing
        #------------------------------------------------------------------
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        #------------------------------------------------------------------
        # Define a Convolutional Neural Network
        #------------------------------------------------------------------

        def Conv3x3BNReLU(in_channels,out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

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

        def VGG16():
            block_nums = [2, 2, 3, 3, 3]
            model = VGGNet(block_nums,7)
            return model

        # #------------------------------------------------------------------
        # # quickly save our trained model
        # #------------------------------------------------------------------
        PATH = 'F:/PythonCode/tmp/wh_cifar_net.pth'
        # torch.save(net.state_dict(), PATH)

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
        # net.load_state_dict(torch.load(PATH,map_location='cpu'))
        if(torch.cuda.is_available):
            net.to(device)

        #------------------------------------------------------------------
        # the network performs on the whole dataset
        #------------------------------------------------------------------
        FP = 0
        FN = 0
        PP = 0
        PN = 0
        TP = 0
        TN = 0
        RP = 0
        RN = 0
        with torch.no_grad():
            for data in testloader:
                images = data['image']
                labels = data['label']
                if(torch.cuda.is_available):
                    images, labels = data['image'].to(device), data['label'].to(device)
                outputs = net(images)
                print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                if(predicted.item()==0):
                    PP +=1
                if(predicted.item()==1):
                    PN +=1
                if(labels.item()==0):
                    RP +=1
                if(labels.item()==1):
                    RN +=1
                if(predicted.item()==0 and labels.item()!=0):
                    FP += 1
                if(predicted.item()!=0 and labels.item()==0):
                    FN += 1
                if(predicted.item()==0 and labels.item()==0):
                    TP += 1
                if(predicted.item()==1 and labels.item()==1):
                    TN += 1
                

        print('FP: %d FN: %d TP: %d TN: %d PP: %d PN: %d RP: %d RN: %d' % (FP,FN,TP,TN,PP,PN,RP,RN))
        ACC = (TP+TN)/(PP+PN)
        recall = TP/(TP+FN)
        print('ACC:%.4f' % (ACC))
        print('recall:%.4f' % (recall))
