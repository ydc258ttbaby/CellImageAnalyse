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

''' create dataset with-transform and imshow '''
transformed_dataset_train = CellsLabelDataset(csv_file='/home/ydc/vggNet/data/523_576_FourClassify/cells_label_train.csv',
                                            root_dir='/home/ydc/vggNet/data/523_576_FourClassify/imgTotal400/',
                                            transform=transforms.Compose([
                                                Rescale(250),
                                               RandomCrop(224),
                                               ToTensor()
                                            ]))
transformed_dataset_test = CellsLabelDataset(csv_file='/home/ydc/vggNet/data/523_576_FourClassify/cells_label_test.csv',
                                            root_dir='/home/ydc/vggNet/data/523_576_FourClassify/imgTotal400/',
                                            transform=transforms.Compose([
                                                Rescale(250),
                                               RandomCrop(224),
                                               ToTensor()
                                            ]))

# for i in range(len(transformed_dataset_train)):
#     sample = transformed_dataset_train[i]

#     print(i, sample['image'].shape, sample['label'].shape)

#     if i == 3:
#         break

trainloader = DataLoader(transformed_dataset_train, batch_size=1,
                        shuffle=True, num_workers=2)
testloader = DataLoader(transformed_dataset_test, batch_size=1,
                        shuffle=True, num_workers=2)
classes = ('cancer','impurity','lymph','mesothelial')

if __name__ == '__main__':
    
    #------------------------------------------------------------------
    # Define a CUDA device
    #------------------------------------------------------------------
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    #------------------------------------------------------------------
    # Loading and normalizing CIFAR10
    #------------------------------------------------------------------
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

    #------------------------------------------------------------------
    # show some of the training images
    #------------------------------------------------------------------
    
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    
    # dataiter = iter(trainloader)    # get some random training images
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images)) # show images
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4))) # print labels

    #------------------------------------------------------------------
    # Define a Convolutional Neural Network
    #------------------------------------------------------------------
    
    class Vgg16(torch.nn.Module):
        def __init__(self):
            super(Vgg16, self).__init__()
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

            self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

            self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        def forward(self, X):
            h = F.relu(self.conv1_1(X))
            h = F.relu(self.conv1_2(h))
            relu1_2 = h
            h = F.max_pool2d(h, kernel_size=2, stride=2)

            h = F.relu(self.conv2_1(h))
            h = F.relu(self.conv2_2(h))
            relu2_2 = h
            h = F.max_pool2d(h, kernel_size=2, stride=2)

            h = F.relu(self.conv3_1(h))
            h = F.relu(self.conv3_2(h))
            h = F.relu(self.conv3_3(h))
            relu3_3 = h
            h = F.max_pool2d(h, kernel_size=2, stride=2)

            h = F.relu(self.conv4_1(h))
            h = F.relu(self.conv4_2(h))
            h = F.relu(self.conv4_3(h))
            relu4_3 = h

            return [relu1_2, relu2_2, relu3_3, relu4_3]

    class VGG_16(nn.Module):
        def __init__(self):
            super(VGG_16, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 113),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 113),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, 2, 57),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 2, 57),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 29),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 2, 29),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 2, 29),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(256, 512, 3, 2, 15),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 2, 15),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 2, 15),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(512, 512, 3, 2, 8),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 2, 8),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 2, 8),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.layer6 = nn.Sequential(
                nn.Linear(7*7*512, 4096),
                nn.Linear(4096, 4096),
                nn.Linear(4096, 1000)
            )
    
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = x.view(x.size(0), -1)
            x = self.layer6(x)
            output = F.softmax(x, dim=1)
            return output

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
        model = VGGNet(block_nums,4)
        return model

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.pool = nn.MaxPool2d(2, 2)
            self.conv1_1 = nn.Conv2d(1, 64, 3)
            self.conv1_2 = nn.Conv2d(64,64, 3)
            self.conv2_1 = nn.Conv2d(64, 128, 3)
            self.conv2_2 = nn.Conv2d(128, 128, 3)
            self.conv3_1 = nn.Conv2d(128, 256, 3)
            self.conv3_2 = nn.Conv2d(256, 256, 3)
            self.conv3_3 = nn.Conv2d(256, 256, 3)
            self.conv4_1 = nn.Conv2d(256, 512, 3)
            self.conv4_2 = nn.Conv2d(512, 512, 3)
            self.conv4_3 = nn.Conv2d(512, 512, 3)
            self.conv5_1 = nn.Conv2d(512, 512, 3)
            self.conv5_2 = nn.Conv2d(512, 512, 3)
            self.conv5_3 = nn.Conv2d(512, 512, 3)
            self.fc1 = nn.Linear(4096, 4096)
            self.fc2 = nn.Linear(4096, 1000)
            self.fc3 = nn.Linear(1000, 4)

        def forward(self, x):
            # print(x.shape)
            x = self.pool(F.relu(self.conv1_2(self.conv1_1(x))))
            x = self.pool(F.relu(self.conv2_2(self.conv2_1(x))))
            x = self.pool(F.relu(self.conv3_3(self.conv3_2(self.conv3_1(x)))))
            x = self.pool(F.relu(self.conv4_3(self.conv4_2(self.conv4_1(x)))))
            x = self.pool(F.relu(self.conv5_3(self.conv5_2(self.conv5_1(x)))))
            x = x.view(-1, 4096)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = VGG16()
    # PATH = './cifar_net.pth'
    # net.load_state_dict(torch.load(PATH))
    if(torch.cuda.is_available):
        net.to(device)
    
    #------------------------------------------------------------------
    # Define a Loss function and optimizer
    #------------------------------------------------------------------
   

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00003, momentum=0.9) # four classify up to 73% and 91%
    # optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)

    #------------------------------------------------------------------
    # Train the network
    #------------------------------------------------------------------
    print('Start Training')
    startTime = time.time()
    for epoch in range(15):  # loop over the dataset multiple times

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
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
    print('time: %.3f' % (time.time()-startTime))
    # print('Finished Training')

    #------------------------------------------------------------------
    # quickly save our trained model
    #------------------------------------------------------------------
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    #------------------------------------------------------------------
    # Test the network on the test data
    #------------------------------------------------------------------
    dataiter = iter(testloader)
    images= dataiter.next()['image']
    labels =  dataiter.next()['label']
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    #------------------------------------------------------------------
    # load the model
    #------------------------------------------------------------------
    net = VGG16()
    net.load_state_dict(torch.load(PATH))
    if(torch.cuda.is_available):
        net.to(device)

    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)

    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(1)))

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
    print('wrongiscacer: %d canceriswrong: %d total: %d' % (count_wrongiscancer,count_canceriswrong,total))
    print('Accuracy : %d %%' % (
        100 * correct / total))
    print('Cancer Accuracy : %d %%' % (
    100 * (1-(count_wrongiscancer+count_canceriswrong) / total)))

    #------------------------------------------------------------------
    # lookat the performs of each class
    #------------------------------------------------------------------
    """
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in testloader:
            images = data['image']
            labels = data['label']
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    """