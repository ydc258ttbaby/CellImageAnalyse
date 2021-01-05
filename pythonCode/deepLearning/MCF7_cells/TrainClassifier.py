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
        image = np.array([image])
        image = image.transpose((1,2,0))
        label = self.label_frame.iloc[idx, 1]
        label = np.array([label])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample
# ''' create dataset without-transform and imshow '''
# cell_dataset = CellsLabelDataset(csv_file='data/cells/cells_label.csv',
#                                     root_dir='data/cells/imgTotal/')
# for i in range(len(cell_dataset)):
#     sample = cell_dataset[i]

#     # print(i, sample['image'].shape, sample['label'].shape)

#     if i == 3:
#         break
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
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).long()}

''' create dataset with-transform and imshow '''
transformed_dataset_train = CellsLabelDataset(csv_file='data/cells/cells_label_train.csv',
                                            root_dir='data/cells/imgTotal/',
                                            transform=transforms.Compose([
                                               Rescale(64),
                                               RandomCrop(32),
                                               ToTensor()
                                            ]))
transformed_dataset_test = CellsLabelDataset(csv_file='data/cells/cells_label_test.csv',
                                            root_dir='data/cells/imgTotal/',
                                            transform=transforms.Compose([
                                               Rescale(64),
                                               RandomCrop(32),
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
classes = ('noise','before','after')

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
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 48, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(48, 96, 5)
            self.fc1 = nn.Linear(2400, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 3)

        def forward(self, x):
            # print(x.shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 2400)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()
    if(torch.cuda.is_available):
        net.to(device)
    
    #------------------------------------------------------------------
    # Define a Loss function and optimizer
    #------------------------------------------------------------------
   

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #------------------------------------------------------------------
    # Train the network
    #------------------------------------------------------------------
    print('Start Training')
    startTime = time.time()
    for epoch in range(20):  # loop over the dataset multiple times

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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
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
    net = Net()
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
    with torch.no_grad():
        for data in testloader:
            images = data['image']
            labels = data['label']
            if(torch.cuda.is_available):
                images, labels = data['image'].to(device), data['label'].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy : %d %%' % (
        100 * correct / total))

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