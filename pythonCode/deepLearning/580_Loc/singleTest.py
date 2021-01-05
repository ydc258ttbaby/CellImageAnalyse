import os
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import struct

if __name__ == '__main__':
 
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv1d(1, 16, 100, 50)
            self.pool = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(16, 32, 50,20)
            self.fc1 = nn.Linear(6368, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 4)

        def forward(self, x):
            # print(x.shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 6368)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    PATH = './cifar_net.pth'
    net.load_state_dict(torch.load(PATH))

    with torch.no_grad():
            
        file_path = "C:\\Users\\dcyang\\data\\580\\Rawdata\\580_107.Wfm.bin"
        data_bin = open(file_path, 'rb+')
        data_size = os.path.getsize(file_path)
        data_list = []
        for i in range(int(data_size/4)):
            data_i = data_bin.read(4) # 每次输出一个字节
            num = struct.unpack('f', data_i) # B 无符号整数，b 有符号整数
            data_list.append(num[0])

        data_bin.close()
        data = np.array([data_list])
        mean = np.mean(data)
        data = data - mean
        std = np.std(data)
        data = data/std
        data = np.array([data])
        images = torch.from_numpy(data).float()
        outputs = net(images)
        output = outputs.numpy()
                