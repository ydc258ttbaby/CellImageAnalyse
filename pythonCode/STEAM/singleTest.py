import os
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import struct
import myFunction as MF
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
    netPATH = "C:\\Users\\86133\\data\\cellLoc_net.pth"
    net.load_state_dict(torch.load(netPATH,map_location='cpu'))

    with torch.no_grad():
            
        file_path = "D:\\天津\\原始数据\\580\\split\\580_107.Wfm.bin"
        data = MF.bin_data_read(file_path)
        data = data - np.mean(data)
        data = data/np.std(data)
        data = np.array([data])
        data = torch.from_numpy(data).float()
        outputs = net(data)
        output = outputs.numpy()
        print(output)
                