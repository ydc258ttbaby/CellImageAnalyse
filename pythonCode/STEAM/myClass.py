import pyvisa as visa
class RTO():
    '''
    rto = RTO('192.168.1.1',5)
    rto.open()
    rto.AcAndSave()
    rto.close()
    '''
    def __init__(self,ip,aCount=5,saveName='C:\\Users\\Public\\Documents\\Rohde-Schwarz\\RTx\\RefWaveforms\\wave_default.bin'):
        self.ip = ip
        self.addr = 'TCPIP::%s::inst0::INSTR' % self.ip
        self.aCount = aCount
        self.saveName = saveName
        self.resourceManager = visa.ResourceManager()
    def open(self):
        self.instance = self.resourceManager.open_resource(self.addr,write_termination= '\r', read_termination='\r')
        print(self.instance.query('*IDN?'))
    def close(self):
        self.instance.close()
    def AcAndSave(self):
        self.instance.write('EXPort:WAVeform:RAW OFF')
        self.instance.write('EXPort:WAVeform:INCXvalues OFF')
        self.instance.write('EXPort:WAVeform:FASTexport ON')
        self.instance.write('CHANnel1:WAVeform1:STATe 1')
        self.instance.write('EXPort:WAVeform:SOURce C1W1')
        self.instance.write('EXPort:WAVeform:SCOPe WFM')
        self.instance.write('EXPort:WAVeform:DLOGging ON')
        self.instance.write('ACQuire:COUNt %d' % self.aCount)
        self.instance.write('EXPort:WAVeform:NAME "%s"' % self.saveName)
        print('SAVE NAME "%s"' % self.saveName)
        self.instance.write('RUNSingle')

import serial
class Serialport():
    '''
    ser = Serialport('com7',115200)
    ser.open()
    ser.run()
    time.sleep(3)
    ser.stop()
    ser.close()
    '''
    def __init__(self,port,baud=115200):
        self.port = port
        self.baud = baud
        self.com = serial.Serial(self.port,self.baud)
    def open(self):
        if not self.com.isOpen():
            self.com.open()
    def close(self):
        self.com.close()
    def run(self):
        self.com.write('run\r'.encode())
    def stop(self):
        self.com.write('stop\r'.encode())

import torch.nn as nn
import torch.nn.functional as F
import torch
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