import os
import struct
import time
file_path = 'C:/Users/dcyang/data/580/Rawdata/580_1.Wfm.bin'
data_bin = open(file_path, 'rb+')
data_size = os.path.getsize(file_path)
print(data_size)
t1 = time.time()
data_list = []
for i in range(int(data_size/4)):
    data_i = data_bin.read(4) # 每次输出一个字节
    #print(data_i)
    #print(i)
    num = struct.unpack('f', data_i) # B 无符号整数，b 有符号整数
    data_list.append(num[0])
    #print(num)
data_bin.close()
print(len(data_list))
print(time.time()-t1)

import csv
import numpy as np

file_path = "E:\\我的坚果云\\19-10-30_10_2.Wfm.csv"
t1 = time.time()
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    data = np.asarray(list(reader))
    print(np.shape(data))
print(time.time()-t1)
