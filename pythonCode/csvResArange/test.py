import os
import csv 
import numpy as np
from functools import reduce
file_dir = "D:\\神经网络\\csv12191316"
classNum = 7
namelist = ['']
totalRes = np.array([np.arange(classNum)])
# print(totalRes)

def str2int(s):
    return reduce(lambda x,y:x*10+y, map(lambda s:{'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}[s], s))

# print(totalRes)
for root, dirs, files in os.walk(file_dir):
    for fileName in files:
        if fileName[-7:] == 'res.csv':
            sampleIndex = fileName[:-8]

            perSampleRes = np.zeros(shape=(1,classNum),dtype='int')
            
            namelist.append(sampleIndex)
            
            filePath = root+'\\'+fileName
            # print(sampleIndex)
            with open(filePath, 'r', newline='') as res_csvfile:
                res_csv_reader = csv.reader(res_csvfile)
                
                for i,row in enumerate(res_csv_reader):
                    if i > 0:
                        # print(row[0])
                        perSampleRes[0][str2int(row[0])] += 1
            
            totalRes = np.append(totalRes,perSampleRes,axis=0)

totalRes = totalRes.astype(int).T
print(namelist)
print(totalRes)

with open("D:\\神经网络\\RGB400TotalSevenClassiftRes12191216.csv", 'w', newline='') as res_csvfile:
    res_csv_writer = csv.writer(res_csvfile)
    res_csv_writer.writerow(namelist)
    for row in totalRes:
        res_csv_writer.writerow(row)
