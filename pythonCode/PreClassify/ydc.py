import pandas as pd
import os
# import xlwt
count=0
fileRoots = []
fileCounts =[]
for root, dirs, files in os.walk(r"F:\tianjin\ImageData\Sixth\AfterCrop\210668"):
    count=0
    fileRoots.append(root)
    
    for each in files:
        count += 1
    fileCounts.append(count)


print(fileRoots)
print(fileCounts)
dic1 = {"tt":1,"ydc":2,"ty":3}
dic2 = {"tt":1,"ty":3}
results = pd.DataFrame({'dic1':dic1})
results.insert(1,'dic2',[12,12,12])
# results = pd.DataFrame({'目录':fileRoots,'A文件个数':fileCounts,'B文件个数':fileCounts})
print(results)
# results.to_excel('D:\list2.xls')



