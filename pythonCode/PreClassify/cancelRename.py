import os  
import numpy as np
import csv

def cancelRename(imgfile):
    
    file_dir = imgfile
    for root, dirs, files in os.walk(file_dir):
        for fileName in files:
            newName = fileName[2:]
            os.rename(file_dir+'\\'+fileName,file_dir+'\\'+newName)
            

fileList = ['210058']
for file in fileList:
    imgfile = "F:\\武汉\\图像数据\\武汉第六次图像数据\\%s\\未分类" %file
    cancelRename(imgfile)
    print("completed ")