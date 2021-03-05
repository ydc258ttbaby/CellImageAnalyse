import os  
import numpy as np
import csv

def cancelRename(imgfile):
    
    file_dir = imgfile
    for root, dirs, files in os.walk(file_dir):
        for fileName in files:
            newName = fileName[2:]
            os.rename(file_dir+'\\'+fileName,file_dir+'\\'+newName)
            

fileList = ['210668']
for file in fileList:
    imgfile = "F:\\北京\\图像数据\\北京第六次图像数据\\%s\\未分类" %file
    cancelRename(imgfile)