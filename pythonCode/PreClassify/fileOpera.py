import os
import shutil
import pandas as pd
import csv
import pygal
def moveFileToChildFile(file_dir,child_dir):
    bFirst = True
    for root, dirs, files in os.walk(file_dir): 
        if(bFirst):
            bFirst = False
            for dir in dirs:
                print(dir)
                shutil.move(file_dir+'\\'+ dir,file_dir + '\\%s' % child_dir)
                newFileDir = file_dir + '\\' + dir
                os.mkdir(newFileDir)
                shutil.move(file_dir + '\\%s' % child_dir,newFileDir)
# moveFileToChildFile(r"F:\tianjin\ImageData\Third\LinearNor\209116",'unclass')

# 删除整个文件夹
# filePath = r"E:\Tianjin\temp\wave20210311_T115207_160_49.Wfm.bin"
# shutil.rmtree(filePath)
# os.remove(filePath)

# 将file_dir下面所有文件，移动到another_dir目录下
def moveFileToAnotherFile(file_dir,another_dir):
    bFirst = True
    for root,dirs,files in os.walk(file_dir):
        for file in files:
            shutil.move(file_dir+'\\'+file,another_dir)   
# 单个文件夹文件的移动
moveFileToAnotherFile(r"F:\tianjin\ImageData\Sixth\210962\label_1",r"F:\tianjin\ImageData\Sixth\210962\unclass")
# 多个文件夹文件循环移动
# fileList = ['210962','210965']
# for file in fileList:
#     file_fir = 'F:\\tianjin\\ImageData\\Sixth\\AfterCrop\\%s\\label_1' % file
#     another_dir ='F:\\tianjin\\ImageData\\Sixth\\AfterCrop\\%s\\unclass' % file
#     moveFileToAnotherFile(file_fir,another_dir)

# 将root_fi\file_name改为root_fir\new_name
def rename(root_fir,file_name,new_name):
    if os.path.exists(os.path.join(root_fir,file_name)):
        os.rename(os.path.join(root_fir,file_name),os.path.join(root_fir,new_name))

# root_dir = r"F:\beijing\ImageData\Third\LinearNor"
# fileList = []
# for root,dirs,files in os.walk(root_dir):
#     fileList = dirs
#     break
# for file in fileList:
#     root_fir = os.path.join(root_dir,file)
#     file_name = r"大细胞"
#     new_name = r"bigcell"
#     rename(root_fir,file_name,new_name)
#     file_name = r"非大细胞"
#     new_name = r"nonbigcell"
#     rename(root_fir,file_name,new_name)
#     file_name = r"淋巴"
#     new_name = r"lp"
#     rename(root_fir,file_name,new_name)
#     file_name = r"未分类"
#     new_name = r"unclass"
#     rename(root_fir,file_name,new_name)
#     file_name = r"小颗粒"
#     new_name = r"particle"
#     rename(root_fir,file_name,new_name)
#     file_name = r"杂质"
#     new_name = r"impurity"
#     rename(root_fir,file_name,new_name)
#     file_name = r"空图"
#     new_name = r"blank"
#     rename(root_fir,file_name,new_name)
#     file_name = r"癌细胞"
#     new_name = r"bigCell"
#     rename(root_fir,file_name,new_name)
#     file_name = r"间皮细胞"
#     new_name = r"bigCell"
#     rename(root_fir,file_name,new_name)
#     file_name = r"超大细胞"
#     new_name = r"unknown"
#     rename(root_fir,file_name,new_name)
#     file_name = r"清晰大细胞"
#     new_name = r"bigCell"
#     rename(root_fir,file_name,new_name)

# rootList = []
# rootrootPath = r"F:\beijing\ImageData"
# for root,dirs,files in os.walk(rootrootPath):
#     rootList = (dirs)
#     break  
# indexName = rootList
# seriesList = []
# fileTotalIndex = []
# for Name in indexName:
#     rootPath = rootrootPath + "\\" + Name + "\\LinearNor"
#     # rootPath = rootrootPath + "\\" + Name
#     print(rootPath)
#     fileIndexs = []
#     for root,dirs,files in os.walk(rootPath):
#         fileIndexs = (dirs)
#         break
#     fileTotalIndex.extend(fileIndexs)
#     for fileIndex in fileIndexs:
#         file_dir = os.path.join(rootPath,fileIndex)
#         # print(file_dir)
#         bFirst = 1
#         root_dir =''
#         child_dir_list = []
#         nums_list = []

#         for root, dirs, files in os.walk(file_dir):
#             if (bFirst):
#                 root_dir = root
#                 child_dir_list = dirs
#                 # print(dirs)
#                 bFirst = 0
#             else:
#                 nums_list.append(len(files))
#         val = pd.Series(nums_list,index=child_dir_list)
#         seriesList.append(val)
#         # dataFrame = pd.DataFrame.from_dict(dic,orient="index",columns=[fileIndex])
# print(seriesList)
# print(fileTotalIndex)
# dataFrame = pd.DataFrame(seriesList,index = fileTotalIndex)
# print(dataFrame)
# dataFrame.to_csv('F:\\DeepLearningRes\\CellNum\\beijng.csv')

