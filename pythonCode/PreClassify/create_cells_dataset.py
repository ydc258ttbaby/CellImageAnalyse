from skimage import io
import os  
import numpy as np
import csv
import random
 
def csv_writer(file_dir,train_csv_writer,test_csv_writer,verify_csv_writer,total_csv_writer,label):
    for root, dirs, files in os.walk(file_dir):
        for fileName in files:
            imageNameAndLabel = [fileName] + [label]
            total_csv_writer.writerow(imageNameAndLabel)
            randomNum = random.randint(0,9)
            if randomNum <= 1:
                test_csv_writer.writerow(imageNameAndLabel)
            else:
                if randomNum <= 3:
                    verify_csv_writer.writerow(imageNameAndLabel)
                else:
                    train_csv_writer.writerow(imageNameAndLabel)




csvName = 'lskj'
CSVPathName = "F:\\DeepLearningRes\\TwoPreClassify"

with  \
    open("%s\\%s_train.csv"  % (CSVPathName,csvName), 'w', newline='') as train_csvfile, \
    open("%s\\%s_test.csv"   % (CSVPathName,csvName), 'w', newline='') as test_csvfile, \
    open("%s\\%s_verify.csv" % (CSVPathName,csvName), 'w', newline='') as verify_csvfile, \
    open("%s\\%s_total.csv"  % (CSVPathName,csvName), 'w', newline='') as total_csvfile \
                :
    print('ydc')
    header = ['image_name']+['label']
    train_csv_writer = csv.writer(train_csvfile)
    train_csv_writer.writerow(header)
    test_csv_writer = csv.writer(test_csvfile)
    test_csv_writer.writerow(header)
    verify_csv_writer = csv.writer(verify_csvfile)
    verify_csv_writer.writerow(header)
    total_csv_writer = csv.writer(total_csvfile)
    total_csv_writer.writerow(header)

    dir_list = ["F:\\天津\\图像数据\\天津第六次图像数据\\剪裁后\\210694\\空图",\
                "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\大细胞",\
                "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\空图",\
                "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\淋巴",\
                "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\其他",\
                "F:\\北京\\图像数据\\北京第六次图像数据\\210668\\杂质"]
    label_list = [0,1,0,0,0,0]
    for dir,label in zip(dir_list,label_list):
        csv_writer(dir,train_csv_writer,test_csv_writer,verify_csv_writer,total_csv_writer,label)
    
    # 空图-0 大细胞-1 淋巴-2 杂质-3 其他-4
    print("create dataset completed !!! ")