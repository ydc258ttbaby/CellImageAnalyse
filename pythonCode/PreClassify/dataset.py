import os  
import numpy as np
import csv
import random
import config
csvName = config.DatasetCSVName
CSVPathName = config.taskRootDir
dir_list = config.DatasetImgDirList
label_list = config.DatasetLabelList

if(not(os.path.exists(CSVPathName))):
    os.mkdir(CSVPathName)

def csv_writer(file_dir,train_csv_writer,test_csv_writer,verify_csv_writer,total_csv_writer,label):
    for root, dirs, files in os.walk(file_dir):
        for fileName in files:
            fileFullName = os.path.join(file_dir,fileName)
            imageNameAndLabel = [fileFullName] + [label]
            total_csv_writer.writerow(imageNameAndLabel)
            randomNum = random.randint(0,9)
            if randomNum <= 1:
                test_csv_writer.writerow(imageNameAndLabel)
            else:
                if randomNum <= 3:
                    verify_csv_writer.writerow(imageNameAndLabel)
                else:
                    train_csv_writer.writerow(imageNameAndLabel)

with  \
    open("%s\\%s_train.csv"  % (CSVPathName,csvName), 'w', newline='') as train_csvfile, \
    open("%s\\%s_test.csv"   % (CSVPathName,csvName), 'w', newline='') as test_csvfile, \
    open("%s\\%s_verify.csv" % (CSVPathName,csvName), 'w', newline='') as verify_csvfile, \
    open("%s\\%s_total.csv"  % (CSVPathName,csvName), 'w', newline='') as total_csvfile \
                :
    print('create dataset')
    header = ['image_name']+['label']
    train_csv_writer = csv.writer(train_csvfile)
    train_csv_writer.writerow(header)
    test_csv_writer = csv.writer(test_csvfile)
    test_csv_writer.writerow(header)
    verify_csv_writer = csv.writer(verify_csvfile)
    verify_csv_writer.writerow(header)
    total_csv_writer = csv.writer(total_csvfile)
    total_csv_writer.writerow(header)

    for dir,label in zip(dir_list,label_list):
        csv_writer(dir,train_csv_writer,test_csv_writer,verify_csv_writer,total_csv_writer,label)
    
    print("create dataset completed !!! ")