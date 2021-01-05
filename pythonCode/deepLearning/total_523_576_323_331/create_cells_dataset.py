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
            # if randomNum <= 1:
            #     test_csv_writer.writerow(imageNameAndLabel)
            # else:
            #     if randomNum <= 3:
            #         verify_csv_writer.writerow(imageNameAndLabel)
            #     else:
            #         train_csv_writer.writerow(imageNameAndLabel)

csvName = 'bj_209172'
with  \
    open("D:\\TotalData\\32umtotal\\%s_train.csv" % csvName, 'w', newline='') as train_csvfile, \
    open("D:\\TotalData\\32umtotal\\%s_test.csv" % csvName, 'w', newline='') as test_csvfile, \
        open("D:\\TotalData\\32umtotal\\%s_verify.csv" % csvName, 'w', newline='') as verify_csvfile, \
            open("D:\\TotalData\\32umtotal\\%s_total.csv" % csvName, 'w', newline='') as total_csvfile \
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
    # csv_writer("D:\\武汉\\第三次图像数据\\208894\\Rename\\间皮细胞",train_csv_writer,test_csv_writer,verify_csv_writer,total_csv_writer,1)
    # csv_writer("D:\\武汉\\第三次图像数据\\209116\\Rename\\癌细胞",train_csv_writer,test_csv_writer,verify_csv_writer,total_csv_writer,0)
    csv_writer("E:\\清华\\32um\\209172\\癌细胞",train_csv_writer,test_csv_writer,verify_csv_writer,total_csv_writer,0)

