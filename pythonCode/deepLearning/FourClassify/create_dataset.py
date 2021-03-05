
import os  
import numpy as np
import csv
import random
 
def csv_writer(file_dir,test_csv_writer,label):
    for root, dirs, files in os.walk(file_dir):
        for fileName in files:
            imageNameAndLabel = [fileName] + [label]
            test_csv_writer.writerow(imageNameAndLabel)
def create_dataset(file):
    with open("D:\\神经网络\\csv12281514\\%s.csv" %file, 'w', newline='') as test_csvfile:
        print('ydc')
        test_csv_writer = csv.writer(test_csvfile)
        header = ['image_name']+['label']
        test_csv_writer.writerow(header)
        csv_writer("E:\\清华\\32um\\%s\\" %file,test_csv_writer,0)

