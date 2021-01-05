from skimage import io
import os  
import numpy as np
import csv
import random
 
def csv_writer(file_dir,train_csv_writer,test_csv_writer,label):
    for root, dirs, files in os.walk(file_dir):
        for fileName in files:
            imageNameAndLabel = [fileName] + [label]
            if random.randint(0,9) > 0:
                train_csv_writer.writerow(imageNameAndLabel)
            else:
                test_csv_writer.writerow(imageNameAndLabel)

with open('data/cells/cells_label_train.csv', 'w', newline='') as train_csvfile,open('data/cells/cells_label_test.csv', 'w', newline='') as test_csvfile:
    train_csv_writer = csv.writer(train_csvfile)
    test_csv_writer = csv.writer(test_csvfile)
    header = ['image_name']+['label']
    train_csv_writer.writerow(header)
    test_csv_writer.writerow(header)
    csv_writer('data/cells/imgTotalBefore',train_csv_writer,test_csv_writer,1)
    csv_writer('data/cells/imgTotalAfter',train_csv_writer,test_csv_writer,2)
    csv_writer('data/cells/imgTotalNoise',train_csv_writer,test_csv_writer,0)

