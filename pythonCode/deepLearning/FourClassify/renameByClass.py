import os  
import numpy as np
import csv
def rename(file):
    with open("D:\\神经网络\\csv12281514\\%s_res.csv" %file, 'r', newline='') as res_csvfile:
        res_csv_reader = csv.reader(res_csvfile)
        rows = []
        for row in res_csv_reader:
            rows.append(row)
        print(rows)
        i = 1
        file_dir = "E:\\清华\\32um\\%s\\" %file
        for root, dirs, files in os.walk(file_dir):
            for fileName in files:
                newName = (rows[i][0]+'_'+fileName)
                os.rename(file_dir+'\\'+fileName,file_dir+'\\'+newName)
                i+=1
# rename('229')