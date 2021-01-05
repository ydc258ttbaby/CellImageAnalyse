import csv
import random
i = 1
with open('data/580/cells_label_train.csv', 'w', newline='') as train_csvfile,\
    open('data/580/cells_label_test.csv', 'w', newline='') as test_csvfile,\
    open('data/580/580LabelBBOXNoneZero.csv', 'r', newline='') as total_csvfile:
    print('ydc')
    train_csv_writer = csv.writer(train_csvfile)
    test_csv_writer = csv.writer(test_csvfile)
    header = ['filename']+['x']+['y']+['w']+['h']+['isNoise']
    train_csv_writer.writerow(header)
    test_csv_writer.writerow(header)
    total_csv_reader = csv.reader(total_csvfile)
    print(type(total_csv_reader))
    for row in total_csv_reader:
        #print(row)
        
        # if i > 2500:
        #     break
        # i += 1
        
        if random.randint(0,9) > 0:
            train_csv_writer.writerow(row)
        else:
            test_csv_writer.writerow(row)
  