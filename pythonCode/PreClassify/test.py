import os
file_dir = r"F:\DeepLearningRes\BigCell"
for root, dirs, files in os.walk(file_dir):
    for fileName in files:
        print(os.path.join(file_dir,fileName))

