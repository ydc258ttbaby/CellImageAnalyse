from skimage import io
import os
import numpy as np

file_dir = "F:\\PythonCode"
for root, dirs, files in os.walk(file_dir):
    for fileName in files:
        print(fileName)