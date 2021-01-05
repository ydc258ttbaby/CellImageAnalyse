import myFunction as MF
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
from scipy.fftpack import fft,ifft
import myFunction as MF
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from myClass import Net,Net2
file_path = "F:\\天津\\原始数据\\208\\wave20201123_T112443_50_1.Wfm.bin"
t = time.time()
rawData = MF.bin_data_read(file_path)
print(time.time()-t)
