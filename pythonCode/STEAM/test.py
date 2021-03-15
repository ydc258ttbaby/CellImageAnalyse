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
file_path = r"F:\tianjin\RawData\Seventh\211116\wave20210310_T135028_160_1.Wfm.bin"
MF.ImageTransRecoverySave(file_path,\
                            dataNum=160,\
                            imgName=r"E:\test\Image",\
                            # displayImage=True,\
                            # display=True,\
                            # widPar=1.0,\
                            # passBandPar=0.5\
                            )
print(time.time()-t)
