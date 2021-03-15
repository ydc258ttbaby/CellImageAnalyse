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
from skimage.io import imshow as skiImshow
t = time.time()
file_path = r"F:\tianjin\RawData\Seventh\211116\wave20210310_T135028_160_1.Wfm.bin"
MF.BinDataToCropImage(
                            file_path = file_path,\
                            imgSavePath = r"E:\test",\
                            midTargetValue = 160,\
                            bLinearNor = False,\
                            crop_H = 32,\
                            full_H = 56,\
                            bPreCrop = True,\
                            # widPar=1.0,\
                            # passBandPar=0.5\
                            )
print("   Total time: %.2f s" % (time.time()-t))


# a= np.array([range(1,10)])
# b = a
# c = np.dot(a.T,b)
# print(c)

# d = c[1:10:2,1:5]
# print(d)

# def f(a):
#     a = a + 1
# a = np.arange([10,10])
# print(a)
# # print(np.where(a>5 and a < 9))
# b = np.where(a>5,(255-160)*(a-5)/(10-5)+160,(160-0)*(a-0)/(5-0)+0)
# print(b)

# b = np.where(a>5,a+1,a-1)
# print(b)

# imgPath = r"E:\test\Image_20.png"
# image = Image.open(imgPath)
# # image.show()
# imageNp = np.array(image)
# print(np.max(imageNp))
# print(np.shape(imageNp))
# cropImg = MF.f_imgCrop(imageNp,32,56,6)
# skiImshow(cropImg)
# plt.show()

