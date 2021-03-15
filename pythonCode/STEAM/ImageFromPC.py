import myFunction as MF
import time
file_path = 'D:/天津/原始数据/580/split/580_2.Wfm.bin'
file_path = "C:\\ydc\学习\\实验室\\天津院\\示波器数据\\Data\\goodRes\\wave101.Wfm.bin"
file_path = "Z:\\RTx\\RefWaveforms\\wave_20201112_171146_5.Wfm.bin"

netPATH = "C:\\Users\\86133\\data\\cellLoc_net.pth"
netPATH = "C:\\ydc\\坚果云文件\\我的坚果云\\two_cifar_net.pth"
t1 = time.time()
import os
bin_path = "F:\\天津\\原始数据\\208\\"
bin_list = [os.path.join(bin_path, i) for i in os.listdir(bin_path) if i[-8:] == '.Wfm.bin']
# print(bin_list )
i = 0
for file_path in bin_list:
    i += 1
    # file_path = "D:\\天津\\原始数据\\580\\split\\580_%d.Wfm.bin" %(i+1)
    # file_path = "D:\\天津\\原始数据\\580\\wave20201023_T142146_200_1.Wfm.bin"
    dataNum = 50
    MF.BinDataToCropImage(file_path,\
                                dataNum=dataNum,\
                                # displayImage=True,\
                                # display=True,\
                                # widPar=0.5,\
                                passBandPar=1.0,\
                                netPATH = netPATH,\
                                useNet = False,\
                                imgName="F:\\天津\\图像数据\\208\\"+file_path[19:-8]
                                    )
    # break
print(time.time()-t1)

