import matplotlib.pyplot as plt
import time 
import numpy as np
from myClass import RTO,Serialport
import myFunction as MF



acNum = 5 # 每个bin文件包含的图像个数
count = 0 
acCount = 1 # 每次采集的bin文件个数
widPar = 0.5 
passBandPar = 1.0
display = False
bRealtimeRecovery = False
ser = Serialport('com7',115200)
print(ser)
ser.open()
ser.run()

while(count < acCount):
    timeStr = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    binFileName = 'C:\\Users\\Public\\Documents\\Rohde-Schwarz\\RTx\\RefWaveforms\\wave_%s_%d.bin' % (timeStr,acNum)
    rto = RTO('192.168.1.1',acNum,binFileName)
    rto.open()
    rto.AcAndSave()
    ser.stop()
    if ~bRealtimeRecovery:
        time.sleep(1)
    else:
        file_path = 'Z:\\RTx\\RefWaveforms\\wave_%s_%d.Wfm.bin' % (timeStr,acNum)
        MF.ImageTransRecoverySave(file_path,\
                                    dataNum=acNum,\
                                    # displayImage=True,\
                                    # display=True,\
                                    # widPar=1.0,\
                                    # passBandPar=0.5\
                                    )

    rto.close()
    ser.run()
    count += 1

ser.close()


