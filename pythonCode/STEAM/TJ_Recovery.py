import os
import myFunction as MF
import shutil

binFilePath = r"E:\Tianjin\temp"
moveDesPath = r"E:\Tianjin\原始数据\211220"
imgSavePath = r"E:\Tianjin\图像数据\211220"

if os.path.exists(moveDesPath) == False:
    os.makedirs(moveDesPath)
if os.path.exists(imgSavePath) == False:
    os.makedirs(imgSavePath)

for file in os.listdir(binFilePath):
    if file.endswith(".Wfm.bin"):
         
        file_path = os.path.join(binFilePath,file)
        
        print(file_path) 
        MF.BinDataToCropImage(\
                                file_path=file_path,\
                                imgSavePath = imgSavePath,\
                                midTargetValue = 160,\
                                bLinearNor = False,\
                                crop_H = 32,\
                                full_H = 56,\
                                bPreCrop = True,\
                                    )
        
        moveDesFile = os.path.join(moveDesPath,file)
        shutil.move(file_path,moveDesFile)
        