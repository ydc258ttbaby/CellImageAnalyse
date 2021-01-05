import os

fileList = ['331','208879','208894','209116','209172']

for file in fileList:
    counter = 1
    filePath = root_dir="F:\\天津\\图像数据\\%s\\crop32\\" %file
    for root, dirs, files in os.walk(filePath):
        for fileName in files:
            os.rename(root+fileName,root+file+'_'+str(counter)+'.png')
            counter += 1
            
            print(root+fileName)
            print(root+file+'_'+str(counter)+'.png')