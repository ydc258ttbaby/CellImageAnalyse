import os

def rename(path):
    i = 0
    '该文件夹下所有的文件（包括文件夹）'
    FileList = os.listdir(path)
    '遍历所有文件'
    for files in FileList:
        '原来的文件路径'
        oldDirPath = os.path.join(path, files)
        '如果是文件夹则递归调用'
        if os.path.isdir(oldDirPath):
            rename(oldDirPath)
        '文件名'
        fileName = os.path.splitext(files)[0]
        '文件扩展名'
        fileType = os.path.splitext(files)[1]
        '新的文件路径'
        fileName = 'imageNoise_'
        newDirPath = os.path.join(path, fileName + str(i) + fileType)
        '重命名'
        os.rename(oldDirPath, newDirPath)
        i += 1

path = 'data/cells/imgTotalNoise'
rename(path)