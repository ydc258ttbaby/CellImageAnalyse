from PIL import Image
import os

file_dir = "F:\\武汉\\北大胸腔第二次数据\\RGB400TotalClassifyRes\\阳性样本人工分类_323_331"
count = 1
scale = 0.5
for root, dirs, files in os.walk(file_dir):
    # for fileName in files:
    #     print(fileName)
    for name in files:
        filename = os.path.join(root, name)
        print(filename)
        count += 1
        if(os.path.isfile(filename)):
            img = Image.open(filename)
            width = int(img.size[0]*scale)
            height = int(img.size[1]*scale)
            img = img.resize((width, height), Image.ANTIALIAS)
            img.save(filename)
print(count)