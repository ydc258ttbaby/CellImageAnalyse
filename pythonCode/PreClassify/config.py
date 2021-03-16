# [train]
DatasetCSVName = 'lskj'
DatasetImgDirList = [\
                     r"F:\tianjin\ImageData\Sixth\210668\blank",\
                     r"F:\beijing\ImageData\Sixth\210668\impurity",\
                     r"F:\beijing\ImageData\Sixth\210668\particle" ,\
                     r"F:\tianjin\ImageData\Sixth\210668\nonbigcell",\
                     r"F:\tianjin\ImageData\Sixth\210668\lp" ,\
                    
                     r"F:\tianjin\ImageData\Sixth\210832\bigcell",\
                     r"F:\tianjin\ImageData\Sixth\210832\impurity",\
                     r"F:\tianjin\ImageData\Sixth\210832\nonbigcell" ,\
                     r"F:\tianjin\ImageData\Sixth\210832\particle",\
                     r"F:\tianjin\ImageData\Sixth\210832\unknown" ,\
                    
                     r"F:\tianjin\ImageData\Sixth\210857\bigcell",\
                     r"F:\tianjin\ImageData\Sixth\210857\BlankAndParticle" ,\
                     r"F:\tianjin\ImageData\Sixth\210857\impurity",\
                     r"F:\tianjin\ImageData\Sixth\210857\lp" ,\
                         
                     r"F:\tianjin\ImageData\Sixth\210857\nonbigcell" ,\
                     r"F:\tianjin\ImageData\Sixth\210857\particle",\

                     r"F:\tianjin\ImageData\Sixth\210880\bigcell" ,\
                     r"F:\tianjin\ImageData\Sixth\210880\BlankAndParticle",\
                     r"F:\tianjin\ImageData\Sixth\210880\nonbigcell" ,\
                         
                     r"F:\tianjin\ImageData\Sixth\210962\truebigcell",\
                     r"F:\tianjin\ImageData\Sixth\210962\nonbigcell" ,\
                         
                     r"F:\tianjin\ImageData\Sixth\210965\truebigcell",\
                     r"F:\tianjin\ImageData\Sixth\210965\nonbigcell" ,\
                         
                     r"F:\tianjin\ImageData\Sixth\211039\nonbigcell",\
                    ]   
DatasetLabelList = [0,0,0,0,0, \
                    1,0,0,0,0,\
                    1,0,0,0, \
                    0,0,\
                    1,0,0,\
                    1,0,\
                    1,0,\
                    0\
                        ]

TrainLabelCSVFileName = 'lskj_train.csv'
TestLabelCSVFileName = 'lskj_test.csv'
VerifyLabelCSVFileName = 'lskj_verify.csv'
TotalLabelCSVFileName = 'lskj_total.csv'

# [test]

# root_dir = r"F:\tianjin\ImageData\Fifth"
# fileList = []
# import os
# for root,dirs,files in os.walk(root_dir):
#     fileList = dirs
#     break

fileList = ["210962","210965"]
TestImgFilePath = r"F:\tianjin\ImageData\Sixth\%s\unclass"
# TestImgFilePath = r"F:\beijing\ImageData\Sixth\%s\unclass"
moveDesName = 'label_1'

# [public]
taskRootDir = r'F:\DeepLearningRes\TJBigCell'
pthName = "classify.pth"