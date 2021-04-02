# [train]
DatasetCSVName = 'lskj_pos_bc'
DatasetImgDirList = [\
                    #  r"F:\wuhan\ImageData\Sixth\211039\bigbigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\211058\bigcell",\
                    #  r"F:\wuhan\ImageData\Seventh\211087\bigcell",\
                    #  r"F:\wuhan\ImageData\Seventh\211094\bigcell",\
                    #  r"F:\wuhan\ImageData\Seventh\211116\bigcell",\
                    #  r"F:\wuhan\ImageData\Seventh\211220\bigcell",\
                    #  r"F:\wuhan\ImageData\Seventh\211239\bigcell",\

                    #  r"F:\wuhan\ImageData\Sixth\210668\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210962\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210965\bigcell",\
                    #  r"F:\wuhan\ImageData\Third\209116\bigCell"\

                    #  r"F:\wuhan\ImageData\Seventh\211091\bigcell",\
                    #  r"F:\wuhan\ImageData\Seventh\211219\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210668\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210857\bigcell",\
                    #  r"F:\wuhan\ImageData\Third\209172\bigCell"\
                         
                    #  r"F:\wuhan\ImageData\Sixth\210668\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210694\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210857\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210880\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210883\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210962\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\210965\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\211039\bigcell",\
                    #  r"F:\wuhan\ImageData\Sixth\211058\bigcell",\

                    r"F:\wuhan\ImageData\Seventh\211087\bigcell",\
                    r"F:\wuhan\ImageData\Seventh\211087\nonbigcell",\
                    r"F:\wuhan\ImageData\Seventh\211087\label2",\
                    r"F:\wuhan\ImageData\Seventh\211087\unclass",\
                        
                    r"F:\wuhan\ImageData\Seventh\211091\bigcell",\
                    r"F:\wuhan\ImageData\Seventh\211091\nonbigcell",\
                    r"F:\wuhan\ImageData\Seventh\211091\label2",\
                    r"F:\wuhan\ImageData\Seventh\211091\unclass",\
                        
                    r"F:\wuhan\ImageData\Seventh\211116\bigcell",\
                    r"F:\wuhan\ImageData\Seventh\211116\nonbigcell",\
                    r"F:\wuhan\ImageData\Seventh\211116\unclass",\


                    ]   
DatasetLabelList = [\
                    # 0,0,1,0,1,0,0,0,1,1,1\
                    # 1,0,0,0,1\
                    # 1,1,1,1,1,1,1,1,1\
                    1,0,0,0,1,0,0,0,1,0,0 \
                        ]

TrainLabelCSVFileName = 'lskj_train.csv'
# TestLabelCSVFileName = 'lskj_test.csv'
# TestLabelCSVFileName = 'lskj_trueTest_total.csv' # true test
TestLabelCSVFileName = 'lskj_pos_bc_total.csv' # true test
VerifyLabelCSVFileName = 'lskj_verify.csv'
TotalLabelCSVFileName = 'lskj_total.csv'

# [test]
# import os
# root_dir = r"F:\beijing\ImageData\DrugSensitivity"
# fileList = os.listdir(root_dir)

fileList = ["211239","211220"]
TestImgFilePath = r"F:\wuhan\ImageData\Seventh\%s\unclass"
# TestImgFilePath = r"F:\beijing\ImageData\Sixth\%s\unclass"
moveDesName = 'label_1'

# [public]
taskRootDir = r'F:\DeepLearningRes\WHSevenBigCell'
pthName = "classify.pth"