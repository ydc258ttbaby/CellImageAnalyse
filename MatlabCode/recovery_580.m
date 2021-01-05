
close all;
clc;
tic
j=1;

filePathList = ["E:\清华\原始数据\208879\";
                "E:\清华\原始数据\208894\";
                "E:\清华\原始数据\209116\";
                "E:\清华\原始数据\209172\"
                ];
fileDesPath = "E:\天津\原始数据\209116\";
imgDesPath = "E:\清华\图像数据\";

if exist(fileDesPath,'dir')==0
    mkdir(fileDesPath);
end   
if exist(imgDesPath,'dir')==0
    mkdir(imgDesPath);
end   

for h = 1:length(filePathList)
    srcPath = filePathList(h);
    srcnamelist = dir(strcat(srcPath,'*.bin'));
    LengthOfNamelist = length(srcnamelist);
    for i = 1:LengthOfNamelist
        
        disp(strcat(num2str(i),'/',num2str(LengthOfNamelist)))
        filename = strcat(srcPath,srcnamelist(i).name);
%         filename = "E:\清华\原始数据\209116\102452.bin";
        data = f_readData(filename);
%         disp('load complete')
%         movefile(strcat(srcPath,srcnamelist(i).name),strcat(fileDesPath,srcnamelist(i).name),'f');
%         disp('move complete')
        
        L = length(data);
%         total = getNumFromName(srcnamelist(i).name);
        total = 1;
        for start = 0:total-1
            partdata = data((start*round(L/total)+1):1:(start+1)*round(L/total));
            partdata = partdata(45:end);
%             partdata = CropData(partdata,400,100,0.4); 
%             partdata = partdata(round(0.45*lengthOfData)-1048576/2:round(0.45*lengthOfData)+1048576/2-1);

            lengthOfData = length(partdata);
            if lengthOfData > 0
%                 [Image, RepRate] = ImageRecoveryModify(partdata, 20000000,1,0.5,1);
                Image = ImageRecoveryTHU(partdata);
%                 figure
%                 imagesc(Image)
                [row,col,dep] = size(Image);
                ImageNor = f_imgNormalize(Image);
%                 RSGrayImage = imresize(NorImg,[row 2.5*0.35*col]);
                ImageRes = imresize(ImageNor,[2*row 0.5*0.35*col]);
                [n,m,l] = size(ImageRes);
              
%                 Image = f_imgRemoveDCStripe(Image);
                ImageCrop = f_imgCrop(ImageRes,40,150,5);
                ImageCrop = imresize(ImageCrop,[500,500]);
                charSrcPath = char(srcPath);
%                 figure
%                 imshow(ImageCrop)
                imwrite(ImageCrop,strcat(imgDesPath,charSrcPath(end-6:end-1),'_',srcnamelist(i).name(1:end-4),'.png'));
                j=j+1;
            else
                disp('no single')
            end
        end
    end
end
toc
% close all;
% beep