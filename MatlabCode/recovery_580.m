
close all;
clc;
tic
j=1;

filePathList = ["Z:\RTx\RefWaveforms\"];
% filePathList = ["D:\tianjin\"];
fileDesPath ="D:\tianjin\";
imgDesPath = "G:\天津\图像数据\test\";

if exist(fileDesPath,'dir')==0
    mkdir(fileDesPath);
end   
if exist(imgDesPath,'dir')==0
    mkdir(imgDesPath);
end   

for h = 1:length(filePathList)
    srcPath = filePathList(h);
    srcnamelist = dir(strcat(srcPath,'*.Wfm.bin'));
    LengthOfNamelist = length(srcnamelist);
    for i = 1:LengthOfNamelist
        
        disp(strcat(num2str(i),'/',num2str(LengthOfNamelist)))
        filename = strcat(srcPath,srcnamelist(i).name);

        data = f_readData(filename);
        disp('load complete')
        movefile(strcat(srcPath,srcnamelist(i).name),strcat(fileDesPath,srcnamelist(i).name),'f');
        disp('move complete')
        
        L = length(data);
%         total = getNumFromName(srcnamelist(i).name);
        total = 160;
        for start = 0:total-1
            partdata = data((start*round(L/total)+1):1:(start+1)*round(L/total));
            partdata = partdata(30:end);
%             partdata = CropData(partdata,400,100,0.4); 
%             partdata = partdata(round(0.45*lengthOfData)-1048576/2:round(0.45*lengthOfData)+1048576/2-1);

            lengthOfData = length(partdata);
            if lengthOfData > 0
                [Image, RepRate] = ImageRecoveryModify(partdata, 20000000,1,0.5,1);
                [row,col,dep] = size(Image);
                NorImg = f_imgNormalize(Image);
                RSGrayImage = imresize(NorImg,[row 2.5*0.35*col]);
                [n,m,l] = size(RSGrayImage);
                %RSGrayImage = imcrop(RSGrayImage, round([1, n/4, m, n/2]));
                
                resImg = f_imgRemoveDCStripe(RSGrayImage);
                
                imwrite(resImg,strcat(imgDesPath,srcnamelist(i).name,'_',num2str(lengthOfData),'_',num2str(j),'.png'));
                j=j+1;
            else
                disp('no single')
            end
        end
    end
end
toc
close all;
beep