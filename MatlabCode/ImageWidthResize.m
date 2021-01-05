clear all
close all
tic
filePath = "D:\天津\图像数据\249_40k\160\";
srcnamelist = dir(strcat(filePath,'*.png'));

for i = 1:length(srcnamelist)
    disp(strcat(num2str(i),'/',num2str(length(srcnamelist))))
    filename = strcat(filePath,srcnamelist(i).name);
    src = imread(filename);
    [m,n,l] = size(src);
    if(n>1.5*m)
       RSGrayImage = imresize(src,[m round(n/2)]); 
       imwrite(RSGrayImage,filename);
    end
    
end