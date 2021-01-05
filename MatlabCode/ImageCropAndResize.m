clear all
close all
tic
srcPath = "D:\武汉\胸腔积液\523\rgb\";
desPath = "D:\武汉\胸腔积液\523\rgb400\";
srcnamelist = dir(strcat(srcPath,'*.png'));
for i = 1:length(srcnamelist)
    disp(strcat(num2str(i),'/',num2str(length(srcnamelist))))
    filename = strcat(srcPath,srcnamelist(i).name);
    src = imread(filename);
    [m,n,l] = size(src);
    cropImg = imcrop(src, round([250,250,999,999]));
    RSGrayImage = imresize(cropImg,[400 400]);
    imwrite(RSGrayImage,strcat(desPath,srcnamelist(i).name));
end