close all
tic
rootPath = "D:\武汉\北大胸腔第二次数据\";
desRootPath = "D:\武汉\北大胸腔第二次数据\RGB400Total\";
imgName = '208_';
filenamelist = dir(rootPath);
for file_index = 3:length(filenamelist)
    filePath = strcat(rootPath,filenamelist(file_index).name,'\');
    srcnamelist = dir(strcat(filePath,'*.tiff'));
    desPath = strcat(desRootPath,filenamelist(file_index).name);
    if exist(desPath,'dir')~=0
        rmdir(desPath, 's')
    end   
    mkdir(desPath);
    
    for i = 1:length(srcnamelist)
        disp(strcat(num2str(i),'/',num2str(length(srcnamelist))))
        filename = strcat(filePath,srcnamelist(i).name);
        src = imread(filename);
        RSGrayImage = imresize(src,[400 400]);
        resImage = zeros(400,400,3);
        for k = 1:3
           resImage(:,:,k) =  RSGrayImage;
        end
        resImage = uint8(resImage);
        imwrite(resImage,strcat(desPath,'\',imgName,srcnamelist(i).name(1:end-4),'png'));
    end
end


toc